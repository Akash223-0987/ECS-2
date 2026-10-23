# Model Integration Contract: Student Issue Priority Classification

This document specifies the engineering contract, model specifications, allowed priority labels, artifact format, and integration boundaries for predicting `priority_level` from student support `issue_transcription`.

---

## 1. Executive Summary & Compatibility Findings

### Existing Repository Model Compatibility
1. **Primary DistilBERT Classifier (`model_loader.py`)**:
   - **Target Labels**: `{"NO": 0, "YES": 1, "REVIEW": 2}`
   - **Task**: Follow-up advisor triage (whether human follow-up callback is necessary).
   - **Compatibility with Priority**: **INCOMPATIBLE**. Does not output priority levels.
2. **Auxiliary Intent Classifier (`intent_classifier.py` / `models/auxiliary-intent-baseline`)**:
   - **Target Labels**: `["payment_issue", "get_refund", "contact_customer_service", "recover_password"]`
   - **Task**: Customer support topic classification from Bitext dataset.
   - **Compatibility with Priority**: **INCOMPATIBLE**. Topic intents do not map to priority levels.
3. **Previous Priority Engine (`priority_engine.py`)**:
   - Priority was previously computed purely via non-ML heuristic regex scoring (urgency, impact, actionability).
4. **Current Status**:
   - No trained model currently exists in the repository that predicts `priority_level`.
   - The ML service implementation uses a **mock model / test double / heuristic fallback** (`PriorityModelLoader`) with explicit status reporting (`is_trained_model = False`) so the Google Sheets watcher functions end-to-end immediately without fake confidence claims.
   - Once the GPU training teammate completes fine-tuning, the saved artifact can simply be placed in the designated directory (`./models/distilbert-priority/` or specified via `PRIORITY_MODEL_PATH`) without modifying any watcher or service code.

---

## 2. Model Integration Contract

### Input Specification
- **Field Name**: `issue_transcription` (also accepts alias `issue_description` for backwards compatibility).
- **Type**: `str`
- **Description**: Textual narrative or speech-to-text transcript of the student inquiry recorded during call session.
- **Preprocessing Required**:
  - Stripping leading and trailing whitespace (`text.strip()`).
  - Handling empty or whitespace-only inputs gracefully (empty inputs must default to `LOW` or `NOT_APPLICABLE` with `EMPTY_TRANSCRIPTION` status).
  - Truncation to max 512 tokens using standard Hugging Face DistilBERT tokenizer (`padding=True, truncation=True, max_length=512, return_tensors="pt"`).
  - PII (student names, phone numbers, registration numbers) MUST be excluded from predictive features.

### Output Specification
- **Field Name**: `priority_level`
- **Type**: `str`
- **Allowed Priority Labels**:
  The project requires 3 categorical triage priority levels:
  - `HIGH`: Time-sensitive blockers, imminent deadlines, payment debited without receipt, exam/portal errors.
  - `MEDIUM`: Important inquiries requiring standard resolution (scholarships, document verification, hostel room changes).
  - `LOW`: General information inquiries, course details, campus queries, or non-urgent follow-ups.

### Canonical Label-to-ID Mapping
The trained model `config.json` **MUST** include the following exact mapping:

```json
{
  "label2id": {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2
  },
  "id2label": {
    "0": "LOW",
    "1": "MEDIUM",
    "2": "HIGH"
  }
}
```

*(Note: Case-insensitive Title-case values `"Low"`, `"Medium"`, `"High"` are also accepted by the loader).*

---

## 3. Artifact Format & Expected Directory Structure

The trained model artifact must be placed in:
`ml_service/models/distilbert-priority/` (or path set by `PRIORITY_MODEL_PATH`).

Expected directory files:
```text
ml_service/models/distilbert-priority/
├── config.json              # Model configuration containing label2id and id2label
├── model.safetensors        # Model weights (or pytorch_model.bin)
├── tokenizer.json           # Fast tokenizer configuration
├── tokenizer_config.json    # Tokenizer settings
├── special_tokens_map.json  # Special token mappings
└── vocab.txt                # WordPiece vocabulary
```

### Alternative Classical Baseline Format
If a classical Scikit-Learn baseline is trained instead of a Transformer:
```text
ml_service/models/priority-baseline/
├── baseline_pipeline.joblib # Trained Pipeline(TfidfVectorizer, LogisticRegression)
└── label_mapping.json       # JSON mapping with {"classes": ["LOW", "MEDIUM", "HIGH"]}
```

---

## 4. Expected Model Loading Method

The model loader is isolated in [`ml_service.priority_model.PriorityModelLoader`](file:///e:/ECS/ECS-2/ml_service/priority_model.py).

### How the Loader Loads the Artifact:
```python
from ml_service.priority_model import PriorityModelLoader

loader = PriorityModelLoader(model_dir="./models/distilbert-priority")
success = loader.load_model()

if success:
    print("Model loaded successfully with validated priority labels!")
else:
    print(f"Fallback active: {loader.status_reason}")
```

### Loading Lifecycle & Guardrails:
1. **Lazy Imports**: `torch` and `transformers` are imported dynamically.
2. **Offline Mode**: Operates in `local_files_only=True` mode; never attempts internet downloads.
3. **Label Verification**: Confirms that `id2label` contains allowed priority labels (`{"HIGH", "MEDIUM", "LOW"}`). If the directory contains a model with labels like `{"YES", "NO", "REVIEW"}` or `{"payment_issue", ...}`, it is flagged as `INCOMPATIBLE_LABELS` and rejected.
4. **Isolated Integration**: The Google Sheets watcher depends solely on `PriorityPredictor.predict_priority()`. When the trained weights are dropped into `./models/distilbert-priority/`, the loader automatically picks them up on restart without rewriting the watcher.

---

## 5. Example Inference Call and Return Structure

### Inference Call
```python
from ml_service.predictor import PriorityPredictor
from ml_service.schemas import PriorityInferenceInput, PriorityInferenceOutput

predictor = PriorityPredictor()

# Call with plain string
output = predictor.predict_priority(
    "Tuition fee was debited twice from my account and today is the deadline to register!"
)

# Or call with PriorityInferenceInput schema
input_data = PriorityInferenceInput(
    issue_transcription="Can I get information on the elective courses for semester 4?",
    ticket_id="John12",
)
output = predictor.predict_priority(input_data)
```

### Return Structure (`PriorityInferenceOutput`)
```json
{
  "priority_level": "HIGH",
  "confidence": 0.9421,
  "model_status": "READY",
  "is_trained_model": true,
  "status_reason": "Inference from fine-tuned DistilBERT priority model"
}
```

### Fallback Return Structure (When Model Is Pending GPU Training)
```json
{
  "priority_level": "HIGH",
  "confidence": null,
  "model_status": "PENDING_TRAINED_MODEL",
  "is_trained_model": false,
  "status_reason": "Pending GPU training: Heuristic rule-based priority estimate active"
}
```

---

## 6. Training Checkpoint Validation Script
To verify that your trained checkpoint is compatible before handoff, run:
```bash
python -c "
from ml_service.priority_contract import validate_priority_checkpoint_directory
valid, errors, meta = validate_priority_checkpoint_directory('./models/distilbert-priority')
print('Valid:', valid)
if errors:
    print('Errors:', errors)
"
```
