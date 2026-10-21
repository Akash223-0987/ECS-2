# Student Support ML Module (Foundation & Model Loader)

Standalone Python Machine Learning module for student support ticket triage and analysis (VIT-AP University).

---

## 1. Module Purpose

This module establishes the data contracts, validation pipelines, configuration structures, model loader, and prediction interfaces for automated support ticket triage.

The architecture is designed around **DistilBERT** sequence classification. It uses **lazy imports** for `transformers` and `torch` so the module can run, be tested, and perform rule-based validation on machines without GPU hardware or deep learning packages installed. When a task-specific fine-tuned model checkpoint is loaded, it performs inference; otherwise, it guarantees safe, explainable `REVIEW` fallback routing.

---

## 2. Directory Structure

```
ml_service/
├── .gitignore               # Ignores caches, virtualenvs, and model weight binaries
├── pyproject.toml           # Package metadata and pytest configuration
├── requirements.txt         # Minimal dependency list (pydantic, dateutil, pytest)
├── README.md                # Module documentation and usage guide
├── LABELLING_GUIDELINES.md  # Comprehensive human annotation guidelines and rubrics
├── TRAINING_CONTRACT.md     # Engineering specification for teammate training on GPU
├── __init__.py              # Package exports
├── config.py                # Configuration for Hugging Face models, devices, and paths
├── schemas.py               # Pydantic v2 schemas for TicketInput and TicketPrediction
├── dataset_schema.py        # Versioned training example schema, PII scrubber, and manifest
├── training_contract.py     # Canonical label mappings, hyperparameters, and report schemas
├── prepare_intent_dataset.py# Data preparation for 4-class auxiliary intent dataset
├── train_intent_baseline.py # Training & evaluation for TF-IDF + Logistic Regression baseline
├── intent_classifier.py     # AuxiliaryIntentClassifier interface for baseline/transformer models
├── validator.py             # Validation logic for timestamps, text, and PII handling
├── model_loader.py          # Hugging Face DistilBERT model loader with lazy imports & safety checks
├── decision_engine.py       # Follow-up decision engine decoupling inference from business rules
├── priority_engine.py       # Conservative priority assessment engine (urgency, interest, scoring)
├── predictor.py             # Triage prediction interface coordinating schemas, engine, and loader
├── data/
│   └── auxiliary_intent/    # Processed JSONL splits (train, val, test) and audit manifest
├── models/
│   └── auxiliary-intent-baseline/ # Serialized CPU baseline pipeline and metadata
└── tests/
    ├── __init__.py
    ├── conftest.py          # Pytest path discovery configuration
    ├── fixtures.py          # Synthetic mock test fixtures (clearly labelled, non-training)
    ├── test_schemas.py      # Tests for input/output schemas and serialization
    ├── test_dataset_schema.py # Tests for training schemas, PII scrubbing, and split leakage
    ├── test_training_contract.py # Tests for contract consistency, hyperparameters, and validation
    ├── test_intent_classifier.py # Tests for intent classification, dataset splits, and safety isolation
    ├── test_validator.py    # Tests for timestamp parsing and empty descriptions
    ├── test_predictor.py    # Tests for baseline placeholder inference and safety constraints
    ├── test_model_loader.py # Tests for model loader, mock injection, and label validation
    ├── test_decision_engine.py # Tests for business rules, evidence weights, and conflicts
    ├── test_priority_engine.py # Tests for urgency, interest, priority scoring, and boundary safety
    └── test_e2e_pipeline.py    # End-to-end pipeline validation across all execution branches
```

---

## 3. Input & Output Contracts

### Input Schema (`TicketInput`)
Accepts records aligned with upstream voice agents (LiveKit), Google Sheets, and n8n workflows:

| Field | Type | Description |
|---|---|---|
| `ticket_id` | `Optional[str]` | Unique session ticket identifier (e.g., `John1`) |
| `date_and_time` | `Optional[Union[datetime, str]]` | ISO-8601 or formatted creation timestamp |
| `student_name` | `Optional[str]` | Student full name *(PII - excluded from ML features)* |
| `application_number` | `Optional[str]` | Application/registration number *(PII - excluded from ML features)* |
| `registration_number` | `Optional[str]` | Alias automatically synced to `application_number` |
| `phone_number` | `Optional[str]` | Student contact number *(PII - excluded from ML features)* |
| `issue_description` | `Optional[str]` | Textual inquiry details. **Primary NLP feature.** |
| `priority_level` | `Optional[str]` | Legacy hardcoded field (`"High"`). **Excluded from training labels.** |
| `status` | `Optional[str]` | Current ticket state (`"Open"`, `"Pending"`, etc.) |

### Output Schema (`TicketPrediction`)
Standardized response guaranteed for every ticket evaluation:

| Field | Type / Values | Description |
|---|---|---|
| `ticket_id` | `Optional[str]` | Pass-through ticket identifier |
| `follow_up_required` | `YES`, `NO`, `REVIEW` | Whether human intervention is needed |
| `interest_level` | `HIGH`, `MEDIUM`, `LOW`, `UNKNOWN` | Assessed academic/program interest |
| `urgency` | `HIGH`, `MEDIUM`, `LOW`, `UNKNOWN` | Assessed urgency based on inquiry |
| `follow_up_reason` | `Optional[str]` | Human-readable explanation of decision |
| `priority_score` | `Optional[float]` (0.0 to 100.0) | Numerical score, or `null` when undefensible |
| `priority` | `HIGH`, `MEDIUM`, `LOW`, `NOT_APPLICABLE` | Categorical priority |
| `recommended_action` | `str` | Actionable next step for support staff |
| `model_status` | `str` | Operational status (see below) |
| `predicted_intent` | `Optional[str]` | Optional auxiliary intent classification (`payment_issue`, `get_refund`, etc.) |
| `intent_confidence` | `Optional[float]` (0.0 to 1.0) | Optional uncalibrated confidence score for auxiliary intent |

### Operational Status Codes & Safe Fallback Explanations (`ModelStatus`)

Every prediction returns an explicit diagnostic `model_status`. When conditions prevent confident automated inference, the system safely falls back to human `REVIEW` with a `null` priority score:

| Status Code | Type | Trigger Condition | Pipeline Action | Operational Remedy |
|---|---|---|---|---|
| `READY` | Active | Task-specific fine-tuned model loaded with verified labels (`YES`, `NO`, `REVIEW`), or rule-based engines operating normally. | Evaluates text, produces decision and defensible score if scorable. | Normal automated triage. |
| `PLACEHOLDER_PENDING_MODEL` | Fallback | Predictor running without model weights or injected fine-tuned classifier. | Routes to `REVIEW`, sets `priority_score: null`. | Standard queue triage until task-specific model is trained. |
| `GENERIC_MODEL_NOT_FINE_TUNED` | Guardrail | Loaded model contains generic pretrained labels (e.g. `LABEL_0`, `LABEL_1`, `POSITIVE`, `NEGATIVE`). | Strictly rejects generic labels, forces `REVIEW`, sets `priority_score: null`. | Fine-tune model on domain-specific student support dataset. |
| `DEPENDENCIES_UNAVAILABLE` | Fallback | Optional ML packages (`torch`, `transformers`) are not installed on the execution environment. | Intercepts gracefully via lazy imports, forces `REVIEW`. | Run `pip install torch transformers` on inference machines. |
| `MODEL_NOT_FOUND` | Fallback | Configured local checkpoint directory path does not exist on disk. | Intercepts missing files safely, forces `REVIEW`. | Verify checkpoint directory path in `MLServiceConfig`. |
| `NOT_INITIALIZED` | Diagnostic | `HuggingFaceModelLoader` instantiated but `load_model()` has not been called. | Blocks uninitialized inference, forces `REVIEW`. | Call `loader.load_model()` prior to serving inference requests. |
| `EMPTY_DESCRIPTION_REVIEW` | Guardrail | Input `issue_description` is `None`, empty string, or whitespace-only. | Bypasses NLP inference, routes directly to `REVIEW`. | Contact student or inspect ticketing portal to obtain query text. |
| `INVALID_TIMESTAMP_REVIEW` | Guardrail | Creation timestamp is corrupted, non-standard, or unparseable. | Catches format error, forces `REVIEW` without crashing. | Verify timestamp formatting from upstream Google Sheets / n8n webhook. |
| `VALIDATION_ERROR` | Guardrail | Input payload fails Pydantic schema validation. | Rejects malformed payload, routes to `REVIEW`. | Inspect upstream integration payload structure. |

---

## 4. Follow-Up Decision Engine (`decision_engine.py`)

The `FollowUpDecisionEngine` decouples raw model inference from business decisions, applying conservative evidence-based reasoning:

- **Decision Outcomes**:
  - `YES`: Sufficient evidence supports a follow-up (explicit callback request, confirmed urgent blocker, or corroborated fine-tuned model prediction).
  - `NO`: Reliable evidence establishes that no further contact is needed (explicit confirmation of resolution in text AND closed ticket status).
  - `REVIEW`: Evidence is missing, ambiguous, brief, conflicting, or the classifier is unready.
- **Evidence & Conflict Resolution**:
  - **Open status is insufficient**: An open status alone never forces a `YES`.
  - **Constant High priority is ignored**: The legacy `priority_level: High` field does not independently influence follow-up eligibility.
  - **Closed status is insufficient**: A closed status alone without confirming resolution text produces `REVIEW`, not `NO`. Missing information is never inferred as resolution.
  - **Conflict detection**: If text contradicts status (e.g. status closed but student reports ongoing blocker) or contradicts classifier predictions, the engine automatically flags a conflict and routes to `REVIEW`.
  - **Injectable classifier interface**: Supports duck-typed or `HuggingFaceModelLoader` classifiers. When an approved classifier is not available, limitations are explicitly recorded.
  - **PII exclusion**: Student names, phone numbers, application numbers, and ticket IDs are strictly excluded from predictive text analysis.

---

## 5. Conservative Priority Assessment Engine (`priority_engine.py`)

The `PriorityAssessmentEngine` provides dedicated priority scoring, urgency assessment, and student expressed interest evaluation, strictly separated from model inference and follow-up eligibility.

> [!IMPORTANT]
> **DISCLAIMER & PRELIMINARY STATUS:**
> The criteria, weights, and scoring thresholds defined herein are preliminary heuristic rules developed for automated baseline triage. They **DO NOT** represent officially approved or institutional policies of VIT-AP University. They are not machine-learned or statistically calibrated, and they make no claim of production readiness or calibrated accuracy.

### 5.1 Urgency Criteria (`UrgencyLevel`)
Evaluates time-sensitivity based strictly on textual evidence in the inquiry description:
- **`HIGH`** (40.0 - 50.0 pts):
  - Imminent hard deadlines (`"deadline is today"`, `"due tomorrow"`, `"within 24 hours"`, exam today/tomorrow). **(50.0 pts)**
  - Urgent application/financial blockers (`"fee was deducted twice"`, `"payment deducted but failing"`, `"portal gave an error"`, `"account blocked"`). **(40.0 pts)**
- **`MEDIUM`** (25.0 pts):
  - Unresolved issues with meaningful but non-imminent deadlines or consequences (`"scholarship verification"`, `"branch change deadline next month"`, `"grade appeal"`, `"hostel allotment"`).
- **`LOW`** (10.0 pts):
  - Non-urgent or routine requests with no indicated deadline or serious consequence (`"syllabus"`, `"library timings"`, `"bus pass schedule"`, `"general query"`, `"no rush"`).
- **`UNKNOWN`** (0.0 pts):
  - Insufficient or ambiguous information to assess urgency (e.g., missing text, brief queries like `"fees"`).

### 5.2 Expressed Student Interest Criteria (`InterestLevel`)
> [!NOTE]
> **Strict Independence from Urgency (Task 4):**
> `interest_level` is **never inferred from urgency**. In student support tickets, an urgent financial blocker does not signify high student interest. Interest is assigned **only** when the inquiry provides explicit, defensible evidence of the student's expressed interest/intent:
- **`HIGH`**: Student explicitly expresses keen interest, commitment, or eager intent to enroll/attend (`"eager to join"`, `"want to accept my admission offer"`, `"my top choice"`).
- **`MEDIUM`**: Student expresses exploratory interest or curriculum curiosity (`"interested in knowing more about MBA"`, `"considering applying"`).
- **`LOW`**: Student explicitly expresses withdrawal, declining admission, or disinterest (`"wish to withdraw application"`, `"joining another college"`, `"cancel my admission"`).
- **`UNKNOWN`**: Default for all routine support tickets (technical glitches, refunds, fee receipts, routine questions).

### 5.3 Transparent Scoring Model (Factors, Weights & Thresholds)
When a ticket is eligible for follow-up (`YES`), priority is computed using transparent, additive factors bounded to `[0.0, 100.0]`:

$$\text{Priority Score} = \text{Urgency Score} (50\text{ max}) + \text{Impact Score} (30\text{ max}) + \text{Actionability Score} (20\text{ max})$$

| Factor | Sub-Criteria | Points Contributed |
|---|---|---|
| **Urgency** (max 50 pts) | Imminent hard deadline / emergency | +50.0 pts |
| | Urgent blocker without same-day deadline | +40.0 pts |
| | Medium urgency (upcoming consequence) | +25.0 pts |
| | Low urgency (routine / non-urgent) | +10.0 pts |
| | Unknown urgency | 0.0 pts |
| **Impact / Severity** (max 30 pts) | Critical blocker (fee deducted twice, portal lockout, exam access) | +30.0 pts |
| | Moderate impact (scholarship, branch change, grade appeal) | +15.0 pts |
| | Routine impact (syllabus, bus routes, calendar) | +5.0 pts |
| | Neutral / Unidentified impact | 0.0 pts |
| **Actionability** (max 20 pts) | Explicit student callback request (`"please call me back"`) | +20.0 pts |
| | Ongoing unresolved issue without explicit callback request | +10.0 pts |
| | Routine or self-service query | 0.0 pts |

**Categorical Priority Mapping (`PriorityCategory`):**
- **`HIGH`**: Score $\ge 70.0$
- **`MEDIUM`**: $40.0 \le \text{Score} < 70.0$
- **`LOW`**: $0.0 \le \text{Score} < 40.0$
- **`NOT_APPLICABLE`**: Follow-up is `NO` or `REVIEW`, or score cannot be defensibly calculated.

### 5.4 Strict Follow-Up Decision Respect & Safety Rules
1. **`NO` Follow-Up Required**:
   - Must result in `priority: NOT_APPLICABLE` and `priority_score: null`.
   - **Outbound call recommendation is strictly suppressed**: `recommended_action: "No advisor follow-up required: Issue is confirmed resolved or closed."`
2. **`REVIEW` Follow-Up Required**:
   - Must remain a review case. It is **never converted to automatic call eligibility** by high urgency or a high score.
   - Must result in `priority: NOT_APPLICABLE` and `priority_score: null`.
   - `recommended_action: "Manual advisor review required: Inspect ticket details before taking operational action."`
3. **`YES` Follow-Up Required**:
   - Receives priority assessment only if evidence is sufficient.
   - If description is ambiguous or only says `"Please call me back"` without issue context, score returns `null` and priority returns `NOT_APPLICABLE` with an explanatory rationale.
4. **Urgency vs Score Distinction**:
   - A `HIGH` urgency ticket **does not automatically receive a score of 100.0**. For example, a high-urgency issue with no same-day deadline and no callback request receives a defensible score of 80.0.
5. **PII and Constant Priority Invariance**:
   - Student identifiers (`student_name`, `application_number`, `registration_number`, `phone_number`, `ticket_id`) and legacy `priority_level: High` are strictly excluded from predictive features. Changing them produces identical scores, urgency, and priority.

### 5.5 Operational Recommended Actions

| Decision / Priority Tier | Recommended Action | Semantics |
|---|---|---|
| `NO` / `NOT_APPLICABLE` | `"No advisor follow-up required: Issue is confirmed resolved or closed."` | Ticket resolved; suppresses outbound calling. |
| `REVIEW` / `NOT_APPLICABLE` | `"Manual advisor review required: Inspect ticket details before taking operational action."` | Ambiguous, brief, or conflicting signals; requires staff review before contact. |
| `YES` / `HIGH` | `"Urgent advisor callback: Contact student promptly to resolve time-sensitive blocker."` | Priority $\ge 70.0$; time-sensitive blocker or imminent deadline. |
| `YES` / `MEDIUM` | `"Standard advisor callback: Contact student during standard support hours to resolve issue."` | Priority $40.0 - 69.9$; unresolved standard inquiry. |
| `YES` / `LOW` | `"Routine support outreach: Contact student or provide email guidance as queue capacity permits."` | Priority $< 40.0$; routine query with callback request. |
| `YES` / Insufficient Evidence | `"Advisor follow-up required: Contact student to gather issue details (context missing in ticket)."` | Callback requested but issue details missing; score is null. |

### 5.6 Code Example: Using Priority Assessment Engine

```python
from ml_service import (
    FollowUpDecisionEngine,
    PriorityAssessmentEngine,
    StudentSupportPredictor,
    TicketInput,
)

# 1. Standalone assessment
engine = PriorityAssessmentEngine()
ticket = TicketInput(
    ticket_id="VIT-101",
    issue_description="Deadline is tomorrow and fee was deducted twice, unable to submit. Please call me back urgently.",
    status="Open",
)
assessment = engine.assess(ticket)
print(f"Urgency: {assessment.urgency}")                  # HIGH
print(f"Interest: {assessment.interest_level}")          # UNKNOWN (not inferred from urgency!)
print(f"Priority Score: {assessment.priority_score}")    # 100.0
print(f"Priority Category: {assessment.priority}")       # HIGH
print(f"Action: {assessment.recommended_action}")

# 2. Integrated with StudentSupportPredictor
dec_engine = FollowUpDecisionEngine()
prio_engine = PriorityAssessmentEngine(decision_engine=dec_engine)
predictor = StudentSupportPredictor(decision_engine=dec_engine, priority_engine=prio_engine)

prediction = predictor.predict(ticket)
print(prediction.model_dump_json(indent=2))
```

---

## 6. End-to-End Pipeline Usage & Architecture (`predictor.py`)

The end-to-end triage prediction pipeline coordinates all modules into a unified, type-safe interface via `StudentSupportPredictor.create_default_pipeline()`:

```
[Incoming Payload (dict or TicketInput)]
                   │
                   ▼
       1. TicketValidator (validator.py)
          ├── Validates & sanitizes schema
          ├── Strips PII from predictive text
          ├── Checks empty text -> (EMPTY_DESCRIPTION_REVIEW)
          └── Validates timestamp -> (INVALID_TIMESTAMP_REVIEW)
                   │
                   ▼
       2. FollowUpDecisionEngine (decision_engine.py)
          ├── Interrogates fine-tuned model or heuristics
          ├── Rejects generic DistilBERT labels
          ├── Detects workflow conflicts (status vs text)
          └── Produces FollowUpDecision (YES, NO, REVIEW)
                   │
                   ▼
       3. PriorityAssessmentEngine (priority_engine.py)
          ├── Assesses urgency (HIGH, MEDIUM, LOW, UNKNOWN)
          ├── Assesses expressed interest (strictly independent)
          ├── Respects follow-up decision (NO -> NOT_APPLICABLE, REVIEW -> null)
          └── Computes priority score [0.0 - 100.0] & action
                   │
                   ▼
       4. Structured Output Serialization (schemas.py)
          └── Returns validated TicketPrediction (dict / JSON)
```

### Concise End-to-End Usage Example

```python
from ml_service import StudentSupportPredictor, TicketInput

# 1. Instantiate the fully-wired end-to-end pipeline in one line
predictor = StudentSupportPredictor.create_default_pipeline()

# 2. Evaluate a raw ticket input (dict or TicketInput)
raw_ticket = {
    "ticket_id": "VIT-2026-001",
    "student_name": "Rahul Varma",          # PII safely excluded from predictive features
    "phone_number": "9123456780",           # PII safely excluded
    "application_number": "26BCE123",       # PII safely excluded
    "date_and_time": "2026-10-21T10:30:00",
    "issue_description": (
        "Tuition fee was deducted twice from my account and portal gave an error. "
        "Deadline is tomorrow, please call me back urgently."
    ),
    "priority_level": "High",               # Legacy constant safely ignored
    "status": "Open",
}

# 3. Predict structured triage outcome
prediction = predictor.predict(raw_ticket)

# 4. Access structured, schema-validated results
print(f"Ticket ID:           {prediction.ticket_id}")
print(f"Follow-Up Required:  {prediction.follow_up_required}")   # YES
print(f"Urgency:             {prediction.urgency}")              # HIGH
print(f"Expressed Interest:  {prediction.interest_level}")       # UNKNOWN (not inferred from urgency)
print(f"Priority Score:      {prediction.priority_score}")       # 100.0
print(f"Priority Tier:       {prediction.priority}")             # HIGH
print(f"Recommended Action:  {prediction.recommended_action}")   # "Urgent advisor callback..."
print(f"Model Status:        {prediction.model_status}")         # "PLACEHOLDER_PENDING_MODEL" or "READY"

# 5. Serialize to JSON for downstream APIs (n8n, webhook, database)
json_payload = prediction.model_dump_json(indent=2)
print(json_payload)
```

---

## 7. Dataset Schema, Annotation Guidelines & Data Preparation Workflow

For full annotation rubrics and instructions, see [LABELLING_GUIDELINES.md](file:///e:/ECS/ECS-2/ml_service/LABELLING_GUIDELINES.md).

### 7.1 Versioned Training Schema (`dataset_schema.py`)
Training examples adhere to a strict versioned schema (`schema_version: "1.0.0"`):

| Field | Type | Description |
|---|---|---|
| `example_id` | `str` | Stable unique identifier for the example (e.g., `ex_2026_0001`). |
| `text` | `str` | Sanitized ticket narrative used as model input. **Strictly devoid of PII and notes.** |
| `label` | `YES`, `NO`, `REVIEW` | Human-reviewed ground truth label. |
| `label_provenance` | `LabelProvenance` | `HUMAN_SINGLE_REVIEW`, `HUMAN_CONSENSUS`, `ADJUDICATED_EXPERT`, `HISTORICAL_AUDIT`. |
| `adjudication_status` | `AdjudicationStatus` | `APPROVED`, `PENDING_REVIEW`, `FLAGGED_CONFLICT`, `REJECTED`, `EXCLUDED`. |
| `notes` | `Optional[str]` | Reviewer notes and rationale, stored **strictly separately** from model input text. |
| `group_id` | `Optional[str]` | Anonymized student or case grouping key to prevent split leakage. |
| `split` | `Optional[SplitType]` | Assigned data partition: `TRAIN`, `VALIDATION`, or `TEST`. |
| `created_at` | `Optional[datetime]` | Timestamp for temporal split ordering. |
| `metadata` | `Dict[str, Any]` | Operational metadata (reviewer ID, source ticket ID) isolated from text. |

### 7.2 Strict PII Protection & Scrubbing
To comply with the Digital Personal Data Protection (DPDP) Act 2023, FERPA, and institutional privacy standards:
1. **Never copy raw student names, phone numbers, application/registration numbers, or email addresses into the model input text.**
2. The `sanitize_text_for_training()` utility replaces identifiers with standardized placeholder tokens:
   - Student Name $\rightarrow$ `[STUDENT_NAME]`
   - Phone Number $\rightarrow$ `[PHONE_NUMBER]`
   - Application / Registration Number $\rightarrow$ `[APPLICATION_NUMBER]`
   - Email Address $\rightarrow$ `[EMAIL]`
3. `TrainingExample` validators inspect text and reject submissions containing unredacted phone numbers, emails, or student IDs.

### 7.3 Labelling Guidelines & Ground-Truth Protocol
- **`YES` (Follow-up Required):** Sufficient evidence indicates that human follow-up or advisor intervention is needed (explicit callback request, active financial deduction failure, course registration blocker, exam admit card issue).
- **`NO` (No Follow-up Required):** Sufficient evidence establishes that no further follow-up is needed (explicit confirmation in student's own words that the issue is resolved and everything is working, explicit statement that no help is needed).
  - *Note:* A closed ticket status in CRM without student confirmation text must **never** be labelled `NO`; it must be labelled `REVIEW`. Missing information is never inferred as resolution.
- **`REVIEW` (Ambiguous, Fragmented, Conflicting):** Text lacks sufficient evidence (single keywords like *"fees"*, truncated messages, contradictory signals between status and student narrative).
- **Mandatory Ground-Truth Rule:** Training ground truth **MUST** be assigned by qualified human reviewers or historical institutional audits. Automatically generated heuristic labels or unverified model predictions are **strictly prohibited** from training ground truth to avoid feedback loops.

### 7.4 Handling Operational Challenges
- **Class Imbalance:** Natural label distributions in student ticketing are preserved during annotation. Imbalance is compensated mathematically via class-weighted cross-entropy loss in training, not by distorting human labels.
- **Duplicate Tickets:** Students opening multiple tickets for the same issue are assigned a shared `group_id` (case hash). Exact duplicate texts are deduplicated within training splits.
- **Conflicting Labels:** Disagreements between annotators are flagged as `FLAGGED_CONFLICT` and escalated to senior support supervisors (`ADJUDICATED_EXPERT`).
- **Exclusion Rules:** Non-English text, internal test tickets (*"test 123"*), unredactable passwords/credentials, and corrupted telephony transcripts are assigned `adjudication_status: EXCLUDED` and omitted from training.

### 7.5 Leakage-Resistant Dataset Partitioning
- **Group-Based Splitting:** Related tickets from the same student or inquiry are assigned the same `group_id`. All records sharing a `group_id` must reside in the **same split**. `DatasetManifest.validate_split_leakage()` guarantees that no `group_id` or duplicate text spans multiple splits.
- **Temporal Split:** When timestamps are available, data is ordered chronologically so the Validation and Test sets reflect later time windows, mimicking real production distribution drift.
- **Untouched Test Set:** The 15% Test partition must remain completely untouched during model selection and hyperparameter tuning.

---

## 8. Training & Evaluation Contract (GPU Execution Specification)

For full execution details and code templates, see [TRAINING_CONTRACT.md](file:///e:/ECS/ECS-2/ml_service/TRAINING_CONTRACT.md).

### 8.1 Model Architecture & Canonical Label Mapping
- **Base Architecture:** `distilbert-base-uncased` (66M parameters).
- **Classification Head:** 3-class sequence classification (`num_labels=3`).
- **Non-Negotiable Label Mapping:**
  ```python
  LABEL2ID = {"NO": 0, "YES": 1, "REVIEW": 2}
  ID2LABEL = {0: "NO", 1: "YES", 2: "REVIEW"}
  ```
  *(Mismatched or generic labels such as `LABEL_0` will be rejected by `HuggingFaceModelLoader` under `GENERIC_MODEL_NOT_FINE_TUNED`.)*

### 8.2 Standardized Hyperparameter Baseline (`TrainingHyperparameters`)
- **Random Seed:** `42` (fixed across PyTorch, NumPy, and Python for exact reproducibility).
- **Max Sequence Length:** `512` (handles long student narratives without truncation).
- **Learning Rate:** `2e-5` with AdamW optimizer ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$).
- **Batch Size:** `16` (or `32` if $\ge$ 16GB VRAM).
- **Epochs:** `4` with early stopping on validation Macro-F1.
- **Warmup Ratio:** `0.1` (10% of total steps with linear decay).
- **Weight Decay:** `0.01` L2 regularization.
- **Precision:** `fp16=True` (or `bf16=True` on RTX 30/40 series GPUs).
- **Loss Function:** Class-weighted cross-entropy ($w_c = \frac{N}{3 \cdot N_c}$) to handle imbalance.

### 8.3 Evaluation Metrics Hierarchy (`EvaluationReport`)
Evaluations on the untouched test split must be exported to `EvaluationReport`:
1. **Macro-F1 (Target $\ge$ 0.85):** Primary model selection metric across `NO`, `YES`, and `REVIEW`.
2. **Precision on `YES` (Target $\ge$ 0.90):** Minimizing **False Positives on `YES`** is critical. A false `YES` triggers an unneeded outbound phone call, wasting counselor time and frustrating resolved students.
3. **Recall on `NO` (Target $\ge$ 0.85):** Ensures resolved inquiries are recognized, suppressing unneeded calls.
4. **Treatment of `REVIEW`:** Measured via `review_precision`. When uncertain, predicting `REVIEW` is safe and desired.
5. **3x3 Confusion Matrix:** Fully audited grid across `[NO, YES, REVIEW]`.

### 8.4 Checkpoint Artifact Layout & Verification
Trained artifacts are saved in `./models/distilbert-student-support/`:
```
models/distilbert-student-support/
├── config.json               # Contains id2label and label2id
├── model.safetensors         # Model weight binary
├── tokenizer.json            # Fast tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
├── vocab.txt
└── training_metadata.json    # Complete lineage manifest (hyperparameters, metrics, commit)
```

Verify checkpoint integrity programmatically before deployment:
```python
from ml_service.training_contract import validate_checkpoint_directory

is_valid, errors = validate_checkpoint_directory("./models/distilbert-student-support")
assert is_valid, f"Checkpoint invalid: {errors}"
```

### 8.5 Safety Guardrail Invariant
**Model predictions do NOT bypass business safety guardrails:**
1. Predictions are passed to `FollowUpDecisionEngine`, which checks for conflicts. If a model predicts `YES` on a ticket confirming resolution with closed status, the engine flags conflict and forces `REVIEW`.
2. Tickets with `follow_up_required = NO` or `REVIEW` receive `priority: NOT_APPLICABLE` and have outbound call recommendations strictly suppressed.

---

## 9. Auxiliary Support Intent Classification (Bitext 4-Class Pipeline)

An independent auxiliary intent classification pipeline categorizes support inquiries into 4 legitimate support topics from the public Bitext dataset (`bitext/Bitext-customer-support-llm-chatbot-training-dataset`):

1. `payment_issue` (fee payment failures, debit errors, transaction issues)
2. `get_refund` (tuition/caution deposit refund requests)
3. `contact_customer_service` (requests to speak with a human advisor or support officer)
4. `recover_password` (portal login credential recovery, password/PIN resets)

> [!IMPORTANT]
> **STRICT INDEPENDENCE & ETHICAL BOUNDARY:**  
> - Uses **ONLY** `instruction` as input text and `intent` as target label.
> - Intent labels represent auxiliary topic categorization and are **NEVER mapped to follow-up triage labels (`YES`, `NO`, `REVIEW`)**.
> - Intent predictions **do NOT alter** `follow_up_required`, `urgency`, `priority_score`, `priority` tier, or outbound call recommendations.
> - An explicitly resolved ticket with closed status continues to receive `follow_up_required = NO`, `priority = NOT_APPLICABLE`, and outbound call suppression, even if its intent is classified as `payment_issue`.

### 9.1 Prepared Dataset Partitions (`data/auxiliary_intent/`)
Prepared via `prepare_intent_dataset.py` with text normalization, zero duplicate leakage across splits, and an auditable manifest (`intent_dataset_manifest.json`):

| Partition | Total Samples | `payment_issue` | `get_refund` | `contact_customer_service` | `recover_password` |
|---|:---:|:---:|:---:|:---:|:---:|
| **Train (70%)** | 2,715 | 695 | 638 | 691 | 691 |
| **Validation (15%)** | 582 | 149 | 136 | 148 | 149 |
| **Held-out Test (15%)** | 582 | 149 | 137 | 148 | 148 |
| **Total Unique** | **3,879** | **993** | **911** | **987** | **988** |

- **Duplicate Handling:** 79 raw duplicates and 112 normalized duplicates were identified; 0 conflicting labels were found. Duplicate normalized texts were pruned prior to splitting, guaranteeing **0 cross-split leakage**.

### 9.2 Measured CPU Baseline Performance (TF-IDF + Logistic Regression)
Trained via `train_intent_baseline.py` on the Train split, tuned on Validation, and evaluated **once** on the untouched Test split ($n=582$):

- **Model Type:** Classical CPU Baseline (TF-IDF 1-2 grams + Logistic Regression, $C=1.0$). *(Clearly identified as a baseline, NOT DistilBERT).*
- **Artifacts Location:** `./models/auxiliary-intent-baseline/`
- **Measured Held-out Test Metrics:**
  - **Overall Test Accuracy:** `99.83%` (581 / 582 correct)
  - **Macro-F1:** `0.9983`
  - **Weighted-F1:** `0.9983`
- **Per-Class Breakdown on Test Split:**
  - `payment_issue`: Precision = 1.000, Recall = 0.993, F1 = 0.997 ($n=149$)
  - `get_refund`: Precision = 1.000, Recall = 1.000, F1 = 1.000 ($n=137$)
  - `contact_customer_service`: Precision = 0.993, Recall = 1.000, F1 = 0.997 ($n=148$)
  - `recover_password`: Precision = 1.000, Recall = 1.000, F1 = 1.000 ($n=148$)

### 9.3 Inference Usage with AuxiliaryIntentClassifier

```python
from ml_service import AuxiliaryIntentClassifier, StudentSupportPredictor, TicketInput

# 1. Instantiate and load auxiliary intent classifier
intent_clf = AuxiliaryIntentClassifier(model_dir="./models/auxiliary-intent-baseline")
intent_clf.load_model()

# 2. Wire into triage pipeline
pipeline = StudentSupportPredictor.create_default_pipeline(intent_classifier=intent_clf)

# 3. Predict on ticket
ticket = TicketInput(
    ticket_id="VIT-101",
    issue_description="Payment of 25000 was deducted twice from my account during fee submission.",
)
prediction = pipeline.predict(ticket)

print(prediction.predicted_intent)    # "payment_issue"
print(prediction.intent_confidence)   # e.g., 0.9854 (uncalibrated probability)
print(prediction.follow_up_required)  # Evaluated independently by decision engine
```

---

## 10. Current Limitations & Safety Guardrails

- **No Automatic Downloads:** `local_files_only=True` is enforced by default. Ordinary imports and tests will never make external HTTP requests to Hugging Face Hub.
- **Zero Hallucinated Predictions:** If a model is not fine-tuned, not found, or missing dependencies, the module routes tickets to `REVIEW`. It never invents confidence scores or labels.
- **Strict Generic Label Rejection:** Generic labels (e.g. `LABEL_0`, `LABEL_1`, `POSITIVE`, `NEGATIVE`) are never mapped to `YES`, `NO`, or `REVIEW`.
- **Heuristic Priority Scoring:** Priority scores are rule-based and transparently calculated from urgency, blocker severity, and callback requests. They are NOT machine-learned or statistically calibrated. Tickets with insufficient evidence return `priority_score: null`.
- **PII Protection:** Student identifiers (`student_name`, `application_number`, `registration_number`, `phone_number`) and legacy `priority_level` are strictly excluded from model input text and scoring.
- **Safe Fallbacks:** Corrupted timestamps and empty descriptions are intercepted gracefully without throwing unhandled exceptions.

---

## 11. Setup & Running Unit Tests (Development Environment)

The development test suite uses dependency injection, synthetic test fixtures, and mocks. It runs entirely without PyTorch, GPU, or internet access:

```bash
# 1. Install minimal lightweight requirements:
pip install -r requirements.txt

# 2. Run all unit and end-to-end tests (151 passing tests):
python -m pytest ml_service/tests -v
```

---

## 12. Quick-Start Guide: Deploying and Running the Model on a GPU Machine

When deploying a fine-tuned model checkpoint on a GPU-enabled laptop/workstation:

### Step 1: Install GPU Dependencies
```bash
# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate
```

### Step 2: Initialize and Run Inference on GPU
Load the fine-tuned checkpoint explicitly using `HuggingFaceModelLoader`:

```python
from ml_service import (
    MLServiceConfig,
    HuggingFaceModelLoader,
    StudentSupportPredictor,
    TicketInput,
)

# 1. Configure for local checkpoint and GPU device
config = MLServiceConfig(
    model_name="./models/distilbert-student-support",
    device="cuda",          # Explicit GPU selection ('cuda' or 'cuda:0')
    local_files_only=True,  # Prevent Hub network calls
)

# 2. Explicitly load the model
loader = HuggingFaceModelLoader(config=config, device="cuda")
success = loader.load_model()
print(f"Model loaded: {success}, Status: {loader.status}, Device: {loader.device}")

# 3. Create fully-wired end-to-end triage pipeline
predictor = StudentSupportPredictor.create_default_pipeline(config=config, model_loader=loader)

# 4. Predict on a ticket
ticket = TicketInput(
    ticket_id="VIT-001",
    issue_description="Need urgent assistance with tuition fee refund before semester starts.",
)
prediction = predictor.predict(ticket)
print(prediction.model_dump_json(indent=2))
```

The loader automatically verifies that `id2label` matches `{0: 'NO', 1: 'YES', 2: 'REVIEW'}`, moves tensors to `cuda`, and produces predictions with verified confidence scores while preserving rule-based safety guardrails.

