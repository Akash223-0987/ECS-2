# Training and Evaluation Contract for DistilBERT Classifier

**Target Hardware:** Dedicated GPU Laptop / Workstation (NVIDIA CUDA 12.x)  
**Base Architecture:** Hugging Face `distilbert-base-uncased`  
**Task:** 3-Class Sequence Classification for Support Ticket Follow-Up Triage  
**Target Output Checkpoint:** `./models/distilbert-student-support/`  
**Contract Version:** `1.0.0`

---

## 1. Executive Summary for GPU Teammate

This document specifies the exact technical requirements, label encodings, hyperparameter baselines, directory layouts, and evaluation standards for fine-tuning the student support ticket classifier on a GPU laptop.

Adherence to this contract ensures that the trained model checkpoint will load cleanly into the `ml_service.HuggingFaceModelLoader` without runtime label validation errors or configuration mismatches.

---

## 2. Canonical Label-to-ID Mapping

The classification head **MUST** be instantiated with the following exact label-to-id and id-to-label mappings:

```python
LABEL2ID = {
    "NO": 0,
    "YES": 1,
    "REVIEW": 2,
}

ID2LABEL = {
    0: "NO",
    1: "YES",
    2: "REVIEW",
}
```

> [!CAUTION]
> **DO NOT USE GENERIC OR NUMERIC LABELS:**  
> If the model is saved with generic labels (such as `LABEL_0`, `LABEL_1`, `LABEL_2` or `POSITIVE`, `NEGATIVE`), the inference loader will **strictly reject the model** under `ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED` and all tickets will default to `REVIEW`. The labels in `config.json` must be string `NO`, `YES`, `REVIEW`.

---

## 3. Dataset Format and Split Strategy

### 3.1 Input Format
Training data must be provided as a JSONL file adhering to `TrainingExample` / `DatasetManifest`:

```json
{
  "example_id": "ex_2026_0042",
  "text": "Payment was deducted twice from my bank account during fee submission. Portal shows unpaid.",
  "label": "YES",
  "label_provenance": "HUMAN_SINGLE_REVIEW",
  "adjudication_status": "APPROVED",
  "group_id": "case_hash_89a3f2",
  "split": "TRAIN"
}
```

### 3.2 Leakage-Resistant Splitting
- **Split Proportions:** 70% Train, 15% Validation, 15% Test.
- **Group-Based Splitting:** Multiple tickets belonging to the same student or case must share the same `group_id`. **All examples sharing a `group_id` must reside in the same split.** No `group_id` may cross between Train, Validation, and Test splits.
- **Temporal Integrity:** When timestamps are available, order tickets chronologically so the Validation and Test sets reflect later time windows, mimicking real-world deployment distribution shifts.
- **Untouched Test Set:** The 15% Test partition must remain sealed until final hyperparameter selection is complete.

---

## 4. Handling Class Imbalance

In ticket triage datasets, `YES` or `REVIEW` examples often exceed explicit `NO` resolutions. To prevent the model from ignoring the minority class, apply **class-weighted cross-entropy loss** during training:

```python
import torch
import torch.nn as nn
from transformers import Trainer

# Compute inverse frequency weights
# class_counts = [count_no, count_yes, count_review]
# total_samples = sum(class_counts)
# weights = [total_samples / (3.0 * count) for count in class_counts]
class_weights = torch.tensor([w_no, w_yes, w_review], dtype=torch.float).to("cuda")

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

---

## 5. Recommended Reproducible Hyperparameters

To guarantee reproducible training, initialize all seeds and use the baseline parameters verified in `TrainingHyperparameters`:

| Parameter | Recommended Value | Rationale |
|---|---|---|
| **Base Model** | `distilbert-base-uncased` | Lightweight (66M params), fast inference on CPU/GPU, strong text classification baseline. |
| **Random Seed** | `42` | Set across `torch.manual_seed(42)`, `numpy.random.seed(42)`, and Python `random.seed(42)`. |
| **Max Sequence Length** | `512` | Captures long student explanations without truncation. |
| **Batch Size** | `16` (or `32` if $\ge$ 16GB VRAM) | Stable gradients on laptop GPUs (e.g. RTX 3060 / 4060 / 4070). |
| **Learning Rate** | `2e-5` | Standard AdamW rate for transformer fine-tuning. |
| **Optimizer** | `AdamW` ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$) | Standard optimizer for Hugging Face Transformers. |
| **Weight Decay** | `0.01` | L2 regularization to prevent overfitting on specialized phrasing. |
| **Warmup Ratio** | `0.1` (10% of total steps) | Prevents early gradient shocks to pretrained representations. |
| **Number of Epochs** | `4` | Sufficient convergence; stop early if validation Macro-F1 plateaus. |
| **Precision** | `fp16=True` (or `bf16=True` on RTX 30/40) | Reduces VRAM by 50% and accelerates throughput by 2-3x. |
| **Evaluation Strategy** | `evaluation_strategy="epoch"` | Evaluate validation metrics at every epoch. |
| **Best Model Selection** | `metric_for_best_model="eval_macro_f1"` | Select checkpoint maximizing unweighted Macro-F1. |

---

## 6. Evaluation Standards & Metrics

Every trained model candidate must be evaluated on the untouched Test split. The evaluation report must be exported as JSON adhering to `ml_service.training_contract.EvaluationReport`.

### 6.1 Priority Metrics Hierarchy

1. **Macro-F1 (Target $\ge$ 0.85):**  
   The unweighted arithmetic mean of F1 scores across `NO`, `YES`, and `REVIEW`. Prevents the model from scoring high by merely predicting the majority class.
2. **Precision on `YES` (Target $\ge$ 0.90):**  
   Minimizing **False Positives on `YES`** is critical. A false `YES` triggers an automated outbound phone call to a student who does not need assistance, wasting university advisor time and degrading the student experience.
3. **Recall on `NO` (Target $\ge$ 0.85):**  
   Ensures that resolved inquiries are reliably recognized, suppressing unneeded calls.
4. **Precision on `REVIEW` (Target $\ge$ 0.80):**  
   Validates that when the model routes a case to `REVIEW`, the ticket is indeed ambiguous, incomplete, or borderline.
5. **3x3 Confusion Matrix:**  
   Ordered by `[NO, YES, REVIEW]` for both rows (ground truth) and columns (predicted):
   ```
                   Predicted NO    Predicted YES   Predicted REVIEW
   True NO              TN              FP_yes          FP_rev
   True YES             FN_no           TP_yes          FN_rev
   True REVIEW          F_rev_no        F_rev_yes       TP_rev
   ```

---

## 7. Required Checkpoint Directory Structure

Upon completion of training, save model artifacts inside `./models/distilbert-student-support/`. The directory **MUST** contain:

```
models/distilbert-student-support/
├── config.json               # Model config containing id2label and label2id
├── model.safetensors         # Model weight binary (or pytorch_model.bin)
├── tokenizer.json            # Fast tokenizer definition
├── tokenizer_config.json     # Tokenizer configuration
├── special_tokens_map.json   # Special tokens mapping
├── vocab.txt                 # Tokenizer vocabulary
└── training_metadata.json    # Lineage manifest matching TrainingMetadata schema
```

### 7.1 Schema for `training_metadata.json`
```json
{
  "model_name": "distilbert-student-support",
  "model_version": "1.0.0",
  "base_model": "distilbert-base-uncased",
  "trained_at": "2026-10-22T10:00:00Z",
  "git_commit_hash": "a1b2c3d4e5f6...",
  "hyperparameters": {
    "base_model": "distilbert-base-uncased",
    "max_seq_length": 512,
    "learning_rate": 0.00002,
    "batch_size": 16,
    "num_epochs": 4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "random_seed": 42,
    "fp16": true,
    "class_weights_strategy": "balanced",
    "metric_for_best_model": "macro_f1"
  },
  "dataset_manifest_id": "vit_support_dataset_v1.0.0",
  "dataset_version": "1.0.0",
  "train_samples_count": 1400,
  "val_samples_count": 300,
  "test_samples_count": 300,
  "evaluation_summary": {
    "model_name": "distilbert-student-support",
    "model_version": "1.0.0",
    "dataset_version": "1.0.0",
    "evaluated_at": "2026-10-22T11:00:00Z",
    "test_samples_count": 300,
    "macro_f1": 0.885,
    "weighted_f1": 0.892,
    "accuracy": 0.890,
    "per_class": {
      "NO": {"precision": 0.88, "recall": 0.86, "f1_score": 0.87, "support": 90},
      "YES": {"precision": 0.92, "recall": 0.90, "f1_score": 0.91, "support": 130},
      "REVIEW": {"precision": 0.85, "recall": 0.89, "f1_score": 0.87, "support": 80}
    },
    "confusion_matrix": [
      [77, 3, 10],
      [4, 117, 9],
      [6, 3, 71]
    ],
    "false_yes_count": 6,
    "false_no_count": 10,
    "review_precision": 0.85,
    "notes": "Trained with balanced cross-entropy on CUDA RTX 4070."
  },
  "label2id": {"NO": 0, "YES": 1, "REVIEW": 2},
  "id2label": {0: "NO", 1: "YES", 2: "REVIEW"},
  "institutional_disclaimer": "Trained for student-support triage guidance. Model predictions provide advisory signals and do NOT bypass institutional business rules or human oversight."
}
```

---

## 8. Verification Procedure Prior to Submission

Before handing off the model directory, run the verification utility:

```python
from ml_service.training_contract import validate_checkpoint_directory
from ml_service.model_loader import HuggingFaceModelLoader

checkpoint_dir = "./models/distilbert-student-support"

# 1. Validate artifact structure and metadata
is_valid, errors = validate_checkpoint_directory(checkpoint_dir, require_metadata=True)
if not is_valid:
    print(f"FAILED checkpoint validation: {errors}")
else:
    print("PASSED checkpoint validation!")

# 2. Test loading through HuggingFaceModelLoader
loader = HuggingFaceModelLoader(model_name_or_path=checkpoint_dir, device="cuda")
loaded = loader.load_model()
print(f"Loaded: {loaded}, Status: {loader.status}, Fine-Tuned: {loader.is_fine_tuned}")
assert loader.is_fine_tuned is True, "Model is not recognized as fine-tuned!"
```

---

## 9. Safety Invariant Reminder

**Model predictions DO NOT bypass safety guardrails.**  
Even if the learned DistilBERT model outputs `YES` with high confidence, the prediction is passed to `FollowUpDecisionEngine` and `PriorityAssessmentEngine`:
1. If text confirms resolution and the ticket is closed, the decision engine catches the conflict and forces `REVIEW`.
2. Tickets with `follow_up_required = NO` or `REVIEW` are strictly assigned `priority: NOT_APPLICABLE` and will **never** trigger automated outbound calls.
3. Outbound calls are never initiated without defensible priority justification.
