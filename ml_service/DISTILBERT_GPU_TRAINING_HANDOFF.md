# DistilBERT GPU Training Handoff Guide

**Target Task:** 4-Class Auxiliary Support Intent Classification
**Target Hardware:** NVIDIA GPU Laptop / Workstation with CUDA
**Base Model:** Hugging Face `distilbert-base-uncased`
**Target Checkpoint Directory:** `ml_service/models/distilbert-intent-classifier/`
**Handoff Date:** October 2026

---

## 1. Objective and Boundary Reminder

You are fine-tuning `distilbert-base-uncased` to classify customer support inquiries into 4 legitimate support topics from the public Bitext dataset.

> [!IMPORTANT]
> **AUXILIARY BOUNDARY:**
> This classifier predicts **intent topic tags only** (`payment_issue`, `get_refund`, `contact_customer_service`, `recover_password`).
> It **does NOT** predict follow-up triage (`YES`/`NO`/`REVIEW`) or priority scores. Follow-up and priority decisions remain strictly governed by the safety-guarded decision engine.

---

## 2. Dataset Files & Locations

The dataset has already been filtered, normalized, deduplicated, and split with random seed `42` (zero text overlap across splits):

* **Train Split ($70\%$):** `ml_service/data/auxiliary_intent/train.jsonl` ($n = 2,715$)
* **Validation Split ($15\%$):** `ml_service/data/auxiliary_intent/val.jsonl` ($n = 582$)
* **Held-out Test Split ($15\%$):** `ml_service/data/auxiliary_intent/test.jsonl` ($n = 582$)
* **Dataset Manifest:** `ml_service/data/auxiliary_intent/intent_dataset_manifest.json`

Each line in the JSONL files is structured as:

```json
{
  "example_id": "bitext_train_0001",
  "text": "i cant make a payment where can i notify of an error",
  "normalized_text": "i cant make a payment where can i notify of an error",
  "intent": "payment_issue",
  "label_id": 0,
  "split": "train"
}
```

**Input feature:** `text` (or `instruction`)
**Target label:** `label_id` (integer `0..3`) or `intent` (string)

---

## 3. Canonical Label-to-ID Contract

The model **MUST** be instantiated with the following exact mapping in `config.json`:

```python
label2id = {
    "payment_issue": 0,
    "get_refund": 1,
    "contact_customer_service": 2,
    "recover_password": 3,
}

id2label = {
    0: "payment_issue",
    1: "get_refund",
    2: "contact_customer_service",
    3: "recover_password",
}
```

---

## 4. Complete Training Script (`train_distilbert_gpu.py`)

Run this script on your GPU laptop:

```python
import json
from pathlib import Path
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# 1. Define paths and mappings
DATA_DIR = Path("ml_service/data/auxiliary_intent")
OUTPUT_DIR = Path("ml_service/models/distilbert-intent-classifier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

label2id = {
    "payment_issue": 0,
    "get_refund": 1,
    "contact_customer_service": 2,
    "recover_password": 3,
}
id2label = {v: k for k, v in label2id.items()}

# 2. Load dataset partitions
def load_jsonl(path: Path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                texts.append(item["text"])
                labels.append(item["label_id"])
    return Dataset.from_dict({"text": texts, "label": labels})

train_dataset = load_jsonl(DATA_DIR / "train.jsonl")
val_dataset = load_jsonl(DATA_DIR / "val.jsonl")
test_dataset = load_jsonl(DATA_DIR / "test.jsonl")

# 3. Tokenize
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")

train_dataset = train_dataset.map(tokenize_batch, batched=True)
val_dataset = val_dataset.map(tokenize_batch, batched=True)
test_dataset = test_dataset.map(tokenize_batch, batched=True)

# 4. Initialize DistilBERT model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4,
    id2label=id2label,
    label2id=label2id,
)

# 5. Training Arguments (optimized for GPU)
training_args = TrainingArguments(
    output_dir="./tmp_checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=50,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 6. Train and Save Final Model
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"DistilBERT model artifacts successfully saved to: {OUTPUT_DIR}")
```

---

## 5. Expected Output Artifacts

Upon completion, `ml_service/models/distilbert-intent-classifier/` should contain:

- `config.json` (containing `id2label` and `label2id`)
- `model.safetensors` (or `pytorch_model.bin`)
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `vocab.txt`

---

## 6. Verification After Saving Artifacts

Once you copy the model directory into `ml_service/models/distilbert-intent-classifier/`, test it immediately:

```python
from ml_service.intent_classifier import AuxiliaryIntentClassifier

classifier = AuxiliaryIntentClassifier(model_dir="ml_service/models/distilbert-intent-classifier")
loaded = classifier.load_model()
print(f"Loaded successfully: {loaded}, Model Type: {classifier.model_type}")

pred = classifier.predict_intent("Fee deduction failed during payment portal submission")
print(f"Predicted Intent: {pred.predicted_intent}, Confidence: {pred.intent_confidence}")
assert pred.predicted_intent == "payment_issue"
```

The system will automatically recognize the DistilBERT model and serve intent predictions through `StudentSupportPredictor`.
