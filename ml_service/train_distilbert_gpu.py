"""Training Script for Auxiliary DistilBERT Support Intent Classifier.

Fine-tunes Hugging Face `distilbert-base-uncased` to classify customer support
inquiries into 4 auxiliary support topics from the curated Bitext dataset:
1. payment_issue
2. get_refund
3. contact_customer_service
4. recover_password

Saves complete Hugging Face model artifacts to:
`ml_service/models/distilbert-intent-classifier/`

BOUNDARIES & SAFETY CONTRACT:
- Auxiliary topic classification only.
- Intent predictions NEVER map to follow-up triage (YES/NO/REVIEW) or priority scores.
- Adheres strictly to the canonical label2id and id2label mappings.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# 1. Define paths and canonical label mappings
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "auxiliary_intent"
OUTPUT_DIR = BASE_DIR / "models" / "distilbert-intent-classifier"
TMP_CHECKPOINT_DIR = BASE_DIR / "tmp_checkpoints"

LABEL2ID: Dict[str, int] = {
    "payment_issue": 0,
    "get_refund": 1,
    "contact_customer_service": 2,
    "recover_password": 3,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}
TARGET_INTENTS = ["payment_issue", "get_refund", "contact_customer_service", "recover_password"]


def load_jsonl_dataset(path: Path) -> Dataset:
    """Load JSONL partition into a Hugging Face Dataset."""
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                texts.append(item["text"])
                labels.append(item["label_id"])
    return Dataset.from_dict({"text": texts, "label": labels})


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute accuracy and Macro/Weighted F1 for model evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def train_distilbert_classifier(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete fine-tuning pipeline for DistilBERT intent classifier."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training device: {device.upper()}")
    if device == "cuda":
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load splits
    print("[*] Loading dataset splits...")
    train_dataset = load_jsonl_dataset(data_dir / "train.jsonl")
    val_dataset = load_jsonl_dataset(data_dir / "val.jsonl")
    test_dataset = load_jsonl_dataset(data_dir / "test.jsonl")
    print(f"    Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 2. Tokenizer and Tokenization
    print("[*] Loading tokenizer 'distilbert-base-uncased'...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. Model Initialization
    print("[*] Initializing DistilBERT sequence classification head (4 classes)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 4. Training Arguments
    use_fp16 = torch.cuda.is_available()
    total_train_steps = (len(train_tokenized) // batch_size) * num_epochs
    warmup_steps = max(1, int(total_train_steps * 0.1))

    training_args = TrainingArguments(
        output_dir=str(TMP_CHECKPOINT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        fp16=use_fp16,
        logging_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed=random_seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 5. Train
    print("[*] Starting model training...")
    train_result = trainer.train()
    print("[*] Model training complete!")

    # 6. Final Evaluation on Held-Out Test Set
    print("[*] Evaluating on untouched held-out test split...")
    test_preds_raw = trainer.predict(test_tokenized)
    test_logits = test_preds_raw.predictions
    test_pred_labels = np.argmax(test_logits, axis=-1)
    true_labels = np.array(test_dataset["label"])

    test_acc = float(accuracy_score(true_labels, test_pred_labels))
    test_macro_f1 = float(f1_score(true_labels, test_pred_labels, average="macro"))
    test_weighted_f1 = float(f1_score(true_labels, test_pred_labels, average="weighted"))

    report_dict = classification_report(
        true_labels,
        test_pred_labels,
        target_names=TARGET_INTENTS,
        output_dict=True,
    )
    conf_mat = confusion_matrix(true_labels, test_pred_labels).tolist()

    # 7. Save Model Artifacts
    print(f"[*] Saving model artifacts to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    manifest: Dict[str, Any] = {
        "model_architecture": "distilbert-base-uncased",
        "model_type": "distilbert",
        "task": "auxiliary_intent_classification",
        "disclaimer": (
            "This model provides auxiliary topic tagging only. Predictions MUST NOT be used "
            "for follow-up triage (YES/NO/REVIEW), priority scoring, or outbound call decisions."
        ),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "hyperparameters": {
            "base_model": "distilbert-base-uncased",
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": 0.01,
            "warmup_steps": warmup_steps,
            "fp16": use_fp16,
            "random_seed": random_seed,
        },
        "dataset_split_counts": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "held_out_test_evaluation": {
            "test_samples_count": len(test_dataset),
            "accuracy": test_acc,
            "macro_f1": test_macro_f1,
            "weighted_f1": test_weighted_f1,
            "classification_report": report_dict,
            "confusion_matrix": conf_mat,
            "class_order": TARGET_INTENTS,
        },
    }

    with open(output_dir / "distilbert_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 8. Clean up temporary checkpoints
    if TMP_CHECKPOINT_DIR.exists():
        shutil.rmtree(TMP_CHECKPOINT_DIR, ignore_errors=True)

    print("\n================ FINAL EVALUATION RESULTS ================")
    print(f"Test Accuracy:    {test_acc * 100:.2f}%")
    print(f"Test Macro-F1:    {test_macro_f1:.4f}")
    print(f"Test Weighted-F1: {test_weighted_f1:.4f}")
    for intent in TARGET_INTENTS:
        m = report_dict[intent]
        print(f"  {intent:24s} Precision: {m['precision']:.3f} | Recall: {m['recall']:.3f} | F1: {m['f1-score']:.3f}")
    print("==========================================================\n")

    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train DistilBERT auxiliary intent classifier")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is not available")
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() returned False!")
    train_distilbert_classifier()
