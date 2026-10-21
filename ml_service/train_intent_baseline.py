"""Training script for Auxiliary Support Intent Classification Baseline.

Trains a lightweight TF-IDF + Logistic Regression baseline on the 4-class
Bitext support intent dataset (train.jsonl), validates on val.jsonl,
and evaluates performance once on the held-out test.jsonl split.

IMPORTANT IDENTIFICATION & BOUNDARY:
- This is a classical ML baseline (TF-IDF + Logistic Regression), NOT a DistilBERT model.
- Model artifacts are stored in `models/auxiliary-intent-baseline/`, strictly
  separate from future DistilBERT artifact directories.
- Intent predictions provide auxiliary topic tagging only and NEVER influence
  follow-up triage (YES, NO, REVIEW), priority scoring, or outbound call decisions.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from .prepare_intent_dataset import INTENT_ID2LABEL, INTENT_LABEL2ID, TARGET_INTENTS

DATA_DIR = Path(__file__).resolve().parent / "data" / "auxiliary_intent"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "auxiliary-intent-baseline"


def load_split(jsonl_path: Path) -> Tuple[List[str], List[str], List[int]]:
    """Load texts, intent strings, and label IDs from a JSONL partition."""
    texts: List[str] = []
    labels: List[str] = []
    label_ids: List[int] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(item["intent"])
            label_ids.append(item["label_id"])

    return texts, labels, label_ids


def train_and_evaluate_baseline(
    data_dir: Path = DATA_DIR,
    model_output_dir: Path = MODEL_DIR,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Train TF-IDF + Logistic Regression on train split, select on val, evaluate on test."""
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load train, val, and test splits
    train_texts, train_labels, train_ids = load_split(data_dir / "train.jsonl")
    val_texts, val_labels, val_ids = load_split(data_dir / "val.jsonl")
    test_texts, test_labels, test_ids = load_split(data_dir / "test.jsonl")

    # 2. Validation model selection across regularization strengths
    c_candidates = [0.1, 1.0, 5.0, 10.0]
    best_c = 1.0
    best_val_macro_f1 = -1.0

    for c in c_candidates:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
            ("clf", LogisticRegression(C=c, max_iter=1000, random_state=random_seed)),
        ])
        pipe.fit(train_texts, train_labels)
        val_preds = pipe.predict(val_texts)
        val_f1 = float(f1_score(val_labels, val_preds, average="macro"))
        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_c = c

    # 3. Fit optimal pipeline on train split
    final_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=best_c, max_iter=1000, random_state=random_seed)),
    ])
    final_pipeline.fit(train_texts, train_labels)

    # 4. Final evaluation ONCE on held-out test split
    test_preds = final_pipeline.predict(test_texts)
    test_probs = final_pipeline.predict_proba(test_texts)

    test_acc = float(accuracy_score(test_labels, test_preds))
    test_macro_f1 = float(f1_score(test_labels, test_preds, average="macro"))
    test_weighted_f1 = float(f1_score(test_labels, test_preds, average="weighted"))

    # Detailed classification report
    report_dict = classification_report(
        test_labels,
        test_preds,
        labels=TARGET_INTENTS,
        output_dict=True,
    )

    per_class_metrics: Dict[str, Dict[str, Any]] = {}
    for intent in TARGET_INTENTS:
        if intent in report_dict:
            per_class_metrics[intent] = {
                "precision": float(report_dict[intent]["precision"]),
                "recall": float(report_dict[intent]["recall"]),
                "f1_score": float(report_dict[intent]["f1-score"]),
                "support": int(report_dict[intent]["support"]),
            }

    # Confusion matrix (ordered by TARGET_INTENTS)
    conf_mat = confusion_matrix(test_labels, test_preds, labels=TARGET_INTENTS).tolist()

    # 5. Save model pipeline
    model_path = model_output_dir / "baseline_pipeline.joblib"
    joblib.dump(final_pipeline, model_path)

    # 6. Save label mapping
    mapping_path = model_output_dir / "label_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({
            "label2id": INTENT_LABEL2ID,
            "id2label": INTENT_ID2LABEL,
            "classes": TARGET_INTENTS,
        }, f, indent=2)

    # 7. Save training manifest
    manifest: Dict[str, Any] = {
        "model_architecture": "TF-IDF (1-2 grams, max_features=10000) + Logistic Regression",
        "model_type": "baseline_cpu",
        "disclaimer": (
            "This is a lightweight classical CPU baseline for auxiliary intent classification. "
            "It is NOT a fine-tuned DistilBERT transformer model. Predictions are auxiliary topic "
            "tags and MUST NOT be used for follow-up triage or outbound call decisions."
        ),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "random_seed": random_seed,
        "selected_hyperparameters": {
            "C": best_c,
            "max_iter": 1000,
            "ngram_range": [1, 2],
            "max_features": 10000,
            "sublinear_tf": True,
            "val_macro_f1_during_tuning": best_val_macro_f1,
        },
        "dataset_split_counts": {
            "train": len(train_texts),
            "val": len(val_texts),
            "test": len(test_texts),
        },
        "held_out_test_evaluation": {
            "test_samples_count": len(test_texts),
            "accuracy": test_acc,
            "macro_f1": test_macro_f1,
            "weighted_f1": test_weighted_f1,
            "per_class": per_class_metrics,
            "confusion_matrix": conf_mat,
            "class_order": TARGET_INTENTS,
        },
        "artifacts": {
            "pipeline": "baseline_pipeline.joblib",
            "label_mapping": "label_mapping.json",
        },
    }

    manifest_path = model_output_dir / "baseline_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    print("Training TF-IDF + Logistic Regression baseline on train split...")
    eval_summary = train_and_evaluate_baseline()
    print("Baseline training and evaluation complete!")
    test_metrics = eval_summary["held_out_test_evaluation"]
    print(f"Held-out Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Held-out Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    for cls_name, mets in test_metrics["per_class"].items():
        print(f"  {cls_name}: P={mets['precision']:.3f}, R={mets['recall']:.3f}, F1={mets['f1_score']:.3f} (n={mets['support']})")
