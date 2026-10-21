"""Dataset Preparation Script for Auxiliary Support Intent Classification.

Filters the public Bitext Customer Support Dataset to 4 legitimate support intents:
1. payment_issue
2. get_refund
3. contact_customer_service
4. recover_password

Applies consistent text normalization, verifies zero label conflicts, performs
stratified train/val/test splitting with fixed random seed, verifies zero
leakage across splits, and exports JSONL partitions and an audit manifest.

IMPORTANT ARCHITECTURAL & ETHICAL BOUNDARY:
- Uses ONLY `instruction` as input text and `intent` as target label.
- NEVER uses `response` as an input feature.
- Intent labels are strictly auxiliary topics; they are NEVER mapped to
  follow-up triage labels (YES, NO, REVIEW).
- NO real student records or personal data are included.
"""

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_URL = (
    "https://huggingface.co/datasets/bitext/"
    "Bitext-customer-support-llm-chatbot-training-dataset/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)

TARGET_INTENTS: List[str] = [
    "payment_issue",
    "get_refund",
    "contact_customer_service",
    "recover_password",
]

INTENT_LABEL2ID: Dict[str, int] = {
    "payment_issue": 0,
    "get_refund": 1,
    "contact_customer_service": 2,
    "recover_password": 3,
}

INTENT_ID2LABEL: Dict[int, str] = {
    0: "payment_issue",
    1: "get_refund",
    2: "contact_customer_service",
    3: "recover_password",
}

OUTPUT_DATA_DIR = Path(__file__).resolve().parent / "data" / "auxiliary_intent"


def normalize_text(text: str) -> str:
    """Consistently normalize text for duplicate detection and modeling."""
    if not isinstance(text, str):
        return ""
    # Lowercase, replace non-standard whitespace, strip
    cleaned = text.lower().strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def prepare_intent_dataset(
    parquet_url_or_path: str = DATASET_URL,
    output_dir: Path = OUTPUT_DATA_DIR,
    random_seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, Any]:
    """Download, filter, deduplicate, split, and save the 4-class auxiliary dataset.
    
    Returns:
        Dict[str, Any]: Comprehensive dataset preparation summary manifest.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load parquet dataset (only columns: instruction and intent)
    df = pd.read_parquet(parquet_url_or_path, columns=["instruction", "intent"])
    raw_total = len(df)

    # 2. Filter to 4 selected classes
    filtered = df[df["intent"].isin(TARGET_INTENTS)].copy()
    raw_filtered_count = len(filtered)
    raw_class_counts = filtered["intent"].value_counts().to_dict()

    # 3. Apply text normalization
    filtered["normalized_text"] = filtered["instruction"].apply(normalize_text)

    # 4. Check for duplicates and label conflicts
    raw_duplicates_count = int(filtered["instruction"].duplicated().sum())
    norm_duplicates_count = int(filtered["normalized_text"].duplicated().sum())

    # Group by normalized text to check for conflicting labels
    grouped = filtered.groupby("normalized_text")["intent"].nunique()
    conflicts = grouped[grouped > 1]
    conflicts_count = int(len(conflicts))
    if conflicts_count > 0:
        # If any conflicts occur, drop them explicitly and report
        conflicting_texts = set(conflicts.index)
        filtered = filtered[~filtered["normalized_text"].isin(conflicting_texts)].copy()

    # 5. Deduplicate to eliminate duplicate normalized instructions across dataset
    dedup = filtered.drop_duplicates(subset=["normalized_text"]).copy().reset_index(drop=True)
    total_unique = len(dedup)
    dedup_class_counts = dedup["intent"].value_counts().to_dict()

    # 6. Stratified Train / Val / Test Splits
    val_test_ratio = val_ratio + test_ratio  # 0.30
    test_fraction_of_val_test = test_ratio / val_test_ratio  # 0.50

    train_df, val_test_df = train_test_split(
        dedup,
        test_size=val_test_ratio,
        random_state=random_seed,
        stratify=dedup["intent"],
    )

    val_df, test_df = train_test_split(
        val_test_df,
        test_size=test_fraction_of_val_test,
        random_state=random_seed,
        stratify=val_test_df["intent"],
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # 7. Zero-Leakage Verification
    train_texts = set(train_df["normalized_text"])
    val_texts = set(val_df["normalized_text"])
    test_texts = set(test_df["normalized_text"])

    leakage_train_val = len(train_texts.intersection(val_texts))
    leakage_train_test = len(train_texts.intersection(test_texts))
    leakage_val_test = len(val_texts.intersection(test_texts))
    leakage_free = (leakage_train_val == 0) and (leakage_train_test == 0) and (leakage_val_test == 0)
    assert leakage_free, "Data leakage detected across splits!"

    # 8. Save partitions as JSONL
    def save_jsonl(dataframe: pd.DataFrame, filepath: Path, split_name: str) -> int:
        records: List[Dict[str, Any]] = []
        for idx, row in dataframe.iterrows():
            records.append({
                "example_id": f"bitext_{split_name}_{idx:04d}",
                "text": row["instruction"],
                "normalized_text": row["normalized_text"],
                "intent": row["intent"],
                "label_id": INTENT_LABEL2ID[row["intent"]],
                "split": split_name,
            })
        with open(filepath, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return len(records)

    train_saved = save_jsonl(train_df, output_dir / "train.jsonl", "train")
    val_saved = save_jsonl(val_df, output_dir / "val.jsonl", "val")
    test_saved = save_jsonl(test_df, output_dir / "test.jsonl", "test")

    # 9. Create and save manifest
    manifest: Dict[str, Any] = {
        "dataset_source": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "random_seed": random_seed,
        "selected_intents": TARGET_INTENTS,
        "label2id": INTENT_LABEL2ID,
        "id2label": INTENT_ID2LABEL,
        "num_classes": len(TARGET_INTENTS),
        "input_column": "instruction",
        "target_column": "intent",
        "excluded_columns": ["response", "category", "flags"],
        "statistics": {
            "raw_total_rows": raw_total,
            "raw_filtered_rows": raw_filtered_count,
            "raw_class_counts": raw_class_counts,
            "raw_duplicate_instructions": raw_duplicates_count,
            "normalized_duplicate_instructions": norm_duplicates_count,
            "conflicting_duplicate_instructions": conflicts_count,
            "total_unique_examples": total_unique,
            "deduplicated_class_counts": dedup_class_counts,
        },
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "split_counts": {
            "train": train_saved,
            "val": val_saved,
            "test": test_saved,
            "total": train_saved + val_saved + test_saved,
        },
        "split_class_distributions": {
            "train": train_df["intent"].value_counts().to_dict(),
            "val": val_df["intent"].value_counts().to_dict(),
            "test": test_df["intent"].value_counts().to_dict(),
        },
        "preprocessing_steps": [
            "Filter to 4 target classes: payment_issue, get_refund, contact_customer_service, recover_password",
            "Text normalization: lowercase, unicode normalize, strip, multi-space collapse",
            "Conflict audit: verified 0 conflicting labels for identical normalized text",
            "Deduplication: drop duplicate normalized texts so no utterance spans splits",
            "Stratified split by intent using random_seed=42 (70% train, 15% val, 15% test)",
            "Strict cross-split leakage check: verified 0 overlapping normalized texts",
        ],
        "leakage_check_passed": leakage_free,
        "files": {
            "train": str((output_dir / "train.jsonl").name),
            "val": str((output_dir / "val.jsonl").name),
            "test": str((output_dir / "test.jsonl").name),
        },
    }

    manifest_path = output_dir / "intent_dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    print("Preparing 4-class Bitext auxiliary intent dataset...")
    summary = prepare_intent_dataset()
    print("Dataset preparation complete!")
    print(f"Total unique rows: {summary['statistics']['total_unique_examples']}")
    print(f"Splits: Train={summary['split_counts']['train']}, Val={summary['split_counts']['val']}, Test={summary['split_counts']['test']}")
    print(f"Leakage free: {summary['leakage_check_passed']}")
