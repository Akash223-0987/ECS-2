"""Model Integration Contract and Specifications for Priority Prediction.

Defines the contract, canonical mappings, validation rules, and schemas
for fine-tuning and deploying a priority classification model in ml_service.

PURPOSE:
Provides an unambiguous specification for the teammate executing model training
on GPU hardware.

CRITICAL INVARIANTS:
1. Expected Input: `issue_transcription` (raw or normalized text).
2. Expected Output: `priority_level` with one of the allowed labels.
3. Allowed Priority Labels:
   - {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
   - {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
4. Separation of Concerns:
   - The watcher monitors the Google Sheet and passes text to the inference interface.
   - The model loading code is fully isolated so the trained artifact can be dropped in
     without modifying the watcher or polling service.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field

# ==============================================================================
# Canonical Priority Label-to-ID Contract
# ==============================================================================

MODEL_FAMILY: str = "distilbert-base-uncased"

PRIORITY_LABEL2ID: Dict[str, int] = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
}

PRIORITY_ID2LABEL: Dict[int, str] = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH",
}

NUM_PRIORITY_LABELS: int = 3

ALLOWED_PRIORITY_LABELS: Set[str] = {"HIGH", "MEDIUM", "LOW"}
MINIMAL_PRIORITY_LABELS: Set[str] = {"HIGH", "LOW"}

DEFAULT_PRIORITY_MODEL_DIR: str = "./models/distilbert-priority"

REQUIRED_CHECKPOINT_FILES: Set[str] = {
    "config.json",
}

WEIGHT_FILE_CANDIDATES: Set[str] = {
    "model.safetensors",
    "pytorch_model.bin",
}

TOKENIZER_FILE_CANDIDATES: Set[str] = {
    "tokenizer.json",
    "vocab.txt",
    "tokenizer_config.json",
}


def normalize_priority_label(label: str, title_case: bool = False) -> str:
    """Normalize arbitrary string inputs to canonical allowed priority labels.
    
    Args:
        label: Raw label string (e.g., 'high', 'HIGH', 'High', 'critical').
        title_case: If True, returns Title-cased string ('High'); otherwise UPPER ('HIGH').
        
    Returns:
        Canonical label string ('HIGH', 'MEDIUM', 'LOW').
    """
    cleaned = str(label).strip().upper()
    if cleaned in ("HIGH", "CRITICAL", "URGENT"):
        target = "HIGH"
    elif cleaned in ("MEDIUM", "NORMAL", "MODERATE"):
        target = "MEDIUM"
    elif cleaned in ("LOW", "MINOR", "TRIVIAL"):
        target = "LOW"
    else:
        target = "LOW"

    return target.capitalize() if title_case else target


def validate_priority_checkpoint_directory(
    directory: Union[str, Path]
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validate that a candidate model checkpoint adheres to the priority contract.
    
    Checks:
    1. Directory exists.
    2. config.json is present and parseable.
    3. At least one model weights file is present.
    4. At least one tokenizer artifact is present.
    5. id2label contains validated priority labels (HIGH, MEDIUM, LOW).
    
    Returns:
        Tuple[bool, List[str], Dict[str, Any]]: (is_valid, error_list, metadata)
    """
    path = Path(directory)
    errors: List[str] = []
    metadata: Dict[str, Any] = {}

    if not path.is_dir():
        return False, [f"Checkpoint path does not exist or is not a directory: {path}"], {}

    files = {p.name for p in path.iterdir() if p.is_file()}

    # 1. Required config
    if "config.json" not in files:
        errors.append(f"Missing required file: config.json in {path}")
    else:
        try:
            with open(path / "config.json", "r", encoding="utf-8") as f:
                cfg = json.load(f)
            metadata["config"] = cfg
            id2label = cfg.get("id2label", {})
            metadata["id2label"] = id2label
            labels = {str(v).strip().upper() for v in id2label.values()}
            metadata["labels"] = list(labels)

            if not labels:
                errors.append("config.json does not define id2label mapping.")
            elif not (ALLOWED_PRIORITY_LABELS.issubset(labels) or labels == ALLOWED_PRIORITY_LABELS or MINIMAL_PRIORITY_LABELS.issubset(labels)):
                errors.append(
                    f"id2label labels {sorted(list(labels))} do not match allowed priority labels "
                    f"{sorted(list(ALLOWED_PRIORITY_LABELS))}."
                )
        except Exception as e:
            errors.append(f"Failed to parse config.json: {e}")

    # 2. Weights file
    has_weights = any(w in files for w in WEIGHT_FILE_CANDIDATES)
    if not has_weights:
        errors.append(
            f"No weights file found in {path}. Expected one of {WEIGHT_FILE_CANDIDATES}"
        )

    # 3. Tokenizer file
    has_tokenizer = any(t in files for t in TOKENIZER_FILE_CANDIDATES)
    if not has_tokenizer:
        errors.append(
            f"No tokenizer file found in {path}. Expected one of {TOKENIZER_FILE_CANDIDATES}"
        )

    is_valid = len(errors) == 0
    return is_valid, errors, metadata
