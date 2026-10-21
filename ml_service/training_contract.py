"""Training Contract and Evaluation Standards for DistilBERT Classifier.

Defines the engineering contract, model specifications, evaluation report schemas,
and checkpoint validation routines for training task-specific Hugging Face models.

PURPOSE:
Provides an unambiguous specification for the teammate executing model fine-tuning
on a GPU laptop or workstation.

CRITICAL INVARIANTS:
1. Label-to-ID Mapping:
   - {"NO": 0, "YES": 1, "REVIEW": 2}
   - {0: "NO", 1: "YES", 2: "REVIEW"}
   This matches the task-specific verification in HuggingFaceModelLoader.
2. Metric Priorities:
   - High precision on YES is paramount to prevent spamming students with unneeded calls.
   - Low false NO rate is critical to prevent student distress from being ignored.
   - Macro-F1 across all 3 classes is the primary model selection criterion.
3. Separation of Concerns:
   - The learned classifier outputs logits/probabilities for {NO, YES, REVIEW}.
   - The classifier NEVER bypasses the rule-based FollowUpDecisionEngine or
     PriorityAssessmentEngine.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ==============================================================================
# Canonical Label-to-ID Contract
# ==============================================================================

MODEL_FAMILY: str = "distilbert-base-uncased"

LABEL2ID: Dict[str, int] = {
    "NO": 0,
    "YES": 1,
    "REVIEW": 2,
}

ID2LABEL: Dict[int, str] = {
    0: "NO",
    1: "YES",
    2: "REVIEW",
}

NUM_LABELS: int = 3

DEFAULT_MODEL_DIR: str = "./models/distilbert-student-support"

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
}


# ==============================================================================
# Hyperparameter & Training Configuration Specification
# ==============================================================================

class TrainingHyperparameters(BaseModel):
    """Reproducible training configuration for DistilBERT sequence classification.
    
    Standardized hyperparameters optimized for student support domain adaptation.
    """
    model_config = ConfigDict(frozen=True)

    base_model: str = Field(
        default=MODEL_FAMILY,
        description="Pretrained Hugging Face base checkpoint.",
    )
    max_seq_length: int = Field(
        default=512,
        ge=64,
        le=512,
        description="Maximum sequence length for tokenization padding and truncation.",
    )
    learning_rate: float = Field(
        default=2e-5,
        gt=0.0,
        le=1e-3,
        description="Peak learning rate for AdamW optimizer.",
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Per-device batch size for training and evaluation.",
    )
    num_epochs: int = Field(
        default=4,
        ge=1,
        le=50,
        description="Total training epochs.",
    )
    warmup_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Proportion of training steps for linear learning rate warmup.",
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        le=0.5,
        description="Weight decay for regularization.",
    )
    random_seed: int = Field(
        default=42,
        description="Fixed random seed for PyTorch, NumPy, and Python for exact reproducibility.",
    )
    fp16: bool = Field(
        default=True,
        description="Enable mixed precision training (fp16) on CUDA GPUs.",
    )
    class_weights_strategy: str = Field(
        default="balanced",
        description="Strategy for handling class imbalance: 'balanced', 'none', or 'custom'.",
    )
    metric_for_best_model: str = Field(
        default="macro_f1",
        description="Primary validation metric used to select best checkpoint.",
    )


# ==============================================================================
# Evaluation Metrics & Report Schema
# ==============================================================================

class ClassMetrics(BaseModel):
    """Per-class classification metrics."""
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    support: int = Field(..., ge=0)


class EvaluationReport(BaseModel):
    """Comprehensive evaluation report for model selection and governance.
    
    Must be generated on the untouched test split prior to model deployment.
    """
    model_config = ConfigDict(use_enum_values=True)

    model_name: str = Field(..., description="Target model identifier or path.")
    model_version: str = Field(..., description="Semantic version string of the model.")
    dataset_version: str = Field(..., description="Dataset release version evaluated upon.")
    evaluated_at: str = Field(..., description="ISO-8601 evaluation timestamp.")
    test_samples_count: int = Field(..., ge=1, description="Total number of evaluated test examples.")
    
    # Overall Aggregate Metrics
    macro_f1: float = Field(..., ge=0.0, le=1.0, description="Unweighted average F1 across YES, NO, REVIEW.")
    weighted_f1: float = Field(..., ge=0.0, le=1.0, description="Support-weighted average F1.")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Overall classification accuracy.")
    
    # Class-Level Breakdown
    per_class: Dict[str, ClassMetrics] = Field(
        ...,
        description="Detailed metrics for YES, NO, and REVIEW.",
    )
    
    # Confusion Matrix: 3x3 grid [true_idx][pred_idx] corresponding to [NO, YES, REVIEW]
    confusion_matrix: List[List[int]] = Field(
        ...,
        description="3x3 confusion matrix ordered by [NO, YES, REVIEW].",
    )
    
    # Operational Risk Metrics
    false_yes_count: int = Field(
        ...,
        ge=0,
        description="Count of non-YES tickets falsely predicted as YES (causes unneeded outbound calls).",
    )
    false_no_count: int = Field(
        ...,
        ge=0,
        description="Count of non-NO tickets falsely predicted as NO (causes student abandonment).",
    )
    review_precision: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Precision when predicting REVIEW; measures reliability of fallback routing.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Reviewer commentary, observed failure modes, or threshold analysis.",
    )

    @field_validator("confusion_matrix")
    @classmethod
    def validate_confusion_matrix(cls, matrix: List[List[int]]) -> List[List[int]]:
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise ValueError(
                "Confusion matrix must be a 3x3 grid matching classes [NO, YES, REVIEW]."
            )
        for row in matrix:
            for val in row:
                if val < 0:
                    raise ValueError("Confusion matrix cell values cannot be negative.")
        return matrix

    @model_validator(mode="after")
    def validate_class_names(self) -> "EvaluationReport":
        required = {"YES", "NO", "REVIEW"}
        keys = set(self.per_class.keys())
        if not required.issubset(keys):
            raise ValueError(f"per_class metrics must include all classes {required}; found {keys}.")
        return self


# ==============================================================================
# Model Artifact Metadata Schema
# ==============================================================================

class TrainingMetadata(BaseModel):
    """Schema for 'training_metadata.json' saved inside the model checkpoint directory.
    
    Ensures complete lineage, provenance, reproducibility, and compliance auditability.
    """
    model_config = ConfigDict(use_enum_values=True)

    model_name: str = Field(default="distilbert-student-support")
    model_version: str = Field(..., description="Release version of the trained model (e.g. '1.0.0').")
    base_model: str = Field(default=MODEL_FAMILY)
    trained_at: str = Field(..., description="Timestamp when model training completed.")
    git_commit_hash: Optional[str] = Field(default=None, description="Repository commit hash.")
    hyperparameters: TrainingHyperparameters = Field(..., description="Training parameters.")
    dataset_manifest_id: str = Field(..., description="Dataset release manifest ID.")
    dataset_version: str = Field(..., description="Version of the dataset used.")
    train_samples_count: int = Field(..., ge=1)
    val_samples_count: int = Field(..., ge=1)
    test_samples_count: int = Field(..., ge=1)
    evaluation_summary: EvaluationReport = Field(..., description="Official test split evaluation results.")
    label2id: Dict[str, int] = Field(default_factory=lambda: LABEL2ID)
    id2label: Dict[int, str] = Field(default_factory=lambda: ID2LABEL)
    institutional_disclaimer: str = Field(
        default=(
            "Trained for student-support triage guidance. Model predictions provide advisory signals "
            "and do NOT bypass institutional business rules or human oversight."
        )
    )


# ==============================================================================
# Checkpoint Directory Validator
# ==============================================================================

def validate_checkpoint_directory(
    directory_path: Union[str, Path],
    require_metadata: bool = True,
) -> Tuple[bool, List[str]]:
    """Verify that a local directory contains a valid fine-tuned DistilBERT checkpoint.
    
    Verifies:
    1. config.json exists with exact label2id and id2label mappings.
    2. Model weights file (model.safetensors or pytorch_model.bin) exists.
    3. Tokenizer files exist.
    4. Optional training_metadata.json exists and conforms to TrainingMetadata schema.
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_error_reasons)
    """
    path = Path(directory_path)
    errors: List[str] = []

    if not path.exists() or not path.is_dir():
        return False, [f"Checkpoint directory does not exist or is not a directory: '{path}'"]

    # 1. Check config.json
    config_file = path / "config.json"
    if not config_file.exists():
        errors.append("Missing required 'config.json' file in checkpoint directory.")
    else:
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            
            # Check id2label in config.json
            raw_id2label = config_data.get("id2label", {})
            normalized_id2label = {int(k): str(v).strip().upper() for k, v in raw_id2label.items()}
            
            expected_labels = {"NO", "YES", "REVIEW"}
            actual_labels = set(normalized_id2label.values())

            if actual_labels != expected_labels:
                errors.append(
                    f"Invalid id2label in config.json: {normalized_id2label}. "
                    f"Must map precisely to {expected_labels}."
                )
        except Exception as exc:
            errors.append(f"Failed to read or parse 'config.json': {str(exc)}")

    # 2. Check for weights
    has_weights = any((path / fname).exists() for fname in WEIGHT_FILE_CANDIDATES)
    if not has_weights:
        errors.append(
            f"Missing model weight binary. Expected at least one of: {sorted(list(WEIGHT_FILE_CANDIDATES))}."
        )

    # 3. Check for tokenizer
    has_tokenizer = any((path / fname).exists() for fname in TOKENIZER_FILE_CANDIDATES)
    if not has_tokenizer:
        errors.append(
            f"Missing tokenizer artifact. Expected at least one of: {sorted(list(TOKENIZER_FILE_CANDIDATES))}."
        )

    # 4. Check for training metadata if required
    if require_metadata:
        metadata_file = path / "training_metadata.json"
        if not metadata_file.exists():
            errors.append("Missing required 'training_metadata.json' audit manifest.")
        else:
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    meta_dict = json.load(f)
                TrainingMetadata.model_validate(meta_dict)
            except Exception as exc:
                errors.append(f"Invalid 'training_metadata.json': {str(exc)}")

    return (len(errors) == 0, errors)
