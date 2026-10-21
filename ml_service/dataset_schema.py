"""Dataset Schema and Validation for Student Support ML Training.

Defines versioned schemas, data structures, and validation rules for training
examples and dataset releases for the DistilBERT sequence classifier.

CRITICAL ETHICAL, ARCHITECTURAL, & PRIVACY CONSTRAINTS:
1. PII Exclusion: Student names, phone numbers, application numbers, registration
   numbers, and ticket IDs MUST NEVER be included in the model input text.
2. Human-Reviewed Ground Truth: Training labels MUST be assigned by authorized human
   reviewers using ticket context and institutional policy. Automatically generated
   heuristic labels from rule engines or unverified model predictions are strictly
   PROHIBITED from serving as training ground truth.
3. Separation of Notes & Metadata: Reviewer notes and operational metadata are stored
   strictly separately from model input text.
4. Split Leakage Resistance: Group identifiers (anonymized student/case hashes)
   ensure related tickets from the same case never cross train/validation/test splits.
"""

from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .schemas import FollowUpRequired


# ==============================================================================
# Enumerations for Dataset Provenance, Adjudication, and Splits
# ==============================================================================

class LabelProvenance(str, Enum):
    """Auditable provenance of how the training ground-truth label was established.
    
    Training ground truth must always stem from qualified human review or historical
    audits, never unverified heuristic rule generation.
    """
    HUMAN_SINGLE_REVIEW = "HUMAN_SINGLE_REVIEW"
    HUMAN_CONSENSUS = "HUMAN_CONSENSUS"
    ADJUDICATED_EXPERT = "ADJUDICATED_EXPERT"
    HISTORICAL_AUDIT = "HISTORICAL_AUDIT"


class AdjudicationStatus(str, Enum):
    """Lifecycle status of the annotation record."""
    APPROVED = "APPROVED"
    PENDING_REVIEW = "PENDING_REVIEW"
    FLAGGED_CONFLICT = "FLAGGED_CONFLICT"
    REJECTED = "REJECTED"
    EXCLUDED = "EXCLUDED"


class SplitType(str, Enum):
    """Dataset partition assignment."""
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


# ==============================================================================
# PII Detection and Sanitization Helpers
# ==============================================================================

PHONE_REGEX = re.compile(
    r"(?:\+?91[\-\s]?)?[6-9]\d{9}\b|"  # Indian 10-digit mobile numbers
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",  # Standard 10-digit
    re.IGNORECASE,
)

EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    re.IGNORECASE,
)

APPLICATION_NO_REGEX = re.compile(
    r"\b(?:20\d{2}[A-Z]{3,6}\d{4,5}|APP\d{5,8}|REG\d{5,8}|VDP\d{5,8})\b",
    re.IGNORECASE,
)

RAW_TICKET_ID_REGEX = re.compile(
    r"\b(?:TICK(?:ET)?[-_]?\d+|[A-Za-z]+\d{2,})\b",
    re.IGNORECASE,
)


def detect_potential_pii(text: str) -> List[str]:
    """Inspect text for potential personal identifiable information (PII).
    
    Returns:
        List of identified PII categories found in the text.
    """
    findings: List[str] = []
    if PHONE_REGEX.search(text):
        findings.append("PHONE_NUMBER")
    if EMAIL_REGEX.search(text):
        findings.append("EMAIL_ADDRESS")
    if APPLICATION_NO_REGEX.search(text):
        findings.append("APPLICATION_OR_REG_NUMBER")
    return findings


def sanitize_text_for_training(
    raw_text: str,
    student_name: Optional[str] = None,
    phone_number: Optional[str] = None,
    application_number: Optional[str] = None,
) -> str:
    """Sanitize ticket text by scrubbing detected and provided PII.
    
    Replaces personal identifiers with standardized placeholder tokens.
    """
    if not raw_text:
        return ""

    sanitized = raw_text

    # 1. Scrub pattern-matched emails first to protect email addresses from name collision
    sanitized = EMAIL_REGEX.sub("[EMAIL]", sanitized)

    # 2. Scrub explicit phone number if provided
    if phone_number and phone_number.strip():
        sanitized = re.sub(re.escape(phone_number.strip()), "[PHONE_NUMBER]", sanitized)

    # 3. Scrub explicit application number if provided
    if application_number and application_number.strip():
        sanitized = re.sub(re.escape(application_number.strip()), "[APPLICATION_NUMBER]", sanitized)

    # 4. Scrub pattern-matched phone numbers and application numbers
    sanitized = PHONE_REGEX.sub("[PHONE_NUMBER]", sanitized)
    sanitized = APPLICATION_NO_REGEX.sub("[APPLICATION_NUMBER]", sanitized)

    # 5. Scrub explicit student name tokens using word boundaries
    if student_name and student_name.strip():
        name_tokens = [t for t in re.split(r"\s+", student_name.strip()) if len(t) > 2]
        for token in name_tokens:
            sanitized = re.sub(r"\b" + re.escape(token) + r"\b", "[STUDENT_NAME]", sanitized, flags=re.IGNORECASE)

    # Normalize whitespace
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


# ==============================================================================
# Training Example Schema
# ==============================================================================

class TrainingExample(BaseModel):
    """Schema for a single curated, human-reviewed training example.
    
    Adheres to strict privacy, provenance, and data separation standards.
    """
    model_config = ConfigDict(use_enum_values=True)

    example_id: str = Field(
        ...,
        description="Stable unique identifier for the example (e.g., 'ex_2026_0001').",
    )
    text: str = Field(
        ...,
        description="Sanitized ticket narrative used as model input. Strictly devoid of PII and notes.",
    )
    label: FollowUpRequired = Field(
        ...,
        description="Human-reviewed ground truth label: YES, NO, or REVIEW.",
    )
    label_provenance: LabelProvenance = Field(
        ...,
        description="Auditable human review provenance. Heuristic auto-labels are disallowed.",
    )
    adjudication_status: AdjudicationStatus = Field(
        default=AdjudicationStatus.APPROVED,
        description="Review status (e.g., APPROVED, FLAGGED_CONFLICT, PENDING_REVIEW, EXCLUDED).",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Reviewer rationale or annotations. Stored strictly separately from model text.",
    )
    group_id: Optional[str] = Field(
        default=None,
        description="Anonymized student or case grouping key to prevent data leakage across splits.",
    )
    split: Optional[SplitType] = Field(
        default=None,
        description="Assigned data split: TRAIN, VALIDATION, or TEST.",
    )
    created_at: Optional[Union[datetime, str]] = Field(
        default=None,
        description="Creation timestamp of ticket or annotation for temporal split analysis.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary operational metadata (e.g., reviewer_id, source_ticket_id) isolated from text.",
    )

    @field_validator("example_id")
    @classmethod
    def validate_example_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("example_id must not be empty.")
        return v.strip()

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Model input text must not be empty or whitespace.")
        cleaned = v.strip()
        # Enforce basic PII check
        pii_found = detect_potential_pii(cleaned)
        if pii_found:
            raise ValueError(
                f"Model input text contains potential unredacted PII: {pii_found}. "
                "Scrub all identifiers before creating training examples."
            )
        return cleaned

    @model_validator(mode="after")
    def validate_separation_and_provenance(self) -> "TrainingExample":
        # 1. Ensure notes are not accidentally concatenated into model input text
        if self.notes and self.notes.strip():
            if self.notes.strip() in self.text:
                raise ValueError(
                    "Reviewer notes must not be leaked into the model input text. "
                    "Keep notes strictly in the 'notes' field."
                )

        # 2. Reject non-human or unverified provenance
        valid_provenance = {
            LabelProvenance.HUMAN_SINGLE_REVIEW,
            LabelProvenance.HUMAN_CONSENSUS,
            LabelProvenance.ADJUDICATED_EXPERT,
            LabelProvenance.HISTORICAL_AUDIT,
            LabelProvenance.HUMAN_SINGLE_REVIEW.value,
            LabelProvenance.HUMAN_CONSENSUS.value,
            LabelProvenance.ADJUDICATED_EXPERT.value,
            LabelProvenance.HISTORICAL_AUDIT.value,
        }
        if self.label_provenance not in valid_provenance:
            raise ValueError(
                f"Invalid label provenance '{self.label_provenance}'. "
                "Training ground truth MUST be verified by qualified human review or historical audit."
            )

        return self


# ==============================================================================
# Dataset Manifest Schema & Split Management
# ==============================================================================

class DatasetManifest(BaseModel):
    """Versioned container and manifest for training dataset releases.
    
    Provides leakage-resistant partitioning, distribution metrics, and serialization.
    """
    model_config = ConfigDict(use_enum_values=True)

    dataset_id: str = Field(
        ...,
        description="Unique dataset release identifier (e.g., 'vit_support_dataset_v1.0.0').",
    )
    schema_version: str = Field(
        default="1.0.0",
        description="Dataset manifest schema specification version.",
    )
    name: str = Field(
        ...,
        description="Descriptive title of the dataset collection.",
    )
    description: str = Field(
        ...,
        description="Detailed scope, institutional context, and provenance description.",
    )
    created_at: Union[datetime, str] = Field(
        ...,
        description="Dataset creation timestamp.",
    )
    split_strategy: str = Field(
        default="group_based",
        description="Splitting methodology: 'group_based', 'time_based', or 'stratified_group'.",
    )
    examples: List[TrainingExample] = Field(
        default_factory=list,
        description="List of validated training examples.",
    )

    def get_split(self, split: Union[SplitType, str]) -> List[TrainingExample]:
        """Filter examples belonging to a designated split."""
        target = split.value if isinstance(split, SplitType) else str(split).upper()
        return [ex for ex in self.examples if ex.split == target]

    def get_label_distribution(
        self, split: Optional[Union[SplitType, str]] = None
    ) -> Dict[str, int]:
        """Compute label counts across all examples or a specific split."""
        target_examples = self.get_split(split) if split else self.examples
        counts: Dict[str, int] = {
            FollowUpRequired.YES.value: 0,
            FollowUpRequired.NO.value: 0,
            FollowUpRequired.REVIEW.value: 0,
        }
        for ex in target_examples:
            lbl = ex.label.value if isinstance(ex.label, FollowUpRequired) else str(ex.label)
            if lbl in counts:
                counts[lbl] += 1
            else:
                counts[lbl] = 1
        return counts

    def validate_split_leakage(self) -> Tuple[bool, List[str]]:
        """Verify that no grouping identifier (case/student) spans multiple splits.
        
        Returns:
            Tuple[bool, List[str]]: (is_leakage_free, list_of_violation_reasons)
        """
        violations: List[str] = []
        group_split_map: Dict[str, Set[str]] = {}
        text_split_map: Dict[str, Set[str]] = {}

        for ex in self.examples:
            if not ex.split:
                continue
            split_val = ex.split.value if isinstance(ex.split, SplitType) else str(ex.split)

            # Check group_id leakage
            if ex.group_id:
                group_split_map.setdefault(ex.group_id, set()).add(split_val)

            # Check exact duplicate text leakage across splits
            norm_text = re.sub(r"\s+", " ", ex.text.strip().lower())
            text_split_map.setdefault(norm_text, set()).add(split_val)

        for gid, splits in group_split_map.items():
            if len(splits) > 1:
                violations.append(
                    f"Group ID leakage: '{gid}' is present in multiple splits: {sorted(list(splits))}."
                )

        for txt, splits in text_split_map.items():
            if len(splits) > 1:
                snippet = txt[:40] + "..." if len(txt) > 40 else txt
                violations.append(
                    f"Exact text duplicate leakage across splits: '{snippet}' present in: {sorted(list(splits))}."
                )

        return (len(violations) == 0, violations)

    def to_jsonl(self, filepath: Union[str, Path]) -> int:
        """Export examples to a JSON Lines (JSONL) file for training tools.
        
        Returns:
            int: Number of examples exported.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for ex in self.examples:
                f.write(ex.model_dump_json() + "\n")
                count += 1
        return count

    @classmethod
    def from_jsonl(
        cls,
        filepath: Union[str, Path],
        dataset_id: str,
        name: str,
        description: str,
        split_strategy: str = "group_based",
    ) -> "DatasetManifest":
        """Load examples from a JSON Lines (JSONL) file into a DatasetManifest."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"JSONL dataset file not found: {filepath}")

        examples: List[TrainingExample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    data = json.loads(clean_line)
                    examples.append(TrainingExample.model_validate(data))
                except Exception as exc:
                    raise ValueError(f"Error parsing JSONL line {line_no}: {str(exc)}") from exc

        return cls(
            dataset_id=dataset_id,
            name=name,
            description=description,
            created_at=datetime.utcnow().isoformat(),
            split_strategy=split_strategy,
            examples=examples,
        )

    def to_hf_dict(self, split: Optional[Union[SplitType, str]] = None) -> Dict[str, List[Any]]:
        """Format dataset examples into a dictionary ready for Hugging Face datasets.
        
        Maps labels to canonical task integers: {"NO": 0, "YES": 1, "REVIEW": 2}.
        """
        label2id = {"NO": 0, "YES": 1, "REVIEW": 2}
        target_examples = self.get_split(split) if split else self.examples

        texts: List[str] = []
        labels: List[str] = []
        label_ids: List[int] = []
        example_ids: List[str] = []

        for ex in target_examples:
            lbl_str = ex.label.value if isinstance(ex.label, FollowUpRequired) else str(ex.label)
            texts.append(ex.text)
            labels.append(lbl_str)
            label_ids.append(label2id.get(lbl_str, 2))
            example_ids.append(ex.example_id)

        return {
            "example_id": example_ids,
            "text": texts,
            "label": labels,
            "label_id": label_ids,
        }
