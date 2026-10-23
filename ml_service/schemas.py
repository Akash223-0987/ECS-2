"""Input and Output Schemas for Student Support ML Module.

Defines Pydantic models for type safety, validation, and serialization.

IMPORTANT ETHICAL & ARCHITECTURAL CONSTRAINTS:
1. PII Exclusion: Student identifiers (`student_name`, `application_number`,
   `registration_number`) and contact info (`phone_number`) are included in the
   input schema for operational record-keeping only. They MUST NEVER be used as
   predictive features by any ML model to avoid bias and privacy leakage.
2. Legacy Label Exclusion: The existing `priority_level` field in incoming tickets
   is statically recorded as "High" by upstream conversational flows. It MUST NOT
   be used as a training label or ground truth.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ==============================================================================
# Enumerations for Structured Output
# ==============================================================================

class FollowUpRequired(str, Enum):
    """Indicates whether human follow-up or advisor intervention is needed."""
    YES = "YES"
    NO = "NO"
    REVIEW = "REVIEW"


class InterestLevel(str, Enum):
    """Assessed student interest level in academic/hostel/admission programs."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class UrgencyLevel(str, Enum):
    """Assessed urgency level based on ticket content and deadlines."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class PriorityCategory(str, Enum):
    """Categorical triage priority assignment."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# Alias for simplified naming per GitHub Issue #7
PriorityLevel = PriorityCategory


class ModelStatus(str, Enum):
    """Operational status of the ML prediction."""
    PLACEHOLDER_PENDING_MODEL = "PLACEHOLDER_PENDING_MODEL"
    EMPTY_DESCRIPTION_REVIEW = "EMPTY_DESCRIPTION_REVIEW"
    INVALID_TIMESTAMP_REVIEW = "INVALID_TIMESTAMP_REVIEW"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_INITIALIZED = "NOT_INITIALIZED"
    DEPENDENCIES_UNAVAILABLE = "DEPENDENCIES_UNAVAILABLE"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    GENERIC_MODEL_NOT_FINE_TUNED = "GENERIC_MODEL_NOT_FINE_TUNED"
    INCOMPATIBLE_LABELS = "INCOMPATIBLE_LABELS"
    FALLBACK_HEURISTIC = "FALLBACK_HEURISTIC"
    MOCK_MODEL = "MOCK_MODEL"
    EMPTY_TRANSCRIPTION = "EMPTY_TRANSCRIPTION"
    READY = "READY"


# ==============================================================================
# Input Schema
# ==============================================================================

class TicketInput(BaseModel):
    """Schema for incoming student support tickets.
    
    Fields match the upstream LiveKit / Google Sheets / n8n schema.
    Optional fields are safely defaulted to None so missing values do not crash
    the ingestion pipeline.
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    ticket_id: Optional[str] = Field(
        default=None,
        description="Unique ticket identifier generated during student session (e.g., 'John1').",
    )
    date_and_time: Optional[Union[datetime, str]] = Field(
        default=None,
        description="Timestamp string or datetime of ticket creation.",
    )
    student_name: Optional[str] = Field(
        default=None,
        description="Student full name. PII - MUST NOT be used as a predictive feature.",
    )
    application_number: Optional[str] = Field(
        default=None,
        description="Application or registration number. PII - MUST NOT be used as a predictive feature.",
    )
    registration_number: Optional[str] = Field(
        default=None,
        description="Alias for application_number frequently used in university records.",
    )
    phone_number: Optional[str] = Field(
        default=None,
        description="Student contact phone number. PII - MUST NOT be used as a predictive feature.",
    )
    issue_description: Optional[str] = Field(
        default=None,
        description="Primary textual narrative of student's query. Main NLP feature.",
    )
    priority_level: Optional[str] = Field(
        default=None,
        description="Legacy hardcoded priority ('High'). MUST NOT be used as training ground truth.",
    )
    status: Optional[str] = Field(
        default=None,
        description="Current ticket lifecycle status (e.g., 'Open', 'Resolved').",
    )

    @model_validator(mode="before")
    @classmethod
    def sync_registration_and_application_number(cls, data: Any) -> Any:
        """Allow registration_number as a drop-in alias for application_number."""
        if isinstance(data, dict):
            app_no = data.get("application_number")
            reg_no = data.get("registration_number")
            if not app_no and reg_no:
                data["application_number"] = reg_no
            elif not reg_no and app_no:
                data["registration_number"] = app_no
        return data


# ==============================================================================
# Structured Output Schema
# ==============================================================================

class TicketPrediction(BaseModel):
    """Structured output schema returned by the Student Support ML Module.
    
    Guarantees consistent contract across all triage predictions.
    """
    model_config = ConfigDict(use_enum_values=True)

    ticket_id: Optional[str] = Field(
        default=None,
        description="Identifier of the ticket corresponding to this prediction.",
    )
    follow_up_required: FollowUpRequired = Field(
        default=FollowUpRequired.REVIEW,
        description="Whether advisor follow-up is necessary: YES, NO, or REVIEW.",
    )
    interest_level: InterestLevel = Field(
        default=InterestLevel.UNKNOWN,
        description="Assessed student interest level: HIGH, MEDIUM, LOW, or UNKNOWN.",
    )
    urgency: UrgencyLevel = Field(
        default=UrgencyLevel.UNKNOWN,
        description="Assessed issue urgency: HIGH, MEDIUM, LOW, or UNKNOWN.",
    )
    follow_up_reason: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of why follow-up was flagged or reviewed.",
    )
    priority_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Defensible numerical priority score [0.0 - 100.0], or null when undefensible.",
    )
    priority: PriorityCategory = Field(
        default=PriorityCategory.NOT_APPLICABLE,
        description="Categorical priority assignment: HIGH, MEDIUM, LOW, or NOT_APPLICABLE.",
    )
    recommended_action: str = Field(
        ...,
        description="Concrete recommended next operational action for support staff.",
    )
    model_status: str = Field(
        ...,
        description="Diagnostic state of model processing (e.g., 'PLACEHOLDER_PENDING_MODEL').",
    )
    predicted_intent: Optional[str] = Field(
        default=None,
        description="Auxiliary intent classification (e.g., 'payment_issue', 'get_refund').",
    )
    intent_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Uncalibrated confidence score for auxiliary intent.",
    )


# ==============================================================================
# Simplified Priority Schemas (GitHub Issue #7)
# ==============================================================================

class PriorityInferenceInput(BaseModel):
    """Minimal schema for priority inference input.
    
    Accepts issue_transcription (primary) and supports issue_description as an alias.
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    issue_transcription: str = Field(
        ...,
        alias="issue_description",
        description="Primary textual narrative or transcription of student inquiry.",
    )
    ticket_id: Optional[str] = Field(
        default=None,
        description="Optional ticket identifier for correlation and traceability.",
    )

    @field_validator("issue_transcription", mode="before")
    @classmethod
    def coerce_to_string(cls, v: Any) -> str:
        """Coerce null or numeric transcriptions cleanly to string."""
        if v is None:
            return ""
        return str(v).strip()


class PriorityInferenceOutput(BaseModel):
    """Minimal schema for priority inference output.
    
    Returns priority_level (HIGH, MEDIUM, LOW), with explicit indication of whether
    a fine-tuned model or test-double/fallback produced the output.
    """
    model_config = ConfigDict(extra="ignore")

    priority_level: str = Field(
        ...,
        description="Predicted priority level (HIGH, MEDIUM, LOW).",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Inference confidence score [0.0 - 1.0] if model outputs probabilities.",
    )
    model_status: str = Field(
        default="READY",
        description="Operational model status (READY, PENDING_TRAINED_MODEL, etc.).",
    )
    is_trained_model: bool = Field(
        default=False,
        description="True if prediction was generated by a validated fine-tuned ML model, False if fallback/mock.",
    )
    status_reason: Optional[str] = Field(
        default=None,
        description="Diagnostic explanation of model or fallback state.",
    )

