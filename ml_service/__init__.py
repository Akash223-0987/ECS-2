"""Standalone Machine Learning Module for Student Support Ticket Triage.

Provides schema validation, PII safety guardrails, configuration, and a minimal
prediction interface for ticket classification.
"""

from .config import MLServiceConfig
from .decision_engine import (
    FollowUpDecision,
    FollowUpDecisionEngine,
)
from .model_loader import (
    HuggingFaceModelLoader,
    check_dependencies,
)
from .predictor import (
    BasePredictor,
    PlaceholderPredictor,
    PriorityPredictor,
    StudentSupportPredictor,
)
from .priority_model import (
    PriorityModelLoader,
)
from .dataset_schema import (
    AdjudicationStatus,
    DatasetManifest,
    LabelProvenance,
    SplitType,
    TrainingExample,
    detect_potential_pii,
    sanitize_text_for_training,
)
from .priority_engine import (
    PriorityAssessment,
    PriorityAssessmentEngine,
    ScoreBreakdown,
)
from .schemas import (
    FollowUpRequired,
    InterestLevel,
    ModelStatus,
    PriorityCategory,
    PriorityInferenceInput,
    PriorityInferenceOutput,
    PriorityLevel,
    TicketInput,
    TicketPrediction,
    UrgencyLevel,
)
from .training_contract import (
    ID2LABEL,
    LABEL2ID,
    ClassMetrics,
    EvaluationReport,
    TrainingHyperparameters,
    TrainingMetadata,
    validate_checkpoint_directory,
)
from .intent_classifier import (
    AuxiliaryIntentClassifier,
    IntentModelStatus,
    IntentPrediction,
)
from .sheets_config import SheetsWatcherConfig
from .sheets_watcher import (
    GoogleSheetsClientInterface,
    GoogleSheetsWatcher,
    GSpreadSheetsClient,
    MockGoogleSheetsClient,
    RowUpdateReport,
)
from .validator import (
    TicketValidator,
    ValidationResult,
    validate_ticket_input,
)

__version__ = "0.1.0"

__all__ = [
    "MLServiceConfig",
    "TicketInput",
    "TicketPrediction",
    "FollowUpRequired",
    "InterestLevel",
    "UrgencyLevel",
    "PriorityCategory",
    "ModelStatus",
    "TicketValidator",
    "ValidationResult",
    "validate_ticket_input",
    "BasePredictor",
    "PlaceholderPredictor",
    "StudentSupportPredictor",
    "PriorityPredictor",
    "PriorityModelLoader",
    "HuggingFaceModelLoader",
    "check_dependencies",
    "FollowUpDecision",
    "FollowUpDecisionEngine",
    "PriorityAssessment",
    "PriorityAssessmentEngine",
    "ScoreBreakdown",
    "PriorityLevel",
    "PriorityInferenceInput",
    "PriorityInferenceOutput",
    "LabelProvenance",
    "AdjudicationStatus",
    "SplitType",
    "TrainingExample",
    "DatasetManifest",
    "detect_potential_pii",
    "sanitize_text_for_training",
    "LABEL2ID",
    "ID2LABEL",
    "TrainingHyperparameters",
    "ClassMetrics",
    "EvaluationReport",
    "TrainingMetadata",
    "validate_checkpoint_directory",
    "AuxiliaryIntentClassifier",
    "IntentModelStatus",
    "IntentPrediction",
    "SheetsWatcherConfig",
    "GoogleSheetsClientInterface",
    "GoogleSheetsWatcher",
    "GSpreadSheetsClient",
    "MockGoogleSheetsClient",
    "RowUpdateReport",
]
