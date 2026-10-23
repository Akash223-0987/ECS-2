"""Prediction interface for Student Support ML Module.

Provides a minimal, safe prediction contract that returns clearly identified
placeholder or REVIEW results.

CRITICAL CONSTRAINTS:
- No pretrained models or weights are loaded.
- No priority scoring rules are implemented yet; `priority_score` remains null
  whenever defensible calculation is not possible.
- Predictions are NEVER invented or hallucinated.
- Student identifiers (name, phone, registration) and legacy `priority_level`
  are excluded from inference.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from .config import MLServiceConfig
from .schemas import (
    FollowUpRequired,
    InterestLevel,
    ModelStatus,
    PriorityCategory,
    PriorityInferenceInput,
    PriorityInferenceOutput,
    TicketInput,
    TicketPrediction,
    UrgencyLevel,
)
from .validator import TicketValidator, ValidationResult, validate_ticket_input


class BasePredictor(ABC):
    """Abstract base class for student support prediction interfaces."""

    @abstractmethod
    def predict(self, ticket: Union[TicketInput, Dict[str, Any], Any]) -> TicketPrediction:
        """Process a single ticket and return a structured prediction."""
        raise NotImplementedError

    @abstractmethod
    def predict_batch(
        self, tickets: List[Union[TicketInput, Dict[str, Any], Any]]
    ) -> List[TicketPrediction]:
        """Process multiple tickets in batch."""
        raise NotImplementedError


class StudentSupportPredictor(BasePredictor):
    """Minimal foundation predictor for Student Support ticket triage.
    
    In this foundation stage, returns defensible placeholder/REVIEW results
    identifying that model training and weights deployment are pending.
    """

    def __init__(
        self,
        config: Optional[MLServiceConfig] = None,
        model_loader: Optional[Any] = None,
        decision_engine: Optional[Any] = None,
        priority_engine: Optional[Any] = None,
        intent_classifier: Optional[Any] = None,
    ) -> None:
        self.config = config or MLServiceConfig()
        self.model_loader = model_loader
        self.decision_engine = decision_engine
        self.priority_engine = priority_engine
        self.intent_classifier = intent_classifier

        # Cross-wire dependencies if provided
        if self.decision_engine is not None and self.model_loader is not None:
            if getattr(self.decision_engine, "classifier", None) is None:
                self.decision_engine.set_classifier(self.model_loader)
        if self.priority_engine is not None and self.decision_engine is not None:
            if getattr(self.priority_engine, "decision_engine", None) is None:
                self.priority_engine.decision_engine = self.decision_engine

    @classmethod
    def create_default_pipeline(
        cls,
        config: Optional[MLServiceConfig] = None,
        model_loader: Optional[Any] = None,
        classifier: Optional[Any] = None,
        intent_classifier: Optional[Any] = None,
    ) -> "StudentSupportPredictor":
        """Factory method to instantiate a fully-wired end-to-end triage predictor.
        
        Coordinates:
        - Input validation & PII sanitization
        - Model loader / classifier inference
        - Follow-up decision engine
        - Conservative priority assessment engine
        - Optional auxiliary intent classifier
        """
        from .decision_engine import FollowUpDecisionEngine
        from .priority_engine import PriorityAssessmentEngine

        cfg = config or MLServiceConfig()
        target_classifier = classifier or model_loader
        dec_engine = FollowUpDecisionEngine(classifier=target_classifier, config=cfg)
        prio_engine = PriorityAssessmentEngine(config=cfg, decision_engine=dec_engine)
        return cls(
            config=cfg,
            model_loader=model_loader,
            decision_engine=dec_engine,
            priority_engine=prio_engine,
            intent_classifier=intent_classifier,
        )

    def _attach_auxiliary_intent(
        self,
        prediction: TicketPrediction,
        ticket: Union[TicketInput, Dict[str, Any], Any],
    ) -> TicketPrediction:
        """Attach auxiliary intent prediction without modifying follow-up triage or priority.
        
        Preserves complete independence: follow_up_required, priority_score,
        priority category, urgency, and recommended actions are strictly untouched.
        """
        if self.intent_classifier is None:
            return prediction

        is_loaded = getattr(self.intent_classifier, "is_loaded", False)
        if not is_loaded:
            return prediction

        # Extract raw issue description
        raw_text = None
        if isinstance(ticket, TicketInput):
            raw_text = ticket.issue_description
        elif isinstance(ticket, dict):
            raw_text = ticket.get("issue_description")
        elif hasattr(ticket, "issue_description"):
            raw_text = getattr(ticket, "issue_description", None)

        if raw_text and str(raw_text).strip():
            try:
                res = self.intent_classifier.predict_intent(str(raw_text))
                if res and res.predicted_intent:
                    prediction.predicted_intent = res.predicted_intent
                    prediction.intent_confidence = res.intent_confidence
            except Exception:
                pass

        return prediction

    def predict(self, ticket: Union[TicketInput, Dict[str, Any], Any]) -> TicketPrediction:
        """Perform triage prediction and optionally attach auxiliary intent."""
        prediction = self._predict_core(ticket)
        return self._attach_auxiliary_intent(prediction, ticket)

    def _predict_core(self, ticket: Union[TicketInput, Dict[str, Any], Any]) -> TicketPrediction:
        """Perform safe validation and produce a structured triage placeholder or inference."""
        validation: ValidationResult = validate_ticket_input(ticket)
        cleaned = validation.cleaned_ticket

        # Extract ticket_id safely even if validation failed
        ticket_id: Optional[str] = None
        if cleaned and cleaned.ticket_id:
            ticket_id = cleaned.ticket_id
        elif isinstance(ticket, dict) and ticket.get("ticket_id"):
            ticket_id = str(ticket.get("ticket_id"))
        elif isinstance(ticket, TicketInput) and ticket.ticket_id:
            ticket_id = ticket.ticket_id

        # Case 1: Empty or missing issue description
        if validation.empty_description:
            return TicketPrediction(
                ticket_id=ticket_id,
                follow_up_required=FollowUpRequired.REVIEW,
                interest_level=InterestLevel.UNKNOWN,
                urgency=UrgencyLevel.UNKNOWN,
                follow_up_reason=(
                    "Manual review required: Issue description is missing or empty. "
                    "Cannot perform semantic inference without student query text."
                ),
                priority_score=None,
                priority=PriorityCategory.NOT_APPLICABLE,
                recommended_action=(
                    "Route to support staff to contact student for issue details."
                ),
                model_status=ModelStatus.EMPTY_DESCRIPTION_REVIEW.value,
            )

        # Case 2: Invalid timestamp provided
        if validation.invalid_timestamp:
            return TicketPrediction(
                ticket_id=ticket_id,
                follow_up_required=FollowUpRequired.REVIEW,
                interest_level=InterestLevel.UNKNOWN,
                urgency=UrgencyLevel.UNKNOWN,
                follow_up_reason=(
                    f"Manual review required: Invalid timestamp provided. "
                    f"Errors: {'; '.join(validation.errors)}"
                ),
                priority_score=None,
                priority=PriorityCategory.NOT_APPLICABLE,
                recommended_action=(
                    "Verify ticket creation date and inspect issue manually."
                ),
                model_status=ModelStatus.INVALID_TIMESTAMP_REVIEW.value,
            )

        # Case 3: Other validation failures
        if not validation.is_valid:
            return TicketPrediction(
                ticket_id=ticket_id,
                follow_up_required=FollowUpRequired.REVIEW,
                interest_level=InterestLevel.UNKNOWN,
                urgency=UrgencyLevel.UNKNOWN,
                follow_up_reason=(
                    f"Input validation failure: {'; '.join(validation.errors)}"
                ),
                priority_score=None,
                priority=PriorityCategory.NOT_APPLICABLE,
                recommended_action=(
                    "Investigate ticket input formatting and triage manually."
                ),
                model_status=ModelStatus.VALIDATION_ERROR.value,
            )

        # Case 3.5: Explicit decision engine injection (with optional priority engine)
        if self.decision_engine is not None:
            decision = self.decision_engine.evaluate(ticket, validation)
            if self.priority_engine is not None:
                assessment = self.priority_engine.assess(
                    cleaned or ticket,
                    follow_up_decision=decision,
                    validation_result=validation,
                )
                return TicketPrediction(
                    ticket_id=ticket_id,
                    follow_up_required=decision.decision,
                    interest_level=assessment.interest_level,
                    urgency=assessment.urgency,
                    follow_up_reason=decision.reason,
                    priority_score=assessment.priority_score,
                    priority=assessment.priority,
                    recommended_action=assessment.recommended_action,
                    model_status=decision.classifier_status or ModelStatus.READY.value,
                )
            action_map = {
                FollowUpRequired.YES: "Advisor follow-up required. Contact student to resolve issue.",
                FollowUpRequired.NO: "No advisor follow-up required. Automated or self-service issue.",
                FollowUpRequired.REVIEW: "Manual advisor review recommended to assess complex student inquiry.",
            }
            return TicketPrediction(
                ticket_id=ticket_id,
                follow_up_required=decision.decision,
                interest_level=InterestLevel.UNKNOWN,
                urgency=UrgencyLevel.UNKNOWN,
                follow_up_reason=decision.reason,
                priority_score=None,
                priority=PriorityCategory.NOT_APPLICABLE,
                recommended_action=action_map.get(decision.decision, "Manual triage recommended."),
                model_status=decision.classifier_status or ModelStatus.READY.value,
            )

        # Case 3.6: Standalone priority engine injection
        if self.priority_engine is not None:
            return self.priority_engine.assess_ticket(
                cleaned or ticket,
                validation_result=validation,
                model_status=ModelStatus.READY.value,
            )

        # Case 4: Model loader integration
        if self.model_loader is not None:
            loader_status = getattr(self.model_loader, "status", None)
            is_fine_tuned = getattr(self.model_loader, "is_fine_tuned", False)
            status_reason = getattr(
                self.model_loader,
                "status_reason",
                "Model is currently unavailable for inference.",
            )

            if loader_status == ModelStatus.NOT_INITIALIZED:
                return TicketPrediction(
                    ticket_id=ticket_id,
                    follow_up_required=FollowUpRequired.REVIEW,
                    interest_level=InterestLevel.UNKNOWN,
                    urgency=UrgencyLevel.UNKNOWN,
                    follow_up_reason=f"Model not initialized: {status_reason}",
                    priority_score=None,
                    priority=PriorityCategory.NOT_APPLICABLE,
                    recommended_action="Initialize and load model weights before inference, or triage manually.",
                    model_status=ModelStatus.NOT_INITIALIZED.value,
                )

            if loader_status == ModelStatus.DEPENDENCIES_UNAVAILABLE:
                return TicketPrediction(
                    ticket_id=ticket_id,
                    follow_up_required=FollowUpRequired.REVIEW,
                    interest_level=InterestLevel.UNKNOWN,
                    urgency=UrgencyLevel.UNKNOWN,
                    follow_up_reason=f"ML dependencies missing: {status_reason}",
                    priority_score=None,
                    priority=PriorityCategory.NOT_APPLICABLE,
                    recommended_action="Install optional ML dependencies (torch, transformers) or triage manually.",
                    model_status=ModelStatus.DEPENDENCIES_UNAVAILABLE.value,
                )

            if loader_status == ModelStatus.MODEL_NOT_FOUND:
                return TicketPrediction(
                    ticket_id=ticket_id,
                    follow_up_required=FollowUpRequired.REVIEW,
                    interest_level=InterestLevel.UNKNOWN,
                    urgency=UrgencyLevel.UNKNOWN,
                    follow_up_reason=f"Model weights not found: {status_reason}",
                    priority_score=None,
                    priority=PriorityCategory.NOT_APPLICABLE,
                    recommended_action="Deploy fine-tuned model checkpoint or triage manually.",
                    model_status=ModelStatus.MODEL_NOT_FOUND.value,
                )

            # Task 6 & 7: Distinguish generic pretrained model vs fine-tuned
            if loader_status == ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED or not is_fine_tuned:
                return TicketPrediction(
                    ticket_id=ticket_id,
                    follow_up_required=FollowUpRequired.REVIEW,
                    interest_level=InterestLevel.UNKNOWN,
                    urgency=UrgencyLevel.UNKNOWN,
                    follow_up_reason=(
                        "Generic pretrained model detected: Not fine-tuned on student support labels. "
                        "Generic labels must never be mapped to YES, NO, or REVIEW decisions."
                    ),
                    priority_score=None,
                    priority=PriorityCategory.NOT_APPLICABLE,
                    recommended_action="Fine-tune model on student support dataset before enabling automated triage.",
                    model_status=ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED.value,
                )

            # Ready fine-tuned model inference
            if loader_status == ModelStatus.READY and is_fine_tuned and cleaned and cleaned.issue_description:
                inference = self.model_loader.classify_text(cleaned.issue_description)
                inf_status = inference.get("status")
                if inf_status != ModelStatus.READY:
                    return TicketPrediction(
                        ticket_id=ticket_id,
                        follow_up_required=FollowUpRequired.REVIEW,
                        interest_level=InterestLevel.UNKNOWN,
                        urgency=UrgencyLevel.UNKNOWN,
                        follow_up_reason=inference.get("status_reason", "Inference unready."),
                        priority_score=None,
                        priority=PriorityCategory.NOT_APPLICABLE,
                        recommended_action="Manual triage recommended due to inference status.",
                        model_status=getattr(inf_status, "value", str(inf_status)),
                    )

                raw_label = inference.get("label", "REVIEW")
                score = inference.get("score")
                if raw_label in FollowUpRequired._value2member_map_:
                    fu_enum = FollowUpRequired(raw_label)
                else:
                    fu_enum = FollowUpRequired.REVIEW

                action_map = {
                    FollowUpRequired.YES: "Advisor follow-up required. Contact student to resolve issue.",
                    FollowUpRequired.NO: "No advisor follow-up required. Automated or self-service issue.",
                    FollowUpRequired.REVIEW: "Manual advisor review recommended to assess complex student inquiry.",
                }

                score_str = f"{score:.2f}" if score is not None else "N/A"
                follow_up_reason = (
                    f"Classified by fine-tuned DistilBERT model '{self.config.model_name}' "
                    f"(decision: {fu_enum.value}, confidence: {score_str})."
                )

                if self.priority_engine is not None:
                    assessment = self.priority_engine.assess(
                        cleaned,
                        follow_up_decision=fu_enum,
                        validation_result=validation,
                    )
                    return TicketPrediction(
                        ticket_id=ticket_id,
                        follow_up_required=fu_enum,
                        interest_level=assessment.interest_level,
                        urgency=assessment.urgency,
                        follow_up_reason=follow_up_reason,
                        priority_score=assessment.priority_score,
                        priority=assessment.priority,
                        recommended_action=assessment.recommended_action,
                        model_status=ModelStatus.READY.value,
                    )

                return TicketPrediction(
                    ticket_id=ticket_id,
                    follow_up_required=fu_enum,
                    interest_level=InterestLevel.UNKNOWN,
                    urgency=UrgencyLevel.UNKNOWN,
                    follow_up_reason=follow_up_reason,
                    priority_score=None,  # Null when priority engine is not injected
                    priority=PriorityCategory.NOT_APPLICABLE,
                    recommended_action=action_map.get(fu_enum, "Manual triage recommended."),
                    model_status=ModelStatus.READY.value,
                )

        # Case 5: Default placeholder inference pending model training/weights
        return TicketPrediction(
            ticket_id=ticket_id,
            follow_up_required=FollowUpRequired.REVIEW,
            interest_level=InterestLevel.UNKNOWN,
            urgency=UrgencyLevel.UNKNOWN,
            follow_up_reason=(
                f"Placeholder prediction: Model '{self.config.model_name}' "
                "is not yet trained or deployed. Ticket routed for human review."
            ),
            priority_score=None,  # Null per constraint: cannot be calculated defensibly yet
            priority=PriorityCategory.NOT_APPLICABLE,
            recommended_action=(
                "Standard support queue triage until ML model deployment is complete."
            ),
            model_status=ModelStatus.PLACEHOLDER_PENDING_MODEL.value,
        )

    def predict_batch(
        self, tickets: List[Union[TicketInput, Dict[str, Any], Any]]
    ) -> List[TicketPrediction]:
        """Process a list of tickets sequentially."""
        return [self.predict(t) for t in tickets]

    def predict_priority(
        self, input_data: Union[str, PriorityInferenceInput, Dict[str, Any], Any]
    ) -> PriorityInferenceOutput:
        """Convenience method to extract priority_level from ticket or transcription."""
        predictor = PriorityPredictor(config=self.config)
        return predictor.predict_priority(input_data)


# Alias for backwards compatibility / descriptive clarity
PlaceholderPredictor = StudentSupportPredictor


# ==============================================================================
# Simplified Priority Predictor (GitHub Issue #7)
# ==============================================================================

class PriorityPredictor(BasePredictor):
    """Simplified priority inference interface for student support inquiries.
    
    Accepts `issue_transcription` and outputs `priority_level` with operational metadata.
    Completely isolated from obsolete PII processing, dual-model intent classifying,
    and complex triage Pydantic schemas.
    """

    def __init__(
        self,
        config: Optional[MLServiceConfig] = None,
        model_loader: Optional[Any] = None,
        model_dir: Optional[str] = None,
    ) -> None:
        from .priority_model import PriorityModelLoader

        self.config = config or MLServiceConfig()
        if model_loader is not None:
            self.model_loader = model_loader
        else:
            target_dir = model_dir or self.config.model_dir
            self.model_loader = PriorityModelLoader(config=self.config, model_dir=target_dir)
            self.model_loader.load_model()

    def predict_priority(
        self, input_data: Union[str, PriorityInferenceInput, Dict[str, Any], Any]
    ) -> PriorityInferenceOutput:
        """Predict priority level from transcription text.
        
        Args:
            input_data: String transcript, PriorityInferenceInput instance,
                        or dictionary containing 'issue_transcription' or 'issue_description'.
                        
        Returns:
            PriorityInferenceOutput with priority_level (HIGH, MEDIUM, LOW).
        """
        transcription: str = ""
        ticket_id: Optional[str] = None

        if isinstance(input_data, str):
            transcription = input_data
        elif isinstance(input_data, PriorityInferenceInput):
            transcription = input_data.issue_transcription
            ticket_id = input_data.ticket_id
        elif isinstance(input_data, dict):
            transcription = str(
                input_data.get("issue_transcription")
                or input_data.get("issue_description")
                or ""
            )
            ticket_id = input_data.get("ticket_id")
        elif hasattr(input_data, "issue_transcription"):
            transcription = str(getattr(input_data, "issue_transcription", "") or "")
            ticket_id = getattr(input_data, "ticket_id", None)
        elif hasattr(input_data, "issue_description"):
            transcription = str(getattr(input_data, "issue_description", "") or "")
            ticket_id = getattr(input_data, "ticket_id", None)

        res = self.model_loader.classify_priority(transcription)

        return PriorityInferenceOutput(
            priority_level=res.get("priority_level", "LOW"),
            confidence=res.get("confidence"),
            model_status=res.get("status", "READY"),
            is_trained_model=res.get("is_trained_model", False),
            status_reason=res.get("status_reason"),
        )

    def predict(self, ticket: Union[TicketInput, Dict[str, Any], Any]) -> Any:
        """BasePredictor compatibility: invokes predict_priority."""
        return self.predict_priority(ticket)

    def predict_batch(
        self, tickets: List[Union[TicketInput, Dict[str, Any], Any]]
    ) -> List[PriorityInferenceOutput]:
        """Process a list of tickets or transcriptions sequentially."""
        return [self.predict_priority(t) for t in tickets]

