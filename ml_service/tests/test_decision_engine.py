"""Unit tests for FollowUpDecisionEngine.

Verifies:
- YES: Explicit callback requests, confirmed urgent blockers, corroborated classifier YES.
- NO: Confirmed resolution in text AND closed status. Never inferred from missing info.
- REVIEW: Missing text, ambiguous brief descriptions, open status alone, closed status alone,
  conflicting evidence, and generic/unavailable classifiers.
- Constant 'High' priority_level does NOT force a YES.
- Student identifiers (PII) are strictly excluded from predictive features.
- Model inference is decoupled from business decisions.
"""

from unittest.mock import MagicMock
import pytest

from ml_service.config import MLServiceConfig
from ml_service.decision_engine import FollowUpDecision, FollowUpDecisionEngine
from ml_service.predictor import StudentSupportPredictor
from ml_service.schemas import (
    FollowUpRequired,
    ModelStatus,
    PriorityCategory,
    TicketInput,
    TicketPrediction,
)


class TestFollowUpDecisionEngine:
    """Test suite for FollowUpDecisionEngine rules, evidence analysis, and edge cases."""

    # ==========================================================================
    # YES Cases
    # ==========================================================================

    def test_explicit_callback_request_yields_yes(self):
        """Task 10: Explicit callback request provides sufficient evidence for YES."""
        engine = FollowUpDecisionEngine()
        ticket = TicketInput(
            ticket_id="T001",
            issue_description="I was charged twice for tuition fees. Please call me back urgently.",
            status="Open",
        )
        decision = engine.evaluate(ticket)
        assert decision.decision == FollowUpRequired.YES
        assert "Explicit student callback request" in decision.reason or "explicitly requested" in decision.reason
        assert any("callback request" in ev.lower() for ev in decision.evidence)

    def test_various_callback_phrasings_yield_yes(self):
        """Verify multiple natural phrasing variations of callback requests trigger YES."""
        engine = FollowUpDecisionEngine()
        phrases = [
            "Requesting a call from an academic advisor regarding grade appeal.",
            "Need someone to contact me by phone regarding scholarship deadline.",
            "Waiting for a callback regarding hostel allocation.",
            "Please reach out to me asap to fix my registration blocker.",
            "I need to speak with an advisor about my semester drop.",
        ]
        for phrase in phrases:
            ticket = TicketInput(issue_description=phrase, status="Open")
            decision = engine.evaluate(ticket)
            assert decision.decision == FollowUpRequired.YES, f"Failed for phrase: '{phrase}'"

    def test_unresolved_blocker_in_open_ticket_yields_yes(self):
        """Task 10: Specific unresolved transaction blocker in active ticket supports YES."""
        engine = FollowUpDecisionEngine()
        ticket = TicketInput(
            ticket_id="T002",
            issue_description="Fee was deducted but payment status is still failing and portal threw an error.",
            status="Open",
        )
        decision = engine.evaluate(ticket)
        assert decision.decision == FollowUpRequired.YES
        assert any("unresolved" in ev.lower() or "blocker" in ev.lower() for ev in decision.evidence)

    def test_task_specific_classifier_corroborated_yes(self):
        """Task 4: Corroborated task-specific fine-tuned classifier prediction yields YES."""
        mock_classifier = MagicMock()
        mock_classifier.status = ModelStatus.READY
        mock_classifier.is_fine_tuned = True
        mock_classifier.classify_text.return_value = {
            "status": ModelStatus.READY,
            "label": "YES",
            "score": 0.94,
        }

        engine = FollowUpDecisionEngine(classifier=mock_classifier)
        ticket = TicketInput(
            ticket_id="T003",
            issue_description="Inquiry about financial aid eligibility and special consideration.",
            status="Open",
        )
        decision = engine.evaluate(ticket)
        assert decision.decision == FollowUpRequired.YES
        assert decision.classifier_used is True
        assert decision.confidence == 0.94
        assert "Validated model classified YES" in decision.reason

    # ==========================================================================
    # NO Cases
    # ==========================================================================

    def test_confirmed_resolution_with_closed_status_yields_no(self):
        """Task 9: Reliable evidence confirms resolution in text AND closed ticket status."""
        engine = FollowUpDecisionEngine()
        ticket = TicketInput(
            ticket_id="T004",
            issue_description="The issue has been resolved successfully. Received my refund, no further help needed.",
            status="Closed",
        )
        decision = engine.evaluate(ticket)
        assert decision.decision == FollowUpRequired.NO
        assert "Reliable evidence establishes that no further follow-up is needed" in decision.reason
        assert any("confirmation of resolved issue" in ev.lower() for ev in decision.evidence)

    def test_closed_status_alone_without_text_evidence_yields_review(self):
        """Task 9: Resolution cannot be inferred from closed status alone or missing text."""
        engine = FollowUpDecisionEngine()
        ticket = TicketInput(
            ticket_id="T005",
            issue_description="Inquiry regarding curriculum elective courses for next year.",
            status="Closed",
        )
        decision = engine.evaluate(ticket)
        # Must NOT assume NO without explicit confirmation of resolution
        assert decision.decision == FollowUpRequired.REVIEW
        assert decision.is_conservative_fallback is True
        assert "text does not confirm resolution" in decision.reason.lower()

    def test_missing_info_never_inferred_as_resolution(self):
        """Task 9: Vague inquiry in closed ticket does not produce NO."""
        engine = FollowUpDecisionEngine()
        ticket = TicketInput(
            ticket_id="T006",
            issue_description="Grade inquiry",
            status="Resolved",
        )
        decision = engine.evaluate(ticket)
        assert decision.decision == FollowUpRequired.REVIEW

    # ==========================================================================
    # REVIEW Cases & Conservative Defaults
    # ==========================================================================

    def test_empty_description_yields_review(self):
        """Task 8: Empty issue description produces conservative REVIEW."""
        engine = FollowUpDecisionEngine()
        ticket = TicketInput(ticket_id="T007", issue_description="")
        decision = engine.evaluate(ticket)
        assert decision.decision == FollowUpRequired.REVIEW
        assert decision.is_conservative_fallback is True
        assert "missing or empty" in decision.reason.lower()

    def test_whitespace_only_description_yields_review(self):
        """Task 8: Whitespace-only text produces conservative REVIEW."""
        engine = FollowUpDecisionEngine()
        ticket = TicketInput(ticket_id="T008", issue_description="   \n\t  ")
        decision = engine.evaluate(ticket)
        assert decision.decision == FollowUpRequired.REVIEW
        assert decision.is_conservative_fallback is True

    def test_none_description_yields_review(self):
        """Task 8: None text produces conservative REVIEW."""
        engine = FollowUpDecisionEngine()
        ticket = TicketInput(ticket_id="T009", issue_description=None)
        decision = engine.evaluate(ticket)
        assert decision.decision == FollowUpRequired.REVIEW

    def test_brief_ambiguous_text_yields_review(self):
        """Task 8: Short, vague queries lack sufficient evidence and produce REVIEW."""
        engine = FollowUpDecisionEngine()
        vague_queries = ["fees", "help", "hostel", "registration query", "hello sir"]
        for q in vague_queries:
            ticket = TicketInput(issue_description=q, status="Open")
            decision = engine.evaluate(ticket)
            assert decision.decision == FollowUpRequired.REVIEW, f"Failed for query: '{q}'"
            assert "ambiguous" in decision.reason.lower() or "brief" in decision.reason.lower()

    def test_open_status_alone_does_not_force_yes(self):
        """Task 6: An open status alone is insufficient to classify as YES."""
        engine = FollowUpDecisionEngine()
        ticket = TicketInput(
            ticket_id="T010",
            issue_description="Seeking clarification on syllabus for second semester math.",
            status="Open",
            priority_level="Normal",
        )
        decision = engine.evaluate(ticket)
        # Standard informational inquiry without callback request or trained model must be REVIEW
        assert decision.decision == FollowUpRequired.REVIEW
        assert decision.decision != FollowUpRequired.YES

    def test_constant_high_priority_does_not_force_yes(self):
        """Task 7: Constant 'High' priority_level must not independently force YES."""
        engine = FollowUpDecisionEngine()
        ticket_high = TicketInput(
            ticket_id="T011",
            issue_description="Seeking clarification on syllabus for second semester math.",
            status="Open",
            priority_level="High",  # Legacy constant
        )
        decision = engine.evaluate(ticket_high)
        assert decision.decision == FollowUpRequired.REVIEW
        assert decision.decision != FollowUpRequired.YES

    def test_priority_level_variations_do_not_alter_decision(self):
        """Task 7: Priority level changes must not independently alter follow-up eligibility."""
        engine = FollowUpDecisionEngine()
        desc = "General inquiry about holiday schedule."
        for p_level in ["High", "Low", "Critical", None, ""]:
            ticket = TicketInput(issue_description=desc, status="Open", priority_level=p_level)
            decision = engine.evaluate(ticket)
            assert decision.decision == FollowUpRequired.REVIEW

    # ==========================================================================
    # Conflicting Evidence Handling
    # ==========================================================================

    def test_closed_status_with_unresolved_request_conflict_yields_review(self):
        """Task 8 & 14: Closed status with unresolved problem in text triggers conflict REVIEW."""
        engine = FollowUpDecisionEngine()
        ticket = TicketInput(
            ticket_id="T012",
            issue_description="Ticket is marked closed but my issue is not resolved, still waiting for refund.",
            status="Closed",
        )
        decision = engine.evaluate(ticket)
        assert decision.decision == FollowUpRequired.REVIEW
        assert len(decision.conflicts) > 0
        assert "Conflicting signals detected" in decision.reason

    def test_closed_status_with_callback_request_conflict_yields_review(self):
        """Closed status with explicit callback request in text triggers conflict REVIEW."""
        engine = FollowUpDecisionEngine()
        ticket = TicketInput(
            ticket_id="T013",
            issue_description="Please call me back immediately regarding the portal issue.",
            status="Closed",
        )
        decision = engine.evaluate(ticket)
        assert decision.decision == FollowUpRequired.REVIEW
        assert len(decision.conflicts) > 0

    def test_text_contradiction_resolved_and_callback_yields_review(self):
        """Text containing both resolved phrasing and callback request triggers conflict REVIEW."""
        engine = FollowUpDecisionEngine()
        ticket = TicketInput(
            ticket_id="T014",
            issue_description="Everything is clear now, but please call me back tomorrow anyway.",
            status="Open",
        )
        decision = engine.evaluate(ticket)
        assert decision.decision == FollowUpRequired.REVIEW
        assert len(decision.conflicts) > 0
        assert any("contradictory" in c.lower() for c in decision.conflicts)

    def test_classifier_conflict_with_explicit_callback_yields_review(self):
        """Task 14: If classifier predicts NO but text explicitly requests a call, conflict triggers REVIEW."""
        mock_classifier = MagicMock()
        mock_classifier.status = ModelStatus.READY
        mock_classifier.is_fine_tuned = True
        mock_classifier.classify_text.return_value = {
            "status": ModelStatus.READY,
            "label": "NO",
            "score": 0.88,
        }

        engine = FollowUpDecisionEngine(classifier=mock_classifier)
        ticket = TicketInput(
            ticket_id="T015",
            issue_description="Please call me back regarding my fee receipt.",
            status="Open",
        )
        decision = engine.evaluate(ticket)
        assert decision.decision == FollowUpRequired.REVIEW
        assert len(decision.conflicts) > 0
        assert any("classifier predicted no" in c.lower() for c in decision.conflicts)

    # ==========================================================================
    # Classifier Decoupling & Safety Checks
    # ==========================================================================

    def test_generic_pretrained_labels_never_used_and_yields_review(self):
        """Task 5: Pretrained DistilBERT generic labels (LABEL_0, POSITIVE) are strictly rejected."""
        mock_generic_loader = MagicMock()
        mock_generic_loader.status = ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED
        mock_generic_loader.is_fine_tuned = False

        engine = FollowUpDecisionEngine(classifier=mock_generic_loader)
        ticket = TicketInput(
            ticket_id="T016",
            issue_description="Standard academic calendar inquiry.",
            status="Open",
        )
        decision = engine.evaluate(ticket)
        assert decision.decision == FollowUpRequired.REVIEW
        assert decision.classifier_used is False
        assert decision.classifier_status == ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED.value
        assert any("rejected" in ev.lower() or "generic" in ev.lower() for ev in decision.evidence)

    def test_unavailable_classifier_makes_limitations_explicit(self):
        """Task 11: When no task-specific classifier is available, limitations are explicit."""
        engine = FollowUpDecisionEngine(classifier=None)
        ticket = TicketInput(
            ticket_id="T017",
            issue_description="Inquiry on elective subject choices.",
            status="Open",
        )
        decision = engine.evaluate(ticket)
        assert decision.decision == FollowUpRequired.REVIEW
        assert decision.is_conservative_fallback is True
        assert "limitations" in decision.reason.lower() or "no validated" in decision.reason.lower()

    def test_runtime_classifier_injection(self):
        """Task 4: set_classifier allows dynamic runtime dependency injection."""
        engine = FollowUpDecisionEngine()
        assert engine.classifier is None

        mock_clf = MagicMock()
        mock_clf.status = ModelStatus.READY
        mock_clf.is_fine_tuned = True
        mock_clf.classify_text.return_value = {
            "status": ModelStatus.READY,
            "label": "YES",
            "score": 0.95,
        }

        engine.set_classifier(mock_clf)
        ticket = TicketInput(
            issue_description="Hostel room maintenance issue escalation.",
            status="Open",
        )
        decision = engine.evaluate(ticket)
        assert decision.decision == FollowUpRequired.YES
        assert decision.classifier_used is True

    # ==========================================================================
    # PII Guardrail Verification
    # ==========================================================================

    def test_pii_fields_do_not_influence_decisions(self):
        """Task 12: Changing student identifiers and phone numbers does not alter decision."""
        engine = FollowUpDecisionEngine()
        desc = "Inquiry regarding bus pass renewal."

        ticket_1 = TicketInput(
            ticket_id="ID-001",
            student_name="Alice Smith",
            application_number="APP111",
            phone_number="9998887776",
            issue_description=desc,
            status="Open",
        )
        ticket_2 = TicketInput(
            ticket_id="ID-999",
            student_name="Bob Jones",
            application_number="REG999",
            phone_number="1112223334",
            issue_description=desc,
            status="Open",
        )
        dec_1 = engine.evaluate(ticket_1)
        dec_2 = engine.evaluate(ticket_2)

        assert dec_1.decision == dec_2.decision
        assert dec_1.reason == dec_2.reason
        assert dec_1.evidence == dec_2.evidence

    # ==========================================================================
    # Integration with Predictor
    # ==========================================================================

    def test_predictor_with_injected_decision_engine(self):
        """Task 15: StudentSupportPredictor integrates seamlessly with FollowUpDecisionEngine."""
        engine = FollowUpDecisionEngine()
        predictor = StudentSupportPredictor(decision_engine=engine)

        ticket = TicketInput(
            ticket_id="VIT888",
            issue_description="Please call me back immediately regarding admission fee payment.",
            status="Open",
        )
        prediction: TicketPrediction = predictor.predict(ticket)

        assert prediction.ticket_id == "VIT888"
        assert prediction.follow_up_required == FollowUpRequired.YES
        assert prediction.priority_score is None  # Score remains null per constraint
        assert prediction.priority == PriorityCategory.NOT_APPLICABLE
        assert "Advisor follow-up required" in prediction.recommended_action
