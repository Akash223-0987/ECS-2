"""Unit tests for PriorityAssessmentEngine.

Verifies:
- Clearly urgent unresolved issues with imminent deadlines.
- High urgency does NOT automatically assign a score of 100.
- Medium and low urgency scoring and categorization.
- Unknown urgency for ambiguous or insufficient descriptions.
- Student expressed interest levels (HIGH, MEDIUM, LOW) strictly separated from urgency (UNKNOWN default).
- Explicitly resolved tickets (NO follow-up) produce NOT_APPLICABLE priority and no call recommendation.
- REVIEW decisions remain review cases and never trigger automatic call recommendations.
- Missing, empty, and whitespace descriptions return null scores with clear rationale.
- Legacy 'priority_level: High' constant invariance.
- PII exclusion (name, phone, registration, ticket_id) invariance.
- Valid score boundaries [0.0, 100.0].
- Insufficient evidence returning a null priority_score.
- Recommended actions that strictly preserve follow-up eligibility.
- Integration with StudentSupportPredictor.
"""

from unittest.mock import MagicMock
import pytest

from ml_service.config import MLServiceConfig
from ml_service.decision_engine import FollowUpDecision, FollowUpDecisionEngine
from ml_service.predictor import StudentSupportPredictor
from ml_service.priority_engine import (
    PriorityAssessment,
    PriorityAssessmentEngine,
    ScoreBreakdown,
)
from ml_service.schemas import (
    FollowUpRequired,
    InterestLevel,
    ModelStatus,
    PriorityCategory,
    TicketInput,
    TicketPrediction,
    UrgencyLevel,
)


class TestPriorityAssessmentEngine:
    """Comprehensive test suite for PriorityAssessmentEngine."""

    # ==========================================================================
    # 1. Urgent, Medium, Low, and Unknown Urgency Tests
    # ==========================================================================

    def test_clearly_urgent_unresolved_issue_with_imminent_deadline(self):
        """Urgent issue with imminent deadline, payment failure, and callback yields score 100 and HIGH priority."""
        engine = PriorityAssessmentEngine()
        ticket = TicketInput(
            ticket_id="URG-001",
            issue_description="Deadline is tomorrow and fee was deducted twice, unable to submit application. Please call me back urgently.",
            status="Open",
        )
        assessment = engine.assess(ticket)

        assert assessment.urgency == UrgencyLevel.HIGH
        assert assessment.priority == PriorityCategory.HIGH
        assert assessment.priority_score == 100.0
        assert assessment.is_defensible is True
        assert "Urgent advisor callback" in assessment.recommended_action
        assert assessment.score_breakdown is not None
        assert assessment.score_breakdown.urgency_score == 50.0
        assert assessment.score_breakdown.impact_score == 30.0
        assert assessment.score_breakdown.actionability_score == 20.0

    def test_high_urgency_without_automatic_score_of_100(self):
        """Task 8: High urgency must NOT automatically assign 100 to every case."""
        engine = PriorityAssessmentEngine()
        # Urgent blocker without explicit imminent deadline and without explicit callback request
        ticket = TicketInput(
            ticket_id="URG-002",
            issue_description="Payment was deducted but status is still failing and portal threw an error.",
            status="Open",
        )
        assessment = engine.assess(ticket)

        assert assessment.urgency == UrgencyLevel.HIGH
        assert assessment.priority == PriorityCategory.HIGH
        # High urgency (40 pts) + Critical Impact (30 pts) + Unresolved Issue (10 pts) = 80.0 pts != 100.0
        assert assessment.priority_score == 80.0
        assert assessment.priority_score < 100.0

    def test_medium_urgency_unresolved_issue(self):
        """Task 3 & 10: Medium urgency with ongoing process consequence and callback."""
        engine = PriorityAssessmentEngine()
        ticket = TicketInput(
            ticket_id="MED-001",
            issue_description="Need assistance with scholarship application document verification for next month's deadline. Please call me back.",
            status="Open",
        )
        assessment = engine.assess(ticket)

        assert assessment.urgency == UrgencyLevel.MEDIUM
        assert assessment.priority == PriorityCategory.MEDIUM
        # Medium Urgency (25 pts) + Moderate Impact (15 pts) + Callback (20 pts) = 60.0 pts
        assert assessment.priority_score == 60.0
        assert "Standard advisor callback" in assessment.recommended_action

    def test_low_urgency_routine_inquiry(self):
        """Task 3 & 10: Low urgency routine request with callback."""
        engine = PriorityAssessmentEngine()
        ticket = TicketInput(
            ticket_id="LOW-001",
            issue_description="Can you please share the bus route schedule and library timings? General inquiry, no rush. Please reach out to me.",
            status="Open",
        )
        assessment = engine.assess(ticket)

        assert assessment.urgency == UrgencyLevel.LOW
        assert assessment.priority == PriorityCategory.LOW
        # Low Urgency (10 pts) + Routine Impact (5 pts) + Callback (20 pts) = 35.0 pts
        assert assessment.priority_score == 35.0
        assert "Routine support outreach" in assessment.recommended_action

    def test_unknown_urgency_due_to_vague_or_ambiguous_text(self):
        """Task 3 & 10: Ambiguous or brief text yields UNKNOWN urgency."""
        engine = PriorityAssessmentEngine()
        ticket = TicketInput(
            ticket_id="UNK-001",
            issue_description="fees query",
            status="Open",
        )
        assessment = engine.assess(ticket)

        assert assessment.urgency == UrgencyLevel.UNKNOWN
        # Ambiguous ticket triggers REVIEW in follow-up, so priority is NOT_APPLICABLE and score is None
        assert assessment.priority == PriorityCategory.NOT_APPLICABLE
        assert assessment.priority_score is None

    # ==========================================================================
    # 2. Student Expressed Interest Tests (Task 4)
    # ==========================================================================

    def test_interest_level_high_explicit_expression(self):
        """Task 4: High interest assigned ONLY when student explicitly expresses eagerness/intent to join."""
        engine = PriorityAssessmentEngine()
        ticket = TicketInput(
            ticket_id="INT-001",
            issue_description="I am eager to join VIT-AP and want to accept my admission offer. Deadline is tomorrow, please call me.",
            status="Open",
        )
        assessment = engine.assess(ticket)

        assert assessment.interest_level == InterestLevel.HIGH
        assert assessment.urgency == UrgencyLevel.HIGH

    def test_interest_level_medium_exploratory_expression(self):
        """Task 4: Medium interest assigned when student expresses exploratory interest."""
        engine = PriorityAssessmentEngine()
        ticket = TicketInput(
            ticket_id="INT-002",
            issue_description="I am considering applying and interested in knowing more about MBA specializations. Please call me back.",
            status="Open",
        )
        assessment = engine.assess(ticket)

        assert assessment.interest_level == InterestLevel.MEDIUM

    def test_interest_level_low_withdrawal_expression(self):
        """Task 4: Low interest assigned when student explicitly expresses withdrawal or disinterest."""
        engine = PriorityAssessmentEngine()
        ticket = TicketInput(
            ticket_id="INT-003",
            issue_description="I wish to withdraw my application as I am joining another university. Please call me back regarding my refund.",
            status="Open",
        )
        assessment = engine.assess(ticket)

        assert assessment.interest_level == InterestLevel.LOW

    def test_interest_level_not_inferred_from_urgency_defaults_unknown(self):
        """Task 4: Critical safety test - Urgency must NEVER dictate interest level. Routine support defaults to UNKNOWN."""
        engine = PriorityAssessmentEngine()
        # High urgency ticket without expressed interest
        ticket = TicketInput(
            ticket_id="INT-004",
            issue_description="Payment deducted twice, portal gave an error and deadline is today! Please call me back immediately.",
            status="Open",
        )
        assessment = engine.assess(ticket)

        # Urgency is HIGH, but Interest MUST be UNKNOWN!
        assert assessment.urgency == UrgencyLevel.HIGH
        assert assessment.interest_level == InterestLevel.UNKNOWN

    # ==========================================================================
    # 3. Follow-Up Decision Respect & Safety Rules (Task 7 & 10)
    # ==========================================================================

    def test_explicitly_resolved_ticket_yields_not_applicable_priority_and_no_call(self):
        """Task 7 & 10: Follow-up NO must result in priority: NOT_APPLICABLE and NEVER recommend a call."""
        engine = PriorityAssessmentEngine()
        ticket = TicketInput(
            ticket_id="RES-001",
            issue_description="The fee issue has been resolved successfully. Received my refund receipt, no further help needed.",
            status="Closed",
        )
        assessment = engine.assess(ticket)

        assert assessment.priority == PriorityCategory.NOT_APPLICABLE
        assert assessment.priority_score is None
        # Strict safety check: Must NEVER recommend an outbound call
        assert "callback" not in assessment.recommended_action.lower()
        assert "call" not in assessment.recommended_action.lower()
        assert "no advisor follow-up required" in assessment.recommended_action.lower()

    def test_review_decision_remains_review_and_never_triggers_call(self):
        """Task 7 & 10: REVIEW decision must NOT be converted to automatic call eligibility."""
        engine = PriorityAssessmentEngine()
        # Conflicting signals: Closed status but reports ongoing blocker
        ticket = TicketInput(
            ticket_id="REV-001",
            issue_description="Ticket is marked closed but my issue is not resolved, still waiting for refund.",
            status="Closed",
        )
        assessment = engine.assess(ticket)

        assert assessment.priority == PriorityCategory.NOT_APPLICABLE
        assert assessment.priority_score is None
        assert "manual advisor review required" in assessment.recommended_action.lower()
        # Must not schedule an automatic call
        assert "urgent advisor callback" not in assessment.recommended_action.lower()

    def test_insufficient_evidence_callback_without_issue_context_returns_null_score(self):
        """Task 6 & 10: When student requests callback with no issue context, score must be null."""
        engine = PriorityAssessmentEngine()
        ticket = TicketInput(
            ticket_id="INS-001",
            issue_description="Please call me back.",
            status="Open",
        )
        assessment = engine.assess(ticket)

        assert assessment.urgency == UrgencyLevel.UNKNOWN
        assert assessment.priority_score is None
        assert assessment.priority == PriorityCategory.NOT_APPLICABLE
        assert assessment.is_defensible is False
        assert "insufficient evidence" in assessment.rationale.lower()

    # ==========================================================================
    # 4. Missing, Empty, and Ambiguous Text Handling (Task 10)
    # ==========================================================================

    def test_empty_description_returns_null_score_and_not_applicable(self):
        """Empty description produces safe fallback with null score."""
        engine = PriorityAssessmentEngine()
        ticket = TicketInput(ticket_id="EMP-001", issue_description="")
        assessment = engine.assess(ticket)

        assert assessment.urgency == UrgencyLevel.UNKNOWN
        assert assessment.interest_level == InterestLevel.UNKNOWN
        assert assessment.priority_score is None
        assert assessment.priority == PriorityCategory.NOT_APPLICABLE
        assert assessment.is_defensible is False

    def test_whitespace_only_description_returns_null_score(self):
        """Whitespace-only description produces safe fallback with null score."""
        engine = PriorityAssessmentEngine()
        ticket = TicketInput(ticket_id="EMP-002", issue_description="   \n\t  ")
        assessment = engine.assess(ticket)

        assert assessment.priority_score is None
        assert assessment.priority == PriorityCategory.NOT_APPLICABLE

    def test_none_description_returns_null_score(self):
        """None description produces safe fallback with null score."""
        engine = PriorityAssessmentEngine()
        ticket = TicketInput(ticket_id="EMP-003", issue_description=None)
        assessment = engine.assess(ticket)

        assert assessment.priority_score is None
        assert assessment.priority == PriorityCategory.NOT_APPLICABLE

    # ==========================================================================
    # 5. Invariance & Ethical Guardrails (Task 5 & 10)
    # ==========================================================================

    def test_constant_high_priority_invariance(self):
        """Task 5 & 10: Changing priority_level does not alter score, urgency, or priority."""
        engine = PriorityAssessmentEngine()
        desc = "Need assistance with scholarship application document verification for next month's deadline. Please call me back."

        ticket_high = TicketInput(issue_description=desc, status="Open", priority_level="High")
        ticket_low = TicketInput(issue_description=desc, status="Open", priority_level="Low")
        ticket_none = TicketInput(issue_description=desc, status="Open", priority_level=None)

        res_high = engine.assess(ticket_high)
        res_low = engine.assess(ticket_low)
        res_none = engine.assess(ticket_none)

        assert res_high.priority_score == res_low.priority_score == res_none.priority_score
        assert res_high.priority == res_low.priority == res_none.priority
        assert res_high.urgency == res_low.urgency == res_none.urgency
        assert res_high.recommended_action == res_low.recommended_action

    def test_pii_exclusion_invariance(self):
        """Task 5 & 10: Changing student identifiers, contact info, or ticket ID does not alter assessment."""
        engine = PriorityAssessmentEngine()
        desc = "Deadline is tomorrow and fee was deducted twice, unable to submit application. Please call me back urgently."

        ticket_a = TicketInput(
            ticket_id="TICKET-001",
            student_name="Alice Smith",
            application_number="APP12345",
            registration_number="REG12345",
            phone_number="9876543210",
            issue_description=desc,
            status="Open",
        )
        ticket_b = TicketInput(
            ticket_id="TICKET-999",
            student_name="Bob Johnson",
            application_number="APP99999",
            registration_number="REG99999",
            phone_number="1234567890",
            issue_description=desc,
            status="Open",
        )

        res_a = engine.assess(ticket_a)
        res_b = engine.assess(ticket_b)

        assert res_a.priority_score == res_b.priority_score
        assert res_a.priority == res_b.priority
        assert res_a.urgency == res_b.urgency
        assert res_a.interest_level == res_b.interest_level
        assert res_a.recommended_action == res_b.recommended_action

    # ==========================================================================
    # 6. Score Boundaries & Calculation Validity (Task 10)
    # ==========================================================================

    def test_scores_at_valid_boundaries(self):
        """Task 10: Scores must be strictly bounded within [0.0, 100.0]."""
        engine = PriorityAssessmentEngine()

        # Maximum possible score: Imminent deadline (50) + Critical impact (30) + Callback (20) = 100.0
        max_ticket = TicketInput(
            issue_description="Deadline is today, exam is today and fee was deducted twice, portal gave an error. Please call me back immediately.",
            status="Open",
        )
        max_res = engine.assess(max_ticket)
        assert max_res.priority_score == 100.0
        assert max_res.priority == PriorityCategory.HIGH

        # Minimal valid score for low urgency query forced YES
        min_ticket = TicketInput(
            issue_description="Curious about campus tour schedule and gym timings. General question, no rush.",
            status="Open",
        )
        # Follow-up engine would treat routine inquiry without callback as REVIEW, but if explicitly evaluated with YES:
        min_res = engine.assess(min_ticket, follow_up_decision=FollowUpRequired.YES)
        assert min_res.priority_score is not None
        assert 0.0 <= min_res.priority_score <= 100.0
        assert min_res.priority == PriorityCategory.LOW

    # ==========================================================================
    # 7. Integration & Output Schemas (Task 1 & 9)
    # ==========================================================================

    def test_assess_ticket_returns_valid_ticket_prediction_instance(self):
        """Verify assess_ticket produces a valid TicketPrediction instance complying with schemas."""
        engine = PriorityAssessmentEngine()
        ticket = TicketInput(
            ticket_id="PRED-001",
            issue_description="Deadline is tomorrow and fee was deducted twice, unable to submit application. Please call me back urgently.",
            status="Open",
        )
        prediction = engine.assess_ticket(ticket)

        assert isinstance(prediction, TicketPrediction)
        assert prediction.ticket_id == "PRED-001"
        assert prediction.follow_up_required == FollowUpRequired.YES
        assert prediction.urgency == UrgencyLevel.HIGH
        assert prediction.priority_score == 100.0
        assert prediction.priority == PriorityCategory.HIGH
        assert "Urgent advisor callback" in prediction.recommended_action

    def test_predictor_with_injected_decision_and_priority_engines(self):
        """Verify StudentSupportPredictor coordinates decision_engine and priority_engine seamlessly."""
        dec_engine = FollowUpDecisionEngine()
        prio_engine = PriorityAssessmentEngine(decision_engine=dec_engine)
        predictor = StudentSupportPredictor(decision_engine=dec_engine, priority_engine=prio_engine)

        ticket = TicketInput(
            ticket_id="INTG-001",
            issue_description="Tuition fee was deducted twice and portal gave an error. Please call me back immediately.",
            status="Open",
        )
        pred = predictor.predict(ticket)

        assert pred.ticket_id == "INTG-001"
        assert pred.follow_up_required == FollowUpRequired.YES
        assert pred.urgency == UrgencyLevel.HIGH
        assert pred.priority == PriorityCategory.HIGH
        assert pred.priority_score is not None
        assert pred.priority_score >= 70.0
        assert "Urgent advisor callback" in pred.recommended_action

    def test_recommended_actions_never_bypass_follow_up_eligibility(self):
        """Task 10: Recommended actions must strictly respect follow-up eligibility.
        - NO: Never recommends a callback or human intervention.
        - REVIEW: Never automatically schedules a callback.
        - YES: Schedules appropriate tiered follow-up.
        """
        engine = PriorityAssessmentEngine()

        # Ticket with resolved status and resolution text
        ticket_no = TicketInput(
            issue_description="The issue has been resolved successfully. Received my refund, no further help needed.",
            status="Closed",
        )
        res_no = engine.assess(ticket_no)
        assert res_no.priority == PriorityCategory.NOT_APPLICABLE
        assert res_no.priority_score is None
        assert "no advisor follow-up required" in res_no.recommended_action.lower()
        assert "call" not in res_no.recommended_action.lower()

        # Ticket in REVIEW due to ambiguous text
        ticket_review = TicketInput(
            issue_description="Need advice on general elective courses.",
            status="Open",
        )
        res_review = engine.assess(ticket_review)
        assert res_review.priority == PriorityCategory.NOT_APPLICABLE
        assert res_review.priority_score is None
        assert "manual advisor review required" in res_review.recommended_action.lower()
        assert "urgent advisor callback" not in res_review.recommended_action.lower()

        # Ticket with YES eligibility
        ticket_yes = TicketInput(
            issue_description="Please call me back immediately regarding my tuition fee refund blocker.",
            status="Open",
        )
        res_yes = engine.assess(ticket_yes)
        assert res_yes.priority in {PriorityCategory.HIGH, PriorityCategory.MEDIUM}
        assert res_yes.priority_score is not None
        assert "callback" in res_yes.recommended_action.lower()

    def test_generic_classifier_rejection_preserves_review_and_null_score(self):
        """Task 9: Generic DistilBERT models (LABEL_0, POSITIVE) are rejected by decision engine,
        resulting in REVIEW, NOT_APPLICABLE priority, and null priority_score.
        """
        mock_generic_loader = MagicMock()
        mock_generic_loader.status = ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED
        mock_generic_loader.is_fine_tuned = False

        dec_engine = FollowUpDecisionEngine(classifier=mock_generic_loader)
        prio_engine = PriorityAssessmentEngine(decision_engine=dec_engine)

        ticket = TicketInput(
            ticket_id="GEN-001",
            issue_description="Tuition fee inquiry with general academic question.",
            status="Open",
        )
        pred = prio_engine.assess_ticket(ticket)

        assert pred.follow_up_required == FollowUpRequired.REVIEW
        assert pred.priority == PriorityCategory.NOT_APPLICABLE
        assert pred.priority_score is None
        assert "manual advisor review required" in pred.recommended_action.lower()
