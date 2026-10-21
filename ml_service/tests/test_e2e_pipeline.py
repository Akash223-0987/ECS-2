"""End-to-End Pipeline Validation Tests for Student Support ML Service.

Validates the complete execution flow from raw input (dict or TicketInput)
through validation, follow-up decision, priority assessment, and output
serialization against the TicketPrediction schema.

Covers:
- Empty and whitespace issue descriptions
- Ambiguous descriptions (e.g., "fees")
- Explicit callback requests with sufficient context
- Explicitly resolved tickets with closed status
- Open tickets with no actionable evidence
- Urgent unresolved blockers (verifying score != 100 automatic)
- Conflicting resolution and callback evidence
- Conflicting closed status and unresolved blocker evidence
- Missing and corrupted timestamps
- Generic pretrained model label rejection (LABEL_0, POSITIVE)
- Unavailable model dependencies and missing model checkpoint artifacts
- Full TicketPrediction schema compliance, enum validity, and score bounds [0.0, 100.0]
- PII and constant 'priority_level: High' invariance
- Absolute suppression of automatic calls for NO and REVIEW decisions
- Follow-up YES with insufficient evidence returning null priority_score
- Batch processing and JSON serialization round-trip
"""

import json
from unittest.mock import MagicMock
import pytest

from ml_service.config import MLServiceConfig
from ml_service.decision_engine import FollowUpDecisionEngine
from ml_service.model_loader import HuggingFaceModelLoader
from ml_service.predictor import StudentSupportPredictor
from ml_service.priority_engine import PriorityAssessmentEngine
from ml_service.schemas import (
    FollowUpRequired,
    InterestLevel,
    ModelStatus,
    PriorityCategory,
    TicketInput,
    TicketPrediction,
    UrgencyLevel,
)
from ml_service.tests.fixtures import (
    ALL_SYNTHETIC_FIXTURES,
    FIXTURE_AMBIGUOUS_FEES,
    FIXTURE_CALLBACK_WITHOUT_ISSUE_DETAILS,
    FIXTURE_CLOSED_STATUS_WITH_UNRESOLVED_CONFLICT,
    FIXTURE_CONFLICT_RESOLVED_AND_CALLBACK,
    FIXTURE_EMPTY_DESCRIPTION,
    FIXTURE_EXPLICIT_CALLBACK_WITH_CONTEXT,
    FIXTURE_EXPLICITLY_RESOLVED,
    FIXTURE_INVALID_TIMESTAMP,
    FIXTURE_MISSING_TIMESTAMP,
    FIXTURE_OPEN_NO_ACTIONABLE_EVIDENCE,
    FIXTURE_URGENT_UNRESOLVED_BLOCKER,
    FIXTURE_WHITESPACE_DESCRIPTION,
)


class TestEndToEndPipeline:
    """Test suite validating the end-to-end triage prediction pipeline."""

    @pytest.fixture
    def e2e_pipeline(self) -> StudentSupportPredictor:
        """Create a fully wired end-to-end pipeline predictor."""
        return StudentSupportPredictor.create_default_pipeline()

    # ==========================================================================
    # 1. Core End-to-End Test Scenarios (Task 3)
    # ==========================================================================

    def test_e2e_empty_issue_description(self, e2e_pipeline: StudentSupportPredictor):
        """Case 1: Empty description routes safely to REVIEW with null priority score."""
        pred = e2e_pipeline.predict(FIXTURE_EMPTY_DESCRIPTION)

        assert isinstance(pred, TicketPrediction)
        assert pred.ticket_id == FIXTURE_EMPTY_DESCRIPTION["ticket_id"]
        assert pred.follow_up_required == FollowUpRequired.REVIEW
        assert pred.interest_level == InterestLevel.UNKNOWN
        assert pred.urgency == UrgencyLevel.UNKNOWN
        assert pred.priority_score is None
        assert pred.priority == PriorityCategory.NOT_APPLICABLE
        assert pred.model_status == ModelStatus.EMPTY_DESCRIPTION_REVIEW.value
        assert "missing or empty" in pred.follow_up_reason.lower()
        assert "contact student for issue details" in pred.recommended_action.lower()

    def test_e2e_whitespace_issue_description(self, e2e_pipeline: StudentSupportPredictor):
        """Case 1b: Whitespace-only description routes safely to REVIEW."""
        pred = e2e_pipeline.predict(FIXTURE_WHITESPACE_DESCRIPTION)

        assert pred.follow_up_required == FollowUpRequired.REVIEW
        assert pred.priority_score is None
        assert pred.priority == PriorityCategory.NOT_APPLICABLE
        assert pred.model_status == ModelStatus.EMPTY_DESCRIPTION_REVIEW.value

    def test_e2e_ambiguous_description_fees(self, e2e_pipeline: StudentSupportPredictor):
        """Case 2: Ambiguous text ('fees') lacks sufficient context and routes to REVIEW."""
        pred = e2e_pipeline.predict(FIXTURE_AMBIGUOUS_FEES)

        assert pred.follow_up_required == FollowUpRequired.REVIEW
        assert pred.urgency == UrgencyLevel.UNKNOWN
        assert pred.priority_score is None
        assert pred.priority == PriorityCategory.NOT_APPLICABLE
        assert "manual advisor review required" in pred.recommended_action.lower()
        # Ensure automatic callback is never recommended for ambiguous inquiry
        assert "urgent advisor callback" not in pred.recommended_action.lower()

    def test_e2e_explicit_callback_with_sufficient_context(self, e2e_pipeline: StudentSupportPredictor):
        """Case 3: Explicit callback request with detailed blocker yields YES and scorable priority."""
        pred = e2e_pipeline.predict(FIXTURE_EXPLICIT_CALLBACK_WITH_CONTEXT)

        assert pred.follow_up_required == FollowUpRequired.YES
        assert pred.urgency == UrgencyLevel.HIGH
        assert pred.priority == PriorityCategory.HIGH
        assert pred.priority_score is not None
        assert pred.priority_score >= 70.0
        assert "urgent advisor callback" in pred.recommended_action.lower()

    def test_e2e_explicitly_resolved_issue_with_closed_status(self, e2e_pipeline: StudentSupportPredictor):
        """Case 4: Resolved text AND closed status yields NO and suppresses outbound calling."""
        pred = e2e_pipeline.predict(FIXTURE_EXPLICITLY_RESOLVED)

        assert pred.follow_up_required == FollowUpRequired.NO
        assert pred.priority == PriorityCategory.NOT_APPLICABLE
        assert pred.priority_score is None
        assert "no advisor follow-up required" in pred.recommended_action.lower()
        assert "call" not in pred.recommended_action.lower()

    def test_e2e_open_ticket_with_no_actionable_evidence(self, e2e_pipeline: StudentSupportPredictor):
        """Case 5: Open ticket without callback or blocker text routes safely to REVIEW."""
        pred = e2e_pipeline.predict(FIXTURE_OPEN_NO_ACTIONABLE_EVIDENCE)

        assert pred.follow_up_required == FollowUpRequired.REVIEW
        assert pred.priority == PriorityCategory.NOT_APPLICABLE
        assert pred.priority_score is None
        assert "manual advisor review required" in pred.recommended_action.lower()

    def test_e2e_urgent_unresolved_blocker(self, e2e_pipeline: StudentSupportPredictor):
        """Case 6: Urgent unresolved payment blocker without callback yields YES and score < 100."""
        pred = e2e_pipeline.predict(FIXTURE_URGENT_UNRESOLVED_BLOCKER)

        assert pred.follow_up_required == FollowUpRequired.YES
        assert pred.urgency == UrgencyLevel.HIGH
        assert pred.priority == PriorityCategory.HIGH
        assert pred.priority_score is not None
        # Must not automatically assign 100 to every high-urgency ticket
        assert pred.priority_score == 80.0
        assert pred.priority_score < 100.0

    def test_e2e_conflicting_resolution_and_callback_evidence(self, e2e_pipeline: StudentSupportPredictor):
        """Case 7: Contradictory text (resolved + callback) triggers conflict and routes to REVIEW."""
        pred = e2e_pipeline.predict(FIXTURE_CONFLICT_RESOLVED_AND_CALLBACK)

        assert pred.follow_up_required == FollowUpRequired.REVIEW
        assert pred.priority == PriorityCategory.NOT_APPLICABLE
        assert pred.priority_score is None
        assert "conflicting" in pred.follow_up_reason.lower()

    def test_e2e_closed_status_with_unresolved_conflict(self, e2e_pipeline: StudentSupportPredictor):
        """Case 7b: Closed workflow status with active blocker in text triggers conflict REVIEW."""
        pred = e2e_pipeline.predict(FIXTURE_CLOSED_STATUS_WITH_UNRESOLVED_CONFLICT)

        assert pred.follow_up_required == FollowUpRequired.REVIEW
        assert pred.priority == PriorityCategory.NOT_APPLICABLE
        assert pred.priority_score is None
        assert "conflicting" in pred.follow_up_reason.lower()

    def test_e2e_missing_timestamp_safe_handling(self, e2e_pipeline: StudentSupportPredictor):
        """Case 8a: Missing timestamp is safely handled as a non-blocking warning."""
        pred = e2e_pipeline.predict(FIXTURE_MISSING_TIMESTAMP)

        assert isinstance(pred, TicketPrediction)
        assert pred.follow_up_required == FollowUpRequired.YES
        assert pred.priority == PriorityCategory.MEDIUM
        assert pred.priority_score is not None

    def test_e2e_invalid_timestamp_safe_handling(self, e2e_pipeline: StudentSupportPredictor):
        """Case 8b: Corrupted timestamp intercepts safely and routes to REVIEW."""
        pred = e2e_pipeline.predict(FIXTURE_INVALID_TIMESTAMP)

        assert pred.follow_up_required == FollowUpRequired.REVIEW
        assert pred.priority_score is None
        assert pred.priority == PriorityCategory.NOT_APPLICABLE
        assert pred.model_status == ModelStatus.INVALID_TIMESTAMP_REVIEW.value
        assert "invalid timestamp" in pred.follow_up_reason.lower()

    # ==========================================================================
    # 2. Model Loader Fallback & Generic Rejection Scenarios (Task 3 & 9)
    # ==========================================================================

    def test_e2e_generic_pretrained_model_labels_rejection(self):
        """Case 9: Generic pretrained DistilBERT labels (LABEL_0, POSITIVE) are strictly rejected."""
        mock_loader = MagicMock()
        mock_loader.status = ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED
        mock_loader.is_fine_tuned = False

        pipeline = StudentSupportPredictor.create_default_pipeline(model_loader=mock_loader)
        ticket = TicketInput(
            ticket_id="GEN-E2E-01",
            issue_description="Tuition fee inquiry regarding next semester.",
            status="Open",
        )
        pred = pipeline.predict(ticket)

        assert pred.follow_up_required == FollowUpRequired.REVIEW
        assert pred.priority == PriorityCategory.NOT_APPLICABLE
        assert pred.priority_score is None
        assert pred.model_status == ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED.value

    def test_e2e_unavailable_model_dependencies(self):
        """Case 10a: Missing ML dependencies (torch/transformers) produce safe diagnostic status."""
        mock_loader = MagicMock()
        mock_loader.status = ModelStatus.DEPENDENCIES_UNAVAILABLE
        mock_loader.is_fine_tuned = False

        pipeline = StudentSupportPredictor.create_default_pipeline(model_loader=mock_loader)
        ticket = TicketInput(
            ticket_id="DEP-E2E-01",
            issue_description="Elective course registration question.",
            status="Open",
        )
        pred = pipeline.predict(ticket)

        assert pred.follow_up_required == FollowUpRequired.REVIEW
        assert pred.priority_score is None
        assert pred.priority == PriorityCategory.NOT_APPLICABLE
        assert pred.model_status == ModelStatus.DEPENDENCIES_UNAVAILABLE.value

    def test_e2e_missing_model_checkpoint_artifact(self):
        """Case 10b: Non-existent model directory produces safe MODEL_NOT_FOUND diagnostic status."""
        mock_loader = MagicMock()
        mock_loader.status = ModelStatus.MODEL_NOT_FOUND
        mock_loader.is_fine_tuned = False

        pipeline = StudentSupportPredictor.create_default_pipeline(model_loader=mock_loader)
        ticket = TicketInput(
            ticket_id="MIA-E2E-01",
            issue_description="Hostel room allotment status query.",
            status="Open",
        )
        pred = pipeline.predict(ticket)

        assert pred.follow_up_required == FollowUpRequired.REVIEW
        assert pred.priority_score is None
        assert pred.priority == PriorityCategory.NOT_APPLICABLE
        assert pred.model_status == ModelStatus.MODEL_NOT_FOUND.value

    # ==========================================================================
    # 3. Schema Verification & Output Serialization (Task 4)
    # ==========================================================================

    def test_e2e_schema_validation_and_field_completeness_all_fixtures(
        self, e2e_pipeline: StudentSupportPredictor
    ):
        """Task 4: Verify every fixture produces a valid TicketPrediction instance adhering to schema constraints."""
        for fixture in ALL_SYNTHETIC_FIXTURES:
            pred = e2e_pipeline.predict(fixture)

            # Type and schema contract
            assert isinstance(pred, TicketPrediction)

            # Enum membership validation
            assert isinstance(pred.follow_up_required, FollowUpRequired) or pred.follow_up_required in FollowUpRequired._value2member_map_
            assert isinstance(pred.interest_level, InterestLevel) or pred.interest_level in InterestLevel._value2member_map_
            assert isinstance(pred.urgency, UrgencyLevel) or pred.urgency in UrgencyLevel._value2member_map_
            assert isinstance(pred.priority, PriorityCategory) or pred.priority in PriorityCategory._value2member_map_

            # Priority score constraint [0.0 - 100.0] or None
            if pred.priority_score is not None:
                assert isinstance(pred.priority_score, (int, float))
                assert 0.0 <= pred.priority_score <= 100.0

            # Mandatory text fields
            assert isinstance(pred.recommended_action, str)
            assert len(pred.recommended_action.strip()) > 0
            assert isinstance(pred.model_status, str)
            assert len(pred.model_status.strip()) > 0

            # Serialization round-trip validation
            dumped_dict = pred.model_dump()
            assert isinstance(dumped_dict, dict)
            dumped_json = pred.model_dump_json()
            assert isinstance(dumped_json, str)

            # Deserialize and verify round-trip integrity
            re_parsed = TicketPrediction.model_validate_json(dumped_json)
            assert re_parsed.follow_up_required == pred.follow_up_required
            assert re_parsed.priority_score == pred.priority_score
            assert re_parsed.priority == pred.priority

    # ==========================================================================
    # 4. PII & Constant Priority Invariance (Task 5)
    # ==========================================================================

    def test_e2e_pii_and_legacy_priority_invariance(self, e2e_pipeline: StudentSupportPredictor):
        """Task 5: Modifying PII fields or priority_level must NOT alter triage decisions or priority scores."""
        desc = "Deadline is tomorrow and fee was deducted twice, unable to submit application. Please call me back urgently."

        variations = [
            {"student_name": "Alice Green", "application_number": "APP001", "phone_number": "9111111111", "priority_level": "High", "ticket_id": "T1"},
            {"student_name": "Bob Brown", "application_number": "REG999", "phone_number": "9222222222", "priority_level": "Low", "ticket_id": "T2"},
            {"student_name": "Charlie Davis", "application_number": "APP888", "phone_number": "9333333333", "priority_level": "Critical", "ticket_id": "T3"},
            {"student_name": None, "application_number": None, "phone_number": None, "priority_level": None, "ticket_id": None},
            {"student_name": "", "application_number": "", "phone_number": "", "priority_level": "", "ticket_id": ""},
        ]

        predictions = []
        for var in variations:
            payload = {
                "issue_description": desc,
                "status": "Open",
                **var,
            }
            predictions.append(e2e_pipeline.predict(payload))

        baseline = predictions[0]
        for pred in predictions[1:]:
            assert pred.follow_up_required == baseline.follow_up_required
            assert pred.interest_level == baseline.interest_level
            assert pred.urgency == baseline.urgency
            assert pred.priority_score == baseline.priority_score
            assert pred.priority == baseline.priority
            assert pred.recommended_action == baseline.recommended_action

    # ==========================================================================
    # 5. Follow-Up Eligibility Enforcement on Actions (Task 6)
    # ==========================================================================

    def test_e2e_follow_up_no_and_review_never_trigger_call(self, e2e_pipeline: StudentSupportPredictor):
        """Task 6: NO and REVIEW must strictly prevent automatic call recommendations."""
        # Test all fixtures that result in NO or REVIEW
        no_or_review_fixtures = [
            FIXTURE_EXPLICITLY_RESOLVED,                  # NO
            FIXTURE_AMBIGUOUS_FEES,                       # REVIEW
            FIXTURE_OPEN_NO_ACTIONABLE_EVIDENCE,          # REVIEW
            FIXTURE_CONFLICT_RESOLVED_AND_CALLBACK,       # REVIEW
            FIXTURE_CLOSED_STATUS_WITH_UNRESOLVED_CONFLICT, # REVIEW
            FIXTURE_EMPTY_DESCRIPTION,                    # REVIEW
        ]

        for fix in no_or_review_fixtures:
            pred = e2e_pipeline.predict(fix)
            if pred.follow_up_required == FollowUpRequired.NO:
                assert pred.priority == PriorityCategory.NOT_APPLICABLE
                assert pred.priority_score is None
                assert "no advisor follow-up required" in pred.recommended_action.lower()
                assert "call" not in pred.recommended_action.lower()
            elif pred.follow_up_required == FollowUpRequired.REVIEW:
                assert pred.priority == PriorityCategory.NOT_APPLICABLE
                assert pred.priority_score is None
                # Must not schedule an automatic call
                assert "urgent advisor callback" not in pred.recommended_action.lower()
                assert "standard advisor callback" not in pred.recommended_action.lower()
                assert "outbound call" not in pred.recommended_action.lower()

    # ==========================================================================
    # 6. Follow-Up YES with Insufficient Evidence (Task 7)
    # ==========================================================================

    def test_e2e_follow_up_yes_insufficient_evidence_yields_null_score(
        self, e2e_pipeline: StudentSupportPredictor
    ):
        """Task 7: Follow-up YES does not guarantee a non-null score if issue context is missing."""
        pred = e2e_pipeline.predict(FIXTURE_CALLBACK_WITHOUT_ISSUE_DETAILS)

        assert pred.follow_up_required == FollowUpRequired.YES
        assert pred.urgency == UrgencyLevel.UNKNOWN
        assert pred.priority_score is None
        assert pred.priority == PriorityCategory.NOT_APPLICABLE
        assert "advisor follow-up required" in pred.recommended_action.lower()
        assert "context missing" in pred.recommended_action.lower()

    # ==========================================================================
    # 7. Batch Prediction Coherence (Task 2)
    # ==========================================================================

    def test_e2e_batch_prediction_coherence(self, e2e_pipeline: StudentSupportPredictor):
        """Task 2: Batch prediction processes diverse records sequentially and preserves order."""
        batch = [
            FIXTURE_EXPLICIT_CALLBACK_WITH_CONTEXT,
            FIXTURE_EXPLICITLY_RESOLVED,
            FIXTURE_EMPTY_DESCRIPTION,
            FIXTURE_URGENT_UNRESOLVED_BLOCKER,
        ]
        results = e2e_pipeline.predict_batch(batch)

        assert len(results) == 4
        assert results[0].follow_up_required == FollowUpRequired.YES
        assert results[0].priority == PriorityCategory.HIGH
        assert results[1].follow_up_required == FollowUpRequired.NO
        assert results[1].priority == PriorityCategory.NOT_APPLICABLE
        assert results[2].follow_up_required == FollowUpRequired.REVIEW
        assert results[2].priority == PriorityCategory.NOT_APPLICABLE
        assert results[3].follow_up_required == FollowUpRequired.YES
        assert results[3].priority == PriorityCategory.HIGH
