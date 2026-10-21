"""Unit tests for StudentSupportPredictor placeholder interface."""

import pytest

from ml_service.config import MLServiceConfig
from ml_service.predictor import PlaceholderPredictor, StudentSupportPredictor
from ml_service.schemas import (
    FollowUpRequired,
    InterestLevel,
    ModelStatus,
    PriorityCategory,
    TicketInput,
    TicketPrediction,
    UrgencyLevel,
)


class TestStudentSupportPredictor:
    """Test suite for predictor initialization, safety guarantees, and placeholder inference."""

    def test_initialization_without_model_download(self):
        """Verify predictor initializes cleanly without loading model weights or downloading."""
        config = MLServiceConfig(model_name="distilbert-base-uncased")
        predictor = StudentSupportPredictor(config=config)
        assert predictor.config.model_name == "distilbert-base-uncased"
        assert predictor.config.placeholder_mode is True

    def test_alias_placeholder_predictor(self):
        """Verify PlaceholderPredictor is an alias of StudentSupportPredictor."""
        assert PlaceholderPredictor is StudentSupportPredictor

    def test_valid_input_returns_explicit_placeholder_review(self):
        """Verify valid input produces structured REVIEW placeholder without inventing scores."""
        predictor = StudentSupportPredictor()
        ticket = TicketInput(
            ticket_id="VIT202",
            student_name="Rahul Varma",
            application_number="26BCE123",
            phone_number="9123456780",
            issue_description="Requesting information on branch change eligibility and deadlines.",
            priority_level="High",
            status="Open",
        )
        prediction: TicketPrediction = predictor.predict(ticket)

        assert isinstance(prediction, TicketPrediction)
        assert prediction.ticket_id == "VIT202"
        # No hallucinated conclusions; must be explicit REVIEW
        assert prediction.follow_up_required == FollowUpRequired.REVIEW
        assert prediction.interest_level == InterestLevel.UNKNOWN
        assert prediction.urgency == UrgencyLevel.UNKNOWN
        # Defensible constraint: score must be null until defensible scoring exists
        assert prediction.priority_score is None
        assert prediction.priority == PriorityCategory.NOT_APPLICABLE
        assert prediction.model_status == ModelStatus.PLACEHOLDER_PENDING_MODEL.value
        assert "not yet trained or deployed" in prediction.follow_up_reason

    def test_dict_input_returns_prediction(self):
        """Verify dictionary input works smoothly with predict()."""
        predictor = StudentSupportPredictor()
        payload = {
            "ticket_id": "VIT500",
            "issue_description": "Can I submit my TC certificate two weeks late?",
        }
        pred = predictor.predict(payload)
        assert pred.ticket_id == "VIT500"
        assert pred.follow_up_required == FollowUpRequired.REVIEW
        assert pred.priority_score is None

    def test_empty_description_safe_review_handling(self):
        """Verify empty issue description produces safe REVIEW without error."""
        predictor = StudentSupportPredictor()
        ticket = TicketInput(
            ticket_id="VIT303",
            issue_description="",
        )
        pred = predictor.predict(ticket)
        assert pred.ticket_id == "VIT303"
        assert pred.follow_up_required == FollowUpRequired.REVIEW
        assert pred.priority_score is None
        assert pred.model_status == ModelStatus.EMPTY_DESCRIPTION_REVIEW.value
        assert "missing or empty" in pred.follow_up_reason.lower()

    def test_invalid_timestamp_safe_review_handling(self):
        """Verify corrupted timestamp produces safe REVIEW without error."""
        predictor = StudentSupportPredictor()
        ticket = TicketInput(
            ticket_id="VIT404",
            issue_description="Fee refund inquiry after withdrawal.",
            date_and_time="invalid-date-string-321",
        )
        pred = predictor.predict(ticket)
        assert pred.ticket_id == "VIT404"
        assert pred.follow_up_required == FollowUpRequired.REVIEW
        assert pred.priority_score is None
        assert pred.model_status == ModelStatus.INVALID_TIMESTAMP_REVIEW.value
        assert "invalid timestamp" in pred.follow_up_reason.lower()

    def test_batch_prediction(self):
        """Verify predict_batch processes a collection of inputs preserving order."""
        predictor = StudentSupportPredictor()
        batch = [
            {"ticket_id": "B1", "issue_description": "Query 1 regarding semester schedule"},
            {"ticket_id": "B2", "issue_description": ""},
            {"ticket_id": "B3", "issue_description": "Query 3 regarding bus passes"},
        ]
        results = predictor.predict_batch(batch)
        assert len(results) == 3
        assert results[0].ticket_id == "B1"
        assert results[0].model_status == ModelStatus.PLACEHOLDER_PENDING_MODEL.value
        assert results[1].ticket_id == "B2"
        assert results[1].model_status == ModelStatus.EMPTY_DESCRIPTION_REVIEW.value
        assert results[2].ticket_id == "B3"
        assert results[2].model_status == ModelStatus.PLACEHOLDER_PENDING_MODEL.value

    def test_no_bias_on_identifiers_or_legacy_priority(self):
        """Verify changing student identifiers or legacy priority does not alter placeholder output."""
        predictor = StudentSupportPredictor()
        ticket_a = TicketInput(
            ticket_id="T1",
            student_name="Student A",
            phone_number="1111111111",
            priority_level="High",
            issue_description="Need scholarship guidance",
        )
        ticket_b = TicketInput(
            ticket_id="T1",
            student_name="Student B",
            phone_number="9999999999",
            priority_level="Normal",
            issue_description="Need scholarship guidance",
        )
        res_a = predictor.predict(ticket_a)
        res_b = predictor.predict(ticket_b)

        assert res_a.priority_score == res_b.priority_score is None
        assert res_a.follow_up_required == res_b.follow_up_required
        assert res_a.interest_level == res_b.interest_level
        assert res_a.urgency == res_b.urgency
