"""Unit tests for input and output schemas in ml_service."""

from datetime import datetime
import pytest
from pydantic import ValidationError

from ml_service.schemas import (
    FollowUpRequired,
    InterestLevel,
    ModelStatus,
    PriorityCategory,
    TicketInput,
    TicketPrediction,
    UrgencyLevel,
)


class TestTicketInputSchema:
    """Test suite for TicketInput schema validation and aliases."""

    def test_full_input_instantiation(self):
        """Verify all fields populate properly when provided."""
        ticket = TicketInput(
            ticket_id="John1",
            date_and_time="2026-10-20T14:30:00",
            student_name="John Doe",
            application_number="APP123456",
            phone_number="9876543210",
            issue_description="Seeking clarification on hostel allotment fees.",
            priority_level="High",
            status="Open",
        )
        assert ticket.ticket_id == "John1"
        assert ticket.student_name == "John Doe"
        assert ticket.application_number == "APP123456"
        assert ticket.phone_number == "9876543210"
        assert ticket.issue_description == "Seeking clarification on hostel allotment fees."
        assert ticket.priority_level == "High"
        assert ticket.status == "Open"

    def test_missing_optional_fields_safe(self):
        """Verify that missing optional fields default safely to None."""
        ticket = TicketInput()
        assert ticket.ticket_id is None
        assert ticket.date_and_time is None
        assert ticket.student_name is None
        assert ticket.application_number is None
        assert ticket.phone_number is None
        assert ticket.issue_description is None
        assert ticket.priority_level is None
        assert ticket.status is None

    def test_registration_number_alias_sync(self):
        """Verify registration_number populates application_number if application_number is missing."""
        ticket = TicketInput(
            registration_number="REG-9999",
            issue_description="Course registration portal error",
        )
        assert ticket.application_number == "REG-9999"
        assert ticket.registration_number == "REG-9999"

    def test_extra_fields_ignored_safely(self):
        """Verify extra unexpected fields from upstream do not cause validation error."""
        data = {
            "ticket_id": "Alice2",
            "issue_description": "Fee payment failed",
            "unrelated_field_123": "unexpected_data",
        }
        ticket = TicketInput(**data)
        assert ticket.ticket_id == "Alice2"
        assert not hasattr(ticket, "unrelated_field_123") or getattr(ticket, "unrelated_field_123", None) is None


class TestTicketPredictionSchema:
    """Test suite for TicketPrediction structured output schema."""

    def test_valid_prediction_structure(self):
        """Verify valid structured prediction output matches required fields."""
        pred = TicketPrediction(
            ticket_id="John1",
            follow_up_required=FollowUpRequired.REVIEW,
            interest_level=InterestLevel.UNKNOWN,
            urgency=UrgencyLevel.UNKNOWN,
            follow_up_reason="Manual verification required",
            priority_score=None,
            priority=PriorityCategory.NOT_APPLICABLE,
            recommended_action="Route to human advisor",
            model_status=ModelStatus.PLACEHOLDER_PENDING_MODEL.value,
        )
        assert pred.ticket_id == "John1"
        assert pred.follow_up_required == "REVIEW"
        assert pred.interest_level == "UNKNOWN"
        assert pred.urgency == "UNKNOWN"
        assert pred.priority_score is None
        assert pred.priority == "NOT_APPLICABLE"
        assert pred.recommended_action == "Route to human advisor"
        assert pred.model_status == "PLACEHOLDER_PENDING_MODEL"

    def test_valid_priority_score_boundaries(self):
        """Verify priority_score accepts valid floats between 0.0 and 100.0."""
        # 0.0 lower bound
        pred_low = TicketPrediction(
            recommended_action="Triage",
            model_status="TEST",
            priority_score=0.0,
        )
        assert pred_low.priority_score == 0.0

        # 100.0 upper bound
        pred_high = TicketPrediction(
            recommended_action="Triage",
            model_status="TEST",
            priority_score=100.0,
        )
        assert pred_high.priority_score == 100.0

        # Midpoint
        pred_mid = TicketPrediction(
            recommended_action="Triage",
            model_status="TEST",
            priority_score=55.5,
        )
        assert pred_mid.priority_score == 55.5

    def test_invalid_priority_score_rejected(self):
        """Verify priority_score rejects values < 0 or > 100."""
        with pytest.raises(ValidationError):
            TicketPrediction(
                recommended_action="Triage",
                model_status="TEST",
                priority_score=-1.0,
            )

        with pytest.raises(ValidationError):
            TicketPrediction(
                recommended_action="Triage",
                model_status="TEST",
                priority_score=100.1,
            )

    def test_serialization_dict_and_json(self):
        """Verify model serializes cleanly to python dict and json string."""
        pred = TicketPrediction(
            ticket_id="T100",
            follow_up_required=FollowUpRequired.NO,
            interest_level=InterestLevel.HIGH,
            urgency=UrgencyLevel.LOW,
            follow_up_reason="Automated resolution",
            priority_score=25.0,
            priority=PriorityCategory.LOW,
            recommended_action="Send automated email confirmation",
            model_status="TEST_STATUS",
        )
        dump = pred.model_dump()
        assert dump["ticket_id"] == "T100"
        assert dump["follow_up_required"] == "NO"
        assert dump["priority_score"] == 25.0
        assert isinstance(pred.model_dump_json(), str)
