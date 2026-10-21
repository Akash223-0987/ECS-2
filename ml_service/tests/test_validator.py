"""Unit tests for input validation and timestamp parsing in ml_service."""

from datetime import datetime
import pytest

from ml_service.schemas import TicketInput
from ml_service.validator import TicketValidator, ValidationResult, validate_ticket_input


class TestTicketValidator:
    """Test suite for TicketValidator logic and edge cases."""

    def test_valid_ticket_passes_validation(self):
        """Verify complete ticket with valid fields passes validation."""
        ticket = TicketInput(
            ticket_id="VIT101",
            date_and_time="2026-10-20T10:00:00",
            student_name="Aarav Sharma",
            application_number="26BCE001",
            phone_number="9988776655",
            issue_description="Inquiry regarding B.Tech Computer Science hostel allotment.",
            priority_level="High",
            status="Open",
        )
        result = validate_ticket_input(ticket)
        assert result.is_valid is True
        assert result.empty_description is False
        assert result.invalid_timestamp is False
        assert len(result.errors) == 0
        assert isinstance(result.parsed_timestamp, datetime)
        assert result.cleaned_ticket is not None
        assert result.cleaned_ticket.ticket_id == "VIT101"

    def test_valid_ticket_with_minimal_fields(self):
        """Verify ticket with only issue_description passes validation safely."""
        ticket = TicketInput(
            issue_description="How do I pay my term fees online?",
        )
        result = validate_ticket_input(ticket)
        assert result.is_valid is True
        assert result.empty_description is False
        assert result.parsed_timestamp is None
        # Non-blocking warnings are present for missing optional fields
        assert any("date_and_time was not provided" in w for w in result.warnings)
        assert any("ticket_id is missing" in w for w in result.warnings)

    def test_empty_string_issue_description(self):
        """Verify empty string issue description is flagged as error."""
        ticket = TicketInput(
            ticket_id="T01",
            issue_description="",
        )
        result = validate_ticket_input(ticket)
        assert result.is_valid is False
        assert result.empty_description is True
        assert any("Missing or empty issue_description" in err for err in result.errors)

    def test_whitespace_only_issue_description(self):
        """Verify whitespace-only issue description is treated as empty description."""
        ticket = TicketInput(
            ticket_id="T02",
            issue_description="   \n\t   ",
        )
        result = validate_ticket_input(ticket)
        assert result.is_valid is False
        assert result.empty_description is True
        assert any("Missing or empty issue_description" in err for err in result.errors)

    def test_none_issue_description(self):
        """Verify None issue description is treated as empty description."""
        ticket = TicketInput(
            ticket_id="T03",
            issue_description=None,
        )
        result = validate_ticket_input(ticket)
        assert result.is_valid is False
        assert result.empty_description is True

    def test_valid_iso_timestamp_parsing(self):
        """Verify ISO-8601 timestamps parse cleanly."""
        dt, err = TicketValidator.parse_timestamp("2026-10-20T14:30:00")
        assert err is None
        assert dt == datetime(2026, 10, 20, 14, 30, 0)

    def test_valid_custom_date_format_parsing(self):
        """Verify common custom date format parses cleanly."""
        dt, err = TicketValidator.parse_timestamp("2026-10-20 09:15:00")
        assert err is None
        assert dt == datetime(2026, 10, 20, 9, 15, 0)

    def test_datetime_object_preserved(self):
        """Verify datetime object passed directly is handled without re-parsing."""
        now = datetime.now()
        dt, err = TicketValidator.parse_timestamp(now)
        assert err is None
        assert dt == now

    def test_none_or_empty_timestamp_is_safe(self):
        """Verify None or empty string timestamp returns None without error."""
        dt, err = TicketValidator.parse_timestamp(None)
        assert dt is None
        assert err is None

        dt2, err2 = TicketValidator.parse_timestamp("   ")
        assert dt2 is None
        assert err2 is None

    def test_invalid_corrupted_timestamp_handled_safely(self):
        """Verify invalid timestamp string reports error safely without raising unhandled exception."""
        dt, err = TicketValidator.parse_timestamp("not-a-valid-date-format-999")
        assert dt is None
        assert err is not None
        assert "Invalid timestamp format" in err or "Unable to parse" in err

        # Validate whole ticket with corrupted timestamp
        ticket = TicketInput(
            issue_description="Valid student query about exam schedule",
            date_and_time="corrupted-date-xyz",
        )
        result = validate_ticket_input(ticket)
        assert result.is_valid is False
        assert result.invalid_timestamp is True
        assert result.empty_description is False
        assert len(result.errors) > 0

    def test_dict_input_coercion(self):
        """Verify raw dictionary input is automatically parsed and validated."""
        raw_dict = {
            "ticket_id": "D101",
            "issue_description": "Can I change my hostel room preference?",
            "registration_number": "26BCS002",
        }
        result = validate_ticket_input(raw_dict)
        assert result.is_valid is True
        assert result.cleaned_ticket.application_number == "26BCS002"

    def test_none_input_handled_safely(self):
        """Verify None input does not crash the validator."""
        result = validate_ticket_input(None)
        assert result.is_valid is False
        assert result.empty_description is True
        assert "Input payload cannot be None." in result.errors
