"""Input validation and sanitization for Student Support ML Module.

Implements safe validation for incoming ticket data, specifically handling:
- Empty or whitespace-only issue descriptions.
- Invalid, corrupted, or non-standard timestamps.
- Missing optional fields.
- Type coercion from dictionaries or external JSON payloads.
"""

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Union

try:
    from dateutil import parser as dateutil_parser
except ImportError:
    dateutil_parser = None

from .schemas import TicketInput


@dataclass
class ValidationResult:
    """Detailed result of ticket data validation.
    
    Attributes:
        is_valid: True if input meets all requirements for ML inference.
        cleaned_ticket: Sanitized TicketInput object if valid or partially valid.
        parsed_timestamp: Standardized datetime object if date_and_time was provided and valid.
        errors: List of blocking error messages that prevent ML inference.
        warnings: Non-blocking warning messages (e.g., missing optional fields).
        empty_description: Flag indicating if issue description was missing/empty.
        invalid_timestamp: Flag indicating if a provided timestamp could not be parsed.
    """
    is_valid: bool
    cleaned_ticket: Optional[TicketInput] = None
    parsed_timestamp: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    empty_description: bool = False
    invalid_timestamp: bool = False


class TicketValidator:
    """Validates and sanitizes incoming ticket records before ML inference."""

    @staticmethod
    def parse_timestamp(value: Any) -> tuple[Optional[datetime], Optional[str]]:
        """Safely parse a timestamp into a datetime object without raising exceptions.
        
        Returns:
            tuple: (parsed_datetime_or_None, error_message_or_None)
        """
        if value is None:
            return None, None

        if isinstance(value, datetime):
            return value, None

        if not isinstance(value, str):
            return None, f"Timestamp must be string or datetime, received {type(value).__name__}"

        stripped_val = value.strip()
        if not stripped_val:
            return None, None

        # 1. Attempt ISO 8601 parsing via standard library
        try:
            # Handle trailing 'Z' if present
            iso_val = stripped_val.replace("Z", "+00:00")
            return datetime.fromisoformat(iso_val), None
        except (ValueError, TypeError):
            pass

        # 2. Attempt dateutil parser if available
        if dateutil_parser is not None:
            try:
                dt = dateutil_parser.parse(stripped_val)
                return dt, None
            except (ValueError, TypeError, OverflowError) as exc:
                return None, f"Unable to parse timestamp '{value}': {str(exc)}"

        # 3. Fallback standard formats
        common_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %I:%M %p",
            "%m/%d/%Y %I:%M %p",
            "%d-%m-%Y %H:%M:%S",
        ]
        for fmt in common_formats:
            try:
                return datetime.strptime(stripped_val, fmt), None
            except (ValueError, TypeError):
                continue

        return None, f"Invalid timestamp format: '{value}'. Expected standard date or ISO format."

    @classmethod
    def validate(cls, raw_input: Union[TicketInput, Dict[str, Any], Any]) -> ValidationResult:
        """Validate input data against schemas and operational safety rules.
        
        Rules enforced:
        - Must be coercible to TicketInput.
        - `issue_description` must be non-empty and contain meaningful text.
        - `date_and_time`, if provided, must be a parseable timestamp.
        - Missing optional fields are handled safely and noted as warnings if relevant.
        """
        errors: List[str] = []
        warnings: List[str] = []
        empty_description = False
        invalid_timestamp = False

        if raw_input is None:
            return ValidationResult(
                is_valid=False,
                errors=["Input payload cannot be None."],
                empty_description=True,
            )

        # 1. Coerce to TicketInput model
        ticket: TicketInput
        if isinstance(raw_input, TicketInput):
            ticket = raw_input
        elif isinstance(raw_input, dict):
            try:
                ticket = TicketInput(**raw_input)
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Schema validation error: {str(e)}"],
                )
        else:
            return ValidationResult(
                is_valid=False,
                errors=[f"Expected TicketInput or dict, got {type(raw_input).__name__}."],
            )

        # 2. Validate issue_description
        raw_desc = ticket.issue_description
        if raw_desc is None or not isinstance(raw_desc, str) or not raw_desc.strip():
            empty_description = True
            errors.append(
                "Missing or empty issue_description: Cannot perform NLP classification "
                "without textual description."
            )
            cleaned_desc = None
        else:
            cleaned_desc = raw_desc.strip()

        # 3. Validate timestamp safely
        parsed_dt, ts_err = cls.parse_timestamp(ticket.date_and_time)
        if ts_err is not None:
            invalid_timestamp = True
            errors.append(ts_err)
        elif ticket.date_and_time is None:
            warnings.append("date_and_time was not provided; timestamp is null.")

        # 4. Record warnings for missing metadata
        if not ticket.ticket_id:
            warnings.append("ticket_id is missing or null.")
        if not ticket.application_number and not ticket.registration_number:
            warnings.append("Student application/registration number is missing.")

        # 5. Build sanitized copy
        sanitized_ticket = TicketInput(
            ticket_id=ticket.ticket_id.strip() if ticket.ticket_id else None,
            date_and_time=parsed_dt if parsed_dt is not None else ticket.date_and_time,
            student_name=ticket.student_name.strip() if ticket.student_name else None,
            application_number=ticket.application_number.strip() if ticket.application_number else None,
            phone_number=ticket.phone_number.strip() if ticket.phone_number else None,
            issue_description=cleaned_desc,
            priority_level=ticket.priority_level,
            status=ticket.status.strip() if ticket.status else None,
        )

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            cleaned_ticket=sanitized_ticket,
            parsed_timestamp=parsed_dt,
            errors=errors,
            warnings=warnings,
            empty_description=empty_description,
            invalid_timestamp=invalid_timestamp,
        )


def validate_ticket_input(raw_input: Union[TicketInput, Dict[str, Any], Any]) -> ValidationResult:
    """Convenience functional wrapper around TicketValidator.validate."""
    return TicketValidator.validate(raw_input)
