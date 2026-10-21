"""Synthetic test fixtures for ML Service unit and end-to-end testing.

================================================================================
CRITICAL DISCLAIMER & NOTICE:
These fixtures are synthetic, hand-crafted mock test cases created exclusively
for unit and regression testing of software components in `ml_service`.

THEY DO NOT REPRESENT:
- Real student inquiries, records, or personal data (PII).
- Historical university datasets or Google Sheets records.
- Statistical training, fine-tuning, or evaluation datasets.
- Proof of triage accuracy or production readiness.

All names, application IDs, telephone numbers, and query texts are entirely fictional.
================================================================================
"""

from typing import Any, Dict

# 1. Empty and whitespace descriptions
FIXTURE_EMPTY_DESCRIPTION: Dict[str, Any] = {
    "ticket_id": "SYN-EMP-01",
    "student_name": "Student A",
    "application_number": "APP00001",
    "phone_number": "9000000001",
    "issue_description": "",
    "priority_level": "High",
    "status": "Open",
}

FIXTURE_WHITESPACE_DESCRIPTION: Dict[str, Any] = {
    "ticket_id": "SYN-EMP-02",
    "student_name": "Student B",
    "application_number": "APP00002",
    "phone_number": "9000000002",
    "issue_description": "   \n\t  ",
    "priority_level": "High",
    "status": "Open",
}

# 2. Ambiguous and brief description
FIXTURE_AMBIGUOUS_FEES: Dict[str, Any] = {
    "ticket_id": "SYN-AMB-01",
    "student_name": "Student C",
    "application_number": "APP00003",
    "phone_number": "9000000003",
    "issue_description": "fees",
    "priority_level": "High",
    "status": "Open",
}

# 3. Explicit callback request with sufficient issue context
FIXTURE_EXPLICIT_CALLBACK_WITH_CONTEXT: Dict[str, Any] = {
    "ticket_id": "SYN-CB-01",
    "student_name": "Student D",
    "application_number": "APP00004",
    "phone_number": "9000000004",
    "issue_description": "Tuition fee was deducted twice from my bank account and portal gave an error. Please call me back urgently.",
    "priority_level": "High",
    "status": "Open",
}

# 4. Explicitly resolved issue with appropriate closed status
FIXTURE_EXPLICITLY_RESOLVED: Dict[str, Any] = {
    "ticket_id": "SYN-RES-01",
    "student_name": "Student E",
    "application_number": "APP00005",
    "phone_number": "9000000005",
    "issue_description": "The fee receipt issue has been resolved successfully. Received my refund, no further help needed.",
    "priority_level": "High",
    "status": "Closed",
}

# 5. Open ticket with no actionable evidence (informational query)
FIXTURE_OPEN_NO_ACTIONABLE_EVIDENCE: Dict[str, Any] = {
    "ticket_id": "SYN-OPN-01",
    "student_name": "Student F",
    "application_number": "APP00006",
    "phone_number": "9000000006",
    "issue_description": "General inquiry regarding the academic calendar and syllabus for next semester.",
    "priority_level": "High",
    "status": "Open",
}

# 6. Urgent unresolved blocker (without explicit callback request)
FIXTURE_URGENT_UNRESOLVED_BLOCKER: Dict[str, Any] = {
    "ticket_id": "SYN-URG-01",
    "student_name": "Student G",
    "application_number": "APP00007",
    "phone_number": "9000000007",
    "issue_description": "Payment was deducted but status is still failing and portal threw an error.",
    "priority_level": "High",
    "status": "Open",
}

# 7. Conflicting resolution and callback evidence
FIXTURE_CONFLICT_RESOLVED_AND_CALLBACK: Dict[str, Any] = {
    "ticket_id": "SYN-CNF-01",
    "student_name": "Student H",
    "application_number": "APP00008",
    "phone_number": "9000000008",
    "issue_description": "The issue has been resolved successfully, but please call me back tomorrow anyway.",
    "priority_level": "High",
    "status": "Open",
}

# 8. Missing and invalid timestamps
FIXTURE_INVALID_TIMESTAMP: Dict[str, Any] = {
    "ticket_id": "SYN-TS-01",
    "student_name": "Student I",
    "application_number": "APP00009",
    "phone_number": "9000000009",
    "date_and_time": "invalid-timestamp-2026-corrupt",
    "issue_description": "Tuition fee payment confirmation pending.",
    "priority_level": "High",
    "status": "Open",
}

FIXTURE_MISSING_TIMESTAMP: Dict[str, Any] = {
    "ticket_id": "SYN-TS-02",
    "student_name": "Student J",
    "application_number": "APP00010",
    "phone_number": "9000000010",
    "date_and_time": None,
    "issue_description": "Need assistance with scholarship application document verification for next month's deadline. Please call me back.",
    "priority_level": "High",
    "status": "Open",
}

# 9. Callback requested without issue details (insufficient evidence)
FIXTURE_CALLBACK_WITHOUT_ISSUE_DETAILS: Dict[str, Any] = {
    "ticket_id": "SYN-INS-01",
    "student_name": "Student K",
    "application_number": "APP00011",
    "phone_number": "9000000011",
    "issue_description": "Please call me back.",
    "priority_level": "High",
    "status": "Open",
}

# 10. Expressed interest variations
FIXTURE_HIGH_INTEREST_EXPRESSED: Dict[str, Any] = {
    "ticket_id": "SYN-INT-01",
    "student_name": "Student L",
    "application_number": "APP00012",
    "phone_number": "9000000012",
    "issue_description": "I am eager to join VIT-AP and want to accept my admission offer. Deadline is tomorrow, please call me.",
    "priority_level": "High",
    "status": "Open",
}

FIXTURE_LOW_INTEREST_EXPRESSED: Dict[str, Any] = {
    "ticket_id": "SYN-INT-02",
    "student_name": "Student M",
    "application_number": "APP00013",
    "phone_number": "9000000013",
    "issue_description": "I wish to withdraw my application as I am joining another university. Please call me back regarding my refund.",
    "priority_level": "High",
    "status": "Open",
}

# 11. Closed status with ongoing unresolved blocker conflict
FIXTURE_CLOSED_STATUS_WITH_UNRESOLVED_CONFLICT: Dict[str, Any] = {
    "ticket_id": "SYN-CNF-02",
    "student_name": "Student N",
    "application_number": "APP00014",
    "phone_number": "9000000014",
    "issue_description": "Ticket is marked closed but my issue is not resolved, payment was deducted twice.",
    "priority_level": "High",
    "status": "Closed",
}

# Collection of all test fixtures for batch and serialization verification
ALL_SYNTHETIC_FIXTURES = [
    FIXTURE_EMPTY_DESCRIPTION,
    FIXTURE_WHITESPACE_DESCRIPTION,
    FIXTURE_AMBIGUOUS_FEES,
    FIXTURE_EXPLICIT_CALLBACK_WITH_CONTEXT,
    FIXTURE_EXPLICITLY_RESOLVED,
    FIXTURE_OPEN_NO_ACTIONABLE_EVIDENCE,
    FIXTURE_URGENT_UNRESOLVED_BLOCKER,
    FIXTURE_CONFLICT_RESOLVED_AND_CALLBACK,
    FIXTURE_INVALID_TIMESTAMP,
    FIXTURE_MISSING_TIMESTAMP,
    FIXTURE_CALLBACK_WITHOUT_ISSUE_DETAILS,
    FIXTURE_HIGH_INTEREST_EXPRESSED,
    FIXTURE_LOW_INTEREST_EXPRESSED,
    FIXTURE_CLOSED_STATUS_WITH_UNRESOLVED_CONFLICT,
]
