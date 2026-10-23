"""Unit and Integration Tests for Google Sheets Watcher Service.

Tests:
1. MockGoogleSheetsClient operations (get_all_values, update_cell).
2. Header column resolution (transcription, priority, ticket_id, status).
3. Null/missing priority detection.
4. Detection of new rows and writing predicted priority back to the correct cell.
5. Invariance of rows with existing priority (not overwritten).
6. Duplicate processing prevention across poll cycles.
7. Safe error handling (transient network errors, malformed rows).
8. Polling loop lifecycle and graceful termination.
9. Priority case formatting (UPPER vs TITLE).
"""

import logging
import pytest

from ml_service.predictor import PriorityPredictor
from ml_service.schemas import PriorityInferenceOutput
from ml_service.sheets_config import SheetsWatcherConfig
from ml_service.sheets_watcher import (
    GoogleSheetsWatcher,
    MockGoogleSheetsClient,
    RowUpdateReport,
)


@pytest.fixture
def sample_headers():
    return [
        "ticket_id",
        "date_and_time",
        "student_name",
        "registration_number",
        "phone_number",
        "issue_transcription",
        "priority_level",
        "status",
    ]


@pytest.fixture
def sample_sheet_data(sample_headers):
    return [
        sample_headers,
        [
            "John1",
            "2026-10-23 10:00:00",
            "John Doe",
            "REG101",
            "9876543210",
            "Fee debited twice and deadline is today!",
            "",  # Missing priority
            "Open",
        ],
        [
            "John2",
            "2026-10-23 10:05:00",
            "Jane Smith",
            "REG102",
            "9876543211",
            "General question regarding university campus map and facilities",
            "null",  # Missing priority as string 'null'
            "Open",
        ],
        [
            "John3",
            "2026-10-23 10:10:00",
            "Bob Davis",
            "REG103",
            "9876543212",
            "Hostel room change request",
            "High",  # Pre-existing priority: MUST NOT be overwritten
            "Open",
        ],
    ]


class TestMockGoogleSheetsClient:
    """Validate in-memory mock client operations."""

    def test_get_all_values_and_update_cell(self):
        client = MockGoogleSheetsClient([["H1", "H2"], ["R1C1", "R1C2"]])
        values = client.get_all_values()
        assert len(values) == 2
        assert values[1][1] == "R1C2"

        # Update cell at row 2, col 2 (1-indexed)
        client.update_cell(row=2, col=2, value="UPDATED")
        updated_values = client.get_all_values()
        assert updated_values[1][1] == "UPDATED"
        assert len(client.update_log) == 1
        assert client.update_log[0] == {"row": 2, "col": 2, "value": "UPDATED"}


class TestHeaderResolution:
    """Validate dynamic column matching for diverse sheet schemas."""

    def test_standard_headers(self, sample_headers):
        client = MockGoogleSheetsClient()
        watcher = GoogleSheetsWatcher(client=client)
        indices = watcher.resolve_column_indices(sample_headers)

        assert indices["transcription"] == 6  # issue_transcription
        assert indices["priority"] == 7       # priority_level
        assert indices["ticket_id"] == 1      # ticket_id
        assert indices["status"] == 8         # status

    def test_alias_headers(self):
        headers = ["id", "issue_description", "priority", "date"]
        client = MockGoogleSheetsClient()
        watcher = GoogleSheetsWatcher(client=client)
        indices = watcher.resolve_column_indices(headers)

        assert indices["transcription"] == 2  # issue_description
        assert indices["priority"] == 3       # priority
        assert indices["ticket_id"] == 1      # id

    def test_case_insensitive_matching(self):
        headers = ["Ticket ID", "Issue Transcription", "Priority Level"]
        client = MockGoogleSheetsClient()
        watcher = GoogleSheetsWatcher(client=client)
        indices = watcher.resolve_column_indices(headers)

        assert indices["ticket_id"] == 1
        assert indices["transcription"] == 2
        assert indices["priority"] == 3


class TestMissingPriorityDetection:
    """Validate identification of unassigned priority cells."""

    def test_missing_values(self):
        client = MockGoogleSheetsClient()
        watcher = GoogleSheetsWatcher(client=client)

        assert watcher.is_priority_missing(None) is True
        assert watcher.is_priority_missing("") is True
        assert watcher.is_priority_missing("   ") is True
        assert watcher.is_priority_missing("null") is True
        assert watcher.is_priority_missing("None") is True
        assert watcher.is_priority_missing("N/A") is True
        assert watcher.is_priority_missing("-") is True

    def test_populated_values(self):
        client = MockGoogleSheetsClient()
        watcher = GoogleSheetsWatcher(client=client)

        assert watcher.is_priority_missing("HIGH") is False
        assert watcher.is_priority_missing("High") is False
        assert watcher.is_priority_missing("MEDIUM") is False
        assert watcher.is_priority_missing("Low") is False


class TestWatcherExecution:
    """Validate continuous watcher polling and sheet updates."""

    def test_poll_once_detects_and_updates_missing_rows(self, sample_sheet_data):
        client = MockGoogleSheetsClient(initial_data=sample_sheet_data)
        config = SheetsWatcherConfig(priority_case="UPPER")
        watcher = GoogleSheetsWatcher(client=client, config=config)

        reports = watcher.poll_once()

        # Should update exactly rows 2 and 3; row 4 already has "High"
        assert len(reports) == 2

        # Row 2 was high urgency: "Fee debited twice and deadline is today!"
        r2 = next(r for r in reports if r.row_index == 2)
        assert r2.predicted_priority == "HIGH"
        assert r2.status == "SUCCESS"

        # Row 3 was routine query: "General question regarding campus map"
        r3 = next(r for r in reports if r.row_index == 3)
        assert r3.predicted_priority == "LOW"
        assert r3.status == "SUCCESS"

        # Verify client values were actually updated in memory
        updated_data = client.get_all_values()
        assert updated_data[1][6] == "HIGH"   # Row 2, Col 7 (priority_level)
        assert updated_data[2][6] == "LOW"    # Row 3, Col 7 (priority_level)
        assert updated_data[3][6] == "High"   # Row 4 preserved intact!

    def test_duplicate_prevention(self, sample_sheet_data):
        """A second poll cycle must not re-process or re-write already updated rows."""
        client = MockGoogleSheetsClient(initial_data=sample_sheet_data)
        watcher = GoogleSheetsWatcher(client=client)

        first_reports = watcher.poll_once()
        assert len(first_reports) == 2
        initial_updates_count = len(client.update_log)
        assert initial_updates_count == 2

        # Second poll
        second_reports = watcher.poll_once()
        assert len(second_reports) == 0
        # No additional writes made to the sheet
        assert len(client.update_log) == initial_updates_count

    def test_title_case_formatting(self, sample_sheet_data):
        client = MockGoogleSheetsClient(initial_data=sample_sheet_data)
        config = SheetsWatcherConfig(priority_case="TITLE")
        watcher = GoogleSheetsWatcher(client=client, config=config)

        reports = watcher.poll_once()
        assert len(reports) == 2
        assert reports[0].predicted_priority == "High"
        assert reports[1].predicted_priority == "Low"

        # In sheet:
        updated_data = client.get_all_values()
        assert updated_data[1][6] == "High"
        assert updated_data[2][6] == "Low"

    def test_safe_error_handling_on_individual_row_failure(self, sample_headers):
        """Failure to update one cell should not crash or stop other rows."""
        sheet_data = [
            sample_headers,
            ["T1", "time", "Name", "Reg", "Phone", "Issue 1 deadline today", "", "Open"],
            ["T2", "time", "Name", "Reg", "Phone", "Issue 2 general info", "", "Open"],
        ]
        client = MockGoogleSheetsClient(initial_data=sheet_data)

        # Mock client to fail only on row 2
        original_update = client.update_cell
        def flaky_update(row, col, value):
            if row == 2:
                raise RuntimeError("Simulated API rate limit on row 2")
            return original_update(row, col, value)

        client.update_cell = flaky_update

        watcher = GoogleSheetsWatcher(client=client)
        reports = watcher.poll_once()

        assert len(reports) == 2
        # Row 2 failed safely
        r2 = next(r for r in reports if r.row_index == 2)
        assert r2.status == "ERROR"
        assert "Simulated API rate limit" in r2.error_message

        # Row 3 succeeded
        r3 = next(r for r in reports if r.row_index == 3)
        assert r3.status == "SUCCESS"
        assert r3.predicted_priority == "LOW"

    def test_graceful_stop_and_max_cycles(self, sample_sheet_data):
        client = MockGoogleSheetsClient(initial_data=sample_sheet_data)
        config = SheetsWatcherConfig(poll_interval_seconds=0.01)
        watcher = GoogleSheetsWatcher(client=client, config=config)

        # Run 2 cycles and exit
        watcher.start_watching(max_cycles=2)
        assert watcher.stop_requested is False

        # Stop explicitly
        watcher.stop()
        assert watcher.stop_requested is True

    def test_run_once_config(self, sample_sheet_data):
        client = MockGoogleSheetsClient(initial_data=sample_sheet_data)
        config = SheetsWatcherConfig(run_once=True, poll_interval_seconds=0.01)
        watcher = GoogleSheetsWatcher(client=client, config=config)

        # Should run 1 cycle and return
        watcher.start_watching()
        assert len(client.update_log) == 2

    def test_empty_transcription_is_skipped_until_populated(self, sample_headers):
        """Rows with blank transcription (e.g., in progress typing) must not be stamped LOW."""
        sheet_data = [
            sample_headers,
            ["T1", "time", "Alice", "Reg1", "Phone1", "", "", "Open"],  # Blank transcription
            ["T2", "time", "Bob", "Reg2", "Phone2", "   ", "", "Open"],  # Whitespace only
        ]
        client = MockGoogleSheetsClient(initial_data=sheet_data)
        watcher = GoogleSheetsWatcher(client=client)

        reports = watcher.poll_once()
        assert len(reports) == 0
        assert len(client.update_log) == 0
        # Priority cells remain empty
        assert client.data[1][6] == ""
        assert client.data[2][6] == ""

        # Now simulate user typing text into T1
        client.data[1][5] = "Fee deducted twice and urgent registration deadline today!"
        second_reports = watcher.poll_once()
        assert len(second_reports) == 1
        assert second_reports[0].ticket_id == "T1"
        assert second_reports[0].predicted_priority == "HIGH"
        assert client.data[1][6] == "HIGH"
        # T2 still skipped
        assert client.data[2][6] == ""

    def test_re_evaluates_if_priority_is_cleared(self, sample_headers):
        """If user clears the priority column in the sheet, it must re-evaluate."""
        sheet_data = [
            sample_headers,
            ["T1", "time", "Alice", "Reg1", "Phone1", "Hostel room change request", "LOW", "Open"],
        ]
        client = MockGoogleSheetsClient(initial_data=sheet_data)
        watcher = GoogleSheetsWatcher(client=client)

        # Priority is already LOW; should do nothing
        assert len(watcher.poll_once()) == 0

        # User clears priority cell in sheet
        client.data[1][6] = ""
        reports = watcher.poll_once()
        assert len(reports) == 1
        assert reports[0].predicted_priority == "MEDIUM"
        assert client.data[1][6] == "MEDIUM"

