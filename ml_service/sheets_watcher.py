"""Continuous Google Sheets Watcher Service for Priority Inference.

Monitors a Google Sheet in real time, detects rows with missing or null priority,
extracts `issue_transcription`, runs priority inference, and updates the `priority_level` cell.

FEATURES:
- Configurable polling interval and exponential backoff on transient errors.
- Duplicate-processing prevention via state tracking and row validation.
- Safe error handling: individual row failures or intermittent network disconnects
  never crash the watcher daemon.
- Isolated model execution using PriorityPredictor.
- Clean client abstraction supporting real Google Sheets (`gspread`) and in-memory mocks.
- Graceful shutdown handling on SIGINT/SIGTERM.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .predictor import PriorityPredictor
from .priority_contract import normalize_priority_label
from .schemas import PriorityInferenceInput, PriorityInferenceOutput
from .sheets_config import SheetsWatcherConfig

logger = logging.getLogger("ml_service.sheets_watcher")


# ==============================================================================
# Client Interfaces & Implementations
# ==============================================================================

class GoogleSheetsClientInterface(ABC):
    """Abstract interface for Google Sheet operations."""

    @abstractmethod
    def get_all_values(self) -> List[List[str]]:
        """Fetch all rows from the worksheet as a 2D list of strings."""
        raise NotImplementedError

    @abstractmethod
    def update_cell(self, row: int, col: int, value: str) -> bool:
        """Update a single cell at 1-indexed (row, col) with string value."""
        raise NotImplementedError

    @abstractmethod
    def batch_update_cells(self, updates: List[Tuple[int, int, str]]) -> bool:
        """Update multiple cells: list of (1-indexed row, 1-indexed col, value)."""
        raise NotImplementedError


class MockGoogleSheetsClient(GoogleSheetsClientInterface):
    """In-memory Google Sheets client for offline testing and verification."""

    def __init__(self, initial_data: Optional[List[List[str]]] = None) -> None:
        # 0-indexed list of lists internally, converted from 1-indexed calls
        self.data: List[List[str]] = [list(r) for r in (initial_data or [])]
        self.update_log: List[Dict[str, Any]] = []
        self.should_raise_error: Optional[Exception] = None

    def get_all_values(self) -> List[List[str]]:
        if self.should_raise_error:
            err = self.should_raise_error
            self.should_raise_error = None
            raise err
        # Return a deep copy
        return [list(r) for r in self.data]

    def update_cell(self, row: int, col: int, value: str) -> bool:
        if self.should_raise_error:
            err = self.should_raise_error
            self.should_raise_error = None
            raise err

        r_idx = row - 1
        c_idx = col - 1

        # Expand rows if needed
        while len(self.data) <= r_idx:
            self.data.append([])

        # Expand columns if needed
        while len(self.data[r_idx]) <= c_idx:
            self.data[r_idx].append("")

        self.data[r_idx][c_idx] = str(value)
        self.update_log.append({"row": row, "col": col, "value": value})
        return True

    def batch_update_cells(self, updates: List[Tuple[int, int, str]]) -> bool:
        for r, c, val in updates:
            self.update_cell(r, c, val)
        return True


class GSpreadSheetsClient(GoogleSheetsClientInterface):
    """Production Google Sheets client wrapping gspread."""

    def __init__(self, config: SheetsWatcherConfig) -> None:
        self.config = config
        self.gc: Any = None
        self.spreadsheet: Any = None
        self.worksheet: Any = None
        self._authenticate_and_open()

    def _authenticate_and_open(self) -> None:
        """Authenticate with Google APIs and open the worksheet."""
        try:
            import gspread
        except ImportError as e:
            raise ImportError(
                "gspread is required for GSpreadSheetsClient. Install with 'pip install gspread google-auth'."
            ) from e

        # 1. Authenticate (supports both Service Account and OAuth2 Desktop Client)
        if self.config.credentials_json:
            creds_dict = json.loads(self.config.credentials_json)
            if "installed" in creds_dict or "web" in creds_dict:
                import tempfile
                with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tf:
                    json.dump(creds_dict, tf)
                    temp_path = tf.name
                self.gc = gspread.oauth(credentials_filename=temp_path)
            else:
                self.gc = gspread.service_account_from_dict(creds_dict)
        elif self.config.credentials_path and os.path.exists(self.config.credentials_path):
            try:
                with open(self.config.credentials_path, "r", encoding="utf-8") as f:
                    creds_data = json.load(f)
            except Exception:
                creds_data = {}

            if "installed" in creds_data or "web" in creds_data:
                os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
                token_dir = Path(self.config.credentials_path).parent
                token_path = os.getenv("GOOGLE_OAUTH_TOKEN_PATH") or str(token_dir / "authorized_user.json")
                self.gc = gspread.oauth(
                    scopes=["https://www.googleapis.com/auth/spreadsheets"],
                    credentials_filename=self.config.credentials_path,
                    authorized_user_filename=token_path,
                )
            else:
                self.gc = gspread.service_account(filename=self.config.credentials_path)
        else:
            # Fall back to default application credentials or environment
            self.gc = gspread.service_account()

        # 2. Open spreadsheet by ID or Name
        if self.config.spreadsheet_id:
            self.spreadsheet = self.gc.open_by_key(self.config.spreadsheet_id)
        elif self.config.spreadsheet_name:
            self.spreadsheet = self.gc.open(self.config.spreadsheet_name)
        else:
            raise ValueError(
                "Neither spreadsheet_id nor spreadsheet_name was provided in SheetsWatcherConfig."
            )

        # 3. Open worksheet
        try:
            self.worksheet = self.spreadsheet.worksheet(self.config.worksheet_name)
        except Exception:
            # Try index 0 as fallback
            self.worksheet = self.spreadsheet.get_worksheet(0)

    def get_all_values(self) -> List[List[str]]:
        return self.worksheet.get_all_values()

    def update_cell(self, row: int, col: int, value: str) -> bool:
        self.worksheet.update_cell(row, col, value)
        return True

    def batch_update_cells(self, updates: List[Tuple[int, int, str]]) -> bool:
        import gspread
        cells = [
            gspread.Cell(row=r, col=c, value=val)
            for r, c, val in updates
        ]
        self.worksheet.update_cells(cells)
        return True


# ==============================================================================
# Row Update Report Dataclass
# ==============================================================================

@dataclass
class RowUpdateReport:
    """Diagnostic details for an individual row processed by the watcher."""
    row_index: int
    ticket_id: Optional[str]
    transcription_snippet: str
    predicted_priority: str
    confidence: Optional[float]
    model_status: str
    is_trained_model: bool
    status: str  # "SUCCESS", "SKIPPED", "ERROR"
    error_message: Optional[str] = None


# ==============================================================================
# Google Sheets Watcher Service
# ==============================================================================

class GoogleSheetsWatcher:
    """Continuous watcher daemon for detecting new rows and assigning priority."""

    def __init__(
        self,
        client: GoogleSheetsClientInterface,
        predictor: Optional[PriorityPredictor] = None,
        config: Optional[SheetsWatcherConfig] = None,
        logger_instance: Optional[logging.Logger] = None,
    ) -> None:
        self.client = client
        self.config = config or SheetsWatcherConfig()
        self.logger = logger_instance or logger
        self.predictor = predictor or PriorityPredictor(model_dir=self.config.model_dir)

        # Deduplication and in-flight tracking
        self.processed_keys: Set[str] = set()
        self.in_flight_keys: Set[str] = set()
        self.stop_requested: bool = False

    def is_priority_missing(self, value: Optional[str]) -> bool:
        """Determine if a cell contains a null, empty, or unassigned priority."""
        if value is None:
            return True
        val_str = str(value).strip().lower()
        return val_str in ("", "null", "none", "n/a", "-", "undefined", "nil")

    def resolve_column_indices(self, headers: List[str]) -> Dict[str, Optional[int]]:
        """Identify 1-indexed column numbers from header row values.
        
        Matches header names dynamically against config candidate lists.
        """
        normalized_headers = [str(h).strip().lower().replace(" ", "_") for h in headers]
        indices: Dict[str, Optional[int]] = {
            "transcription": None,
            "priority": None,
            "ticket_id": None,
            "status": None,
        }

        # 1. Match transcription column
        for cand in self.config.transcription_columns:
            cand_norm = cand.lower().replace(" ", "_")
            if cand_norm in normalized_headers:
                indices["transcription"] = normalized_headers.index(cand_norm) + 1
                break

        # 2. Match priority column
        for cand in self.config.priority_columns:
            cand_norm = cand.lower().replace(" ", "_")
            if cand_norm in normalized_headers:
                indices["priority"] = normalized_headers.index(cand_norm) + 1
                break

        # 3. Match ticket_id column
        for cand in self.config.ticket_id_columns:
            cand_norm = cand.lower().replace(" ", "_")
            if cand_norm in normalized_headers:
                indices["ticket_id"] = normalized_headers.index(cand_norm) + 1
                break

        # 4. Match status column
        for cand in self.config.status_columns:
            cand_norm = cand.lower().replace(" ", "_")
            if cand_norm in normalized_headers:
                indices["status"] = normalized_headers.index(cand_norm) + 1
                break

        return indices

    def _generate_row_key(
        self, row_idx: int, ticket_id: Optional[str], transcription: str
    ) -> str:
        """Create a stable fingerprint for a row to prevent duplicate updates."""
        clean_trans = str(transcription or "").strip()
        snippet_hash = hashlib.sha256(clean_trans.encode("utf-8")).hexdigest()[:12]
        if ticket_id and str(ticket_id).strip():
            return f"ticket:{str(ticket_id).strip()}:{snippet_hash}"
        # Fallback to row index + text hash
        return f"row:{row_idx}:{snippet_hash}"

    def poll_once(self) -> List[RowUpdateReport]:
        """Perform a single inspection pass of the Google Sheet.
        
        Returns:
            List of RowUpdateReport objects for all processed rows.
        """
        reports: List[RowUpdateReport] = []

        try:
            rows = self.client.get_all_values()
        except Exception as e:
            self.logger.error("Failed to fetch values from Google Sheet: %s", e)
            raise

        if not rows:
            self.logger.debug("Google Sheet is empty; waiting for rows.")
            return reports

        headers = rows[0]
        col_indices = self.resolve_column_indices(headers)

        trans_col = col_indices.get("transcription")
        prio_col = col_indices.get("priority")
        id_col = col_indices.get("ticket_id")

        if trans_col is None:
            self.logger.warning(
                "Could not find transcription column in sheet headers: %s. Expected one of: %s",
                headers,
                self.config.transcription_columns,
            )
            return reports

        if prio_col is None:
            self.logger.warning(
                "Could not find priority column in sheet headers: %s. Expected one of: %s",
                headers,
                self.config.priority_columns,
            )
            return reports

        # Iterate through rows (1-indexed row 2 is rows[1])
        for row_offset, row in enumerate(rows[1:], start=2):
            # Safe extraction of cell values (handles short rows)
            prio_val = row[prio_col - 1] if len(row) >= prio_col else ""
            trans_val = row[trans_col - 1] if len(row) >= trans_col else ""
            ticket_id = (row[id_col - 1] if id_col and len(row) >= id_col else "") or None

            # Skip if priority is already assigned
            if not self.is_priority_missing(prio_val):
                row_key = self._generate_row_key(row_offset, ticket_id, trans_val)
                self.processed_keys.discard(row_key)
                continue

            # Skip if issue transcription is empty or whitespace only.
            # Avoids prematurely evaluating blank rows or rows being actively typed.
            if not trans_val or not str(trans_val).strip():
                self.logger.debug(
                    "Skipping row %d: issue transcription is empty (waiting for input).",
                    row_offset,
                )
                continue

            # Check deduplication fingerprint
            row_key = self._generate_row_key(row_offset, ticket_id, trans_val)
            if row_key in self.processed_keys or row_key in self.in_flight_keys:
                continue

            self.in_flight_keys.add(row_key)
            snippet = (trans_val[:40] + "...") if len(trans_val) > 40 else trans_val

            self.logger.info(
                "Detected row %d with missing priority (ticket_id: %s, snippet: '%s')",
                row_offset,
                ticket_id or "N/A",
                snippet,
            )

            try:
                # 1. Run inference
                prediction: PriorityInferenceOutput = self.predictor.predict_priority(
                    PriorityInferenceInput(
                        issue_transcription=trans_val,
                        ticket_id=ticket_id,
                    )
                )

                # 2. Format priority string
                raw_priority = prediction.priority_level
                formatted_priority = normalize_priority_label(
                    raw_priority,
                    title_case=(self.config.priority_case == "TITLE"),
                )

                # 3. Write prediction back to Google Sheet
                self.client.update_cell(
                    row=row_offset,
                    col=prio_col,
                    value=formatted_priority,
                )

                # 4. Mark completed in state cache
                self.processed_keys.add(row_key)
                self.in_flight_keys.remove(row_key)

                self.logger.info(
                    "Successfully updated row %d: priority_level = '%s' "
                    "(model_status: %s, is_trained: %s)",
                    row_offset,
                    formatted_priority,
                    prediction.model_status,
                    prediction.is_trained_model,
                )

                reports.append(
                    RowUpdateReport(
                        row_index=row_offset,
                        ticket_id=ticket_id,
                        transcription_snippet=snippet,
                        predicted_priority=formatted_priority,
                        confidence=prediction.confidence,
                        model_status=prediction.model_status,
                        is_trained_model=prediction.is_trained_model,
                        status="SUCCESS",
                    )
                )

            except Exception as exc:
                self.in_flight_keys.remove(row_key)
                self.logger.error(
                    "Failed to process row %d (ticket_id: %s): %s",
                    row_offset,
                    ticket_id,
                    exc,
                )
                reports.append(
                    RowUpdateReport(
                        row_index=row_offset,
                        ticket_id=ticket_id,
                        transcription_snippet=snippet,
                        predicted_priority="UNKNOWN",
                        confidence=None,
                        model_status="ERROR",
                        is_trained_model=False,
                        status="ERROR",
                        error_message=str(exc),
                    )
                )

        return reports

    def stop(self) -> None:
        """Signal the watcher daemon to stop polling gracefully."""
        self.logger.info("Shutdown requested for GoogleSheetsWatcher.")
        self.stop_requested = True

    def start_watching(self, max_cycles: Optional[int] = None) -> None:
        """Run continuous polling loop.
        
        Args:
            max_cycles: Optional limit on poll cycles (primarily for testing).
        """
        # Register signal handlers for clean exit
        try:
            signal.signal(signal.SIGINT, lambda s, f: self.stop())
            signal.signal(signal.SIGTERM, lambda s, f: self.stop())
        except (ValueError, AttributeError):
            # Not in main thread or platform unsupported
            pass

        self.logger.info(
            "Starting GoogleSheetsWatcher daemon (poll_interval=%.1fs, priority_case=%s)",
            self.config.poll_interval_seconds,
            self.config.priority_case,
        )

        cycle_count = 0
        consecutive_errors = 0

        while not self.stop_requested:
            cycle_count += 1
            start_time = time.time()

            try:
                reports = self.poll_once()
                consecutive_errors = 0
                if reports:
                    self.logger.info("Cycle %d: updated %d rows.", cycle_count, len(reports))
            except Exception as exc:
                consecutive_errors += 1
                backoff = min(
                    60.0,
                    self.config.retry_delay_seconds * (2 ** min(consecutive_errors - 1, 5)),
                )
                self.logger.warning(
                    "Error during poll cycle %d: %s. Backing off for %.1fs (consecutive errors: %d)",
                    cycle_count,
                    exc,
                    backoff,
                    consecutive_errors,
                )
                time.sleep(backoff)
                continue

            if self.config.run_once:
                self.logger.info("RUN_ONCE enabled; exiting after single poll.")
                break

            if max_cycles is not None and cycle_count >= max_cycles:
                self.logger.info("Reached max_cycles (%d); exiting loop.", max_cycles)
                break

            # Sleep remainder of poll interval in short chunks for responsive shutdown
            elapsed = time.time() - start_time
            remaining = max(0.0, self.config.poll_interval_seconds - elapsed)
            while remaining > 0 and not self.stop_requested:
                chunk = min(0.5, remaining)
                time.sleep(chunk)
                remaining -= chunk

        self.logger.info("GoogleSheetsWatcher stopped successfully.")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main() -> None:
    """CLI runner to launch the Google Sheets Watcher service continuously."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = SheetsWatcherConfig()
    logger.info("Initializing Google Sheets Watcher Service...")

    # Validate that credentials and sheet identifier are configured
    if not (config.credentials_path or config.credentials_json):
        logger.warning(
            "No Google credentials provided via GOOGLE_APPLICATION_CREDENTIALS or "
            "GOOGLE_SHEETS_CREDENTIALS_JSON. Using mock client for demonstration."
        )
        sample_sheet = [
            ["ticket_id", "date_and_time", "student_name", "registration_number", "phone_number", "issue_transcription", "priority_level", "status"],
            ["John1", "2026-10-23 10:00:00", "John Doe", "REG101", "9876543210", "Payment debited twice but admission portal says fee pending deadline today", "", "Open"],
            ["John2", "2026-10-23 10:05:00", "Jane Smith", "REG102", "9876543211", "Could you please tell me what courses are available for the next semester?", "", "Open"],
            ["John3", "2026-10-23 10:10:00", "Bob Davis", "REG103", "9876543212", "Hostel room change request pending for verification", "Medium", "Open"],
        ]
        client = MockGoogleSheetsClient(initial_data=sample_sheet)
    else:
        client = GSpreadSheetsClient(config=config)

    watcher = GoogleSheetsWatcher(client=client, config=config)
    watcher.start_watching()


if __name__ == "__main__":
    main()
