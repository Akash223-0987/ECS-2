"""Configuration for Google Sheets Watcher Service.

Provides declarative configuration settings for connecting to Google Sheets,
polling intervals, duplicate prevention, and priority inference.
"""

from dataclasses import dataclass, field
import os
from typing import List, Optional


@dataclass
class SheetsWatcherConfig:
    """Settings for the continuous Google Sheets watcher service.
    
    Attributes:
        spreadsheet_name: Title of the Google Spreadsheet to watch.
        spreadsheet_id: Unique ID of the Google Spreadsheet (from URL).
        worksheet_name: Title or name of the specific worksheet tab (default: 'Sheet1').
        credentials_path: Filepath to Google Service Account JSON credentials.
        credentials_json: Raw JSON string of Google Service Account credentials (from env).
        poll_interval_seconds: Number of seconds to wait between polling cycles.
        batch_size: Max rows to inspect or update per poll cycle.
        max_retries: Max retry attempts on transient network or API failures.
        retry_delay_seconds: Base delay in seconds for exponential backoff on retries.
        run_once: When True, performs a single polling pass and exits (useful for tests/CLI).
        priority_case: Formatting of output priority string ('UPPER' for HIGH, 'TITLE' for High).
        model_dir: Optional explicit path to trained priority model checkpoint.
        transcription_columns: Column name candidates for extracting issue transcription.
        priority_columns: Column name candidates for reading/writing priority level.
        ticket_id_columns: Column name candidates for ticket identifier.
    """
    spreadsheet_name: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_SHEET_NAME") or os.getenv("SPREADSHEET_NAME")
    )
    spreadsheet_id: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_SHEET_ID") or os.getenv("SPREADSHEET_ID")
    )
    worksheet_name: str = field(
        default_factory=lambda: os.getenv("GOOGLE_SHEET_WORKSHEET") or os.getenv("WORKSHEET_NAME", "Sheet1")
    )
    credentials_path: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("SERVICE_ACCOUNT_FILE")
    )
    credentials_json: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_SHEETS_CREDENTIALS_JSON")
    )
    poll_interval_seconds: float = field(
        default_factory=lambda: float(os.getenv("POLL_INTERVAL_SECONDS", "5.0"))
    )
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("BATCH_SIZE", "50"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "3"))
    )
    retry_delay_seconds: float = field(
        default_factory=lambda: float(os.getenv("RETRY_DELAY_SECONDS", "2.0"))
    )
    run_once: bool = field(
        default_factory=lambda: os.getenv("RUN_ONCE", "false").lower() in ("true", "1", "yes")
    )
    priority_case: str = field(
        default_factory=lambda: os.getenv("PRIORITY_CASE", "UPPER").upper()
    )
    model_dir: Optional[str] = field(
        default_factory=lambda: os.getenv("PRIORITY_MODEL_PATH") or os.getenv("PRIORITY_MODEL_DIR")
    )
    transcription_columns: List[str] = field(
        default_factory=lambda: [
            "issue_transcription",
            "issue_description",
            "transcription",
            "issue",
            "query",
            "description",
        ]
    )
    priority_columns: List[str] = field(
        default_factory=lambda: [
            "priority_level",
            "priority",
            "priority_score",
        ]
    )
    ticket_id_columns: List[str] = field(
        default_factory=lambda: [
            "ticket_id",
            "id",
            "ticket",
        ]
    )
    status_columns: List[str] = field(
        default_factory=lambda: [
            "status",
            "ticket_status",
        ]
    )
