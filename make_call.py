"""Outbound SIP dialer using LiveKit and Google Sheets.

This module reads student applicant contact and verification records from a
Google Sheet, ensures a valid outbound SIP trunk exists on LiveKit Cloud via
the Vobiz carrier gateway, and concurrently dispatches verification calls
respecting concurrency limits and carrier rate-limiting (CPS) constraints.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
import gspread
from livekit import api
from livekit.protocol.sip import (
    CreateSIPOutboundTrunkRequest,
    CreateSIPParticipantRequest,
    ListSIPOutboundTrunkRequest,
    ListSIPOutboundTrunkResponse,
    SIPOutboundTrunkInfo,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("OutboundDialer")

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

VOBIZ_SIP_DOMAIN = os.environ["VOBIZ_SIP_DOMAIN"]
VOBIZ_USERNAME = os.environ["VOBIZ_USERNAME"]
VOBIZ_PASSWORD = os.environ["VOBIZ_PASSWORD"]
VOBIZ_CALLER_ID = os.environ["VOBIZ_CALLER_ID"]

GOOGLE_SERVICE_ACCOUNT_FILE = os.environ.get(
    "GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json"
)
SPREADSHEET_NAME = os.environ.get("SPREADSHEET_NAME", "vitap_database")
WORKSHEET_NAME = os.environ.get("WORKSHEET_NAME", "students")

MAX_CONCURRENT_CALLS = int(os.environ.get("MAX_CONCURRENT_CALLS", "10"))
CALLS_PER_SECOND = float(os.environ.get("CALLS_PER_SECOND", "2.0"))


def normalize_e164(raw_phone: Any) -> str:
    """Normalize raw Indian phone numbers into standard E.164 format.

    Args:
        raw_phone (Any): The unparsed phone number representation, which can
            be a string, integer, or other scalar value.

    Returns:
        str: Standardized E.164 phone string starting with '+' prefix.
    """
    digits = "".join(filter(str.isdigit, str(raw_phone)))
    if len(digits) == 10:
        return f"+91{digits}"
    elif len(digits) == 12 and digits.startswith("91"):
        return f"+{digits}"
    elif str(raw_phone).strip().startswith("+"):
        return str(raw_phone).strip()
    return f"+{digits}"


def fetch_sheet_records() -> List[Dict[str, Any]]:
    """Fetch all applicant rows from the designated Google Sheet.

    Connects to Google Sheets using a service account credentials file,
    retrieves all student rows from the specified worksheet, and normalizes
    the phone numbers into E.164 format.

    Returns:
        List[Dict[str, Any]]: A list of valid applicant records containing
        callable phone numbers.
    """
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(
        GOOGLE_SERVICE_ACCOUNT_FILE, scopes=scopes
    )
    client = gspread.authorize(creds)
    sheet = client.open(SPREADSHEET_NAME).worksheet(WORKSHEET_NAME)
    records = sheet.get_all_records()

    valid_targets = []
    for row in records:
        phone = row.get("phone_number")
        if phone:
            row["e164_phone"] = normalize_e164(phone)
            valid_targets.append(row)

    logger.info(
        "Retrieved %d callable records from '%s/%s'",
        len(valid_targets),
        SPREADSHEET_NAME,
        WORKSHEET_NAME,
    )
    return valid_targets


async def get_or_create_sip_trunk(lk: api.LiveKitAPI) -> str:
    """Find or provision the Vobiz Outbound SIP Trunk on LiveKit Cloud.

    Queries LiveKit for existing outbound SIP trunks matching the configured
    Vobiz domain. If none exists, creates and configures a new trunk.

    Args:
        lk (api.LiveKitAPI): Initialized LiveKit API client instance.

    Returns:
        str: The SIP trunk identifier string.
    """
    clean_domain = VOBIZ_SIP_DOMAIN.replace("sip:", "").strip("/")

    existing_trunks: ListSIPOutboundTrunkResponse = (
        await lk.sip.list_sip_outbound_trunk(ListSIPOutboundTrunkRequest())
    )

    for trunk in existing_trunks.items:
        if trunk.address == clean_domain:
            logger.info(
                "Found existing outbound SIP trunk: %s", trunk.sip_trunk_id
            )
            return trunk.sip_trunk_id

    logger.info(
        "Creating new LiveKit outbound SIP trunk targeting domain: %s",
        clean_domain,
    )
    trunk_info = SIPOutboundTrunkInfo(
        name="Vobiz Outbound Trunk",
        address=clean_domain,
        auth_username=VOBIZ_USERNAME,
        auth_password=VOBIZ_PASSWORD,
        numbers=[VOBIZ_CALLER_ID],
    )
    request = CreateSIPOutboundTrunkRequest(trunk=trunk_info)
    created_trunk = await lk.sip.create_sip_outbound_trunk(request)
    logger.info("Provisioned trunk ID: %s", created_trunk.sip_trunk_id)
    return created_trunk.sip_trunk_id


class AsyncTokenBucketRateLimiter:
    """Enforce non-blocking CPS constraints so carrier SBCs do not drop
    packets.

    Attributes:
        interval (float): Minimum duration in seconds required between calls.
        last_timestamp (float): Monotonic timestamp of the last acquired token.
    """

    def __init__(self, rate: float) -> None:
        """Initialize the rate limiter with a maximum rate per second.

        Args:
            rate (float): Maximum requests or calls permitted per second.
        """
        self.interval = 1.0 / rate
        self.last_timestamp = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait asynchronously until enough time has elapsed to make a call."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_timestamp
            if elapsed < self.interval:
                await asyncio.sleep(self.interval - elapsed)
            self.last_timestamp = time.monotonic()


async def execute_verification_call(
    lk: api.LiveKitAPI,
    trunk_id: str,
    record: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    rate_limiter: AsyncTokenBucketRateLimiter,
) -> None:
    """Dispatch an outbound verification call to a single student participant.

    Acquires concurrency and rate limiting slots, constructs dynamic session
    room parameters and student metadata, and creates a SIP participant via
    LiveKit.

    Args:
        lk (api.LiveKitAPI): Initialized LiveKit API client instance.
        trunk_id (str): Outbound SIP trunk ID used to route the call.
        record (Dict[str, Any]): Applicant record containing contact and
            verification metadata.
        semaphore (asyncio.Semaphore): Concurrency limiter restricting active
            calls.
        rate_limiter (AsyncTokenBucketRateLimiter): Token-bucket rate limiter
            ensuring safe calls-per-second throughput.

    Returns:
        None
    """
    async with semaphore:
        await rate_limiter.acquire()

        target_phone = record["e164_phone"]
        app_id = str(record.get("application_number", "UNKNOWN"))
        student_name = record.get("student_name", "Student")
        issue_context = record.get("issue_transcription", "")
        ticket_id = record.get("ticket_id", "")

        session_room = f"verify-{app_id}-{int(time.time())}"
        participant_id = f"sip-{target_phone}"

        metadata_dict = {
            "student_name": student_name,
            "application_number": app_id,
            "issue_transcription": issue_context,
            "ticket_id": ticket_id,
        }

        logger.info(
            "Dialing %s (%s) for room %s",
            student_name,
            target_phone,
            session_room,
        )

        call_request = CreateSIPParticipantRequest(
            sip_trunk_id=trunk_id,
            sip_call_to=target_phone,
            sip_number=VOBIZ_CALLER_ID,
            room_name=session_room,
            participant_identity=participant_id,
            participant_name=student_name,
            participant_metadata=json.dumps(metadata_dict),
            participant_attributes={
                k: str(v) for k, v in metadata_dict.items()
            },
            wait_until_answered=False,
        )

        try:
            participant = await lk.sip.create_sip_participant(call_request)
            logger.info(
                "Dispatched call to %s -> Participant ID: %s",
                target_phone,
                participant.participant_id,
            )
        except Exception as exc:
            logger.error(
                "Failed to place call to %s: %s", target_phone, str(exc)
            )


async def main() -> None:
    """Execute the end-to-end outbound dialing orchestrator.

    Fetches applicant records from Google Sheets, provisions or verifies the
    SIP trunk, configures concurrency and rate limiting primitives, and
    dispatches all calls concurrently.

    Returns:
        None
    """
    records = fetch_sheet_records()
    if not records:
        logger.warning("No records found to call. Exiting.")
        return

    lk_client = api.LiveKitAPI(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
    )

    try:
        trunk_id = await get_or_create_sip_trunk(lk_client)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
        rate_limiter = AsyncTokenBucketRateLimiter(CALLS_PER_SECOND)

        # Dispatches all calls simultaneously within the semaphore &
        # rate limiter constraints
        tasks = [
            execute_verification_call(
                lk_client, trunk_id, record, semaphore, rate_limiter
            )
            for record in records
        ]

        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("All outbound verification calls have been dispatched.")
    finally:
        await lk_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
