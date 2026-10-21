"""Follow-Up Decision Engine for Student Support Ticket Triage.

Separates business decision rules from model inference and applies conservative,
evidence-based reasoning to determine whether a student-support ticket requires
a follow-up call.

CORE BEHAVIOR:
- YES: Sufficient, reliable evidence supports a follow-up (e.g. explicit callback
  request, verified urgent blocker, or corroborated task-specific classifier prediction).
- NO: Reliable evidence establishes that no further contact is needed (e.g. explicitly
  confirmed resolution in text AND closed ticket status). Missing info is never
  assumed to mean resolved.
- REVIEW: Evidence is missing, ambiguous, brief, conflicting, or model is unready.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. Model inference is decoupled from business decisions.
2. Supports an injectable classifier interface (duck-typed or HuggingFaceModelLoader).
3. Generic pretrained labels (LABEL_0, POSITIVE, etc.) are strictly rejected.
4. Neither 'Open' status nor 'High' priority_level independently forces a YES.
5. Student identifiers (name, phone, registration, ticket_id) are strictly excluded
   from predictive text features.
6. Engine limitations are made explicit when an approved classifier is absent.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .config import MLServiceConfig
from .schemas import FollowUpRequired, ModelStatus, TicketInput
from .validator import ValidationResult, validate_ticket_input


# ==============================================================================
# Evidence Pattern Definitions
# ==============================================================================

CALLBACK_PATTERNS: List[str] = [
    r"\b(?:please\s+)?call(?:\s+me)?\s+back\b",
    r"\brequest(?:ing)?\s+(?:a\s+)?call(?:back)?\b",
    r"\bneed\s+(?:an?\s+)?(?:advisor|agent|officer|someone)\s+to\s+(?:call|reach|contact)\b",
    r"\breach\s+out\s+to\s+me\b",
    r"\bwaiting\s+for\s+(?:a\s+)?call(?:back)?\b",
    r"\bcontact\s+me\s+(?:by\s+phone|urgently|at|asap)\b",
    r"\bspeak\s+with\s+(?:an?\s+)?(?:advisor|human|agent|counselor|officer)\b",
    r"\btalk\s+to\s+(?:an?\s+)?(?:advisor|human|agent|counselor|officer)\b",
]

UNRESOLVED_PATTERNS: List[str] = [
    r"\b(?:issue|problem|ticket)\s+(?:is\s+)?not\s+resolved\b",
    r"\bstill\s+(?:not\s+working|failing|pending|unresolved|stuck)\b",
    r"\bpayment\s+(?:was\s+)?deducted\s+but\b",
    r"\bfee\s+(?:was\s+)?deducted\s+but\b",
    r"\bdeadline\s+(?:is\s+)?(?:today|tomorrow|approaching|urgent)\b",
    r"\bunable\s+to\s+(?:access|login|pay|register|upload|submit)\b",
    r"\bhave\s+not\s+received\s+(?:my\s+)?(?:refund|receipt|id|hall\s*ticket|admit\s*card)\b",
    r"\bportal\s+(?:gives|shows|threw|displays)\s+(?:an?\s+)?error\b",
]

RESOLVED_PATTERNS: List[str] = [
    r"\b(?:issue|problem|query)\s+(?:is|has\s+been)\s+(?:resolved|fixed|solved)\b",
    r"\bresolved\s+successfully\b",
    r"\bno\s+further\s+(?:help|assistance|contact|follow-up|follow\s*up)\s+(?:is\s+)?needed\b",
    r"\beverything\s+is\s+(?:clear|sorted|working|fine)\s+now\b",
    r"\ball\s+(?:set|sorted|good)\s+now\b",
    r"\breceived\s+(?:my\s+)?(?:refund|receipt|hall\s*ticket|admit\s*card)\s+successfully\b",
    r"\bissue\s+has\s+been\s+sorted\b",
    r"\bno\s+longer\s+need\s+(?:help|assistance)\b",
]

CLOSED_STATUSES: Set[str] = {"resolved", "closed", "completed", "done"}
OPEN_STATUSES: Set[str] = {"open", "pending", "in progress", "new", "assigned", "reopened"}


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class FollowUpDecision:
    """Detailed result of follow-up evaluation.
    
    Attributes:
        decision: The categorical follow-up requirement (YES, NO, REVIEW).
        confidence: Numerical confidence score if calculable, else None.
        reason: Human-readable rationale for the decision.
        evidence: List of identified positive/negative evidence items.
        conflicts: List of conflicting signals detected during evaluation.
        classifier_status: Operational status of the classifier used.
        classifier_used: Whether an approved task-specific ML classifier contributed.
        is_conservative_fallback: True if the decision fell back to REVIEW.
    """
    decision: FollowUpRequired
    confidence: Optional[float]
    reason: str
    evidence: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    classifier_status: Optional[str] = None
    classifier_used: bool = False
    is_conservative_fallback: bool = False


# ==============================================================================
# Decision Engine
# ==============================================================================

class FollowUpDecisionEngine:
    """Evaluates student support tickets and determines follow-up necessity.
    
    LIMITATIONS & DISCLAIMER:
    Heuristic rules in this engine provide defensive baseline routing.
    They have not been statistically trained or calibrated on historical datasets.
    Production triage should pair this engine with an approved, task-specific
    fine-tuned model.
    """

    def __init__(
        self,
        classifier: Optional[Any] = None,
        config: Optional[MLServiceConfig] = None,
    ) -> None:
        self.classifier = classifier
        self.config = config or MLServiceConfig()

    def set_classifier(self, classifier: Any) -> None:
        """Inject or update the classifier used for text evaluation."""
        self.classifier = classifier

    def evaluate(
        self,
        ticket: Union[TicketInput, Dict[str, Any], Any],
        validation_result: Optional[ValidationResult] = None,
    ) -> FollowUpDecision:
        """Evaluate a ticket and determine follow-up requirement.
        
        Args:
            ticket: Raw dictionary or TicketInput instance.
            validation_result: Optional precomputed ValidationResult.
            
        Returns:
            FollowUpDecision with decision (YES/NO/REVIEW), reason, and evidence trail.
        """
        # 1. Validation check
        val = validation_result or validate_ticket_input(ticket)
        cleaned = val.cleaned_ticket

        # Case: Empty or missing description
        if val.empty_description:
            return FollowUpDecision(
                decision=FollowUpRequired.REVIEW,
                confidence=None,
                reason=(
                    "Insufficient evidence: Issue description is missing or empty. "
                    "Cannot assess follow-up necessity without student query text."
                ),
                evidence=["Empty or whitespace-only issue_description"],
                classifier_status=ModelStatus.EMPTY_DESCRIPTION_REVIEW.value,
                is_conservative_fallback=True,
            )

        # Case: Corrupt or unparseable timestamp
        if val.invalid_timestamp:
            return FollowUpDecision(
                decision=FollowUpRequired.REVIEW,
                confidence=None,
                reason=(
                    f"Manual review required: Corrupt or invalid ticket timestamp. "
                    f"Errors: {'; '.join(val.errors)}"
                ),
                evidence=["Invalid date_and_time format"],
                classifier_status=ModelStatus.INVALID_TIMESTAMP_REVIEW.value,
                is_conservative_fallback=True,
            )

        # Case: Other schema validation failure
        if not val.is_valid or cleaned is None or not cleaned.issue_description:
            return FollowUpDecision(
                decision=FollowUpRequired.REVIEW,
                confidence=None,
                reason=f"Validation failure: {'; '.join(val.errors)}",
                evidence=["Schema validation error"],
                classifier_status=ModelStatus.VALIDATION_ERROR.value,
                is_conservative_fallback=True,
            )

        # 2. Text extraction (Guaranteed: Student identifiers are strictly excluded)
        text = cleaned.issue_description.strip()
        raw_status = (cleaned.status or "").strip().lower()
        is_closed_status = raw_status in CLOSED_STATUSES
        is_open_status = raw_status in OPEN_STATUSES

        evidence: List[str] = []
        conflicts: List[str] = []

        # 3. Pattern signal detection
        has_callback = any(re.search(p, text, re.IGNORECASE) for p in CALLBACK_PATTERNS)
        has_unresolved = any(re.search(p, text, re.IGNORECASE) for p in UNRESOLVED_PATTERNS)
        has_resolved = any(re.search(p, text, re.IGNORECASE) for p in RESOLVED_PATTERNS)

        if has_callback:
            evidence.append("Explicit student callback request detected in text")
        if has_unresolved:
            evidence.append("Unresolved issue or urgent blocker phrasing detected in text")
        if has_resolved:
            evidence.append("Explicit confirmation of resolved issue detected in text")
        if is_closed_status:
            evidence.append(f"Ticket workflow status is closed ('{cleaned.status}')")
        elif is_open_status:
            evidence.append(f"Ticket workflow status is active ('{cleaned.status}')")

        # 4. Brevity & Ambiguity check
        words = text.split()
        is_too_brief = len(words) < 3 or len(text) < 15
        if is_too_brief and not has_callback:
            evidence.append("Issue description is very brief or lacks actionable context")

        # 5. Classifier query with generic model protection
        classifier_status = ModelStatus.PLACEHOLDER_PENDING_MODEL.value
        classifier_used = False
        classifier_label: Optional[str] = None
        classifier_score: Optional[float] = None

        if self.classifier is not None:
            c_status = getattr(self.classifier, "status", None)
            is_fine_tuned = getattr(self.classifier, "is_fine_tuned", False)

            if c_status in {
                ModelStatus.NOT_INITIALIZED,
                ModelStatus.DEPENDENCIES_UNAVAILABLE,
                ModelStatus.MODEL_NOT_FOUND,
            }:
                classifier_status = getattr(c_status, "value", str(c_status))
                evidence.append(f"Classifier unavailable: status={classifier_status}")
            elif c_status == ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED or not is_fine_tuned:
                classifier_status = ModelStatus.GENERIC_MODEL_NOT_FINE_TUNED.value
                evidence.append(
                    "Classifier rejected: Loaded model has generic labels and is not fine-tuned "
                    "for student support follow-up. Generic labels are excluded."
                )
            elif c_status == ModelStatus.READY and is_fine_tuned:
                inf = self.classifier.classify_text(text)
                if inf.get("status") == ModelStatus.READY:
                    classifier_used = True
                    classifier_status = ModelStatus.READY.value
                    classifier_label = inf.get("label")
                    classifier_score = inf.get("score")
                    evidence.append(
                        f"Task-specific classifier predicted '{classifier_label}' "
                        f"(confidence: {classifier_score:.2f})"
                    )
            elif c_status is not None:
                classifier_status = getattr(c_status, "value", str(c_status))
                evidence.append(f"Classifier unavailable: status={classifier_status}")

        # 6. Conflict detection
        # Conflict A: Closed status but student explicitly requested callback or reports unresolved issue
        if is_closed_status and (has_callback or has_unresolved):
            conflicts.append(
                "Ticket marked closed/resolved in workflow, but text reports an unresolved issue "
                "or explicit callback request"
            )

        # Conflict B: Text contains both resolved language and callback/unresolved language
        if has_resolved and (has_callback or has_unresolved):
            conflicts.append(
                "Text contains contradictory phrasing: reports resolution but also includes "
                "callback request or ongoing blocker"
            )

        # Conflict C: Model predicted NO, but text has explicit callback request
        if classifier_label == "NO" and has_callback:
            conflicts.append(
                "Classifier predicted NO follow-up, but student explicitly requested a callback in text"
            )

        # Conflict D: Model predicted YES, but text confirms resolution and ticket is closed
        if classifier_label == "YES" and has_resolved and is_closed_status:
            conflicts.append(
                "Classifier predicted YES follow-up, but student confirmed full resolution and ticket is closed"
            )

        # If any conflicts detected -> force safe REVIEW
        if conflicts:
            return FollowUpDecision(
                decision=FollowUpRequired.REVIEW,
                confidence=None,
                reason=f"Conflicting signals detected: {'; '.join(conflicts)}. Escalated for manual review.",
                evidence=evidence,
                conflicts=conflicts,
                classifier_status=classifier_status,
                classifier_used=classifier_used,
                is_conservative_fallback=True,
            )

        # 7. Evaluate for NO
        # Requirement: Reliable evidence that no further contact is needed.
        # Must NOT infer resolution from missing information or open/closed status alone.
        if has_resolved and is_closed_status and not has_callback and not has_unresolved:
            return FollowUpDecision(
                decision=FollowUpRequired.NO,
                confidence=0.90 if classifier_used and classifier_label == "NO" else None,
                reason=(
                    "Reliable evidence establishes that no further follow-up is needed: "
                    "Student explicitly confirmed resolution and ticket is closed."
                ),
                evidence=evidence,
                classifier_status=classifier_status,
                classifier_used=classifier_used,
            )

        if classifier_used and classifier_label == "NO" and has_resolved:
            return FollowUpDecision(
                decision=FollowUpRequired.NO,
                confidence=classifier_score,
                reason=(
                    f"Validated task-specific model classified NO (confidence: {classifier_score:.2f}) "
                    "with corroborating resolution text."
                ),
                evidence=evidence,
                classifier_status=classifier_status,
                classifier_used=classifier_used,
            )

        # 8. Evaluate for YES
        # Requirement: Sufficient evidence supports follow-up.
        # Explicit callback request is direct evidence
        if has_callback:
            return FollowUpDecision(
                decision=FollowUpRequired.YES,
                confidence=classifier_score if classifier_used and classifier_label == "YES" else None,
                reason=(
                    "Sufficient evidence supports follow-up: Student explicitly requested "
                    "a phone callback or advisor contact."
                ),
                evidence=evidence,
                classifier_status=classifier_status,
                classifier_used=classifier_used,
            )

        # Validated classifier predicted YES with corroborating open/actionable context
        if classifier_used and classifier_label == "YES" and not is_closed_status and not is_too_brief:
            return FollowUpDecision(
                decision=FollowUpRequired.YES,
                confidence=classifier_score,
                reason=(
                    f"Sufficient evidence supports follow-up: Validated model classified YES "
                    f"(confidence: {classifier_score:.2f}) with corroborating open inquiry."
                ),
                evidence=evidence,
                classifier_status=classifier_status,
                classifier_used=classifier_used,
            )

        # Clear unresolved blocker in active ticket with adequate description
        if has_unresolved and is_open_status and not is_too_brief and not classifier_used:
            return FollowUpDecision(
                decision=FollowUpRequired.YES,
                confidence=None,
                reason=(
                    "Sufficient evidence supports follow-up: Concrete unresolved blocker "
                    "detected in active support ticket."
                ),
                evidence=evidence,
                classifier_status=classifier_status,
                classifier_used=False,
            )

        # 9. Conservative Default -> REVIEW
        # Applies to:
        # - Too brief / ambiguous descriptions (e.g. "fees", "help")
        # - Open status alone without explicit callback or confirmed blocker
        # - Closed status alone without text confirming resolution
        # - Absent or generic classifier where heuristics cannot defensibly decide
        if is_too_brief:
            reason = (
                "Ambiguous evidence: Issue description is brief or lacks sufficient context "
                "to defensibly determine follow-up necessity. Escalated for human review."
            )
        elif is_closed_status and not has_resolved:
            reason = (
                "Ambiguous evidence: Ticket status is closed but text does not confirm resolution. "
                "Resolution cannot be assumed from status or missing text alone."
            )
        elif not classifier_used:
            reason = (
                "Conservative fallback: No validated task-specific classifier is available and "
                "text contains neither an explicit callback request nor confirmed resolution. "
                "Limitations: Rules are heuristic and not calibrated."
            )
        else:
            reason = (
                f"Classifier returned '{classifier_label or 'REVIEW'}'; "
                "routed for manual review."
            )

        return FollowUpDecision(
            decision=FollowUpRequired.REVIEW,
            confidence=classifier_score if classifier_used else None,
            reason=reason,
            evidence=evidence,
            classifier_status=classifier_status,
            classifier_used=classifier_used,
            is_conservative_fallback=True,
        )
