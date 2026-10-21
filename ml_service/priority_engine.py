"""Priority Assessment Engine for Student Support Ticket Triage.

Separates priority scoring, urgency assessment, and student interest evaluation
from model inference and follow-up eligibility determination. Applies conservative,
transparent, and evidence-based heuristic rules to assign defensible urgency levels,
expressed student interest levels, priority scores [0.0 - 100.0], categorical priority,
and recommended next operational actions.

IMPORTANT DISCLAIMER & INSTITUTIONAL CONTEXT:
The rules, criteria, weights, and scoring thresholds implemented in this module
are preliminary heuristic baselines developed for transparent triage demonstration.
They DO NOT represent officially approved or validated institutional policies of
VIT-AP University. They are NOT machine-learned or statistically calibrated on
historical datasets. They must not be claimed as accurate, optimal, or production-ready.

CORE ARCHITECTURAL & ETHICAL CONSTRAINTS:
1. PII Exclusion: Student identifiers (`student_name`, `application_number`,
   `registration_number`, `ticket_id`) and contact info (`phone_number`) are
   strictly excluded from predictive text analysis to prevent bias and privacy leakage.
2. Legacy Priority Invariance: The legacy `priority_level` field (always "High" in
   upstream feeds) is strictly ignored and must not influence scoring.
3. Separation of Concerns: Priority assessment is kept separate from model inference
   and follow-up eligibility.
4. Follow-Up Decision Respect:
   - NO follow-up tickets MUST receive `priority: NOT_APPLICABLE` and MUST NEVER
     trigger an outbound call recommendation.
   - REVIEW tickets MUST remain review cases and MUST NOT be converted into automatic
     call eligibility by a high urgency or high score.
   - YES follow-up tickets receive priority scoring ONLY if evidence is sufficient.
5. Expressed Interest Independence: `interest_level` is NEVER inferred from urgency.
   It is evaluated strictly on explicit student expressions of interest/intent.
   Support tickets without explicit expressed interest default to `UNKNOWN`.
6. Transparent Scoring: Factors and weights are fully documented and auditable.
7. Safe Null Fallback: If defensible evidence is absent, ambiguous, or insufficient,
   `priority_score` returns `None` with an explicit explanatory rationale.
8. Urgency vs Score Distinction: High urgency does NOT automatically yield a score of 100.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import MLServiceConfig
from .decision_engine import (
    CALLBACK_PATTERNS,
    FollowUpDecision,
    FollowUpDecisionEngine,
)
from .schemas import (
    FollowUpRequired,
    InterestLevel,
    ModelStatus,
    PriorityCategory,
    TicketInput,
    TicketPrediction,
    UrgencyLevel,
)
from .validator import ValidationResult, validate_ticket_input


# ==============================================================================
# Evidence Pattern Definitions for Urgency, Interest, and Impact
# ==============================================================================

# Urgency Patterns
IMMINENT_DEADLINE_PATTERNS: List[str] = [
    r"\bdeadline\s+(?:is\s+)?(?:today|tomorrow|tonight|approaching|urgent|imminent)\b",
    r"\b(?:last|final)\s+date\s+(?:is\s+)?(?:today|tomorrow|tonight|approaching)\b",
    r"\b(?:today|tomorrow)\s+is\s+the\s+(?:last|final)\s+date\b",
    r"\bwithin\s+(?:24|48)\s*hours?\b",
    r"\b(?:closes|expires|due)\s+(?:today|tomorrow|tonight|in\s+a\s+few\s+hours)\b",
    r"\bexam\s+(?:is\s+)?(?:today|tomorrow)\b",
    r"\bemergency\b",
]

URGENT_BLOCKER_PATTERNS: List[str] = [
    r"\b(?:payment|fee|tuition)\s+(?:was\s+)?(?:deducted|debited|charged)(?:\s+(?:twice|double|but))?\b",
    r"\bcharged\s+(?:twice|double)\b",
    r"\bunable\s+to\s+(?:submit|register|pay)\b",
    r"\bportal\s+(?:gives|shows|threw|displays|gave|returned|has)\s+(?:an?\s+)?error\b",
    r"\bcannot\s+(?:download|access)\s+(?:hall\s*ticket|admit\s*card)\b",
    r"\bimmediate\s+(?:assistance|action|help|escalation)\s+(?:needed|required)\b",
    r"\badmission\s+cancelled\b",
    r"\baccount\s+blocked\b",
    r"\bstranded\s+at\s+hostel\b",
]

MEDIUM_URGENCY_PATTERNS: List[str] = [
    r"\bdeadline\s+(?:next\s+week|in\s+\d+\s+days|next\s+month)\b",
    r"\bbranch\s+change\b",
    r"\bscholarship\b",
    r"\bfee\s+refund\b",
    r"\bgrade\s+(?:appeal|correction|re-evaluation|re-checking)\b",
    r"\bhostel\s+(?:allotment|allocation|room\s+change)\b",
    r"\bdocument\s+verification\b",
    r"\bunresolved\b",
    r"\bstill\s+(?:not\s+working|failing|pending|unresolved|stuck)\b",
    r"\bunable\s+to\s+(?:access|login|register|view)\b",
    r"\bhave\s+not\s+received\b",
    r"\bcertificate\s+(?:verification|submission)\b",
]

LOW_URGENCY_PATTERNS: List[str] = [
    r"\b(?:syllabus|curriculum|academic\s+calendar|holiday\s+schedule|bus\s+(?:pass|routes?)|library\s+timings?|sports|gym|wifi|canteen)\b",
    r"\b(?:general\s+(?:inquiry|query|question)|just\s+curious|whenever\s+possible|no\s+rush|at\s+your\s+convenience)\b",
    r"\b(?:campus\s+tour|visit|club\s+activities|orientation\s+schedule)\b",
]

# Student Expressed Interest Patterns (Task 4: Strictly separated from urgency)
HIGH_INTEREST_PATTERNS: List[str] = [
    r"\b(?:eager|keen|excited|thrilled)\s+to\s+(?:join|enroll|attend|study)\b",
    r"\b(?:deeply|very|strongly)\s+interested\s+in\s+(?:joining|enrolling|taking\s+admission|the\s+program)\b",
    r"\b(?:want|wish|intend)\s+to\s+(?:accept\s+my\s+admission\s+offer|confirm\s+my\s+seat|take\s+admission)\b",
    r"\bcommitted\s+to\s+joining\b",
    r"\bmy\s+top\s+choice\s+university\b",
    r"\bready\s+to\s+pay\s+admission\s+fee\s+to\s+confirm\s+seat\b",
]

MEDIUM_INTEREST_PATTERNS: List[str] = [
    r"\binterested\s+in\s+(?:knowing\s+more\s+about|learning\s+about|applying\s+for)\b",
    r"\bconsidering\s+(?:applying|enrolling|joining)\b",
    r"\bexploring\s+(?:admission\s+options|programs?|courses?)\b",
    r"\bwould\s+like\s+information\s+on\s+(?:admission|programs?|b\.?tech|m\.?tech|mba|phd)\b",
    r"\bcurious\s+about\s+(?:specializations?|curriculum|dual\s+degree)\b",
]

LOW_INTEREST_PATTERNS: List[str] = [
    r"\b(?:not|no\s+longer)\s+interested\s+(?:in\s+joining|anymore|in\s+this\s+program)\b",
    r"\b(?:cancel|withdraw)\s+(?:my\s+)?(?:admission|application|seat)\b",
    r"\bjoining\s+(?:another|different)\s+(?:college|university|institute)\b",
    r"\bdecided\s+not\s+to\s+join\b",
    r"\bdecline\s+(?:the\s+)?admission\s+offer\b",
]

# Impact / Blocker Severity Patterns
CRITICAL_IMPACT_PATTERNS: List[str] = [
    r"\b(?:payment|fee|tuition)\s+(?:was\s+)?(?:deducted|debited|charged)(?:\s+(?:twice|double|but))?\b",
    r"\bcharged\s+(?:twice|double)\b",
    r"\badmission\s+cancelled\b",
    r"\baccount\s+blocked\b",
    r"\bexam\s+(?:is\s+)?(?:today|tomorrow)\b",
    r"\bcannot\s+(?:download|access)\s+(?:hall\s*ticket|admit\s*card)\b",
    r"\bunable\s+to\s+(?:submit|register|pay|upload)\b",
    r"\bstranded\s+at\s+hostel\b",
    r"\bportal\s+(?:gives|shows|threw|displays|gave|returned|has)\s+(?:an?\s+)?error\b",
]

MODERATE_IMPACT_PATTERNS: List[str] = [
    r"\bscholarship\b",
    r"\bbranch\s+change\b",
    r"\bgrade\s+appeal\b",
    r"\bfee\s+refund\b",
    r"\bhostel\b",
    r"\bunable\s+to\s+(?:access|login|register)\b",
    r"\bdocument\s+verification\b",
]

ROUTINE_IMPACT_PATTERNS: List[str] = [
    r"\b(?:syllabus|curriculum|calendar|bus|library|timings|sports|club|wifi|canteen)\b",
    r"\bgeneral\s+(?:inquiry|query|question)\b",
]


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class ScoreBreakdown:
    """Detailed transparent breakdown of factors and weights contributing to priority score.
    
    Attributes:
        urgency_score: Points contributed by assessed urgency [0.0 - 50.0].
        impact_score: Points contributed by blocker / issue impact [0.0 - 30.0].
        actionability_score: Points contributed by student callback / engagement [0.0 - 20.0].
        total_score: Resulting composite score [0.0 - 100.0], or None if undefensible.
        factors_applied: List of human-readable factor descriptions applied during scoring.
    """
    urgency_score: float = 0.0
    impact_score: float = 0.0
    actionability_score: float = 0.0
    total_score: Optional[float] = None
    factors_applied: List[str] = field(default_factory=list)


@dataclass
class PriorityAssessment:
    """Comprehensive priority assessment result for a student support ticket.
    
    Attributes:
        urgency: Assessed urgency level (HIGH, MEDIUM, LOW, UNKNOWN).
        interest_level: Assessed student expressed interest (HIGH, MEDIUM, LOW, UNKNOWN).
        priority_score: Numerical priority score [0.0 - 100.0], or None when undefensible.
        priority: Categorical priority assignment (HIGH, MEDIUM, LOW, NOT_APPLICABLE).
        recommended_action: Specific, defensible next action for support operations.
        rationale: Explanatory rationale detailing why the score/action was assigned.
        evidence: Concrete textual evidence snippets detected during evaluation.
        score_breakdown: Transparent numerical breakdown of score components.
        is_defensible: Flag indicating whether assessment was backed by sufficient evidence.
    """
    urgency: UrgencyLevel
    interest_level: InterestLevel
    priority_score: Optional[float]
    priority: PriorityCategory
    recommended_action: str
    rationale: str
    evidence: List[str] = field(default_factory=list)
    score_breakdown: Optional[ScoreBreakdown] = None
    is_defensible: bool = True


# ==============================================================================
# Priority Assessment Engine
# ==============================================================================

class PriorityAssessmentEngine:
    """Evaluates student support tickets for urgency, expressed interest, and priority.
    
    Provides transparent, rule-based scoring and operational recommendations
    while strictly respecting follow-up decisions, excluding PII, and avoiding
    spurious claims of machine-learned precision.
    """

    # Scoring weights
    MAX_URGENCY_POINTS: float = 50.0
    MAX_IMPACT_POINTS: float = 30.0
    MAX_ACTIONABILITY_POINTS: float = 20.0

    # Categorical Priority Thresholds
    HIGH_PRIORITY_THRESHOLD: float = 70.0
    MEDIUM_PRIORITY_THRESHOLD: float = 40.0

    def __init__(
        self,
        config: Optional[MLServiceConfig] = None,
        decision_engine: Optional[Any] = None,
    ) -> None:
        self.config = config or MLServiceConfig()
        self.decision_engine = decision_engine or FollowUpDecisionEngine(config=self.config)

    def assess_urgency(self, text: str) -> Tuple[UrgencyLevel, List[str], float]:
        """Assess issue urgency based on temporal constraints and blocker severity.
        
        Criteria:
        - HIGH: Clearly stated imminent deadline (today/tomorrow/within 48h) or serious
          time-sensitive application blocker (e.g., fee deducted twice, account locked).
          Score: 50.0 for imminent hard deadlines; 40.0 for urgent blockers without hard deadline.
        - MEDIUM: Unresolved issue with meaningful but not imminent deadline or consequence
          (e.g., refund pending, scholarship verification, branch change next month).
          Score: 25.0 points.
        - LOW: Non-urgent inquiry with no indicated deadline or serious consequence
          (e.g., syllabus, library timings, bus routes, holiday calendar).
          Score: 10.0 points.
        - UNKNOWN: Insufficient or ambiguous information to assess urgency.
          Score: 0.0 points.
          
        Returns:
            Tuple[UrgencyLevel, List[str], float]: (urgency_level, evidence_list, urgency_score)
        """
        evidence: List[str] = []

        # Check imminent deadline (50.0 points)
        imminent_matches = [p for p in IMMINENT_DEADLINE_PATTERNS if re.search(p, text, re.IGNORECASE)]
        if imminent_matches:
            evidence.append("Imminent deadline or time-critical emergency detected in text")
            return UrgencyLevel.HIGH, evidence, 50.0

        # Check urgent blocker without hard deadline (40.0 points)
        blocker_matches = [p for p in URGENT_BLOCKER_PATTERNS if re.search(p, text, re.IGNORECASE)]
        if blocker_matches:
            evidence.append("Urgent time-sensitive application or financial blocker detected in text")
            return UrgencyLevel.HIGH, evidence, 40.0

        # Check medium urgency (25.0 points)
        medium_matches = [p for p in MEDIUM_URGENCY_PATTERNS if re.search(p, text, re.IGNORECASE)]
        if medium_matches:
            evidence.append("Unresolved administrative or academic inquiry with non-imminent consequence detected")
            return UrgencyLevel.MEDIUM, evidence, 25.0

        # Check low urgency (10.0 points)
        low_matches = [p for p in LOW_URGENCY_PATTERNS if re.search(p, text, re.IGNORECASE)]
        if low_matches:
            evidence.append("Routine informational inquiry with no stated deadline or blocker detected")
            return UrgencyLevel.LOW, evidence, 10.0

        # Default fallback: Unknown urgency (0.0 points)
        evidence.append("Ambiguous or insufficient text to establish urgency level")
        return UrgencyLevel.UNKNOWN, evidence, 0.0

    def assess_interest(self, text: str) -> Tuple[InterestLevel, List[str]]:
        """Assess student expressed interest in academic / admission programs.
        
        TASK 4 CONSTRAINT:
        Do not infer `interest_level` from urgency. Assign HIGH, MEDIUM, or LOW
        strictly when the student description provides defensible evidence of
        expressed interest. Otherwise return UNKNOWN.
        
        Criteria:
        - HIGH: Explicit statement of enthusiasm, commitment, or eager intent to enroll.
        - MEDIUM: Expressed curiosity or exploratory inquiry about program offerings.
        - LOW: Explicit statement of disinterest, withdrawal, or joining another college.
        - UNKNOWN: All routine student-support tickets without explicit expressed interest.
        
        Returns:
            Tuple[InterestLevel, List[str]]: (interest_level, evidence_list)
        """
        evidence: List[str] = []

        # 1. High Interest Check
        high_matches = [p for p in HIGH_INTEREST_PATTERNS if re.search(p, text, re.IGNORECASE)]
        if high_matches:
            evidence.append("Student explicitly expressed keen interest / commitment to enroll")
            return InterestLevel.HIGH, evidence

        # 2. Low Interest Check (Withdrawal / Disinterest)
        low_matches = [p for p in LOW_INTEREST_PATTERNS if re.search(p, text, re.IGNORECASE)]
        if low_matches:
            evidence.append("Student explicitly expressed withdrawal, declining admission, or disinterest")
            return InterestLevel.LOW, evidence

        # 3. Medium Interest Check (Exploration)
        medium_matches = [p for p in MEDIUM_INTEREST_PATTERNS if re.search(p, text, re.IGNORECASE)]
        if medium_matches:
            evidence.append("Student explicitly expressed exploratory interest in program offerings")
            return InterestLevel.MEDIUM, evidence

        # 4. Default: UNKNOWN (Must not infer interest from urgency or routine queries)
        evidence.append("No explicit expression of student interest/intent detected; defaulted to UNKNOWN")
        return InterestLevel.UNKNOWN, evidence

    def assess_impact(self, text: str) -> Tuple[str, List[str], float]:
        """Assess operational and financial impact / severity of the ticket issue.
        
        Returns:
            Tuple[str, List[str], float]: (impact_tier, evidence_list, impact_score)
        """
        evidence: List[str] = []

        if any(re.search(p, text, re.IGNORECASE) for p in CRITICAL_IMPACT_PATTERNS):
            evidence.append("Critical impact: Financial transaction failure or operational lockout")
            return "CRITICAL", evidence, self.MAX_IMPACT_POINTS  # 30.0

        if any(re.search(p, text, re.IGNORECASE) for p in MODERATE_IMPACT_PATTERNS):
            evidence.append("Moderate impact: Academic progression, scholarship, or housing inquiry")
            return "MODERATE", evidence, 15.0

        if any(re.search(p, text, re.IGNORECASE) for p in ROUTINE_IMPACT_PATTERNS):
            evidence.append("Low impact: Routine informational or campus life query")
            return "ROUTINE", evidence, 5.0

        evidence.append("Neutral impact: General support inquiry without acute severity markers")
        return "NEUTRAL", evidence, 0.0

    def assess_actionability(self, text: str, has_callback: bool) -> Tuple[str, List[str], float]:
        """Assess student request actionability and direct advisor engagement requirement.
        
        Returns:
            Tuple[str, List[str], float]: (actionability_tier, evidence_list, actionability_score)
        """
        evidence: List[str] = []

        if has_callback:
            evidence.append("Direct actionability: Student explicitly requested phone callback or advisor contact")
            return "DIRECT_CALLBACK_REQUEST", evidence, self.MAX_ACTIONABILITY_POINTS  # 20.0

        # Check if unresolved problem without explicit callback phrasing
        if any(re.search(p, text, re.IGNORECASE) for p in URGENT_BLOCKER_PATTERNS + MEDIUM_URGENCY_PATTERNS):
            evidence.append("Indirect actionability: Ongoing unresolved issue requiring advisor investigation")
            return "UNRESOLVED_ISSUE", evidence, 10.0

        evidence.append("Standard actionability: Informational inquiry; self-service or queue response sufficient")
        return "ROUTINE", evidence, 0.0

    def assess(
        self,
        ticket: Union[TicketInput, Dict[str, Any], Any],
        follow_up_decision: Optional[Union[FollowUpDecision, FollowUpRequired, str]] = None,
        validation_result: Optional[ValidationResult] = None,
    ) -> PriorityAssessment:
        """Perform conservative, defensible priority assessment for a student ticket.
        
        Args:
            ticket: Raw dictionary or TicketInput instance.
            follow_up_decision: Optional precomputed FollowUpDecision or FollowUpRequired.
            validation_result: Optional precomputed ValidationResult.
            
        Returns:
            PriorityAssessment: Structured evaluation containing urgency, interest,
            score, categorical priority, and recommended action.
        """
        # 1. Validation check
        val = validation_result or validate_ticket_input(ticket)
        cleaned = val.cleaned_ticket

        # Case 1: Missing or empty description
        if val.empty_description or cleaned is None or not cleaned.issue_description:
            return PriorityAssessment(
                urgency=UrgencyLevel.UNKNOWN,
                interest_level=InterestLevel.UNKNOWN,
                priority_score=None,
                priority=PriorityCategory.NOT_APPLICABLE,
                recommended_action="Manual advisor review required: Issue description is missing or empty.",
                rationale=(
                    "Insufficient evidence: Issue description is missing or empty. "
                    "Cannot defensibly calculate urgency, interest, or priority score."
                ),
                evidence=["Empty or whitespace-only issue_description"],
                is_defensible=False,
            )

        # Case 2: Corrupt or unparseable timestamp
        if val.invalid_timestamp:
            return PriorityAssessment(
                urgency=UrgencyLevel.UNKNOWN,
                interest_level=InterestLevel.UNKNOWN,
                priority_score=None,
                priority=PriorityCategory.NOT_APPLICABLE,
                recommended_action="Manual advisor review required: Invalid ticket creation timestamp.",
                rationale=(
                    f"Manual review required: Corrupt timestamp prevents temporal triage. "
                    f"Errors: {'; '.join(val.errors)}"
                ),
                evidence=["Invalid date_and_time format"],
                is_defensible=False,
            )

        # 2. Text Extraction & Sanitization
        # PII exclusion guaranteed: Only issue_description text is used for inference.
        # Legacy priority_level is strictly ignored.
        text = cleaned.issue_description.strip()
        words = text.split()
        is_too_brief = len(words) < 3 or len(text) < 15

        # 3. Determine Follow-Up Decision Context
        fu_decision: FollowUpRequired
        fu_reason: str = ""
        fu_conflicts: List[str] = []

        if follow_up_decision is not None:
            if isinstance(follow_up_decision, FollowUpDecision):
                fu_decision = follow_up_decision.decision
                fu_reason = follow_up_decision.reason
                fu_conflicts = follow_up_decision.conflicts
            elif isinstance(follow_up_decision, FollowUpRequired):
                fu_decision = follow_up_decision
            elif isinstance(follow_up_decision, str):
                try:
                    fu_decision = FollowUpRequired(follow_up_decision.upper())
                except ValueError:
                    fu_decision = FollowUpRequired.REVIEW
            else:
                fu_decision = FollowUpRequired.REVIEW
        else:
            evaluated_fu = self.decision_engine.evaluate(cleaned, val)
            fu_decision = evaluated_fu.decision
            fu_reason = evaluated_fu.reason
            fu_conflicts = evaluated_fu.conflicts

        # 4. Assess Urgency and Expressed Interest independently
        urgency, urgency_ev, urgency_pts = self.assess_urgency(text)
        interest, interest_ev = self.assess_interest(text)

        # 5. Respect Follow-Up Decision (Tasks 7 & 8)

        # ----------------------------------------------------------------------
        # Case A: Follow-up is NO
        # Requirement: NO must result in priority: NOT_APPLICABLE; must not trigger a call recommendation.
        # ----------------------------------------------------------------------
        if fu_decision == FollowUpRequired.NO:
            return PriorityAssessment(
                urgency=urgency,
                interest_level=interest,
                priority_score=None,
                priority=PriorityCategory.NOT_APPLICABLE,
                recommended_action="No advisor follow-up required: Issue is confirmed resolved or closed.",
                rationale=(
                    "Follow-up decision is NO: Ticket issue is confirmed resolved or closed. "
                    "Priority scoring is not applicable and outbound call recommendation is suppressed."
                ),
                evidence=urgency_ev + interest_ev + ["Follow-up decision is NO; outbound call suppressed"],
                score_breakdown=ScoreBreakdown(
                    urgency_score=0.0,
                    impact_score=0.0,
                    actionability_score=0.0,
                    total_score=None,
                    factors_applied=["Score withheld: Follow-up required is NO"],
                ),
                is_defensible=True,
            )

        # ----------------------------------------------------------------------
        # Case B: Follow-up is REVIEW
        # Requirement: REVIEW must remain a review case. It must not be converted
        # into automatic call eligibility by a high score.
        # ----------------------------------------------------------------------
        if fu_decision == FollowUpRequired.REVIEW:
            review_reason = (
                f"Follow-up decision is REVIEW ({fu_reason or 'ambiguous, brief, or conflicting signals'}). "
                "Automatic call recommendation and priority scoring are withheld pending human review."
            )
            return PriorityAssessment(
                urgency=urgency,
                interest_level=interest,
                priority_score=None,
                priority=PriorityCategory.NOT_APPLICABLE,
                recommended_action="Manual advisor review required: Inspect ticket details before taking operational action.",
                rationale=review_reason,
                evidence=urgency_ev + interest_ev + (fu_conflicts or ["Follow-up decision is REVIEW"]),
                score_breakdown=ScoreBreakdown(
                    urgency_score=0.0,
                    impact_score=0.0,
                    actionability_score=0.0,
                    total_score=None,
                    factors_applied=["Score withheld: Follow-up decision is REVIEW"],
                ),
                is_defensible=True,
            )

        # ----------------------------------------------------------------------
        # Case C: Follow-up is YES
        # Requirement: YES may receive a priority assessment if evidence is sufficient.
        # ----------------------------------------------------------------------
        has_callback = any(re.search(p, text, re.IGNORECASE) for p in CALLBACK_PATTERNS)

        # Compute Transparent Score
        impact_tier, impact_ev, impact_pts = self.assess_impact(text)
        action_tier, action_ev, action_pts = self.assess_actionability(text, has_callback)

        # Sub-case: Insufficient evidence to calculate defensible priority score
        # (e.g. Student said "Please call me back", but provided no context or severity on the issue)
        if urgency == UrgencyLevel.UNKNOWN and impact_tier == "NEUTRAL":
            return PriorityAssessment(
                urgency=UrgencyLevel.UNKNOWN,
                interest_level=interest,
                priority_score=None,
                priority=PriorityCategory.NOT_APPLICABLE,
                recommended_action="Advisor follow-up required: Contact student to gather issue details (context missing in ticket).",
                rationale=(
                    "Insufficient evidence: Student explicitly requested follow-up, but ticket "
                    "lacks specific details regarding the underlying issue. "
                    "Priority score cannot be defensibly calculated."
                ),
                evidence=urgency_ev + interest_ev + ["Issue description is brief / context missing"],
                score_breakdown=ScoreBreakdown(
                    urgency_score=0.0,
                    impact_score=0.0,
                    actionability_score=0.0,
                    total_score=None,
                    factors_applied=["Score withheld: Insufficient issue details to calculate priority"],
                ),
                is_defensible=False,
            )

        raw_score = urgency_pts + impact_pts + action_pts
        # Bound score strictly to [0.0, 100.0]
        final_score = round(max(0.0, min(100.0, float(raw_score))), 1)

        factors_applied = [
            f"Urgency factor: {urgency.value} (+{urgency_pts:.1f} pts)",
            f"Impact factor: {impact_tier} (+{impact_pts:.1f} pts)",
            f"Actionability factor: {action_tier} (+{action_pts:.1f} pts)",
        ]

        # Categorical Priority Assignment
        if final_score >= self.HIGH_PRIORITY_THRESHOLD:
            priority_cat = PriorityCategory.HIGH
            recommended_action = "Urgent advisor callback: Contact student promptly to resolve time-sensitive blocker."
        elif final_score >= self.MEDIUM_PRIORITY_THRESHOLD:
            priority_cat = PriorityCategory.MEDIUM
            recommended_action = "Standard advisor callback: Contact student during standard support hours to resolve issue."
        else:
            priority_cat = PriorityCategory.LOW
            recommended_action = "Routine support outreach: Contact student or provide email guidance as queue capacity permits."

        rationale = (
            f"Defensible priority score {final_score}/100 ({priority_cat.value}) based on: "
            f"urgency={urgency.value} ({urgency_pts:.0f} pts), "
            f"impact={impact_tier} ({impact_pts:.0f} pts), "
            f"actionability={action_tier} ({action_pts:.0f} pts). "
            "Transparent heuristic evaluation; not statistically calibrated."
        )

        all_evidence = urgency_ev + interest_ev + impact_ev + action_ev

        return PriorityAssessment(
            urgency=urgency,
            interest_level=interest,
            priority_score=final_score,
            priority=priority_cat,
            recommended_action=recommended_action,
            rationale=rationale,
            evidence=all_evidence,
            score_breakdown=ScoreBreakdown(
                urgency_score=urgency_pts,
                impact_score=impact_pts,
                actionability_score=action_pts,
                total_score=final_score,
                factors_applied=factors_applied,
            ),
            is_defensible=True,
        )

    def assess_ticket(
        self,
        ticket: Union[TicketInput, Dict[str, Any], Any],
        follow_up_decision: Optional[Union[FollowUpDecision, FollowUpRequired, str]] = None,
        validation_result: Optional[ValidationResult] = None,
        model_status: Optional[str] = None,
    ) -> TicketPrediction:
        """Convenience method to evaluate ticket and produce a full TicketPrediction instance.
        
        Args:
            ticket: Raw dictionary or TicketInput instance.
            follow_up_decision: Optional precomputed follow-up decision.
            validation_result: Optional precomputed validation result.
            model_status: Optional model status override.
            
        Returns:
            TicketPrediction: Fully populated, validated structured output.
        """
        val = validation_result or validate_ticket_input(ticket)
        cleaned = val.cleaned_ticket

        # Extract ticket_id safely
        ticket_id: Optional[str] = None
        if cleaned and cleaned.ticket_id:
            ticket_id = cleaned.ticket_id
        elif isinstance(ticket, dict) and ticket.get("ticket_id"):
            ticket_id = str(ticket.get("ticket_id"))
        elif isinstance(ticket, TicketInput) and ticket.ticket_id:
            ticket_id = ticket.ticket_id

        # Determine follow-up decision
        fu_decision_obj: Optional[FollowUpDecision] = None
        if follow_up_decision is None:
            fu_decision_obj = self.decision_engine.evaluate(ticket, val)
            fu_enum = fu_decision_obj.decision
            fu_reason = fu_decision_obj.reason
            status = model_status or fu_decision_obj.classifier_status or ModelStatus.READY.value
        elif isinstance(follow_up_decision, FollowUpDecision):
            fu_decision_obj = follow_up_decision
            fu_enum = fu_decision_obj.decision
            fu_reason = fu_decision_obj.reason
            status = model_status or fu_decision_obj.classifier_status or ModelStatus.READY.value
        elif isinstance(follow_up_decision, FollowUpRequired):
            fu_enum = follow_up_decision
            fu_reason = f"Explicitly configured follow-up decision: {fu_enum.value}"
            status = model_status or ModelStatus.READY.value
        else:
            try:
                fu_enum = FollowUpRequired(str(follow_up_decision).upper())
            except ValueError:
                fu_enum = FollowUpRequired.REVIEW
            fu_reason = f"Explicitly configured follow-up decision: {fu_enum.value}"
            status = model_status or ModelStatus.READY.value

        assessment = self.assess(ticket, follow_up_decision=fu_decision_obj or fu_enum, validation_result=val)

        return TicketPrediction(
            ticket_id=ticket_id,
            follow_up_required=fu_enum,
            interest_level=assessment.interest_level,
            urgency=assessment.urgency,
            follow_up_reason=fu_reason or assessment.rationale,
            priority_score=assessment.priority_score,
            priority=assessment.priority,
            recommended_action=assessment.recommended_action,
            model_status=status,
        )
