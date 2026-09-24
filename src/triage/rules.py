"""Gate 1 — deterministic rules.

Zero-cost pattern matching for routine ticket archetypes whose full verdict
(type, category, priority, assignment group) is decidable from vocabulary
alone. Everything else returns None and passes down to Gate 2.

Only ticket text is inspected — never the ticket_id prefix.
"""

import re
import time
from dataclasses import dataclass

from triage.models import (
    LABEL_FIELDS,
    AssignmentGroup,
    Category,
    Priority,
    TicketInput,
    TicketType,
    TriageVerdict,
)


@dataclass(frozen=True)
class _Rule:
    """A declarative rule: if any pattern matches, emit this fixed verdict."""

    name: str
    patterns: tuple[re.Pattern[str], ...]
    type: TicketType
    category: Category
    assignment_group: AssignmentGroup
    priority: Priority
    confidence: float
    rationale: str


def _compile(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p) for p in patterns)


_RULES: tuple[_Rule, ...] = (
    _Rule(
        name="password_reset",
        patterns=_compile(
            r"\bpassword\b",
            r"\blocked\s+out\b",
            r"\block(?:ed)?\s+(?:myself|me|us|them)\s+out\b",
            r"\bunlock(?:ed|ing)?\s+(?:my\s+|the\s+|our\s+)?(?:account|user|profile|login|logon)\b",
            r"\b(?:account|user|login|logon)\s+unlock",
        ),
        type=TicketType.SERVICE_REQUEST,
        category=Category.PASSWORD_ACCOUNT,
        assignment_group=AssignmentGroup.SERVICE_DESK,
        priority=Priority.P4,
        confidence=0.99,
        rationale="Matched password reset / account unlock pattern; routine Service Desk request.",
    ),
    _Rule(
        name="access_request",
        patterns=_compile(
            r"\bgrant\s+access\b",
            r"\brequest(?:ing|ed)?\s+access\b",
            r"\bneed(?:s|ing)?\s+access\b",
            r"\brequest(?:ing|ed)?\s+(?:a\s+|an\s+)?role\b",
        ),
        type=TicketType.SERVICE_REQUEST,
        category=Category.ACCESS_AUTHORIZATION,
        assignment_group=AssignmentGroup.SECURITY,
        priority=Priority.P4,
        confidence=0.85,
        rationale="Matched access-provisioning request pattern; routed to Security.",
    ),
    _Rule(
        name="transport_request",
        patterns=_compile(
            r"\bimport\s+transport\b",
            r"\bmove\s+transport\b",
            r"\btransport\s+request\b",
            r"\btransport\s+to\s+(?:qas|prd)\b",
        ),
        type=TicketType.SERVICE_REQUEST,
        category=Category.TRANSPORT_CHANGE,
        assignment_group=AssignmentGroup.BASIS,
        priority=Priority.P4,
        confidence=0.95,
        rationale="Matched transport import/release request; routed to Basis.",
    ),
)

# Urgency/impact/scope signals. If any appear, Gate 1 abstains even when a rule
# matches — a routine archetype accompanied by severity language is not P4.
# Deliberately excludes helpdesk vocabulary (error, fail, unable, cannot): those
# describe *why* the request exists, not its severity, and vetoing them would
# gut coverage of genuine routine tickets.
_URGENCY_VETO = _compile(
    r"\burgent\w*\b",
    r"\bcritical\b",
    r"\boutage\b",
    r"\bdown\b",
    r"\bshut\s?down\b",
    r"\bhalted\b",
    r"\bcrash",
    r"\ball\s+users\b",
    r"\beveryone\b",
    r"\b(?:whole|entire)\s+(?:team|department|site|company|plant|office|shift)\b",
    r"\b(?:company|site|plant|department)[- ]wide\b",
    # "our warehouse team has been unable", "the department cannot" — a
    # possessive/demonstrative + org unit + verb claims wider-than-one scope.
    r"\b(?:my|our|the|this)\s+(?:\w+\s+){0,2}"
    r"(?:team|department|company|office|site|plant)\s+"
    r"(?:is|are|was|were|has|have|cannot|can't|cant)\b",
    r"\b(?:several|multiple|many|numerous)\s+"
    r"(?:users|colleagues|employees|people|departments|teams)\b",
    r"\b(?:production|system|sap)\s+(?:is|was|went)\s+(?:down|offline|unavailable)\b",
    r"\bproduction\s+(?:halted|down|stopped|outage|impact|issue|problem)\b",
    r"\bmonth[\s-]?end\b",
    r"\byear[\s-]?end\b",
    r"\bdeadline\b",
    r"\basap\b",
    r"\bimmediat\w*\b",
)

# Signals of a *different* ticket domain. A routine archetype co-occurring with
# these is a collision ("password reset + not authorized in ME21N"), not a
# routine request — abstain rather than pick a side.
_DOMAIN_CONFLICT = _compile(
    r"\bnot\s+authorized\b",
    r"\bauthori[sz]ation\s+fail",
    r"\bshort\s+dump\b",
    r"\bdump\b",
    r"\bst22\b",
    r"\bidoc\b",
    r"\bwe02\b",
    r"\bpurchase\s+order\b",
    r"\bme21n\b",
    r"\binvoice\b",
    r"\bpricing\b",
    r"\bmaster\s+data\b",
    r"\bbatch\s+job\b",
    r"\bbackground\s+job\b",
    r"\binterface\b",
    r"\bspool\b",
)


def apply_rules(ticket: TicketInput) -> TriageVerdict | None:
    """Return a rules verdict for deterministic tickets, else None."""
    start = time.perf_counter()
    text = f"{ticket.short_description} {ticket.description}".lower()

    fired = [rule for rule in _RULES if any(p.search(text) for p in rule.patterns)]

    # Exactly one rule must fire on a clean ticket: 0 matches → nothing
    # deterministic; >1 → domain collision; veto/conflict → non-routine
    # signals present.
    if len(fired) != 1 or any(p.search(text) for p in (*_URGENCY_VETO, *_DOMAIN_CONFLICT)):
        return None

    rule = fired[0]
    return TriageVerdict(
        ticket_id=ticket.ticket_id,
        type=rule.type,
        category=rule.category,
        priority=rule.priority,
        assignment_group=rule.assignment_group,
        confidence=rule.confidence,
        decided_by="rules",
        label_source=dict.fromkeys(LABEL_FIELDS, "rules"),
        rationale=rule.rationale,
        similar_ticket_ids=[],
        cost_usd=0.0,
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )
