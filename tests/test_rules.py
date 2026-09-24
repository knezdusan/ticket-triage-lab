"""Tests for Gate 1 deterministic rules."""

from datetime import UTC, datetime

from triage.models import (
    AssignmentGroup,
    Category,
    Priority,
    TicketInput,
    TicketType,
    TriageVerdict,
)
from triage.rules import apply_rules


def make_ticket(
    short_desc: str,
    desc: str,
    ticket_id: str = "INC000001",
    sap_module: str | None = None,
) -> TicketInput:
    """Helper to create minimal TicketInput for testing."""
    return TicketInput(
        ticket_id=ticket_id,
        created_at=datetime.now(UTC),
        short_description=short_desc,
        description=desc,
        requester="tester@example.com",
        sap_module=sap_module,
    )


class TestGate1Rules:
    def test_password_reset_matches(self):
        ticket = make_ticket(
            short_desc="Password reset required for user JSMITH",
            desc="I have locked myself out of SAP GUI after 3 incorrect attempts. Please unlock.",
        )
        verdict = apply_rules(ticket)

        assert verdict is not None
        assert isinstance(verdict, TriageVerdict)
        assert verdict.decided_by == "rules"
        assert verdict.type == TicketType.SERVICE_REQUEST
        assert verdict.category == Category.PASSWORD_ACCOUNT
        assert verdict.assignment_group == AssignmentGroup.SERVICE_DESK
        assert verdict.priority == Priority.P4
        assert verdict.confidence >= 0.95
        assert verdict.cost_usd == 0.0
        assert verdict.latency_ms >= 0.0

    def test_routine_access_request_matches(self):
        ticket = make_ticket(
            short_desc="Please grant access to VA01",
            desc="New team member needs standard role for sales order entry. "
            "Please request access to VA01.",
        )
        verdict = apply_rules(ticket)

        assert verdict is not None
        assert verdict.decided_by == "rules"
        assert verdict.type == TicketType.SERVICE_REQUEST
        assert verdict.category == Category.ACCESS_AUTHORIZATION
        assert verdict.assignment_group == AssignmentGroup.SECURITY
        assert verdict.priority == Priority.P4

    def test_short_dump_is_passed_down(self):
        # Even though category is clear (short_dump), priority and assignment group
        # are unknown/variable. Gate 1 must return None!
        ticket = make_ticket(
            short_desc="ST22 runtime error in billing",
            desc="Getting MESSAGE_TYPE_X short dump when executing VF01. Production is impacted.",
        )
        verdict = apply_rules(ticket)
        assert verdict is None

    def test_collision_returns_none(self):
        # Mentions password reset AND an authorization issue/tcode error
        ticket = make_ticket(
            short_desc="Password reset and ME21N purchase order error",
            desc="I need my password reset but also getting not authorized "
            "error in purchasing ME21N.",
        )
        verdict = apply_rules(ticket)
        assert verdict is None

    def test_outage_or_high_urgency_words_prevent_p4_rules(self):
        # A password issue accompanied by critical outage phrasing should not be auto-closed as P4
        ticket = make_ticket(
            short_desc="Password locked out during critical month end plant shutdown",
            desc="Entire shift locked out. Production halted. Need urgent fix.",
        )
        verdict = apply_rules(ticket)
        # Should pass down to similarity or model to evaluate true priority
        assert verdict is None

    def test_vague_ticket_returns_none(self):
        ticket = make_ticket(
            short_desc="System is slow today",
            desc="Transactions take a long time to load in PRD.",
        )
        assert apply_rules(ticket) is None
