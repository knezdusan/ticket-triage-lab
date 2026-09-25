"""Tests for Gate 3 — mocked ChatModel, no Azure calls."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from triage.classifier import _schema_for, classify_ticket_gate3
from triage.models import (
    AssignmentGroup,
    Category,
    Level,
    Priority,
    TicketInput,
    TicketType,
    TriageVerdict,
)


class FakeChat:
    """Records the schema/messages it was called with; returns a canned
    parsed object or raises, like the real complete_structured."""

    def __init__(self, parsed: BaseModel | None = None, error: Exception | None = None):
        self.parsed = parsed
        self.error = error
        self.last_schema = None
        self.last_messages = None
        self.calls = 0
        self.last_kwargs = None
        self.last_usage = SimpleNamespace(prompt_tokens=400, completion_tokens=80)

    def complete_structured(self, messages, schema, **kwargs):
        self.calls += 1
        self.last_messages = messages
        self.last_schema = schema
        self.last_kwargs = kwargs
        if self.error:
            raise self.error
        return self.parsed


def make_ticket() -> TicketInput:
    return TicketInput(
        ticket_id="INC000777",
        created_at=datetime.now(UTC),
        short_description="ST22 dump in billing",
        description="MESSAGE_TYPE_X short dump when executing VF01 for one user.",
        requester="tester@example.com",
    )


def make_locked(**labels) -> TriageVerdict:
    """A partial verdict as Gate 2 would emit it."""
    sources = {f: "similarity" for f, v in labels.items() if v is not None}
    return TriageVerdict(
        ticket_id="INC000777",
        confidence=0.9,
        decided_by="similarity",
        label_source=sources,
        rationale="Top-1 similarity 0.87.",
        similar_ticket_ids=["INC000031"],
        cost_usd=0.00002,
        **labels,
    )


class TestSchemaFactory:
    def test_full_fallback_asks_everything(self):
        schema = _schema_for(frozenset({"type", "category", "assignment_group"}))
        fields = set(schema.model_fields)
        assert fields == {
            "type",
            "category",
            "assignment_group",
            "impact",
            "urgency",
            "rationale",
        }

    def test_locked_fields_not_in_schema(self):
        schema = _schema_for(frozenset({"type"}))
        fields = set(schema.model_fields)
        assert "type" in fields
        assert "category" not in fields
        assert "assignment_group" not in fields

    def test_severity_only_when_all_locked(self):
        schema = _schema_for(frozenset())
        assert set(schema.model_fields) == {"impact", "urgency", "rationale"}


class TestClassifyGate3:
    def test_full_fallback(self):
        schema = _schema_for(frozenset({"type", "category", "assignment_group"}))
        parsed = schema(
            type=TicketType.INCIDENT,
            category=Category.SHORT_DUMP,
            assignment_group=AssignmentGroup.BASIS,
            impact=Level.HIGH,
            urgency=Level.MEDIUM,
            rationale="Runtime error affecting one team's billing run.",
        )
        chat = FakeChat(parsed)
        verdict = classify_ticket_gate3(make_ticket(), chat)

        assert chat.calls == 1
        assert verdict is not None
        assert verdict.decided_by == "model"
        assert verdict.type == TicketType.INCIDENT
        assert verdict.category == Category.SHORT_DUMP
        assert verdict.assignment_group == AssignmentGroup.BASIS
        # high impact + medium urgency → P2 per the matrix
        assert verdict.priority == Priority.P2
        assert verdict.impact == Level.HIGH
        assert verdict.urgency == Level.MEDIUM
        assert verdict.label_source == {
            "type": "model",
            "category": "model",
            "priority": "model",
            "assignment_group": "model",
        }
        assert verdict.cost_usd > 0.0
        assert verdict.latency_ms >= 0.0

    def test_partial_merge_keeps_locked_values(self):
        locked = make_locked(
            category=Category.SHORT_DUMP,
            assignment_group=AssignmentGroup.BASIS,
        )
        schema = _schema_for(frozenset({"type"}))
        parsed = schema(
            type=TicketType.INCIDENT,
            impact=Level.HIGH,
            urgency=Level.LOW,
            rationale="Single-user dump, no urgency pressure in text.",
        )
        chat = FakeChat(parsed)
        verdict = classify_ticket_gate3(make_ticket(), chat, locked)

        assert verdict is not None
        assert verdict.decided_by == "cascade"
        # Locked fields unchanged with similarity provenance.
        assert verdict.category == Category.SHORT_DUMP
        assert verdict.assignment_group == AssignmentGroup.BASIS
        assert verdict.label_source["category"] == "similarity"
        assert verdict.label_source["assignment_group"] == "similarity"
        # New fields sourced to the model; priority derived.
        assert verdict.type == TicketType.INCIDENT
        assert verdict.label_source["type"] == "model"
        assert verdict.label_source["priority"] == "model"
        assert verdict.priority == Priority.P3  # high + low
        assert verdict.impact == Level.HIGH
        assert verdict.urgency == Level.LOW
        # Carried context from Gate 2.
        assert verdict.similar_ticket_ids == ["INC000031"]
        assert verdict.cost_usd > locked.cost_usd
        assert verdict.confidence == pytest.approx(min(0.9, 0.8))

    def test_context_injection_in_prompt(self):
        locked = make_locked(category=Category.BATCH_JOB)
        schema = _schema_for(frozenset({"type", "assignment_group"}))
        parsed = schema(
            type=TicketType.INCIDENT,
            assignment_group=AssignmentGroup.BASIS,
            impact=Level.LOW,
            urgency=Level.LOW,
            rationale="Overnight batch job failed once.",
        )
        chat = FakeChat(parsed)
        classify_ticket_gate3(make_ticket(), chat, locked)

        user_msg = chat.last_messages[1]["content"]
        assert "batch_job" in user_msg
        assert "Already verified" in user_msg

    def test_fully_locked_verdict_short_circuits(self):
        locked = make_locked(
            type=TicketType.SERVICE_REQUEST,
            category=Category.PASSWORD_ACCOUNT,
            priority=Priority.P4,
            assignment_group=AssignmentGroup.SERVICE_DESK,
        )
        chat = FakeChat()
        verdict = classify_ticket_gate3(make_ticket(), chat, locked)
        assert chat.calls == 0
        assert verdict is locked

    def test_severity_only_call_when_three_locked(self):
        locked = make_locked(
            type=TicketType.INCIDENT,
            category=Category.SHORT_DUMP,
            assignment_group=AssignmentGroup.BASIS,
        )
        schema = _schema_for(frozenset())
        parsed = schema(
            impact=Level.HIGH,
            urgency=Level.HIGH,
            rationale="Plant-wide outage in progress.",
        )
        chat = FakeChat(parsed)
        verdict = classify_ticket_gate3(make_ticket(), chat, locked)

        assert chat.calls == 1
        assert set(chat.last_schema.model_fields) == {"impact", "urgency", "rationale"}
        assert verdict is not None
        assert verdict.priority == Priority.P1
        assert verdict.label_source["priority"] == "model"

    def test_refusal_propagates(self):
        chat = FakeChat(error=ValueError("structured output not parsed (refusal: x)"))
        with pytest.raises(ValueError, match="refusal"):
            classify_ticket_gate3(make_ticket(), chat)


class TestFewShotExamples:
    def test_examples_rendered_into_prompt(self):
        from triage.models import LabelledTicket

        example = LabelledTicket(
            ticket_id="INC000045",
            created_at=datetime.now(UTC),
            short_description="Printer spool error in warehouse",
            description="Spool request stuck in SP01.",
            requester="history@example.com",
            type=TicketType.INCIDENT,
            category=Category.OUTPUT_PRINTING,
            impact=Level.LOW,
            urgency=Level.LOW,
            priority=Priority.P4,
            assignment_group=AssignmentGroup.BASIS,
            tier="L1",
            resolution_notes="Spool request reprocessed; output delivered.",
        )
        schema = _schema_for(frozenset({"type", "category", "assignment_group"}))
        parsed = schema(
            type=TicketType.INCIDENT,
            category=Category.OUTPUT_PRINTING,
            assignment_group=AssignmentGroup.BASIS,
            impact=Level.LOW,
            urgency=Level.LOW,
            rationale="Matches resolved spool example.",
        )
        chat = FakeChat(parsed)
        verdict = classify_ticket_gate3(make_ticket(), chat, examples=[example])

        user_msg = chat.last_messages[1]["content"]
        assert "Printer spool error in warehouse" in user_msg
        assert "priority=P4" in user_msg
        assert "Spool request reprocessed" in user_msg
        assert verdict is not None

    def test_no_examples_omits_block(self):
        schema = _schema_for(frozenset({"type", "category", "assignment_group"}))
        chat = FakeChat(
            schema(
                type=TicketType.INCIDENT,
                category=Category.SHORT_DUMP,
                assignment_group=AssignmentGroup.ABAP,
                impact=Level.LOW,
                urgency=Level.LOW,
                rationale="No neighbours.",
            )
        )
        classify_ticket_gate3(make_ticket(), chat)
        assert "reference tickets" not in chat.last_messages[1]["content"]


def test_gate3_pins_temperature_zero():
    """Determinism: default temperature caused a 7pt eval swing; pin it."""
    schema = _schema_for(frozenset({"type", "category", "assignment_group"}))
    chat = FakeChat(
        schema(
            type=TicketType.INCIDENT,
            category=Category.SHORT_DUMP,
            assignment_group=AssignmentGroup.ABAP,
            impact=Level.LOW,
            urgency=Level.LOW,
            rationale="Deterministic call.",
        )
    )
    classify_ticket_gate3(make_ticket(), chat)
    assert chat.last_kwargs["temperature"] == 0.0
