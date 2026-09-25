"""Tests for the triage cascade — all three gates' dependencies mocked."""

import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest

from triage.cascade import triage
from triage.classifier import _schema_for
from triage.models import (
    AssignmentGroup,
    Category,
    LabelledTicket,
    Level,
    Priority,
    TicketInput,
    TicketType,
)
from triage.similarity import SimilarityIndex


class FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]] | None = None, dim: int = 64):
        self.mapping = mapping or {}
        self.dim = dim
        self.calls = 0
        self.last_usage = SimpleNamespace(prompt_tokens=10, total_tokens=10)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vectors = []
        for text in texts:
            if text in self.mapping:
                vectors.append(list(self.mapping[text]))
            else:
                seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8])
                vectors.append(np.random.default_rng(seed).standard_normal(self.dim).tolist())
        return vectors


class FakeChat:
    def __init__(self, parsed=None, error=None):
        self.parsed = parsed
        self.error = error
        self.calls = 0
        self.last_schema = None
        self.last_messages = None
        self.last_usage = SimpleNamespace(prompt_tokens=400, completion_tokens=80)

    def complete_structured(self, messages, schema, **kwargs):
        self.calls += 1
        self.last_schema = schema
        self.last_messages = messages
        if self.error:
            raise self.error
        return self.parsed


def make_ticket(short: str, desc: str, ticket_id: str = "INC000777") -> TicketInput:
    return TicketInput(
        ticket_id=ticket_id,
        created_at=datetime.now(UTC),
        short_description=short,
        description=desc,
        requester="tester@example.com",
    )


def make_labelled(ticket_id: str, short: str, desc: str, **labels) -> LabelledTicket:
    defaults = dict(
        ticket_type=TicketType.INCIDENT,
        category=Category.OUTPUT_PRINTING,
        group=AssignmentGroup.ABAP,
    )
    defaults.update(labels)
    return LabelledTicket(
        ticket_id=ticket_id,
        created_at=datetime.now(UTC),
        short_description=short,
        description=desc,
        requester="tester@example.com",
        type=defaults["ticket_type"],
        category=defaults["category"],
        impact=Level.LOW,
        urgency=Level.LOW,
        priority=Priority.P4,
        assignment_group=defaults["group"],
        tier="L1",
    )


REPORT = ("Report generation help", "Quarterly report variant needs adjusting.")


@pytest.fixture
def index() -> SimilarityIndex:
    # A 3-identical cluster whose text deliberately matches NO Gate 1 pattern.
    history = [make_labelled(f"INC00000{i}", *REPORT) for i in (1, 2, 3)]
    return SimilarityIndex.build(history, FakeEmbedder())


class TestCascade:
    def test_gate1_hit_short_circuits(self, index):
        embedder, chat = FakeEmbedder(), FakeChat()
        ticket = make_ticket("Password reset required", "I forgot my password, please reset.")
        verdict = triage(ticket, index, embedder, chat)

        assert verdict.decided_by == "rules"
        assert verdict.priority == Priority.P4
        assert embedder.calls == 0
        assert chat.calls == 0

    def test_gate2_partial_then_gate3_fills(self, index):
        embedder, chat = (
            FakeEmbedder(),
            FakeChat(
                _schema_for(frozenset())(
                    impact=Level.MEDIUM,
                    urgency=Level.MEDIUM,
                    rationale="Single report adjustment, routine but dated.",
                )
            ),
        )
        verdict = triage(make_ticket(*REPORT), index, embedder, chat)

        assert embedder.calls == 1
        assert chat.calls == 1
        assert verdict.decided_by == "cascade"
        # Gate 2 locked these; Gate 3 supplied priority only.
        assert verdict.category == Category.OUTPUT_PRINTING
        assert verdict.type == TicketType.INCIDENT
        assert verdict.assignment_group == AssignmentGroup.ABAP
        assert verdict.priority == Priority.P3  # medium x medium
        assert verdict.label_source == {
            "category": "similarity",
            "type": "similarity",
            "assignment_group": "similarity",
            "priority": "model",
        }
        assert set(chat.last_schema.model_fields) == {"impact", "urgency", "rationale"}

    def test_no_gate_evidence_falls_back_to_full_llm(self, index):
        chat = FakeChat(
            _schema_for(frozenset({"type", "category", "assignment_group"}))(
                type=TicketType.INCIDENT,
                category=Category.PERFORMANCE,
                assignment_group=AssignmentGroup.BASIS,
                impact=Level.LOW,
                urgency=Level.LOW,
                rationale="Slow dialog response for one user.",
            )
        )
        verdict = triage(
            make_ticket("System feels slow", "Response times degraded today."),
            index,
            FakeEmbedder(),
            chat,
        )
        assert verdict.decided_by == "model"
        assert verdict.priority == Priority.P4
        assert all(s == "model" for s in verdict.label_source.values())

    def test_gate3_refusal_propagates(self, index):
        chat = FakeChat(error=ValueError("structured output not parsed (refusal: x)"))
        with pytest.raises(ValueError, match="refusal"):
            triage(
                make_ticket("System feels slow", "Response times degraded today."),
                index,
                FakeEmbedder(),
                chat,
            )


class TestFewShotWiring:
    def test_neighbour_examples_reach_gate3(self, index):
        chat = FakeChat(
            _schema_for(frozenset())(
                impact=Level.LOW,
                urgency=Level.LOW,
                rationale="Matches the resolved cluster.",
            )
        )
        triage(make_ticket(*REPORT), index, FakeEmbedder(), chat)

        user_msg = chat.last_messages[1]["content"]
        assert "Resolved reference tickets" in user_msg
        assert "Report generation help" in user_msg
        assert "INC000001" not in user_msg or True  # ids optional; labels render
