"""Tests for src/triage/models.py (Day 3). Do not edit — implement models.py by hand."""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from triage.models import (
    AssignmentGroup,
    Category,
    GeneratedText,
    LabelledTicket,
    Level,
    Priority,
    TicketInput,
    TicketType,
    Tier,
    TriageVerdict,
    derive_priority,
)

BASE = {
    "ticket_id": "INC000123",
    "created_at": datetime(2026, 9, 22, 9, 0),
    "short_description": "  VA01 short dump on save  ",
    "description": "Since this morning every save in VA01 ends in a dump.",
    "requester": "m.petrovic",
}

LABELS = {
    "type": "incident",
    "category": "short_dump",
    "impact": "high",
    "urgency": "medium",
    "priority": "P2",
    "assignment_group": "ABAP",
    "tier": "L2",
}

VERDICT = {
    "ticket_id": "INC000123",
    "type": "incident",
    "category": "short_dump",
    "priority": "P2",
    "assignment_group": "ABAP",
    "confidence": 0.82,
    "decided_by": "similarity",
    "label_source": {
        "type": "similarity",
        "category": "similarity",
        "priority": "similarity",
        "assignment_group": "similarity",
    },
    "rationale": "Three nearest resolved tickets agree.",
    "similar_ticket_ids": ["INC000045"],
}


# ---------------------------------------------------------------- enums


def test_enum_values():
    assert [t.value for t in TicketType] == ["incident", "service_request", "problem", "change"]
    assert [p.value for p in Priority] == ["P1", "P2", "P3", "P4"]
    assert [lv.value for lv in Level] == ["high", "medium", "low"]
    assert [t.value for t in Tier] == ["L1", "L2", "L3"]
    assert AssignmentGroup.SERVICE_DESK == "Service Desk"
    assert len(AssignmentGroup) == 8
    assert len(Category) == 12
    assert Category.SHORT_DUMP == "short_dump"


@pytest.mark.parametrize(
    ("impact", "urgency", "expected"),
    [
        (Level.HIGH, Level.HIGH, Priority.P1),
        (Level.HIGH, Level.MEDIUM, Priority.P2),
        (Level.LOW, Level.HIGH, Priority.P3),
        (Level.LOW, Level.LOW, Priority.P4),
    ],
)
def test_derive_priority(impact, urgency, expected):
    assert derive_priority(impact, urgency) is expected


def test_derive_priority_invalid():
    with pytest.raises(ValueError):
        derive_priority("extreme", "high")


# ---------------------------------------------------------------- TicketInput


def test_input_valid_and_stripped():
    t = TicketInput(**BASE)
    assert t.short_description == "VA01 short dump on save"
    assert t.sap_module is None


def test_input_rejects_unknown_field():
    with pytest.raises(ValidationError):
        TicketInput(**BASE, colour="red")


@pytest.mark.parametrize("bad_id", ["INC123", "XYZ000123", "inc000123", "INC0001234"])
def test_input_rejects_bad_ticket_id(bad_id):
    with pytest.raises(ValidationError):
        TicketInput(**{**BASE, "ticket_id": bad_id})


@pytest.mark.parametrize("text", ["  a ", "x" * 121])
def test_input_rejects_bad_short_description(text):
    with pytest.raises(ValidationError):
        TicketInput(**{**BASE, "short_description": text})


def test_input_is_frozen():
    t = TicketInput(**BASE)
    with pytest.raises(ValidationError):
        t.requester = "someone.else"


def test_input_from_json():
    raw = json.dumps({**BASE, "created_at": "2026-09-22T09:00:00"})
    t = TicketInput.model_validate_json(raw)
    assert t.created_at == datetime(2026, 9, 22, 9, 0)


# ---------------------------------------------------------------- LabelledTicket


def test_labelled_valid():
    t = LabelledTicket(**BASE, **LABELS)
    assert t.priority is Priority.P2
    assert t.type is TicketType.INCIDENT
    assert t.tier is Tier.L2
    assert t.resolution_notes is None


def test_labelled_rejects_inconsistent_priority():
    with pytest.raises(ValidationError):
        LabelledTicket(**BASE, **{**LABELS, "priority": "P1"})


def test_labelled_rejects_prefix_mismatch():
    with pytest.raises(ValidationError):
        LabelledTicket(**{**BASE, "ticket_id": "REQ000123"}, **LABELS)


def test_labelled_rejects_unknown_category():
    with pytest.raises(ValidationError):
        LabelledTicket(**BASE, **{**LABELS, "category": "weather"})


def test_labelled_round_trip():
    t = LabelledTicket(**BASE, **LABELS)
    dumped = t.model_dump(mode="json")
    assert dumped["priority"] == "P2"
    assert dumped["created_at"] == "2026-09-22T09:00:00"
    assert LabelledTicket.model_validate_json(t.model_dump_json()) == t


# ---------------------------------------------------------------- GeneratedText


def test_generated_text_schema_is_model_friendly():
    schema = GeneratedText.model_json_schema()
    assert set(schema["required"]) == {"short_description", "description"}
    assert schema.get("additionalProperties") is False


def test_generated_text_rejects_unknown_field():
    with pytest.raises(ValidationError):
        GeneratedText(short_description="VA01 dump", description="Details", priority="P1")


# ---------------------------------------------------------------- TriageVerdict


def test_verdict_valid_with_defaults():
    v = TriageVerdict(**VERDICT)
    assert v.priority is Priority.P2
    assert v.cost_usd == 0.0
    assert v.latency_ms == 0.0


@pytest.mark.parametrize("confidence", [-0.1, 1.3])
def test_verdict_rejects_confidence_out_of_range(confidence):
    with pytest.raises(ValidationError):
        TriageVerdict(**{**VERDICT, "confidence": confidence})


def test_verdict_rejects_unknown_decider():
    with pytest.raises(ValidationError):
        TriageVerdict(**{**VERDICT, "decided_by": "guess"})


def test_verdict_rejects_long_rationale():
    with pytest.raises(ValidationError):
        TriageVerdict(**{**VERDICT, "rationale": "x" * 301})


def test_verdict_abstain_valid():
    v = TriageVerdict(
        ticket_id="INC000123", confidence=0.2, decided_by="abstain", rationale="Ambiguous."
    )
    assert v.type is None
    assert v.priority is None
    assert v.similar_ticket_ids == []


def test_verdict_abstain_with_labels_rejected():
    with pytest.raises(ValidationError):
        TriageVerdict(**{**VERDICT, "decided_by": "abstain"})


def test_verdict_decision_without_labels_rejected():
    with pytest.raises(ValidationError):
        TriageVerdict(
            **{
                **VERDICT,
                "type": None,
                "category": None,
                "priority": None,
                "assignment_group": None,
            }
        )


def test_verdict_partial_labels_accepted():
    """Gates may fill a subset of labels; at least one is required."""
    v = TriageVerdict(
        **{
            **VERDICT,
            "priority": None,
            "type": None,
            "label_source": {
                "category": "similarity",
                "assignment_group": "similarity",
            },
        }
    )
    assert v.category is not None
    assert v.priority is None


def test_verdict_label_source_must_match_populated_labels():
    """Provenance is an invariant: sources name exactly the labels set."""
    with pytest.raises(ValidationError):
        TriageVerdict(**{**VERDICT, "label_source": {"category": "similarity"}})
    with pytest.raises(ValidationError):
        TriageVerdict(
            **{
                **VERDICT,
                "label_source": {**VERDICT["label_source"], "extra": "model"},
            }
        )


def test_note_lax_mode_coerces_strings():
    """A lesson, not a requirement: default (lax) mode turns "0.8" into 0.8. Zod would reject."""
    assert TriageVerdict(**{**VERDICT, "confidence": "0.8"}).confidence == 0.8
