"""Tests for src/triage/datagen.py. No network — ChatModel is faked."""

from collections import Counter
from pathlib import Path

import pytest

from triage.datagen import (
    _prompt,
    build_plan,
    build_stress_plan,
    generate_records,
    load_existing_ids,
    short_description_valid,
    split_plan,
    text_has_leak,
)
from triage.llm import TransientLLMError
from triage.models import (
    Category,
    GeneratedText,
    LabelledTicket,
    Level,
    Priority,
    TicketType,
    derive_priority,
)
from triage.taxonomy import (
    IMPACT_SITUATIONS,
    MASTER_DATA_GROUP_BY_MODULE,
    TYPE_MIX,
    URGENCY_SITUATIONS,
)

SEED = 42


class FakeChat:
    """Drop-in for ChatModel: returns canned GeneratedText, counts calls."""

    def __init__(self, responses: list[GeneratedText] | None = None) -> None:
        self.calls = 0
        self._responses = responses or []

    def complete_structured(self, messages: list[dict], schema: type, **kwargs: object):
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return GeneratedText(
            short_description=f"System issue number {self.calls}",
            description="Users report the transaction hangs on save since this morning.",
        )


def _planned(n: int = 300):
    return build_plan(n, SEED)


# ---------------------------------------------------------------- plan


def test_plan_is_deterministic():
    assert build_plan(300, SEED) == build_plan(300, SEED)
    assert build_plan(300, SEED) != build_plan(300, SEED + 1)


def test_plan_respects_n():
    assert len(build_plan(50, SEED)) == 50


def test_type_distribution_within_tolerance():
    counts = Counter(t.type for t in _planned())
    for ticket_type, share in TYPE_MIX.items():
        expected = 300 * share
        assert abs(counts[ticket_type] - expected) <= max(10, expected * 0.35), ticket_type


def test_every_priority_matches_matrix():
    for t in _planned():
        assert derive_priority(t.impact, t.urgency) is t.priority


def test_master_data_group_follows_module():
    master_data = [t for t in _planned() if t.category is Category.MASTER_DATA]
    assert master_data, "expected master_data tickets in the plan"
    for t in master_data:
        assert t.assignment_group == MASTER_DATA_GROUP_BY_MODULE[t.sap_module]


def test_ticket_id_prefix_matches_type():
    prefix = {
        TicketType.INCIDENT: "INC",
        TicketType.SERVICE_REQUEST: "REQ",
        TicketType.PROBLEM: "PRB",
        TicketType.CHANGE: "CHG",
    }
    for t in _planned():
        assert t.ticket_id.startswith(prefix[t.type])


def test_duplicates_share_labels():
    plan = _planned()
    clustered = [t for t in plan if t.cluster_id]
    assert clustered, "expected some near-duplicate tickets"
    for dup in clustered:
        members = [t for t in plan if t.cluster_id == dup.cluster_id]
        assert len(members) >= 2
        for other in members:
            if other is dup:
                continue
            assert (dup.type, dup.category, dup.priority, dup.assignment_group, dup.tier) == (
                other.type,
                other.category,
                other.priority,
                other.assignment_group,
                other.tier,
            )
            assert dup.requester != other.requester or dup.created_at != other.created_at


# ---------------------------------------------------------------- leaks


@pytest.mark.parametrize(
    ("short", "desc"),
    [
        ("System down", "This is a P1 for us"),
        ("Urgent", "Please raise the priority now"),
        ("Login fails", "Business impact is huge"),
        ("VA01 dump", "what is the urgency here?"),
    ],
)
def test_leak_check_flags_labels(short, desc):
    assert text_has_leak(short, desc) is True


@pytest.mark.parametrize(
    ("short", "desc"),
    [
        ("VA01 short dump on save", "Every save in VA01 ends in a dump since Monday."),
        ("ME21N access", "Please grant display access for purchasing."),
        ("P10 report slow", "Report P10 takes ten minutes."),  # P10 is not P1-P4
    ],
)
def test_leak_check_allows_clean_text(short, desc):
    assert text_has_leak(short, desc) is False


# ---------------------------------------------------------------- split


def test_clusters_never_split():
    plan = _planned()
    history, evaluation = split_plan(plan, SEED)
    eval_ids = {t.ticket_id for t in evaluation}
    for t in plan:
        if not t.cluster_id:
            continue
        members = [m for m in plan if m.cluster_id == t.cluster_id]
        in_eval = {m.ticket_id in eval_ids for m in members}
        assert len(in_eval) == 1, f"cluster {t.cluster_id} split across sets"


def test_split_is_stratified_and_covers_all():
    plan = _planned()
    history, evaluation = split_plan(plan, SEED)
    assert len(history) + len(evaluation) == len(plan)
    eval_types = Counter(t.type for t in evaluation)
    assert eval_types[TicketType.INCIDENT] > 0
    assert eval_types[TicketType.SERVICE_REQUEST] > 0


# ------------------------------------------------------------- generation


def test_generate_writes_valid_records(tmp_path: Path):
    plan = build_plan(10, SEED)
    out = tmp_path / "tickets.jsonl"
    records = generate_records(plan, FakeChat(), out, progress=lambda m: None)
    assert len(records) == 10
    for line in out.read_text().splitlines():
        LabelledTicket.model_validate_json(line)  # raises if malformed


def test_resume_skips_existing_ids(tmp_path: Path):
    plan = build_plan(10, SEED)
    out = tmp_path / "tickets.jsonl"
    chat = FakeChat()
    first = generate_records(plan, chat, out, progress=lambda m: None)
    calls_after_first = chat.calls

    second = generate_records(
        plan, chat, out, existing_ids=load_existing_ids(out), progress=lambda m: None
    )
    assert second == []
    assert chat.calls == calls_after_first
    assert len(out.read_text().splitlines()) == len(first)


def test_leaking_text_is_retried_then_skipped(tmp_path: Path):
    leaky = GeneratedText(short_description="P1 outage", description="please raise the priority")
    chat = FakeChat(responses=[leaky, leaky, leaky])
    records = generate_records(
        build_plan(1, SEED), chat, tmp_path / "t.jsonl", progress=lambda m: None
    )
    assert records == []
    assert chat.calls == 3


def test_transient_error_retries_then_aborts(tmp_path: Path):
    class FlakyChat(FakeChat):
        def complete_structured(self, messages, schema, **kwargs):
            self.calls += 1
            raise TransientLLMError("rate limited")

    chat = FlakyChat()
    with pytest.raises(TransientLLMError):
        generate_records(build_plan(1, SEED), chat, tmp_path / "t.jsonl", progress=lambda m: None)
    assert chat.calls == 3  # retry(times=3)


def test_load_existing_ids_roundtrip(tmp_path: Path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"ticket_id": "INC000001"}\n\n{"ticket_id": "REQ000002"}\n')
    assert load_existing_ids(path) == {"INC000001", "REQ000002"}
    assert load_existing_ids(tmp_path / "missing.jsonl") == set()


# ------------------------------------------------------------- signal text


def test_situation_phrasings_are_leak_free():
    for phrasings in (*IMPACT_SITUATIONS.values(), *URGENCY_SITUATIONS.values()):
        for phrase in phrasings:
            assert text_has_leak(phrase, "") is False, phrase


def test_prompt_carries_situation_not_labels():
    for ticket in _planned(40):
        content = _prompt(ticket)[1]["content"]
        assert ticket.impact_text in content
        assert ticket.urgency_text in content
        # Labels never reach the model: no level names, no priority value.
        assert ticket.priority.value not in content
        assert ticket.impact.value not in content.lower().split()
        assert ticket.urgency.value not in content.lower().split()


def test_prompt_forbids_verbatim_situation_reuse():
    content = _prompt(_planned()[0])[1]["content"]
    assert "requester's own words" in content
    assert "verbatim" in content


def test_impact_guards_only_at_matching_levels():
    for ticket in _planned():
        content = _prompt(ticket)[1]["content"]
        assert ("Exactly one person is affected" in content) == (ticket.impact is Level.LOW)
        assert ("other groups are working normally" in content) == (ticket.impact is Level.MEDIUM)


def test_impact_and_urgency_guards_combine():
    both = [t for t in _planned() if t.impact is Level.LOW and t.urgency is Level.LOW]
    assert both  # the seeded plan must actually contain the combination
    for ticket in both:
        content = _prompt(ticket)[1]["content"]
        assert "must not mention any deadline" in content
        assert "Exactly one person is affected" in content


# ------------------------------------------------------------- length check


@pytest.mark.parametrize(
    "raw", ["ab", "  ", "x" * 121, "ok" + " " * 200, "VA01 dump\nsecond line", "text\rmore"]
)
def test_short_description_valid_rejects(raw):
    assert short_description_valid(raw) is False


@pytest.mark.parametrize("raw", ["abc", " VA01 dump ", "x" * 120])
def test_short_description_valid_accepts(raw):
    assert short_description_valid(raw) is True


def test_long_short_description_retries(tmp_path: Path):
    too_long = GeneratedText(short_description="x" * 200, description="details")
    ok = GeneratedText(short_description="VA01 dump on save", description="details")
    chat = FakeChat(responses=[too_long, ok])
    records = generate_records(
        build_plan(1, SEED), chat, tmp_path / "t.jsonl", progress=lambda m: None
    )
    assert len(records) == 1
    assert records[0].short_description == "VA01 dump on save"
    assert chat.calls == 2


# ------------------------------------------------------------- stress set


def _stress(n: int = 40):
    main_plan = _planned()
    counters = Counter(t.ticket_id[:3] for t in main_plan)
    return main_plan, build_stress_plan(n, SEED + 1000, counters)


def test_stress_plan_exactly_half_p1():
    for n in (40, 41):
        _, stress = _stress(n)
        p1 = [t for t in stress if t.priority is Priority.P1]
        p2 = [t for t in stress if t.priority is Priority.P2]
        assert len(p1) == (n + 1) // 2
        assert len(p2) == n // 2


def test_stress_p1_is_always_incident_high_high():
    _, stress = _stress()
    for t in stress:
        if t.priority is Priority.P1:
            assert t.type is TicketType.INCIDENT
            assert t.impact is Level.HIGH
            assert t.urgency is Level.HIGH


def test_stress_plan_is_deterministic_and_disjoint():
    main_a, stress_a = _stress()
    _, stress_b = _stress()
    assert stress_a == stress_b
    assert build_stress_plan(10, SEED + 1001, Counter()) != build_stress_plan(
        10, SEED + 1000, Counter()
    )
    main_ids = {t.ticket_id for t in main_a}
    assert not {t.ticket_id for t in stress_a} & main_ids
    history, evaluation = split_plan(main_a, SEED)
    split_ids = {t.ticket_id for t in history + evaluation}
    assert not {t.ticket_id for t in stress_a} & split_ids
    assert all(t.cluster_id is None for t in stress_a)


def test_stress_ids_continue_sequence():
    main_plan, stress = _stress(10)
    counters = Counter(t.ticket_id[:3] for t in main_plan)
    for t in stress:
        seq = int(t.ticket_id[3:])
        base = counters[t.ticket_id[:3]]
        assert base < seq <= base + 10
