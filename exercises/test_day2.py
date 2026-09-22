"""Tests for Day 2. Do not edit — make them pass by editing day2.py."""

import asyncio
import dataclasses
import inspect
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from day2 import (  # noqa: E402
    Priority,
    Ticket,
    TicketLoadError,
    any_p1,
    chunks,
    classify_all,
    count_by_priority,
    derive_priority,
    extract_tcodes,
    group_by_assignment,
    index_by_id,
    load_tickets,
    retry,
    short_descriptions,
    sla_due,
    sort_by_urgency,
    timer,
    unique_modules,
)

TICKETS = [
    {"ticket_id": "INC001", "short_description": "VA01 short dump on save",
     "priority": "P2", "assignment_group": "SD", "sap_module": "SD"},
    {"ticket_id": "REQ002", "short_description": "Access to ME21N",
     "priority": "P4", "assignment_group": "Security", "sap_module": "MM"},
    {"ticket_id": "INC003", "short_description": "Invoice posting stopped",
     "priority": "P1", "assignment_group": "FI", "sap_module": "FI"},
    {"ticket_id": "INC004", "short_description": "Report slow",
     "priority": "P3", "assignment_group": "Basis", "sap_module": None},
    {"ticket_id": "REQ005", "short_description": "Password reset",
     "priority": "P4", "assignment_group": "Service Desk", "sap_module": None},
    {"ticket_id": "INC006", "short_description": "Pricing wrong on order",
     "priority": "P3", "assignment_group": "SD", "sap_module": "SD"},
]


# ---------------------------------------------------------------- b1

def test_b1_count_by_priority():
    assert count_by_priority(TICKETS) == {"P1": 1, "P2": 1, "P3": 2, "P4": 2}


def test_b1_count_by_priority_empty():
    assert count_by_priority([]) == {}


# ---------------------------------------------------------------- b2

def test_b2_short_descriptions():
    assert short_descriptions(TICKETS, "P4") == ["Access to ME21N", "Password reset"]


def test_b2_index_by_id():
    index = index_by_id(TICKETS)
    assert set(index) == {t["ticket_id"] for t in TICKETS}
    assert index["INC003"]["assignment_group"] == "FI"


def test_b2_unique_modules():
    assert unique_modules(TICKETS) == {"SD", "MM", "FI"}


def test_b2_any_p1():
    assert any_p1(TICKETS) is True
    assert any_p1([t for t in TICKETS if t["priority"] != "P1"]) is False


# ---------------------------------------------------------------- b3

def test_b3_group_by_assignment():
    assert group_by_assignment(TICKETS) == {
        "SD": ["INC001", "INC006"],
        "Security": ["REQ002"],
        "FI": ["INC003"],
        "Basis": ["INC004"],
        "Service Desk": ["REQ005"],
    }


def test_b3_extract_tcodes():
    text = "Error in VA01 and ME21N, see ST22. Retried VA01. P1 from SAP."
    assert extract_tcodes(text) == ["VA01", "ME21N", "ST22"]


def test_b3_extract_tcodes_ignores_lowercase():
    assert extract_tcodes("tried va01 then SU01") == ["SU01"]


def test_b3_sla_due():
    created = datetime(2026, 9, 21, 10, 0)
    assert sla_due(created, "P1") == datetime(2026, 9, 21, 14, 0)
    assert sla_due(created, "P4") == datetime(2026, 9, 24, 10, 0)


def test_b3_sla_due_unknown_priority():
    with pytest.raises(ValueError):
        sla_due(datetime(2026, 9, 21), "P9")


# ---------------------------------------------------------------- b4

def test_b4_priority_is_a_string():
    assert Priority.P1 == "P1"
    assert [p.value for p in Priority] == ["P1", "P2", "P3", "P4"]


@pytest.mark.parametrize(
    ("impact", "urgency", "expected"),
    [
        ("high", "high", "P1"),
        ("high", "medium", "P2"),
        ("medium", "high", "P2"),
        ("high", "low", "P3"),
        ("medium", "medium", "P3"),
        ("low", "high", "P3"),
        ("medium", "low", "P4"),
        ("low", "medium", "P4"),
        ("low", "low", "P4"),
    ],
)
def test_b4_derive_priority(impact, urgency, expected):
    assert derive_priority(impact, urgency) == Priority(expected)


def test_b4_derive_priority_invalid():
    with pytest.raises(ValueError):
        derive_priority("extreme", "high")


def test_b4_ticket_is_frozen_dataclass():
    t = Ticket("INC001", "VA01 short dump", Priority.P2)
    assert dataclasses.is_dataclass(t)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.priority = Priority.P1  # type: ignore[misc]


def test_b4_ticket_is_urgent():
    assert Ticket("A", "x", Priority.P1).is_urgent is True
    assert Ticket("B", "x", Priority.P2).is_urgent is True
    assert Ticket("C", "x", Priority.P3).is_urgent is False


def test_b4_ticket_str():
    assert str(Ticket("INC001", "VA01 short dump", Priority.P2)) == "[P2] INC001: VA01 short dump"


def test_b4_sort_by_urgency():
    tickets = [
        Ticket("A", "x", Priority.P3),
        Ticket("B", "y", Priority.P1),
        Ticket("C", "z", Priority.P3),
        Ticket("D", "w", Priority.P2),
    ]
    assert [t.ticket_id for t in sort_by_urgency(tickets)] == ["B", "D", "A", "C"]


# ---------------------------------------------------------------- b5

def test_b5_load_tickets(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"ticket_id": "A"}\n\n{"ticket_id": "B"}\n')
    assert load_tickets(path) == [{"ticket_id": "A"}, {"ticket_id": "B"}]


def test_b5_load_tickets_bad_line(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"ticket_id": "A"}\n{broken\n')
    with pytest.raises(TicketLoadError, match="2"):
        load_tickets(path)


def test_b5_load_tickets_missing_file(tmp_path):
    with pytest.raises(TicketLoadError) as exc_info:
        load_tickets(tmp_path / "nope.jsonl")
    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_b5_timer():
    with timer() as t:
        sum(range(10_000))
    assert "ms" in t
    assert t["ms"] >= 0


# ---------------------------------------------------------------- b6

def test_b6_retry_succeeds_after_failures():
    calls = {"n": 0}

    @retry(times=3)
    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("temporary")
        return a + b

    assert flaky(2, 3) == 5
    assert calls["n"] == 3


def test_b6_retry_gives_up():
    calls = {"n": 0}

    @retry(times=3)
    def always_fails():
        calls["n"] += 1
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        always_fails()
    assert calls["n"] == 3


def test_b6_retry_only_listed_exceptions():
    calls = {"n": 0}

    @retry(times=3, exceptions=(ConnectionError,))
    def bad_input():
        calls["n"] += 1
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        bad_input()
    assert calls["n"] == 1


def test_b6_retry_preserves_name():
    @retry(times=2)
    def my_func():
        return 1

    assert my_func.__name__ == "my_func"


# ---------------------------------------------------------------- b7

def test_b7_classify_all_order_and_cap():
    state = {"now": 0, "peak": 0}

    async def fake_classify(text: str) -> str:
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0.01)
        state["now"] -= 1
        return text.upper()

    texts = [f"t{i}" for i in range(10)]
    result = asyncio.run(classify_all(texts, fake_classify, max_concurrency=3))
    assert result == [t.upper() for t in texts]
    assert state["peak"] == 3


# ---------------------------------------------------------------- b8

def test_b8_chunks():
    assert list(chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(chunks([], 3)) == []


def test_b8_chunks_is_generator():
    assert inspect.isgenerator(chunks([1, 2], 1))


def test_b8_chunks_invalid_size():
    with pytest.raises(ValueError):
        list(chunks([1], 0))
