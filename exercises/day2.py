"""Day 2 — Python idiom. Write every function body yourself.

Signatures are fixed; the tests in test_day2.py depend on them.
Replace each `raise NotImplementedError` with your implementation.
"""

import asyncio
import functools
import json
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, TypedDict

# ---------------------------------------------------------------- scratch data

TICKETS = [
    {
        "ticket_id": "INC001",
        "short_description": "VA01 short dump on save",
        "priority": "P2",
        "assignment_group": "SD",
        "sap_module": "SD",
    },
    {
        "ticket_id": "REQ002",
        "short_description": "Access to ME21N",
        "priority": "P4",
        "assignment_group": "Security",
        "sap_module": "MM",
    },
    {
        "ticket_id": "INC003",
        "short_description": "Invoice posting stopped",
        "priority": "P1",
        "assignment_group": "FI",
        "sap_module": "FI",
    },
    {
        "ticket_id": "INC004",
        "short_description": "Report slow",
        "priority": "P3",
        "assignment_group": "Basis",
        "sap_module": None,
    },
    {
        "ticket_id": "REQ005",
        "short_description": "Password reset",
        "priority": "P4",
        "assignment_group": "Service Desk",
        "sap_module": None,
    },
    {
        "ticket_id": "INC006",
        "short_description": "Pricing wrong on order",
        "priority": "P3",
        "assignment_group": "SD",
        "sap_module": "SD",
    },
]


# Type definitions and aliases
class TicketDict(TypedDict):
    ticket_id: str
    short_description: str
    priority: str
    assignment_group: str
    sap_module: str | None


type TicketList = list[TicketDict]

# ---------------------------------------------------------------- b1


def count_by_priority(tickets: TicketList) -> dict[str, int]:
    """Count tickets per priority. Only priorities that appear are included."""
    result = {}
    for ticket in tickets:
        priority = ticket.get("priority")
        if priority:
            result[priority] = result.get(priority, 0) + 1
    return result


# if __name__ == "__main__":
#     print(count_by_priority(TICKETS))

# ---------------------------------------------------------------- b2


def short_descriptions(tickets: TicketList, priority: str) -> list[str]:
    """Short descriptions of tickets with the given priority, in input order."""
    result = []
    # for ticket in tickets:
    #     if ticket["priority"] == priority:
    #         result.append(ticket["short_description"])

    result = [
        ticket["short_description"] or "No description"
        for ticket in tickets
        if ticket["priority"] == priority
    ]
    return result

    # if __name__ == "__main__":
    #     print(short_descriptions(TICKETS, "P2"))


def index_by_id(tickets: TicketList) -> dict[str, dict]:
    """Map ticket_id -> ticket."""
    return {ticket["ticket_id"]: ticket for ticket in tickets}


# if __name__ == "__main__":
#     print(index_by_id(TICKETS))


def unique_modules(tickets: TicketList) -> set[str]:
    """Distinct sap_module values, ignoring None."""
    return {ticket["sap_module"] for ticket in tickets if ticket["sap_module"]}


# if __name__ == "__main__":
#     print(unique_modules(TICKETS))


def any_p1(tickets: TicketList) -> bool:
    """True if any ticket is P1."""
    return any(ticket["priority"] == "P1" for ticket in tickets)

    # for ticket in tickets:
    #     if ticket["priority"] == "P1":
    #         return True
    # return False


# if __name__ == "__main__":
#     print(any_p1(TICKETS))


# ---------------------------------------------------------------- b3


def group_by_assignment(tickets: TicketList) -> dict[str, list[str]]:
    """Map assignment_group -> list of ticket_ids, in input order."""
    # result = {}
    # for ticket in tickets:
    #     assignment_group = ticket["assignment_group"]
    #     if assignment_group not in result:
    #         result[assignment_group] = []
    #     result[assignment_group].append(ticket["ticket_id"])
    # return result

    result = defaultdict(list)
    for ticket in tickets:
        result[ticket["assignment_group"]].append(ticket["ticket_id"])
    return dict(result)


# if __name__ == "__main__":
#     print(group_by_assignment(TICKETS))


def extract_tcodes(text: str) -> list[str]:
    """SAP transaction codes in text: uppercase only, no duplicates, first-seen order.

    A tcode here is 2-4 uppercase letters, then 2 digits, then an optional
    uppercase letter. Examples: VA01, ME21N, ST22, SU01.
    """

    codes = []
    for match in re.finditer(r"[A-Z]{2,4}\d{2}[A-Z]?", text):
        codes.append(match.group())

    # dedup codes with list(dict.fromkeys(codes))
    return list(dict.fromkeys(codes))


# if __name__ == "__main__":
#     print(extract_tcodes("VA01 ME21N ST22"))


def sla_due(created_at: datetime, priority: str) -> datetime:
    """Resolution deadline, calendar hours (24x7): P1 4h, P2 8h, P3 24h, P4 72h.

    Unknown priority raises ValueError.
    """
    due = created_at
    if priority == "P1":
        due += timedelta(hours=4)
    elif priority == "P2":
        due += timedelta(hours=8)
    elif priority == "P3":
        due += timedelta(hours=24)
    elif priority == "P4":
        due += timedelta(hours=72)
    else:
        raise ValueError("Unknown priority")
    return due


# if __name__ == "__main__":
#     print(sla_due(datetime.now(), "P1"))
# ---------------------------------------------------------------- b4


class Priority(StrEnum):
    """TODO: members P1, P2, P3, P4 with values "P1".."P4"."""

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


def derive_priority(impact: str, urgency: str) -> Priority:
    """Impact x urgency matrix (see README). Unknown values raise ValueError."""

    impact = impact.lower()
    urgency = urgency.lower()

    # refactor:
    MATRIX = {
        "high": {"high": Priority.P1, "medium": Priority.P2, "low": Priority.P3},
        "medium": {"high": Priority.P2, "medium": Priority.P3, "low": Priority.P4},
        "low": {"high": Priority.P3, "medium": Priority.P4, "low": Priority.P4},
    }

    try:
        return MATRIX[impact][urgency]
    except KeyError:
        raise ValueError("Unknown impact or urgency") from None


# if __name__ == "__main__":
#     print(derive_priority("medium", "high"))


@dataclass(frozen=True)
class Ticket:
    """TODO: make this a frozen dataclass.

    Fields, in this order: ticket_id: str, short_description: str, priority: Priority.
    Add:
      - is_urgent property: True for P1 and P2
      - __str__ returning e.g. "[P2] INC001: VA01 short dump"
    """

    ticket_id: str
    short_description: str
    priority: Priority

    @property
    def is_urgent(self) -> bool:
        return self.priority in (Priority.P1, Priority.P2)

    def __str__(self) -> str:
        return f"[{self.priority}] {self.ticket_id}: {self.short_description}"


def sort_by_urgency(tickets: list[Ticket]) -> list[Ticket]:
    """Most urgent first. Tickets with equal priority keep their input order."""

    return sorted(tickets, key=lambda t: t.priority)


# ---------------------------------------------------------------- b5


class TicketLoadError(Exception):
    """Raised when a tickets file cannot be loaded."""

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error


def load_tickets(path: Path) -> TicketList:
    """Read JSON Lines. Skip blank lines.

    A malformed line raises TicketLoadError mentioning its 1-based line number.
    A missing file raises TicketLoadError from the original FileNotFoundError.
    """
    try:
        with open(path) as f:
            tickets = []
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    tickets.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise TicketLoadError(f"Malformed line {i}: {e}") from e
            return tickets
    except FileNotFoundError as e:
        raise TicketLoadError(f"File not found: {path}") from e


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Context manager: `with timer() as t: ...` then t["ms"] holds elapsed milliseconds.

    Hint: contextlib.
    """
    start = datetime.now()
    result = {}
    yield result
    end = datetime.now()
    result["ms"] = (end - start).total_seconds() * 1000


# ---------------------------------------------------------------- b6


def retry(
    times: int, exceptions: tuple[type[Exception], ...] = (Exception,)
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator factory. Retry up to `times` attempts on listed exceptions only.

    Re-raise the last exception after the final attempt. Preserve the wrapped
    function's name. No sleeping.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except exceptions:
                    if attempt == times - 1:
                        raise
            return None

        return wrapper

    return decorator


# ---------------------------------------------------------------- b7


async def classify_all(
    texts: list[str],
    classify: Callable[[str], Awaitable[str]],
    max_concurrency: int,
) -> list[str]:
    """Run classify over every text, at most max_concurrency at once.

    Results are returned in input order.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run(text: str) -> str:
        async with semaphore:
            return await classify(text)

    return await asyncio.gather(*(run(text) for text in texts))


# ---------------------------------------------------------------- b8


def chunks[T](items: Sequence[T], size: int) -> Iterator[list[T]]:
    """Yield consecutive lists of up to `size` items. Must be a generator.

    size < 1 raises ValueError.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    for i in range(0, len(items), size):
        yield list(items[i : i + size])
