"""Gate 3 — LLM classification for whatever the earlier gates left open.

The cascade is an assembly line: Gate 3 receives the labels Gate 1/2 already
locked and is asked ONLY for the rest, plus impact and urgency. Priority is
never asked of the model — it is derived mechanically via derive_priority(),
keeping severity narrative and final priority mathematically consistent.

The structured-output schema is built per missing-label subset, so the model
physically cannot return a field that is already locked — anchoring is
enforced by the schema, not just the prompt.
"""

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, create_model

from triage.llm import ChatModel, cost_usd
from triage.models import (
    LABEL_FIELDS,
    AssignmentGroup,
    Category,
    Level,
    TicketInput,
    TicketType,
    TriageVerdict,
    derive_priority,
)

_OPTIONAL_FIELD_TYPES: dict[str, type] = {
    "type": TicketType,
    "category": Category,
    "assignment_group": AssignmentGroup,
}

_SYSTEM_PROMPT = """You are an SAP AMS ticket triage specialist applying ITIL conventions.

TICKET TYPE:
- incident: something is broken, malfunctioning, or an unplanned interruption —
  errors, dumps, failed jobs, "cannot do X".
- service_request: a standard user request — access, password reset, master
  data change, transport import, routine question. Nothing is broken. A ticket
  phrased as a request ("please reset", "kindly provide", "request access") is
  a service_request even when the user sounds urgent.
- problem: a root-cause investigation of recurring or linked incidents.
- change: a planned modification to configuration or code.

IMPACT — who is affected (independent of urgency):
- high: a whole site, plant, department, or all users are affected; a critical
  business process is halted.
- medium: a team or several users are affected; other groups work normally or
  a workaround exists.
- low: exactly one user is affected; nobody else.

URGENCY — how time-critical (independent of impact):
- high: work has stopped right now, production is standing still, or a hard
  deadline (today, closing, go-live) is blocked until this is fixed.
- medium: work is slowed, or a deadline later this week is at risk; people can
  still work.
- low: no deadline at risk, no time pressure; it can wait.

Judge impact and urgency INDEPENDENTLY from the narrative — a company-wide
issue with no deadline is high impact + low urgency, not automatically urgent.

Fields already verified by earlier gates are stated in the ticket context —
treat them as ground truth and do not reconsider them."""

_SCHEMA_CACHE: dict[frozenset[str], type[BaseModel]] = {}


def _schema_for(missing: frozenset[str]) -> type[BaseModel]:
    """Structured-output schema asking only for `missing` labels + severity."""
    if missing in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[missing]
    fields: dict[str, object] = {
        name: (field_type, ...)
        for name, field_type in _OPTIONAL_FIELD_TYPES.items()
        if name in missing
    }
    fields["impact"] = (Level, ...)
    fields["urgency"] = (Level, ...)
    fields["rationale"] = (str, ...)
    schema = create_model(  # type: ignore[call-overload]
        f"Gate3Output_{len(missing)}",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    _SCHEMA_CACHE[missing] = schema
    return schema


def _locked_labels(verdict: TriageVerdict | None) -> dict[str, object]:
    if verdict is None:
        return {}
    return {f: getattr(verdict, f) for f in LABEL_FIELDS if getattr(verdict, f) is not None}


def classify_ticket_gate3(
    ticket: TicketInput,
    chat: ChatModel,
    locked: TriageVerdict | None = None,
    *,
    system_prompt: str = _SYSTEM_PROMPT,
) -> TriageVerdict | None:
    """Fill the labels `locked` left open; return a merged verdict.

    `locked` is the partial verdict from Gate 1/2 (or None for a full
    fallback). Locked values are echoed back as context and win any
    disagreement — the schema doesn't even offer those fields.
    """
    start = time.perf_counter()
    known = _locked_labels(locked)
    missing = frozenset(f for f in LABEL_FIELDS if f not in known and f != "priority")
    # Priority is always Gate 3's job (derived, not asked).
    if not missing and locked is not None and locked.priority is not None:
        return locked

    context = "; ".join(f"{f}={v!r}" for f, v in sorted(known.items())) if known else "none"
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Ticket:\n"
                f"Short: {ticket.short_description}\n\n"
                f"{ticket.description}\n\n"
                f"Already verified (ground truth, do not change): {context}.\n"
                f"Provide only the fields requested by the schema."
            ),
        },
    ]

    schema = _schema_for(missing)
    parsed = chat.complete_structured(messages, schema)

    labels: dict[str, object] = dict(known)
    label_source: dict[str, Literal["rules", "similarity", "model"]] = dict(
        locked.label_source if locked else {}
    )
    for field in missing:
        labels[field] = getattr(parsed, field)
        label_source[field] = "model"
    labels["priority"] = derive_priority(parsed.impact, parsed.urgency)
    label_source["priority"] = "model"

    usage = chat.last_usage
    call_cost = cost_usd(usage.prompt_tokens, usage.completion_tokens) if usage else 0.0
    carried_cost = locked.cost_usd if locked else 0.0
    confidence = min(locked.confidence, 0.8) if locked else 0.8

    return TriageVerdict(
        ticket_id=ticket.ticket_id,
        type=labels.get("type"),
        category=labels.get("category"),
        priority=labels.get("priority"),
        assignment_group=labels.get("assignment_group"),
        impact=parsed.impact,
        urgency=parsed.urgency,
        confidence=confidence,
        decided_by="cascade" if locked else "model",
        label_source=label_source,
        rationale=parsed.rationale[:300],
        similar_ticket_ids=list(locked.similar_ticket_ids) if locked else [],
        cost_usd=carried_cost + call_cost,
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )
