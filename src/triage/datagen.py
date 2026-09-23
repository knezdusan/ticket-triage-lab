"""Synthetic ticket generator. Run: uv run python -m triage.datagen

Code chooses every label; the model writes only the ticket text. Resumable:
records append to data/tickets.jsonl and IDs already present are skipped.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from triage.llm import ChatModel, TransientLLMError, get_chat_model
from triage.models import (
    AssignmentGroup,
    Category,
    GeneratedText,
    LabelledTicket,
    Level,
    Priority,
    TicketType,
    Tier,
    derive_priority,
)
from triage.taxonomy import (
    ASSIGNMENT_GROUPS_BY_CATEGORY,
    CATEGORIES_BY_TYPE,
    DATE_RANGE,
    DUPLICATE_RATE,
    HOUR_WEIGHTS,
    IMPACT_SITUATIONS,
    LOW_IMPACT_GUARD,
    LOW_URGENCY_GUARD,
    MASTER_DATA_GROUP_BY_MODULE,
    MEDIUM_IMPACT_GUARD,
    PREFIX_BY_TYPE,
    PRIORITY_CELLS,
    PRIORITY_MIX_BY_TYPE,
    PROBLEM_TIER,
    REQUESTERS,
    RESOLUTION_NOTES_BY_CATEGORY,
    SAP_MODULES_BY_CATEGORY,
    STYLE_MIX,
    TIER_BY_GROUP,
    TYPE_MIX,
    URGENCY_SITUATIONS,
    WEEKEND_PROB,
    WRITING_STYLES,
)
from triage.utils import retry

MAX_ATTEMPTS = 3
DEFAULT_SEED = 42
EVAL_SHARE = 0.20


# ---------------------------------------------------------------- plan


@dataclass(frozen=True)
class PlannedTicket:
    """One planned ticket: every label chosen in code before any model call."""

    ticket_id: str
    created_at: datetime
    requester: str
    type: TicketType
    category: Category
    impact: Level
    urgency: Level
    priority: Priority
    assignment_group: AssignmentGroup
    tier: Tier
    sap_module: str | None
    style: str
    impact_text: str
    urgency_text: str
    cluster_id: str | None


def _weighted_choice[K](rng: random.Random, weights: dict[K, float]) -> K:
    return rng.choices(list(weights), weights=list(weights.values()), k=1)[0]


def _sample_created_at(rng: random.Random) -> datetime:
    start, end = DATE_RANGE
    days = (end - start).days
    while True:
        day = start + timedelta(days=rng.randint(0, days))
        if day.weekday() < 5 or rng.random() < WEEKEND_PROB:
            break
    hour = _weighted_choice(rng, HOUR_WEIGHTS)
    return datetime.combine(day, datetime.min.time()).replace(
        hour=hour, minute=rng.randint(0, 59), second=rng.randint(0, 59)
    )


def _sample_labels(
    rng: random.Random,
) -> tuple[TicketType, Category, Level, Level, Priority]:
    ticket_type = _weighted_choice(rng, TYPE_MIX)
    priority = _weighted_choice(rng, PRIORITY_MIX_BY_TYPE[ticket_type])
    impact, urgency = rng.choice(PRIORITY_CELLS[priority])
    if derive_priority(impact, urgency) is not priority:
        raise ValueError(f"plan/matrix mismatch: {impact} x {urgency} != {priority}")
    category = _weighted_choice(rng, CATEGORIES_BY_TYPE[ticket_type])
    return ticket_type, category, impact, urgency, priority


def _counters_from_plan(plan: list[PlannedTicket]) -> Counter[str]:
    return Counter(PREFIX_BY_TYPE[t.type] for t in plan)


def _build_ticket(
    rng: random.Random,
    counters: Counter[str],
    *,
    ticket_type: TicketType,
    category: Category,
    impact: Level,
    urgency: Level,
    priority: Priority,
    cluster_id: str | None,
    src: PlannedTicket | None = None,
) -> PlannedTicket:
    """Assemble one PlannedTicket. With `src` set (near-duplicate), group,
    tier and module are cloned from it and the requester is resampled until
    it differs."""
    if src is not None:
        sap_module, assignment_group, tier = src.sap_module, src.assignment_group, src.tier
    else:
        sap_module = rng.choice(SAP_MODULES_BY_CATEGORY[category])
        if category is Category.MASTER_DATA:
            # group follows the module, not an independent draw
            assignment_group = MASTER_DATA_GROUP_BY_MODULE[sap_module]
        else:
            assignment_group = _weighted_choice(rng, ASSIGNMENT_GROUPS_BY_CATEGORY[category])
        tier = (
            PROBLEM_TIER if ticket_type is TicketType.PROBLEM else TIER_BY_GROUP[assignment_group]
        )

    prefix = PREFIX_BY_TYPE[ticket_type]
    counters[prefix] += 1
    requester = rng.choice(REQUESTERS)
    while src is not None and requester == src.requester:
        requester = rng.choice(REQUESTERS)  # a duplicate must come from someone else
    return PlannedTicket(
        ticket_id=f"{prefix}{counters[prefix]:06d}",
        created_at=_sample_created_at(rng),
        requester=requester,
        type=ticket_type,
        category=category,
        impact=impact,
        urgency=urgency,
        priority=priority,
        assignment_group=assignment_group,
        tier=tier,
        sap_module=sap_module,
        style=_weighted_choice(rng, STYLE_MIX),
        impact_text=rng.choice(IMPACT_SITUATIONS[impact]),
        urgency_text=rng.choice(URGENCY_SITUATIONS[urgency]),
        cluster_id=cluster_id,
    )


def build_plan(n: int = 300, seed: int = DEFAULT_SEED) -> list[PlannedTicket]:
    """Deterministic sampling plan: all labels, styles and cluster links."""
    rng = random.Random(seed)
    counters: Counter[str] = Counter()
    cluster_of: dict[str, str] = {}  # ticket_id -> cluster_id
    plan: list[PlannedTicket] = []

    for _ in range(n):
        src: PlannedTicket | None = None
        if plan and rng.random() < DUPLICATE_RATE:
            # Near-duplicate: same labels as an earlier ticket, new text,
            # requester and timestamp.
            src = rng.choice(plan)
            cluster_id = cluster_of.get(src.ticket_id)
            if cluster_id is None:
                cluster_id = f"CLU{len(set(cluster_of.values())) + 1:03d}"
                cluster_of[src.ticket_id] = cluster_id
                plan[plan.index(src)] = replace(src, cluster_id=cluster_id)
        else:
            ticket_type, category, impact, urgency, priority = _sample_labels(rng)
            cluster_id = None
        if src is not None:
            ticket_type, category = src.type, src.category
            impact, urgency, priority = src.impact, src.urgency, src.priority
        ticket = _build_ticket(
            rng,
            counters,
            ticket_type=ticket_type,
            category=category,
            impact=impact,
            urgency=urgency,
            priority=priority,
            cluster_id=cluster_id,
            src=src,
        )
        if cluster_id is not None:
            cluster_of[ticket.ticket_id] = cluster_id
        plan.append(ticket)
    return plan


def build_stress_plan(n: int, seed: int, counters: Counter[str]) -> list[PlannedTicket]:
    """P1/P2-only eval stress set: ceil(n/2) P1, the rest P2, fixed by
    construction rather than sampled. Every P1 is an incident (the only type
    with P1 mass, at high x high). P2 types are drawn weighted by TYPE_MIX
    among the types that allow it. IDs continue the main plan's sequence.
    No duplicates, no clusters.
    """
    rng = random.Random(seed)
    counters = Counter(counters)
    priorities = [Priority.P1] * ((n + 1) // 2) + [Priority.P2] * (n // 2)
    rng.shuffle(priorities)

    plan: list[PlannedTicket] = []
    for priority in priorities:
        types = {t: w for t, w in TYPE_MIX.items() if PRIORITY_MIX_BY_TYPE[t].get(priority, 0) > 0}
        ticket_type = _weighted_choice(rng, types)
        impact, urgency = rng.choice(PRIORITY_CELLS[priority])
        if derive_priority(impact, urgency) is not priority:
            raise ValueError(f"plan/matrix mismatch: {impact} x {urgency} != {priority}")
        category = _weighted_choice(rng, CATEGORIES_BY_TYPE[ticket_type])
        plan.append(
            _build_ticket(
                rng,
                counters,
                ticket_type=ticket_type,
                category=category,
                impact=impact,
                urgency=urgency,
                priority=priority,
                cluster_id=None,
            )
        )
    return plan


# ---------------------------------------------------------------- text


_LEAK_TOKEN_RE = re.compile(r"\bP[1-4]\b")
_LEAK_WORD_RE = re.compile(r"\b(?:priority|urgency|impact)\b", re.IGNORECASE)


def text_has_leak(short_description: str, description: str) -> bool:
    """True if the generated text states priority, impact or urgency."""
    text = f"{short_description}\n{description}"
    return bool(_LEAK_TOKEN_RE.search(text) or _LEAK_WORD_RE.search(text))


def short_description_valid(raw: str) -> bool:
    """True if the stripped short_description is a single line of 3-120 chars."""
    return "\n" not in raw and "\r" not in raw and 3 <= len(raw.strip()) <= 120


def _prompt(ticket: PlannedTicket) -> list[dict]:
    module = ticket.sap_module or "not specified"
    guards = []
    if ticket.urgency is Level.LOW:
        guards.append(LOW_URGENCY_GUARD)
    if ticket.impact is Level.LOW:
        guards.append(LOW_IMPACT_GUARD)
    elif ticket.impact is Level.MEDIUM:
        guards.append(MEDIUM_IMPACT_GUARD)
    guard = f" {' '.join(guards)}" if guards else ""
    return [
        {
            "role": "system",
            "content": (
                "You write realistic SAP support ticket text for a synthetic "
                "training dataset. Output only the requested fields."
            ),
        },
        {
            "role": "user",
            "content": (
                "Write a realistic ticket for an SAP AMS service desk.\n"
                f"Kind: {ticket.type.value}\n"
                f"About: {ticket.category.value}\n"
                f"SAP module: {module}\n"
                f"Writing style: {WRITING_STYLES[ticket.style]}\n"
                f"Situation to convey naturally in the text: {ticket.impact_text}; "
                f"{ticket.urgency_text}.\n"
                "Convey that situation in the requester's own words through "
                "concrete facts about who is affected and what is blocked — "
                "do NOT reuse the phrasing above verbatim or near-verbatim.\n"
                "Rules: plausible SAP vocabulary (tcodes, modules, clients) is "
                "welcome. The short_description is a single line, at most 120 "
                "characters. NEVER state or imply a ticket ranking or severity "
                "code — no P-numbers — and never use the words 'priority', "
                "'urgency' or 'impact' anywhere in the output."
                f"{guard}"
            ),
        },
    ]


# --------------------------------------------------------------- generate


def load_existing_ids(path: Path) -> set[str]:
    """Ticket IDs already written to a JSONL file (resume support)."""
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            ids.add(json.loads(line)["ticket_id"])
    return ids


def generate_records(
    plan: list[PlannedTicket],
    chat: ChatModel,
    out_path: Path,
    existing_ids: set[str] | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    progress: Callable[[str], None] = print,
) -> list[LabelledTicket]:
    """Generate text for each planned ticket, validate, append to JSONL.

    Returns the records written this run. Skipped tickets (leak or
    validation failure after max_attempts) are reported via `progress`.
    A TransientLLMError that survives its retries aborts the run; re-run
    to resume — written IDs are skipped.
    """
    existing_ids = existing_ids or set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[LabelledTicket] = []

    @retry(times=3, exceptions=(TransientLLMError,))
    def _text(messages: list[dict]) -> GeneratedText:
        return chat.complete_structured(messages, GeneratedText)

    written = skipped = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for ticket in plan:
            if ticket.ticket_id in existing_ids:
                continue
            record: LabelledTicket | None = None
            for attempt in range(1, max_attempts + 1):
                text = _text(_prompt(ticket))
                if text_has_leak(text.short_description, text.description):
                    progress(f"{ticket.ticket_id}: leak in text (attempt {attempt})")
                    continue
                if not short_description_valid(text.short_description):
                    progress(
                        f"{ticket.ticket_id}: short_description length "
                        f"{len(text.short_description.strip())} (attempt {attempt})"
                    )
                    continue
                try:
                    record = LabelledTicket(
                        ticket_id=ticket.ticket_id,
                        created_at=ticket.created_at,
                        short_description=text.short_description,
                        description=text.description,
                        requester=ticket.requester,
                        sap_module=ticket.sap_module,
                        type=ticket.type,
                        category=ticket.category,
                        impact=ticket.impact,
                        urgency=ticket.urgency,
                        priority=ticket.priority,
                        assignment_group=ticket.assignment_group,
                        tier=ticket.tier,
                    )
                    break
                except ValidationError as exc:
                    progress(f"{ticket.ticket_id}: validation failed (attempt {attempt}): {exc}")
            if record is None:
                skipped += 1
                progress(f"{ticket.ticket_id}: skipped after {max_attempts} attempts")
                continue
            fh.write(record.model_dump_json() + "\n")
            fh.flush()
            records.append(record)
            written += 1
            if written % 25 == 0:
                progress(f"written {written} records")
    progress(f"done: {written} written, {skipped} skipped")
    return records


# ---------------------------------------------------------------- split


def split_plan(
    plan: list[PlannedTicket],
    seed: int = DEFAULT_SEED,
    eval_share: float = EVAL_SHARE,
) -> tuple[list[PlannedTicket], list[PlannedTicket]]:
    """Stratified history/eval split. A cluster never crosses the boundary."""
    rng = random.Random(seed)
    groups: dict[str, list[PlannedTicket]] = defaultdict(list)
    for ticket in plan:
        groups[ticket.cluster_id or ticket.ticket_id].append(ticket)

    by_type: dict[TicketType, list[str]] = defaultdict(list)
    for key, members in groups.items():
        by_type[members[0].type].append(key)

    eval_keys: set[str] = set()
    for keys in by_type.values():
        keys.sort()
        rng.shuffle(keys)
        eval_keys.update(keys[: max(1, round(len(keys) * eval_share))])

    history = [t for k, ts in groups.items() if k not in eval_keys for t in ts]
    evaluation = [t for k, ts in groups.items() if k in eval_keys for t in ts]
    return history, evaluation


def _write_jsonl(path: Path, records: list[LabelledTicket]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")


def _with_resolution_note(ticket: LabelledTicket, rng: random.Random) -> LabelledTicket:
    note = rng.choice(RESOLUTION_NOTES_BY_CATEGORY[ticket.category])
    return ticket.model_copy(update={"resolution_notes": note})


# --------------------------------------------------------------- report


def _counts(items: list, key: Callable[[object], str]) -> dict[str, int]:
    return dict(sorted(Counter(key(t) for t in items).items()))


def report_markdown(
    plan: list[PlannedTicket],
    history: list[PlannedTicket],
    evaluation: list[PlannedTicket],
    *,
    dry_run: bool,
) -> str:
    clusters = {t.cluster_id for t in plan if t.cluster_id}
    title = "planned (dry run — no tickets generated)" if dry_run else "generated"
    lines = [
        "# Synthetic ticket data report",
        "",
        f"Scope: {title}. Planned tickets: {len(plan)}.",
        "",
        "## Counts by type",
    ]
    for key, n in _counts(plan, lambda t: t.type.value).items():
        lines.append(f"- {key}: {n}")
    for label, attr in (
        ("priority", "priority"),
        ("category", "category"),
        ("assignment_group", "assignment_group"),
        ("tier", "tier"),
    ):
        lines += ["", f"## Counts by {label}"]
        for key, n in _counts(plan, lambda t, a=attr: getattr(t, a).value).items():
            lines.append(f"- {key}: {n}")
    lines += [
        "",
        "## Split",
        f"- history.jsonl: {len(history)} tickets",
        f"- eval.jsonl: {len(evaluation)} tickets",
        "",
        "## Near-duplicate clusters",
        f"- {len(clusters)} clusters covering {sum(1 for t in plan if t.cluster_id)} tickets",
    ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="triage.datagen")
    parser.add_argument("--n", type=int, default=300, help="number of tickets to plan")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, call no model")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--stress",
        type=int,
        default=0,
        help="extra P1/P2-only tickets written to eval_p1.jsonl",
    )
    args = parser.parse_args(argv)

    plan = build_plan(args.n, args.seed)
    stress_plan = (
        build_stress_plan(args.stress, args.seed + 1000, _counters_from_plan(plan))
        if args.stress
        else []
    )

    if args.dry_run:
        history, evaluation = split_plan(plan, args.seed)
        print(report_markdown(plan, history, evaluation, dry_run=True))
        if stress_plan:
            n_p1 = sum(t.priority is Priority.P1 for t in stress_plan)
            print(
                f"Stress set: {len(stress_plan)} extra tickets — exactly {n_p1} P1 "
                f"(all incidents) + {len(stress_plan) - n_p1} P2 -> eval_p1.jsonl; "
                "excluded from all counts above."
            )
        return

    tickets_path = args.data_dir / "tickets.jsonl"
    existing = load_existing_ids(tickets_path)
    if existing:
        print(f"resuming: {len(existing)} ticket IDs already in {tickets_path}")

    chat = get_chat_model()
    generate_records(plan, chat, tickets_path, existing)

    # Split covers every record in tickets.jsonl — including ones written by
    # earlier (resumed) runs. Parse each line once.
    records: dict[str, LabelledTicket] = {}
    for line in tickets_path.read_text().splitlines():
        if line.strip():
            parsed = LabelledTicket.model_validate_json(line)
            records[parsed.ticket_id] = parsed

    history_plan, eval_plan = split_plan(plan, args.seed)
    rng = random.Random(args.seed)
    history = [
        _with_resolution_note(records[t.ticket_id], rng)
        for t in history_plan
        if t.ticket_id in records
    ]
    evaluation = [records[t.ticket_id] for t in eval_plan if t.ticket_id in records]

    _write_jsonl(args.data_dir / "history.jsonl", history)
    _write_jsonl(args.data_dir / "eval.jsonl", evaluation)

    clusters: dict[str, list[str]] = defaultdict(list)
    for t in plan:
        if t.cluster_id:
            clusters[t.cluster_id].append(t.ticket_id)
    (args.data_dir / "clusters.json").write_text(
        json.dumps(clusters, indent=2) + "\n", encoding="utf-8"
    )

    stress_written = 0
    if stress_plan:
        stress_path = args.data_dir / "eval_p1.jsonl"
        generate_records(stress_plan, chat, stress_path, load_existing_ids(stress_path))
        stress_written = len(load_existing_ids(stress_path))

    report = report_markdown(plan, history_plan, eval_plan, dry_run=False)
    if stress_plan:
        n_p1 = sum(t.priority is Priority.P1 for t in stress_plan)
        report += (
            "\n## Priority stress set\n"
            f"- eval_p1.jsonl: {stress_written} tickets — exactly {n_p1} P1 and\n"
            f"  {stress_written - n_p1} P2. Every P1 is an incident by construction\n"
            "  (P1 requires impact=high x urgency=high, and only incidents carry\n"
            "  P1 mass). Deliberately non-representative: exists only to measure\n"
            "  priority confusion. Excluded from every distribution count above.\n"
        )
    (args.data_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(
        f"wrote {len(records)} tickets total, "
        f"split {len(history)}/{len(evaluation)} into {args.data_dir}"
    )


if __name__ == "__main__":
    sys.exit(main())
