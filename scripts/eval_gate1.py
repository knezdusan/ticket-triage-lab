"""Measure Gate 1 coverage and label precision over labelled history.

Usage: uv run python scripts/eval_gate1.py data/history.jsonl

Coverage  = fraction of tickets where apply_rules returns a verdict.
Precision = among decided tickets, fraction where each predicted label
            matches the ground-truth label (all four must match for a
            fully correct verdict).
No model calls.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from triage.models import LabelledTicket
from triage.rules import apply_rules

_FIELDS = ("type", "category", "priority", "assignment_group")


def _load(path: Path) -> list[LabelledTicket]:
    return [
        LabelledTicket.model_validate(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def evaluate(path: Path) -> None:
    tickets = _load(path)
    decided: list[tuple[LabelledTicket, object]] = []
    for ticket in tickets:
        verdict = apply_rules(ticket)
        if verdict is not None:
            decided.append((ticket, verdict))

    coverage = len(decided) / len(tickets) if tickets else 0.0
    print(f"audited {len(tickets)} tickets from {path}")
    print(f"coverage: {len(decided)}/{len(tickets)} ({coverage:.1%})\n")

    if not decided:
        print("no verdicts returned")
        return

    field_hits = Counter()
    full_hits = 0
    misses: list[tuple[str, dict[str, object]]] = []
    for ticket, verdict in decided:
        diffs = {
            f: {"truth": getattr(ticket, f), "pred": getattr(verdict, f)}
            for f in _FIELDS
            if getattr(verdict, f) != getattr(ticket, f)
        }
        if diffs:
            misses.append((ticket.ticket_id, diffs))
        else:
            full_hits += 1
        for f in _FIELDS:
            field_hits[f] += f not in diffs

    n = len(decided)
    print(f"full-verdict precision: {full_hits}/{n} ({full_hits / n:.1%})")
    for f in _FIELDS:
        print(f"  {f:<18} {field_hits[f]}/{n} ({field_hits[f] / n:.1%})")

    print(f"\nmislabelled verdicts: {len(misses)}")
    for ticket_id, diffs in misses[:10]:
        print(f"  {ticket_id}: {diffs}")

    truth_cats = Counter(t.category for t, _ in decided)
    print("\ndecided tickets by true category:")
    for cat, count in truth_cats.most_common():
        print(f"  {cat}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="eval_gate1")
    parser.add_argument("path", type=Path, help="labelled JSONL file")
    args = parser.parse_args()
    if not args.path.exists():
        sys.exit(f"no such file: {args.path}")
    evaluate(args.path)


if __name__ == "__main__":
    main()
