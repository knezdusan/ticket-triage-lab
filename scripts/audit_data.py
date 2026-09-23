"""Audit generated ticket text against the taxonomy phrasings and guards.

Usage: uv run python scripts/audit_data.py data/tickets.jsonl

Reads JSONL, prints counts and up to 5 example ticket_ids per finding.
No model calls.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from triage.taxonomy import IMPACT_SITUATIONS, URGENCY_SITUATIONS

_PHRASINGS = [p for ps in IMPACT_SITUATIONS.values() for p in ps] + [
    p for ps in URGENCY_SITUATIONS.values() for p in ps
]

# Words that must not appear in a low-urgency ticket's text.
_DEADLINE_RE = re.compile(r"\b(?:urgent|immediately|today|deadline|asap|critical)\b", re.IGNORECASE)

# Wider-than-one-person scope that must not appear in a low-impact ticket.
# Deliberately NOT a bare-word list: "team"/"department" alone fire on
# salutations ("Hello Team") and singular references ("a team member"),
# so every alternative encodes the claim itself — a quantifier, a
# possessive + verb, or an explicit wide-scope phrase.
_LOW_IMPACT_RE = re.compile(
    r"\b(?:"
    # "the whole department", "entire plant", ...
    r"(?:whole|entire)\s+(?:team|department|site|company|plant|office)\b"
    # "our team is unable", "the department cannot", "my company has ..."
    r"|(?:my|our|the|this)\s+(?:team|department|company|office)\s+"
    r"(?:is|are|was|were|has|have|can't|cant|cannot)\b"
    # "all users", "all of us", "all the employees"
    r"|all\s+(?:(?:of|the)\s+){0,2}(?:users|employees|staff|colleagues|us)\b"
    r"|everyone\b"
    # "company-wide", "site wide"
    r"|(?:company|site|plant|department)[- ]wide\b"
    # "across the company"
    r"|across\s+the\s+(?:company|site|plant|department|organisation|organization)\b"
    # "several colleagues", "many users", "multiple departments"
    r"|(?:several|multiple|many|numerous)\s+"
    r"(?:users|colleagues|employees|people|departments|teams)\b"
    # "production is down", "the system was offline", "SAP is down"
    r"|(?:production|system|sap)\s+(?:is|was|went)\s+(?:down|offline|unavailable)\b"
    r")",
    re.IGNORECASE,
)

# Company-/site-wide scope that must not appear in a medium-impact ticket.
_MEDIUM_IMPACT_RE = re.compile(
    r"\b(?:company-wide|site-wide|whole company|entire (?:site|company|plant|department)|"
    r"everyone|all users|production is down|system is down)\b",
    re.IGNORECASE,
)

_EXAMPLE_LIMIT = 5


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _text(ticket: dict[str, Any]) -> str:
    return f"{ticket['short_description']}\n{ticket['description']}"


def _examples(ids: list[str]) -> str:
    return ", ".join(ids[:_EXAMPLE_LIMIT]) if ids else "-"


def audit(path: Path) -> None:
    tickets = _load(path)
    print(f"audited {len(tickets)} tickets from {path}\n")

    verbatim = [
        t["ticket_id"] for t in tickets if any(p.lower() in _text(t).lower() for p in _PHRASINGS)
    ]
    print(f"verbatim situation phrasings: {len(verbatim)}")
    print(f"  e.g. {_examples(verbatim)}")

    low = [t for t in tickets if t.get("urgency") == "low"]
    pressured = [t["ticket_id"] for t in low if _DEADLINE_RE.search(_text(t))]
    print(f"\nlow-urgency tickets with deadline/urgency words: {len(pressured)} of {len(low)}")
    print(f"  e.g. {_examples(pressured)}")

    low_impact = [t for t in tickets if t.get("impact") == "low"]
    overbroad = [t["ticket_id"] for t in low_impact if _LOW_IMPACT_RE.search(_text(t))]
    print(f"\nlow-impact tickets claiming wider scope: {len(overbroad)} of {len(low_impact)}")
    print(f"  e.g. {_examples(overbroad)}")

    medium_impact = [t for t in tickets if t.get("impact") == "medium"]
    sitewide = [t["ticket_id"] for t in medium_impact if _MEDIUM_IMPACT_RE.search(_text(t))]
    print(
        "\nmedium-impact tickets claiming site/company-wide scope: "
        f"{len(sitewide)} of {len(medium_impact)}"
    )
    print(f"  e.g. {_examples(sitewide)}")

    lengths = [len(t["short_description"].strip()) for t in tickets]
    buckets = Counter(
        "<=40" if n <= 40 else "41-80" if n <= 80 else "81-120" if n <= 120 else ">120"
        for n in lengths
    )
    print(
        f"\nshort_description length: min={min(lengths)} max={max(lengths)} "
        f"mean={sum(lengths) / len(lengths):.0f}"
    )
    for bucket in ("<=40", "41-80", "81-120", ">120"):
        print(f"  {bucket}: {buckets.get(bucket, 0)}")

    dup_descriptions = {
        text: n for text, n in Counter(t["short_description"] for t in tickets).items() if n > 1
    }
    dup_ids = [t["ticket_id"] for t in tickets if t["short_description"] in dup_descriptions]
    print(
        f"\nduplicate short_descriptions: {len(dup_ids)} tickets "
        f"share {len(dup_descriptions)} texts"
    )
    print(f"  e.g. {_examples(dup_ids)}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="audit_data")
    parser.add_argument("path", type=Path, help="JSONL file to audit")
    args = parser.parse_args()
    if not args.path.exists():
        sys.exit(f"no such file: {args.path}")
    audit(args.path)


if __name__ == "__main__":
    main()
