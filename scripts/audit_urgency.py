"""Adjudicate Gate-3 urgency misses: dump verdicts, print each mismatch.

Usage: uv run python scripts/audit_urgency.py

NEEDS AZURE CREDENTIALS (~$0.10). Writes every verdict to
data/eval_verdicts.jsonl so further analysis is free. For each urgency
mismatch it prints the model's impact/urgency/priority, the truth, the
rationale, and the ticket text — enough to decide whether the model or the
synthetic label is at fault.
"""

import argparse
import json
import re
from pathlib import Path

from triage.cascade import triage
from triage.llm import EmbeddingModel, get_chat_model, get_embedding_model
from triage.models import LABEL_FIELDS, LabelledTicket, TriageVerdict
from triage.similarity import SimilarityIndex

# Phrasings the taxonomy reserves for high/medium urgency. A "low" label on
# text containing these is a label defect by construction (see taxonomy.py
# URGENCY_SITUATIONS / LOW_URGENCY_GUARD).
_HIGH = re.compile(
    r"\b(?:stopped|standing still|halted|production is down|system is down|"
    r"deadline today|asap|immediately|right now|cannot proceed|can't proceed|"
    r"work has stopped|blocking production)\b",
    re.IGNORECASE,
)
_MEDIUM = re.compile(
    r"\b(?:deadline|later this week|this week|at risk|slowing down|slowed)\b",
    re.IGNORECASE,
)


class CachedEmbedder:
    def __init__(self, embedder: EmbeddingModel) -> None:
        self._embedder = embedder
        self._cache: dict[str, list[float]] = {}
        self.last_usage = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            if text not in self._cache:
                self._cache[text] = self._embedder.embed([text])[0]
                self.last_usage = self._embedder.last_usage
            out.append(self._cache[text])
        return out


def _load(path: Path) -> list[LabelledTicket]:
    return [
        LabelledTicket.model_validate(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _text_implied(text: str) -> str:
    if _HIGH.search(text):
        return "high"
    if _MEDIUM.search(text):
        return "medium"
    return "low"


def _record(ticket: LabelledTicket, verdict: TriageVerdict) -> dict:
    return {
        "ticket_id": ticket.ticket_id,
        "short_description": ticket.short_description,
        "description": ticket.description,
        "truth": {f: str(getattr(ticket, f)) for f in LABEL_FIELDS}
        | {"impact": str(ticket.impact), "urgency": str(ticket.urgency)},
        "verdict": verdict.model_dump(mode="json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="audit_urgency")
    parser.add_argument("--eval", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--history", type=Path, default=Path("data/history.jsonl"))
    parser.add_argument("--cache", type=Path, default=Path("data/history_index.npz"))
    parser.add_argument("--out", type=Path, default=Path("data/eval_verdicts.jsonl"))
    args = parser.parse_args()

    history = _load(args.history)
    tickets = _load(args.eval)
    embedder = CachedEmbedder(get_embedding_model())
    chat = get_chat_model()
    index = SimilarityIndex.load_or_build(args.cache, history, embedder)

    records = []
    misses = []
    for i, ticket in enumerate(tickets, 1):
        verdict = triage(ticket, index, embedder, chat)
        records.append(_record(ticket, verdict))
        if verdict.urgency is not None and verdict.urgency != ticket.urgency:
            misses.append((ticket, verdict))
        print(f"[{i:>3}/{len(tickets)}] {ticket.ticket_id} done")

    args.out.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(f"\nwrote {len(records)} verdicts -> {args.out}")
    print(f"urgency misses: {len(misses)} of {len(records)}\n")

    for ticket, verdict in misses:
        text = f"{ticket.short_description} {ticket.description}"
        implied = _text_implied(text)
        fault = (
            "LABEL?"
            if ticket.urgency.value == "low" and implied != "low"
            else ("MODEL" if implied == ticket.urgency.value else "UNCLEAR")
        )
        print(f"{ticket.ticket_id}  [{fault}]")
        print(
            f"  truth: impact={ticket.impact.value} urgency={ticket.urgency.value} "
            f"priority={ticket.priority.value}"
        )
        print(
            f"  model: impact={verdict.impact and verdict.impact.value} "
            f"urgency={verdict.urgency and verdict.urgency.value} "
            f"priority={verdict.priority and verdict.priority.value} "
            f"(text-implied urgency={implied})"
        )
        print(f"  rationale: {verdict.rationale[:200]}")
        print(f"  text: {ticket.short_description}")
        print(f"        {ticket.description[:280]}")
        print()


if __name__ == "__main__":
    main()
