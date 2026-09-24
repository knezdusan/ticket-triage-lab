"""Run the full cascade over data/eval.jsonl and report the numbers that matter.

Usage: uv run python scripts/eval_cascade.py

NEEDS AZURE CREDENTIALS (embeddings + chat). ~60 tickets, ~$0.10 spend.
The history index at data/history_index.npz is reused if valid.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from triage.cascade import triage
from triage.llm import EmbeddingModel, get_chat_model, get_embedding_model
from triage.models import LABEL_FIELDS, LabelledTicket, TriageVerdict
from triage.similarity import SimilarityIndex


class CachedEmbedder:
    """One embedding per text, remembered."""

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


def report(results: list[tuple[LabelledTicket, TriageVerdict]]) -> None:
    n = len(results)
    print(f"\n{'=' * 74}")
    print(f"CASCADE EVAL — {n} tickets")
    print(f"{'=' * 74}")

    # 1. Per-field accuracy
    hits = Counter()
    for truth, verdict in results:
        hits.update(f for f in LABEL_FIELDS if getattr(verdict, f) == getattr(truth, f))
    all4 = sum(all(getattr(v, f) == getattr(t, f) for f in LABEL_FIELDS) for t, v in results)
    print("\nper-field accuracy:")
    for f in LABEL_FIELDS:
        print(f"  {f:<18} {hits[f]}/{n} ({hits[f] / n:.1%})")
    print(f"  {'ALL FOUR':<18} {all4}/{n} ({all4 / n:.1%})")

    # 1b. Gate-3 severity inputs — which axis is failing priority?
    for axis in ("impact", "urgency"):
        scored = [(t, v) for t, v in results if getattr(v, axis) is not None]
        if scored:
            correct = sum(getattr(v, axis) == getattr(t, axis) for t, v in scored)
            print(
                f"  {axis:<18} {correct}/{len(scored)} ({correct / len(scored):.1%}) [gate-3 only]"
            )

    # 2. Provenance breakdown
    source_counts = Counter()
    for _, verdict in results:
        source_counts.update(verdict.label_source.values())
    decided_by = Counter(v.decided_by for _, v in results)
    print("\nlabels filled, by gate:")
    for source in ("rules", "similarity", "model"):
        print(f"  {source:<12} {source_counts[source]:>4} fields")
    print("verdicts, by decider:", dict(decided_by))

    # 3. Accuracy by source — does each gate earn its keep?
    field_hits: dict[str, Counter] = defaultdict(Counter)
    field_totals: dict[str, Counter] = defaultdict(Counter)
    for truth, verdict in results:
        for f in LABEL_FIELDS:
            source = verdict.label_source.get(f)
            if source is None:
                continue
            field_totals[f][source] += 1
            if getattr(verdict, f) == getattr(truth, f):
                field_hits[f][source] += 1
    print("\naccuracy by field x source:")
    print(f"  {'field':<18} {'source':<11} {'acc':>8} {'n':>4}")
    for f in LABEL_FIELDS:
        for source in ("rules", "similarity", "model"):
            total = field_totals[f][source]
            if total:
                acc = field_hits[f][source] / total
                print(f"  {f:<18} {source:<11} {acc:>7.1%} {total:>4}")

    # 4. Cost & latency
    total_cost = sum(v.cost_usd for _, v in results)
    latencies = sorted(v.latency_ms for _, v in results)
    print("\ncost & latency:")
    print(f"  total cost:    ${total_cost:.4f}")
    print(f"  mean latency:  {sum(latencies) / n:.0f} ms")
    print(f"  p50/p95:       {latencies[n // 2]:.0f} / {latencies[int(n * 0.95) - 1]:.0f} ms")

    # 5. Misses worth reading
    misses = [
        (t.ticket_id, f, getattr(t, f), getattr(v, f))
        for t, v in results
        for f in LABEL_FIELDS
        if getattr(v, f) != getattr(t, f)
    ]
    print(f"\nmislabels: {len(misses)}")
    for ticket_id, f, truth, pred in misses[:15]:
        print(f"  {ticket_id} {f}: truth={truth} pred={pred}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="eval_cascade")
    parser.add_argument("--eval", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--history", type=Path, default=Path("data/history.jsonl"))
    parser.add_argument("--cache", type=Path, default=Path("data/history_index.npz"))
    args = parser.parse_args()

    history = _load(args.history)
    eval_tickets = _load(args.eval)
    embedder = CachedEmbedder(get_embedding_model())
    chat = get_chat_model()

    index = SimilarityIndex.load_or_build(args.cache, history, embedder)
    print(f"index: {index.matrix.shape} | eval tickets: {len(eval_tickets)}")

    results = []
    for i, ticket in enumerate(eval_tickets, 1):
        verdict = triage(ticket, index, embedder, chat)
        results.append((ticket, verdict))
        print(
            f"[{i:>3}/{len(eval_tickets)}] {ticket.ticket_id} "
            f"{verdict.decided_by:<10} {verdict.priority} "
            f"{verdict.category} -> {verdict.assignment_group} "
            f"(${verdict.cost_usd:.5f}, {verdict.latency_ms:.0f}ms)"
        )

    report(results)


if __name__ == "__main__":
    main()
