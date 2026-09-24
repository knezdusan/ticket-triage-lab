"""Calibrate Gate 2: sweep (k, threshold) over history.jsonl, leave-one-out.

Usage: uv run python scripts/eval_gate2.py

Builds (or loads) the embedding index at data/history_index.npz, then for each
ticket pretends it is new: embeds it once, excludes its own row from the index,
and evaluates every (k, threshold) config. Reports coverage and full-verdict
precision, split by whether a cluster twin (data/clusters.json) was among the
neighbours — that split separates memorization from generalization.

NEEDS AZURE CREDENTIALS on first run (embeds 240 tickets); afterwards the
cache serves everything offline.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from triage.llm import EmbeddingModel, get_embedding_model
from triage.models import LabelledTicket
from triage.similarity import SimilarityIndex, apply_similarity

_FIELDS = ("type", "category", "priority", "assignment_group")
_KS = (1, 3, 5)
_THRESHOLDS = (0.75, 0.78, 0.80, 0.82, 0.85)


class CachedEmbedder:
    """One embedding per text, remembered — the sweep re-queries each ticket
    at every config but must only pay for the vector once."""

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


def _twins(clusters_path: Path) -> dict[str, set[str]]:
    """ticket_id -> ids of its cluster-mates."""
    if not clusters_path.exists():
        return {}
    clusters = json.loads(clusters_path.read_text())
    twins: dict[str, set[str]] = {}
    for members in clusters.values():
        for member in members:
            twins.setdefault(member, set()).update(m for m in members if m != member)
    return twins


def _field_hits(ticket: LabelledTicket, verdict) -> Counter:
    return Counter(f for f in _FIELDS if getattr(verdict, f) == getattr(ticket, f))


def _correct(ticket: LabelledTicket, verdict) -> bool:
    return all(getattr(verdict, f) == getattr(ticket, f) for f in _FIELDS)


def sweep(index: SimilarityIndex, embedder, twins: dict[str, set[str]]) -> None:
    tickets = index.tickets
    print(f"tickets: {len(tickets)} | k in {_KS} | thresholds {_THRESHOLDS}\n")
    print(
        f"{'k':>2} {'theta':>6} {'cov':>6} {'type':>6} {'cat':>6} {'prio':>6} "
        f"{'group':>6} {'all4':>6} {'twin%':>6} {'nov-all4':>9}"
    )

    for k in _KS:
        for theta in _THRESHOLDS:
            decided = []
            for ticket in tickets:
                verdict = apply_similarity(
                    ticket,
                    index,
                    embedder,
                    k=k,
                    threshold=theta,
                    exclude_ids=frozenset({ticket.ticket_id}),
                )
                if verdict is not None:
                    decided.append((ticket, verdict))

            n = len(decided)
            cov = n / len(tickets)
            field_hits = Counter()
            for ticket, verdict in decided:
                field_hits.update(_field_hits(ticket, verdict))
            all4 = sum(_correct(t, v) for t, v in decided)

            with_twin = {
                id(t)
                for t, v in decided
                if twins.get(t.ticket_id, set()) & set(v.similar_ticket_ids)
            }
            novel = [(t, v) for t, v in decided if id(t) not in with_twin]
            nov_all4 = sum(_correct(t, v) for t, v in novel)
            twin_share = len(with_twin) / n if n else 0.0

            def pct(hits: int, denom: int) -> float:
                return hits / denom if denom else 0.0

            print(
                f"{k:>2} {theta:>6.2f} {cov:>6.1%} "
                f"{pct(field_hits['type'], n):>6.1%} {pct(field_hits['category'], n):>6.1%} "
                f"{pct(field_hits['priority'], n):>6.1%} "
                f"{pct(field_hits['assignment_group'], n):>6.1%} "
                f"{pct(all4, n):>6.1%} {twin_share:>6.0%} {pct(nov_all4, len(novel)):>9.1%}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="eval_gate2")
    parser.add_argument("--history", type=Path, default=Path("data/history.jsonl"))
    parser.add_argument("--cache", type=Path, default=Path("data/history_index.npz"))
    parser.add_argument("--clusters", type=Path, default=Path("data/clusters.json"))
    args = parser.parse_args()

    tickets = _load(args.history)
    twins = _twins(args.clusters)
    embedder = CachedEmbedder(get_embedding_model())

    index = SimilarityIndex.load_or_build(args.cache, tickets, embedder)
    print(f"index: {index.matrix.shape} (fingerprint {index.fingerprint[:12]}…)")

    sweep(index, embedder, twins)

    print("\nscore histogram of top-1 neighbours (k=1, no threshold):")
    buckets = Counter()
    for ticket in tickets:
        vec = np.asarray(
            embedder.embed([f"{ticket.short_description} {ticket.description}"])[0],
            dtype=np.float32,
        )
        vec = vec / np.linalg.norm(vec)
        scores = index.matrix @ vec
        scores[[i for i, t in enumerate(tickets) if t.ticket_id == ticket.ticket_id]] = -np.inf
        bucket = f"{int(float(scores.max()) * 20) / 20:.2f}"
        buckets[bucket] += 1
    for bucket in sorted(buckets):
        print(f"  {bucket}: {'#' * buckets[bucket]} {buckets[bucket]}")


if __name__ == "__main__":
    main()
