"""Gate 2 — embedding similarity against historical labelled tickets.

Embeds the incoming ticket, finds its top-k nearest neighbours in a cached,
unit-normalized index of history vectors, and emits a verdict only when the
closest match clears a similarity threshold AND the neighbours agree on all
four labels. Ambiguous or distant queries return None and pass down to Gate 3.

Only ticket text is embedded — never the ticket_id prefix.
"""

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np

from triage.llm import EMBEDDING_COST_PER_TOKEN, EmbeddingModel
from triage.models import LabelledTicket, TicketInput, TriageVerdict

type LabelField = Literal["type", "category", "priority", "assignment_group"]

_LABEL_FIELDS: tuple[LabelField, ...] = ("type", "category", "priority", "assignment_group")


def _text(ticket: TicketInput) -> str:
    """Option C: embed short description + description together."""
    return f"{ticket.short_description} {ticket.description}"


def _fingerprint(tickets: list[LabelledTicket]) -> str:
    """Content hash over id+text so a stale cache can never silently serve."""
    digest = hashlib.sha256()
    for t in tickets:
        digest.update(f"{t.ticket_id}|{t.short_description}|{t.description}\n".encode())
    return digest.hexdigest()


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to unit length; zero vectors stay zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


class SimilarityIndex:
    """A unit-normalized embedding matrix over labelled history."""

    def __init__(
        self,
        tickets: list[LabelledTicket],
        matrix: np.ndarray,
        fingerprint: str,
    ) -> None:
        self.tickets = tickets
        self.matrix = matrix
        self.fingerprint = fingerprint

    @classmethod
    def build(cls, tickets: list[LabelledTicket], embedder: EmbeddingModel) -> "SimilarityIndex":
        """Embed all tickets and normalize rows to unit length."""
        vectors = embedder.embed([_text(t) for t in tickets])
        matrix = _normalize(np.asarray(vectors, dtype=np.float32))
        return cls(tickets, matrix, _fingerprint(tickets))

    def save(self, path: Path) -> None:
        """Persist vectors + metadata to a single .npz (no pickle)."""
        meta = json.dumps(
            {
                "fingerprint": self.fingerprint,
                "tickets": [t.model_dump(mode="json") for t in self.tickets],
            }
        ).encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            vectors=self.matrix.astype(np.float32),
            meta=np.frombuffer(meta, dtype=np.uint8),
        )

    @classmethod
    def load(cls, path: Path, tickets: list[LabelledTicket]) -> "SimilarityIndex | None":
        """Load a cached index, or None if missing/stale for these tickets."""
        if not path.exists():
            return None
        data = np.load(path)
        meta = json.loads(bytes(data["meta"].tobytes()))
        if meta["fingerprint"] != _fingerprint(tickets):
            return None
        matrix = np.asarray(data["vectors"], dtype=np.float32)
        return cls(tickets, matrix, meta["fingerprint"])

    @classmethod
    def load_or_build(
        cls, path: Path, tickets: list[LabelledTicket], embedder: EmbeddingModel
    ) -> "SimilarityIndex":
        """Serve the cache if valid, else embed history and refresh it."""
        index = cls.load(path, tickets)
        if index is None:
            index = cls.build(tickets, embedder)
            index.save(path)
        return index

    def neighbours(
        self,
        vector: np.ndarray,
        k: int,
        exclude_ids: frozenset[str] | None = None,
    ) -> list[tuple[LabelledTicket, float]]:
        """Cosine similarity via dot product over unit-normalized rows."""
        scores = self.matrix @ vector
        if exclude_ids:
            scores = scores.copy()
            for i, t in enumerate(self.tickets):
                if t.ticket_id in exclude_ids:
                    scores[i] = -np.inf
        top = np.argsort(scores)[::-1][:k]
        return [(self.tickets[i], float(scores[i])) for i in top if scores[i] > -np.inf]


def apply_similarity(
    ticket: TicketInput,
    index: SimilarityIndex,
    embedder: EmbeddingModel,
    *,
    k: int = 3,
    threshold: float = 0.85,
    consensus: float = 1.0,
    exclude_ids: frozenset[str] | None = None,
) -> TriageVerdict | None:
    """Return a verdict when the top-k neighbours are close AND agree.

    threshold — minimum top-1 cosine similarity.
    consensus — fraction of the k neighbours that must share each label
                (1.0 = unanimous; 0.8 with k=5 = 4-of-5 supermajority).
    exclude_ids — history ticket_ids to skip (leave-one-out evaluation).
    """
    start = time.perf_counter()
    if not index.tickets:
        return None
    k = min(k, len(index.tickets))

    vector = _normalize(np.asarray(embedder.embed([_text(ticket)]), dtype=np.float32))[0]
    neighbours = index.neighbours(vector, k, exclude_ids)

    # Threshold check: the single closest match must be near enough.
    if not neighbours:
        return None
    top1 = neighbours[0][1]
    if top1 < threshold:
        return None

    # Consensus check: each label field independently needs a quorum.
    quorum = math.ceil(consensus * len(neighbours))
    labels: dict[LabelField, object] = {}
    for field in _LABEL_FIELDS:
        value, count = Counter(getattr(t, field) for t, _ in neighbours).most_common(1)[0]
        if count < quorum:
            return None
        labels[field] = value

    agreeing = [
        (t, score)
        for t, score in neighbours
        if all(getattr(t, f) == labels[f] for f in _LABEL_FIELDS)
    ]
    confidence = float(np.mean([score for _, score in agreeing]))

    usage = getattr(embedder, "last_usage", None)
    cost = (usage.prompt_tokens * EMBEDDING_COST_PER_TOKEN) if usage else 0.0

    return TriageVerdict(
        ticket_id=ticket.ticket_id,
        type=labels["type"],
        category=labels["category"],
        priority=labels["priority"],
        assignment_group=labels["assignment_group"],
        confidence=confidence,
        decided_by="similarity",
        label_source=dict.fromkeys(_LABEL_FIELDS, "similarity"),
        rationale=(
            f"Top-1 similarity {top1:.2f} >= {threshold}; "
            f"{len(agreeing)}/{len(neighbours)} neighbours agree on all labels."
        ),
        similar_ticket_ids=[t.ticket_id for t, _ in agreeing],
        cost_usd=cost,
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )


def apply_similarity_partial(
    ticket: TicketInput,
    index: SimilarityIndex,
    embedder: EmbeddingModel,
    *,
    threshold: float = 0.75,
    category_threshold: float | None = None,
    k: int = 3,
    exclude_ids: frozenset[str] | None = None,
) -> TriageVerdict | None:
    """Per-field Gate 2: emit only the labels the evidence supports.

    Evidence bars (from the eval_gate2 sweep over history.jsonl):
      - category: top-1 score >= category_threshold (default `threshold`).
      - type, assignment_group: unanimous across the top-k neighbours.
      - priority: never emitted — embeddings capture topic, not severity.
    Top-1 below `threshold` means we don't know this ticket at all → None.
    """
    start = time.perf_counter()
    if not index.tickets:
        return None
    cat_bar = category_threshold if category_threshold is not None else threshold
    k = min(k, len(index.tickets))

    vector = _normalize(np.asarray(embedder.embed([_text(ticket)]), dtype=np.float32))[0]
    neighbours = index.neighbours(vector, k, exclude_ids)
    if not neighbours or neighbours[0][1] < threshold:
        return None

    top1_ticket, top1 = neighbours[0]
    labels: dict[str, object] = {}
    if top1 >= cat_bar:
        labels["category"] = top1_ticket.category
    for field in ("type", "assignment_group"):
        if len(neighbours) == k and len({getattr(t, field) for t, _ in neighbours}) == 1:
            labels[field] = getattr(top1_ticket, field)
    if not labels:
        return None

    agreeing = [
        (t, score) for t, score in neighbours if all(getattr(t, f) == labels[f] for f in labels)
    ]
    usage = getattr(embedder, "last_usage", None)
    cost = (usage.prompt_tokens * EMBEDDING_COST_PER_TOKEN) if usage else 0.0

    return TriageVerdict(
        ticket_id=ticket.ticket_id,
        type=labels.get("type"),
        category=labels.get("category"),
        priority=None,
        assignment_group=labels.get("assignment_group"),
        confidence=top1,
        decided_by="similarity",
        label_source={f: "similarity" for f in labels},
        rationale=(
            f"Top-1 similarity {top1:.2f} >= {threshold}; "
            f"locked {sorted(labels)} (priority left for downstream)."
        ),
        similar_ticket_ids=[t.ticket_id for t, _ in agreeing],
        cost_usd=cost,
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )
