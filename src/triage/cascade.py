"""The triage cascade: Gate 1 rules -> Gate 2 similarity -> Gate 3 model.

An assembly line, not a contest: each gate fills the labels it can prove and
hands the partial verdict down. A ticket returns early only when all four
labels are populated; otherwise Gate 3 fills the remainder and derives
priority from impact x urgency.
"""

from triage.classifier import classify_ticket_gate3
from triage.llm import ChatModel, EmbeddingModel
from triage.models import LABEL_FIELDS, TicketInput, TriageVerdict
from triage.rules import apply_rules
from triage.similarity import SimilarityIndex, apply_similarity_partial


def _merge_partials(
    base: TriageVerdict | None, incoming: TriageVerdict | None
) -> TriageVerdict | None:
    """Overlay two partial verdicts; earlier gates win on any overlap."""
    if base is None:
        return incoming
    if incoming is None:
        return base

    labels = {
        f: getattr(base, f) if getattr(base, f) is not None else getattr(incoming, f)
        for f in LABEL_FIELDS
    }
    if not any(v is not None for v in labels.values()):
        return None
    return TriageVerdict(
        ticket_id=base.ticket_id,
        **labels,
        confidence=min(base.confidence, incoming.confidence),
        decided_by="cascade",
        label_source={**incoming.label_source, **base.label_source},
        rationale=f"{base.rationale} | {incoming.rationale}"[:300],
        similar_ticket_ids=list(
            dict.fromkeys(base.similar_ticket_ids + incoming.similar_ticket_ids)
        ),
        cost_usd=base.cost_usd + incoming.cost_usd,
        latency_ms=base.latency_ms + incoming.latency_ms,
    )


def _complete(verdict: TriageVerdict | None) -> bool:
    return verdict is not None and all(getattr(verdict, f) is not None for f in LABEL_FIELDS)


def triage(
    ticket: TicketInput,
    index: SimilarityIndex,
    embedder: EmbeddingModel,
    chat: ChatModel,
    *,
    sim_threshold: float = 0.75,
    sim_k: int = 3,
) -> TriageVerdict:
    """Run the full assembly-line cascade on one ticket."""
    verdict = apply_rules(ticket)
    if _complete(verdict):
        return verdict

    partial = apply_similarity_partial(ticket, index, embedder, threshold=sim_threshold, k=sim_k)
    merged = _merge_partials(verdict, partial)
    if _complete(merged):
        return merged

    examples = (
        [
            neighbour
            for tid in merged.similar_ticket_ids
            if (neighbour := index.by_id(tid)) is not None
        ]
        if merged is not None
        else None
    )
    result = classify_ticket_gate3(ticket, chat, merged, examples=examples)
    if result is not None:
        return result
    return TriageVerdict(
        ticket_id=ticket.ticket_id,
        confidence=0.0,
        decided_by="abstain",
        rationale="All gates declined to decide.",
        cost_usd=(merged.cost_usd if merged else 0.0),
    )
