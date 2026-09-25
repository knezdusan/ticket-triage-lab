"""Tests for Gate 2 similarity — mocked embeddings, no Azure calls."""

import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest

from triage.models import (
    AssignmentGroup,
    Category,
    LabelledTicket,
    Level,
    Priority,
    TicketInput,
    TicketType,
    TriageVerdict,
)
from triage.similarity import (
    SimilarityIndex,
    apply_similarity,
    apply_similarity_partial,
)


class FakeEmbedder:
    """Deterministic embedder: mapped texts get fixed vectors, everything
    else gets a hash-seeded Gaussian vector so distinct texts are
    near-orthogonal in high dimensions."""

    def __init__(self, mapping: dict[str, list[float]] | None = None, dim: int = 64):
        self.mapping = mapping or {}
        self.dim = dim
        self.last_usage = SimpleNamespace(prompt_tokens=10, total_tokens=10)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if text in self.mapping:
                vectors.append(list(self.mapping[text]))
            else:
                seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8])
                vectors.append(np.random.default_rng(seed).standard_normal(self.dim).tolist())
        return vectors


def make_labelled(
    ticket_id: str,
    short: str,
    desc: str,
    *,
    ticket_type: TicketType = TicketType.SERVICE_REQUEST,
    category: Category = Category.PASSWORD_ACCOUNT,
    impact: Level = Level.LOW,
    urgency: Level = Level.LOW,
    priority: Priority = Priority.P4,
    group: AssignmentGroup = AssignmentGroup.SERVICE_DESK,
) -> LabelledTicket:
    return LabelledTicket(
        ticket_id=ticket_id,
        created_at=datetime.now(UTC),
        short_description=short,
        description=desc,
        requester="tester@example.com",
        type=ticket_type,
        category=category,
        impact=impact,
        urgency=urgency,
        priority=priority,
        assignment_group=group,
        tier="L1",
    )


def make_query(short: str, desc: str) -> TicketInput:
    return TicketInput(
        ticket_id="REQ999999",
        created_at=datetime.now(UTC),
        short_description=short,
        description=desc,
        requester="tester@example.com",
    )


PASSWORD = ("Password reset needed", "Please reset my SAP password.")
TRANSPORT = ("Transport import", "Please import transport TRK123 to QAS.")
ACCESS = ("Access request", "Please grant access to VA01 for a new user.")
DUMP = ("ST22 dump", "Runtime error in VA01 billing.")


@pytest.fixture
def history() -> list[LabelledTicket]:
    # Three identical password tickets form a cluster; the rest are distinct.
    return [
        make_labelled("REQ000001", *PASSWORD),
        make_labelled("REQ000002", *PASSWORD),
        make_labelled("REQ000003", *PASSWORD),
        make_labelled(
            "REQ000010",
            *TRANSPORT,
            category=Category.TRANSPORT_CHANGE,
            group=AssignmentGroup.BASIS,
        ),
        make_labelled(
            "REQ000020",
            *ACCESS,
            category=Category.ACCESS_AUTHORIZATION,
            group=AssignmentGroup.SECURITY,
        ),
        make_labelled(
            "INC000030",
            *DUMP,
            ticket_type=TicketType.INCIDENT,
            category=Category.SHORT_DUMP,
            impact=Level.HIGH,
            urgency=Level.LOW,
            priority=Priority.P3,
            group=AssignmentGroup.BASIS,
        ),
    ]


@pytest.fixture
def index(history: list[LabelledTicket]) -> SimilarityIndex:
    return SimilarityIndex.build(history, FakeEmbedder())


class TestIndex:
    def test_build_normalizes_rows(self, index: SimilarityIndex):
        norms = np.linalg.norm(index.matrix, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(index.tickets)))

    def test_save_load_roundtrip(self, history, index: SimilarityIndex, tmp_path):
        path = tmp_path / "index.npz"
        index.save(path)
        loaded = SimilarityIndex.load(path, history)
        assert loaded is not None
        np.testing.assert_allclose(loaded.matrix, index.matrix)
        assert loaded.fingerprint == index.fingerprint

    def test_load_rejects_stale_history(self, history, index: SimilarityIndex, tmp_path):
        path = tmp_path / "index.npz"
        index.save(path)
        changed = history[:-1] + [make_labelled("REQ000040", "New", "Different text.")]
        assert SimilarityIndex.load(path, changed) is None

    def test_load_missing_file(self, history, tmp_path):
        assert SimilarityIndex.load(tmp_path / "nope.npz", history) is None


class TestApplySimilarity:
    def test_identical_neighbour_emits_verdict(self, index, history):
        verdict = apply_similarity(
            make_query(*PASSWORD), index, FakeEmbedder(), k=1, threshold=0.85
        )
        assert isinstance(verdict, TriageVerdict)
        assert verdict.decided_by == "similarity"
        assert verdict.type == TicketType.SERVICE_REQUEST
        assert verdict.category == Category.PASSWORD_ACCOUNT
        assert verdict.priority == Priority.P4
        assert verdict.assignment_group == AssignmentGroup.SERVICE_DESK
        assert verdict.confidence == pytest.approx(1.0)
        # All three cluster tickets tie at score ~1.0; argsort tie order is
        # unspecified, so accept any of them.
        assert verdict.similar_ticket_ids[0] in {"REQ000001", "REQ000002", "REQ000003"}
        assert verdict.cost_usd > 0.0
        assert verdict.latency_ms >= 0.0

    def test_distant_query_returns_none(self, index):
        query = make_query("Completely unrelated", "Nothing like any history.")
        assert apply_similarity(query, index, FakeEmbedder(), k=3) is None

    def test_unanimous_cluster_emits_all_ids(self, index):
        verdict = apply_similarity(
            make_query(*PASSWORD), index, FakeEmbedder(), k=3, threshold=0.85
        )
        assert verdict is not None
        assert set(verdict.similar_ticket_ids) == {"REQ000001", "REQ000002", "REQ000003"}

    def test_unanimous_consensus_rejects_disagreement(self):
        # Top-1 is an exact match, but the other two neighbours carry
        # different labels — unanimous k=3 must abstain.
        history = [
            make_labelled("REQ000001", *PASSWORD),
            make_labelled(
                "REQ000010",
                *TRANSPORT,
                category=Category.TRANSPORT_CHANGE,
                group=AssignmentGroup.BASIS,
            ),
            make_labelled(
                "REQ000020",
                *ACCESS,
                category=Category.ACCESS_AUTHORIZATION,
                group=AssignmentGroup.SECURITY,
            ),
        ]
        index = SimilarityIndex.build(history, FakeEmbedder())
        assert apply_similarity(make_query(*PASSWORD), index, FakeEmbedder(), k=3) is None

    def test_supermajority_consensus_emits_verdict(self):
        # Same setup, but consensus=0.67 with k=3 tolerates one dissenter.
        history = [
            make_labelled("REQ000001", *PASSWORD),
            make_labelled("REQ000002", *PASSWORD),
            make_labelled(
                "REQ000010",
                *TRANSPORT,
                category=Category.TRANSPORT_CHANGE,
                group=AssignmentGroup.BASIS,
            ),
        ]
        index = SimilarityIndex.build(history, FakeEmbedder())
        verdict = apply_similarity(
            make_query(*PASSWORD), index, FakeEmbedder(), k=3, consensus=2 / 3
        )
        assert verdict is not None
        assert verdict.category == Category.PASSWORD_ACCOUNT
        assert set(verdict.similar_ticket_ids) == {"REQ000001", "REQ000002"}

    def test_threshold_boundary(self, index):
        # float32 dot of identical unit vectors ≈ 0.99999994, not 1.0.
        verdict = apply_similarity(
            make_query(*PASSWORD), index, FakeEmbedder(), k=1, threshold=0.999
        )
        assert verdict is not None
        # A near-orthogonal query (~N(0, 0.125) scores) fails threshold 0.5.
        query = make_query("Unrelated", "Orthogonal text.")
        assert apply_similarity(query, index, FakeEmbedder(), k=1, threshold=0.5) is None

    def test_partial_match_below_threshold(self, index):
        # Query shares ~50% direction with password cluster → score ~0.71 < 0.85.
        embedder = FakeEmbedder({f"{PASSWORD[0]} {PASSWORD[1]}": [1.0] + [0.0] * 63})
        index = SimilarityIndex.build(index.tickets, embedder)
        hybrid = make_query("Hybrid", "Half password half other.")
        embedder.mapping[f"{hybrid.short_description} {hybrid.description}"] = [
            1.0,
            1.0,
        ] + [0.0] * 62
        assert apply_similarity(hybrid, index, embedder, k=1) is None


class TestApplySimilarityPartial:
    def test_locks_everything_except_priority_on_unanimous_cluster(self, index):
        verdict = apply_similarity_partial(
            make_query(*PASSWORD), index, FakeEmbedder(), threshold=0.75
        )
        assert verdict is not None
        assert verdict.decided_by == "similarity"
        assert verdict.category == Category.PASSWORD_ACCOUNT
        assert verdict.type == TicketType.SERVICE_REQUEST
        assert verdict.assignment_group == AssignmentGroup.SERVICE_DESK
        assert verdict.priority is None
        assert verdict.label_source == {
            "category": "similarity",
            "type": "similarity",
            "assignment_group": "similarity",
        }
        assert verdict.confidence == pytest.approx(1.0)

    def test_locks_only_agreed_fields(self):
        # Same category/type across top-3, but groups differ → group stays None.
        a = "Password reset alpha", "Reset please."
        b = "Password reset beta", "Reset me too."
        c = "Password reset gamma", "And me."
        mapping = {
            f"{a[0]} {a[1]}": [1.0, 0.0, 0.0, 0.0] + [0.0] * 60,
            f"{b[0]} {b[1]}": [0.9, 0.1, 0.0, 0.0] + [0.0] * 60,
            f"{c[0]} {c[1]}": [0.8, 0.2, 0.0, 0.0] + [0.0] * 60,
        }
        history = [
            make_labelled("REQ000001", *a),
            make_labelled("REQ000002", *b, group=AssignmentGroup.SECURITY),
            make_labelled("REQ000003", *c, group=AssignmentGroup.BASIS),
        ]
        index = SimilarityIndex.build(history, FakeEmbedder(mapping))
        verdict = apply_similarity_partial(
            make_query(*a), index, FakeEmbedder(mapping), threshold=0.75
        )
        assert verdict is not None
        assert verdict.category == Category.PASSWORD_ACCOUNT
        assert verdict.type == TicketType.SERVICE_REQUEST
        assert verdict.assignment_group is None
        assert verdict.priority is None
        assert verdict.label_source == {
            "category": "similarity",
            "type": "similarity",
        }

    def test_below_floor_returns_none(self, index):
        query = make_query("Completely unrelated", "Nothing like any history.")
        assert apply_similarity_partial(query, index, FakeEmbedder()) is None

    def test_category_bar_above_floor(self, index):
        # top-1 clears the floor but not the higher category bar →
        # only unanimous type/group may be emitted.
        verdict = apply_similarity_partial(
            make_query(*PASSWORD),
            index,
            FakeEmbedder(),
            threshold=0.75,
            category_threshold=1.0,  # float32 top-1 is ~0.99999994 < 1.0
        )
        assert verdict is not None
        assert verdict.category is None
        assert verdict.type == TicketType.SERVICE_REQUEST
        assert verdict.assignment_group == AssignmentGroup.SERVICE_DESK


class TestIndexLookup:
    def test_by_id_resolves(self, index):
        ticket = index.by_id("REQ000001")
        assert ticket is not None
        assert ticket.category == Category.PASSWORD_ACCOUNT

    def test_by_id_unknown_returns_none(self, index):
        assert index.by_id("INC999999") is None
