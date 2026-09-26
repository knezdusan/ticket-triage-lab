"""Tests for the Azure AI Search backend — SDK fully mocked, zero network."""

import math
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
    TicketType,
)
from triage.search_index import VECTOR_DIMENSIONS, AzureSearchIndex


class FakeIndexClient:
    """Records the index schema it was asked to create."""

    def __init__(self) -> None:
        self.created: list = []

    def create_or_update_index(self, index) -> None:
        self.created.append(index)


class FakeSearchClient:
    """Stores uploaded docs and answers vector queries with real cosine.

    Mimics the service: score is rescaled to 1 / (2 - cos), results are
    ordered best-first, and k_nearest_neighbors bounds the count.
    """

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.upload_calls = 0

    def upload_documents(self, docs: list[dict]) -> None:
        self.upload_calls += 1
        for doc in docs:
            self.docs[doc["ticket_id"]] = doc

    def search(self, *, vector_queries, top, **_kwargs):
        vector = np.asarray(vector_queries[0].vector, dtype=np.float64)
        vector /= np.linalg.norm(vector)

        def score(doc: dict) -> float:
            v = np.asarray(doc["embedding"], dtype=np.float64)
            v /= np.linalg.norm(v)
            return 1.0 / (2.0 - float(vector @ v))

        ranked = sorted(self.docs.values(), key=score, reverse=True)
        return [{**doc, "@search.score": score(doc)} for doc in ranked[:top]]


class FakeEmbedder:
    """Deterministic near-orthogonal vectors per text."""

    dim = 64

    def __init__(self) -> None:
        self.last_usage = SimpleNamespace(prompt_tokens=0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.standard_normal(self.dim).tolist()


def make_labelled(ticket_id: str, short: str, desc: str, **over) -> LabelledTicket:
    fields = dict(
        ticket_id=ticket_id,
        created_at=datetime.now(UTC),
        short_description=short,
        description=desc,
        requester="tester@example.com",
        type=TicketType.INCIDENT,
        category=Category.OUTPUT_PRINTING,
        impact=Level.LOW,
        urgency=Level.LOW,
        priority=Priority.P4,
        assignment_group=AssignmentGroup.BASIS,
        tier="L1",
    )
    fields.update(over)
    return LabelledTicket(**fields)


@pytest.fixture
def backend():
    index_client = FakeIndexClient()
    search_client = FakeSearchClient()
    return AzureSearchIndex(
        SimpleNamespace(
            azure_search_index_name="test-index",
            azure_search_endpoint=None,
            azure_search_key=None,
        ),
        index_client=index_client,
        search_client=search_client,
    )


class TestIndexCreation:
    def test_schema_has_vector_profile_and_fields(self, backend):
        backend.create_or_update_index()
        index = backend._index_client.created[0]

        assert index.name == "test-index"
        names = {f.name: f for f in index.fields}
        assert names["ticket_id"].key is True
        vector = names["embedding"]
        assert vector.vector_search_dimensions == VECTOR_DIMENSIONS
        assert vector.vector_search_profile_name is not None
        for label in ("type", "category", "priority", "assignment_group"):
            assert label in names
            assert names[label].filterable is True
        algorithm = index.vector_search.algorithms[0]
        assert algorithm.parameters.metric == "cosine"


class TestUpload:
    def test_uploads_all_fields_and_vector(self, backend):
        ticket = make_labelled("INC000001", "Printer spool error", "Spool stuck in SP01.")
        backend.upload_tickets([ticket], FakeEmbedder())

        doc = backend._search_client.docs["INC000001"]
        assert doc["type"] == "incident"
        assert doc["priority"] == "P4"
        assert len(doc["embedding"]) == FakeEmbedder.dim

    def test_batches_at_100(self, backend):
        tickets = [make_labelled(f"INC000{i:03}", f"T-{i}", f"Body {i}") for i in range(205)]
        backend.upload_tickets(tickets, FakeEmbedder())
        assert backend._search_client.upload_calls == 3
        assert len(backend._search_client.docs) == 205

    def test_naive_created_at_gets_utc_offset(self, backend):
        # history.jsonl carries naive timestamps; Edm.DateTimeOffset needs Z.
        naive = datetime(2026, 7, 5, 8, 32, 38)  # noqa: DTZ001 — fixture mirrors data
        ticket = make_labelled("INC000003", "Naive ts", "Body", created_at=naive)
        backend.upload_tickets([ticket], FakeEmbedder())
        doc = backend._search_client.docs["INC000003"]
        assert doc["created_at"] == "2026-07-05T08:32:38Z"

    def test_by_id_resolves_uploaded(self, backend):
        ticket = make_labelled("INC000002", "Query", "Description")
        backend.upload_tickets([ticket], FakeEmbedder())
        assert backend.by_id("INC000002") is ticket
        assert backend.by_id("INC999999") is None


class TestSearch:
    def test_returns_labelled_tickets_ordered_by_similarity(self, backend):
        near = make_labelled("INC000010", "near me", "same body")
        far = make_labelled("INC000011", "different", "other body")
        backend.upload_tickets([near, far], FakeEmbedder())

        # Query = near's own vector → near must rank first.
        vector = FakeEmbedder()._vec("near me same body")
        hits = backend.search_similar(vector, k=2)

        assert hits[0][0].ticket_id == "INC000010"
        assert isinstance(hits[0][0], LabelledTicket)
        assert hits[0][1] > hits[1][1]

    def test_scores_are_rescaled_cosine(self, backend):
        ticket = make_labelled("INC000012", "self", "identical")
        backend.upload_tickets([ticket], FakeEmbedder())
        vector = FakeEmbedder()._vec("self identical")

        _, score = backend.search_similar(vector, k=1)[0]
        # Self-query: cosine 1.0 → Azure (1+1)/2 = 1.0
        assert math.isclose(score, 1.0, abs_tol=1e-6)

    def test_roundtrip_reconstructs_all_fields(self, backend):
        ticket = make_labelled(
            "REQ000007",
            "Password reset",
            "Account locked.",
            type=TicketType.SERVICE_REQUEST,
            category=Category.PASSWORD_ACCOUNT,
            assignment_group=AssignmentGroup.SERVICE_DESK,
            resolution_notes="Unlocked via SU01.",
        )
        backend.upload_tickets([ticket], FakeEmbedder())
        vector = FakeEmbedder()._vec("Password reset Account locked.")

        found, _ = backend.search_similar(vector, k=1)[0]
        assert found == ticket


class TestMissingSettings:
    def test_requires_endpoint_and_key_without_clients(self):
        settings = SimpleNamespace(
            azure_search_index_name="x",
            azure_search_endpoint=None,
            azure_search_key=None,
        )
        with pytest.raises(RuntimeError, match="AZURE_SEARCH"):
            AzureSearchIndex(settings)
