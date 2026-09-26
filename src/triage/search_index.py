"""Azure AI Search retrieval backend — a hosted stand-in for SimilarityIndex.

Same shape as the numpy index: declare a schema once, upload embedded
tickets, query by vector for top-k neighbours. This exists to build fluency
with the service, not because Gate 2 needs it at n=240 — do not wire it
into the cascade without measuring.

Boundary rule: `azure-search-documents` is imported only in this module,
the same way `llm.py` owns the OpenAI SDK.

Score caveat: with the cosine metric, Azure rescales raw cosine `s` to
`1 / (2 - s)` in `@search.score` (verified live against the numpy index —
e.g. cos 0.7620 -> 0.8077). Ordering is preserved, but values are NOT
comparable to SimilarityIndex's raw cosine; recover it as `2 - 1/score`.
"""

from datetime import datetime

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from triage.config import Settings, get_settings
from triage.llm import EmbeddingModel
from triage.models import (
    AssignmentGroup,
    Category,
    LabelledTicket,
    Level,
    Priority,
    TicketType,
    Tier,
)
from triage.similarity import _text
from triage.utils import chunks

VECTOR_DIMENSIONS = 1536
_VECTOR_FIELD = "embedding"
_VECTOR_PROFILE = "tickets-vector-profile"
_HNSW_ALGORITHM = "tickets-hnsw"
_BATCH_SIZE = 100


class AzureSearchIndex:
    """Vector index over labelled history, hosted on Azure AI Search."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        index_client: SearchIndexClient | None = None,
        search_client: SearchClient | None = None,
    ) -> None:
        settings = settings or get_settings()
        self.index_name = settings.azure_search_index_name
        if index_client is not None and search_client is not None:
            self._index_client = index_client
            self._search_client = search_client
        else:
            if not settings.azure_search_endpoint or not settings.azure_search_key:
                raise RuntimeError(
                    "AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY are required "
                    "for the AzureSearchIndex backend."
                )
            credential = AzureKeyCredential(settings.azure_search_key)
            self._index_client = SearchIndexClient(settings.azure_search_endpoint, credential)
            self._search_client = SearchClient(
                settings.azure_search_endpoint, self.index_name, credential
            )
        # In-memory mirror of uploaded tickets for O(1) few-shot lookup,
        # mirroring SimilarityIndex.by_id — populated by upload_tickets.
        self._by_id: dict[str, LabelledTicket] = {}

    def create_or_update_index(self) -> None:
        """Declare the index schema: searchable text, filterable labels, one
        1536-dim vector field wired to an HNSW/cosine profile."""
        label_fields = [
            SimpleField(name=name, type=SearchFieldDataType.String, filterable=True)
            for name in (
                "type",
                "category",
                "impact",
                "urgency",
                "priority",
                "assignment_group",
                "tier",
            )
        ]
        fields = [
            SimpleField(
                name="ticket_id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchableField(name="short_description", type=SearchFieldDataType.String),
            SearchableField(name="description", type=SearchFieldDataType.String),
            SearchableField(name="resolution_notes", type=SearchFieldDataType.String),
            SimpleField(name="requester", type=SearchFieldDataType.String),
            SimpleField(name="sap_module", type=SearchFieldDataType.String, filterable=True),
            SimpleField(
                name="created_at",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True,
            ),
            *label_fields,
            SearchField(
                name=_VECTOR_FIELD,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=VECTOR_DIMENSIONS,
                vector_search_profile_name=_VECTOR_PROFILE,
            ),
        ]
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name=_HNSW_ALGORITHM,
                        parameters=HnswParameters(metric="cosine"),
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name=_VECTOR_PROFILE,
                        algorithm_configuration_name=_HNSW_ALGORITHM,
                    )
                ],
            ),
        )
        self._index_client.create_or_update_index(index)

    @staticmethod
    def _to_document(ticket: LabelledTicket, vector: list[float]) -> dict:
        created = ticket.created_at
        doc = {
            "ticket_id": ticket.ticket_id,
            "short_description": ticket.short_description,
            "description": ticket.description,
            "requester": ticket.requester,
            # Edm.DateTimeOffset rejects naive ISO strings; this dataset's
            # timestamps are UTC, so a missing offset means "Z".
            "created_at": (
                created.isoformat() if created.tzinfo is not None else created.isoformat() + "Z"
            ),
            "type": ticket.type.value,
            "category": ticket.category.value,
            "impact": ticket.impact.value,
            "urgency": ticket.urgency.value,
            "priority": ticket.priority.value,
            "assignment_group": ticket.assignment_group.value,
            "tier": ticket.tier.value,
            _VECTOR_FIELD: vector,
        }
        if ticket.sap_module:
            doc["sap_module"] = ticket.sap_module
        if ticket.resolution_notes:
            doc["resolution_notes"] = ticket.resolution_notes
        return doc

    @staticmethod
    def _to_ticket(doc: dict) -> LabelledTicket:
        return LabelledTicket(
            ticket_id=doc["ticket_id"],
            created_at=datetime.fromisoformat(doc["created_at"]),
            short_description=doc["short_description"],
            description=doc["description"],
            requester=doc["requester"],
            sap_module=doc.get("sap_module"),
            type=TicketType(doc["type"]),
            category=Category(doc["category"]),
            impact=Level(doc["impact"]),
            urgency=Level(doc["urgency"]),
            priority=Priority(doc["priority"]),
            assignment_group=AssignmentGroup(doc["assignment_group"]),
            tier=Tier(doc["tier"]),
            resolution_notes=doc.get("resolution_notes"),
        )

    def upload_tickets(self, tickets: list[LabelledTicket], embedder: EmbeddingModel) -> None:
        """Embed and upload tickets in batches of 100 documents."""
        for batch in chunks(tickets, _BATCH_SIZE):
            vectors = embedder.embed([_text(t) for t in batch])
            docs = [self._to_document(t, v) for t, v in zip(batch, vectors, strict=True)]
            self._search_client.upload_documents(docs)
            self._by_id.update({t.ticket_id: t for t in batch})

    def by_id(self, ticket_id: str) -> LabelledTicket | None:
        """O(1) lookup over tickets uploaded through this process."""
        return self._by_id.get(ticket_id)

    def search_similar(
        self, query_vector: list[float], k: int = 3
    ) -> list[tuple[LabelledTicket, float]]:
        """Top-k nearest neighbours by HNSW/cosine vector query.

        Scores are Azure-rescaled ((1 + cosine) / 2), not raw cosine.
        """
        results = self._search_client.search(
            search_text=None,
            vector_queries=[
                VectorizedQuery(
                    vector=list(query_vector),
                    k_nearest_neighbors=k,
                    fields=_VECTOR_FIELD,
                )
            ],
            top=k,
        )
        return [(self._to_ticket(doc), float(doc["@search.score"])) for doc in results]
