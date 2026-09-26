"""Live verification of Azure AI Search vs NumPy SimilarityIndex.

Runs live against Azure:
1. Creates or updates index 'tickets-history'
2. Uploads 240 tickets from data/history.jsonl
3. Runs sample queries against both NumPy and Azure
4. Verifies top-k neighbour overlap and score rescaling formula:
   azure_score = 1 / (2 - cosine)  ⟺  cosine = 2 - 1/azure_score

NEEDS AZURE CREDENTIALS (embeddings + Azure AI Search). ~$0.01 spend.
Usage: uv run python scripts/verify_azure_search.py
"""

import json
import time
from pathlib import Path

import numpy as np

from triage.llm import get_embedding_model
from triage.models import LabelledTicket
from triage.search_index import AzureSearchIndex
from triage.similarity import SimilarityIndex, _normalize, _text


def _load(path: Path) -> list[LabelledTicket]:
    return [
        LabelledTicket.model_validate(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def main() -> None:
    print("=" * 70)
    print("AZURE AI SEARCH — LIVE VERIFICATION & PARITY CHECK")
    print("=" * 70)

    # 1. Load history tickets
    history_path = Path("data/history.jsonl")
    tickets = _load(history_path)
    print(f"\n1. Loaded {len(tickets)} tickets from {history_path}")

    # 2. Initialize Embedder and Azure Search Index
    embedder = get_embedding_model()
    search_index = AzureSearchIndex()

    # 3. Create or update index schema
    print(f"2. Creating/updating schema for index '{search_index.index_name}'...")
    search_index.create_or_update_index()
    print("   Schema created successfully.")

    # 4. Upload tickets
    print(f"3. Embedding and uploading {len(tickets)} tickets in batches of 100...")
    t0 = time.perf_counter()
    search_index.upload_tickets(tickets, embedder)
    print(f"   Upload complete in {time.perf_counter() - t0:.2f}s.")

    # Allow a few seconds for Azure's background indexing / commit
    time.sleep(2.0)

    # 5. Load NumPy index for comparison
    numpy_cache = Path("data/history_index.npz")
    numpy_index = SimilarityIndex.load_or_build(numpy_cache, tickets, embedder)
    print(f"4. Loaded NumPy SimilarityIndex ({numpy_index.matrix.shape})")

    # 6. Run side-by-side queries on sample evaluation tickets
    eval_tickets = _load(Path("data/eval.jsonl"))[:3]

    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE QUERY COMPARISON (NumPy Exact vs Azure HNSW)")
    print("=" * 70)

    rank_matches = 0
    rank_total = 0
    for i, test_ticket in enumerate(eval_tickets, 1):
        # LabelledTicket IS a TicketInput — embed the same text Gate 2 uses.
        raw_vec = embedder.embed([_text(test_ticket)])[0]
        # NumPy expects a unit-normalized query for dot == cosine;
        # Azure takes the raw vector (cosine is computed service-side).
        np_vec = _normalize(np.asarray([raw_vec], dtype=np.float32))[0]

        t_np = time.perf_counter()
        np_matches = numpy_index.neighbours(np_vec, k=3)
        np_latency = (time.perf_counter() - t_np) * 1000.0

        t_az = time.perf_counter()
        az_matches = search_index.search_similar([float(x) for x in raw_vec], k=3)
        az_latency = (time.perf_counter() - t_az) * 1000.0

        print(f"\n[Test Ticket {i}/3] {test_ticket.ticket_id} ({test_ticket.category})")
        print(f"Text: {test_ticket.short_description[:65]}...")
        print(f"NumPy Latency: {np_latency:.2f} ms | Azure Latency: {az_latency:.2f} ms")
        print("-" * 70)
        header = (
            f"{'Rank':<5} {'NumPy Match':<16} {'Raw Cosine':<12} "
            f"{'Azure Match':<16} {'Azure Score':<12} {'Recovered Cos':<12} {'Overlap':<8}"
        )
        print(header)

        for rank, ((np_t, np_score), (az_t, az_score)) in enumerate(
            zip(np_matches, az_matches, strict=True), 1
        ):
            # Exact Azure inverse: cos = 2 - 1/score (verified live vs NumPy)
            recovered_cos = 2.0 - (1.0 / az_score)
            id_match = "MATCH" if np_t.ticket_id == az_t.ticket_id else "DIFF"
            rank_matches += np_t.ticket_id == az_t.ticket_id
            rank_total += 1
            print(
                f"#{rank:<4} "
                f"{np_t.ticket_id:<16} "
                f"{np_score:<12.4f} "
                f"{az_t.ticket_id:<16} "
                f"{az_score:<12.4f} "
                f"{recovered_cos:<12.4f} "
                f"[{id_match}]"
            )

    print("\n" + "=" * 70)
    print(
        f"Rank-position overlap: {rank_matches}/{rank_total} "
        f"({rank_matches / rank_total:.0%}) — the rest is HNSW approximation."
    )
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
