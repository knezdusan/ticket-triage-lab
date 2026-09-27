"""Live verification of Azure Document Intelligence + Structure-Aware Retrieval.

Demonstrates:
1. Live layout analysis of data/ams_sla_runbook.pdf via prebuilt-layout.
2. Structure-aware chunking preserving 2D tables as Markdown.
3. Naive text vs Structure-aware contrast.
4. Vector embedding and semantic search over document chunks.

NEEDS AZURE CREDENTIALS (DocIntel + OpenAI Embeddings).
Usage: uv run python scripts/verify_docintel.py
"""

import time
from pathlib import Path

import numpy as np

from triage.documents import analyze_layout, to_chunks
from triage.llm import get_embedding_model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    pdf_path = Path("data/ams_sla_runbook.pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing {pdf_path}. Run generate_test_pdf.py first.")

    print("=" * 70)
    print("AZURE AI DOCUMENT INTELLIGENCE — LIVE VERIFICATION")
    print("=" * 70)

    # 1. Live Analysis
    print(f"\n1. Sending '{pdf_path}' to Azure Document Intelligence (prebuilt-layout)...")
    t0 = time.perf_counter()
    layout = analyze_layout(pdf_path)
    analysis_time = time.perf_counter() - t0
    print(f"   Analysis completed in {analysis_time:.2f}s.")
    print(f"   Extracted: Title='{layout.title}'")
    print(f"   Markdown: {len(layout.markdown_content)} chars")
    md_path = pdf_path.with_suffix(".md")
    md_path.write_text(layout.markdown_content)
    print(f"   Markdown saved to '{md_path}' for inspection.")

    # 2. Structure-Aware Chunking
    chunks = to_chunks(layout)
    print(f"\n2. Emitted {len(chunks)} structure-aware chunks:")
    for chunk in chunks:
        preview = chunk.content.replace("\n", " ")[:60]
        print(
            f"   - [{chunk.chunk_id}] ({chunk.chunk_type:<5}) "
            f"Heading: '{chunk.heading}' | {preview}..."
        )

    # 3. Fixed-window slice vs Semantic Section Chunk
    table_chunk = next((c for c in chunks if c.chunk_type == "table"), None)
    print("\n" + "=" * 70)
    print("THE CONTRAST: FIXED-WINDOW SLICE vs SEMANTIC SECTION CHUNK")
    print("=" * 70)
    print("\n--- NAIVE 300-CHAR WINDOW (What fixed-size chunking gives you) ---")
    anchor = layout.markdown_content.find("| P1 |")
    if anchor == -1:
        anchor = layout.markdown_content.find("P1")
    naive_slice = layout.markdown_content[max(0, anchor - 100) : anchor + 200]
    print(f'"{naive_slice}..."')
    print("(Notice: sliced mid-table — row detached from its headers and heading)")

    if table_chunk is None:
        print("\n--- WARNING: no table chunk detected — inspect the saved Markdown ---")
    else:
        print("\n--- SEMANTIC SECTION CHUNK (What our pipeline gives you) ---")
        print(table_chunk.content)
        print("(Notice: heading attached, table fully intact, headers with every row)")

    # 4. Semantic Search Test
    print("\n" + "=" * 70)
    print("SEMANTIC RETRIEVAL TEST (Vector Search over Document Chunks)")
    print("=" * 70)

    embedder = get_embedding_model()
    print("\nEmbedding document chunks with text-embedding-3-small...")
    chunk_texts = [c.content for c in chunks]
    chunk_vectors = [np.asarray(v, dtype=np.float32) for v in embedder.embed(chunk_texts)]

    queries = [
        ("What is the target resolution time for a P1 incident?", "table", "2 hours"),
        ("Who approves emergency transport requests to PRD?", "prose", "Basis Lead"),
    ]

    all_passed = True
    for q_text, expected_type, expected_keyword in queries:
        print(f'\nQuery: "{q_text}"')
        q_vec = np.asarray(embedder.embed([q_text])[0], dtype=np.float32)

        # Rank chunks by cosine similarity
        scores = [_cosine_similarity(q_vec, cv) for cv in chunk_vectors]
        ranked_indices = np.argsort(scores)[::-1]

        top_idx = ranked_indices[0]
        top_chunk = chunks[top_idx]
        top_score = scores[top_idx]

        match_ok = expected_keyword.lower() in top_chunk.content.lower()
        type_ok = top_chunk.chunk_type == expected_type
        all_passed &= match_ok and type_ok
        print(
            f"Top Result: [{top_chunk.chunk_id}] ({top_chunk.chunk_type}) - Cosine: {top_score:.4f}"
        )
        print(f"Section Heading: '{top_chunk.heading}'")
        print(f"Content:\n{top_chunk.content}")
        print(
            f"Verification: keyword '{expected_keyword}' {'PASSED' if match_ok else 'FAILED'} "
            f"| expected type '{expected_type}' {'PASSED' if type_ok else 'FAILED'}"
        )

    print("\n" + "=" * 70)
    verdict = "VERIFIED" if all_passed else "PARTIAL — inspect retrieval above"
    print(f"DOCUMENT INTELLIGENCE RETRIEVAL: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
