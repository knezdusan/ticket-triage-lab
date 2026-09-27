"""Azure AI Document Intelligence boundary + semantic Markdown chunking.

Only this module imports `azure-ai-documentintelligence` — the same
boundary rule as `llm.py` (OpenAI SDK) and `search_index.py` (AI Search).

Level 1 approach: request `output_content_format=MARKDOWN` so Azure's
server-side engine resolves reading order and table geometry itself —
headings become `#`/`##`, tables are inlined as Markdown grids (or as
HTML `<table>` markup when a grid is too complex), and cells are never
double-emitted as stray paragraphs. `to_chunks` then does
semantic header splitting: each `#`-`###` section (heading + prose +
intact tables) becomes one atomic chunk. No coordinate math client-side.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat
from azure.core.credentials import AzureKeyCredential

from triage.config import Settings, get_settings

# A new section starts at a Markdown heading (# .. ###) on its own line.
_SECTION_SPLIT = re.compile(r"\n(?=#{1,3}\s+)")
_HEADING_LINE = re.compile(r"^#{1,3}\s+(?P<title>.+?)\s*$")
# Tables appear either as a Markdown grid ("| --- | --- |" separator)
# or as inline HTML when Azure falls back for complex tables ("<table>").
_TABLE_SEPARATOR = re.compile(r"^\|\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$|<table[\s>]", re.MULTILINE)


@dataclass(frozen=True)
class LayoutResult:
    """Clean representation of Document Intelligence layout analysis.

    `markdown_content` is the document rendered server-side as Markdown —
    headings, prose, and inline tables in true reading order.
    """

    title: str | None
    markdown_content: str


@dataclass(frozen=True)
class DocumentChunk:
    """A structure-aware chunk ready for embedding and retrieval."""

    chunk_id: str
    chunk_type: Literal["prose", "table"]
    heading: str | None
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


def analyze_layout(
    file_path: Path,
    settings: Settings | None = None,
    client: DocumentIntelligenceClient | None = None,
) -> LayoutResult:
    """Run prebuilt-layout on a document and return its Markdown rendering."""
    if client is None:
        settings = settings or get_settings()
        if not settings.azure_docintel_endpoint or not settings.azure_docintel_key:
            raise RuntimeError(
                "AZURE_DOCINTEL_ENDPOINT and AZURE_DOCINTEL_KEY are required "
                "for Document Intelligence analysis."
            )
        client = DocumentIntelligenceClient(
            settings.azure_docintel_endpoint,
            AzureKeyCredential(settings.azure_docintel_key),
        )

    with file_path.open("rb") as fh:
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=fh,
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
    result = poller.result()

    # Paragraph roles are still populated alongside Markdown output.
    title = next((p.content for p in result.paragraphs or [] if p.role == "title"), None)
    return LayoutResult(title=title, markdown_content=result.content or "")


def to_chunks(layout: LayoutResult) -> list[DocumentChunk]:
    """Split the Markdown into one atomic chunk per heading section.

    A section's chunk keeps its heading line, its prose, and any inline
    table whole — headings stay attached and tables are never sliced.
    Content before the first heading becomes a heading-less preamble chunk.
    """
    chunks: list[DocumentChunk] = []
    for section in _SECTION_SPLIT.split(layout.markdown_content.strip()):
        content = section.strip()
        if not content:
            continue
        first_line = content.splitlines()[0]
        match = _HEADING_LINE.match(first_line)
        chunks.append(
            DocumentChunk(
                chunk_id=f"chunk-{len(chunks):03d}",
                chunk_type="table" if _TABLE_SEPARATOR.search(content) else "prose",
                heading=match.group("title") if match else None,
                content=content,
            )
        )
    return chunks
