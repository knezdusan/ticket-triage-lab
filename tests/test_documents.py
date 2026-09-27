"""Offline unit tests for Document Intelligence analysis and chunking."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from azure.ai.documentintelligence.models import DocumentContentFormat

from triage.documents import LayoutResult, analyze_layout, to_chunks

_SAMPLE_MARKDOWN = """\
# SAP AMS Service Level Agreement & Operational Runbook

Customer: Global Manufacturing Operations | Provider: NTT DATA AMS

## 1. Service Scope and Operational Objectives

This document establishes the binding Service Level Agreements (SLAs).

## 2. Incident Priority & Response SLA Matrix

All incoming support tickets must be triaged according to the matrix below.

| Priority | Target Response | Target Resolution |
| --- | --- | --- |
| P1 | 15 minutes | 2 hours |
| P2 | 30 minutes | 4 hours |

## 3. Emergency Transport Procedures (STMS)

An Emergency Change Request must be approved by the Basis Lead.
"""


class TestHeaderChunking:
    def test_one_chunk_per_section(self):
        layout = LayoutResult(title="T", markdown_content=_SAMPLE_MARKDOWN)
        chunks = to_chunks(layout)
        # Title+preamble, scope, SLA matrix, STMS = 4 chunks
        assert len(chunks) == 4
        assert [c.chunk_id for c in chunks] == [
            "chunk-000",
            "chunk-001",
            "chunk-002",
            "chunk-003",
        ]

    def test_heading_extracted_and_attached(self):
        layout = LayoutResult(title="T", markdown_content=_SAMPLE_MARKDOWN)
        chunks = to_chunks(layout)
        sla = chunks[2]
        assert sla.heading == "2. Incident Priority & Response SLA Matrix"
        assert sla.content.startswith("## 2. Incident Priority")
        assert "| P1 | 15 minutes | 2 hours |" in sla.content

    def test_table_section_flagged_as_table(self):
        layout = LayoutResult(title="T", markdown_content=_SAMPLE_MARKDOWN)
        chunks = to_chunks(layout)
        assert chunks[2].chunk_type == "table"
        assert all(c.chunk_type == "prose" for c in (chunks[0], chunks[1], chunks[3]))

    def test_html_table_section_flagged_as_table(self):
        # Azure falls back to inline HTML <table> for complex grids.
        md = "## SLA\n\nIntro.\n\n<table>\n<tr><td>P1</td><td>2 hours</td></tr>\n</table>\n"
        chunks = to_chunks(LayoutResult(title=None, markdown_content=md))
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "table"

    def test_preamble_without_heading(self):
        layout = LayoutResult(title=None, markdown_content="Intro text\n\n# Sec\n\nBody")
        chunks = to_chunks(layout)
        assert chunks[0].heading is None
        assert chunks[0].content == "Intro text"

    def test_empty_markdown_yields_no_chunks(self):
        assert to_chunks(LayoutResult(title=None, markdown_content="")) == []


class TestAnalyzeLayoutMocked:
    def test_requests_markdown_and_extracts_title(self, tmp_path: Path):
        test_file = tmp_path / "dummy.pdf"
        test_file.write_bytes(b"%PDF-1.4 dummy")

        mock_result = MagicMock()
        mock_result.content = _SAMPLE_MARKDOWN
        mock_result.paragraphs = [
            MagicMock(role="title", content="SAP AMS Service Level Agreement"),
            MagicMock(role="sectionHeading", content="1. Service Scope"),
        ]

        mock_poller = MagicMock()
        mock_poller.result.return_value = mock_result
        mock_client = MagicMock()
        mock_client.begin_analyze_document.return_value = mock_poller

        layout = analyze_layout(test_file, client=mock_client)

        kwargs = mock_client.begin_analyze_document.call_args.kwargs
        assert kwargs["output_content_format"] == DocumentContentFormat.MARKDOWN
        assert kwargs["model_id"] == "prebuilt-layout"
        assert layout.title == "SAP AMS Service Level Agreement"
        assert "| P1 | 15 minutes | 2 hours |" in layout.markdown_content

    def test_missing_credentials_raises_runtime_error(self, tmp_path: Path):
        test_file = tmp_path / "dummy.pdf"
        test_file.write_bytes(b"%PDF-1.4 dummy")

        mock_settings = MagicMock()
        mock_settings.azure_docintel_endpoint = None
        mock_settings.azure_docintel_key = None

        with pytest.raises(RuntimeError, match="AZURE_DOCINTEL_ENDPOINT"):
            analyze_layout(test_file, settings=mock_settings)
