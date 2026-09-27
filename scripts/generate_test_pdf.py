"""Generate a realistic SAP AMS SLA & Operational Runbook PDF for Document Intelligence testing.

Usage:
  uv run --with reportlab python scripts/generate_test_pdf.py
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_sla_pdf(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0B2545"),
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#134074"),
        spaceBefore=12,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1D2D44"),
        spaceAfter=8,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#0B2545"),
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("SAP AMS Service Level Agreement & Operational Runbook", title_style))
    story.append(
        Paragraph(
            "<b>Customer:</b> Global Manufacturing Operations &nbsp;|&nbsp; "
            "<b>Service Provider:</b> NTT DATA AMS &nbsp;|&nbsp; "
            "<b>Version:</b> 2.4 (Effective: 2026-2027)",
            body_style,
        )
    )
    story.append(Spacer(1, 10))

    # Section 1
    story.append(Paragraph("1. Service Scope and Operational Objectives", h1_style))
    story.append(
        Paragraph(
            "This document establishes the binding Service Level Agreements (SLAs), "
            "classification rubrics, and emergency operational procedures governing the "
            "SAP Application Management Services (AMS) contract. The covered scope "
            "includes production instances of SAP S/4HANA (Client 100/200), covering "
            "functional modules FI (Financials), SD (Sales & Distribution), MM "
            "(Materials Management), as well as technical domains ABAP development "
            "and Basis system administration. The primary objective is to guarantee "
            "business continuity, rapid containment of operational stoppages, and "
            "strict adherence to ITIL-aligned restoration procedures.",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    # Section 2: SLA Table
    story.append(Paragraph("2. Incident Priority & Response SLA Matrix", h1_style))
    story.append(
        Paragraph(
            "All incoming support tickets must be triaged according to the severity "
            "matrix below. Target resolution times represent maximum elapsed time "
            "from ticket creation to technical restoration.",
            body_style,
        )
    )

    table_data = [
        [
            Paragraph("Priority", table_header_style),
            Paragraph("Severity Classification", table_header_style),
            Paragraph("Target Response", table_header_style),
            Paragraph("Target Resolution", table_header_style),
            Paragraph("Escalation Authority", table_header_style),
        ],
        [
            Paragraph("<b>P1</b>", table_cell_style),
            Paragraph("Critical Outage: Core plant or site completely halted", table_cell_style),
            Paragraph("15 minutes", table_cell_style),
            Paragraph("<b>2 hours</b>", table_cell_style),
            Paragraph("L3 On-Call Lead + Service Delivery Manager", table_cell_style),
        ],
        [
            Paragraph("<b>P2</b>", table_cell_style),
            Paragraph("Major Degradation: Entire department workflow blocked", table_cell_style),
            Paragraph("30 minutes", table_cell_style),
            Paragraph("<b>4 hours</b>", table_cell_style),
            Paragraph("L2 Specialist + Module Duty Manager", table_cell_style),
        ],
        [
            Paragraph("<b>P3</b>", table_cell_style),
            Paragraph(
                "Medium Issue: Workaround exists or non-critical task delayed",
                table_cell_style,
            ),
            Paragraph("2 hours", table_cell_style),
            Paragraph("<b>24 hours</b>", table_cell_style),
            Paragraph("Functional Lead (FI/SD/MM/Basis)", table_cell_style),
        ],
        [
            Paragraph("<b>P4</b>", table_cell_style),
            Paragraph(
                "Minor / Routine: Single user inquiry or cosmetic defect",
                table_cell_style,
            ),
            Paragraph("4 hours", table_cell_style),
            Paragraph("<b>72 hours</b>", table_cell_style),
            Paragraph("Service Desk L1 Team", table_cell_style),
        ],
    ]

    sla_table = Table(table_data, colWidths=[50, 150, 80, 85, 165])
    sla_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#134074")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#8DA9C4")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#EEF4F8")),
                ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#EEF4F8")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(sla_table)
    story.append(Spacer(1, 12))

    # Section 3
    story.append(Paragraph("3. Emergency Transport Procedures (STMS)", h1_style))
    story.append(
        Paragraph(
            "Emergency software corrections and hotfixes requiring transport release "
            "outside standard weekly maintenance windows must follow strict governance. "
            "An Emergency Change Request (CHG) must be approved in writing by the Basis "
            "Lead and the functional Module Owner. All emergency transports must be "
            "verified in QAS quality assurance prior to importing into PRD. Bypassing "
            "the standard transport route (DEV -> QAS -> PRD) is strictly prohibited "
            "under any circumstances.",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    # Section 4
    story.append(Paragraph("4. Emergency Contact Directory", h1_style))
    story.append(
        Paragraph(
            "<b>24/7 Global AMS Bridge:</b> +1 (800) 555-0199 &nbsp;|&nbsp; "
            "<b>Major Incident Commander:</b> incident-command@example.com &nbsp;|&nbsp; "
            "<b>Service Delivery Escalations:</b> ams-delivery-lead@example.com",
            body_style,
        )
    )

    doc.build(story)
    print(f"Successfully generated realistic test document: {output_path}")


if __name__ == "__main__":
    generate_sla_pdf(Path("data/ams_sla_runbook.pdf"))
