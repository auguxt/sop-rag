"""
Builds the sample SOP collection used by the assistant.

The editable source of every document lives in data/source_text/*.txt.
This script writes the document collection that is actually ingested into
data/sops/:

  * 5 documents are rendered as PDF files (multi-page, so page references
    can be demonstrated)
  * 5 documents are copied as plain-text files

Run:   python scripts/build_sample_documents.py
Needs: reportlab   (pip install reportlab) - only for (re)building the PDFs.
The ready-made PDFs are already shipped in data/sops/, so this script is
optional.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "source_text"
OUT = ROOT / "data" / "sops"

# Documents that are delivered as PDF; the rest stay as .txt files.
PDF_DOCS = {
    "SOP-QA-001",
    "SOP-QA-002",
    "SOP-CAL-004",
    "SOP-NCR-006",
    "SOP-REL-010",
}


def render_pdf(txt_path: Path, pdf_path: Path) -> None:
    import re

    from reportlab.lib.pagesizes import A5
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.5, leading=12)
    head = ParagraphStyle("head", parent=styles["Heading3"], fontSize=11, spaceBefore=6, spaceAfter=3)
    meta = ParagraphStyle("meta", parent=body, fontSize=8.5, textColor="#444444")

    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    story = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip()
        if not line:
            story.append(Spacer(1, 3))
        elif re.match(r"^\d{1,2}\.\s+[A-Z]", line):
            story.append(Paragraph(esc(line), head))
        elif re.match(r"^[A-Za-z ]+:\s", line) and len(story) < 12:
            story.append(Paragraph(esc(line), meta))
        else:
            story.append(Paragraph(esc(line), body))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.drawString(12 * mm, 8 * mm, f"{txt_path.stem.split('_')[0]}  -  Controlled copy")
        canvas.drawRightString(A5[0] - 12 * mm, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A5,
        leftMargin=12 * mm, rightMargin=12 * mm, topMargin=12 * mm, bottomMargin=15 * mm,
        title=txt_path.stem,
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):
        old.unlink()
    for txt in sorted(SRC.glob("*.txt")):
        doc_id = txt.stem.split("_")[0]
        if doc_id in PDF_DOCS:
            target = OUT / (txt.stem + ".pdf")
            render_pdf(txt, target)
        else:
            target = OUT / txt.name
            shutil.copyfile(txt, target)
        print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
