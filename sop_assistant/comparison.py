"""Requirement 05 - procedure comparison.

Both procedures are retrieved with a metadata filter on ``doc_id`` and
aligned section by section ("Sampling Plan" vs "Sampling Plan"). Every cell
contains text copied from the SOP; when a section exists in only one
document the other cell says "Not specified in <doc>" - nothing is invented.
Numeric values that differ inside aligned sections (AQL, retention years
...) are listed as key differences.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .limits import extract_facts
from .retrieval import Hit, SOPRetriever


def section_items(text: str) -> List[str]:
    """Split a section into its list items / steps (numbering and bullets removed)."""
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^(-|\d{1,2}\.\d{1,2})\s+", "", line)
        items.append(line)
    return items


def _norm_title(section_label: str) -> str:
    title = re.sub(r"^\d+\.\s*", "", section_label).lower()
    return re.sub(r"[^a-z ]", "", title).strip()


def _norm_item(item: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", item.lower()).strip()


def compare_documents(retriever: SOPRetriever, doc_a: str, doc_b: str, question: str = "") -> Dict:
    secs_a = retriever.document_sections(doc_a, question)
    secs_b = retriever.document_sections(doc_b, question)
    if not secs_a or not secs_b:
        missing = doc_a if not secs_a else doc_b
        raise ValueError(f"Document {missing} is not in the index.")

    title_a = secs_a[0].metadata.get("title", doc_a)
    title_b = secs_b[0].metadata.get("title", doc_b)
    map_b: Dict[str, Hit] = {_norm_title(h.section): h for h in secs_b}
    used_b = set()
    rows: List[Dict] = []

    def make_row(ha: Optional[Hit], hb: Optional[Hit]) -> Dict:
        items_a = section_items(ha.text) if ha else []
        items_b = section_items(hb.text) if hb else []
        if ha and hb:
            same = {_norm_item(i) for i in items_a} == {_norm_item(i) for i in items_b}
            status = "Same" if same else "Different"
        else:
            status = f"Only in {doc_a}" if ha else f"Only in {doc_b}"
        return {
            "aspect": re.sub(r"^\d+\.\s*", "", (ha or hb).section),
            "status": status,
            "a_items": items_a or [f"Not specified in {doc_a}."],
            "b_items": items_b or [f"Not specified in {doc_b}."],
            "a_citation": ha.citation() if ha else "",
            "b_citation": hb.citation() if hb else "",
            "a_node": ha.node_id if ha else None,
            "b_node": hb.node_id if hb else None,
        }

    for ha in secs_a:
        hb = map_b.get(_norm_title(ha.section))
        if hb:
            used_b.add(hb.node_id)
        rows.append(make_row(ha, hb))
    for hb in secs_b:
        if hb.node_id not in used_b:
            rows.append(make_row(None, hb))

    # key numeric differences inside aligned sections
    key_differences = []
    by_node = {h.node_id: h for h in secs_a + secs_b}
    for row in rows:
        if not (row["a_node"] and row["b_node"]):
            continue
        fa = extract_facts(by_node[row["a_node"]])
        fb = extract_facts(by_node[row["b_node"]])
        for x in fa:
            for y in fb:
                if x.attribute == y.attribute and x.unit == y.unit and float(x.value) != float(y.value):
                    label = "AQL" if x.unit == "AQL" else x.attribute
                    key_differences.append({
                        "aspect": row["aspect"], "quantity": label,
                        "a_value": x.value if x.unit == "AQL" else x.display_value,
                        "b_value": y.value if y.unit == "AQL" else y.display_value,
                        "a_citation": row["a_citation"], "b_citation": row["b_citation"],
                    })

    return {
        "doc_a": doc_a, "doc_b": doc_b, "title_a": title_a, "title_b": title_b,
        "rows": rows, "key_differences": key_differences,
        "hits": secs_a + secs_b,
    }


def comparison_markdown(comp: Dict) -> str:
    a, b = comp["doc_a"], comp["doc_b"]
    lines = [f"**Comparison: {a} ({comp['title_a']}) vs {b} ({comp['title_b']})**", ""]
    lines.append("Both procedures were retrieved and aligned section by section. Every entry is copied from the "
                 "cited SOP; sections missing from one document are marked *Not specified*.")
    lines.append("")
    if comp["key_differences"]:
        lines.append("**Key documented differences (numeric):**")
        for d in comp["key_differences"]:
            lines.append(f"- {d['aspect']} - {d['quantity']}: **{comp['title_a']}** {d['a_value']} {d['a_citation']} "
                         f"vs **{comp['title_b']}** {d['b_value']} {d['b_citation']}")
        lines.append("")
    lines.append("| Aspect | " + f"{comp['title_a']} ({a})" + " | " + f"{comp['title_b']} ({b})" + " | Status |")
    lines.append("|---|---|---|---|")
    for row in comp["rows"]:
        cell_a = "; ".join(row["a_items"]).replace("|", "/") + (f" {row['a_citation']}" if row["a_citation"] else "")
        cell_b = "; ".join(row["b_items"]).replace("|", "/") + (f" {row['b_citation']}" if row["b_citation"] else "")
        lines.append(f"| **{row['aspect']}** | {cell_a} | {cell_b} | {row['status']} |")
    return "\n".join(lines)
