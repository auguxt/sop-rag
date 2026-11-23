"""Domain vocabulary: defect synonyms and document aliases.

Requirement 03 (defect-specific retrieval): shop-floor users describe the
same defect with many different words ("scuff", "rubbing mark", "oversize",
"no sticker" ...). We map such wording to four canonical defect categories
and *expand* the search query with the vocabulary the SOPs actually use.
The canonical defect code is also used to boost passages that mention it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class DefectCategory:
    key: str
    code: str
    label: str
    patterns: tuple
    expansion: str


DEFECT_CATEGORIES: List[DefectCategory] = [
    DefectCategory(
        key="scratch",
        code="SC-01",
        label="Scratch / surface mark",
        patterns=(
            r"\bscratch\w*", r"\bscuff\w*", r"\bscrap(e|ed|es|ing)\b", r"\babrasion\w*",
            r"\bscor(e|ing) marks?\b", r"\bline marks?\b", r"\bsurface marks?\b",
            r"\brub(bing)? marks?\b", r"\bmarks? on (the )?(part|surface|cover|face)\b",
        ),
        expansion="scratch scuff abrasion line mark surface cosmetic defect SC-01 Class A surface",
    ),
    DefectCategory(
        key="dimensional",
        code="DM-02",
        label="Dimensional mismatch",
        patterns=(
            r"\bdimensional\w*", r"\bout of tolerance\b", r"\bover-?size\w*", r"\bunder-?size\w*",
            r"\bwrong (size|dimension\w*|diameter|length)\b", r"\btoo (big|small|large|long|short|thick|thin)\b",
            r"\bout of spec\w*\b", r"\b(does ?n[o']?t|not|won'?t) fit\b",
            r"\bdimensions? (is |are )?(off|wrong|out|not ok)\b", r"\bmeasur\w* (is |are )?(off|wrong|out)\b",
        ),
        expansion="dimensional mismatch dimension out of tolerance oversize undersize DM-02 stop the machine",
    ),
    DefectCategory(
        key="label",
        code="LB-03",
        label="Missing / incorrect label",
        patterns=(
            r"(?<!calibration )\blabel\w*", r"\bsticker\w*", r"\bunlabel+ed\b", r"\bno tag\b",
            r"\btag (is )?missing\b", r"\bbarcode\w*", r"\bidentification\b", r"\bunidentified\b",
        ),
        expansion="missing label incorrect label LB-03 label verification traceability package HOLD",
    ),
    DefectCategory(
        key="calibration",
        code="CF-04",
        label="Failed calibration",
        patterns=(
            r"\bfail\w*\s+(\w+\s+){0,3}calibrat\w*", r"\bcalibrat\w*\s+(\w+\s+){0,3}fail\w*",
            r"\bout of cal(ibration)?\b", r"\b(calibration|cal)\s+(is\s+|has\s+)?(overdue|expired|lapsed)\b",
            r"\boverdue\b", r"\b(gauge|gage|instrument|caliper|micrometer)s?\s+(has\s+|have\s+)?failed\b",
            r"\bfailed (gauge|gage|instrument)\b",
        ),
        expansion="failed calibration out of calibration CF-04 instrument impact assessment DO NOT USE tag",
    ),
]


# Additional defect types from the defect code reference (not required, but cheap to support)
DEFECT_CATEGORIES += [
    DefectCategory("dent", "DN-05", "Dent", (r"\bdent\w*", r"\bding\w*", r"\bimpact marks?\b"),
                   "dent deformation impact DN-05 Class A surface"),
    DefectCategory("corrosion", "CR-06", "Corrosion", (r"\brust\w*", r"\bcorro\w*", r"\boxidi[sz]\w*", r"\boxidation\b"),
                   "corrosion rust oxidation CR-06 visual inspection"),
    DefectCategory("burr", "BR-07", "Burr", (r"\bburrs?\b", r"\bsharp edges?\b"),
                   "burr sharp edge machining BR-07"),
]


def detect_defects(text: str) -> List[DefectCategory]:
    """Return the defect categories mentioned (in any wording) in `text`."""
    low = text.lower()
    found = []
    for cat in DEFECT_CATEGORIES:
        if any(re.search(p, low) for p in cat.patterns):
            found.append(cat)
    # "failed calibration" questions often also contain the word "label"
    # (calibration label) - a calibration problem is not a labelling problem.
    keys = {c.key for c in found}
    if "calibration" in keys and "label" in keys and not re.search(r"\b(missing|no|wrong|incorrect)\s+label", low):
        found = [c for c in found if c.key != "label"]
    return found


def expand_query(question: str) -> str:
    """Append canonical SOP vocabulary for every recognised defect category."""
    cats = detect_defects(question)
    if not cats:
        return question
    return question + " | " + " ".join(c.expansion for c in cats)


# --------------------------------------------------------------------------
# Document aliases - used by the comparison tool to find which procedures the
# user is talking about ("incoming" -> SOP-QA-001 ...). The order matters
# only for display.
# --------------------------------------------------------------------------
DOC_ALIASES: Dict[str, tuple] = {
    "SOP-QA-001": (r"\bincoming\b", r"\breceiving\b", r"\breceipt\b", r"\bgoods[- ]in\b"),
    "SOP-QA-002": (r"\bfinal\b", r"\bpre-?dispatch\b", r"\boutgoing\b"),
    "SOP-QA-003": (r"\bin-?process\b", r"\bpatrol\b", r"\bfirst[- ]piece\b"),
    "SOP-CAL-004": (r"\bcalibrat\w*\b",),
    "REF-DEF-005": (r"\bdefect codes?\b", r"\bdefect reference\b"),
    "SOP-NCR-006": (r"\bncr\b", r"\bnon-?conformance\b", r"\bnonconform\w*\b", r"\bmrb\b"),
    "CHK-MC-007": (r"\bcnc\b", r"\bmachine checklist\b", r"\bchecklist\b"),
    "SOP-LBL-008": (r"\blabel+ing\b", r"\btraceability\b"),
    "SOP-SURF-009": (r"\bsurface (finish|inspection)\b", r"\bcosmetic inspection\b", r"\bvisual inspection\b"),
    "SOP-REL-010": (r"\brelease\b", r"\bquarantine\b"),
}


def mentioned_documents(text: str) -> List[str]:
    """Document IDs referred to in `text`, by explicit ID or by alias, in order of appearance."""
    low = text.lower()
    hits = []
    for doc_id, patterns in DOC_ALIASES.items():
        positions = [m.start() for p in patterns for m in [re.search(p, low)] if m]
        if doc_id.lower() in low:
            positions.append(low.index(doc_id.lower()))
        if positions:
            hits.append((min(positions), doc_id))
    return [d for _, d in sorted(hits)]
