"""Requirement 06 - safety and uncertainty handling.

For questions that ask for a tolerance, acceptance limit, interval or an
approval rule, the assistant must never guess. This module

* extracts *numeric limit statements* ("maximum length 5 mm", "every 6 months",
  "minimum 1000 lux") from the retrieved passages,
* keeps only the statements about the subject of the question,
* detects inconsistent values for the same quantity across documents,
* detects when the requested item is simply not documented.

Outcomes: ``verified`` | ``conflict`` | ``not_found``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .retrieval import Hit
from .text_utils import FILLER_WORDS, GENERIC_DOMAIN_WORDS, STOPWORDS, split_sentences, stem, tokenize
from .vocabulary import DEFECT_CATEGORIES, detect_defects, mentioned_documents

STAGE_DOCS = {"SOP-QA-001", "SOP-QA-002", "SOP-QA-003"}  # incoming / final / in-process

_UNIT = r"(working days|mm|µm|um|microns?|lux|months?|years?|weeks?|days?|hours?|minutes?|seconds?|percent|%)"
NUM_UNIT_RE = re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>" + _UNIT + r")(?![a-z])", re.I)
AQL_RE = re.compile(r"\bAQL\s+(?P<num>\d+(?:\.\d+)?)", re.I)
COUNT_RE = re.compile(r"\b(?:no more than|not more than|maximum of|minimum of|up to|at most)\s+(?P<num>\d+)\s+(?P<noun>[a-z]+)", re.I)
APPROVAL_RE = re.compile(
    r"\b(may|can|cannot|shall|must|to)\s+(approve|release|authori[sz]e)\b|\bapproves\b|\bauthori[sz](es|ed)\b|"
    r"\bsigns?\s+(off|the release)\b|\bsign-?off from\b|\brelease authority\b", re.I)

# words describing *which* quantity is meant
_ATTRIBUTE_RULES = [
    ("length", r"\blength\b|\blong\b"),
    ("width", r"\bwidth\b|\bwide\b"),
    ("depth", r"\bdepth\b|\bdeep\b"),
    ("distance", r"\bdistance\b"),
    ("retention", r"\bretain\w*|\bretention\b|\bkept\b|\bstored?\b"),
    ("interval", r"\bevery\b|\binterval\b|\bcalibrat\w*|\bdue\b"),
    ("deadline", r"\bwithin\b|\bbefore\b|\bno later\b"),
    ("duration", r"\bat least\b|\bstabilis\w*|\bfor about\b"),
]
_QUERY_ATTRIBUTE_RULES = [
    ("count", r"\bhow many\b(?!\s+(hours|days|months|years|minutes|weeks|mm|lux))"),
    ("illumination", r"\blux\b|\blight\w*|\billumin\w*"),
    ("retention", r"\bretain\w*|\bretention\b|\bkeep\b|\bkept\b|\bstored?\b|\barchiv\w*"),
    ("interval", r"\bhow often\b|\binterval\b|\bfrequen\w*|\bevery\b|\bdue\b"),
    ("deadline", r"\bwithin\b|\bhow soon\b|\bdeadline\b|\btime limit\b|\bhow quickly\b|\bhow many (hours|days)\b"),
    ("aql", r"\baql\b|\bsampl\w*"),
    ("length", r"\blength\b|\bhow long is\b"),
    ("width", r"\bwidth\b|\bhow wide\b"),
    ("depth", r"\bdepth\b|\bhow deep\b"),
    ("distance", r"\bdistance\b"),
]
ATTRIBUTE_WORDS = {"length", "width", "depth", "distance", "lux", "light", "lighting", "illumination",
                   "interval", "retain", "retained", "retention", "aql", "deadline"}


@dataclass
class LimitFact:
    doc_id: str
    title: str
    section: str
    page: str
    sentence: str
    attribute: str
    value: str
    unit: str
    match: int = 0
    unit_text: str = ""
    revision: str = ""
    effective_date: str = ""
    owner: str = ""

    def citation(self) -> str:
        page = f", p.{self.page}" if self.page not in ("n/a", "None", "") else ""
        return f"[{self.doc_id} §{self.section}{page}]"

    @property
    def display_value(self) -> str:
        return f"{self.value} {self.unit_text or self.unit}".strip()

    def to_dict(self) -> Dict:
        return dict(self.__dict__, citation=self.citation())


@dataclass
class LimitCheck:
    status: str                       # verified | conflict | not_found
    kind: str                         # numeric | approval
    attribute: Optional[str]
    facts: List[LimitFact] = field(default_factory=list)
    unknown_terms: List[str] = field(default_factory=list)
    subject_terms: List[str] = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
def _norm_unit(unit: str) -> str:
    u = unit.lower()
    if u in ("%", "percent"):
        return "percent"
    if u in ("um", "µm", "micron", "microns"):
        return "µm"
    return u.rstrip("s") if u not in ("lux", "working days") else u


def _attribute_for(sentence: str, start: int, unit: str) -> str:
    if unit == "lux":
        return "illumination"
    clause_start = max(sentence.rfind(",", 0, start), sentence.rfind(";", 0, start), sentence.rfind(":", 0, start))
    window = sentence[clause_start + 1: start].lower()
    for name, pattern in _ATTRIBUTE_RULES:
        if re.search(pattern, window):
            return name
    # nothing in the clause - look at the whole sentence before the number
    window = sentence[:start].lower()
    for name, pattern in _ATTRIBUTE_RULES:
        if re.search(pattern, window):
            return name
    return "value"


def extract_facts(hit: Hit) -> List[LimitFact]:
    facts: List[LimitFact] = []
    meta = hit.metadata
    for sent in split_sentences(hit.text):
        for m in NUM_UNIT_RE.finditer(sent):
            unit = _norm_unit(m.group("unit"))
            facts.append(LimitFact(
                doc_id=hit.doc_id, title=meta.get("title", ""), section=hit.section, page=hit.page,
                sentence=sent, attribute=_attribute_for(sent, m.start(), unit), value=m.group("num"), unit=unit,
                unit_text=m.group("unit"),
                revision=meta.get("revision", ""), effective_date=meta.get("effective_date", ""),
                owner=meta.get("owner", "")))
        for m in COUNT_RE.finditer(sent):
            noun = m.group("noun").lower()
            if noun in ("mm", "lux", "percent") or NUM_UNIT_RE.match(m.group(0)[m.start("num") - m.start():]):
                continue
            facts.append(LimitFact(
                doc_id=hit.doc_id, title=meta.get("title", ""), section=hit.section, page=hit.page,
                sentence=sent, attribute="count", value=m.group("num"), unit=stem(noun), unit_text=noun,
                revision=meta.get("revision", ""), effective_date=meta.get("effective_date", ""),
                owner=meta.get("owner", "")))
        for m in AQL_RE.finditer(sent):
            facts.append(LimitFact(
                doc_id=hit.doc_id, title=meta.get("title", ""), section=hit.section, page=hit.page,
                sentence=sent, attribute="aql", value=m.group("num"), unit="AQL", unit_text="(AQL)",
                revision=meta.get("revision", ""), effective_date=meta.get("effective_date", ""),
                owner=meta.get("owner", "")))
    return facts


def query_attribute(question: str) -> Optional[str]:
    low = question.lower()
    if re.search(r"\bhow long\b", low) and re.search(r"record|retain|keep|kept|form|report|certificate", low):
        return "retention"
    for name, pattern in _QUERY_ATTRIBUTE_RULES:
        if re.search(pattern, low):
            return name
    return None


def _defect_words(question: str) -> Set[str]:
    """Stems of the words that were recognised as a defect synonym."""
    low = question.lower()
    words: Set[str] = set()
    for cat in DEFECT_CATEGORIES:
        for p in cat.patterns:
            for m in re.finditer(p, low):
                words |= {stem(t) for t in tokenize(m.group(0))}
    return words


def subject_terms(question: str) -> List[str]:
    """Informative stems of the question (attribute words such as 'length' included)."""
    out = []
    for tok in tokenize(question):
        if tok in STOPWORDS:
            continue
        if (tok in GENERIC_DOMAIN_WORDS and tok not in ATTRIBUTE_WORDS) or tok in FILLER_WORDS:
            continue
        if re.fullmatch(r"\d+(\.\d+)?", tok):
            continue
        out.append(stem(tok))
    for cat in detect_defects(question):
        out.append(stem(cat.key))
    return list(dict.fromkeys(out))


def unknown_terms(question: str, corpus_vocab: Set[str]) -> List[str]:
    """Question words that occur in *no* indexed document (e.g. 'torque', 'flatness')."""
    defect_words = _defect_words(question)
    unknown = []
    for tok in tokenize(question):
        if tok in STOPWORDS or tok in GENERIC_DOMAIN_WORDS or tok in FILLER_WORDS or re.fullmatch(r"\d+(\.\d+)?", tok):
            continue
        s = stem(tok)
        if s in defect_words or s in corpus_vocab:
            continue
        unknown.append(tok)
    return list(dict.fromkeys(unknown))


def _score(sentence: str, subject: Set[str]) -> int:
    sent_stems = {stem(t) for t in tokenize(sentence)}
    return len(subject & sent_stems)


def check_limits(question: str, hits: List[Hit], corpus_vocab: Set[str]) -> LimitCheck:
    approval = bool(re.search(r"\b(approv\w*|authori[sz]\w*|sign[- ]?off|concession|who can release|who may release)\b",
                              question.lower()))
    attribute = None if approval else query_attribute(question)
    subject = subject_terms(question)
    unknown = unknown_terms(question, corpus_vocab)
    kind = "approval" if approval else "numeric"

    if unknown:
        return LimitCheck(status="not_found", kind=kind, attribute=attribute, unknown_terms=unknown,
                          subject_terms=subject,
                          reason=f"The term(s) {', '.join(repr(u) for u in unknown)} do not appear in any indexed document.")

    subj = set(subject)
    must_have = {stem(c.key) for c in detect_defects(question)}

    if approval:
        candidates: List[LimitFact] = []
        for h in hits:
            if re.search(r"purpose|scope", h.section.lower()):
                continue
            for sent in split_sentences(h.text):
                if APPROVAL_RE.search(sent):
                    candidates.append(LimitFact(doc_id=h.doc_id, title=h.metadata.get("title", ""), section=h.section,
                                                page=h.page, sentence=sent, attribute="approval", value="", unit="",
                                                match=_score(sent, subj)))
    else:
        candidates = [f for h in hits for f in extract_facts(h)]
        for f in candidates:
            f.match = _score(f.sentence, subj)
        if attribute:
            candidates = [f for f in candidates if f.attribute == attribute]

    if must_have:
        candidates = [f for f in candidates if must_have & {stem(t) for t in tokenize(f.sentence)}]
    candidates = [f for f in candidates if f.match > 0]
    # If the user explicitly names an inspection stage / document ("final inspection records"),
    # prefer statements from that document - but only if it actually contains one.
    stage_docs = {d for d in mentioned_documents(question) if d in STAGE_DOCS or d.lower() in question.lower()}
    if stage_docs and any(f.doc_id in stage_docs for f in candidates):
        candidates = [f for f in candidates if f.doc_id in stage_docs]
    if not candidates:
        what = "approval rule" if approval else ("limit / tolerance" + (f" ({attribute})" if attribute else ""))
        return LimitCheck(status="not_found", kind=kind, attribute=attribute, subject_terms=subject,
                          reason=f"No retrieved passage states a {what} for this subject.")

    best = max(f.match for f in candidates)
    # keep the statements that match the subject best (one word of slack for longer subjects)
    keep = [f for f in candidates if f.match >= (best - 1 if best >= 3 else best)]
    # de-duplicate identical statements
    seen, facts = set(), []
    for f in sorted(keep, key=lambda f: (-f.match, f.doc_id)):
        key = (f.doc_id, f.section, f.sentence, f.value, f.unit)
        if key not in seen:
            seen.add(key)
            facts.append(f)

    if not approval:
        # conflict = same quantity (attribute + unit) with different values in different documents
        groups: Dict[tuple, List[LimitFact]] = {}
        for f in facts:
            groups.setdefault((f.attribute, f.unit), []).append(f)
        for (attr, unit), group in groups.items():
            values = {float(f.value) for f in group}
            docs = {f.doc_id for f in group}
            if len(values) > 1 and len(docs) > 1:
                conflicting = group
                return LimitCheck(status="conflict", kind=kind, attribute=attr, facts=conflicting,
                                  subject_terms=subject,
                                  reason=f"Documents state different values for the same {attr} ({unit}).")
    return LimitCheck(status="verified", kind=kind, attribute=attribute, facts=facts, subject_terms=subject)


def _jaccard_without_numbers(a: str, b: str) -> float:
    sa = {stem(t) for t in tokenize(a) if t not in STOPWORDS and not re.fullmatch(r"\d+(\.\d+)?", t)}
    sb = {stem(t) for t in tokenize(b) if t not in STOPWORDS and not re.fullmatch(r"\d+(\.\d+)?", t)}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def find_cross_document_conflicts(used: List[Hit], corpus: List[Hit], min_similarity: float = 0.5) -> List[tuple]:
    """Consistency check for any answer: every numeric statement in the passages used for the
    answer is compared with similar statements in *other* documents of the whole collection.
    Returns a list of ``(fact_in_answer, conflicting_fact)`` pairs."""
    conflicts = []
    seen = set()
    corpus_facts = [f for h in corpus for f in extract_facts(h)]
    for h in used:
        for fa in extract_facts(h):
            for fb in corpus_facts:
                if fb.doc_id == fa.doc_id or fb.attribute != fa.attribute or fb.unit != fa.unit:
                    continue
                if fa.doc_id in STAGE_DOCS and fb.doc_id in STAGE_DOCS:
                    continue  # incoming / in-process / final are different stages: different values are expected
                if float(fb.value) == float(fa.value):
                    continue
                if _jaccard_without_numbers(fa.sentence, fb.sentence) < min_similarity:
                    continue
                key = tuple(sorted([(fa.doc_id, fa.sentence), (fb.doc_id, fb.sentence)]))
                if key not in seen:
                    seen.add(key)
                    conflicts.append((fa, fb))
    return conflicts
