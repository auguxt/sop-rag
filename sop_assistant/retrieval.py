"""Semantic retrieval + transparent re-ranking (requirements 02, 03, 04).

1. The question is expanded with defect vocabulary (vocabulary.expand_query).
2. The LlamaIndex vector retriever returns the `candidate_k` most similar
   section nodes (cosine similarity of embeddings).
3. Candidates are re-ranked with small, explainable boosts:
      * keyword overlap between question and passage
      * passage mentions the canonical defect code / defect word
      * section intent ("evidence" -> Acceptable Evidence, "who/inform" ->
        Escalation / Notification, "steps/how" -> Steps / Procedure)
      * the passage belongs to a document the user named ("final", "incoming")
Every hit keeps both the raw vector similarity and the final score so the UI
can show *why* a passage was chosen.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from .text_utils import content_stems, stem_set, tokenize
from .vocabulary import detect_defects, expand_query, mentioned_documents

SECTION_INTENTS = [
    # (question pattern, section-title pattern, boost)
    (r"\b(evidence|proof|record\w*|document\w* (needed|required)|show that)\b", r"evidence|records", 0.35),
    (r"\b(who|inform|notify|notif\w*|escalat\w*|report (it|to)|tell|contact)\b", r"escalation|notification|responsib|handling", 0.35),
    (r"\b(steps?|how (do|to|should)|procedure|process|what (do|should) i do|what now|next|handle|handling)\b",
     r"steps|procedure", 0.3),
    (r"\b(steps?|how (do|to|should)|procedure|process|what (do|should) i do|what now|next|handle|handling|check)\b",
     r"handling|workflow|missing|checks", 0.2),
    (r"\b(sampl\w*|aql|how many parts)\b", r"sampling|frequency", 0.35),
    (r"\b(interval|how often|frequency|due)\b", r"interval|frequency|gauge", 0.3),
    (r"\b(limit|tolerance|acceptance|accept\w*|criteria|maximum|minimum|allowed|lux|lighting)\b",
     r"limits|criteria|conditions|codes", 0.25),
    (r"\b(approv\w*|authori[sz]\w*|release|sign[- ]?off)\b", r"authority|release|responsib", 0.3),
    (r"\b(defect code|code|classif\w*)\b", r"codes|classification", 0.3),
]


@dataclass
class Hit:
    node_id: str
    text: str
    metadata: Dict
    vector_score: float
    score: float = 0.0
    boosts: Dict[str, float] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.metadata.get("doc_id", "")

    @property
    def section(self) -> str:
        return self.metadata.get("section", "")

    @property
    def page(self) -> str:
        return str(self.metadata.get("page_label", "n/a"))

    def citation(self) -> str:
        page = f", p.{self.page}" if self.page not in ("n/a", "None", "") else ""
        return f"[{self.doc_id} §{self.section}{page}]"

    def to_source(self, role: str = "cited") -> Dict:
        """role: cited (used in the answer) | retrieved (shown for context) | related (closest, not supporting)"""
        return {
            "node_id": self.node_id,
            "doc_id": self.doc_id,
            "title": self.metadata.get("title", ""),
            "section": self.section,
            "page": self.page,
            "file_name": self.metadata.get("file_name", ""),
            "vector_score": round(self.vector_score, 4),
            "score": round(self.score, 4),
            "boosts": {k: round(v, 3) for k, v in self.boosts.items()},
            "text": self.text,
            "role": role,
        }


class SOPRetriever:
    def __init__(self, index, candidate_k: int = 15):
        self.index = index
        self.candidate_k = candidate_k

    # ------------------------------------------------------------------
    def _vector_search(self, query: str, k: int, doc_id: Optional[str] = None) -> List[Hit]:
        kwargs = {"similarity_top_k": k}
        if doc_id:
            kwargs["filters"] = MetadataFilters(filters=[ExactMatchFilter(key="doc_id", value=doc_id)])
        retriever = self.index.as_retriever(**kwargs)
        results = retriever.retrieve(query)
        return [Hit(node_id=r.node.node_id, text=r.node.get_content(), metadata=dict(r.node.metadata),
                    vector_score=float(r.score or 0.0)) for r in results]

    def retrieve(self, question: str, top_k: int = 5, doc_id: Optional[str] = None) -> List[Hit]:
        """Expanded semantic search + re-ranking. Returns the best `top_k` hits."""
        query = expand_query(question)
        if not tokenize(query):
            return []
        hits = self._vector_search(query, max(self.candidate_k, top_k), doc_id=doc_id)
        if not hits:
            return []
        self._rerank(question, hits)
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def document_sections(self, doc_id: str, question: str = "") -> List[Hit]:
        """All sections of one document (metadata-filtered retrieval), in section order."""
        hits = self._vector_search(question or doc_id, 100, doc_id=doc_id)
        hits.sort(key=lambda h: int(h.metadata.get("section_number", 0)))
        return hits

    # ------------------------------------------------------------------
    def _rerank(self, question: str, hits: List[Hit]) -> None:
        q_low = question.lower()
        q_stems = set(content_stems(expand_query(question)))
        defects = detect_defects(question)
        named_docs = set(mentioned_documents(question))
        intents = [(sec_pat, boost) for q_pat, sec_pat, boost in SECTION_INTENTS if re.search(q_pat, q_low)]

        scores = [h.vector_score for h in hits]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        for h in hits:
            boosts: Dict[str, float] = {}
            boosts["vector"] = (h.vector_score - lo) / span  # 0..1 within the candidate set
            p_stems = stem_set(h.text + " " + h.section)
            if q_stems:
                boosts["keyword"] = 0.6 * len(q_stems & p_stems) / len(q_stems)
            text_low = h.text.lower()
            for cat in defects:
                if cat.code.lower() in text_low or cat.key[:7] in text_low:
                    boosts["defect"] = 0.3
            sec_low = h.section.lower()
            intent_boost = min(0.6, sum(b for pat, b in intents if re.search(pat, sec_low)))
            if intent_boost:
                boosts["section_intent"] = intent_boost
            if h.doc_id in named_docs:
                boosts["named_document"] = 0.3
            if re.search(r"purpose|scope", sec_low):
                boosts["generic_section_penalty"] = -0.4
            h.boosts = boosts
            h.score = sum(boosts.values())
