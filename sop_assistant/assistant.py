"""The Manufacturing Quality SOP Assistant - orchestration layer.

    question
       |
       v
    router (rules, optional LLM) --------------------------+
       |                     |                              |
       v                     v                              v
  procedural_explanation  document_comparison   missing_information_check
  (retrieve + summarise)  (retrieve 2 docs,     (verify limit / approval rule:
                           align sections)        verified | conflict | not found)
       |                     |                              |
       +---------------------+------------------------------+
       v
    guardrail (no "I inspected / released" claims) + disclaimer
       v
    query log (JSONL)  ->  Answer (markdown + sources panel data)

The three tools are registered as LlamaIndex ``FunctionTool`` objects, i.e.
the same interface an LLM agent would use.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

from llama_index.core.tools import FunctionTool

from .comparison import compare_documents, comparison_markdown
from .config import AppConfig, load_config
from .embeddings import get_embed_model
from .generation import (compose_limit_answer, compose_not_found, compose_procedure, compose_release_refusal)
from .guardrails import sanitize
from .ingest import build_or_load_index
from .limits import check_limits, find_cross_document_conflicts, unknown_terms
from .llm import get_llm
from .models import (ACTION_REFUSED, CONFLICT, DISCLAIMER, ERROR, GROUNDED, NOT_VERIFIABLE, Answer, RouteDecision)
from .monitoring import QueryLogger
from .retrieval import Hit, SOPRetriever
from .router import route
from .text_utils import FILLER_WORDS, GENERIC_DOMAIN_WORDS, STOPWORDS, stem_set, tokenize
from .vocabulary import detect_defects, mentioned_documents

log = logging.getLogger(__name__)


class SOPAssistant:
    def __init__(self, config: Optional[AppConfig] = None, rebuild: bool = False):
        self.cfg = config or load_config()
        self.embed_model, self.embed_name = get_embed_model(self.cfg.embed_backend, self.cfg.hf_model)
        self.index, nodes, self.index_info = build_or_load_index(
            self.cfg.data_dir, self.cfg.storage_dir, self.embed_model, self.embed_name,
            persist=self.cfg.persist_index, rebuild=rebuild)
        self.nodes = nodes
        self.retriever = SOPRetriever(self.index, candidate_k=self.cfg.candidate_k)
        self.llm, self.llm_name = get_llm(self.cfg.llm_backend, self.cfg.openai_model, self.cfg.ollama_model)
        self.logger = QueryLogger(self.cfg.log_file)

        # lookup tables
        self.all_hits: Dict[str, Hit] = {
            n.node_id: Hit(node_id=n.node_id, text=n.get_content(), metadata=dict(n.metadata), vector_score=0.0)
            for n in nodes}
        self.doc_titles: Dict[str, str] = {h.doc_id: h.metadata.get("title", "") for h in self.all_hits.values()}
        self.corpus_vocab = set()
        for h in self.all_hits.values():
            self.corpus_vocab |= stem_set(h.text + " " + h.metadata.get("title", "") + " " + h.section)

        # agent tools
        self.tools: Dict[str, FunctionTool] = {
            "procedural_explanation": FunctionTool.from_defaults(
                fn=self.procedural_explanation, name="procedural_explanation",
                description="Explain inspection steps, acceptable evidence or who to inform, as a numbered, "
                            "cited summary of the retrieved SOP sections."),
            "document_comparison": FunctionTool.from_defaults(
                fn=self.document_comparison, name="document_comparison",
                description="Retrieve two procedures and show their differences section by section."),
            "missing_information_check": FunctionTool.from_defaults(
                fn=self.missing_information_check, name="missing_information_check",
                description="Verify a tolerance, acceptance limit, interval or approval rule. Returns the documented "
                            "value with citations, both values when documents conflict, or a cannot-verify answer."),
        }

    # ------------------------------------------------------------------ info
    @property
    def documents(self) -> List[Dict]:
        docs: Dict[str, Dict] = {}
        for h in self.all_hits.values():
            d = docs.setdefault(h.doc_id, {"doc_id": h.doc_id, "title": h.metadata.get("title", ""),
                                           "file_name": h.metadata.get("file_name", ""),
                                           "file_type": h.metadata.get("file_type", ""),
                                           "revision": h.metadata.get("revision", ""), "sections": 0})
            d["sections"] += 1
        return sorted(docs.values(), key=lambda d: d["doc_id"])

    # ------------------------------------------------------------------ main entry point
    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        question = (question or "").strip()
        if not question:
            ans = Answer(question, "empty", "none", NOT_VERIFIABLE, "Please type a question about an SOP.")
            ans.processing_ms = (time.perf_counter() - t0) * 1000
            return ans

        decision: RouteDecision = route(question, self.llm)
        try:
            if decision.question_type == "release_action":
                ans = self.release_refusal(question)
            else:
                output = self.tools[decision.route].call(question=question)
                ans = output.raw_output
                if not isinstance(ans, Answer):
                    raise RuntimeError(f"Tool {decision.route} returned an unexpected result: {output.content}")
        except Exception as exc:  # never crash the UI / CLI
            log.exception("Error while answering")
            ans = Answer(question, decision.question_type, decision.route, ERROR,
                         f"An internal error occurred while answering: {exc}")

        ans.question_type = decision.question_type
        ans.route = decision.route
        ans.route_reason = f"[{decision.router}] {decision.reason}"
        clean, removed = sanitize(ans.answer_markdown)
        if removed:
            ans.warnings.append(f"Guardrail removed {len(removed)} unsupported claim(s).")
        ans.answer_markdown = clean + "\n\n---\n*" + DISCLAIMER + "*"
        ans.processing_ms = (time.perf_counter() - t0) * 1000
        self.logger.log(ans, self.embed_name)
        return ans

    # ------------------------------------------------------------------ tools
    def _out_of_scope(self, question: str) -> Tuple[bool, List[str]]:
        """(True, unknown_terms) if most informative words of the question are not in any SOP."""
        if detect_defects(question):
            return False, []
        named_documents = mentioned_documents(question)
        informative = {t for t in tokenize(question)
                       if t not in STOPWORDS and t not in GENERIC_DOMAIN_WORDS and t not in FILLER_WORDS
                       and not t.replace(".", "").isdigit()}
        unknown = unknown_terms(question, self.corpus_vocab)
        known = len(informative) - len(unknown)
        if not informative and named_documents:   # e.g. "How is in-process inspection done?"
            return False, []
        if known <= 0 or len(unknown) > known:
            return True, unknown
        return False, []

    def procedural_explanation(self, question: str) -> Answer:
        """Numbered, cited summary of the most relevant SOP section(s)."""
        hits = self.retriever.retrieve(question, top_k=self.cfg.top_k)
        out_of_scope, unknown = self._out_of_scope(question)
        if not hits or out_of_scope:
            return Answer(question, "procedure", "missing_information_check", NOT_VERIFIABLE,
                          compose_not_found(question, unknown, hits),
                          sources=[h.to_source("related") for h in hits[:3]])
        md, used, generator = compose_procedure(question, hits, self.all_hits, self.doc_titles,
                                                self.llm, self.llm_name)
        # consistency check: do the limits quoted in this answer contradict another document?
        warnings = []
        conflicts = find_cross_document_conflicts(used, list(self.all_hits.values()))
        if conflicts:
            md += "\n\n**Consistency check - inconsistent limits between documents (no value selected):**"
            for fa, fb in conflicts:
                md += (f"\n- {fa.doc_id} states **{fa.display_value}** (\"{fa.sentence}\") {fa.citation()} but "
                       f"{fb.doc_id} states **{fb.display_value}** (\"{fb.sentence}\") {fb.citation()}. "
                       "Resolve with the document owners / Quality Manager before relying on either value.")
                warnings.append(f"Inconsistent {fa.attribute}: {fa.doc_id}={fa.display_value} vs {fb.doc_id}={fb.display_value}")
                conflict_hit = self.all_hits.get(f"{fb.doc_id}::S{fb.section.split('.')[0]}")
                if conflict_hit and conflict_hit.node_id not in {u.node_id for u in used}:
                    used.append(conflict_hit)
        used_ids = {h.node_id for h in used}
        sources = [h.to_source("cited") for h in used]
        sources += [h.to_source("retrieved") for h in hits if h.node_id not in used_ids]
        return Answer(question, "procedure", "procedural_explanation", GROUNDED, md,
                      sources=sources, generator=generator, warnings=warnings)

    def document_comparison(self, question: str) -> Answer:
        """Compare two procedures section by section."""
        docs = [d for d in mentioned_documents(question) if d in self.doc_titles]
        if len(docs) == 1:  # only one procedure named - do not guess the second one
            options = "\n".join(f"- {d} - {t}" for d, t in sorted(self.doc_titles.items()) if d != docs[0])
            return Answer(question, "comparison", "document_comparison", NOT_VERIFIABLE,
                          f"Only one procedure was identified ({docs[0]} - {self.doc_titles[docs[0]]}). "
                          f"Which procedure should it be compared with?\n\n{options}")
        if len(docs) < 2:  # nothing named: fall back to the two best matching documents
            for h in self.retriever.retrieve(question, top_k=10):
                if h.doc_id not in docs:
                    docs.append(h.doc_id)
                if len(docs) == 2:
                    break
        if len(docs) < 2:
            return Answer(question, "comparison", "document_comparison", NOT_VERIFIABLE,
                          "Two procedures are needed for a comparison, but only one could be identified. "
                          "Please name both procedures (e.g. 'incoming' and 'final inspection').")
        comp = compare_documents(self.retriever, docs[0], docs[1], question)
        hits = comp.pop("hits")
        return Answer(question, "comparison", "document_comparison", GROUNDED, comparison_markdown(comp),
                      sources=[h.to_source("cited") for h in hits], comparison=comp)

    def missing_information_check(self, question: str) -> Answer:
        """Verify a limit / tolerance / approval rule; never guess."""
        hits = self.retriever.retrieve(question, top_k=10)
        check = check_limits(question, hits, self.corpus_vocab)
        status = {"verified": GROUNDED, "conflict": CONFLICT, "not_found": NOT_VERIFIABLE}[check.status]
        md = compose_limit_answer(question, check, hits)
        if check.status == "not_found":
            sources = [h.to_source("related") for h in hits[:3]]
        else:
            fact_keys = {(f.doc_id, f.section) for f in check.facts}
            sources = [h.to_source("cited") for h in hits if (h.doc_id, h.section) in fact_keys]
            sources += [h.to_source("retrieved") for h in hits[:5] if (h.doc_id, h.section) not in fact_keys]
        ans = Answer(question, "limit_or_approval", "missing_information_check", status, md, sources=sources,
                     limit_facts=[f.to_dict() for f in check.facts])
        if check.status == "conflict":
            ans.warnings.append(check.reason)
        return ans

    def release_refusal(self, question: str) -> Answer:
        """Guardrail: the assistant never approves/releases; it points to the documented release step."""
        release_hits = self.retriever.document_sections("SOP-REL-010", question)
        md = compose_release_refusal(release_hits)
        cited = [h for h in release_hits if any(k in h.section.lower() for k in ("authority", "prerequisite", "release steps"))]
        return Answer(question, "release_action", "procedural_explanation", ACTION_REFUSED, md,
                      sources=[h.to_source("cited") for h in cited])
