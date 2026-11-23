"""Optional LLM support.

The assistant works completely WITHOUT an LLM (extractive, fully grounded
summaries). If you want more natural wording you can plug in an LLM:

    SOP_LLM_BACKEND=openai   (pip install llama-index-llms-openai, set OPENAI_API_KEY)
    SOP_LLM_BACKEND=ollama   (pip install llama-index-llms-ollama, run `ollama serve`)

Any problem (package missing, no key, server down) -> ``None`` is returned and
the extractive generator is used, so the program never crashes because of the LLM.
LLM output is validated (every step must cite a source and overlap with it);
invalid output is discarded in favour of the extractive answer.
"""
from __future__ import annotations

import logging
import os
import re
from typing import List, Optional, Tuple

from .retrieval import Hit
from .text_utils import token_overlap

log = logging.getLogger(__name__)


def get_llm(backend: str, openai_model: str = "gpt-4o-mini", ollama_model: str = "llama3.1"):
    """Return ``(llm_or_None, name)``."""
    backend = (backend or "none").lower().strip()
    if backend in ("", "none", "off", "false"):
        return None, "none"
    try:
        if backend == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            from llama_index.llms.openai import OpenAI

            llm = OpenAI(model=openai_model, temperature=0)
            return llm, f"openai:{openai_model}"
        if backend == "ollama":
            from llama_index.llms.ollama import Ollama

            llm = Ollama(model=ollama_model, request_timeout=120.0, temperature=0)
            llm.complete("Reply with OK.")  # connectivity check
            return llm, f"ollama:{ollama_model}"
        raise ValueError(f"unknown LLM backend {backend!r}")
    except Exception as exc:
        log.warning("LLM backend %r unavailable (%s) - using extractive answers.", backend, exc)
        return None, "none"


PROMPT = """You are a manufacturing quality SOP assistant. Answer the question using ONLY the numbered
sources below. Write a short numbered list of steps. End EVERY line with the source tag it came from,
e.g. [S1]. Do not add steps, values or tolerances that are not in the sources. Never say that you
inspected a part or approved/released a batch.

{sources}

Question: {question}
Numbered answer:"""


def llm_numbered_summary(llm, question: str, hits: List[Hit]) -> Optional[List[Tuple[str, Hit]]]:
    """Ask the LLM for a cited numbered summary. Returns ``[(step_text, hit), ...]`` or None if invalid."""
    sources = "\n\n".join(f"[S{i}] ({h.doc_id} {h.section})\n{h.text}" for i, h in enumerate(hits, start=1))
    try:
        raw = llm.complete(PROMPT.format(sources=sources, question=question)).text
    except Exception as exc:
        log.warning("LLM call failed (%s)", exc)
        return None
    steps: List[Tuple[str, Hit]] = []
    for line in raw.splitlines():
        m = re.match(r"^\s*\d+[.)]\s+(.*)$", line)
        if not m:
            continue
        body = m.group(1).strip()
        tags = [int(t) for t in re.findall(r"\[S(\d+)\]", body)]
        if not tags or any(t < 1 or t > len(hits) for t in tags):
            return None                       # uncited / invalid citation -> reject the whole answer
        text = re.sub(r"\s*\[S\d+\]", "", body).strip()
        hit = hits[tags[0] - 1]
        if token_overlap(text, hit.text) < 0.5:
            return None                       # not supported by the cited passage
        steps.append((text, hit))
    return steps or None
