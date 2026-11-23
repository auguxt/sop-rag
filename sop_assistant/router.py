"""Requirement 07 - lightweight agent routing.

Every question is routed to exactly one tool:

    procedural_explanation     how-to / steps / evidence / who-to-inform questions
    document_comparison        "what is different between X and Y"
    missing_information_check  tolerance / limit / interval / approval-rule questions:
                               the value is only given if it is documented,
                               otherwise a missing-information response is produced

Requests that ask the assistant itself to approve, release or inspect
something are tagged ``release_action`` and are always refused (the assistant
only points to the documented release step).

The default router is rule based (deterministic and easy to explain). If an
LLM is configured it is asked first and the rules are the fallback.
"""
from __future__ import annotations

import re
from typing import Optional

from .models import RouteDecision
from .vocabulary import DEFECT_CATEGORIES, detect_defects, mentioned_documents

ROUTE_FOR_TYPE = {
    "procedure": "procedural_explanation",
    "comparison": "document_comparison",
    "limit_or_approval": "missing_information_check",
    "release_action": "procedural_explanation",
}

_COMPARISON = re.compile(r"\b(differen\w*|compare\w*|comparison|versus|vs\.?|contrast|distinguish)\b", re.I)
_LIMIT = re.compile(
    r"\b(tolerance\w*|limits?|maximum|minimum|max|min|allowed|allowable|permissible|threshold\w*|"
    r"how (long|often|many|much|wide|deep|soon|far|big|small|hot|cold)|interval|frequency|lux|aql|"
    r"temperature|humidity|pressure|distance|length|width|depth|diameter|roughness|flatness|torque|"
    r"acceptance (limit|criteria|value)|acceptable (length|width|depth|size|value|number)|"
    r"approv\w*|authori[sz]\w*|sign[- ]?off|concession)\b", re.I)
_HELP = re.compile(r"\b(what (should|do|must|can) (i|we) do|what now|what next|how (do|should) (i|we) (handle|proceed)|"
                   r"procedure|steps?|handle|handling)\b", re.I)
_ACTION_VERB = r"(approve|release|pass|accept|sign[- ]?off|ship|dispatch|clear|ok|okay)"
_RELEASE_ACTION = [
    re.compile(r"^\s*(please\s+)?" + _ACTION_VERB + r"\b(?!\s+(steps?|procedure|process|prerequisites?|authority|rules?|criteria)\b)", re.I),  # "Approve batch 42"
    re.compile(r"\b(can|could|would|will)\s+you\s+(please\s+)?" + _ACTION_VERB + r"\b", re.I),       # "Can you release ..."
    re.compile(r"\b(is|are)\s+(this|the|these|my|our)?\s*(batch|lot|part|parts|shipment|order)\b.*\b(ok|okay|good|fine|acceptable|approved|released|ready)\b", re.I),
    re.compile(r"\b(is|are)\s+(batch|lot)\s+[\w-]+\s+.*\b(ok|okay|good|fine|acceptable|approved|released|ready)\b", re.I),
    re.compile(r"\b(have|did)\s+you\s+(inspect|check|measure|approve|release)\w*\b", re.I),
    re.compile(r"\b(mark|declare)\s+(it|this|the batch|the lot)\s+(as\s+)?(ok|passed|accepted|released)\b", re.I),
]


def rule_based_route(question: str) -> RouteDecision:
    q = question.strip()
    docs = mentioned_documents(q)
    for pattern in _RELEASE_ACTION:
        if pattern.search(q) and re.search(r"\b(batch|lot|part|parts|shipment|order|production|it|this|b-\d+)\b", q, re.I):
            return RouteDecision("release_action", ROUTE_FOR_TYPE["release_action"],
                                 "The user asks the assistant to approve/release/inspect; this is not permitted.", docs)
    if _COMPARISON.search(q):
        return RouteDecision("comparison", ROUTE_FOR_TYPE["comparison"],
                             "Comparison wording detected (different / compare / vs).", docs)
    # Words that belong to a defect description ("out of tolerance", "oversize") are not a
    # request for a limit, and "what should I do" questions ask for the procedure.
    q_wo_defects = q
    for cat in DEFECT_CATEGORIES:
        for p in cat.patterns:
            q_wo_defects = re.sub(p, " ", q_wo_defects, flags=re.I)
    asks_for_procedure = bool(_HELP.search(q)) and bool(detect_defects(q))
    if _LIMIT.search(q_wo_defects) and not asks_for_procedure:
        return RouteDecision("limit_or_approval", ROUTE_FOR_TYPE["limit_or_approval"],
                             "Asks for a tolerance, limit, interval or approval rule - must be verified in the SOPs.", docs)
    return RouteDecision("procedure", ROUTE_FOR_TYPE["procedure"],
                         "General procedural question (steps / evidence / escalation).", docs)


_LLM_PROMPT = """You are a router for a manufacturing quality SOP assistant.
Classify the user question into exactly one label:
- procedure: asks how to do something, inspection steps, evidence, who to inform, what to do about a defect
- comparison: asks for the differences / comparison between two procedures
- limit_or_approval: asks for a tolerance, acceptance limit, numeric value, interval, or who may approve something
- release_action: asks YOU (the assistant) to approve, release, pass, ship or inspect a batch/part
Answer with the label only.

Question: {question}
Label:"""


def llm_route(question: str, llm) -> Optional[RouteDecision]:
    try:
        raw = llm.complete(_LLM_PROMPT.format(question=question)).text.strip().lower()
    except Exception:
        return None
    for label in ("release_action", "limit_or_approval", "comparison", "procedure"):
        if label in raw:
            return RouteDecision(label, ROUTE_FOR_TYPE[label], f"LLM router label: {label}",
                                 mentioned_documents(question), router="llm")
    return None


def route(question: str, llm=None) -> RouteDecision:
    rules = rule_based_route(question)
    # Safety first: the LLM may only re-classify questions the rules consider a plain
    # procedure question. Release actions, comparisons and limit / approval questions
    # detected by the rules are never overridden (a limit question must always be verified).
    if rules.question_type != "procedure" or llm is None:
        return rules
    decision = llm_route(question, llm)
    return decision or rules
