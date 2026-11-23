"""Plain data objects returned by the assistant."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Answer status values (also used in the logs and the evaluation)
GROUNDED = "grounded"              # answer built only from retrieved SOP text
NOT_VERIFIABLE = "not_verifiable"  # requested limit / rule / topic is not documented
CONFLICT = "conflict"              # documents disagree - both shown, none selected
ACTION_REFUSED = "action_refused"  # user asked the assistant to approve / release / inspect
ERROR = "error"                    # unexpected internal error (never crashes the app)

DISCLAIMER = ("This assistant only locates and summarises controlled SOP content. It has not inspected any "
              "part and cannot approve or release production batches - release authority is defined in "
              "SOP-REL-010. Open the cited passage and confirm it before acting.")


@dataclass
class RouteDecision:
    question_type: str          # procedure | comparison | limit_or_approval | release_action
    route: str                  # procedural_explanation | document_comparison | missing_information_check
    reason: str
    doc_ids: List[str] = field(default_factory=list)
    router: str = "rules"       # rules | llm


@dataclass
class Answer:
    question: str
    question_type: str
    route: str
    status: str
    answer_markdown: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    comparison: Optional[Dict[str, Any]] = None
    limit_facts: List[Dict[str, Any]] = field(default_factory=list)
    route_reason: str = ""
    processing_ms: float = 0.0
    generator: str = "extractive"     # extractive | llm:<name>
    warnings: List[str] = field(default_factory=list)

    @property
    def source_ids(self) -> List[str]:
        return list(dict.fromkeys(s["doc_id"] for s in self.sources if s.get("role") == "cited"))

    @property
    def cited_node_ids(self) -> List[str]:
        return [s["node_id"] for s in self.sources if s.get("role") == "cited"]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source_ids"] = self.source_ids
        return d
