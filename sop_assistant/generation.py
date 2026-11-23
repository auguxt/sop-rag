"""Answer composition (requirements 02, 04, 06, 07).

The default generator is *extractive*: every numbered step is a sentence or
list item copied from a retrieved SOP section, followed by its citation
``[DOC-ID §section, p.page]``. This guarantees grounded answers without any
LLM or API key. If an LLM is configured it may re-word the steps, but its
output is validated and rejected when a step is uncited or unsupported.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .comparison import section_items
from .limits import LimitCheck
from .llm import llm_numbered_summary
from .retrieval import Hit
from .vocabulary import detect_defects, mentioned_documents

MAX_SECTIONS = 3
MAX_STEPS = 14
STAGE_DOCS = {"SOP-QA-001", "SOP-QA-002", "SOP-QA-003"}  # incoming / final / in-process


def select_hits(hits: List[Hit], question: str = "", ratio: float = 0.85) -> List[Hit]:
    """Choose the passages the answer is built from.

    * always the best hit;
    * runners-up only if they score >= ratio * best AND were boosted for a
      reason of their own (matching section intent or defect code) - this keeps
      generic sections such as 'Purpose' out of the answer;
    * if the user named a procedure ("final", "incoming") and the best hit is
      from another document, the best intent-matching section of the named
      procedure is added as well.
    """
    if not hits:
        return []
    best = hits[0].score
    chosen = [hits[0]]
    for h in hits[1:]:
        if len(chosen) >= MAX_SECTIONS:
            break
        own_reason = h.boosts.get("section_intent", 0) > 0 or h.boosts.get("defect", 0) > 0
        if h.score >= ratio * best and own_reason and "generic_section_penalty" not in h.boosts:
            chosen.append(h)
    named = mentioned_documents(question)
    stage_named = [d for d in named if d in STAGE_DOCS]
    if stage_named and hits[0].doc_id in stage_named:
        # the user asked about a specific inspection stage: do not mix in other stages
        chosen = [h for h in chosen if h.doc_id in stage_named or h.doc_id not in STAGE_DOCS]
    if named and hits[0].doc_id not in named and len(chosen) < MAX_SECTIONS:
        for h in hits[1:]:
            if h.doc_id in named and h.boosts.get("section_intent", 0) > 0 and h not in chosen:
                chosen.append(h)
                break
    return chosen


def defect_reference_lines(question: str, all_hits: Dict[str, Hit]) -> List[Tuple[str, Hit]]:
    """For recognised defects, the matching line of the defect code reference (REF-DEF-005)."""
    out = []
    for cat in detect_defects(question):
        for hit in all_hits.values():
            if hit.doc_id != "REF-DEF-005":
                continue
            for item in section_items(hit.text):
                if item.startswith(cat.code):
                    out.append((item, hit))
    return out


def compose_procedure(question: str, hits: List[Hit], all_hits: Dict[str, Hit], doc_titles: Dict[str, str],
                      llm=None, llm_name: str = "none") -> Tuple[str, List[Hit], str]:
    selected = select_hits(hits, question)
    steps: List[Tuple[str, Hit]] = []
    generator = "extractive"
    if llm is not None:
        llm_steps = llm_numbered_summary(llm, question, selected)
        if llm_steps:
            steps, generator = llm_steps, f"llm:{llm_name}"
    defect_lines = defect_reference_lines(question, all_hits)
    if not steps:
        for h in selected:
            if defect_lines and h.doc_id == "REF-DEF-005" and "codes" in h.section.lower():
                continue  # the matching defect-code line is already shown above the steps
            for item in section_items(h.text):
                steps.append((item, h))
        if not steps:  # only the defect-code section was relevant
            steps = list(defect_lines)
    steps = steps[:MAX_STEPS]

    lines: List[str] = []
    used = [h for h in selected if any(h.node_id == sh.node_id for _, sh in steps)]
    if defect_lines:
        lines.append("**Recognised defect type (from the defect code reference):**")
        for text, h in defect_lines:
            lines.append(f"- {text} {h.citation()}")
            if h.node_id not in {u.node_id for u in used}:
                used.append(h)
        lines.append("")

    lines.append("**Documented procedure (summary of the retrieved SOP text):**")
    current = None
    for i, (text, h) in enumerate(steps, start=1):
        if h.node_id != current:
            current = h.node_id
            lines.append(f"*{h.doc_id} - {h.metadata.get('title', '')}, {h.section}*")
        lines.append(f"{i}. {text} {h.citation()}")

    refs = []
    for h in used:
        for ref in filter(None, h.metadata.get("references", "").split(",")):
            if ref not in refs and ref not in {u.doc_id for u in used}:
                refs.append(ref)
    if refs:
        lines.append("")
        lines.append("**Next documented step / related procedures referenced in these passages:**")
        for ref in refs:
            lines.append(f"- {ref} - {doc_titles.get(ref, 'see document')}")
    return "\n".join(lines), used, generator


def compose_limit_answer(question: str, check: LimitCheck, related: List[Hit]) -> str:
    lines: List[str] = []
    if check.status == "verified":
        if check.kind == "approval":
            lines.append("**Documented approval / authority rule:**")
        else:
            lines.append("**Documented value(s) found in the SOPs:**")
        for i, f in enumerate(check.facts, start=1):
            value = "" if check.kind == "approval" else f"**{f.display_value}** - "
            lines.append(f"{i}. {value}\"{f.sentence}\" {f.citation()}")
        values = {(f.value, f.unit) for f in check.facts if f.value}
        if len(check.facts) > 1 and len(values) == 1:
            lines.append("")
            lines.append("All retrieved sources state the same value.")
        return "\n".join(lines)

    if check.status == "conflict":
        lines.append("**Inconsistent limits found - the value cannot be verified.**")
        lines.append("")
        lines.append(f"The documents state different values for the same {check.attribute}. "
                     "Both sources are shown below; the assistant does **not** select a value without evidence "
                     "of which document prevails.")
        lines.append("")
        lines.append("| Source | Revision / effective date | Stated value | Exact wording |")
        lines.append("|---|---|---|---|")
        for f in check.facts:
            lines.append(f"| {f.citation()} {f.title} | Rev {f.revision or '?'} / {f.effective_date or '?'} | "
                         f"**{f.display_value}** | {f.sentence} |")
        owners = sorted({f"{f.doc_id} (owner: {f.owner})" for f in check.facts if f.owner})
        lines.append("")
        lines.append("**Suggested next step:** raise the discrepancy with the document owners"
                     + (f" - {', '.join(owners)} -" if owners else "")
                     + " or the Quality Manager and treat affected parts as suspect (HOLD) until the controlled "
                     "documents are aligned. Revision data is shown for reference only.")
        return "\n".join(lines)

    # not_found
    lines.append("**This cannot be verified from the indexed SOPs.**")
    lines.append("")
    if check.unknown_terms:
        lines.append(f"- {check.reason}")
    else:
        lines.append(f"- {check.reason}")
    lines.append("- No value is given because none is documented; do not assume a tolerance, acceptance limit or "
                 "approval rule. Check the part drawing / control plan or ask the responsible Quality Engineer.")
    if related:
        lines.append("")
        lines.append("Closest passages retrieved (they do **not** contain the requested value):")
        for h in related[:3]:
            lines.append(f"- {h.doc_id} - {h.metadata.get('title', '')}, {h.section} {h.citation()}")
    return "\n".join(lines)


def compose_not_found(question: str, unknown: List[str], related: List[Hit]) -> str:
    lines = ["**No SOP content found for this question - it cannot be answered from the indexed documents.**", ""]
    if unknown:
        lines.append(f"- The term(s) {', '.join(repr(u) for u in unknown)} do not appear in any indexed document.")
    lines.append("- The assistant only answers from the plant's controlled SOPs. Please rephrase, or contact the "
                 "Quality Engineer if the topic is not yet covered by a procedure.")
    if related:
        lines.append("")
        lines.append("Closest passages (low relevance):")
        for h in related[:3]:
            lines.append(f"- {h.doc_id} - {h.metadata.get('title', '')}, {h.section} {h.citation()}")
    return "\n".join(lines)


def compose_release_refusal(release_hits: List[Hit]) -> str:
    lines = ["**I can't do that.** I am a document assistant: I have not inspected any part and I cannot approve, "
             "pass or release a batch.", ""]
    authority = [h for h in release_hits if "authority" in h.section.lower()]
    steps = [h for h in release_hits if "release steps" in h.section.lower()]
    prereq = [h for h in release_hits if "prerequisite" in h.section.lower()]
    if authority:
        lines.append("**Who may release (documented):**")
        for item in section_items(authority[0].text):
            lines.append(f"- {item} {authority[0].citation()}")
        lines.append("")
    if prereq:
        lines.append("**Release prerequisites to check:**")
        for i, item in enumerate(section_items(prereq[0].text), start=1):
            lines.append(f"{i}. {item} {prereq[0].citation()}")
        lines.append("")
    if steps:
        first = section_items(steps[0].text)[0]
        lines.append(f"**Next documented step:** {first} {steps[0].citation()}")
    return "\n".join(lines)
