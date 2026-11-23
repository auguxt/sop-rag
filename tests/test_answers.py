"""Requirements 02/04/05/06/07 - procedures, citations, comparison, missing data, conflicts, routing."""
import re

import pytest

from sop_assistant.comparison import section_items
from sop_assistant.models import ACTION_REFUSED, CONFLICT, DISCLAIMER, GROUNDED, NOT_VERIFIABLE
from sop_assistant.router import rule_based_route

CITE = re.compile(r"\[((?:SOP|REF|CHK)-[A-Z]+-\d{3}) §([^\],]+)")


# ---------------------------------------------------------------- routing (07)
@pytest.mark.parametrize("question, qtype", [
    ("What are the steps for final inspection?", "procedure"),
    ("Who should I inform when a defect is found?", "procedure"),
    ("What is different between incoming and final inspection?", "comparison"),
    ("Compare in-process and final inspection", "comparison"),
    ("What is the torque tolerance for M8 bolts?", "limit_or_approval"),
    ("How often must calipers be calibrated?", "limit_or_approval"),
    ("Who approves the release of a batch?", "limit_or_approval"),
    ("Please approve batch B-2231 for shipment.", "release_action"),
    ("Can you release this lot?", "release_action"),
    ("Is batch 45 ok to ship?", "release_action"),
    ("Have you inspected lot 7?", "release_action"),
    ("Release steps for a batch", "procedure"),
])
def test_router(question, qtype):
    assert rule_based_route(question).question_type == qtype


# ---------------------------------------------------------------- procedure + citations (02/04)
def test_procedure_answer_is_numbered_and_every_step_is_cited(assistant):
    ans = assistant.ask("What are the steps for incoming material inspection?")
    assert ans.status == GROUNDED and ans.route == "procedural_explanation"
    steps = [l for l in ans.answer_markdown.splitlines() if re.match(r"^\d+\. ", l)]
    assert len(steps) >= 8
    assert all(CITE.search(s) for s in steps)


def test_every_step_text_comes_from_its_cited_passage(assistant):
    ans = assistant.ask("Who should I inform when a defect is found during final inspection?")
    passages = {(s["doc_id"], s["section"]): s["text"] for s in ans.sources}
    for line in ans.answer_markdown.splitlines():
        m = re.match(r"^\d+\. (.*) \[((?:SOP|REF|CHK)-[A-Z]+-\d{3}) §([^\],]+)", line)
        if m:
            text, doc, sec = m.group(1), m.group(2), m.group(3).strip()
            assert text in passages[(doc, sec)], line


def test_sources_contain_inspectable_passages(assistant):
    ans = assistant.ask("What evidence is acceptable to show final inspection was completed?")
    cited = [s for s in ans.sources if s["role"] == "cited"]
    assert cited and all(s["text"] and s["file_name"] and s["section"] for s in cited)
    assert "SOP-QA-002" in ans.source_ids


def test_defect_answer_shows_defect_code(assistant):
    ans = assistant.ask("A box arrived with no sticker on it. What should I do?")
    assert "LB-03" in ans.answer_markdown and "SOP-LBL-008" in ans.source_ids


# ---------------------------------------------------------------- comparison (05)
def test_comparison_retrieves_both_procedures(assistant):
    ans = assistant.ask("What is different between incoming and final inspection?")
    assert ans.route == "document_comparison" and ans.status == GROUNDED
    comp = ans.comparison
    assert (comp["doc_a"], comp["doc_b"]) == ("SOP-QA-001", "SOP-QA-002")
    assert {"SOP-QA-001", "SOP-QA-002"} <= set(ans.source_ids)
    aspects = [r["aspect"] for r in comp["rows"]]
    for expected in ("Sampling Plan", "Inspection Steps", "Acceptable Evidence", "Defect Handling and Escalation"):
        assert expected in aspects
    aql = [d for d in comp["key_differences"] if d["quantity"] == "AQL"]
    assert aql and aql[0]["a_value"] == "1.0" and aql[0]["b_value"] == "0.65"


def test_comparison_does_not_invent_steps(assistant):
    ans = assistant.ask("What is different between incoming and final inspection?")
    texts = {s["node_id"]: s["text"] for s in ans.sources}
    for row in ans.comparison["rows"]:
        for side in ("a", "b"):
            node = row[f"{side}_node"]
            if node is None:
                assert row[f"{side}_items"][0].startswith("Not specified")
            else:
                assert row[f"{side}_items"] == section_items(texts[node])


def test_comparison_marks_sections_missing_in_one_document(assistant):
    ans = assistant.ask("Compare the calibration SOP and the CNC checklist")
    statuses = [r["status"] for r in ans.comparison["rows"]]
    assert any(s.startswith("Only in") for s in statuses)


# ---------------------------------------------------------------- missing data / conflicts (06)
def test_missing_tolerance_is_not_verifiable(assistant):
    ans = assistant.ask("What is the torque tolerance for M8 fasteners on the gearbox cover?")
    assert ans.status == NOT_VERIFIABLE
    assert "cannot be verified" in ans.answer_markdown
    assert not re.search(r"\d+(\.\d+)?\s*(Nm|N·m|mm)", ans.answer_markdown)
    assert ans.source_ids == []        # nothing is cited as support


def test_missing_approval_rule_is_not_verifiable(assistant):
    ans = assistant.ask("Who can approve overtime for the inspectors?")
    assert ans.status == NOT_VERIFIABLE


def test_contradictory_scratch_limit_shows_both_sources(assistant):
    ans = assistant.ask("What is the maximum allowed scratch length on a visible Class A surface?")
    assert ans.status == CONFLICT
    assert {"REF-DEF-005", "SOP-SURF-009"} <= set(ans.source_ids)
    values = {(f["doc_id"], f["value"]) for f in ans.limit_facts}
    assert ("REF-DEF-005", "5") in values and ("SOP-SURF-009", "3") in values
    assert "does **not** select" in ans.answer_markdown


def test_contradictory_calibration_interval(assistant):
    ans = assistant.ask("How often must vernier calipers be calibrated?")
    assert ans.status == CONFLICT
    assert {"SOP-CAL-004", "CHK-MC-007"} <= set(ans.source_ids)


def test_documented_limit_is_verified_with_citation(assistant):
    ans = assistant.ask("What minimum lighting level is required for visual inspection?")
    assert ans.status == GROUNDED
    assert "1000 lux" in ans.answer_markdown and CITE.search(ans.answer_markdown)


def test_consistency_check_finds_cross_document_conflicts(assistant):
    from sop_assistant.limits import find_cross_document_conflicts

    corpus = list(assistant.all_hits.values())
    pairs = find_cross_document_conflicts([assistant.all_hits["SOP-CAL-004::S4"]], corpus)
    assert {(a.doc_id, a.value, b.doc_id, b.value) for a, b in pairs} == {("SOP-CAL-004", "6", "CHK-MC-007", "12")}
    pairs = find_cross_document_conflicts([assistant.all_hits["SOP-SURF-009::S5"]], corpus)
    assert ("SOP-SURF-009", "3", "REF-DEF-005", "5") in {(a.doc_id, a.value, b.doc_id, b.value) for a, b in pairs}
    # different inspection stages may legitimately differ (AQL 1.0 vs 0.65) - not a conflict
    assert find_cross_document_conflicts([assistant.all_hits["SOP-QA-001::S4"]], corpus) == []


def test_out_of_scope_question(assistant):
    ans = assistant.ask("What is the canteen menu for today?")
    assert ans.status == NOT_VERIFIABLE and ans.source_ids == []


@pytest.mark.parametrize("question", ["", "   ", "?", "hello"])
def test_degenerate_input_does_not_crash(assistant, question):
    ans = assistant.ask(question)
    assert ans.status == NOT_VERIFIABLE


# ---------------------------------------------------------------- guardrails (07)
def test_release_request_is_refused_with_next_documented_step(assistant):
    ans = assistant.ask("Please approve batch B-2231 for shipment.")
    assert ans.status == ACTION_REFUSED
    assert "SOP-REL-010" in ans.source_ids
    assert "Next documented step" in ans.answer_markdown


def test_answers_never_claim_inspection_or_release(assistant):
    from sop_assistant.guardrails import find_forbidden_claims

    for q in ["Please approve batch B-2231 for shipment.", "Release steps for a batch",
              "Who approves the release of a batch?", "What are the steps for final inspection?"]:
        ans = assistant.ask(q)
        assert find_forbidden_claims(ans.answer_markdown) == []
        assert DISCLAIMER in ans.answer_markdown
