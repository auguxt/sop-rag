"""Requirements 02/03 - section retrieval and defect-specific retrieval with different wording."""
import pytest

from sop_assistant.vocabulary import detect_defects, expand_query


@pytest.mark.parametrize("question, defect", [
    ("There are scratches on the housing", "scratch"),
    ("I see scuff marks on the face", "scratch"),
    ("Part is oversize", "dimensional"),
    ("the diameter is out of tolerance", "dimensional"),
    ("dimensional mismatch on the bore", "dimensional"),
    ("box has no sticker", "label"),
    ("missing label on the bin", "label"),
    ("the micrometer failed its calibration", "calibration"),
    ("gauge is out of calibration", "calibration"),
])
def test_defect_terms_are_recognised(question, defect):
    assert defect in [c.key for c in detect_defects(question)]


def test_calibration_label_is_not_a_label_defect():
    keys = [c.key for c in detect_defects("check the calibration label")]
    assert "label" not in keys


def test_query_expansion_adds_canonical_vocabulary():
    expanded = expand_query("box has no sticker")
    assert "LB-03" in expanded and "missing label" in expanded


@pytest.mark.parametrize("question, expected_docs", [
    ("I found scuff marks on the cover", {"SOP-SURF-009", "REF-DEF-005"}),
    ("Parts are coming out oversize on the CNC", {"SOP-QA-003", "CHK-MC-007", "REF-DEF-005"}),
    ("A package arrived with no sticker", {"SOP-LBL-008"}),
    ("Our micrometer failed its calibration check", {"SOP-CAL-004"}),
])
def test_defect_questions_retrieve_relevant_sop_in_top3(assistant, question, expected_docs):
    hits = assistant.retriever.retrieve(question, top_k=3)
    assert {h.doc_id for h in hits} & expected_docs


@pytest.mark.parametrize("question, doc_id, section_word", [
    ("What evidence is acceptable for final inspection?", "SOP-QA-002", "Evidence"),
    ("What are the steps for incoming material inspection?", "SOP-QA-001", "Steps"),
    ("What do I do when an instrument fails calibration?", "SOP-CAL-004", "Failed Calibration"),
    ("Who must be notified about an in-process defect?", "SOP-NCR-006", "Notification"),
])
def test_section_level_retrieval(assistant, question, doc_id, section_word):
    top = assistant.retriever.retrieve(question, top_k=3)
    assert any(h.doc_id == doc_id and section_word in h.section for h in top), [(h.doc_id, h.section) for h in top]


def test_metadata_filter_returns_only_one_document(assistant):
    hits = assistant.retriever.document_sections("SOP-QA-002")
    assert hits and {h.doc_id for h in hits} == {"SOP-QA-002"}
    assert [int(h.metadata["section_number"]) for h in hits] == sorted(int(h.metadata["section_number"]) for h in hits)


def test_hits_carry_scores_and_citations(assistant):
    hit = assistant.retriever.retrieve("incoming inspection steps", top_k=1)[0]
    assert hit.citation().startswith("[SOP-QA-001 §")
    assert "vector" in hit.boosts
