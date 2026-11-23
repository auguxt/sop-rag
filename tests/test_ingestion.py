"""Requirement 01 - ingestion of PDF + text files, document identity, section labels, index."""
import re

from sop_assistant.ingest import build_nodes, build_or_load_index, load_documents, parse_sections


def test_collection_has_8_to_12_documents_in_pdf_and_text(assistant):
    docs = assistant.documents
    assert 8 <= len(docs) <= 12
    types = {d["file_type"] for d in docs}
    assert "pdf" in types and "txt" in types


def test_pdf_loaded_page_by_page(config):
    pages = load_documents(config.data_dir)
    pdf_pages = [d for d in pages if d.metadata["file_type"] == "pdf"]
    assert pdf_pages, "no PDF pages loaded"
    assert all(isinstance(d.metadata["page"], int) and d.metadata["page"] >= 1 for d in pdf_pages)
    assert any(d.metadata["page"] == 2 for d in pdf_pages), "multi-page PDFs expected"


def test_every_node_keeps_document_identity_and_section_label(assistant):
    for node in assistant.nodes:
        md = node.metadata
        assert re.fullmatch(r"(SOP|REF|CHK)-[A-Z]+-\d{3}", md["doc_id"])
        assert md["title"]
        assert re.match(r"^\d+\. \S", md["section"]), md["section"]
        assert md["page_label"]
        assert node.node_id.startswith(md["doc_id"] + "::S")


def test_expected_sections_are_parsed(assistant):
    sections = {(n.metadata["doc_id"], n.metadata["section"]) for n in assistant.nodes}
    assert ("SOP-QA-001", "5. Inspection Steps") in sections
    assert ("SOP-QA-002", "6. Acceptable Evidence") in sections
    assert ("SOP-CAL-004", "6. Failed Calibration Handling") in sections
    assert ("REF-DEF-005", "3. Defect Codes") in sections


def test_section_spanning_two_pdf_pages_has_page_range(assistant):
    ranges = [h for h in assistant.all_hits.values() if "-" in h.page]
    assert ranges, "at least one PDF section continues on the next page"
    assert assistant.all_hits["SOP-QA-001::S5"].page == "1-2"


def test_wrapped_pdf_lines_are_rejoined(assistant):
    text = assistant.all_hits["SOP-QA-001::S5"].text
    steps = [l for l in text.splitlines() if re.match(r"^5\.\d ", l)]
    assert len(steps) == 8                      # 5.1 ... 5.8, one logical line each


def test_txt_sections_have_no_page(assistant):
    assert assistant.all_hits["REF-DEF-005::S3"].page == "n/a"


def test_parse_sections_header_and_headings():
    header, sections = parse_sections([(1, "Document ID: SOP-X-001\nTitle: Demo\n1. Purpose\nText line\nwrapped\n2. Steps\n2.1 Do A\n2.2 Do B")])
    assert header["Document ID"] == "SOP-X-001" and header["Title"] == "Demo"
    assert [s.label for s in sections] == ["1. Purpose", "2. Steps"]
    assert sections[0].lines == ["Text line wrapped"]
    assert sections[1].lines == ["2.1 Do A", "2.2 Do B"]


def test_index_is_persisted_and_reloaded(config, assistant):
    index, nodes, info = build_or_load_index(config.data_dir, config.storage_dir, assistant.embed_model,
                                             assistant.embed_name, persist=True)
    assert info["loaded_from_storage"] is True
    assert len(nodes) == len(assistant.nodes)


def test_build_nodes_counts_match(config):
    nodes = build_nodes(load_documents(config.data_dir))
    assert len({n.metadata["doc_id"] for n in nodes}) == 10
    assert len(nodes) > 50
