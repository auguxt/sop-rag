"""Guardrail unit tests and monitoring log (requirement 08)."""
import json

from sop_assistant.guardrails import REPLACEMENT, sanitize


def test_sanitize_removes_forbidden_claims():
    text, removed = sanitize("I have inspected the part and the batch B-12 is approved for shipment.")
    assert removed and REPLACEMENT in text
    assert "I have inspected the part" not in text


def test_sanitize_keeps_normal_text():
    original = "Submit the inspection package to the Quality Manager for the release decision."
    text, removed = sanitize(original)
    assert text == original and removed == []


def test_every_question_is_logged(assistant):
    before = len(assistant.logger.read())
    assistant.ask("What are the steps for incoming material inspection?")
    rows = assistant.logger.read()
    assert len(rows) == before + 1
    rec = rows[-1]
    for key in ("timestamp", "question", "question_type", "route", "answer_status", "source_ids", "processing_ms"):
        assert key in rec
    assert rec["question_type"] == "procedure" and rec["answer_status"] == "grounded"
    assert "SOP-QA-001" in rec["source_ids"]
    assert rec["processing_ms"] > 0
    json.dumps(rec)  # serialisable


def test_log_summary(assistant):
    assistant.ask("What is the canteen menu for today?")
    summary = assistant.logger.summary()
    assert summary["total"] >= 1 and "not_verifiable" in summary["by_status"]
