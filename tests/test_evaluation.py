"""Requirement 08 - the full evaluation set (>= 8 questions, incl. missing tolerance and contradiction)."""
from evaluation.run_eval import evaluate, load_questions


def test_question_set_covers_required_cases():
    qs = load_questions()
    assert len(qs) >= 8
    cats = " ".join(q["category"] for q in qs)
    assert "missing tolerance" in cats and "contradictory" in cats


def test_all_evaluation_questions_pass(assistant):
    result = evaluate(assistant)
    failed = [r for r in result["rows"] if not r["passed"]]
    assert not failed, failed
    s = result["summary"]
    assert s["retrieval_hit_at_3"] == 1.0 and s["citation_correctness"] == 1.0 and s["grounded_outcome_rate"] == 1.0
