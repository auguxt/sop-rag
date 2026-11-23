"""Requirement 08 - quality checks.

Runs every question in evaluation/test_questions.json through the assistant
and measures:

* Routing accuracy      - question type predicted by the router == expected
* Retrieval relevance   - Hit@3 and MRR of the relevant documents in the
                          re-ranked retrieval results
* Citation correctness  - (a) every citation in the answer points to a passage
                          that was really retrieved and is in the sources panel,
                          (b) the required documents are cited,
                          (c) "not verifiable" answers cite no value-bearing source
* Grounded outcome      - answer status == expected status, every numbered /
                          bulleted statement is supported by its cited passage
                          (token overlap >= 0.8) and required phrases are present

Results: evaluation/results/eval_results.csv, eval_results.json, eval_report.md
Run:     python -m evaluation.run_eval            (from the project folder)
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sop_assistant.models import Answer  # noqa: E402
from sop_assistant.text_utils import token_overlap  # noqa: E402

QUESTIONS_FILE = ROOT / "evaluation" / "test_questions.json"
RESULTS_DIR = ROOT / "evaluation" / "results"

CITATION_RE = re.compile(r"\[((?:SOP|REF|CHK)-[A-Z]+-\d{3}) §([^\],]+)(?:, p\.([\d-]+))?\]")


def load_questions(path: Path = QUESTIONS_FILE) -> List[Dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _clean_statement(text: str) -> str:
    text = re.sub(r"\*\*[^*]+\*\*\s+-\s+", "", text)  # "**5 mm** - " value prefix
    text = text.replace("**", "").replace('"', "").strip()
    return text


def check_citations(ans: Answer) -> Dict:
    """Every [DOC §section] in the answer must match a source in the sources panel."""
    by_key = {(s["doc_id"], s["section"]): s for s in ans.sources}
    cites = CITATION_RE.findall(ans.answer_markdown)
    valid = [c for c in cites if (c[0], c[1].strip()) in by_key]
    page_ok = all(
        (not c[2]) or by_key[(c[0], c[1].strip())]["page"] == c[2] for c in valid
    )
    return {"n_citations": len(cites), "n_valid": len(valid), "all_valid": len(valid) == len(cites), "pages_ok": page_ok}


def groundedness(ans: Answer) -> float:
    """Fraction of cited statements that are supported by the passage they cite."""
    by_key = {(s["doc_id"], s["section"]): s["text"] for s in ans.sources}
    supported = total = 0
    for line in ans.answer_markdown.splitlines():
        m = re.match(r"^\s*(?:\d+\.|-)\s+(.*)$", line)
        if not m or line.lstrip().startswith("- The term"):
            continue
        body = m.group(1)
        cites = CITATION_RE.findall(body)
        if not cites:
            continue
        if re.match(r"^(SOP|REF|CHK)-[A-Z]+-\d{3} - ", body):
            continue  # pointer to a related document ("closest passages"), not a claim
        if len(cites) >= 2:
            # comparison / conflict line: every number before a citation must be in that cited passage
            segments = CITATION_RE.split(body)
            ok = True
            # re.split with 3 groups -> [text, doc, section, page, text, doc, section, page, ...]
            for i in range(0, len(segments) - 3, 4):
                seg, doc, section = segments[i], segments[i + 1], segments[i + 2].strip()
                passage = by_key.get((doc, section), "")
                numbers = re.findall(r"\d+(?:\.\d+)?", re.sub(r"(SOP|REF|CHK|QF)-[A-Z]*-?\d+", "", seg))
                if not passage or any(n not in passage for n in numbers):
                    ok = False
            total += 1
            supported += int(ok)
            continue
        statement = _clean_statement(CITATION_RE.sub("", m.group(1)))
        doc, section = cites[-1][0], cites[-1][1].strip()
        passage = by_key.get((doc, section), "")
        total += 1
        if passage and token_overlap(statement, passage) >= 0.8:
            supported += 1
    return 1.0 if total == 0 else supported / total


def evaluate(assistant, questions: List[Dict] = None) -> Dict:
    questions = questions or load_questions()
    rows = []
    for q in questions:
        ans: Answer = assistant.ask(q["question"])
        # --- retrieval relevance (independent of the answer composer)
        retrieved = [h.doc_id for h in assistant.retriever.retrieve(q["question"], top_k=5)]
        relevant = set(q.get("relevant_docs", []))
        if relevant:
            ranks = [i + 1 for i, d in enumerate(retrieved) if d in relevant]
            hit3 = bool(ranks and ranks[0] <= 3)
            mrr = 1.0 / ranks[0] if ranks else 0.0
        else:
            hit3, mrr = None, None
        # --- citation correctness
        cit = check_citations(ans)
        cited_docs = set(ans.source_ids)
        need_all = set(q.get("must_cite", []))
        need_any = set(q.get("must_cite_any", []))
        cites_required = need_all <= cited_docs and (not need_any or bool(need_any & cited_docs))
        if q["expected_status"] == "not_verifiable":
            cites_required = len(cited_docs) == 0
        citation_correct = cit["all_valid"] and cit["pages_ok"] and cites_required
        # --- grounded outcome
        g = groundedness(ans)
        text = ans.answer_markdown
        contains_ok = all(p.lower() in text.lower() for p in q.get("must_contain", []))
        not_contains_ok = not any(p.lower() in text.lower() for p in q.get("must_not_contain", []))
        status_ok = ans.status == q["expected_status"]
        grounded_ok = status_ok and g >= 0.999 and contains_ok and not_contains_ok
        route_ok = ans.question_type == q["expected_type"]
        rows.append({
            "id": q["id"], "category": q["category"], "question": q["question"],
            "expected_type": q["expected_type"], "question_type": ans.question_type, "route": ans.route,
            "route_ok": route_ok,
            "expected_status": q["expected_status"], "answer_status": ans.status, "status_ok": status_ok,
            "retrieved_top5": retrieved, "hit_at_3": hit3, "mrr": mrr,
            "cited_docs": sorted(cited_docs), "n_citations": cit["n_citations"],
            "citation_correct": citation_correct,
            "groundedness": round(g, 3), "phrases_ok": contains_ok and not_contains_ok,
            "grounded_outcome_ok": grounded_ok,
            "passed": route_ok and citation_correct and grounded_ok and (hit3 is not False),
            "processing_ms": round(ans.processing_ms, 1),
        })

    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(float(v) for v in vals) / len(vals), 3) if vals else None

    summary = {
        "questions": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "routing_accuracy": avg("route_ok"),
        "retrieval_hit_at_3": avg("hit_at_3"),
        "retrieval_mrr": avg("mrr"),
        "citation_correctness": avg("citation_correct"),
        "grounded_outcome_rate": avg("grounded_outcome_ok"),
        "mean_groundedness": avg("groundedness"),
        "avg_processing_ms": avg("processing_ms"),
        "embed_model": assistant.embed_name,
        "generator": "llm:" + assistant.llm_name if assistant.llm else "extractive",
    }
    return {"summary": summary, "rows": rows}


def write_reports(result: Dict, out_dir: Path = RESULTS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, s = result["rows"], result["summary"]
    (out_dir / "eval_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "eval_results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (";".join(v) if isinstance(v, list) else v) for k, v in r.items()})

    def yn(v):
        return "n/a" if v is None else ("PASS" if v else "FAIL")

    lines = ["# Evaluation report - Manufacturing Quality SOP Assistant", "",
             f"Embedding model: `{s['embed_model']}`  |  Generator: `{s['generator']}`", "",
             "| Metric | Value |", "|---|---|"]
    for k in ("questions", "passed", "routing_accuracy", "retrieval_hit_at_3", "retrieval_mrr",
              "citation_correctness", "grounded_outcome_rate", "mean_groundedness", "avg_processing_ms"):
        lines.append(f"| {k} | {s[k]} |")
    lines += ["", "| ID | Category | Type (exp -> got) | Status (exp -> got) | Hit@3 | MRR | Citations | Grounded | ms | Result |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['category']} | {r['expected_type']} -> {r['question_type']} | "
            f"{r['expected_status']} -> {r['answer_status']} | {yn(r['hit_at_3'])} | "
            f"{'n/a' if r['mrr'] is None else round(r['mrr'], 2)} | {yn(r['citation_correct'])} | "
            f"{yn(r['grounded_outcome_ok'])} ({r['groundedness']}) | {r['processing_ms']} | "
            f"{'PASS' if r['passed'] else 'FAIL'} |")
    (out_dir / "eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import logging

    logging.basicConfig(level=logging.WARNING)
    from sop_assistant.assistant import SOPAssistant

    assistant = SOPAssistant()
    result = evaluate(assistant)
    write_reports(result)
    s = result["summary"]
    print(f"\nEmbedding model: {s['embed_model']}   generator: {s['generator']}\n")
    print(f"{'ID':<4} {'Type':<18} {'Status':<15} {'Hit@3':<6} {'Cite':<5} {'Grnd':<5} {'ms':>6}  Result")
    for r in result["rows"]:
        h = "n/a" if r["hit_at_3"] is None else ("yes" if r["hit_at_3"] else "NO")
        print(f"{r['id']:<4} {r['question_type']:<18} {r['answer_status']:<15} {h:<6} "
              f"{'ok' if r['citation_correct'] else 'BAD':<5} {'ok' if r['grounded_outcome_ok'] else 'BAD':<5} "
              f"{r['processing_ms']:>6}  {'PASS' if r['passed'] else 'FAIL'}")
    print("\nSummary:")
    for k, v in s.items():
        print(f"  {k:<24} {v}")
    print(f"\nReports written to {RESULTS_DIR}")
    return 0 if s["passed"] == s["questions"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
