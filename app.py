"""Streamlit user interface (requirement 09).

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from sop_assistant.assistant import SOPAssistant
from sop_assistant.models import ACTION_REFUSED, CONFLICT, ERROR, GROUNDED, NOT_VERIFIABLE

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Quality SOP Assistant", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading SOP documents and vector index ...")
def get_assistant(rebuild: bool = False) -> SOPAssistant:
    return SOPAssistant(rebuild=rebuild)


def example_questions():
    path = ROOT / "evaluation" / "test_questions.json"
    try:
        return [q["question"] for q in json.loads(path.read_text(encoding="utf-8"))]
    except Exception:
        return ["What are the steps for incoming material inspection?"]


STATUS_STYLE = {
    GROUNDED: ("success", "Grounded in SOP text"),
    CONFLICT: ("warning", "Inconsistent documents - no value selected"),
    NOT_VERIFIABLE: ("info", "Cannot be verified from the SOPs"),
    ACTION_REFUSED: ("error", "Action refused - assistant cannot approve / release / inspect"),
    ERROR: ("error", "Internal error"),
}


def set_question(q: str):
    st.session_state["question"] = q
    st.session_state["auto_ask"] = True


assistant = get_assistant()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("System")
    st.write(f"**Embeddings:** `{assistant.embed_name}`")
    st.write(f"**Answer generator:** `{'LLM: ' + assistant.llm_name if assistant.llm else 'extractive (no LLM)'}`")
    st.write(f"**Index:** {len(assistant.documents)} documents, {len(assistant.nodes)} section nodes "
             f"({'loaded from storage' if assistant.index_info['loaded_from_storage'] else 'freshly built'})")
    with st.expander("Indexed documents"):
        st.dataframe(pd.DataFrame(assistant.documents), hide_index=True)
    if st.button("Rebuild index"):
        get_assistant.clear()
        get_assistant(rebuild=True)
        st.rerun()

    st.header("Example questions")
    for i, q in enumerate(example_questions()):
        st.button(q, key=f"ex{i}", on_click=set_question, args=(q,))

# ----------------------------------------------------------------------------- main
st.title("🔎 Manufacturing Quality SOP Assistant")
st.caption("Finds and summarises inspection procedures from the plant's controlled SOPs, with citations. "
           "It does **not** inspect parts and does **not** approve production releases.")

tab_ask, tab_eval, tab_log = st.tabs(["Ask a question", "Quality checks", "Query log"])

with tab_ask:
    with st.form("ask_form", clear_on_submit=False):
        question = st.text_input("Your question", key="question",
                                 placeholder="e.g. Who should I inform when a defect is found during final inspection?")
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted or st.session_state.pop("auto_ask", False):
        if question.strip():
            with st.spinner("Retrieving SOP passages ..."):
                st.session_state["answer"] = assistant.ask(question)

    ans = st.session_state.get("answer")
    if ans is not None:
        kind, label = STATUS_STYLE.get(ans.status, ("info", ans.status))
        getattr(st, kind)(f"**{label}**  ·  type: `{ans.question_type}`  ·  route: `{ans.route}`  ·  "
                          f"{ans.processing_ms:.0f} ms")
        for w in ans.warnings:
            st.warning(w)

        col_answer, col_sources = st.columns([3, 2], gap="large")
        with col_answer:
            st.subheader("Answer")
            if ans.comparison:
                comp = ans.comparison
                st.markdown(f"**{comp['doc_a']} ({comp['title_a']})  vs  {comp['doc_b']} ({comp['title_b']})**")
                if comp["key_differences"]:
                    st.markdown("**Key documented differences (numeric)**")
                    st.dataframe(pd.DataFrame([{
                        "Aspect": d["aspect"], "Quantity": d["quantity"],
                        comp["title_a"]: f"{d['a_value']}  {d['a_citation']}",
                        comp["title_b"]: f"{d['b_value']}  {d['b_citation']}"} for d in comp["key_differences"]]),
                        hide_index=True)
                st.markdown("**Section-by-section comparison** (text copied from the SOPs)")
                head_a, head_b = st.columns(2)
                head_a.markdown(f"#### {comp['title_a']}")
                head_b.markdown(f"#### {comp['title_b']}")
                for row in comp["rows"]:
                    st.markdown(f"**{row['aspect']}** — *{row['status']}*")
                    ca, cb = st.columns(2)
                    ca.markdown("\n".join(f"- {i}" for i in row["a_items"]) + f"\n\n`{row['a_citation']}`"
                                if row["a_citation"] else f"_{row['a_items'][0]}_")
                    cb.markdown("\n".join(f"- {i}" for i in row["b_items"]) + f"\n\n`{row['b_citation']}`"
                                if row["b_citation"] else f"_{row['b_items'][0]}_")
                    st.divider()
                with st.expander("Plain-text version of the answer"):
                    st.markdown(ans.answer_markdown)
            else:
                st.markdown(ans.answer_markdown)
            with st.expander("Why this route?"):
                st.write(ans.route_reason)
                st.write(f"Generator: `{ans.generator}`")

        with col_sources:
            st.subheader("Sources")
            st.caption("Open a passage to check it before following an instruction.")
            if not ans.sources:
                st.write("No supporting passage (nothing in the SOPs supports an answer).")
            role_label = {"cited": "cited in answer", "retrieved": "retrieved, not used",
                          "related": "closest match - does NOT contain the answer"}
            for i, s in enumerate(ans.sources, start=1):
                page = f", p.{s['page']}" if s["page"] not in ("n/a", "None") else ""
                title = f"[{i}] {s['doc_id']} · {s['section']}{page} — {role_label.get(s['role'], s['role'])}"
                with st.expander(title, expanded=(s["role"] == "cited" and i <= 2)):
                    st.markdown(f"**{s['title']}**  \nFile: `{s['file_name']}`")
                    st.text(s["text"])
                    st.caption(f"vector similarity {s['vector_score']} · final score {s['score']} · boosts {s['boosts']}")
                    path = assistant.cfg.data_dir / s["file_name"]
                    if path.exists():
                        st.download_button("Open source document", data=path.read_bytes(), file_name=s["file_name"],
                                           key=f"dl{i}-{s['node_id']}")

with tab_eval:
    st.subheader("Automated quality checks")
    st.write("Runs the test questions (missing tolerance, contradictory instructions, comparison, defect synonyms, "
             "release request ...) and measures routing, retrieval relevance, citation correctness and grounded outcome.")
    if st.button("Run evaluation", type="primary"):
        from evaluation.run_eval import evaluate, write_reports

        with st.spinner("Running evaluation ..."):
            result = evaluate(assistant)
            write_reports(result)
        st.session_state["eval"] = result
    result = st.session_state.get("eval")
    if result:
        s = result["summary"]
        c = st.columns(5)
        c[0].metric("Passed", f"{s['passed']}/{s['questions']}")
        c[1].metric("Routing accuracy", s["routing_accuracy"])
        c[2].metric("Retrieval Hit@3", s["retrieval_hit_at_3"])
        c[3].metric("Citation correctness", s["citation_correctness"])
        c[4].metric("Grounded outcome", s["grounded_outcome_rate"])
        df = pd.DataFrame(result["rows"])[["id", "category", "question_type", "answer_status", "hit_at_3", "mrr",
                                            "citation_correct", "groundedness", "grounded_outcome_ok",
                                            "processing_ms", "passed"]]
        st.dataframe(df, hide_index=True)
        st.caption("Reports saved to evaluation/results/ (eval_report.md, eval_results.csv, eval_results.json)")

with tab_log:
    st.subheader("Monitoring log")
    summary = assistant.logger.summary()
    if summary.get("total"):
        c = st.columns(3)
        c[0].metric("Questions logged", summary["total"])
        c[1].metric("Average time (ms)", summary["avg_ms"])
        c[2].metric("Max time (ms)", summary["max_ms"])
        st.write("By status:", summary["by_status"])
        st.write("By question type:", summary["by_type"])
        rows = assistant.logger.read(last=50)[::-1]
        st.dataframe(pd.DataFrame(rows)[["timestamp", "question", "question_type", "route", "answer_status",
                                         "source_ids", "processing_ms"]], hide_index=True)
        st.caption(f"Full log: {assistant.cfg.log_file}")
    else:
        st.write("No questions logged yet.")
