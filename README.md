# Manufacturing Quality SOP Assistant
**Individual RAG case study · Primary framework: LlamaIndex**

This assistant answers questions from a plant's quality SOPs (inspection procedures, defect codes, calibration and non-conformance workflows). It:

* finds the right procedure, even when the question uses different words ("scuff marks", "oversize", "no sticker")
* summarises the procedure as numbered steps, and cites every step `[SOP-QA-001 §5. Inspection Steps, p.1-2]`
* shows the supporting passage in a sources panel, so you can check it before acting
* compares two procedures side by side, without inventing steps
* says **"cannot be verified"** when a tolerance or approval rule is not documented
* shows **both sources** when documents contradict each other, and does not pick a value
* **never** claims to have inspected a part, and never approves or releases a batch

It runs **without an API key and without an LLM**. Answers are extractive: every sentence comes from a retrieved SOP passage. An LLM can be added if you want (see Configuration).

> **New here?** Read `PROJECT_EXPLANATION.md` (simple English) and `SAMPLE_QUESTIONS.md` (47 tested questions).

---

## 1. Quick start

```bash
# 1. create a virtual environment (Python 3.10 - 3.12 recommended)
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 2. install
pip install -r requirements.txt

# 3. run the web app (question box + answer + sources panel)
streamlit run app.py

# 4. or use the command line
python cli.py "What are the steps for incoming material inspection?" --sources
python cli.py --interactive

# 5. automated tests and evaluation
python -m pytest                    # 72 tests
python -m evaluation.run_eval       # 16 test questions -> evaluation/results/
```

On the first run the HuggingFace embedding model `BAAI/bge-small-en-v1.5` (~130 MB) is downloaded. If it can't be installed or downloaded (for example, a lab with no internet), the program switches to its built-in **offline embeddings** by itself, so it still runs. To force offline mode: `python cli.py --embed local ...` or set `SOP_EMBED_BACKEND=local`.

> Tip for the viva: run `python cli.py --rebuild --list-docs` once before the viva. This builds the index, saves it in `storage/`, and lists the 10 documents.

---

## 2. Project structure

```
sop_rag/
├── app.py                       Streamlit UI: question box, answer, sources panel, quality checks, log
├── cli.py                       Command-line interface
├── requirements.txt
├── data/
│   ├── sops/                    THE DOCUMENT COLLECTION THAT IS INDEXED (5 PDF + 5 TXT)
│   └── source_text/             editable text source of every document
├── scripts/build_sample_documents.py   regenerates data/sops (PDFs via reportlab)
├── sop_assistant/
│   ├── config.py                settings (env-var overridable)
│   ├── ingest.py                load PDF/TXT -> parse sections -> TextNodes -> VectorStoreIndex (persisted)
│   ├── embeddings.py            HuggingFace embeddings + offline LocalHashEmbedding fallback
│   ├── vocabulary.py            defect synonyms (scratch, dimensional, label, calibration ...) + doc aliases
│   ├── retrieval.py             vector search + explainable re-ranking, metadata-filtered retrieval
│   ├── router.py                agent router: procedural / comparison / missing-information
│   ├── generation.py            numbered, cited summaries; missing-info, conflict and refusal answers
│   ├── comparison.py            section-by-section comparison of two procedures
│   ├── limits.py                tolerance / approval-rule verification, conflict detection
│   ├── guardrails.py            blocks "I inspected / approved / released" claims
│   ├── llm.py                   optional LLM (OpenAI / Ollama) with output validation
│   ├── monitoring.py            JSONL query log (type, sources, status, time)
│   ├── models.py                Answer / RouteDecision data classes and status values
│   └── assistant.py             orchestration: router -> tool -> guardrail -> log
├── evaluation/
│   ├── test_questions.json      16 test questions with expected outcomes
│   ├── run_eval.py              routing, Hit@3/MRR, citation correctness, groundedness
│   └── results/                 eval_report.md, eval_results.csv, eval_results.json
├── tests/                       pytest suite (ingestion, retrieval, comparison, missing data, UI ...)
├── storage/                     persisted vector index (auto-created)
└── logs/query_log.jsonl         monitoring log (auto-created)
```

---

## 3. Architecture

```
            PDF (pypdf, per page) + TXT
                       │
        ┌──────────────▼──────────────┐
        │ ingest.py                   │  header -> doc_id, title, revision
        │ rebuild wrapped lines       │  "5. Inspection Steps" -> section
        │ one TextNode per section    │  metadata: doc_id, section, page range, file
        └──────────────┬──────────────┘
                       │ embeddings (bge-small / offline hashing)
               VectorStoreIndex (LlamaIndex, persisted in storage/)
                       │
question ──► router.py ─┼─► procedural_explanation ─► retrieve + rerank ─► numbered cited steps
                        ├─► document_comparison ────► retrieve both docs (metadata filter) ─► aligned table
                        ├─► missing_information_check ► extract limit statements ─► verified | conflict | cannot verify
                        └─► release_action ─────────► refuse + documented next step (SOP-REL-010)
                       │
               guardrails.py (no "I inspected/released" claims) + disclaimer
                       │
               monitoring.py (JSONL log)  ──►  UI / CLI  (answer + sources panel)
```

The three tools are registered as LlamaIndex `FunctionTool`s, the same interface an LLM agent would use.

---

## 4. Requirement → implementation map

| # | Requirement | Where / how |
|---|---|---|
| 01 | 8–12 documents, PDF + text, keep document identity & section labels, embeddings, semantic index | `data/sops` has 10 docs (5 PDF, 5 TXT). `ingest.py` reads PDFs page by page with pypdf, parses the header (Document ID, Title, Revision) and the numbered sections, and creates one `TextNode` per section with `doc_id / title / section / page_label / file_name` metadata. These go into a LlamaIndex `VectorStoreIndex`, persisted with a fingerprint. |
| 02 | Procedure lookup, numbered summaries from retrieved content only | `assistant.procedural_explanation` → `generation.compose_procedure`. Every numbered step is a list item copied from a retrieved section. Section-intent re-ranking covers "evidence", "who to inform" and "steps" questions. |
| 03 | Defect-specific retrieval with different wording | `vocabulary.py` maps synonyms to scratch SC-01, dimensional mismatch DM-02, missing label LB-03, failed calibration CF-04 (plus dent, corrosion, burr). The query is expanded with the SOP vocabulary, passages containing the defect code are boosted, and the defect-code line from REF-DEF-005 is shown. |
| 04 | Citations with document/section/page; inspect the passage | Every step ends with `[DOC §section, p.x]`. The sources panel (UI expanders / `--sources` in the CLI) shows the full passage, file name, scores and a download of the source document. |
| 05 | Comparison of incoming vs final inspection | `comparison.py` retrieves both documents with a `doc_id` metadata filter and aligns their sections ("Sampling Plan" ↔ "Sampling Plan"). It shows a two-column view plus numeric key differences (AQL 1.0 vs 0.65). Sections missing from one document are marked "Not specified". Nothing is invented. |
| 06 | Missing limits → "cannot be verified"; inconsistent limits → show both | `limits.py` extracts numeric statements (value + unit + attribute) and approval sentences, then returns `verified`, `conflict` (both sources, revision info, no value chosen) or `not_found`. A cross-document consistency check also warns inside procedure answers. |
| 07 | Lightweight agent routing; may suggest next step; never claim inspection/release | `router.py` (rules, optional LLM) routes to 3 tools. Release/approval requests are refused, with the documented next step from SOP-REL-010. `guardrails.py` removes forbidden claims from any output (including LLM output), and a disclaimer is added to every answer. |
| 08 | ≥8 test questions incl. missing tolerance + contradiction; relevance, citation, groundedness; logging | `evaluation/test_questions.json` (16 questions) and `run_eval.py` measure routing accuracy, Hit@3/MRR, citation correctness and groundedness. `monitoring.py` logs question type, route, source IDs, answer status and processing time to `logs/query_log.jsonl`. |
| 09 | Question box, readable answer + sources panel; automated tests | `app.py` (Streamlit: Ask / Quality checks / Query log tabs). `tests/` holds 72 pytest tests covering ingestion, section retrieval, comparisons, missing-data handling, conflicts, guardrails, logging, evaluation and the UI (Streamlit AppTest). |

---

## 5. Document collection

| ID | Title | Format | Notes |
|---|---|---|---|
| SOP-QA-001 | Incoming Material Inspection | PDF | AQL 1.0, QF-101, records 3 years |
| SOP-QA-002 | Final Inspection | PDF | AQL 0.65, QF-201, 1000 lux, records 5 years |
| SOP-QA-003 | In-Process Inspection | TXT | first-piece / patrol every 2 h, DM-02 stop machine |
| SOP-CAL-004 | Calibration of Measuring Instruments | PDF | calipers **every 6 months**, failed-calibration handling |
| REF-DEF-005 | Defect Code Reference | TXT | SC-01 … BR-07, scratch Class A **max length 5 mm** |
| SOP-NCR-006 | Non-Conformance Workflow | PDF | NCR steps, notification matrix, MRB |
| CHK-MC-007 | CNC Machine Quality Checklist | TXT | calipers **every 12 months** ← contradicts SOP-CAL-004 |
| SOP-LBL-008 | Labelling and Traceability | TXT | missing / incorrect label LB-03 |
| SOP-SURF-009 | Surface Finish and Cosmetic Inspection | TXT | scratch Class A **max length 3 mm** ← contradicts REF-DEF-005 |
| SOP-REL-010 | Batch Release and Quarantine Control | PDF | only the Quality Manager may release |

Two contradictions are deliberate test cases: scratch length (5 mm vs 3 mm) and caliper calibration interval (6 vs 12 months). No document mentions torque, flatness or roughness, so those questions test missing tolerances.

---

## 6. Evaluation results (`python -m evaluation.run_eval`)

| Metric | Result |
|---|---|
| Questions / passed | 16 / 16 |
| Routing accuracy | 1.0 |
| Retrieval Hit@3 / MRR | 1.0 / 1.0 |
| Citation correctness | 1.0 |
| Grounded-outcome rate | 1.0 |
| Average processing time | ~30 ms (bge-small, CPU) / ~7 ms (offline) |

These numbers are the same with both embedding back-ends. The full per-question table is in `evaluation/results/eval_report.md`.

---

## 7. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `SOP_EMBED_BACKEND` | `auto` | `auto` (HuggingFace, falling back to offline), `huggingface`, `local` |
| `SOP_HF_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `SOP_LLM_BACKEND` | `none` | `none` (extractive), `openai` (needs `OPENAI_API_KEY` and `pip install llama-index-llms-openai`), `ollama` (needs `pip install llama-index-llms-ollama` and a running Ollama server) |
| `SOP_DATA_DIR` | `data/sops` | document folder (add your own PDF/TXT here, then `--rebuild`) |
| `SOP_STORAGE_DIR` | `storage` | persisted index |
| `SOP_LOG_FILE` | `logs/query_log.jsonl` | monitoring log |

When an LLM is configured, its numbered answer is accepted only if every step cites a source `[S#]` and has at least 50 % word overlap with that source. Otherwise the extractive answer is used. Tolerance, comparison and release questions always stay on the deterministic tools.

---

## 8. Adding your own documents

Put `.pdf` / `.txt` / `.md` files in `data/sops/` using this layout, then run `python cli.py --rebuild --list-docs`:

```
Document ID: SOP-XX-011
Title: My Procedure
Revision: A

1. Purpose
...
2. Inspection Steps
2.1 First step ...
2.2 Second step ...
```

---

## 9. Known limitations

* The router and the limit extractor are rule-based. They are transparent and easy to test, but they only understand the wording patterns listed in `router.py` / `limits.py`.
* The offline embedding is lexical (hashing) rather than truly semantic. Synonym expansion makes up for this on defect terms, and the HuggingFace model is used whenever it is available.
* The assistant only reports what the SOPs say. It does not resolve contradictions or judge parts.
