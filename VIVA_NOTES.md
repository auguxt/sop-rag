# Viva preparation notes

## A. Demo script (about 8 minutes)

Run these in order. For the CLI add `--sources` to show the passages. In the UI, use the sidebar example buttons.

| # | Type in | What to point out |
|---|---|---|
| 0 | `python cli.py --rebuild --list-docs` | 10 documents (5 PDF, 5 TXT), 67 section nodes, embedding model name, index saved to `storage/` |
| 1 | What are the steps for incoming material inspection? | numbered steps, each with `[SOP-QA-001 §5. Inspection Steps, p.1-2]`, related SOPs as next documented step |
| 2 | What evidence is acceptable to show final inspection was completed? | section-intent re-ranking picks *Acceptable Evidence* |
| 3 | Who should I inform when a defect is found during final inspection? | notification matrix (SOP-NCR-006) + final inspection escalation (SOP-QA-002) |
| 4 | I found some scuff marks on the cover surface, what do I do? | "scuff" recognised as scratch, SC-01 |
| 5 | A box arrived with no sticker on it. What should I do? | "no sticker" recognised as missing label, LB-03 |
| 6 | Our micrometer failed its calibration check - what now? | failed calibration CF-04 → SOP-CAL-004 §6 |
| 7 | What is different between incoming and final inspection? | two-column comparison, AQL 1.0 vs 0.65, nothing invented |
| 8 | What is the torque tolerance for M8 fasteners on the gearbox cover? | **cannot be verified**: 'torque' is in no document, no value given |
| 9 | What is the maximum allowed scratch length on a visible Class A surface? | **conflict**: REF-DEF-005 says 5 mm, SOP-SURF-009 says 3 mm, both shown, none selected |
| 10 | Please approve batch B-2231 for shipment. | **refused**, release authority + next documented step from SOP-REL-010 |
| 11 | UI tab *Quality checks* → Run evaluation | 16/16, Hit@3, MRR, citation correctness, groundedness |
| 12 | UI tab *Query log* (or open `logs/query_log.jsonl`) | question type, route, source IDs, status, ms |
| 13 | `python -m pytest` | 72 tests pass |

---

## B. Concepts you must be able to explain

**RAG (Retrieval-Augmented Generation).** Before answering, the system *retrieves* the relevant passages from a document collection and builds the answer only from them. Benefits: answers are grounded and can be cited, the knowledge is up to date (just re-index), and hallucination drops.

**Chunking.** Documents are split into retrievable units. We chunk **by SOP section** (one node = one numbered section) instead of fixed-size windows, because sections are meaningful units (*Inspection Steps*, *Acceptable Evidence*) and give precise citations. Sections longer than 2,500 characters would be split again with LlamaIndex's `SentenceSplitter`.

**Embeddings.** A model maps text to a vector, and texts with similar meaning get vectors that point in similar directions. We use `BAAI/bge-small-en-v1.5` (384 dimensions). The offline fallback `LocalHashEmbedding` hashes word stems, bigrams and character 4-grams into 1,024 dimensions (MD5, so it is deterministic across runs).

**Cosine similarity.** `cos(a,b) = a·b / (|a||b|)` measures how close two vectors point. The vector store returns the nodes with the highest cosine similarity to the question vector.

**VectorStoreIndex (LlamaIndex).** Stores nodes and their embeddings. `index.as_retriever(similarity_top_k=k, filters=...)` returns the top-k nodes. **Metadata filters** (`ExactMatchFilter(key="doc_id", ...)`) restrict the search to one document; the comparison uses this.

**Metadata.** Each node keeps `doc_id, title, section, section_number, page_label, file_name, revision, defect_codes, references`. This is what "preserve document identity and section labels" means, and it is what makes citations possible. Unneeded keys are excluded from the embedding text (`excluded_embed_metadata_keys`).

**Persistence.** The index is saved in `storage/` with a fingerprint (MD5 of the files + embedding model + parser version). If anything changes, the index is rebuilt automatically.

**Query expansion.** Users say "scuff" while the SOP says "scratch". `vocabulary.expand_query` appends the SOP terms and the defect code to the question before embedding it.

**Re-ranking.** The vector search returns 15 candidates. Each gets a final score = normalised vector score + keyword overlap + defect-code boost + section-intent boost (sum, capped at 0.6) + named-document boost − Purpose/Scope penalty. All boosts are shown in the sources panel, so the ranking is explainable.

**Routing / lightweight agent.** A router picks one tool per question: `procedural_explanation`, `document_comparison` or `missing_information_check`, plus a refusal path for release actions. The tools are LlamaIndex `FunctionTool`s. The router is rule-based (deterministic, testable). An LLM can be asked only for questions the rules classify as plain procedure, so it can never skip the limit check or the release refusal.

**Grounding & hallucination control.**
1. Extractive generation: answer sentences are copied from the passages.
2. Optional LLM output is validated (each step must cite `[S#]` and overlap ≥ 50 % with that source), otherwise it is rejected.
3. Limit questions are answered only if a numeric statement about the same subject exists.
4. Guardrail regexes remove "I inspected / approved / released" claims.

**Missing information.** If a question word (e.g. *torque*, *flatness*) appears in **no** document, or no retrieved sentence states a value for the subject, the answer is "cannot be verified". No value is given.

**Conflict detection.** Each numeric statement is turned into `(attribute, value, unit)`, e.g. `(length, 5, mm)`. If statements about the same subject and attribute have different values in **different documents**, that is a conflict: both are shown with revision/date, and none is selected. Different inspection stages (incoming vs final) may legitimately differ, so they are not compared with each other.

**Evaluation metrics.**
* *Routing accuracy*: predicted question type == expected.
* *Hit@3*: a relevant document is in the top 3 retrieved.
* *MRR* (mean reciprocal rank): average of 1/rank of the first relevant document.
* *Citation correctness*: every `[DOC §section]` in the answer matches a passage in the sources panel (page included), the required documents are cited, and "not verifiable" answers cite nothing as support.
* *Groundedness*: share of cited statements whose words (≥ 80 % overlap) occur in the cited passage. For comparison/conflict lines, every number must occur in its cited passage.

---

## C. Code walk-through (what each file does)

| File | Key functions | Explain as |
|---|---|---|
| `ingest.py` | `load_documents`, `_logical_lines`, `parse_sections`, `build_nodes`, `build_or_load_index` | PDF text wraps lines, so `_logical_lines` joins a line onto the previous one unless it starts a heading (`5. Title`), a step (`5.1 Text`) or a bullet. Sections remember which pages they came from, giving page ranges like `1-2`. |
| `embeddings.py` | `LocalHashEmbedding`, `get_embed_model` | Subclass of LlamaIndex `BaseEmbedding`. `auto` tries HuggingFace and falls back to local on any exception. |
| `vocabulary.py` | `detect_defects`, `expand_query`, `mentioned_documents` | regex synonym lists per defect category; document aliases ("incoming" → SOP-QA-001) |
| `retrieval.py` | `SOPRetriever.retrieve`, `_rerank`, `document_sections`, `Hit.citation` | vector search → re-ranking; metadata-filtered retrieval |
| `router.py` | `rule_based_route`, `llm_route`, `route` | regex patterns for release actions, comparison wording and limit wording |
| `generation.py` | `select_hits`, `compose_procedure`, `compose_limit_answer`, `compose_release_refusal` | builds the markdown answers. `select_hits` keeps the best section plus close runners-up that were boosted for a reason of their own. |
| `comparison.py` | `compare_documents`, `section_items`, `comparison_markdown` | aligns sections by normalised title; "Same / Different / Only in X" |
| `limits.py` | `extract_facts`, `check_limits`, `unknown_terms`, `find_cross_document_conflicts` | regex `number + unit`, attribute from the words before the number, subject matching, conflict grouping |
| `guardrails.py` | `sanitize` | regex list of forbidden claims |
| `monitoring.py` | `QueryLogger.log/read/summary` | thread-safe JSONL append |
| `assistant.py` | `SOPAssistant.ask` and the 4 tool methods | glue code: route → tool → guardrail → disclaimer → log; exceptions are caught so the UI never crashes |
| `app.py` | Streamlit UI | `st.cache_resource` loads the index once; columns for answer + sources; expanders show passages |
| `evaluation/run_eval.py` | `evaluate`, `check_citations`, `groundedness` | metrics described above |

---

## D. Likely viva questions

1. **Why LlamaIndex?** It provides document/node abstractions with metadata, embedding integrations, a vector index with metadata filters and persistence, and tool abstractions for agents, which is everything this RAG pipeline needs.
2. **Why section-level chunks?** Precise citations, and complete steps stay together. Fixed-size chunks would cut a procedure in half.
3. **How do you make sure the answer is not hallucinated?** Extractive answers, validated LLM output, limit verification, guardrails and a groundedness metric in the evaluation.
4. **What happens if a tolerance is not in the documents?** The `missing_information_check` tool finds no matching numeric statement (or an unknown term) and returns "cannot be verified" with the closest passages, labelled as *not* containing the value.
5. **What if two documents disagree?** Status `conflict`: both values, exact wording, revision and date are shown, and the user is told to escalate to the document owners. No value is selected.
6. **How does "scuff marks" find the scratch procedure?** Regex synonym detection → query expansion with "scratch … SC-01" → defect-code boost in re-ranking → SC-01 line shown from REF-DEF-005.
7. **Can the assistant release a batch?** No. The router detects action requests (refusal path), SOP-REL-010 itself says software tools cannot release, the guardrail removes such claims, and every answer carries a disclaimer.
8. **What is the difference between the vector score and the final score?** The vector score is raw cosine similarity. The final score adds the explainable boosts. Both are shown in the sources panel.
9. **How would you scale this?** Use a persistent vector database (Chroma, Qdrant, pgvector), incremental ingestion, a cross-encoder re-ranker, and an LLM router/summariser behind the same validation.
10. **What do the tests cover?** Ingestion (PDF pages, sections, page ranges, persistence), defect-synonym retrieval, section retrieval, metadata filters, routing, comparisons (no invented steps), missing data, conflicts, guardrails, logging, the full evaluation set and the Streamlit UI.

---

## E. Troubleshooting

| Problem | Fix |
|---|---|
| No internet / HuggingFace install fails | Nothing to do: `auto` falls back to offline embeddings. To force it: `SOP_EMBED_BACKEND=local` or `--embed local` |
| Changed documents are not picked up | `python cli.py --rebuild` (or the *Rebuild index* button). The fingerprint normally detects changes automatically. |
| `ModuleNotFoundError: sop_assistant` | Run the commands from the project folder (`cd sop_rag`) |
| Streamlit port busy | `streamlit run app.py --server.port 8502` |
| Windows: activation blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in PowerShell, or use `cmd` |
