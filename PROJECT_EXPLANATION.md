# Project Explanation (Simple English)
## Manufacturing Quality SOP Assistant

---

## 1. What is the problem?

A small factory has many quality documents (called **SOPs**, Standard Operating Procedures). For example:

- how to check raw material when it arrives
- how to check finished products before shipping
- what to do when a measuring tool fails calibration
- defect codes (scratch, wrong size, missing label ...)
- what to do when a bad part is found

A **new quality engineer** does not know all these documents. Finding the correct step takes a long time.

## 2. What did we build?

We built an **AI assistant**. The engineer types a question in normal English, and the assistant:

1. **finds** the correct SOP and section
2. **gives the steps** as a numbered list
3. **shows the source** (document name, section, page) for every step
4. **shows the original text** so the engineer can check it
5. **says "cannot be verified"** if the answer is not in the documents
6. **shows both documents** if two documents say different things
7. **never approves or releases a batch.** Only the Quality Manager can do that.

Example:

> **Question:** A box arrived with no sticker on it. What should I do?
>
> **Answer:**
> Recognised defect: LB-03 Missing or incorrect label [REF-DEF-005 §3]
> 1. Stop the movement of the package and tag it HOLD. [SOP-LBL-008 §4]
> 2. Do not create a replacement label from memory ... [SOP-LBL-008 §4]
> 3. Inform the Stores In-charge and the Line Quality Inspector. [SOP-LBL-008 §4]
> 4. Raise an NCR as per SOP-NCR-006 ... [SOP-LBL-008 §4]

The user said "sticker", the document says "label", and the assistant still found the right procedure.

---

## 3. What is RAG? (the main idea)

**RAG = Retrieval-Augmented Generation**

In simple words: **first search, then answer.**

```
Question  ──►  SEARCH the documents  ──►  take the best paragraphs  ──►  make the answer only from them
```

Why not just ask ChatGPT?
- ChatGPT does not know **our factory's** documents.
- ChatGPT may **invent** an answer (this is called *hallucination*).
- In quality work a wrong number (tolerance) can be dangerous.

With RAG, every answer comes from **our own documents**, and we can show **where** it came from.

---

## 4. How does it work? (step by step)

### Step 1: Read the documents (`ingest.py`)
- We have **10 documents**: 5 PDF files and 5 text files.
- The program reads every PDF page by page (using the `pypdf` library).
- It finds the **document ID** (e.g. SOP-QA-001), the **title** and the **sections** (e.g. "5. Inspection Steps").
- Each section becomes **one small piece**, called a **node** or **chunk**.
- Each piece remembers: document ID, title, section name, page number and file name. This is how we can show sources later.

> Why cut by section? A section is a complete idea (all the steps together). Citations are also exact: "Section 5, page 1-2".

### Step 2: Convert text to numbers (`embeddings.py`)
- A computer cannot understand words directly, so we convert each piece of text into a list of numbers called an **embedding** (a vector).
- Texts with **similar meaning** get **similar numbers**.
- We use the free model **BAAI/bge-small-en-v1.5**.
- If there is no internet, the program uses its **own offline method**, so it always runs.

### Step 3: Store them in an index (`ingest.py`)
- All pieces and their numbers are stored in a **Vector Index** (from the **LlamaIndex** library).
- The index is saved in the `storage/` folder, so the next start is fast.

### Step 4: Understand the question (`router.py`)
The assistant first decides **what type of question** it is:

| Question type | Example | What it does |
|---|---|---|
| **Procedure** | "What are the steps for final inspection?" | numbered steps |
| **Comparison** | "What is different between incoming and final inspection?" | side-by-side table |
| **Limit / tolerance / approval** | "What is the maximum scratch length?" | checks if the value is really written in the documents |
| **Release request** | "Please approve batch 45" | politely **refuses** |

This decision-making part is the **"lightweight agent"**.

### Step 5: Search (`retrieval.py`)
- The question is converted to numbers in the same way.
- The index finds the pieces with the **closest numbers** (most similar meaning). This is measured with **cosine similarity**.
- Then we **re-rank** (re-order) the results with simple extra points:
  - the same important words appear in the piece
  - the piece contains the defect code (e.g. SC-01)
  - the section name matches the question ("evidence" → *Acceptable Evidence* section, "who to inform" → *Escalation* section)
  - the user named the document ("final" → Final Inspection SOP)

### Step 6: Understand shop-floor words (`vocabulary.py`)
People use different words for the same thing. We map them:

| User says | Assistant understands |
|---|---|
| scuff, scrape, rubbing mark | **Scratch** (SC-01) |
| oversize, undersize, out of tolerance, wrong size | **Dimensional mismatch** (DM-02) |
| no sticker, label missing, barcode wrong | **Missing label** (LB-03) |
| failed calibration, overdue, out of cal | **Failed calibration** (CF-04) |

### Step 7: Make the answer (`generation.py`)
- The answer is built **only from the found sections**. Sentences are copied, not invented.
- Every step gets a **citation** like `[SOP-QA-001 §5. Inspection Steps, p.1-2]`.
- Documents that are mentioned in the passage (like "see SOP-NCR-006") are shown as the **next documented step**.

### Step 8: Safety checks (`limits.py`, `guardrails.py`)
- **Missing value:** if someone asks for a torque tolerance and no document has it, the answer is **"This cannot be verified"**. No number is guessed.
- **Contradiction:** the Defect Code Reference says the max scratch length is **5 mm**, but the Surface Inspection SOP says **3 mm**. The assistant shows **both** and does **not choose** one.
- **No false claims:** the assistant never says "I inspected the part" or "the batch is released". A warning note is added under every answer.

### Step 9: Record everything (`monitoring.py`)
Every question is saved in `logs/query_log.jsonl` with:
question type, documents used, answer status, and time taken (milliseconds).

### Step 10: Test the quality (`evaluation/`, `tests/`)
- **16 test questions** with the correct expected result (`evaluation/test_questions.json`).
- They include one **missing tolerance** question and two **contradiction** questions.
- We measure:
  - **Routing accuracy**: was the question type detected correctly?
  - **Hit@3**: was the correct document in the top 3 search results?
  - **Citation correctness**: does every citation point to a real passage?
  - **Groundedness**: is every answer sentence really present in the source?
- Result: **16 / 16 passed.**
- **72 automatic tests** (pytest) check the whole program.

### Step 11: The screen (`app.py`)
A web page made with **Streamlit**:
- a **question box**
- the **answer** on the left
- the **sources panel** on the right (click to read the original text)
- tab **Quality checks**: run the 16 test questions
- tab **Query log**: see all past questions

---

## 5. Full flow in one picture

```
 10 SOP documents (PDF + TXT)
          │
          ▼
  Read → cut into sections → convert to numbers → store in Vector Index
                                                          │
 User question ──► Router decides type ───────────────────┤
                      │                                   │
     ┌────────────────┼──────────────────┬────────────────┘
     ▼                ▼                  ▼                 ▼
 Procedure       Comparison        Limit check       Release request
 (steps)         (2 documents)     (found / missing  (refuse + show
                                    / contradiction)  who can release)
     └────────────────┴──────────────────┴─────────────────┘
                      ▼
       Safety check + warning note → save to log
                      ▼
        Answer + sources shown on screen
```

---

## 6. Technologies used

| Tool | Used for |
|---|---|
| **Python** | programming language |
| **LlamaIndex** | documents, nodes, vector index, retriever, metadata filter, tools |
| **HuggingFace bge-small** | embeddings (text → numbers) |
| **pypdf** | reading PDF files |
| **Streamlit** | web user interface |
| **pytest** | automatic testing |
| **reportlab** | creating the sample PDF files |

No paid API key is needed.

---

## 7. How to run

```bash
pip install -r requirements.txt          # install once
streamlit run app.py                     # open the web app
python cli.py --interactive              # or ask in the terminal
python -m pytest                         # run 72 tests
python -m evaluation.run_eval            # run 16 test questions
```

---

## 8. Short answers for the viva

| Question | Simple answer |
|---|---|
| What is RAG? | First search the documents, then answer only from the found text. |
| What is an embedding? | A list of numbers that represents the meaning of a text. |
| What is a vector index? | A store of embeddings that can quickly find the most similar texts. |
| What is cosine similarity? | A score that shows how similar two embeddings are (closer to 1 = more similar). |
| What is a chunk / node? | A small piece of a document. Here one piece = one SOP section. |
| What is metadata? | Extra information saved with each piece: document ID, section, page. |
| What is hallucination? | When an AI invents an answer that is not true. |
| How do you stop hallucination? | Answers are copied from the documents, limits are checked, and the sources are shown. |
| What is the router / agent? | The part that decides the question type and picks the right tool. |
| What if information is missing? | It says "cannot be verified" and gives no number. |
| What if documents disagree? | It shows both values with sources and does not choose. |
| Can it release a batch? | No. Only the Quality Manager can (SOP-REL-010). The assistant refuses. |
| Why LlamaIndex? | It gives ready-made tools for loading documents, indexing, searching and filtering. |
| How do you test it? | 16 evaluation questions + 72 pytest tests. |
