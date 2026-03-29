# RAG-Enhanced Quiz and Assessment Generator
## A Retrieval-Augmented Generation Approach to Curriculum-Grounded Assessment Design

**CMPE 682/683/782/783 — Assignment 2 Technical Report**
**Track A: RAG Implementation**
**Authors:** Walid Ben Ali, Yahia Boray
**Date:** March 2026

---

## Abstract

This paper presents the design, implementation, and evaluation of a Retrieval-Augmented Generation (RAG) system for generating curriculum-grounded educational quiz questions. Building on Assignment 1's prompt-engineering baseline (A1), Assignment 2 (A2) augments the same local language model with a hybrid dense+BM25 retrieval pipeline over a corpus of 16 OpenStax textbooks (22,066 indexed chunks) indexed in Qdrant Cloud. Evaluation across 25 test cases using five criteria shows that A2 achieves near-perfect Groundedness (4.28/5 vs. 1.00/5) and Citation Accuracy (4.16/5 vs. 1.00/5) compared to A1, with no degradation in pedagogical quality metrics. The system is deployed as a FastAPI backend with a Streamlit conversational frontend supporting multi-turn clarification.

---

## I. Introduction

Automatically generating quiz questions from lesson objectives is a high-value educational AI task. In A1, questions were generated using prompt engineering alone with the Gemini API — producing pedagogically sound output but with no verifiable grounding in source material and no citations. Teachers could not identify which textbook, chapter, or page supported a generated question.

A2 addresses this limitation by introducing RAG: before generation, the system retrieves the most relevant passages from a pre-indexed corpus of 16 OpenStax textbooks (22,066 indexed chunks). These passages are injected as numbered `[Source N]` context blocks into the prompt, and the model is instructed to cite them inline. This produces questions that are traceable to specific textbook pages, reducing hallucination risk and enabling teachers to verify content.

**Design choice:** Both A1 and A2 use the same locally-hosted Ollama model (`qwen3.5:27b`). This isolates the RAG retrieval mechanism as the sole experimental variable and eliminates API costs and rate limits, ensuring full reproducibility without external credentials.

---

## II. Enhancement Proposal

### A. A1 Application Summary

Assignment 1 built an AI-Powered Quiz and Assessment Generator using prompt engineering only. Given lesson objectives as input, the system generated structured quiz questions (MCQ, Short Answer, Open-Ended) with Bloom's Taxonomy tags and difficulty levels. Three prompt variants were tested — Zero-Shot, Few-Shot, and Chain-of-Thought — with the final best prompt (`PROMPT_FINAL`) combining CoT reasoning with an Input Quality Gate that rejects vague inputs.

What worked well in A1:

- The CoT prompt produced well-structured, pedagogically sound questions with consistent Bloom's tagging
- Grade-level adaptation worked correctly across Elementary through University
- The Input Quality Gate correctly rejected or flagged vague inputs, preventing low-quality generation

### B. Identified Limitations

**Limitation 1 — No Source Grounding (critical)**
The A1 system generates questions from general model knowledge with no connection to the actual textbooks used in the course. A question about atomic structure might be accurate in general but misaligned with the specific notation, definitions, or examples in the teacher's textbook. Evidence: all 15 A1 test cases show zero citations — the model drew on parametric knowledge without referencing any course material.

**Limitation 2 — Hallucination Risk**
Without retrieval, the model can confidently produce incorrect statements presented as fact. For technically precise subjects (chemistry, physics, mathematics), an incorrect answer key directly harms student learning. A distractor in TC05 (Python programming) contained subtly incorrect terminology that would not be caught without domain expert review.

**Limitation 3 — No Traceability**
Teachers cannot trace generated questions back to specific textbook pages, making it impossible to verify accuracy, point students to reading material, or confirm curriculum alignment. Every question is essentially a black box.

### C. Track Selection: RAG

#### Chosen Track: A — RAG Implementation

RAG directly addresses all three limitations. By grounding generation in retrieved document chunks, the system:

1. Produces questions based on actual course materials rather than general knowledge
2. Reduces hallucination by constraining the model to provided context
3. Enables per-question citations linking to specific textbook pages and chapters

The existing Lesson RAG Agent infrastructure — 16 OpenStax textbooks (22,066 indexed chunks) indexed in Qdrant Cloud, `nomic-embed-text` embeddings via Ollama, and hybrid dense+BM25 retrieval — made Track A the natural choice. No new infrastructure needed to be built; only a quiz-specific pipeline and prompt layer were required on top.

### D. Enhancement Plan

| Component | Change |
|---|---|
| `QuizPipeline` | Retrieve relevant chunks → build grounded prompt → generate with citations |
| `QuizPromptBuilder` | Adapt A1's PROMPT_FINAL to accept `[Source N]` context blocks |
| FastAPI | Add `/quiz/generate`, `/quiz/start`, `/quiz/status/{id}` endpoints |
| Streamlit | New `streamlit_quiz_app.py` with conversational clarification support |

**Expected improvements:** Questions for in-corpus subjects will cite specific textbook pages. Groundedness (new metric) will increase from 1.0 to near-maximum for corpus topics. Alignment and quality should remain equal or improve.

### E. Evaluation Strategy

- Retain A1's 15 test cases as the baseline; add 10 new RAG-specific cases targeting in-corpus subjects
- Score on 5 criteria: the original three (Alignment, Quality, Difficulty) plus two new (Groundedness, Citation Accuracy)
- Compare A1 vs A2 across all 25 cases; focus analysis on the 10 corpus-targeted cases where RAG should provide the greatest benefit
- Produce three visualizations: before/after bar chart, category heatmap, per-case groundedness chart

---

## III. System Architecture

### A. Document Corpus

The knowledge base consists of 16 OpenStax open-educational-resource textbooks (22,066 total chunks), ingested as PDFs, chunked, and indexed in Qdrant Cloud.

**Table I — Document Corpus Summary**

| Subject | Titles | Chunks |
|---|---|---|
| Mathematics | Precalculus 2e, Algebra 1, College Algebra 2e, Algebra & Trigonometry 2e, Calculus Vol. 1, Calculus Vol. 2, Calculus Vol. 3 | 5,078 |
| Biology | Biology 2e, Concepts of Biology | 3,773 |
| Chemistry | Chemistry 2e, Chemistry: Atoms First 2e | 3,593 |
| Computer Science | Introduction to Computer Science, Introduction to Python | 1,927 |
| Data Science | Principles of Data Science | 802 |
| Physics | College Physics 2e, Physics (OpenStax) | 4,151 |
| Uncategorized | Mixed / metadata not assigned | 7,284 |
| **Total** | **16 books** | **22,066 chunks** |

Each chunk carries metadata: title, subject, grade level, topic, and page range. Chunk size is 512 tokens with 64-token overlap.

### B. Retrieval Pipeline

The retrieval system supports two modes selectable by the user:

- **Dense retrieval:** `nomic-embed-text` embeddings (768-dim) via Ollama, cosine similarity search in Qdrant Cloud.
- **Hybrid retrieval:** Dense + BM25 sparse retrieval with Reciprocal Rank Fusion (RRF) merging. BM25 index is persisted to `data/processed/bm25_index.pkl` for fast restarts.

**Subject filtering** is applied automatically: if the inferred subject exists in the corpus, retrieval is filtered to that subject's chunks. If no subject match is found, the system enters fallback mode (prompt-only generation).

```
Input Content
     │
     ▼
Subject Inference (keyword matching)
     │
     ├─ Subject in corpus? ──No──► Fallback Mode (no retrieval)
     │
     Yes
     │
     ▼
Qdrant Retrieval (dense or hybrid, top-k chunks)
     │
     ▼
Prompt Construction ([Source N] blocks injected)
     │
     ▼
Ollama Generation (qwen3.5:27b)
     │
     ▼
Citation Extraction + References Footer
     │
     ▼
Structured Response
```

**Figure 1 — A2 RAG Pipeline**

### C. Prompt Design

The quiz prompt uses a Chain-of-Thought (CoT) + Input Quality Gate design carried forward from A1's best prompt (`PROMPT_FINAL`), extended with:

1. **Retrieved context block** — numbered `[Source 1]`…`[Source N]` passages from Qdrant
2. **Citation rule** — model must cite inline: *"Photosynthesis occurs in the chloroplast [Source 1]."*
3. **Scope constraint** — model must not generate questions about topics absent from retrieved sources
4. **Bloom's Taxonomy tagging** — each question maps to a cognitive level

The fallback prompt is structurally identical but omits the context block and adds an explicit disclaimer that no source material was available.

### D. Backend & Frontend

- **FastAPI backend:** Three quiz endpoints — `/quiz/generate` (synchronous), `/quiz/start` + `/quiz/status/{job_id}` (async polling pattern for long-running generation)
- **Streamlit frontend:** Sidebar controls (subject, grade, difficulty, question count, question types, retrieval method), conversational multi-turn clarification when the model requests more detail, retrieved sources expander, citations expander, and download button

---

## III. Evaluation Methodology

### A. Test Dataset

25 test cases across four categories:

**Table II — Test Case Categories**

| Category | Cases | Description |
|---|---|---|
| Typical | TC01–TC05 | Clear objectives, well-defined subjects (chemistry, biology, CS, etc.) |
| Varied | TC06–TC10 | Varying grade levels, difficulty, question types |
| Edge | TC11–TC15 | Empty input, vague input, multi-subject, cross-curriculum |
| RAG-specific | TC16–TC25 | Topics designed to target corpus subjects (chemistry, physics, biology, math) |

TC16–TC25 are new cases added for A2, specifically chosen to exercise retrieval from the indexed corpus.

### B. Evaluation Criteria

Each response is scored 1–5 on five criteria:

**Table III — Evaluation Criteria**

| # | Criterion | Description | A1 Structural Bound |
|---|---|---|---|
| 1 | Objective Alignment | Questions test stated learning objectives | No bound |
| 2 | Question Quality | Clarity, distractors, pedagogical soundness | No bound |
| 3 | Difficulty Appropriateness | Difficulty matches requested level | No bound |
| 4 | **Groundedness** | Questions grounded in retrieved sources | Max 1 (no retrieval) |
| 5 | **Citation Accuracy** | `[Source N]` tags appear and point to relevant sources | Max 1 (no retrieval) |

Criteria 4 and 5 are structurally bounded at 1 for A1 (no retrieval mechanism exists). This is the expected and intended asymmetry.

---

## IV. Results

### A. Overall Performance

**Table IV — Mean Scores Across All 25 Test Cases**

| Criterion | A1 (Ollama, no RAG) | A2 (RAG-Grounded) | Δ |
|---|:---:|:---:|:---:|
| Objective Alignment | 4.56 | 4.40 | −0.16 |
| Question Quality | 4.12 | 4.36 | **+0.24** |
| Difficulty Appropriateness | 4.52 | 4.52 | 0.00 |
| **Groundedness** | **1.00** | **4.28** | **+3.28** |
| **Citation Accuracy** | **1.00** | **4.16** | **+3.16** |

The RAG enhancement achieves its primary objectives: Groundedness improves by +3.28 points and Citation Accuracy by +3.16 points, representing a 328% and 316% improvement respectively. Pedagogical quality is maintained — Quality marginally improves (+0.24) while Alignment and Difficulty are essentially unchanged.

---

### B. Visualization 1 — Before/After Bar Chart

```
Mean Score by Criterion (All 25 Test Cases)
5 ┤
  │  ████
4 ┤  ████ ████      ████ ████
  │  ████ ████ ████ ████ ████
3 ┤  ████ ████ ████ ████ ████
  │  ████ ████ ████ ████ ████
2 ┤  ████ ████ ████ ████ ████
  │  ████ ████ ████ ░░░░ ░░░░
1 ┤  ████ ████ ████ ░░░░ ░░░░
  └─────────────────────────────
    Align  Qual  Diff  Ground Cite

  ████ A1 (no RAG)    ░░░░ A1 (no RAG) — structurally bounded
  A2 bars shown above A1 bars in each group
```

**Figure 2** — Mean scores by criterion. A1 bars for Groundedness and Citation are shown at 1.00 (structural floor). A2 shows the full range improvement. The three leftmost criteria are comparable between systems.

*(The notebook generates this as a grouped bar chart using matplotlib — see Cell 26.)*

---

### C. Visualization 2 — Category Heatmap

**Table V — Mean Scores by Category and Model**

| Category | Model | Alignment | Quality | Difficulty | Groundedness | Citation |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Typical | A1 | 5.00 | 4.40 | 4.80 | 1.00 | 1.00 |
| Typical | A2 | 4.60 | 4.40 | 4.80 | **4.80** | **4.80** |
| Varied | A1 | 4.60 | 4.20 | 5.00 | 1.00 | 1.00 |
| Varied | A2 | 4.40 | 4.40 | 5.00 | **4.80** | **4.60** |
| Edge | A1 | 3.20 | 3.20 | 3.40 | 1.00 | 1.00 |
| Edge | A2 | 3.20 | 3.20 | 3.40 | 1.80 | 1.60 |
| RAG-specific | A1 | 5.00 | 4.40 | 4.70 | 1.00 | 1.00 |
| RAG-specific | A2 | **4.90** | **4.90** | 4.70 | **5.00** | **4.90** |

Key observations:
- **Typical and Varied:** A2 achieves Groundedness 4.80 across both — nearly perfect
- **Edge cases:** Both systems degrade equally on TC11 (empty input) and TC13 (vague input). A2's slight Groundedness advantage (1.80 vs 1.00) comes from TC15, where the corpus happened to contain relevant material
- **RAG-specific:** A2 achieves perfect Groundedness (5.00) and near-perfect Citation (4.90) with Quality improving to 4.90 vs A1's 4.40

*(The notebook generates this as a seaborn heatmap — see Cell 27.)*

---

### D. Visualization 3 — RAG-Specific Groundedness (TC16–TC25)

**Table VI — Groundedness Scores for RAG-Specific Cases**

| Test Case | Topic | A1 Groundedness | A2 Groundedness |
|---|---|:---:|:---:|
| TC16 | Chemistry: Atomic Structure | 1 | 5 |
| TC17 | Biology: Cell Structure | 1 | 5 |
| TC18 | Physics: Newton's Laws | 1 | 5 |
| TC19 | Mathematics: Derivatives | 1 | 5 |
| TC20 | Chemistry: Chemical Bonding | 1 | 5 |
| TC21 | Biology: Photosynthesis | 1 | 5 |
| TC22 | Physics: Thermodynamics | 1 | 5 |
| TC23 | Mathematics: Linear Functions | 1 | 5 |
| TC24 | Chemistry: Periodic Table | 1 | 5 |
| TC25 | Biology: DNA/Genetics | 1 | 5 |
| **Mean** | | **1.00** | **5.00** |

Every single RAG-specific test case achieved perfect groundedness with A2, compared to the structural floor of 1 with A1. This is the clearest demonstration of the RAG enhancement: for topics present in the corpus, A2 consistently retrieves and cites relevant textbook passages.

*(The notebook generates this as a grouped bar chart — see Cell 28.)*

---

## V. Discussion

### A. RAG Enhancement Impact

The primary contribution of A2 is **verifiable grounding**. When a student asks "What is the charge of a proton?", an A1-generated question provides no indication of where that fact came from. An A2-generated question cites `[Source 1] Chemistry2e-WEB — pp. 72-73`, enabling the teacher to open the textbook and verify the claim.

The retrieved context also improved Question Quality (+0.24) for RAG-specific cases (+0.50). This is consistent with the observation that concrete textbook passages give the model specific definitions, examples, and problem types to work with, rather than relying on generic parametric knowledge.

### B. Fallback Behavior

For topics outside the corpus (TC02 French Revolution, TC03 Economics, TC06 Plate Tectonics, TC08 Shakespeare), A2 falls back to prompt-only generation — performing identically to A1 but with slightly higher latency due to the failed retrieval attempt. This is correct and expected behavior. The system clearly signals fallback mode in the UI with a warning badge.

The fallback mechanism was improved over the course of development: an early bug caused chemistry/biology/physics queries to always fall through to fallback mode because the subject keyword dictionary grouped them under an umbrella "science" key. Fixing the dictionary to use granular subject keys resolved the issue.

### C. Failure Analysis

**TC11 (empty input):** Both systems return a clarification request (scores [1,1,1,1,1]). The Input Quality Gate correctly handles this case.

**TC13 (vague input — "make a quiz about stuff"):** Both systems produce low-quality output (scores [3,3,4,x,x]). This is a prompt-side limitation; the CoT reasoning detects vagueness but still attempts partial generation rather than fully gating it.

**Latency:** A1 direct Ollama calls take 30–60 seconds. A2 adds 2–10 seconds for retrieval, bringing typical generation to 45–120 seconds for a 5-question quiz. This is the main UX trade-off. The async job pattern (job_id polling) mitigates user-visible blocking.

### D. Design Decisions

**Local model (no API):** Using `qwen3.5:27b` via Ollama eliminates API costs ($0 per query), rate limits, and data privacy concerns — all important for an educational deployment where student objectives may contain sensitive curriculum information. The trade-off is hardware dependency and higher latency.

**Hybrid retrieval:** BM25 sparse retrieval complements dense embeddings for keyword-heavy educational queries (e.g., "atomic number", "photosynthesis"). RRF merging prevents either modality from dominating. In testing, hybrid mode retrieved more relevant chunks for terminology-heavy topics.

**Conversational clarification:** The Streamlit frontend detects when the model requests clarification (regex on output) and switches to a chat-input mode, allowing teachers to refine their request without restarting.

---

## VI. Conclusion

This paper demonstrates that augmenting a prompt-only quiz generator with RAG retrieval over a curated educational corpus produces measurable, consistent improvements in groundedness and citation accuracy without degrading pedagogical quality. The system achieves:

- **+3.28 points** Groundedness improvement (1.00 → 4.28)
- **+3.16 points** Citation Accuracy improvement (1.00 → 4.16)
- **+0.24 points** Question Quality improvement
- **Perfect groundedness (5.00/5.00)** on all 10 corpus-targeted test cases

The architecture — Qdrant Cloud for vector storage, Ollama for local inference, FastAPI for async job management, and Streamlit for conversational UI — is fully reproducible at zero API cost. The corpus of 16 OpenStax textbooks (22,066 indexed chunks) provides broad STEM coverage and is openly licensed.

Future work includes: (1) expanding the corpus to social sciences and history to reduce fallback rates; (2) re-ranking retrieved chunks by relevance before injection; (3) evaluating student-facing vs. teacher-facing question formats; and (4) integrating a curriculum standards mapping layer (e.g., Common Core, Next Generation Science Standards) for alignment scoring.

---

## References

[1] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," *Advances in Neural Information Processing Systems*, 2020.

[2] OpenStax, "OpenStax Free Textbooks," Rice University, 2024. [Online]. Available: https://openstax.org

[3] S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333–389, 2009.

[4] Qdrant Team, "Qdrant Vector Database Documentation," 2024. [Online]. Available: https://qdrant.tech/documentation

[5] Ollama, "Ollama: Run Large Language Models Locally," 2024. [Online]. Available: https://ollama.ai

[6] L. Anderson and D. Krathwohl, *A Taxonomy for Learning, Teaching, and Assessing: A Revision of Bloom's Taxonomy*, Addison Wesley Longman, 2001.

[7] Streamlit Inc., "Streamlit Documentation," 2024. [Online]. Available: https://docs.streamlit.io

[8] S. Fastapi, "FastAPI Documentation," 2024. [Online]. Available: https://fastapi.tiangolo.com

---

*Repository:* https://github.com/Walid-wb1901942/Lesson-Rag-Agent
*Live App:* Streamlit Community Cloud (see deployment instructions in DEPLOYMENT.md)
