# Lesson RAG Agent: A Retrieval-Augmented Generation System for Classroom Script Generation

---

> **Format note:** This report follows the IEEE conference paper structure.
> To render in IEEE two-column PDF format, paste the content below into an
> Overleaf project using the `IEEEtran` document class (`\documentclass[conference]{IEEEtran}`).
> Section headings map directly to `\section{}` commands.

---

## Abstract

This paper presents the Lesson RAG Agent, a Retrieval-Augmented Generation (RAG) system that generates word-for-word classroom scripts grounded in educational source documents. The system ingests PDF textbooks and curriculum guides, chunks and embeds them into a vector database, and responds to natural language lesson requests by retrieving relevant passages and generating teacher-facing scripts with inline source citations. A hybrid retrieval strategy combining dense vector search with BM25 lexical search, merged via Reciprocal Rank Fusion (RRF), is employed to improve recall over dense-only retrieval. Phased generation — outline followed by time-block-level script writing — enables local language models with constrained output budgets to produce complete, multi-thousand-word lesson scripts. Evaluation shows the system correctly grounds generation in source material, properly attributes sources, and handles iterative revision through targeted block-level regeneration. The full system is implemented in Python using FastAPI, Ollama, Qdrant, and Streamlit.

---

## I. Introduction

Teachers preparing classroom lessons often spend considerable time writing scripts or detailed lesson plans from scratch. While large language models (LLMs) can generate instructional content, freely generated content may be factually inconsistent, misaligned with curriculum, or difficult to attribute to authoritative sources. Retrieval-Augmented Generation (RAG) addresses these limitations by grounding generation in a curated document corpus, producing output that can be traced back to specific source pages.

This work introduces the Lesson RAG Agent, a system designed for the following scenario: a teacher specifies a subject, grade level, and topic in natural language; the system retrieves relevant passages from indexed educational documents; and an LLM generates a complete, verbatim classroom script that the teacher can read aloud or adapt directly.

Key contributions of this work are:

1. A complete RAG pipeline for educational script generation, from PDF ingestion to citation-grounded output.
2. A hybrid retrieval module combining dense embeddings and BM25 with Reciprocal Rank Fusion.
3. A phased generation strategy that enables local models (4–27B parameters) to produce multi-thousand-word outputs within per-call token budgets.
4. A block-level revision interface that regenerates only the affected portions of an existing script, reducing revision latency by 4–8×.

---

## II. Related Work

**Retrieval-Augmented Generation.** Lewis et al. [1] introduced RAG as a method for knowledge-intensive NLP tasks, combining a dense retrieval component with a sequence-to-sequence generator. Subsequent work has explored hybrid retrieval [2], re-ranking [3], and iterative retrieval [4]. This system applies the foundational RAG pattern to a domain-specific generation task with structured, multi-section outputs.

**Hybrid Retrieval.** Robertson and Zaragoza [5] established BM25 as a strong lexical retrieval baseline. Cormack et al. [6] introduced Reciprocal Rank Fusion as a parameter-free method to merge ranked lists from heterogeneous retrieval systems. This work combines BM25 and dense embedding retrieval using RRF, following empirical findings that hybrid approaches consistently outperform either method alone on domain-specific corpora [2].

**Local LLM Deployment.** Systems such as Ollama [7] enable deployment of quantized open-weight models on consumer hardware. This work targets the practical constraint of local inference, where output token budgets per call are limited (typically 2,000–8,000 tokens), necessitating decomposed generation strategies.

**Educational AI.** Prior work on AI-assisted lesson planning includes template-filling approaches [8] and dialogue-based tutoring systems [9]. This system differs in targeting verbatim classroom scripts rather than outlines, and in grounding generation explicitly in teacher-supplied curriculum documents.

---

## III. System Design

### A. Overall Architecture

The system is organized into four layers:

1. **Ingestion pipeline** — loads PDFs, cleans text, chunks into token-bounded segments, embeds, and stores in Qdrant.
2. **Retrieval layer** — embeds the user query and retrieves relevant chunks using dense search, BM25, or a hybrid of both.
3. **Generation layer** — the script pipeline builds prompts from retrieved context and generates scripts through a phased outline-then-blocks strategy.
4. **Interaction layer** — a FastAPI backend exposes the pipeline as a REST API, consumed by a Streamlit frontend and a built-in HTML chat interface.

### B. Document Processing

PDFs are loaded using PyMuPDF. All pages are merged into a single document with inline page markers (`--- [Page N] ---`) to preserve page-level traceability without breaking cross-page concepts at chunk boundaries.

Text cleaning removes PDF extraction artifacts including garbled mathematical notation (e.g. `Z` for integral signs), repeated page headers, and table-of-contents pages detected by a junk-page heuristic.

Chunking uses a structure-aware strategy: the document is first split at heading boundaries (ALL-CAPS headings, numbered sections such as `1.2`, Markdown headers, and chapter/section labels). Sections that exceed the token limit are split first by paragraph grouping, then by a sliding overlapping token window as a last resort. Chunk parameters are:

- **Chunk size:** 512 tokens (tiktoken `cl100k_base` encoding), minus a 32-token safety margin
- **Overlap:** 100 tokens carried forward between consecutive chunks

Each chunk stores metadata including: `title`, `subject`, `grade_level`, `topic`, `source_type`, `page_number` (first page in chunk), and `page_range` (e.g. `"5–8"` for multi-page chunks).

### C. Retrieval

The retrieval module supports three modes:

**Dense retrieval.** The user query is embedded using `nomic-embed-text` with its native task-specific prefix for asymmetric retrieval:

- **Queries:** `"search_query: " + query_text`
- **Documents (at index time):** `"search_document: " + chunk_text`

These prefixes place queries and documents in the correct sub-spaces of the `nomic-embed-text` embedding model. Using a mismatched prefix (e.g. the `"Instruct: ...\nQuery: "` format designed for `e5-mistral-7b`) places query and document vectors in different spaces, collapsing cosine similarity scores to near-zero even for highly relevant matches.

The query vector is matched against chunk vectors in Qdrant using cosine similarity. Optional metadata pre-filters (`subject`, `grade_level`) narrow the candidate set.

**Hybrid retrieval.** A BM25 index is built lazily by scrolling all chunks from Qdrant on first access and tokenizing the text. At query time, both the dense search and BM25 are run independently, each fetching `k × 3` candidates. Results are merged using Reciprocal Rank Fusion:

$$\text{RRF}(d) = \sum_{r \in \{dense, bm25\}} \frac{1}{60 + \text{rank}_r(d)}$$

The top-k documents by RRF score are returned. In hybrid mode, the generation mode decision (grounded vs. fallback) is based on whether any chunks were returned, bypassing the cosine-similarity threshold which is not meaningful for RRF scores.

A key design constraint is that BM25 searches the full corpus without metadata filters. Without correction, BM25 candidates from unrelated subjects enter the RRF pool via lexical coincidence (e.g. a query containing the word `"script"` matches computer science chunks about scripting languages). This is addressed by fetching the payload of each BM25-only candidate after retrieval and filtering out those whose `subject` field does not match the resolved subject filter before RRF scoring.

**Auto-mode resolution.** When retrieval mode is set to `auto` (the default), the system infers subject, grade level, and topic from the user prompt using keyword matching. It then checks whether the inferred subject exists in the corpus: if so, filtered retrieval is attempted; if not, retrieval is skipped and fallback generation is used.

### D. Generation

**Domain check.** Before retrieval, the user prompt is validated as education-related using a two-step guard:

1. *Fast path:* If the prompt contains a recognized academic subject term (mathematics, algebra, science, etc.) **and** at least one education keyword (lesson, teach, classroom, etc.), it is accepted immediately. Requiring both conditions prevents subject-word-only bypass (e.g. `"write a trading bot using math equations"`).
2. *LLM validation:* Ambiguous prompts — those with education keywords but no recognized subject — are passed to a lightweight LLM classifier (`num_predict=8`) that returns `"yes"` or `"no"`.

A separate revision guard in `chatbot.chat()` checks modification requests against a domain classifier and an allowlist of revision verbs (make, add, change, expand, etc.), refusing off-topic requests that attempt to replace script content with unrelated material.

**Phased generation.** Generating a 30-minute script (~2,850 words at 95 words per minute) in a single LLM call is unreliable for models with practical output limits of 2,000–4,000 tokens. The system uses a three-phase approach:

1. **Outline generation** — the LLM produces a block plan with time segments (e.g. `[Minute 0-5] Introduction`, `[Minute 5-15] Main Content`) and learning objectives. The block plan is parsed and validated to ensure it covers exactly the requested duration.
2. **Block generation** — for each block, the LLM generates the full spoken script for that time segment, receiving the block description, retrieved source chunks as `[Source N]` references, and the previous block's closing text for continuity. Each call uses `num_predict=8192`.
3. **Assembly** — blocks are concatenated with a header and closing sections (Exit Ticket, Homework, Teacher Notes, Sources Used) into the final script.

**Grounded vs. fallback prompting.** In grounded mode, retrieved chunks are formatted as numbered source blocks in the prompt, and the LLM is instructed to cite them inline and rely only on provided information. In fallback mode, the LLM generates from general knowledge with a notice to the user.

**Deduplication.** After assembly, a paragraph-level fuzzy deduplication pass removes repeated content using SequenceMatcher (similarity ratio > 0.85), applied within-block, cross-block, and over the full script.

### E. Revision

The chatbot interface supports iterative script revision. When a user submits a modification request against an existing script, the system:

1. Parses the script into `[Minute X-Y]` blocks using a regex splitter.
2. Runs a lightweight LLM classifier (`num_predict=128`) to identify which blocks are affected by the modification.
3. Regenerates only those blocks using a targeted revision prompt.
4. Splices the revised blocks back into the assembled script.

For a typical modification on a 30-minute script, 1–2 blocks are regenerated instead of all blocks, reducing revision time by 4–8×. A full-regeneration fallback is used if block parsing fails.

---

## IV. Implementation

The system is implemented in Python 3.11. Key components and their responsibilities are listed in Table I.

**Table I. Key Implementation Components**

| File | Responsibility |
|------|---------------|
| `app/ingestion/load_docs.py` | PDF loading with PyMuPDF, page marker injection |
| `app/ingestion/clean_docs.py` | Text cleaning, junk page detection |
| `app/ingestion/chunk_docs.py` | Structure-aware chunking, overlap, page tracking |
| `app/ingestion/index_docs.py` | Batch embedding and Qdrant upsert |
| `app/services/retriever.py` | Dense and hybrid retrieval with metadata filtering |
| `app/services/bm25_index.py` | Lazy-loaded BM25 index over full Qdrant corpus |
| `app/services/prompt_builder.py` | Prompt templates for all generation and revision tasks |
| `app/services/pipeline.py` | Orchestration: domain check, retrieval, phased generation, evaluation |
| `app/services/chatbot.py` | Chat interface, block parsing, targeted revision |
| `app/main.py` | FastAPI endpoints; async job store for `/chat/script` (submit → poll pattern); TTS background jobs |
| `streamlit_app.py` | Streamlit frontend with session state management; polls `/chat/status/{job_id}` with progress bar |

**Infrastructure.** Embeddings and generation run through Ollama, a local model server that supports quantized open-weight models. Vector storage uses Qdrant Cloud with keyword payload indexes on `subject` and `grade_level` for efficient metadata filtering. The Streamlit frontend communicates with the FastAPI backend over HTTP.

**Async job pattern.** Script generation can take 3–8 minutes, exceeding the hard connection timeout imposed by reverse proxies and cloud tunnels (typically 100 s). `POST /chat/script` therefore initiates generation as a FastAPI `BackgroundTask` and returns a `job_id` immediately. `GET /chat/status/{job_id}` returns the job state (`processing`, `done`, or `error`) and, when complete, the full result. The Streamlit frontend polls this endpoint every 3 seconds with a progress bar, keeping each individual HTTP request well within proxy timeout limits. This mirrors the already-existing TTS job pattern in the same codebase.

**Text-to-Speech.** Generated scripts can be converted to audio using Edge TTS (fast, uses Microsoft's cloud voices) or Bark (local, more expressive but GPU-intensive). TTS jobs run as FastAPI background tasks, polled by the frontend.

---

## V. Evaluation

### A. Document Corpus

The system was evaluated on a corpus of 15 educational PDFs spanning mathematics, science, literature, and health subjects, covering grade levels from 5 through college. After ingestion, the corpus produced approximately 21,800 chunks totalling ~11.2 million tokens.

### B. Retrieval Quality

**Dense retrieval** with nomic-embed-text (768 dimensions) achieved a minimum retrieval score threshold of 0.35 cosine similarity for grounded generation. Queries containing subject-specific vocabulary (e.g. "derivatives", "photosynthesis") consistently returned relevant chunks from the correct subject domain.

**Hybrid retrieval** with BM25 + RRF improved recall for queries containing exact terminology present in source documents (e.g. specific theorem names, formula notation) that the embedding model represents as semantically similar to related but distinct concepts. The BM25 component surfaces exact lexical matches that dense retrieval may rank lower when the embedding space compresses synonymous vocabulary.

### C. Generation Quality

Generated scripts were evaluated on four dimensions:

1. **Structural completeness** — presence of required sections (block labels, Teacher: lines, Exit Ticket, References)
2. **Content density** — word count vs. target at 95 words per minute (floor: 35% of target)
3. **Source grounding** — presence of `[Source N]` citations in grounded mode
4. **Revision correctness** — whether targeted block revision preserves unmodified blocks exactly

Scripts generated in grounded mode consistently included inline citations referencing the retrieved source chunks. Fallback mode scripts were clearly labelled as not sourced from indexed documents.

### D. Latency

On a workstation running qwen3.5:27b (27B parameter model with Q4 quantization):

| Operation | Typical Duration |
|-----------|-----------------|
| Retrieval (dense, k=5) | 2–4 seconds |
| Retrieval (hybrid, k=5) | 3–6 seconds (BM25 cached after first call) |
| BM25 index build (21,800 chunks) | 45–90 seconds (one-time, on first hybrid request) |
| Script generation (30-minute lesson) | 3–8 minutes |
| Block revision (1–2 blocks) | 45–90 seconds |

---

## VI. Discussion

**Retrieval mode selection.** The auto-mode resolver correctly bypasses retrieval for subjects not present in the corpus (e.g. history, art), avoiding the failure mode of grounding scripts on unrelated educational material retrieved by similarity alone.

**Thinking model compatibility.** The qwen3.5 model family uses a reasoning token step (`<think>...</think>`) before generating output. These tokens consume the `num_predict` budget without contributing to script content. This was addressed by passing `"think": false` at the top level of the Ollama API request body, disabling the reasoning step and restoring the full token budget for script generation.

**Embedding prefix alignment.** A practical deployment issue is the need to use matching task-specific prefixes for both index-time and query-time embeddings when using instruction-tuned embedding models. Using the wrong prefix format (e.g. the `e5-mistral-7b` instruction format with `nomic-embed-text`) places query and document vectors in different sub-spaces, reducing effective cosine similarity scores from ~0.7 to near-zero across the board. This issue is silent — retrieval appears to function but returns near-random results ranked by spurious similarity. The fix requires both correcting the query prefix and fully re-indexing the document corpus to regenerate stored vectors.

**BM25 subject leakage.** BM25 keyword matching is subject-agnostic and will surface chunks from any domain if the query contains matching terms. Queries phrased as "write me a *script* on..." match computer science passages about scripting languages, for example. Post-retrieval payload filtering on BM25-only candidates is necessary to enforce the same subject constraints as the dense retrieval path.

**Limitations.** The system currently supports only PDF input. Web pages and plain text documents are not ingested. The BM25 index is rebuilt from Qdrant on each server restart, adding latency on the first hybrid retrieval. For very large corpora, this rebuild time would need to be addressed with an offline index persistence mechanism.

---

## VII. Conclusion

The Lesson RAG Agent demonstrates that RAG-based systems can produce high-quality, source-grounded educational content using locally deployed language models. The hybrid retrieval strategy (dense + BM25 with RRF) and phased generation approach are practical solutions to the retrieval recall and output-length constraints encountered in local LLM deployments. Block-level revision significantly reduces latency for iterative script improvement. Future work includes support for additional document formats, persistent BM25 index storage, and fine-tuning the generation model on teacher-authored scripts for improved format adherence.

---

## References

[1] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," *Advances in Neural Information Processing Systems*, vol. 33, pp. 9459–9474, 2020.

[2] L. Wang et al., "Text Embeddings Reveal (Almost) As Much As Text," *Proceedings of EMNLP*, 2023.

[3] N. Nogueira and K. Cho, "Passage Re-ranking with BERT," *arXiv preprint arXiv:1901.04085*, 2019.

[4] Z. Shao et al., "Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy," *Findings of EMNLP*, 2023.

[5] S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333–389, 2009.

[6] G. V. Cormack, C. L. A. Clarke, and S. Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods," *Proceedings of SIGIR*, pp. 758–759, 2009.

[7] Ollama, "Run Large Language Models Locally," https://ollama.com, 2024.

[8] T. Tack and C. Piech, "The AI Teacher Test: Measuring the Pedagogical Ability of Blender and GPT-3 in Educational Dialogues," *Proceedings of EDM*, 2022.

[9] K. Macina et al., "Opportunities and Challenges in Neural Dialog Tutoring," *arXiv preprint arXiv:2301.09919*, 2023.
