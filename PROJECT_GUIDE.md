# Project Guide

## Purpose
This file is the day-to-day operating guide for the Lesson RAG Agent.

Use it for:
- how to run the project
- how the code is organized
- how the main data flow works
- what to update when behavior changes

This file should be updated whenever the project workflow, commands, architecture, or operating assumptions change.

## What The Project Does
The project generates literal, teacher-facing classroom scripts from educational source documents.

It works in two major stages:
1. Ingestion
   The system reads PDFs, cleans the text, chunks it, embeds it, and stores the chunks in Qdrant.
2. Retrieval + generation
   The system embeds a teacher request, retrieves relevant chunks from Qdrant, and asks Ollama to generate a lesson.

## Requirements
- A working Python virtual environment
- Ollama running locally
- The embedding model installed in Ollama
- A working Qdrant instance
- Valid settings in `.env`

## Environment Setup
The main settings live in `.env`.

Important values:
- `OLLAMA_BASE_URL`
- `OLLAMA_GENERATION_MODEL`
- `OLLAMA_EMBEDDING_MODEL`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION`
- `CHUNK_SIZE_TOKENS` (default: 512)
- `CHUNK_OVERLAP_TOKENS` (default: 100)
- `EMBEDDING_BATCH_SIZE`
- `QDRANT_UPSERT_BATCH_SIZE`

If `QDRANT_URL` or `QDRANT_API_KEY` are placeholders, indexing and retrieval will fail.
If large ingestion runs hit Ollama request-size limits, reduce `EMBEDDING_BATCH_SIZE`.

## How To Run The Project
Activate the virtual environment first:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 1. Full ingestion
Use this to process all PDFs in `data/raw`:

```powershell
python -m app.ingestion.run_ingestion
```

### 2. Index one new PDF
Use this when you add one new document and want to add it to the existing collection:

```powershell
python -m app.ingestion.run_ingestion --pdf data/raw/your_new_file.pdf
```

### 3. Reindex one existing PDF
Use this when a document already exists and you want its old chunks removed before inserting the new version:

```powershell
python -m app.ingestion.run_ingestion --pdf data/raw/your_existing_file.pdf --reindex
```

### 4. Run the API server
Use this when you want to interact with the chatbot continuously instead of editing test files:

```powershell
uvicorn app.main:app --reload
```

Default local URLs:
- Chat UI: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

Browser workflow:
1. Open `http://127.0.0.1:8000`
2. Enter the lesson request in the chat box
3. Review the generated word-for-word classroom script
4. Ask for modifications in follow-up messages
5. Keep revising until the script fits the teacher's needs

The browser UI only shows the lesson duration control. The system automatically infers subject, grade level, and topic from the user's message and picks the best retrieval strategy.

Main POST endpoints:
- `POST /chat/script`
- `POST /agent/run`
- `POST /lessons/generate`

Retrieval modes:
- `auto` (default)
  infers subject, grade, and topic from the prompt, tries filtered retrieval if all three are present and exist in the corpus, falls back to all mode if corpus has data for that subject, or skips retrieval entirely (leading to fallback generation) if the inferred subject is not in the corpus at all
- `filtered`
  uses `subject` and `grade_level` as hard metadata filters
- `all`
  ignores metadata filters and retrieves from the full indexed document set

Recommended chatbot flow:
1. Send an initial request with only `message`
2. Receive a generated lesson script and a follow-up prompt
3. Send a second request with:
   - `message` as the modification request
   - `current_script` as the current lesson script
   - `original_request` as the first user request

Initial request example:

```json
{
  "message": "Create a 40-minute Grade 6 science lesson script on ecosystems and food chains with a class activity and short exit ticket."
}
```

The system will automatically infer `subject: "science"`, `grade_level: "6"`, `topic: "ecosystems and food chains"` and pick the best retrieval mode.

You can still override these explicitly if needed:

```json
{
  "message": "Create a 40-minute lesson script on ecosystems.",
  "subject": "science",
  "grade_level": "6",
  "retrieval_mode": "filtered"
}
```

Revision request example:

```json
{
  "message": "Please make the opening more interactive and reduce the difficulty for weaker students.",
  "original_request": "Create a 40-minute Grade 6 science lesson script on ecosystems and food chains with a class activity and short exit ticket.",
  "current_script": "PASTE_THE_CURRENT_SCRIPT_HERE"
}
```

### 5. Run retrieval smoke test
Use this to check whether retrieval is returning expected chunks:

```powershell
python -m tests.test_retrieval
```

### 6. Run lesson-script generation smoke test
Use this to test end-to-end retrieval and lesson-script generation:

```powershell
python -m tests.test_generate_lesson
```

### 7. Run pipeline smoke test
Use this to test the pipeline entrypoint and inspect its trace:

```powershell
python -m tests.test_pipeline
```

### 8. Run refusal-mode smoke test
Use this to confirm that off-domain prompts are rejected:

```powershell
python -m tests.test_refusal_mode
```

### 9. Run embedding smoke test
Use this to verify Ollama embedding is responding:

```powershell
python -m tests.test_embeddings
```

### 10. Run API smoke test
Use this to check the FastAPI app contract locally without starting a separate server process:

```powershell
python -m tests.test_api
```

## Source Data Workflow
### Add a new PDF
1. Put the PDF in `data/raw`
2. Add metadata for that file in `data/raw/metadata.json`
3. Run single-PDF indexing or full ingestion

Metadata maintenance helper:

```powershell
python -m app.ingestion.metadata_tools
python -m app.ingestion.metadata_tools --write-template
python -m app.ingestion.metadata_tools --write-llm-template --limit 3
```

What it does:
- audits which PDFs are missing metadata entries
- writes `data/raw/metadata_template.json` with suggested metadata for missing files
- can write `data/raw/metadata_llm_template.json` using sample PDF contents and Ollama
- lets you copy useful entries from the template into `data/raw/metadata.json`

Recommended metadata strategy:
1. Prioritize the PDFs you actually want the chatbot to retrieve well
2. Fill these fields first:
   - `subject`
   - `grade_level`
   - `topic`
   - `source_type`
3. Use the heuristic template or LLM template as a starting point, then correct the guesses manually
4. Reindex important PDFs after metadata updates

Notes on the LLM template:
- it is slower than filename-based guessing
- it uses sample PDF contents, so it can often infer better topics and source types
- it should still be reviewed manually before copying into `metadata.json`

### Update an existing PDF
1. Replace or edit the file in `data/raw`
2. Update its metadata if needed
3. Run the `--reindex` command for that PDF

## How The Code Works
### Configuration
[`app/config.py`](/c:/Users/walee/Desktop/Masters%20Degree/LLM/lesson-rag-agent/app/config.py)
- Loads settings from `.env`
- Controls Ollama, Qdrant, and chunking behavior

### Schemas
[`app/schemas.py`](/c:/Users/walee/Desktop/Masters%20Degree/LLM/lesson-rag-agent/app/schemas.py)
- Defines the core document and chunk models
- Keeps ingestion and indexing data structured
- `grade_level` can be either a single grade string or a list of grades for multi-grade sources

### Ingestion
[`app/ingestion/load_docs.py`](/c:/Users/walee/Desktop/Masters%20Degree/LLM/lesson-rag-agent/app/ingestion/load_docs.py)
- Loads PDFs and metadata
- Merges all pages of each PDF into a single document with `--- [Page N] ---` markers between pages
- Builds one deterministic document ID per PDF (not per page)

[`app/ingestion/metadata_tools.py`](/c:/Users/walee/Desktop/Masters%20Degree/LLM/lesson-rag-agent/app/ingestion/metadata_tools.py)
- Audits missing metadata entries
- Generates a suggested `metadata_template.json` for PDFs not yet described in `metadata.json`
- Can also generate an LLM-assisted `metadata_llm_template.json` based on PDF contents

[`app/ingestion/clean_docs.py`](/c:/Users/walee/Desktop/Masters%20Degree/LLM/lesson-rag-agent/app/ingestion/clean_docs.py)
- Cleans extracted PDF text
- Removes low-value junk content

[`app/ingestion/chunk_docs.py`](/c:/Users/walee/Desktop/Masters%20Degree/LLM/lesson-rag-agent/app/ingestion/chunk_docs.py)
- Uses structure-aware splitting: splits on headings (ALL-CAPS, numbered sections, markdown headers, chapter/section labels) rather than only paragraph boundaries
- Page markers are preserved inline for traceability but do not drive split points
- Each chunk records `page_number` (first page in the chunk) and `page_range` (e.g. `"5-8"`) in metadata
- Splits oversized sections first by paragraph, then by overlapping token windows
- Builds deterministic chunk IDs

[`app/ingestion/index_docs.py`](/c:/Users/walee/Desktop/Masters%20Degree/LLM/lesson-rag-agent/app/ingestion/index_docs.py)
- Embeds chunks
- Sends them to Qdrant
- Supports single-file indexing and reindexing
- Upserts points in batches during large ingestion runs

[`app/ingestion/run_ingestion.py`](/c:/Users/walee/Desktop/Masters%20Degree/LLM/lesson-rag-agent/app/ingestion/run_ingestion.py)
- Main ingestion entrypoint
- Supports full ingestion and single-file ingestion
- Writes per-document debug chunk files under `data/processed/chunks_debug/`

### Services
[`app/services/ollama_client.py`](/c:/Users/walee/Desktop/Masters%20Degree/LLM/lesson-rag-agent/app/services/ollama_client.py)
- Talks to Ollama for embeddings and generation
- Batches embedding requests so large ingestion runs do not send every chunk in one `/embed` request

[`app/services/qdrant_client.py`](/c:/Users/walee/Desktop/Masters%20Degree/LLM/lesson-rag-agent/app/services/qdrant_client.py)
- Creates the Qdrant client
- Ensures the collection and payload indexes exist
- Deletes chunks by `source_path` during reindex

[`app/services/retriever.py`](/c:/Users/walee/Desktop/Masters%20Degree/LLM/lesson-rag-agent/app/services/retriever.py)
- Embeds the query
- Applies stable metadata filters
- Uses topic as query context by default instead of exact topic filtering
- Can also run in `all` mode to retrieve from the entire collection without metadata filters
- Supports `retrieval_method="hybrid"`: runs dense search and BM25 independently (each fetching `k×3` candidates), merges results with Reciprocal Rank Fusion (RRF, k=60), and returns top-k

[`app/services/prompt_builder.py`](/c:/Users/walee/Desktop/Masters%20Degree/LLM/lesson-rag-agent/app/services/prompt_builder.py)
- Builds grounded prompts for literal, word-for-word classroom scripts when retrieval is strong
- Builds fallback prompts for literal, word-for-word classroom scripts when retrieval is weak
- Provides phased generation prompts: outline planning, per-block script writing, and closing sections
- Provides targeted revision prompts: `build_block_identification_prompt()` (identifies which blocks a modification affects) and `build_block_revision_prompt()` (revises a single block)

[`app/services/pipeline.py`](/c:/Users/walee/Desktop/Masters%20Degree/LLM/lesson-rag-agent/app/services/pipeline.py)
- Main pipeline entrypoint (`ScriptPipeline`)
- Orchestrates:
  - domain check
  - retrieval (dense or hybrid dense+BM25 with RRF)
  - generation mode selection
  - phased generation (outline → block-by-block → closing → assembly)
  - lesson evaluation and normalization
- Returns an `agent_trace` showing which steps ran

[`app/services/bm25_index.py`](/c:/Users/walee/Desktop/Masters%20Degree/LLM/lesson-rag-agent/app/services/bm25_index.py)
- Lazy-loaded BM25 index (`BM25Index`) built by scrolling all chunks from Qdrant on first access
- Thread-safe singleton via `get_bm25_index()`; invalidated with `invalidate_bm25_index()` after reindex

[`app/services/chatbot.py`](/c:/Users/walee/Desktop/Masters%20Degree/LLM/lesson-rag-agent/app/services/chatbot.py)
- Provides a chatbot-style workflow for lesson scripts
- Supports:
  - initial script generation
  - follow-up script revision using targeted block revision: parses the script into `[Minute X-Y]` blocks, identifies which block(s) the user's modification affects, regenerates only those blocks, and splices them back in
  - falls back to full script regeneration if block parsing fails
- Returns a follow-up message inviting the user to request modifications

### API Layer
[`app/main.py`](/c:/Users/walee/Desktop/Masters%20Degree/LLM/lesson-rag-agent/app/main.py)
- Exposes the agent through FastAPI
- Provides:
  - `GET /` for the lightweight chat UI
  - `GET /health`
  - `POST /chat/script`
  - `POST /agent/run`
  - `POST /lessons/generate`

## Current Runtime Logic
### Retrieval
Default hard filters:
- `subject`
- `grade_level`

Grade behavior:
- `grade_level` metadata can be a single value like `"6"`
- or a list like `["6", "7", "8"]`
- the retrieval filter can still ask for `grade_level="6"` and match multi-grade chunks that include grade 6

Topic behavior:
- `topic` is not used as a hard filter by default
- it is added to the semantic query context instead

Retrieval method behavior:
- `dense` (default): cosine similarity search against Qdrant
- `hybrid`: dense + BM25 merged with Reciprocal Rank Fusion (RRF, k=60) — better for exact terminology matches

Retrieval mode behavior:
- `filtered` keeps the hard metadata filters above
- `all` ignores those metadata filters and searches across the full corpus
- `all` is useful when metadata is incomplete or when you want broad discovery

### Generation Modes
`grounded`
- used when retrieved chunks are available and strong enough

`fallback`
- used when the request is education-related but retrieval is weak

`refuse`
- used when the request is unrelated to education

Script behavior:
- output is intended to be read aloud from start to finish
- teacher spoken lines are expected in quotation marks
- the script is organized into minute blocks that may vary in length
- the output is no longer just a lesson outline
- if no grade level is supplied, the script must say `Grade Level: Unspecified`
- if the subject field is blank but the prompt clearly names a subject like mathematics or science, the backend will infer that subject for the script header
- the output is now teacher-facing only and should not include `Students:` or `Student Lines:` sections
- each block is expected to contain approximately 95 words per minute of block duration, with at least `max(3, ceil(minutes*0.6))` Teacher: lines
- the evaluator checks both total script word count and per-block word count against 35% of the target as a density floor
- the backend performs a cleanup pass on generated output to remove draft-style preambles, repair blank headers where possible, and normalize fallback source notes
- script generation uses a phased approach: outline first, then each block individually, then closing sections — this ensures the model can produce enough content for long scripts

### Pipeline Flow
`ScriptPipeline.run()` executes these steps in order:
1. `domain_checker` — rejects non-education requests using a two-step check: subject keyword pre-screen (math, algebra, science, etc.) first, then education keyword gate, then LLM-based validation via Ollama (with `num_predict=8`) to confirm the request is genuinely about classroom education
2. `param_resolver` — infers subject, grade level, and topic from the prompt if not explicitly provided, then picks the best retrieval mode (subject in corpus? → filtered → all; subject NOT in corpus? → skip retrieval → fallback generation)
3. `retriever` — fetches relevant chunks from Qdrant using dense or hybrid (dense+BM25+RRF) retrieval
4. `outline_generator` — creates a structured lesson outline with block plan
5. `block_generator` (repeated per block) — writes the full script for each time block individually
   - blocks longer than 5 minutes are automatically split into sub-blocks (max 5 minutes each) so each LLM call stays within the model's practical output range
   - sub-blocks receive the previous block text for continuity
6. `closing_generator` — writes Exit Ticket, Homework, Teacher Notes, Sources Used
7. Assembly — combines header, blocks, and closing into the final script
8. `evaluator` — checks structure, constraints, and density

## Smoke Tests
Current smoke scripts:
- `tests/test_embeddings.py`
- `tests/test_index_qdrant.py`
- `tests/test_retrieval.py`
- `tests/test_generate_lesson.py`
- `tests/test_pipeline.py`
- `tests/test_api.py`
- `tests/test_chatbot_api.py`
- `tests/test_frontend.py`
- `tests/test_refusal_mode.py`
- `tests/test_single_pdf_reindex.py`

These are mainly manual verification scripts, not full assertion-based automated tests.

## When You Make Changes
Update this file when any of the following change:
- run commands
- ingestion flow
- retrieval logic
- generation logic
- required environment variables
- project structure

Also update:
- `README.md` for the project log and architecture history

## Change Log
## 2026-03-11
- Change: created `PROJECT_GUIDE.md` as the operating guide for running the project and understanding the code flow.
- Reason: `README.md` is serving as the project log, but the project also needs a dedicated guide focused on usage and execution.
- Follow-up: keep this file updated whenever project workflow or behavior changes.

- Change: documented multi-grade metadata support for `grade_level` and clarified how grade filtering should work for books that span multiple grades.
- Reason: the project needs to support educational sources that are not limited to exactly one grade.
- Follow-up: validate retrieval behavior on a larger multi-grade corpus and adjust metadata conventions if needed.

- Change: updated the ingestion guide to use per-document debug chunk JSON files under `data/processed/chunks_debug/` instead of a single overwritten debug file.
- Reason: per-document debug files are easier to inspect later and preserve document-level references across ingestion runs.
- Follow-up: decide later whether stale debug files for removed source documents should be cleaned automatically.

- Change: documented the new `agent.py` orchestration layer, added the direct agent smoke-test command, and clarified that `lesson_generator.py` now wraps the agent.
- Reason: the project now has an explicit agent entrypoint and the operating guide needs to reflect the current runtime flow.
- Follow-up: document any future evaluator or revision-logic upgrades here as the agent becomes more capable.

- Change: documented the FastAPI entrypoint in `app/main.py`, including the `uvicorn` run command and the main API endpoints for continuous use.
- Reason: the project now supports repeated interaction through an API instead of relying only on smoke scripts.
- Follow-up: add auth, structured error handling, and frontend integration details if the API becomes team-facing.

- Change: updated the guide to reflect that the system now generates teacher-facing lesson scripts rather than only lesson plans.
- Reason: the actual output requirement is a classroom script a teacher can follow.
- Follow-up: document any future script-format refinements here if the output structure changes again.

- Change: documented the chatbot-style lesson-script workflow and the `/chat/script` API endpoint for iterative script customization.
- Reason: the preferred interaction model is now conversational revision rather than a one-shot agent run.
- Follow-up: add frontend conversation examples here if the team builds a chat UI on top of the API.

- Change: documented the new lightweight browser chat UI at `/` and added the frontend smoke test.
- Reason: the project now has a small frontend so teachers can use the chatbot without interacting with raw API endpoints.
- Follow-up: refine the UI later if the team wants richer chat history, authentication, or saved sessions.

- Change: updated the guide to reflect that the generated output should now be a literal, word-for-word classroom script with minute blocks and spoken teacher lines.
- Reason: the actual teaching need is a script a teacher can read aloud from start to finish, not a semi-scripted outline.
- Follow-up: refine the script structure further if the team wants even tighter minute-by-minute pacing.

- Change: documented Ollama embedding batching and Qdrant upsert batching for large ingestion runs.
- Reason: full-library ingestion can send thousands of chunks, which is too large for a single Ollama `/embed` request.
- Follow-up: tune `EMBEDDING_BATCH_SIZE` and `QDRANT_UPSERT_BATCH_SIZE` based on local performance and model limits.

- Change: updated chunking so oversized paragraphs are split before indexing, keeping final chunks within the embedding token budget.
- Reason: some source pages contained very long paragraphs, which created chunks that exceeded the Ollama embedding context length.
- Follow-up: monitor chunk quality on math-heavy PDFs and refine splitting if token-window cuts become too awkward semantically.

- Change: documented the metadata audit/template helper so missing PDF metadata can be managed incrementally instead of manually from scratch.
- Reason: only a subset of PDFs currently have metadata, and retrieval quality depends on gradually covering the important ones.
- Follow-up: improve the filename-based metadata guesses if the corpus grows or naming patterns change.

- Change: documented the LLM-assisted metadata template workflow based on PDF contents.
- Reason: content-based suggestions can produce better metadata than filename heuristics alone, especially for topic and source-type inference.
- Follow-up: compare LLM-suggested metadata against manual labels on a few PDFs and tune the prompt if the guesses are noisy.

- Change: documented the new `retrieval_mode` option so the chatbot and API can either search with metadata filters or retrieve from all indexed documents.
- Reason: some lesson requests need broad corpus search, especially while metadata coverage is still incomplete.
- Follow-up: add an automatic retry path later if filtered retrieval returns weak results.

- Change: added strict validation for `Filtered Mode`, including required fields and checks that the selected subject and grade values exist in the indexed corpus.
- Reason: filtered retrieval should fail fast when the request is incomplete or points to values that are not actually represented in the dataset.
- Follow-up: add autocomplete or selectable indexed values later if users still enter invalid filters often.

- Change: tightened the classroom-script pacing rules so scripts can use variable-length minute blocks, keep `Grade Level: Unspecified` when no grade is supplied, and include enough dialogue and pause cues to fill each block proportionally to its duration.
- Reason: the output needs to behave like a real classroom script, not a thin outline with minute labels.
- Follow-up: tune the timing heuristic against real teacher reviews and classroom pacing.

- Change: updated lesson generation so subject can be inferred from the prompt when the subject form field is blank, and removed student-response sections from the target script format.
- Reason: the user prompt often already states the subject, and the product requirement is now a direct teacher read-aloud script rather than teacher-plus-student scripted dialogue.
- Follow-up: extend subject inference only if more subjects or domain labels are added later.

- Change: added a final script-normalization pass to clean draft-style output, remove stray student-dialogue sections, fill missing header values like lesson title where possible, and stop fallback mode from inventing source names.
- Reason: prompt instructions alone were not enough to keep the returned script consistently inside the required format.
- Follow-up: move toward more structured generation later if cleanup logic becomes too broad.

- Change: updated the quality gate so scripts that still fail timing or formatting checks after revision attempts are rejected with a clear generation error instead of being returned to the user.
- Reason: a visibly broken script is worse than an explicit failure because it looks usable when it is not.
- Follow-up: add a stronger repair-only pass if the current model still fails too often on long lesson scripts.

- Change: removed `curriculum` from the default browser chat UI while keeping API support for it.
- Reason: it is not needed for the common workflow and was adding clutter to the frontend.
- Follow-up: if curriculum-specific retrieval becomes important later, add it back under an advanced options section rather than the default form.

- Change: changed the browser retrieval-mode control from free text to a fixed dropdown with `Filtered Mode` and `All Mode`.
- Reason: the frontend should only expose the two supported retrieval behaviors and avoid invalid user-entered values.
- Follow-up: if more retrieval strategies are added later, keep using fixed selections rather than open text entry.

## 2026-03-13
- Change: added explicit words-per-minute density requirements to all generation, revision, and chatbot revision prompts so the LLM has concrete word count targets per block and for the full script.
- Reason: scripts were passing structural checks but were too sparse to fill the requested duration when read aloud.
- Details:
  - `prompt_builder.py` now includes `format_density_guidance()` which calculates total and per-block word targets at 95 words per minute
  - Block templates now show per-block word count and minimum Teacher: line targets
  - The evaluator in `agent.py` now checks total script word count against the duration target (75% floor)
  - Per-block evaluation uses the same density floor and reports actual vs. target word counts in failure messages
  - Revision prompts now tell the model exactly how many words each failing block needs
  - Teacher-line minimums raised from `max(3, ceil(minutes/4)+1)` to `max(4, minutes*2)` to require denser dialogue
- Follow-up: tune `WORDS_PER_MINUTE_TARGET` (95), `DENSITY_FLOOR` (0.55), and `DENSITY_CEILING` (1.4) after reviewing output against real classroom pacing.

- Change: set `num_predict: 8192` and `num_ctx: 16384` in the Ollama `/generate` call so the model can produce full-length scripts, and increased timeout to 600s.
- Reason: the default `llama3.2` output limit is 128 tokens (~100 words), which was the root cause of all generation failures regardless of prompt instructions.
- Follow-up: make `num_predict` and `num_ctx` configurable via `.env` if needed for different models.

- Change: calibrated density evaluation: `DENSITY_FLOOR` to 0.35, teacher-lines to `max(3, ceil(minutes*0.6))`, separated per-block density checks into non-blocking `density_warnings` that guide revisions without preventing output, added `density_ratio` to evaluation results, and added a user-visible density notice when script coverage is below 85%.
- Reason: per-block failures were blocking output even when the overall script structure was correct. The revision loop now uses density warnings to push the model (observed 729 -> 3732 words across passes) while still returning usable scripts.
- Follow-up: raise the density floor after switching to a larger generation model.

- Change: replaced hard `generation_error` failure with a best-attempt strategy. The revision loop tracks the best script across all attempts and returns it with a density warning instead of refusing output. Added `_eval_quality_score()` to rank attempts by completeness, constraints, and density.
- Reason: `llama3.2` output is nondeterministic and sometimes cannot pass all checks within the revision budget. A thin script with a warning is always more useful than an error.
- Follow-up: consider removing the `generation_error` code path or reserving it for zero-output edge cases.

- Change: removed `app/nodes/` (empty LangGraph stubs), `app/services/embeddings.py` (empty), and unused deps from `requirements.txt` (`langgraph`, `langchain-*`, `sentence-transformers`, `python-docx`). Added `requests` explicitly.
- Reason: none were used in the codebase.
- Follow-up: none.

- Change: `get_qdrant_client()` is now a module-level singleton instead of creating a new client per call.
- Reason: validation was creating 4+ redundant connections to Qdrant Cloud per request.
- Follow-up: none.

- Change: replaced single-shot generation + revision loop with phased generation. The agent now generates an outline, then fills each time block individually, then generates closing sections, and assembles the final script. Each LLM call produces ~500-1000 words instead of trying to generate ~3800 words at once.
- Reason: `llama3.2` (3B) cannot produce a full-length script in one pass. Splitting into smaller prompts keeps each call within the model's practical output range.
- Details:
  - `prompt_builder.py`: added `build_outline_prompt()`, `build_block_prompt()`, `build_closing_sections_prompt()`
  - `agent.py`: added `generate_phased()` method, wired it into `run()`, removed the old revision loop
  - `chatbot.py`: removed `generation_error` from handled modes (phased generation always returns output)
  - Agent trace now shows `outline_generator`, per-block `block_generator` entries, and `closing_generator`
- Follow-up: test end-to-end through the UI and tune block word targets.

- Change: added math notation cleanup to ingestion (`clean_math_notation()` in `clean_docs.py`) and a spoken-math rule to all prompts so scripts write math as words instead of symbols. Added truncation detection in block generation.
- Reason: PDF-extracted math was garbled (Z for integral, broken fractions) and unusable for read-aloud scripts or TTS.
- Follow-up: reindex math PDFs with `--reindex` to apply cleanup. Test TTS output quality.

- Change: redesigned the frontend with a dark theme, Inter + JetBrains Mono fonts, typing indicator, copy-to-clipboard, suggestion chips, auto-resize textarea, Enter-to-send, and a duration input that auto-injects into prompts. Old UI preserved at `/old` (`index_old.html`).
- Reason: improve visual polish and usability for extended lesson scripting sessions.
- Follow-up: gather feedback; remove `/old` route if the new UI is kept.

## 2026-03-15
- Change: merged PDF pages into whole-document text before chunking, with `--- [Page N] ---` markers preserved inline for traceability. Each PDF now produces one `RawDocument` instead of one per page.
- Reason: page-level chunking was splitting concepts that span page boundaries. Cross-page content never landed in the same chunk, and overlap only worked within a single page.
- Follow-up: reindex all PDFs to apply the new chunking. Old page-level chunks in Qdrant should be cleared first.

- Change: replaced paragraph-only splitting with structure-aware section splitting. The chunker now splits on ALL-CAPS headings, numbered sections (e.g. `1.2`), markdown headings, and chapter/section labels before falling back to paragraph splitting.
- Reason: paragraph splitting (`\n\n`) is too granular for well-structured educational documents. Heading-based splitting keeps semantically related content together within each chunk.
- Follow-up: monitor chunk quality across different PDF structures and add more heading patterns if needed.

- Change: increased `CHUNK_SIZE_TOKENS` from 300 to 512 and `CHUNK_OVERLAP_TOKENS` from 80 to 100.
- Reason: 300 tokens was too small for `nomic-embed-text` (8192-token context). Larger chunks provide richer semantic context per embedding and reduce the number of retrieval hits needed to cover a topic.
- Follow-up: consider going to 1024 tokens if retrieval quality improves further with larger chunks.

- Change: added `page_range` field to `ChunkMetadata` and to the retrieval response metadata. Each chunk now records which pages it spans (e.g. `"5-8"`). The prompt builder now shows `Pages:` instead of `Page:` in source references.
- Reason: with merged-document chunking, a single chunk can span multiple pages. The page range gives teachers and developers a reference back to the source PDF.
- Follow-up: none.

- Change: added overlapping token-window splitting to `_split_by_token_window()` for oversized sections that can't be split by paragraph.
- Reason: the previous `split_large_paragraph` used non-overlapping windows, which could cut sentences mid-thought at boundaries.
- Follow-up: none.

- Change: added automatic sub-block splitting to `generate_phased()` in `agent.py`. Blocks longer than 5 minutes are split into sub-blocks (max 5 minutes each) so each LLM call only needs to produce ~475 words. Sub-blocks get the previous block text for continuity. Short trailing remainders (< 3 minutes) are merged into the previous sub-block to avoid tiny fragments.
- Reason: `llama3.2` (3B) produces short output regardless of prompt instructions. The fix that worked before was splitting work into smaller LLM calls. A 10-minute block asking for ~950 words often came back with ~200 words, but two 5-minute calls each producing ~400 words gives ~800 words total.
- Follow-up: tune `max_minutes_per_call` if a larger model is used later.

- Change: fixed `is_mostly_junk()` in `clean_docs.py` so the "contents" and "index" heuristics only apply to short texts (< 25 lines). Previously these checks could discard entire merged documents that happened to contain those words.
- Reason: after merging pages into whole documents, the `c2integral.pdf` was being filtered out because the merged text contained "index" on its own line.
- Follow-up: none.

- Change: added automatic retrieval mode (`auto`) as the new default. The agent now infers subject, grade level, and topic from the user's natural language prompt, tries filtered retrieval if all three are present and exist in the corpus, and falls back to all mode otherwise. Removed subject, grade level, topic, retrieval limit, and retrieval mode controls from the browser UI. The frontend now only shows the duration setting.
- Reason: teachers should not need to understand retrieval modes or manually fill metadata fields. The system should figure out the best retrieval strategy from the prompt.
- Follow-up: improve prompt parsing for edge cases. Consider using the LLM for more robust extraction if keyword matching is insufficient.

- Change: added `_parse_and_validate_block_plan()` to enforce the requested duration on the outline's block plan. Blocks that exceed the total duration are clipped, duplicate block ranges are removed, and invalid plans fall back to a sensible default. The plan is guaranteed to start at minute 0 and end at the requested total.
- Reason: the LLM was generating outlines with blocks extending far beyond the requested duration (e.g. blocks up to [Minute 25-30] for a 15-minute lesson) and sometimes duplicating block ranges.
- Follow-up: none.

- Change: fixed irrelevant source retrieval in auto mode. When the system infers a subject from the prompt but that subject does not exist in the Qdrant corpus, auto mode now skips retrieval entirely and goes straight to fallback generation instead of falling back to "all" mode (which would retrieve the closest but irrelevant content from unrelated subjects).
- Reason: falling back to "all" mode for unknown subjects caused the system to retrieve and ground scripts on unrelated source material, producing misleading output.
- Details:
  - `_resolve_retrieval_params()` in `agent.py`: added a `field_value_exists("subject", resolved_subject)` check before trying "all" mode. If the subject is not in the corpus, sets mode to `"skip"`.
  - `run()` in `agent.py`: when mode is `"skip"`, bypasses retrieval and uses empty chunks (leading to fallback generation).
  - Same fix applied in `chatbot.py` `revise_script()`.
  - Auto mode resolution order is now: subject exists in corpus? try filtered, fall back to all. Subject NOT in corpus? skip retrieval, use fallback generation.
- Follow-up: none.

- Change: replaced simple keyword-based domain check with LLM-validated domain check in `check_education_domain()` in `app/services/agent.py`.
- Reason: the old keyword check only looked for education-related words like "lesson" or "teach" in the prompt, so requests like "write me a winning trading bot (40-minute lesson)" would pass the gate and generate a script.
- Details:
  - Step 1 (fast path): quick keyword pre-screen. If no education keywords are found at all, refuse immediately without calling the LLM.
  - Step 2 (LLM validation): sends a short classifier prompt to Ollama asking whether the request is genuinely about classroom education. Uses `num_predict=8` to keep latency low. Only proceeds if the LLM confirms "yes".
  - This prevents non-educational topics (trading bots, gambling strategies, etc.) from generating scripts just because they happen to include words like "lesson" or "teach".
- Follow-up: monitor false-positive and false-negative rates on real prompts and tune the classifier prompt if needed.

## 2026-03-27
- Change: renamed `agent.py` → `pipeline.py`, `LessonPlanningAgent` → `ScriptPipeline`, `AgentConfig` → `PipelineConfig`. Renamed `test_agent.py` → `test_pipeline.py`.
- Reason: the orchestrator is a deterministic pipeline (domain check → retrieval → generation → evaluate), not a true agent with tool selection or reasoning loops. The name was misleading.
- Follow-up: none.

- Change: implemented hybrid retrieval — dense vector search and BM25 lexical search merged with Reciprocal Rank Fusion (RRF, k=60). Added `app/services/bm25_index.py` (lazy BM25 singleton). Added `retrieval_method` parameter to `retrieve_chunks()`, `ScriptPipeline.run()`, all chatbot methods, API schemas, and Streamlit sidebar.
- Reason: dense-only retrieval misses exact lexical matches (e.g. specific theorem names, formula notation). Hybrid + RRF consistently improves recall on domain-specific corpora.
- Follow-up: add `rank-bm25` to requirements and install it (`pip install rank-bm25`).

- Change: fixed domain check refusing valid education prompts like "write me a math script on algebra". Subject keywords (math, algebra, science, etc.) are now checked before the education keyword gate.
- Reason: "math", "algebra", and "script" are not in EDUCATION_KEYWORDS, so the prompt was refused at the keyword gate before subject detection ran.
- Follow-up: none.

- Change: fixed hybrid retrieval always returning fallback mode. RRF scores max ~0.033 (far below the 0.35 cosine threshold). When `retrieval_method="hybrid"`, generation mode is now decided by whether any chunks were returned, not by score threshold.
- Reason: the cosine similarity threshold is not meaningful for RRF scores.
- Follow-up: none.

- Change: improved Streamlit frontend. Removed `st.container(height=550)` — messages now render inline with `st.chat_input()` pinned to viewport. Replaced `st.code()` with a pre-wrap HTML div. Replaced `st.text()` with `st.markdown()` for retrieved chunk text. Added Top-k Chunks slider (3/5/7) and Retrieval Method dropdown (dense/hybrid) to sidebar.
- Reason: fixed horizontal scroll on long script lines and auto-scroll failures on new messages.
- Follow-up: none.

- Change: deleted dead code and artifacts: `app/services/lesson_generator.py`, `app/static/index_old.html`, `/old` route in `main.py`, audio artifacts in `app/static/audio/`, debug chunk files in `data/processed/chunks_debug/`, `data/samples/metadata_llm_template.json`, and `build_lesson_prompt()` unused wrapper in `prompt_builder.py`.
- Reason: none of these were referenced by any active code path.
- Follow-up: none.

- Change: rewrote the script revision flow in `chatbot.py` to use targeted block revision instead of full-script regeneration.
- Reason: sending the entire script to the LLM for regeneration on every modification was extremely slow for large scripts.
- Details:
  - `app/services/chatbot.py`: complete rewrite of `revise_script()`. Added `_parse_script_blocks()` (splits script into header, `[Minute X-Y]` blocks, and footer), `_reassemble_script()` (splices revised blocks back), and `_full_revision_fallback()` (for scripts that cannot be parsed into blocks). Removed calls to `normalize_lesson_text()` and `evaluate_lesson()` during revision since the structure is already valid from initial generation.
  - `app/services/prompt_builder.py`: added `build_block_identification_prompt()` (quick classifier using `num_predict=128` to identify which blocks a modification affects) and `build_block_revision_prompt()` (revises a single block with the modification request).
  - For a typical modification like "make the introduction more engaging" on a 40-minute script with 8 blocks, only 1-2 blocks are regenerated instead of all 8, yielding roughly 4-8x faster revisions.
  - Falls back to full regeneration if the script cannot be parsed into blocks.
- Follow-up: none.

## 2026-03-28
- Change: async job pattern for `/chat/script` endpoint. `POST /chat/script` now returns a `job_id` immediately. `GET /chat/status/{job_id}` returns status (`processing` / `done` / `error`) and the full result when done. Frontend polls every 3 s with a progress bar.
- Reason: Cloudflare tunnel (and any reverse proxy) imposes a ~100 s hard timeout on open connections. Lesson generation takes 4–6 minutes, causing 524 errors. The async pattern keeps every individual HTTP request short.
- Details:
  - `app/main.py`: added `_script_jobs` dict, `_run_script()` background task, changed `/chat/script` to use `BackgroundTasks`, added `GET /chat/status/{job_id}`.
  - `streamlit_app.py`: `call_chat_api()` now submits the job then polls `/chat/status/{job_id}` with a `st.progress` bar (300 iterations × 3 s = 15-minute ceiling).
- Follow-up: none.

- Change: fixed `st.secrets.get()` crash when no `.streamlit/secrets.toml` exists locally.
- Reason: Streamlit raises `StreamlitSecretNotFoundError` when no secrets file is present at all, rather than returning the default value.
- Details: wrapped `st.secrets.get()` in try/except; created `.streamlit/secrets.toml` (gitignored) for local dev with `API_BASE = "http://localhost:8000"`.
- Follow-up: when deploying to Streamlit Cloud, set `API_BASE` in the app's Secrets settings UI.

- Change: tightened domain filter to close subject-keyword bypass and added revision guard.
- Reason: prompts like `"write a trading bot using math equations"` bypassed LLM validation because `"equation"` matched the subject keyword fast-path. Revision requests had no domain check at all.
- Details:
  - `pipeline.py`: subject keyword fast-path now requires BOTH a subject term AND an education keyword (e.g. lesson, teach, classroom).
  - `chatbot.py`: `chat()` checks off-topic revision requests against a revision-verb allowlist (`make`, `add`, `change`, `expand`, etc.) and refuses if neither education-related nor a recognised revision command.
- Follow-up: monitor false-positive refusals on legitimate short revision commands.

- Change: fixed BM25 subject filter bypass in hybrid retrieval.
- Reason: BM25 searched the full corpus without subject filtering. Queries containing words like `"script"` matched CS chunks about scripting languages, which entered the RRF pool and appeared in results alongside calculus chunks.
- Details: `retriever.py` `_hybrid_retrieve()` now accepts `subject` and `grade_level` params; BM25-only candidates have their payloads fetched and filtered by subject/grade before entering the RRF pool. Grade-level stored as list or string is handled.
- Follow-up: none.

- Change: fixed embedding vector space mismatch — restored retrieval scores to ~0.7 range.
- Reason: the query instruction prefix was `"Instruct: Given a teacher's request...\nQuery: "` (designed for e5-mistral-7b), but the embedding model is `nomic-embed-text`. Documents were indexed with plain text. The mismatched formats put query and document vectors in different spaces, producing similarity scores of ~0.016.
- Details:
  - `retriever.py`: changed `_QUERY_INSTRUCTION` to `_QUERY_PREFIX = "search_query: "` (nomic-embed-text native prefix).
  - `index_docs.py`: documents now indexed with `"search_document: " + chunk.text`.
  - **Requires full re-indexing**: `python -m app.ingestion.run_ingestion`
- Follow-up: after re-indexing, confirm scores return to ~0.7 for relevant queries.

- Change: score display now labels RRF scores explicitly (e.g. `0.0164 (RRF)` vs plain cosine `0.732`).
- Reason: RRF scores are intrinsically small fractions and look broken compared to cosine similarity scores without context.
- Follow-up: none.
