# Lesson RAG Agent

A Retrieval-Augmented Generation system that ingests educational PDFs and generates word-for-word classroom scripts grounded in real source material. Built with FastAPI, Ollama, and Qdrant.

## Architecture

```
+------------------+     +-------------------+     +------------------+
|   PDF Documents  |     |   Streamlit /      |     |   FastAPI API    |
|   (data/raw/)    |     |   Browser UI       |     |   (app/main.py)  |
+--------+---------+     +--------+----------+     +--------+---------+
         |                        |                          |
         v                        v                          v
+------------------+     +-------------------+     +------------------+
|   Ingestion      |     |   LessonScript    |     |   Script         |
|   Pipeline       |     |   Chatbot         |     |   Pipeline       |
|                  |     | (app/services/    |     | (app/services/   |
| 1. Load PDFs     |     |  chatbot.py)      |     |  pipeline.py)    |
| 2. Clean text    |     +--------+----------+     +--------+---------+
| 3. Chunk         |              |                          |
| 4. Embed + Index |              v                          v
+--------+---------+     +-------------------+     +------------------+
         |               |   Prompt Builder  |     |   Retriever      |
         v               | (app/services/    |     | (app/services/   |
+------------------+     |  prompt_builder.py|     |  retriever.py)   |
|   Qdrant Vector  |<--->+-------------------+     +--------+---------+
|   Database       |              |                          |
|                  |              v                          v
+------------------+     +-------------------+     +------------------+
                         |   Ollama LLM      |     |   Ollama         |
                         |   (Generation)    |     |   (Embeddings)   |
                         +-------------------+     +------------------+
```

### Pipeline Flow

```
User Request
    |
    v
[Domain Check] -- not education --> [Refuse]
    |
    v (education-related)
[Auto-Resolve Retrieval Params]  (infer subject / grade / topic from prompt)
    |
    v
[Retrieve Chunks from Qdrant]  (dense or hybrid dense+BM25 with RRF)
    |
    +--> chunks found (score >= threshold) --> [Grounded Mode]
    |
    +--> no matches / low score            --> [Fallback Mode]
    |
    v
[Phased Generation]
    |
    v
[1. Generate Outline] --> [2. Generate Block-by-Block] --> [3. Deduplicate]
    |
    v
[4. Assemble Script + Citations + References]
    |
    v
[5. Evaluate Quality]
    |
    v
[Return Script with Source Attribution]
```

## Features

- **RAG-Grounded Generation** — Scripts are grounded in retrieved educational documents with inline `[Source N]` citations
- **Hybrid Retrieval** — Dense vector search and BM25 lexical search merged with Reciprocal Rank Fusion (RRF); configurable top-k (k = 3, 5, 7)
- **Phased Generation** — Outline first, then block-by-block to handle local model output limits
- **Three Generation Modes** — Grounded (from sources), Fallback (general knowledge), Refuse (non-educational)
- **Iterative Revision** — Chat-style interface for modifying generated scripts block-by-block
- **Source Attribution** — Inline citations with a References section linking to source documents
- **Text-to-Speech** — Generate audio from scripts using Edge TTS or Bark
- **Auto Retrieval** — Infers subject, grade level, and topic from natural language prompts
- **Metadata-Aware Ingestion** — PDF metadata (subject, grade, topic) extracted via heuristics or LLM

## Tech Stack

| Component          | Technology                                      |
|--------------------|-------------------------------------------------|
| Backend API        | FastAPI                                         |
| LLM                | Ollama (qwen3.5:27b recommended; qwen2.5:7b for low-resource) |
| Embeddings         | Ollama (nomic-embed-text or qwen3-embedding:0.6b) |
| Vector Database    | Qdrant (Cloud or local Docker)                  |
| Lexical Search     | BM25 (rank-bm25) merged with dense via RRF      |
| Frontend           | Streamlit + HTML/JS chat UI                     |
| TTS                | Edge TTS / Bark                                 |
| Language           | Python 3.11+                                    |

## Setup Instructions

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- [Qdrant](https://qdrant.tech/) (Cloud account or local Docker instance)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/lesson-rag-agent.git
cd lesson-rag-agent
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull Ollama Models

```bash
# Generation model — use a larger model if your hardware allows
ollama pull qwen2.5:7b

# Embedding model
ollama pull nomic-embed-text
```

### 5. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
OLLAMA_BASE_URL=http://localhost:11434/api
OLLAMA_GENERATION_MODEL=qwen2.5:7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_api_key
QDRANT_COLLECTION=lesson_docs
CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_TOKENS=100
```

### 6. Add Source Documents

Place educational PDFs in `data/raw/` and add entries to `data/raw/metadata.json`:

```json
{
  "Algebra_1.pdf": {
    "title": "Algebra 1 Textbook",
    "subject": "mathematics",
    "grade_level": "8",
    "topic": "algebra",
    "source_type": "textbook",
    "language": "en"
  }
}
```

To auto-generate metadata suggestions using the LLM:

```bash
python -m app.ingestion.metadata_tools --write-llm-template
```

### 7. Run Ingestion

```bash
# Ingest all PDFs
python -m app.ingestion.run_ingestion

# Ingest a single PDF
python -m app.ingestion.run_ingestion --pdf data/raw/your_file.pdf

# Reindex an existing PDF (removes old chunks first)
python -m app.ingestion.run_ingestion --pdf data/raw/your_file.pdf --reindex
```

### 8. Start the Application

**FastAPI backend:**

```bash
uvicorn app.main:app --reload
```

API runs at `http://localhost:8000` with a built-in chat UI at `/` and Swagger docs at `/docs`.

**Streamlit frontend:**

```bash
streamlit run streamlit_app.py
```

Runs at `http://localhost:8501` — requires the FastAPI backend to be running.

## API Endpoints

| Method | Endpoint             | Description                                   |
|--------|----------------------|-----------------------------------------------|
| POST   | `/agent/run`         | Generate a lesson script                      |
| POST   | `/lessons/generate`  | Alias for `/agent/run`                        |
| POST   | `/chat/script`       | Chat-style generation and revision            |
| POST   | `/tts/generate`      | Start text-to-speech audio generation         |
| GET    | `/tts/status/{id}`   | Check TTS job status                          |
| GET    | `/tts/download/{id}` | Download generated audio file                 |
| GET    | `/health`            | Health check                                  |

### Example Request

```bash
curl -X POST http://localhost:8000/chat/script \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a 20-minute lesson on quadratic equations for grade 8",
    "retrieval_limit": 5,
    "retrieval_method": "hybrid"
  }'
```

### Request Parameters (`/chat/script`)

| Parameter          | Type    | Default    | Description                                              |
|--------------------|---------|------------|----------------------------------------------------------|
| `message`          | string  | required   | Natural language lesson request                          |
| `subject`          | string  | auto       | Override inferred subject                                |
| `grade_level`      | string  | auto       | Override inferred grade                                  |
| `retrieval_limit`  | int     | 5          | Number of chunks to retrieve (3, 5, or 7 recommended)   |
| `retrieval_mode`   | string  | `"auto"`   | `"auto"`, `"filtered"`, or `"all"`                       |
| `retrieval_method` | string  | `"dense"`  | `"dense"` (cosine only) or `"hybrid"` (dense + BM25 RRF) |
| `current_script`   | string  | null       | Existing script to revise (revision mode)                |
| `original_request` | string  | null       | Original prompt used to generate `current_script`        |

## Project Structure

```
lesson-rag-agent/
├── app/
│   ├── main.py                  # FastAPI application and endpoints
│   ├── config.py                # Environment settings (from .env)
│   ├── schemas.py               # Pydantic request/response models
│   ├── services/
│   │   ├── pipeline.py          # Script generation pipeline (domain check, retrieval, generation)
│   │   ├── chatbot.py           # Conversational script generation and revision
│   │   ├── prompt_builder.py    # All LLM prompt templates and citation formatting
│   │   ├── retriever.py         # Dense + hybrid retrieval with metadata filtering
│   │   ├── bm25_index.py        # Lazy-loaded BM25 index for hybrid search
│   │   ├── ollama_client.py     # Ollama API client (generate + embed)
│   │   ├── qdrant_client.py     # Qdrant connection and collection management
│   │   └── tts.py               # Text-to-speech generation (Edge TTS / Bark)
│   ├── ingestion/
│   │   ├── run_ingestion.py     # CLI entry point for document ingestion
│   │   ├── load_docs.py         # PDF loading with page markers and metadata merging
│   │   ├── clean_docs.py        # Text cleaning and junk page removal
│   │   ├── chunk_docs.py        # Structure-aware chunking (512 tokens, 100 overlap)
│   │   ├── index_docs.py        # Embedding and Qdrant indexing
│   │   └── metadata_tools.py    # Metadata inference (heuristic + LLM)
│   └── static/
│       └── index.html           # Built-in browser chat UI
├── data/
│   └── raw/                     # Source PDFs and metadata.json
├── tests/
│   ├── test_pipeline.py         # Script pipeline smoke test
│   ├── test_api.py              # FastAPI endpoint smoke test
│   ├── test_chatbot_api.py      # Chat endpoint smoke test
│   ├── test_embeddings.py       # Embedding smoke test
│   ├── test_frontend.py         # Frontend smoke test
│   ├── test_generate_lesson.py  # End-to-end generation smoke test
│   ├── test_index_qdrant.py     # Qdrant indexing smoke test
│   ├── test_refusal_mode.py     # Domain refusal smoke test
│   ├── test_retrieval.py        # Retrieval smoke test
│   └── test_single_pdf_reindex.py # Single PDF reindex smoke test
├── streamlit_app.py             # Streamlit frontend
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
└── README.md
```

## Key Design Decisions

1. **Phased generation over one-shot** — Local models (4–27B parameters) cannot reliably generate 3000+ word scripts in a single call. The system generates an outline, then fills each time block individually (up to 30 minutes per LLM call at 8192 tokens), and assembles the result.

2. **Hybrid retrieval with RRF** — Dense vector search captures semantic similarity; BM25 captures exact lexical matches (e.g. specific formulas or topic names). Reciprocal Rank Fusion (k=60) merges both ranked lists without needing score normalisation.

3. **Three-layer deduplication** — Local models tend to repeat content. The system applies within-block, cross-block, and full-script paragraph-level deduplication using fuzzy matching (SequenceMatcher ratio > 0.85).

4. **Citation system** — When generation is grounded in retrieved chunks, the LLM is instructed to cite sources inline as `[Source 1]`, `[Source 2]`. A References section with source metadata is appended to the script.

5. **Auto retrieval mode** — Instead of requiring users to specify subject/grade/topic, the system infers these from the natural language prompt and picks the best retrieval strategy automatically (filtered → all → skip).

6. **Block-level revision** — When a user requests changes, only the affected time blocks are regenerated rather than the entire script, yielding 4–8× faster revisions.

## Running Smoke Tests

```bash
# Verify embeddings are working
python -m tests.test_embeddings

# Verify retrieval is returning relevant chunks
python -m tests.test_retrieval

# End-to-end lesson generation
python -m tests.test_generate_lesson

# Verify non-educational prompts are refused
python -m tests.test_refusal_mode

# API contract check
python -m tests.test_api
```
