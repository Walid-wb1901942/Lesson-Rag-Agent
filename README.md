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
|   Ingestion      |     |   LessonScript    |     |   Lesson         |
|   Pipeline       |     |   Chatbot         |     |   Planning Agent |
|                  |     | (app/services/    |     | (app/services/   |
| 1. Load PDFs     |     |  chatbot.py)      |     |  agent.py)       |
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
[Retrieve Chunks from Qdrant]
    |
    +--> chunks found --> [Grounded Mode]
    |
    +--> no matches   --> [Fallback Mode]
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

- **RAG-Grounded Generation**: Scripts are grounded in retrieved educational documents with inline `[Source N]` citations
- **Phased Generation**: Outline first, then block-by-block to handle small model output limits
- **Three Generation Modes**: Grounded (from sources), Fallback (general knowledge), Refuse (non-educational)
- **Iterative Revision**: Chat-style interface for modifying generated scripts block-by-block
- **Source Attribution**: Inline citations with a References section linking to source documents
- **Text-to-Speech**: Generate audio from scripts using Edge TTS or Bark
- **Auto Retrieval**: Infers subject, grade level, and topic from natural language prompts
- **Metadata-Aware Ingestion**: PDF metadata (subject, grade, topic) extracted via heuristics or LLM

## Tech Stack

| Component         | Technology                          |
|--------------------|-------------------------------------|
| Backend API        | FastAPI                             |
| LLM                | Ollama (qwen2.5:7b / gemma3:4b)    |
| Embeddings         | Ollama (nomic-embed-text)           |
| Vector Database    | Qdrant (Cloud or local)             |
| Frontend           | Streamlit + HTML/JS chat UI         |
| TTS                | Edge TTS / Bark                     |
| Language           | Python 3.11+                        |

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
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 5. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your Qdrant credentials:

```env
OLLAMA_BASE_URL=http://localhost:11434/api
OLLAMA_GENERATION_MODEL=qwen2.5:7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_api_key
QDRANT_COLLECTION=lesson_docs
```

### 6. Add Source Documents

Place educational PDFs in `data/raw/` and create metadata entries in `data/raw/metadata.json`:

```json
{
  "Algebra_1.pdf": {
    "title": "Algebra 1 Textbook",
    "subject": "mathematics",
    "grade_level": "8",
    "curriculum": "general",
    "topic": "algebra",
    "source_type": "textbook",
    "language": "en"
  }
}
```

Alternatively, auto-generate metadata suggestions:

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

The API runs at `http://localhost:8000` with a built-in chat UI.

**Streamlit frontend (alternative UI):**

```bash
streamlit run streamlit_app.py
```

The Streamlit app runs at `http://localhost:8501` and connects to the FastAPI backend.

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
    "message": "Create a 20-minute lesson on quadratic equations for grade 8"
  }'
```

## Project Structure

```
lesson-rag-agent/
├── app/
│   ├── main.py                  # FastAPI application and endpoints
│   ├── config.py                # Environment settings (from .env)
│   ├── schemas.py               # Pydantic request/response models
│   ├── services/
│   │   ├── agent.py             # Main orchestration (domain check, retrieval, generation)
│   │   ├── chatbot.py           # Conversational script generation and revision
│   │   ├── prompt_builder.py    # All LLM prompt templates and citation formatting
│   │   ├── retriever.py         # Qdrant vector search with metadata filtering
│   │   ├── ollama_client.py     # Ollama API client (generate + embed)
│   │   ├── qdrant_client.py     # Qdrant connection and collection management
│   │   ├── tts.py               # Text-to-speech generation (Edge TTS / Bark)
│   │   └── lesson_generator.py  # Legacy compatibility wrapper
│   ├── ingestion/
│   │   ├── run_ingestion.py     # CLI entry point for document ingestion
│   │   ├── load_docs.py         # PDF loading with metadata merging
│   │   ├── clean_docs.py        # Text cleaning and junk removal
│   │   ├── chunk_docs.py        # Structure-aware document chunking
│   │   ├── index_docs.py        # Embedding and Qdrant indexing
│   │   └── metadata_tools.py    # Metadata inference (heuristic + LLM)
│   └── static/
│       └── index.html           # Built-in browser chat UI
├── data/
│   └── raw/                     # Source PDFs and metadata.json
├── streamlit_app.py             # Streamlit frontend
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
└── README.md
```

## Key Design Decisions

1. **Phased generation over one-shot**: Small local models (4-8B parameters) cannot generate 3000+ word scripts in a single call. The system generates an outline, then fills each time block individually (max 5 minutes per LLM call), and assembles the result.

2. **Three-layer deduplication**: Small models tend to repeat content. The system applies within-block, cross-block, and full-script paragraph-level deduplication using fuzzy matching (SequenceMatcher).

3. **Citation system**: When generation is grounded in retrieved chunks, the LLM is instructed to cite sources inline as `[Source 1]`, `[Source 2]`. A References section with source metadata is appended to the script.

4. **Auto retrieval mode**: Instead of requiring users to specify subject/grade/topic, the system infers these from the natural language prompt and picks the best retrieval strategy automatically.

5. **Block-level revision**: When a user requests changes, only the affected time blocks are regenerated rather than the entire script, yielding 4-8x faster revisions.
