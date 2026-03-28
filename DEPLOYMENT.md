# Deployment Guide

## Architecture Constraints

This application has two components with different hosting requirements:

| Component | Requirement | Cloud-hostable? |
|-----------|-------------|-----------------|
| Streamlit frontend | Any Python host | Yes — Streamlit Community Cloud (free) |
| FastAPI + Ollama backend | Local GPU/CPU for LLM inference | No — must run on your machine or a GPU server |
| Qdrant vector database | Already cloud-hosted | Yes — Qdrant Cloud |

The recommended deployment for demonstration is:

- **Streamlit frontend** → Streamlit Community Cloud (public URL, free)
- **FastAPI + Ollama backend** → your local machine, exposed via `ngrok` tunnel
- **Qdrant** → Qdrant Cloud (already configured)

---

## Option A: Local Demo (Recommended for Submission)

Run everything locally. This is the simplest and most reliable option for showing the system working.

```bash
# Terminal 1 — start Ollama (if not already running as a service)
ollama serve

# Terminal 2 — start FastAPI backend
uvicorn app.main:app --reload

# Terminal 3 — start Streamlit
streamlit run streamlit_app.py
```

Access at `http://localhost:8501`.

---

## Option B: Streamlit Cloud + ngrok Tunnel (Public URL)

This gives you a shareable public URL for the Streamlit UI while keeping Ollama running locally.

### Step 1 — Expose the backend with ngrok

Install ngrok from https://ngrok.com, then:

```bash
# Start your FastAPI backend
uvicorn app.main:app

# In a separate terminal, expose port 8000
ngrok http 8000
```

ngrok will print a public URL like `https://abc123.ngrok-free.app`. Copy it.

### Step 2 — Update Streamlit to use the ngrok URL

In `streamlit_app.py`, change:

```python
API_BASE = "http://localhost:8000"
```

to:

```python
API_BASE = "https://abc123.ngrok-free.app"  # your ngrok URL
```

### Step 3 — Deploy Streamlit to Streamlit Community Cloud

1. Push your repository to GitHub (ensure `.env` is in `.gitignore` — it already is)
2. Go to https://share.streamlit.io and sign in with GitHub
3. Click **New app**, select your repo, set the main file to `streamlit_app.py`
4. Under **Advanced settings → Secrets**, add your environment variables:

```toml
[secrets]
API_BASE = "https://abc123.ngrok-free.app"
```

5. Update `streamlit_app.py` to read the API base from secrets:

```python
import streamlit as st
API_BASE = st.secrets.get("API_BASE", "http://localhost:8000")
```

6. Deploy. Streamlit Community Cloud will give you a public URL.

**Note:** The ngrok tunnel must stay active on your local machine for the deployed Streamlit app to reach your backend.

---

## Option C: Full Cloud Deployment (GPU Server)

For a persistent public deployment without keeping your local machine on:

1. Rent a GPU instance on [RunPod](https://runpod.io), [vast.ai](https://vast.ai), or AWS EC2 (g4dn.xlarge)
2. Install Ollama and pull your models on the server
3. Deploy FastAPI with a process manager:

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

4. Point your Streamlit app (local or Streamlit Cloud) at the server's public IP.

**Estimated cost:** ~$0.20–0.50/hour for a GPU instance sufficient for qwen2.5:7b.

---

## Environment Variables for Deployment

Never commit `.env` to your repository. Use the host's secrets/environment variable system instead.

Required variables:

```
OLLAMA_BASE_URL=http://localhost:11434/api
OLLAMA_GENERATION_MODEL=qwen2.5:7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_api_key
QDRANT_COLLECTION=lesson_docs
CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_TOKENS=100
EMBEDDING_BATCH_SIZE=64
QDRANT_UPSERT_BATCH_SIZE=32
```
