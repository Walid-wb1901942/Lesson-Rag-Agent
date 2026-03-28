"""Streamlit frontend for the Lesson RAG Agent.

Connects to the FastAPI backend to generate and revise classroom scripts.
Run with: streamlit run streamlit_app.py
Requires the FastAPI server: uvicorn app.main:app
"""

import re
import streamlit as st
import requests
import time

try:
    API_BASE = st.secrets.get("API_BASE", "http://localhost:8000")
except Exception:
    API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Lesson RAG Agent",
    page_icon="📚",
    layout="wide",
)

# ── Session state ──────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_script" not in st.session_state:
    st.session_state.current_script = None
if "original_request" not in st.session_state:
    st.session_state.original_request = None
if "citations" not in st.session_state:
    st.session_state.citations = []
if "generation_mode" not in st.session_state:
    st.session_state.generation_mode = None
if "source_notice" not in st.session_state:
    st.session_state.source_notice = None
if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = []
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📚 Lesson RAG Agent")
    st.caption("Generate classroom scripts grounded in your documents")

    st.divider()

    duration = st.slider("Duration (minutes)", 5, 60, 40, step=5)

    subject = st.selectbox(
        "Subject",
        ["Auto-detect", "Mathematics", "Science", "Literature", "Health"],
    )
    subject_value = None if subject == "Auto-detect" else subject.lower()

    grade_level = st.text_input("Grade Level (optional)", placeholder="e.g. 8")
    grade_value = grade_level.strip() or None

    retrieval_limit = st.select_slider(
        "Top-k Chunks",
        options=[3, 5, 7],
        value=5,
        help="Number of source chunks retrieved per query (k=3, 5, or 7).",
    )

    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        ["auto", "filtered", "all"],
        help="auto: infers filters from your prompt. filtered: strict metadata match. all: search entire corpus.",
    )

    retrieval_method = st.selectbox(
        "Retrieval Method",
        ["dense", "hybrid"],
        help="dense: cosine similarity only. hybrid: dense + BM25 merged with RRF.",
    )

    st.divider()

    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        if resp.status_code == 200:
            st.success("Backend connected", icon="✅")
        else:
            st.error("Backend returned error", icon="❌")
    except requests.ConnectionError:
        st.error("Backend not running. Start with:\n`uvicorn app.main:app`", icon="❌")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_script = None
            st.session_state.original_request = None
            st.session_state.citations = []
            st.session_state.generation_mode = None
            st.session_state.source_notice = None
            st.session_state.retrieved_chunks = []
            st.session_state.audio_bytes = None
            st.rerun()
    with col_b:
        if st.button(
            "📝 New Script",
            use_container_width=True,
            disabled=not bool(st.session_state.current_script),
        ):
            st.session_state.current_script = None
            st.session_state.original_request = None
            st.session_state.audio_bytes = None
            st.rerun()


# ── Helper functions ───────────────────────────────────────────────────────

_DURATION_RE = re.compile(r'\b\d+\s*[-–]?\s*minute', re.IGNORECASE)
_NEW_SCRIPT_KEYWORDS = (
    "new script", "new lesson", "generate another", "create another",
    "write another", "different lesson", "different script", "another script",
)


def call_chat_api(message: str) -> tuple[dict | None, str]:
    """Submit a script generation job and poll until complete.

    Returns (result, effective_message) where effective_message is the actual
    message sent to the API (may include injected duration suffix).
    """
    if st.session_state.current_script and any(kw in message.lower() for kw in _NEW_SCRIPT_KEYWORDS):
        st.session_state.current_script = None
        st.session_state.original_request = None
        st.session_state.audio_bytes = None

    payload = {
        "message": message,
        "retrieval_mode": retrieval_mode,
        "retrieval_limit": retrieval_limit,
        "retrieval_method": retrieval_method,
    }

    if subject_value:
        payload["subject"] = subject_value
    if grade_value:
        payload["grade_level"] = grade_value

    if st.session_state.current_script:
        payload["current_script"] = st.session_state.current_script
        payload["original_request"] = st.session_state.original_request
    else:
        # Only inject slider duration if the prompt doesn't already specify one
        if not _DURATION_RE.search(message):
            message = f"{message} ({duration}-minute lesson)"
            payload["message"] = message

    try:
        # Submit the job — returns immediately with a job_id
        resp = requests.post(f"{API_BASE}/chat/script", json=payload, timeout=30)
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

        # Poll until done
        progress = st.progress(0, text="Generating script...")
        for i in range(300):  # up to 15 minutes (3s × 300)
            time.sleep(3)
            status_resp = requests.get(f"{API_BASE}/chat/status/{job_id}", timeout=10)
            status = status_resp.json()
            progress.progress(min((i + 1) * 1, 95), text="Generating script...")
            if status["status"] == "done":
                progress.progress(100, text="Done!")
                return status["result"], message
            if status["status"] == "error":
                st.error(f"Generation failed: {status.get('error', 'Unknown error')}")
                return None, message
        st.warning("Generation timed out after 15 minutes.")
        return None, message

    except requests.ConnectionError:
        st.error("Cannot connect to backend. Is it running?")
        return None, message
    except requests.HTTPError as e:
        st.error(f"API error: {e}")
        return None, message


def generate_tts(script: str) -> bytes | None:
    """Start TTS generation, poll until complete, and return audio bytes."""
    try:
        resp = requests.post(
            f"{API_BASE}/tts/generate",
            json={"script": script, "engine": "edge"},
            timeout=30,
        )
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

        progress = st.progress(0, text="Generating audio...")
        for i in range(120):
            time.sleep(2)
            status = requests.get(f"{API_BASE}/tts/status/{job_id}", timeout=10).json()
            progress.progress(min((i + 1) * 2, 95), text="Generating audio...")
            if status["status"] == "done":
                progress.progress(100, text="Audio ready!")
                audio_resp = requests.get(f"{API_BASE}/tts/download/{job_id}", timeout=30)
                return audio_resp.content
            if status["status"] == "error":
                st.error(f"TTS failed: {status.get('error', 'Unknown error')}")
                return None
        st.warning("TTS timed out")
        return None
    except requests.ConnectionError:
        st.error("Cannot connect to backend for TTS.")
        return None


# ── Global CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Wrap long lines in chat messages */
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span {
    word-wrap: break-word;
    overflow-wrap: break-word;
    white-space: pre-wrap;
}
/* Remove default max-width cap on chat message content */
[data-testid="stChatMessageContent"] {
    max-width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ── Main chat area ─────────────────────────────────────────────────────────

st.header("Classroom Script Generator")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Describe the lesson you want..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result, sent_message = call_chat_api(prompt)

        if result:
            chat_mode = result.get("chat_mode", "")
            lesson_text = result.get("lesson_text", "")
            assistant_msg = result.get("assistant_message", "")
            source_notice = result.get("source_notice", "")
            gen_mode = result.get("generation_mode", "")
            citations = result.get("citations", [])
            chunks = result.get("retrieved_chunks", [])

            if chat_mode not in ("refuse", "input_error"):
                st.session_state.current_script = lesson_text
                if not st.session_state.original_request:
                    st.session_state.original_request = sent_message
                st.session_state.citations = citations
                st.session_state.generation_mode = gen_mode
                st.session_state.source_notice = source_notice
                st.session_state.retrieved_chunks = chunks
                st.session_state.audio_bytes = None

            if chat_mode in ("refuse", "input_error"):
                st.warning(lesson_text)
                display_text = lesson_text
            else:
                st.info(f"**Mode:** {gen_mode} | {source_notice}")
                st.markdown(assistant_msg)
                display_text = assistant_msg

            st.session_state.messages.append({"role": "assistant", "content": display_text})

# ── Script display area ────────────────────────────────────────────────────

if st.session_state.current_script:
    st.divider()

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.subheader("Generated Script")
    with col2:
        st.download_button(
            "⬇️ Download",
            data=st.session_state.current_script,
            file_name="lesson_script.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col3:
        if st.button("🔊 Generate Audio", use_container_width=True):
            audio_bytes = generate_tts(st.session_state.current_script)
            if audio_bytes:
                st.session_state.audio_bytes = audio_bytes
                st.rerun()

    if st.session_state.audio_bytes:
        st.audio(st.session_state.audio_bytes, format="audio/mp3")

    script_safe = (
        st.session_state.current_script
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    st.markdown(
        f'<div style="'
        f'white-space:pre-wrap;'
        f'word-wrap:break-word;'
        f'overflow-wrap:break-word;'
        f'font-family:monospace;'
        f'font-size:0.85rem;'
        f'line-height:1.7;'
        f'padding:1rem 1.25rem;'
        f'background:#f8f9fa;'
        f'border:1px solid #dee2e6;'
        f'border-radius:0.5rem;'
        f'max-height:600px;'
        f'overflow-y:auto;'
        f'">{script_safe}</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.citations:
        with st.expander("📝 Citations Used", expanded=False):
            for cite in st.session_state.citations:
                parts = [f"**[Source {cite['source_number']}]** {cite.get('title', 'Untitled')}"]
                if cite.get("pages"):
                    parts.append(f"pp. {cite['pages']}")
                details = []
                if cite.get("subject"):
                    details.append(cite["subject"])
                if cite.get("grade_level"):
                    details.append(f"Grade {cite['grade_level']}")
                if cite.get("topic"):
                    details.append(cite["topic"])
                if details:
                    parts.append(f"({', '.join(details)})")
                st.markdown(" \u2014 ".join(parts))

    if st.session_state.retrieved_chunks:
        with st.expander(f"📚 Retrieved Sources ({len(st.session_state.retrieved_chunks)} chunks)", expanded=False):
            for i, chunk in enumerate(st.session_state.retrieved_chunks, 1):
                meta = chunk.get("metadata", {})
                score = chunk.get("score", 0)
                title = meta.get("title", "Untitled")
                pages = meta.get("page_range") or meta.get("page_number") or "N/A"
                score_label = f"{score:.4f} (RRF)" if score < 0.1 else f"{score:.3f}"
                st.markdown(f"**[Source {i}]** {title} (pp. {pages}) \u2014 Score: {score_label}")
                st.markdown(chunk.get("text", "")[:300] + "...")
                st.divider()
