"""Streamlit frontend for the Lesson RAG Agent.

Connects to the FastAPI backend to generate and revise classroom scripts.
Run with: streamlit run streamlit_app.py
Requires the FastAPI server: uvicorn app.main:app
"""

import streamlit as st
import requests
import time

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

    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        ["auto", "filtered", "all"],
        help="auto: infers filters from your prompt. filtered: strict metadata match. all: search entire corpus.",
    )

    st.divider()

    # Health check
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        if resp.status_code == 200:
            st.success("Backend connected", icon="✅")
        else:
            st.error("Backend returned error", icon="❌")
    except requests.ConnectionError:
        st.error("Backend not running. Start with:\n`uvicorn app.main:app`", icon="❌")

    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_script = None
        st.session_state.original_request = None
        st.session_state.citations = []
        st.session_state.generation_mode = None
        st.session_state.source_notice = None
        st.session_state.retrieved_chunks = []
        st.rerun()


# ── Helper functions ───────────────────────────────────────────────────────

def call_chat_api(message: str) -> dict | None:
    """Send a message to the chat/script endpoint."""
    payload = {
        "message": message,
        "retrieval_mode": retrieval_mode,
        "retrieval_limit": 5,
    }

    if subject_value:
        payload["subject"] = subject_value
    if grade_value:
        payload["grade_level"] = grade_value

    if st.session_state.current_script:
        payload["current_script"] = st.session_state.current_script
        payload["original_request"] = st.session_state.original_request
    else:
        # Inject duration into the prompt for initial generation
        if f"{duration}-minute" not in message and f"{duration} minute" not in message:
            message = f"{message} ({duration}-minute lesson)"
            payload["message"] = message

    try:
        resp = requests.post(
            f"{API_BASE}/chat/script",
            json=payload,
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("Cannot connect to backend. Is it running?")
        return None
    except requests.HTTPError as e:
        st.error(f"API error: {e}")
        return None


def generate_tts(script: str) -> str | None:
    """Start TTS generation and poll until complete. Returns job_id."""
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
                return job_id
            if status["status"] == "error":
                st.error(f"TTS failed: {status.get('error', 'Unknown error')}")
                return None
        st.warning("TTS timed out")
        return None
    except requests.ConnectionError:
        st.error("Cannot connect to backend for TTS.")
        return None


# ── Main chat area ─────────────────────────────────────────────────────────

st.header("Classroom Script Generator")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Describe the lesson you want..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Generating lesson script..." if not st.session_state.current_script else "Revising script..."):
            result = call_chat_api(prompt)

        if result:
            chat_mode = result.get("chat_mode", "")
            lesson_text = result.get("lesson_text", "")
            assistant_msg = result.get("assistant_message", "")
            source_notice = result.get("source_notice", "")
            gen_mode = result.get("generation_mode", "")
            citations = result.get("citations", [])
            chunks = result.get("retrieved_chunks", [])

            # Update state
            if chat_mode not in ("refuse", "input_error"):
                st.session_state.current_script = lesson_text
                if not st.session_state.original_request:
                    st.session_state.original_request = prompt
                st.session_state.citations = citations
                st.session_state.generation_mode = gen_mode
                st.session_state.source_notice = source_notice
                st.session_state.retrieved_chunks = chunks

            # Display response
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

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Generated Script")

    with col2:
        if st.button("🔊 Generate Audio", use_container_width=True):
            job_id = generate_tts(st.session_state.current_script)
            if job_id:
                st.markdown(
                    f"[Download Audio]({API_BASE}/tts/download/{job_id})"
                )

    st.code(st.session_state.current_script, language=None)

    # Citations
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
                st.markdown(" — ".join(parts))

    # Retrieved chunks
    if st.session_state.retrieved_chunks:
        with st.expander(f"📚 Retrieved Sources ({len(st.session_state.retrieved_chunks)} chunks)", expanded=False):
            for i, chunk in enumerate(st.session_state.retrieved_chunks, 1):
                meta = chunk.get("metadata", {})
                score = chunk.get("score", 0)
                title = meta.get("title", "Untitled")
                pages = meta.get("page_range") or meta.get("page_number") or "N/A"
                st.markdown(f"**[Source {i}]** {title} (pp. {pages}) — Score: {score:.3f}")
                st.text(chunk.get("text", "")[:300] + "...")
                st.divider()
