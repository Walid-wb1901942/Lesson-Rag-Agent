"""Streamlit frontend for the Lesson RAG Agent.

Connects to the FastAPI backend to generate and revise classroom scripts.
Run with: streamlit run streamlit_app.py
Requires the FastAPI server: uvicorn app.main:app
"""

import re
import streamlit as st
import requests
import time

API_BASE = st.secrets.get("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="Lesson RAG Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ──────────────────────────────────────────────────────────

for key, default in [
    ("messages", []),
    ("current_script", None),
    ("original_request", None),
    ("citations", []),
    ("generation_mode", None),
    ("source_notice", None),
    ("retrieved_chunks", []),
    ("audio_bytes", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Global CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base typography ── */
html, body, [class*="css"] { font-family: "Inter", "Segoe UI", sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    border-right: 1px solid #4338ca;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #e0e7ff !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #fff !important; }
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stTextInput > div > div > input,
section[data-testid="stSidebar"] .stSlider { background: #312e81 !important; }
section[data-testid="stSidebar"] hr { border-color: #4338ca !important; opacity: 0.5; }

/* ── Chat messages ── */
[data-testid="stChatMessageContent"] {
    max-width: 100%;
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
}
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span {
    word-wrap: break-word;
    overflow-wrap: break-word;
    white-space: pre-wrap;
}
/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
[data-testid="stChatMessageContent"] {
    background: #eef2ff;
    border-left: 3px solid #4f46e5;
}
/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
[data-testid="stChatMessageContent"] {
    background: #f0fdf4;
    border-left: 3px solid #059669;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.18s ease;
    border: none;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(79,70,229,0.25);
}
section[data-testid="stSidebar"] .stButton > button {
    background: #4338ca;
    color: #fff !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #4f46e5;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-weight: 500;
    padding: 0.4rem 1rem;
}

/* ── Script container ── */
.script-box {
    font-family: "Georgia", "Times New Roman", serif;
    font-size: 0.9rem;
    line-height: 1.85;
    padding: 1.5rem 2rem;
    background: #fafafa;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    max-height: 620px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-wrap: break-word;
    color: #111827;
}
.script-box::-webkit-scrollbar { width: 6px; }
.script-box::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 3px; }
.script-box::-webkit-scrollbar-thumb { background: #94a3b8; border-radius: 3px; }

/* ── Mode badges ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    margin-right: 0.4rem;
}
.badge-grounded  { background: #d1fae5; color: #065f46; }
.badge-fallback  { background: #fef3c7; color: #92400e; }
.badge-refuse    { background: #fee2e2; color: #991b1b; }
.badge-revised   { background: #e0e7ff; color: #3730a3; }
.badge-generated { background: #d1fae5; color: #065f46; }

/* ── Source card ── */
.source-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
}
.source-card .source-title { font-weight: 600; font-size: 0.88rem; color: #1e293b; }
.source-card .source-meta  { font-size: 0.78rem; color: #64748b; margin-top: 0.2rem; }
.source-card .source-text  { font-size: 0.82rem; color: #374151; margin-top: 0.5rem; line-height: 1.6; }

/* ── Stats row ── */
.stats-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 0.75rem;
    flex-wrap: wrap;
    align-items: center;
}
.stat-chip {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.3rem 0.75rem;
    font-size: 0.78rem;
    color: #475569;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────

_DURATION_RE = re.compile(r'\b\d+\s*[-–]?\s*minute', re.IGNORECASE)
_NEW_SCRIPT_KEYWORDS = (
    "new script", "new lesson", "generate another", "create another",
    "write another", "different lesson", "different script", "another script",
)

_MODE_BADGE = {
    "grounded":  '<span class="badge badge-grounded">Grounded</span>',
    "fallback":  '<span class="badge badge-fallback">Fallback</span>',
    "refuse":    '<span class="badge badge-refuse">Refused</span>',
    "revised":   '<span class="badge badge-revised">Revised</span>',
    "generated": '<span class="badge badge-generated">Generated</span>',
}


def mode_badge(mode: str) -> str:
    return _MODE_BADGE.get(mode, f'<span class="badge badge-fallback">{mode}</span>')


def call_chat_api(message: str) -> tuple[dict | None, str]:
    """Send a message to the chat/script endpoint.

    Returns (result, effective_message) where effective_message is the actual
    message sent to the API (may include injected duration suffix).
    """
    if st.session_state.current_script and any(
        kw in message.lower() for kw in _NEW_SCRIPT_KEYWORDS
    ):
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
        if not _DURATION_RE.search(message):
            message = f"{message} ({duration}-minute lesson)"
            payload["message"] = message

    try:
        resp = requests.post(
            f"{API_BASE}/chat/script",
            json=payload,
            timeout=1200,
        )
        resp.raise_for_status()
        return resp.json(), message
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


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📚 Lesson RAG Agent")
    st.caption("Generate classroom scripts grounded in your documents")

    st.divider()

    st.markdown("**Lesson Settings**")
    duration = st.slider("Duration (minutes)", 5, 60, 40, step=5)

    subject = st.selectbox(
        "Subject",
        ["Auto-detect", "Mathematics", "Science", "Literature", "Health"],
    )
    subject_value = None if subject == "Auto-detect" else subject.lower()

    grade_level = st.text_input("Grade Level", placeholder="e.g. 8  (optional)")
    grade_value = grade_level.strip() or None

    st.divider()

    st.markdown("**Retrieval Settings**")
    retrieval_limit = st.select_slider(
        "Top-k Chunks",
        options=[3, 5, 7],
        value=5,
        help="Number of source chunks retrieved per query.",
    )
    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        ["auto", "filtered", "all"],
        help="auto — infers filters from your prompt.\nfiltered — strict metadata match.\nall — search entire corpus.",
    )
    retrieval_method = st.selectbox(
        "Retrieval Method",
        ["dense", "hybrid"],
        help="dense — cosine similarity only.\nhybrid — dense + BM25 merged with RRF.",
    )

    st.divider()

    # Backend status
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        if resp.status_code == 200:
            st.success("Backend connected", icon="✅")
        else:
            st.error("Backend error", icon="❌")
    except requests.ConnectionError:
        st.error("Backend offline\n`uvicorn app.main:app`", icon="❌")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 New Chat", use_container_width=True):
            for k in ("messages", "citations", "retrieved_chunks"):
                st.session_state[k] = []
            for k in ("current_script", "original_request", "generation_mode",
                      "source_notice", "audio_bytes"):
                st.session_state[k] = None
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

# ── Page header ────────────────────────────────────────────────────────────

st.markdown("""
<div style="
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 60%, #4338ca 100%);
    padding: 1.25rem 1.75rem;
    border-radius: 12px;
    margin-bottom: 1.25rem;
">
    <h2 style="color:#fff; margin:0; font-size:1.4rem; font-weight:700; letter-spacing:-0.01em;">
        Classroom Script Generator
    </h2>
    <p style="color:#c7d2fe; margin:0.25rem 0 0; font-size:0.85rem;">
        Describe the lesson you want — the system will retrieve relevant source material
        and generate a word-for-word classroom script.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Chat history ───────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────

if prompt := st.chat_input(
    "Describe the lesson you want..." if not st.session_state.current_script
    else "Request a revision or ask for a new script..."
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        spinner_text = (
            "Revising script..." if st.session_state.current_script
            else "Retrieving sources and generating script..."
        )
        with st.spinner(spinner_text):
            result, sent_message = call_chat_api(prompt)

        if result:
            chat_mode   = result.get("chat_mode", "")
            lesson_text = result.get("lesson_text", "")
            asst_msg    = result.get("assistant_message", "")
            source_notice = result.get("source_notice", "")
            gen_mode    = result.get("generation_mode", "")
            citations   = result.get("citations", [])
            chunks      = result.get("retrieved_chunks", [])

            if chat_mode not in ("refuse", "input_error"):
                st.session_state.current_script  = lesson_text
                if not st.session_state.original_request:
                    st.session_state.original_request = sent_message
                st.session_state.citations        = citations
                st.session_state.generation_mode  = gen_mode
                st.session_state.source_notice    = source_notice
                st.session_state.retrieved_chunks = chunks
                st.session_state.audio_bytes      = None

            if chat_mode in ("refuse", "input_error"):
                st.warning(lesson_text)
                display_text = lesson_text
            else:
                # Mode badge + source notice
                badge_html = mode_badge(gen_mode)
                st.markdown(
                    f'{badge_html} <span style="font-size:0.82rem;color:#6b7280;">{source_notice}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(asst_msg)
                display_text = asst_msg

            st.session_state.messages.append({"role": "assistant", "content": display_text})

# ── Script panel ───────────────────────────────────────────────────────────

if st.session_state.current_script:
    st.divider()

    script = st.session_state.current_script
    word_count = len(script.split())
    est_minutes = round(word_count / 95)
    n_sources = len(st.session_state.retrieved_chunks)
    n_citations = len(st.session_state.citations)

    # Header row: title + action buttons
    hcol1, hcol2, hcol3 = st.columns([4, 1, 1])
    with hcol1:
        gen_mode = st.session_state.generation_mode or ""
        badge_html = mode_badge(gen_mode)
        st.markdown(
            f'<span style="font-size:1.1rem;font-weight:700;color:#1e293b;">Generated Script</span>'
            f'&nbsp;&nbsp;{badge_html}',
            unsafe_allow_html=True,
        )
        # Stats chips
        st.markdown(
            f'<div class="stats-row">'
            f'<span class="stat-chip">📝 {word_count:,} words</span>'
            f'<span class="stat-chip">⏱ ~{est_minutes} min read-aloud</span>'
            f'<span class="stat-chip">📚 {n_sources} sources retrieved</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with hcol2:
        st.download_button(
            "⬇️ Download",
            data=script,
            file_name="lesson_script.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with hcol3:
        if st.button("🔊 Audio", use_container_width=True):
            audio_bytes = generate_tts(script)
            if audio_bytes:
                st.session_state.audio_bytes = audio_bytes
                st.rerun()

    if st.session_state.audio_bytes:
        st.audio(st.session_state.audio_bytes, format="audio/mp3")

    # Tabs: Script | Citations | Sources
    tab_labels = [
        "📄 Script",
        f"📝 Citations ({n_citations})",
        f"📚 Sources ({n_sources})",
    ]
    tab_script, tab_citations, tab_sources = st.tabs(tab_labels)

    with tab_script:
        script_safe = (
            script
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        st.markdown(f'<div class="script-box">{script_safe}</div>', unsafe_allow_html=True)

    with tab_citations:
        if st.session_state.citations:
            for cite in st.session_state.citations:
                title   = cite.get("title", "Untitled")
                pages   = cite.get("pages")
                subject = cite.get("subject", "")
                grade   = cite.get("grade_level", "")
                topic   = cite.get("topic", "")
                num     = cite.get("source_number", "?")

                meta_parts = []
                if subject:
                    meta_parts.append(subject.capitalize())
                if grade:
                    meta_parts.append(f"Grade {grade}")
                if topic:
                    meta_parts.append(topic)
                meta_str = " · ".join(meta_parts) if meta_parts else "No metadata"
                pages_str = f"pp. {pages}" if pages else "page unknown"

                st.markdown(
                    f'<div class="source-card">'
                    f'<div class="source-title">[Source {num}] {title}</div>'
                    f'<div class="source-meta">{pages_str} &nbsp;·&nbsp; {meta_str}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No citations — script was generated in fallback mode.")

    with tab_sources:
        if st.session_state.retrieved_chunks:
            for i, chunk in enumerate(st.session_state.retrieved_chunks, 1):
                meta  = chunk.get("metadata", {})
                score = chunk.get("score", 0)
                title = meta.get("title", "Untitled")
                pages = meta.get("page_range") or meta.get("page_number") or "N/A"
                subj  = meta.get("subject", "")
                grade = meta.get("grade_level", "")
                preview = chunk.get("text", "")[:280].replace("<", "&lt;").replace(">", "&gt;")

                meta_parts = []
                if subj:
                    meta_parts.append(subj.capitalize())
                if grade:
                    grade_str = "/".join(grade) if isinstance(grade, list) else str(grade)
                    meta_parts.append(f"Grade {grade_str}")
                meta_str = " · ".join(meta_parts) if meta_parts else ""

                st.markdown(
                    f'<div class="source-card">'
                    f'<div class="source-title">[Source {i}] {title}'
                    f'<span style="float:right;font-size:0.75rem;color:#94a3b8;font-weight:400;">'
                    f'score {score:.3f}</span></div>'
                    f'<div class="source-meta">pp. {pages}'
                    + (f' &nbsp;·&nbsp; {meta_str}' if meta_str else '') +
                    f'</div>'
                    f'<div class="source-text">{preview}…</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No sources retrieved — script was generated in fallback mode.")
