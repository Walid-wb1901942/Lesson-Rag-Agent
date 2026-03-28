"""Streamlit frontend for the RAG-Grounded Quiz Generator.

Connects to the FastAPI backend (/quiz/start + /quiz/status) to generate
assessment questions grounded in the indexed document corpus.

Run with: streamlit run streamlit_quiz_app.py
Requires the FastAPI server: uvicorn app.main:app
"""

import streamlit as st
import requests
import time

try:
    API_BASE = st.secrets.get("API_BASE", "http://localhost:8000")
except Exception:
    API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Quiz Generator — RAG",
    page_icon="📝",
    layout="wide",
)

# ── Session state ──────────────────────────────────────────────────────────

for key, default in [
    ("quiz_text", None),
    ("generation_mode", None),
    ("source_notice", None),
    ("retrieved_chunks", []),
    ("citations", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar controls ───────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Quiz Settings")

    subject = st.selectbox(
        "Subject",
        ["(auto-detect)", "mathematics", "chemistry", "biology", "physics",
         "computer science", "data science", "literature", "health", "science"],
    )
    subject_val = None if subject == "(auto-detect)" else subject

    grade_level = st.selectbox(
        "Grade Level",
        ["(any)", "Elementary", "Middle School", "High School",
         "Grade 9", "Grade 10", "Grade 11", "Grade 12", "College / University"],
    )
    grade_val = None if grade_level == "(any)" else grade_level

    num_questions = st.slider("Number of Questions", min_value=3, max_value=15, value=5)

    difficulty = st.selectbox("Difficulty", ["Mixed", "Easy", "Medium", "Hard"])

    st.markdown("**Question Types**")
    use_mcq = st.checkbox("Multiple Choice (MCQ)", value=True)
    use_sa = st.checkbox("Short Answer", value=True)
    use_oe = st.checkbox("Open-Ended", value=True)
    selected_types = ", ".join(
        t for t, on in [("MCQ", use_mcq), ("Short Answer", use_sa), ("Open-Ended", use_oe)] if on
    ) or "MCQ"

    st.markdown("---")
    st.markdown("**Retrieval Settings**")
    retrieval_method = st.selectbox("Retrieval Method", ["dense", "hybrid"])
    retrieval_limit = st.slider("Top-k chunks", min_value=3, max_value=10, value=5)

    st.markdown("---")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        st.success("API: connected ✓")
    except Exception:
        st.error("API: unreachable ✗")

# ── Helper: submit job and poll ────────────────────────────────────────────

def call_quiz_api(payload: dict) -> dict | None:
    """Submit a quiz job and poll until complete. Returns result dict or None."""
    try:
        resp = requests.post(f"{API_BASE}/quiz/start", json=payload, timeout=30)
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
    except Exception as e:
        st.error(f"Failed to start job: {e}")
        return None

    progress = st.progress(0, text="Retrieving relevant source chunks…")
    start = time.time()
    max_wait = 300  # 5-minute cap

    for i in range(max_wait):
        time.sleep(2)
        try:
            status_resp = requests.get(f"{API_BASE}/quiz/status/{job_id}", timeout=10).json()
        except Exception:
            continue

        elapsed = time.time() - start
        pct = min(int((elapsed / max_wait) * 95), 95)
        label = "Generating grounded questions…" if elapsed > 8 else "Retrieving relevant source chunks…"
        progress.progress(pct, text=label)

        if status_resp["status"] == "done":
            progress.progress(100, text="Done!")
            return status_resp["result"]
        if status_resp["status"] == "error":
            st.error(f"Generation error: {status_resp.get('error', 'unknown')}")
            return None

    st.error("Timed out waiting for quiz generation.")
    return None

# ── Main UI ────────────────────────────────────────────────────────────────

st.title("📝 AI Quiz Generator")
st.caption("Generates assessment questions grounded in your document corpus via RAG.")

content = st.text_area(
    "Lesson Objectives or Content",
    height=180,
    placeholder=(
        "Paste lesson objectives or a brief content description here.\n\n"
        "Example: Students will understand the structure of an atom, including "
        "protons, neutrons, and electrons. They should know the charges of each "
        "subatomic particle and how they are arranged in the nucleus and orbitals."
    ),
)

generate_clicked = st.button("Generate Quiz", type="primary", use_container_width=True)

if generate_clicked:
    if not content.strip():
        st.warning("Please enter lesson objectives or content before generating.")
    else:
        st.session_state.quiz_text = None
        st.session_state.generation_mode = None
        st.session_state.source_notice = None
        st.session_state.retrieved_chunks = []
        st.session_state.citations = []

        payload = {
            "content": content.strip(),
            "subject": subject_val,
            "grade_level": grade_val,
            "num_questions": num_questions,
            "difficulty": difficulty,
            "question_types": selected_types,
            "retrieval_limit": retrieval_limit,
            "retrieval_mode": "auto",
            "retrieval_method": retrieval_method,
        }

        result = call_quiz_api(payload)
        if result:
            st.session_state.quiz_text = result.get("quiz_text", "")
            st.session_state.generation_mode = result.get("generation_mode", "")
            st.session_state.source_notice = result.get("source_notice", "")
            st.session_state.retrieved_chunks = result.get("retrieved_chunks", [])
            st.session_state.citations = result.get("citations", [])

# ── Results ────────────────────────────────────────────────────────────────

if st.session_state.quiz_text:
    mode = st.session_state.generation_mode
    notice = st.session_state.source_notice

    if mode == "grounded":
        st.success(f"✅ Grounded — {notice}")
    else:
        st.warning(f"⚠️ Fallback — {notice}")

    st.markdown("---")
    st.subheader("Generated Quiz")
    st.markdown(st.session_state.quiz_text)

    # Download button
    st.download_button(
        label="⬇️ Download Quiz (.txt)",
        data=st.session_state.quiz_text,
        file_name="quiz.txt",
        mime="text/plain",
    )

    # Retrieved sources
    chunks = st.session_state.retrieved_chunks
    if chunks:
        with st.expander(f"📚 Retrieved Sources ({len(chunks)} chunks)", expanded=False):
            for i, chunk in enumerate(chunks, 1):
                meta = chunk.get("metadata", {})
                score = chunk.get("score", 0)
                score_label = f"{score:.4f} (RRF)" if score < 0.1 else f"{score:.3f}"
                st.markdown(
                    f"**[Source {i}]** {meta.get('title', 'Untitled')} "
                    f"— pp. {meta.get('page_range') or meta.get('page_number', '?')} "
                    f"| {meta.get('subject', '')} "
                    f"| Score: {score_label}"
                )
                with st.expander("Show chunk text", expanded=False):
                    st.text(chunk.get("text", "")[:800])

    # Citations
    citations = st.session_state.citations
    if citations:
        with st.expander(f"🔖 Citations ({len(citations)})", expanded=False):
            for c in citations:
                st.markdown(
                    f"**[Source {c['source_number']}]** {c.get('title', 'Untitled')} "
                    f"pp. {c.get('pages', '?')} — {c.get('subject', '')} "
                    f"Grade {c.get('grade_level', '?')}"
                )
