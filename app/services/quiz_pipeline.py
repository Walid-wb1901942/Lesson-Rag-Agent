"""RAG-grounded quiz generation pipeline.

Flow:
  1. Infer subject from content keywords (reuses SUBJECT_KEYWORDS from pipeline.py)
  2. Resolve retrieval mode (auto / filtered / all / skip)
  3. Retrieve relevant chunks from Qdrant
  4. Choose generation mode (grounded if chunks found, fallback otherwise)
  5. Build prompt and generate quiz text via Ollama
  6. Extract inline [Source N] citations and build References section
  7. Return structured result dict
"""
from __future__ import annotations

from app.services.ollama_client import generate_text
from app.services.prompt_builder import (
    build_references_section,
    extract_citations,
)
from app.services.qdrant_client import (
    field_value_exists,
    filter_has_matches,
)
from app.services.quiz_prompt_builder import (
    build_fallback_quiz_prompt,
    build_grounded_quiz_prompt,
)
from app.services.retriever import retrieve_chunks


# ---------------------------------------------------------------------------
# Subject inference (mirrors logic in ScriptPipeline._infer_subject)
# ---------------------------------------------------------------------------

_SUBJECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "mathematics": (
        "math", "mathematics", "algebra", "calculus", "geometry",
        "integral", "quadratic", "equation", "trigonometry", "precalculus",
    ),
    "chemistry": (
        "chemistry", "atomic number", "atomic mass", "molecule",
        "bond", "element", "compound", "periodic table", "atom",
    ),
    "biology": (
        "biology", "cell", "organism", "genetics", "evolution",
        "ecosystem", "photosynthesis", "dna",
    ),
    "physics": (
        "physics", "force", "energy", "velocity", "motion",
        "momentum", "gravity", "thermodynamics",
    ),
    "computer science": ("computer science", "programming", "coding", "algorithm"),
    "data science": ("data science",),
    "science": ("science",),
    "literature": ("literature", "english", "poetry", "novel", "grammar"),
    "health": ("health", "wellness", "nutrition", "hygiene"),
}


def _infer_subject(text: str) -> str | None:
    lower = text.lower()
    for subject, keywords in _SUBJECT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return subject
    return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class QuizPipeline:
    """Lightweight RAG pipeline for grounded quiz question generation."""

    def run(
        self,
        content: str,
        subject: str | None = None,
        grade_level: str | None = None,
        num_questions: int = 5,
        difficulty: str = "Mixed",
        question_types: str = "MCQ, Short Answer, Open-Ended",
        retrieval_limit: int = 5,
        retrieval_mode: str = "auto",
        retrieval_method: str = "dense",
    ) -> dict:
        """Run the full quiz generation pipeline and return a result dict.

        Returns:
            content            - original input content
            generation_mode    - "grounded" | "fallback"
            source_notice      - human-readable explanation of the mode
            retrieved_chunks   - list of chunk dicts used as context
            quiz_text          - generated quiz with optional References section
            citations          - list of Citation-compatible dicts
            agent_trace        - list of trace event dicts
        """
        trace: list[dict] = []

        # 1. Subject inference
        resolved_subject = subject or _infer_subject(content)
        resolved_grade = grade_level

        trace.append({
            "tool": "subject_inference",
            "resolved_subject": resolved_subject,
            "resolved_grade": resolved_grade,
        })

        # 2. Resolve retrieval mode
        resolved_mode = retrieval_mode
        if retrieval_mode == "auto":
            if resolved_subject and not field_value_exists("subject", resolved_subject):
                resolved_mode = "skip"
            elif resolved_subject and filter_has_matches(
                subject=resolved_subject, grade_level=resolved_grade
            ):
                resolved_mode = "filtered"
            else:
                resolved_mode = "all"

        trace.append({"tool": "mode_resolver", "retrieval_mode": resolved_mode})

        # 3. Retrieve chunks
        retrieved_chunks: list[dict] = []
        if resolved_mode != "skip":
            retrieved_chunks = retrieve_chunks(
                query=content,
                limit=retrieval_limit,
                subject=resolved_subject,
                grade_level=resolved_grade,
                retrieval_mode=resolved_mode,
                retrieval_method=retrieval_method,
            )

        trace.append({
            "tool": "retriever",
            "retrieval_mode": resolved_mode,
            "retrieval_method": retrieval_method,
            "num_chunks": len(retrieved_chunks),
            "top_score": retrieved_chunks[0]["score"] if retrieved_chunks else None,
        })

        # 4. Generation mode
        generation_mode = "grounded" if retrieved_chunks else "fallback"

        if generation_mode == "grounded":
            source_notice = (
                "Grounded — quiz questions are drawn from retrieved source material "
                f"({len(retrieved_chunks)} chunks retrieved via {retrieval_method} retrieval)."
            )
        else:
            reason = (
                "subject not found in corpus"
                if resolved_mode == "skip"
                else "no matching chunks returned"
            )
            source_notice = (
                f"Fallback — no relevant source material found ({reason}). "
                "Questions generated from general model knowledge."
            )

        trace.append({"tool": "mode_selector", "generation_mode": generation_mode})

        # 5. Build prompt and generate
        if generation_mode == "grounded":
            prompt = build_grounded_quiz_prompt(
                content=content,
                chunks=retrieved_chunks,
                subject=resolved_subject,
                grade_level=resolved_grade,
                num_questions=num_questions,
                difficulty=difficulty,
                question_types=question_types,
            )
        else:
            prompt = build_fallback_quiz_prompt(
                content=content,
                subject=resolved_subject,
                grade_level=resolved_grade,
                num_questions=num_questions,
                difficulty=difficulty,
                question_types=question_types,
            )

        quiz_text = generate_text(prompt, num_predict=2048)
        trace.append({"tool": "generator", "generation_mode": generation_mode})

        # 6. Citations and references
        citations = extract_citations(quiz_text, retrieved_chunks)
        references = build_references_section(retrieved_chunks)
        if references:
            quiz_text = f"{quiz_text}\n\n{references}"

        return {
            "content": content,
            "generation_mode": generation_mode,
            "source_notice": source_notice,
            "retrieved_chunks": retrieved_chunks,
            "quiz_text": quiz_text,
            "citations": citations,
            "agent_trace": trace,
        }
