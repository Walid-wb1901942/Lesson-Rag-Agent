"""Prompt templates for the RAG-grounded quiz generation pipeline.

The grounded prompt incorporates retrieved document chunks as [Source N] context
blocks and instructs the model to cite them inline.  The fallback prompt generates
questions from general knowledge when no relevant corpus content is found.

Both prompts use the same Chain-of-Thought + Input Quality Gate design from A1's
PROMPT_FINAL, so output format and reasoning depth are consistent across modes.
"""
from __future__ import annotations

from app.services.prompt_builder import format_retrieved_chunks


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_BLOOM_LEVELS = "Remember, Understand, Apply, Analyze, Evaluate, Create"

_OUTPUT_FORMAT = """
## Output Format
For each question provide:
- **Question number and type** (MCQ / Short Answer / Open-Ended)
- **Difficulty level** (Easy / Medium / Hard)
- **Question text**
- For MCQs: four options labeled A–D with the correct answer marked ✓
- For Short Answer: a model answer (1–2 sentences)
- For Open-Ended: key points the response should address
- **Bloom's Taxonomy level** ({bloom})
""".strip().format(bloom=_BLOOM_LEVELS)

_QUALITY_GUIDELINES = """
## Quality Guidelines
- MCQ distractors must be plausible and grounded in common student misconceptions — not obviously wrong.
- Questions must align directly with the stated objectives; do not test content outside scope.
- Use clear, unambiguous language appropriate for the target student level.
- Vary question types and cognitive levels across the set.
- Include a balanced difficulty distribution unless a specific level is requested.
""".strip()

_CONSTRAINTS = """
## Constraints
- Do NOT include culturally biased, offensive, or exclusionary content.
- Do NOT create trick questions designed to confuse rather than assess.
- Do NOT generate content inappropriate for the specified age group.
- If the input is too vague, ask for clarification rather than guessing.
""".strip()

_DISCLAIMER = (
"All generated questions are drafts for teacher review. "
"Teachers should verify accuracy, adjust difficulty for their specific class, "
"and ensure alignment with their curriculum standards before use."
)

_INPUT_GATE = """
## First: Check Input Quality
Before generating ANY questions, evaluate the input:
- Does it contain specific learning objectives or identifiable topic content?
- Is there enough detail to create meaningful assessment questions?

If the input lacks specific objectives (e.g. "teach about science" or "make a quiz"),
DO NOT generate questions. Instead respond with:
"I'd like to help you create a great assessment! Could you provide more detail about:
1. The specific topic or chapter being covered
2. Key concepts students should have learned
3. The grade level or course name"

Only proceed if the input is sufficiently specific.
""".strip()

_COT_STEPS = """
## Your Process (Follow These Steps)
Before generating questions, complete these reasoning steps and show your work:

**Step 1 — Content Analysis:**
Identify the 3–5 key concepts or learning outcomes in the provided content.
List them explicitly.

**Step 2 — Assessment Planning:**
For each key concept decide:
- What Bloom's Taxonomy cognitive level is most appropriate?
- What question type (MCQ / Short Answer / Open-Ended) best fits?
- What common misconceptions do students have about this concept?

**Step 3 — Question Generation:**
Generate questions based on your analysis. Each question must map back to a
specific key concept identified in Step 1.
""".strip()

_CITATION_RULE_GROUNDED = (
    "When a question or answer draws on a retrieved source, cite it inline "
    "using [Source N] notation immediately after the relevant sentence. "
    "Example: 'Photosynthesis occurs in the chloroplast [Source 1].'"
)

_CITATION_RULE_FALLBACK = (
    "This quiz is generated from general knowledge because no relevant source "
    "material was found in the document corpus. Do not fabricate source citations."
)


# ---------------------------------------------------------------------------
# Public prompt builders
# ---------------------------------------------------------------------------

def build_grounded_quiz_prompt(
    content: str,
    chunks: list[dict],
    subject: str | None,
    grade_level: str | None,
    num_questions: int,
    difficulty: str,
    question_types: str,
) -> str:
    """Build a quiz generation prompt grounded in retrieved document chunks.

    Retrieved chunks are injected as numbered [Source N] blocks.  The model is
    instructed to cite sources inline and base questions on the provided material.
    """
    grade_str = grade_level or "Unspecified"
    subject_str = subject.title() if subject else "Unspecified"
    context_block = format_retrieved_chunks(chunks)

    return f"""You are an experienced educational assessment designer.

{_INPUT_GATE}

{_COT_STEPS}

{_OUTPUT_FORMAT}

{_QUALITY_GUIDELINES}

{_CONSTRAINTS}

## Citation Rule
{_CITATION_RULE_GROUNDED}

Also add a **Mapped concept** field to each question (which key concept from Step 1 this assesses).

## Retrieved Source Material
The following passages were retrieved from the document corpus. Base your questions
on this material and cite sources inline as instructed.

{context_block}

## Important Note
{_DISCLAIMER}

---

Please generate {num_questions} assessment questions based on the following:

**Lesson Content / Objectives:**
{content}

**Parameters:**
- Number of questions: {num_questions}
- Difficulty: {difficulty}
- Question types to include: {question_types}
- Target student level: {grade_str}
- Subject: {subject_str}

Please complete Steps 1–3 above, then generate the questions now."""


def build_fallback_quiz_prompt(
    content: str,
    subject: str | None,
    grade_level: str | None,
    num_questions: int,
    difficulty: str,
    question_types: str,
) -> str:
    """Build a fallback quiz prompt when no relevant corpus content was retrieved.

    Produces the same output format as the grounded prompt but notes that questions
    are drawn from general model knowledge rather than document sources.
    """
    grade_str = grade_level or "Unspecified"
    subject_str = subject.title() if subject else "Unspecified"

    return f"""You are an experienced educational assessment designer.

{_INPUT_GATE}

{_COT_STEPS}

{_OUTPUT_FORMAT}

{_QUALITY_GUIDELINES}

{_CONSTRAINTS}

## Citation Rule
{_CITATION_RULE_FALLBACK}

Also add a **Mapped concept** field to each question (which key concept from Step 1 this assesses).

## Important Note
{_DISCLAIMER}

**Note — Fallback Mode:** No relevant source material was found in the document
corpus for this topic. Questions are generated from general model knowledge.
Be explicit that this is fallback output and do not fabricate textbook citations.

---

Please generate {num_questions} assessment questions based on the following:

**Lesson Content / Objectives:**
{content}

**Parameters:**
- Number of questions: {num_questions}
- Difficulty: {difficulty}
- Question types to include: {question_types}
- Target student level: {grade_str}
- Subject: {subject_str}

Please complete Steps 1–3 above, then generate the questions now."""
