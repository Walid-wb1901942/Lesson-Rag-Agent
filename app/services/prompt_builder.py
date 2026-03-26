WORDS_PER_MINUTE_TARGET = 95


def _outline_block_examples(total_minutes: int) -> str:
    """Generate duration-appropriate block plan examples for the outline prompt."""
    if total_minutes <= 10:
        blocks = [(0, total_minutes, "Full lesson")]
    elif total_minutes <= 20:
        blocks = [
            (0, 5, "Opening and introduction"),
            (5, total_minutes, "Core instruction and closing"),
        ]
    elif total_minutes <= 40:
        mid = total_minutes - 10
        blocks = [
            (0, 5, "Opening"),
            (5, mid, "Core instruction"),
            (mid, total_minutes, "Review and closing"),
        ]
    else:
        mid = total_minutes // 2
        blocks = [
            (0, 5, "Opening"),
            (5, mid, "Core instruction part 1"),
            (mid, total_minutes - 10, "Core instruction part 2"),
            (total_minutes - 10, total_minutes, "Review and closing"),
        ]
    return "\n".join(
        f"[Minute {s}-{e}] <{desc}>" for s, e, desc in blocks
    )


def extract_citations(lesson_text: str, chunks: list[dict]) -> list[dict]:
    """Extract [Source N] references from generated text and map to chunk metadata."""
    if not chunks:
        return []

    import re as _re
    used_sources = set(int(m) for m in _re.findall(r"\[Source\s+(\d+)\]", lesson_text))

    citations = []
    for i, chunk in enumerate(chunks, start=1):
        if i not in used_sources:
            continue
        meta = chunk["metadata"]
        citations.append({
            "source_number": i,
            "title": meta.get("title"),
            "pages": meta.get("page_range") or str(meta.get("page_number") or ""),
            "subject": meta.get("subject"),
            "grade_level": format_metadata_value(meta.get("grade_level")),
            "topic": meta.get("topic"),
        })

    return citations


SPOKEN_MATH_RULE = (
    "- CRITICAL: This script will be read aloud and converted to audio. "
    "ALL mathematical notation MUST be written as spoken words, not symbols. "
    'Write "the integral from negative one to one of three x squared minus five, dx" '
    'instead of "integral (3x^2 - 5)dx". '
    'Write "x squared" not "x2" or "x^2". '
    'Write "x cubed" not "x3" or "x^3". '
    'Write "one half" not "1/2". '
    'Write "is not equal to" not "!=" or "=/=". '
    'Write "the fraction a over b" not "a/b" when it is a complex fraction. '
    "Never use raw LaTeX, integral signs, summation signs, or symbolic math notation in Teacher: lines. "
    "The teacher must be able to read every line naturally without interpreting symbols."
)


def format_time_block_template(duration_minutes: int | None) -> str:
    """Generate example [Minute X-Y] block templates with word/line targets."""
    total_minutes = duration_minutes or 40

    if total_minutes <= 20:
        example_blocks = [(0, 5), (5, total_minutes)]
    elif total_minutes <= 40:
        example_blocks = [(0, 5), (5, 15), (15, total_minutes - 10), (total_minutes - 10, total_minutes)]
    else:
        midpoint = total_minutes // 2
        example_blocks = [(0, 5), (5, 15), (15, midpoint), (midpoint, total_minutes - 10), (total_minutes - 10, total_minutes)]

    parts = []
    for start, end in example_blocks:
        if start >= end:
            continue
        block_minutes = end - start
        word_target = block_minutes * WORDS_PER_MINUTE_TARGET
        teacher_lines = max(3, int(block_minutes * 0.6))
        parts.append(
            f"""[Minute {start}-{end}]  (approx. {word_target} words, at least {teacher_lines} Teacher: lines)
Teacher: "..."
Teacher: "..."
Teacher: "..."
Teacher: "..."
[Pause 20 seconds]
Teacher: "..."
"""
        )

    parts.append("[Continue with additional minute blocks as needed so the full script covers the requested duration exactly with no gaps or overlaps]")
    return "\n".join(parts)


def format_density_guidance(duration_minutes: int | None) -> str:
    """Generate word-count density requirements for the given lesson duration."""
    total_minutes = duration_minutes or 40
    total_words = total_minutes * WORDS_PER_MINUTE_TARGET
    return (
        f"CRITICAL DENSITY REQUIREMENT: The full script must contain approximately {total_words} words of content "
        f"({WORDS_PER_MINUTE_TARGET} words per minute x {total_minutes} minutes). "
        f"Each minute block must contain roughly {WORDS_PER_MINUTE_TARGET} words per minute of block duration. "
        f"A 5-minute block needs ~{5 * WORDS_PER_MINUTE_TARGET} words. "
        f"A 10-minute block needs ~{10 * WORDS_PER_MINUTE_TARGET} words. "
        f"A 20-minute block needs ~{20 * WORDS_PER_MINUTE_TARGET} words. "
        "Every Teacher: line should be a full sentence or multi-sentence paragraph in quotation marks, not a short phrase. "
        "Include detailed explanations, worked examples with step-by-step narration, questions to the class with pause cues, "
        "and transition statements between topics."
    )


def format_grade_line(grade_level: str | None) -> str:
    """Format grade level for display, defaulting to 'Unspecified'."""
    return grade_level or "Unspecified"


def format_subject_line(subject: str | None) -> str:
    """Format subject for display, defaulting to 'Unspecified'."""
    return subject or "Unspecified"


def format_metadata_value(value: str | list[str] | None) -> str:
    """Convert a metadata value (string, list, or None) to display string."""
    if isinstance(value, list):
        return ", ".join(value)
    return value or ""


def format_retrieved_chunks(chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered [Source N] blocks for LLM prompts."""
    if not chunks:
        return "No retrieved source material was available."

    formatted = []

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        formatted.append(
            f"""[Source {i}]
Title: {meta.get("title")}
Pages: {meta.get("page_range") or meta.get("page_number") or "unknown"}
Subject: {meta.get("subject")}
Grade: {format_metadata_value(meta.get("grade_level"))}
Topic: {meta.get("topic")}

Content:
{chunk["text"]}
"""
        )

    return "\n\n".join(formatted)


def build_references_section(chunks: list[dict]) -> str:
    """Build a References section from retrieved chunks for appending to lesson scripts.

    Groups chunks from the same document title and shows page ranges.
    """
    if not chunks:
        return ""

    # Group chunks by title, collecting source numbers and page info
    from collections import OrderedDict
    groups: OrderedDict[str, dict] = OrderedDict()

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        title = meta.get("title") or "Untitled"
        pages = meta.get("page_range") or (str(meta["page_number"]) if meta.get("page_number") else None)

        if title not in groups:
            groups[title] = {
                "source_numbers": [],
                "pages": set(),
                "subject": meta.get("subject") or "",
                "grade": format_metadata_value(meta.get("grade_level")),
                "topic": meta.get("topic") or "",
            }
        groups[title]["source_numbers"].append(i)
        if pages:
            groups[title]["pages"].add(pages)

    lines = ["References:"]
    for title, info in groups.items():
        source_labels = ", ".join(f"[Source {n}]" for n in info["source_numbers"])
        parts = [f"{source_labels} {title}"]

        if info["pages"]:
            sorted_pages = sorted(info["pages"])
            parts.append(f"pp. {', '.join(sorted_pages)}")

        detail_parts = []
        if info["subject"]:
            detail_parts.append(info["subject"])
        if info["grade"]:
            detail_parts.append(f"Grade {info['grade']}")
        if info["topic"]:
            detail_parts.append(info["topic"])
        if detail_parts:
            parts.append(f"({', '.join(detail_parts)})")

        lines.append(f"- {' \u2014 '.join(parts)}")

    return "\n".join(lines)


def build_grounded_lesson_prompt(
    user_prompt: str,
    retrieved_chunks: list[dict],
    subject: str | None = None,
    grade_level: str | None = None,
    duration_minutes: int | None = None,
) -> str:
    context = format_retrieved_chunks(retrieved_chunks)
    block_template = format_time_block_template(duration_minutes)
    duration_line = f"{duration_minutes or 40} minutes"
    density_guidance = format_density_guidance(duration_minutes)

    return f"""
You are an expert classroom script writer.

Your task is to create a literal, word-for-word classroom script that a teacher can read aloud from start to finish.

{density_guidance}

Important rules:
- Use the retrieved material as your main source of truth.
- Do not invent curriculum facts that are not supported by the retrieved material.
- Write spoken teacher lines in quotation marks. Each Teacher: line must be a full paragraph of spoken dialogue, not a short phrase.
- Make the script classroom-ready, age-appropriate, and easy to follow live.
- Include brief stage directions in square brackets only when necessary.
- Do not return a high-level lesson plan. Return a read-aloud script.
- Return only the final classroom script. Do not add intro text such as "Here is the revised draft".
- Use minute blocks labeled like `[Minute 0-5]`, `[Minute 5-15]`, or `[Minute 15-35]`.
- You may choose variable-length blocks such as 5, 10, 20, or other sensible section lengths.
- The minute blocks must cover the full requested duration exactly, with no gaps, overlaps, or extra time beyond the requested total.
- Each block must contain multiple separate `Teacher:` lines and pause cues such as `[Pause 20 seconds]` or `[Think time 30 seconds]` where useful so the pacing feels realistic.
- The script should be almost entirely teacher wording. Do not add `Students:` sections, `Student Lines:`, or bullet-point mock dialogue.
- Do not invent source titles. If no relevant retrieved material exists, say so plainly in `Sources Used`.
- If the request does not explicitly provide a grade level, write exactly `Grade Level: Unspecified`.
- The lesson title must not be blank.
{CITATION_RULE_GROUNDED}
{SPOKEN_MATH_RULE}

User request:
{user_prompt}

Retrieved source material:
{context}

Return the result in this exact format:

Lesson Title:
Grade Level: {format_grade_line(grade_level)}
Subject: {format_subject_line(subject)}
Duration: {duration_line}

Materials Needed:
- ...
- ...

Teaching Goals:
- ...
- ...
- ...

Full Classroom Script:
{block_template}

Exit Ticket:
Teacher: "..."

Homework:
Teacher: "..."

Teacher Notes:
- ...
- ...

Sources Used:
- ...
"""


def build_fallback_lesson_prompt(
    user_prompt: str,
    subject: str | None = None,
    grade_level: str | None = None,
    duration_minutes: int | None = None,
) -> str:
    block_template = format_time_block_template(duration_minutes)
    duration_line = f"{duration_minutes or 40} minutes"
    density_guidance = format_density_guidance(duration_minutes)
    return f"""
You are an expert classroom script writer.

The user's request is education-related, but there is not enough relevant retrieved source material to ground the answer in the project's document collection.

{density_guidance}

Important rules:
- Create a literal, word-for-word teacher script using general teaching knowledge and standard classroom practice.
- Do not claim that retrieved documents supported the answer.
- Be explicit that this is a fallback classroom script generated without relevant source-backed retrieval.
- Write spoken teacher lines in quotation marks. Each Teacher: line must be a full paragraph of spoken dialogue, not a short phrase.
- Include brief stage directions in square brackets only when necessary.
- Do not return a high-level lesson plan. Return a read-aloud script.
- Return only the final classroom script. Do not add intro text such as "Here is the revised draft".
- Use minute blocks labeled like `[Minute 0-5]`, `[Minute 5-15]`, or `[Minute 15-35]`.
- You may choose variable-length blocks such as 5, 10, 20, or other sensible section lengths.
- The minute blocks must cover the full requested duration exactly, with no gaps, overlaps, or extra time beyond the requested total.
- Each block must contain multiple separate `Teacher:` lines and pause cues such as `[Pause 20 seconds]` or `[Think time 30 seconds]` where useful so the pacing feels realistic.
- The script should be almost entirely teacher wording. Do not add `Students:` sections, `Student Lines:`, or bullet-point mock dialogue.
- In fallback mode, do not invent textbooks, handouts, or other source titles in `Sources Used`.
- If the request does not explicitly provide a grade level, write exactly `Grade Level: Unspecified`.
- The lesson title must not be blank.
{CITATION_RULE_FALLBACK}
{SPOKEN_MATH_RULE}

User request:
{user_prompt}

Return the result in this exact format:

Lesson Title:
Grade Level: {format_grade_line(grade_level)}
Subject: {format_subject_line(subject)}
Duration: {duration_line}

Materials Needed:
- ...
- ...

Teaching Goals:
- ...
- ...
- ...

Full Classroom Script:
{block_template}

Exit Ticket:
Teacher: "..."

Homework:
Teacher: "..."

Teacher Notes:
- ...
- ...

Sources Used:
- No relevant retrieved source material was available. This classroom script was generated as fallback output.
"""


def build_script_revision_prompt(
    original_request: str,
    current_script: str,
    modification_request: str,
    retrieved_chunks: list[dict],
    subject: str | None = None,
    grade_level: str | None = None,
    duration_minutes: int | None = None,
) -> str:
    context = format_retrieved_chunks(retrieved_chunks)
    duration_line = f"{duration_minutes or 40} minutes"
    density_guidance = format_density_guidance(duration_minutes)

    return f"""
You are an expert classroom script writer.

Your task is to revise an existing classroom script based on the user's requested modifications.

{density_guidance}

Important rules:
- Preserve useful parts of the current script unless the user asked to change them.
- Apply the requested modifications clearly and directly.
- If retrieved source material is available, keep the revision aligned with it.
- Return a complete revised read-aloud classroom script, not notes about changes.
- Every line of spoken text must begin with "Teacher:" followed by ONLY the exact words the teacher says out loud.
- Do NOT include stage directions, body language, or instructions to the teacher.
- Do NOT use bullet lists or numbered lists. Write everything as full spoken paragraphs (4-6 sentences minimum per Teacher: paragraph).
- Include pause cues like [Pause 20 seconds] or [Think time 30 seconds] between Teacher: paragraphs.
- Do NOT add Students: lines or mock student dialogue.
- Return only the final classroom script. Do not add intro text such as "Here is the revised draft".
- Do NOT repeat yourself. Every paragraph must introduce NEW content.
- Do NOT wrap up, conclude, or summarise at the end of the script unless the user explicitly asked for a closing.
- Do NOT say "let's revisit" or reference previous classes. This is a single continuous lesson.
{CITATION_RULE_GROUNDED if retrieved_chunks else CITATION_RULE_FALLBACK}
{SPOKEN_MATH_RULE}

Original request:
{original_request}

User modification request:
{modification_request}

Retrieved source material:
{context}

Current classroom script:
{current_script}

Return the revised script in this exact format and NOTHING else:

Lesson: <title>
Grade: {format_grade_line(grade_level)} | Subject: {format_subject_line(subject)} | Duration: {duration_line}
Source: <"Retrieved from documents" or "Generated (no matching documents found)">

Teacher: "..."
[Pause 20 seconds]
Teacher: "..."
[Think time 30 seconds]
Teacher: "..."

Write ONLY the header and Teacher: lines with pause cues. Do NOT include Materials Needed, Teaching Goals, Exit Ticket, Homework, Teacher Notes, or Sources Used sections.
- ...
"""


def build_block_identification_prompt(
    modification_request: str,
    block_labels: list[str],
) -> str:
    labels_list = "\n".join(f"- {label}" for label in block_labels)
    return f"""You are a classifier. A user wants to modify a classroom lesson script.

The script has these time blocks:
{labels_list}

User's modification request:
{modification_request}

Which blocks need to change? Reply with ONLY the block labels that must be rewritten, one per line.
If ALL blocks are affected (e.g. "change the tone everywhere"), reply with: ALL
If only specific blocks are affected (e.g. "make the introduction more engaging"), reply with only those block labels.
Reply with nothing else — no explanation, no numbering."""


def build_block_revision_prompt(
    block_label: str,
    block_text: str,
    block_minutes: int,
    modification_request: str,
    retrieved_chunks: list[dict],
    generation_mode: str,
) -> str:
    word_target = block_minutes * WORDS_PER_MINUTE_TARGET
    min_words = int(word_target * 0.8)
    context = ""
    citation_rule = CITATION_RULE_FALLBACK
    if generation_mode == "grounded" and retrieved_chunks:
        context = f"\nRetrieved source material:\n{format_retrieved_chunks(retrieved_chunks)}\n"
        citation_rule = CITATION_RULE_GROUNDED

    return f"""You are an expert classroom script writer.

Revise ONLY the following time block of a classroom script based on the user's modification request.
{context}
Time block: {block_label}
Duration: {block_minutes} minutes

CRITICAL LENGTH REQUIREMENT:
- This block represents {block_minutes} minutes of spoken classroom time at ~95 words per minute.
- You MUST write at least {min_words} words (minimum) and aim for {word_target} words.
- A teacher speaks roughly one full paragraph per minute. For {block_minutes} minutes you need {block_minutes} or more full Teacher: paragraphs.
- If your output is shorter than {min_words} words, it is TOO SHORT and will not fill the time.

Current block content:
{block_text}

User's modification request:
{modification_request}

IMPORTANT — This script will be read aloud by a narrator. Write ONLY the words the teacher speaks out loud.

Rules:
- Apply the requested modification to this block.
- Start with the block header: {block_label}
- Every line must begin with "Teacher:" followed by ONLY the exact words the teacher says out loud.
- Do NOT include stage directions, body language, or instructions to the teacher.
- Do NOT use bullet lists. Write everything as full spoken paragraphs (4-6 sentences minimum).
- Include pause cues like [Pause 20 seconds] or [Think time 30 seconds] between Teacher: paragraphs.
- Do NOT add Students: lines or mock student dialogue.
- Do NOT add intro text like "Here is the revised block".
- Do NOT include Exit Ticket, Homework, Teacher Notes, or Sources Used sections.
- Do NOT wrap up, conclude, or summarise at the end of this block. This block flows directly into the next one.
- Do NOT repeat yourself. Every paragraph must introduce NEW content.
{citation_rule}
- Write ONLY this one revised block, nothing else.
{SPOKEN_MATH_RULE}
"""


def build_lesson_prompt(user_prompt: str, retrieved_chunks: list[dict]) -> str:
    return build_grounded_lesson_prompt(user_prompt, retrieved_chunks)


# ---------------------------------------------------------------------------
# Phased generation prompts (outline then block-by-block)
# ---------------------------------------------------------------------------

def build_outline_prompt(
    user_prompt: str,
    retrieved_chunks: list[dict],
    generation_mode: str,
    subject: str | None = None,
    grade_level: str | None = None,
    duration_minutes: int | None = None,
) -> str:
    total = duration_minutes or 40
    context = format_retrieved_chunks(retrieved_chunks) if generation_mode == "grounded" else ""
    context_section = f"\nRetrieved source material:\n{context}\n" if context else ""
    source_rule = (
        "- Use the retrieved material to decide what content belongs in each block."
        if generation_mode == "grounded"
        else "- Use general teaching knowledge to plan the content."
    )

    return f"""You are an expert classroom script planner.

Create a structured lesson outline for the following request.
Return ONLY the outline in the exact format shown below. Do not write the full script yet.

{source_rule}
- Choose sensible minute blocks that cover exactly {total} minutes with no gaps or overlaps.
- Each block description should be 1-2 sentences explaining what the teacher will cover.

User request:
{user_prompt}
{context_section}
Return this exact format and nothing else:

Lesson Title: <title>
Grade Level: {format_grade_line(grade_level)}
Subject: {format_subject_line(subject)}
Duration: {total} minutes

Materials Needed:
- ...

Teaching Goals:
- ...
- ...
- ...

Block Plan:
{_outline_block_examples(total)}

Exit Ticket: <1-sentence description>
Homework: <1-sentence description>
"""


CITATION_RULE_GROUNDED = (
    "- CITATIONS: When you use facts, definitions, examples, or explanations from the retrieved source material, "
    "cite inline using [Source 1], [Source 2], etc. matching the source numbers above. "
    "Place the citation at the end of the Teacher: line that uses that information. "
    "Do NOT cite every line — only cite where you directly draw on a specific source."
)

CITATION_RULE_FALLBACK = (
    "- NOTE: No relevant source documents were retrieved for this lesson. "
    "All content is generated from general teaching knowledge. Do not fabricate source references."
)


def build_block_prompt(
    block_label: str,
    block_description: str,
    block_minutes: int,
    outline: str,
    previous_blocks_text: str,
    retrieved_chunks: list[dict],
    generation_mode: str,
) -> str:
    word_target = block_minutes * WORDS_PER_MINUTE_TARGET
    context = ""
    citation_rule = CITATION_RULE_FALLBACK
    if generation_mode == "grounded" and retrieved_chunks:
        context = f"\nRetrieved source material:\n{format_retrieved_chunks(retrieved_chunks)}\n"
        citation_rule = CITATION_RULE_GROUNDED

    previous_section = ""
    if previous_blocks_text:
        previous_section = f"\nPrevious blocks already written (for continuity):\n{previous_blocks_text}\n"

    min_words = int(word_target * 0.8)

    return f"""You are an expert classroom script writer.

Write ONLY the script content for the time block below. This is one section of a larger lesson.

Lesson outline:
{outline}
{context}{previous_section}
Time block to write: {block_label}
Block plan: {block_description}
Duration: {block_minutes} minutes

CRITICAL LENGTH REQUIREMENT:
- This block represents {block_minutes} minutes of spoken classroom time at ~95 words per minute.
- You MUST write at least {min_words} words (minimum) and aim for {word_target} words.
- A teacher speaks roughly one full paragraph per minute. For {block_minutes} minutes you need {block_minutes} or more full Teacher: paragraphs.
- If your output is shorter than {min_words} words, it is TOO SHORT and will not fill the time.

IMPORTANT — This script will be read aloud by a narrator. Write ONLY the words the teacher speaks out loud to students. Nothing else.

Rules:
- Start with the block header: {block_label}
- Every line must begin with "Teacher:" followed by ONLY the exact words the teacher says out loud.
- Do NOT include stage directions, body language, or instructions to the teacher (like "stand at the front", "write on the board", "walk around the room", "pause to gather thoughts"). Those are NOT spoken words.
- Do NOT use bullet lists or numbered lists. Write everything as full spoken paragraphs (4-6 sentences minimum per Teacher: paragraph).
- Include pause cues like [Pause 20 seconds] or [Think time 30 seconds] on their own line between Teacher: paragraphs.
- Include detailed explanations, worked examples, questions to the class, and smooth transitions.
- Do NOT add Students: lines or mock student dialogue.
- Do NOT add intro text like "Here is the script".
- Do NOT include Exit Ticket, Homework, Teacher Notes, or Sources Used sections.
- Do NOT wrap up, conclude, or summarise at the end of this block. This block flows directly into the next one as part of one continuous lesson. Do not say things like "that concludes", "let's wrap up", "we'll continue next class", or "see you then".
- Do NOT say "let's revisit" or reference previous classes. This is a single continuous lesson.
- Do NOT repeat yourself. Every paragraph must introduce NEW content, examples, or activities. Never restate the same explanation twice.
{citation_rule}
- Write ONLY this one block, nothing else.
{SPOKEN_MATH_RULE}
"""


def build_closing_sections_prompt(
    outline: str,
    generation_mode: str,
    retrieved_chunks: list[dict],
) -> str:
    source_line = (
        "- List the actual source titles from the retrieved material."
        if generation_mode == "grounded" and retrieved_chunks
        else "- Write: No relevant retrieved source material was available. This classroom script was generated as fallback output."
    )

    return f"""You are an expert classroom script writer.

Write ONLY the closing sections for this lesson script. The main classroom script blocks are already written.

Lesson outline:
{outline}

Write these sections and nothing else:

Exit Ticket:
Teacher: "<full paragraph of what the teacher says for the exit ticket>"

Homework:
Teacher: "<full paragraph of what the teacher assigns>"

Teacher Notes:
- <practical note 1>
- <practical note 2>

Sources Used:
{source_line}

Rules:
- Each Teacher: line must be a full multi-sentence paragraph in quotation marks.
- Return ONLY these four sections, no other text.
"""
