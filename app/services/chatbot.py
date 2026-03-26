import re

from app.services.agent import LessonPlanningAgent
from app.services.ollama_client import generate_text
from app.services.prompt_builder import (
    build_block_identification_prompt,
    build_block_prompt,
    build_block_revision_prompt,
    build_references_section,
    extract_citations,
)
from app.services.retriever import retrieve_chunks


FOLLOW_UP_PROMPT = (
    "If you want any changes, tell me what to modify in the script and I will revise it."
)

# Regex to split script into blocks: [Minute X-Y]
_BLOCK_RE = re.compile(r"(\[Minute\s+\d+\s*-\s*\d+\])")
# Extract start/end from a block label
_BLOCK_RANGE_RE = re.compile(r"\[Minute\s+(\d+)\s*-\s*(\d+)\]")


def _parse_script_blocks(script: str) -> list[dict]:
    """Split a full script into header, body blocks, and footer.

    Returns a list of dicts:
      {"type": "header", "text": ...}
      {"type": "block", "label": "[Minute 0-5]", "start": 0, "end": 5, "text": ...}
      {"type": "footer", "text": ...}
    """
    # Try the old format first: "Full Classroom Script:"
    script_marker = "Full Classroom Script:"
    marker_idx = script.find(script_marker)

    if marker_idx == -1:
        # New clean format — no "Full Classroom Script:" marker.
        # Check for [Minute X-Y] labels anywhere in the script.
        if not _BLOCK_RE.search(script):
            # No block labels at all — return as header-only (triggers full revision)
            return [{"type": "header", "text": script}]
        # Has block labels but no "Full Classroom Script:" — find first block label
        first_block = _BLOCK_RE.search(script)
        header = script[: first_block.start()].rstrip()
        body = script[first_block.start() :]
    else:
        header = script[: marker_idx + len(script_marker)]
        body = script[marker_idx + len(script_marker) :]

    # Find the footer sections (Exit Ticket, Homework, Teacher Notes, Sources Used)
    footer_markers = ["Exit Ticket:", "Homework:", "Teacher Notes:", "Sources Used:"]
    footer_start = len(body)
    for fm in footer_markers:
        idx = body.find(fm)
        if idx != -1 and idx < footer_start:
            footer_start = idx

    script_body = body[:footer_start]
    footer = body[footer_start:]

    # Split body into blocks
    parts = _BLOCK_RE.split(script_body)
    blocks = [{"type": "header", "text": header}]

    i = 0
    # Skip any text before the first block label
    if parts and not _BLOCK_RE.match(parts[0]):
        # Prepend to header if there's leading text
        if parts[0].strip():
            blocks[0]["text"] += "\n" + parts[0]
        i = 1

    while i < len(parts):
        label = parts[i]
        text = parts[i + 1] if i + 1 < len(parts) else ""
        m = _BLOCK_RANGE_RE.match(label)
        if m:
            blocks.append({
                "type": "block",
                "label": label,
                "start": int(m.group(1)),
                "end": int(m.group(2)),
                "text": label + text,
            })
        i += 2

    if footer.strip():
        blocks.append({"type": "footer", "text": footer})

    return blocks


def _find_missing_blocks(
    block_parts: list[dict], total_minutes: int
) -> list[tuple[int, int]]:
    """Find gaps in the block timeline.

    For example, if blocks are [0-5] and [15-20] with total=20,
    returns [(5, 15)] as the missing range.
    """
    sorted_blocks = sorted(block_parts, key=lambda b: b["start"])
    gaps: list[tuple[int, int]] = []

    # Check gap before first block
    if sorted_blocks and sorted_blocks[0]["start"] > 0:
        gaps.append((0, sorted_blocks[0]["start"]))

    # Check gaps between blocks
    for i in range(len(sorted_blocks) - 1):
        current_end = sorted_blocks[i]["end"]
        next_start = sorted_blocks[i + 1]["start"]
        if next_start > current_end:
            gaps.append((current_end, next_start))

    # Check gap after last block
    if sorted_blocks and sorted_blocks[-1]["end"] < total_minutes:
        gaps.append((sorted_blocks[-1]["end"], total_minutes))

    return gaps


def _reassemble_script(parts: list[dict]) -> str:
    """Reassemble parsed script parts back into a full script string."""
    sections = []
    for part in parts:
        if part["type"] == "header":
            sections.append(part["text"])
        elif part["type"] == "block":
            sections.append(part["text"])
        elif part["type"] == "footer":
            sections.append(part["text"])
    return "\n\n".join(sections)


class LessonScriptChatbot:
    """Conversational interface for generating and revising lesson scripts."""

    def __init__(self, agent: LessonPlanningAgent | None = None):
        """Initialize with an optional pre-configured agent."""
        self.agent = agent or LessonPlanningAgent()

    def generate_script(
        self,
        user_prompt: str,
        subject: str | None = None,
        grade_level: str | None = None,
        curriculum: str | None = None,
        topic: str | None = None,
        retrieval_limit: int = 5,
        retrieval_mode: str = "auto",
    ) -> dict:
        """Generate a new lesson script via the agent pipeline."""
        result = self.agent.run(
            user_prompt=user_prompt,
            subject=subject,
            grade_level=grade_level,
            curriculum=curriculum,
            topic=topic,
            retrieval_limit=retrieval_limit,
            retrieval_mode=retrieval_mode,
        )

        if result["generation_mode"] in {"refuse", "input_error"}:
            assistant_message = result["lesson_text"]
            follow_up_prompt = None
            chat_mode = result["generation_mode"]
        else:
            assistant_message = (
                "I generated a lesson script based on your request. " + FOLLOW_UP_PROMPT
            )
            follow_up_prompt = FOLLOW_UP_PROMPT
            chat_mode = "generated"

        return {
            "chat_mode": chat_mode,
            "assistant_message": assistant_message,
            "follow_up_prompt": follow_up_prompt,
            **result,
        }

    def revise_script(
        self,
        modification_request: str,
        current_script: str,
        original_request: str,
        subject: str | None = None,
        grade_level: str | None = None,
        curriculum: str | None = None,
        topic: str | None = None,
        retrieval_limit: int = 5,
        retrieval_mode: str = "auto",
    ) -> dict:
        """Revise an existing script by regenerating only the affected blocks."""
        trace: list[dict] = [
            {"tool": "chat_revision_request", "modification_request": modification_request},
        ]

        # Resolve retrieval params
        resolved_subject, resolved_grade, resolved_topic, resolved_mode = (
            self.agent._resolve_retrieval_params(
                user_prompt=original_request,
                subject=subject,
                grade_level=grade_level,
                topic=topic,
                retrieval_mode=retrieval_mode,
            )
        )

        if resolved_mode == "skip":
            retrieved_chunks = []
        else:
            retrieval_query = f"{original_request}\nRequested revision: {modification_request}"
            retrieved_chunks = retrieve_chunks(
                query=retrieval_query,
                limit=retrieval_limit,
                subject=resolved_subject,
                grade_level=resolved_grade,
                curriculum=curriculum,
                topic=resolved_topic,
                retrieval_mode=resolved_mode,
            )
        trace.append({"tool": "retriever", "chunks": retrieved_chunks})

        generation_mode = "grounded" if retrieved_chunks else "fallback"

        # Parse the current script into blocks
        parsed = _parse_script_blocks(current_script)
        block_parts = [p for p in parsed if p["type"] == "block"]

        # Extract duration from original request for gap detection
        constraints = self.agent._extract_constraints(original_request)
        total_minutes = constraints["duration_minutes"] or 40

        if not block_parts:
            # No parseable blocks — fall back to full regeneration
            revised_script = self._full_revision_fallback(
                modification_request=modification_request,
                current_script=current_script,
                original_request=original_request,
                retrieved_chunks=retrieved_chunks,
                generation_mode=generation_mode,
                subject=resolved_subject,
                grade_level=resolved_grade,
            )
            trace.append({"tool": "full_revision_fallback"})
        else:
            # Detect and fill missing blocks (gaps in the timeline)
            missing_blocks = _find_missing_blocks(block_parts, total_minutes)
            if missing_blocks:
                # Build a minimal outline from existing blocks for context
                outline_summary = "\n".join(
                    f"{p['label']} {p.get('text', '')[:100]}..." for p in block_parts
                )
                for gap_start, gap_end in missing_blocks:
                    gap_label = f"[Minute {gap_start}-{gap_end}]"
                    gap_minutes = gap_end - gap_start

                    # Find blocks before and after this gap for continuity
                    prev_text = ""
                    next_text = ""
                    for p in block_parts:
                        if p["end"] <= gap_start:
                            prev_text = p["text"]
                        if p["start"] >= gap_end and not next_text:
                            next_text = p["text"]

                    context_parts = []
                    if prev_text:
                        context_parts.append(f"Previous block (for continuity):\n{prev_text[-500:]}")
                    if next_text:
                        context_parts.append(f"Next block (to transition into):\n{next_text[:500]}")
                    surrounding_context = "\n\n".join(context_parts)

                    gap_prompt = build_block_prompt(
                        block_label=gap_label,
                        block_description="Continued instruction (filling missing section)",
                        block_minutes=gap_minutes,
                        outline=outline_summary,
                        previous_blocks_text=surrounding_context,
                        retrieved_chunks=retrieved_chunks,
                        generation_mode=generation_mode,
                    )
                    gap_text = generate_text(gap_prompt)
                    if gap_label not in gap_text:
                        gap_text = f"{gap_label}\n{gap_text}"

                    new_block = {
                        "type": "block",
                        "label": gap_label,
                        "start": gap_start,
                        "end": gap_end,
                        "text": gap_text.strip(),
                    }

                    # Insert in the right position
                    insert_idx = next(
                        (i for i, p in enumerate(parsed) if p["type"] == "block" and p["start"] >= gap_end),
                        len(parsed) - (1 if parsed and parsed[-1]["type"] == "footer" else 0),
                    )
                    parsed.insert(insert_idx, new_block)
                    block_parts.append(new_block)

                    trace.append({
                        "tool": "block_gap_filler",
                        "block": gap_label,
                        "words": len(gap_text.split()),
                    })

            # Step 1: Identify which blocks need revision
            block_labels = [p["label"] for p in parsed if p["type"] == "block"]
            id_prompt = build_block_identification_prompt(modification_request, block_labels)
            id_response = generate_text(id_prompt, num_predict=128).strip()
            trace.append({"tool": "block_identifier", "response": id_response})

            # Parse which blocks to revise
            if "ALL" in id_response.upper():
                target_labels = set(block_labels)
            else:
                target_labels = set()
                for label in block_labels:
                    if label in id_response:
                        target_labels.add(label)
                # If LLM didn't return any valid labels but there were no
                # missing blocks filled, revise all. If we just filled gaps,
                # the gap filling itself is the revision — skip further edits.
                if not target_labels and not missing_blocks:
                    target_labels = set(block_labels)

            # Step 2: Revise only the targeted blocks
            for part in parsed:
                if part["type"] != "block" or part["label"] not in target_labels:
                    continue

                block_minutes = part["end"] - part["start"]
                rev_prompt = build_block_revision_prompt(
                    block_label=part["label"],
                    block_text=part["text"],
                    block_minutes=block_minutes,
                    modification_request=modification_request,
                    retrieved_chunks=retrieved_chunks,
                    generation_mode=generation_mode,
                )
                revised_block = generate_text(rev_prompt)

                # Ensure block label is present
                if part["label"] not in revised_block:
                    revised_block = f"{part['label']}\n{revised_block}"

                # Trim any LLM preamble before the block label
                label_idx = revised_block.find(part["label"])
                if label_idx > 0:
                    revised_block = revised_block[label_idx:]

                part["text"] = revised_block.strip()
                trace.append({
                    "tool": "block_reviser",
                    "block": part["label"],
                    "words": len(revised_block.split()),
                })

            # Reassemble, deduplicate, and normalize
            revised_script = _reassemble_script(parsed)
            revised_script = self.agent._remove_script_repetitions(
                revised_script, trace,
            )
            revised_script = self.agent.normalize_lesson_text(
                lesson_text=revised_script,
                user_prompt=original_request,
                generation_mode=generation_mode,
                subject=resolved_subject,
                grade_level=resolved_grade,
                retrieved_chunks=retrieved_chunks,
            )

        # Append references section if we have retrieved chunks
        references = build_references_section(retrieved_chunks)
        if references:
            revised_script = f"{revised_script}\n\n{references}"

        return {
            "chat_mode": "revised",
            "assistant_message": (
                "I revised the lesson script based on your requested changes. "
                + FOLLOW_UP_PROMPT
            ),
            "follow_up_prompt": FOLLOW_UP_PROMPT,
            "user_prompt": original_request,
            "generation_mode": generation_mode,
            "source_notice": (
                "Revision used retrieved source material."
                if retrieved_chunks
                else "Revision used model fallback."
            ),
            "retrieved_chunks": retrieved_chunks,
            "prompt_used": None,
            "lesson_text": revised_script,
            "evaluation": None,
            "agent_trace": trace,
            "citations": extract_citations(revised_script, retrieved_chunks),
        }

    def _full_revision_fallback(
        self,
        modification_request: str,
        current_script: str,
        original_request: str,
        retrieved_chunks: list[dict],
        generation_mode: str,
        subject: str | None = None,
        grade_level: str | None = None,
    ) -> str:
        """Fallback: revise the whole script in one LLM call when blocks can't be parsed."""
        from app.services.prompt_builder import build_script_revision_prompt

        constraints = self.agent._extract_constraints(original_request)
        resolved_subject = subject or self.agent._infer_subject(original_request)

        prompt = build_script_revision_prompt(
            original_request=original_request,
            current_script=current_script,
            modification_request=modification_request,
            retrieved_chunks=retrieved_chunks,
            subject=resolved_subject,
            grade_level=grade_level,
            duration_minutes=constraints["duration_minutes"],
        )
        revised = generate_text(prompt)
        return self.agent.normalize_lesson_text(
            lesson_text=revised,
            user_prompt=original_request,
            generation_mode=generation_mode,
            subject=resolved_subject,
            grade_level=grade_level,
            retrieved_chunks=retrieved_chunks,
        )

    def chat(
        self,
        message: str,
        current_script: str | None = None,
        original_request: str | None = None,
        subject: str | None = None,
        grade_level: str | None = None,
        curriculum: str | None = None,
        topic: str | None = None,
        retrieval_limit: int = 5,
        retrieval_mode: str = "auto",
    ) -> dict:
        """Route a chat message to generation or revision based on context."""
        if current_script:
            return self.revise_script(
                modification_request=message,
                current_script=current_script,
                original_request=original_request or message,
                subject=subject,
                grade_level=grade_level,
                curriculum=curriculum,
                topic=topic,
                retrieval_limit=retrieval_limit,
                retrieval_mode=retrieval_mode,
            )

        return self.generate_script(
            user_prompt=message,
            subject=subject,
            grade_level=grade_level,
            curriculum=curriculum,
            topic=topic,
            retrieval_limit=retrieval_limit,
            retrieval_mode=retrieval_mode,
        )
