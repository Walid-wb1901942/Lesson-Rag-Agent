import re
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from math import ceil
from dataclasses import dataclass

from app.services.ollama_client import generate_text
from app.services.prompt_builder import (
    build_block_prompt,
    build_fallback_lesson_prompt,
    build_grounded_lesson_prompt,
    build_outline_prompt,
    build_references_section,
    extract_citations,
    format_grade_line,
    format_subject_line,
)
from app.services.qdrant_client import (
    collection_has_points,
    field_value_exists,
    filter_has_matches,
    list_payload_values,
)
from app.services.retriever import retrieve_chunks

EDUCATION_KEYWORDS = (
    "lesson",
    "teach",
    "teacher",
    "student",
    "classroom",
    "school",
    "grade",
    "curriculum",
    "assessment",
    "homework",
    "activity",
    "learning",
    "education",
    "subject",
    "instructor",
)
MIN_RETRIEVAL_SCORE = 0.35
REQUIRED_SCRIPT_SECTIONS = (
    "Lesson:",
    "Teacher:",
)
WORDS_PER_MINUTE = 95
DENSITY_FLOOR = 0.35
DENSITY_CEILING = 1.5
SUBJECT_KEYWORDS = {
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
    "science": ("science",),  # generic fallback for unspecified science
    "literature": ("literature", "english", "poetry", "novel", "grammar"),
    "health": ("health", "wellness", "nutrition", "hygiene"),
}


@dataclass
class PipelineConfig:
    """Configuration for the lesson generation pipeline."""

    retrieval_limit: int = 5
    max_revision_passes: int = 4
    min_retrieval_score: float = MIN_RETRIEVAL_SCORE


class ScriptPipeline:
    """Orchestration pipeline for RAG-based lesson script generation."""

    def __init__(self, config: PipelineConfig | None = None):
        """Initialize the pipeline with optional configuration overrides."""
        self.config = config or PipelineConfig()

    def check_education_domain(self, user_prompt: str) -> dict:
        """Verify the request is education-related using keywords and LLM validation."""
        lower = user_prompt.lower()

        # Fast path: if prompt contains a known academic subject AND at least one
        # education keyword, accept immediately (e.g. "write a math script on algebra").
        # Requires BOTH to prevent subject-word-only bypasses like "trading bot using math".
        all_subject_terms = [
            term for terms in SUBJECT_KEYWORDS.values() for term in terms
        ]
        has_subject = any(term in lower for term in all_subject_terms)
        has_keyword = any(keyword in lower for keyword in EDUCATION_KEYWORDS)

        if has_subject and has_keyword:
            return {"tool": "domain_checker", "is_education_related": True}

        # Fast path: if no education keywords at all, refuse immediately
        if not has_keyword:
            return {"tool": "domain_checker", "is_education_related": False}

        # Ambiguous case: has education keywords but no recognized subject.
        # Use a quick LLM call to verify (e.g. "trading bot lesson").
        validation_prompt = (
            "You are a classifier. Decide whether the following user request "
            "is genuinely asking for a classroom lesson plan or teaching script "
            "for a real educational subject (e.g. math, science, literature, history, health, etc.).\n\n"
            "A request is NOT educational if the core topic is something like trading, "
            "gambling, hacking, weapons, or any non-academic subject — even if the user "
            "uses words like 'lesson' or 'teach'.\n\n"
            f"User request: {user_prompt}\n\n"
            "Reply with ONLY 'yes' or 'no'."
        )
        response = generate_text(validation_prompt, num_predict=8).strip().lower()
        is_education_related = response.startswith("yes")

        return {
            "tool": "domain_checker",
            "is_education_related": is_education_related,
            "llm_validation": response,
        }

    def retrieve_context(
        self,
        user_prompt: str,
        subject: str | None = None,
        grade_level: str | None = None,
        topic: str | None = None,
        retrieval_limit: int | None = None,
        retrieval_mode: str = "filtered",
        retrieval_method: str = "dense",
    ) -> dict:
        """Retrieve relevant document chunks from Qdrant for the given query."""
        limit = retrieval_limit or self.config.retrieval_limit
        chunks = retrieve_chunks(
            query=user_prompt,
            limit=limit,
            subject=subject,
            grade_level=grade_level,
            topic=topic,
            retrieval_mode=retrieval_mode,
            retrieval_method=retrieval_method,
        )
        return {
            "tool": "retriever",
            "retrieval_mode": retrieval_mode,
            "retrieval_method": retrieval_method,
            "chunks": chunks,
            "top_score": chunks[0]["score"] if chunks else None,
        }

    def choose_generation_mode(
        self, retrieved_chunks: list[dict], retrieval_method: str = "dense"
    ) -> str:
        """Select grounded or fallback mode based on retrieval quality.

        For hybrid retrieval, RRF scores are always < 0.05 (1/(60+rank)) so the
        cosine-similarity threshold does not apply — any returned chunks are grounded.
        """
        if not retrieved_chunks:
            return "fallback"
        if retrieval_method == "hybrid":
            return "grounded"
        if retrieved_chunks[0]["score"] >= self.config.min_retrieval_score:
            return "grounded"
        return "fallback"

    def build_generation_prompt(
        self,
        user_prompt: str,
        generation_mode: str,
        retrieved_chunks: list[dict],
        subject: str | None = None,
        grade_level: str | None = None,
        duration_minutes: int | None = None,
    ) -> str:
        """Build a grounded or fallback lesson generation prompt."""
        resolved_subject = subject or self._infer_subject(user_prompt)

        if generation_mode == "grounded":
            return build_grounded_lesson_prompt(
                user_prompt,
                retrieved_chunks,
                subject=resolved_subject,
                grade_level=grade_level,
                duration_minutes=duration_minutes,
            )
        return build_fallback_lesson_prompt(
            user_prompt,
            subject=resolved_subject,
            grade_level=grade_level,
            duration_minutes=duration_minutes,
        )

    def generate_lesson_text(self, prompt: str) -> dict:
        """Generate lesson text from a single prompt via Ollama."""
        return {
            "tool": "generator",
            "lesson_text": generate_text(prompt),
        }

    def generate_phased(
        self,
        user_prompt: str,
        generation_mode: str,
        retrieved_chunks: list[dict],
        subject: str | None = None,
        grade_level: str | None = None,
        duration_minutes: int | None = None,
        trace: list[dict] | None = None,
    ) -> dict:
        """Generate a script in phases: outline, block-by-block, then assemble."""
        if trace is None:
            trace = []

        total = duration_minutes or 40
        resolved_subject = subject or self._infer_subject(user_prompt)

        # Step 1: Generate the outline
        outline_prompt = build_outline_prompt(
            user_prompt=user_prompt,
            retrieved_chunks=retrieved_chunks,
            generation_mode=generation_mode,
            subject=resolved_subject,
            grade_level=grade_level,
            duration_minutes=total,
        )
        outline = generate_text(outline_prompt, num_predict=1024)
        trace.append({"tool": "outline_generator", "outline": outline})

        # Parse blocks from the outline and enforce the requested duration
        blocks_plan = self._parse_and_validate_block_plan(outline, total)

        # Step 2: Build a flat ordered list of all sub-blocks first, then generate
        # them all in parallel. Each block receives the full outline as context
        # (which describes every block's topic), so concurrent generation works
        # without needing the actual text of preceding blocks.
        # The deduplication pass in Step 3 handles any content overlap.
        max_minutes_per_call = 30
        all_sub_blocks: list[tuple[int, int, str]] = []  # (sub_start, sub_end, sub_desc)
        for start, end, description in blocks_plan:
            block_minutes = end - start
            if block_minutes <= max_minutes_per_call:
                all_sub_blocks.append((start, end, description))
            else:
                cursor = start
                part_num = 0
                while cursor < end:
                    sub_end = min(cursor + max_minutes_per_call, end)
                    if sub_end == end or (end - sub_end) < 3:
                        sub_end = end
                    label_hint = (
                        f"{description} (part {part_num + 1})"
                        if sub_end != end else f"{description} (final part)"
                    )
                    all_sub_blocks.append((cursor, sub_end, label_hint))
                    part_num += 1
                    cursor = sub_end

        def _generate_sub_block(args: tuple) -> tuple:
            idx, sub_start, sub_end, sub_desc = args
            sub_label = f"[Minute {sub_start}-{sub_end}]"
            sub_minutes = sub_end - sub_start
            sub_prompt = build_block_prompt(
                block_label=sub_label,
                block_description=sub_desc,
                block_minutes=sub_minutes,
                outline=outline,
                previous_blocks_text="",
                retrieved_chunks=retrieved_chunks,
                generation_mode=generation_mode,
            )
            # Scale token budget to block duration: ~127 tokens/minute at 95 wpm,
            # with a 40% overhead buffer. Floor at 1024 for very short blocks.
            block_num_predict = max(1024, int(sub_minutes * WORDS_PER_MINUTE / 0.75 * 1.4))
            text = generate_text(sub_prompt, num_predict=block_num_predict)
            if sub_label not in text:
                text = f"{sub_label}\n{text}"
            text = self._trim_to_current_block(text, sub_label)
            text = self._remove_repetitions(text)
            text = self._trim_truncated_block(text)
            return idx, sub_label, sub_minutes, text.strip()

        indexed = [(i, s, e, d) for i, (s, e, d) in enumerate(all_sub_blocks)]
        max_workers = min(len(indexed), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            raw_results = list(executor.map(_generate_sub_block, indexed))

        raw_results.sort(key=lambda r: r[0])
        generated_blocks: list[str] = []
        for _, sub_label, sub_minutes, sub_text in raw_results:
            generated_blocks.append(sub_text)
            trace.append({
                "tool": "block_generator",
                "block": sub_label,
                "words": len(sub_text.split()),
                "target_words": sub_minutes * WORDS_PER_MINUTE,
            })

        # Step 3: Remove cross-block duplicates (model sometimes repeats blocks
        # with minor wording variations). Uses fuzzy similarity to catch near-dupes.
        unique_blocks: list[str] = []
        seen_normalized: list[str] = []
        for block in generated_blocks:
            normalized = re.sub(r"\[Minute\s+\d+\s*-\s*\d+\]\s*", "", block).strip()
            normalized = " ".join(normalized.split())
            is_dup = False
            for prev in seen_normalized:
                ratio = SequenceMatcher(None, prev, normalized).ratio()
                if ratio > 0.55:
                    is_dup = True
                    trace.append({
                        "tool": "duplicate_block_removed",
                        "similarity": round(ratio, 2),
                        "preview": block[:80],
                    })
                    break
            if not is_dup:
                seen_normalized.append(normalized)
                unique_blocks.append(block)
        generated_blocks = unique_blocks

        # Step 4: Assemble — script-only output for clean TTS narration
        full_script_body = "\n\n".join(generated_blocks)
        # Strip [Minute X-Y] labels — they were only needed during generation
        full_script_body = re.sub(
            r"^\[Minute\s+\d+\s*-\s*\d+\]\s*\n?", "", full_script_body, flags=re.MULTILINE
        )
        # Remove duplicate Teacher: paragraphs across the full script
        full_script_body = self._remove_script_repetitions(full_script_body, trace)

        title_match = re.search(r"^Lesson Title:\s*(.+)$", outline, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "Lesson Script"

        source_label = (
            "Source: Retrieved from documents"
            if generation_mode == "grounded"
            else "Source: Generated (no matching documents found)"
        )

        header = (
            f"Lesson: {title}\n"
            f"Grade: {format_grade_line(grade_level)} | "
            f"Subject: {format_subject_line(resolved_subject)} | "
            f"Duration: {total} minutes\n"
            f"{source_label}"
        )

        assembled = f"{header}\n\n{full_script_body}".strip()

        # Append references section if we have retrieved chunks
        references = build_references_section(retrieved_chunks)
        if references:
            assembled = f"{assembled}\n\n{references}"

        return {
            "tool": "phased_generator",
            "lesson_text": assembled,
            "outline": outline,
            "prompt": outline_prompt,
        }

    def evaluate_lesson(
        self,
        user_prompt: str,
        lesson_text: str,
        generation_mode: str,
        retrieved_chunks: list[dict],
        subject: str | None = None,
        grade_level: str | None = None,
    ) -> dict:
        """Evaluate a generated script for completeness, density, and constraints."""
        missing_sections = [
            section for section in REQUIRED_SCRIPT_SECTIONS if section not in lesson_text
        ]
        constraints = self._extract_constraints(user_prompt)
        missing_constraints = []
        duration_minutes = constraints["duration_minutes"] or 40
        title_value = self._extract_header_value(lesson_text, "Lesson Title:")
        grade_value = self._extract_header_value(lesson_text, "Grade Level:")
        resolved_subject = subject or self._infer_subject(user_prompt)

        if not title_value:
            missing_constraints.append("lesson_title")

        if constraints["duration_text"] and constraints["duration_text"] not in lesson_text:
            missing_constraints.append(f"duration:{constraints['duration_text']}")
        if grade_level and grade_level not in grade_value:
            missing_constraints.append(f"grade_level:{grade_level}")
        if not grade_level and grade_value != "Unspecified":
            missing_constraints.append("grade_level:Unspecified")
        if resolved_subject:
            subject_value = self._extract_header_value(lesson_text, "Subject:")
            if resolved_subject.lower() not in subject_value.lower():
                missing_constraints.append(f"subject:{resolved_subject}")
        if constraints["needs_activity"] and not re.search(
            r"\b(activity|practice|exercise|task|pair work|group work)\b",
            lesson_text,
            re.IGNORECASE,
        ):
            missing_constraints.append("activity")
        if constraints["needs_exit_ticket"] and "Exit Ticket:" not in lesson_text:
            missing_constraints.append("exit_ticket")
        if '"' not in lesson_text:
            missing_constraints.append("quoted_teacher_lines")
        if "Timeline:" in lesson_text or re.search(r"^\s*[\*\-]\s*Minute", lesson_text, re.MULTILINE):
            missing_constraints.append("timeline_outline_section")
        if "Student Lines:" in lesson_text or "Students:" in lesson_text:
            missing_constraints.append("student_dialogue_format")
        if "Short Block:" in lesson_text:
            missing_constraints.append("outline_markers")
        if lesson_text.lstrip().lower().startswith("here is"):
            missing_constraints.append("extra_intro_text")

        total_words = len(re.findall(r"\b\w+\b", lesson_text))
        target_total_words = duration_minutes * WORDS_PER_MINUTE
        if total_words < int(target_total_words * DENSITY_FLOOR):
            missing_constraints.append(
                f"total_word_count(have ~{total_words}, need ~{target_total_words})"
            )

        timeline_issues = self._evaluate_script_timeline(lesson_text, duration_minutes)
        if timeline_issues:
            missing_constraints.extend(timeline_issues)

        block_issues = self._evaluate_script_blocks(lesson_text, duration_minutes)
        density_warnings: list[str] = []
        if block_issues:
            density_warnings.extend(block_issues)

        completeness_score = (
            len(REQUIRED_SCRIPT_SECTIONS) - len(missing_sections)
        ) / len(REQUIRED_SCRIPT_SECTIONS)
        grounded_support = (
            generation_mode != "grounded"
            or (
                bool(retrieved_chunks)
                and retrieved_chunks[0]["score"] >= self.config.min_retrieval_score
            )
        )

        density_ratio = round(total_words / target_total_words, 2) if target_total_words else 1.0

        passed = (
            not missing_sections
            and not missing_constraints
            and grounded_support
        )

        return {
            "tool": "evaluator",
            "passed": passed,
            "completeness_score": round(completeness_score, 2),
            "missing_sections": missing_sections,
            "missing_constraints": missing_constraints,
            "density_warnings": density_warnings,
            "density_ratio": density_ratio,
            "grounded_support_ok": grounded_support,
        }

    def revise_lesson(
        self,
        user_prompt: str,
        draft_text: str,
        evaluation: dict,
        generation_mode: str,
        retrieved_chunks: list[dict],
        subject: str | None = None,
        grade_level: str | None = None,
    ) -> dict:
        """Revise a draft script to fix evaluation failures."""
        missing_sections = ", ".join(evaluation["missing_sections"]) or "none"
        missing_constraints = ", ".join(evaluation["missing_constraints"]) or "none"
        density_warnings = ", ".join(evaluation.get("density_warnings", [])) or "none"
        constraints = self._extract_constraints(user_prompt)
        duration = constraints["duration_minutes"] or 40
        context_prompt = self.build_generation_prompt(
            user_prompt=user_prompt,
            generation_mode=generation_mode,
            retrieved_chunks=retrieved_chunks,
            subject=subject,
            grade_level=grade_level,
            duration_minutes=duration,
        )
        revision_prompt = f"""
{context_prompt}

You already produced a draft lesson script. Revise it so it fully satisfies the required structure and the constraints listed below.

The target density is {WORDS_PER_MINUTE} words per minute. For a {duration}-minute script, the total script body should contain approximately {duration * WORDS_PER_MINUTE} words.
If a block is flagged as too short, expand it with more detailed teacher dialogue, worked examples, questioning sequences, and pause cues until it reaches the target word count shown in the constraint.
Each Teacher: line must be a full multi-sentence paragraph of spoken dialogue, not a one-line phrase.

Missing sections:
{missing_sections}

Missing constraints:
{missing_constraints}

Density warnings (blocks that need more content):
{density_warnings}

Current draft:
{draft_text}
"""
        return {
            "tool": "reviser",
            "lesson_text": generate_text(revision_prompt),
            "prompt": revision_prompt,
        }

    def _resolve_retrieval_params(
        self,
        user_prompt: str,
        subject: str | None,
        grade_level: str | None,
        topic: str | None,
        retrieval_mode: str,
    ) -> tuple[str | None, str | None, str | None, str]:
        """Infer missing retrieval params from the prompt and pick the best retrieval mode.

        For auto mode:
        1. Infer subject/grade/topic from the prompt if not explicitly provided
        2. Try filtered if all three are present and exist in the corpus
        3. Fall back to all mode otherwise
        """
        resolved_subject = subject or self._infer_subject(user_prompt)
        resolved_grade = grade_level or self._infer_grade_level(user_prompt)
        resolved_topic = topic or self._infer_topic(user_prompt)

        if retrieval_mode not in ("auto", "filtered", "all"):
            retrieval_mode = "auto"

        if retrieval_mode == "auto":
            if resolved_subject and not field_value_exists("subject", resolved_subject):
                # Subject doesn't exist in the corpus at all — skip retrieval
                # to avoid pulling irrelevant content from unrelated subjects.
                retrieval_mode = "skip"
            elif resolved_subject:
                # Subject is known — always filter by it so unrelated subjects
                # (e.g. chemistry when the request is mathematics) are excluded.
                # Grade is used as an additional filter only when present.
                if filter_has_matches(subject=resolved_subject, grade_level=resolved_grade):
                    retrieval_mode = "filtered"
                else:
                    retrieval_mode = "all"
            else:
                retrieval_mode = "all"

        return resolved_subject, resolved_grade, resolved_topic, retrieval_mode

    def run(
        self,
        user_prompt: str,
        subject: str | None = None,
        grade_level: str | None = None,
        topic: str | None = None,
        retrieval_limit: int | None = None,
        retrieval_mode: str = "auto",
        retrieval_method: str = "dense",
    ) -> dict:
        """Execute the full agent pipeline: domain check, retrieve, generate, evaluate."""
        trace: list[dict] = []

        domain_result = self.check_education_domain(user_prompt)
        trace.append(domain_result)
        if not domain_result["is_education_related"]:
            refusal_text = (
                "This project only handles education-related requests. "
                "Please ask for lesson planning, teaching materials, curriculum support, or other classroom-focused help."
            )
            return {
                "user_prompt": user_prompt,
                "generation_mode": "refuse",
                "source_notice": "Request refused because it is not education-related.",
                "retrieved_chunks": [],
                "prompt_used": None,
                "lesson_text": refusal_text,
                "evaluation": None,
                "agent_trace": trace,
                "citations": [],
            }

        # Resolve retrieval parameters automatically
        resolved_subject, resolved_grade, resolved_topic, resolved_mode = self._resolve_retrieval_params(
            user_prompt=user_prompt,
            subject=subject,
            grade_level=grade_level,
            topic=topic,
            retrieval_mode=retrieval_mode,
        )
        trace.append({
            "tool": "param_resolver",
            "subject": resolved_subject,
            "grade_level": resolved_grade,
            "topic": resolved_topic,
            "retrieval_mode": resolved_mode,
            "inferred": {
                "subject": subject is None and resolved_subject is not None,
                "grade_level": grade_level is None and resolved_grade is not None,
                "topic": topic is None and resolved_topic is not None,
            },
        })

        # Skip strict validation in auto mode — the resolver already handled it
        if resolved_mode == "filtered":
            validation_result = self.validate_retrieval_request(
                retrieval_mode=resolved_mode,
                subject=resolved_subject,
                grade_level=resolved_grade,
                topic=resolved_topic,
            )
            trace.append(validation_result)
            if not validation_result["passed"]:
                # Auto-downgrade to all mode instead of erroring
                resolved_mode = "all"
                trace.append({"tool": "mode_fallback", "reason": validation_result["message"], "new_mode": "all"})

        if resolved_mode == "skip":
            # Subject not in corpus — go straight to fallback, no retrieval
            trace.append({"tool": "retriever_skip", "reason": f"Subject '{resolved_subject}' not found in corpus"})
            retrieved_chunks = []
        else:
            retrieval_result = self.retrieve_context(
                user_prompt=user_prompt,
                subject=resolved_subject,
                grade_level=resolved_grade,
                topic=resolved_topic,
                retrieval_limit=retrieval_limit,
                retrieval_mode=resolved_mode,
                retrieval_method=retrieval_method,
            )
            trace.append(retrieval_result)
            retrieved_chunks = retrieval_result["chunks"]

        generation_mode = self.choose_generation_mode(retrieved_chunks, retrieval_method)
        source_notice = (
            "Lesson generated using retrieved source material from the vector store."
            if generation_mode == "grounded"
            else (
                "No sufficiently relevant retrieved source material was available. "
                "Lesson generated using model fallback rather than RAG-grounded sources."
            )
        )

        constraints = self._extract_constraints(user_prompt)

        # Use phased generation: outline → block-by-block → closing → assemble
        phased_result = self.generate_phased(
            user_prompt=user_prompt,
            generation_mode=generation_mode,
            retrieved_chunks=retrieved_chunks,
            subject=resolved_subject,
            grade_level=resolved_grade,
            duration_minutes=constraints["duration_minutes"],
            trace=trace,
        )
        prompt_used = phased_result.get("prompt")

        lesson_text = self.normalize_lesson_text(
            lesson_text=phased_result["lesson_text"],
            user_prompt=user_prompt,
            generation_mode=generation_mode,
            subject=resolved_subject,
            grade_level=resolved_grade,
            retrieved_chunks=retrieved_chunks,
        )
        evaluation = self.evaluate_lesson(
            user_prompt=user_prompt,
            lesson_text=lesson_text,
            generation_mode=generation_mode,
            retrieved_chunks=retrieved_chunks,
            subject=resolved_subject,
            grade_level=resolved_grade,
        )
        trace.append(evaluation)

        density_ratio = evaluation.get("density_ratio", 1.0)
        if density_ratio < 0.85:
            density_pct = int(density_ratio * 100)
            source_notice += (
                f" Note: the script covers approximately {density_pct}% of the target duration density. "
                "You may want to ask for more detail in specific sections."
            )

        citations = extract_citations(lesson_text, retrieved_chunks)

        return {
            "user_prompt": user_prompt,
            "generation_mode": generation_mode,
            "source_notice": source_notice,
            "retrieved_chunks": retrieved_chunks,
            "prompt_used": prompt_used,
            "lesson_text": lesson_text,
            "citations": citations,
            "evaluation": evaluation,
            "agent_trace": trace,
        }

    def validate_retrieval_request(
        self,
        retrieval_mode: str,
        subject: str | None = None,
        grade_level: str | None = None,
        topic: str | None = None,
    ) -> dict:
        """Validate that filtered retrieval inputs exist in the Qdrant corpus."""
        if retrieval_mode != "filtered":
            return {
                "tool": "input_validator",
                "passed": True,
                "message": "Retrieval input accepted.",
            }

        missing_fields = []
        if not subject:
            missing_fields.append("subject")
        if not grade_level:
            missing_fields.append("grade_level")
        if not topic:
            missing_fields.append("topic")

        if missing_fields:
            return {
                "tool": "input_validator",
                "passed": False,
                "message": (
                    "Filtered Mode requires subject, grade level, and topic to be filled. "
                    f"Missing: {', '.join(missing_fields)}."
                ),
            }

        if not collection_has_points():
            return {
                "tool": "input_validator",
                "passed": False,
                "message": "No indexed source documents are available yet. Run ingestion before using Filtered Mode.",
            }

        if not field_value_exists("subject", subject):
            available_subjects = ", ".join(list_payload_values("subject"))
            return {
                "tool": "input_validator",
                "passed": False,
                "message": (
                    f"The selected subject '{subject}' does not exist in the indexed document set. "
                    f"Available subjects include: {available_subjects or 'none found'}."
                ),
            }

        if not field_value_exists("grade_level", grade_level):
            available_grades = ", ".join(list_payload_values("grade_level"))
            return {
                "tool": "input_validator",
                "passed": False,
                "message": (
                    f"The selected grade level '{grade_level}' does not exist in the indexed document set. "
                    f"Available grade values include: {available_grades or 'none found'}."
                ),
            }

        if not filter_has_matches(
            subject=subject,
            grade_level=grade_level,
        ):
            return {
                "tool": "input_validator",
                "passed": False,
                "message": (
                    "The selected filtered combination does not match any indexed documents. "
                    "Try a different grade, subject, or switch to All Mode."
                ),
            }

        return {
            "tool": "input_validator",
            "passed": True,
            "message": "Retrieval input accepted.",
        }

    def _build_input_error_response(self, user_prompt: str, message: str, trace: list[dict]) -> dict:
        """Build a standardized error response for invalid retrieval inputs."""
        return {
            "user_prompt": user_prompt,
            "generation_mode": "input_error",
            "source_notice": "Request could not be processed because the retrieval inputs were invalid.",
            "retrieved_chunks": [],
            "prompt_used": None,
            "lesson_text": message,
            "evaluation": None,
            "agent_trace": trace,
            "citations": [],
        }

    def _extract_header_value(self, text: str, header: str) -> str:
        """Extract the value after a header line like 'Grade Level: 8'."""
        pattern = rf"^{re.escape(header)}[ \t]*(.*)$"
        match = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip() if match else ""

    def _evaluate_script_blocks(self, lesson_text: str, duration_minutes: int) -> list[str]:
        """Check each script block for density and teacher-line requirements."""
        issues: list[str] = []
        blocks = self._extract_script_blocks(lesson_text)

        for block in blocks:
            block_label = f"[Minute {block['start']}-{block['end']}]"
            block_text = block["text"]
            block_minutes = block["end"] - block["start"]
            teacher_lines = len(re.findall(r"Teacher:\s*\".*?\"", block_text, re.DOTALL))
            words = len(re.findall(r"\b\w+\b", block_text))
            pause_seconds = sum(
                int(seconds)
                for seconds in re.findall(r"\[(?:Pause|Think time)\s+(\d+)\s+seconds\]", block_text, re.IGNORECASE)
            )
            estimated_seconds = int((words / WORDS_PER_MINUTE) * 60) + pause_seconds
            target_seconds = block_minutes * 60
            target_words = block_minutes * WORDS_PER_MINUTE
            required_teacher_lines = max(3, ceil(block_minutes * 0.6))

            if teacher_lines < required_teacher_lines:
                issues.append(
                    f"teacher_lines:{block_label}(have {teacher_lines}, need {required_teacher_lines})"
                )
            if estimated_seconds < int(target_seconds * DENSITY_FLOOR):
                issues.append(
                    f"short_block:{block_label}(have ~{words} words, need ~{target_words})"
                )
            if estimated_seconds > int(target_seconds * DENSITY_CEILING):
                issues.append(
                    f"long_block:{block_label}(have ~{words} words, target ~{target_words})"
                )

        return issues

    def _evaluate_script_timeline(self, lesson_text: str, duration_minutes: int) -> list[str]:
        """Verify script blocks cover the full duration with no gaps or overlaps."""
        blocks = self._extract_script_blocks(lesson_text)
        if not blocks:
            return ["minute_blocks"]

        issues: list[str] = []

        if blocks[0]["start"] != 0:
            issues.append("timeline_start")

        previous_end = 0
        for block in blocks:
            if block["start"] != previous_end:
                issues.append("timeline_gap_or_overlap")
                break
            if block["end"] <= block["start"]:
                issues.append("timeline_invalid_block")
                break
            previous_end = block["end"]

        if blocks[-1]["end"] != duration_minutes:
            issues.append("timeline_total_duration")

        return issues

    def _extract_script_blocks(self, lesson_text: str) -> list[dict]:
        """Parse [Minute X-Y] blocks from lesson text with their content."""
        matches = list(re.finditer(r"\[Minute (\d+)-(\d+)\]", lesson_text))
        blocks: list[dict] = []

        for index, match in enumerate(matches):
            start = int(match.group(1))
            end = int(match.group(2))
            content_start = match.end()
            content_end = matches[index + 1].start() if index + 1 < len(matches) else lesson_text.find("Exit Ticket:")
            if content_end == -1:
                content_end = len(lesson_text)
            blocks.append(
                {
                    "start": start,
                    "end": end,
                    "text": lesson_text[content_start:content_end],
                }
            )

        return blocks

    def _extract_constraints(self, user_prompt: str) -> dict:
        """Extract duration, activity, and exit ticket requirements from the prompt."""
        duration_match = re.search(r"(\d+)\s*-\s*minute|(\d+)\s*minute", user_prompt, re.IGNORECASE)
        duration = None
        if duration_match:
            duration = int(next(group for group in duration_match.groups() if group is not None))
        lower = user_prompt.lower()
        return {
            "duration_minutes": duration,
            "duration_text": f"{duration} minutes" if duration is not None else None,
            "needs_activity": "activity" in lower,
            "needs_exit_ticket": "exit ticket" in lower,
        }

    def _infer_subject(self, user_prompt: str) -> str | None:
        """Infer the academic subject from keywords in the prompt."""
        lower = user_prompt.lower()
        for subject, keywords in SUBJECT_KEYWORDS.items():
            if any(keyword in lower for keyword in keywords):
                return subject
        return None

    def _infer_grade_level(self, user_prompt: str) -> str | None:
        """Extract grade level from patterns like 'grade 8' or '8th grade'."""
        lower = user_prompt.lower()
        match = re.search(r"grade\s*(\d+)", lower)
        if match:
            return match.group(1)
        match = re.search(r"class\s*(\d+)", lower)
        if match:
            return match.group(1)
        match = re.search(r"(\d+)(?:st|nd|rd|th)\s+grade", lower)
        if match:
            return match.group(1)
        return None

    def _infer_topic(self, user_prompt: str) -> str | None:
        """Extract the lesson topic from 'lesson on/about <topic>' patterns."""
        lower = user_prompt.lower()
        # Look for "on <topic>" or "about <topic>" patterns
        match = re.search(r"(?:lesson|script|class)\s+(?:on|about)\s+(.+?)(?:\s+for\s+|\s+with\s+|\s+that\s+|\s*\(|\s*$)", lower)
        if match:
            topic = match.group(1).strip().rstrip(".,;:")
            if topic and len(topic) > 2:
                return topic
        return None

    def normalize_lesson_text(
        self,
        lesson_text: str,
        user_prompt: str,
        generation_mode: str,
        subject: str | None = None,
        grade_level: str | None = None,
        retrieved_chunks: list[dict] | None = None,
    ) -> str:
        """Clean up LLM output artifacts while preserving the script content."""
        text = lesson_text.replace("\r\n", "\n").strip()

        lines = text.splitlines()
        cleaned_lines: list[str] = []
        skip_student_block = False

        for line in lines:
            stripped = line.strip()

            # Remove student dialogue blocks
            if stripped.startswith("Student Lines:"):
                skip_student_block = True
                continue
            if skip_student_block:
                if stripped.startswith("Teacher:") or stripped.startswith("[Minute "):
                    skip_student_block = False
                else:
                    continue

            if stripped.startswith("Students:"):
                continue
            if stripped.startswith("- Student"):
                continue
            if stripped.lower().startswith("here is "):
                continue

            # Remove sections we no longer include
            if stripped.startswith(("Exit Ticket:", "Homework:", "Teacher Notes:", "Sources Used:")):
                break

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def _parse_and_validate_block_plan(
        self, outline: str, total_minutes: int
    ) -> list[tuple[int, int, str]]:
        """Parse block plan from outline and enforce the requested duration.

        Fixes common LLM mistakes:
        - Blocks that exceed the requested total duration are clipped
        - Duplicate block ranges are removed
        - If the parsed plan is invalid, a sensible default plan is returned
        """
        matches = list(re.finditer(r"\[Minute (\d+)-(\d+)\]\s*(.*)", outline))

        if not matches:
            return self._default_block_plan(total_minutes)

        raw_blocks = [
            (int(m.group(1)), int(m.group(2)), m.group(3).strip())
            for m in matches
        ]

        # Remove duplicates (same start-end range)
        seen_ranges: set[tuple[int, int]] = set()
        deduped: list[tuple[int, int, str]] = []
        for start, end, desc in raw_blocks:
            if (start, end) not in seen_ranges:
                seen_ranges.add((start, end))
                deduped.append((start, end, desc))

        # Clip blocks to the requested total duration
        validated: list[tuple[int, int, str]] = []
        for start, end, desc in deduped:
            if start >= total_minutes:
                break
            clipped_end = min(end, total_minutes)
            if clipped_end > start:
                validated.append((start, clipped_end, desc))

        if not validated:
            return self._default_block_plan(total_minutes)

        # Ensure the plan starts at 0
        if validated[0][0] != 0:
            validated[0] = (0, validated[0][1], validated[0][2])

        # Fill gaps between blocks (e.g., 0-5 then 15-25 → insert 5-15)
        gap_filled: list[tuple[int, int, str]] = []
        for i, (start, end, desc) in enumerate(validated):
            if gap_filled and start > gap_filled[-1][1]:
                gap_start = gap_filled[-1][1]
                gap_filled.append((gap_start, start, "Continued instruction"))
            gap_filled.append((start, end, desc))
        validated = gap_filled

        # Ensure the plan ends at total_minutes
        last_start, last_end, last_desc = validated[-1]
        if last_end < total_minutes:
            validated[-1] = (last_start, total_minutes, last_desc)

        return validated

    def _default_block_plan(self, total_minutes: int) -> list[tuple[int, int, str]]:
        """Return a sensible default block plan when the LLM outline is invalid."""
        if total_minutes <= 10:
            return [(0, total_minutes, "Full lesson")]
        if total_minutes <= 20:
            return [
                (0, 5, "Opening"),
                (5, total_minutes, "Core instruction and closing"),
            ]
        return [
            (0, 5, "Opening"),
            (5, total_minutes - 10, "Core instruction"),
            (total_minutes - 10, total_minutes, "Review and closing"),
        ]

    def _trim_to_current_block(self, block_text: str, current_label: str) -> str:
        """If the LLM wrote past the assigned block into the next one, cut it off."""
        # Find any [Minute X-Y] label that isn't the current one
        next_block = re.search(r"\[Minute\s+\d+\s*-\s*\d+\]", block_text)
        if next_block:
            # Skip past the current label itself
            first_match_start = next_block.start()
            if block_text[first_match_start:].startswith(current_label):
                # Look for the NEXT block label after the current one
                remainder = block_text[first_match_start + len(current_label):]
                next_block = re.search(r"\[Minute\s+\d+\s*-\s*\d+\]", remainder)
                if next_block:
                    cut_point = first_match_start + len(current_label) + next_block.start()
                    return block_text[:cut_point].rstrip()
            else:
                # The first label found isn't ours — something went wrong, keep as-is
                pass
        return block_text

    def _trim_truncated_block(self, block_text: str) -> str:
        """Trim blocks that end mid-sentence back to the last complete sentence."""
        stripped = block_text.rstrip()
        truncated = not stripped.endswith((".", '"', ")", "]", "!", "?", "seconds]"))
        if truncated:
            last_period = stripped.rfind(". ")
            last_quote = stripped.rfind('"')
            cut_at = max(last_period, last_quote)
            if cut_at > len(stripped) // 2:
                return stripped[: cut_at + 1]
        return block_text

    def _remove_script_repetitions(
        self, script_body: str, trace: list[dict] | None = None,
    ) -> str:
        """Remove duplicate Teacher: paragraphs across the full assembled script.

        Splits on Teacher: line boundaries (not blank lines) so it works
        regardless of how the model formats whitespace between paragraphs.
        Uses fuzzy matching so paraphrased repetitions are also caught.
        """
        # Split into segments at each "Teacher:" line boundary
        lines = script_body.split("\n")
        segments: list[str] = []
        current: list[str] = []

        for line in lines:
            if line.strip().startswith("Teacher:") and current:
                segments.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            segments.append("\n".join(current))

        seen: list[str] = []
        kept: list[str] = []
        removed = 0

        for seg in segments:
            stripped = seg.strip()
            if not stripped:
                continue

            # Only deduplicate Teacher: segments
            if not stripped.startswith("Teacher:"):
                kept.append(seg)
                continue

            # Strip pause/think cues for comparison so timing differences
            # don't prevent matching otherwise-identical content
            speech_only = re.sub(
                r"\[(?:Pause|Think time|Think-time)\s+\d+\s*seconds?\s*\]",
                "", stripped, flags=re.IGNORECASE,
            )
            normalized = " ".join(speech_only.split()).lower()

            if len(normalized) < 80:
                kept.append(seg)
                seen.append(normalized)
                continue

            is_dup = False
            for prev in seen:
                ratio = SequenceMatcher(None, prev, normalized).ratio()
                if ratio > 0.6:
                    is_dup = True
                    break

            if is_dup:
                removed += 1
            else:
                seen.append(normalized)
                kept.append(seg)

        if removed and trace is not None:
            trace.append({
                "tool": "script_paragraph_dedup",
                "paragraphs_removed": removed,
            })

        return "\n\n".join(kept)

    def _remove_repetitions(self, block_text: str) -> str:
        """Detect and remove repeated paragraphs within a single block.

        Splits on Teacher: line boundaries (not blank lines) and uses fuzzy
        matching. Skips duplicates instead of truncating to avoid content loss
        that causes the next block to regenerate similar material.
        """
        lines = block_text.split("\n")
        segments: list[str] = []
        current: list[str] = []

        for line in lines:
            if line.strip().startswith("Teacher:") and current:
                segments.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            segments.append("\n".join(current))

        seen: list[str] = []
        kept: list[str] = []
        for seg in segments:
            stripped = seg.strip()
            if not stripped.startswith("Teacher:"):
                kept.append(seg)
                continue

            normalized = " ".join(stripped.split()).lower()
            if len(normalized) < 60:
                kept.append(seg)
                seen.append(normalized)
                continue

            is_dup = False
            for prev in seen:
                ratio = SequenceMatcher(None, prev, normalized).ratio()
                if ratio > 0.65:
                    is_dup = True
                    break

            if not is_dup:
                seen.append(normalized)
                kept.append(seg)
            # Skip duplicate — do NOT break, keep processing remaining segments

        return "\n\n".join(kept)

    def _build_default_title(self, user_prompt: str, subject: str | None) -> str:
        """Generate a default lesson title from the prompt or subject."""
        if subject:
            return f"{subject.title()} Lesson Script"
        return "Lesson Script"
