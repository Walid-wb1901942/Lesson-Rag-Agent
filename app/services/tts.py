"""Text-to-speech service supporting Bark (emotional) and Edge-TTS (fast)."""

import asyncio
import re
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path


# ---- Shared config ----

# Regex patterns for parsing the script
_PAUSE_RE = re.compile(
    r"\[\s*(?:Pause|Think time|Think-time)\s+(\d+)\s*seconds?\s*\]",
    re.IGNORECASE,
)
_TEACHER_RE = re.compile(r"^Teacher:\s*", re.MULTILINE)
_BLOCK_LABEL_RE = re.compile(r"^\[Minute\s+\d+\s*-\s*\d+\]\s*$", re.MULTILINE)
_STAGE_DIRECTION_RE = re.compile(
    r"^\(.*?\)\s*$|^\[(?!Minute).*?\]\s*$", re.MULTILINE
)


def parse_script_for_speech(script: str) -> list[dict]:
    """Parse a lesson script into a sequence of speech and silence segments.

    Returns a list of dicts:
      {"type": "speech", "text": "..."}
      {"type": "silence", "seconds": 20}
    """
    segments: list[dict] = []

    # Extract only the Full Classroom Script body + closing sections
    marker = "Full Classroom Script:"
    start_idx = script.find(marker)
    if start_idx != -1:
        body = script[start_idx + len(marker):]
    else:
        body = script

    # Remove display math blocks — these are visual (board/slides), not spoken
    body = re.sub(r'\\\[.*?\\\]', '', body, flags=re.DOTALL)
    body = re.sub(r'\$\$.*?\$\$', '', body, flags=re.DOTALL)
    # Remove equation continuation lines (lines starting with = sign)
    body = re.sub(r'^\s*=\s.*$', '', body, flags=re.MULTILINE)

    lines = body.split("\n")
    current_speech = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if _BLOCK_LABEL_RE.match(stripped):
            if current_speech:
                segments.append({"type": "speech", "text": " ".join(current_speech)})
                current_speech = []
            continue

        pause_match = _PAUSE_RE.search(stripped)
        if pause_match:
            if current_speech:
                segments.append({"type": "speech", "text": " ".join(current_speech)})
                current_speech = []
            segments.append({"type": "silence", "seconds": int(pause_match.group(1))})
            text_before = stripped[:pause_match.start()].strip()
            if text_before:
                text_before = _TEACHER_RE.sub("", text_before).strip().strip('"')
                if text_before:
                    segments.insert(-1, {"type": "speech", "text": text_before})
            continue

        if _STAGE_DIRECTION_RE.match(stripped):
            continue

        if stripped.startswith(("- ", "* ", "**")) and "Teacher:" not in stripped:
            continue

        if "Teacher:" in stripped:
            text = _TEACHER_RE.sub("", stripped).strip().strip('"')
            if text:
                if current_speech:
                    segments.append({"type": "speech", "text": " ".join(current_speech)})
                    current_speech = []
                current_speech.append(text)
        elif stripped.startswith('"') or (
            current_speech
            and not stripped.startswith(
                ("Exit Ticket:", "Homework:", "Teacher Notes:", "Sources Used:")
            )
        ):
            text = stripped.strip('"')
            if text:
                current_speech.append(text)
        elif stripped.startswith(("Exit Ticket:", "Homework:")):
            if current_speech:
                segments.append({"type": "speech", "text": " ".join(current_speech)})
                current_speech = []
        elif stripped.startswith(("Teacher Notes:", "Sources Used:")):
            if current_speech:
                segments.append({"type": "speech", "text": " ".join(current_speech)})
                current_speech = []
            break

    if current_speech:
        segments.append({"type": "speech", "text": " ".join(current_speech)})

    return segments


# ---- Edge-TTS engine (fast, no GPU needed) ----

# Good teaching voices:
# en-US-GuyNeural      — warm male, supports styles
# en-US-JennyNeural    — clear female, supports styles
# en-GB-RyanNeural     — British male
EDGE_VOICE = "en-US-GuyNeural"
EDGE_RATE = "-25%"  # slower rate to match natural teacher speaking pace (~95 wpm)


def _preprocess_for_speech(text: str) -> str:
    """Convert mathematical notation, LaTeX, and symbols to speakable words."""
    # ---- Step 1: Strip LaTeX math delimiters ----
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("$$", "").replace("$", "")
    # LaTeX spacing commands
    text = text.replace("\\,", " ")
    text = text.replace("\\;", " ")
    text = text.replace("\\!", "")
    text = text.replace("\\quad", " ")
    text = text.replace("\\qquad", " ")

    # ---- Step 2: Trig & common math functions (before generic \cmd removal) ----
    # With parentheses: sin(x) or \sin(x) → "sine of x"
    text = re.sub(r'\\?\bsin\s*\(([^)]*)\)', r' sine of \1 ', text)
    text = re.sub(r'\\?\bcos\s*\(([^)]*)\)', r' cosine of \1 ', text)
    text = re.sub(r'\\?\btan\s*\(([^)]*)\)', r' tangent of \1 ', text)
    text = re.sub(r'\\?\blog\s*\(([^)]*)\)', r' log of \1 ', text)
    text = re.sub(r'\\?\bln\s*\(([^)]*)\)', r' natural log of \1 ', text)
    # Without parentheses: \sin x → "sine"
    text = re.sub(r'\\sin\b', ' sine ', text)
    text = re.sub(r'\\cos\b', ' cosine ', text)
    text = re.sub(r'\\tan\b', ' tangent ', text)
    text = re.sub(r'\\log\b', ' log ', text)
    text = re.sub(r'\\ln\b', ' natural log ', text)

    # ---- Step 3: Multi-pass LaTeX resolution (handles nesting) ----
    # Use [^{}]* to match innermost groups first; repeat to resolve outer layers
    for _ in range(3):
        text = re.sub(r'\\sqrt\{([^{}]*)\}', r' the square root of \1 ', text)
        text = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r' \1 over \2 ', text)
        # Integral with braced limits: \int_{a}^{b}
        text = re.sub(r'\\int_\{([^{}]*)\}\^\{([^{}]*)\}', r' the integral from \1 to \2 of ', text)
        # Integral with bare limits: \int_a^b
        text = re.sub(r'\\int_(\S+?)\^(\S+?)\b', r' the integral from \1 to \2 of ', text)
        # Braced superscript/subscript
        text = re.sub(r'\^\{([^{}]*)\}', r' to the power of \1 ', text)
        text = re.sub(r'_\{([^{}]*)\}', r' sub \1 ', text)

    # ---- Step 3b: Bare (un-braced) superscript/subscript: x^2, x_n ----
    text = re.sub(r'\^(\d+)', r' to the power of \1 ', text)
    text = re.sub(r'_(\w)', r' sub \1 ', text)

    # ---- Step 4: Remaining specific LaTeX commands ----
    text = re.sub(r'\\int', ' the integral of ', text)
    text = re.sub(r'\\cdot', ' times ', text)
    text = re.sub(r'\\times', ' times ', text)
    text = re.sub(r'\\pm', ' plus or minus ', text)
    text = re.sub(r'\\infty', ' infinity ', text)
    text = re.sub(r'\\pi\b', ' pi ', text)
    text = re.sub(r'\\sum', ' the sum of ', text)
    text = re.sub(r'\\geq', ' greater than or equal to ', text)
    text = re.sub(r'\\leq', ' less than or equal to ', text)
    text = re.sub(r'\\neq', ' does not equal ', text)
    text = re.sub(r'\\approx', ' approximately equals ', text)

    # Remove \left, \right, \Big, etc.
    text = re.sub(r'\\(?:left|right|Big|big|bigg|Bigg)\s*[|()[\]{}.]?', '', text)
    # \text{...}, \mathrm{...} → keep content
    text = re.sub(r'\\(?:text|mathrm|mathbf|mathit|textbf)\{([^}]*)\}', r'\1', text)
    # Remove remaining \command sequences
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    # Clean up braces and remaining backslashes
    text = text.replace("{", "").replace("}", "").replace("\\", "")

    # ---- Step 5: Differential notation ----
    # "dx" → "d x" so TTS says "dee ex" not "ducks"
    text = re.sub(r'\bd([xyzt])\b', r'd \1', text)

    # ---- Step 6: Strip markdown formatting ----
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'\1', text)

    # ---- Step 7: Unicode math symbols ----
    text = text.replace("×", " times ")
    text = text.replace("÷", " divided by ")
    text = text.replace("≠", " does not equal ")
    text = text.replace("≤", " less than or equal to ")
    text = text.replace("≥", " greater than or equal to ")
    text = text.replace("²", " squared")
    text = text.replace("³", " cubed")
    text = text.replace("π", " pi ")
    text = text.replace("∫", " the integral of ")
    text = text.replace("∑", " the sum of ")
    text = text.replace("±", " plus or minus ")
    text = text.replace("∞", " infinity ")
    text = text.replace("√", " the square root of ")

    # ---- Step 8: ASCII math operators in numeric context ----
    text = re.sub(r'(\d)\s*\*\s*(\d)', r'\1 times \2', text)
    text = re.sub(r'(\d)\s*\+\s*(\d)', r'\1 plus \2', text)
    text = re.sub(r'(\d)\s+-\s+(\d)', r'\1 minus \2', text)
    text = re.sub(r'(\d)\s*/\s*(\d)', r'\1 divided by \2', text)
    text = re.sub(r'(\d)\s*\^\s*(\d)', r'\1 to the power of \2', text)
    text = re.sub(r'(\d)\s*=\s*', r'\1 equals ', text)

    # ---- Step 9: Clean up ----
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Matches sentence-ending punctuation (with optional closing quote) followed by whitespace.
# Uses a capturing group instead of a lookbehind to avoid variable-width lookbehind errors.
_SENTENCE_BOUNDARY_RE = re.compile(r'([.!?]["\']?)\s+')


def _split_at_sentence_boundary(text: str, max_chars: int) -> tuple[str, str]:
    """Split text into (head, tail) where head is at most max_chars long.

    Splits at the last sentence boundary before max_chars.  Falls back to the
    last whitespace if no sentence boundary is found.
    """
    if len(text) <= max_chars:
        return text, ""
    window = text[:max_chars]
    last_end = -1
    for m in _SENTENCE_BOUNDARY_RE.finditer(window):
        last_end = m.end(1)  # position after punctuation+quote, before the whitespace
    if last_end > 0:
        return text[:last_end].strip(), text[last_end:].strip()
    last_space = window.rfind(" ")
    if last_space > 0:
        return text[:last_space].strip(), text[last_space:].strip()
    return window, text[max_chars:]


async def _async_generate_edge_chunked(segments: list[dict], mp3_path: str) -> None:
    """Generate audio segment-by-segment and concatenate MP3 bytes.

    This avoids Edge-TTS skipping content when text is too long, and
    preserves actual pause durations between speech segments.  Chunks are split
    at sentence boundaries so Edge-TTS never cuts a word or sentence in half.
    """
    import edge_tts

    # Build text chunks of reasonable size (~3000 chars each)
    MAX_CHUNK = 3000
    chunks: list[str] = []
    current_chunk = ""

    for seg in segments:
        if seg["type"] == "silence":
            # Each "..." adds ~1-2s of natural pause at the slowed rate
            pause_count = max(1, seg["seconds"] // 2)
            piece = " ... " * pause_count
        elif seg["type"] == "speech":
            piece = _preprocess_for_speech(seg["text"]) + " "
        else:
            continue

        combined = current_chunk + piece
        if len(combined) > MAX_CHUNK and current_chunk.strip():
            # Flush current_chunk at a sentence boundary
            head, tail = _split_at_sentence_boundary(current_chunk.strip(), MAX_CHUNK)
            chunks.append(head)
            # Any tail left over from splitting becomes the start of the next chunk
            current_chunk = (tail + " " + piece.strip()).strip() + " " if tail else piece
        else:
            current_chunk = combined

    # Flush remainder, splitting oversized final chunks at sentence boundaries
    remaining = current_chunk.strip()
    while remaining:
        if len(remaining) <= MAX_CHUNK:
            chunks.append(remaining)
            break
        head, remaining = _split_at_sentence_boundary(remaining, MAX_CHUNK)
        chunks.append(head)

    # Generate each chunk separately and concatenate raw MP3 bytes
    all_audio = b""
    for chunk_text in chunks:
        communicate = edge_tts.Communicate(chunk_text, EDGE_VOICE, rate=EDGE_RATE)
        async for data in communicate.stream():
            if data["type"] == "audio":
                all_audio += data["data"]

    with open(mp3_path, "wb") as f:
        f.write(all_audio)


def generate_lesson_audio_edge(script: str, output_path: str | Path) -> Path:
    """Generate audio from a lesson script using Edge-TTS (fast).

    Produces a single MP3 file with speech and built-in pauses.
    Generates in chunks to avoid skipping on long scripts.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mp3_path = output_path.with_suffix(".mp3")

    segments = parse_script_for_speech(script)
    if not segments:
        raise ValueError("No speakable content found in the script.")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_async_generate_edge_chunked(segments, str(mp3_path)))
    finally:
        loop.close()

    return mp3_path


# ---- Bark engine (emotional, needs GPU) ----

BARK_SPEAKER = "v2/en_speaker_6"
BARK_MAX_CHUNK_CHARS = 250

_bark_models_loaded = False


def _ensure_bark_models():
    global _bark_models_loaded
    if not _bark_models_loaded:
        from bark.generation import preload_models
        preload_models()
        _bark_models_loaded = True


def _split_text_into_chunks(text: str, max_chars: int = BARK_MAX_CHUNK_CHARS) -> list[str]:
    """Split long text into Bark-friendly chunks at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if not sentence.strip():
            continue
        if len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence

    if current.strip():
        chunks.append(current.strip())

    final: list[str] = []
    for chunk in chunks:
        if len(chunk) > max_chars:
            words = chunk.split()
            part = ""
            for word in words:
                if len(part) + len(word) + 1 > max_chars and part:
                    final.append(part.strip())
                    part = word
                else:
                    part = f"{part} {word}" if part else word
            if part.strip():
                final.append(part.strip())
        else:
            final.append(chunk)

    return final


def generate_lesson_audio_bark(script: str, output_path: str | Path) -> Path:
    """Generate audio from a lesson script using Bark TTS (emotional)."""
    from bark import generate_audio, SAMPLE_RATE

    _ensure_bark_models()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segments = parse_script_for_speech(script)
    audio_pieces: list[np.ndarray] = []

    for segment in segments:
        if segment["type"] == "silence":
            silence = np.zeros(int(SAMPLE_RATE * segment["seconds"]), dtype=np.float32)
            audio_pieces.append(silence)
        elif segment["type"] == "speech":
            chunks = _split_text_into_chunks(_preprocess_for_speech(segment["text"]))
            for chunk in chunks:
                audio = generate_audio(chunk, history_prompt=BARK_SPEAKER)
                audio_pieces.append(audio)
                audio_pieces.append(np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.float32))

    if not audio_pieces:
        raise ValueError("No speakable content found in the script.")

    full_audio = np.concatenate(audio_pieces)
    max_val = np.max(np.abs(full_audio))
    if max_val > 0:
        full_audio = full_audio / max_val * 0.95

    wav_path = output_path.with_suffix(".wav")
    wavfile.write(str(wav_path), SAMPLE_RATE, (full_audio * 32767).astype(np.int16))
    return wav_path


# ---- Public API ----

def generate_lesson_audio(
    script: str, output_path: str | Path, engine: str = "edge"
) -> Path:
    """Generate audio from a lesson script.

    Args:
        script: The full lesson script text.
        output_path: Where to save the output file.
        engine: "edge" (fast, no GPU) or "bark" (emotional, needs GPU).

    Returns:
        Path to the generated audio file.
    """
    if engine == "bark":
        return generate_lesson_audio_bark(script, output_path)
    return generate_lesson_audio_edge(script, output_path)
