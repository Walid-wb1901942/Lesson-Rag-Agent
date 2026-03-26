import re
from uuid import NAMESPACE_URL, uuid5

import tiktoken
from app.config import settings
from app.schemas import RawDocument, DocumentChunk, ChunkMetadata

ENCODING = tiktoken.get_encoding("cl100k_base")
TOKEN_SAFETY_MARGIN = 32

PAGE_MARKER_RE = re.compile(r"---\s*\[Page\s+(\d+)\]\s*---")

# Patterns that indicate a section/heading boundary (used for structure-aware splitting)
HEADING_PATTERNS = [
    re.compile(r"^#{1,6}\s+", re.MULTILINE),                     # Markdown headings
    re.compile(r"^(?:Chapter|Section|Unit|Part)\s+\d", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\d+\.\d+[\.\s]", re.MULTILINE),                # Numbered sections like 1.2 or 1.2.3
    re.compile(r"^[A-Z][A-Z\s]{4,}$", re.MULTILINE),             # ALL-CAPS headings (5+ chars)
]


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string using tiktoken."""
    return len(ENCODING.encode(text))


def extract_page_number(text: str) -> int | None:
    """Return the first page marker number found in a text block, or None."""
    match = PAGE_MARKER_RE.search(text)
    return int(match.group(1)) if match else None


def extract_page_range(text: str) -> str | None:
    """Return a string like '5-8' showing all page markers present in a text block."""
    pages = [int(m.group(1)) for m in PAGE_MARKER_RE.finditer(text)]
    if not pages:
        return None
    if len(pages) == 1:
        return str(pages[0])
    return f"{pages[0]}-{pages[-1]}"


def split_sections(text: str) -> list[str]:
    """Split text into sections using structural heading markers.

    Page markers (--- [Page N] ---) are kept inline as content for page tracking,
    but are NOT used as split points. Only headings drive the section boundaries.
    Falls back to paragraph splitting if no heading markers are found.
    """
    split_positions: set[int] = set()

    for pattern in HEADING_PATTERNS:
        for match in pattern.finditer(text):
            # Skip heading matches that fall inside a page marker line
            line_start = text.rfind("\n", 0, match.start()) + 1
            line = text[line_start: text.find("\n", match.start())].strip()
            if PAGE_MARKER_RE.match(line):
                continue
            split_positions.add(match.start())

    if not split_positions:
        # No heading markers found — fall back to paragraph splitting
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    positions = sorted(split_positions)
    sections: list[str] = []

    # Include any text before the first heading
    if positions[0] > 0:
        preamble = text[: positions[0]].strip()
        if preamble:
            sections.append(preamble)

    for i, pos in enumerate(positions):
        start = pos
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)

    return sections


def split_large_section(section: str, max_tokens: int) -> list[str]:
    """Split an oversized section into smaller pieces.

    First tries paragraph-level splitting within the section.
    Falls back to token-window splitting if paragraphs are still too large.
    """
    if count_tokens(section) <= max_tokens:
        return [section]

    # Try paragraph-level splitting first
    paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        # Re-group paragraphs into pieces that fit within max_tokens
        pieces: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = count_tokens(para)
            if current_parts and current_tokens + para_tokens > max_tokens:
                pieces.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0
            current_parts.append(para)
            current_tokens += para_tokens

        if current_parts:
            pieces.append("\n\n".join(current_parts))

        # Check if any piece is still too large and needs token-window splitting
        result: list[str] = []
        for piece in pieces:
            if count_tokens(piece) <= max_tokens:
                result.append(piece)
            else:
                result.extend(_split_by_token_window(piece, max_tokens))
        return result

    # Single large paragraph — split by token window
    return _split_by_token_window(section, max_tokens)


def _split_by_token_window(text: str, max_tokens: int) -> list[str]:
    """Split text into overlapping token windows."""
    tokens = ENCODING.encode(text)
    overlap = min(max_tokens // 4, settings.CHUNK_OVERLAP_TOKENS)
    step = max(1, max_tokens - overlap)
    parts: list[str] = []
    for start in range(0, len(tokens), step):
        piece = ENCODING.decode(tokens[start: start + max_tokens]).strip()
        if piece:
            parts.append(piece)
        if start + max_tokens >= len(tokens):
            break
    return parts


def normalize_sections(sections: list[str], max_tokens: int) -> list[str]:
    """Split any oversized sections so all fit within the token limit."""
    normalized: list[str] = []
    for section in sections:
        normalized.extend(split_large_section(section, max_tokens))
    return normalized


def chunk_token_limit() -> int:
    """Return the effective chunk token limit after subtracting the safety margin."""
    return max(1, settings.CHUNK_SIZE_TOKENS - TOKEN_SAFETY_MARGIN)


def build_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Generate a deterministic UUID for a chunk based on document ID and index."""
    return str(uuid5(NAMESPACE_URL, f"{doc_id}#chunk={chunk_index}"))


def chunk_document(doc: RawDocument) -> list[DocumentChunk]:
    """Split a document into token-limited chunks with overlap and page tracking."""
    max_chunk_tokens = chunk_token_limit()
    sections = normalize_sections(
        split_sections(doc.text),
        max_tokens=max_chunk_tokens,
    )

    chunks: list[DocumentChunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    chunk_index = 0

    for section in sections:
        section_tokens = count_tokens(section)

        if current_parts and current_tokens + section_tokens > max_chunk_tokens:
            chunk_text = "\n\n".join(current_parts).strip()
            page_range = extract_page_range(chunk_text)
            first_page = extract_page_number(chunk_text)

            chunks.append(
                DocumentChunk(
                    chunk_id=build_chunk_id(doc.doc_id, chunk_index),
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        **{k: v for k, v in doc.metadata.model_dump().items() if k != "page_number"},
                        doc_id=doc.doc_id,
                        chunk_index=chunk_index,
                        page_number=first_page,
                        page_range=page_range,
                    ),
                )
            )
            chunk_index += 1

            overlap_parts = []
            overlap_tokens = 0

            for part in reversed(current_parts):
                part_tokens = count_tokens(part)
                if overlap_tokens + part_tokens > settings.CHUNK_OVERLAP_TOKENS:
                    break
                overlap_parts.insert(0, part)
                overlap_tokens += part_tokens

            if overlap_tokens + section_tokens > max_chunk_tokens:
                overlap_parts = []
                overlap_tokens = 0

            current_parts = overlap_parts
            current_tokens = overlap_tokens

        current_parts.append(section)
        current_tokens += section_tokens

    if current_parts:
        chunk_text = "\n\n".join(current_parts).strip()
        page_range = extract_page_range(chunk_text)
        first_page = extract_page_number(chunk_text)

        chunks.append(
            DocumentChunk(
                chunk_id=build_chunk_id(doc.doc_id, chunk_index),
                text=chunk_text,
                metadata=ChunkMetadata(
                    **{k: v for k, v in doc.metadata.model_dump().items() if k != "page_number"},
                    doc_id=doc.doc_id,
                    chunk_index=chunk_index,
                    page_number=first_page,
                    page_range=page_range,
                ),
            )
        )

    return chunks


def chunk_documents(docs: list[RawDocument]) -> list[DocumentChunk]:
    """Chunk all documents and return a flat list of DocumentChunk objects."""
    all_chunks: list[DocumentChunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
