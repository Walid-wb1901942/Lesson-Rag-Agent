import re
from app.schemas import RawDocument


def clean_math_notation(text: str) -> str:
    """Fix common math artifacts from PDF text extraction."""

    # Integral sign: standalone "Z" used as integral symbol in LaTeX PDFs.
    # Match Z that appears at line start or after whitespace, followed by
    # typical integral patterns (limits, function expressions, dx).
    # First handle "Z b\na" style (definite integral with limits on separate lines)
    text = re.sub(r'\bZ\s+(\d+)\s*\n\s*(\d+)\s*\n', r'integral from \2 to \1 of ', text)
    text = re.sub(r'\bZ\s+(\S+)\s*\n\s*(\S+)\s*\n', r'integral from \2 to \1 of ', text)
    # "Z\n" at start of expression (indefinite integral)
    text = re.sub(r'\bZ\n', 'integral of ', text)
    # Remaining standalone Z before dx patterns
    text = re.sub(r'\bZ\s+(?=\()', 'integral of ', text)
    text = re.sub(r'\bZ\s+(?=\[)', 'integral of ', text)
    text = re.sub(r'\bZ\s+(?=\w+dx)', 'integral of ', text)

    # Not-equal sign
    text = text.replace("̸=", " != ")
    text = text.replace("̸= ", " != ")

    # Common Unicode math symbols to readable forms
    text = text.replace("∴", "therefore")
    text = text.replace("∵", "because")

    # Superscript digits (e.g. x² x³) - keep these, they're readable
    # But fix "x2" patterns that should be "x^2" when followed by operators/spaces
    # This is too ambiguous to auto-fix reliably, so we leave it for the prompt

    return text


def clean_text(text: str) -> str:
    """Normalize whitespace, remove page numbers, and fix math notation artifacts."""
    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove standalone page numbers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove common page labels
    text = re.sub(r"Page\s+\d+\s*(of\s+\d+)?", "", text, flags=re.IGNORECASE)

    # Fix common math extraction artifacts
    text = clean_math_notation(text)

    # Remove repeated blank lines again after cleanup
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_mostly_junk(text: str) -> bool:
    """Check if text is too short or looks like a table of contents / index page."""
    stripped = text.strip()
    lower = stripped.lower()

    # Skip very small useless documents
    if len(stripped) < 40:
        return True

    # The following checks only apply to short texts (single-page level).
    # For merged whole-document texts these patterns will appear as part of
    # legitimate content and should not disqualify the entire document.
    if len(stripped.splitlines()) < 25:
        if "contents" in lower:
            return True
        if lower.startswith("index") or "\nindex\n" in f"\n{lower}\n":
            return True

    return False


def clean_documents(docs: list[RawDocument]) -> list[RawDocument]:
    """Clean all documents, removing junk pages and normalizing text."""
    cleaned: list[RawDocument] = []

    for doc in docs:
        cleaned_text = clean_text(doc.text)
        if cleaned_text and not is_mostly_junk(cleaned_text):
            cleaned.append(
                RawDocument(
                    doc_id=doc.doc_id,
                    text=cleaned_text,
                    metadata=doc.metadata,
                )
            )

    return cleaned