from pathlib import Path
import json
from uuid import NAMESPACE_URL, uuid5

import fitz  # PyMuPDF

from app.schemas import RawDocument, DocumentMetadata

PAGE_MARKER_TEMPLATE = "\n\n--- [Page {page_number}] ---\n\n"


def load_metadata_manifest(folder: str) -> dict:
    """Load the metadata.json manifest from the given folder."""
    manifest_path = Path(folder) / "metadata.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def normalize_source_path(path: Path, source_root: Path | None = None) -> str:
    """Convert a file path to a relative string for use as a source identifier."""
    if source_root is not None:
        try:
            return str(path.relative_to(source_root))
        except ValueError:
            pass
    return str(path)


def build_doc_id(source_path: str) -> str:
    """Generate a deterministic UUID for a document based on its source path."""
    return str(uuid5(NAMESPACE_URL, source_path))


def load_pdf(path: Path, manifest: dict, source_root: Path | None = None) -> list[RawDocument]:
    """Load a single PDF, merge all pages, and return as a RawDocument."""
    file_meta = manifest.get(path.name, {})
    source_path = normalize_source_path(path, source_root=source_root)

    pdf = fitz.open(path)
    page_texts: list[str] = []
    for page_index in range(len(pdf)):
        page = pdf[page_index]
        text = page.get_text("text")
        page_number = page_index + 1

        if text and text.strip():
            marker = PAGE_MARKER_TEMPLATE.format(page_number=page_number)
            page_texts.append(f"{marker}{text}")
    pdf.close()

    if not page_texts:
        return []

    merged_text = "".join(page_texts).strip()

    return [
        RawDocument(
            doc_id=build_doc_id(source_path),
            text=merged_text,
            metadata=DocumentMetadata(
                source_path=source_path,
                file_name=path.name,
                file_type="pdf",
                title=file_meta.get("title", path.stem),
                subject=file_meta.get("subject"),
                grade_level=file_meta.get("grade_level"),
                topic=file_meta.get("topic"),
                source_type=file_meta.get("source_type", "pdf"),
                language=file_meta.get("language", "en"),
                page_number=None,
            ),
        )
    ]


def load_documents_from_folder(folder: str) -> list[RawDocument]:
    """Load all PDFs from a folder and return them as RawDocument objects."""
    folder_path = Path(folder)
    all_docs: list[RawDocument] = []
    manifest = load_metadata_manifest(folder)

    for path in folder_path.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".pdf":
            all_docs.extend(load_pdf(path, manifest, source_root=folder_path))

    return all_docs


def load_documents_from_pdf(path: str, metadata_folder: str = "data/raw") -> list[RawDocument]:
    """Load a single PDF with metadata from the manifest."""
    pdf_path = Path(path)
    metadata_root = Path(metadata_folder)
    manifest = load_metadata_manifest(str(metadata_root))

    return load_pdf(pdf_path, manifest, source_root=metadata_root)
