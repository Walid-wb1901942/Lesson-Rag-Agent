import argparse
import json
from collections import defaultdict
from pathlib import Path

from app.ingestion.index_docs import index_chunks, index_pdf, reindex_pdf
from app.ingestion.load_docs import load_documents_from_folder
from app.ingestion.clean_docs import clean_documents
from app.ingestion.chunk_docs import chunk_documents


RAW_DIR = "data/raw"
PROCESSED_DIR = Path("data/processed")
DEBUG_DIR = PROCESSED_DIR / "chunks_debug"


def serialize_chunk(chunk) -> dict:
    """Convert a DocumentChunk to a JSON-serializable dict for debug output."""
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "metadata": chunk.metadata.model_dump(),
    }


def build_debug_file_path(source_path: str) -> Path:
    """Build the debug output path for a source document's chunks."""
    relative_source = Path(source_path)
    return (DEBUG_DIR / relative_source).with_suffix(".chunks.json")


def write_debug_chunks(chunks) -> None:
    """Write per-document debug JSON files with chunk content and metadata."""
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    chunks_by_source: dict[str, list] = defaultdict(list)
    for chunk in chunks:
        chunks_by_source[chunk.metadata.source_path].append(chunk)

    for source_path, source_chunks in chunks_by_source.items():
        debug_file = build_debug_file_path(source_path)
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        with debug_file.open("w", encoding="utf-8") as f:
            json.dump(
                [serialize_chunk(chunk) for chunk in source_chunks],
                f,
                ensure_ascii=False,
                indent=2,
            )


def run_full_ingestion() -> None:
    """Run the complete ingestion pipeline: load, clean, chunk, index all PDFs."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Loading PDFs...")
    raw_docs = load_documents_from_folder(RAW_DIR)
    print(f"Loaded {len(raw_docs)} documents (one per PDF).")

    print("2. Cleaning text...")
    cleaned_docs = clean_documents(raw_docs)
    print(f"Cleaned {len(cleaned_docs)} documents.")

    print("3. Chunking text...")
    chunks = chunk_documents(cleaned_docs)
    print(f"Created {len(chunks)} chunks.")

    write_debug_chunks(chunks)
    print(f"Wrote debug chunk files to {DEBUG_DIR}")

    print("4. Indexing into Qdrant...")
    index_chunks(chunks)

    print("Done.")


def run_single_pdf(pdf_path: str, reindex: bool) -> None:
    """Index or reindex a single PDF file."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    action = "Reindexing" if reindex else "Indexing"
    print(f"{action} single PDF: {pdf_path}")
    chunks = reindex_pdf(pdf_path, metadata_folder=RAW_DIR) if reindex else index_pdf(
        pdf_path, metadata_folder=RAW_DIR
    )
    print(f"Created {len(chunks)} chunks.")
    write_debug_chunks(chunks)
    print(f"Wrote debug chunk files to {DEBUG_DIR}")
    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the ingestion script."""
    parser = argparse.ArgumentParser(description="Run full or single-file ingestion.")
    parser.add_argument(
        "--pdf",
        help="Optional path to a single PDF to index into the existing collection.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Delete existing chunks for the given PDF source_path before indexing it again.",
    )
    return parser


def main():
    """Entry point for the ingestion CLI."""
    args = build_parser().parse_args()
    if args.pdf:
        run_single_pdf(args.pdf, reindex=args.reindex)
        return

    run_full_ingestion()


if __name__ == "__main__":
    main()
