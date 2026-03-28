from pathlib import Path

from qdrant_client.models import PointStruct

from app.schemas import DocumentChunk
from app.ingestion.chunk_docs import chunk_documents
from app.ingestion.clean_docs import clean_documents
from app.ingestion.load_docs import load_documents_from_pdf, normalize_source_path
from app.services.ollama_client import embed_texts
from app.services.qdrant_client import (
    delete_chunks_by_source_path,
    ensure_collection,
    get_qdrant_client,
)
from app.config import settings


def index_chunks(chunks: list[DocumentChunk]) -> None:
    """Embed and upsert document chunks into Qdrant in batches."""
    if not chunks:
        print("No chunks to index.")
        return

    client = get_qdrant_client()
    upsert_batch_size = max(1, settings.QDRANT_UPSERT_BATCH_SIZE)
    collection_ready = False
    indexed_count = 0

    for start in range(0, len(chunks), upsert_batch_size):
        chunk_batch = chunks[start : start + upsert_batch_size]
        texts = ["search_document: " + chunk.text for chunk in chunk_batch]
        vectors = embed_texts(texts)

        if not collection_ready:
            ensure_collection(len(vectors[0]))
            collection_ready = True

        points = []
        for chunk, vector in zip(chunk_batch, vectors):
            payload = chunk.metadata.model_dump()
            payload["text"] = chunk.text
            payload["chunk_id"] = chunk.chunk_id

            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
        )
        indexed_count += len(points)

    print(f"Indexed {indexed_count} chunks.")


def index_pdf(path: str, metadata_folder: str = "data/raw") -> list[DocumentChunk]:
    """Load, clean, chunk, and index a single PDF into Qdrant."""
    raw_docs = load_documents_from_pdf(path, metadata_folder=metadata_folder)
    cleaned_docs = clean_documents(raw_docs)
    chunks = chunk_documents(cleaned_docs)
    index_chunks(chunks)
    return chunks


def reindex_pdf(path: str, metadata_folder: str = "data/raw") -> list[DocumentChunk]:
    """Delete old chunks for a PDF and re-index it from scratch."""
    metadata_root = Path(metadata_folder)
    source_path = normalize_source_path(Path(path), source_root=metadata_root)
    delete_chunks_by_source_path(source_path)
    return index_pdf(path, metadata_folder=metadata_folder)
