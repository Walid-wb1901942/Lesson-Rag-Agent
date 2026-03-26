from pathlib import Path

from app.ingestion.index_docs import reindex_pdf


def main():
    pdf_path = Path("data/raw/c2integral.pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"Expected sample PDF at {pdf_path}")

    chunks = reindex_pdf(str(pdf_path))

    print("=" * 100)
    print("REINDEXED PDF")
    print("=" * 100)
    print("Path:", pdf_path)
    print("Chunks indexed:", len(chunks))
    if chunks:
        print("First chunk id:", chunks[0].chunk_id)
        print("First chunk source:", chunks[0].metadata.source_path)


if __name__ == "__main__":
    main()
