from app.ingestion.index_docs import index_chunks
from app.schemas import ChunkMetadata, DocumentChunk


def main():
    chunks = [
        DocumentChunk(
            chunk_id="6ea48630-2be2-460f-8e19-3d57a2ba33e1",
            text="Photosynthesis is the process by which green plants make food using sunlight, carbon dioxide, and water.",
            metadata=ChunkMetadata(
                source_path="tests/manual/grade6_science_notes.pdf",
                file_name="grade6_science_notes.pdf",
                file_type="pdf",
                title="Grade 6 Science Notes",
                subject="science",
                grade_level="6",
                topic="photosynthesis",
                source_type="notes",
                language="en",
                page_number=1,
                doc_id="4300902e-26db-4a26-8eec-df1c362f8c52",
                chunk_index=0,
            ),
        ),
        DocumentChunk(
            chunk_id="30a46774-238f-4d67-ae86-cc89d634e596",
            text="A food chain shows how energy passes from one organism to another in an ecosystem.",
            metadata=ChunkMetadata(
                source_path="tests/manual/grade6_science_notes.pdf",
                file_name="grade6_science_notes.pdf",
                file_type="pdf",
                title="Grade 6 Science Notes",
                subject="science",
                grade_level=["6", "7"],
                topic="ecosystems",
                source_type="notes",
                language="en",
                page_number=2,
                doc_id="ea716706-8ec5-42b0-8765-c3564c46ebc6",
                chunk_index=0,
            ),
        ),
        DocumentChunk(
            chunk_id="2ef85b9f-d6e4-4b66-a1a2-f15a0e23e0dd",
            text="Evaporation is when liquid water changes into water vapor due to heat.",
            metadata=ChunkMetadata(
                source_path="tests/manual/grade5_science_notes.pdf",
                file_name="grade5_science_notes.pdf",
                file_type="pdf",
                title="Grade 5 Science Notes",
                subject="science",
                grade_level="5",
                topic="water cycle",
                source_type="notes",
                language="en",
                page_number=1,
                doc_id="5c4ef1d6-4baa-4724-907f-25096ba4a8ee",
                chunk_index=0,
            ),
        ),
    ]

    index_chunks(chunks)


if __name__ == "__main__":
    main()
