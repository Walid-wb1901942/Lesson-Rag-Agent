from app.services.retriever import retrieve_chunks


def format_metadata_value(value):
    if isinstance(value, list):
        return ", ".join(value)
    return value


def search(
    query: str,
    limit: int = 5,
    subject: str | None = None,
    grade_level: str | None = None,
    topic: str | None = None,
    use_topic_filter: bool = False,
):
    results = retrieve_chunks(
        query=query,
        limit=limit,
        subject=subject,
        grade_level=grade_level,
        topic=topic,
        use_topic_filter=use_topic_filter,
    )

    if not results:
        print("No retrieval results.")
        return

    for i, chunk in enumerate(results, start=1):
        metadata = chunk["metadata"]
        print("=" * 80)
        print(f"Result {i}")
        print("Score:", chunk["score"])
        print("Title:", metadata.get("title"))
        print("Page:", metadata.get("page_number"))
        print("Subject:", metadata.get("subject"))
        print("Grade:", format_metadata_value(metadata.get("grade_level")))
        print("Topic:", metadata.get("topic"))
        print("Source path:", metadata.get("source_path"))
        print("Text preview:")
        print(chunk["text"][:800])
        print()


if __name__ == "__main__":
    search(
        query="Find the areas A1, A2 and A3 in a mathematics lesson",
        limit=5,
        subject="mathematics",
        grade_level="12",
        topic="integral calculus",
    )
