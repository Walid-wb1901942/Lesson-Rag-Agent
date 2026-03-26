from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.services.ollama_client import embed_texts
from app.services.qdrant_client import get_qdrant_client
from app.config import settings


def retrieve_chunks(
    query: str,
    limit: int = 5,
    subject: str | None = None,
    grade_level: str | None = None,
    curriculum: str | None = None,
    topic: str | None = None,
    use_topic_filter: bool = False,
    retrieval_mode: str = "filtered",
) -> list[dict]:
    """Embed a query and search Qdrant for matching document chunks."""
    query_text = f"{query}\nTopic focus: {topic}" if topic else query
    query_vector = embed_texts([query_text])[0]
    client = get_qdrant_client()

    conditions = []
    use_filters = retrieval_mode != "all"

    if use_filters and subject:
        conditions.append(FieldCondition(key="subject", match=MatchValue(value=subject)))
    if use_filters and grade_level:
        conditions.append(FieldCondition(key="grade_level", match=MatchValue(value=grade_level)))
    if use_filters and curriculum:
        conditions.append(FieldCondition(key="curriculum", match=MatchValue(value=curriculum)))
    if use_filters and topic and use_topic_filter:
        conditions.append(FieldCondition(key="topic", match=MatchValue(value=topic)))

    query_filter = Filter(must=conditions) if conditions else None

    results = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
    )

    retrieved = []
    for point in results.points:
        payload = point.payload
        retrieved.append(
            {
                "score": point.score,
                "text": payload.get("text", ""),
                "metadata": {
                    "title": payload.get("title"),
                    "page_number": payload.get("page_number"),
                    "page_range": payload.get("page_range"),
                    "subject": payload.get("subject"),
                    "grade_level": payload.get("grade_level"),
                    "curriculum": payload.get("curriculum"),
                    "topic": payload.get("topic"),
                    "source_type": payload.get("source_type"),
                    "source_path": payload.get("source_path"),
                },
            }
        )

    return retrieved
