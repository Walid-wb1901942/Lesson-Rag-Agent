from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)
from app.config import settings

FILTERABLE_PAYLOAD_FIELDS = (
    "subject",
    "grade_level",
    "topic",
    "language",
    "source_type",
    "file_name",
    "source_path",
    "doc_id",
    "page_range",
)


_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Return a singleton Qdrant client instance."""
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
    return _client


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Check whether a Qdrant collection exists."""
    try:
        collections = client.get_collections().collections
        names = {c.name for c in collections}
        return collection_name in names
    except Exception:
        return False


def ensure_payload_indexes(client: QdrantClient) -> None:
    """Create keyword indexes on all filterable payload fields."""
    for field_name in FILTERABLE_PAYLOAD_FIELDS:
        client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
        )


def ensure_collection(vector_size: int) -> None:
    """Create the Qdrant collection if it doesn't exist, then ensure indexes."""
    client = get_qdrant_client()
    if not collection_exists(client, settings.QDRANT_COLLECTION):
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    ensure_payload_indexes(client)


def delete_chunks_by_source_path(source_path: str) -> None:
    """Delete all indexed chunks for a given source PDF path."""
    client = get_qdrant_client()
    if not collection_exists(client, settings.QDRANT_COLLECTION):
        return

    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION,
        field_name="source_path",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    source_filter = Filter(
        must=[FieldCondition(key="source_path", match=MatchValue(value=source_path))]
    )
    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=source_filter,
        wait=True,
    )


def collection_has_points() -> bool:
    """Check if the collection contains at least one indexed point."""
    client = get_qdrant_client()
    if not collection_exists(client, settings.QDRANT_COLLECTION):
        return False

    points, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return bool(points)


def field_value_exists(field_name: str, value: str) -> bool:
    """Check if a specific value exists for a payload field in the collection."""
    client = get_qdrant_client()
    if not collection_exists(client, settings.QDRANT_COLLECTION):
        return False

    points, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key=field_name, match=MatchValue(value=value))]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return bool(points)


def filter_has_matches(
    subject: str | None = None,
    grade_level: str | None = None,
) -> bool:
    """Check if any indexed documents match the given metadata filter combination."""
    client = get_qdrant_client()
    if not collection_exists(client, settings.QDRANT_COLLECTION):
        return False

    conditions = []
    if subject:
        conditions.append(FieldCondition(key="subject", match=MatchValue(value=subject)))
    if grade_level:
        conditions.append(FieldCondition(key="grade_level", match=MatchValue(value=grade_level)))

    points, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=Filter(must=conditions) if conditions else None,
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return bool(points)


def list_payload_values(field_name: str, max_values: int = 20) -> list[str]:
    """List unique values for a payload field across all indexed documents."""
    client = get_qdrant_client()
    if not collection_exists(client, settings.QDRANT_COLLECTION):
        return []

    values: set[str] = set()
    offset = None

    while len(values) < max_values:
        points, offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            offset=offset,
            limit=256,
            with_payload=[field_name],
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            payload_value = point.payload.get(field_name)
            if isinstance(payload_value, list):
                for item in payload_value:
                    if item is not None:
                        values.add(str(item))
            elif payload_value is not None:
                values.add(str(payload_value))

            if len(values) >= max_values:
                break

        if offset is None:
            break

    return sorted(values)
