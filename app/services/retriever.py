from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.services.ollama_client import embed_texts
from app.services.qdrant_client import get_qdrant_client
from app.config import settings


_QUERY_PREFIX = "search_query: "  # nomic-embed-text task prefix for queries


def _point_to_chunk(point, score: float) -> dict:
    payload = point.payload or {}
    return {
        "score": score,
        "text": payload.get("text", ""),
        "metadata": {
            "title": payload.get("title"),
            "page_number": payload.get("page_number"),
            "page_range": payload.get("page_range"),
            "subject": payload.get("subject"),
            "grade_level": payload.get("grade_level"),
            "topic": payload.get("topic"),
            "source_type": payload.get("source_type"),
            "source_path": payload.get("source_path"),
        },
    }


def retrieve_chunks(
    query: str,
    limit: int = 5,
    subject: str | None = None,
    grade_level: str | None = None,
    topic: str | None = None,
    use_topic_filter: bool = False,
    retrieval_mode: str = "filtered",
    retrieval_method: str = "dense",
) -> list[dict]:
    """Embed a query and search Qdrant for matching document chunks.

    retrieval_method:
      "dense"  — cosine similarity search only (default)
      "hybrid" — dense + BM25, merged with Reciprocal Rank Fusion (RRF)
    """
    query_text = f"{query}\nTopic focus: {topic}" if topic else query
    query_vector = embed_texts([_QUERY_PREFIX + query_text])[0]
    client = get_qdrant_client()

    conditions = []
    use_filters = retrieval_mode != "all"

    if use_filters and subject:
        conditions.append(FieldCondition(key="subject", match=MatchValue(value=subject)))
    if use_filters and grade_level:
        conditions.append(FieldCondition(key="grade_level", match=MatchValue(value=grade_level)))
    if use_filters and topic and use_topic_filter:
        conditions.append(FieldCondition(key="topic", match=MatchValue(value=topic)))

    query_filter = Filter(must=conditions) if conditions else None

    if retrieval_method == "hybrid":
        return _hybrid_retrieve(
            query=query,
            query_vector=query_vector,
            query_filter=query_filter,
            subject=subject if use_filters else None,
            grade_level=grade_level if use_filters else None,
            limit=limit,
            client=client,
        )

    # Dense-only retrieval
    results = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
    )

    return [_point_to_chunk(point, point.score) for point in results.points]


def _hybrid_retrieve(
    query: str,
    query_vector: list[float],
    query_filter,
    subject: str | None,
    grade_level: str | None,
    limit: int,
    client,
) -> list[dict]:
    """Hybrid dense + BM25 retrieval merged with Reciprocal Rank Fusion (k=60).

    Fetches `limit * 3` candidates from each method so the merged top-`limit`
    results are drawn from a wider pool.

    BM25-only candidates are filtered by subject/grade_level to prevent
    unrelated subject documents from entering the pool via keyword matching.
    """
    from app.services.bm25_index import get_bm25_index

    fetch_k = limit * 3

    # 1. Dense retrieval — broader candidate set (Qdrant filter applied here)
    dense_results = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        limit=fetch_k,
        query_filter=query_filter,
    )

    # Build id -> chunk dict from dense results
    id_to_chunk: dict[str, dict] = {}
    dense_ids: list[str] = []
    for point in dense_results.points:
        chunk_id = str(point.id)
        dense_ids.append(chunk_id)
        id_to_chunk[chunk_id] = _point_to_chunk(point, point.score)

    # 2. BM25 retrieval — searches full corpus, filter applied below
    bm25_index = get_bm25_index()
    bm25_hits = bm25_index.search(query, top_k=fetch_k)
    bm25_ids = [cid for cid, _ in bm25_hits]

    # 3. Fetch payloads for BM25-only hits and apply subject/grade filter
    bm25_only_ids = [cid for cid in bm25_ids if cid not in id_to_chunk]
    if bm25_only_ids:
        fetched = client.retrieve(
            collection_name=settings.QDRANT_COLLECTION,
            ids=bm25_only_ids,
            with_payload=True,
            with_vectors=False,
        )
        for point in fetched:
            chunk_id = str(point.id)
            payload = point.payload or {}
            # Skip BM25-only hits that don't match the subject/grade filter
            if subject and payload.get("subject") != subject:
                continue
            if grade_level:
                stored_grade = payload.get("grade_level")
                # grade_level may be stored as a list or a string
                if isinstance(stored_grade, list):
                    if grade_level not in stored_grade:
                        continue
                elif stored_grade != grade_level:
                    continue
            id_to_chunk[chunk_id] = _point_to_chunk(point, 0.0)

    # 4. Reciprocal Rank Fusion
    K = 60
    rrf: dict[str, float] = {}
    for rank, chunk_id in enumerate(dense_ids):
        rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (K + rank + 1)
    for rank, chunk_id in enumerate(bm25_ids):
        rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (K + rank + 1)

    # 5. Sort by RRF score, return top `limit`
    top_ids = sorted(rrf, key=lambda cid: rrf[cid], reverse=True)[:limit]
    result = []
    for chunk_id in top_ids:
        if chunk_id in id_to_chunk:
            chunk = dict(id_to_chunk[chunk_id])
            chunk["score"] = rrf[chunk_id]
            result.append(chunk)

    return result
