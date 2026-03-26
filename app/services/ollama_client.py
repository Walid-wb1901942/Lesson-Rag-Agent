import re

import requests

from app.config import settings


def _raise_with_response_details(response: requests.Response) -> None:
    """Raise an HTTPError with the response body included in the message."""
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text.strip()
        detail = f"{exc}"
        if body:
            detail = f"{detail}\nOllama response body: {body}"
        raise requests.HTTPError(detail, response=response) from exc


def generate_text(
    prompt: str,
    num_predict: int = 4096,
    num_ctx: int | None = None,
) -> str:
    """Generate text from a prompt using Ollama with auto-sized context window."""
    # Auto-size context window: prompt tokens + output headroom.
    # Conservative estimate: 1 token ≈ 3 chars. Add buffer for safety.
    if num_ctx is None:
        estimated_prompt_tokens = len(prompt) // 2 + 512
        num_ctx = min(estimated_prompt_tokens + num_predict, 32768)

    response = requests.post(
        f"{settings.OLLAMA_BASE_URL}/generate",
        json={
            "model": settings.OLLAMA_GENERATION_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
        },
        timeout=600,
    )
    _raise_with_response_details(response)
    data = response.json()
    text = data["response"]
    # Strip <think>...</think> tags from models like qwen3 that use thinking mode
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using Ollama's /embed endpoint."""
    response = requests.post(
        f"{settings.OLLAMA_BASE_URL}/embed",
        json={
            "model": settings.OLLAMA_EMBEDDING_MODEL,
            "input": texts,
        },
        timeout=300,
    )
    _raise_with_response_details(response)
    data = response.json()
    return data["embeddings"]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in batches, returning one embedding vector per text."""
    if not texts:
        return []

    batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        all_embeddings.extend(_embed_batch(batch))

    return all_embeddings
