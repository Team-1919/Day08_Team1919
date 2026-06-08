"""Task 5 - Semantic search over the local chunk index."""

from __future__ import annotations

import math

from .task4_chunking_indexing import embed_text, get_chunks


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Return top chunks by cosine similarity to the query embedding."""
    if top_k <= 0:
        return []

    query_embedding = embed_text(query)
    results: list[dict] = []
    for chunk in get_chunks():
        embedding = chunk.get("embedding") or embed_text(chunk["content"])
        score = _cosine(query_embedding, embedding)
        if score <= 0:
            continue
        results.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    for result in semantic_search("hinh phat ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
