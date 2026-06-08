"""Task 7 - Local reranking methods."""

from __future__ import annotations

import math

from .task4_chunking_indexing import embed_text, tokenize


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Lightweight cross-encoder-style reranker.

    It re-scores each candidate by query term coverage plus the original retrieval
    score, giving a deterministic offline substitute for API rerankers.
    """
    query_terms = set(tokenize(query))
    if not query_terms:
        return candidates[:top_k]

    reranked: list[dict] = []
    for item in candidates:
        doc_terms = set(tokenize(item.get("content", "")))
        coverage = len(query_terms & doc_terms) / len(query_terms)
        density = len(query_terms & doc_terms) / (len(doc_terms) or 1)
        original = float(item.get("score", 0.0))
        score = 0.75 * coverage + 0.15 * density + 0.10 * _normalize_score(original)
        reranked.append({**item, "score": float(score)})

    reranked.sort(key=lambda result: result["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Select candidates that balance relevance and diversity."""
    if not candidates:
        return []

    vectors = [item.get("embedding") or embed_text(item.get("content", "")) for item in candidates]
    selected: list[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_idx = None
        best_score = float("-inf")
        for idx in remaining:
            relevance = _cosine(query_embedding, vectors[idx])
            diversity_penalty = max((_cosine(vectors[idx], vectors[j]) for j in selected), default=0.0)
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_idx = idx
                best_score = mmr_score
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [{**candidates[idx], "score": float(candidates[idx].get("score", 0.0))} for idx in selected]


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Merge ranked lists with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = _key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items[key] = item

    merged = []
    for key, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True):
        merged.append({**items[key], "score": float(score)})
        if len(merged) >= top_k:
            break
    return merged


def rerank(query: str, candidates: list[dict], top_k: int = 5, method: str = "cross_encoder") -> list[dict]:
    """Unified reranking interface."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        return rerank_mmr(embed_text(query), candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k)
    raise ValueError(f"Unknown rerank method: {method}")


def _key(item: dict) -> str:
    metadata = item.get("metadata", {})
    return f"{metadata.get('source', '')}:{metadata.get('chunk_index', '')}:{item.get('content', '')[:80]}"


def _normalize_score(score: float) -> float:
    return score / (abs(score) + 1.0)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Toi tang tru ma tuy", "score": 0.8, "metadata": {}},
        {"content": "Nghe si bi bat vi ma tuy", "score": 0.6, "metadata": {}},
    ]
    for result in rerank("hinh phat ma tuy", dummy_candidates, top_k=2):
        print(f"[{result['score']:.3f}] {result['content']}")
