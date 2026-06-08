"""
Task 7 — Reranking Module.

Hỗ trợ:
    - cross_encoder: local CrossEncoder (multilingual), fallback keyword scoring
    - mmr: Maximal Marginal Relevance
    - rrf: Reciprocal Rank Fusion
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        )
    return _cross_encoder


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """Rerank bằng Jina API (nếu có key) hoặc local CrossEncoder."""
    jina_key = os.getenv("JINA_API_KEY", "")
    if jina_key and "xxx" not in jina_key:
        try:
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {jina_key}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": top_k,
                },
                timeout=30,
            )
            response.raise_for_status()
            reranked = response.json()["results"]
            return [
                {**candidates[r["index"]], "score": float(r["relevance_score"])}
                for r in reranked
            ]
        except Exception:
            pass

    if not candidates:
        return []

    try:
        model = _get_cross_encoder()
        pairs = [[query, c["content"]] for c in candidates]
        scores = model.predict(pairs)
        scored = []
        for candidate, score in zip(candidates, scores):
            item = candidate.copy()
            item["score"] = float(score)
            scored.append(item)
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]
    except Exception:
        return _rerank_keyword_fallback(query, candidates, top_k)


def _rerank_keyword_fallback(
    query: str, candidates: list[dict], top_k: int
) -> list[dict]:
    query_terms = set(query.lower().split())
    scored = []
    for candidate in candidates:
        content_terms = set(candidate["content"].lower().split())
        overlap = len(query_terms & content_terms)
        score = overlap / max(len(query_terms), 1) + candidate.get("score", 0) * 0.1
        item = candidate.copy()
        item["score"] = score
        scored.append(item)
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Maximal Marginal Relevance — relevance + diversity."""
    selected_indices = []
    remaining = list(range(len(candidates)))
    results = []

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            relevance = _cosine_sim(
                query_embedding, candidates[idx].get("embedding", [])
            )
            max_sim_to_selected = 0.0
            for sel_idx in selected_indices:
                sim = _cosine_sim(
                    candidates[idx].get("embedding", []),
                    candidates[sel_idx].get("embedding", []),
                )
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * max_sim_to_selected
            )
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining.remove(best_idx)
        item = candidates[best_idx].copy()
        item["score"] = best_score
        results.append(item)

    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """Reciprocal Rank Fusion — gộp nhiều ranked lists."""
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """Unified reranking interface."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        from src.task5_semantic_search import embed_text

        query_embedding = embed_text(query)
        return rerank_mmr(query_embedding, candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
