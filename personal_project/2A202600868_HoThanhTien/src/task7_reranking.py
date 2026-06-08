"""
Task 7 — Reranking Module.

3 phương pháp:
- rerank_mmr: Maximal Marginal Relevance - tối ưu relevance + diversity
- rerank_rrf: Reciprocal Rank Fusion - gộp nhiều rankers
- rerank_cross_encoder: mock dựa trên query-candidate cosine sim
  (vì không có Jina API key / không down model 500MB+ trong test)

MMR và RRF hoạt động pure local. Cross-encoder là approximation.
"""

import sys
from typing import Optional

# Fix Windows encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np


# =============================================================================
# MMR — Maximal Marginal Relevance
# =============================================================================

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity giữa 2 vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content', 'score', 'embedding', 'metadata'}
                    'embedding' bắt buộc cho MMR.
        top_k: Số kết quả
        lambda_param: 1.0 = pure relevance, 0.0 = pure diversity

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    q_emb = np.array(query_embedding, dtype=np.float32)
    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            cand_emb = np.array(candidates[idx].get("embedding", []), dtype=np.float32)
            if cand_emb.size == 0:
                # Nếu thiếu embedding, dùng score gốc
                relevance = float(candidates[idx].get("score", 0))
            else:
                relevance = _cosine_sim(q_emb, cand_emb)

            # Max similarity tới đã chọn
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sel_emb = np.array(
                    candidates[sel_idx].get("embedding", []), dtype=np.float32
                )
                if cand_emb.size > 0 and sel_emb.size > 0:
                    max_sim_to_selected = max(
                        max_sim_to_selected, _cosine_sim(cand_emb, sel_emb)
                    )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = dict(candidates[idx])
        item["score"] = best_score if idx == selected[-1] else item.get("score", 0)
        results.append(item)
    return results


# =============================================================================
# RRF — Reciprocal Rank Fusion
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: Mỗi list từ 1 ranker (semantic, lexical, ...)
        top_k: Số kết quả cuối
        k: Smoothing constant (default=60, từ Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for content, score in sorted_items[:top_k]:
        item = dict(content_map[content])
        item["score"] = float(score)
        results.append(item)
    return results


# =============================================================================
# Cross-Encoder (mock local) — dùng cosine sim giữa query embedding và chunk
# =============================================================================

def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Mock cross-encoder reranking sử dụng sentence-transformer cosine sim.

    Lý do mock: không có Jina API key, không download model 500MB+ cho test.
    Đây là approximation: cosine(query_embed, chunk_embed) * length_penalty.
    Thực tế sẽ dùng cross-encoder model thật để score (query, chunk) pair.

    Args:
        query: Câu truy vấn
        candidates: List of {'content', 'score', 'embedding'?, 'metadata'}
        top_k: Số kết quả

    Returns:
        List of top_k candidates re-scored, sorted desc.
    """
    if not candidates:
        return []

    # Nếu candidates có embedding, dùng. Nếu không, embed query và so sánh với content embedding
    from sentence_transformers import SentenceTransformer

    # Lazy load model
    if not hasattr(rerank_cross_encoder, "_model"):
        rerank_cross_encoder._model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    model = rerank_cross_encoder._model

    q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]

    scored = []
    for cand in candidates:
        cand_emb = np.array(cand.get("embedding", []), dtype=np.float32)
        if cand_emb.size == 0:
            # Embed content on-the-fly (chậm hơn)
            cand_emb = model.encode(
                [cand["content"]], normalize_embeddings=True, show_progress_bar=False
            )[0]

        sim = _cosine_sim(q_emb, cand_emb)
        # Length penalty: ưu tiên chunks không quá dài (>2000 chars) cũng không quá ngắn (<50)
        length = len(cand["content"])
        if length < 50:
            length_penalty = 0.5
        elif length > 2000:
            length_penalty = 0.8
        else:
            length_penalty = 1.0
        score = sim * length_penalty
        scored.append((score, cand))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, cand in scored[:top_k]:
        item = dict(cand)
        item["score"] = float(score)
        results.append(item)
    return results


# =============================================================================
# Unified interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
    query_embedding: Optional[list[float]] = None,
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số kết quả
        method: "cross_encoder" | "mmr" | "rrf"
        query_embedding: Bắt buộc cho MMR, optional cho cross_encoder.

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        if query_embedding is None:
            raise ValueError("MMR can query_embedding")
        return rerank_mmr(query_embedding, candidates, top_k)
    elif method == "rrf":
        raise ValueError("RRF can nhieu ranked lists, dung rerank_rrf truc tiep")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    print("=== Cross-encoder (mock) ===")
    results = rerank("hinh phat tang tru ma tuy", dummy_candidates, top_k=2)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['content']}")

    print("\n=== RRF ===")
    rrf = rerank_rrf([dummy_candidates, dummy_candidates[:2]], top_k=2)
    for r in rrf:
        print(f"  [{r['score']:.3f}] {r['content']}")
