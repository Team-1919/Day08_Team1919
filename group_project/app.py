"""
Flask API server — RAG Pipeline v2.

Kết nối các module trong src/ với giao diện web (index.html).

Chạy:
    pip install flask
    python app.py          # mặc định port 5000
    python app.py --port 5001
"""

from __future__ import annotations

import os
import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

app = Flask(__name__, static_folder=str(BASE_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# CORS — cho phép index.html gọi API khi mở trực tiếp qua file://
# ─────────────────────────────────────────────────────────────────────────────

@app.after_request
def _add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def _options(path=""):
    return jsonify({}), 200


# ─────────────────────────────────────────────────────────────────────────────
# STATIC — phục vụ index.html tại /
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR), "index.html")


# ─────────────────────────────────────────────────────────────────────────────
# LAZY-LOAD: FAISS index + BM25 index
# ─────────────────────────────────────────────────────────────────────────────

FAISS_INDEX_PATH = BASE_DIR / "data" / ".faiss.index"
FAISS_META_PATH  = BASE_DIR / "data" / ".faiss.meta.pkl"

_faiss_index = None
_faiss_meta:  list[dict] = []
_embedding_model = None
_bm25 = None
_bm25_meta: list[dict] = []


def _load_faiss() -> bool:
    global _faiss_index, _faiss_meta
    if _faiss_index is not None:
        return True
    if not FAISS_INDEX_PATH.exists() or not FAISS_META_PATH.exists():
        return False
    try:
        import faiss  # noqa
        _faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        _faiss_meta  = pickle.loads(FAISS_META_PATH.read_bytes())
        print(f"[OK] FAISS index loaded: {_faiss_index.ntotal} vectors, "
              f"{len(_faiss_meta)} chunks")
        return True
    except Exception as exc:
        print(f"[WARN] FAISS load failed: {exc}")
        return False


def _get_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer  # noqa
        _embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _embedding_model


def _load_bm25() -> bool:
    global _bm25, _bm25_meta
    if _bm25 is not None:
        return True
    if not FAISS_META_PATH.exists():
        return False
    try:
        from rank_bm25 import BM25Okapi  # noqa
        chunks = pickle.loads(FAISS_META_PATH.read_bytes())
        _bm25_meta = chunks
        tokenized = [c["content"].lower().split() for c in chunks]
        _bm25 = BM25Okapi(tokenized)
        print(f"[OK] BM25 index built: {len(_bm25_meta)} chunks")
        return True
    except Exception as exc:
        print(f"[WARN] BM25 build failed: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _semantic_search(query: str, top_k: int = 10) -> list[dict]:
    if not _load_faiss():
        raise RuntimeError("FAISS index not available. Run task4 first.")
    model = _get_model()
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = _faiss_index.search(q_emb, min(top_k, _faiss_index.ntotal))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_faiss_meta):
            continue
        chunk = _faiss_meta[int(idx)]
        results.append({
            "content": chunk["content"],
            "score": float(score),
            "metadata": chunk.get("metadata", {}),
        })
    return results


def _lexical_search(query: str, top_k: int = 10) -> list[dict]:
    if not _load_bm25():
        raise RuntimeError("BM25 index not available. Run task4 first.")
    tokens = query.lower().split()
    raw_scores = _bm25.get_scores(tokens)
    top_indices = np.argsort(raw_scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        sc = float(raw_scores[int(idx)])
        if sc <= 0:
            continue
        chunk = _bm25_meta[int(idx)]
        results.append({
            "content": chunk["content"],
            "score": sc,
            "metadata": chunk.get("metadata", {}),
        })
    return results


def _rrf_merge(ranked_lists: list[list[dict]], top_k: int, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion."""
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict]  = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"][:300]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for content_key, score in sorted_items[:top_k]:
        item = dict(content_map[content_key])
        item["rrf_score"] = score
        item["score"] = score
        results.append(item)
    return results


def _jina_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict] | None:
    jina_key = os.getenv("JINA_API_KEY", "")
    if not jina_key:
        return None
    try:
        import requests as req  # noqa
        resp = req.post(
            "https://api.jina.ai/v1/rerank",
            headers={
                "Authorization": f"Bearer {jina_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k,
            },
            timeout=15,
        )
        resp.raise_for_status()
        reranked = resp.json()["results"]
        return [
            {**candidates[r["index"]], "score": r["relevance_score"]}
            for r in reranked
        ]
    except Exception as exc:
        print(f"[WARN] Jina rerank failed: {exc}")
        return None


def _reorder_lost_in_middle(chunks: list[dict]) -> list[dict]:
    """
    Tránh 'lost in the middle': chunks quan trọng ở đầu và cuối.
    [1,2,3,4,5] → [1,3,5,4,2]
    """
    n = len(chunks)
    if n <= 2:
        return chunks
    first_half  = chunks[::2]         # 0, 2, 4 …
    second_half = chunks[1::2][::-1]  # 1, 3, 5 … đảo ngược
    return first_half + second_half


def _retrieve(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.3,
    use_reranking: bool = True,
) -> dict:
    """Full retrieval pipeline. Returns {results, log, ...}."""
    faiss_ok = _load_faiss()
    bm25_ok  = _load_bm25()
    log: list[str] = []

    if not faiss_ok and not bm25_ok:
        raise RuntimeError("No index available. Run task4 first.")

    # Step 1 — Dense search
    dense_results: list[dict] = []
    if faiss_ok:
        try:
            dense_results = _semantic_search(query, top_k=top_k * 2)
            log.append(f"🔵 Semantic Search → {len(dense_results)} results")
        except Exception as exc:
            log.append(f"🔵 Semantic Search failed: {exc}")

    # Step 2 — BM25 search
    sparse_results: list[dict] = []
    if bm25_ok:
        try:
            sparse_results = _lexical_search(query, top_k=top_k * 2)
            log.append(f"🟠 BM25 → {len(sparse_results)} results")
        except Exception as exc:
            log.append(f"🟠 BM25 failed: {exc}")

    # Step 3 — RRF merge
    if dense_results and sparse_results:
        merged = _rrf_merge([dense_results, sparse_results], top_k=top_k * 2)
        log.append(f"⚖️  RRF Merge → {len(merged)} candidates")
    elif dense_results:
        merged = dense_results[:top_k * 2]
    elif sparse_results:
        merged = sparse_results[:top_k * 2]
    else:
        raise RuntimeError("No results from any search method")

    # Step 4 — Rerank
    if use_reranking and merged:
        reranked = _jina_rerank(query, merged, top_k)
        if reranked:
            merged = reranked
            log.append("✅ Reranked via Jina API")
        else:
            merged = sorted(merged, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
            log.append("⚖️  Sorted by score (no Jina key)")
    else:
        merged = sorted(merged, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    # Step 5 — Threshold check → PageIndex fallback
    best_score    = merged[0]["score"] if merged else 0.0
    fallback_used = False

    if best_score < score_threshold:
        pageindex_key = os.getenv("PAGEINDEX_API_KEY", "")
        if pageindex_key:
            try:
                from pageindex import PageIndex  # noqa
                pi = PageIndex(api_key=pageindex_key)
                pi_results = pi.query(query=query, top_k=top_k)
                merged = [
                    {
                        "content": r.text,
                        "score":   r.score,
                        "metadata": getattr(r, "metadata", {}),
                        "source":  "pageindex",
                    }
                    for r in pi_results
                ]
                fallback_used = True
                log.append(
                    f"⚠️  Score {best_score:.3f} < threshold {score_threshold} "
                    f"→ PageIndex fallback activated"
                )
            except Exception as exc:
                log.append(f"⚠️  Score {best_score:.3f} < threshold {score_threshold} "
                            f"(PageIndex failed: {exc})")
        else:
            log.append(
                f"⚠️  Score {best_score:.3f} < threshold {score_threshold} "
                f"(no PageIndex API key configured)"
            )

    for item in merged:
        item.setdefault("source", "pageindex" if fallback_used else "hybrid")

    return {
        "results":        merged[:top_k],
        "log":            log,
        "best_score":     best_score,
        "fallback_used":  fallback_used,
        "score_threshold": score_threshold,
    }


SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta   = chunk.get("metadata", {})
        source = meta.get("source", f"Nguồn {i}")
        dtype  = meta.get("type", "document")
        parts.append(
            f"[Tài liệu {i} | Nguồn: {source} | Loại: {dtype}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def _generate(query: str, top_k: int = 5, score_threshold: float = 0.3) -> dict:
    retrieval = _retrieve(query, top_k=top_k, score_threshold=score_threshold)
    chunks    = retrieval["results"]

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
            "log": retrieval["log"],
        }

    reordered = _reorder_lost_in_middle(chunks)
    context   = _format_context(reordered)

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        try:
            from openai import OpenAI  # noqa
            client = OpenAI(api_key=openai_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Context:\n{context}\n\n---\n\nCâu hỏi: {query}"},
                ],
                temperature=0.3,
                top_p=0.9,
                max_tokens=800,
            )
            answer = completion.choices[0].message.content
        except Exception as exc:
            answer = (
                f"[OpenAI lỗi: {exc}]\n\n"
                f"Dựa trên context tìm được:\n\n"
                + "\n\n".join(
                    f"[{c.get('metadata',{}).get('source','?')}] {c['content'][:200]}…"
                    for c in reordered[:3]
                )
            )
    else:
        # Không có OpenAI key — trả về context đã format kèm ghi chú
        answer = (
            "⚠️ Chưa cấu hình OPENAI_API_KEY. "
            "Dưới đây là các đoạn văn bản liên quan được tìm thấy:\n\n"
            + "\n\n".join(
                f"**[{c.get('metadata',{}).get('source','?')}]** {c['content'][:300]}…"
                for c in reordered
            )
        )

    return {
        "answer":           answer,
        "sources":          chunks,
        "retrieval_source": "pageindex" if retrieval["fallback_used"] else "hybrid",
        "log":              retrieval["log"],
        "reordered_chunks": [
            {
                "content":  ch["content"][:120],
                "source":   ch.get("metadata", {}).get("source", "?"),
                "original_rank": chunks.index(ch) + 1 if ch in chunks else 0,
                "reordered_rank": i + 1,
            }
            for i, ch in enumerate(reordered)
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    """Trả về trạng thái của tất cả các thành phần."""
    faiss_ok = _load_faiss()
    bm25_ok  = _load_bm25() if faiss_ok else False
    return jsonify({
        "faiss_index":    faiss_ok,
        "faiss_chunks":   len(_faiss_meta) if faiss_ok else 0,
        "bm25":           bm25_ok,
        "openai_key":     bool(os.getenv("OPENAI_API_KEY")),
        "jina_key":       bool(os.getenv("JINA_API_KEY")),
        "pageindex_key":  bool(os.getenv("PAGEINDEX_API_KEY")),
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    })


@app.route("/api/search/semantic", methods=["POST"])
def api_semantic_search():
    """Task 5 — Semantic Search."""
    data  = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 10))
    if not query:
        return jsonify({"error": "query is required"}), 400
    try:
        results = _semantic_search(query, top_k=top_k)
        return jsonify(results)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/search/lexical", methods=["POST"])
def api_lexical_search():
    """Task 6 — Lexical Search (BM25)."""
    data  = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 10))
    if not query:
        return jsonify({"error": "query is required"}), 400
    try:
        results = _lexical_search(query, top_k=top_k)
        return jsonify(results)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/rerank", methods=["POST"])
def api_rerank():
    """Task 7 — Reranking (Jina API hoặc sort-by-score fallback)."""
    data       = request.get_json(silent=True) or {}
    query      = data.get("query", "").strip()
    candidates = data.get("candidates", [])
    top_k      = int(data.get("top_k", 5))
    if not query or not candidates:
        return jsonify({"error": "query and candidates are required"}), 400
    reranked = _jina_rerank(query, candidates, top_k)
    if reranked:
        return jsonify({"results": reranked, "method": "jina"})
    sorted_cands = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
    return jsonify({"results": sorted_cands[:top_k], "method": "score_sort"})


@app.route("/api/retrieve", methods=["POST"])
def api_retrieve():
    """Task 9 — Full Retrieval Pipeline."""
    data            = request.get_json(silent=True) or {}
    query           = data.get("query", "").strip()
    top_k           = int(data.get("top_k", 5))
    score_threshold = float(data.get("score_threshold", 0.3))
    use_reranking   = bool(data.get("use_reranking", True))
    if not query:
        return jsonify({"error": "query is required"}), 400
    try:
        result = _retrieve(query, top_k=top_k,
                           score_threshold=score_threshold,
                           use_reranking=use_reranking)
        return jsonify(result)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Task 10 — Generation with Citation."""
    data            = request.get_json(silent=True) or {}
    query           = data.get("query", "").strip()
    top_k           = int(data.get("top_k", 5))
    score_threshold = float(data.get("score_threshold", 0.3))
    if not query:
        return jsonify({"error": "query is required"}), 400
    try:
        result = _generate(query, top_k=top_k, score_threshold=score_threshold)
        return jsonify(result)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline v2 — Flask API")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print("=" * 60)
    print("RAG Pipeline v2 — Flask API Server")
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  FAISS index: {FAISS_INDEX_PATH}")
    print(f"  OpenAI key:  {'✓' if os.getenv('OPENAI_API_KEY') else '✗ (không cấu hình)'}")
    print(f"  Jina key:    {'✓' if os.getenv('JINA_API_KEY') else '✗ (không cấu hình)'}")
    print("=" * 60)

    app.run(host=args.host, port=args.port, debug=True)
