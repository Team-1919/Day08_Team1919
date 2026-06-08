"""
Task 5 — Semantic Search Module.

Stack: paraphrase-multilingual-MiniLM-L12-v2 + FAISS tại data/faiss_index/
"""

import os
import pickle
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

_DATA_DIR = Path(__file__).parent.parent / "data"
FAISS_INDEX_PATH = _DATA_DIR / "faiss_index" / "index.faiss"
FAISS_META_PATH = _DATA_DIR / "faiss_index" / "meta.pkl"

_faiss_index = None
_faiss_meta: list[dict] | None = None


def _load_faiss():
    global _faiss_index, _faiss_meta
    if _faiss_index is not None and _faiss_meta is not None:
        return _faiss_index, _faiss_meta

    from src.task4_chunking_indexing import faiss_index_exists, run_pipeline

    if not faiss_index_exists():
        run_pipeline()

    import faiss

    _faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
    _faiss_meta = pickle.loads(FAISS_META_PATH.read_bytes())
    return _faiss_index, _faiss_meta


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng cosine similarity trên FAISS.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    index, meta = _load_faiss()
    if index.ntotal == 0 or not meta:
        return []

    from src._embeddings import embed_text

    query_vec = np.array([embed_text(query)], dtype=np.float32)
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = meta[idx]
        results.append({
            "content": chunk["content"],
            "score": float(score),
            "metadata": chunk.get("metadata", {}),
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
