"""
Task 5 — Semantic Search Module (Dense Retrieval trên FAISS).

Load FAISS index + chunks metadata, embed query, search top_k.
"""

import pickle
import sys
from pathlib import Path
from typing import Optional

# Fix Windows encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

# Paths
SRC_DIR = Path(__file__).parent
FAISS_INDEX_PATH = SRC_DIR.parent / "data" / ".faiss.index"
FAISS_META_PATH = SRC_DIR.parent / "data" / ".faiss.meta.pkl"

# Embedding model (phải khớp với Task 4)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# Module-level cache để không load lại mỗi lần gọi
_index = None
_chunks = None
_model = None


def _load_resources():
    """Lazy load FAISS index + chunks + model (chỉ load 1 lần)."""
    global _index, _chunks, _model
    if _index is not None:
        return

    if not FAISS_INDEX_PATH.exists() or not FAISS_META_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index chua duoc tao. Hay chay task4_chunking_indexing.py truoc."
        )

    import faiss
    from sentence_transformers import SentenceTransformer

    _index = faiss.read_index(str(FAISS_INDEX_PATH))
    _chunks = pickle.loads(FAISS_META_PATH.read_bytes())
    _model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  [OK] Loaded FAISS index ({_index.ntotal} vectors) + model")


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng FAISS cosine similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,  # Cosine similarity (0-1)
            'metadata': dict
        }
        Sorted by score descending.
    """
    _load_resources()

    if not query or not query.strip():
        return []

    # Embed query (normalized để dùng Inner Product như cosine)
    query_embedding = _model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    # Search FAISS
    top_k = min(top_k, _index.ntotal)
    scores, indices = _index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_chunks):
            continue
        chunk = _chunks[idx]
        results.append({
            "content": chunk["content"],
            "score": float(score),  # Inner product of normalized vectors = cosine
            "metadata": chunk.get("metadata", {}),
        })

    return results


if __name__ == "__main__":
    test_queries = [
        "hinh phat cho toi tang tru ma tuy",
        "Nghi dinh 57/2022 danh muc chat ma tuy",
        "Chau Viet Cuong su dung ma tuy",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = semantic_search(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] {r['metadata'].get('source', '?')}: {r['content'][:80]}...")
