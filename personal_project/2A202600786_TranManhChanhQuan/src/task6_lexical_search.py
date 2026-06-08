"""
Task 6 — Lexical Search Module (BM25).

Sử dụng rank_bm25 trên cùng corpus với Task 4.
Tokenize đơn giản bằng .lower().split() (phù hợp tiếng Việt không dấu
và cả có dấu; không dùng underthesea để tránh thêm dependency nặng).
"""

import pickle
import re
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

SRC_DIR = Path(__file__).parent
FAISS_META_PATH = SRC_DIR.parent / "data" / ".faiss.meta.pkl"

# Module-level cache
_bm25 = None
_chunks = None


def _tokenize(text: str) -> list[str]:
    """
    Tokenize đơn giản: lowercase + tách theo ký tự không phải chữ/số.
    Giữ cả tiếng Việt có dấu.
    """
    text = text.lower()
    # Tách theo ký tự không phải chữ cái (có dấu) hoặc số
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    # Lọc bỏ token quá ngắn
    return [t for t in tokens if len(t) > 1]


def _load_resources():
    """Lazy load chunks + build BM25 index."""
    global _bm25, _chunks
    if _bm25 is not None:
        return

    if not FAISS_META_PATH.exists():
        raise FileNotFoundError(
            f"FAISS meta chua ton tai. Hay chay task4 truoc."
        )

    from rank_bm25 import BM25Okapi

    _chunks = pickle.loads(FAISS_META_PATH.read_bytes())
    tokenized_corpus = [_tokenize(c["content"]) for c in _chunks]
    _bm25 = BM25Okapi(tokenized_corpus)
    print(f"  [OK] Built BM25 index on {len(_chunks)} chunks")


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,  # BM25 score
            'metadata': dict
        }
        Sorted by BM25 score descending.
    """
    _load_resources()

    if not query or not query.strip():
        return []

    tokenized_query = _tokenize(query)
    scores = _bm25.get_scores(tokenized_query)

    # Lấy top_k indices với score > 0
    top_k = min(top_k, len(scores))
    # argsort descending
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            break  # Vì sorted desc, các phần tử sau cũng <= 0
        chunk = _chunks[idx]
        results.append({
            "content": chunk["content"],
            "score": score,
            "metadata": chunk.get("metadata", {}),
        })

    return results


if __name__ == "__main__":
    test_queries = [
        "Điều 248 tàng trữ trái phép chất ma tuý",
        "Nghi dinh 57 danh muc",
        "Chau Viet Cuong ma tuy",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = lexical_search(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] {r['metadata'].get('source', '?')}: {r['content'][:80]}...")
