"""
Task 6 — Lexical Search Module (BM25).

Stack: rank-bm25, corpus lưu tại data/bm25_corpus.pkl
"""

import pickle

from src.task4_chunking_indexing import (
    BM25_CORPUS_PATH,
    chunk_documents,
    load_documents,
    save_bm25_corpus,
)

CORPUS: list[dict] = []
_bm25_index = None


def build_and_save_bm25_corpus(chunks: list[dict] | None = None) -> list[dict]:
    """Xây dựng và lưu corpus chunks cho BM25."""
    global CORPUS, _bm25_index
    if chunks is None:
        chunks = chunk_documents(load_documents())

    save_bm25_corpus(chunks)
    corpus = pickle.loads(BM25_CORPUS_PATH.read_bytes())
    CORPUS = corpus
    _bm25_index = None
    return corpus


def _ensure_corpus() -> list[dict]:
    global CORPUS
    if CORPUS:
        return CORPUS

    if BM25_CORPUS_PATH.exists():
        CORPUS = pickle.loads(BM25_CORPUS_PATH.read_bytes())
        return CORPUS

    return build_and_save_bm25_corpus()


def build_bm25_index(corpus: list[dict]):
    """Xây dựng BM25 index từ corpus."""
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _get_bm25_index():
    global _bm25_index
    corpus = _ensure_corpus()
    if _bm25_index is None:
        _bm25_index = build_bm25_index(corpus)
    return _bm25_index, corpus


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Returns:
        List of {'content', 'score', 'metadata'} sorted by score descending.
    """
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    bm25, corpus = _get_bm25_index()
    if not corpus:
        return []

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    import numpy as np

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"],
            })
    return results


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
