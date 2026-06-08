"""Task 6 - Lexical search with BM25 and a small local fallback."""

from __future__ import annotations

import math
from collections import Counter

from .task4_chunking_indexing import get_chunks, tokenize

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


CORPUS: list[dict] = []


def build_bm25_index(corpus: list[dict]):
    """Build a BM25 index from chunk dictionaries."""
    tokenized = [tokenize(doc["content"]) for doc in corpus]
    if BM25Okapi:
        return BM25Okapi(tokenized)
    return _SimpleBM25(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Search chunks by lexical BM25 score."""
    if top_k <= 0:
        return []

    corpus = _load_corpus()
    if not corpus:
        return []

    bm25 = build_bm25_index(corpus)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)

    results: list[dict] = []
    for idx, score in ranked[:top_k]:
        if score <= 0:
            continue
        results.append(
            {
                "content": corpus[idx]["content"],
                "score": float(score),
                "metadata": corpus[idx].get("metadata", {}),
            }
        )
    return results


def _load_corpus() -> list[dict]:
    global CORPUS
    if not CORPUS:
        CORPUS = [
            {"content": chunk["content"], "metadata": chunk.get("metadata", {})}
            for chunk in get_chunks()
        ]
    return CORPUS


class _SimpleBM25:
    """Tiny BM25 implementation used only when rank-bm25 is unavailable."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.freqs = [Counter(doc) for doc in tokenized_corpus]
        df = Counter()
        for doc in tokenized_corpus:
            df.update(set(doc))
        total = len(tokenized_corpus)
        self.idf = {term: math.log(1 + (total - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for idx, freqs in enumerate(self.freqs):
            score = 0.0
            dl = self.doc_len[idx] or 1
            for term in query_tokens:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += self.idf.get(term, 0.0) * (tf * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


if __name__ == "__main__":
    for result in lexical_search("ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
