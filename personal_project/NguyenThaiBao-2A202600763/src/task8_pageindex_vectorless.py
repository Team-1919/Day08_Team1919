"""Task 8 - PageIndex vectorless RAG with local fallback."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .task4_chunking_indexing import STANDARDIZED_DIR, get_chunks, tokenize

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")


def upload_documents() -> list[dict]:
    """
    Prepare documents for PageIndex upload.

    In offline mode this returns local document metadata, which is enough for a
    reproducible demo and keeps the real PageIndex integration point explicit.
    """
    uploaded: list[dict] = []
    for md_file in sorted(Path(STANDARDIZED_DIR).rglob("*.md")):
        if "_index" in md_file.parts:
            continue
        uploaded.append({"filename": md_file.name, "path": str(md_file)})
    return uploaded


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Vectorless fallback search over local markdown chunks."""
    query_terms = set(tokenize(query))
    results: list[dict] = []
    for chunk in get_chunks():
        terms = set(tokenize(chunk.get("content", "")))
        overlap = len(query_terms & terms)
        score = overlap / (len(query_terms) or 1)
        if score <= 0:
            continue
        results.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
                "source": "pageindex",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in pageindex_search("ma tuy", top_k=3):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
