"""
Task 4 - Chunking and local indexing.

This implementation keeps the pipeline local so it can run in class without paid
APIs. It uses recursive character chunking, Ollama embeddings when available,
and deterministic hashing embeddings as a fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

import requests


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = STANDARDIZED_DIR / "_index"
INDEX_PATH = INDEX_DIR / "chunks.json"

# Recursive chunking is robust for mixed legal/news markdown. 500 chars keeps
# chunks focused enough for retrieval while 50 chars preserves boundary context.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Prefer Ollama local embeddings when an embedding model is available. The code
# falls back to deterministic hashing so tests still pass without Ollama.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBEDDING_MODEL = f"ollama:{OLLAMA_EMBED_MODEL}"
# nomic-embed-text returns 768-dimensional vectors; the hashing fallback uses
# the same dimension so the index shape remains stable.
EMBEDDING_DIM = 768
VECTOR_STORE = "local-json"
_OLLAMA_AVAILABLE: bool | None = None


def load_documents() -> list[dict]:
    """Read all markdown files from data/standardized/."""
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if "_index" in md_file.parts:
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        rel = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        header_metadata = _extract_markdown_metadata(content)
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(rel).replace("\\", "/"),
                    "type": doc_type,
                    **header_metadata,
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents into size-bounded overlapping chunks."""
    chunks: list[dict] = []
    for doc in documents:
        splits = _split_text(doc["content"])
        for idx, text in enumerate(splits):
            chunks.append(
                {
                    "content": text,
                    "metadata": {**doc.get("metadata", {}), "chunk_index": idx},
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Attach Ollama embeddings when available, otherwise hashing embeddings."""
    model_name = get_embedding_model_name()
    if _is_ollama_embedding_available():
        for start in range(0, len(chunks), 16):
            batch = chunks[start : start + 16]
            embeddings = _ollama_embed_batch([chunk["content"] for chunk in batch])
            if not embeddings:
                embeddings = [_hash_embedding(chunk["content"]) for chunk in batch]
                model_name = "local-hashing-tfidf"
            for chunk, embedding in zip(batch, embeddings):
                chunk["embedding"] = embedding
                chunk["embedding_model"] = model_name
    else:
        for chunk in chunks:
            chunk["embedding"] = _hash_embedding(chunk["content"])
            chunk["embedding_model"] = model_name
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> Path:
    """Persist chunks to a local JSON index."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            "content": c["content"],
            "metadata": c.get("metadata", {}),
            "embedding": c.get("embedding", []),
            "embedding_model": c.get("embedding_model", get_embedding_model_name()),
        }
        for c in chunks
    ]
    INDEX_PATH.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    return INDEX_PATH


def get_chunks() -> list[dict]:
    """Load chunks from the local index, or build them from markdown files."""
    if INDEX_PATH.exists():
        try:
            chunks = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            expected_model = get_embedding_model_name()
            if chunks and chunks[0].get("embedding_model") == expected_model:
                return chunks
        except json.JSONDecodeError:
            pass

    docs = load_documents()
    chunks = embed_chunks(chunk_documents(docs))
    index_to_vectorstore(chunks)
    return chunks


def run_pipeline() -> None:
    """Run load -> chunk -> embed -> local index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")
    embed_chunks(chunks)
    index_path = index_to_vectorstore(chunks)
    print(f"Indexed to {index_path}")


def _split_text(text: str) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        window = text[start:end]
        if end < len(text):
            split_at = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"), window.rfind(" "))
            if split_at > CHUNK_SIZE * 0.5:
                end = start + split_at + 1
                window = text[start:end]
        chunk = window.strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def _extract_markdown_metadata(content: str) -> dict:
    metadata: dict[str, str] = {}
    source_match = re.search(r"^\*\*Source:\*\*\s*(.+)$", content, flags=re.MULTILINE)
    crawled_match = re.search(r"^\*\*Crawled:\*\*\s*(.+)$", content, flags=re.MULTILINE)
    if source_match:
        metadata["source_url"] = source_match.group(1).strip()
    if crawled_match:
        crawled = crawled_match.group(1).strip()
        metadata["crawl_date"] = crawled
        year_match = re.search(r"(20\d{2}|19\d{2})", crawled)
        if year_match:
            metadata["year"] = year_match.group(1)
    return metadata


def _hash_embedding(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % EMBEDDING_DIM
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[idx] += sign
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def embed_text(text: str) -> list[float]:
    """Embed text with Ollama if possible, falling back to local hashing."""
    if _is_ollama_embedding_available():
        embeddings = _ollama_embed_batch([text])
        if embeddings:
            return embeddings[0]
    return _hash_embedding(text)


def _ollama_embed_batch(texts: list[str]) -> list[list[float]]:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": texts},
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        return [[float(value) for value in embedding] for embedding in data.get("embeddings", [])]
    except requests.RequestException:
        return []


def get_embedding_model_name() -> str:
    """Return the active embedding backend name."""
    return f"ollama:{OLLAMA_EMBED_MODEL}" if _is_ollama_embedding_available() else "local-hashing-tfidf"


def _is_ollama_embedding_available() -> bool:
    global _OLLAMA_AVAILABLE
    if _OLLAMA_AVAILABLE is not None:
        return _OLLAMA_AVAILABLE

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": "health check"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        _OLLAMA_AVAILABLE = bool(data.get("embeddings"))
    except requests.RequestException:
        _OLLAMA_AVAILABLE = False
    return _OLLAMA_AVAILABLE


def tokenize(text: str) -> list[str]:
    import unicodedata

    lowered = text.lower()
    normalized = unicodedata.normalize("NFD", lowered)
    no_accents = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return re.findall(r"\w+", no_accents, flags=re.UNICODE)


if __name__ == "__main__":
    run_pipeline()
