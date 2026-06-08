"""
Task 4 — Chunking & Indexing vào FAISS vector store.

Stack thống nhất:
    - Embedding: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    - Vector store: FAISS local (faiss-cpu), index tại data/faiss_index/
"""

import pickle
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
DATA_DIR = Path(__file__).parent.parent / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "index.faiss"
FAISS_META_PATH = FAISS_INDEX_DIR / "meta.pkl"
CHUNKS_CACHE_PATH = DATA_DIR / ".chunks_cache.pkl"
BM25_CORPUS_PATH = DATA_DIR / "bm25_corpus.pkl"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
VECTOR_STORE = "faiss"

_embedding_model = None


def load_documents() -> list[dict]:
    """Đọc toàn bộ markdown files từ data/standardized/."""
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
                "path": str(md_file.relative_to(STANDARDIZED_DIR.parent)),
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents theo RecursiveCharacterTextSplitter."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def _get_embedding_model():
    global _embedding_model
    from sentence_transformers import SentenceTransformer

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def embed_text(text: str) -> list[float]:
    """Embed một đoạn text bằng model thống nhất."""
    model = _get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.astype(np.float32).tolist()


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed toàn bộ chunks; cache vào .chunks_cache.pkl khi cần."""
    if CHUNKS_CACHE_PATH.exists():
        try:
            cached = pickle.loads(CHUNKS_CACHE_PATH.read_bytes())
            if len(cached) == len(chunks) and all(
                cached[i]["content"] == chunks[i]["content"]
                for i in range(min(3, len(chunks)))
            ):
                return cached
        except Exception:
            pass

    model = _get_embedding_model()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=len(texts) > 32,
        batch_size=32,
        normalize_embeddings=True,
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.astype(np.float32).tolist()

    CHUNKS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_CACHE_PATH.write_bytes(pickle.dumps(chunks))
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks vào FAISS index tại data/faiss_index/."""
    import faiss

    if not chunks or "embedding" not in chunks[0]:
        raise ValueError("Chunks chưa được embed. Gọi embed_chunks() trước.")

    embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    meta = [
        {"content": c["content"], "metadata": c["metadata"]}
        for c in chunks
    ]
    FAISS_META_PATH.write_bytes(pickle.dumps(meta))


def save_bm25_corpus(chunks: list[dict]):
    """Lưu corpus chunks cho BM25 tại data/bm25_corpus.pkl."""
    corpus = [{"content": c["content"], "metadata": c["metadata"]} for c in chunks]
    BM25_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BM25_CORPUS_PATH.write_bytes(pickle.dumps(corpus))


def faiss_index_exists() -> bool:
    return FAISS_INDEX_PATH.exists() and FAISS_META_PATH.exists()


def run_pipeline(force_reindex: bool = False):
    """Chạy pipeline: load → chunk → embed → index FAISS."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing (FAISS)")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE} → {FAISS_INDEX_DIR}")
    print("=" * 50)

    if force_reindex:
        for path in (CHUNKS_CACHE_PATH, FAISS_INDEX_PATH, FAISS_META_PATH):
            if path.exists():
                path.unlink()

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print(f"[OK] Indexed to FAISS: {FAISS_INDEX_PATH}")

    save_bm25_corpus(chunks)
    print(f"[OK] Saved BM25 corpus: {BM25_CORPUS_PATH}")


if __name__ == "__main__":
    run_pipeline()
