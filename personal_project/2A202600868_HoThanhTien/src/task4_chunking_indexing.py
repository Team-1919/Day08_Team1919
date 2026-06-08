"""
Task 4 — Chunking & Indexing vào FAISS vector store.

Stack: FAISS + sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
- Multilingual model tốt cho tiếng Việt, ~471MB, 384 dim
- FAISS đơn giản, chạy local, không cần Docker
- Cache chunks + embeddings vào pickle để load nhanh cho test lặp lại
"""

import pickle
import sys
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_DIR = Path(__file__).parent.parent / "data"
CACHE_PATH = CACHE_DIR / ".chunks_cache.pkl"
FAISS_INDEX_PATH = CACHE_DIR / ".faiss.index"
FAISS_META_PATH = CACHE_DIR / ".faiss.meta.pkl"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# Chunking strategy: RecursiveCharacterTextSplitter
# - Phổ biến, an toàn, hoạt động tốt với nhiều loại văn bản
# - Ưu tiên split theo đoạn văn (\n\n) -> câu -> từ
# - Tốt cho cả văn bản pháp luật (có cấu trúc điều/khoản) và bài báo
CHUNK_SIZE = 500        # chars. ~500 chars ~ 1-2 đoạn pháp luật, vừa đủ cho context window
CHUNK_OVERLAP = 80      # 16% overlap. Giữ ngữ cảnh ranh giới giữa 2 chunks
CHUNKING_METHOD = "recursive"

# Embedding model: paraphrase-multilingual-MiniLM-L12-v2
# - Multilingual, tốt cho tiếng Việt
# - 384 dim, nhẹ (~471MB)
# - Faster hơn BAAI/bge-m3 (1024 dim, ~2GB)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# Vector store: FAISS (in-memory + IndexFlatIP cho cosine similarity)
# - Đơn giản, chạy local, không cần Docker
# - IndexFlatIP cho inner product (tương đương cosine khi vectors đã normalize)
VECTOR_STORE = "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        # Xác định doc_type dựa trên đường dẫn (legal/ hoặc news/)
        doc_type = "legal" if "legal" in str(md_file) else "news"
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
    """
    Chunk documents theo RecursiveCharacterTextSplitter.
    """
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


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng sentence-transformers.
    Caching: nếu chunks đã embed rồi, load từ cache.
    """
    from sentence_transformers import SentenceTransformer

    # Check cache
    if CACHE_PATH.exists():
        try:
            cached = pickle.loads(CACHE_PATH.read_bytes())
            # Verify cache cùng số chunks
            if len(cached) == len(chunks):
                # Quick verify content
                if all(
                    cached[i]["content"] == chunks[i]["content"]
                    for i in range(min(3, len(chunks)))
                ):
                    print(f"  [OK] Loaded {len(cached)} chunks tu cache")
                    return cached
        except Exception as e:
            print(f"  [WARN] Cache loi ({e}), re-embed...")

    # Load model và embed
    print(f"  Loading model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,  # quan trọng cho cosine similarity
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.astype(np.float32).tolist()

    # Save cache
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_bytes(pickle.dumps(chunks))
    print(f"  [OK] Saved cache: {CACHE_PATH}")
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào FAISS index.
    - IndexFlatIP: inner product (cosine sim khi vectors đã normalize)
    - Lưu riêng FAISS index và metadata để load nhanh.
    """
    import faiss

    if not chunks or "embedding" not in chunks[0]:
        raise ValueError("Chunks chưa được embed. Gọi embed_chunks() trước.")

    embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)

    # Build FAISS index (Inner Product = cosine khi vectors đã normalize)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    print(f"  [OK] FAISS index co {index.ntotal} vectors")

    # Save
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    FAISS_META_PATH.write_bytes(pickle.dumps(chunks))
    print(f"  [OK] Saved FAISS index: {FAISS_INDEX_PATH}")
    print(f"  [OK] Saved metadata: {FAISS_META_PATH}")


def run_pipeline(force_reindex: bool = False):
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing (FAISS)")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    if force_reindex:
        for p in [CACHE_PATH, FAISS_INDEX_PATH, FAISS_META_PATH]:
            if p.exists():
                p.unlink()
        print("[OK] Cleared cache + FAISS index")

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("[OK] Indexed to FAISS")


if __name__ == "__main__":
    run_pipeline()
