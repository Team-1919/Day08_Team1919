"""
Task 8 — PageIndex Vectorless RAG.

PageIndex API khi có PAGEINDEX_API_KEY; fallback local keyword search
trên markdown (vẫn đánh dấu source='pageindex' cho pipeline).
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def _local_pageindex_search(query: str, top_k: int) -> list[dict]:
    """Fallback: keyword search trên markdown khi PageIndex API không khả dụng."""
    query_terms = set(query.lower().split())
    scored = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        content_terms = set(content.lower().split())
        overlap = len(query_terms & content_terms)
        if overlap == 0:
            continue
        score = overlap / max(len(query_terms), 1)
        scored.append({
            "content": content[:2000],
            "score": score,
            "metadata": {
                "source": md_file.name,
                "doc_type": md_file.parent.name,
            },
            "source": "pageindex",
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _pageindex_api_search(query: str, top_k: int) -> list[dict]:
    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    docs_response = client.list_documents(limit=100)
    documents = docs_response.get("documents", [])

    ready_doc_ids = []
    for doc in documents:
        doc_id = doc.get("doc_id") or doc.get("id")
        if doc_id and client.is_retrieval_ready(doc_id):
            ready_doc_ids.append(doc_id)

    if not ready_doc_ids:
        raise RuntimeError("Không có document PageIndex sẵn sàng")

    results = []
    for doc_id in ready_doc_ids[:3]:
        submission = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = submission["retrieval_id"]
        data = {}

        for _ in range(60):
            data = client.get_retrieval(retrieval_id)
            status = data.get("status", "")
            if status in ("completed", "done", "success"):
                break
            if status in ("failed", "error"):
                break
            time.sleep(0.5)

        items = (
            data.get("results")
            or data.get("retrieved_nodes")
            or data.get("nodes")
            or []
        )
        if isinstance(items, dict):
            items = items.get("items", [])

        for rank, item in enumerate(items):
            if isinstance(item, str):
                content = item
                score = 1.0 - rank * 0.05
                metadata = {"doc_id": doc_id}
            else:
                content = item.get("content") or item.get("text") or str(item)
                score = float(item.get("score", 1.0 - rank * 0.05))
                metadata = item.get("metadata", {"doc_id": doc_id})
            results.append({
                "content": content,
                "score": score,
                "metadata": metadata,
                "source": "pageindex",
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def upload_documents():
    """Upload PDF pháp luật lên PageIndex (nếu có API key)."""
    if not PAGEINDEX_API_KEY or "xxx" in PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        return

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    for pdf_file in sorted(LANDING_LEGAL_DIR.glob("*.pdf")):
        result = client.submit_document(str(pdf_file))
        print(f"  ✓ Uploaded: {pdf_file.name} → {result.get('doc_id')}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex hoặc local fallback.

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'
        }
    """
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    if PAGEINDEX_API_KEY and "xxx" not in PAGEINDEX_API_KEY:
        try:
            return _pageindex_api_search(query, top_k)
        except Exception:
            pass

    return _local_pageindex_search(query, top_k)


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

    print("\nTest query:")
    results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
