"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    # TODO: Implement upload
    #
    # Tham khảo: https://github.com/VectifyAI/PageIndex
    #
    # from pageindex import PageIndex
    #
    # pi = PageIndex(api_key=PAGEINDEX_API_KEY)
    #
    # for md_file in STANDARDIZED_DIR.rglob("*.md"):
    #     content = md_file.read_text(encoding="utf-8")
    #     pi.upload(
    #         content=content,
    #         metadata={"filename": md_file.name, "type": md_file.parent.name}
    #     )
    #     print(f"  ✓ Uploaded: {md_file.name}")
    raise NotImplementedError("Implement upload_documents")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval: keyword overlap search trên standardized markdown files.
    Mock implementation khi không có PageIndex API key.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'
        }
    """
    if PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndex  # type: ignore
            pi = PageIndex(api_key=PAGEINDEX_API_KEY)
            api_results = pi.query(query=query, top_k=top_k)
            return [
                {
                    "content": r.text,
                    "score": float(r.score),
                    "metadata": r.metadata,
                    "source": "pageindex",
                }
                for r in api_results
            ]
        except Exception:
            pass  # Fall through to local mock

    # Local mock: keyword overlap (Jaccard-like) trên markdown files
    query_tokens = set(query.lower().split())
    if not query_tokens:
        return []

    results = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue

        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 50]
        for para in paragraphs:
            para_tokens = set(para.lower().split())
            overlap = len(query_tokens & para_tokens)
            if overlap > 0:
                score = overlap / (len(query_tokens) + len(para_tokens) - overlap + 1)
                results.append(
                    {
                        "content": para[:600],
                        "score": float(score),
                        "metadata": {
                            "source": md_file.name,
                            "type": md_file.parent.name,
                        },
                        "source": "pageindex",
                    }
                )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


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
