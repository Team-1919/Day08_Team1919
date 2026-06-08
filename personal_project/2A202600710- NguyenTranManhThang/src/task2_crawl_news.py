"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    # Ví dụ:
    # "https://vnexpress.net/...",
    # "https://tuoitre.vn/...",
    # "https://thanhnien.vn/...",
    "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
    "https://vnexpress.net/ca-si-miu-le-bi-bat-qua-tang-dung-ma-tuy-o-bai-bien-5072657.html",
    "https://tuoitre.vn/bat-ca-si-long-nhat-va-ca-si-son-ngoc-minh-vi-lien-quan-ma-tuy-20260520082138943.htm",
    "https://tuoitre.vn/ca-si-long-nhat-khai-su-dung-ma-tuy-da-cung-quan-ly-20260520132251413.htm",
    "https://vnexpress.net/ma-tuy-trong-loi-song-showbiz-5074606.html",
]


def _parse_crawl_result(url: str, result) -> dict:
    """Chuyển CrawlResult thành dict metadata + nội dung."""
    if not result.success:
        raise RuntimeError(f"Crawl thất bại: {url} — {result.error_message}")

    metadata = result.metadata or {}
    title = metadata.get("title") or metadata.get("og:title") or "Unknown"
    content = result.markdown or result.cleaned_html or ""

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content,
    }


async def crawl_article(url: str, crawler=None) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    # TODO: Implement crawling logic
    # async with AsyncWebCrawler() as crawler:
    #     result = await crawler.arun(url=url)
    #     return {
    #         "url": url,
    #         "title": result.metadata.get("title", "Unknown"),
    #         "date_crawled": datetime.now().isoformat(),
    #         "content_markdown": result.markdown,
    #     }
    if crawler is not None:
        result = await crawler.arun(url=url)
        return _parse_crawl_result(url, result)

    async with AsyncWebCrawler() as new_crawler:
        result = await new_crawler.arun(url=url)
        return _parse_crawl_result(url, result)


async def crawl_all(skip_existing: bool = True):
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        for i, url in enumerate(ARTICLE_URLS, 1):
            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename

            if skip_existing and filepath.exists() and filepath.stat().st_size > 500:
                print(f"[{i}/{len(ARTICLE_URLS)}] ⊘ Bỏ qua (đã có): {filepath}")
                continue

            print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
            article = await crawl_article(url, crawler=crawler)

            # Lưu file JSON
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
