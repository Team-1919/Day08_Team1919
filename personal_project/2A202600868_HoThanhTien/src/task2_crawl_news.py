"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Crawl từ VietnamNet, VnExpress, Tuổi Trẻ bằng Crawl4AI.
Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content_markdown).
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from crawl4ai import AsyncWebCrawler

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Danh sách URL bài báo (đã verify crawl được bằng Crawl4AI)
ARTICLE_URLS = [
    # Châu Việt Cường
    "https://vietnamnet.vn/chau-viet-cuong-sa-nga-truoc-lenh-bat-giu-4572589.html",
    "https://vnexpress.net/chau-viet-cuong-bi-bat-qua-bai-hat-3728897.html",
    # Trang Moon / Trang Khàn
    "https://vietnamnet.vn/trang-moon-va-qua-khu-dung-ngoai-nghe-56123.html",
    "https://tuoitre.vn/trang-khan-khoe-son-tren-facebook-20170807094542398.htm",
    # Cao Thái Sơn / Hương Giang / Phạm Anh Khoa
    "https://vietnamnet.vn/huong-giang-idol-phan-hoi-tin-don-ma-tuy-451234.html",
    "https://vietnamnet.vn/pham-anh-khoa-toi-khong-su-dung-ma-tuy-398765.html",
    "https://vietnamnet.vn/trang-khan-len-tieng-viec-bi-to-su-dung-ma-tuy-470383.html",
]


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Thu muc: {DATA_DIR}")


async def crawl_article(crawler, url: str) -> dict | None:
    """Crawl 1 bài báo, trả về dict chứa metadata + content."""
    try:
        result = await crawler.arun(url=url, timeout=30)
        if not (result.success and result.markdown and len(result.markdown) > 500):
            print(f"  [WARN] Crawl fail hoac qua ngan: {url}")
            return None

        # Lấy title từ metadata (nếu có)
        title = ""
        if result.metadata and isinstance(result.metadata, dict):
            title = result.metadata.get("title", "") or ""

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown,
        }
    except Exception as e:
        print(f"  [ERR] Loi crawl {url}: {e}")
        return None


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    async with AsyncWebCrawler(verbose=False) as crawler:
        for i, url in enumerate(ARTICLE_URLS, 1):
            print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
            article = await crawl_article(crawler, url)
            if article is None:
                continue

            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            size_kb = filepath.stat().st_size / 1024
            print(f"  [OK] Saved: {filename} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("[WARN] Hay dien ARTICLE_URLS truoc khi chay!")
    else:
        asyncio.run(crawl_all())
        print(f"\n[OK] Hoan tat! Files trong {DATA_DIR}:")
        for f in sorted(DATA_DIR.iterdir()):
            if f.is_file():
                print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
