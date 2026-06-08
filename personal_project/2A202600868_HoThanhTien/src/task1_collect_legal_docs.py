"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Crawl từ thuvienphapluat.vn bằng Crawl4AI, lưu dưới dạng .md
(đã thử nghiệm: thuvienphapluat.vn KHÔNG có direct PDF, nhưng
crawl được toàn văn dạng HTML/Markdown).
"""

import asyncio
import sys
from pathlib import Path

# Fix Windows console encoding for Vietnamese
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from crawl4ai import AsyncWebCrawler

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Danh sách URL văn bản pháp luật cần crawl (đã verify còn sống)
LEGAL_URLS = {
    "luat-73-2021-phong-chong-ma-tuy.md": (
        "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Luat-Phong-chong-ma-tuy-2021-445185.aspx"
    ),
    "nghi-dinh-105-2021-huong-dan-luat.md": (
        "https://thuvienphapluat.vn/van-ban/Thu-tuc-hanh-chinh/Nghi-dinh-105-2021-ND-CP-huong-dan-Luat-Phong-chong-ma-tuy-2021-499880.aspx"
    ),
    "bo-luat-hinh-su-2015.md": (
        "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Bo-luat-hinh-su-2015-100296.aspx"
    ),
    "nghi-dinh-57-2022-danh-muc-chat-ma-tuy.md": (
        "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Nghi-dinh-57-2022-ND-CP-ve-danh-muc-chat-ma-tuy-498770.aspx"
    ),
}


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Thư mục đã sẵn sàng: {DATA_DIR}")


async def crawl_legal_doc(crawler, url: str) -> str | None:
    """Crawl 1 URL, trả về markdown hoặc None nếu fail."""
    try:
        result = await crawler.arun(url=url, timeout=30)
        if result.success and result.markdown and len(result.markdown) > 1000:
            return result.markdown
        print(f"  [WARN] Crawl fail hoac qua ngan: {url}")
        return None
    except Exception as e:
        print(f"  [ERR] Loi crawl {url}: {e}")
        return None


async def collect_all():
    """Crawl toàn bộ văn bản pháp luật."""
    setup_directory()

    async with AsyncWebCrawler(verbose=False) as crawler:
        for i, (filename, url) in enumerate(LEGAL_URLS.items(), 1):
            print(f"[{i}/{len(LEGAL_URLS)}] Crawling: {filename}")
            content = await crawl_legal_doc(crawler, url)
            if content is None:
                continue

            filepath = DATA_DIR / filename
            filepath.write_text(content, encoding="utf-8")
            size_kb = filepath.stat().st_size / 1024
            print(f"  [OK] Saved: {filepath.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    asyncio.run(collect_all())
    print(f"\n[OK] Hoan tat! Files trong {DATA_DIR}:")
    for f in sorted(DATA_DIR.iterdir()):
        if f.is_file():
            print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
