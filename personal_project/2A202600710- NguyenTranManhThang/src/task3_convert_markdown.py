"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import asyncio
import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
MIN_CONTENT_LENGTH = 200

# Một số PDF từ vanban.chinhphu.vn là bản scan (không có text layer).
LEGAL_FALLBACK_URLS = {
    "nghi-dinh-105-2021": "https://luatvietnam.vn/an-ninh-trat-tu/nghi-dinh-105-2021-nd-cp-213690-d1.html",
    "nghi-dinh-57-2022": (
        "https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2022/8/37734/"
        "41623-1-2022709-71057-2022-nd-cp.pdf"
    ),
}


def _crawl_markdown(url: str) -> str:
    """Crawl nội dung web bằng Crawl4AI khi không convert được từ file."""
    from crawl4ai import AsyncWebCrawler

    async def _run():
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
        if not result.success:
            raise RuntimeError(f"Crawl thất bại: {url} — {result.error_message}")
        return result.markdown or ""

    return asyncio.run(_run())


def _convert_legal_file(md: MarkItDown, filepath: Path) -> str:
    """Convert file pháp luật, fallback sang nguồn khác nếu PDF scan rỗng."""
    result = md.convert(str(filepath))
    content = (result.text_content or "").strip()
    if len(content) >= MIN_CONTENT_LENGTH:
        return content

    fallback_url = LEGAL_FALLBACK_URLS.get(filepath.stem)
    if not fallback_url:
        return content

    print(f"  ↻ PDF scan, dùng nguồn thay thế: {fallback_url}")
    if fallback_url.lower().endswith(".pdf"):
        alt = md.convert(fallback_url)
        return (alt.text_content or "").strip()

    return _crawl_markdown(fallback_url).strip()


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    if not legal_dir.exists():
        print("⚠ Không tìm thấy thư mục legal/, bỏ qua.")
        return

    for filepath in sorted(legal_dir.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() not in (".pdf", ".docx"):
            continue

        output_path = output_dir / f"{filepath.stem}.md"
        print(f"Converting: {filepath.name}")
        # TODO: Convert và lưu file
        # result = md.convert(str(filepath))
        # output_path = output_dir / f"{filepath.stem}.md"
        # output_path.write_text(result.text_content, encoding="utf-8")
        # print(f"  ✓ Saved: {output_path}")
        if output_path.exists() and len(output_path.read_text(encoding="utf-8")) > MIN_CONTENT_LENGTH:
            print(f"  ⊘ Bỏ qua (đã có): {output_path}")
            continue

        content = _convert_legal_file(md, filepath)
        output_path.write_text(content, encoding="utf-8")
        print(f"  ✓ Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print("⚠ Không tìm thấy thư mục news/, bỏ qua.")
        return

    for filepath in sorted(news_dir.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() != ".json":
            continue

        output_path = output_dir / f"{filepath.stem}.md"
        print(f"Converting: {filepath.name}")
        # TODO: Đọc JSON, extract content_markdown, lưu thành .md
        # data = json.loads(filepath.read_text(encoding="utf-8"))
        # output_path = output_dir / f"{filepath.stem}.md"
        #
        # # Thêm metadata header
        # header = f"# {data.get('title', 'Unknown')}\n\n"
        # header += f"**Source:** {data.get('url', 'N/A')}\n"
        # header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
        #
        # content = header + data.get("content_markdown", "")
        # output_path.write_text(content, encoding="utf-8")
        # print(f"  ✓ Saved: {output_path}")
        if output_path.exists() and len(output_path.read_text(encoding="utf-8")) > MIN_CONTENT_LENGTH:
            print(f"  ⊘ Bỏ qua (đã có): {output_path}")
            continue

        data = json.loads(filepath.read_text(encoding="utf-8"))
        header = f"# {data.get('title', 'Unknown')}\n\n"
        header += f"**Source:** {data.get('url', 'N/A')}\n"
        header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
        content = header + data.get("content_markdown", "")
        output_path.write_text(content, encoding="utf-8")
        print(f"  ✓ Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
