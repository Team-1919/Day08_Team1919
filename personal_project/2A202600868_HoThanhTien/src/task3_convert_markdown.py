"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

- Legal files: đã là .md (do Crawl4AI trả về markdown) -> copy sang standardized/
- News files: .json chứa content_markdown -> extract + thêm header
"""

import json
import shutil
import sys
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """
    Legal docs đã ở dạng .md (Crawl4AI output).
    Copy sang standardized/legal/.
    """
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md_files = list(legal_dir.glob("*.md"))
    for filepath in md_files:
        print(f"Converting: {filepath.name}")
        output_path = output_dir / f"{filepath.stem}.md"
        shutil.copy2(filepath, output_path)
        size_kb = output_path.stat().st_size / 1024
        print(f"  [OK] Saved: {output_path.name} ({size_kb:.1f} KB)")


def convert_news_articles():
    """
    News JSON chứa content_markdown -> extract + thêm header.
    """
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(news_dir.glob("*.json"))
    for filepath in json_files:
        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))

        title = data.get("title", "Unknown")
        url = data.get("url", "N/A")
        date = data.get("date_crawled", "N/A")
        content = data.get("content_markdown", "")

        # Build header
        header = f"# {title}\n\n"
        header += f"**Source:** {url}\n"
        header += f"**Crawled:** {date}\n\n---\n\n"

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(header + content, encoding="utf-8")
        size_kb = output_path.stat().st_size / 1024
        print(f"  [OK] Saved: {output_path.name} ({size_kb:.1f} KB)")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print(f"\n[OK] Done! Output tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
