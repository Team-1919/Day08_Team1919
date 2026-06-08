"""
Task 3 - Convert files from data/landing/ to Markdown.

Legal PDF/DOCX files are converted with MarkItDown when it is installed. News
JSON files are converted by extracting readable article text from the stored HTML.
Output keeps the source folder structure under data/standardized/.
"""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> None:
    """Convert PDF/DOCX files in data/landing/legal/ to markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = MarkItDown() if MarkItDown else None

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            continue

        print(f"Converting: {filepath.name}")
        output_path = output_dir / f"{filepath.stem}.md"

        if converter:
            result = converter.convert(str(filepath))
            content = result.text_content.strip()
        elif filepath.suffix.lower() == ".pdf" and PdfReader:
            content = _pdf_to_markdown(filepath)
        else:
            content = _fallback_legal_markdown(filepath)

        output_path.write_text(content + "\n", encoding="utf-8")
        print(f"  Saved: {output_path}")


def convert_news_articles() -> None:
    """Convert crawled JSON news articles in data/landing/news/ to markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        body = data.get("content_markdown") or _html_to_markdown(data.get("html", ""))

        header = f"# {data.get('title') or filepath.stem}\n\n"
        header += f"**Source:** {data.get('url', 'N/A')}\n\n"
        header += f"**Crawled:** {data.get('crawl_date', data.get('date_crawled', 'N/A'))}\n\n"
        if data.get("description"):
            header += f"**Description:** {data['description']}\n\n"
        header += "---\n\n"

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(header + body.strip() + "\n", encoding="utf-8")
        print(f"  Saved: {output_path}")


def convert_all() -> None:
    """Convert all supported landing files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\nDone! Output at:", OUTPUT_DIR)


def _fallback_legal_markdown(filepath: Path) -> str:
    """Create useful markdown metadata when PDF extraction is unavailable."""
    title = filepath.stem.replace("-", " ").title()
    return (
        f"# {title}\n\n"
        f"**Source file:** {filepath.name}\n\n"
        f"**Document type:** legal\n\n"
        f"**Original format:** {filepath.suffix.lower().lstrip('.')}\n\n"
        "This markdown file was generated from the legal source document in "
        "`data/landing/legal/`. Install `markitdown` and rerun Task 3 to extract "
        "the full PDF/DOCX text. The source file is preserved for later indexing "
        "and verification.\n"
    )


def _pdf_to_markdown(filepath: Path) -> str:
    """Extract PDF text with pypdf when MarkItDown is unavailable."""
    reader = PdfReader(str(filepath))
    title = filepath.stem.replace("-", " ").title()
    extracted_chars = 0
    parts = [
        f"# {title}",
        "",
        f"**Source file:** {filepath.name}",
        "",
        "**Document type:** legal",
        "",
        "---",
        "",
    ]

    for page_number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        text = _clean_pdf_text(text)
        if text:
            extracted_chars += len(text)
            parts.append(f"## Page {page_number}")
            parts.append("")
            parts.append(text)
            parts.append("")

    if extracted_chars < 200:
        return _fallback_legal_markdown(filepath)

    return "\n".join(parts).strip()


def _clean_pdf_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _html_to_markdown(html: str) -> str:
    """Extract readable article text from stored HTML and format it as Markdown."""
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    candidates = soup.find_all(["article", "main"])
    root = max(candidates, key=lambda tag: len(tag.get_text(" ", strip=True)), default=soup)

    lines: list[str] = []
    for tag in root.find_all(["h1", "h2", "h3", "p", "li"]):
        text = _clean_text(tag.get_text(" ", strip=True))
        if not text or len(text) < 20:
            continue

        if tag.name == "h1":
            lines.append(f"# {text}")
        elif tag.name == "h2":
            lines.append(f"## {text}")
        elif tag.name == "h3":
            lines.append(f"### {text}")
        elif tag.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line)

    return "\n\n".join(deduped)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    convert_all()
