import os
import json
from datetime import datetime
from crawl4ai import AsyncWebCrawler


async def crawl_article(url: str, output_dir: str):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url="https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov")

        data = {
            "url": url,
            "crawl_date": datetime.now().isoformat(),
            "title": result.metadata.get("title", ""),
            "content": result.markdown
        }

        filename = data["title"][:50].replace("/", "-") + ".json"

        with open(
            os.path.join(output_dir, filename),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)