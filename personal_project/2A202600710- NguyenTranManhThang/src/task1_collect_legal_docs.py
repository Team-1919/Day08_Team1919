"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.

Gợi ý nguồn:
    - https://thuvienphapluat.vn
    - https://vanban.chinhphu.vn
    - https://luatvietnam.vn

Gợi ý văn bản:
    - Luật Phòng, chống ma tuý 2021 (73/2021/QH15)
    - Nghị định 105/2021/NĐ-CP
    - Bộ luật Hình sự 2015 (sửa đổi 2017) - Chương XX
    - Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


# TODO: Tải file PDF/DOCX về DATA_DIR
# Có thể tải thủ công hoặc viết script download nếu có direct link.
#
# Ví dụ nếu có direct link:
#
# import requests
#
# def download_file(url: str, filename: str):
#     response = requests.get(url)
#     filepath = DATA_DIR / filename
#     filepath.write_bytes(response.content)
#     print(f"✓ Đã tải: {filepath}")

import requests

# Văn bản pháp luật từ vanban.chinhphu.vn (Cổng TTĐT Chính phủ)
LEGAL_DOCUMENTS = [
    {
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/73luat.pdf",
        "filename": "luat-phong-chong-ma-tuy-2021.pdf",
        "title": "Luật Phòng, chống ma túy 2021 (73/2021/QH14)",
    },
    {
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2021/12/105.signed_02.pdf",
        "filename": "nghi-dinh-105-2021.pdf",
        "title": "Nghị định 105/2021/NĐ-CP hướng dẫn Luật Phòng chống ma túy",
    },
    {
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/08/57-cp.signed.pdf",
        "filename": "nghi-dinh-57-2022.pdf",
        "title": "Nghị định 57/2022/NĐ-CP về danh mục chất ma túy",
    },
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


def download_file(url: str, filename: str) -> Path:
    """Tải file từ URL và lưu vào DATA_DIR."""
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()

    filepath = DATA_DIR / filename
    filepath.write_bytes(response.content)
    print(f"✓ Đã tải: {filepath} ({len(response.content):,} bytes)")
    return filepath


def download_all(skip_existing: bool = True) -> list[Path]:
    """Tải toàn bộ văn bản pháp luật trong LEGAL_DOCUMENTS."""
    setup_directory()
    downloaded: list[Path] = []

    for doc in LEGAL_DOCUMENTS:
        filepath = DATA_DIR / doc["filename"]
        if skip_existing and filepath.exists() and filepath.stat().st_size > 1024:
            print(f"⊘ Bỏ qua (đã có): {filepath}")
            downloaded.append(filepath)
            continue

        print(f"↓ Đang tải: {doc['title']}")
        downloaded.append(download_file(doc["url"], doc["filename"]))

    return downloaded


if __name__ == "__main__":
    download_all()
