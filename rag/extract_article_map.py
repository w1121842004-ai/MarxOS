import os

OCR_CACHE_DIR = "data/ocr_cache"


def read_pages(volume_name, start_page=1, end_page=30):
    texts = []

    for page_num in range(start_page, end_page + 1):
        path = os.path.join(
            OCR_CACHE_DIR,
            volume_name,
            f"page_{page_num}.txt"
        )

        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if text:
            texts.append(f"\n===== 第 {page_num} 页 =====\n{text}")

    return "\n".join(texts)


if __name__ == "__main__":
    volume_name = "me24"

    catalog_text = read_pages(
        volume_name,
        start_page=1,
        end_page=30
    )

    print(catalog_text[:5000])