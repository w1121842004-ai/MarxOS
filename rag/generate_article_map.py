import json
import os
import re
from pathlib import Path

import fitz

from build_vectorstore_from_cache import BOOK_MAPPING, is_me_volume, normalize_digits


DATA_DIR = "data"
ARTICLE_MAP_PATH = "rag/article_map.json"
MAX_TOC_PAGES = int(os.getenv("MAX_TOC_PAGES", "40"))

WATERMARK_MARKERS = [
    "本PDF文件",
    "S22PDF",
    "pdf@",
    "pdfFactory",
    "fineprint",
]


def clean_line(line):
    line = normalize_digits(line)
    line = re.sub(r"\s+", "", line)
    line = line.strip(" \t\r\n·.。…—-")

    return line


def is_noise(line):
    if not line:
        return True

    if any(marker in line for marker in WATERMARK_MARKERS):
        return True

    if set(line) <= {"…", ".", "·", "-", "—", "－"}:
        return True

    if re.fullmatch(r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+", line):
        return True

    return False


def page_range_from_line(line):
    match = re.fullmatch(r"(?P<start>\d{1,4})(?:[—－-](?P<end>\d{1,4}))?", line)

    if not match:
        return None

    start = int(match.group("start"))
    end = int(match.group("end") or start)

    if start <= 0 or end <= 0:
        return None

    return start, end


def split_inline_range(line):
    match = re.search(r"(?P<title>.+?)(?P<start>\d{1,4})(?:[—－-](?P<end>\d{1,4}))?$", line)

    if not match:
        return None

    title = match.group("title").strip(" …—－-")
    start = int(match.group("start"))
    end = int(match.group("end") or start)

    if len(title) < 2 or start <= 0 or end <= 0:
        return None

    return title, start, end


def clean_title(title):
    title = re.sub(r"[.。…·—－-]+$", "", title)
    title = title.strip(" ，。；：:、（）()[]【】")

    return title[:120]


def extract_toc_entries(pdf_path):
    doc = fitz.open(str(pdf_path))
    entries = []
    title_buffer = []

    for page_index in range(min(MAX_TOC_PAGES, len(doc))):
        text = doc.load_page(page_index).get_text("text")
        raw_lines = text.splitlines()
        cleaned_lines = [clean_line(line) for line in raw_lines]
        has_toc_label = any("目录" in line or "目錄" in line or "目次" in line for line in cleaned_lines)
        leader_count = sum(1 for line in raw_lines if "…" in line or "……" in line)
        looks_like_toc = has_toc_label or leader_count >= 3

        if not looks_like_toc:
            continue

        for raw_line in raw_lines:
            line = clean_line(raw_line)

            if is_noise(line) or line in {"目录", "目錄", "目次"}:
                continue

            page_range = page_range_from_line(line)

            if page_range and title_buffer:
                title = clean_title("".join(title_buffer))
                title_buffer = []

                if title:
                    entries.append(
                        {
                            "title": title,
                            "start_printed_page": page_range[0],
                            "end_printed_page": page_range[1],
                        }
                    )
                continue

            inline = split_inline_range(line)

            if inline:
                title, start, end = inline
                title = clean_title(title)

                if title:
                    entries.append(
                        {
                            "title": title,
                            "start_printed_page": start,
                            "end_printed_page": end,
                        }
                    )
                title_buffer = []
                continue

            if len(line) <= 80:
                title_buffer.append(line)

    doc.close()

    return normalize_ranges(entries)


def normalize_ranges(entries):
    deduped = []
    seen = set()

    for entry in entries:
        key = (entry["title"], entry["start_printed_page"], entry["end_printed_page"])

        if key in seen:
            continue

        seen.add(key)
        deduped.append(entry)

    deduped.sort(key=lambda item: (item["start_printed_page"], item["end_printed_page"], item["title"]))

    for index, entry in enumerate(deduped):
        if entry["end_printed_page"] < entry["start_printed_page"]:
            entry["end_printed_page"] = entry["start_printed_page"]

        if entry["end_printed_page"] == entry["start_printed_page"] and index + 1 < len(deduped):
            next_start = deduped[index + 1]["start_printed_page"]

            if next_start > entry["start_printed_page"]:
                entry["end_printed_page"] = next_start - 1

    return deduped


def iter_target_pdfs():
    for pdf_path in sorted(Path(DATA_DIR).rglob("*.pdf")):
        if is_me_volume(pdf_path.name):
            yield pdf_path


def main():
    article_map = {}

    for pdf_path in iter_target_pdfs():
        entries = extract_toc_entries(pdf_path)
        source = pdf_path.name
        article_map[source] = {
            "book": BOOK_MAPPING.get(source, source.replace(".pdf", "")),
            "entries": entries,
        }
        print(f"{source}: {len(entries)} entries")

    with open(ARTICLE_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(article_map, f, ensure_ascii=False, indent=2)

    print(f"\narticle map saved: {ARTICLE_MAP_PATH}")


if __name__ == "__main__":
    main()
