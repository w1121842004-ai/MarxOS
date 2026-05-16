import json
import os
import re
from pathlib import Path

import fitz

from build_vectorstore_from_cache import BOOK_MAPPING, is_me_volume, normalize_digits


DATA_DIR = "data"
ARTICLE_MAP_PATH = os.getenv("ARTICLE_MAP_PATH", "rag/article_map.json")
MAX_TOC_PAGES = int(os.getenv("MAX_TOC_PAGES", "40"))
TARGET_PDFS = {
    name.strip()
    for name in os.getenv("TARGET_PDFS", "").split(",")
    if name.strip()
}

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


def strip_trailing_subtitle(title):
    title = clean_title(title)
    match = re.match(r"(?P<parent>.+?)[［\[][^］\]]+[］\]]$", title)

    if not match:
        return None

    parent = clean_title(match.group("parent"))

    if len(parent) < 2:
        return None

    return parent


def looks_like_child_entry(title):
    title = clean_title(title)

    if re.match(r"^[［\[][^］\]]+[］\]]$", title):
        return True

    if title in {"导言", "序言", "前言", "跋", "附录", "论文"}:
        return True

    if re.match(r"^[一二三四五六七八九十]+[、.．]", title):
        return True

    return False


def looks_like_new_major_entry(title):
    title = clean_title(title)

    if not title:
        return False

    if title in {"注释", "人名索引", "名目索引", "期刊索引"}:
        return True

    if title.endswith("索引") or title.endswith("年表"):
        return True

    if title.startswith("《") and not title.endswith("》的总计划草案") and "第一页" not in title:
        return True

    if re.match(r"^第[一二三四五六七八九十百]+[编部分]", title):
        return True

    return False


def add_parent_entries(entries):
    enriched = list(entries)

    for index, entry in enumerate(entries):
        parent_title = strip_trailing_subtitle(entry["title"])
        if not parent_title:
            continue

        end_page = entry["end_printed_page"]

        for next_entry in entries[index + 1:]:
            next_title = clean_title(next_entry["title"])

            if looks_like_new_major_entry(next_title) and parent_title not in next_title:
                break

            if parent_title in next_title or looks_like_child_entry(next_title):
                end_page = max(end_page, next_entry["end_printed_page"])
                continue

            # Short unquoted titles immediately after a bracketed section are
            # often children of the same parent, e.g. 导言 under 自然辩证法.
            if len(next_title) <= 24 and not next_title.startswith("《"):
                end_page = max(end_page, next_entry["end_printed_page"])
                continue

            break

        enriched.append(
            {
                "title": parent_title,
                "start_printed_page": entry["start_printed_page"],
                "end_printed_page": end_page,
            }
        )

    return enriched


def can_be_parent(entry):
    title = clean_title(entry["title"])

    if not title:
        return False

    if looks_like_child_entry(title):
        return False

    return entry["end_printed_page"] > entry["start_printed_page"]


def entry_width(entry):
    return entry["end_printed_page"] - entry["start_printed_page"]


def find_parent_entry(entry, entries):
    candidates = []

    for candidate in entries:
        if candidate is entry:
            continue

        if not can_be_parent(candidate):
            continue

        if candidate["title"] == entry["title"]:
            continue

        if candidate["start_printed_page"] > entry["start_printed_page"]:
            continue

        if candidate["end_printed_page"] < entry["end_printed_page"]:
            continue

        if entry_width(candidate) <= entry_width(entry):
            continue

        candidates.append(candidate)

    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda item: (
            entry_width(item),
            -item["start_printed_page"],
            item["title"],
        ),
    )[0]


def annotate_hierarchy(entries):
    annotated = []

    for entry in entries:
        annotated.append(
            {
                **entry,
                "level": 1,
                "parent": None,
            }
        )

    # Multiple passes let nested levels settle even when the parent appears
    # after a child in sorted page order, as with synthesized parent titles.
    for _ in range(4):
        changed = False

        for entry in annotated:
            parent = find_parent_entry(entry, annotated)
            parent_title = parent["title"] if parent else None
            parent_level = parent["level"] if parent else 0
            level = min(parent_level + 1, 6) if parent else 1

            if entry["parent"] != parent_title or entry["level"] != level:
                entry["parent"] = parent_title
                entry["level"] = level
                changed = True

        if not changed:
            break

    return annotated


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

    return annotate_hierarchy(normalize_ranges(add_parent_entries(entries)))


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
        if TARGET_PDFS and pdf_path.name not in TARGET_PDFS:
            continue

        if TARGET_PDFS or is_me_volume(pdf_path.name):
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
