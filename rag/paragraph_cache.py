from __future__ import annotations

import json
import os
import re
from pathlib import Path

from rag.build_vectorstore_from_cache import (
    OCR_CACHE_DIR,
    document_from_cache,
    page_num_from_cache_file,
)


CORE_PARAGRAPH_SOURCES = [
    *(f"mea{i:02d}.pdf" for i in range(1, 11)),
    *(f"mes{i:02d}.pdf" for i in range(1, 5)),
]

SENTENCE_END_RE = re.compile(r"[。！？；!?;][”’」』）)]?$")
HEADING_PREFIX_RE = re.compile(r"^(第[一二三四五六七八九十百零〇\d]+[章节篇部编]|[一二三四五六七八九十]+[、.．])")


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def paragraph_cache_sources(value: str | None = None) -> list[str]:
    value = value if value is not None else os.getenv("TARGET_PDFS", "")
    sources = [item.strip() for item in value.split(",") if item.strip()]
    return sources or CORE_PARAGRAPH_SOURCES


def iter_source_cache_files(source: str, ocr_cache_dir: str | Path = OCR_CACHE_DIR) -> list[Path]:
    stem = source.replace(".pdf", "")
    source_dir = Path(ocr_cache_dir) / stem
    if not source_dir.exists():
        return []

    return sorted(
        source_dir.glob("page_*.txt"),
        key=lambda path: page_num_from_cache_file(str(path)) or 0,
    )


def is_noise_line(line: str, metadata: dict) -> bool:
    line = str(line or "").strip()
    if not line:
        return True

    compact = compact_text(line)
    if not compact:
        return True

    if len(compact) <= 2 and not re.search(r"[\u4e00-\u9fff]", compact):
        return True

    titles = [
        metadata.get("section"),
        metadata.get("article"),
        metadata.get("chapter"),
        metadata.get("book"),
    ]
    if any(compact == compact_text(title) for title in titles if title):
        return True

    if re.fullmatch(r"[-—_·•.。,\s\d]+", line):
        return True

    return False


def line_starts_new_paragraph(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped:
        return False

    if HEADING_PREFIX_RE.match(stripped):
        return True

    return bool(re.match(r"^[“\"《（(]?[一二三四五六七八九十]+[、.．]", stripped))


def line_ends_paragraph(line: str) -> bool:
    return bool(SENTENCE_END_RE.search(str(line or "").strip()))


def join_paragraph_lines(lines: list[str]) -> str:
    text = "".join(line.strip() for line in lines if line.strip())
    text = re.sub(r"\s+", "", text)
    return text


def split_page_paragraphs(text: str, metadata: dict, min_chars: int = 35) -> list[str]:
    raw_lines = [line.strip() for line in str(text or "").splitlines()]
    lines = [line for line in raw_lines if not is_noise_line(line, metadata)]
    paragraphs = []
    current = []

    for line in lines:
        if current and line_starts_new_paragraph(line):
            paragraph = join_paragraph_lines(current)
            if paragraph:
                paragraphs.append(paragraph)
            current = [line]
            continue

        current.append(line)
        paragraph = join_paragraph_lines(current)
        if len(paragraph) >= min_chars and line_ends_paragraph(line):
            paragraphs.append(paragraph)
            current = []

    if current:
        paragraph = join_paragraph_lines(current)
        if paragraph:
            paragraphs.append(paragraph)

    return paragraphs


def is_incomplete_paragraph(text: str) -> bool:
    text = str(text or "").strip()
    if not text:
        return False
    return not bool(SENTENCE_END_RE.search(text))


def paragraph_record(source: str, doc, text: str, index_on_page: int) -> dict:
    metadata = dict(doc.metadata)
    pdf_page = metadata.get("pdf_page")
    citation_page = metadata.get("citation_page")

    return {
        "source": source,
        "book": metadata.get("book"),
        "article": metadata.get("article"),
        "section": metadata.get("section"),
        "paragraph_text": text,
        "paragraph_char_count": len(text),
        "paragraph_index_on_page": index_on_page,
        "pdf_page_start": pdf_page,
        "pdf_page_end": pdf_page,
        "printed_page_start": metadata.get("printed_page"),
        "printed_page_end": metadata.get("printed_page"),
        "citation_page_start": citation_page,
        "citation_page_end": citation_page,
        "citation_page_type": metadata.get("citation_page_type"),
        "page_span": [pdf_page],
        "page_type": metadata.get("page_type"),
    }


def merge_records(left: dict, right: dict) -> dict:
    merged = dict(left)
    merged["paragraph_text"] = left["paragraph_text"] + right["paragraph_text"]
    merged["paragraph_char_count"] = len(merged["paragraph_text"])
    merged["pdf_page_end"] = right.get("pdf_page_end")
    merged["printed_page_end"] = right.get("printed_page_end")
    merged["citation_page_end"] = right.get("citation_page_end")
    merged["page_span"] = list(dict.fromkeys((left.get("page_span") or []) + (right.get("page_span") or [])))
    merged["cross_page"] = True
    return merged


def build_paragraph_records_for_source(source: str, ocr_cache_dir: str | Path = OCR_CACHE_DIR) -> list[dict]:
    records = []
    title_context = {}
    page_sequence_context = {}
    pending = None

    for cache_path in iter_source_cache_files(source, ocr_cache_dir):
        doc = document_from_cache(str(cache_path), title_context, page_sequence_context)
        if doc is None:
            continue

        paragraphs = split_page_paragraphs(doc.page_content, doc.metadata)
        page_records = [
            paragraph_record(source, doc, paragraph, index)
            for index, paragraph in enumerate(paragraphs, start=1)
        ]

        if pending and page_records:
            page_records[0] = merge_records(pending, page_records[0])
            pending = None

        if page_records and is_incomplete_paragraph(page_records[-1]["paragraph_text"]):
            pending = page_records.pop()

        records.extend(page_records)

    if pending:
        pending["cross_page"] = False
        records.append(pending)

    for index, record in enumerate(records, start=1):
        record["paragraph_index"] = index
        record["paragraph_id"] = f"{source}#p{index:06d}"

    return records


def write_paragraph_cache(
    output_path: str | Path,
    sources: list[str] | None = None,
    ocr_cache_dir: str | Path = OCR_CACHE_DIR,
) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sources = sources or paragraph_cache_sources()
    summary = {"sources": {}, "paragraphs": 0}

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for source in sources:
            records = build_paragraph_records_for_source(source, ocr_cache_dir)
            summary["sources"][source] = len(records)
            summary["paragraphs"] += len(records)
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return summary


def read_paragraph_cache(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
