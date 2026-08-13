from __future__ import annotations

import json
import os
import re
from pathlib import Path

from langchain_core.documents import Document
from marxos.data.document_contract import normalize_document_record, stable_paragraph_id

from rag.build_vectorstore_from_cache import (
    OCR_CACHE_DIR,
    document_from_cache,
    page_num_from_cache_file,
)


CORE_PARAGRAPH_SOURCES = [
    *(f"mea{i:02d}.pdf" for i in range(1, 11)),
    *(f"mes{i:02d}.pdf" for i in range(1, 5)),
]

SENTENCE_END_RE = re.compile(r"[\u3002\uff01\uff1f!?;\uff1b][\u201d\u2019\u3001\uff09\)]?$")
HEADING_PREFIX_RE = re.compile(
    r"^("
    r"[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e]+[\u3001\.\uff0e]"
    r"|[\uff08\(]?[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e]+[\uff09\)]"
    r"|\d+[\u3001\.\uff0e]"
    r"|\u7b2c[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\d]+[\u7ae0\u8282\u5377\u7bc7]"
    r")"
)


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

    if re.fullmatch(r"[!！\.。·•…_\-\s]{4,}", line):
        return True

    if re.search(r"[!！]{5,}", line):
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

    return bool(re.match(r"^[\u25cf\u25cb\u2022\-]\s*", stripped))


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


def locate_paragraph_on_page(page_text: str, paragraph_text: str, cursor: int = 0) -> tuple[int | None, int | None, int | None, int | None]:
    raw_lines = str(page_text or "").splitlines()
    compact_to_raw = []
    compact_chars = []
    raw_offset = 0

    for line_index, line in enumerate(raw_lines, start=1):
        for char_index, char in enumerate(line):
            if char.strip():
                compact_chars.append(char)
                compact_to_raw.append((raw_offset + char_index, line_index))
        raw_offset += len(line) + 1

    compact_page = "".join(compact_chars)
    compact_para = compact_text(paragraph_text)
    if not compact_page or not compact_para:
        return None, None, None, None

    start = compact_page.find(compact_para, max(cursor, 0))
    if start < 0:
        start = compact_page.find(compact_para)
    if start < 0:
        return None, None, None, None

    end = start + len(compact_para) - 1
    char_start, line_start = compact_to_raw[start]
    char_end, line_end = compact_to_raw[min(end, len(compact_to_raw) - 1)]
    return char_start, char_end + 1, line_start, line_end


def paragraph_record(source: str, doc, text: str, index_on_page: int, char_start=None, char_end=None, line_start=None, line_end=None) -> dict:
    metadata = dict(doc.metadata)
    pdf_page = metadata.get("pdf_page")
    citation_page = metadata.get("citation_page")

    page_id = f"{source}#pdf{pdf_page}" if pdf_page is not None else None
    return normalize_document_record({
        "source": source,
        "book": metadata.get("book"),
        "article": metadata.get("article"),
        "section": metadata.get("section"),
        "paragraph_text": text,
        "paragraph_char_count": len(text),
        "paragraph_index_on_page": index_on_page,
        "char_start": char_start,
        "char_end": char_end,
        "line_start": line_start,
        "line_end": line_end,
        "pdf_page_start": pdf_page,
        "pdf_page_end": pdf_page,
        "printed_page_start": metadata.get("printed_page"),
        "printed_page_end": metadata.get("printed_page"),
        "citation_page_start": citation_page,
        "citation_page_end": citation_page,
        "citation_page_type": metadata.get("citation_page_type"),
        "page_span": [pdf_page],
        "spans": [{
            "page_id": page_id,
            "pdf_page": pdf_page,
            "char_start": char_start,
            "char_end": char_end,
            "line_start": line_start,
            "line_end": line_end,
        }],
        "page_type": metadata.get("page_type"),
        "text_source": metadata.get("text_source"),
        "page_number_source": metadata.get("page_number_source"),
        "cleaning_reasons": metadata.get("cleaning_reasons"),
    }, retrieval_unit="paragraph")


def merge_records(left: dict, right: dict) -> dict:
    merged = dict(left)
    merged["paragraph_text"] = left["paragraph_text"] + right["paragraph_text"]
    merged["paragraph_char_count"] = len(merged["paragraph_text"])
    merged["pdf_page_end"] = right.get("pdf_page_end")
    merged["printed_page_end"] = right.get("printed_page_end")
    merged["citation_page_end"] = right.get("citation_page_end")
    merged["page_span"] = list(dict.fromkeys((left.get("page_span") or []) + (right.get("page_span") or [])))
    merged["spans"] = [dict(span) for span in (left.get("spans") or []) + (right.get("spans") or [])]
    merged["source_page_ids"] = list(
        dict.fromkeys(
            span.get("page_id")
            for span in merged["spans"]
            if span.get("page_id")
        )
    )
    merged["char_end"] = right.get("char_end")
    merged["line_end"] = right.get("line_end")
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
            if pending:
                pending["cross_page"] = False
                records.append(pending)
                pending = None
            continue
        cleaning_reasons = str(doc.metadata.get("cleaning_reasons") or "")
        if "many_line_end_pages" in cleaning_reasons and doc.metadata.get("printed_page") is None:
            continue

        paragraphs = split_page_paragraphs(doc.page_content, doc.metadata)
        page_records = []
        cursor = 0
        for index, paragraph in enumerate(paragraphs, start=1):
            char_start, char_end, line_start, line_end = locate_paragraph_on_page(doc.page_content, paragraph, cursor)
            if char_end is not None:
                cursor = char_end
            page_records.append(
                paragraph_record(
                    source,
                    doc,
                    paragraph,
                    index,
                    char_start=char_start,
                    char_end=char_end,
                    line_start=line_start,
                    line_end=line_end,
                )
            )

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
        record["paragraph_id"] = stable_paragraph_id(record)
        record["parent_paragraph_id"] = record["paragraph_id"]

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


def paragraph_record_to_document(record: dict) -> Document:
    metadata = {
        key: value
        for key, value in dict(record).items()
        if key != "paragraph_text" and value is not None
    }
    metadata["retrieval_unit"] = "paragraph"
    metadata["page"] = record.get("citation_page_start") or record.get("pdf_page_start")
    metadata["pdf_page"] = record.get("pdf_page_start")
    metadata["pdf_page_end"] = record.get("pdf_page_end")
    metadata["printed_page"] = record.get("printed_page_start")
    metadata["printed_page_end"] = record.get("printed_page_end")
    metadata["citation_page"] = record.get("citation_page_start")
    metadata["citation_page_end"] = record.get("citation_page_end")
    metadata["char_start"] = record.get("char_start")
    metadata["char_end"] = record.get("char_end")
    metadata["line_start"] = record.get("line_start")
    metadata["line_end"] = record.get("line_end")

    if record.get("citation_page_start") != record.get("citation_page_end"):
        metadata["page_range"] = f"{record.get('citation_page_start')}-{record.get('citation_page_end')}"
    elif record.get("pdf_page_start") != record.get("pdf_page_end"):
        metadata["page_range"] = f"PDF {record.get('pdf_page_start')}-{record.get('pdf_page_end')}"

    return Document(
        page_content=record.get("paragraph_text") or "",
        metadata=metadata,
    )
