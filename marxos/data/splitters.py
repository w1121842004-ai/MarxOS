from __future__ import annotations

import re

from rag.semantic_retrieval import build_semantic_child_documents
from marxos.data.document_contract import normalize_document_record

SPLIT_RE = re.compile(r"([^。！？；;!?]+[。！？；;!?]?)")


def same_parent_group(left: dict, right: dict) -> bool:
    return all(
        str(left.get(key) or "") == str(right.get(key) or "")
        for key in ("source", "book", "article", "section")
    )


def starts_new_parent(record: dict, current: list[dict], max_chars: int) -> bool:
    if not current:
        return False
    if not same_parent_group(current[-1], record):
        return True
    current_chars = sum(int(item.get("paragraph_char_count") or len(str(item.get("paragraph_text") or ""))) for item in current)
    next_chars = int(record.get("paragraph_char_count") or len(str(record.get("paragraph_text") or "")))
    if current_chars + next_chars > max_chars:
        return True
    previous_page = current[-1].get("pdf_page_end")
    current_page = record.get("pdf_page_start")
    if previous_page is not None and current_page is not None:
        try:
            if int(current_page) - int(previous_page) > 1:
                return True
        except (TypeError, ValueError):
            pass
    return False


def split_long_text(text: str, max_chars: int) -> list[str]:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []

    parts = [match.group(1).strip() for match in SPLIT_RE.finditer(text) if match.group(1).strip()]
    if not parts:
        return [text[index:index + max_chars] for index in range(0, len(text), max_chars)]

    chunks = []
    current = ""
    for part in parts:
        if len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(part[index:index + max_chars] for index in range(0, len(part), max_chars))
            continue
        if current and len(current) + len(part) > max_chars:
            chunks.append(current)
            current = part
        else:
            current += part
    if current:
        chunks.append(current)
    return chunks


def split_long_records(records: list[dict], max_chars: int) -> list[dict]:
    split_records = []
    for record in records:
        text = str(record.get("paragraph_text") or "")
        chunks = split_long_text(text, max_chars=max_chars)
        if len(chunks) <= 1:
            split_records.append(record)
            continue
        for index, chunk in enumerate(chunks, start=1):
            clone = dict(record)
            clone["paragraph_text"] = chunk
            clone["paragraph_char_count"] = len(chunk)
            clone["paragraph_id"] = f"{record.get('paragraph_id')}#s{index:03d}"
            clone["split_from_paragraph_id"] = record.get("paragraph_id")
            clone["split_part_index"] = index
            clone["split_part_total"] = len(chunks)
            split_records.append(clone)
    return split_records


def semantic_parent_record(records: list[dict], parent_index: int) -> dict:
    first = records[0]
    last = records[-1]
    text_parts = [
        str(record.get("paragraph_text") or "").strip()
        for record in records
        if str(record.get("paragraph_text") or "").strip()
    ]
    text = "\n\n".join(text_parts)
    parent_id = f"{first.get('source')}#sp{parent_index:06d}"
    return normalize_document_record({
        "source": first.get("source"),
        "book": first.get("book"),
        "article": first.get("article"),
        "section": first.get("section"),
        "paragraph_text": text,
        "paragraph_char_count": len(text),
        "paragraph_index": parent_index,
        "paragraph_id": parent_id,
        "semantic_parent_id": parent_id,
        "child_source_paragraph_ids": [record.get("paragraph_id") for record in records],
        "child_source_paragraph_start": first.get("paragraph_id"),
        "child_source_paragraph_end": last.get("paragraph_id"),
        "source_paragraph_count": len(records),
        "pdf_page_start": first.get("pdf_page_start"),
        "pdf_page_end": last.get("pdf_page_end"),
        "printed_page_start": first.get("printed_page_start"),
        "printed_page_end": last.get("printed_page_end"),
        "citation_page_start": first.get("citation_page_start"),
        "citation_page_end": last.get("citation_page_end"),
        "citation_page_type": first.get("citation_page_type"),
        "page_span": list(
            dict.fromkeys(
                page
                for record in records
                for page in (record.get("page_span") or [])
                if page is not None
            )
        ),
        "page_type": first.get("page_type"),
        "text_source": first.get("text_source"),
        "page_number_source": first.get("page_number_source"),
        "cleaning_reasons": first.get("cleaning_reasons"),
        "char_start": first.get("char_start"),
        "char_end": last.get("char_end"),
        "line_start": first.get("line_start"),
        "line_end": last.get("line_end"),
        "cross_page": any(bool(record.get("cross_page")) for record in records),
        "retrieval_unit": "semantic_parent",
    }, retrieval_unit="semantic_parent")


def build_semantic_parent_records(records: list[dict], max_chars: int) -> list[dict]:
    records = split_long_records(records, max_chars=max_chars)
    parents = []
    current: list[dict] = []
    parent_index = 1
    for record in records:
        if starts_new_parent(record, current, max_chars):
            parents.append(semantic_parent_record(current, parent_index))
            parent_index += 1
            current = []
        current.append(record)
    if current:
        parents.append(semantic_parent_record(current, parent_index))
    return parents


__all__ = [
    "build_semantic_child_documents",
    "build_semantic_parent_records",
    "split_long_records",
]
