"""Deterministic, traceable retrieval records derived from ParagraphRecord v2."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Iterator


RETRIEVAL_RECORD_VERSION = "retrieval-record/v2"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_retrieval_id(parent_id: str, start: int, end: int, text: str) -> str:
    """Return a content-sensitive ID independent of iteration order."""
    identity = {
        "parent_paragraph_id": str(parent_id),
        "parent_char_start": int(start),
        "parent_char_end": int(end),
        "indexed_text_sha256": _sha256(text),
        "retrieval_record_version": RETRIEVAL_RECORD_VERSION,
    }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "ret_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _validate_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")


def _chunk_offsets(text_length: int, chunk_size: int, chunk_overlap: int) -> Iterator[tuple[int, int]]:
    step = chunk_size - chunk_overlap
    start = 0
    while start < text_length:
        end = min(start + chunk_size, text_length)
        yield start, end
        if end == text_length:
            break
        start += step


def build_semantic_child_records(
    paragraph_records: Iterable[dict],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> Iterator[dict]:
    """Yield semantic children whose offsets map exactly to the parent text.

    Input mappings and their nested metadata are never mutated. Records explicitly
    marked ``retrievable=false`` and records without text are intentionally omitted.
    """
    _validate_chunk_parameters(chunk_size, chunk_overlap)
    for input_record in paragraph_records:
        if input_record.get("retrievable") is False:
            continue
        source_text = str(input_record.get("paragraph_text") or input_record.get("text") or "")
        if not source_text:
            continue
        parent_id = str(input_record.get("paragraph_id") or "")
        if not parent_id:
            raise ValueError("retrievable paragraph record is missing paragraph_id")

        source_hash = _sha256(source_text)
        offsets = tuple(_chunk_offsets(len(source_text), chunk_size, chunk_overlap))
        child_total = len(offsets)
        for child_index, (start, end) in enumerate(offsets):
            indexed_text = source_text[start:end]
            retrieval_id = stable_retrieval_id(parent_id, start, end, indexed_text)
            child = copy.deepcopy(input_record)
            child.update(
                {
                    "retrieval_record_version": RETRIEVAL_RECORD_VERSION,
                    "retrieval_unit": "semantic_child",
                    "retrieval_id": retrieval_id,
                    "id": retrieval_id,
                    "parent_paragraph_id": parent_id,
                    "child_index": child_index,
                    "child_total": child_total,
                    "child_chunk_index": child_index,
                    "child_chunk_total": child_total,
                    "paragraph_text": indexed_text,
                    "text": indexed_text,
                    "parent_char_start": start,
                    "parent_char_end": end,
                    "child_char_start": start,
                    "child_char_end": end,
                    "child_chunk_size": len(indexed_text),
                    "indexed_char_start": start,
                    "indexed_char_end": end,
                    "source_char_count": len(source_text),
                    "indexed_char_count": len(indexed_text),
                    "source_text_sha256": source_hash,
                    "indexed_text_sha256": _sha256(indexed_text),
                    "text_was_clipped": False,
                }
            )
            yield child


__all__ = [
    "RETRIEVAL_RECORD_VERSION",
    "build_semantic_child_records",
    "stable_retrieval_id",
]
