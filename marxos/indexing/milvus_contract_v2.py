"""Canonical Milvus v2 storage contract.

This module deliberately has no dependency on the corpus builder.  Milvus is a
derived artifact: records are converted at this boundary and missing nullable
integers use ``-1`` only while stored in Milvus.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


CONTRACT_VERSION = "document-record/v2"
NULL_INT = -1
OPTIONAL_SPARSE_FIELD = "sparse_embedding"

STRING_FIELDS = (
    "id",
    "corpus",
    "author",
    "series",
    "book",
    "volume",
    "work_id",
    "article",
    "section",
    "edition_id",
    "publisher",
    "source",
    "source_file",
    "record_id",
    "paragraph_id",
    "parent_paragraph_id",
    "chunk_id",
    "retrieval_unit",
    "document_record_version",
    "retrieval_record_version",
    "metadata_schema_version",
    "build_id",
    "content_class",
    "quality_status",
    "chunker_version",
    "chunker_config_hash",
    "citation_page_type",
    "page_type",
    "citation_mode",
    "text_source",
    "source_text_hash",
    "indexed_text_hash",
    "page_span_json",
    "source_page_ids_json",
    "cleaning_reasons_json",
    "spans_json",
    "quality_flags_json",
    "bibliography_confidence_json",
    "bibliography_sources_json",
    "provenance_json",
    "text",
)

INT_FIELDS = (
    "publication_year",
    "pdf_page_start",
    "pdf_page_end",
    "printed_page_start",
    "printed_page_end",
    "citation_page_start",
    "citation_page_end",
    "child_chunk_index",
    "child_chunk_total",
    "child_char_start",
    "child_char_end",
    "child_chunk_size",
    "indexed_char_start",
    "indexed_char_end",
)

BOOL_FIELDS = ("is_letter", "text_was_clipped", "retrievable")
CORE_FIELD_NAMES = STRING_FIELDS + INT_FIELDS + BOOL_FIELDS
BASE_OUTPUT_FIELDS = CORE_FIELD_NAMES


@dataclass(frozen=True)
class SchemaBuildResult:
    created: bool
    output_fields: tuple[str, ...]


def text_sha256(text: Any) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def to_milvus_int(value: Any, default: int = NULL_INT) -> int:
    """Encode a nullable integer for Milvus, whose INT64 field is non-null."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def from_milvus_int(value: Any) -> int | None:
    """Decode the v2 null sentinel returned by Milvus."""
    parsed = to_milvus_int(value)
    return None if parsed == NULL_INT else parsed


def _canonical_json(value: Any, empty: Any) -> str:
    normalized = empty if value is None else value
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_record_id(record: Mapping[str, Any]) -> str:
    """Return an explicit immutable ID or derive one from source-local identity.

    The digest excludes display metadata such as title and book so correcting
    labels does not churn IDs.  It includes retrieval position so sibling child
    chunks remain distinct even when their text happens to be identical.
    """
    explicit = str(
        record.get("retrieval_id")
        or record.get("record_id")
        or record.get("retrieval_record_id")
        or ""
    ).strip()
    if explicit:
        return explicit
    identity = {
        "source": str(record.get("source") or record.get("source_file") or ""),
        "retrieval_unit": str(record.get("retrieval_unit") or "paragraph"),
        "pdf_page_start": record.get("pdf_page_start"),
        "pdf_page_end": record.get("pdf_page_end"),
        "spans": record.get("spans") or [],
        "segmentation_version": record.get("segmentation_version") or "paragraph-segmentation/v2",
        "parent_paragraph_id": record.get("parent_paragraph_id"),
        "child_char_start": record.get("child_char_start"),
        "child_char_end": record.get("child_char_end"),
        "text_hash": text_sha256(record.get("source_text") or record.get("paragraph_text") or record.get("text")),
    }
    digest = hashlib.sha256(_canonical_json(identity, {}).encode("utf-8")).hexdigest()[:32]
    return f"mr2_{digest}"


def _string(record: Mapping[str, Any], key: str, default: str = "") -> str:
    value = record.get(key)
    return default if value is None else str(value)


def row_from_record_v2(
    record: Mapping[str, Any],
    embedding: Sequence[float],
    indexed_text: str,
    sparse_embedding: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    """Map an authoritative/retrieval record to the canonical Milvus v2 row."""
    record_id = canonical_record_id(record)
    source_text = str(record.get("source_text") or record.get("paragraph_text") or record.get("text") or "")
    text = str(indexed_text or "")
    paragraph_id = _string(record, "paragraph_id", record_id)
    parent_id = _string(record, "parent_paragraph_id", paragraph_id)
    is_letter = bool(record.get("is_letter") or record.get("no_page_citation"))
    start = to_milvus_int(record.get("indexed_char_start"), 0)
    end = to_milvus_int(record.get("indexed_char_end"), start + len(text))
    provenance = record.get("provenance") or {}

    row: dict[str, Any] = {
        "id": record_id,
        "corpus": _string(record, "corpus", "marx_engels"),
        "author": _string(record, "author", "马克思恩格斯"),
        "series": _string(record, "series"),
        "book": _string(record, "book"),
        "volume": _string(record, "volume"),
        "work_id": _string(record, "work_id"),
        "article": _string(record, "article"),
        "section": _string(record, "section"),
        "edition_id": _string(record, "edition_id"),
        "publisher": _string(record, "publisher"),
        "source": _string(record, "source"),
        "source_file": _string(record, "source_file", _string(record, "source")),
        "record_id": record_id,
        "paragraph_id": paragraph_id,
        "parent_paragraph_id": parent_id,
        "chunk_id": _string(record, "chunk_id", record_id),
        "retrieval_unit": _string(record, "retrieval_unit", "paragraph"),
        # The row shape itself is v2 even when its source was migrated from v1.
        "document_record_version": CONTRACT_VERSION,
        "retrieval_record_version": _string(record, "retrieval_record_version", "retrieval-record/v2"),
        "metadata_schema_version": _string(record, "metadata_schema_version", "milvus-metadata/v2"),
        "build_id": _string(record, "build_id"),
        "content_class": _string(record, "content_class", _string(record, "page_type")),
        "quality_status": _string(record, "quality_status", "passed"),
        "chunker_version": _string(record, "chunker_version", "semantic-child/v2"),
        "chunker_config_hash": _string(record, "chunker_config_hash"),
        "citation_page_type": _string(record, "citation_page_type"),
        "page_type": _string(record, "page_type"),
        "citation_mode": _string(record, "citation_mode", "letter_title_only" if is_letter else "page"),
        "text_source": _string(record, "text_source"),
        "source_text_hash": text_sha256(source_text),
        "indexed_text_hash": text_sha256(text),
        "page_span_json": _canonical_json(record.get("page_span"), []),
        "source_page_ids_json": _canonical_json(record.get("source_page_ids"), []),
        "cleaning_reasons_json": _canonical_json(record.get("cleaning_reasons"), []),
        "spans_json": _canonical_json(record.get("spans"), []),
        "quality_flags_json": _canonical_json(record.get("quality_flags"), []),
        "bibliography_confidence_json": _canonical_json(record.get("bibliography_confidence"), {}),
        "bibliography_sources_json": _canonical_json(record.get("bibliography_sources"), {}),
        "provenance_json": _canonical_json(provenance, {}),
        "text": text,
        "publication_year": to_milvus_int(record.get("publication_year")),
        "pdf_page_start": to_milvus_int(record.get("pdf_page_start")),
        "pdf_page_end": to_milvus_int(record.get("pdf_page_end")),
        "printed_page_start": to_milvus_int(record.get("printed_page_start")),
        "printed_page_end": to_milvus_int(record.get("printed_page_end")),
        "citation_page_start": to_milvus_int(record.get("citation_page_start")),
        "citation_page_end": to_milvus_int(record.get("citation_page_end")),
        "child_chunk_index": to_milvus_int(record.get("child_chunk_index"), 1),
        "child_chunk_total": to_milvus_int(record.get("child_chunk_total"), 1),
        "child_char_start": to_milvus_int(record.get("child_char_start")),
        "child_char_end": to_milvus_int(record.get("child_char_end")),
        "child_chunk_size": to_milvus_int(record.get("child_chunk_size"), len(text)),
        "indexed_char_start": start,
        "indexed_char_end": end,
        "is_letter": is_letter,
        "text_was_clipped": bool(record.get("text_was_clipped", text != source_text)),
        "retrievable": bool(record.get("retrievable", True)),
        "embedding": list(embedding),
    }
    if sparse_embedding is not None:
        row[OPTIONAL_SPARSE_FIELD] = dict(sparse_embedding)
    return row


def _load_data_type():
    try:
        from pymilvus import DataType
    except ImportError as exc:  # pragma: no cover - exercised only by integration callers
        raise RuntimeError("pymilvus is required to create a Milvus schema") from exc
    return DataType


def create_v2_schema(
    client: Any,
    collection_name: str,
    dim: int,
    *,
    data_type: Any = None,
    drop_existing: bool = False,
    enable_sparse: bool = False,
) -> SchemaBuildResult:
    """Create the v2 schema and return its canonical query output fields."""
    if dim <= 0:
        raise ValueError("embedding dimension must be positive")
    output_fields = BASE_OUTPUT_FIELDS + ((OPTIONAL_SPARSE_FIELD,) if enable_sparse else ())
    if client.has_collection(collection_name):
        if not drop_existing:
            return SchemaBuildResult(created=False, output_fields=output_fields)
        client.drop_collection(collection_name)

    dtype = data_type or _load_data_type()
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    for field in STRING_FIELDS:
        max_length = 65535 if field in {"text", "provenance_json", "spans_json"} else 4096
        schema.add_field(field, dtype.VARCHAR, is_primary=field == "id", max_length=max_length)
    for field in INT_FIELDS:
        schema.add_field(field, dtype.INT64)
    for field in BOOL_FIELDS:
        schema.add_field(field, dtype.BOOL)
    schema.add_field("embedding", dtype.FLOAT_VECTOR, dim=dim)
    if enable_sparse:
        schema.add_field(OPTIONAL_SPARSE_FIELD, dtype.SPARSE_FLOAT_VECTOR)

    indexes = client.prepare_index_params()
    indexes.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    if enable_sparse:
        indexes.add_index(
            field_name=OPTIONAL_SPARSE_FIELD,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )
    client.create_collection(collection_name=collection_name, schema=schema, index_params=indexes)
    return SchemaBuildResult(created=True, output_fields=output_fields)


__all__ = [
    "BASE_OUTPUT_FIELDS",
    "BOOL_FIELDS",
    "CONTRACT_VERSION",
    "CORE_FIELD_NAMES",
    "INT_FIELDS",
    "NULL_INT",
    "OPTIONAL_SPARSE_FIELD",
    "STRING_FIELDS",
    "SchemaBuildResult",
    "canonical_record_id",
    "create_v2_schema",
    "from_milvus_int",
    "row_from_record_v2",
    "text_sha256",
    "to_milvus_int",
]
