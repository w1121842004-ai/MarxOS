"""Versioned metadata contract shared by document indexing stages."""

from __future__ import annotations

import re
import hashlib
import json
from collections import Counter
from typing import Iterable


DOCUMENT_RECORD_VERSION = "document-record/v2"
FRONT_MATTER_PAGE_TYPES = {"toc", "title_page", "blank"}
NON_BODY_PAGE_TYPES = FRONT_MATTER_PAGE_TYPES | {
    "notes", "person_index", "subject_index", "literature_index", "newspaper_index",
    "preface_editorial", "copyright", "bibliography", "appendix",
}
RETRIEVABLE_UNITS = {"paragraph", "semantic_parent", "semantic_child", "paragraph_child", "milvus_passage"}

_ARTICLE_POLLUTION_TERMS = (
    "目录", "注释", "人名索引", "名目索引", "主题索引", "文献索引", "报刊索引",
    "书名索引", "事项索引", "索引",
)
_UNKNOWN_ARTICLES = {"未知", "未知篇名", "未识别", "未识别篇名", "unknown", "n/a", "none"}
_MOJIBAKE_MARKERS = ("锟斤拷", "�", "Ã", "Â", "æ–", "å­", "ä¸", "ï¼", "â€")
_ARTICLE_PAGE_TYPES = {
    "注释": "notes",
    "人名索引": "person_index",
    "名目索引": "subject_index",
    "主题索引": "subject_index",
    "事项索引": "subject_index",
    "文献索引": "literature_index",
    "书名索引": "literature_index",
    "报刊索引": "newspaper_index",
    "目录": "toc",
    "索引": "subject_index",
}
_EDITORIAL_ARTICLE_TERMS = ("马克思主义理论研究和建设工程重点项目", "编辑说明", "编者说明")
_EDITORIAL_TEXT_HEADINGS = ("编辑说明", "编者说明", "出版说明")

LINEAGE_FIELDS = (
    "source", "book", "volume", "article", "section", "pdf_page_start", "pdf_page_end",
    "printed_page_start", "printed_page_end", "citation_page_start", "citation_page_end",
    "citation_page_type", "page_span", "page_type", "source_page_ids",
    "document_record_version", "text_source", "page_number_source", "cleaning_reasons",
)


def volume_from_book(book: object) -> str:
    match = re.search(r"第\s*([0-9一二三四五六七八九十百]+)\s*卷", str(book or ""))
    return f"第{match.group(1)}卷" if match else ""


def stable_paragraph_id(record: dict) -> str:
    identity = {
        "source": str(record.get("source") or ""),
        "spans": record.get("spans") or [],
        "text_sha256": hashlib.sha256(str(record.get("paragraph_text") or "").encode("utf-8")).hexdigest(),
        "segmentation_major": 2,
    }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "par_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def normalize_document_record(record: dict, retrieval_unit: str | None = None) -> dict:
    normalized = dict(record)
    unit = retrieval_unit or str(normalized.get("retrieval_unit") or "paragraph")
    paragraph_id = str(normalized.get("paragraph_id") or "")
    source = str(normalized.get("source") or "")
    pages = list(normalized.get("page_span") or [])
    if not pages and normalized.get("pdf_page_start") is not None:
        start = int(normalized["pdf_page_start"])
        end = int(normalized.get("pdf_page_end") or start)
        pages = list(range(start, end + 1)) if end >= start else [start]

    page_type = str(normalized.get("page_type") or "unknown")
    article_kind = _article_pollution_kind(normalized.get("article"))
    article_label = _normalized_label(normalized.get("article"))
    page_type_source = normalized.get("page_type_source") or "page_cache"
    if page_type == "body" and article_kind:
        page_type = _ARTICLE_PAGE_TYPES.get(article_kind, "subject_index")
        page_type_source = "article_policy"
    text = str(normalized.get("paragraph_text") or normalized.get("text") or "")
    text_heading = text.strip().splitlines()[0] if text.strip() else ""
    if page_type == "body" and (
        any(term in article_label for term in _EDITORIAL_ARTICLE_TERMS)
        or text_heading in _EDITORIAL_TEXT_HEADINGS
    ):
        page_type = "preface_editorial"
        page_type_source = "editorial_policy"
    quality_flags = list(normalized.get("quality_flags") or [])
    if any(marker in text for marker in _MOJIBAKE_MARKERS) and "mojibake" not in quality_flags:
        quality_flags.append("mojibake")
    retrievable = bool(normalized.get("retrievable", page_type == "body" and not quality_flags))

    normalized.update(
        {
            "document_record_version": DOCUMENT_RECORD_VERSION,
            "retrieval_unit": unit,
            "volume": normalized.get("volume") or volume_from_book(normalized.get("book")),
            "page_span": pages,
            "source_page_ids": normalized.get("source_page_ids")
            or [f"{source}#pdf{page}" for page in pages if source and page is not None],
            "page_type": page_type,
            "page_type_source": page_type_source,
            "quality_flags": quality_flags,
            "retrievable": retrievable,
        }
    )
    if unit == "paragraph" and paragraph_id:
        normalized["parent_paragraph_id"] = normalized.get("parent_paragraph_id") or paragraph_id
    return normalized


def inherit_record_metadata(parent: dict, child: dict, retrieval_unit: str) -> dict:
    inherited = {field: parent.get(field) for field in LINEAGE_FIELDS if field in parent}
    inherited.update(dict(child))
    inherited["parent_paragraph_id"] = parent.get("parent_paragraph_id") or parent.get("paragraph_id")
    return normalize_document_record(inherited, retrieval_unit=retrieval_unit)


def _issue(
    code: str,
    record: dict,
    field: str,
    message: str,
    severity: str = "error",
    policy: str = "repair_before_rebuild",
) -> dict:
    return {
        "code": code,
        "severity": severity,
        "record_id": record.get("paragraph_id") or record.get("id"),
        "source": record.get("source"),
        "retrieval_unit": record.get("retrieval_unit"),
        "pdf_page_start": record.get("pdf_page_start"),
        "pdf_page_end": record.get("pdf_page_end"),
        "field": field,
        "message": message,
        "policy": policy,
        "exempted": False,
    }


def _normalized_label(value: object) -> str:
    return re.sub(r"[\s·•.。…,:：;；—_\-]+", "", str(value or "")).lower()


def _article_pollution_kind(article: object) -> str | None:
    normalized = _normalized_label(article)
    return next((term for term in _ARTICLE_POLLUTION_TERMS if term in normalized), None)


def _looks_like_orphan_footnote(text: str) -> bool:
    if len(text.strip()) > 400:
        return False
    # Three-plus digit bracketed numbers in this corpus are commonly printed page
    # anchors (for example ［７７２］), not footnote markers.
    return bool(re.match(r"^\s*[\[［【]\s*\d{1,2}\s*[\]］】]", text))


def audit_document_records(records: Iterable[dict]) -> dict:
    materialized = [dict(record) for record in records]
    issues = []
    exemptions: Counter[str] = Counter()
    seen_text: dict[tuple[str, str], dict] = {}
    for record in materialized:
        legacy_pdf_page = record.get("pdf_page")
        if record.get("pdf_page_start") is None and legacy_pdf_page is not None:
            record = {
                **record,
                "pdf_page_start": legacy_pdf_page,
                "pdf_page_end": record.get("pdf_page_end", legacy_pdf_page),
            }
        unit = str(record.get("retrieval_unit") or "paragraph")
        page_type = str(record.get("page_type") or "body")
        eligible = unit in RETRIEVABLE_UNITS and record.get("retrievable", True) is not False
        if eligible and page_type in FRONT_MATTER_PAGE_TYPES:
            issues.append(_issue("FRONT_MATTER_LEAK", record, "page_type", "non-body page entered retrieval data"))
        text = str(record.get("paragraph_text") or record.get("text") or "")
        compact_text = re.sub(r"\s+", "", text)
        if unit in RETRIEVABLE_UNITS and not compact_text:
            issues.append(_issue("EMPTY_RETRIEVAL_TEXT", record, "paragraph_text", "retrieval record has no text"))
        if any(marker in text for marker in _MOJIBAKE_MARKERS):
            if not eligible and "mojibake" in (record.get("quality_flags") or []):
                exemptions["MOJIBAKE_REMAINS"] += 1
            else:
                issues.append(_issue("MOJIBAKE_REMAINS", record, "paragraph_text", "known mojibake marker remains"))
        article = str(record.get("article") or "").strip()
        article_pollution = _article_pollution_kind(article)
        if article_pollution:
            if page_type in NON_BODY_PAGE_TYPES:
                exemptions["ARTICLE_POLLUTION"] += 1
            else:
                issues.append(_issue(
                    "ARTICLE_POLLUTION", record, "article",
                    f"article metadata appears to name a non-body region ({article_pollution})",
                    policy="exclude_from_body_retrieval",
                ))
        if _looks_like_orphan_footnote(text):
            if page_type == "notes" or article_pollution == "注释":
                exemptions["FOOTNOTE_ORPHAN"] += 1
            else:
                issues.append(_issue(
                    "FOOTNOTE_ORPHAN", record, "paragraph_text",
                    "short body record begins with a detached numeric footnote marker",
                    severity="warning", policy="review_or_attach_to_parent",
                ))
        duplicate_key = (str(record.get("source") or ""), compact_text)
        if len(compact_text) >= 12 and duplicate_key in seen_text:
            previous = seen_text[duplicate_key]
            same_page = (
                record.get("pdf_page_start"), record.get("pdf_page_end")
            ) == (
                previous.get("pdf_page_start"), previous.get("pdf_page_end")
            )
            code = "DUPLICATE_TEXT" if same_page else "DUPLICATE_PAGE_TEXT"
            message = "normalized text duplicates an earlier record on the same page" if same_page else "normalized page text repeats on a different page"
            issues.append(_issue(code, record, "paragraph_text", message, severity="warning", policy="deduplicate_or_explain"))
        elif compact_text:
            seen_text[duplicate_key] = record
        if unit in RETRIEVABLE_UNITS and page_type == "body" and not article and not record.get("is_letter"):
            issues.append(_issue("ARTICLE_MISSING", record, "article", "body record has no article metadata"))
        elif unit in RETRIEVABLE_UNITS and page_type == "body" and _normalized_label(article) in _UNKNOWN_ARTICLES:
            issues.append(_issue("ARTICLE_UNKNOWN", record, "article", "body record has an unresolved article placeholder"))
        if unit in RETRIEVABLE_UNITS:
            start, end = record.get("pdf_page_start"), record.get("pdf_page_end")
            if start is None or end is None:
                issues.append(_issue("PAGE_REQUIRED", record, "pdf_page_start", "retrievable record requires a PDF page range"))
            else:
                try:
                    if int(end) < int(start):
                        issues.append(_issue("PAGE_RANGE_REVERSED", record, "pdf_page_end", "PDF page range is reversed"))
                except (TypeError, ValueError):
                    issues.append(_issue("PAGE_REQUIRED", record, "pdf_page_start", "PDF page range must be numeric"))

    def sortable_page(value: object) -> int:
        try:
            return int(value) if value is not None else -1
        except (TypeError, ValueError):
            return -1

    issues.sort(key=lambda item: (
        str(item.get("source") or ""),
        sortable_page(item.get("pdf_page_start")),
        str(item.get("record_id") or ""),
        item["code"],
    ))
    counts = Counter(issue["code"] for issue in issues)
    severity_counts = Counter(issue["severity"] for issue in issues)
    blocking_issues = severity_counts.get("error", 0)
    coverage_fields = (
        "source", "book", "article", "paragraph_id", "retrieval_unit",
        "pdf_page_start", "citation_page_start", "citation_page_type",
        "document_record_version",
    )
    coverage = {
        field: sum(record.get(field) not in (None, "") for record in materialized)
        for field in coverage_fields
    }
    return {
        "schema_version": "document-audit/v1",
        "policy_version": DOCUMENT_RECORD_VERSION,
        "summary": {
            "records": len(materialized),
            "issues": len(issues),
            "issues_by_code": dict(sorted(counts.items())),
            "issues_by_severity": dict(sorted(severity_counts.items())),
            "blocking_issues": blocking_issues,
            "exemptions": sum(exemptions.values()),
            "exemptions_by_code": dict(sorted(exemptions.items())),
            "passed": blocking_issues == 0,
            "readable": (
                f"{len(materialized)} records; {blocking_issues} blocking error"
                f"{'s' if blocking_issues != 1 else ''}; {severity_counts.get('warning', 0)} warning"
                f"{'s' if severity_counts.get('warning', 0) != 1 else ''}; {sum(exemptions.values())} exemptions"
            ),
            "field_coverage": coverage,
        },
        "issues": issues,
    }


__all__ = [
    "DOCUMENT_RECORD_VERSION",
    "LINEAGE_FIELDS",
    "audit_document_records",
    "inherit_record_metadata",
    "normalize_document_record",
    "stable_paragraph_id",
    "volume_from_book",
]
