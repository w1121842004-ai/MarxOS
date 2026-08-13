"""Deterministic bibliography enrichment for authoritative v2 records.

The work catalog is authoritative for editions and canonical works.  The
article map can identify an edition-local article, but is deliberately not
allowed to invent a canonical ``work_id``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping


BIBLIOGRAPHY_MATCH_VERSION = "bibliography-match/v2"
_NON_BODY_TYPES = {
    "blank",
    "toc",
    "title_page",
    "notes",
    "person_index",
    "subject_index",
    "literature_index",
    "newspaper_index",
}
_NON_BODY_TITLES = {"目录", "注释", "注解", "编者说明"}
_NON_BODY_TITLE_SUFFIXES = ("人名索引", "名目索引", "文献索引", "报刊索引", "主题索引")
_BIBLIOGRAPHY_FIELDS = (
    "series",
    "volume",
    "edition_id",
    "publisher",
    "publication_year",
    "work_id",
    "article_id",
)


def _normalize_title(value: Any) -> str:
    return re.sub(r"[\s《》〈〉「」『』【】\[\]（）()·,，。:：;；—－_-]", "", str(value or ""))


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _record_page(record: Mapping[str, Any]) -> int | None:
    printed = _integer(record.get("printed_page_start"))
    if printed is not None:
        return printed
    if record.get("citation_page_type") == "printed_page":
        return _integer(record.get("citation_page_start") or record.get("citation_page"))
    return None


def _edition_id(key: str, edition: Mapping[str, Any]) -> str:
    year = _integer(edition.get("year"))
    value = f"{key}-{year}" if year is not None else str(key)
    declared = str(edition.get("edition") or "")
    match = re.search(r"第\s*(\d+)\s*版", declared)
    return f"{value}-{match.group(1)}e" if match else value


def _article_id(source: str, title: str, start: int, end: int) -> str:
    identity = "\x1f".join((source, _normalize_title(title), str(start), str(end)))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return f"article_v2_{digest}"


def _is_non_body(record: Mapping[str, Any]) -> bool:
    page_type = str(record.get("page_type") or "").strip().lower()
    article = _normalize_title(record.get("article"))
    polluted_title = article in _NON_BODY_TITLES or article.endswith(_NON_BODY_TITLE_SUFFIXES)
    return page_type in _NON_BODY_TYPES or polluted_title


@dataclass(frozen=True)
class _EditionMatch:
    key: str
    payload: Mapping[str, Any]
    volume: str


@dataclass(frozen=True)
class _ArticleMatch:
    title: str
    start: int
    end: int
    work_id: str = ""
    canonical_title: str = ""


class BibliographyIndex:
    """Immutable lookup index built from versioned catalog mappings."""

    def __init__(self, work_catalog: Mapping[str, Any], article_map: Mapping[str, Any]):
        self._editions_by_source = self._index_editions(work_catalog)
        self._works_by_source = self._index_works(work_catalog)
        self._articles_by_source = self._index_articles(article_map)

    @staticmethod
    def _index_editions(work_catalog: Mapping[str, Any]) -> dict[str, _EditionMatch]:
        matches: dict[str, _EditionMatch] = {}
        for key, payload in (work_catalog.get("editions") or {}).items():
            if not isinstance(payload, Mapping):
                continue
            for volume, volume_data in (payload.get("volume_map") or {}).items():
                if not isinstance(volume_data, Mapping):
                    continue
                source = str(volume_data.get("source") or "").strip()
                if source and source not in matches:
                    matches[source] = _EditionMatch(str(key), payload, f"第{volume}卷")
        return matches

    @staticmethod
    def _index_works(work_catalog: Mapping[str, Any]) -> dict[str, tuple[_ArticleMatch, ...]]:
        matches: dict[str, list[_ArticleMatch]] = {}
        for work in work_catalog.get("works") or []:
            if not isinstance(work, Mapping):
                continue
            work_id = str(work.get("work_id") or "").strip()
            aliases = tuple(str(value) for value in (work.get("aliases") or []))
            canonical_title = str(work.get("title") or "")
            for edition in (work.get("editions") or {}).values():
                if not isinstance(edition, Mapping):
                    continue
                source = str(edition.get("source") or "").strip()
                start = _integer(edition.get("start_page"))
                end = _integer(edition.get("end_page"))
                title = str(edition.get("article_title") or canonical_title).strip()
                if not source or start is None or end is None or not title or not work_id:
                    continue
                all_titles = (title, canonical_title, *aliases)
                for candidate_title in dict.fromkeys(all_titles):
                    if candidate_title:
                        matches.setdefault(source, []).append(
                            _ArticleMatch(candidate_title, start, end, work_id, title)
                        )
        return {
            source: tuple(sorted(values, key=lambda item: (item.start, item.end, item.work_id, item.title)))
            for source, values in matches.items()
        }

    @staticmethod
    def _index_articles(article_map: Mapping[str, Any]) -> dict[str, tuple[_ArticleMatch, ...]]:
        matches: dict[str, tuple[_ArticleMatch, ...]] = {}
        for source, source_map in article_map.items():
            if not isinstance(source_map, Mapping):
                continue
            values: list[_ArticleMatch] = []
            for entry in source_map.get("entries") or []:
                if not isinstance(entry, Mapping):
                    continue
                title = str(entry.get("title") or "").strip()
                start = _integer(entry.get("start_printed_page"))
                end = _integer(entry.get("end_printed_page"))
                if title and start is not None and end is not None:
                    values.append(_ArticleMatch(title, start, end))
            matches[str(source)] = tuple(sorted(values, key=lambda item: (item.start, item.end, item.title)))
        return matches

    @staticmethod
    def _find_article(
        candidates: tuple[_ArticleMatch, ...], article: str, page: int | None
    ) -> _ArticleMatch | None:
        normalized = _normalize_title(article)
        if not normalized or page is None:
            return None
        exact = [
            candidate
            for candidate in candidates
            if candidate.start <= page <= candidate.end
            and _normalize_title(candidate.title) == normalized
        ]
        if not exact:
            return None
        return min(exact, key=lambda item: (item.end - item.start, item.start, item.title, item.work_id))

    @staticmethod
    def _find_unique_work_by_page(
        candidates: tuple[_ArticleMatch, ...], page: int | None
    ) -> _ArticleMatch | None:
        if page is None:
            return None
        by_work: dict[str, _ArticleMatch] = {}
        for candidate in candidates:
            if candidate.work_id and candidate.start <= page <= candidate.end:
                by_work.setdefault(candidate.work_id, candidate)
        return next(iter(by_work.values())) if len(by_work) == 1 else None

    def enrich(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Return a new record with deterministic bibliography values and evidence."""
        source = str(record.get("source") or record.get("source_file") or "").strip()
        article = str(record.get("article") or "").strip()
        page = _record_page(record)
        values: dict[str, Any] = {
            "series": "",
            "volume": "",
            "edition_id": "",
            "publisher": "",
            "publication_year": None,
            "work_id": "",
            "article_id": "",
        }
        confidence = {field: 0.0 for field in _BIBLIOGRAPHY_FIELDS}
        sources = {field: "unknown" for field in _BIBLIOGRAPHY_FIELDS}

        edition = self._editions_by_source.get(source)
        if edition is not None:
            edition_values = {
                "series": str(edition.payload.get("name") or ""),
                "volume": edition.volume,
                "edition_id": _edition_id(edition.key, edition.payload),
                "publisher": str(edition.payload.get("publisher") or ""),
                "publication_year": _integer(edition.payload.get("year")),
            }
            for field, value in edition_values.items():
                values[field] = value
                confidence[field] = 1.0 if value not in (None, "") else 0.0
                sources[field] = "work_catalog:edition_source" if value not in (None, "") else "unknown"

        if _is_non_body(record):
            for field in ("work_id", "article_id"):
                sources[field] = "policy:non_body"
            return {
                **record,
                **values,
                "bibliography_match_version": BIBLIOGRAPHY_MATCH_VERSION,
                "bibliography_confidence": confidence,
                "bibliography_sources": sources,
            }

        work_candidates = self._works_by_source.get(source, ())
        work = self._find_article(work_candidates, article, page)
        work_source = "work_catalog:source+printed_page+article"
        work_confidence = 1.0
        if work is None:
            work = self._find_unique_work_by_page(work_candidates, page)
            work_source = "work_catalog:unique_source_page"
            work_confidence = 0.9
        article_match = self._find_article(self._articles_by_source.get(source, ()), article, page)
        selected = work or article_match
        if work is not None:
            values["work_id"] = work.work_id
            confidence["work_id"] = work_confidence
            sources["work_id"] = work_source
        if selected is not None:
            values["article_id"] = _article_id(
                source, selected.canonical_title or selected.title, selected.start, selected.end
            )
            confidence["article_id"] = work_confidence if work is not None else 0.95
            sources["article_id"] = (
                work_source if work is not None else "article_map:exact_range"
            )

        return {
            **record,
            **values,
            "bibliography_match_version": BIBLIOGRAPHY_MATCH_VERSION,
            "bibliography_confidence": confidence,
            "bibliography_sources": sources,
        }
