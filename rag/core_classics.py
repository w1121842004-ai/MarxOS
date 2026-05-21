import json
import re
from functools import lru_cache
from pathlib import Path


DEFAULT_CORE_CLASSICS_PATH = Path(__file__).with_name("core_classics.json")
DEFAULT_CORE_BIBLIOGRAPHY_PATH = Path(__file__).with_name("core_bibliography_catalog.json")


def normalize_for_match(text):
    text = str(text or "")
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", text).lower()


@lru_cache(maxsize=4)
def load_core_classics(path=None):
    path = Path(path) if path else DEFAULT_CORE_CLASSICS_PATH

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        classics = json.load(f)

    for classic in classics:
        aliases = {classic.get("title", "")}
        aliases.update(classic.get("aliases") or [])
        classic["_normalized_aliases"] = {
            normalize_for_match(alias)
            for alias in aliases
            if normalize_for_match(alias)
        }
        classic["_normalized_quotes"] = [
            normalize_for_match(quote)
            for quote in classic.get("quotes") or []
            if normalize_for_match(quote)
        ]
        classic["entries"] = sorted(
            classic.get("entries") or [],
            key=lambda entry: entry.get("priority", 99),
        )

    return classics


@lru_cache(maxsize=4)
def load_core_bibliography(path=None):
    path = Path(path) if path else DEFAULT_CORE_BIBLIOGRAPHY_PATH

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        sections = json.load(f)

    classics_by_id = {classic.get("id"): classic for classic in load_core_classics()}
    normalized_sections = []
    for section in sections:
        works = []
        for work in sorted(section.get("works") or [], key=lambda item: item.get("priority", 99)):
            classic = classics_by_id.get(work.get("classic_id"))
            if not classic:
                continue
            works.append(
                {
                    "classic_id": classic.get("id"),
                    "title": classic.get("title"),
                    "author": classic.get("author"),
                    "work_year": classic.get("work_year"),
                    "work_type": classic.get("work_type"),
                    "entries": classic.get("entries") or [],
                    "priority": work.get("priority", 99),
                    "role": work.get("role", "foundation"),
                }
            )

        normalized_sections.append(
            {
                "id": section.get("id"),
                "label": section.get("label"),
                "description": section.get("description", ""),
                "works": works,
            }
        )

    return normalized_sections


def match_core_classic(text, path=None):
    normalized_text = normalize_for_match(text)

    if not normalized_text:
        return None

    for classic in load_core_classics(path):
        for alias in classic.get("_normalized_aliases", set()):
            if alias and (alias in normalized_text or normalized_text in alias):
                return classic

    for classic in load_core_classics(path):
        for quote in classic.get("_normalized_quotes", []):
            if quote and (quote in normalized_text or normalized_text in quote):
                return classic

    return None


def classic_entries_for_query(text, path=None):
    classic = match_core_classic(text, path)

    if not classic:
        return []

    return [
        {
            "source": entry["source"],
            "article": entry.get("article") or classic["title"],
            "start_page": entry["start_page"],
            "end_page": entry["end_page"],
            "classic_id": classic["id"],
            "classic_title": classic["title"],
            "classic_author": classic.get("author"),
            "classic_work_year": classic.get("work_year"),
            "classic_work_type": classic.get("work_type"),
            "entry_type": entry.get("entry_type"),
            "priority": entry.get("priority", 99),
        }
        for entry in classic.get("entries") or []
    ]
