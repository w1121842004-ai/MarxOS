import json
import re
from functools import lru_cache
from pathlib import Path


DEFAULT_CORE_CLASSICS_PATH = Path(__file__).with_name("core_classics.json")


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
