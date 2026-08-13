from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from marxos.config import get_settings

SETTINGS = get_settings()
PAGE_MAP_PATH = ROOT_DIR / os.getenv("PAGE_MAP_PATH", SETTINGS.corpus.page_map_path)
ARTICLE_MAP_PATH = ROOT_DIR / os.getenv("ARTICLE_MAP_PATH", SETTINGS.corpus.article_map_path)


PAGE_ENTRY_SCHEMA = {
    "type": "object",
    "required": ["pdf_page", "printed_page", "page_type", "source"],
    "properties": {
        "pdf_page": {"type": "integer", "minimum": 1},
        "printed_page": {"type": ["integer", "null"], "minimum": 1},
        "page_type": {"type": ["string", "null"]},
        "source": {"type": ["string", "null"]},
    },
    "additionalProperties": True,
}

PAGE_SOURCE_SCHEMA = {
    "type": "object",
    "required": ["source", "total_pages", "mapped_pages", "pages"],
    "properties": {
        "source": {"type": "string", "minLength": 1},
        "total_pages": {"type": "integer", "minimum": 0},
        "mapped_pages": {"type": "integer", "minimum": 0},
        "pages": {
            "type": "object",
            "additionalProperties": PAGE_ENTRY_SCHEMA,
        },
    },
    "additionalProperties": True,
}

PAGE_MAP_SCHEMA = {
    "type": "object",
    "required": ["version", "ocr_cache_dir", "sources"],
    "properties": {
        "version": {"type": "integer", "minimum": 1},
        "ocr_cache_dir": {"type": "string", "minLength": 1},
        "sources": {
            "type": "object",
            "additionalProperties": PAGE_SOURCE_SCHEMA,
        },
    },
    "additionalProperties": True,
}

ARTICLE_ENTRY_SCHEMA = {
    "type": "object",
    "required": [
        "title",
        "start_printed_page",
        "end_printed_page",
        "level",
        "parent",
    ],
    "properties": {
        "title": {"type": "string"},
        "start_printed_page": {"type": "integer", "minimum": 1},
        "end_printed_page": {"type": "integer", "minimum": 1},
        "level": {"type": "integer", "minimum": 1},
        "parent": {"type": ["string", "null"]},
    },
    "additionalProperties": True,
}

ARTICLE_SOURCE_SCHEMA = {
    "type": "object",
    "required": ["book", "entries"],
    "properties": {
        "book": {"type": "string"},
        "entries": {
            "type": "array",
            "items": ARTICLE_ENTRY_SCHEMA,
        },
    },
    "additionalProperties": True,
}

ARTICLE_MAP_SCHEMA = {
    "type": "object",
    "additionalProperties": ARTICLE_SOURCE_SCHEMA,
}


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_validation_errors(schema: dict, payload: object):
    validator = Draft202012Validator(schema)
    return sorted(validator.iter_errors(payload), key=lambda error: list(error.path))


def format_path(error) -> str:
    if not error.path:
        return "<root>"
    return ".".join(str(part) for part in error.path)


def validate_named_map(name: str, path: Path, schema: dict) -> int:
    if not path.exists():
        print(f"[FAIL] {name}: file not found at {path}")
        return 1

    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] {name}: could not load JSON from {path}: {exc}")
        return 1

    errors = iter_validation_errors(schema, payload)
    if errors:
        print(f"[FAIL] {name}: schema validation failed for {path}")
        for error in errors[:10]:
            print(f"  - {format_path(error)}: {error.message}")
        if len(errors) > 10:
            print(f"  - ... {len(errors) - 10} more errors omitted")
        return 1

    print(f"[PASS] {name}: schema valid ({path})")
    if name == "page_map":
        source_count = len((payload.get("sources") or {}))  # type: ignore[union-attr]
        print(f"       sources={source_count}")
    elif name == "article_map":
        source_count = len(payload)  # type: ignore[arg-type]
        print(f"       sources={source_count}")
    return 0


def main() -> int:
    checks = [
        ("page_map", PAGE_MAP_PATH, PAGE_MAP_SCHEMA),
        ("article_map", ARTICLE_MAP_PATH, ARTICLE_MAP_SCHEMA),
    ]
    failed = 0
    for name, path, schema in checks:
        failed += validate_named_map(name, path, schema)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
