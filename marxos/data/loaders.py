from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from marxos.config import get_settings


def read_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> Iterable[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_merged_article_map(
    primary_path: str | Path | None = None,
    extra_paths: str | Iterable[str | Path] | None = None,
) -> dict:
    settings = get_settings()
    primary = str(primary_path or settings.corpus.article_map_path)
    extras = extra_paths if extra_paths is not None else settings.corpus.article_map_extra_paths
    if isinstance(extras, str):
        extra_list = [path for path in extras.split(os.pathsep) if path]
    else:
        extra_list = list(extras or [])

    merged = {}
    for index, path in enumerate([primary, *extra_list]):
        data = read_json(path, default={})
        if not isinstance(data, dict):
            continue
        if index == 0:
            merged.update(data)
        else:
            for source, payload in data.items():
                merged.setdefault(source, payload)
    return merged


def load_topic_catalog(path: str | Path | None = None) -> list:
    settings = get_settings()
    data = read_json(path or settings.corpus.topic_catalog_path, default=[])
    return data if isinstance(data, list) else []

