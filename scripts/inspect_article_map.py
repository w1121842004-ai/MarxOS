from __future__ import annotations

import json
import os
from pathlib import Path

from marxos.config import get_settings

SETTINGS = get_settings()
ARTICLE_MAP_PATH = Path(os.getenv("ARTICLE_MAP_PATH", SETTINGS.corpus.article_map_path))


def load_article_map() -> dict:
    with ARTICLE_MAP_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    article_map = load_article_map()
    total_entries = 0
    missing_level = []
    missing_parent_key = []

    for source, payload in article_map.items():
        for entry in payload.get("entries", []):
            total_entries += 1

            if "level" not in entry:
                missing_level.append((source, entry.get("title")))

            if "parent" not in entry:
                missing_parent_key.append((source, entry.get("title")))

    print(f"sources: {len(article_map)}")
    print(f"entries: {total_entries}")
    print(f"missing level: {len(missing_level)}")
    print(f"missing parent key: {len(missing_parent_key)}")

    sample_source = "me20.pdf"
    sample_titles = {"自然辩证法", "［论文］", "运动的基本形式", "第三编社会主义", "二、理论"}
    print(f"\n[{sample_source}] hierarchy sample")

    for entry in article_map.get(sample_source, {}).get("entries", []):
        if entry.get("title") in sample_titles:
            print(
                f"level={entry.get('level')} parent={entry.get('parent')} "
                f"title={entry.get('title')} pages={entry.get('start_printed_page')}-{entry.get('end_printed_page')}"
            )


if __name__ == "__main__":
    main()
