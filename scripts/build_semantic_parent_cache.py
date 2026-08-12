from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from marxos.config import get_settings
from marxos.data.splitters import build_semantic_parent_records, split_long_records
from rag.paragraph_cache import read_paragraph_cache


SETTINGS = get_settings()
DEFAULT_INPUT = ROOT_DIR / SETTINGS.corpus.paragraph_cache_path
DEFAULT_OUTPUT = ROOT_DIR / SETTINGS.corpus.semantic_parent_cache_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build semantic parent cache from paragraph cache.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-chars", type=int, default=1000)
    args = parser.parse_args()

    records = read_paragraph_cache(args.input)
    parents = build_semantic_parent_records(records, max_chars=args.max_chars)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in parents:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "input": args.input,
        "output": str(output),
        "paragraphs": len(records),
        "paragraph_units_after_long_split": sum(1 for _ in split_long_records(records, max_chars=args.max_chars)),
        "semantic_parents": len(parents),
        "max_chars": args.max_chars,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
