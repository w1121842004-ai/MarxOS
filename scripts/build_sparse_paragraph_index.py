from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.semantic_retrieval import (  # noqa: E402
    DEFAULT_LIGHT_SPARSE_INDEX_PATH,
    DEFAULT_PARAGRAPH_CACHE_PATH,
    write_light_sparse_paragraph_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build lightweight sparse paragraph index.")
    parser.add_argument("--paragraph-cache", default=str(DEFAULT_PARAGRAPH_CACHE_PATH))
    parser.add_argument("--output", default=str(DEFAULT_LIGHT_SPARSE_INDEX_PATH))
    args = parser.parse_args()

    started = time.perf_counter()
    summary = write_light_sparse_paragraph_index(
        output_path=args.output,
        paragraph_cache_path=args.paragraph_cache,
    )
    summary["elapsed_sec"] = round(time.perf_counter() - started, 3)
    summary["size_mb"] = round(summary["size_bytes"] / 1024 / 1024, 2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
