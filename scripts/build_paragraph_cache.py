from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from marxos.config import get_settings
from rag.paragraph_cache import paragraph_cache_sources, write_paragraph_cache


SETTINGS = get_settings()
DEFAULT_OUTPUT = Path(SETTINGS.corpus.paragraph_cache_path)


def main() -> None:
    output_path = Path(os.getenv("PARAGRAPH_CACHE_PATH", str(DEFAULT_OUTPUT)))
    sources = paragraph_cache_sources()
    summary = write_paragraph_cache(output_path, sources=sources)

    print(f"paragraph cache written: {output_path}")
    print(f"sources: {len(summary['sources'])}")
    print(f"paragraphs: {summary['paragraphs']}")
    for source, count in summary["sources"].items():
        print(f"{source}: {count}")


if __name__ == "__main__":
    main()
