from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app


OCR_CACHE_DIR = Path(os.getenv("OCR_CACHE_DIR", app.OCR_CACHE_DIR))
OUTPUT_PATH = Path(os.getenv("PAGE_MAP_PATH", "data/page_map.json"))
TARGET_PDFS = {
    item.strip()
    for item in os.getenv("TARGET_PDFS", "").split(",")
    if item.strip()
}


def page_num_from_path(path: Path) -> int | None:
    match = re.search(r"page_(\d+)\.json$", path.name)
    return int(match.group(1)) if match else None


def iter_sources() -> list[Path]:
    sources = [path for path in OCR_CACHE_DIR.iterdir() if path.is_dir()]
    if TARGET_PDFS:
        stems = {source.replace(".pdf", "") for source in TARGET_PDFS}
        sources = [path for path in sources if path.name in stems]
    return sorted(sources, key=lambda path: path.name)


def build_source_map(source_dir: Path) -> dict:
    source = f"{source_dir.name}.pdf"
    pages = {}
    total = 0
    mapped = 0

    for page_path in sorted(source_dir.glob("page_*.json"), key=lambda path: page_num_from_path(path) or 0):
        pdf_page = page_num_from_path(page_path)
        if pdf_page is None:
            continue
        total += 1
        printed_page = app.infer_printed_page_from_ocr_cache({"source": source, "pdf_page": pdf_page})
        try:
            payload = json.loads(page_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if printed_page is not None:
            mapped += 1
        pages[str(pdf_page)] = {
            "pdf_page": pdf_page,
            "printed_page": printed_page,
            "page_type": payload.get("page_type"),
            "source": "ocr_margin" if printed_page is not None else None,
        }

    return {
        "source": source,
        "total_pages": total,
        "mapped_pages": mapped,
        "pages": pages,
    }


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "version": 1,
        "ocr_cache_dir": OCR_CACHE_DIR.as_posix(),
        "sources": {},
    }

    for source_dir in iter_sources():
        source_map = build_source_map(source_dir)
        result["sources"][source_map["source"]] = source_map
        print(
            f"{source_map['source']}: {source_map['mapped_pages']}/{source_map['total_pages']} pages mapped",
            flush=True,
        )

    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"page map written: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
