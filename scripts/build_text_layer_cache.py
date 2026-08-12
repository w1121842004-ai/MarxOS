from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import fitz

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.academic_text_cleaner import clean_academic_page
from rag.build_vectorstore_from_cache import BOOK_MAPPING


DATA_DIR = Path(os.getenv("PDF_DATA_DIR", "data/marx_engels全集"))
OUTPUT_DIR = Path(os.getenv("TEXT_LAYER_CACHE_DIR", "data/ocr_cache_text_layer"))
LEGACY_OCR_CACHE_DIR = Path(os.getenv("LEGACY_OCR_CACHE_DIR", "data/ocr_cache"))
START_PAGE = int(os.getenv("START_PAGE", "1"))
END_PAGE = os.getenv("END_PAGE")
END_PAGE = int(END_PAGE) if END_PAGE else None
PDF_NAME = os.getenv("PDF_NAME", "").strip()
TARGET_PDFS = {
    item.strip()
    for item in os.getenv("TARGET_PDFS", "").split(",")
    if item.strip()
}
SKIP_PDFS = {
    item.strip()
    for item in os.getenv("SKIP_PDFS", "capital.pdf").split(",")
    if item.strip()
}
OVERWRITE = os.getenv("OVERWRITE_TEXT_CACHE") == "1"
MIN_TEXT_CHARS = int(os.getenv("TEXT_LAYER_MIN_LENGTH", "80"))
PROGRESS_EVERY = int(os.getenv("PROGRESS_EVERY", "100"))
ME_VOLUMES_ONLY = os.getenv("ME_VOLUMES_ONLY") == "1"
FALLBACK_LEGACY_SHORT_TEXT = os.getenv("FALLBACK_LEGACY_SHORT_TEXT", "1") != "0"


def is_me_volume(filename: str) -> bool:
    stem = filename.lower().replace(".pdf", "")
    if not re.fullmatch(r"me\d{2}[abc]?", stem):
        return False
    return 1 <= int(stem[2:4]) <= 50


def should_process_pdf(filename: str) -> bool:
    if PDF_NAME and filename != PDF_NAME:
        return False
    if TARGET_PDFS and filename not in TARGET_PDFS:
        return False
    if ME_VOLUMES_ONLY and not is_me_volume(filename):
        return False
    if filename in SKIP_PDFS:
        return False
    return filename.endswith(".pdf")


def iter_pdf_paths() -> list[Path]:
    return sorted(path for path in DATA_DIR.rglob("*.pdf") if should_process_pdf(path.name))


def page_paths(source: str, page_num: int) -> tuple[Path, Path]:
    stem = source.replace(".pdf", "")
    base = OUTPUT_DIR / stem
    return base / f"page_{page_num}.txt", base / f"page_{page_num}.json"


def write_page(source: str, page_num: int, payload: dict) -> None:
    txt_path, json_path = page_paths(source, page_num)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(payload.get("cleaned_text") or "", encoding="utf-8", newline="\n")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_legacy_page_text(source: str, page_num: int) -> str:
    stem = source.replace(".pdf", "")
    json_path = LEGACY_OCR_CACHE_DIR / stem / f"page_{page_num}.json"
    txt_path = LEGACY_OCR_CACHE_DIR / stem / f"page_{page_num}.txt"
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            return payload.get("raw_text") or payload.get("cleaned_text") or ""
        except (OSError, json.JSONDecodeError):
            return ""
    if txt_path.exists():
        try:
            return txt_path.read_text(encoding="utf-8")
        except OSError:
            return ""
    return ""


def add_legacy_page_candidates(payload: dict, source: str, page_num: int, book_title: str) -> None:
    if payload.get("page_number_candidates"):
        return
    legacy_text = read_legacy_page_text(source, page_num)
    if not legacy_text:
        return
    legacy = clean_academic_page(
        legacy_text,
        source=source,
        page_num=page_num,
        book_title=book_title,
    )
    candidates = legacy.get("page_number_candidates") or []
    if not candidates:
        return
    payload["page_number_candidates"] = candidates
    payload["legacy_footer_text"] = legacy.get("footer_text") or ""
    payload.setdefault("reasons", []).append("legacy_page_number_candidates")


def write_legacy_fallback_page(source: str, page_num: int, pdf_path: Path, book_title: str) -> bool:
    legacy_text = read_legacy_page_text(source, page_num)
    if not legacy_text:
        return False
    payload = clean_academic_page(
        legacy_text,
        source=source,
        page_num=page_num,
        book_title=book_title,
    )
    if not (payload.get("cleaned_text") or "").strip():
        return False
    payload["text_source"] = "legacy_ocr_fallback"
    payload["pdf_path"] = str(pdf_path)
    payload.setdefault("reasons", []).append("text_layer_short_legacy_fallback")
    write_page(source, page_num, payload)
    return True


def build_pdf_cache(pdf_path: Path) -> dict:
    source = pdf_path.name
    summary = {
        "source": source,
        "pages_total": 0,
        "pages_written": 0,
        "pages_skipped_existing": 0,
        "pages_skipped_short_text": 0,
        "pages_legacy_fallback": 0,
    }
    book_title = BOOK_MAPPING.get(source, source)

    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count
        summary["pages_total"] = total_pages
        last_page = min(END_PAGE or total_pages, total_pages)
        if START_PAGE > last_page:
            return summary

        for page_num in range(START_PAGE, last_page + 1):
            txt_path, json_path = page_paths(source, page_num)
            if txt_path.exists() and json_path.exists() and not OVERWRITE:
                summary["pages_skipped_existing"] += 1
                continue

            raw_text = doc.load_page(page_num - 1).get_text("text")
            if len(re.sub(r"\s+", "", raw_text or "")) < MIN_TEXT_CHARS:
                if FALLBACK_LEGACY_SHORT_TEXT and write_legacy_fallback_page(source, page_num, pdf_path, book_title):
                    summary["pages_written"] += 1
                    summary["pages_legacy_fallback"] += 1
                else:
                    summary["pages_skipped_short_text"] += 1
                continue

            payload = clean_academic_page(
                raw_text,
                source=source,
                page_num=page_num,
                book_title=book_title,
            )
            add_legacy_page_candidates(payload, source, page_num, book_title)
            payload["text_source"] = "pdf_text_layer"
            payload["pdf_path"] = str(pdf_path)
            write_page(source, page_num, payload)
            summary["pages_written"] += 1

            if PROGRESS_EVERY > 0 and page_num % PROGRESS_EVERY == 0:
                print(f"{source}: page {page_num}/{last_page}, written={summary['pages_written']}", flush=True)

    return summary


def main() -> int:
    if START_PAGE < 1:
        raise ValueError("START_PAGE must be >= 1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_paths = iter_pdf_paths()
    print(f"pdfs: {len(pdf_paths)}")
    print(f"output: {OUTPUT_DIR}")

    total_written = 0
    summaries = []
    for pdf_path in pdf_paths:
        print(f"\n===== {pdf_path.name} =====")
        summary = build_pdf_cache(pdf_path)
        summaries.append(summary)
        total_written += summary["pages_written"]
        print(
            "done: "
            f"written={summary['pages_written']} "
            f"existing={summary['pages_skipped_existing']} "
            f"legacy={summary['pages_legacy_fallback']} "
            f"short={summary['pages_skipped_short_text']} "
            f"total={summary['pages_total']}",
            flush=True,
        )

    manifest = {
        "version": 1,
        "builder": "scripts/build_text_layer_cache.py",
        "ocr_cache_dir": str(OUTPUT_DIR),
        "data_dir": str(DATA_DIR),
        "sources": summaries,
        "pages_written": total_written,
    }
    manifest_path = OUTPUT_DIR / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmanifest: {manifest_path}")
    print(f"pages_written: {total_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
