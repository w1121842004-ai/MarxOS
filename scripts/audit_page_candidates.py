from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
RAG_DIR = ROOT_DIR / "rag"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from rag.build_vectorstore_from_cache import (  # noqa: E402
    infer_page_metadata,
    is_plausible_for_pdf_page,
    is_valid_printed_page,
    normalize_digits,
    strip_pdf_boilerplate,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


OCR_CACHE_DIR = Path(os.getenv("OCR_CACHE_DIR", "data/ocr_cache"))
ARTICLE_MAP_PATH = Path(os.getenv("ARTICLE_MAP_PATH", "rag/article_map_core.json"))


def compact(text: str, limit: int = 110) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def load_article_map() -> dict:
    with ARTICLE_MAP_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def candidate_numbers_from_margin(text: str, pdf_page: int) -> list[tuple[int, str, str]]:
    normalized_text = normalize_digits(strip_pdf_boilerplate(text))
    lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
    margin_lines = lines[:6] + lines[-6:]
    candidates = []

    for line in margin_lines:
        compact_line = re.sub(r"\s+", "", line)
        matches = []
        if re.fullmatch(r"[/\\_. -]*(\d{1,4})[/\\_. -]*", compact_line):
            matches.append(re.search(r"\d{1,4}", compact_line).group(0))
        matches.extend(re.findall(r"(?<!\d)(\d{1,4})(?!\d)", compact_line))

        for raw in matches:
            page = int(raw)
            if not is_valid_printed_page(page):
                continue
            if not is_plausible_for_pdf_page(page, pdf_page):
                continue
            candidates.append((page, raw, line))

    seen = set()
    unique = []
    for page, raw, line in candidates:
        key = (page, line)
        if key in seen:
            continue
        seen.add(key)
        unique.append((page, raw, line))

    return unique


def article_hits(article_map: dict, source: str, printed_page: int | None) -> list[str]:
    if printed_page is None:
        return []

    source_map = article_map.get(source) or {}
    hits = []
    for entry in source_map.get("entries", []):
        start = entry.get("start_printed_page")
        end = entry.get("end_printed_page")
        if start is None or end is None:
            continue
        if start <= printed_page <= end:
            hits.append(
                f"{entry.get('title')} [{start}-{end}] level={entry.get('level')}"
            )

    hits.sort(key=lambda item: len(item))
    return hits


def load_page(source_stem: str, page_num: int) -> dict:
    path = OCR_CACHE_DIR / source_stem / f"page_{page_num}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def audit_range(source: str, start: int, end: int) -> None:
    article_map = load_article_map()
    source_stem = source.replace(".pdf", "")

    for page_num in range(start, end + 1):
        page = load_page(source_stem, page_num)
        text = page.get("cleaned_text") or page.get("text") or ""
        inferred_printed, inferred_article = infer_page_metadata(text, "fallback", page_num)
        candidates = candidate_numbers_from_margin(text, page_num)
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        print(f"\n===== {source} pdf_page={page_num} =====")
        print(f"page_type={page.get('page_type')} cache_page_num={page.get('page_num')}")
        print(f"infer_page_metadata: printed_page={inferred_printed}, article={inferred_article}")
        print("margin_candidates:")
        for candidate, raw, line in candidates:
            print(f"- page={candidate} raw={raw!r} line={compact(line)}")

        print("article_map_hits_for_inferred:")
        for hit in article_hits(article_map, source, inferred_printed)[:6]:
            print(f"- {hit}")

        print("head:")
        for line in lines[:4]:
            print(f"- {compact(line)}")
        print("tail:")
        for line in lines[-4:]:
            print(f"- {compact(line)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="PDF source, for example mea01.pdf")
    parser.add_argument("--start", type=int, required=True, help="First PDF page")
    parser.add_argument("--end", type=int, required=True, help="Last PDF page")
    args = parser.parse_args()
    audit_range(args.source, args.start, args.end)


if __name__ == "__main__":
    main()
