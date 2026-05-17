from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
RAG_DIR = ROOT_DIR / "rag"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from rag.build_vectorstore_from_cache import (  # noqa: E402
    is_plausible_for_pdf_page,
    is_valid_printed_page,
    normalize_digits,
    strip_pdf_boilerplate,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


OCR_CACHE_DIR = Path(os.getenv("OCR_CACHE_DIR", "data/ocr_cache"))


def compact(text: str, limit: int = 72) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def page_num_from_path(path: Path) -> int | None:
    match = re.search(r"page_(\d+)\.(?:json|txt)$", path.name)
    return int(match.group(1)) if match else None


def load_page(source_stem: str, pdf_page: int) -> dict:
    json_path = OCR_CACHE_DIR / source_stem / f"page_{pdf_page}.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    txt_path = OCR_CACHE_DIR / source_stem / f"page_{pdf_page}.txt"
    with txt_path.open("r", encoding="utf-8", errors="replace") as f:
        return {"cleaned_text": f.read(), "page_type": ""}


def iter_pdf_pages(source_stem: str, first_pages: int) -> list[int]:
    source_dir = OCR_CACHE_DIR / source_stem
    pages = []
    seen = set()
    for path in source_dir.glob("page_*.*"):
        page = page_num_from_path(path)
        if page is not None and page not in seen:
            seen.add(page)
            pages.append(page)

    return sorted(pages)[:first_pages]


def margin_lines(text: str, width: int) -> list[str]:
    normalized_text = normalize_digits(strip_pdf_boilerplate(text))
    lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
    return lines[:width] + lines[-width:]


def strict_margin_candidates(text: str, pdf_page: int, width: int) -> list[tuple[int, str]]:
    # Production-safe candidates: only a whole margin line that is just a page number.
    candidates = []
    for line in margin_lines(text, width):
        compact_line = re.sub(r"\s+", "", line)
        match = re.fullmatch(r"[/\\_. -]*(\d{1,4})[/\\_. -]*", compact_line)
        if not match:
            continue

        page = int(match.group(1))
        if not is_valid_printed_page(page):
            continue
        if not is_plausible_for_pdf_page(page, pdf_page):
            continue

        candidates.append((page, line))

    return candidates


def risky_embedded_candidates(text: str, pdf_page: int, width: int) -> list[tuple[int, str]]:
    # Diagnostic only. These numbers often come from notes or manuscript page marks.
    candidates = []
    for line in margin_lines(text, width):
        compact_line = re.sub(r"\s+", "", line)
        for raw in re.findall(r"(?<!\d)(\d{1,4})(?!\d)", compact_line):
            page = int(raw)
            if is_valid_printed_page(page) and is_plausible_for_pdf_page(page, pdf_page):
                candidates.append((page, line))

    return candidates


def collect_observations(source_stem: str, first_pages: int, margin_width: int, include_embedded: bool) -> list[dict]:
    observations = []
    for pdf_page in iter_pdf_pages(source_stem, first_pages):
        page = load_page(source_stem, pdf_page)
        text = page.get("cleaned_text") or page.get("text") or ""
        candidates = strict_margin_candidates(text, pdf_page, margin_width)
        if include_embedded:
            candidates.extend(risky_embedded_candidates(text, pdf_page, margin_width))

        unique = []
        seen = set()
        for printed_page, line in candidates:
            key = (printed_page, line)
            if key not in seen:
                seen.add(key)
                unique.append((printed_page, line))

        observations.append(
            {
                "pdf_page": pdf_page,
                "page_type": page.get("page_type") or "",
                "candidates": unique,
            }
        )

    return observations


def longest_consecutive_run(observations: list[dict]) -> dict | None:
    by_offset = defaultdict(list)
    for obs in observations:
        pdf_page = obs["pdf_page"]
        for printed_page, line in obs["candidates"]:
            offset = pdf_page - printed_page
            by_offset[offset].append((pdf_page, printed_page, line))

    best = None
    for offset, rows in by_offset.items():
        rows = sorted(rows)
        current = []
        previous_pdf = None
        previous_printed = None
        for pdf_page, printed_page, line in rows:
            if previous_pdf == pdf_page - 1 and previous_printed == printed_page - 1:
                current.append((pdf_page, printed_page, line))
            else:
                if best is None or len(current) > len(best["rows"]):
                    best = {"offset": offset, "rows": current}
                current = [(pdf_page, printed_page, line)]
            previous_pdf = pdf_page
            previous_printed = printed_page

        if best is None or len(current) > len(best["rows"]):
            best = {"offset": offset, "rows": current}

    return best


def summarize_offsets(observations: list[dict]) -> Counter:
    counter = Counter()
    for obs in observations:
        pdf_page = obs["pdf_page"]
        for printed_page, _line in obs["candidates"]:
            counter[pdf_page - printed_page] += 1
    return counter


def detect_source(source: str, first_pages: int, margin_width: int, min_run: int, include_embedded: bool) -> None:
    source_stem = source.replace(".pdf", "")
    observations = collect_observations(source_stem, first_pages, margin_width, include_embedded)
    best = longest_consecutive_run(observations)
    offsets = summarize_offsets(observations)
    candidate_pages = sum(1 for obs in observations if obs["candidates"])

    print(f"\n===== {source_stem}.pdf =====")
    print(f"scanned_pages={len(observations)} candidate_pages={candidate_pages} include_embedded={include_embedded}")
    print("top_offsets:", ", ".join(f"{offset}:{count}" for offset, count in offsets.most_common(5)) or "none")

    if not best or len(best["rows"]) < min_run:
        print(f"verdict=NO_STABLE_START min_run={min_run}")
        return

    rows = best["rows"]
    start_pdf, start_printed, _ = rows[0]
    print(
        f"verdict=STABLE_START offset={best['offset']} "
        f"start_pdf={start_pdf} start_printed={start_printed} run_len={len(rows)}"
    )
    print("evidence:")
    for pdf_page, printed_page, line in rows[:8]:
        print(f"- pdf={pdf_page} printed={printed_page} line={compact(line)}")


def parse_sources(value: str | None) -> list[str]:
    if value:
        return [item.strip().replace(".pdf", "") for item in value.split(",") if item.strip()]

    return [path.name for path in sorted(OCR_CACHE_DIR.iterdir()) if path.is_dir()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", help="Comma separated source stems or pdf names, for example me01,mea01,capital")
    parser.add_argument("--first-pages", type=int, default=30)
    parser.add_argument("--margin-width", type=int, default=6)
    parser.add_argument("--min-run", type=int, default=3)
    parser.add_argument("--include-embedded", action="store_true", help="Diagnostic only: include embedded numbers in margin lines")
    args = parser.parse_args()

    for source in parse_sources(args.sources):
        detect_source(source, args.first_pages, args.margin_width, args.min_run, args.include_embedded)


if __name__ == "__main__":
    main()
