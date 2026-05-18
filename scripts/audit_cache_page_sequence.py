from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
import json


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.build_vectorstore_from_cache import document_from_cache  # noqa: E402
from rag.build_vectorstore_from_cache import page_num_from_cache_file  # noqa: E402


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


OCR_CACHE_DIR = Path(os.getenv("OCR_CACHE_DIR", "data/ocr_cache"))


def source_from_cache_dir(path: Path) -> str:
    return f"{path.name}.pdf"


def iter_source_cache_files(source: str) -> list[Path]:
    source_stem = source.replace(".pdf", "")
    source_dir = OCR_CACHE_DIR / source_stem
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    return sorted(
        source_dir.glob("page_*.txt"),
        key=lambda path: page_num_from_cache_file(str(path)) or 0,
    )


def compact(text: str, limit: int = 80) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def load_page_type(source: str, pdf_page: int) -> str:
    source_stem = source.replace(".pdf", "")
    path = OCR_CACHE_DIR / source_stem / f"page_{pdf_page}.json"
    if not path.exists():
        return "missing"

    with path.open("r", encoding="utf-8") as f:
        return json.load(f).get("page_type") or "body"


def explain_break_by_skipped_pages(source: str, previous: dict, row: dict) -> tuple[str | None, str]:
    pdf_delta = row["pdf_page"] - previous["pdf_page"]
    printed_delta = row["printed_page"] - previous["printed_page"]
    skipped_types = [
        load_page_type(source, page)
        for page in range(previous["pdf_page"] + 1, row["pdf_page"])
    ]
    non_counted = sum(1 for page_type in skipped_types if page_type in {"title_page", "blank", "toc"})
    expected_delta = pdf_delta - non_counted

    if "title_page" in skipped_types:
        return "title_boundary", ",".join(skipped_types)

    if printed_delta == expected_delta:
        return "skipped_pages", ",".join(skipped_types)

    return None, ",".join(skipped_types)


def audit_source(source: str, show_limit: int = 80) -> None:
    title_context = {}
    page_sequence_context = {}
    rows = []

    for cache_path in iter_source_cache_files(source):
        doc = document_from_cache(str(cache_path), title_context, page_sequence_context)
        if doc is None:
            continue

        metadata = doc.metadata
        rows.append(
            {
                "pdf_page": metadata.get("pdf_page"),
                "printed_page": metadata.get("printed_page"),
                "source": metadata.get("page_number_source"),
                "citation_type": metadata.get("citation_page_type"),
                "text": doc.page_content,
            }
        )

    rows.sort(key=lambda row: row["pdf_page"] or 0)
    counts = Counter(row["source"] or "unknown" for row in rows)
    printed_rows = [row for row in rows if row["printed_page"] is not None]
    pdf_only_rows = [row for row in rows if row["printed_page"] is None]

    breaks = []
    explained_breaks = Counter()
    previous = None
    for row in printed_rows:
        if previous:
            pdf_delta = row["pdf_page"] - previous["pdf_page"]
            printed_delta = row["printed_page"] - previous["printed_page"]
            if pdf_delta > 0 and printed_delta != pdf_delta:
                explanation, skipped_types = explain_break_by_skipped_pages(source, previous, row)
                if explanation:
                    explained_breaks[explanation] += 1
                else:
                    breaks.append((previous, row, pdf_delta, printed_delta, skipped_types))
        previous = row

    print(f"\n===== {source} =====")
    print(f"docs={len(rows)} printed={len(printed_rows)} pdf_only={len(pdf_only_rows)}")
    print("page_number_source:", " ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    explained_text = " ".join(
        f"{key}={value}" for key, value in sorted(explained_breaks.items())
    )
    print(f"sequence_breaks={len(breaks)} explained={explained_text or 'none'}")

    if pdf_only_rows:
        print("pdf_only examples:")
        for row in pdf_only_rows[:show_limit]:
            print(
                f"- pdf={row['pdf_page']} source={row['source']} "
                f"text={compact(row['text'])}"
            )

    if breaks:
        print("sequence break examples:")
        for previous, row, pdf_delta, printed_delta, skipped_types in breaks[:show_limit]:
            print(
                f"- pdf {previous['pdf_page']}->{row['pdf_page']} "
                f"printed {previous['printed_page']}->{row['printed_page']} "
                f"delta {pdf_delta}/{printed_delta} source={row['source']} "
                f"skipped=[{skipped_types}]"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        default="mea01.pdf,mea02.pdf,mea03.pdf,mea04.pdf,mea05.pdf,mea06.pdf,mea07.pdf,mea08.pdf,mea09.pdf,mea10.pdf",
        help="Comma-separated PDF sources to audit.",
    )
    parser.add_argument("--show-limit", type=int, default=40)
    args = parser.parse_args()

    for source in [item.strip() for item in args.sources.split(",") if item.strip()]:
        audit_source(source, show_limit=args.show_limit)


if __name__ == "__main__":
    main()
