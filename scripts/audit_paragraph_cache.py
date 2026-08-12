from __future__ import annotations

import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.paragraph_cache import read_paragraph_cache


DEFAULT_CACHE = Path(os.getenv("PARAGRAPH_CACHE_PATH", "data/paragraph_cache.jsonl"))
CHECK_TERMS = [
    "国家是社会在一定发展阶段上的产物",
    "剩余价值",
    "劳动过程",
]
SUSPICIOUS_MARKERS = ["目录", "目次", "索引", "注释", "编者注", "选编说明"]


def compact(text: str, limit: int = 160) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def has_suspicious_text(record: dict) -> bool:
    title_text = " ".join(
        str(record.get(key) or "")
        for key in ["article", "section"]
    )
    paragraph = str(record.get("paragraph_text") or "")
    return any(marker in title_text or marker in paragraph[:80] for marker in SUSPICIOUS_MARKERS)


def looks_too_short(record: dict) -> bool:
    return int(record.get("paragraph_char_count") or 0) < 25


def audit_cache(path: Path) -> int:
    records = read_paragraph_cache(path)
    by_source = defaultdict(list)
    for record in records:
        by_source[record.get("source")].append(record)

    print(f"paragraph cache: {path}")
    print(f"total paragraphs: {len(records)}")
    print(f"sources: {len(by_source)}")

    suspicious = [record for record in records if has_suspicious_text(record)]
    too_short = [record for record in records if looks_too_short(record)]
    cross_page = [record for record in records if record.get("cross_page")]
    lengths = [int(record.get("paragraph_char_count") or 0) for record in records]

    if lengths:
        lengths_sorted = sorted(lengths)
        print(
            "lengths: "
            f"min={lengths_sorted[0]}, "
            f"p50={lengths_sorted[len(lengths_sorted)//2]}, "
            f"max={lengths_sorted[-1]}"
        )

    print(f"cross_page paragraphs: {len(cross_page)}")
    print(f"short paragraphs (<25 chars): {len(too_short)}")
    print(f"suspicious title/text markers: {len(suspicious)}")

    print("\nper source:")
    for source in sorted(by_source):
        source_records = by_source[source]
        source_cross = sum(1 for record in source_records if record.get("cross_page"))
        print(f"- {source}: {len(source_records)} paragraphs, cross_page={source_cross}")

    print("\nterm checks:")
    failed_terms = []
    for term in CHECK_TERMS:
        hits = [
            record for record in records
            if re.sub(r"\s+", "", term) in re.sub(r"\s+", "", record.get("paragraph_text") or "")
        ]
        print(f"- {term}: {len(hits)} hits")
        if hits:
            first = hits[0]
            print(
                f"  first={first.get('source')} "
                f"pdf={first.get('pdf_page_start')}-{first.get('pdf_page_end')} "
                f"paragraph={first.get('paragraph_index')} "
                f"text={compact(first.get('paragraph_text'))}"
            )
        else:
            failed_terms.append(term)

    if suspicious:
        print("\nsuspicious examples:")
        for record in suspicious[:10]:
            print(
                f"- {record.get('source')} pdf={record.get('pdf_page_start')} "
                f"article={record.get('article')} text={compact(record.get('paragraph_text'), 120)}"
            )

    if too_short:
        short_by_source = Counter(record.get("source") for record in too_short)
        print("\nshort paragraph counts:")
        for source, count in short_by_source.most_common(10):
            print(f"- {source}: {count}")

    if failed_terms:
        print("\nmissing required term checks:")
        for term in failed_terms:
            print(f"- {term}")
        return 1

    print("\nparagraph cache audit passed.")
    return 0


def main() -> None:
    path = Path(os.getenv("PARAGRAPH_CACHE_PATH", str(DEFAULT_CACHE)))
    if not path.exists():
        raise FileNotFoundError(f"Paragraph cache not found: {path}")
    raise SystemExit(audit_cache(path))


if __name__ == "__main__":
    main()
