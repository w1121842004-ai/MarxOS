from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
OCR_CACHE_DIR = ROOT_DIR / "data" / "ocr_cache"
REPORT_PATH = ROOT_DIR / "logs" / "ocr_printed_page_audit.json"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app  # noqa: E402


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def page_num(path: Path) -> int:
    match = re.search(r"page_(\d+)\.json$", path.name)
    return int(match.group(1)) if match else -1


def edge_printed_page(path: Path) -> int | None:
    metadata = {"source": f"{path.parent.name}.pdf", "pdf_page": page_num(path)}
    return app.infer_printed_page_from_ocr_cache(metadata)


def compact(text: object, limit: int = 120) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def load_text(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return app.repair_mojibake(payload.get("cleaned_text") or payload.get("raw_text") or "")


def audit_source(source_stem: str) -> dict:
    source_dir = OCR_CACHE_DIR / source_stem
    paths = sorted(source_dir.glob("page_*.json"), key=page_num)
    rows = []

    for path in paths:
        pdf = page_num(path)
        printed = edge_printed_page(path)
        rows.append(
            {
                "pdf_page": pdf,
                "printed_page": printed,
                "offset": pdf - printed if printed is not None else None,
                "text": compact(load_text(path)),
            }
        )

    printed_rows = [row for row in rows if row["printed_page"] is not None]
    offsets = Counter(row["offset"] for row in printed_rows)
    dominant_offset, dominant_count = offsets.most_common(1)[0] if offsets else (None, 0)

    jumps = []
    previous = None
    for row in printed_rows:
        if previous:
            pdf_delta = row["pdf_page"] - previous["pdf_page"]
            printed_delta = row["printed_page"] - previous["printed_page"]
            if pdf_delta > 0 and printed_delta not in {pdf_delta, 0}:
                jumps.append(
                    {
                        "from_pdf": previous["pdf_page"],
                        "to_pdf": row["pdf_page"],
                        "from_printed": previous["printed_page"],
                        "to_printed": row["printed_page"],
                        "pdf_delta": pdf_delta,
                        "printed_delta": printed_delta,
                        "text": row["text"],
                    }
                )
        previous = row

    outliers = [
        row for row in printed_rows
        if dominant_offset is not None and row["offset"] != dominant_offset
    ]

    return {
        "source": f"{source_stem}.pdf",
        "total_pages": len(rows),
        "printed_detected": len(printed_rows),
        "printed_missing": len(rows) - len(printed_rows),
        "dominant_offset": dominant_offset,
        "dominant_offset_count": dominant_count,
        "offset_variants": dict(sorted(offsets.items())),
        "outlier_count": len(outliers),
        "outlier_examples": outliers[:20],
        "jump_count": len(jumps),
        "jump_examples": jumps[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        default="mea01,mea02,mea03,mea04,mea05,mea06,mea07,mea08,mea09,mea10,mes01,mes02,mes03,mes04",
    )
    parser.add_argument("--report", default=str(REPORT_PATH))
    args = parser.parse_args()

    sources = [item.strip().replace(".pdf", "") for item in args.sources.split(",") if item.strip()]
    results = [audit_source(source) for source in sources]

    print("source total detected missing dominant_offset variants outliers jumps")
    for item in results:
        print(
            f"{item['source']} {item['total_pages']} {item['printed_detected']} "
            f"{item['printed_missing']} {item['dominant_offset']} "
            f"{len(item['offset_variants'])} {item['outlier_count']} {item['jump_count']}"
        )

    by_health = defaultdict(int)
    for item in results:
        if item["printed_detected"] == 0:
            by_health["no_printed_pages"] += 1
        elif item["outlier_count"] > item["printed_detected"] * 0.25:
            by_health["unstable_offset"] += 1
        else:
            by_health["usable"] += 1

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"summary": dict(by_health), "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Report: {report_path}")
    print(f"Summary: {dict(by_health)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
