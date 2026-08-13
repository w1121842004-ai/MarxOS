#!/usr/bin/env python3
"""全集印刷页码 OCR 探针：顶部带 + 水平位置判别。

发现（2026-08-13）：当前 PDF 的印刷页码在页面上部——偶数页左上（x≈250-300）、
奇数页右上（x≈950-1000），页边码贴左边距（x≈85-100），两者位置可区分。
页映射：cache page_N ↔ PDF 第 N-1 页（fitz 索引 N-1）。

指标：数字行检出率、位置分类（printed/margin/unknown）、相邻页一致性、
单页耗时（RapidOCR）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_VERSION = "quanji-pagenum-ocr-probe/v1"
DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def classify_number(x: float, width: float) -> str:
    """按水平位置分类：贴左边距=边码，内侧/右侧=印刷页码。"""
    ratio = x / max(width, 1)
    if ratio < 0.12:
        return "margin"
    if ratio < 0.75:
        return "printed_inner"
    return "printed_outer"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "data/marx_engels全集")
    parser.add_argument("--sources", default="me23,me03,me39a")
    parser.add_argument("--pages", type=int, default=50)
    parser.add_argument("--skip", type=int, default=10)
    parser.add_argument("--band-bottom", type=float, default=0.22, help="顶部带下界（页高比例）")
    parser.add_argument("--dpi", type=int, default=216)
    parser.add_argument("--report", type=Path, default=ROOT / "logs/quanji_pagenum_ocr_probe.json")
    args = parser.parse_args()

    import fitz
    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR()
    zoom = args.dpi / 72.0

    per_source = []
    for source in args.sources.split(","):
        source = source.strip()
        pdf_path = args.pdf_dir / f"{source}.pdf"
        if not pdf_path.exists():
            print(f"[skip] {source}: PDF missing", flush=True)
            continue
        doc = fitz.open(str(pdf_path))
        sample = list(range(args.skip, min(args.skip + args.pages, len(doc) - 1)))
        detected = []
        times = []
        for cache_page in sample:
            page = doc[cache_page - 1]
            rect = page.rect
            clip = fitz.Rect(0, 0, rect.width, rect.height * args.band_bottom)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                pix.save(handle.name)
                image_path = handle.name
            started = time.perf_counter()
            try:
                result, _ = ocr(image_path)
            finally:
                Path(image_path).unlink(missing_ok=True)
            elapsed = time.perf_counter() - started
            times.append(elapsed)
            numbers = []
            for box, text, _score in result or []:
                compact = re.sub(r"\s+", "", str(text or "").translate(DIGITS))
                match = re.fullmatch(r"(\d{1,4})", compact)
                if not match:
                    continue
                x = box[0][0]
                numbers.append({
                    "value": int(match.group(1)),
                    "x": round(x, 0),
                    "class": classify_number(x, pix.width),
                    "text": text,
                })
            numbers.sort(key=lambda item: item["class"] != "margin")
            detected.append((cache_page, numbers))
        doc.close()

        printed = [
            (page, next((n for n in numbers if n["class"] != "margin"), None))
            for page, numbers in detected
        ]
        printed_values = [(page, n["value"]) for page, n in printed if n]
        consistent_pairs = sum(
            1 for index in range(1, len(printed_values))
            if abs((printed_values[index][1] - printed_values[index - 1][1])
                   - (printed_values[index][0] - printed_values[index - 1][0])) <= 2
        )
        stats = {
            "source": source,
            "pages_probed": len(sample),
            "printed_digit_rate": round(len(printed_values) / len(sample), 4) if sample else 0.0,
            "consistency_rate": round(consistent_pairs / max(len(printed_values) - 1, 1), 4),
            "avg_seconds_per_page": round(sum(times) / len(times), 2) if times else 0.0,
            "samples": [
                {"cache_page": page, "numbers": numbers[:4]}
                for page, numbers in detected[:10]
            ],
        }
        per_source.append(stats)
        print(
            f"[{source}] pages={len(sample)} printed_digit={stats['printed_digit_rate']:.2f} "
            f"consistency={stats['consistency_rate']:.2f} avg={stats['avg_seconds_per_page']}s/page",
            flush=True,
        )
        for page, numbers in detected[:6]:
            print(f"    p{page}: {[(n['value'], n['class']) for n in numbers]}", flush=True)

    report = {
        "schema_version": REPORT_VERSION,
        "sources": per_source,
        "summary": {
            item["source"]: {
                "printed_digit_rate": item["printed_digit_rate"],
                "consistency_rate": item["consistency_rate"],
                "avg_seconds_per_page": item["avg_seconds_per_page"],
            }
            for item in per_source
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
