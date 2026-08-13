#!/usr/bin/env python3
"""全集印刷页码证据提取：顶部带 RapidOCR + 位置判别 + checkpoint 续跑。

每页渲染顶部带（页高 0-0.22）跑 RapidOCR，按水平位置分类数字行：
  margin  贴左边距（x 比例 < 0.12）——边码，排除
  printed 内侧（0.12-0.75）或外侧（>=0.75）——印刷页码（奇偶页交替角位）
偶数/奇数 cache 页的期望角位：奇→内侧(左)，偶→外侧(右)（实测 me23）。

写回 page JSON：page_number_ocr = {"printed": {...}|null, "numbers": [...],
"expected_corner_match": bool}。已有该字段的页跳过（checkpoint）。
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

REPORT_VERSION = "quanji-page-evidence/v1"
DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def classify(x: float, width: float) -> str:
    ratio = x / max(width, 1)
    if ratio < 0.12:
        return "margin"
    if ratio < 0.75:
        return "printed_inner"
    return "printed_outer"


def expected_corner(cache_page: int, klass: str) -> bool:
    if cache_page % 2 == 1:
        return klass == "printed_inner"
    return klass == "printed_outer"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="", help="逗号分隔；默认全部 me 卷")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/ocr_cache_text_layer")
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "data/marx_engels全集")
    parser.add_argument("--band-bottom", type=float, default=0.22)
    parser.add_argument("--also-bottom", action="store_true", help="书信卷：追加页脚带扫描（0.75-0.98）")
    parser.add_argument("--dpi", type=int, default=216)
    parser.add_argument("--report", type=Path, default=ROOT / "logs/quanji_page_evidence.json")
    args = parser.parse_args()

    import fitz
    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR()
    zoom = args.dpi / 72.0

    sources = (
        [item.strip() for item in args.sources.split(",") if item.strip()]
        if args.sources
        else sorted(
            item.name
            for item in args.cache_dir.iterdir()
            if item.is_dir() and re.match(r"^me\d", item.name)
        )
    )

    per_source = []
    for source in sources:
        source_dir = args.cache_dir / source
        pdf_path = args.pdf_dir / f"{source}.pdf"
        page_paths = sorted(
            source_dir.glob("page_*.json"),
            key=lambda path: int(path.stem.removeprefix("page_")),
        )
        if not page_paths or not pdf_path.exists():
            print(f"[skip] {source}: missing pages or PDF", flush=True)
            continue
        doc = fitz.open(str(pdf_path))
        processed = 0
        printed_count = 0
        skipped = 0
        timings: list[float] = []
        for path in page_paths:
            cache_page = int(path.stem.removeprefix("page_"))
            try:
                page_json = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if "page_number_ocr" in page_json:
                skipped += 1
                continue
            if cache_page - 1 >= len(doc):
                continue
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
            timings.append(time.perf_counter() - started)

            numbers = []
            for box, text, _score in result or []:
                compact = re.sub(r"\s+", "", str(text or "").translate(DIGITS))
                match = re.fullmatch(r"(\d{1,4})", compact)
                if not match:
                    continue
                klass = classify(box[0][0], pix.width)
                numbers.append({
                    "value": int(match.group(1)),
                    "x": round(box[0][0], 0),
                    "class": klass,
                    "text": str(text).strip(),
                })
            printed_candidates = [n for n in numbers if n["class"] != "margin"]
            printed = None
            expected_match = False
            if printed_candidates:
                # 优先期望角位，其次任意角位。
                preferred = [n for n in printed_candidates if expected_corner(cache_page, n["class"])]
                chosen = preferred[0] if preferred else printed_candidates[0]
                printed = {"value": chosen["value"], "x": chosen["x"], "class": chosen["class"]}
                expected_match = bool(preferred)
            page_json["page_number_ocr"] = {
                "printed": printed,
                "numbers": numbers,
                "expected_corner_match": expected_match,
            }

            bottom_numbers = None
            if args.also_bottom:
                clip_bottom = fitz.Rect(0, rect.height * 0.75, rect.width, rect.height * 0.98)
                pix_bottom = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip_bottom)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                    pix_bottom.save(handle.name)
                    bottom_image_path = handle.name
                started = time.perf_counter()
                try:
                    bottom_result, _ = ocr(bottom_image_path)
                finally:
                    Path(bottom_image_path).unlink(missing_ok=True)
                timings.append(time.perf_counter() - started)
                bottom_digits = []
                for box, text, _score in bottom_result or []:
                    compact = re.sub(r"\s+", "", str(text or "").translate(DIGITS))
                    match = re.fullmatch(r"(\d{1,4})", compact)
                    if match:
                        bottom_digits.append({
                            "value": int(match.group(1)),
                            "x": round(box[0][0], 0),
                            "text": str(text).strip(),
                        })
                bottom_numbers = bottom_digits
            if bottom_numbers is not None:
                page_json["page_number_ocr_bottom"] = bottom_numbers
            path.write_text(json.dumps(page_json, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            processed += 1
            printed_count += int(printed is not None)
            if processed % 100 == 0:
                print(f"  [{source}] {processed}/{len(page_paths)} pages, printed={printed_count}", flush=True)
        doc.close()
        stats = {
            "source": source,
            "pages_processed": processed,
            "pages_skipped": skipped,
            "printed_rate": round(printed_count / max(processed, 1), 4),
            "avg_seconds_per_page": round(sum(timings) / len(timings), 2) if timings else 0.0,
        }
        per_source.append(stats)
        print(
            f"[{source}] processed={processed} skipped={skipped} "
            f"printed_rate={stats['printed_rate']:.2f} avg={stats['avg_seconds_per_page']}s/page",
            flush=True,
        )

    report = {
        "schema_version": REPORT_VERSION,
        "sources": per_source,
        "summary": {
            item["source"]: {
                "printed_rate": item["printed_rate"],
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
