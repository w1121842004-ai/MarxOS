#!/usr/bin/env python3
"""全集页脚条 OCR 探针：渲染页脚带 → PaddleOCR → 数字行提取。

衡量：
  1. 单页 OCR 耗时（CPU）
  2. 页脚带内数字行检出率
  3. 相邻页数字一致性（|Δ值 − Δ页| ≤ 2 的占比）

页映射：cache page_N ↔ PDF 第 N-1 页（fitz 索引 N-1，缓存从第 2 页起）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_VERSION = "quanji-footer-ocr-probe/v1"
DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def digit_values(text: str) -> list[int]:
    compact = re.sub(r"\s+", "", str(text or "").translate(DIGITS))
    values = []
    for match in re.finditer(r"\d{1,4}", compact):
        if "页" in compact or "版" in compact:
            continue
        values.append(int(match.group(0)))
    return values


def footer_number_from_ocr(lines: list[list]) -> tuple[int | None, str]:
    """从 PaddleOCR 行结果提取页脚数字：取垂直位置最靠下的纯数字行。"""
    candidates = []
    for line in lines:
        text = line[1][0]
        y = line[0][0][1]
        values = digit_values(text)
        if not values:
            continue
        candidates.append((y, values, text))
    if not candidates:
        return None, "no_digit_line"
    # 最靠下的数字行优先（页脚带内）。
    candidates.sort(key=lambda item: -item[0])
    y, values, text = candidates[0]
    return values[0], text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "data/marx_engels全集")
    parser.add_argument("--sources", default="me23,me39a")
    parser.add_argument("--pages", type=int, default=50)
    parser.add_argument("--skip", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--band-top", type=float, default=0.86, help="页脚带起点（页高比例）")
    parser.add_argument("--batch", type=int, default=8, help="每批页面数（PaddleOCR 批量推理）")
    parser.add_argument("--report", type=Path, default=ROOT / "logs/quanji_footer_ocr_probe.json")
    args = parser.parse_args()

    import fitz
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)
    zoom = args.dpi / 72.0
    import tempfile

    per_source = []
    for source in args.sources.split(","):
        source = source.strip()
        pdf_path = args.pdf_dir / f"{source}.pdf"
        if not pdf_path.exists():
            print(f"[skip] {source}: PDF missing", flush=True)
            continue
        doc = fitz.open(str(pdf_path))
        # 缓存页 N ↔ PDF 第 N-1 页（缓存从 PDF 第 2 页开始编号）。
        sample = list(range(args.skip, min(args.skip + args.pages, len(doc) - 1)))
        detected = []
        times = []
        for batch_start in range(0, len(sample), args.batch):
            batch_pages = sample[batch_start:batch_start + args.batch]
            image_paths = []
            rendered = []
            for cache_page in batch_pages:
                page = doc[cache_page - 1]
                rect = page.rect
                clip = fitz.Rect(0, rect.height * args.band_top, rect.width, rect.height)
                started = time.perf_counter()
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                    pix.save(handle.name)
                    image_paths.append(handle.name)
                rendered.append(started)
            batch_started = time.perf_counter()
            results = ocr.ocr(image_paths, cls=False)
            batch_elapsed = time.perf_counter() - batch_started
            for cache_page, result in zip(batch_pages, results or []):
                lines = result or []
                number, evidence = footer_number_from_ocr(lines)
                detected.append((cache_page, number, evidence))
                times.append(batch_elapsed / len(batch_pages))
            for image_path in image_paths:
                Path(image_path).unlink(missing_ok=True)
            print(f"  [{source}] batch {batch_start // args.batch + 1}: "
                  f"{len(batch_pages)} pages in {batch_elapsed:.1f}s "
                  f"({batch_elapsed / len(batch_pages):.1f}s/page)", flush=True)
        doc.close()

        values = [(page, number) for page, number, _evidence in detected if number is not None]
        consistent_pairs = sum(
            1 for index in range(1, len(values))
            if abs((values[index][1] - values[index - 1][1]) - (values[index][0] - values[index - 1][0])) <= 2
        )
        stats = {
            "source": source,
            "pages_probed": len(sample),
            "digit_line_rate": round(len(values) / len(sample), 4) if sample else 0.0,
            "consistency_rate": round(consistent_pairs / max(len(values) - 1, 1), 4),
            "avg_seconds_per_page": round(sum(times) / len(times), 2) if times else 0.0,
            "samples": [{"cache_page": p, "number": n, "evidence": e} for p, n, e in detected[:10]],
        }
        per_source.append(stats)
        print(
            f"[{source}] pages={len(sample)} digit_line={stats['digit_line_rate']:.2f} "
            f"consistency={stats['consistency_rate']:.2f} avg={stats['avg_seconds_per_page']}s/page",
            flush=True,
        )
        for p, n, e in detected[:8]:
            print(f"    cache_page={p} -> {n} ({e!r})", flush=True)

    report = {
        "schema_version": REPORT_VERSION,
        "dpi": args.dpi,
        "sources": per_source,
        "summary": {item["source"]: {k: item[k] for k in ("digit_line_rate", "consistency_rate", "avg_seconds_per_page")} for item in per_source},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
