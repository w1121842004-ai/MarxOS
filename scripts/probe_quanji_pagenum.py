#!/usr/bin/env python3
"""全集页码识别探针：坐标法 vs 现有缓存候选。

用 pymupdf 提取每页顶部区域（页眉带）的独立数字文本作为"位置页码"候选，
与 data/ocr_cache_text_layer 中的 page_number_candidates 对比。

探针目标：验证坐标法能否把全集卷的页码识别率从 0.3-0.7 提升到 ≥0.95。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_VERSION = "quanji-pagenum-probe/v1"
PAGE_NUMBER_RE = re.compile(r"^\s*([0-9０-９]{1,4})\s*$")


def normalize_digits(text: str) -> int | None:
    text = str(text or "").strip()
    mapping = str.maketrans("０１２３４５６７８９", "0123456789")
    text = text.translate(mapping).strip()
    if not text.isdigit():
        return None
    return int(text)


def header_band_number(page) -> tuple[int | None, str]:
    """页面上方 25% 带内的数字（多为边码/书信编号，非印刷页码）。"""
    try:
        blocks = page.get_text("dict", sort=True)["blocks"]
    except Exception:
        return None, "extract_failed"
    page_height = page.rect.height
    band_bottom = page_height * 0.25
    row_spans: dict[int, list[tuple[float, str]]] = {}
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                y = span["bbox"][1]
                if y > band_bottom:
                    continue
                text = span["text"]
                if not text or any(char not in "0123456789０１２３４５６７８９" for char in text.strip()):
                    continue
                key = int(y // 6)
                row_spans.setdefault(key, []).append((span["bbox"][0], text.strip()))
    candidates = []
    for key, spans in row_spans.items():
        spans.sort(key=lambda item: item[0])
        joined = "".join(text for _, text in spans)
        number = normalize_digits(joined)
        if number is None:
            continue
        candidates.append((number, joined, key))
    if not candidates:
        return None, "no_header_number"
    candidates.sort(key=lambda item: item[2])
    number, joined, _ = candidates[0]
    return number, joined


def footer_band_number(page) -> tuple[int | None, str]:
    """页面下方 15% 带内的独立数字行 = 印刷页码（全集排版惯例）。

    页码在页脚（footer whole_line），数字逐位分 span，需按行合并。
    """
    try:
        blocks = page.get_text("dict", sort=True)["blocks"]
    except Exception:
        return None, "extract_failed"
    page_height = page.rect.height
    band_top = page_height * 0.85
    row_spans: dict[int, list[tuple[float, str]]] = {}
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                y = span["bbox"][1]
                if y < band_top:
                    continue
                text = span["text"]
                if not text or any(char not in "0123456789０１２３４５６７８９" for char in text.strip()):
                    continue
                key = int(y // 6)
                row_spans.setdefault(key, []).append((span["bbox"][0], text.strip()))
    candidates = []
    for key, spans in row_spans.items():
        spans.sort(key=lambda item: item[0])
        joined = "".join(text for _, text in spans)
        number = normalize_digits(joined)
        if number is None:
            continue
        candidates.append((number, joined, key))
    if not candidates:
        return None, "no_footer_number"
    candidates.sort(key=lambda item: item[2])
    number, joined, _ = candidates[0]
    return number, joined


def cached_candidates(cache_dir: Path, source: str, page_index: int) -> list[dict]:
    path = cache_dir / source / f"page_{page_index}.json"
    if not path.exists():
        return []
    try:
        page = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return page.get("page_number_candidates") or []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "data/marx_engels全集")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/ocr_cache_text_layer")
    parser.add_argument("--sources", default="me23,me03,me39a")
    parser.add_argument("--pages", type=int, default=100, help="每源探针页数")
    parser.add_argument("--skip", type=int, default=10, help="跳过的前置页数（封面/扉页/目录）")
    parser.add_argument("--report", type=Path, default=ROOT / "logs/quanji_pagenum_probe.json")
    args = parser.parse_args()

    try:
        import fitz  # noqa: F401
    except ImportError:
        print("pymupdf (fitz) is required", file=sys.stderr)
        return 2

    per_source = []
    for source in args.sources.split(","):
        source = source.strip()
        pdf_path = args.pdf_dir / f"{source}.pdf"
        if not pdf_path.exists():
            print(f"[skip] {source}: PDF missing", flush=True)
            continue
        doc = fitz.open(str(pdf_path))
        sample = range(args.skip, min(args.skip + args.pages, len(doc)))
        footer_hits = 0
        header_hits = 0
        cache_hits = 0
        footer_cache_agree = 0
        footer_only_pages = []
        details = []
        for page_index in sample:
            page = doc[page_index]
            footer_number, footer_evidence = footer_band_number(page)
            header_number, _header_evidence = header_band_number(page)
            cached = cached_candidates(args.cache_dir, source, page_index)
            cached_number = None
            for candidate in cached:
                cached_number = normalize_digits(str(candidate.get("printed_page") or ""))
                if cached_number is not None:
                    break
            footer_hits += int(footer_number is not None)
            header_hits += int(header_number is not None)
            cache_hits += int(cached_number is not None)
            if footer_number is not None and cached_number is not None:
                footer_cache_agree += int(footer_number == cached_number)
            if footer_number is not None and cached_number is None:
                footer_only_pages.append((page_index, footer_number))
            if len(details) < 6:
                details.append({
                    "page_index": page_index,
                    "footer": footer_number,
                    "header": header_number,
                    "cached_printed": cached_number,
                })
        doc.close()
        result = {
            "source": source,
            "pages_probed": len(sample),
            "footer_hit_rate": round(footer_hits / len(sample), 4) if sample else 0.0,
            "header_hit_rate": round(header_hits / len(sample), 4) if sample else 0.0,
            "cached_hit_rate": round(cache_hits / len(sample), 4) if sample else 0.0,
            "footer_cache_agreement_rate": round(footer_cache_agree / len(sample), 4) if sample else 0.0,
            "footer_only_count": len(footer_only_pages),
            "footer_only_pages": footer_only_pages[:5],
            "samples": details,
        }
        per_source.append(result)
        print(
            f"[{source}] pages={len(sample)} footer={result['footer_hit_rate']:.2f} "
            f"header={result['header_hit_rate']:.2f} cached={result['cached_hit_rate']:.2f} "
            f"footer≈cached={result['footer_cache_agreement_rate']:.2f} footer_only={result['footer_only_count']}",
            flush=True,
        )
        for detail in details:
            print(f"    pdf={detail['page_index']} footer={detail['footer']} "
                  f"header={detail['header']} cached={detail['cached_printed']}", flush=True)

    report = {
        "schema_version": REPORT_VERSION,
        "sources": per_source,
        "summary": {
            result["source"]: {
                "footer_hit_rate": result["footer_hit_rate"],
                "cached_hit_rate": result["cached_hit_rate"],
                "footer_cache_agreement_rate": result["footer_cache_agreement_rate"],
                "footer_only_count": result["footer_only_count"],
            }
            for result in per_source
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
