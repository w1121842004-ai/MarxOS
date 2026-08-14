#!/usr/bin/env python3
"""全集 OCR/文本层缓存质量体检。

按源（77 个 me* 目录 + 未缓存的 PDF）输出机器可读质量指标并分档：
  A  简体、空页率低、页码识别率高、杂质密度低 → 可直接进 v2 管线
  B  繁体，或页码/杂质中等 → 需 LLM 纠错或局部重做
  C  空页多、页码率低、杂质/乱码高 → 建议重识别
  UNCACHED  无页面缓存（书信卷/索引卷等）

分档阈值是初版经验值，可在 --tier-empty/--tier-page/--tier-garbage 调整。
输出报告 logs/quanji_cache_audit.json（含逐源明细与分档汇总）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_VERSION = "quanji-cache-audit/v1"

# 高区分度繁体字集合（避开简体中同样常用的歧义字）。
TRADITIONAL_MARKERS = set(
    "馬讀讓來動後為國書時經與爾無學門問見說語論類關處議義實業廣傳張陳劉戰獨對錯變發從眾體會長電鐵嚴產資階級織線約設識記認訴譯釋詞試評調談請誰閱聞觀現視覺頭腦難隱險隊陽陰際濟滿漢淚滅燈靈煉煙燒熱愛爭靜節藝蘇藥蘭蟲雖歲歸鄉邊達進遠運過還這連選遺郵鄰鐘點羅舊曉錢園團歡權雙斷斷勝務嗎碼離頓"
)

NON_HAN_PUNCT = re.compile(r"[^一-鿿0-9A-Za-z]")
REPLACEMENT_JUNK = re.compile("[\ufffd\ue000-\uf8ff]")


def han_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(re.findall(r"[一-鿿]", text)) / len(text)


def traditional_ratio(text: str) -> float:
    han = re.findall(r"[一-鿿]", text)
    if not han:
        return 0.0
    hits = sum(1 for char in han if char in TRADITIONAL_MARKERS)
    return hits / len(han)


def garbage_ratio(text: str) -> float:
    """非汉字非标点的异常字符比例（乱码/替换符/私有区字符）。"""
    if not text:
        return 0.0
    junk = len(REPLACEMENT_JUNK.findall(text))
    return junk / max(len(text), 1)


def repeated_line_ratio(text: str) -> float:
    """同页重复行占比（OCR 重复/扫描重影的代理指标）。"""
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 6]
    if not lines:
        return 0.0
    counts = Counter(lines)
    duplicates = sum(count - 1 for count in counts.values() if count >= 3)
    return duplicates / len(lines)


def audit_source(cache_dir: Path, source: str, limit: int | None) -> dict:
    source_dir = cache_dir / source
    page_paths = sorted(
        source_dir.glob("page_*.json"),
        key=lambda path: int(path.stem.removeprefix("page_")),
    )
    if limit:
        page_paths = page_paths[:limit]

    total_chars = 0
    han_chars = 0
    trad_chars = 0
    junk_chars = 0
    empty_pages = 0
    pages_with_page_number = 0
    whole_line_pages = 0
    repeated_dup = 0
    total_lines = 0
    page_types: Counter[str] = Counter()
    text_sources: Counter[str] = Counter()
    char_counts: list[int] = []

    for path in page_paths:
        try:
            page = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            empty_pages += 1
            continue
        text = str(page.get("cleaned_text") or page.get("raw_text") or "")
        if not text.strip():
            empty_pages += 1
            continue
        total_chars += len(text)
        han = re.findall(r"[一-鿿]", text)
        han_chars += len(han)
        trad_chars += sum(1 for char in han if char in TRADITIONAL_MARKERS)
        junk_chars += len(REPLACEMENT_JUNK.findall(text))
        candidates = page.get("page_number_candidates") or []
        if candidates:
            pages_with_page_number += 1
            if any(str(item.get("reason") or "") == "whole_line" for item in candidates):
                whole_line_pages += 1
        page_types[str(page.get("page_type") or "unknown")] += 1
        text_sources[str(page.get("text_source") or "unknown")] += 1
        char_counts.append(len(han))
        lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 6]
        total_lines += len(lines)
        counts = Counter(lines)
        repeated_dup += sum(count - 1 for count in counts.values() if count >= 3)

    pages = len(page_paths)
    return {
        "source": source,
        "pages_checked": pages,
        "empty_pages": empty_pages,
        "empty_rate": round(empty_pages / pages, 4) if pages else 1.0,
        "total_chars": total_chars,
        "han_chars": han_chars,
        "avg_han_per_page": round(han_chars / pages, 1) if pages else 0.0,
        "traditional_ratio": round(trad_chars / han_chars, 5) if han_chars else 0.0,
        "junk_ratio": round(junk_chars / total_chars, 5) if total_chars else 0.0,
        "page_number_rate": round(pages_with_page_number / pages, 4) if pages else 0.0,
        "page_number_whole_line_rate": round(whole_line_pages / pages, 4) if pages else 0.0,
        "repeated_line_ratio": round(repeated_dup / total_lines, 5) if total_lines else 0.0,
        "page_types": dict(sorted(page_types.items())),
        "text_sources": dict(sorted(text_sources.items())),
        "min_page_chars": min(char_counts) if char_counts else 0,
        "max_page_chars": max(char_counts) if char_counts else 0,
    }


LETTER_SOURCE_PATTERN = re.compile(r"^(me2[7-9][ab]?|me3[0-9][ab]?|letter\d+)$")
DOC_SOURCE_PATTERN = re.compile(r"^(mega1-mega2|meid|.*index.*)$")


def tier_of(source: str, metrics: dict, thresholds: dict) -> str:
    # 书信卷：无页眉页码是版式特征，页码率不作为质量判据，单列 L。
    if LETTER_SOURCE_PATTERN.match(source):
        return "L"
    # 编辑说明/总索引/附件卷：不切块入库，单列 DOC。
    if DOC_SOURCE_PATTERN.match(source):
        return "DOC"
    if metrics["empty_rate"] > thresholds["empty"]:
        return "C"
    if metrics["junk_ratio"] > thresholds["junk"]:
        return "C"
    if metrics["page_number_rate"] < thresholds["page"]:
        return "C"
    if metrics["traditional_ratio"] > thresholds["traditional"]:
        return "B"
    if metrics["empty_rate"] <= thresholds["empty_a"] and metrics["page_number_rate"] >= thresholds["page_a"]:
        return "A"
    return "B"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/ocr_cache_text_layer")
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "data/marx_engels全集")
    parser.add_argument("--report", type=Path, default=ROOT / "logs/quanji_cache_audit.json")
    parser.add_argument("--limit", type=int, default=0, help="每源最多检查页数（0=全部）")
    parser.add_argument("--tier-empty", type=float, default=0.08, help="C 档空页率阈值")
    parser.add_argument("--tier-page", type=float, default=0.30, help="C 档页码识别率阈值（全集卷普遍偏低，校准为 0.30）")
    parser.add_argument("--tier-junk", type=float, default=0.03, help="C 档杂质密度阈值")
    parser.add_argument("--tier-traditional", type=float, default=0.003, help="繁体判定阈值（繁体字/汉字）")
    parser.add_argument("--tier-page-a", type=float, default=0.70, help="A 档页码识别率阈值")
    parser.add_argument("--tier-empty-a", type=float, default=0.01, help="A 档空页率阈值")
    args = parser.parse_args()

    pdfs = sorted(path.stem for path in args.pdf_dir.glob("*.pdf"))
    cached_sources = sorted(
        item.name
        for item in args.cache_dir.iterdir()
        if item.is_dir() and next(item.glob("page_*.json"), None)
    )
    thresholds = {
        "empty": args.tier_empty,
        "page": args.tier_page,
        "junk": args.tier_junk,
        "traditional": args.tier_traditional,
        "page_a": args.tier_page_a,
        "empty_a": args.tier_empty_a,
    }

    per_source: list[dict] = []
    for source in cached_sources:
        metrics = audit_source(args.cache_dir, source, args.limit or None)
        metrics["tier"] = tier_of(source, metrics, thresholds)
        per_source.append(metrics)
        print(
            f"[{metrics['tier']}] {source:16s} pages={metrics['pages_checked']:5d} "
            f"empty={metrics['empty_rate']:.3f} pagenum={metrics['page_number_rate']:.2f} "
            f"junk={metrics['junk_ratio']:.4f} trad={metrics['traditional_ratio']:.4f} "
            f"avg_han={metrics['avg_han_per_page']:.0f}",
            flush=True,
        )

    uncached = sorted(set(pdfs) - set(cached_sources))
    for source in uncached:
        print(f"[UNCACHED] {source}", flush=True)

    tier_summary = Counter(item["tier"] for item in per_source)
    report = {
        "schema_version": REPORT_VERSION,
        "cache_dir": str(args.cache_dir),
        "thresholds": thresholds,
        "summary": {
            "cached_sources": len(per_source),
            "uncached_pdfs": len(uncached),
            "uncached_list": uncached,
            "tiers": dict(sorted(tier_summary.items())),
        },
        "sources": per_source,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n=== 分档汇总 ===")
    for tier, count in sorted(tier_summary.items()):
        print(f"{tier}: {count}")
    print(f"UNCACHED: {len(uncached)} -> {', '.join(uncached)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
