#!/usr/bin/env python3
"""构建 OCR 缓存的汉字双字倒排索引，用于引文子句级全库定位。

索引结构（pickle）：
  {
    "pages": [(source, pdf_page), ...],          # page_id 表
    "bigrams": {bigram: [page_id, ...], ...},    # 汉字双字 → 页面列表
    "version": "quote-index/v1",
  }

全库约 48K 页，构建约 2-4 分钟；查询时对引文子句取双字集合求页面交集，
再对候选页做精确 fuzzy 校验，避免线性扫库。
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INDEX_VERSION = "quote-index/v1"
HAN_RE = re.compile(r"[一-鿿]")


def han_bigrams(text: str) -> set[str]:
    han = "".join(HAN_RE.findall(str(text or "")))
    return {han[i:i + 2] for i in range(len(han) - 1)}


def build_index(cache_dir: Path) -> dict:
    pages: list[tuple[str, int]] = []
    page_ids: dict[tuple[str, int], int] = {}
    bigrams: dict[str, list[int]] = defaultdict(list)

    sources = sorted(
        item.name for item in cache_dir.iterdir()
        if item.is_dir() and next(item.glob("page_*.json"), None)
    )
    for source_index, source in enumerate(sources):
        source_dir = cache_dir / source
        page_paths = sorted(
            source_dir.glob("page_*.json"),
            key=lambda path: int(path.stem.removeprefix("page_")),
        )
        for path in page_paths:
            try:
                page = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            text = page.get("cleaned_text") or page.get("raw_text") or ""
            page_id = len(pages)
            pages.append((source, int(path.stem.removeprefix("page_"))))
            page_ids[(source, int(path.stem.removeprefix("page_")))] = page_id
            for bigram in han_bigrams(text):
                bigrams[bigram].append(page_id)
        if (source_index + 1) % 10 == 0:
            print(f"  indexed {source_index + 1}/{len(sources)} sources", flush=True)

    return {
        "version": INDEX_VERSION,
        "pages": pages,
        "page_ids": page_ids,
        "bigrams": {bigram: sorted(set(ids)) for bigram, ids in bigrams.items()},
    }


def query_index(index: dict, clause: str, top: int = 80) -> list[tuple[str, int]]:
    """返回与子句共享双字最多的候选页（source, pdf_page）。"""
    clause_bigrams = han_bigrams(clause)
    if not clause_bigrams:
        return []
    scores: dict[int, int] = defaultdict(int)
    for bigram in clause_bigrams:
        for page_id in index["bigrams"].get(bigram, ())[:400]:
            scores[page_id] += 1
    ranked = sorted(scores.items(), key=lambda item: -item[1])[:top]
    pages = index["pages"]
    return [pages[page_id] for page_id, _score in ranked]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/ocr_cache_text_layer")
    parser.add_argument("--output", type=Path, default=ROOT / "data/quote_index.pkl")
    args = parser.parse_args()

    index = build_index(args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(index, handle, protocol=4)
    print(f"索引完成：{len(index['pages'])} 页，{len(index['bigrams'])} 个双字，"
          f"输出 {args.output}（{args.output.stat().st_size // 1024 // 1024}MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
