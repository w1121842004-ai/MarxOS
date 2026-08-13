#!/usr/bin/env python3
"""全集页码识别 V2：锚点链选择 + 插值 + article_map 校验。

设计（针对全集页脚页码的十位数 OCR 噪声与边码/书信编号混杂）：

1. 多候选提取：每页从 footer_text/header_text（region 已知）与旧 OCR 候选
   中提取全部数字候选，按 region 分类（footer=印刷页优先，header=边码）。
2. 锚点链 DP：跨页 Viterbi 选择，transition 奖励 ±1 连续，region 加权；
   连续 ≥3 页的链段成为锚点。
3. 插值：锚点之间的页面按 pdf 页距线性插值，confidence 分级。
4. 校验：锚点与插值结果必须在 article_map 的印刷页范围内且单调递增；
   违反的源不写回并列入失败清单。

写回：page JSON 的 page_number_candidates 重排（V2 结果居首，原候选后置），
并新增 page_number_v2 字段（printed_page/confidence/method/region）。
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

REPORT_VERSION = "quanji-pagemap-v2/v1"

DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def normalize_digits(text: str) -> str:
    return str(text or "").translate(DIGITS).replace(" ", "")


def edge_numbers(text: str) -> list[tuple[int, str]]:
    """从页眉/页脚行提取首尾数字（与 page_number_detection 同规则）。"""
    compact = re.sub(r"\s+", "", normalize_digits(text))
    if not compact:
        return []
    if "页" in compact or "版" in compact:
        return []
    if re.search(r"\d{1,4}\s*[-—－]\s*\d{1,4}", compact):
        return []
    results = []
    whole = re.fullmatch(r"[/\\_.\-—－ ]*(\d{1,4})[/\\_.\-—－ ]*", compact)
    if whole:
        results.append((int(whole.group(1)), "whole_line"))
    else:
        start = re.match(r"^[/\\_.\-—－ ]*(\d{1,4})(?!\d)", compact)
        end = re.search(r"(?<!\d)(\d{1,4})[/\\_.\-—－ ]*$", compact)
        if start:
            results.append((int(start.group(1)), "line_start"))
        if end:
            results.append((int(end.group(1)), "line_end"))
    return results


def plausible(printed: int, pdf_page: int, bounds: tuple[int, int] | None = None) -> bool:
    """候选合理性：优先用 article_map 全局印刷页界校验。

    分卷（me14b 从 897 页起、me25b 从 525 页起）的 printed - pdf 偏移远超
    单卷常规窗口，pdf 相对窗口会误杀正确页码；全局界才是正确判据。
    """
    if bounds is not None:
        return bounds[0] - 5 <= printed <= bounds[1] + 5
    return -60 <= printed - pdf_page <= 160


def collect_candidates(
    page: dict,
    pdf_page: int,
    bounds: tuple[int, int] | None = None,
    use_layout_fields: bool = False,
) -> list[dict]:
    """region 分类的多候选提取。

    use_layout_fields=False（全集卷）：只用旧 OCR cleaner 的 region 分离候选——
    text_layer 缓存的 footer_text 混入了边码/书信编号，不可信。
    use_layout_fields=True（文集/选集直通卷）：v1 布局字段本身可信。
    """
    candidates: list[dict] = []
    seen: set[tuple[int, str]] = set()

    def add(value: int, region: str, reason: str, line: str = "", trusted: bool = False):
        # 信任分级：legacy cleaner 的 region 分离候选用全局界（分卷偏移大），
        # text_layer/v1 候选只用 pdf 相对窗口（混入边码噪声）。
        if trusted:
            if bounds is None or not (bounds[0] - 5 <= value <= bounds[1] + 5):
                return
        else:
            if not plausible(value, pdf_page, None):
                return
        key = (value, region)
        if key in seen:
            return
        seen.add(key)
        candidates.append({"printed_page": value, "region": region, "reason": reason, "line": line})

    if use_layout_fields:
        for value, reason in edge_numbers(str(page.get("footer_text") or "")):
            add(value, "footer", f"footer_{reason}", str(page.get("footer_text") or ""))
        for value, reason in edge_numbers(str(page.get("header_text") or "")):
            add(value, "header", f"header_{reason}", str(page.get("header_text") or ""))
        for candidate in page.get("page_number_candidates") or []:
            value = candidate.get("printed_page")
            if isinstance(value, int):
                add(value, str(candidate.get("region") or "unknown"), "v1_candidate")
    # 旧 OCR 缓存候选（clean_academic_page 的 header/footer 分离结果），可信。
    legacy = page.get("_legacy_cleaned") or {}
    for value, reason in edge_numbers(str(legacy.get("footer_text") or "")):
        add(value, "footer", f"legacy_footer_{reason}", trusted=True)
    for value, reason in edge_numbers(str(legacy.get("header_text") or "")):
        add(value, "header", f"legacy_header_{reason}", trusted=True)
    for candidate in legacy.get("page_number_candidates") or []:
        value = candidate.get("printed_page")
        if isinstance(value, int):
            add(value, str(candidate.get("region") or "unknown"), "legacy_candidate", trusted=True)
    # 顶部带 OCR 证据（RapidOCR，当前 PDF 权威来源）：printed 为最强证据，
    # margin（边码）仅作备用候选。
    ocr_evidence = page.get("page_number_ocr") or {}
    printed = ocr_evidence.get("printed") or {}
    if isinstance(printed.get("value"), int):
        add(printed["value"], "printed_top", "ocr_top_printed", trusted=True)
    for number in ocr_evidence.get("numbers") or []:
        if number.get("class") == "margin" and isinstance(number.get("value"), int):
            add(number["value"], "header", "ocr_margin", trusted=True)
    # 书信卷：页脚带 OCR（印刷页在页脚的概率高）。
    for number in page.get("page_number_ocr_bottom") or []:
        if isinstance(number.get("value"), int):
            add(number["value"], "footer", "ocr_bottom", trusted=True)
    return candidates


REGION_BONUS = {"printed_top": 1.0, "footer": 0.6, "header": 0.0, "unknown": 0.2}
GAP_COST = -0.6


def transition_cost(delta: int) -> float:
    if delta == 1:
        return 0.0
    if delta == 0:
        return -0.4  # 插画/重复页
    if delta == 2:
        return -0.3
    if delta == 3:
        return -0.8
    return -4.0


def select_chain(candidates_by_page: list[list[dict]]) -> list[dict | None]:
    """Viterbi：返回每页选中的候选（None=该页无候选/跳过）。"""
    # states: (prev_index, score)
    n = len(candidates_by_page)
    dp: list[list[tuple[int, float] | None]] = [[] for _ in range(n)]
    best_end: tuple[int, int, float] | None = None

    for i, candidates in enumerate(candidates_by_page):
        if not candidates:
            continue
        for j, candidate in enumerate(candidates):
            bonus = REGION_BONUS.get(candidate["region"], 0.0)
            best_prev: tuple[int, float] | None = None
            for pi in range(max(0, i - 3), i):
                if not dp[pi]:
                    continue
                for pj, prev in enumerate(dp[pi]):
                    if prev is None:
                        continue
                    score = prev[1] + transition_cost(candidate["printed_page"] - candidates_by_page[pi][pj]["printed_page"])
                    if best_prev is None or score > best_prev[1]:
                        best_prev = (pi * 64 + pj, score)
            if best_prev is None:
                dp[i].append((j, bonus + GAP_COST))
            else:
                dp[i].append((best_prev[0], best_prev[1] + bonus))
            state_key = i * 64 + j
            if best_end is None or dp[i][j][1] > best_end[2]:
                best_end = (i, j, dp[i][j][1])

    selection: list[dict | None] = [None] * n
    if best_end is None:
        return selection
    i, j, _ = best_end
    while i >= 0:
        if not dp[i] or j >= len(dp[i]):
            break
        selection[i] = candidates_by_page[i][j]
        prev = dp[i][j][0]
        if prev == j:
            break
        i, j = prev // 64, prev % 64
    return selection


def build_anchors(candidates_by_page: list[list[dict]]) -> list[tuple[int, int, str]]:
    """稀疏锚点：候选页两两一致性校验。

    候选页 (i, v_i) 与下一个候选页 (j, v_j) 一致 ⇔ |v_j - v_i - (j - i)| <= 2。
    与相邻候选页都一致的页面成为锚点；每页在一致约束下选择 region 加权
    最优的候选值。
    """
    candidate_pages = [(i, items) for i, items in enumerate(candidates_by_page) if items]
    if not candidate_pages:
        return [], []

    def choose(page_items: list[dict], preferred: dict[int, int]) -> dict:
        scored = []
        for item in page_items:
            bonus = REGION_BONUS.get(item["region"], 0.0)
            consistency = 0
            for delta, expected in preferred.items():
                if abs(item["printed_page"] - expected) <= 2:
                    consistency += 1
            scored.append((consistency, bonus, item))
        scored.sort(key=lambda entry: (-entry[0], -entry[1]))
        return scored[0][2]

    # 第一遍：相邻候选页一致性检查，确定可用候选值集合。
    consistent = {i: False for i, _ in candidate_pages}
    for position in range(len(candidate_pages) - 1):
        i, items_i = candidate_pages[position]
        j, items_j = candidate_pages[position + 1]
        delta_pages = j - i
        for item_i in items_i:
            for item_j in items_j:
                if abs((item_j["printed_page"] - item_i["printed_page"]) - delta_pages) <= 2:
                    consistent[i] = True
                    consistent[j] = True
                    break
            if consistent[i]:
                break

    anchors: list[tuple[int, int, str]] = []
    for position, (i, items) in enumerate(candidate_pages):
        if not consistent[i]:
            continue
        preferred: dict[int, int] = {}
        if position > 0:
            prev_i, prev_items = candidate_pages[position - 1]
            if consistent[prev_i]:
                for item in prev_items:
                    preferred.setdefault(i - prev_i, item["printed_page"] + (i - prev_i))
        if position < len(candidate_pages) - 1:
            next_i, next_items = candidate_pages[position + 1]
            if consistent[next_i]:
                for item in next_items:
                    preferred.setdefault(-(next_i - i), item["printed_page"] - (next_i - i))
        chosen = choose(items, preferred)
        anchors.append((i, chosen["printed_page"], "sparse"))

    # 保留全部 ≥3 的一致链段：扉页/序言/正文各有一套偏移，段内插值才正确，
    # 段间页面（插图/无页码页）留空。跨段拼接会产生递减链，必须避免。
    # 返回值 (anchors, segment_span)，segments 用于限定插值范围。
    segments: list[list[tuple[int, int, str]]] = []
    current: list[tuple[int, int, str]] = []
    for anchor in anchors:
        if current:
            prev_i, prev_v, _ = current[-1]
            i, value, _kind = anchor
            if abs((value - prev_v) - (i - prev_i)) <= 2:
                current.append(anchor)
            else:
                if len(current) >= 3:
                    segments.append(current)
                current = [anchor]
        else:
            current = [anchor]
    if len(current) >= 3:
        segments.append(current)
    flattened = [anchor for segment in segments for anchor in segment]
    spans = [(segment[0][0], segment[-1][0]) for segment in segments]
    return flattened, spans


def interpolate_page(anchors: list[tuple[int, int, str]], index: int) -> dict | None:
    """在两锚点之间线性插值。"""
    before = [anchor for anchor in anchors if anchor[0] < index]
    after = [anchor for anchor in anchors if anchor[0] > index]
    if not before or not after:
        return None
    left_page, left_value, _ = max(before, key=lambda item: item[0])
    right_page, right_value, _ = min(after, key=lambda item: item[0])
    span = right_page - left_page
    if span <= 0:
        return None
    interpolated = left_value + (right_value - left_value) * (index - left_page) / span
    rounded = round(interpolated)
    if abs(interpolated - rounded) > 0.45:
        return None
    return {
        "printed_page": rounded,
        "confidence": 0.8,
        "method": "interpolated",
        "region": "footer",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="", help="逗号分隔；默认全部 me*/mes*/mea* 缓存源")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/ocr_cache_text_layer")
    parser.add_argument("--legacy-cache", type=Path, default=ROOT / "data/ocr_cache")
    parser.add_argument("--article-map", type=Path, default=ROOT / "rag/article_map.json")
    parser.add_argument("--write", action="store_true", help="写回 page JSON（未通过校验的源不写）")
    parser.add_argument("--report", type=Path, default=ROOT / "logs/quanji_pagemap_v2.json")
    args = parser.parse_args()

    article_map = json.loads(args.article_map.read_text(encoding="utf-8"))
    sources = [item.strip() for item in args.sources.split(",") if item.strip()] if args.sources else sorted(
        item.name for item in args.cache_dir.iterdir() if item.is_dir()
    )

    per_source = []
    failed_sources = []
    for source in sources:
        source_dir = args.cache_dir / source
        page_paths = sorted(
            source_dir.glob("page_*.json"),
            key=lambda path: int(path.stem.removeprefix("page_")),
        )
        if not page_paths:
            continue
        from rag.academic_text_cleaner import clean_academic_page

        pages = []
        for path in page_paths:
            try:
                page = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                page = {"page_num": int(path.stem.removeprefix("page_"))}
            page_num = page.get("page_num") or int(path.stem.removeprefix("page_"))
            legacy_txt = args.legacy_cache / source / f"page_{page_num}.txt"
            if legacy_txt.exists():
                try:
                    page["_legacy_cleaned"] = clean_academic_page(
                        legacy_txt.read_text(encoding="utf-8"),
                        source=f"{source}.pdf",
                        page_num=page_num,
                        book_title=page.get("book_title") or source,
                    )
                except Exception:
                    page["_legacy_cleaned"] = {}
            pages.append(page)

        map_entry = article_map.get(f"{source}.pdf") or {}
        map_entries = map_entry.get("entries") or []
        bounds = None
        if map_entries:
            starts = [int(entry.get("start_printed_page") or 0) for entry in map_entries]
            ends = [int(entry.get("end_printed_page") or 0) for entry in map_entries]
            bounds = (min(starts), max(ends))

        # DOC 卷（编辑说明/总目录）不参与页码构建。
        if source in {"mega1-mega2", "meid"} or "index" in source:
            per_source.append({"source": source, "pages": len(page_paths), "skipped": "doc"})
            print(f"[{source}] skipped (doc volume)", flush=True)
            continue

        candidates_by_page = [
            collect_candidates(
                page,
                page.get("page_num") or index,
                bounds,
                use_layout_fields=True,
            )
            for index, page in enumerate(pages)
        ]

        # v1 直通：文集/选集卷是生产基线，v1 页码已经过验证，直接保持不动。
        # 全集卷在 v1 覆盖 ≥0.97 且断点较少（容忍卷内偏移段切换）时也直通。
        v1_values = []
        for index, page in enumerate(pages):
            for candidate in page.get("page_number_candidates") or []:
                value = candidate.get("printed_page")
                if isinstance(value, int) and plausible(value, index, bounds):
                    v1_values.append((index, value))
                    break
        v1_breaks = sum(
            1 for k in range(1, len(v1_values))
            if abs(v1_values[k][1] - v1_values[k - 1][1]) > 2
        )
        v1_coverage = len(v1_values) / len(pages) if pages else 0.0
        v1_break_limit = max(4, len(pages) // 100)
        if source.startswith(("mea", "mes")) or (v1_coverage >= 0.97 and v1_breaks <= v1_break_limit):
            per_source.append({
                "source": source,
                "pages": len(pages),
                "v1_passthrough": True,
                "coverage": round(v1_coverage, 4),
                "passed": True,
            })
            print(f"[{source}] v1 passthrough (coverage={v1_coverage:.2f} breaks={v1_breaks})", flush=True)
            continue

        anchors, spans = build_anchors(candidates_by_page)
        anchors_by_segment = []
        for start, end in spans:
            anchors_by_segment.append([
                (i, value, kind) for i, value, kind in anchors if start <= i <= end
            ])

        # 组装最终页码：锚点页用选中值；段内插值；段外页面留空
        # （宁可无页码，不可错页码）。
        final: list[dict | None] = [None] * len(pages)
        for i, value, _kind in anchors:
            final[i] = {
                "printed_page": value,
                "confidence": 1.0,
                "method": "anchor_sparse",
                "region": "footer",
            }
        for segment_anchors in anchors_by_segment:
            for i in range(segment_anchors[0][0], segment_anchors[-1][0] + 1):
                if final[i] is None:
                    final[i] = interpolate_page(segment_anchors, i)

        # 校验：单调性 + article_map 范围
        values = [(i, item["printed_page"]) for i, item in enumerate(final) if item]
        monotonic_breaks = sum(
            1 for index in range(1, len(values))
            if values[index][1] <= values[index - 1][1]
        )
        in_bounds = True
        if bounds and values:
            in_bounds = bounds[0] - 3 <= values[0][1] and values[-1][1] <= bounds[1] + 3

        coverage = len(values) / len(pages) if pages else 0.0
        anchor_count = len(anchors)
        stats = {
            "source": source,
            "pages": len(pages),
            "candidate_pages": sum(1 for items in candidates_by_page if items),
            "anchors": anchor_count,
            "coverage": round(coverage, 4),
            "monotonic_breaks": monotonic_breaks,
            "article_map_bounds": bounds,
            "in_bounds": in_bounds,
        }
        is_letter_volume = bool(re.match(r"^me2[7-9][ab]?$|^me3[0-9][ab]?$", source))
        coverage_ok = coverage >= (0.25 if is_letter_volume else 0.80)
        passed = in_bounds and coverage_ok and (not values or monotonic_breaks <= max(2, len(pages) // 200))
        if is_letter_volume and passed and coverage < 0.80:
            stats["letters_partial"] = True
        stats["passed"] = passed
        per_source.append(stats)
        if not passed:
            failed_sources.append(source)
        print(
            f"[{source}] pages={len(pages)} candidates={stats['candidate_pages']} "
            f"anchors={anchor_count} coverage={stats['coverage']:.2f} "
            f"breaks={monotonic_breaks} bounds={bounds} in_bounds={in_bounds} passed={passed}",
            flush=True,
        )

        if args.write and passed:
            for i, page in enumerate(pages):
                if final[i] is None:
                    continue
                winner = final[i]
                original = page.get("page_number_candidates") or []
                alternates = [
                    candidate for candidate in original
                    if candidate.get("printed_page") != winner["printed_page"]
                ]
                page["page_number_candidates"] = [
                    {
                        "printed_page": winner["printed_page"],
                        "reason": f"quanji_v2_{winner['method']}",
                        "line": "",
                        "region": winner.get("region", "footer"),
                    }
                ] + alternates
                page["page_number_v2"] = winner
            for path, page in zip(page_paths, pages):
                if any(key in page for key in ("page_number_v2",)):
                    page.pop("_legacy_cleaned", None)
                    path.write_text(json.dumps(page, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    report = {
        "schema_version": REPORT_VERSION,
        "sources": per_source,
        "failed_sources": failed_sources,
        "summary": {
            "total": len(per_source),
            "passed": sum(1 for item in per_source if item.get("passed")),
            "failed": failed_sources,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n=== 汇总 === passed={report['summary']['passed']}/{report['summary']['total']}")
    if failed_sources:
        print(f"failed: {failed_sources}")
    return 1 if failed_sources else 0


if __name__ == "__main__":
    raise SystemExit(main())
