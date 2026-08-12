from __future__ import annotations


def _volume_from_source(source: str) -> str:
    source = str(source or "").lower().replace(".pdf", "")
    if not source.startswith("me"):
        return source
    number = source[2:4]
    suffix = source[4:]
    try:
        volume = int(number)
    except ValueError:
        return source
    suffix_text = {"a": "上", "b": "下", "c": "附"}.get(suffix, "")
    return f"第{volume}卷{f'({suffix_text})' if suffix_text else ''}"


def build_ambiguous_locator_answer(query: str, constraints: dict, limit: int = 10) -> str:
    if not constraints.get("ambiguous_locator"):
        return ""
    entries = constraints.get("entries") or []
    if len(entries) < 2:
        return ""

    seen = set()
    candidates = []
    for entry in entries:
        key = (
            entry.get("source"),
            entry.get("start_page"),
            entry.get("end_page"),
            entry.get("article") or entry.get("classic_title"),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(entry)

    if len(candidates) < 2:
        return ""

    title = constraints.get("title") or candidates[0].get("article") or "该标题"
    lines = [
        f"《{title}》存在同名或近似同名篇目，当前问题不足以唯一确定一个出处。建议先按候选定位：",
        "",
    ]
    for index, entry in enumerate(candidates[:limit], start=1):
        source = entry.get("source") or ""
        volume = _volume_from_source(source)
        article = entry.get("article") or entry.get("classic_title") or title
        start = entry.get("start_page")
        end = entry.get("end_page")
        page_text = f"第{start}页" if start == end else f"第{start}-{end}页"
        lines.append(f"{index}. 《马克思恩格斯全集》{volume}，《{article}》，{page_text}。")

    lines.extend(
        [
            "",
            "你可以补充卷册、年份、上下文关键词，或指定上面某一个候选，我再按该候选继续检索原文并回答。",
        ]
    )
    return "\n".join(lines)
