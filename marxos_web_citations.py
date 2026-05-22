from __future__ import annotations

import json
import re
from pathlib import Path


def citation_from_evidence(item, index):
    evidence = item.get("evidence") or []
    if not isinstance(evidence, list) or not evidence:
        return None
    selected = evidence[index - 1] if 0 <= index - 1 < len(evidence) else evidence[0]
    source = selected.get("source") or selected.get("source_file")
    page = selected.get("printed_page") or selected.get("citation_page")
    if not source or page is None:
        return None
    try:
        page = int(page)
    except (TypeError, ValueError):
        return None
    return {
        "index": index,
        "body": selected.get("citation") or selected.get("sentence_citation") or "",
        "source": source,
        "page": page,
        "pdf_page": selected.get("pdf_page"),
        "excerpt": selected.get("excerpt") or "",
    }


def requested_citation_index(query):
    query = query or ""
    number_words = [
        (1, ["第一", "第1", "1号", "1条"]),
        (2, ["第二", "第两", "第2", "2号", "2条"]),
        (3, ["第三", "第3", "3号", "3条"]),
        (4, ["第四", "第4", "4号", "4条"]),
        (5, ["第五", "第5", "5号", "5条"]),
    ]
    for index, markers in number_words:
        if any(marker in query for marker in markers):
            return index
    match = re.search(r"(\d+)\s*[号条段]", query)
    return int(match.group(1)) if match else 1


def requested_citation_indices(query):
    query = query or ""
    hits = []
    for match in re.finditer(r"第\s*(\d+)\s*[条段句]", query):
        hits.append(int(match.group(1)))
    for match in re.finditer(r"(\d+)\s*[条段句]", query):
        value = int(match.group(1))
        if value not in hits:
            hits.append(value)
    return hits


def parse_citation_line(line):
    match = re.match(r"\s*(\d+)[\.\u3001]\s*(.+)", line)
    if not match:
        return None
    index = int(match.group(1))
    body = match.group(2).strip()
    series_match = re.search(r"《(马克思恩格斯(?:文集|选集))》第(\d+)卷", body)
    page_match = re.search(r"第(\d+)页", body)
    if not series_match or not page_match:
        return None
    series, volume = series_match.group(1), int(series_match.group(2))
    prefix = "mea" if "文集" in series else "mes"
    return {
        "index": index,
        "body": body,
        "source": f"{prefix}{volume:02d}.pdf",
        "page": int(page_match.group(1)),
    }


def parse_last_citations(text):
    citations = {}
    for line in (text or "").splitlines():
        parsed = parse_citation_line(line)
        if parsed:
            citations[parsed["index"]] = parsed
    return citations


def load_ocr_text(source, pdf_page, ocr_cache_dir: Path, repair_mojibake):
    path = ocr_cache_dir / source.replace(".pdf", "") / f"page_{pdf_page}.json"
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return repair_mojibake(payload.get("cleaned_text") or payload.get("raw_text") or "")


def find_pdf_page_by_printed_page(source, printed_page, ocr_cache_dir: Path, infer_printed_page_from_ocr_cache):
    source_dir = ocr_cache_dir / source.replace(".pdf", "")
    if not source_dir.exists():
        return None
    paths = sorted(
        source_dir.glob("page_*.json"),
        key=lambda path: int(re.search(r"page_(\d+)", path.name).group(1)),
    )
    for path in paths:
        pdf_page = int(re.search(r"page_(\d+)", path.name).group(1))
        inferred = infer_printed_page_from_ocr_cache({"source": source, "pdf_page": pdf_page})
        if inferred == printed_page:
            return pdf_page
    return None


def paragraphs_from_text(text):
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    paragraphs = []
    current = []
    for line in lines:
        current.append(line)
        if line.endswith(("。", "！", "？", "。”")) and len("".join(current)) >= 120:
            paragraphs.append("".join(current))
            current = []
    if current:
        paragraphs.append("".join(current))
    return paragraphs


def answer_citation_followup(
    query,
    history,
    is_contextual_followup_fn,
    last_bot_item_fn,
    last_bot_message_fn,
    ocr_cache_dir: Path,
    repair_mojibake,
    infer_printed_page_from_ocr_cache,
):
    if not is_contextual_followup_fn(query):
        return None
    needed_markers = ["出处", "段落", "摘", "原文", "哪段"]
    if not any(marker in query for marker in needed_markers):
        return None

    requested_index_value = requested_citation_index(query)
    last_bot = last_bot_item_fn(history)
    citation = citation_from_evidence(last_bot, requested_index_value)
    if not citation:
        citations = parse_last_citations(last_bot_message_fn(history))
        if not citations:
            return None
        citation = citations.get(requested_index_value) or citations.get(1)
    if not citation:
        return None

    pdf_page = None
    try:
        pdf_page = int(citation.get("pdf_page")) if citation.get("pdf_page") is not None else None
    except (TypeError, ValueError):
        pdf_page = None
    if pdf_page is None:
        pdf_page = find_pdf_page_by_printed_page(
            citation["source"],
            citation["page"],
            ocr_cache_dir,
            infer_printed_page_from_ocr_cache,
        )
    if pdf_page is None:
        return (
            f"我没有在本地 OCR 页码映射中定位到脚注 {citation['index']} 的原页：{citation['body']}\n\n"
            "这说明上一条回答的页码需要重新核对；本轮不会编造段落。"
        )

    text = load_ocr_text(citation["source"], pdf_page, ocr_cache_dir, repair_mojibake)
    paragraphs = paragraphs_from_text(text)
    excerpt = "\n\n".join(paragraphs[:2]).strip()
    if len(excerpt) > 900:
        excerpt = excerpt[:900].rstrip() + "......"
    if not excerpt:
        excerpt = "该页 OCR 文本为空，需要重新 OCR 或核对原 PDF。"

    return (
        f"按上一条回答的脚注 {citation['index']} 定位：{citation['body']}\n\n"
        f"本地 OCR 对应到 {citation['source']} 的第 {pdf_page} 个图像页，识别出的印刷页为第 {citation['page']} 页。\n\n"
        "原页摘录如下：\n\n"
        f"> {excerpt}\n\n"
        "说明：如果上一条正文里的那句话是概括句，而不是原著逐字引文，我这里只给出脚注页的 OCR 原文，不把概括句伪装成原文。"
    )


def answer_evidence_page_followup(query, history, is_contextual_followup_fn, last_bot_item_fn):
    normalized = query or ""
    explicit_indices = requested_citation_indices(query)
    if not is_contextual_followup_fn(query):
        has_page_request = any(marker in normalized for marker in ["页", "页码", "升序", "排序", "列出来"]) and any(
            marker in normalized for marker in ["证据", "页", "页码"]
        )
        if not explicit_indices and not has_page_request:
            return None
        if "页" not in normalized and "页码" not in normalized:
            return None

    if "页" not in normalized and "页码" not in normalized:
        return None

    last_bot = last_bot_item_fn(history)
    evidence = last_bot.get("evidence") or []
    if not isinstance(evidence, list) or not evidence:
        return None

    numbered_requests = ["前3", "前三", "三条", "3条", "三点", "3点"]
    wants_sorted = any(marker in normalized for marker in ["升序", "排序", "列出来", "全部", "所有", "单独"])
    wants_pages = any(marker in normalized for marker in ["哪一页", "页码", "分别", "页"])
    if not wants_pages:
        return None

    items = []
    for item in evidence:
        page = item.get("printed_page") or item.get("citation_page")
        if page is None:
            continue
        items.append(item)
    if not items:
        return None

    if explicit_indices:
        selected = []
        for index in explicit_indices:
            if 1 <= index <= len(evidence):
                item = evidence[index - 1]
                page = item.get("printed_page") or item.get("citation_page")
                if page is not None:
                    selected.append(item)
        if selected:
            items = selected
    elif any(marker in normalized for marker in numbered_requests):
        items = items[:3]
    elif wants_sorted:
        items = sorted(items, key=lambda item: int(item.get("printed_page") or item.get("citation_page") or 0))
    else:
        items = items[: min(5, len(items))]

    lines = ["根据上一条回答中的直接证据，相关页码如下：", ""]
    for index, item in enumerate(items, start=1):
        citation = item.get("detailed_citation") or item.get("citation") or ""
        page = item.get("printed_page") or item.get("citation_page")
        lines.append(f"{index}. 第{page}页。{citation}")
    return "\n".join(lines)
