from __future__ import annotations

import re
from dataclasses import dataclass

from rag.clean_ocr_text import (
    clean_body_line,
    clean_ocr_page,
    clean_toc_line,
    extract_author_candidate,
    extract_title_candidate,
    is_header_footer_line,
    is_title_page,
    normalize_ocr_text,
    toc_score,
)
from rag.page_number_detection import margin_page_candidates


FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
BOILERPLATE_RE = re.compile(
    r"(?:本\s*PDF\s*文件|S22PDF|pdf@|home\.icm\.ac\.cn|pdfFactory|fineprint).*$",
    re.IGNORECASE,
)
INLINE_FOOTER_RE = re.compile(
    r"(?P<body>.*?)(?P<page>[０１２３４５６７８９\d]{1,4})"
    r"(?P<label>[\u4e00-\u9fff·《》〈〉、（）()\s]{2,40})$"
)


@dataclass
class LineRegion:
    text: str
    region: str
    reason: str | None = None


def normalize_digits(text: str) -> str:
    return str(text or "").translate(FULLWIDTH_DIGITS)


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def split_inline_footer(line: str, pdf_page: int | None = None) -> tuple[str, str | None]:
    """Split a one-line PDF/OCR footer glued onto body text.

    Some existing cache pages end like ``正文……８４反杜林论第一编哲学本PDF文件...``.
    The cleaner keeps the body and exposes ``８４反杜林论第一编哲学`` as footer
    metadata so page inference can still use it.
    """
    line = str(line or "").strip()
    if not line:
        return "", None

    line = BOILERPLATE_RE.sub("", line).strip()
    match = INLINE_FOOTER_RE.fullmatch(line)
    if not match:
        return line, None

    body = match.group("body").strip()
    footer = (match.group("page") + match.group("label")).strip()
    page = int(normalize_digits(match.group("page")))
    if not (0 < page <= 1200):
        return line, None
    if pdf_page is not None and not (-80 <= pdf_page - page <= 220):
        return line, None
    if len(body) < 80:
        return line, None
    return body, footer


def classify_margin_lines(lines: list[str], pdf_page: int | None = None) -> tuple[list[LineRegion], str, str]:
    regions: list[LineRegion] = []
    header_lines: list[str] = []
    footer_lines: list[str] = []
    total = len(lines)

    for index, raw_line in enumerate(lines):
        line = str(raw_line or "").strip()
        if not line:
            continue

        body_line, inline_footer = split_inline_footer(line, pdf_page=pdf_page)
        if inline_footer:
            if body_line:
                regions.append(LineRegion(body_line, "body", "inline_footer_split"))
            footer_lines.append(inline_footer)
            regions.append(LineRegion(inline_footer, "footer", "inline_footer_split"))
            continue

        compact = compact_text(line)
        near_top = index < 3
        near_bottom = index >= max(0, total - 3)
        page_like = bool(re.fullmatch(r"[-—–]?\d{1,4}[-—–]?", normalize_digits(compact)))
        header_footer = is_header_footer_line(line)

        if near_top and (page_like or header_footer):
            header_lines.append(line)
            regions.append(LineRegion(line, "header", "margin"))
        elif near_bottom and (page_like or header_footer):
            footer_lines.append(line)
            regions.append(LineRegion(line, "footer", "margin"))
        else:
            regions.append(LineRegion(line, "body"))

    return regions, "\n".join(header_lines), "\n".join(footer_lines)


def merge_split_digit_page_lines(text: str, pdf_page: int | None = None) -> str:
    """Merge page numbers split into one digit per line, e.g. ``８\n４``."""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return text
    if not all(re.fullmatch(r"[０-９\d]", line) for line in lines):
        return text

    merged = "".join(lines)
    try:
        page = int(normalize_digits(merged))
    except ValueError:
        return text
    if not (0 < page <= 1200):
        return text
    if pdf_page is not None and not (-80 <= pdf_page - page <= 220):
        return text
    return merged


def permissive_margin_page_candidates(header_text: str = "", footer_text: str = "", pdf_page: int | None = None) -> list[dict]:
    candidates: list[dict] = []
    for region, text in [("header", header_text), ("footer", footer_text)]:
        for line in str(text or "").splitlines():
            compact = re.sub(r"\s+", "", normalize_digits(line))
            match = re.fullmatch(r"[-—–]?(\d{1,4})[-—–]?", compact)
            reason = "whole_line"
            if not match and region == "footer":
                match = re.match(r"^(\d{1,4})(?=[\u4e00-\u9fff·《》〈〉、（）()])", compact)
                reason = "line_start"
            if not match:
                continue
            page = int(match.group(1))
            if not (0 < page <= 1200):
                continue
            if 1700 <= page <= 2099:
                continue
            if pdf_page is not None and not (-80 <= pdf_page - page <= 220):
                continue
            candidates.append({"printed_page": page, "reason": reason, "line": line, "region": region})

    seen = set()
    unique = []
    for candidate in candidates:
        key = (candidate["printed_page"], candidate["region"], candidate["line"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def clean_academic_page(raw_text: str, source: str | None = None, page_num: int | None = None, book_title: str | None = None) -> dict:
    normalized = normalize_ocr_text(raw_text)
    raw_lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    regions, header_text, footer_text = classify_margin_lines(raw_lines, pdf_page=page_num)
    header_text = merge_split_digit_page_lines(header_text, pdf_page=page_num)
    footer_text = merge_split_digit_page_lines(footer_text, pdf_page=page_num)
    body_lines = [item.text for item in regions if item.region == "body"]

    score, reasons = toc_score(body_lines)
    title_candidate = extract_title_candidate(body_lines)
    author_candidate = extract_author_candidate(body_lines)
    title_page = is_title_page(body_lines, title_candidate, author_candidate)

    if not body_lines:
        page_type = "blank"
    elif score >= 4:
        page_type = "toc"
    elif title_page:
        page_type = "title_page"
        reasons.append("short_title_page")
    else:
        page_type = "body"

    if page_type == "toc":
        title_candidate = None
        author_candidate = None

    cleaned_lines: list[str] = []
    for line in body_lines:
        if is_header_footer_line(line):
            continue
        cleaned = clean_toc_line(line) if page_type == "toc" else clean_body_line(line)
        if cleaned:
            cleaned_lines.append(cleaned)

    cleaned_text = "\n".join(cleaned_lines)
    if not cleaned_text and raw_text:
        fallback = clean_ocr_page(raw_text, source=source, page_num=page_num, book_title=book_title)
        cleaned_text = fallback.get("cleaned_text") or ""
        reasons.extend(fallback.get("reasons") or [])

    layout_lines = [
        {
            "text": item.text,
            "region": item.region,
            "reason": item.reason,
        }
        for item in regions
    ]
    body_text = "\n".join(item.text for item in regions if item.region == "body")
    page_number_candidates = margin_page_candidates(header_text, footer_text, pdf_page=page_num)
    if not page_number_candidates:
        page_number_candidates = permissive_margin_page_candidates(header_text, footer_text, pdf_page=page_num)

    return {
        "raw_text": raw_text or "",
        "cleaned_text": cleaned_text,
        "page_type": page_type,
        "is_toc": page_type == "toc",
        "is_title_page": page_type == "title_page",
        "title_candidate": title_candidate,
        "author_candidate": author_candidate,
        "source": source,
        "page_num": page_num,
        "book_title": book_title,
        "reasons": list(dict.fromkeys(reasons)),
        "header_text": header_text,
        "body_text": body_text,
        "footer_text": footer_text,
        "layout_lines": layout_lines,
        "page_number_candidates": page_number_candidates,
        "cleaner": "academic_text_layer_v1",
    }
