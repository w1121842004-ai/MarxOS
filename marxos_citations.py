from __future__ import annotations

import re


def citation_page_label(metadata, normalize_metadata, clean_text):
    metadata = normalize_metadata(metadata)
    citation_page = metadata.get("citation_page")
    printed_page = metadata.get("printed_page")
    pdf_page = metadata.get("pdf_page")

    if printed_page is not None:
        return f"第{clean_text(printed_page)}页"
    if citation_page is not None:
        return f"第{clean_text(citation_page)}页"
    if pdf_page is not None:
        return f"第{clean_text(pdf_page)}页"
    return "页码不详"


def source_page_label(metadata, normalize_metadata, clean_text):
    metadata = normalize_metadata(metadata)
    return citation_page_label(metadata, normalize_metadata, clean_text)


def format_citation(metadata, include_article, normalize_metadata, normalize_book_parts, clean_text):
    metadata = normalize_metadata(metadata)
    author, title, volume, year = normalize_book_parts(metadata)
    article = clean_text(metadata.get("section") or metadata.get("article"), "")
    if metadata.get("no_page_citation"):
        letter_title = clean_text(metadata.get("letter_title") or article, "")
        volume_text = volume if volume else ""
        if letter_title and letter_title != title:
            return f"《{title}》{volume_text}，《{letter_title}》"
        return f"《{title}》{volume_text}"
    author_text = f"{author}，" if author else ""
    volume_text = volume if volume else ""
    article_text = ""
    if include_article and article and article != title:
        article_text = f"，《{article}》"
    year_text = year if year else ""
    page_text = citation_page_label(metadata, normalize_metadata, clean_text)
    return f"{author_text}《{title}》{volume_text}{article_text}，北京：人民出版社{year_text}，{page_text}。"


def evidence_from_doc(
    doc,
    index,
    normalize_metadata,
    clean_text,
    compact_preview,
    format_citation_fn,
):
    metadata = normalize_metadata(doc.metadata)
    content = clean_text(doc.page_content, "")
    return {
        "id": f"E{index}",
        "citation": format_citation_fn(metadata, include_article=False),
        "detailed_citation": format_citation_fn(metadata, include_article=True),
        "sentence_citation": format_citation_fn(metadata, include_article=False),
        "source": metadata.get("source"),
        "source_file": metadata.get("source_file") or metadata.get("source"),
        "series": metadata.get("series"),
        "volume": metadata.get("volume"),
        "article": metadata.get("article") or metadata.get("section"),
        "section": metadata.get("section"),
        "paragraph_id": metadata.get("paragraph_id"),
        "line_start": metadata.get("line_start"),
        "line_end": metadata.get("line_end"),
        "char_start": metadata.get("char_start"),
        "char_end": metadata.get("char_end"),
        "printed_page": metadata.get("printed_page"),
        "citation_page": metadata.get("citation_page"),
        "pdf_page": metadata.get("pdf_page") or metadata.get("page"),
        "match_type": metadata.get("match_type"),
        "confidence": metadata.get("confidence"),
        "is_letter": metadata.get("is_letter"),
        "letter_title": metadata.get("letter_title"),
        "no_page_citation": metadata.get("no_page_citation"),
        "citation_mode": metadata.get("citation_mode"),
        "excerpt": compact_preview(content, limit=240),
    }


def evidence_from_docs(
    docs,
    limit,
    normalize_metadata,
    clean_text,
    compact_preview,
    format_citation_fn,
):
    evidence = []
    seen = set()
    for doc in docs[:limit]:
        item = evidence_from_doc(
            doc,
            index=len(evidence) + 1,
            normalize_metadata=normalize_metadata,
            clean_text=clean_text,
            compact_preview=compact_preview,
            format_citation_fn=format_citation_fn,
        )
        key = (
            item.get("source"),
            item.get("printed_page"),
            item.get("citation_page"),
            item.get("article"),
            item.get("excerpt")[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        evidence.append(item)
    return evidence


def extract_answer_citation_lines(answer, normalize_final_answer):
    normalized = normalize_final_answer(answer)
    citations = []
    for line in normalized.splitlines():
        match = re.match(r"\s*(?:\d+[.\u3001]|\(\d+\))\s*(.+?第\d+页。?)\s*$", line)
        if match:
            citations.append(match.group(1).strip())
    return citations


def citation_match_key(citation, normalize_for_match):
    return normalize_for_match(citation or "")


def evidence_matches_citation(item, citation, normalize_for_match):
    citation_key = citation_match_key(citation, normalize_for_match)
    if not citation_key:
        return False

    candidates = [
        item.get("citation"),
        item.get("sentence_citation"),
        item.get("detailed_citation"),
    ]
    for candidate in candidates:
        candidate_key = citation_match_key(candidate, normalize_for_match)
        if candidate_key and (candidate_key in citation_key or citation_key in candidate_key):
            return True

    page_match = re.search(r"第(\d+)页", citation or "")
    citation_page = str(page_match.group(1)) if page_match else ""
    item_pages = {
        str(item.get("printed_page") or ""),
        str(item.get("citation_page") or ""),
    }
    item_pages.discard("")
    if citation_page and citation_page in item_pages:
        series = citation_match_key(item.get("series") or "", normalize_for_match)
        source = citation_match_key(item.get("source") or item.get("source_file") or "", normalize_for_match)
        article = citation_match_key(item.get("article") or item.get("section") or "", normalize_for_match)
        if (
            (series and series in citation_key)
            or (source and source in citation_key)
            or (article and article in citation_key)
        ):
            return True

    return False


def filter_evidence_to_answer(answer, evidence, fallback_limit, normalize_final_answer, normalize_for_match):
    evidence = evidence or []
    citations = extract_answer_citation_lines(answer, normalize_final_answer)
    if not citations:
        return [{**item, "id": f"E{index}"} for index, item in enumerate(evidence[:fallback_limit], start=1)]

    matched = []
    seen = set()
    for citation in citations:
        for item in evidence:
            if not evidence_matches_citation(item, citation, normalize_for_match):
                continue
            key = (
                item.get("source"),
                item.get("printed_page"),
                item.get("citation_page"),
                item.get("paragraph_id"),
                item.get("excerpt"),
            )
            if key in seen:
                continue
            seen.add(key)
            matched.append({**item, "answer_citation": citation})
            break

    if not matched:
        return [{**item, "id": f"E{index}"} for index, item in enumerate(evidence[:fallback_limit], start=1)]

    return [{**item, "id": f"E{index}"} for index, item in enumerate(matched, start=1)]


def _split_answer_body_and_citations(answer):
    marker_re = re.compile(r"(?im)^\s*(?:\*+)?\s*引(?:文|用)注释\s*(?:\*+)?\s*$")
    lines = answer.splitlines()
    for index, line in enumerate(lines):
        if marker_re.match(line):
            body = "\n".join(lines[:index]).rstrip()
            return body, True
    return answer.rstrip(), False


def _preferred_citation_text(item):
    return (
        item.get("answer_citation")
        or item.get("detailed_citation")
        or item.get("citation")
        or item.get("sentence_citation")
    )


def _build_citation_section_lines(evidence, limit):
    lines = ["*引文注释*"]
    for index, item in enumerate((evidence or [])[:limit], start=1):
        citation = _preferred_citation_text(item)
        if not citation:
            continue
        lines.append(f"{index}. {citation}")
    return lines if len(lines) > 1 else []


def repair_answer_citations(
    answer,
    evidence,
    fallback_limit,
    normalize_final_answer,
    normalize_for_match,
):
    normalized = normalize_final_answer(answer)
    evidence = evidence or []
    if not normalized.strip() or not evidence:
        return normalized

    citation_lines = extract_answer_citation_lines(normalized, normalize_final_answer)
    matched = []
    seen = set()
    for citation in citation_lines:
        for item in evidence:
            if not evidence_matches_citation(item, citation, normalize_for_match):
                continue
            key = (
                item.get("source"),
                item.get("printed_page"),
                item.get("citation_page"),
                item.get("paragraph_id"),
                item.get("excerpt"),
            )
            if key in seen:
                continue
            seen.add(key)
            matched.append({**item, "answer_citation": citation})
            break

    replacement = matched if matched and len(matched) == len(citation_lines) else evidence[:fallback_limit]
    section_lines = _build_citation_section_lines(replacement, fallback_limit)
    if not section_lines:
        return normalized

    body, _had_marker = _split_answer_body_and_citations(normalized)
    if body.strip():
        return body.rstrip() + "\n\n" + "\n".join(section_lines)
    return "\n".join(section_lines)


def audit_answer_citations(answer, evidence, normalize_final_answer, normalize_for_match):
    normalized = normalize_final_answer(answer)
    issues = []
    forbidden = ["PDF第", "pdf_page", "PDF page", "打印页低信任"]
    for token in forbidden:
        if token in normalized:
            issues.append({"type": "forbidden_token", "token": token})

    evidence_citations = {item.get("citation") for item in evidence or []}
    evidence_citations |= {item.get("sentence_citation") for item in evidence or []}
    evidence_citations |= {item.get("detailed_citation") for item in evidence or []}
    evidence_citations = {item for item in evidence_citations if item}
    citation_lines = extract_answer_citation_lines(normalized, normalize_final_answer)

    if citation_lines and not evidence_citations:
        issues.append({"type": "citation_without_verified_evidence"})

    for citation in citation_lines:
        if evidence_citations and not any(
            evidence_matches_citation(item, citation, normalize_for_match) for item in evidence or []
        ):
            issues.append({"type": "citation_not_in_evidence", "citation": citation})

    return {
        "ok": not issues,
        "issues": issues,
        "evidence_count": len(evidence or []),
        "answer": normalized,
    }
