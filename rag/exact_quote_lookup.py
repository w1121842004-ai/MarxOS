import json
import re
from pathlib import Path

from langchain_core.documents import Document

from rag.core_classics import classic_entries_for_query, load_core_classics


DEFAULT_OCR_CACHE_DIR = Path("data/ocr_cache")
BOOK_BY_SOURCE = {
    **{f"mea{i:02d}.pdf": f"马克思恩格斯文集 第{i}卷" for i in range(1, 11)},
    **{f"mes{i:02d}.pdf": f"马克思恩格斯选集 第{i}卷" for i in range(1, 5)},
}
PREFERRED_CLASSIC_IDS_BY_QUOTE = {
    "\u5168\u4e16\u754c\u65e0\u4ea7\u8005\u8054\u5408\u8d77\u6765": {"communist_manifesto"},
}
STRICT_SCOPED_CLASSIC_IDS = {"critique_gotha_programme"}


def normalize_quote(text):
    text = str(text or "")
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", text).lower()


def extract_query_quote(query):
    query = str(query or "").strip()

    for pattern in [r"“([^”]{4,})”", r'"([^"]{4,})"', r"『([^』]{4,})』", r"「([^」]{4,})」"]:
        match = re.search(pattern, query)
        if match:
            return match.group(1).strip()

    query = re.sub(r"(出自哪里|出自哪|在哪一页|哪一页|哪页|页码|这句话|请给出|准确页码|原文|出处)", "", query)
    query = query.strip(" ？?。，“”\"'：:")

    return query if len(normalize_quote(query)) >= 5 else ""


def cache_path_for_page(ocr_cache_dir, source, pdf_page):
    stem = source.replace(".pdf", "")
    return Path(ocr_cache_dir) / stem / f"page_{pdf_page}.json"


def iter_candidate_pages(query, ocr_cache_dir=DEFAULT_OCR_CACHE_DIR, scoped=True):
    entries = classic_entries_for_query(query)

    if entries and scoped:
        for entry in entries:
            for pdf_page in range(entry["start_page"], entry["end_page"] + 1):
                yield entry, cache_path_for_page(ocr_cache_dir, entry["source"], pdf_page)
        return

    for path in Path(ocr_cache_dir).glob("*/page_*.json"):
        yield None, path


def snippet_around(text, query, window=180):
    normalized_chars = []
    original_positions = []

    for index, char in enumerate(text):
        normalized = normalize_quote(char)
        if not normalized:
            continue

        normalized_chars.append(normalized)
        original_positions.append(index)

    normalized_text = "".join(normalized_chars)
    normalized_query = normalize_quote(query)
    index = normalized_text.find(normalized_query)

    if index < 0:
        return " ".join(text.split())[: window * 2]

    original_index = original_positions[index]

    start = max(0, original_index - window)
    end = min(len(text), original_index + len(query) + window)

    return " ".join(text[start:end].split())


def entry_covers_pdf_page(entry, source, pdf_page, before=20, after=80):
    if entry.get("source") != source or pdf_page is None:
        return False

    return entry["start_page"] - before <= pdf_page <= entry["end_page"] + after


def metadata_from_page(page, entry, path, preferred_entries=None):
    source = page.get("source") or (entry or {}).get("source") or f"{path.parent.name}.pdf"
    pdf_page = page.get("page_num")

    if pdf_page is None:
        match = re.search(r"page_(\d+)\.json$", path.name)
        pdf_page = int(match.group(1)) if match else None

    if entry is None:
        for candidate in preferred_entries or []:
            if entry_covers_pdf_page(candidate, source, pdf_page):
                entry = candidate
                break

    if entry is None:
        for classic in load_core_classics():
            for candidate in classic.get("entries") or []:
                if not entry_covers_pdf_page(candidate, source, pdf_page):
                    continue

                entry = {
                    **candidate,
                    "classic_id": classic.get("id"),
                    "classic_title": classic.get("title"),
                    "classic_author": classic.get("author"),
                    "classic_work_year": classic.get("work_year"),
                    "classic_work_type": classic.get("work_type"),
                }
                break

            if entry is not None:
                break

    metadata = {
        "book": BOOK_BY_SOURCE.get(source) or page.get("book_title") or source,
        "article": (entry or {}).get("article") or page.get("title_candidate") or (entry or {}).get("classic_title"),
        "section": (entry or {}).get("article") or page.get("title_candidate"),
        "page": pdf_page,
        "printed_page": None,
        "pdf_page": pdf_page,
        "citation_page": pdf_page,
        "citation_page_type": "pdf_page",
        "source": source,
        "ocr": True,
        "match_type": "exact_quote",
        "confidence": 1.0,
        "classic_id": (entry or {}).get("classic_id"),
        "classic_title": (entry or {}).get("classic_title"),
        "classic_author": (entry or {}).get("classic_author"),
        "classic_work_year": (entry or {}).get("classic_work_year"),
        "classic_work_type": (entry or {}).get("classic_work_type"),
        "entry_type": (entry or {}).get("entry_type"),
        "entry_priority": (entry or {}).get("priority"),
    }

    return metadata


def hit_quality_rank(page, text, normalized_quote):
    normalized_text = normalize_quote(text)
    quote_index = normalized_text.find(normalized_quote)
    quote_ratio = quote_index / max(len(normalized_text), 1) if quote_index >= 0 else 1
    title_candidate = str(page.get("title_candidate") or "")
    title_norm = normalize_quote(title_candidate)

    if page.get("page_type") in {"toc", "title_page"}:
        return 4

    if len(normalized_text) <= len(normalized_quote) + 20:
        return 4

    if any(marker in title_norm for marker in ["索引", "目录", "注释"]):
        return 3

    if any(marker in title_norm for marker in ["序言", "说明", "编者"]):
        return 2

    annotation_markers = [
        normalize_quote(marker)
        for marker in [
            "重要著作",
            "基本理论的重要著作",
            "本文节选自",
            "选编说明",
            "编者注",
        ]
    ]
    if any(marker and marker in normalized_text[:250] for marker in annotation_markers):
        return 3

    if re.match(r"^\d+", title_norm) and "是恩格斯" in title_norm:
        return 3

    ending_markers = [
        normalize_quote(marker)
        for marker in [
            "共产党人不屑于隐瞒自己的观点和意图",
            "无产者在这个革命中失去的只是锁链",
            "获得的将是整个世界",
        ]
    ]
    if any(marker and marker in normalized_text for marker in ending_markers):
        return 0

    if quote_ratio < 0.15:
        return 2

    return 1


def collect_hits(query, normalized_quote, ocr_cache_dir, scoped):
    quote = extract_query_quote(query)
    preferred_entries = classic_entries_for_query(query)
    preferred_classic_ids = {
        entry.get("classic_id") for entry in preferred_entries if entry.get("classic_id")
    }
    for quote, classic_ids in PREFERRED_CLASSIC_IDS_BY_QUOTE.items():
        if quote and (quote in normalized_quote or normalized_quote in quote):
            preferred_classic_ids.update(classic_ids)

    hits = []
    seen = set()

    for entry, path in iter_candidate_pages(query, ocr_cache_dir, scoped=scoped):
        if not path.exists():
            continue

        with path.open("r", encoding="utf-8") as f:
            page = json.load(f)

        if page.get("page_type") == "toc":
            continue

        cleaned_text = page.get("cleaned_text") or ""
        if normalized_quote not in normalize_quote(cleaned_text):
            continue

        metadata = metadata_from_page(page, entry, path, preferred_entries=preferred_entries)
        metadata["lookup_scope"] = "core_classic" if scoped and entry else "global"
        key = (metadata["source"], metadata.get("pdf_page"))
        if key in seen:
            continue

        seen.add(key)
        priority = (entry or {}).get("priority", metadata.get("entry_priority") or 99)
        quality = hit_quality_rank(page, cleaned_text, normalized_quote)
        hits.append(
            (
                (quality, priority),
                Document(
                    page_content=snippet_around(cleaned_text, quote),
                    metadata=metadata,
                ),
            )
        )

    if preferred_classic_ids:
        trusted_hits = [
            hit for hit in hits
            if hit[1].metadata.get("classic_id") in preferred_classic_ids
        ]
        if trusted_hits:
            hits = trusted_hits

    hits.sort(key=lambda item: (item[0], item[1].metadata.get("source", ""), item[1].metadata.get("pdf_page") or 0))

    return hits


def exact_quote_lookup(query, ocr_cache_dir=DEFAULT_OCR_CACHE_DIR, limit=5):
    quote = extract_query_quote(query)
    normalized_quote = normalize_quote(quote)

    if len(normalized_quote) < 5:
        return []

    preferred_entries = classic_entries_for_query(query)
    hits = collect_hits(query, normalized_quote, ocr_cache_dir, scoped=True)

    # For selected classics with known noisy cross-book contamination, require
    # scoped confirmation and avoid global fallback.
    preferred_ids = {
        entry.get("classic_id") for entry in preferred_entries if entry.get("classic_id")
    }
    if preferred_ids & STRICT_SCOPED_CLASSIC_IDS and not hits:
        return []

    global_hits = collect_hits(query, normalized_quote, ocr_cache_dir, scoped=False)

    if global_hits:
        merged = {}
        for sort_key, doc in hits + global_hits:
            key = (doc.metadata.get("source"), doc.metadata.get("pdf_page"))
            previous = merged.get(key)
            if previous is None or sort_key < previous[0]:
                merged[key] = (sort_key, doc)
        hits = sorted(
            merged.values(),
            key=lambda item: (
                item[0],
                item[1].metadata.get("source", ""),
                item[1].metadata.get("pdf_page") or 0,
            ),
        )

    return [doc for _, doc in hits[:limit]]
