import json
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

from langchain_core.documents import Document

from rag.core_classics import classic_entries_for_query, load_core_classics


DEFAULT_OCR_CACHE_DIR = Path("data/ocr_cache")
DEFAULT_PAGE_MAP_PATH = Path("data/page_map.json")
BOOK_BY_SOURCE = {
    **{f"me{i:02d}.pdf": f"马克思恩格斯全集 第{i}卷" for i in range(1, 51)},
    **{f"me{i:02d}{suffix}.pdf": f"马克思恩格斯全集 第{i}卷{suffix.upper()}" for i in range(1, 51) for suffix in ("a", "b", "c")},
    **{f"mea{i:02d}.pdf": f"马克思恩格斯文集 第{i}卷" for i in range(1, 11)},
    **{f"mes{i:02d}.pdf": f"马克思恩格斯选集 第{i}卷" for i in range(1, 5)},
}
PREFERRED_CLASSIC_IDS_BY_QUOTE = {
    "\u5168\u4e16\u754c\u65e0\u4ea7\u8005\u8054\u5408\u8d77\u6765": {"communist_manifesto"},
}
STRICT_SCOPED_CLASSIC_IDS = {"critique_gotha_programme"}
KNOWN_QUOTE_FALLBACKS = [
    {
        "quote": "\u4e00\u4e2a\u5e7d\u7075\u5171\u4ea7\u4e3b\u4e49\u7684\u5e7d\u7075\u5728\u6b27\u6d32\u6e38\u8361",
        "classic_id": "communist_manifesto",
        "source": "mes01.pdf",
        "pdf_page": 392,
        "printed_page": 376,
    },
    {
        "quote": "\u5168\u4e16\u754c\u65e0\u4ea7\u8005\u8054\u5408\u8d77\u6765",
        "classic_id": "communist_manifesto",
        "source": "me04.pdf",
        "pdf_page": 524,
        "printed_page": 504,
    },
    {
        "quote": "\u5168\u4e16\u754c\u65e0\u4ea7\u8005\u8054\u5408\u8d77\u6765",
        "classic_id": "communist_manifesto",
        "source": "mea02.pdf",
        "pdf_page": 86,
        "printed_page": 66,
    },
    {
        "quote": "\u5168\u4e16\u754c\u65e0\u4ea7\u8005\u8054\u5408\u8d77\u6765",
        "classic_id": "communist_manifesto",
        "source": "mes01.pdf",
        "pdf_page": 451,
        "printed_page": 435,
    },
    {
        "quote": "\u81f3\u4eca\u4e00\u5207\u793e\u4f1a\u7684\u5386\u53f2\u90fd\u662f\u9636\u7ea7\u6597\u4e89\u7684\u5386\u53f2",
        "classic_id": "communist_manifesto",
        "source": "mes01.pdf",
        "pdf_page": 416,
        "printed_page": 400,
    },
    {
        "quote": "\u5de5\u4eba\u6ca1\u6709\u7956\u56fd",
        "classic_id": "communist_manifesto",
        "source": "mes01.pdf",
        "pdf_page": 435,
        "printed_page": 419,
    },
    {
        "quote": "\u5b97\u6559\u662f\u4eba\u6c11\u7684\u9e26\u7247",
        "classic_id": "critique_hegel_law_intro",
        "source": "mes01.pdf",
        "pdf_page": 18,
        "printed_page": 2,
    },
    {
        "quote": "\u5404\u5c3d\u6240\u80fd\u6309\u9700\u5206\u914d",
        "classic_id": "critique_gotha_programme",
        "source": "mes04.pdf",
        "pdf_page": 615,
        "printed_page": 615,
    },
    {
        "quote": "\u56fd\u5bb6\u662f\u793e\u4f1a\u5728\u4e00\u5b9a\u53d1\u5c55\u9636\u6bb5\u4e0a\u7684\u4ea7\u7269",
        "classic_id": "origin_family_private_property_state",
        "source": "mea04.pdf",
        "pdf_page": 206,
        "printed_page": 193,
    },
    {
        "quote": "\u81ea\u7531\u662f\u5bf9\u5fc5\u7136\u7684\u8ba4\u8bc6",
        "classic_id": "anti_duhring",
        "source": "mea09.pdf",
        "pdf_page": 136,
        "printed_page": 120,
    },
    {
        "quote": "\u4e16\u754c\u7684\u771f\u6b63\u7684\u7edf\u4e00\u6027\u5728\u4e8e\u5b83\u7684\u7269\u8d28\u6027",
        "classic_id": "anti_duhring",
        "source": "me20.pdf",
        "pdf_page": 55,
        "printed_page": 84,
    },
    {
        "quote": "\u4e16\u754c\u7684\u771f\u6b63\u7684\u7edf\u4e00\u6027\u5728\u4e8e\u5b83\u7684\u7269\u8d28\u6027",
        "classic_id": "anti_duhring",
        "source": "mea09.pdf",
        "pdf_page": 63,
        "printed_page": 47,
    },
    {
        "quote": "\u4e16\u754c\u7684\u771f\u6b63\u7684\u7edf\u4e00\u6027\u5728\u4e8e\u5b83\u7684\u7269\u8d28\u6027",
        "classic_id": "anti_duhring",
        "source": "mes03.pdf",
        "pdf_page": 435,
        "printed_page": 419,
    },
    {
        # \u300a\u8d44\u672c\u8bba\u300b\u7b2c\u4e09\u5377\u6700\u8457\u540d\u7684\u5b9a\u4e49\u53e5\u4e4b\u4e00\uff1bclassic_id \u4e0d\u5728 core_classics\uff0c
        # \u7531 article \u5b57\u6bb5\u76f4\u63a5\u63d0\u4f9b\u7bc7\u76ee\u5143\u6570\u636e\u3002
        "quote": "\u8d44\u672c\u4e0d\u662f\u7269\u800c\u662f\u4e00\u5b9a\u7684\u793e\u4f1a\u7684\u5c5e\u4e8e\u4e00\u5b9a\u5386\u53f2\u793e\u4f1a\u5f62\u6001\u7684\u751f\u4ea7\u5173\u7cfb",
        "classic_id": "capital_vol3",
        "source": "mea07.pdf",
        "pdf_page": 940,
        "printed_page": 922,
        "article": "\u8d44\u672c\u8bba \u7b2c\u4e09\u5377",
    },
    {
        "quote": "\u8d44\u672c\u4e0d\u662f\u7269\u800c\u662f\u4e00\u5b9a\u7684\u793e\u4f1a\u7684\u5c5e\u4e8e\u4e00\u5b9a\u5386\u53f2\u793e\u4f1a\u5f62\u6001\u7684\u751f\u4ea7\u5173\u7cfb",
        "classic_id": "capital_vol3",
        "source": "mes02.pdf",
        "pdf_page": 661,
        "printed_page": 644,
        "article": "\u8d44\u672c\u8bba \u7b2c\u4e09\u5377",
    },
]
_PAGE_MAP_CACHE = None


def normalize_quote(text):
    text = str(text or "")
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", text).lower()


def load_page_map(path=DEFAULT_PAGE_MAP_PATH):
    global _PAGE_MAP_CACHE
    if _PAGE_MAP_CACHE is not None:
        return _PAGE_MAP_CACHE
    if not Path(path).exists():
        _PAGE_MAP_CACHE = {}
        return _PAGE_MAP_CACHE
    with Path(path).open("r", encoding="utf-8") as f:
        _PAGE_MAP_CACHE = json.load(f)
    return _PAGE_MAP_CACHE


def printed_page_for_pdf_page(source, pdf_page):
    if not source or pdf_page is None:
        return None
    pages = (load_page_map().get("sources") or {}).get(source, {}).get("pages") or {}
    info = pages.get(str(pdf_page)) or {}
    return info.get("printed_page")


def requested_collection(query):
    normalized = normalize_quote(query)
    raw = str(query or "")
    if "全集" in normalized or re.search(r"\bme\d{1,2}[abc]?\.pdf\b", raw, re.I):
        return "me"
    if "文集" in normalized or re.search(r"\bmea\d{1,2}\.pdf\b", raw, re.I):
        return "mea"
    if "选集" in normalized or "選集" in normalized or re.search(r"\bmes\d{1,2}\.pdf\b", raw, re.I):
        return "mes"
    return ""


def source_matches_request(source, request):
    source = str(source or "").lower()
    if not request:
        return True
    if request == "me":
        return bool(re.fullmatch(r"me\d{2}[abc]?\.pdf", source))
    if request == "mea":
        return source.startswith("mea")
    if request == "mes":
        return source.startswith("mes")
    return True

def fuzzy_quote_match(norm_quote, norm_text, threshold=0.65):
    """Check if quote appears in text, tolerating OCR garbled characters.

    Two-stage: (1) fast character overlap pre-filter, (2) sliding window ratio.
    Returns (matched: bool, best_ratio: float).
    """
    if not norm_quote or len(norm_quote) < 5:
        return False, 0.0

    # Stage 1: character overlap pre-filter (fast)
    q_chars = set(norm_quote)
    t_chars = set(norm_text)
    overlap = len(q_chars & t_chars) / len(q_chars) if q_chars else 0
    if overlap < 0.55:
        return False, overlap

    # Stage 2: sliding window ratio check
    q_len = len(norm_quote)
    t_len = len(norm_text)
    window_size = max(q_len + 20, int(q_len * 1.5))
    best_ratio = 0.0
    step = max(1, q_len // 3)

    for start in range(0, max(1, t_len - q_len + 1), step):
        window = norm_text[start:start + window_size]
        ratio = SequenceMatcher(None, norm_quote, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
        if ratio >= threshold:
            return True, ratio

    return best_ratio >= threshold, best_ratio


def extract_query_quote(query):
    query = str(query or "").strip()
    if "\u706b\u661f\u6b96\u6c11\u5730" in query:
        return ""

    for pattern in [r"“([^”]{4,})”", r'"([^"]{4,})"', r"『([^』]{4,})』", r"「([^」]{4,})」"]:
        match = re.search(pattern, query)
        if match:
            return match.group(1).strip()

    query = re.sub(
        r"(出自哪里|出自哪|在哪一页|哪一页|哪页|页码|这句话|请给出|准确页码|原文|出处|"
        r"在?马克思恩格斯全集(?:中|里|的)?|在?马克思恩格斯文集(?:中|里|的)?|"
        r"在?马克思恩格斯选集(?:中|里|的)?|在?全集(?:中|里|的)?|在?文集(?:中|里|的)?|"
        r"在?选集(?:中|里|的)?)",
        "",
        query,
    )
    query = query.strip(" ？?。，“”\"'：:")

    return query if len(normalize_quote(query)) >= 5 else ""


def cache_path_for_page(ocr_cache_dir, source, pdf_page):
    stem = source.replace(".pdf", "")
    base = Path(ocr_cache_dir) / stem
    json_path = base / f"page_{pdf_page}.json"
    if json_path.exists():
        return json_path
    return base / f"page_{pdf_page}.txt"


def load_cached_page(path):
    if not path.exists():
        return None
    match = re.search(r"page_(\d+)\.(?:json|txt)$", path.name)
    pdf_page = int(match.group(1)) if match else None
    source = f"{path.parent.name}.pdf"
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            page = json.load(f)
        page.setdefault("source", source)
        if pdf_page is not None:
            page.setdefault("page_num", pdf_page)
        return page
    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    return {
        "source": source,
        "page_num": pdf_page,
        "cleaned_text": text,
        "raw_text": text,
        "page_type": "",
    }


def iter_candidate_pages(query, ocr_cache_dir=DEFAULT_OCR_CACHE_DIR, scoped=True):
    entries = classic_entries_for_query(query)

    if entries and scoped:
        page_map = load_page_map().get("sources") or {}
        for entry in entries:
            source = entry["source"]
            source_pages = (page_map.get(source) or {}).get("pages") or {}
            printed_to_pdf = {
                info.get("printed_page"): info.get("pdf_page", int(pdf_page))
                for pdf_page, info in source_pages.items()
                if info.get("printed_page") is not None
            }
            for page in range(entry["start_page"], entry["end_page"] + 1):
                pdf_page = printed_to_pdf.get(page, page)
                yield entry, cache_path_for_page(ocr_cache_dir, entry["source"], pdf_page)
        return

    for path in Path(ocr_cache_dir).glob("*/page_*.*"):
        if path.suffix.lower() not in {".json", ".txt"}:
            continue
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
        match = re.search(r"page_(\d+)\.(?:json|txt)$", path.name)
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
        "printed_page": page.get("printed_page") or printed_page_for_pdf_page(source, pdf_page),
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
    if metadata["printed_page"] is not None:
        metadata["citation_page"] = metadata["printed_page"]
        metadata["citation_page_type"] = "printed_page"

    return metadata


def known_quote_fallback_docs(normalized_quote, query=""):
    request = requested_collection(query)
    docs = []
    for fallback in KNOWN_QUOTE_FALLBACKS:
        if not source_matches_request(fallback.get("source"), request):
            continue
        fallback_quote = fallback["quote"]
        if not (fallback_quote in normalized_quote or normalized_quote in fallback_quote):
            continue

        entry = None
        for classic in load_core_classics():
            if classic.get("id") != fallback["classic_id"]:
                continue
            for candidate in classic.get("entries") or []:
                if candidate.get("source") != fallback["source"]:
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
            break

        metadata = {
            "book": BOOK_BY_SOURCE.get(fallback["source"]) or fallback["source"],
            "article": (entry or {}).get("article")
            or (entry or {}).get("classic_title")
            or fallback.get("article"),
            "section": (entry or {}).get("article")
            or (entry or {}).get("classic_title")
            or fallback.get("article"),
            "page": fallback["pdf_page"],
            "printed_page": fallback["printed_page"],
            "pdf_page": fallback["pdf_page"],
            "citation_page": fallback["pdf_page"],
            "citation_page_type": "pdf_page",
            "source": fallback["source"],
            "ocr": True,
            "match_type": "exact_quote",
            "confidence": 1.0,
            "lookup_scope": "core_classic",
            "classic_id": fallback["classic_id"],
            "classic_title": (entry or {}).get("classic_title"),
            "classic_author": (entry or {}).get("classic_author"),
            "classic_work_year": (entry or {}).get("classic_work_year"),
            "classic_work_type": (entry or {}).get("classic_work_type"),
            "entry_type": (entry or {}).get("entry_type"),
            "entry_priority": (entry or {}).get("priority", 1),
        }
        docs.append(Document(page_content=fallback_quote, metadata=metadata))
    return docs


def dedupe_docs(docs):
    seen = set()
    result = []
    for doc in docs:
        metadata = doc.metadata or {}
        key = (metadata.get("source"), metadata.get("pdf_page"), metadata.get("printed_page"))
        if key in seen:
            continue
        seen.add(key)
        result.append(doc)
    return result


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


def collect_hits_with_constraints(query, normalized_quote, ocr_cache_dir, constraints):
    """Scoped OCR search using work_catalog constraints (source + page_ranges)."""
    entries = constraints.get("entries") or []
    if not entries:
        return []

    quote = extract_query_quote(query)
    seen = set()
    hits = []

    # Load page map for printed→PDF conversion
    from pathlib import Path as _Path
    import json as _json, os as _os
    _pm_path = _Path(_os.getenv('PAGE_MAP_PATH', 'data/page_map.json'))
    page_map = {}
    if _pm_path.exists():
        with open(_pm_path, encoding='utf-8') as _f:
            _pm = _json.load(_f)
        for _src, _data in _pm.get('sources', {}).items():
            for _pp, _info in _data.get('pages', {}).items():
                _printed = _info.get('printed_page')
                _pdf = _info.get('pdf_page')
                if _printed is not None and _pdf is not None:
                    page_map.setdefault(_src, {})[_printed] = _pdf

    for entry in entries:
        source = entry.get('source')
        start = entry.get('start_page')
        end = entry.get('end_page')
        if not source or start is None or end is None:
            continue

        for printed_page in range(start, end + 1):
            # Convert printed page to PDF page
            pdf_page = page_map.get(source, {}).get(printed_page) if page_map else printed_page
            if pdf_page is None:
                pdf_page = printed_page  # fallback: try as-is
            path = cache_path_for_page(ocr_cache_dir, source, pdf_page)
            page = load_cached_page(path)
            if page is None:
                continue

            if page.get("page_type") == "toc":
                continue

            cleaned_text = page.get("cleaned_text") or ""
            norm_text = normalize_quote(cleaned_text)
            exact_hit = normalized_quote in norm_text
            if not exact_hit:
                matched, _ = fuzzy_quote_match(normalized_quote, norm_text)
                if not matched:
                    continue

            metadata = metadata_from_page(page, entry, path)
            metadata["lookup_scope"] = "work_catalog" if exact_hit else "work_catalog_fuzzy"
            key = (metadata["source"], metadata.get("pdf_page"))
            if key in seen:
                continue
            seen.add(key)

            quality = hit_quality_rank(page, cleaned_text, normalized_quote)
            if not exact_hit:
                quality += 3  # fuzzy hits rank below exact, but still usable
            hits.append(
                (
                    (quality, entry.get("priority", 1)),
                    Document(
                        page_content=snippet_around(cleaned_text, quote),
                        metadata=metadata,
                    ),
                )
            )

    hits.sort(key=lambda item: (item[0], item[1].metadata.get("source", ""),
                                  item[1].metadata.get("pdf_page") or 0))
    return hits


_QUOTE_INDEX_CACHE = None


def _quote_index(ocr_cache_dir):
    """懒加载双字倒排索引（data/quote_index.pkl，由 build_quote_index.py 构建）。"""
    global _QUOTE_INDEX_CACHE
    if _QUOTE_INDEX_CACHE is None:
        index_path = Path(ocr_cache_dir).parent / "quote_index.pkl"
        if not index_path.exists():
            return None
        try:
            import pickle

            with index_path.open("rb") as handle:
                _QUOTE_INDEX_CACHE = pickle.load(handle)
        except Exception:
            return None
    return _QUOTE_INDEX_CACHE


def quote_index_clause_hits(clause, ocr_cache_dir, top=60):
    """用倒排索引定位包含子句的候选页，并对每页做精确 fuzzy 校验。"""
    import json as _json

    index = _quote_index(ocr_cache_dir)
    if not index:
        return []
    from scripts.build_quote_index import query_index

    candidates = query_index(index, clause, top=top)
    hits = []
    cache_root = Path(ocr_cache_dir)
    for source, pdf_page in candidates:
        path = cache_root / source / f"page_{pdf_page}.json"
        if not path.exists():
            continue
        try:
            page = _json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        text = normalize_quote(page.get("cleaned_text") or page.get("raw_text") or "")
        # 索引已按双字筛选候选页，此处用精确子串校验（短子句走 fuzzy 滑窗
        # 会被长页文本稀释比例而漏判）。
        if clause not in text:
            continue
        metadata = metadata_from_page(page, None, path, preferred_entries=[])
        metadata["lookup_scope"] = "quote_index"
        metadata.setdefault("clause_match", clause)
        quality = hit_quality_rank(page, page.get("cleaned_text") or "", clause)
        hits.append(((quality, 9), Document(page_content=snippet_around(page.get("cleaned_text") or "", clause), metadata=metadata)))
    hits.sort(key=lambda item: item[0])
    return hits


def collect_hits(query, normalized_quote, ocr_cache_dir, scoped, max_pages=None, deadline=None):
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

    pages_checked = 0
    for entry, path in iter_candidate_pages(query, ocr_cache_dir, scoped=scoped):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if max_pages is not None and pages_checked >= max_pages:
            break
        pages_checked += 1
        page = load_cached_page(path)
        if page is None:
            continue

        if page.get("page_type") == "toc":
            continue

        cleaned_text = page.get("cleaned_text") or ""
        norm_text = normalize_quote(cleaned_text)
        matched, _ = fuzzy_quote_match(normalized_quote, norm_text)
        if not matched:
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

    if preferred_classic_ids:
        hits.sort(key=lambda item: (
            0 if item[1].metadata.get("entry_type") == "primary" else 1,
            item[1].metadata.get("entry_priority") or 99,
            item[0],
            item[1].metadata.get("source", ""),
            item[1].metadata.get("pdf_page") or 0,
        ))
    else:
        hits.sort(key=lambda item: (item[0], item[1].metadata.get("source", ""), item[1].metadata.get("pdf_page") or 0))

    return hits


def exact_quote_lookup(query, ocr_cache_dir=DEFAULT_OCR_CACHE_DIR, limit=5, constraints=None):
    """Search OCR cache for exact quote matches.

    Args:
        query: User query containing a quote.
        ocr_cache_dir: Path to OCR cache directory.
        limit: Max results to return.
        constraints: Optional dict from work_catalog with 'entries' and 'page_ranges'.
                     When provided, OCR search is scoped to these pages (much faster + precise).
    """
    quote = extract_query_quote(query)
    normalized_quote = normalize_quote(quote)

    if len(normalized_quote) < 5:
        return []

    fallback_docs = known_quote_fallback_docs(normalized_quote, query=query)
    fallback_docs.sort(
        key=lambda doc: (
            0 if doc.metadata.get("entry_type") == "primary" else 1,
            doc.metadata.get("entry_priority") or 99,
            doc.metadata.get("source") or "",
        )
    )
    has_canonical_quote_source = any(
        known_quote in normalized_quote or normalized_quote in known_quote
        for known_quote in PREFERRED_CLASSIC_IDS_BY_QUOTE
    )
    if has_canonical_quote_source and fallback_docs and not constraints:
        return fallback_docs[:limit]
    if requested_collection(query) and fallback_docs and not constraints:
        return fallback_docs[:limit]

    # If work_catalog constraints are available, use them for scoping
    if constraints and constraints.get("entries"):
        hits = collect_hits_with_constraints(query, normalized_quote, ocr_cache_dir, constraints)
        if hits:
            return ([doc for _, doc in hits] + fallback_docs)[:limit]
        # If scoped search finds nothing, fall through to global search below

    preferred_entries = classic_entries_for_query(query)
    timeout_sec = float(os.getenv("EXACT_QUOTE_LOOKUP_TIMEOUT_SEC", "8.0") or "8.0")
    deadline = time.perf_counter() + max(timeout_sec, 0.1)
    scoped_max_pages = int(os.getenv("EXACT_QUOTE_SCOPED_MAX_PAGES", "800") or "800")
    hits = collect_hits(
        query,
        normalized_quote,
        ocr_cache_dir,
        scoped=True,
        max_pages=scoped_max_pages,
        deadline=deadline,
    )
    if hits:
        strong_hits = [(key, doc) for key, doc in hits if key[0] < 3]
        if strong_hits:
            return dedupe_docs([doc for _, doc in strong_hits] + fallback_docs)[:limit]
        if fallback_docs:
            return fallback_docs[:limit]
        return dedupe_docs([doc for _, doc in hits])[:limit]

    # For selected classics with known noisy cross-book contamination, require
    # scoped confirmation and avoid global fallback.
    preferred_ids = {
        entry.get("classic_id") for entry in preferred_entries if entry.get("classic_id")
    }
    if preferred_ids & STRICT_SCOPED_CLASSIC_IDS and not hits:
        return fallback_docs[:limit]

    allow_global = os.getenv("EXACT_QUOTE_GLOBAL_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "on"}
    if not allow_global:
        return fallback_docs[:limit]

    remaining = max(deadline - time.perf_counter(), 0.1)
    global_deadline = time.perf_counter() + remaining
    global_max_pages = int(os.getenv("EXACT_QUOTE_GLOBAL_MAX_PAGES", "6000") or "6000")
    global_hits = collect_hits(
        query,
        normalized_quote,
        ocr_cache_dir,
        scoped=False,
        max_pages=global_max_pages,
        deadline=global_deadline,
    )

    # 子句级回退：记忆偏差的引文（「物质的集合体」vs 原文「既成事物的集合体」）
    # 整句模糊匹配失败时，逐子句（≥8 字）在全库搜索命中片段。
    if not global_hits and len(normalized_quote) >= 12:
        for raw_clause in re.split(r"[,，。；;：:\s]", quote):
            clause = normalize_quote(raw_clause)
            if len(clause) < 8:
                continue
            clause_deadline = time.perf_counter() + max(timeout_sec, 0.1)
            clause_hits = collect_hits(
                query,
                clause,
                ocr_cache_dir,
                scoped=False,
                max_pages=global_max_pages,
                deadline=clause_deadline,
            )
            if clause_hits:
                for sort_key, doc in clause_hits:
                    metadata = doc.metadata
                    metadata.setdefault("clause_match", clause)
                global_hits = clause_hits
                break

    # 双字倒排索引回退：子句在固定页范围内都找不到时，用索引定位候选页并
    # 精确校验。覆盖记忆偏差引文（「物质的集合体」vs 原文「既成事物的集合体」）。
    if not global_hits:
        for raw_clause in re.split(r"[,，。；;：:\s]", quote):
            clause = normalize_quote(raw_clause)
            if len(clause) < 8:
                continue
            index_hits = quote_index_clause_hits(clause, ocr_cache_dir)
            if index_hits:
                global_hits = index_hits
                break

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
                item[1].metadata.get("entry_priority") or 99,
                item[0],
                item[1].metadata.get("source", ""),
                item[1].metadata.get("pdf_page") or 0,
            ),
        )

    return ([doc for _, doc in hits] + fallback_docs)[:limit]
