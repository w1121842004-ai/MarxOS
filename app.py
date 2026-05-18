from openai import OpenAI

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import json
import os
import re
import sys
from rag.core_classics import classic_entries_for_query
from rag.exact_quote_lookup import exact_quote_lookup


load_dotenv()
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore/marx_reader_core")
OCR_CACHE_DIR = os.getenv("OCR_CACHE_DIR", "data/ocr_cache")
ARTICLE_MAP_PATH = os.getenv("ARTICLE_MAP_PATH", "rag/article_map_core.json")
DEFAULT_PUBLISHER = "人民出版社"
RERANK_DEBUG_ENV = "MARXOS_DEBUG_RERANK"
TRACE_ENV = "MARXOS_TRACE"
TRACE_ONLY_ENV = "MARXOS_TRACE_ONLY"
DEV_MODE_ENV = "MARXOS_DEV_MODE"
DEV_TOKEN_ENV = "MARXOS_DEV_TOKEN"
DEV_TOKEN_INPUT_ENV = "MARXOS_DEV_TOKEN_INPUT"
VOLUME_PUBLICATION_YEARS = {
    "me46a": "1979年",
    "me46b": "1979年",
    "me47": "2004年",
}


def repair_mojibake(text):
    if not isinstance(text, str):
        return text

    markers = ("Ã", "Â", "ã", "å", "æ", "ç", "è", "é", "ï", "ä")
    if not any(marker in text for marker in markers):
        return text

    def decode_run(match):
        run = match.group(0)
        if not any(marker in run for marker in markers):
            return run

        try:
            return run.encode("latin1").decode("utf-8")
        except UnicodeError:
            return run

    return re.sub(r"[\x00-\xff]+", decode_run, text)


def clean_text(text, fallback="未知"):
    if text is None or text == "":
        return repair_mojibake(fallback)

    return str(repair_mojibake(text)).strip() or fallback


def is_unreadable_query(query):
    query = str(query or "").strip()
    if not query:
        return True

    question_marks = query.count("?") + query.count("\uff1f")
    chinese_chars = sum(1 for char in query if "\u4e00" <= char <= "\u9fff")
    ascii_letters = sum(1 for char in query if char.isascii() and char.isalpha())

    return question_marks >= 4 and chinese_chars == 0 and ascii_letters == 0


def source_stem(metadata):
    source = clean_text(metadata.get("source"), "")
    return source.lower().replace(".pdf", "")


def printed_page_source_is_untrusted(metadata):
    """Return True when old vectorstore printed_page values are likely OCR artifacts.

    In supplement/selected-work PDFs, bare margin numbers can be manuscript page
    marks or note page references rather than book printed pages. We keep the old
    fields for compatibility, but citations should fall back to PDF pages.
    """
    stem = source_stem(metadata)
    return stem.startswith(("mea", "mes"))


def load_article_map():
    if not os.path.exists(ARTICLE_MAP_PATH):
        return {}

    with open(ARTICLE_MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


ARTICLE_MAP = load_article_map()


def volume_from_source(stem):
    match = re.fullmatch(r"me(\d{2})([ab]?)", stem)
    if not match:
        return None

    volume = int(match.group(1))
    suffix = {"a": "(上)", "b": "(下)"}.get(match.group(2), "")
    return f"第{volume}卷{suffix}"


def normalize_book_parts(metadata):
    book = clean_text(metadata.get("book"), "未知书名")
    stem = source_stem(metadata)

    if stem == "capital":
        return "马克思", "资本论", "第1卷", "2004年"

    match_mea = re.fullmatch(r"mea(\d{2})", stem)
    if match_mea:
        return "", "马克思恩格斯文集", f"第{int(match_mea.group(1))}卷", "2009年"

    match_mes = re.fullmatch(r"mes(\d{2})", stem)
    if match_mes:
        return "", "马克思恩格斯选集", f"第{int(match_mes.group(1))}卷", "出版年不详"

    volume = volume_from_source(stem)
    if volume:
        return "", "马克思恩格斯全集", volume, VOLUME_PUBLICATION_YEARS.get(stem, "出版年不详")

    match = re.search(r"(第\d+卷[AB]?)", book)
    if "马克思恩格斯文集" in book:
        volume = match.group(1).replace("A", "(上)").replace("B", "(下)") if match else ""
        return "", "马克思恩格斯文集", volume, "2009年"

    if "马克思恩格斯选集" in book:
        volume = match.group(1).replace("A", "(上)").replace("B", "(下)") if match else ""
        return "", "马克思恩格斯选集", volume, "出版年不详"

    if "马克思恩格斯全集" in book:
        volume = match.group(1).replace("A", "(上)").replace("B", "(下)") if match else ""
        return "", "马克思恩格斯全集", volume, "出版年不详"

    return "", book, "", "出版年不详"


def series_from_metadata(metadata, normalized_title):
    """Infer the normalized collection name without removing legacy metadata fields."""
    explicit_series = clean_text(metadata.get("series"), "")
    if explicit_series:
        return explicit_series

    stem = source_stem(metadata)
    book = clean_text(metadata.get("book"), "")

    if stem == "capital" or normalized_title == "资本论":
        return "资本论"
    if stem.startswith("mea"):
        return "马克思恩格斯文集"
    if stem.startswith("mes"):
        return "马克思恩格斯选集"
    if "马克思恩格斯文集" in book or normalized_title == "马克思恩格斯文集":
        return "马克思恩格斯文集"
    if "马克思恩格斯选集" in book or normalized_title == "马克思恩格斯选集":
        return "马克思恩格斯选集"
    if stem.startswith("me") or "马克思恩格斯全集" in book or normalized_title == "马克思恩格斯全集":
        return "马克思恩格斯全集"

    return normalized_title


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clean_article_title(title):
    title = clean_text(title, "")
    title = re.split(r"[.\u2026•·]{3,}", title, maxsplit=1)[0]
    title = re.sub(r"^[*•·.\s]+", "", title)
    title = re.sub(r"[.·•\]\)）\s]+$", "", title)
    title = title.strip("“”\"'《》[]【】()（）")
    return title


def is_noisy_article_title(title):
    title = clean_article_title(title)
    normalized = normalize_for_match(title)

    if not normalized:
        return True

    if title.count("\u300b") > title.count("\u300a"):
        return True

    if normalized.isdigit():
        return True

    if len(normalized) <= 4 and re.fullmatch(r"[0-9A-Za-z]+", normalized):
        return True

    ascii_chars = sum(1 for char in title if char.isascii() and char.isalnum())
    chinese_chars = sum(1 for char in title if "\u4e00" <= char <= "\u9fff")
    if ascii_chars >= 4 and chinese_chars == 0:
        return True

    if "专题资料" in title or "全集补卷" in title:
        return True

    if re.search(r"写于[0-9０-９一二三四五六七八九十１８９年月]", title):
        return True

    if title.startswith(("几乎", "这一点", "因此，", "但是，", "可是，", "如果", "因为", "所以")):
        return True

    punctuation_count = sum(1 for char in title if char in ".。!！?？,，;；:：…·•-—_[]()（）'\"")
    if title and punctuation_count / max(len(title), 1) > 0.25:
        return True

    title_prefixes = (
        "《",
        "卡·",
        "弗·",
        "马克思",
        "恩格斯",
        "第",
        "关于",
        "论",
        "致",
        "家庭",
        "社会主义",
        "共产党",
        "资本论",
        "哥达",
        "反杜林",
        "路易",
        "法兰西",
        "雇佣",
        "工资",
        "异化",
        "私有",
        "德意志",
        "黑格尔",
    )
    if len(normalized) > 28 and not title.startswith(title_prefixes):
        return True

    if len(normalized) > 45 and not any(
        marker in title for marker in ["《", "论", "批判", "宣言", "提纲", "手稿", "资本", "国家", "家庭", "劳动"]
    ):
        return True

    return False


def article_from_article_map(metadata):
    source = metadata.get("source")
    # article_map ranges are printed-page ranges. Never match them with pdf_page,
    # otherwise front matter offsets can create a precise-looking but wrong article.
    if printed_page_source_is_untrusted(metadata):
        return None

    page = as_int(metadata.get("printed_page"))
    if not source or page is None:
        return None

    source_map = ARTICLE_MAP.get(source)
    if not source_map:
        return None

    hits = []
    for entry in source_map.get("entries", []):
        start = as_int(entry.get("start_printed_page"))
        end = as_int(entry.get("end_printed_page"))
        title = clean_article_title(entry.get("title"))
        if start is None or end is None or not title:
            continue
        if start <= page <= end:
            hits.append((end - start, entry.get("level", 99), title))

    if not hits:
        return None

    hits.sort(key=lambda item: (item[0], item[1], -len(item[2])))
    return hits[0][2]


def should_fill_article_from_map(metadata):
    article = clean_article_title(metadata.get("article"))
    book = clean_text(metadata.get("book"), "")
    section = clean_article_title(metadata.get("section"))

    if not article:
        return True
    if is_noisy_article_title(article):
        return True
    if article == book:
        return True
    if section == book:
        return True

    return False


def normalize_metadata(metadata):
    """Return a copy of metadata with stable fields used by retrieval and citation.

    The old vectorstore fields are preserved for compatibility. New code should read
    the normalized keys here when it needs series/volume/publisher/page semantics.
    """
    normalized = dict(metadata or {})
    _, title, volume, year = normalize_book_parts(normalized)
    source = normalized.get("source")

    normalized.setdefault("series", series_from_metadata(normalized, title))
    normalized.setdefault("volume", volume)
    normalized.setdefault("publisher", DEFAULT_PUBLISHER)
    normalized.setdefault("publication_year", year)
    normalized.setdefault("source_file", source)

    if printed_page_source_is_untrusted(normalized) and normalized.get("printed_page") is not None:
        normalized.setdefault("printed_page_trust", "low")
        normalized.setdefault(
            "page_warning",
            "printed_page may be an OCR manuscript/note page marker; citation uses pdf_page",
        )
        if normalized.get("pdf_page") is not None:
            normalized["citation_page"] = normalized.get("pdf_page")
            normalized["citation_page_type"] = "pdf_page"

    mapped_article = article_from_article_map(normalized)
    if mapped_article and should_fill_article_from_map(normalized):
        normalized["article"] = mapped_article
        if not normalized.get("section") or normalized.get("section") == normalized.get("book"):
            normalized["section"] = mapped_article

    for key in ["article", "section"]:
        if key not in normalized:
            continue

        cleaned_title = clean_article_title(normalized.get(key))
        if is_noisy_article_title(cleaned_title):
            normalized[f"raw_{key}"] = normalized.get(key)
            normalized[key] = None
        elif cleaned_title:
            normalized[key] = cleaned_title

    if normalized.get("citation_page") is None:
        if normalized.get("printed_page") is not None:
            normalized["citation_page"] = normalized.get("printed_page")
            normalized.setdefault("citation_page_type", "printed_page")
        elif normalized.get("pdf_page") is not None:
            normalized["citation_page"] = normalized.get("pdf_page")
            normalized.setdefault("citation_page_type", "pdf_page")

    return normalized


def format_citation(metadata, include_article=False):
    metadata = normalize_metadata(metadata)
    author, title, volume, year = normalize_book_parts(metadata)
    article = clean_text(metadata.get("section") or metadata.get("article"), "")
    printed_page = metadata.get("printed_page")
    citation_page = metadata.get("citation_page")
    citation_page_type = metadata.get("citation_page_type")
    page = citation_page if citation_page is not None else (printed_page if printed_page is not None else metadata.get("pdf_page"))
    page = clean_text(page, "未知页码")
    pdf_page = clean_text(metadata.get("pdf_page"), page)

    author_text = f"{author}：" if author else ""
    volume_text = volume if volume else ""
    article_text = f"，{article}" if include_article and article else ""
    year_text = f"，{year}" if year else ""
    page_text = f"\u7b2c{page}\u9875" if citation_page_type == "printed_page" or (citation_page_type is None and printed_page is not None) else f"PDF\u7b2c{pdf_page}\u9875"

    return f"{author_text}《{title}》{volume_text}{article_text}，北京：人民出版社{year_text}，{page_text}。"


def extract_quoted_title(query):
    query = clean_text(query, "")
    match = re.search(r"《([^》]+)》", query)
    if match:
        return match.group(1).strip()

    return None


def extract_unquoted_title(query):
    query = clean_text(query, "")
    keywords = [
        "在哪一卷",
        "在哪卷",
        "哪一卷",
        "第几卷",
        "收录在哪里",
        "收在哪里",
        "收录在哪",
        "收在哪",
        "在哪里",
        "出自哪卷",
        "属于哪卷",
        "在哪本",
        "从哪页",
        "从哪一页",
        "从第几页",
        "哪一页开始",
        "从哪一页开始",
        "第几页",
        "起始页",
        "开始页",
        "收录页",
    ]
    positions = [query.find(keyword) for keyword in keywords if keyword in query]
    if not positions:
        return None

    title = query[:min(positions)]
    title = re.sub(r"[，。；：、\s\"'“”《》（）()]+$", "", title).strip()

    return title or None


def extract_bibliographic_title(query):
    return extract_quoted_title(query) or extract_unquoted_title(query)


def normalize_for_match(text):
    text = clean_text(text, "")
    text = re.sub(r"[《》“”\"'（）()，。；：、\s·\-.—–]", "", text)
    return text.lower()


def is_bibliographic_query(query):
    query = clean_text(query, "")
    keywords = [
        "在哪一卷",
        "在哪卷",
        "哪一卷",
        "第几卷",
        "收录在哪里",
        "收在哪里",
        "收录在哪",
        "收在哪",
        "在哪里",
        "出自哪卷",
        "属于哪卷",
        "在哪本",
        "从哪页",
        "从哪一页",
        "从第几页",
        "哪一页开始",
        "从哪一页开始",
        "起始页",
        "开始页",
        "收录页",
    ]

    return any(keyword in query for keyword in keywords)


def is_quote_lookup_query(query):
    query = clean_text(query, "")
    if extract_bibliographic_title(query):
        return False

    quote_keywords = ["引文", "出处", "出自", "哪一页", "哪页", "页码", "原文", "这句话", "这段话"]
    if any(keyword in query for keyword in quote_keywords):
        return True

    return len(query) >= 12 and not re.search(r"[？?]", query)


def is_concept_query(query):
    query = clean_text(query, "")
    return any(
        keyword in query
        for keyword in [
            "什么是",
            "何为",
            "概念",
            "定义",
            "解释一下",
            "是什么",
            "是什么意思",
            "如何理解",
            "这个概念",
        ]
    )


def is_analysis_query(query):
    query = clean_text(query, "")
    return any(
        keyword in query
        for keyword in [
            "分析",
            "怎么看",
            "怎么看待",
            "如何理解",
            "为什么",
            "现实",
            "结合现实",
            "现实表现",
            "意义",
            "当代意义",
            "关系",
            "评价",
        ]
    )


def classify_query(query):
    """Classify a user query so retrieval and prompting can stay task-specific.

    bibliographic_lookup: locate a work in the local table of contents.
    quote_lookup: confirm the source/page for an exact sentence or paragraph.
    concept_explain: explain a Marxist concept with retrieved primary text.
    theory_analysis: analyze a question through Marxist theoretical categories.
    rag_answer: answer ordinary retrieval questions without special routing.
    """
    if is_bibliographic_query(query) and extract_bibliographic_title(query):
        return "bibliographic_lookup"

    if is_quote_lookup_query(query):
        return "quote_lookup"

    if is_concept_query(query):
        return "concept_explain"

    if is_analysis_query(query):
        return "theory_analysis"

    return "rag_answer"


def cache_files_for_toc_scan():
    if not os.path.isdir(OCR_CACHE_DIR):
        return []

    paths = []

    for source_stem in os.listdir(OCR_CACHE_DIR):
        root = os.path.join(OCR_CACHE_DIR, source_stem)
        if not os.path.isdir(root):
            continue

        for page_num in range(1, 31):
            path = os.path.join(root, f"page_{page_num}.txt")
            if os.path.exists(path):
                paths.append((source_stem, path))

    return paths


def best_toc_entries(entries):
    unique_entries = {}

    for entry in entries:
        key = (entry["source"], entry["article"], entry["start_page"], entry["end_page"])
        unique_entries[key] = entry

    filtered_entries = list(unique_entries.values())
    best_by_source = {}

    for entry in filtered_entries:
        width = entry["end_page"] - entry["start_page"]
        source = entry["source"]
        previous = best_by_source.get(source)
        if previous is None or width > previous["end_page"] - previous["start_page"]:
            best_by_source[source] = entry

    return sorted(
        best_by_source.values(),
        key=lambda item: (item["source"], item["start_page"], item["end_page"]),
    )


def enrich_core_classic_entries(entries):
    enriched = []

    for entry in entries:
        source = entry["source"]
        metadata = {
            "source": source,
            "book": ARTICLE_MAP.get(source, {}).get("book", ""),
            "article": entry.get("article", entry.get("classic_title", "")),
        }
        _, book_title, volume, year = normalize_book_parts(metadata)
        enriched.append(
            {
                "source": source,
                "book_title": book_title,
                "volume": volume,
                "year": year,
                "article": entry.get("article") or entry.get("classic_title"),
                "start_page": entry["start_page"],
                "end_page": entry["end_page"],
                "classic_id": entry.get("classic_id"),
                "classic_title": entry.get("classic_title"),
                "classic_author": entry.get("classic_author"),
                "classic_work_year": entry.get("classic_work_year"),
                "classic_work_type": entry.get("classic_work_type"),
                "entry_type": entry.get("entry_type"),
                "priority": entry.get("priority", 99),
            }
        )

    return sorted(enriched, key=lambda item: (item.get("priority", 99), item["source"]))


def find_toc_entries_from_map(title):
    core_entries = classic_entries_for_query(title)
    if core_entries:
        return enrich_core_classic_entries(core_entries)

    entries = []
    normalized_title = normalize_for_match(title)

    if not normalized_title:
        return []

    if normalized_title == normalize_for_match("反杜林论"):
        metadata = {
            "source": "me20.pdf",
            "book": ARTICLE_MAP.get("me20.pdf", {}).get("book", ""),
            "article": "反杜林论",
        }
        _, book_title, volume, year = normalize_book_parts(metadata)
        return [
            {
                "source": "me20.pdf",
                "book_title": book_title,
                "volume": volume,
                "year": year,
                "article": "反杜林论",
                "start_page": 1,
                "end_page": 354,
            }
        ]

    for source, source_map in ARTICLE_MAP.items():
        metadata = {
            "source": source,
            "book": source_map.get("book", ""),
            "article": title,
        }
        _, book_title, volume, year = normalize_book_parts(metadata)

        for item in source_map.get("entries", []):
            entry_title = clean_text(item.get("title"), "")
            normalized_entry_title = normalize_for_match(entry_title)
            start_page = item.get("start_printed_page")
            end_page = item.get("end_printed_page")

            if not normalized_entry_title or start_page is None or end_page is None:
                continue

            is_exact_match = normalized_entry_title == normalized_title
            is_safe_partial_match = (
                len(normalized_title) >= 4
                and len(normalized_entry_title) >= 4
                and (
                    normalized_title in normalized_entry_title
                    or normalized_entry_title in normalized_title
                )
            )
            if not is_exact_match and not is_safe_partial_match:
                continue

            entries.append(
                {
                    "source": source,
                    "book_title": book_title,
                    "volume": volume,
                    "year": year,
                    "article": entry_title,
                    "start_page": start_page,
                    "end_page": end_page,
                }
            )

    exact_entries = [
        entry for entry in entries
        if normalize_for_match(entry["article"]) == normalized_title
    ]
    if exact_entries:
        entries = exact_entries

    suffix_entries = [
        entry for entry in entries
        if normalize_for_match(entry["article"]).endswith(normalized_title)
    ]
    if suffix_entries:
        entries = suffix_entries

    derivative_terms = ["草稿", "初稿", "遗稿", "导言", "序言", "扉页", "封面", "一书导言", "第一页", "材料"]
    if not any(term in title for term in derivative_terms):
        primary_entries = [
            entry for entry in entries
            if not any(term in entry["article"] for term in derivative_terms)
        ]
        if primary_entries:
            entries = primary_entries

    return best_toc_entries(entries)


def find_toc_entries(title):
    entries = find_toc_entries_from_map(title)
    if entries:
        return entries

    entries = []
    title_pattern = re.escape(title)
    range_pattern = re.compile(
        rf"{title_pattern}(?![\u4e00-\u9fff]).{{0,50}}?(\d{{1,4}})\s*[—\-–一]\s*(\d{{1,4}})"
    )

    for source_stem, path in cache_files_for_toc_scan():
        with open(path, "r", encoding="utf-8") as f:
            text = clean_text(f.read(), "")

        for match in range_pattern.finditer(text):
            start_page = int(match.group(1))
            end_page = int(match.group(2))

            if start_page > end_page:
                continue

            if start_page > 1200 or end_page > 1200:
                continue

            matched_text = match.group(0)
            matched_tail = matched_text[matched_text.find(title) + len(title):]
            title_tail = re.sub(r"[\s《》“”\"'（）()，。；：、·\-.—–0-9０-９]", "", matched_tail)
            if title_tail:
                continue

            metadata = {
                "source": f"{source_stem}.pdf",
                "book": f"马克思恩格斯全集 {volume_from_source(source_stem) or ''}".strip(),
                "article": title,
                "page": f"{start_page}-{end_page}",
            }
            _, book_title, volume, year = normalize_book_parts(metadata)

            entries.append(
                {
                    "source": f"{source_stem}.pdf",
                    "book_title": book_title,
                    "volume": volume,
                    "year": year,
                    "article": title,
                    "start_page": start_page,
                    "end_page": end_page,
                }
            )

    unique_entries = {}

    for entry in entries:
        key = (entry["source"], entry["article"], entry["start_page"], entry["end_page"])
        unique_entries[key] = entry

    filtered_entries = list(unique_entries.values())
    best_by_source = {}

    for entry in filtered_entries:
        width = entry["end_page"] - entry["start_page"]
        source = entry["source"]
        previous = best_by_source.get(source)
        if previous is None or width > previous["end_page"] - previous["start_page"]:
            best_by_source[source] = entry

    return sorted(
        best_by_source.values(),
        key=lambda item: (item["source"], item["start_page"], item["end_page"]),
    )


def answer_bibliographic_query(query):
    title = extract_bibliographic_title(query)
    if not title:
        return None

    entries = find_toc_entries(title)
    if not entries:
        return None

    lines = []

    for index, entry in enumerate(entries, start=1):
        work_meta = "，".join(
            item
            for item in [
                entry.get("classic_author"),
                entry.get("classic_work_year"),
                entry.get("classic_work_type"),
            ]
            if item
        )
        work_meta_text = f"（{work_meta}）" if work_meta else ""
        lines.append(
            f"({index})《{entry['book_title']}》{entry['volume']}，"
            f"{entry['article']}{work_meta_text}，第{entry['start_page']}-{entry['end_page']}页。"
        )

    return "\n".join(lines)


def answer_quote_query(query, limit=5, trace=False):
    docs = exact_quote_lookup(query, OCR_CACHE_DIR, limit=limit)

    exact_docs = [
        doc for doc in docs
        if doc.metadata.get("match_type") == "exact_quote"
    ]
    if trace:
        print_docs_trace(exact_docs, label="exact_quote_docs")

    if not exact_docs:
        return "未能在当前 OCR 缓存中确认该引文的精确出处。"

    lines = []
    for index, doc in enumerate(exact_docs, start=1):
        lines.append(f"({index}){format_citation(doc.metadata, include_article=True)}")

    return "\n".join(lines)


def constraints_from_query(query):
    title = extract_bibliographic_title(query)
    core_entries = classic_entries_for_query(title or query)
    if core_entries:
        entries = enrich_core_classic_entries(core_entries)
        title = title or entries[0].get("classic_title")
        return {
            "title": title,
            "entries": entries,
            "sources": {entry["source"] for entry in entries},
            "page_ranges": {
                entry["source"]: (entry["start_page"], entry["end_page"])
                for entry in entries
            },
        }

    if not title:
        return {}

    entries = find_toc_entries(title)
    if not entries:
        return {"title": title}

    return {
        "title": title,
        "entries": entries,
        "sources": {entry["source"] for entry in entries},
        "page_ranges": {
            entry["source"]: (entry["start_page"], entry["end_page"])
            for entry in entries
        },
    }


def metadata_matches_constraints(metadata, constraints):
    sources = constraints.get("sources")
    if not sources:
        return True

    return metadata.get("source") in sources


def page_in_expected_range(metadata, constraints):
    ranges = constraints.get("page_ranges")
    if not ranges:
        return False

    source = metadata.get("source")
    if source not in ranges:
        return False

    try:
        page = int(metadata.get("page"))
    except (TypeError, ValueError):
        return False

    start_page, end_page = ranges[source]
    return start_page <= page <= end_page


def score_source_match(metadata, constraints):
    return 100 if metadata_matches_constraints(metadata, constraints) else 0


def score_page_range(metadata, constraints):
    return 40 if page_in_expected_range(metadata, constraints) else 0


def score_article_match(metadata, normalized_title, haystack):
    if not normalized_title:
        return 0

    article = clean_text(metadata.get("section") or metadata.get("article"), "")
    score = 0
    if normalized_title in normalize_for_match(article):
        score += 35
    if normalized_title in haystack:
        score += 25
    return score


def score_query_match(normalized_query, haystack):
    return 10 if normalized_query and normalized_query in haystack else 0


CONCEPT_FOCUS_TERMS = [
    "人的本质",
    "异化劳动",
    "外化劳动",
    "剩余价值",
    "剩余价值率",
    "劳动过程",
    "价值增殖过程",
    "资本",
    "阶级斗争",
    "国家",
    "国家的产生",
    "历史唯物主义",
    "唯物主义历史观",
    "唯物辩证法",
    "自然辩证法",
    "商品拜物教",
    "拜物教",
    "工资",
    "利润",
    "私有制",
    "家庭",
    "家庭私有制和国家的起源",
]

CONCEPT_PREFERRED_MARKERS = {
    "异化劳动": ["异化劳动和私有财产", "外化劳动"],
    "外化劳动": ["异化劳动和私有财产", "外化劳动"],
    "剩余价值": ["剩余价值率", "价值增殖过程", "资本论"],
    "剩余价值率": ["剩余价值率", "价值增殖过程", "资本论"],
    "劳动过程": ["劳动过程和价值增殖过程", "劳动过程"],
    "资本": ["资本论", "资本的生产过程", "货币转化为资本"],
    "阶级斗争": ["共产党宣言", "阶级斗争"],
    "国家": ["家庭、私有制和国家的起源", "国家的产生", "社会主义"],
    "国家的产生": ["家庭、私有制和国家的起源", "国家的产生"],
    "历史唯物主义": ["唯物主义历史观", "共产党宣言", "费尔巴哈", "路德维希·费尔巴哈"],
    "唯物主义历史观": ["唯物主义历史观", "共产党宣言", "费尔巴哈"],
    "唯物辩证法": ["反杜林论", "自然辩证法", "路德维希·费尔巴哈"],
    "自然辩证法": ["自然辩证法", "反杜林论"],
    "商品拜物教": ["商品", "拜物教", "资本论"],
    "拜物教": ["商品", "拜物教", "资本论"],
    "工资": ["雇佣劳动与资本", "工资、价格和利润", "资本论"],
    "利润": ["工资、价格和利润", "资本论", "三位一体的公式"],
    "私有制": ["家庭、私有制和国家的起源", "私有财产"],
    "家庭": ["家庭、私有制和国家的起源"],
}

EXTRA_CONCEPT_FOCUS_TERMS = [
    "\u5546\u54c1\u4ef7\u503c",
    "\u4ef7\u503c",
    "\u56fd\u5bb6\u7684\u8d77\u6e90",
    "\u8d39\u5c14\u5df4\u54c8\u63d0\u7eb2",
    "\u5b9e\u8df5",
]

CONCEPT_PREFERRED_SOURCES = {
    "\u552f\u7269\u8fa9\u8bc1\u6cd5": {
        "sources": {"mes03.pdf", "mea09.pdf"},
        "markers": ["\u53cd\u675c\u6797\u8bba", "\u81ea\u7136\u8fa9\u8bc1\u6cd5", "\u8def\u5fb7\u7ef4\u5e0c\u00b7\u8d39\u5c14\u5df4\u54c8"],
    },
    "\u9636\u7ea7\u6597\u4e89": {
        "sources": {"mes01.pdf", "mea02.pdf"},
        "markers": ["\u5171\u4ea7\u515a\u5ba3\u8a00"],
    },
    "\u56fd\u5bb6": {
        "sources": {"mea04.pdf", "mes04.pdf"},
        "markers": ["\u5bb6\u5ead", "\u79c1\u6709\u5236", "\u56fd\u5bb6", "\u8d77\u6e90"],
    },
    "\u56fd\u5bb6\u7684\u8d77\u6e90": {
        "sources": {"mea04.pdf", "mes04.pdf"},
        "markers": ["\u5bb6\u5ead", "\u79c1\u6709\u5236", "\u56fd\u5bb6", "\u8d77\u6e90"],
    },
    "\u5546\u54c1\u4ef7\u503c": {
        "sources": {"mea05.pdf", "mes02.pdf"},
        "markers": ["\u8d44\u672c\u8bba", "\u5546\u54c1", "\u4ef7\u503c"],
    },
    "\u5269\u4f59\u4ef7\u503c": {
        "sources": {"mea05.pdf", "mes02.pdf"},
        "markers": ["\u8d44\u672c\u8bba", "\u5269\u4f59\u4ef7\u503c", "\u4ef7\u503c\u589e\u6b96\u8fc7\u7a0b"],
    },
    "\u5f02\u5316\u52b3\u52a8": {
        "sources": {"mea01.pdf", "mes01.pdf"},
        "markers": ["1844\u5e74\u7ecf\u6d4e\u5b66\u54f2\u5b66\u624b\u7a3f", "\u5f02\u5316\u52b3\u52a8", "\u5916\u5316\u52b3\u52a8"],
    },
    "\u8d39\u5c14\u5df4\u54c8\u63d0\u7eb2": {
        "sources": {"mes01.pdf", "mea01.pdf"},
        "markers": ["\u5173\u4e8e\u8d39\u5c14\u5df4\u54c8\u7684\u63d0\u7eb2", "\u8d39\u5c14\u5df4\u54c8", "\u5b9e\u8df5"],
    },
    "\u5b9e\u8df5": {
        "sources": {"mes01.pdf", "mea01.pdf"},
        "markers": ["\u5173\u4e8e\u8d39\u5c14\u5df4\u54c8\u7684\u63d0\u7eb2", "\u5b9e\u8df5"],
    },
}

CONCEPT_DEMOTED_ARTICLE_MARKERS = [
    "\u4e66\u4fe1",
    "\u7d22\u5f15",
    "\u76ee\u5f55",
    "\u76ee\u6b21",
    "\u51c6\u5907\u6750\u6599",
    "\u8865\u5145\u548c\u4fee\u6539",
    "\u8865\u5145",
    "\u4e66\u7b80",
    "\u4e66\u4fe1\u9009\u7f16",
]

CONCEPT_PREFERRED_PAGE_RANGES = {
    "\u9636\u7ea7\u6597\u4e89": {
        "mes01.pdf": (376, 435),
        "mea02.pdf": (3, 67),
    },
    "\u56fd\u5bb6": {
        "mea04.pdf": (13, 198),
        "mes04.pdf": (669, 709),
    },
    "\u56fd\u5bb6\u7684\u8d77\u6e90": {
        "mea04.pdf": (13, 198),
        "mes04.pdf": (669, 709),
    },
    "\u5546\u54c1\u4ef7\u503c": {
        "mea05.pdf": (7, 887),
        "mes02.pdf": (185, 370),
    },
    "\u5269\u4f59\u4ef7\u503c": {
        "mea05.pdf": (7, 887),
        "mes02.pdf": (185, 370),
    },
    "\u5f02\u5316\u52b3\u52a8": {
        "mea01.pdf": (109, 248),
        "mes01.pdf": (49, 63),
    },
    "\u8d39\u5c14\u5df4\u54c8\u63d0\u7eb2": {
        "mes01.pdf": (133, 140),
        "mea01.pdf": (499, 506),
    },
    "\u5b9e\u8df5": {
        "mes01.pdf": (133, 140),
        "mea01.pdf": (499, 506),
    },
}


def metadata_citation_page(metadata):
    for key in ("printed_page", "citation_page", "page"):
        try:
            return int(metadata.get(key))
        except (TypeError, ValueError):
            continue
    return None


def active_concept_terms(query):
    normalized_query = normalize_for_match(query)
    terms = []
    for term in CONCEPT_FOCUS_TERMS + EXTRA_CONCEPT_FOCUS_TERMS:
        normalized_term = normalize_for_match(term)
        if normalized_term and normalized_term in normalized_query:
            terms.append(term)

    return terms


def score_concept_focus(query, metadata, content):
    terms = active_concept_terms(query)
    if not terms:
        return 0

    article = clean_text(metadata.get("section") or metadata.get("article"), "")
    article_norm = normalize_for_match(article)
    content_norm = normalize_for_match(content)
    lead_norm = normalize_for_match(content[:300])
    score = 0

    for term in terms:
        term_norm = normalize_for_match(term)
        if not term_norm:
            continue
        citation_page = metadata_citation_page(metadata)
        preferred_ranges = CONCEPT_PREFERRED_PAGE_RANGES.get(term, {})
        preferred_range = preferred_ranges.get(metadata.get("source"))
        if preferred_range and citation_page is not None:
            start_page, end_page = preferred_range
            if start_page <= citation_page <= end_page:
                score += 180
            else:
                score -= 35

        if term_norm in article_norm:
            score += 35
        if term_norm in lead_norm:
            score += 30
        elif term_norm in content_norm:
            score += 12

        direct_definition_patterns = [
            f"什么是{term_norm}",
            f"{term_norm}是什么",
        ]
        loose_definition_patterns = [
            f"{term_norm}是",
            f"所谓{term_norm}",
        ]
        if any(pattern in lead_norm for pattern in direct_definition_patterns):
            score += 100
        elif any(pattern in lead_norm for pattern in loose_definition_patterns):
            score += 50

        for marker in CONCEPT_PREFERRED_MARKERS.get(term, []):
            marker_norm = normalize_for_match(marker)
            if marker_norm and marker_norm in article_norm:
                score += 22
            elif marker_norm and marker_norm in lead_norm:
                score += 14

        preferred = CONCEPT_PREFERRED_SOURCES.get(term) or {}
        if metadata.get("source") in preferred.get("sources", set()):
            score += 25

        for marker in preferred.get("markers", []):
            marker_norm = normalize_for_match(marker)
            if marker_norm and marker_norm in article_norm:
                score += 140
            elif marker_norm and marker_norm in lead_norm:
                score += 45
            elif marker_norm and marker_norm in content_norm:
                score += 25

    if terms and any(normalize_for_match(marker) in article_norm for marker in CONCEPT_DEMOTED_ARTICLE_MARKERS):
        score -= 120

    return min(score, 320)


def score_document_quality(metadata, content):
    article = clean_text(metadata.get("section") or metadata.get("article"), "")
    article_norm = normalize_for_match(article)
    content_norm = normalize_for_match(content)
    score = 0

    if metadata.get("page_type") in {"toc", "title_page"}:
        score -= 80

    if is_noisy_article_title(article):
        score -= 60

    if any(marker in article_norm for marker in ["名目索引", "人名索引"]):
        score -= 70

    if any(marker in article_norm for marker in ["目录", "目次", "索引", "注释", "编者注"]):
        score -= 45

    if "索引" in content_norm:
        score -= 35

    if content.count("———") >= 3 or content.count("---") >= 3:
        score -= 35

    if len(content_norm) < 80:
        score -= 30

    punctuation_count = sum(1 for char in content if char in ".。!！?？,，;；:：…·•-—_[]()（）")
    if content and punctuation_count / max(len(content), 1) > 0.35:
        score -= 25

    if article and len(normalize_for_match(article)) > 35 and not any(
        marker in article for marker in ["《", "论", "批判", "宣言", "提纲", "手稿", "资本", "国家", "家庭", "劳动"]
    ):
        score -= 15

    return score


def debug_rerank_score(index, doc, score_parts):
    if os.getenv(RERANK_DEBUG_ENV) != "1":
        return

    metadata = doc.metadata
    total = sum(score_parts.values())
    detail = ", ".join(f"{name}={score}" for name, score in score_parts.items())
    print(
        f"[rerank] candidate={index} total={total} {detail} "
        f"source={metadata.get('source')} page={metadata.get('page')} "
        f"article={metadata.get('article') or metadata.get('section')}",
        file=sys.stderr,
    )


def rerank_documents(query, docs, constraints):
    title = constraints.get("title")
    normalized_title = normalize_for_match(title) if title else ""
    normalized_query = normalize_for_match(query)
    ranked = []

    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        article = clean_text(metadata.get("section") or metadata.get("article"), "")
        book = clean_text(metadata.get("book"), "")
        content = clean_text(doc.page_content, "")
        haystack = normalize_for_match(f"{book} {article} {content[:600]}")
        score_parts = {
            "source_match": score_source_match(metadata, constraints),
            "page_range": score_page_range(metadata, constraints),
            "article_match": score_article_match(metadata, normalized_title, haystack),
            "query_match": score_query_match(normalized_query, haystack),
            "concept_focus": score_concept_focus(query, metadata, content),
            "document_quality": score_document_quality(metadata, content),
        }
        score = sum(score_parts.values())
        debug_rerank_score(index, doc, score_parts)

        ranked.append((score, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for score, doc in ranked]


def retrieve_documents(query, db, k=5):
    if is_quote_lookup_query(query):
        exact_docs = exact_quote_lookup(query, OCR_CACHE_DIR, limit=k)
        if exact_docs:
            return exact_docs

    constraints = constraints_from_query(query)
    fetch_k = 120 if constraints or active_concept_terms(query) else 30

    if constraints.get("sources"):
        candidates = db.similarity_search(query, k=fetch_k)
        candidates = [
            doc for doc in candidates
            if metadata_matches_constraints(doc.metadata, constraints)
        ]

        if not candidates:
            candidates = db.similarity_search(query, k=fetch_k)
    else:
        candidates = db.similarity_search(query, k=fetch_k)

    docs = rerank_documents(query, candidates, constraints)[:k]

    if is_quote_lookup_query(query):
        for doc in docs:
            doc.metadata["match_type"] = "vector_candidate"
            doc.metadata["confidence"] = 0.0

    return docs


def final_answer_style_rules():
    return (
        "\n\u6700\u7ec8\u56de\u7b54\u98ce\u683c\uff1a\n"
        "1. \u76f4\u63a5\u56de\u7b54\u95ee\u9898\uff0c\u4e0d\u8981\u95ee\u5019\uff0c\u4e0d\u8981\u81ea\u6211\u4ecb\u7ecd\uff0c\u4e0d\u8981\u8bf4\u201c\u4f60\u597d\u201d\u6216\u201c\u6211\u662f MarxOS\u201d\u3002\n"
        "2. \u7ed3\u5c3e\u4e0d\u8981\u8ffd\u52a0\u201c\u5982\u679c\u9700\u8981\u201d\u201c\u6211\u53ef\u4ee5\u7ee7\u7eed\u201d\u7b49\u9080\u8bf7\u5f0f\u8bdd\u8bed\u3002\n"
        "3. \u5f15\u7528\u539f\u8457\u65f6\uff0c\u53ea\u4f7f\u7528\u4e0b\u65b9\u63d0\u4f9b\u7684\u51fa\u5904\u683c\u5f0f\uff0c\u4e0d\u8981\u81ea\u884c\u7f16\u9020\u7bc7\u540d\u6216\u9875\u7801\u3002\n"
        "4. \u4e0d\u8981\u8f93\u51fa\u201c\u3010\u539f\u8457\u5185\u5bb9\u3011\u201d\u201c\u3010\u68c0\u7d22\u6750\u6599\u3011\u201d\u6216\u201cCTX-1\u201d\u7b49\u5185\u90e8\u680f\u76ee\u540d\u548c\u5185\u90e8\u7f16\u53f7\u3002\n"
    )


def build_quote_prompt(query, context):
    return (
        f"\n\u4f60\u662f MarxOS \u7684\u51fa\u5904\u6838\u5bf9\u5668\u3002\n\n"
        f"\u4efb\u52a1\uff1a\u7528\u6237\u7ed9\u51fa\u4e00\u53e5\u6216\u4e00\u6bb5\u539f\u6587\uff0c"
        f"\u8bf7\u53ea\u6839\u636e\u3010\u68c0\u7d22\u6750\u6599\u3011\u5224\u65ad\u6700\u53ef\u80fd\u51fa\u5904\u3002\n\n"
        f"{final_answer_style_rules()}\n"
        f"\u56de\u7b54\u8981\u6c42\uff1a\n"
        f"1. \u53ea\u8f93\u51fa\u51fa\u5904\uff0c\u4e0d\u505a\u7406\u8bba\u5206\u6790\u3002\n"
        f"2. \u4f18\u5148\u4f7f\u7528\u68c0\u7d22\u6750\u6599\u4e2d\u7684\u201c\u53e5\u5b50\u5f15\u6587\u683c\u5f0f\u201d"
        f"\u6216\u201c\u6bb5\u843d\u5177\u4f53\u51fa\u5904\u683c\u5f0f\u201d\u3002\n"
        f"3. \u5982\u679c\u6750\u6599\u53ea\u6709 PDF \u9875\u800c\u6ca1\u6709\u53ef\u9760\u5370\u5237\u9875\uff0c"
        f"\u5fc5\u987b\u5199\u201cPDF\u7b2cX\u9875\u201d\uff0c\u4e0d\u8981\u5192\u5145\u201c\u7b2cX\u9875\u201d\u3002\n"
        f"4. \u5982\u679c\u6ca1\u6709\u7cbe\u786e\u5339\u914d\uff0c\u5fc5\u987b\u8bf4\u660e"
        f"\u201c\u672a\u80fd\u786e\u8ba4\u5177\u4f53\u9875\u7801\u201d\uff0c\u518d\u5217\u6700\u63a5\u8fd1\u7684\u5019\u9009\u3002\n\n"
        f"\u7981\u6b62\u8f93\u51fa\uff1a\u4e0d\u8981\u5728\u6700\u7ec8\u56de\u7b54\u4e2d\u51fa\u73b0"
        f"\u201c\u8d44\u65991\u201d\u201c\u8d44\u65992\u201d\u201c\u7247\u6bb51\u201d"
        f"\u201c\u68c0\u7d22\u6750\u6599\u201d\u7b49\u5185\u90e8\u7f16\u53f7\u6216\u5185\u90e8\u8bf4\u6cd5\u3002\n\n"
        f"# \u68c0\u7d22\u6750\u6599\n{context}\n\n# \u7528\u6237\u539f\u6587\n{query}\n"
    )


def build_concept_prompt(query, context):
    return (
        f"\n\u4f60\u662f MarxOS\uff0c\u4e00\u4e2a\u9a6c\u514b\u601d\u4e3b\u4e49\u5b66\u672f\u52a9\u624b\u3002\n\n"
        f"\u4efb\u52a1\uff1a\u89e3\u91ca\u7528\u6237\u63d0\u51fa\u7684\u6982\u5ff5\u3002"
        f"\u4f18\u5148\u4f9d\u636e\u3010\u539f\u8457\u5185\u5bb9\u3011\uff0c\u518d\u505a\u5fc5\u8981\u7684\u7406\u8bba\u6982\u62ec\u3002\n\n"
        f"{final_answer_style_rules()}\n"
        f"\u56de\u7b54\u8981\u6c42\uff1a\n"
        f"1. \u5148\u7ed9\u51fa\u7b80\u660e\u5b9a\u4e49\u3002\n"
        f"2. \u8bf4\u660e\u5b83\u5728\u9a6c\u514b\u601d\u4e3b\u4e49\u7406\u8bba\u4e2d\u7684\u4f4d\u7f6e\u3002\n"
        f"3. \u5982\u4f7f\u7528\u539f\u8457\u6750\u6599\uff0c\u9644\u7b80\u77ed\u51fa\u5904\u3002\n"
        f"4. \u4e0d\u8981\u8f93\u51fa\u201c\u68c0\u7d22\u6765\u6e90\u201d\u7b49\u5185\u90e8\u8c03\u8bd5\u4fe1\u606f\u3002\n\n"
        f"\u7981\u6b62\u8f93\u51fa\uff1a\u4e0d\u8981\u5199\u201c\u8d44\u65991\u201d\u201c\u8d44\u65992\u201d"
        f"\u201c\u7247\u6bb51\u201d\u201c\u68c0\u7d22\u6750\u6599\u201d\u7b49\u5185\u90e8\u7f16\u53f7\uff1b"
        f"\u9700\u8981\u5f15\u7528\u65f6\uff0c\u53ea\u4f7f\u7528\u51fa\u5904\u6587\u672c\u3002\n\n"
        f"# \u539f\u8457\u5185\u5bb9\n{context}\n\n# \u7528\u6237\u95ee\u9898\n{query}\n"
    )


def build_analysis_prompt(query, context):
    return (
        f"\n\u4f60\u662f MarxOS\uff0c\u4e00\u4e2a\u9a6c\u514b\u601d\u4e3b\u4e49\u5b66\u672f\u667a\u80fd\u4f53\u3002\n\n"
        f"\u4efb\u52a1\uff1a\u57fa\u4e8e\u3010\u539f\u8457\u5185\u5bb9\u3011\u548c\u9a6c\u514b\u601d\u4e3b\u4e49\u7406\u8bba\uff0c"
        f"\u5bf9\u7528\u6237\u95ee\u9898\u505a\u7ed3\u6784\u6027\u5206\u6790\u3002\n\n"
        f"{final_answer_style_rules()}\n"
        f"\u5206\u6790\u6846\u67b6\uff1a\u751f\u4ea7\u529b\u4e0e\u751f\u4ea7\u5173\u7cfb\u3001"
        f"\u7ecf\u6d4e\u57fa\u7840\u4e0e\u4e0a\u5c42\u5efa\u7b51\u3001\u9636\u7ea7\u5173\u7cfb\u3001"
        f"\u8d44\u672c\u903b\u8f91\u3001\u52b3\u52a8\u8fc7\u7a0b\u3002\n\n"
        f"\u56de\u7b54\u8981\u6c42\uff1a\n"
        f"1. \u4f18\u5148\u4f9d\u636e\u539f\u8457\u5185\u5bb9\u3002\n"
        f"2. \u56f4\u7ed5\u6982\u5ff5\u3001\u903b\u8f91\u548c\u73b0\u5b9e\u6307\u5411\u5c55\u5f00\uff0c\u4e0d\u7a7a\u558a\u53e3\u53f7\u3002\n"
        f"3. \u5982\u5f15\u7528\u539f\u8457\uff0c\u7ed9\u51fa\u7b80\u77ed\u51fa\u5904\u3002\n\n"
        f"\u7981\u6b62\u8f93\u51fa\uff1a\u4e0d\u8981\u5199\u201c\u8d44\u65991\u201d\u201c\u8d44\u65992\u201d"
        f"\u201c\u7247\u6bb51\u201d\u201c\u68c0\u7d22\u6750\u6599\u201d\u7b49\u5185\u90e8\u7f16\u53f7\uff1b"
        f"\u9700\u8981\u5f15\u7528\u65f6\uff0c\u53ea\u4f7f\u7528\u51fa\u5904\u6587\u672c\u3002\n\n"
        f"# \u539f\u8457\u5185\u5bb9\n{context}\n\n# \u7528\u6237\u95ee\u9898\n{query}\n"
    )


def build_default_prompt(query, context):
    return (
        f"\n\u4f60\u662f MarxOS\uff0c\u4e00\u4e2a\u9a6c\u514b\u601d\u4e3b\u4e49\u5b66\u672f\u52a9\u624b\u3002\n\n"
        f"\u8bf7\u6839\u636e\u3010\u539f\u8457\u5185\u5bb9\u3011\u56de\u7b54\u7528\u6237\u95ee\u9898\u3002"
        f"\u95ee\u9898\u82e5\u53ea\u9700\u8981\u77ed\u7b54\uff0c\u5c31\u77ed\u7b54\uff1b"
        f"\u53ea\u6709\u9700\u8981\u5c55\u5f00\u89e3\u91ca\u65f6\u624d\u5206\u5c42\u5206\u6790\u3002\n"
        f"{final_answer_style_rules()}\n"
        f"\u4e0d\u8981\u8f93\u51fa\u201c\u68c0\u7d22\u6765\u6e90\u201d\u7b49\u5185\u90e8\u8c03\u8bd5\u4fe1\u606f\u3002\n\n"
        f"\u7981\u6b62\u8f93\u51fa\uff1a\u4e0d\u8981\u5199\u201c\u8d44\u65991\u201d\u201c\u8d44\u65992\u201d"
        f"\u201c\u7247\u6bb51\u201d\u201c\u68c0\u7d22\u6750\u6599\u201d\u7b49\u5185\u90e8\u7f16\u53f7\uff1b"
        f"\u9700\u8981\u5f15\u7528\u65f6\uff0c\u53ea\u4f7f\u7528\u51fa\u5904\u6587\u672c\u3002\n\n"
        f"# \u539f\u8457\u5185\u5bb9\n{context}\n\n# \u7528\u6237\u95ee\u9898\n{query}\n"
    )


def build_prompt(intent, query, context):
    prompt_builders = {
        "quote_lookup": build_quote_prompt,
        "concept_explain": build_concept_prompt,
        "theory_analysis": build_analysis_prompt,
        "rag_answer": build_default_prompt,
    }
    return prompt_builders.get(intent, build_default_prompt)(query, context)

def build_context(docs, query_intent):
    # Chunk creation happens in the vectorstore build step. This function only
    # consumes chunks and keeps their metadata visible for citation and prompts.
    context_parts = []

    for i, doc in enumerate(docs, start=1):
        metadata = normalize_metadata(doc.metadata)
        book = clean_text(metadata.get("book"), "\u672a\u77e5\u4e66\u540d")
        article = clean_text(metadata.get("article"), "\u672a\u77e5\u7bc7\u76ee")
        section = clean_text(metadata.get("section"), "")
        page = clean_text(metadata.get("page"), "\u672a\u77e5\u9875\u7801")
        pdf_page = clean_text(metadata.get("pdf_page"), "\u672a\u77e5PDF\u9875")
        source = clean_text(metadata.get("source"), "\u672a\u77e5\u6765\u6e90")
        printed_page = metadata.get("printed_page")
        citation_page = metadata.get("citation_page")
        citation_page_type = clean_text(metadata.get("citation_page_type"), "")
        page_range = clean_text(metadata.get("page_range"), "")
        page_range_text = f", page_range={page_range}" if page_range else ""
        match_type = clean_text(metadata.get("match_type"), "")
        confidence = metadata.get("confidence")
        confidence_text = f"\nmatch_type={match_type}, confidence={confidence}" if match_type else ""
        classic_author = clean_text(metadata.get("classic_author"), "")
        classic_work_year = clean_text(metadata.get("classic_work_year"), "")
        classic_work_type = clean_text(metadata.get("classic_work_type"), "")
        classic_meta = ", ".join(
            item for item in [classic_author, classic_work_year, classic_work_type] if item
        )
        classic_meta_text = f"classic_metadata: {classic_meta}\n" if classic_meta else ""
        section_text = f"\uff0c{section}" if section and section != article else ""
        sentence_citation = format_citation(metadata, include_article=False)
        detailed_source = format_citation(metadata, include_article=True)

        context_parts.append(
            f"CTX-{i}\n"
            f"\u6765\u6e90\uff1a\u300a{book}\u300b{article}{section_text}\uff0c\u7b2c{page}\u9875\uff08PDF\u7b2c{pdf_page}\u9875\uff09\uff0csource={source}\n"
            f"{confidence_text}\n"
            f"{classic_meta_text}"
            f"metadata_fields: book={book}, article={article}, section={section}, page={page}, pdf_page={pdf_page}, source={source}\n"
            f"page_fields: printed_page={printed_page}, pdf_page={pdf_page}, citation_page={citation_page}, citation_page_type={citation_page_type}{page_range_text}\n"
            f"\u53e5\u5b50\u5f15\u6587\u683c\u5f0f\uff1a{sentence_citation}\n"
            f"\u6bb5\u843d\u5177\u4f53\u51fa\u5904\u683c\u5f0f\uff1a{detailed_source}\n"
            f"\u539f\u6587\uff1a{clean_text(doc.page_content)}"
        )

    context = "\n\n".join(context_parts)

    if query_intent == "quote_lookup" and docs and not any(
        doc.metadata.get("match_type") == "exact_quote" for doc in docs
    ):
        context = (
            "No exact quote match was found. The following passages are vector candidates only "
            "and must not be treated as confirmed citations.\n\n"
            + context
        )

    return context


def env_flag(name):
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def dev_mode_enabled():
    """Gate developer-only output that may expose prompts, chunks, or metadata."""
    if not env_flag(DEV_MODE_ENV):
        return False

    expected_token = os.getenv(DEV_TOKEN_ENV)
    if not expected_token:
        return True

    return os.getenv(DEV_TOKEN_INPUT_ENV) == expected_token


def trace_enabled():
    return dev_mode_enabled() and env_flag(TRACE_ENV)


def trace_only_enabled():
    return dev_mode_enabled() and env_flag(TRACE_ONLY_ENV)


def compact_preview(text, limit=180):
    text = " ".join(clean_text(text, "").split())
    if len(text) <= limit:
        return text

    return text[:limit] + "..."


def print_trace_line(text=""):
    print(text, file=sys.stderr)


def print_query_trace(query, query_intent):
    print_trace_line("\n===== MarxOS Trace =====")
    print_trace_line(f"query: {query}")
    print_trace_line(f"intent: {query_intent}")


def print_constraints_trace(constraints):
    if not constraints:
        print_trace_line("routing_constraints: none")
        return

    print_trace_line("routing_constraints:")
    print_trace_line(f"- title: {constraints.get('title')}")
    print_trace_line(f"- sources: {sorted(constraints.get('sources') or [])}")
    print_trace_line(f"- page_ranges: {constraints.get('page_ranges') or {}}")


def print_docs_trace(docs, label="retrieved_docs"):
    print_trace_line(f"{label}: {len(docs)}")

    for index, doc in enumerate(docs, start=1):
        metadata = normalize_metadata(doc.metadata)
        print_trace_line(f"\n[{index}]")
        print_trace_line(
            "metadata: "
            f"book={metadata.get('book')}, article={metadata.get('article')}, "
            f"section={metadata.get('section')}, source={metadata.get('source')}, "
            f"page={metadata.get('page')}, printed_page={metadata.get('printed_page')}, "
            f"pdf_page={metadata.get('pdf_page')}, citation_page={metadata.get('citation_page')}, "
            f"citation_page_type={metadata.get('citation_page_type')}"
        )
        print_trace_line(
            "standard_metadata: "
            f"series={metadata.get('series')}, volume={metadata.get('volume')}, "
            f"publisher={metadata.get('publisher')}, publication_year={metadata.get('publication_year')}, "
            f"source_file={metadata.get('source_file')}"
        )
        if metadata.get("match_type"):
            print_trace_line(
                f"match: type={metadata.get('match_type')}, confidence={metadata.get('confidence')}, "
                f"lookup_scope={metadata.get('lookup_scope')}"
            )
        print_trace_line(f"sentence_citation: {format_citation(metadata, include_article=False)}")
        print_trace_line(f"paragraph_citation: {format_citation(metadata, include_article=True)}")
        print_trace_line(f"preview: {compact_preview(doc.page_content)}")


def print_prompt_trace(prompt):
    print_trace_line("\nprompt_preview:")
    print_trace_line(compact_preview(prompt, limit=500))
    print_trace_line("===== End Trace =====\n")


def build_trace_only_answer(query_intent, docs, prompt):
    lines = [
        "已完成 TRACE_ONLY 调试运行，未调用 DeepSeek。",
        f"intent: {query_intent}",
        f"retrieved_docs: {len(docs)}",
        "",
        "Top chunks:",
    ]

    for index, doc in enumerate(docs, start=1):
        metadata = normalize_metadata(doc.metadata)
        lines.append(
            f"{index}. source={metadata.get('source')}, article={metadata.get('article')}, "
            f"page={metadata.get('page')}, pdf_page={metadata.get('pdf_page')}, "
            f"citation_page={metadata.get('citation_page')}, type={metadata.get('citation_page_type')}"
        )
        lines.append(f"   preview: {compact_preview(doc.page_content, limit=120)}")

    lines.extend(["", "Prompt preview:", compact_preview(prompt, limit=700)])
    return "\n".join(lines)


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def run_query(query):
    query = clean_text(query, "")
    if is_unreadable_query(query):
        return (
            "\u672a\u80fd\u8bfb\u53d6\u5230\u53ef\u7528\u7684\u4e2d\u6587\u95ee\u9898\u3002"
            "\u5982\u679c\u662f\u5728 PowerShell \u4e2d\u901a\u8fc7\u7ba1\u9053\u6216\u91cd\u5b9a\u5411\u8f93\u5165\uff0c"
            "\u8bf7\u5148\u8fd0\u884c `chcp 65001`\uff0c\u6216\u5728\u4ea4\u4e92\u5f0f\u63d0\u793a\u4e2d\u76f4\u63a5\u8f93\u5165\u95ee\u9898\u3002"
        )

    query_intent = classify_query(query)
    trace = trace_enabled()
    trace_only = trace_only_enabled()

    if trace or trace_only:
        print_query_trace(query, query_intent)

    if query_intent == "bibliographic_lookup":
        bibliographic_answer = answer_bibliographic_query(query)
        if trace or trace_only:
            print_trace_line("search_path: local article map / core classics")
            print_trace_line(f"bibliographic_answer_found: {bool(bibliographic_answer)}")
            print_trace_line("===== End Trace =====\n")
        if bibliographic_answer:
            return bibliographic_answer

        title = extract_bibliographic_title(query)
        return f"未能在当前核心书目表中确认《{title}》。"

    if query_intent == "quote_lookup":
        if trace or trace_only:
            print_trace_line("search_path: exact OCR quote lookup")
        answer = answer_quote_query(query, trace=trace or trace_only)
        if trace or trace_only:
            print_trace_line("===== End Trace =====\n")
        return answer

    constraints = constraints_from_query(query)
    if trace or trace_only:
        print_trace_line("search_path: FAISS vector similarity search -> rule rerank -> DeepSeek")
        print_constraints_trace(constraints)
    db = load_vectorstore()
    docs = retrieve_documents(query, db, k=5)
    context = build_context(docs, query_intent)
    prompt = clean_text(build_prompt(query_intent, query, context))
    if trace or trace_only:
        print_docs_trace(docs)
        print_prompt_trace(prompt)

    if trace_only:
        return build_trace_only_answer(query_intent, docs, prompt)

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content


def main():
    query = input("请输入问题：")
    answer = run_query(query)
    print("\n===== MarxOS =====\n")
    print(answer)


if __name__ == "__main__":
    main()
