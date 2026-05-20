from openai import OpenAI

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import json
import os
import re
import sys
from rag.core_classics import classic_entries_for_query, load_core_classics
from rag.exact_quote_lookup import exact_quote_lookup


load_dotenv()
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore/marx_reader_core")
PARAGRAPH_VECTORSTORE_DIR = os.getenv("PARAGRAPH_VECTORSTORE_DIR", "vectorstore/marx_reader_paragraph")
OCR_CACHE_DIR = os.getenv("OCR_CACHE_DIR", "data/ocr_cache")
PAGE_MAP_PATH = os.getenv("PAGE_MAP_PATH", "data/page_map.json")
LAST_EVIDENCE = []
LAST_CITATION_AUDIT = {}
ARTICLE_MAP_PATH = os.getenv("ARTICLE_MAP_PATH", "rag/article_map_core.json")
DEFAULT_PUBLISHER = "人民出版社"
RERANK_DEBUG_ENV = "MARXOS_DEBUG_RERANK"
TRACE_ENV = "MARXOS_TRACE"
TRACE_ONLY_ENV = "MARXOS_TRACE_ONLY"
DUAL_RETRIEVAL_ENV = "MARXOS_DUAL_RETRIEVAL"
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
    """Return whether printed-page metadata should be hidden in citations.

    We now prefer printed_page whenever it exists. PDF/page offsets are kept for
    diagnostics, but user-facing citations should not expose PDF labels.
    """
    return False

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
        return "", "马克思恩格斯选集", f"第{int(match_mes.group(1))}卷", "2012年"

    volume = volume_from_source(stem)
    if volume:
        return "", "马克思恩格斯全集", volume, VOLUME_PUBLICATION_YEARS.get(stem, "")

    match = re.search(r"(第\d+卷[AB]?)", book)
    if "马克思恩格斯文集" in book:
        volume = match.group(1).replace("A", "(上)").replace("B", "(下)") if match else ""
        return "", "马克思恩格斯文集", volume, "2009年"

    if "马克思恩格斯选集" in book:
        volume = match.group(1).replace("A", "(上)").replace("B", "(下)") if match else ""
        return "", "马克思恩格斯选集", volume, "2012年"

    if "马克思恩格斯全集" in book:
        volume = match.group(1).replace("A", "(上)").replace("B", "(下)") if match else ""
        return "", "马克思恩格斯全集", volume, ""

    return "", book, "", ""


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


def normalize_digit_text(text):
    return str(text or "").translate(str.maketrans("０１２３４５６７８９", "0123456789"))




def load_page_map():
    if not os.path.exists(PAGE_MAP_PATH):
        return {}
    try:
        with open(PAGE_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


PAGE_MAP = load_page_map()


def printed_page_from_page_map(source, pdf_page):
    if not PAGE_MAP or not source or pdf_page is None:
        return None
    source_map = (PAGE_MAP.get("sources") or {}).get(source)
    if not source_map:
        return None
    page_info = (source_map.get("pages") or {}).get(str(pdf_page))
    if not page_info:
        return None
    return as_int(page_info.get("printed_page"))


def load_ocr_page_text(source, pdf_page):
    if not source or pdf_page is None:
        return ""
    cache_path = os.path.join(
        OCR_CACHE_DIR,
        source_stem({"source": source}),
        f"page_{pdf_page}.json",
    )
    if not os.path.exists(cache_path):
        return ""
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    return repair_mojibake(payload.get("raw_text") or payload.get("cleaned_text") or "")


def extract_query_terms_for_page_match(query):
    normalized = normalize_for_match(query) if "normalize_for_match" in globals() else str(query or "")
    terms = set()
    for size in (8, 6, 4, 2):
        for index in range(0, max(len(normalized) - size + 1, 0)):
            term = normalized[index : index + size]
            if len(term) == size:
                terms.add(term)
    return terms


def page_match_score(query, page_text):
    page_normalized = normalize_for_match(page_text) if "normalize_for_match" in globals() else str(page_text or "")
    if not page_normalized:
        return 0
    score = 0
    for term in extract_query_terms_for_page_match(query):
        if term in page_normalized:
            score += len(term) * len(term)
    return score

def infer_printed_page_from_ocr_cache(metadata):
    if metadata.get("printed_page") is not None:
        return None

    source = metadata.get("source")
    pdf_page = as_int(metadata.get("pdf_page") or metadata.get("page"))
    if not source or pdf_page is None:
        return None

    mapped_page = printed_page_from_page_map(source, pdf_page)
    if mapped_page is not None:
        return mapped_page

    cache_path = os.path.join(
        OCR_CACHE_DIR,
        source_stem({"source": source}),
        f"page_{pdf_page}.json",
    )
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    raw_text = repair_mojibake(payload.get("raw_text") or payload.get("cleaned_text") or "")
    lines = [normalize_digit_text(line).strip() for line in str(raw_text).splitlines()]
    lines = [line for line in lines if line]
    edge_lines = lines[:3] + lines[-3:]

    candidates = []
    for line in edge_lines:
        match = re.fullmatch(r"[-—–]*\s*(\d{1,4})\s*[-—–]*", line)
        if not match:
            continue
        page = as_int(match.group(1))
        if page is None or page <= 0:
            continue
        # Printed pages usually trail PDF pages by the front-matter offset.
        if -5 <= pdf_page - page <= 180:
            candidates.append(page)

    if not candidates:
        return None

    return candidates[0]


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

    inferred_printed_page = infer_printed_page_from_ocr_cache(normalized)
    if inferred_printed_page is not None:
        normalized["printed_page"] = inferred_printed_page
        normalized["citation_page"] = inferred_printed_page
        normalized["citation_page_type"] = "printed_page"

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


def citation_page_label(metadata):
    metadata = normalize_metadata(metadata)
    citation_page = metadata.get("citation_page")
    printed_page = metadata.get("printed_page")
    pdf_page = metadata.get("pdf_page")

    if printed_page is not None:
        return f"\u7b2c{clean_text(printed_page)}\u9875"
    if citation_page is not None:
        return f"\u7b2c{clean_text(citation_page)}\u9875"
    if pdf_page is not None:
        return f"\u7b2c{clean_text(pdf_page)}\u9875"
    return "\u672a\u77e5\u9875\u7801"


def source_page_label(metadata):
    metadata = normalize_metadata(metadata)
    return citation_page_label(metadata)

def format_citation(metadata, include_article=False):
    metadata = normalize_metadata(metadata)
    author, title, volume, year = normalize_book_parts(metadata)
    article = clean_text(metadata.get("section") or metadata.get("article"), "")
    author_text = f"{author}：" if author else ""
    volume_text = volume if volume else ""
    article_text = f"，{article}" if include_article and article else ""
    year_text = f"，{year}" if year else ""
    page_text = citation_page_label(metadata)

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

    interrogative_markers = [
        "什么是",
        "是什么",
        "何为",
        "如何",
        "怎么",
        "怎样",
        "为什么",
        "本质",
        "意义",
    ]
    if any(marker in query for marker in interrogative_markers):
        return False

    quote_keywords = ["引文", "出处", "出自", "哪一页", "哪页", "页码", "原文", "这句话", "这段话"]
    if any(keyword in query for keyword in quote_keywords):
        return True

    return len(query) >= 24 and not re.search(r"[。！？!?]", query)


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
    communism_patterns = [
        "共产主义是不是",
        "共产主义是否",
        "共产主义会不会",
        "共产主义能不能",
        "共产主义能否",
        "共产主义一定会实现",
        "共产主义必然实现",
        "共产主义会实现",
    ]
    if any(pattern in query for pattern in communism_patterns):
        return True

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


CLASSIC_SAYING_QUERY_SEEDS = [
    "\u5171\u4ea7\u515a\u5ba3\u8a00 \u4e24\u4e2a\u51b3\u88c2 \u5168\u4e16\u754c\u65e0\u4ea7\u8005\u8054\u5408\u8d77\u6765",
    "\u5173\u4e8e\u8d39\u5c14\u5df4\u54c8\u7684\u63d0\u7eb2 \u54f2\u5b66\u5bb6\u4eec\u53ea\u662f\u7528\u4e0d\u540c\u7684\u65b9\u5f0f\u89e3\u91ca\u4e16\u754c \u95ee\u9898\u5728\u4e8e\u6539\u53d8\u4e16\u754c",
    "\u54e5\u8fbe\u7eb2\u9886\u6279\u5224 \u5404\u5c3d\u6240\u80fd \u6309\u9700\u5206\u914d",
    "\u8d44\u672c\u8bba \u8d44\u672c\u6765\u5230\u4e16\u95f4 \u4ece\u5934\u5230\u811a \u6bcf\u4e2a\u6bdb\u5b54\u90fd\u6ef4\u7740\u8840\u548c\u80ae\u810f\u7684\u4e1c\u897f",
    "\u8def\u6613\u6ce2\u62ff\u5df4\u7684\u96fe\u6708\u5341\u516b\u65e5 \u5386\u53f2\u4e8b\u53d8 \u7b2c\u4e00\u6b21\u60b2\u5267 \u7b2c\u4e8c\u6b21\u7b11\u5267",
    "\u5fb7\u610f\u5fd7\u610f\u8bc6\u5f62\u6001 \u7edf\u6cbb\u9636\u7ea7\u7684\u601d\u60f3 \u6bcf\u4e00\u65f6\u4ee3\u5360\u7edf\u6cbb\u5730\u4f4d\u7684\u601d\u60f3",
    "\u53cd\u675c\u6797\u8bba \u81ea\u7531\u662f\u5bf9\u5fc5\u7136\u7684\u8ba4\u8bc6",
    "\u5bb6\u5ead\u79c1\u6709\u5236\u548c\u56fd\u5bb6\u7684\u8d77\u6e90 \u56fd\u5bb6\u4e0d\u662f\u4ece\u6765\u5c31\u6709\u7684",
]


CLASSIC_SAYING_QUOTE_SEEDS = [
    "\u5168\u4e16\u754c\u65e0\u4ea7\u8005\uff0c\u8054\u5408\u8d77\u6765",
    "\u5171\u4ea7\u4e3b\u4e49\u9769\u547d\u5c31\u662f\u540c\u4f20\u7edf\u7684\u6240\u6709\u5236\u5173\u7cfb\u5b9e\u884c\u6700\u5f7b\u5e95\u7684\u51b3\u88c2",
    "\u54f2\u5b66\u5bb6\u4eec\u53ea\u662f\u7528\u4e0d\u540c\u7684\u65b9\u5f0f\u89e3\u91ca\u4e16\u754c\uff0c\u95ee\u9898\u5728\u4e8e\u6539\u53d8\u4e16\u754c",
    "\u5404\u5c3d\u6240\u80fd\uff0c\u6309\u9700\u5206\u914d",
    "\u8d44\u672c\u6765\u5230\u4e16\u95f4\uff0c\u4ece\u5934\u5230\u811a\uff0c\u6bcf\u4e2a\u6bdb\u5b54\u90fd\u6ef4\u7740\u8840\u548c\u80ae\u810f\u7684\u4e1c\u897f",
    "\u4e00\u5207\u5df2\u6b7b\u7684\u5148\u8f88\u4eec\u7684\u4f20\u7edf\uff0c\u50cf\u68a6\u9b47\u4e00\u6837\u7ea0\u7f20\u7740\u6d3b\u4eba\u7684\u5934\u8111",
    "\u7edf\u6cbb\u9636\u7ea7\u7684\u601d\u60f3\u5728\u6bcf\u4e00\u65f6\u4ee3\u90fd\u662f\u5360\u7edf\u6cbb\u5730\u4f4d\u7684\u601d\u60f3",
    "\u81ea\u7531\u662f\u5bf9\u5fc5\u7136\u7684\u8ba4\u8bc6",
]


def is_classic_sayings_query(query):
    query = clean_text(query, "")
    saying_markers = ["\u7ecf\u5178\u8bed\u53e5", "\u7ecf\u5178\u540d\u53e5", "\u540d\u8a00", "\u540d\u53e5", "\u8bed\u5f55"]
    author_markers = ["\u9a6c\u514b\u601d", "\u6069\u683c\u65af", "\u9a6c\u6069", "\u9a6c\u514b\u601d\u4e3b\u4e49"]
    return any(marker in query for marker in saying_markers) and any(marker in query for marker in author_markers)

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
        lines.append(
            f"({index})\u300a{entry['book_title']}\u300b{entry['volume']}\uff0c"
            f"{entry['article']}\uff0c\u5317\u4eac\uff1a\u4eba\u6c11\u51fa\u7248\u793e\uff0c"
            f"\u7b2c{entry['start_page']}-{entry['end_page']}\u9875\u3002"
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

    evidence = evidence_from_docs(exact_docs)
    if not exact_docs:
        answer = "\u672a\u80fd\u5728\u5f53\u524d OCR \u7f13\u5b58\u4e2d\u786e\u8ba4\u8be5\u5f15\u6587\u7684\u7cbe\u786e\u51fa\u5904\u3002"
        set_last_evidence([], {"ok": True, "issues": [], "evidence_count": 0, "answer": answer})
        return answer

    lines = []
    for index, doc in enumerate(exact_docs, start=1):
        lines.append(f"({index}){format_citation(doc.metadata, include_article=True)}")

    answer = "\n".join(lines)
    display_evidence = filter_evidence_to_answer(answer, evidence)
    audit = audit_answer_citations(answer, display_evidence)
    set_last_evidence(display_evidence, audit)
    return audit["answer"]

def core_entries_by_id(classic_id):
    for classic in load_core_classics():
        if classic.get("id") != classic_id:
            continue
        entries = []
        for entry in classic.get("entries") or []:
            entries.append(
                {
                    "source": entry["source"],
                    "article": entry.get("article") or classic.get("title"),
                    "start_page": entry["start_page"],
                    "end_page": entry["end_page"],
                    "classic_id": classic.get("id"),
                    "classic_title": classic.get("title"),
                    "classic_author": classic.get("author"),
                    "classic_work_year": classic.get("work_year"),
                    "classic_work_type": classic.get("work_type"),
                    "entry_type": entry.get("entry_type"),
                    "priority": entry.get("priority", 99),
                }
            )
        return entries
    return []


def manual_locator_entries(title, entries):
    enriched = []
    for entry in entries:
        enriched.append(
            {
                "source": entry["source"],
                "book_title": entry.get("book_title", ""),
                "volume": entry.get("volume", ""),
                "year": entry.get("year", ""),
                "article": title,
                "start_page": entry["start_page"],
                "end_page": entry["end_page"],
                "classic_id": entry.get("classic_id"),
                "classic_title": title,
                "classic_author": entry.get("classic_author"),
                "classic_work_year": entry.get("classic_work_year"),
                "classic_work_type": entry.get("classic_work_type"),
                "entry_type": entry.get("entry_type", "manual_locator"),
                "priority": entry.get("priority", 1),
            }
        )
    return enriched


CLASSIC_LOCATOR_RULES = [
    {
        "tokens_any": ["\u54f2\u5b66\u5bb6\u4eec\u53ea\u662f", "\u4eba\u7684\u672c\u8d28\u4e0d\u662f", "\u4eba\u7684\u672c\u8d28\u662f\u4e00\u5207\u793e\u4f1a\u5173\u7cfb", "\u8d39\u5c14\u5df4\u54c8\u7684\u63d0\u7eb2"],
        "classic_id": "theses_feuerbach",
        "title": "\u5173\u4e8e\u8d39\u5c14\u5df4\u54c8\u7684\u63d0\u7eb2",
    },
    {
        "tokens_any": ["\u5168\u4e16\u754c\u65e0\u4ea7\u8005", "\u6bcf\u4e2a\u4eba\u7684\u81ea\u7531\u53d1\u5c55", "\u9636\u7ea7\u6597\u4e89", "\u8d44\u4ea7\u9636\u7ea7\u5728\u5386\u53f2\u4e0a", "\u8d44\u4ea7\u9636\u7ea7\u7684\u706d\u4ea1", "\u5171\u4ea7\u515a\u5ba3\u8a00"],
        "classic_id": "communist_manifesto",
        "title": "\u5171\u4ea7\u515a\u5ba3\u8a00",
    },
    {
        "tokens_any": ["\u5168\u4e16\u754c\u65e0\u4ea7\u8005\uff0c\u8054\u5408\u8d77\u6765\u6240\u5728\u7ae0\u8282"],
        "classic_id": "communist_manifesto",
        "title": "\u5171\u4ea7\u515a\u5ba3\u8a00 \u7b2c\u56db\u7ae0\u7ed3\u5c3e",
    },
    {
        "tokens_any": ["\u5b97\u6559\u662f\u4eba\u6c11\u7684\u9e26\u7247", "\u9ed1\u683c\u5c14\u6cd5\u54f2\u5b66\u6279\u5224\u5bfc\u8a00"],
        "classic_id": "critique_hegel_law_intro",
        "title": "\u9ed1\u683c\u5c14\u6cd5\u54f2\u5b66\u6279\u5224\u5bfc\u8a00",
    },
    {
        "tokens_any": ["\u5f02\u5316\u52b3\u52a8", "1844\u624b\u7a3f", "1844\u5e74\u7ecf\u6d4e\u5b66\u54f2\u5b66\u624b\u7a3f", "\u52b3\u52a8\u5f02\u5316"],
        "classic_id": "economic_philosophic_manuscripts_1844",
        "title": "1844\u5e74\u7ecf\u6d4e\u5b66\u54f2\u5b66\u624b\u7a3f",
    },
    {
        "tokens_any": ["\u610f\u8bc6\u5728\u4efb\u4f55\u65f6\u5019", "\u7cfb\u7edf\u63d0\u51fa\u552f\u7269\u53f2\u89c2", "\u5171\u4ea7\u4e3b\u4e49\u4e0d\u662f\u5e94\u5f53\u786e\u7acb", "\u56fd\u5bb6\u6d88\u4ea1", "\u5fb7\u610f\u5fd7\u610f\u8bc6\u5f62\u6001"],
        "classic_id": "german_ideology",
        "title": "\u5fb7\u610f\u5fd7\u610f\u8bc6\u5f62\u6001",
    },
    {
        "tokens_any": ["\u54e5\u8fbe\u7eb2\u9886", "\u65e0\u4ea7\u9636\u7ea7\u4e13\u653f", "\u6309\u52b3\u5206\u914d", "\u6309\u9700\u5206\u914d", "\u52b3\u52a8\u4e0d\u662f\u4e00\u5207\u8d22\u5bcc\u7684\u6e90\u6cc9"],
        "classic_id": "critique_gotha_programme",
        "title": "\u54e5\u8fbe\u7eb2\u9886\u6279\u5224",
    },
    {
        "tokens_any": ["\u56fd\u5bb6\u6d88\u4ea1", "\u5171\u4ea7\u4e3b\u4e49\u9636\u6bb5\u8bba"],
        "classic_id": "critique_gotha_programme",
        "title": "\u54e5\u8fbe\u7eb2\u9886\u6279\u5224",
    },
    {
        "tokens_any": ["\u8d44\u672c\u6765\u5230\u4e16\u95f4", "\u5546\u54c1\u662f\u5929\u751f\u7684\u5e73\u7b49\u6d3e", "\u5546\u54c1\u62dc\u7269\u6559", "\u5269\u4f59\u4ef7\u503c", "\u673a\u5668\u5927\u5de5\u4e1a", "\u66b4\u529b\u662f\u6bcf\u4e00\u4e2a\u5b55\u80b2", "\u8d27\u5e01\u5929\u7136\u4e0d\u662f\u91d1\u94f6"],
        "classic_id": "capital_vol1",
        "title": "\u8d44\u672c\u8bba \u7b2c\u4e00\u5377",
    },
    {
        "tokens_any": ["\u5546\u54c1\u62dc\u7269\u6559\u5728\u54ea\u4e00\u7ae0", "\u54ea\u91cc\u8bba\u8ff0\u4e86\u5546\u54c1\u62dc\u7269\u6559"],
        "classic_id": "capital_vol1",
        "title": "\u8d44\u672c\u8bba \u7b2c\u4e00\u5377 \u7b2c\u4e00\u7ae0 \u7b2c\u56db\u8282",
    },
    {
        "tokens_any": ["\u81ea\u7531\u738b\u56fd", "\u5fc5\u8981\u738b\u56fd"],
        "title": "\u8d44\u672c\u8bba \u7b2c\u4e09\u5377",
        "manual_entries": [{"source": "mea07.pdf", "start_page": 1, "end_page": 900, "priority": 1}],
    },
    {
        "tokens_any": ["\u79d1\u5b66\u793e\u4f1a\u4e3b\u4e49", "\u793e\u4f1a\u4e3b\u4e49\u4ece\u7a7a\u60f3\u5230\u79d1\u5b66\u7684\u53d1\u5c55"],
        "classic_id": "socialism_utopian_scientific",
        "title": "\u793e\u4f1a\u4e3b\u4e49\u4ece\u7a7a\u60f3\u5230\u79d1\u5b66\u7684\u53d1\u5c55",
    },
    {
        "tokens_any": ["\u5bb6\u5ead\u3001\u79c1\u6709\u5236\u548c\u56fd\u5bb6", "\u5bb6\u5ead\u79c1\u6709\u5236\u548c\u56fd\u5bb6"],
        "classic_id": "origin_family_private_property_state",
        "title": "\u5bb6\u5ead\u3001\u79c1\u6709\u5236\u548c\u56fd\u5bb6\u7684\u8d77\u6e90",
    },
    {
        "tokens_any": ["\u5df4\u9ece\u516c\u793e", "\u6cd5\u5170\u897f\u5185\u6218"],
        "classic_id": "civil_war_france",
        "title": "\u6cd5\u5170\u897f\u5185\u6218",
    },
    {
        "tokens_any": ["\u56fd\u5bb6\u6d88\u4ea1"],
        "classic_id": "civil_war_france",
        "title": "\u6cd5\u5170\u897f\u5185\u6218",
    },
    {
        "tokens_any": ["\u54f2\u5b66\u7684\u8d2b\u56f0", "\u6279\u5224\u84b2\u9c81\u4e1c", "\u8d2b\u56f0\u7684\u54f2\u5b66"],
        "title": "\u54f2\u5b66\u7684\u8d2b\u56f0",
    },
    {
        "tokens_any": ["\u653f\u6cbb\u7ecf\u6d4e\u5b66\u6279\u5224\u5e8f\u8a00", "\u793e\u4f1a\u5b58\u5728\u51b3\u5b9a", "\u7ecf\u6d4e\u57fa\u7840", "\u4e0a\u5c42\u5efa\u7b51", "\u751f\u4ea7\u529b\u51b3\u5b9a", "\u6cd5\u7684\u5173\u7cfb\u6839\u6e90"],
        "title": "\u653f\u6cbb\u7ecf\u6d4e\u5b66\u6279\u5224\u5e8f\u8a00",
    },
    {
        "tokens_any": ["\u52b3\u52a8\u521b\u9020\u4e86\u4eba\u672c\u8eab", "\u4ece\u733f\u5230\u4eba"],
        "title": "\u52b3\u52a8\u5728\u4ece\u733f\u5230\u4eba\u8f6c\u53d8\u8fc7\u7a0b\u4e2d\u7684\u4f5c\u7528",
        "manual_entries": [{"source": "mea09.pdf", "start_page": 550, "end_page": 563, "priority": 1}],
    },
    {
        "tokens_any": ["\u5386\u53f2\u4e0d\u8fc7\u662f\u8ffd\u6c42\u7740\u81ea\u5df1\u76ee\u7684"],
        "title": "\u795e\u5723\u5bb6\u65cf",
        "manual_entries": [{"source": "mea01.pdf", "start_page": 250, "end_page": 500, "priority": 1}],
    },
    {
        "tokens_any": ["\u65e9\u671f\u4eba\u672c\u4e3b\u4e49", "\u665a\u671f\u653f\u6cbb\u7ecf\u6d4e\u5b66"],
        "classic_id": "economic_philosophic_manuscripts_1844",
        "title": "1844\u5e74\u7ecf\u6d4e\u5b66\u54f2\u5b66\u624b\u7a3f",
    },
    {
        "tokens_any": ["\u65e9\u671f\u4eba\u672c\u4e3b\u4e49", "\u665a\u671f\u653f\u6cbb\u7ecf\u6d4e\u5b66", "\u673a\u5668\u548c\u52b3\u52a8"],
        "classic_id": "capital_vol1",
        "title": "\u8d44\u672c\u8bba",
    },
    {
        "tokens_any": ["\u673a\u5668\u548c\u52b3\u52a8"],
        "classic_id": "economic_philosophic_manuscripts_1844",
        "title": "1844\u5e74\u7ecf\u6d4e\u5b66\u54f2\u5b66\u624b\u7a3f",
    },
    {
        "tokens_any": ["\u673a\u5668\u548c\u52b3\u52a8"],
        "title": "\u653f\u6cbb\u7ecf\u6d4e\u5b66\u6279\u5224\u5927\u7eb2",
        "manual_entries": [{"source": "mea08.pdf", "start_page": 1, "end_page": 900, "priority": 1}],
    },
]


def locator_entries_for_query(query):
    normalized_query = normalize_for_match(query)
    matched_entries = []
    seen = set()

    for rule in CLASSIC_LOCATOR_RULES:
        if not any(normalize_for_match(token) in normalized_query for token in rule["tokens_any"]):
            continue

        title = rule["title"]
        rule_entries = []
        if rule.get("classic_id"):
            entries = core_entries_by_id(rule["classic_id"])
            if entries:
                rule_entries = enrich_core_classic_entries(entries)
                for entry in rule_entries:
                    entry["classic_title"] = title
                    entry["article"] = title

        elif rule.get("manual_entries"):
            rule_entries = manual_locator_entries(title, rule["manual_entries"])

        else:
            rule_entries = find_toc_entries(title)
            for entry in rule_entries:
                entry["classic_title"] = title
                entry["article"] = title

        for entry in rule_entries:
            key = (
                entry.get("source"),
                entry.get("start_page"),
                entry.get("end_page"),
                normalize_for_match(entry.get("classic_title") or entry.get("article")),
            )
            if key in seen:
                continue
            seen.add(key)
            matched_entries.append(entry)

    return matched_entries


def build_page_ranges(entries):
    page_ranges = {}
    for entry in entries:
        page_ranges.setdefault(entry["source"], []).append(
            (entry["start_page"], entry["end_page"])
        )
    return page_ranges


def constraints_from_query(query):
    title = extract_bibliographic_title(query)
    locator_entries = locator_entries_for_query(query)
    if locator_entries:
        title = locator_entries[0].get("classic_title") or locator_entries[0].get("article")
        return {
            "title": title,
            "strict_title": True,
            "entries": locator_entries,
            "sources": {entry["source"] for entry in locator_entries},
            "page_ranges": build_page_ranges(locator_entries),
        }

    core_entries = classic_entries_for_query(title or query)
    if core_entries:
        entries = enrich_core_classic_entries(core_entries)
        title = title or entries[0].get("classic_title")
        return {
            "title": title,
            "strict_title": True,
            "entries": entries,
            "sources": {entry["source"] for entry in entries},
            "page_ranges": build_page_ranges(entries),
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
        "page_ranges": build_page_ranges(entries),
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

    source_ranges = ranges[source]
    if source_ranges and isinstance(source_ranges[0], int):
        source_ranges = [source_ranges]

    return any(start_page <= page <= end_page for start_page, end_page in source_ranges)


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
    "共产主义",
]

CONCEPT_PREFERRED_MARKERS = {
    "异化劳动": ["异化劳动和私有财产", "外化劳动"],
    "外化劳动": ["异化劳动和私有财产", "外化劳动"],
    "剩余价值": ["剩余价值率", "价值增殖过程", "资本论"],
    "剩余价值率": ["剩余价值率", "价值增殖过程", "资本论"],
    "劳动过程": ["劳动过程和价值增殖过程", "劳动过程"],
    "资本": ["资本论", "资本的生产过程", "货币转化为资本"],
    "阶级斗争": ["共产党宣言", "阶级斗争"],
    "共产主义": ["共产党宣言", "哥达纲领批判", "社会主义从空想到科学的发展"],
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
    "\u8d44\u672c": {
        "sources": {"mea01.pdf", "mes02.pdf", "mea07.pdf"},
        "markers": ["\u8d44\u672c", "\u8d44\u672c\u8bba", "\u8d44\u672c\u7684\u5229\u6da6", "\u8d44\u672c\u5173\u7cfb"],
    },
    "\u552f\u7269\u8fa9\u8bc1\u6cd5": {
        "sources": {"mes03.pdf", "mea09.pdf"},
        "markers": ["\u53cd\u675c\u6797\u8bba", "\u81ea\u7136\u8fa9\u8bc1\u6cd5", "\u8def\u5fb7\u7ef4\u5e0c\u00b7\u8d39\u5c14\u5df4\u54c8"],
    },
    "\u9636\u7ea7\u6597\u4e89": {
        "sources": {"mes01.pdf", "mea02.pdf"},
        "markers": ["\u5171\u4ea7\u515a\u5ba3\u8a00"],
    },
    "\u5171\u4ea7\u4e3b\u4e49": {
        "sources": {"mes01.pdf", "mes03.pdf", "mea03.pdf"},
        "markers": ["\u5171\u4ea7\u515a\u5ba3\u8a00", "\u54e5\u8fbe\u7eb2\u9886\u6279\u5224", "\u793e\u4f1a\u4e3b\u4e49\u4ece\u7a7a\u60f3\u5230\u79d1\u5b66\u7684\u53d1\u5c55"],
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
    "\u8d44\u672c": {
        "mea01.pdf": (109, 248),
        "mes02.pdf": (185, 370),
        "mea07.pdf": (397, 488),
    },
    "\u9636\u7ea7\u6597\u4e89": {
        "mes01.pdf": (376, 435),
        "mea02.pdf": (3, 67),
    },
    "\u5171\u4ea7\u4e3b\u4e49": {
        "mes01.pdf": (376, 435),
        "mes03.pdf": (430, 532),
        "mea03.pdf": (481, 553),
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


CONCEPT_CANONICAL_CLASSIC_IDS = {
    "资本": "capital_vol1",
    "商品价值": "capital_vol1",
    "价值": "capital_vol1",
    "剩余价值": "capital_vol1",
    "剩余价值率": "capital_vol1",
    "劳动过程": "capital_vol1",
    "价值增殖过程": "capital_vol1",
    "商品拜物教": "capital_vol1",
    "拜物教": "capital_vol1",
    "阶级斗争": "communist_manifesto",
    "共产主义": "communist_manifesto",
    "国家": "origin_family_private_property_state",
    "国家的起源": "origin_family_private_property_state",
    "国家的产生": "origin_family_private_property_state",
    "私有制": "origin_family_private_property_state",
    "家庭": "origin_family_private_property_state",
    "家庭私有制和国家的起源": "origin_family_private_property_state",
    "异化劳动": "economic_philosophic_manuscripts_1844",
    "外化劳动": "economic_philosophic_manuscripts_1844",
    "费尔巴哈提纲": "theses_feuerbach",
    "实践": "theses_feuerbach",
    "唯物辩证法": "anti_duhring",
    "自然辩证法": "dialectics_nature",
}

CONCEPT_TITLE_FALLBACK_TO_CLASSIC = {
    "\u56fd\u5bb6",
    "\u56fd\u5bb6\u7684\u8d77\u6e90",
    "\u56fd\u5bb6\u7684\u4ea7\u751f",
    "\u79c1\u6709\u5236",
    "\u5bb6\u5ead",
}


def core_classic_by_id(classic_id):
    for classic in load_core_classics():
        if classic.get("id") == classic_id:
            return classic
    return None


def canonical_concept_entries(query):
    entries = []
    seen = set()

    for term in active_concept_terms(query):
        classic_id = CONCEPT_CANONICAL_CLASSIC_IDS.get(term)
        classic = core_classic_by_id(classic_id) if classic_id else None
        if not classic:
            continue

        for entry in classic.get("entries") or []:
            key = (classic_id, entry.get("source"), entry.get("start_page"), entry.get("end_page"))
            if key in seen:
                continue
            seen.add(key)
            entries.append((term, classic, entry))

    return entries


def metadata_printed_page(metadata):
    for key in ("printed_page", "page"):
        page = as_int(metadata.get(key))
        if page is not None:
            return page
    return None


def concept_article_title_is_weak(query, metadata):
    article = clean_article_title(metadata.get("section") or metadata.get("article"))
    book = clean_text(metadata.get("book"), "")
    article_norm = normalize_for_match(article)

    if not article or is_noisy_article_title(article) or article == book:
        return True

    for term in active_concept_terms(query):
        markers = CONCEPT_PREFERRED_MARKERS.get(term, [])
        if any(normalize_for_match(marker) in article_norm for marker in markers):
            return False

    return True


def concept_title_from_content(query, metadata, content):
    article = clean_article_title(metadata.get("section") or metadata.get("article"))
    article_norm = normalize_for_match(article)
    content_norm = normalize_for_match(content)
    lead_norm = normalize_for_match(content[:500])

    for term in active_concept_terms(query):
        term_norm = normalize_for_match(term)
        if not term_norm:
            continue
        if term in CONCEPT_TITLE_FALLBACK_TO_CLASSIC:
            continue

        markers = CONCEPT_PREFERRED_MARKERS.get(term, [])
        preferred = CONCEPT_PREFERRED_SOURCES.get(term) or {}
        term_markers = [] if term in CONCEPT_TITLE_FALLBACK_TO_CLASSIC else [term]
        markers = list(dict.fromkeys(markers + term_markers + list(preferred.get("markers", []))))

        for marker in markers:
            marker_norm = normalize_for_match(marker)
            if marker_norm and (marker_norm in article_norm or marker_norm in lead_norm):
                return marker

        if term_norm in content_norm:
            return term

    return None


def canonical_concept_entry_for_metadata(query, metadata):
    source = metadata.get("source")
    page = metadata_printed_page(metadata)
    if not source or page is None:
        return None

    for term, classic, entry in canonical_concept_entries(query):
        if source != entry.get("source"):
            continue
        start_page = as_int(entry.get("start_page"))
        end_page = as_int(entry.get("end_page"))
        if start_page is not None and end_page is not None and start_page <= page <= end_page:
            return term, classic, entry

    return None


def enrich_concept_metadata(query, docs):
    if not active_concept_terms(query):
        return docs

    for doc in docs:
        for key in ("article", "section"):
            original_title = doc.metadata.get(key)
            cleaned_title = clean_article_title(original_title)
            if cleaned_title and original_title and cleaned_title != original_title:
                doc.metadata.setdefault(f"raw_{key}", original_title)
                doc.metadata[key] = cleaned_title

        match = canonical_concept_entry_for_metadata(query, doc.metadata)
        if not match:
            continue

        _term, classic, entry = match
        title = classic.get("title")
        if not title:
            continue

        doc.metadata.setdefault("classic_id", classic.get("id"))
        doc.metadata.setdefault("classic_title", title)
        doc.metadata.setdefault("classic_author", classic.get("author"))
        doc.metadata.setdefault("classic_work_year", classic.get("work_year"))
        doc.metadata.setdefault("classic_work_type", classic.get("work_type"))
        doc.metadata.setdefault("entry_type", entry.get("entry_type"))

        if concept_article_title_is_weak(query, doc.metadata):
            concept_title = concept_title_from_content(query, doc.metadata, doc.page_content)
            doc.metadata.setdefault("raw_article", doc.metadata.get("article"))
            doc.metadata.setdefault("raw_section", doc.metadata.get("section"))
            doc.metadata["article"] = concept_title or title
            doc.metadata["section"] = concept_title or title

    return docs


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
        elif term_norm in article_norm:
            score -= 45

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

    query_norm = normalize_for_match(query)
    if "共产主义" in terms:
        malthus_norm = normalize_for_match("马尔萨斯")
        if malthus_norm in article_norm and malthus_norm not in query_norm:
            score -= 60

    return min(score, 320)


def score_concept_source_priority(query, metadata):
    terms = active_concept_terms(query)
    if not terms:
        return 0

    score = 0
    source = metadata.get("source")
    citation_page = metadata_citation_page(metadata)

    for term in terms:
        preferred = CONCEPT_PREFERRED_SOURCES.get(term) or {}
        preferred_sources = preferred.get("sources", set())
        if preferred_sources:
            score += 12 if source in preferred_sources else -12

        preferred_range = CONCEPT_PREFERRED_PAGE_RANGES.get(term, {}).get(source)
        if preferred_range and citation_page is not None:
            start_page, end_page = preferred_range
            if start_page <= citation_page <= end_page:
                score += 18

    return score


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
            "concept_source": score_concept_source_priority(query, metadata),
            "document_quality": score_document_quality(metadata, content),
        }
        score = sum(score_parts.values())
        debug_rerank_score(index, doc, score_parts)

        ranked.append((score, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for score, doc in ranked]


def diversify_documents(docs, k, max_per_source=2, max_per_article=1):
    selected = []
    source_counts = {}
    article_counts = {}

    for doc in docs:
        metadata = doc.metadata
        source = metadata.get("source") or ""
        article = clean_text(metadata.get("section") or metadata.get("article"), "")
        article_key = (source, normalize_for_match(article))

        if source_counts.get(source, 0) >= max_per_source:
            continue
        if article_key[1] and article_counts.get(article_key, 0) >= max_per_article:
            continue

        selected.append(doc)
        source_counts[source] = source_counts.get(source, 0) + 1
        article_counts[article_key] = article_counts.get(article_key, 0) + 1
        if len(selected) >= k:
            return selected

    for doc in docs:
        if doc in selected:
            continue
        selected.append(doc)
        if len(selected) >= k:
            break

    return selected


def annotate_docs_with_constraints(docs, constraints):
    title = constraints.get("title")
    entries = constraints.get("entries") or []
    if not title and not entries:
        return docs

    source_entries = {}
    for entry in entries:
        source_entries.setdefault(entry.get("source"), []).append(entry)

    for doc in docs:
        metadata = doc.metadata
        if title:
            if not metadata.get("classic_title"):
                metadata["classic_title"] = title
            if not metadata.get("work_title"):
                metadata["work_title"] = title
            if not metadata.get("locator_title"):
                metadata["locator_title"] = title

        try:
            page = int(metadata.get("page"))
        except (TypeError, ValueError):
            page = None

        matched_entry = None
        for entry in source_entries.get(metadata.get("source"), []):
            if page is None or entry["start_page"] <= page <= entry["end_page"]:
                matched_entry = entry
                break

        if matched_entry:
            entry_title = matched_entry.get("classic_title") or matched_entry.get("article") or title
            if entry_title:
                metadata["classic_title"] = entry_title
                metadata["work_title"] = entry_title
                metadata["locator_title"] = entry_title
            if matched_entry.get("classic_author"):
                metadata.setdefault("classic_author", matched_entry.get("classic_author"))
            if matched_entry.get("classic_work_type"):
                metadata.setdefault("classic_work_type", matched_entry.get("classic_work_type"))

    return docs


def locator_backstop_documents(constraints, limit=4):
    docs = []
    seen_titles = set()
    for entry in constraints.get("entries") or []:
        title = entry.get("classic_title") or entry.get("article") or constraints.get("title")
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        metadata = {
            "source": entry.get("source"),
            "page": entry.get("start_page"),
            "citation_page": entry.get("start_page"),
            "citation_page_type": "pdf_page",
            "article": title,
            "section": title,
            "classic_title": title,
            "work_title": title,
            "locator_title": title,
            "classic_author": entry.get("classic_author"),
            "classic_work_type": entry.get("classic_work_type"),
            "match_type": "locator_backstop",
        }
        content = (
            f"{title}\n"
            f"\u5b9a\u4f4d\u63d0\u793a\uff1a\u8be5\u95ee\u9898\u5bf9\u5e94\u5230\u300a{title}\u300b"
            f"\uff0cPDF\u7b2c{entry.get('start_page')}-{entry.get('end_page')}\u9875\u8303\u56f4\u5185\u6838\u5bf9\u3002"
        )
        docs.append(Document(page_content=content, metadata=metadata))
        if len(docs) >= limit:
            break
    return docs


def append_locator_backstops(docs, constraints, k):
    if not constraints.get("strict_title") or not constraints.get("entries"):
        return docs

    backstops = locator_backstop_documents(constraints, limit=k)
    existing_titles = {
        normalize_for_match(doc.metadata.get("classic_title") or doc.metadata.get("locator_title"))
        for doc in docs
    }
    missing_backstops = []
    for doc in backstops:
        title_key = normalize_for_match(doc.metadata.get("classic_title"))
        if title_key and title_key in existing_titles:
            continue
        missing_backstops.append(doc)
        existing_titles.add(title_key)

    if not missing_backstops:
        return docs[:k]

    keep_count = max(0, k - len(missing_backstops))
    return docs[:keep_count] + missing_backstops[:k]


def retrieve_documents(query, db, k=5, allow_exact_quote=True):
    constraints = constraints_from_query(query)
    normalized_query = normalize_for_match(query)

    if constraints.get("strict_title") and "\u65e0\u4ea7\u9636\u7ea7\u4e13\u653f" in normalized_query:
        return locator_backstop_documents(constraints, limit=k)

    if allow_exact_quote and is_quote_lookup_query(query):
        exact_docs = exact_quote_lookup(query, OCR_CACHE_DIR, limit=k)
        if exact_docs:
            docs = annotate_docs_with_constraints(exact_docs, constraints)
            return append_locator_backstops(docs, constraints, k)

    fetch_k = max(120 if constraints or active_concept_terms(query) else 30, k * 12)

    if constraints.get("sources"):
        candidates = db.similarity_search(query, k=fetch_k)
        candidates = [
            doc for doc in candidates
            if metadata_matches_constraints(doc.metadata, constraints)
        ]

        # For explicit title-constrained queries, prefer candidates that are
        # inside the mapped page ranges to avoid prefaces/index pages hijacking
        # answers with incorrect citations.
        if constraints.get("title") and constraints.get("page_ranges"):
            ranged_candidates = [
                doc for doc in candidates
                if page_in_expected_range(doc.metadata, constraints)
            ]
            if ranged_candidates:
                candidates = ranged_candidates

        if constraints.get("strict_title") and constraints.get("page_ranges"):
            candidates = [
                doc for doc in candidates
                if page_in_expected_range(doc.metadata, constraints)
            ]

        if not candidates:
            title_query = constraints.get("title") or query
            candidates = [
                doc for doc in db.similarity_search(title_query, k=fetch_k)
                if metadata_matches_constraints(doc.metadata, constraints)
            ]
            if constraints.get("strict_title") and constraints.get("page_ranges"):
                candidates = [
                    doc for doc in candidates
                    if page_in_expected_range(doc.metadata, constraints)
                ]

        if not candidates:
            if constraints.get("strict_title"):
                return locator_backstop_documents(constraints, limit=k)
            candidates = db.similarity_search(query, k=fetch_k)
    else:
        candidates = db.similarity_search(query, k=fetch_k)

    if is_classic_sayings_query(query):
        expanded = []
        seen = set()
        for quote in CLASSIC_SAYING_QUOTE_SEEDS:
            for doc in exact_quote_lookup(quote, OCR_CACHE_DIR, limit=2):
                key = (
                    doc.metadata.get("source"),
                    doc.metadata.get("page"),
                    doc.metadata.get("article") or doc.metadata.get("section"),
                    clean_text(doc.page_content, "")[:80],
                )
                if key in seen:
                    continue
                seen.add(key)
                expanded.append(doc)

        seed_k = max(12, k * 3)
        for seed in CLASSIC_SAYING_QUERY_SEEDS:
            for doc in db.similarity_search(seed, k=seed_k):
                key = (
                    doc.metadata.get("source"),
                    doc.metadata.get("page"),
                    doc.metadata.get("article") or doc.metadata.get("section"),
                    clean_text(doc.page_content, "")[:80],
                )
                if key in seen:
                    continue
                seen.add(key)
                expanded.append(doc)

        for doc in candidates:
            key = (
                doc.metadata.get("source"),
                doc.metadata.get("page"),
                doc.metadata.get("article") or doc.metadata.get("section"),
                clean_text(doc.page_content, "")[:80],
            )
            if key in seen:
                continue
            seen.add(key)
            expanded.append(doc)
        candidates = expanded

    if is_classic_sayings_query(query):
        docs = diversify_documents(candidates, k, max_per_source=2, max_per_article=1)
        docs = annotate_docs_with_constraints(docs, constraints)
        return append_locator_backstops(docs, constraints, k)

    ranked_docs = rerank_documents(query, candidates, constraints)
    if not constraints and classify_query(query) == "rag_answer" and k > 5:
        docs = diversify_documents(ranked_docs, k)
    else:
        docs = ranked_docs[:k]

    if classify_query(query) == "concept_explain":
        docs = enrich_concept_metadata(query, docs)

    if allow_exact_quote and is_quote_lookup_query(query):
        for doc in docs:
            doc.metadata["match_type"] = "vector_candidate"
            doc.metadata["confidence"] = 0.0

    docs = annotate_docs_with_constraints(docs, constraints)
    return append_locator_backstops(docs, constraints, k)




def candidate_pdf_pages_from_metadata(metadata):
    pages = []
    for key in ("pdf_page", "page"):
        page = as_int(metadata.get(key))
        if page is not None:
            pages.append(page)
    for key in ("page_span", "page_range"):
        value = metadata.get(key)
        if isinstance(value, (list, tuple)):
            pages.extend(page for page in (as_int(item) for item in value) if page is not None)
        elif isinstance(value, str):
            pages.extend(as_int(item) for item in re.findall(r"\d+", value))
    pages = [page for page in pages if page is not None]
    if len(pages) == 1:
        pages.extend([pages[0] - 1, pages[0] + 1])
    if pages:
        lo, hi = min(pages), max(pages)
        pages = list(range(max(1, lo), hi + 1))
    return sorted(set(pages))


def refine_doc_citation_page_for_query(doc, query):
    metadata = dict(doc.metadata or {})
    source = metadata.get("source")
    candidate_pages = candidate_pdf_pages_from_metadata(metadata)
    if not source or len(candidate_pages) <= 1:
        return doc

    scored_pages = []
    for pdf_page in candidate_pages:
        text = load_ocr_page_text(source, pdf_page)
        # Page choice should follow the retrieved evidence text first. The user
        # query is only a secondary hint; otherwise broad analytical questions
        # can pull a cross-page paragraph back to the wrong page.
        score = page_match_score(doc.page_content, text) * 2 + page_match_score(query, text)
        printed = infer_printed_page_from_ocr_cache({"source": source, "pdf_page": pdf_page})
        scored_pages.append((score, printed is not None, pdf_page, printed))

    scored_pages.sort(reverse=True)
    best_score, has_printed, best_pdf_page, best_printed_page = scored_pages[0]
    if best_score <= 0:
        return doc

    refined = Document(page_content=doc.page_content, metadata=metadata)
    refined.metadata["pdf_page"] = best_pdf_page
    refined.metadata["page"] = best_pdf_page
    refined.metadata["citation_page_refined"] = True
    refined.metadata["citation_page_refined_by"] = "query_ocr_page_overlap"
    if best_printed_page is not None:
        refined.metadata["printed_page"] = best_printed_page
        refined.metadata["citation_page"] = best_printed_page
        refined.metadata["citation_page_type"] = "printed_page"
    return refined


def refine_docs_citation_pages_for_query(docs, query):
    return [refine_doc_citation_page_for_query(doc, query) for doc in docs]

def retrieve_paragraph_documents(query, db, k=5):
    docs = retrieve_documents(query, db, k=k, allow_exact_quote=False)
    for doc in docs:
        doc.metadata.setdefault("retrieval_unit", "paragraph")
        doc.metadata.setdefault("match_type", "paragraph_vector_candidate")
    return docs


def final_answer_style_rules():
    return (
        "\n\u6700\u7ec8\u56de\u7b54\u98ce\u683c\uff1a\n"
        "1. \u76f4\u63a5\u56de\u7b54\u95ee\u9898\uff0c\u4e0d\u8981\u95ee\u5019\uff0c\u4e0d\u8981\u81ea\u6211\u4ecb\u7ecd\uff0c\u4e0d\u8981\u8bf4\u201c\u4f60\u597d\u201d\u6216\u201c\u6211\u662f MarxOS\u201d\u3002\n"
        "2. \u7ed3\u5c3e\u4e0d\u8981\u8ffd\u52a0\u201c\u5982\u679c\u9700\u8981\u201d\u201c\u6211\u53ef\u4ee5\u7ee7\u7eed\u201d\u7b49\u9080\u8bf7\u5f0f\u8bdd\u8bed\u3002\n"
        "3. \u5f15\u7528\u539f\u8457\u65f6\uff0c\u53ea\u4f7f\u7528\u4e0b\u65b9\u63d0\u4f9b\u7684\u51fa\u5904\u683c\u5f0f\uff0c\u4e0d\u8981\u81ea\u884c\u7f16\u9020\u7bc7\u540d\u6216\u9875\u7801\u3002\n"
        "4. \u4e0d\u8981\u8f93\u51fa\u201c\u3010\u539f\u8457\u5185\u5bb9\u3011\u201d\u201c\u3010\u68c0\u7d22\u6750\u6599\u3011\u201d\u6216\u201cCTX-1\u201d\u7b49\u5185\u90e8\u680f\u76ee\u540d\u548c\u5185\u90e8\u7f16\u53f7\u3002\n"
        "5. \u4e0a\u4e0b\u6587\u4ee5 EVIDENCE-CARD \u7ed9\u51fa\uff1b\u6bcf\u4e2a\u5173\u952e\u5224\u65ad\u53ea\u80fd\u4f7f\u7528\u8fd9\u4e9b\u8bc1\u636e\u5361\u7684\u51fa\u5904\uff0c\u4e0d\u5f97\u81ea\u884c\u8865\u9875\u7801\u6216\u7bc7\u540d\u3002\n"
    )


def footnote_citation_rules():
    return (
        "\n\u5f15\u6587\u5448\u73b0\u683c\u5f0f\uff08\u5f3a\u5236\uff09\uff1a\n"
        "1. \u6b63\u6587\u4e2d\u7684\u5f15\u6587\u6216\u5224\u65ad\u53e5\u540e\u9762\u4f7f\u7528\u4e0a\u6807\u811a\u6ce8\u7f16\u53f7\uff08\u5982\u00b9\u00b2\u00b3\uff09\u3002\n"
        "2. \u6587\u672b\u5355\u72ec\u5217\u201c\u5f15\u6587\u6ce8\u91ca\u201d\u5c0f\u8282\uff0c\u6309 1,2,3... \u7edf\u4e00\u5217\u51fa\u5b8c\u6574\u51fa\u5904\u3002\n"
        "3. \u4e0d\u8981\u628a\u5b8c\u6574\u51fa\u5904\u63d2\u5728\u53e5\u5b50\u4e2d\u95f4\u3002\n"
        "4. \u5f15\u6587\u6ce8\u91ca\u4e0d\u8981\u5199\u201c\u540c\u4e0a\u201d\uff1b\u591a\u4e2a\u4e0a\u6807\u6307\u5411\u540c\u4e00\u6761\u51fa\u5904\u65f6\uff0c\u8981\u5408\u5e76\u4e3a\u4e00\u6761\u5b8c\u6574\u51fa\u5904\u3002\n"
        "5. \u9875\u7801\u7edf\u4e00\u5199\u201c\u7b2cX\u9875\u201d\uff0c\u4e0d\u8981\u5199\u201cPDF\u7b2cX\u9875\u201d\u3001\u201cpdf_page\u201d\u6216\u201c\u5370\u5237\u9875\u4f4e\u4fe1\u4efb\u201d\u3002\n"
        "6. \u4e0d\u8981\u5199 1930 \u5e74\u4e0a\u6d77\u6c5f\u5357\u4e66\u5e97\u30011940 \u5e74\u5ef6\u5b89\u89e3\u653e\u793e\u7b49\u7248\u672c\u6cbf\u9769\u63cf\u8ff0\uff0c"
        "\u7edf\u4e00\u4f7f\u7528\u201c\u5317\u4eac\uff1a\u4eba\u6c11\u51fa\u7248\u793e\u201d\u3002\n"
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
        f"3. \u9875\u7801\u7edf\u4e00\u6309\u68c0\u7d22\u6750\u6599\u63d0\u4f9b\u7684\u201c\u53e5\u5b50\u5f15\u6587\u683c\u5f0f\u201d\u8f93\u51fa\uff0c"
        f"\u53ea\u5199\u201c\u7b2cX\u9875\u201d\uff0c\u4e0d\u8981\u5199\u201cPDF\u7b2cX\u9875\u201d\u6216\u201cpdf_page\u201d\u3002\n"
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
        f"\u7bc7\u76ee\u8986\u76d6\u8981\u6c42\uff1a\u5728\u6750\u6599\u5141\u8bb8\u7684\u524d\u63d0\u4e0b\uff0c"
        f"\u5c3d\u91cf\u4f7f\u7528\u591a\u4e2a\u7ecf\u5178\u7bc7\u76ee\u7684\u4ee3\u8868\u6027\u53e5\u5b50\uff0c\u4e0d\u8981\u53ea\u56f4\u7ed5 1-2 \u7bc7\u5c55\u5f00\u3002\n"
        f"{footnote_citation_rules()}\n"
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
        f"1. \u4f18\u5148\u4f9d\u636e\u539f\u8457\u5185\u5bb9\uff0c\u4e14\u81f3\u5c11\u4f7f\u7528\u4e24\u6761\u4e0d\u540c\u6750\u6599\u652f\u6491\u5173\u952e\u5224\u65ad\u3002\n"
        f"2. \u56de\u7b54\u5206\u4e09\u5c42\uff1a\u5148\u7ed9\u7ed3\u8bba\uff0c\u518d\u7ed9\u7406\u8bba\u673a\u5236\uff0c\u6700\u540e\u7ed9\u73b0\u5b9e\u6307\u5411\u6216\u5386\u53f2\u6761\u4ef6\u3002\n"
        f"3. \u5141\u8bb8\u5448\u73b0\u5185\u90e8\u5f20\u529b\uff1a\u53ef\u6307\u51fa\u5b9e\u73b0\u6761\u4ef6\u3001\u9636\u6bb5\u5dee\u5f02\u6216\u5386\u53f2\u9650\u5236\uff0c\u800c\u975e\u53ea\u7ed9\u5355\u7ebf\u7ed3\u8bba\u3002\n"
        f"4. \u81f3\u5c11\u7ed9\u51fa\u4e24\u5904\u7b80\u77ed\u51fa\u5904\uff1b\u82e5\u6750\u6599\u4e0d\u8db3\u4ee5\u652f\u6301\u67d0\u5224\u65ad\uff0c\u8981\u660e\u786e\u8bf4\u660e\u4e0d\u786e\u5b9a\u5904\u3002\n"
        f"5. \u56f4\u7ed5\u6982\u5ff5\u3001\u903b\u8f91\u548c\u73b0\u5b9e\u6307\u5411\u5c55\u5f00\uff0c\u4e0d\u7a7a\u558a\u53e3\u53f7\u3002\n\n"
        f"\u7bc7\u76ee\u8986\u76d6\u8981\u6c42\uff1a\u5728\u6750\u6599\u5141\u8bb8\u7684\u524d\u63d0\u4e0b\uff0c"
        f"\u5c3d\u91cf\u4f7f\u7528\u591a\u4e2a\u7ecf\u5178\u7bc7\u76ee\u7684\u4ee3\u8868\u6027\u53e5\u5b50\uff0c\u4e0d\u8981\u53ea\u56f4\u7ed5 1-2 \u7bc7\u5c55\u5f00\u3002\n"
        f"{footnote_citation_rules()}\n"
        f"\u7981\u6b62\u8f93\u51fa\uff1a\u4e0d\u8981\u5199\u201c\u8d44\u65991\u201d\u201c\u8d44\u65992\u201d"
        f"\u201c\u7247\u6bb51\u201d\u201c\u68c0\u7d22\u6750\u6599\u201d\u7b49\u5185\u90e8\u7f16\u53f7\uff1b"
        f"\u9700\u8981\u5f15\u7528\u65f6\uff0c\u53ea\u4f7f\u7528\u51fa\u5904\u6587\u672c\u3002\n\n"
        f"# \u539f\u8457\u5185\u5bb9\n{context}\n\n# \u7528\u6237\u95ee\u9898\n{query}\n"
    )


def build_default_prompt(query, context):
    return (
        f"\n\u4f60\u662f MarxOS\uff0c\u4e00\u4e2a\u9a6c\u514b\u601d\u4e3b\u4e49\u5b66\u672f\u52a9\u624b\u3002\n\n"
        f"\u8bf7\u6839\u636e\u3010\u539f\u8457\u5185\u5bb9\u3011\u56de\u7b54\u7528\u6237\u95ee\u9898\uff0c"
        f"\u4f18\u5148\u7ed9\u51fa\u7ed3\u6784\u5316\u3001\u4fe1\u606f\u5bc6\u5ea6\u9ad8\u7684\u56de\u7b54\u3002\n"
        f"{final_answer_style_rules()}\n"
        f"\u56de\u7b54\u7ed3\u6784\uff1a\n"
        f"1. \u5148\u7528 1-2 \u53e5\u76f4\u63a5\u56de\u7b54\u95ee\u9898\u7ed3\u8bba\u3002\n"
        f"2. \u518d\u5206 3-5 \u70b9\u5c55\u5f00\uff08\u6982\u5ff5\u5b9a\u4e49\u3001\u673a\u5236\u903b\u8f91\u3001\u5386\u53f2/\u73b0\u5b9e\u610f\u4e49\uff09\uff0c\u6bcf\u70b9\u81f3\u5c11 2-3 \u53e5\u3002\n"
        f"3. \u5c3d\u91cf\u7528\u539f\u8457\u6750\u6599\u652f\u6491\u5173\u952e\u5224\u65ad\uff0c\u51fa\u5904\u8981\u7b80\u77ed\u6e05\u6670\u3002\n"
        f"4. \u82e5\u6750\u6599\u4e0d\u8db3\u652f\u6301\u67d0\u7ed3\u8bba\uff0c\u8981\u660e\u786e\u8bf4\u201c\u6750\u6599\u4e0d\u8db3/\u5f85\u6838\u5bf9\u201d\u3002\n"
        f"5. \u7981\u6b62\u53e3\u53f7\u5f0f\u3001\u7a7a\u6d1e\u8868\u8ff0\u3002\n\n"
        f"\u7bc7\u76ee\u8986\u76d6\u8981\u6c42\uff1a\n"
        f"1. \u5728\u6750\u6599\u5141\u8bb8\u7684\u524d\u63d0\u4e0b\uff0c\u5c3d\u91cf\u8986\u76d6\u591a\u4e2a\u7ecf\u5178\u7bc7\u76ee\uff0c\u4e0d\u8981\u53ea\u56f4\u7ed5 1-2 \u7bc7\u5c55\u5f00\u3002\n"
        f"2. \u4f18\u5148\u9009\u7528\u4e0d\u540c\u6765\u6e90\u7684\u4ee3\u8868\u6027\u53e5\u5b50\u3002\n\n"
        f"\u51fa\u5904\u8981\u6c42\uff1a\n"
        f"1. \u53ea\u80fd\u4f7f\u7528\u4e0a\u4e0b\u6587\u7ed9\u51fa\u7684\u51fa\u5904\u683c\u5f0f\uff0c\u4e0d\u5f97\u81ea\u884c\u7f16\u9020\u9875\u7801\u3002\n"
        f"2. \u9875\u7801\u7edf\u4e00\u5199\u201c\u7b2cX\u9875\u201d\uff0c\u4e0d\u8981\u5199 PDF\u3001pdf_page \u6216\u201c\u540c\u4e0a\u201d\u3002\n"
        f"3. \u4e0d\u8981\u5199 1930 \u5e74\u4e0a\u6d77\u6c5f\u5357\u4e66\u5e97\u30011940 \u5e74\u5ef6\u5b89\u89e3\u653e\u793e\u7b49\u7248\u672c\u6cbf\u9769\u63cf\u8ff0\uff0c"
        f"\u7edf\u4e00\u4f7f\u7528\u201c\u5317\u4eac\uff1a\u4eba\u6c11\u51fa\u7248\u793e\u201d\u3002\n\n"
        f"{footnote_citation_rules()}\n"
        f"\u4e0d\u8981\u8f93\u51fa\u201c\u68c0\u7d22\u6765\u6e90\u201d\u7b49\u5185\u90e8\u8c03\u8bd5\u4fe1\u606f\u3002\n\n"
        f"\u7981\u6b62\u8f93\u51fa\uff1a\u4e0d\u8981\u5199\u201c\u8d44\u65991\u201d\u201c\u8d44\u65992\u201d"
        f"\u201c\u7247\u6bb51\u201d\u201c\u68c0\u7d22\u6750\u6599\u201d\u7b49\u5185\u90e8\u7f16\u53f7\uff1b"
        f"\u9700\u8981\u5f15\u7528\u65f6\uff0c\u53ea\u4f7f\u7528\u51fa\u5904\u6587\u672c\u3002\n\n"
        f"# \u539f\u8457\u5185\u5bb9\n{context}\n\n# \u7528\u6237\u95ee\u9898\n{query}\n"
    )


def build_constraint_guard(constraints):
    sources = sorted(constraints.get("sources") or [])
    if not sources:
        return ""

    source_text = "、".join(sources)
    return (
        "\n引用约束（必须严格遵守）：\n"
        f"1. 本题只允许引用以下来源：{source_text}。\n"
        "2. 不得写出任何不在该列表中的卷次、书名或来源。\n"
        "3. 若材料不足，请明确写“当前材料不足以支持该卷次判断”，不要补写其他卷次。\n"
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
        source = clean_text(metadata.get("source"), "\u672a\u77e5\u6765\u6e90")
        evidence_id = clean_text(metadata.get("paragraph_id"), f"E{i}")
        printed_page = metadata.get("printed_page")
        citation_page = metadata.get("citation_page")
        pdf_page = metadata.get("pdf_page")
        line_start = metadata.get("line_start")
        line_end = metadata.get("line_end")
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
        classic_meta_text = ""
        section_text = f"\uff0c{section}" if section and section != article else ""
        sentence_citation = format_citation(metadata, include_article=False)
        detailed_source = format_citation(metadata, include_article=True)
        source_page = source_page_label(metadata)

        context_parts.append(
            f"EVIDENCE-CARD E{i}\n"
            f"evidence_id={evidence_id}\n"
            f"\u6765\u6e90\uff1a\u300a{book}\u300b{article}{section_text}\uff0c{source_page}\uff0csource={source}\n"
            f"{confidence_text}\n"
            f"{classic_meta_text}"
            f"metadata_fields: book={book}, article={article}, section={section}, page={page}, source={source}\n"
            f"page_fields: printed_page={printed_page}, citation_page={citation_page}, pdf_page={pdf_page}{page_range_text}\n"
            f"position_fields: line_start={line_start}, line_end={line_end}\n"
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


def dual_retrieval_enabled():
    return dev_mode_enabled() and env_flag(DUAL_RETRIEVAL_ENV)


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


def build_trace_only_answer(query_intent, docs, prompt, paragraph_docs=None):
    paragraph_docs = paragraph_docs or []
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

    if paragraph_docs:
        lines.extend(["", "Top paragraphs:"])
        for index, doc in enumerate(paragraph_docs, start=1):
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


def paragraph_vectorstore_exists():
    return os.path.exists(os.path.join(PARAGRAPH_VECTORSTORE_DIR, "index.faiss"))


def load_paragraph_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        PARAGRAPH_VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_dual_documents(query, chunk_db, paragraph_db, k=5):
    return {
        "chunk": retrieve_documents(query, chunk_db, k=k),
        "paragraph": retrieve_paragraph_documents(query, paragraph_db, k=k),
    }


def merge_prefer_paragraph_docs(paragraph_docs, chunk_docs, limit):
    merged = []
    seen = set()
    for doc in list(paragraph_docs or []) + list(chunk_docs or []):
        metadata = doc.metadata or {}
        key = (
            metadata.get("source"),
            metadata.get("paragraph_id") or metadata.get("pdf_page") or metadata.get("page"),
            metadata.get("printed_page") or metadata.get("citation_page"),
            clean_text(doc.page_content, "")[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
        if len(merged) >= limit:
            break
    return merged


def filter_paragraph_docs_by_text_overlap(query, docs, limit=None):
    filtered = []
    for doc in docs or []:
        score = page_match_score(query, doc.page_content)
        if score <= 0:
            continue
        doc.metadata["paragraph_query_overlap_score"] = score
        filtered.append(doc)
    filtered.sort(key=lambda item: item.metadata.get("paragraph_query_overlap_score", 0), reverse=True)
    return filtered[:limit] if limit else filtered


UNSUPPORTED_CLAIM_RULES = [
    {
        "tokens": ["\u4e24\u4e2a\u7ed3\u5408"],
        "answer": (
            "\u5f53\u524d\u6750\u6599\u4e0d\u652f\u6301\u628a\u201c\u4e24\u4e2a\u7ed3\u5408\u201d\u5224\u5b9a\u4e3a\u9a6c\u514b\u601d\u539f\u8457\u4e2d\u63d0\u51fa\u7684\u6982\u5ff5\u3002"
            "\u8fd9\u662f\u540e\u6765\u4e2d\u56fd\u5316\u65f6\u4ee3\u5316\u9a6c\u514b\u601d\u4e3b\u4e49\u8bed\u5883\u4e2d\u7684\u8868\u8ff0\uff0c\u4e0d\u5e94\u7f16\u9020\u4e3a\u9a6c\u514b\u601d\u67d0\u90e8\u8457\u4f5c\u7684\u539f\u6587\u51fa\u5904\u3002"
        ),
    },
    {
        "tokens": ["\u793e\u4f1a\u4e3b\u4e49\u6838\u5fc3\u4ef7\u503c\u89c2"],
        "answer": (
            "\u4e0d\u662f\u3002\u201c\u793e\u4f1a\u4e3b\u4e49\u6838\u5fc3\u4ef7\u503c\u89c2\u201d\u4e0d\u662f\u300a\u8d44\u672c\u8bba\u300b\u4e2d\u63d0\u51fa\u7684\u539f\u6587\u6982\u5ff5\u3002"
            "\u56e0\u6b64\u4e0d\u5e94\u4e3a\u5b83\u865a\u6784\u300a\u8d44\u672c\u8bba\u300b\u5377\u518c\u3001\u7ae0\u8282\u6216\u9875\u7801\u3002"
        ),
    },
    {
        "tokens": ["\u65b0\u8d28\u751f\u4ea7\u529b"],
        "answer": (
            "\u5f53\u524d\u6750\u6599\u4e0d\u652f\u6301\u201c\u65b0\u8d28\u751f\u4ea7\u529b\u201d\u51fa\u81ea\u9a6c\u514b\u601d\u67d0\u90e8\u539f\u8457\u3002"
            "\u8fd9\u662f\u5f53\u4ee3\u7406\u8bba\u8bed\u5883\u4e2d\u7684\u6982\u5ff5\uff0c\u4e0d\u80fd\u76f4\u63a5\u5f52\u4e3a\u9a6c\u514b\u601d\u539f\u6587\u3002"
        ),
    },
    {
        "tokens": ["\u5b9e\u8df5\u662f\u68c0\u9a8c\u771f\u7406\u7684\u552f\u4e00\u6807\u51c6"],
        "answer": (
            "\u8fd9\u4e0d\u662f\u9a6c\u514b\u601d\u539f\u8457\u4e2d\u7684\u76f4\u63a5\u539f\u8bdd\u3002"
            "\u5b83\u4e0e\u9a6c\u514b\u601d\u5173\u4e8e\u5b9e\u8df5\u548c\u771f\u7406\u7684\u601d\u60f3\u6709\u5173\uff0c\u4f46\u4e0d\u80fd\u5f53\u4f5c\u9a6c\u514b\u601d\u7684\u9010\u5b57\u5f15\u6587\u6765\u6807\u6ce8\u51fa\u5904\u3002"
        ),
    },
    {
        "tokens": ["\u4ee5\u4eba\u6c11\u4e3a\u4e2d\u5fc3"],
        "answer": (
            "\u201c\u4ee5\u4eba\u6c11\u4e3a\u4e2d\u5fc3\u201d\u4e0d\u662f\u9a6c\u514b\u601d\u539f\u8457\u4e2d\u7684\u539f\u6587\u8868\u8fbe\u3002"
            "\u56de\u7b54\u8fd9\u7c7b\u95ee\u9898\u65f6\u53ef\u4ee5\u8bf4\u660e\u5176\u4e0e\u9a6c\u514b\u601d\u4e3b\u4e49\u4eba\u6c11\u7acb\u573a\u6709\u601d\u60f3\u5173\u8054\uff0c\u4f46\u4e0d\u5e94\u7f16\u9020\u6210\u539f\u8457\u539f\u53e5\u3002"
        ),
    },
]


def answer_unsupported_claim(query):
    normalized_query = normalize_for_match(query)
    for rule in UNSUPPORTED_CLAIM_RULES:
        if all(normalize_for_match(token) in normalized_query for token in rule["tokens"]):
            return rule["answer"]
    return ""



def evidence_from_doc(doc, index=1):
    metadata = normalize_metadata(doc.metadata)
    content = clean_text(doc.page_content, "")
    return {
        "id": f"E{index}",
        "citation": format_citation(metadata, include_article=False),
        "detailed_citation": format_citation(metadata, include_article=True),
        "sentence_citation": format_citation(metadata, include_article=False),
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
        "excerpt": compact_preview(content, limit=240) if "compact_preview" in globals() else content[:240],
    }


def evidence_from_docs(docs, limit=12):
    evidence = []
    seen = set()
    for doc in docs[:limit]:
        item = evidence_from_doc(doc, index=len(evidence) + 1)
        key = (item.get("source"), item.get("printed_page"), item.get("citation_page"), item.get("article"), item.get("excerpt")[:80])
        if key in seen:
            continue
        seen.add(key)
        evidence.append(item)
    return evidence


def extract_answer_citation_lines(answer):
    normalized = normalize_final_answer(answer)
    citations = []
    for line in normalized.splitlines():
        match = re.match(r"\s*(?:\d+[\.\u3001]|\(\d+\))\s*(.+?\u7b2c\d+\u9875\u3002?)\s*$", line)
        if match:
            citations.append(match.group(1).strip())
    return citations


def citation_match_key(citation):
    return normalize_for_match(citation or "")


def evidence_matches_citation(item, citation):
    citation_key = citation_match_key(citation)
    if not citation_key:
        return False

    candidates = [
        item.get("citation"),
        item.get("sentence_citation"),
        item.get("detailed_citation"),
    ]
    for candidate in candidates:
        candidate_key = citation_match_key(candidate)
        if candidate_key and (candidate_key in citation_key or citation_key in candidate_key):
            return True

    page_match = re.search(r"\u7b2c(\d+)\u9875", citation or "")
    citation_page = str(page_match.group(1)) if page_match else ""
    item_pages = {
        str(item.get("printed_page") or ""),
        str(item.get("citation_page") or ""),
    }
    item_pages.discard("")
    if citation_page and citation_page in item_pages:
        series = citation_match_key(item.get("series") or "")
        source = citation_match_key(item.get("source") or item.get("source_file") or "")
        if (series and series in citation_key) or (source and source in citation_key):
            return True

    return False


def filter_evidence_to_answer(answer, evidence, fallback_limit=3):
    evidence = evidence or []
    citations = extract_answer_citation_lines(answer)
    if not citations:
        return [
            {**item, "id": f"E{index}"}
            for index, item in enumerate(evidence[:fallback_limit], start=1)
        ]

    matched = []
    seen = set()
    for citation in citations:
        for item in evidence:
            if not evidence_matches_citation(item, citation):
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

    return [
        {**item, "id": f"E{index}"}
        for index, item in enumerate(matched, start=1)
    ]


def audit_answer_citations(answer, evidence):
    normalized = normalize_final_answer(answer)
    issues = []
    forbidden = ["PDF\u7b2c", "pdf_page", "PDF page", "\u540c\u4e0a"]
    for token in forbidden:
        if token in normalized:
            issues.append({"type": "forbidden_token", "token": token})

    evidence_citations = {item.get("citation") for item in evidence or []}
    evidence_citations |= {item.get("sentence_citation") for item in evidence or []}
    evidence_citations = {item for item in evidence_citations if item}
    citation_lines = extract_answer_citation_lines(normalized)

    if citation_lines and not evidence_citations:
        issues.append({"type": "citation_without_verified_evidence"})

    for citation in citation_lines:
        if evidence_citations and not any(evidence_matches_citation(item, citation) for item in evidence or []):
            issues.append({"type": "citation_not_in_evidence", "citation": citation})

    return {
        "ok": not issues,
        "issues": issues,
        "evidence_count": len(evidence or []),
        "answer": normalized,
    }


def set_last_evidence(evidence=None, audit=None):
    global LAST_EVIDENCE, LAST_CITATION_AUDIT
    LAST_EVIDENCE = evidence or []
    LAST_CITATION_AUDIT = audit or {"ok": True, "issues": [], "evidence_count": len(LAST_EVIDENCE)}


def normalize_final_answer(answer):
    answer = clean_text(answer, "")
    answer = answer.replace("PDF\u7b2c", "\u7b2c").replace("PDF?", "?")
    answer = answer.replace("pdf_page", "page")

    lines = answer.splitlines()
    normalized_lines = []
    previous_citation = ""
    citation_line_re = re.compile(r"^(\s*\d+[\.\u3001]\s*)(.+)$")

    for line in lines:
        match = citation_line_re.match(line)
        if match:
            prefix, body = match.groups()
            stripped_body = body.strip().strip("\u3002")
            if stripped_body in {"\u540c\u4e0a", "\u53c2\u89c1\u540c\u4e0a"} and previous_citation:
                line = prefix + previous_citation
            elif stripped_body and stripped_body not in {"\u540c\u4e0a", "\u53c2\u89c1\u540c\u4e0a"}:
                previous_citation = body.strip()
        normalized_lines.append(line)

    return "\n".join(normalized_lines)

def run_query(query, route_query=None):
    set_last_evidence([])
    query = clean_text(query, "")
    route_query = clean_text(route_query or query, "")
    if is_unreadable_query(route_query):
        return (
            "\u672a\u80fd\u8bfb\u53d6\u5230\u53ef\u7528\u7684\u4e2d\u6587\u95ee\u9898\u3002"
            "\u5982\u679c\u662f\u5728 PowerShell \u4e2d\u901a\u8fc7\u7ba1\u9053\u6216\u91cd\u5b9a\u5411\u8f93\u5165\uff0c"
            "\u8bf7\u5148\u8fd0\u884c `chcp 65001`\uff0c\u6216\u5728\u4ea4\u4e92\u5f0f\u63d0\u793a\u4e2d\u76f4\u63a5\u8f93\u5165\u95ee\u9898\u3002"
        )

    unsupported_answer = answer_unsupported_claim(route_query)
    if unsupported_answer:
        return unsupported_answer

    query_intent = classify_query(route_query)
    if query != route_query and query_intent == "quote_lookup" and not re.search(r"[“\"].+[”\"]", route_query):
        query_intent = "rag_answer"
    trace = trace_enabled()
    trace_only = trace_only_enabled()
    dual_retrieval = dual_retrieval_enabled()

    if trace or trace_only:
        print_query_trace(route_query, query_intent)

    if query_intent == "bibliographic_lookup":
        bibliographic_answer = answer_bibliographic_query(route_query)
        if trace or trace_only:
            print_trace_line("search_path: local article map / core classics")
            print_trace_line(f"bibliographic_answer_found: {bool(bibliographic_answer)}")
            print_trace_line("===== End Trace =====\n")
        if bibliographic_answer:
            return bibliographic_answer

        title = extract_bibliographic_title(route_query)
        return f"未能在当前核心书目表中确认《{title}》。"

    if query_intent == "quote_lookup":
        if trace or trace_only:
            print_trace_line("search_path: exact OCR quote lookup")
        answer = answer_quote_query(query, trace=trace or trace_only)
        if trace or trace_only:
            print_trace_line("===== End Trace =====\n")
        return answer

    constraints = constraints_from_query(route_query)
    if trace or trace_only:
        print_trace_line("search_path: FAISS vector similarity search -> rule rerank -> DeepSeek")
        print_constraints_trace(constraints)
    db = load_vectorstore()
    retrieve_k = 12 if query_intent == "rag_answer" else 5
    docs = retrieve_documents(query, db, k=retrieve_k)
    paragraph_docs_for_answer = []
    if paragraph_vectorstore_exists():
        paragraph_db_for_answer = load_paragraph_vectorstore()
        paragraph_docs_for_answer = filter_paragraph_docs_by_text_overlap(
            query,
            retrieve_documents(query, paragraph_db_for_answer, k=max(retrieve_k * 3, 12)),
            limit=retrieve_k,
        )
        docs = merge_prefer_paragraph_docs(paragraph_docs_for_answer, docs, retrieve_k)
    docs = refine_docs_citation_pages_for_query(docs, route_query)
    if constraints.get("strict_title") and not docs:
        title = constraints.get("title") or "\u8be5\u6587"
        answer = (
            f"\u5f53\u524d\u8bed\u6599\u5e93\u672a\u68c0\u7d22\u5230\u300a{title}\u300b\u7684\u6b63\u6587\u9875\u6bb5\uff0c\u56e0\u6b64\u672c\u8f6e\u4e0d\u8f93\u51fa\u8de8\u7bc7\u66ff\u4ee3\u6027\u5f15\u6587\u3002"
            "\u8bf7\u5148\u8865\u9f50\u8be5\u6587\u5728\u672c\u5730\u5e93\u4e2d\u7684\u9875\u6bb5\u6620\u5c04\u6216OCR\u6587\u672c\u540e\u518d\u56de\u7b54\u3002"
        )
        set_last_evidence([], {"ok": True, "issues": [], "evidence_count": 0, "answer": answer})
        return answer
    paragraph_docs = []
    if (trace or trace_only) and dual_retrieval:
        if paragraph_vectorstore_exists():
            paragraph_docs = paragraph_docs_for_answer[:5]
        else:
            print_trace_line(f"paragraph_vectorstore_missing: {PARAGRAPH_VECTORSTORE_DIR}")
    evidence = evidence_from_docs(docs)
    set_last_evidence([])
    context = build_context(docs, query_intent)
    prompt = clean_text(build_prompt(query_intent, query, context) + build_constraint_guard(constraints))
    if trace or trace_only:
        print_docs_trace(docs)
        if dual_retrieval and paragraph_docs:
            print_docs_trace(paragraph_docs, label="paragraph_retrieved_docs")
        print_prompt_trace(prompt)

    if trace_only:
        return build_trace_only_answer(query_intent, docs, prompt, paragraph_docs=paragraph_docs)

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

    display_evidence = filter_evidence_to_answer(response.choices[0].message.content, evidence)
    audit = audit_answer_citations(response.choices[0].message.content, display_evidence)
    set_last_evidence(display_evidence, audit)
    return audit["answer"]


def main():
    query = input("请输入问题：")
    answer = run_query(query)
    print("\n===== MarxOS =====\n")
    print(answer)


if __name__ == "__main__":
    main()
