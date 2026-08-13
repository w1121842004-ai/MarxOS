from openai import OpenAI

from langchain_core.documents import Document
from dotenv import load_dotenv
import json
import os

# macOS ARM: torch and Milvus Lite (whose HNSW index is FAISS-backed) share the
# same libomp runtime. Spawning OpenMP worker threads in that combination
# segfaults at search time (__kmp_suspend_initialize_thread). Pin to one thread
# BEFORE any faiss/torch import so libomp initializes in single-thread mode.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import re
import sys
import time
from pathlib import Path
import retrieval as retrieval_utils
from marxos.config import get_settings
from marxos import ambiguous as ambiguous_utils
from marxos import phoenix
from marxos import query_intent
from marxos import query_planner
from marxos import trace as trace_utils
from marxos.data.loaders import load_merged_article_map, load_topic_catalog as load_topic_catalog_data
from marxos.book_locator import BookLocator
from marxos.generation import answers as answer_utils
from marxos.generation import citations
from marxos.generation import prompts
from marxos.generation.citation_audit import CitationVerifier
from marxos.generation.llm_client import (
    create_deepseek_client,
    deepseek_extra_body,
    deepseek_flash_model,
    deepseek_model,
    generation_model,
)
from marxos.app import orchestration
from marxos.app.runtime import RuntimeState
from marxos.work_catalog import WorkCatalog
from marxos.relevance_classifier import is_marxism_relevant
from rag.core_classics import classic_entries_for_query, load_core_classics
from rag.exact_quote_lookup import exact_quote_lookup, extract_query_quote
from rag.semantic_retrieval import expand_semantic_parent_docs as expand_semantic_parent_windows
from rag.semantic_retrieval import (
    load_sparse_paragraph_index,
    sparse_retrieve_documents as sparse_parent_retrieval,
)


load_dotenv()
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
SETTINGS = get_settings()


EMBEDDING_MODEL = SETTINGS.models.embedding_model
VECTORSTORE_DIR = SETTINGS.index.vectorstore_dir
PARAGRAPH_VECTORSTORE_DIR = SETTINGS.index.paragraph_vectorstore_dir
VECTOR_BACKEND_ENV = SETTINGS.retrieval.vector_backend_env
MILVUS_URI = SETTINGS.index.milvus_uri
MILVUS_COLLECTION = SETTINGS.index.milvus_collection
MILVUS_EMBEDDING_DEVICE = SETTINGS.models.embedding_device
PARAGRAPH_CACHE_PATH = SETTINGS.corpus.paragraph_cache_path
SEMANTIC_PARENT_CACHE_PATH = SETTINGS.corpus.semantic_parent_cache_path
OCR_CACHE_DIR = SETTINGS.corpus.ocr_cache_dir
PAGE_MAP_PATH = SETTINGS.corpus.page_map_path
SEMANTIC_PARENT_WINDOW = SETTINGS.retrieval.semantic_parent_window
SEMANTIC_CHILD_PARENT_WINDOW = SETTINGS.retrieval.semantic_child_parent_window
LAST_EVIDENCE = []
LAST_CITATION_AUDIT = {}
LAST_TOPIC_INFO = {}
LAST_CRAG_REPORT = {}
LAST_TIMING = {}
# Answer class for the current turn: local_lookup / local_view / refusal /
# llm / out_of_domain / ambiguous_locator / trace_only.
LAST_ANSWER_PATH = ""
ARTICLE_MAP_PATH = SETTINGS.corpus.article_map_path
ARTICLE_MAP_EXTRA_PATHS = SETTINGS.corpus.article_map_extra_paths
TOPIC_CATALOG_PATH = SETTINGS.corpus.topic_catalog_path
DEFAULT_PUBLISHER = "人民出版社"
RERANK_DEBUG_ENV = SETTINGS.retrieval.rerank_debug_env
TRACE_ENV = SETTINGS.trace_env
TRACE_ONLY_ENV = SETTINGS.trace_only_env
DUAL_RETRIEVAL_ENV = SETTINGS.retrieval.dual_retrieval_env
HYBRID_RETRIEVAL_ENV = SETTINGS.retrieval.hybrid_retrieval_env
DEV_MODE_ENV = SETTINGS.dev_mode_env
DEV_TOKEN_ENV = SETTINGS.dev_token_env
DEV_TOKEN_INPUT_ENV = SETTINGS.dev_token_input_env
RUNTIME = RuntimeState(
    embedding_model=EMBEDDING_MODEL,
    vectorstore_dir=VECTORSTORE_DIR,
    paragraph_vectorstore_dir=PARAGRAPH_VECTORSTORE_DIR,
    dev_mode_env=DEV_MODE_ENV,
    dev_token_env=DEV_TOKEN_ENV,
    dev_token_input_env=DEV_TOKEN_INPUT_ENV,
    trace_env=TRACE_ENV,
    trace_only_env=TRACE_ONLY_ENV,
    dual_retrieval_env=DUAL_RETRIEVAL_ENV,
    vector_backend_env=VECTOR_BACKEND_ENV,
    vector_backend_default=SETTINGS.retrieval.vector_backend_default,
    milvus_uri=MILVUS_URI,
    milvus_collection=MILVUS_COLLECTION,
    milvus_embedding_device=MILVUS_EMBEDDING_DEVICE,
    milvus_sparse_provider=SETTINGS.index.milvus_sparse_provider,
    milvus_bm25_stats_path=SETTINGS.index.milvus_bm25_stats_path,
)
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
    return load_merged_article_map(ARTICLE_MAP_PATH, ARTICLE_MAP_EXTRA_PATHS)


ARTICLE_MAP = load_article_map()


def load_topic_catalog():
    return load_topic_catalog_data(TOPIC_CATALOG_PATH)


TOPIC_CATALOG = load_topic_catalog()


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
    cache_base = os.path.join(
        OCR_CACHE_DIR,
        source_stem({"source": source}),
    )
    cache_path = os.path.join(cache_base, f"page_{pdf_page}.json")
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return repair_mojibake(payload.get("raw_text") or payload.get("cleaned_text") or "")
        txt_path = os.path.join(cache_base, f"page_{pdf_page}.txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                return repair_mojibake(f.read())
    except (OSError, json.JSONDecodeError):
        return ""
    return ""


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

    cache_base = os.path.join(
        OCR_CACHE_DIR,
        source_stem({"source": source}),
    )
    try:
        cache_path = os.path.join(cache_base, f"page_{pdf_page}.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            raw_text = repair_mojibake(payload.get("raw_text") or payload.get("cleaned_text") or "")
        else:
            txt_path = os.path.join(cache_base, f"page_{pdf_page}.txt")
            if not os.path.exists(txt_path):
                return None
            with open(txt_path, "r", encoding="utf-8") as f:
                raw_text = repair_mojibake(f.read())
    except (OSError, json.JSONDecodeError):
        return None
    lines = [normalize_digit_text(line).strip() for line in str(raw_text).splitlines()]
    lines = [line for line in lines if line]
    if len(lines) == 1:
        line = lines[0]
        edge_lines = [line[-360:], line[:360]]
    else:
        edge_lines = lines[-3:] + lines[:3]

    candidates = []
    for line in edge_lines:
        matches = list(re.finditer(r"(?<![0-9A-Za-z])(\d{1,4})(?![0-9A-Za-z])", line))
        for match in matches:
            page = as_int(match.group(1))
            if page is None or page <= 0:
                continue
            # Printed pages usually trail PDF pages by the front-matter offset.
            if -80 <= pdf_page - page <= 220:
                candidates.append(page)

    if not candidates:
        return None

    return candidates[0]


def find_pdf_page_by_printed_page(source, printed_page):
    source_dir = Path(OCR_CACHE_DIR) / source.replace(".pdf", "")
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


def clean_article_title(title):
    title = clean_text(title, "")
    title = re.split(r"[.\u2026•·]{3,}", title, maxsplit=1)[0]
    title = re.sub(r"^[*•·.\s]+", "", title)
    title = re.sub(r"[.·•\]\)）\s]+$", "", title)
    title = title.strip("“”\"'《》[]【】()（）")
    return title


TOPIC_TITLE_REWRITES = {
    "弗恩格斯法德农民问题": "法德农民问题",
    "弗恩格斯德国农民战争": "德国农民战争",
    "弗恩格斯关于普鲁士农民的历史": "关于普鲁士农民的历史",
    "卡马克思论土地国有化": "论土地国有化",
    "对农村居民土地的剥夺": "对农村居民土地的剥夺",
    "分成制和农民的小块土地所有制": "分成制和农民的小块土地所有制",
}


# ── Work Catalog ──────────────────────────────────────────────────

_work_catalog = None

def _get_work_catalog():
    global _work_catalog
    if _work_catalog is None:
        _work_catalog = WorkCatalog()
    return _work_catalog


def work_catalog_entries_for_query(query):
    """Match query to work_catalog and return constraint entries."""
    catalog = _get_work_catalog()
    work = catalog.match_query(query, normalize_fn=normalize_for_match)
    if work is None:
        concept_hits = catalog.match_by_concepts(query, normalize_fn=normalize_for_match)
        if concept_hits:
            work = concept_hits[0][0]
    if work is None:
        return []
    return catalog.get_entries(work)


def work_catalog_title_entries_for_query(query):
    """Return work catalog entries only for explicit title or alias mentions."""
    catalog = _get_work_catalog()
    work = catalog.match_title_query(query, normalize_fn=normalize_for_match)
    if work is None:
        return []
    return catalog.get_entries(work)


def work_catalog_title_mentioned(query):
    """True when a full work title (not just an alias) appears verbatim."""
    return _get_work_catalog().has_explicit_title_mention(query, normalize_fn=normalize_for_match)


# ── Book Locator Agent ────────────────────────────────────────────

_book_locator = None

def _get_book_locator():
    global _book_locator
    if _book_locator is None:
        client = create_deepseek_client(max_retries=2, timeout=30.0)
        catalog = _get_work_catalog()
        _book_locator = BookLocator(client, catalog, model=deepseek_flash_model())
    return _book_locator


def book_locator_constraints(query):
    """LLM-driven fallback: when rule-based matching fails, ask DeepSeek."""
    if trace_only_enabled():
        return {}
    locator = _get_book_locator()
    result = locator.get_constraints(query)
    return result if result else {}


# ── Citation Verifier ─────────────────────────────────────────────

_citation_verifier = None

def _get_citation_verifier():
    global _citation_verifier
    if _citation_verifier is None:
        client = create_deepseek_client(max_retries=2, timeout=30.0)
        _citation_verifier = CitationVerifier(client, OCR_CACHE_DIR, model=deepseek_flash_model())
    return _citation_verifier


def verify_citations(answer_text, evidence_cards):
    """Content-level citation verification against OCR text."""
    if trace_only_enabled():
        return None
    verifier = _get_citation_verifier()
    return verifier.verify(answer_text, evidence_cards)


def _retrieval_ctx():
    return {
        "TOPIC_CATALOG": TOPIC_CATALOG,
        "WORK_TITLE_ALIASES": WORK_TITLE_ALIASES,
        "re": re,
        "CONCEPT_CANONICAL_CLASSIC_IDS": CONCEPT_CANONICAL_CLASSIC_IDS,
        "CONCEPT_PREFERRED_MARKERS": CONCEPT_PREFERRED_MARKERS,
        "CONCEPT_PREFERRED_SOURCES": CONCEPT_PREFERRED_SOURCES,
        "OCR_CACHE_DIR": OCR_CACHE_DIR,
        "RERANK_DEBUG_ENV": RERANK_DEBUG_ENV,
        "CLASSIC_SAYING_QUOTE_SEEDS": CLASSIC_SAYING_QUOTE_SEEDS,
        "CLASSIC_SAYING_QUERY_SEEDS": CLASSIC_SAYING_QUERY_SEEDS,
        "normalize_topic_title": normalize_topic_title,
        "normalize_digit_text": normalize_digit_text,
        "normalize_for_match": normalize_for_match,
        "clean_article_title": clean_article_title,
        "clean_text": clean_text,
        "find_toc_entries": find_toc_entries,
        "extract_bibliographic_title": extract_bibliographic_title,
        "work_catalog_entries_for_query": work_catalog_entries_for_query,
        "work_catalog_title_entries_for_query": work_catalog_title_entries_for_query,
        "work_catalog_title_mentioned": work_catalog_title_mentioned,
        "book_locator_constraints": book_locator_constraints,
        "locator_entries_for_query": locator_entries_for_query,
        "classic_entries_for_query": classic_entries_for_query,
        "enrich_core_classic_entries": enrich_core_classic_entries,
        "active_concept_terms": active_concept_terms,
        "core_classic_by_id": core_classic_by_id,
        "metadata_citation_page": metadata_citation_page,
        "as_int": as_int,
        "exact_quote_lookup": exact_quote_lookup,
        "is_quote_lookup_query": is_quote_lookup_query,
        "classify_query": classify_query,
        "enrich_concept_metadata": enrich_concept_metadata,
        "find_pdf_page_by_printed_page": find_pdf_page_by_printed_page,
        "load_ocr_page_text": load_ocr_page_text,
        "page_match_score": page_match_score,
        "infer_printed_page_from_ocr_cache": infer_printed_page_from_ocr_cache,
        "score_concept_focus": score_concept_focus,
        "score_concept_source_priority": score_concept_source_priority,
        "score_document_quality": score_document_quality,
        "is_front_matter_candidate": is_front_matter_candidate,
        "requests_derivative_material": requests_derivative_material,
        "is_classic_sayings_query": is_classic_sayings_query,
        "is_noisy_article_title": is_noisy_article_title,
        "controlled_multi_queries": controlled_multi_queries,
        "expand_semantic_parent_docs": expand_semantic_parent_docs,
        "hybrid_retrieval_enabled": hybrid_retrieval_enabled,
        "sparse_index_ready": sparse_index_ready,
        "sparse_retrieve_documents": sparse_retrieve_documents,
    }


def expand_semantic_parent_docs(docs):
    uses_semantic_child = any(
        (doc.metadata or {}).get("retrieval_unit") == "semantic_child"
        for doc in docs or []
    )
    return expand_semantic_parent_windows(
        docs,
        window=SEMANTIC_CHILD_PARENT_WINDOW if uses_semantic_child else SEMANTIC_PARENT_WINDOW,
        path=SEMANTIC_PARENT_CACHE_PATH if uses_semantic_child else PARAGRAPH_CACHE_PATH,
    )


def hybrid_retrieval_enabled():
    if (
        RUNTIME.vector_backend() == "milvus"
        and os.getenv(
            "MILVUS_HYBRID_SEARCH",
            os.getenv("MARXOS_MILVUS_HYBRID", "1" if SETTINGS.index.milvus_hybrid_search else "0"),
        ).lower()
        in {"1", "true", "yes", "on"}
        and os.getenv("MARXOS_ENABLE_LOCAL_HYBRID_WITH_MILVUS", "0").lower()
        not in {"1", "true", "yes", "on"}
    ):
        return False
    value = os.getenv(HYBRID_RETRIEVAL_ENV, "")
    if not value:
        return True
    return value.lower() in {"1", "true", "yes", "on"}


def sparse_index_ready():
    return load_sparse_paragraph_index.cache_info().currsize > 0


def warm_sparse_index():
    return load_sparse_paragraph_index(PARAGRAPH_CACHE_PATH)


def sparse_retrieve_documents(query, limit=24):
    return sparse_parent_retrieval(
        query,
        limit=limit,
        path=PARAGRAPH_CACHE_PATH,
    )


def normalize_topic_title(title):
    cleaned = clean_article_title(title)
    if not cleaned:
        return ""

    normalized = normalize_for_match(cleaned)
    if normalized in TOPIC_TITLE_REWRITES:
        return TOPIC_TITLE_REWRITES[normalized]

    cleaned = re.sub(r"^[0-9IVXivx一二三四五六七八九十]+[.．、)\s]+", "", cleaned)
    cleaned = re.sub(r"^(卡·马克思|弗·恩格斯)", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


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


DERIVATIVE_TITLE_MARKERS = [
    "序言",
    "导言",
    "前言",
    "后记",
    "跋",
    "附录",
    "编者注",
    "译者注",
    "译后记",
    "出版说明",
    "凡例",
    "说明",
    "绪言",
]

FRONT_MATTER_LEAD_MARKERS = [
    "本版序言",
    "德文版序言",
    "俄文版序言",
    "英文版序言",
    "法文版序言",
    "波兰文版序言",
    "意大利文版序言",
    "译后记",
    "编者注",
    "译者注",
    "出版说明",
    "凡例",
    "本卷的开篇是",
    "在这个马克思主义纲领性文献中",
]


def requests_derivative_material(query, constraints=None):
    haystacks = [clean_text(query, "")]
    constraints = constraints or {}
    haystacks.extend(
        clean_text(constraints.get(key), "")
        for key in ("title", "classic_title", "locator_title")
    )
    normalized = normalize_for_match(" ".join(item for item in haystacks if item))
    if not normalized:
        return False
    return any(normalize_for_match(marker) in normalized for marker in DERIVATIVE_TITLE_MARKERS)


def is_front_matter_candidate(metadata, content, constraints=None):
    if requests_derivative_material("", constraints):
        return False

    metadata = metadata or {}
    title_fields = [
        metadata.get("raw_article"),
        metadata.get("raw_section"),
        metadata.get("section"),
        metadata.get("article"),
        metadata.get("locator_title"),
    ]
    title_norm = normalize_for_match(" ".join(clean_text(item, "") for item in title_fields if item))
    if any(normalize_for_match(marker) in title_norm for marker in DERIVATIVE_TITLE_MARKERS):
        return True

    lead = clean_text(content, "")[:260]
    lead_norm = normalize_for_match(lead)
    if any(normalize_for_match(marker) in lead_norm for marker in FRONT_MATTER_LEAD_MARKERS):
        return True

    if re.search(r"(18|19)\d{2}年.{0,12}(版)?序言", lead):
        return True

    if "——编者注" in lead or "———编者注" in lead:
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
    # Prefer work_catalog title over OCR article_map when available
    work_title = normalized.get("classic_title") or normalized.get("locator_title") or normalized.get("work_title")
    has_work_title = bool(work_title and len(clean_article_title(work_title)) >= 4)
    if mapped_article and should_fill_article_from_map(normalized) and not has_work_title:
        normalized["article"] = mapped_article
        if not normalized.get("section") or normalized.get("section") == normalized.get("book"):
            normalized["section"] = mapped_article
    elif has_work_title and (not normalized.get("article") or is_noisy_article_title(clean_article_title(normalized.get("article","")))):
        normalized["article"] = work_title
        normalized["section"] = work_title

    for key in ["article", "section"]:
        if key not in normalized:
            continue

        cleaned_title = clean_article_title(normalized.get(key))
        if is_noisy_article_title(cleaned_title):
            normalized[f"raw_{key}"] = normalized.get(key)
            normalized[key] = None
        elif cleaned_title:
            normalized[key] = cleaned_title

    # Final override: work_catalog title always wins over OCR article_map
    work_title = normalized.get("classic_title") or normalized.get("locator_title") or normalized.get("work_title")
    if work_title:
        clean_work = clean_article_title(work_title)
        current_article = clean_article_title(normalized.get("article") or "")
        current_section = clean_article_title(normalized.get("section") or "")
        if clean_work and (not current_article or len(current_article) < 4 or
                           current_article == "实践" or  # known bad metadata
                           is_noisy_article_title(current_article)):
            normalized["article"] = work_title
        if clean_work and (not current_section or len(current_section) < 4 or
                           current_section == "实践" or
                           is_noisy_article_title(current_section)):
            normalized["section"] = work_title

    if normalized.get("citation_page") is None:
        if normalized.get("printed_page") is not None:
            normalized["citation_page"] = normalized.get("printed_page")
            normalized.setdefault("citation_page_type", "printed_page")
        elif normalized.get("pdf_page") is not None:
            normalized["citation_page"] = normalized.get("pdf_page")
            normalized.setdefault("citation_page_type", "pdf_page")

    return normalized


def citation_page_label(metadata):
    return citations.citation_page_label(metadata, normalize_metadata, clean_text)


def source_page_label(metadata):
    return citations.source_page_label(metadata, normalize_metadata, clean_text)

def format_citation(metadata, include_article=False):
    return citations.format_citation(
        metadata,
        include_article,
        normalize_metadata,
        normalize_book_parts,
        clean_text,
    )


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
    ) or bool(active_concept_terms(query))


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


# Delegate the public routing helpers to the extracted query-intent module
# while keeping the same app.py call surface for tests and callers.
def extract_quoted_title(query):
    return query_intent.extract_quoted_title(query, clean_text)


def extract_unquoted_title(query):
    return query_intent.extract_unquoted_title(query, clean_text)


def extract_bibliographic_title(query):
    return query_intent.extract_bibliographic_title(query, clean_text)


def normalize_for_match(text):
    return query_intent.normalize_for_match(text, clean_text)


def is_bibliographic_query(query):
    return query_intent.is_bibliographic_query(query, clean_text)


def is_quote_lookup_query(query):
    return query_intent.is_quote_lookup_query(query, clean_text)


def is_analysis_query(query):
    return query_intent.is_analysis_query(query, clean_text)


def is_classic_sayings_query(query):
    return query_intent.is_classic_sayings_query(query, clean_text)


def classify_query(query):
    """Classify a user query with work_catalog-aware routing (v2 scoring).

    Uses the layered-scoring engine from ``marxos.query_intent.classify_query_v2``
    enriched with work_catalog match signals.  Returns a plain string for
    backward compatibility — call ``classify_query_v2()`` for the rich
    ``IntentResult`` with confidence / ambiguity signals.

    Intents: bibliographic_lookup | quote_lookup | concept_explain |
             comparison | deep_analysis | theory_analysis | rag_answer
    """
    if orchestration.is_chitchat_query(query):
        return "chitchat"

    # ── Relevance gate: skip RAG for non-Marxism queries ──
    if not is_marxism_relevant(query):
        return "out_of_domain"

    if is_quote_lookup_query(query) and extract_query_quote(query) and not extract_bibliographic_title(query):
        return "quote_lookup"

    v2 = query_intent.classify_query_v2(query, clean_text)

    # Boost bibliographic / quote confidence when work_catalog confirms a match
    if v2.primary in ("bibliographic_lookup", "quote_lookup"):
        catalog = _get_work_catalog()
        if catalog.match_query(query, normalize_fn=normalize_for_match):
            return v2.primary
        # Degraded: no catalog match → fall through to secondary intent
        if v2.is_ambiguous:
            secondary = v2.secondary_intents(threshold=0.18)
            if secondary:
                return secondary[0][0]

    # Boost concept when concept terms are active
    if v2.primary == "concept_explain" and is_concept_query(query):
        return "concept_explain"

    return v2.primary


def classify_query_v2(query):
    """Rich intent classification with confidence / ambiguity signals.

    Returns ``marxos.query_intent.IntentResult``.  The object compares
    equal to its primary intent string (e.g. ``result == "rag_answer"``)
    for drop-in backward compatibility.
    """
    return query_intent.classify_query_v2(query, clean_text)


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
        key=lambda item: (toc_source_priority(item["source"]), item["source"], item["start_page"], item["end_page"]),
    )


def toc_source_priority(source):
    source = str(source or "").lower()
    if re.fullmatch(r"me\d{2}[abc]?\.pdf", source):
        return 0
    if source.startswith("mea"):
        return 1
    if source.startswith("mes"):
        return 2
    return 3


def requested_capital_volume(title):
    normalized = normalize_for_match(title)
    if "资本论" not in normalized:
        return ""
    volume_markers = {
        "第一卷": "第一卷",
        "第1卷": "第一卷",
        "一卷": "第一卷",
        "第二卷": "第二卷",
        "第2卷": "第二卷",
        "二卷": "第二卷",
        "第三卷": "第三卷",
        "第3卷": "第三卷",
        "三卷": "第三卷",
    }
    for marker, canonical in volume_markers.items():
        if normalize_for_match(marker) in normalized:
            return canonical
    return ""


def filter_capital_volume_entries(title, entries):
    requested_volume = requested_capital_volume(title)
    if not requested_volume:
        return entries
    requested_norm = normalize_for_match(requested_volume)
    return [
        entry for entry in entries
        if requested_norm in normalize_for_match(entry.get("classic_title") or entry.get("article") or "")
    ]


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
        exact_me_entries = [
            entry for entry in exact_entries
            if toc_source_priority(entry.get("source")) == 0
        ]
        any_me_entries = any(toc_source_priority(entry.get("source")) == 0 for entry in entries)
        if exact_me_entries:
            entries = exact_me_entries
        elif not any_me_entries:
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

    map_entries = filter_capital_volume_entries(title, best_toc_entries(entries))
    if map_entries:
        return map_entries

    core_entries = classic_entries_for_query(title)
    if core_entries:
        filtered_core_entries = filter_capital_volume_entries(title, enrich_core_classic_entries(core_entries))
        if filtered_core_entries:
            return filtered_core_entries

    return []


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
    entries = []
    if title:
        entries = find_toc_entries(title)

    if not entries:
        constraints = constraints_from_query(query)
        entries = constraints.get("entries") or []
        title = constraints.get("title") or title

    lines = []

    for index, entry in enumerate(entries, start=1):
        metadata = {
            "source": entry.get("source"),
            "book": entry.get("book_title") or ARTICLE_MAP.get(entry.get("source"), {}).get("book", ""),
            "article": entry.get("article") or title or "",
        }
        _, book_title, volume, year = normalize_book_parts(metadata)
        page = (
            str(entry["start_page"])
            if entry.get("start_page") == entry.get("end_page")
            else f"{entry['start_page']}-{entry['end_page']}"
        )
        lines.append(
            f"({index})\u300a{book_title or entry.get('source', '')}\u300b{volume}\uff0c"
            f"{entry.get('article') or title}\uff0c\u5317\u4eac\uff1a\u4eba\u6c11\u51fa\u7248\u793e{year}\uff0c"
            f"\u7b2c{page}\u9875\u3002"
        )
        if index >= 8:
            break

    return "\n".join(lines) if lines else None


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
    repaired_answer = repair_answer_citations(answer, evidence)
    display_evidence = filter_evidence_to_answer(repaired_answer, evidence)
    audit = audit_answer_citations(repaired_answer, display_evidence)
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
    return retrieval_utils.build_page_ranges(entries)


def dedupe_locator_entries(entries):
    return retrieval_utils.dedupe_locator_entries(entries)


def normalize_topic_entries(entries):
    return retrieval_utils.normalize_topic_entries(entries, _retrieval_ctx())


def topic_matches_query(topic, query):
    return retrieval_utils.topic_matches_query(topic, query, _retrieval_ctx())


def topic_entries_for_query(query):
    return retrieval_utils.topic_entries_for_query(query, _retrieval_ctx())


def topic_info_from_constraints(constraints):
    return retrieval_utils.topic_info_from_constraints(constraints)


def narrow_topic_constraints_by_query(query, constraints):
    return retrieval_utils.narrow_topic_constraints_by_query(
        query,
        constraints,
        _retrieval_ctx(),
    )


WORK_TITLE_ALIASES = {
    "法德农民问题": ["法德农民问题", "农民合作社", "农民问题"],
    "关于费尔巴哈的提纲": ["关于费尔巴哈的提纲", "费尔巴哈提纲", "费尔巴哈", "实践"],
}


def infer_work_title_from_query(query):
    return retrieval_utils.infer_work_title_from_query(query, _retrieval_ctx())


def concept_constraints_from_query(query):
    return retrieval_utils.concept_constraints_from_query(query, _retrieval_ctx())


def constraints_from_query(query):
    return retrieval_utils.constraints_from_query(query, _retrieval_ctx())


def metadata_matches_constraints(metadata, constraints):
    return retrieval_utils.metadata_matches_constraints(metadata, constraints)


def page_in_expected_range(metadata, constraints):
    return retrieval_utils.page_in_expected_range(metadata, constraints, _retrieval_ctx())


def topic_title_allowed(metadata, constraints):
    return retrieval_utils.topic_title_allowed(metadata, constraints, _retrieval_ctx())


def score_source_match(metadata, constraints):
    return retrieval_utils.score_source_match(metadata, constraints)


def score_page_range(metadata, constraints):
    return retrieval_utils.score_page_range(metadata, constraints, _retrieval_ctx())


def score_topic_title_match(metadata, constraints):
    return retrieval_utils.score_topic_title_match(metadata, constraints, _retrieval_ctx())


def score_topic_content_match(metadata, content, constraints):
    return retrieval_utils.score_topic_content_match(
        metadata,
        content,
        constraints,
        _retrieval_ctx(),
    )


def score_article_match(metadata, normalized_title, haystack):
    return retrieval_utils.score_article_match(
        metadata,
        normalized_title,
        haystack,
        _retrieval_ctx(),
    )


def score_query_match(normalized_query, haystack):
    return retrieval_utils.score_query_match(normalized_query, haystack)


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
    classic_norm = normalize_for_match(
        metadata.get("classic_title") or metadata.get("locator_title") or ""
    )

    for term in active_concept_terms(query):
        term_norm = normalize_for_match(term)
        if not term_norm:
            continue
        if term in CONCEPT_TITLE_FALLBACK_TO_CLASSIC:
            continue

        markers = list(CONCEPT_PREFERRED_MARKERS.get(term, []))
        preferred = CONCEPT_PREFERRED_SOURCES.get(term) or {}
        term_markers = [] if term in CONCEPT_TITLE_FALLBACK_TO_CLASSIC else [term]
        markers = list(dict.fromkeys(markers + term_markers + list(preferred.get("markers", []))))

        for marker in markers:
            marker_norm = normalize_for_match(marker)
            if classic_norm and marker_norm == classic_norm:
                continue
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
            if not concept_title:
                concept_title = next(
                    (
                        term
                        for term in active_concept_terms(query)
                        if term not in CONCEPT_TITLE_FALLBACK_TO_CLASSIC
                    ),
                    None,
                )
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

    deduped = []
    seen_norms = []
    for term in sorted(terms, key=lambda item: len(normalize_for_match(item)), reverse=True):
        normalized_term = normalize_for_match(term)
        if any(normalized_term in seen for seen in seen_norms):
            continue
        deduped.append(term)
        seen_norms.append(normalized_term)

    return deduped


def score_concept_focus(query, metadata, content):
    terms = active_concept_terms(query)
    if not terms:
        return 0

    article = clean_text(metadata.get("section") or metadata.get("article"), "")
    article_norm = normalize_for_match(article)
    content_norm = normalize_for_match(content)
    lead_norm = normalize_for_match(content[:300])
    early_body_norm = normalize_for_match(content[:1200])
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
            score += 55
        term_index = content_norm.find(term_norm)
        if term_norm in lead_norm:
            score += 70
        elif term_index != -1:
            score += 55
        else:
            score -= 60

        if term_index != -1:
            if term_index <= 120:
                score += 95
            elif term_index <= 260:
                score += 72
            elif term_index <= 480:
                score += 44
            elif term_index <= 900:
                score += 18
            else:
                score -= 18

        direct_definition_patterns = [
            f"什么是{term_norm}",
            f"{term_norm}是什么",
        ]
        loose_definition_patterns = [
            f"{term_norm}是",
            f"所谓{term_norm}",
            f"叫做{term_norm}",
            f"称为{term_norm}",
            f"就是{term_norm}",
        ]
        if any(pattern in lead_norm for pattern in direct_definition_patterns):
            score += 100
        elif any(pattern in lead_norm for pattern in loose_definition_patterns):
            score += 50
        elif any(pattern in early_body_norm for pattern in direct_definition_patterns):
            score += 75
        elif any(pattern in early_body_norm for pattern in loose_definition_patterns):
            score += 42

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

    return score


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

    if content.startswith(("说明本卷", "本卷收入", "本卷是")):
        score -= 120

    if article == clean_text(metadata.get("book"), "") and metadata_citation_page(metadata) is not None and metadata_citation_page(metadata) <= 10:
        score -= 100

    if is_noisy_article_title(article):
        score -= 60

    if any(marker in article_norm for marker in ["名目索引", "人名索引"]):
        score -= 70

    if any(marker in article_norm for marker in ["目录", "目次", "索引", "注释", "编者注"]):
        score -= 45

    if any(normalize_for_match(marker) in article_norm for marker in DERIVATIVE_TITLE_MARKERS):
        score -= 95

    if is_front_matter_candidate(metadata, content):
        score -= 120

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
    return retrieval_utils.debug_rerank_score(index, doc, score_parts, _retrieval_ctx())


def rerank_documents(query, docs, constraints):
    return retrieval_utils.rerank_documents(query, docs, constraints, _retrieval_ctx())


def diversify_documents(docs, k, max_per_source=2, max_per_article=1, min_distinct_sources=0):
    return retrieval_utils.diversify_documents(
        docs,
        k,
        _retrieval_ctx(),
        max_per_source=max_per_source,
        max_per_article=max_per_article,
        min_distinct_sources=min_distinct_sources,
    )


def annotate_docs_with_constraints(docs, constraints):
    return retrieval_utils.annotate_docs_with_constraints(docs, constraints, _retrieval_ctx())


def select_topic_documents(ranked_docs, constraints, k):
    return retrieval_utils.select_topic_documents(ranked_docs, constraints, k, _retrieval_ctx())


def strict_title_cache_documents(query, constraints, limit=6):
    return retrieval_utils.strict_title_cache_documents(
        query,
        constraints,
        limit,
        _retrieval_ctx(),
    )


def locator_backstop_documents(constraints, limit=4):
    return retrieval_utils.locator_backstop_documents(constraints, limit)


def append_locator_backstops(docs, constraints, k):
    return retrieval_utils.append_locator_backstops(docs, constraints, k, _retrieval_ctx())


def dedupe_documents(docs):
    return retrieval_utils.dedupe_documents(docs, _retrieval_ctx())


def topic_seed_queries(query, constraints):
    return retrieval_utils.topic_seed_queries(query, constraints, _retrieval_ctx())


def concept_seed_queries(query, constraints):
    return retrieval_utils.concept_seed_queries(query, constraints, _retrieval_ctx())


def controlled_multi_queries(query, constraints, ctx=None):
    return retrieval_utils.controlled_multi_queries(query, constraints, ctx or _retrieval_ctx())


def topic_constrained_candidates(query, db, constraints, fetch_k):
    return retrieval_utils.topic_constrained_candidates(
        query,
        db,
        constraints,
        fetch_k,
        _retrieval_ctx(),
    )


def concept_constrained_candidates(query, db, constraints, fetch_k):
    return retrieval_utils.concept_constrained_candidates(
        query,
        db,
        constraints,
        fetch_k,
        _retrieval_ctx(),
    )


def retrieve_documents(query, db, k=5, allow_exact_quote=True, performance=None, strategy=None, variant_retrieval=False):
    ctx = _retrieval_ctx()
    if performance is not None or strategy is not None or variant_retrieval:
        ctx = dict(ctx)
    if performance is not None:
        ctx["hybrid_retrieval"] = bool(performance.get("hybrid_retrieval", True))
    if strategy is not None:
        ctx["strategy"] = strategy
    if variant_retrieval:
        # Planner variants are expansions of a query that already went through
        # constraint building; do not spend an LLM call re-locating works.
        ctx["variant_retrieval"] = True
    return retrieval_utils.retrieve_documents(
        query,
        db,
        k,
        allow_exact_quote,
        ctx,
    )




def candidate_pdf_pages_from_metadata(metadata):
    return retrieval_utils.candidate_pdf_pages_from_metadata(metadata, _retrieval_ctx())


def refine_doc_citation_page_for_query(doc, query):
    return retrieval_utils.refine_doc_citation_page_for_query(doc, query, _retrieval_ctx())


def refine_docs_citation_pages_for_query(docs, query):
    return retrieval_utils.refine_docs_citation_pages_for_query(docs, query, _retrieval_ctx())

def retrieve_paragraph_documents(query, db, k=5):
    return retrieval_utils.retrieve_paragraph_documents(query, db, k, _retrieval_ctx())


def final_answer_style_rules():
    return prompts.final_answer_style_rules()


def footnote_citation_rules():
    return prompts.footnote_citation_rules()


def build_quote_prompt(query, context, mode=None):
    return prompts.build_quote_prompt(query, context, mode=mode)


def build_concept_prompt(query, context, mode=None):
    return prompts.build_concept_prompt(query, context, mode=mode)


def build_analysis_prompt(query, context, mode=None):
    return prompts.build_analysis_prompt(query, context, mode=mode)


def build_default_prompt(query, context, mode=None):
    return prompts.build_default_prompt(query, context, mode=mode)


def build_constraint_guard(constraints):
    return prompts.build_constraint_guard(constraints)


def build_prompt(intent, query, context, mode=None):
    return prompts.build_prompt(intent, query, context, mode=mode)


def build_ambiguous_locator_answer(query, constraints, limit=10):
    return ambiguous_utils.build_ambiguous_locator_answer(query, constraints, limit=limit)

def _clip_context_text(text, limit):
    text = clean_text(text, "")
    if not limit or limit <= 0 or len(text) <= limit:
        return text
    head_limit = max(int(limit * 0.72), 1)
    tail_limit = max(limit - head_limit, 0)
    head = text[:head_limit].rstrip()
    tail = text[-tail_limit:].lstrip() if tail_limit else ""
    if tail:
        return f"{head}\n……（中间内容已按上下文预算省略）……\n{tail}"
    return head


def build_context(docs, query_intent, performance=None):
    # Chunk creation happens in the vectorstore build step. This function only
    # consumes chunks and keeps their metadata visible for citation and prompts.
    performance = performance or {}
    doc_char_limit = int(performance.get("context_doc_char_limit") or 0)
    total_char_limit = int(performance.get("context_total_char_limit") or 0)
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
        letter_notice = ""
        metadata_fields_text = (
            f"metadata_fields: book={book}, article={article}, section={section}, page={page}, source={source}\n"
        )
        page_fields_text = (
            f"page_fields: printed_page={printed_page}, citation_page={citation_page}, pdf_page={pdf_page}{page_range_text}\n"
        )
        if metadata.get("no_page_citation"):
            letter_title = clean_text(metadata.get("letter_title") or article, article)
            source_page = "书信材料"
            letter_notice = (
                f"letter_mode=true, letter_title={letter_title}, "
                "citation_policy=letter_title_only\n"
            )
            metadata_fields_text = (
                f"metadata_fields: book={book}, article={article}, section={section}, source={source}\n"
            )
            page_fields_text = (
                "locator_fields: suppressed_for_letter=true\n"
            )

        clipped_content = _clip_context_text(doc.page_content, doc_char_limit)
        card = (
            f"EVIDENCE-CARD E{i}\n"
            f"evidence_id={evidence_id}\n"
            f"\u6765\u6e90\uff1a\u300a{book}\u300b{article}{section_text}\uff0c{source_page}\uff0csource={source}\n"
            f"{letter_notice}"
            f"{confidence_text}\n"
            f"{classic_meta_text}"
            f"{metadata_fields_text}"
            f"{page_fields_text}"
            f"position_fields: line_start={line_start}, line_end={line_end}\n"
            f"\u53e5\u5b50\u5f15\u6587\u683c\u5f0f\uff1a{sentence_citation}\n"
            f"\u6bb5\u843d\u5177\u4f53\u51fa\u5904\u683c\u5f0f\uff1a{detailed_source}\n"
            f"\u539f\u6587\uff1a{clipped_content}"
        )
        next_total = len("\n\n".join(context_parts + [card]))
        if total_char_limit and context_parts and next_total > total_char_limit:
            break
        if total_char_limit and next_total > total_char_limit:
            card = card[:total_char_limit].rstrip() + "\n……（上下文预算已截断）"
        context_parts.append(card)

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
    return RUNTIME.env_flag(name)


def dev_mode_enabled():
    """Gate developer-only output that may expose prompts, chunks, or metadata."""
    return RUNTIME.dev_mode_enabled()


def trace_enabled():
    return RUNTIME.trace_enabled()


def trace_only_enabled():
    return RUNTIME.trace_only_enabled()


def dual_retrieval_enabled():
    return RUNTIME.dual_retrieval_enabled()


def compact_preview(text, limit=180):
    return trace_utils.compact_preview(text, clean_text, limit=limit)


def print_trace_line(text=""):
    return trace_utils.print_trace_line(text)


def print_query_trace(query, query_intent):
    return trace_utils.print_query_trace(query, query_intent)


def print_constraints_trace(constraints):
    return trace_utils.print_constraints_trace(constraints)


def print_docs_trace(docs, label="retrieved_docs"):
    return trace_utils.print_docs_trace(
        docs,
        normalize_metadata,
        format_citation,
        compact_preview,
        label=label,
    )


def print_prompt_trace(prompt):
    return trace_utils.print_prompt_trace(prompt, compact_preview)


def build_trace_only_answer(query_intent, docs, prompt, paragraph_docs=None):
    return trace_utils.build_trace_only_answer(
        query_intent,
        docs,
        prompt,
        normalize_metadata,
        compact_preview,
        paragraph_docs=paragraph_docs,
    )


def load_embeddings():
    return RUNTIME.load_embeddings()


def embed_query(query: str):
    """Return an embedding vector for *query*.

    Used by the ML intent classifier (v3) to blend rule-based scores with
    a lightweight logistic-regression head.  Raises ``RuntimeError`` if the
    embedding model has not been initialised.
    """
    emb = RUNTIME.load_embeddings()
    return emb.embed_query(query)


def load_vectorstore():
    return RUNTIME.load_vectorstore()


def paragraph_vectorstore_exists():
    return RUNTIME.paragraph_vectorstore_exists()


def load_paragraph_vectorstore():
    return RUNTIME.load_paragraph_vectorstore()


def retrieve_dual_documents(query, chunk_db, paragraph_db, k=5):
    return retrieval_utils.retrieve_dual_documents(
        query,
        chunk_db,
        paragraph_db,
        k,
        _retrieval_ctx(),
    )


def merge_prefer_paragraph_docs(paragraph_docs, chunk_docs, limit):
    return retrieval_utils.merge_prefer_paragraph_docs(
        paragraph_docs,
        chunk_docs,
        limit,
        _retrieval_ctx(),
    )


def filter_paragraph_docs_by_text_overlap(query, docs, limit=None):
    return retrieval_utils.filter_paragraph_docs_by_text_overlap(
        query,
        docs,
        _retrieval_ctx(),
        limit=limit,
    )


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
    return answer_utils.answer_unsupported_claim(
        query,
        UNSUPPORTED_CLAIM_RULES,
        normalize_for_match,
    )



def evidence_from_doc(doc, index=1):
    return citations.evidence_from_doc(
        doc,
        index,
        normalize_metadata,
        clean_text,
        compact_preview,
        format_citation,
    )


def evidence_from_docs(docs, limit=12):
    return citations.evidence_from_docs(
        docs,
        limit,
        normalize_metadata,
        clean_text,
        compact_preview,
        format_citation,
    )


def is_view_list_query(query):
    return answer_utils.is_view_list_query(query, normalize_for_match)


def is_topic_view_list_query(query, constraints):
    return answer_utils.is_topic_view_list_query(query, constraints, normalize_for_match)


def clean_excerpt_for_display(text, article=""):
    return answer_utils.clean_excerpt_for_display(text, clean_text, article=article)


def best_excerpt_span(text, markers, max_len=88):
    return answer_utils.best_excerpt_span(
        text,
        markers,
        clean_text,
        normalize_for_match,
        max_len=max_len,
    )


def summarize_peasant_cooperative_viewpoint(text):
    return answer_utils.summarize_peasant_cooperative_viewpoint(text, normalize_for_match)


def format_topic_viewpoint(item, constraints):
    return answer_utils.format_topic_viewpoint(
        item,
        constraints,
        clean_text,
        normalize_for_match,
    )


def strict_title_answer_evidence(query, constraints, evidence, limit=8):
    return answer_utils.strict_title_answer_evidence(
        query,
        constraints,
        evidence,
        active_concept_terms,
        clean_text,
        normalize_for_match,
        limit=limit,
    )


def build_strict_title_view_list_answer(query, constraints, evidence, limit=8):
    return answer_utils.build_strict_title_view_list_answer(
        query,
        constraints,
        evidence,
        active_concept_terms,
        clean_text,
        normalize_for_match,
        limit=limit,
    )


def topic_direct_evidence(evidence, constraints):
    return answer_utils.topic_direct_evidence(
        evidence,
        constraints,
        clean_text,
        normalize_for_match,
    )


def topic_answer_evidence(evidence, constraints, limit=10):
    return answer_utils.topic_answer_evidence(
        evidence,
        constraints,
        clean_text,
        normalize_for_match,
        limit=limit,
    )


def build_topic_view_list_answer(query, constraints, evidence, limit=8):
    return answer_utils.build_topic_view_list_answer(
        query,
        constraints,
        evidence,
        clean_text,
        normalize_for_match,
        limit=limit,
    )


def extract_answer_citation_lines(answer):
    return citations.extract_answer_citation_lines(answer, normalize_final_answer)


def citation_match_key(citation):
    return citations.citation_match_key(citation, normalize_for_match)


def evidence_matches_citation(item, citation):
    return citations.evidence_matches_citation(item, citation, normalize_for_match)


def filter_evidence_to_answer(answer, evidence, fallback_limit=3):
    return citations.filter_evidence_to_answer(
        answer,
        evidence,
        fallback_limit,
        normalize_final_answer,
        normalize_for_match,
    )


def repair_answer_citations(answer, evidence, fallback_limit=4):
    return citations.repair_answer_citations(
        answer,
        evidence,
        fallback_limit,
        normalize_final_answer,
        normalize_for_match,
    )


def audit_answer_citations(answer, evidence):
    return citations.audit_answer_citations(
        answer,
        evidence,
        normalize_final_answer,
        normalize_for_match,
    )


def set_last_evidence(evidence=None, audit=None):
    global LAST_EVIDENCE, LAST_CITATION_AUDIT
    LAST_EVIDENCE = evidence or []
    LAST_CITATION_AUDIT = audit or {"ok": True, "issues": [], "evidence_count": len(LAST_EVIDENCE)}


def set_last_topic_info(info=None):
    global LAST_TOPIC_INFO
    LAST_TOPIC_INFO = info or {}


def set_last_crag_report(report=None):
    global LAST_CRAG_REPORT
    LAST_CRAG_REPORT = report or {}


def set_last_timing(info=None):
    global LAST_TIMING
    LAST_TIMING = info or {}


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


def performance_settings(mode=None):
    """Return retrieval/generation knobs for web and CLI answer paths."""
    selected = (
        mode
        or os.getenv("MARXOS_PERFORMANCE_MODE", SETTINGS.answer.default_performance_mode)
        or SETTINGS.answer.default_performance_mode
    ).lower()
    presets = {
        "fast": {
            "mode": "fast",
            "retrieve_k": 3,
            "rag_retrieve_k": 5,
            "paragraph_retrieval": False,
            "corrective_retrieval": False,
            "planner_multi_query": False,
            "hybrid_retrieval": False,
            "content_verification": False,
            "max_recovery_rounds": 0,
            "citation_audit_mode": "lightweight",
            "citation_recovery": False,
            "citation_page_refinement": False,
            "context_doc_char_limit": 800,
            "context_total_char_limit": 2500,
            "max_tokens": 700,
            "llm_timeout": 35.0,
        },
        "standard": {
            "mode": "standard",
            "retrieve_k": 4,
            "rag_retrieve_k": 8,
            "paragraph_retrieval": False,
            "corrective_retrieval": True,
            "planner_multi_query": False,
            "hybrid_retrieval": False,
            "content_verification": False,
            "max_recovery_rounds": 0,
            "citation_audit_mode": "lightweight",
            "citation_recovery": False,
            "citation_page_refinement": False,
            "context_doc_char_limit": 1500,
            "context_total_char_limit": 6000,
            "max_tokens": 1100,
            "llm_timeout": 60.0,
        },
        "deep": {
            "mode": "deep",
            "retrieve_k": 5,
            "rag_retrieve_k": 12,
            "paragraph_retrieval": True,
            "corrective_retrieval": True,
            "planner_multi_query": True,
            "hybrid_retrieval": True,
            "content_verification": True,
            "max_recovery_rounds": 2,
            "citation_audit_mode": "deep",
            "citation_recovery": True,
            "citation_page_refinement": True,
            "context_doc_char_limit": 4000,
            "context_total_char_limit": 16000,
            "max_tokens": None,
            "llm_timeout": 120.0,
        },
    }
    return presets.get(selected, presets["deep"])


def run_query(query, route_query=None, force_intent=None, history=None, performance=None):
    global LAST_ANSWER_PATH
    LAST_ANSWER_PATH = ""
    with phoenix.trace_manager.start_as_current_span("marxos.run_query") as root_span:
        try:
            run_started = time.perf_counter()
            timings = {}

            def _mark_phase(phase, started, extra=None):
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                total_ms = int((time.perf_counter() - run_started) * 1000)
                timings[phase] = elapsed_ms
                payload = {
                    "event": "marxos_timing",
                    "phase": phase,
                    "elapsed_ms": elapsed_ms,
                    "total_ms": total_ms,
                }
                if extra:
                    payload.update(extra)
                try:
                    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)
                except BrokenPipeError:
                    pass
                except UnicodeEncodeError:
                    try:
                        print(json.dumps(payload, ensure_ascii=True), file=sys.stderr, flush=True)
                    except BrokenPipeError:
                        pass

            def _mark_event(phase, extra=None):
                payload = {
                    "event": "marxos_timing",
                    "phase": phase,
                    "total_ms": int((time.perf_counter() - run_started) * 1000),
                }
                if extra:
                    payload.update(extra)
                try:
                    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)
                except BrokenPipeError:
                    pass
                except UnicodeEncodeError:
                    try:
                        print(json.dumps(payload, ensure_ascii=True), file=sys.stderr, flush=True)
                    except BrokenPipeError:
                        pass

            perf = performance_settings(performance)
            set_last_evidence([])
            set_last_topic_info({})
            set_last_crag_report({})
            set_last_timing({})

            # Use provided classification function or default
            _classify = classify_query
            if force_intent:
                _classify = lambda q: force_intent

            request = orchestration.prepare_query_request(
                query, route_query, clean_text, is_unreadable_query,
                answer_unsupported_claim, _classify,
            )
            if request.get("early_answer"):
                phoenix.set_attributes(
                    root_span,
                    {
                        "app.query": phoenix.compact_text(query, limit=240),
                        "answer.path": "early_answer",
                        "answer.length": len(request["early_answer"]),
                    },
                )
                return request["early_answer"]

            query = request["query"]
            route_query = request["route_query"]
            query_intent = request["query_intent"]

            # Apply intent-specific retrieval strategy overrides (NEW)
            from marxos.config.retrieval_strategies import get_intent_strategy, apply_strategy
            strategy = get_intent_strategy(query_intent)
            perf = apply_strategy(perf, strategy)

            plan_started = time.perf_counter()
            plan = query_planner.plan_query(route_query, query_intent, history=history or [])
            if plan.standalone_query and plan.standalone_query != route_query:
                route_query = plan.standalone_query
            _mark_phase(
                "query_plan",
                plan_started,
                {
                    "mode": perf.get("mode"),
                    "intent": query_intent,
                    "plan_mode": plan.mode,
                    "variants": len(plan.retrieval_queries),
                },
            )
            trace = trace_enabled()
            trace_only = trace_only_enabled()
            dual_retrieval = dual_retrieval_enabled()

            def _audit_rank(audit_payload):
                issues = audit_payload.get("issues") or []
                return (
                    1 if audit_payload.get("ok") else 0,
                    -len(issues),
                    int(audit_payload.get("evidence_count") or 0),
                )

            def _build_prompt_for_docs(active_docs, span_name):
                phase_started = time.perf_counter()
                with phoenix.trace_manager.start_as_current_span(span_name) as span:
                    context = build_context(active_docs, query_intent, performance=perf)
                    prompt = clean_text(
                        build_prompt(query_intent, query, context, mode=perf.get("mode"))
                        + build_constraint_guard(constraints)
                    )
                    phoenix.set_attributes(
                        span,
                        {
                            "prompt.mode": perf.get("mode") or "",
                            "prompt.length": len(prompt),
                            "prompt.preview": phoenix.compact_text(prompt, limit=240),
                        },
                    )
                _mark_phase(
                    span_name,
                    phase_started,
                    {"prompt_length": len(prompt), "doc_count": len(active_docs or [])},
                )
                return prompt

            def _generate_raw_answer(prompt, span_name):
                phase_started = time.perf_counter()
                with phoenix.trace_manager.start_as_current_span(span_name) as span:
                    client = create_deepseek_client(
                        max_retries=3,
                        timeout=float(perf.get("llm_timeout") or 120.0),
                    )
                    # deep 模式走 pro；fast/standard 走 flash（日常问答降本）。
                    model = generation_model(perf.get("mode"))
                    request_kwargs = {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        "extra_body": deepseek_extra_body(),
                    }
                    if perf.get("max_tokens"):
                        request_kwargs["max_tokens"] = int(perf["max_tokens"])
                    _mark_event(
                        f"{span_name}_start",
                        {
                            "prompt_length": len(prompt or ""),
                            "max_tokens": int(perf.get("max_tokens") or 0),
                            "timeout": float(perf.get("llm_timeout") or 120.0),
                        },
                    )
                    response = client.chat.completions.create(**request_kwargs)
                    raw_answer = response.choices[0].message.content
                    phoenix.set_attributes(
                        span,
                        {
                            "llm.vendor": "deepseek",
                            "llm.api_style": "openai_compatible",
                            "llm.model": model,
                            "llm.output_length": len(raw_answer or ""),
                            "llm.max_tokens": int(perf["max_tokens"] or 0),
                        },
                    )
                _mark_phase(span_name, phase_started, {"answer_length": len(raw_answer or "")})
                return raw_answer

            def _finalize_answer(raw_answer, active_evidence, active_crag_report, span_name, recovery_used=False):
                phase_started = time.perf_counter()
                with phoenix.trace_manager.start_as_current_span(span_name) as span:
                    repaired_answer = repair_answer_citations(raw_answer, active_evidence)
                    display_evidence = filter_evidence_to_answer(repaired_answer, active_evidence)
                    audit = audit_answer_citations(repaired_answer, display_evidence)
                    audit["mode"] = perf.get("citation_audit_mode") or "lightweight"
                    if perf.get("content_verification", True):
                        # Content verification against OCR text is valuable but expensive.
                        content_verify = verify_citations(repaired_answer, display_evidence)
                        if content_verify:
                            audit["content_verification"] = content_verify
                    audit["crag_report"] = dict(active_crag_report or {})
                    audit["crag_recovery_used"] = recovery_used
                    phoenix.set_attributes(
                        span,
                        {
                            "citation.audit_ok": bool(audit.get("ok")),
                            "citation.audit_mode": audit.get("mode") or "",
                            "citation.issue_count": len(audit.get("issues") or []),
                            "answer.length": len(audit["answer"]),
                            "crag.path": (active_crag_report or {}).get("path") or "",
                            "crag.score": int((active_crag_report or {}).get("score") or 0),
                        },
                    )
                    phoenix.set_attributes(span, phoenix.summarize_evidence(display_evidence))
                _mark_phase(
                    span_name,
                    phase_started,
                    {
                        "audit_ok": bool(audit.get("ok")),
                        "issue_count": len(audit.get("issues") or []),
                        "evidence_count": len(display_evidence or []),
                    },
                )
                return audit, display_evidence

            phoenix.set_attributes(
                root_span,
                {
                    "app.query": phoenix.compact_text(query, limit=240),
                    "app.route_query": phoenix.compact_text(route_query, limit=240),
                    "app.query_intent": query_intent,
                    "app.query_plan_mode": plan.mode,
                    "app.query_plan_variants": len(plan.retrieval_queries),
                    "app.trace_enabled": trace,
                    "app.trace_only": trace_only,
                    "app.dual_retrieval": dual_retrieval,
                    "app.performance_mode": perf.get("mode"),
                    "phoenix.enabled": phoenix.trace_manager.enabled(),
                },
            )
            init_error = phoenix.trace_manager.init_error()
            if init_error:
                root_span.add_event(
                    "phoenix.init_warning",
                    {"message": phoenix.compact_text(init_error, limit=240)},
                )

            if trace or trace_only:
                print_query_trace(route_query, query_intent)

            with phoenix.trace_manager.start_as_current_span("marxos.local_lookup") as span:
                phoenix.set_attributes(
                    span,
                    {
                        "app.query_intent": query_intent,
                    },
                )
                local_answer = orchestration.maybe_answer_local_lookup(
                    query,
                    route_query,
                    query_intent,
                    trace,
                    trace_only,
                    answer_bibliographic_query,
                    extract_bibliographic_title,
                    answer_quote_query,
                    print_trace_line,
                )
                if local_answer:
                    phoenix.set_attributes(
                        span,
                        {
                            "answer.path": "local_lookup",
                            "answer.length": len(local_answer),
                        },
                    )
                    phoenix.set_attributes(
                        root_span,
                        {
                            "answer.path": "local_lookup",
                            "answer.length": len(local_answer),
                        },
                    )
                    LAST_ANSWER_PATH = "local_lookup"
                    return local_answer

            # Local and malformed-input paths must resolve before the general
            # out-of-domain LLM fallback. They are deterministic and must not
            # require network credentials merely because relevance is unclear.
            if not is_marxism_relevant(query):
                prompt = (
                    f"用户问题：{query}\n\n"
                    "这是一个与马克思主义专业领域无关的问题。"
                    "请以通用助手的身份简洁回答，不要引用马克思或恩格斯的著作。"
                )
                with phoenix.trace_manager.start_as_current_span("marxos.llm_generate_out_of_domain") as span:
                    client = create_deepseek_client(max_retries=2, timeout=30.0)
                    response = client.chat.completions.create(
                        model=deepseek_flash_model(),
                        messages=[{"role": "user", "content": prompt}],
                        extra_body=deepseek_extra_body(),
                    )
                    raw_answer = response.choices[0].message.content or ""
                    phoenix.set_attributes(
                        span,
                        {
                            "llm.vendor": "deepseek",
                            "llm.model": deepseek_flash_model(),
                            "llm.output_length": len(raw_answer),
                        },
                    )
                phoenix.set_attributes(
                    root_span,
                    {
                        "app.query": phoenix.compact_text(query, limit=240),
                        "answer.path": "out_of_domain",
                        "answer.length": len(raw_answer),
                    },
                )
                timings["total"] = int((time.perf_counter() - run_started) * 1000)
                timings["mode"] = perf.get("mode")
                timings["intent"] = "out_of_domain"
                set_last_timing(timings)
                set_last_evidence([], {"ok": True, "issues": [], "evidence_count": 0})
                LAST_ANSWER_PATH = "out_of_domain"
                return raw_answer

            constraints = constraints_from_query(route_query)
            retrieval_started = time.perf_counter()
            with phoenix.trace_manager.start_as_current_span("marxos.retrieval") as span:
                phoenix.set_attributes(span, phoenix.summarize_constraints(constraints))
                retrieval_state = orchestration.collect_retrieval_materials(
                    query,
                    route_query,
                    query_intent,
                    constraints,
                    PARAGRAPH_VECTORSTORE_DIR,
                    trace,
                    trace_only,
                    topic_info_from_constraints,
                    set_last_topic_info,
                    print_trace_line,
                    print_constraints_trace,
                    load_vectorstore,
                    retrieve_documents,
                    paragraph_vectorstore_exists,
                    load_paragraph_vectorstore,
                    filter_paragraph_docs_by_text_overlap,
                    merge_prefer_paragraph_docs,
                    refine_docs_citation_pages_for_query,
                    evidence_from_docs,
                    is_topic_view_list_query,
                    query_plan=plan.as_dict(),
                    performance=perf,
                    strategy=strategy,
                )
                docs = retrieval_state["docs"]
                evidence = retrieval_state["evidence"]
                crag_report = retrieval_state.get("crag_report") or {}
                set_last_crag_report(crag_report)
                paragraph_docs = retrieval_state["paragraph_docs"] if dual_retrieval else []
                phoenix.set_attributes(span, phoenix.summarize_docs(docs, normalize_metadata))
                if paragraph_docs:
                    phoenix.set_attributes(
                        span,
                        phoenix.summarize_docs(
                            paragraph_docs,
                            normalize_metadata,
                            limit=2,
                        ),
                    )
                phoenix.set_attributes(span, phoenix.summarize_evidence(evidence))
                phoenix.set_attributes(
                    span,
                    {
                        "crag.path": crag_report.get("path") or "",
                        "crag.score": int(crag_report.get("score") or 0),
                        "crag.threshold": int(crag_report.get("threshold") or 0),
                        "crag.ok": bool(crag_report.get("ok", False)),
                    },
                )
            _mark_phase(
                "retrieval",
                retrieval_started,
                {
                    "doc_count": len(docs or []),
                    "evidence_count": len(evidence or []),
                    "crag_path": crag_report.get("path") or "",
                    "crag_score": int(crag_report.get("score") or 0),
                },
            )

            ambiguous_answer = build_ambiguous_locator_answer(route_query, constraints)
            if ambiguous_answer:
                set_last_evidence(evidence, {"ok": True, "issues": [], "evidence_count": len(evidence), "answer": ambiguous_answer})
                phoenix.set_attributes(
                    root_span,
                    {
                        "answer.path": "ambiguous_locator",
                        "answer.length": len(ambiguous_answer),
                    },
                )
                LAST_ANSWER_PATH = "ambiguous_locator"
                return ambiguous_answer

            with phoenix.trace_manager.start_as_current_span("marxos.local_view_answer") as span:
                local_view_answer = orchestration.maybe_answer_local_view_query(
                    query,
                    route_query,
                    query_intent,
                    constraints,
                    docs,
                    evidence,
                    set_last_evidence,
                    filter_evidence_to_answer,
                    audit_answer_citations,
                    is_topic_view_list_query,
                    build_topic_view_list_answer,
                    topic_answer_evidence,
                    is_view_list_query,
                    build_strict_title_view_list_answer,
                    strict_title_answer_evidence,
                )
                if local_view_answer:
                    phoenix.set_attributes(
                        span,
                        {
                            "answer.path": "local_view",
                            "answer.length": len(local_view_answer),
                        },
                    )
                    phoenix.set_attributes(
                        root_span,
                        {
                            "answer.path": "local_view",
                            "answer.length": len(local_view_answer),
                        },
                    )
                    # A local view answer with no retrieved docs is a refusal,
                    # not a structured list answer.
                    LAST_ANSWER_PATH = "refusal" if not docs else "local_view"
                    return local_view_answer

            if not docs:
                # No retrieved documents → deterministic refusal. The LLM must
                # never see an empty context: without evidence cards nothing is
                # verifiable, and prose fabrication would slip past the
                # citation-line audit.
                refusal = answer_utils.answer_insufficient_material(route_query, constraints)
                set_last_evidence([], {"ok": True, "issues": [], "evidence_count": 0, "answer": refusal})
                timings["total"] = int((time.perf_counter() - run_started) * 1000)
                timings["mode"] = perf.get("mode")
                timings["intent"] = query_intent
                set_last_timing(timings)
                LAST_ANSWER_PATH = "refusal"
                _mark_event("run_query_done", {"total_ms": timings["total"], "mode": perf.get("mode"), "path": "refusal"})
                phoenix.set_attributes(
                    root_span,
                    {"answer.path": "refusal", "answer.length": len(refusal)},
                )
                return refusal

            set_last_evidence([])
            prompt = _build_prompt_for_docs(docs, "marxos.prompt_build")
            if trace or trace_only:
                print_docs_trace(docs)
                if dual_retrieval and paragraph_docs:
                    print_docs_trace(paragraph_docs, label="paragraph_retrieved_docs")
                print_prompt_trace(prompt)

            if trace_only:
                answer = build_trace_only_answer(query_intent, docs, prompt, paragraph_docs=paragraph_docs)
                phoenix.set_attributes(
                    root_span,
                    {
                        "answer.path": "trace_only",
                        "answer.length": len(answer),
                    },
                )
                LAST_ANSWER_PATH = "trace_only"
                return answer

            raw_answer = _generate_raw_answer(prompt, "marxos.llm_generate")
            audit, display_evidence = _finalize_answer(
                raw_answer,
                evidence,
                crag_report,
                "marxos.citation_audit",
            )

            recovery_round = 0

            def _try_recover(audit, docs, evidence, paragraph_docs, crag_report, prompt,
                             display_evidence, recovery_reason="citation_format"):
                """Attempt one round of recovery: re-retrieve + regenerate + re-audit."""
                nonlocal recovery_round
                recovery_round += 1
                if recovery_round > 2:
                    return audit, docs, evidence, paragraph_docs, crag_report, prompt, display_evidence
                if recovery_round > int(perf.get("max_recovery_rounds", 2)):
                    return audit, docs, evidence, paragraph_docs, crag_report, prompt, display_evidence

                with phoenix.trace_manager.start_as_current_span(
                    f"marxos.recovery_{recovery_round}_{recovery_reason}"
                ) as span:
                    recovery_state = orchestration.collect_retrieval_materials(
                        query, route_query, query_intent, constraints,
                        PARAGRAPH_VECTORSTORE_DIR, trace, trace_only,
                        topic_info_from_constraints, set_last_topic_info,
                        print_trace_line, print_constraints_trace,
                        load_vectorstore, retrieve_documents,
                        paragraph_vectorstore_exists, load_paragraph_vectorstore,
                        filter_paragraph_docs_by_text_overlap,
                        merge_prefer_paragraph_docs,
                        refine_docs_citation_pages_for_query,
                        evidence_from_docs, is_topic_view_list_query,
                        force_corrective=True,
                        query_plan=plan.as_dict(),
                        performance=perf,
                    )
                    new_docs = recovery_state["docs"]
                    new_evidence = recovery_state["evidence"]
                    new_crag = recovery_state.get("crag_report") or {}
                    new_para = recovery_state["paragraph_docs"] if dual_retrieval else []
                    if not new_docs:
                        # Recovery re-retrieved nothing: refuse deterministically
                        # instead of regenerating against an empty context.
                        refusal = answer_utils.answer_insufficient_material(route_query, constraints)
                        return (
                            {"ok": True, "issues": [], "evidence_count": 0,
                             "answer": refusal, "mode": perf.get("citation_audit_mode") or "lightweight",
                             "crag_report": dict(new_crag or {}), "crag_recovery_used": True},
                            new_docs, new_evidence, new_para, new_crag, "", [],
                        )
                    new_prompt = _build_prompt_for_docs(new_docs, f"marxos.prompt_build_recovery_{recovery_round}")
                    new_raw = _generate_raw_answer(new_prompt, f"marxos.llm_generate_recovery_{recovery_round}")
                    new_audit, new_display = _finalize_answer(
                        new_raw, new_evidence, new_crag,
                        f"marxos.citation_audit_recovery_{recovery_round}",
                        recovery_used=True,
                    )
                    if _audit_rank(new_audit) >= _audit_rank(audit):
                        set_last_crag_report(new_crag)
                        return new_audit, new_docs, new_evidence, new_para, new_crag, new_prompt, new_display
                return audit, docs, evidence, paragraph_docs, crag_report, prompt, display_evidence

            # Round 1: citation format recovery (existing)
            if perf.get("citation_recovery", True) and not audit.get("ok"):
                audit, docs, evidence, paragraph_docs, crag_report, prompt, display_evidence = _try_recover(
                    audit, docs, evidence, paragraph_docs, crag_report, prompt, display_evidence,
                    "citation_format")

            # Round 2: content verification recovery (new)
            content_verify = audit.get("content_verification") or {}
            if perf.get("citation_recovery", True) and content_verify and content_verify.get("total", 0) > 0:
                hall_rate = content_verify.get("hallucinated", 0) / content_verify["total"]
                if hall_rate > 0.5:  # >50% citations flagged → verify-driven recovery
                    audit, docs, evidence, paragraph_docs, crag_report, prompt, display_evidence = _try_recover(
                        audit, docs, evidence, paragraph_docs, crag_report, prompt, display_evidence,
                        "content_verification")

            set_last_evidence(display_evidence, audit)
            timings["total"] = int((time.perf_counter() - run_started) * 1000)
            timings["mode"] = perf.get("mode")
            timings["intent"] = query_intent
            set_last_timing(timings)
            _mark_event("run_query_done", {"total_ms": timings["total"], "mode": perf.get("mode")})
            phoenix.set_attributes(
                root_span,
                {
                    "answer.path": "llm",
                    "answer.length": len(audit["answer"]),
                    "citation.audit_ok": bool(audit.get("ok")),
                    "crag.path": crag_report.get("path") or "",
                    "crag.score": int(crag_report.get("score") or 0),
                    "crag.recovery_used": bool(audit.get("crag_recovery_used", False)),
                },
            )
            LAST_ANSWER_PATH = "llm"
            return audit["answer"]
        except Exception as exc:
            root_span.record_exception(exc)
            raise


def main():
    for line in phoenix.startup_status_lines():
        print(line)
    query = input("请输入问题：")
    answer = run_query(query)
    print("\n===== MarxOS =====\n")
    print(answer)


if __name__ == "__main__":
    main()
