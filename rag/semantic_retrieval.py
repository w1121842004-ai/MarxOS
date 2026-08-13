from __future__ import annotations

import os
import math
import re
from array import array
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import pickle

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from marxos.config import get_settings
from rag.paragraph_cache import paragraph_record_to_document, read_paragraph_cache


SETTINGS = get_settings()
DEFAULT_PARAGRAPH_CACHE_PATH = Path(SETTINGS.corpus.paragraph_cache_path)
DEFAULT_LIGHT_SPARSE_INDEX_PATH = Path(
    os.getenv("SEMANTIC_LIGHT_SPARSE_INDEX_PATH", "data/sparse_paragraph_index_light.pkl")
)
DEFAULT_CHILD_CHUNK_SIZE = SETTINGS.retrieval.semantic_child_chunk_size
DEFAULT_CHILD_CHUNK_OVERLAP = SETTINGS.retrieval.semantic_child_chunk_overlap
DEFAULT_PARENT_WINDOW = SETTINGS.retrieval.semantic_parent_window
SPARSE_TOP_K = int(os.getenv("SEMANTIC_SPARSE_TOP_K", "24"))
SPARSE_RERANK_POOL = int(os.getenv("SEMANTIC_SPARSE_RERANK_POOL", "2000"))
SPARSE_CONTENT_RERANK_POOL = int(os.getenv("SEMANTIC_SPARSE_CONTENT_RERANK_POOL", "300"))
LIGHT_SPARSE_MIN_DF = int(os.getenv("SEMANTIC_LIGHT_SPARSE_MIN_DF", "5"))
LIGHT_SPARSE_MAX_DF_RATIO = float(os.getenv("SEMANTIC_LIGHT_SPARSE_MAX_DF_RATIO", "0.6"))
SPARSE_STOP_PHRASES = (
    "请解释",
    "请说明",
    "请问",
    "请概括",
    "概括",
    "总结",
    "归纳",
    "梳理",
    "为什么说",
    "为什么",
    "到底",
    "这里",
    "那里",
    "这个表述",
    "这个说法",
    "这个词",
    "这个概念",
    "不能",
    "不是",
    "如何",
    "什么是",
    "是什么",
    "怎么",
    "怎样",
    "单纯",
    "简单",
    "直接",
    "问题",
    "观点",
    "看法",
    "主张",
    "在马克思那里",
)


def looks_like_index_or_toc_text(text: str, metadata: dict | None = None) -> bool:
    text = str(text or "")
    metadata = metadata or {}
    article = str(metadata.get("article") or metadata.get("section") or "")
    if metadata.get("page_type") in {"toc", "title_page"}:
        return True
    if any(marker in article for marker in ["目录", "目次", "人名索引", "名目索引", "文献索引", "报刊索引", "地名索引"]):
        return True
    dash_entry_count = text.count("——") + text.count("---")
    if "并见" in text and dash_entry_count >= 1:
        return True
    if dash_entry_count >= 3 and len(text) < 600:
        return True
    punctuation_count = sum(1 for char in text if char in ".。!！?？,，;；:：…·•-—_[]()（）")
    return bool(text and punctuation_count / max(len(text), 1) > 0.42)


def _paragraph_cache_path(path: str | Path | None = None) -> Path:
    return Path(path) if path else DEFAULT_PARAGRAPH_CACHE_PATH


def _light_sparse_index_path(path: str | Path | None = None) -> Path:
    configured = Path(os.getenv("SEMANTIC_LIGHT_SPARSE_INDEX_PATH", str(DEFAULT_LIGHT_SPARSE_INDEX_PATH)))
    if path is None:
        return configured
    cache_path = _paragraph_cache_path(path)
    try:
        if cache_path.resolve() == DEFAULT_PARAGRAPH_CACHE_PATH.resolve():
            return configured
    except OSError:
        if str(cache_path) == str(DEFAULT_PARAGRAPH_CACHE_PATH):
            return configured
    safe_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(cache_path))
    return configured.with_name(f"{configured.stem}_{safe_name}{configured.suffix}")


@lru_cache(maxsize=2)
def load_paragraph_records(path: str | Path | None = None) -> tuple[dict, dict]:
    cache_path = _paragraph_cache_path(path)
    records = read_paragraph_cache(cache_path)
    by_id = {record.get("paragraph_id"): record for record in records if record.get("paragraph_id")}
    by_source = {}
    for record in records:
        source = record.get("source")
        if not source:
            continue
        by_source.setdefault(source, []).append(record)
    return by_id, by_source


def _sparse_record_for_index(record: dict) -> dict:
    return {
        key: record.get(key)
        for key in (
            "source",
            "book",
            "article",
            "section",
            "paragraph_text",
            "paragraph_char_count",
            "paragraph_index_on_page",
            "char_start",
            "char_end",
            "line_start",
            "line_end",
            "pdf_page_start",
            "pdf_page_end",
            "printed_page_start",
            "printed_page_end",
            "citation_page_start",
            "citation_page_end",
            "citation_page_type",
            "page_span",
            "page_type",
            "paragraph_index",
            "paragraph_id",
            "cross_page",
        )
        if record.get(key) is not None
    }


def normalize_sparse_text(text: str) -> str:
    text = str(text or "").lower()
    return re.sub(r"[\s《》“”\"'（）()、，。；：！？\-\.\·]", "", text)


def sparse_query_tokens(text: str) -> list[str]:
    normalized = normalize_sparse_text(text)
    if not normalized:
        return []

    tokens = []
    ascii_tokens = re.findall(r"[a-z0-9]+", normalized)
    tokens.extend(ascii_tokens)

    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
    for run in chinese_runs:
        if len(run) <= 2:
            tokens.append(run)
            continue
        for size in (2, 3):
            if len(run) < size:
                continue
            tokens.extend(run[index:index + size] for index in range(len(run) - size + 1))

    # Keep order stable while deduping for postings lookup.
    seen = set()
    unique_tokens = []
    for token in tokens:
        if not token or token in seen:
            continue
        seen.add(token)
        unique_tokens.append(token)
    return unique_tokens


def sparse_query_phrases(text: str) -> list[str]:
    normalized = normalize_sparse_text(text)
    if not normalized:
        return []

    phrases = []
    quoted = re.findall(r"[“\"]([^”\"]+)[”\"]", str(text or ""))
    for phrase in quoted:
        phrase = normalize_sparse_text(phrase)
        if len(phrase) >= 2:
            phrases.append(phrase)

    candidate = normalized
    for stop in SPARSE_STOP_PHRASES:
        candidate = candidate.replace(normalize_sparse_text(stop), " ")

    for run in re.findall(r"[\u4e00-\u9fff]{2,16}|[a-z0-9]{3,}", candidate):
        run = run.strip()
        if len(run) >= 2:
            phrases.append(run)

    seen = set()
    unique = []
    for phrase in sorted(phrases, key=len, reverse=True):
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        unique.append(phrase)
    return unique[:12]


def _pack_postings(
    postings: dict[str, dict[int, int]],
    doc_count: int,
    min_df: int = LIGHT_SPARSE_MIN_DF,
    max_df_ratio: float = LIGHT_SPARSE_MAX_DF_RATIO,
) -> dict[str, tuple[array, array]]:
    packed = {}
    max_df = int(doc_count * max_df_ratio) if max_df_ratio > 0 else 0
    for token, freqs in postings.items():
        if not freqs:
            continue
        doc_freq = len(freqs)
        if min_df > 1 and doc_freq < min_df:
            continue
        if max_df and doc_freq > max_df:
            continue
        doc_ids = array("I")
        term_freqs = array("H")
        for doc_id, freq in freqs.items():
            doc_ids.append(int(doc_id))
            term_freqs.append(min(int(freq), 65535))
        packed[token] = (doc_ids, term_freqs)
    return packed


def _iter_postings(postings):
    if not postings:
        return ()
    if isinstance(postings, tuple) and len(postings) == 2:
        return zip(postings[0], postings[1])
    return postings


def build_light_sparse_paragraph_index(path: str | Path | None = None) -> dict:
    records = read_paragraph_cache(_paragraph_cache_path(path))
    docs = []
    postings = defaultdict(dict)
    doc_lengths = []

    for record in records:
        doc_id = len(docs)
        sparse_text = " ".join(
            str(part or "")
            for part in (
                record.get("book"),
                record.get("article"),
                record.get("section"),
                record.get("paragraph_text"),
            )
        )
        tokens = sparse_query_tokens(sparse_text)
        if not tokens:
            continue

        term_freqs = Counter(tokens)
        doc_len = sum(term_freqs.values())
        indexed_record = _sparse_record_for_index(record)
        indexed_record["doc_len"] = doc_len
        indexed_record["title_text"] = normalize_sparse_text(
            f"{record.get('article') or ''} {record.get('section') or ''}"
        )
        indexed_record["book_text"] = normalize_sparse_text(record.get("book") or "")
        docs.append(indexed_record)
        doc_lengths.append(doc_len)
        for token, freq in term_freqs.items():
            postings[token][doc_id] = freq

    avgdl = (sum(doc_lengths) / len(doc_lengths)) if doc_lengths else 1.0
    return {
        "schema_version": 1,
        "kind": "light_sparse_paragraph_index",
        "docs": docs,
        "postings": _pack_postings(postings, len(docs)),
        "doc_count": len(docs),
        "avgdl": avgdl or 1.0,
        "min_df": LIGHT_SPARSE_MIN_DF,
        "max_df_ratio": LIGHT_SPARSE_MAX_DF_RATIO,
    }


def write_light_sparse_paragraph_index(
    output_path: str | Path | None = None,
    paragraph_cache_path: str | Path | None = None,
) -> dict:
    output_path = Path(output_path) if output_path else _light_sparse_index_path(paragraph_cache_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index = build_light_sparse_paragraph_index(paragraph_cache_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(output_path)
    return {
        "path": str(output_path),
        "doc_count": index.get("doc_count", 0),
        "posting_terms": len(index.get("postings") or {}),
        "size_bytes": output_path.stat().st_size,
    }


def load_light_sparse_paragraph_index(path: str | Path | None = None) -> dict | None:
    cache_path = _paragraph_cache_path(path)
    index_path = _light_sparse_index_path(path)
    if os.getenv("SEMANTIC_USE_LIGHT_SPARSE_INDEX", "1").lower() not in {"1", "true", "yes", "on"}:
        return None
    if not index_path.exists() or not cache_path.exists():
        return None
    try:
        if index_path.stat().st_mtime < cache_path.stat().st_mtime:
            return None
        with index_path.open("rb") as f:
            index = pickle.load(f)
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
        return None
    if not isinstance(index, dict) or index.get("kind") != "light_sparse_paragraph_index":
        return None
    return index


@lru_cache(maxsize=2)
def load_sparse_paragraph_index(path: str | Path | None = None) -> dict:
    light_index = load_light_sparse_paragraph_index(path)
    if light_index is not None:
        return light_index

    _, by_source = load_paragraph_records(path)
    docs = []
    postings = defaultdict(list)
    doc_lengths = []
    avgdl = 0.0

    for source_records in by_source.values():
        for record in source_records:
            doc = paragraph_record_to_document(record)
            metadata = doc.metadata
            sparse_text = " ".join(
                str(part or "")
                for part in (
                    metadata.get("book"),
                    metadata.get("article"),
                    metadata.get("section"),
                    doc.page_content,
                )
            )
            tokens = sparse_query_tokens(sparse_text)
            if not tokens:
                continue

            term_freqs = Counter(tokens)
            title_text = normalize_sparse_text(
                f"{metadata.get('article') or ''} {metadata.get('section') or ''}"
            )
            docs.append(
                {
                    "document": doc,
                    "term_freqs": term_freqs,
                    "doc_len": sum(term_freqs.values()),
                    "title_text": title_text,
                    "content_text": normalize_sparse_text(doc.page_content),
                    "book_text": normalize_sparse_text(metadata.get("book") or ""),
                }
            )
            doc_id = len(docs) - 1
            doc_lengths.append(docs[doc_id]["doc_len"])
            for token, freq in term_freqs.items():
                postings[token].append((doc_id, freq))

    if doc_lengths:
        avgdl = sum(doc_lengths) / len(doc_lengths)

    return {
        "docs": docs,
        "postings": dict(postings),
        "doc_count": len(docs),
        "avgdl": avgdl or 1.0,
    }


def _child_splitter(
    chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["。", "；", "！", "？", "\n\n", "\n", "，", ""],
    )


def build_semantic_child_documents(
    records: list[dict],
    chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
) -> list[Document]:
    splitter = _child_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    child_docs = []

    for record in records:
        parent_doc = paragraph_record_to_document(record)
        parent_text = parent_doc.page_content
        if not parent_text:
            continue

        split_docs = splitter.split_documents([parent_doc])
        cursor = 0
        total = len(split_docs)
        for index, doc in enumerate(split_docs, start=1):
            chunk_text = str(doc.page_content or "").strip()
            if not chunk_text:
                continue

            char_start = parent_text.find(chunk_text, max(cursor, 0))
            if char_start < 0:
                char_start = parent_text.find(chunk_text)
            char_end = char_start + len(chunk_text) if char_start >= 0 else None
            if char_end is not None:
                cursor = char_end

            metadata = dict(parent_doc.metadata)
            metadata.update(
                {
                    "retrieval_unit": "semantic_child",
                    "parent_paragraph_id": record.get("paragraph_id"),
                    "child_chunk_index": index,
                    "child_chunk_total": total,
                    "child_char_start": char_start,
                    "child_char_end": char_end,
                    "child_chunk_size": len(chunk_text),
                }
            )
            child_docs.append(Document(page_content=chunk_text, metadata=metadata))

    return child_docs


def paragraph_window_document(
    paragraph_id: str,
    window: int = DEFAULT_PARENT_WINDOW,
    path: str | Path | None = None,
    match_doc: Document | None = None,
) -> Document | None:
    by_id, by_source = load_paragraph_records(path)
    center = by_id.get(paragraph_id)
    if not center:
        return None

    source = center.get("source")
    source_records = by_source.get(source) or []
    center_index = int(center.get("paragraph_index") or 0)
    start_index = max(1, center_index - max(window, 0))
    end_index = center_index + max(window, 0)
    window_records = [
        record
        for record in source_records
        if start_index <= int(record.get("paragraph_index") or 0) <= end_index
    ]
    if not window_records:
        window_records = [center]

    text = "\n\n".join(
        str(record.get("paragraph_text") or "").strip()
        for record in window_records
        if str(record.get("paragraph_text") or "").strip()
    )
    if not text:
        return None

    doc = paragraph_record_to_document(center)
    doc.page_content = text
    doc.metadata["retrieval_unit"] = "paragraph_window"
    doc.metadata["paragraph_window_size"] = window
    doc.metadata["parent_paragraph_id"] = paragraph_id
    doc.metadata["window_paragraph_ids"] = [record.get("paragraph_id") for record in window_records]
    doc.metadata["window_paragraph_start"] = window_records[0].get("paragraph_index")
    doc.metadata["window_paragraph_end"] = window_records[-1].get("paragraph_index")
    doc.metadata["pdf_page"] = window_records[0].get("pdf_page_start")
    doc.metadata["pdf_page_end"] = window_records[-1].get("pdf_page_end")
    doc.metadata["printed_page"] = window_records[0].get("printed_page_start")
    doc.metadata["printed_page_end"] = window_records[-1].get("printed_page_end")
    doc.metadata["citation_page"] = window_records[0].get("citation_page_start")
    doc.metadata["citation_page_end"] = window_records[-1].get("citation_page_end")

    if match_doc is not None:
        for key in (
            "match_type",
            "confidence",
            "child_chunk_index",
            "child_chunk_total",
            "child_char_start",
            "child_char_end",
            "paragraph_query_overlap_score",
            "retrieval_backend",
            "milvus_id",
            "vector_score",
        ):
            if key in (match_doc.metadata or {}):
                doc.metadata[key] = match_doc.metadata.get(key)

    return doc


def expand_semantic_parent_docs(
    docs: list[Document],
    window: int = DEFAULT_PARENT_WINDOW,
    path: str | Path | None = None,
) -> list[Document]:
    expanded = []
    seen = set()

    for doc in docs or []:
        metadata = dict(doc.metadata or {})
        parent_id = metadata.get("parent_paragraph_id")
        if metadata.get("retrieval_unit") in {"paragraph_child", "semantic_child", "milvus_passage"} and parent_id:
            parent_doc = paragraph_window_document(parent_id, window=window, path=path, match_doc=doc)
            if parent_doc is not None:
                if (
                    metadata.get("retrieval_unit") == "milvus_passage"
                    and looks_like_index_or_toc_text(parent_doc.page_content, parent_doc.metadata)
                    and not looks_like_index_or_toc_text(doc.page_content, metadata)
                ):
                    parent_doc = None

            if parent_doc is not None:
                if metadata.get("retrieval_unit") == "milvus_passage":
                    parent_doc.metadata["retrieval_unit"] = "milvus_paragraph_window"
                elif metadata.get("retrieval_unit") == "semantic_child":
                    parent_doc.metadata["retrieval_unit"] = "semantic_parent"
                key = (
                    parent_doc.metadata.get("source"),
                    parent_doc.metadata.get("parent_paragraph_id"),
                    parent_doc.metadata.get("window_paragraph_start"),
                    parent_doc.metadata.get("window_paragraph_end"),
                )
                if key in seen:
                    continue
                seen.add(key)
                expanded.append(parent_doc)
                continue

        key = (
            metadata.get("source"),
            metadata.get("paragraph_id") or metadata.get("parent_paragraph_id") or metadata.get("pdf_page"),
            metadata.get("retrieval_unit"),
        )
        if key in seen:
            continue
        seen.add(key)
        expanded.append(doc)

    return expanded


def sparse_retrieve_documents(
    query: str,
    limit: int = SPARSE_TOP_K,
    path: str | Path | None = None,
) -> list[Document]:
    index = load_sparse_paragraph_index(path)
    query_tokens = sparse_query_tokens(query)
    query_phrases = sparse_query_phrases(query)
    if not query_tokens:
        return []

    scores = defaultdict(float)
    k1 = 1.5
    b = 0.75
    doc_count = max(int(index.get("doc_count") or 0), 1)
    avgdl = float(index.get("avgdl") or 1.0)

    for token in query_tokens:
        postings = index["postings"].get(token) or []
        if not postings:
            continue
        doc_freq = len(postings[0]) if isinstance(postings, tuple) else len(postings)
        idf = math.log(1 + ((doc_count - doc_freq + 0.5) / (doc_freq + 0.5)))
        for doc_id, term_freq in _iter_postings(postings):
            entry = index["docs"][doc_id]
            doc_len = entry["doc_len"]
            denom = term_freq + k1 * (1 - b + b * (doc_len / avgdl))
            scores[doc_id] += idf * ((term_freq * (k1 + 1)) / max(denom, 1e-9))

    rough_ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if SPARSE_RERANK_POOL > 0:
        rough_ranked = rough_ranked[:max(limit, SPARSE_RERANK_POOL)]

    ranked = []
    normalized_query = normalize_sparse_text(query)
    for rank_index, (doc_id, score) in enumerate(rough_ranked, start=1):
        entry = index["docs"][doc_id]
        title_text = entry["title_text"]
        inspect_content = SPARSE_CONTENT_RERANK_POOL <= 0 or rank_index <= SPARSE_CONTENT_RERANK_POOL
        content_text = entry.get("content_text")
        if content_text is None and inspect_content:
            content_text = normalize_sparse_text(entry.get("paragraph_text") or "")
        book_text = entry["book_text"]
        title_token_hits = 0
        content_token_hits = 0
        title_phrase_hits = 0
        content_phrase_hits = 0
        book_phrase_hits = 0
        for token in query_tokens:
            if token and token in title_text:
                title_token_hits += 1
                score += 2.2
            elif inspect_content and token and token in content_text:
                content_token_hits += 1
                score += 0.6
        if normalized_query and normalized_query in title_text:
            score += 6.0
        for phrase in query_phrases:
            if len(phrase) < 2:
                continue
            if phrase in title_text:
                title_phrase_hits += 1
                score += min(10.0, 2.2 * len(phrase))
            elif phrase in book_text:
                book_phrase_hits += 1
                score += min(6.0, 1.2 * len(phrase))
            elif inspect_content and len(phrase) <= 12 and phrase in content_text:
                content_phrase_hits += 1
                score += min(8.0, 1.5 * len(phrase))
        ranked.append(
            (
                score,
                entry.get("document") or entry,
                {
                    "sparse_title_token_hits": title_token_hits,
                    "sparse_content_token_hits": content_token_hits,
                    "sparse_title_phrase_hits": title_phrase_hits,
                    "sparse_content_phrase_hits": content_phrase_hits,
                    "sparse_book_phrase_hits": book_phrase_hits,
                },
            )
        )

    ranked.sort(key=lambda item: item[0], reverse=True)

    docs = []
    for score, doc, hit_info in ranked[:limit]:
        if isinstance(doc, Document):
            clone = Document(page_content=doc.page_content, metadata=dict(doc.metadata or {}))
        else:
            clone = paragraph_record_to_document(doc)
        clone.metadata["match_type"] = "sparse_candidate"
        clone.metadata["sparse_score"] = round(score, 4)
        clone.metadata.update(hit_info)
        docs.append(clone)
    return docs
