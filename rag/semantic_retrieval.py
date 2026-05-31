from __future__ import annotations

import os
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.paragraph_cache import paragraph_record_to_document, read_paragraph_cache


DEFAULT_PARAGRAPH_CACHE_PATH = Path(
    os.getenv("PARAGRAPH_CACHE_PATH", "data/paragraph_cache_core.jsonl")
)
DEFAULT_CHILD_CHUNK_SIZE = int(os.getenv("SEMANTIC_CHILD_CHUNK_SIZE", "180"))
DEFAULT_CHILD_CHUNK_OVERLAP = int(os.getenv("SEMANTIC_CHILD_CHUNK_OVERLAP", "40"))
DEFAULT_PARENT_WINDOW = int(os.getenv("SEMANTIC_PARENT_WINDOW", "1"))
SPARSE_TOP_K = int(os.getenv("SEMANTIC_SPARSE_TOP_K", "24"))
SPARSE_STOP_PHRASES = (
    "请解释",
    "请说明",
    "请问",
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
    "在马克思那里",
)


def _paragraph_cache_path(path: str | Path | None = None) -> Path:
    return Path(path) if path else DEFAULT_PARAGRAPH_CACHE_PATH


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


@lru_cache(maxsize=2)
def load_sparse_paragraph_index(path: str | Path | None = None) -> dict:
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
                    "retrieval_unit": "paragraph_child",
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
        if metadata.get("retrieval_unit") == "paragraph_child" and parent_id:
            parent_doc = paragraph_window_document(parent_id, window=window, path=path, match_doc=doc)
            if parent_doc is not None:
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
        doc_freq = len(postings)
        idf = math.log(1 + ((doc_count - doc_freq + 0.5) / (doc_freq + 0.5)))
        for doc_id, term_freq in postings:
            entry = index["docs"][doc_id]
            doc_len = entry["doc_len"]
            denom = term_freq + k1 * (1 - b + b * (doc_len / avgdl))
            scores[doc_id] += idf * ((term_freq * (k1 + 1)) / max(denom, 1e-9))

    ranked = []
    normalized_query = normalize_sparse_text(query)
    for doc_id, score in scores.items():
        entry = index["docs"][doc_id]
        title_text = entry["title_text"]
        content_text = entry["content_text"]
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
            elif token and token in content_text:
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
            elif phrase in content_text:
                content_phrase_hits += 1
                score += min(8.0, 1.5 * len(phrase))
        ranked.append(
            (
                score,
                entry["document"],
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
        clone = Document(page_content=doc.page_content, metadata=dict(doc.metadata or {}))
        clone.metadata["match_type"] = "sparse_candidate"
        clone.metadata["sparse_score"] = round(score, 4)
        clone.metadata.update(hit_info)
        docs.append(clone)
    return docs
