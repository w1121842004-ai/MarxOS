import json
from pathlib import Path

from langchain_core.documents import Document

from marxos_retrieval_constraints import (
    candidate_pdf_pages_from_metadata,
    concept_seed_queries,
    constraints_from_query,
    metadata_matches_constraints,
    page_in_expected_range,
    topic_seed_queries,
)
from marxos_retrieval_ranking import (
    annotate_docs_with_constraints,
    dedupe_documents,
    diversify_documents,
    rerank_documents,
    select_topic_documents,
)


def _helper(ctx, name):
    return ctx[name]


def strict_title_cache_documents(query, constraints, limit, ctx):
    clean_text = _helper(ctx, "clean_text")
    normalize_for_match = _helper(ctx, "normalize_for_match")
    active_concept_terms = _helper(ctx, "active_concept_terms")
    as_int = _helper(ctx, "as_int")
    find_pdf_page_by_printed_page = _helper(ctx, "find_pdf_page_by_printed_page")
    OCR_CACHE_DIR = ctx["OCR_CACHE_DIR"]
    CONCEPT_PREFERRED_MARKERS = ctx["CONCEPT_PREFERRED_MARKERS"]

    docs = []
    seen = set()
    cache_root = Path(OCR_CACHE_DIR)
    title_norm = normalize_for_match(constraints.get("title") or "")
    concept_terms = active_concept_terms(query)
    concept_markers = []
    for term in concept_terms:
        concept_markers.append(term)
        concept_markers.extend(CONCEPT_PREFERRED_MARKERS.get(term, []))
    concept_marker_norms = {
        normalize_for_match(marker)
        for marker in concept_markers
        if normalize_for_match(marker)
        and normalize_for_match(marker) != title_norm
        and normalize_for_match(marker) not in title_norm
    }

    for entry in constraints.get("entries") or []:
        source = entry.get("source")
        start_printed_page = as_int(entry.get("start_page"))
        end_printed_page = as_int(entry.get("end_page"))
        if not source or start_printed_page is None or end_printed_page is None:
            continue

        stem = source.replace(".pdf", "")
        for printed_page in range(start_printed_page, end_printed_page + 1):
            pdf_page = find_pdf_page_by_printed_page(source, printed_page)
            if pdf_page is None:
                continue

            path = cache_root / stem / f"page_{pdf_page}.json"
            if not path.exists():
                continue

            with path.open("r", encoding="utf-8") as f:
                page = json.load(f)

            text = clean_text(page.get("cleaned_text"), "")
            if not text:
                continue
            if concept_marker_norms:
                page_title_norm = normalize_for_match(page.get("title_candidate") or "")
                text_norm = normalize_for_match(text)
                if not any(marker in page_title_norm or marker in text_norm for marker in concept_marker_norms):
                    continue

            metadata = {
                "book": page.get("book_title") or source,
                "article": entry.get("classic_title") or entry.get("article") or constraints.get("title"),
                "section": entry.get("classic_title") or entry.get("article") or constraints.get("title"),
                "page": printed_page,
                "printed_page": printed_page,
                "pdf_page": page.get("page_num") or pdf_page,
                "citation_page": printed_page,
                "citation_page_type": "printed_page",
                "source": source,
                "ocr": True,
                "classic_title": entry.get("classic_title") or constraints.get("title"),
                "work_title": entry.get("classic_title") or constraints.get("title"),
                "locator_title": entry.get("classic_title") or constraints.get("title"),
                "classic_author": entry.get("classic_author"),
                "classic_work_type": entry.get("classic_work_type"),
                "entry_type": entry.get("entry_type"),
                "match_type": "cache_backstop",
            }
            key = (source, printed_page)
            if key in seen:
                continue
            seen.add(key)
            docs.append(Document(page_content=text, metadata=metadata))

    if not docs:
        return []

    ranked = rerank_documents(query, docs, constraints, ctx)
    return ranked[:limit]


def locator_backstop_documents(constraints, limit):
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
            f"定位提示：该问题对应到《{title}》"
            f"，PDF第{entry.get('start_page')}-{entry.get('end_page')}页范围内核对。"
        )
        docs.append(Document(page_content=content, metadata=metadata))
        if len(docs) >= limit:
            break
    return docs


def append_locator_backstops(docs, constraints, k, ctx):
    normalize_for_match = _helper(ctx, "normalize_for_match")
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


def topic_constrained_candidates(query, db, constraints, fetch_k, ctx):
    candidates = []
    seeds = topic_seed_queries(query, constraints, ctx)
    per_seed_k = max(24, fetch_k // max(len(seeds), 1))
    for seed in seeds:
        candidates.extend(db.similarity_search(seed, k=per_seed_k))

    candidates = [
        doc for doc in dedupe_documents(candidates, ctx)
        if metadata_matches_constraints(doc.metadata, constraints)
    ]

    if constraints.get("page_ranges"):
        ranged_candidates = [
            doc for doc in candidates
            if page_in_expected_range(doc.metadata, constraints, ctx)
        ]
        if ranged_candidates:
            candidates = ranged_candidates
    return candidates


def concept_constrained_candidates(query, db, constraints, fetch_k, ctx):
    candidates = []
    seeds = concept_seed_queries(query, constraints, ctx)
    per_seed_k = max(18, fetch_k // max(len(seeds), 1))
    for seed in seeds:
        candidates.extend(db.similarity_search(seed, k=per_seed_k))

    candidates = [
        doc for doc in dedupe_documents(candidates, ctx)
        if metadata_matches_constraints(doc.metadata, constraints)
    ]

    if constraints.get("page_ranges"):
        ranged_candidates = [
            doc for doc in candidates
            if page_in_expected_range(doc.metadata, constraints, ctx)
        ]
        if ranged_candidates:
            candidates = ranged_candidates
    return candidates


def retrieve_documents(query, db, k, allow_exact_quote, ctx):
    normalize_for_match = _helper(ctx, "normalize_for_match")
    active_concept_terms = _helper(ctx, "active_concept_terms")
    exact_quote_lookup = _helper(ctx, "exact_quote_lookup")
    is_quote_lookup_query = _helper(ctx, "is_quote_lookup_query")
    classify_query = _helper(ctx, "classify_query")
    enrich_concept_metadata = _helper(ctx, "enrich_concept_metadata")
    OCR_CACHE_DIR = ctx["OCR_CACHE_DIR"]
    CLASSIC_SAYING_QUOTE_SEEDS = ctx["CLASSIC_SAYING_QUOTE_SEEDS"]
    CLASSIC_SAYING_QUERY_SEEDS = ctx["CLASSIC_SAYING_QUERY_SEEDS"]
    is_classic_sayings_query = _helper(ctx, "is_classic_sayings_query")
    clean_text = _helper(ctx, "clean_text")

    constraints = constraints_from_query(query, ctx)
    normalized_query = normalize_for_match(query)

    if constraints.get("strict_title") and "无产阶级专政" in normalized_query:
        return locator_backstop_documents(constraints, limit=k)

    if allow_exact_quote and is_quote_lookup_query(query):
        exact_docs = exact_quote_lookup(query, OCR_CACHE_DIR, limit=k)
        if exact_docs:
            docs = annotate_docs_with_constraints(exact_docs, constraints, ctx)
            return append_locator_backstops(docs, constraints, k, ctx)

    fetch_k = max(120 if constraints or active_concept_terms(query) else 30, k * 12)

    if constraints.get("sources"):
        if active_concept_terms(query):
            candidates = concept_constrained_candidates(query, db, constraints, fetch_k, ctx)
        elif constraints.get("topic_id"):
            candidates = topic_constrained_candidates(query, db, constraints, fetch_k, ctx)
        else:
            candidates = db.similarity_search(query, k=fetch_k)
            candidates = [
                doc for doc in candidates
                if metadata_matches_constraints(doc.metadata, constraints)
            ]

        if constraints.get("page_ranges"):
            ranged_candidates = [
                doc for doc in candidates
                if page_in_expected_range(doc.metadata, constraints, ctx)
            ]
            if ranged_candidates:
                candidates = ranged_candidates

        if constraints.get("strict_title") and constraints.get("page_ranges"):
            candidates = [
                doc for doc in candidates
                if page_in_expected_range(doc.metadata, constraints, ctx)
            ]

        if constraints.get("strict_title") and active_concept_terms(query):
            candidates = dedupe_documents(
                candidates + strict_title_cache_documents(query, constraints, fetch_k, ctx),
                ctx,
            )
            if constraints.get("page_ranges"):
                ranged_candidates = [
                    doc for doc in candidates
                    if page_in_expected_range(doc.metadata, constraints, ctx)
                ]
                if ranged_candidates:
                    candidates = ranged_candidates

        if not candidates:
            title_query = constraints.get("title") or query
            if constraints.get("topic_id"):
                candidates = topic_constrained_candidates(title_query, db, constraints, fetch_k, ctx)
            else:
                candidates = [
                    doc for doc in db.similarity_search(title_query, k=fetch_k)
                    if metadata_matches_constraints(doc.metadata, constraints)
                ]
            if constraints.get("page_ranges"):
                ranged_candidates = [
                    doc for doc in candidates
                    if page_in_expected_range(doc.metadata, constraints, ctx)
                ]
                if ranged_candidates:
                    candidates = ranged_candidates
            if constraints.get("strict_title") and constraints.get("page_ranges"):
                candidates = [
                    doc for doc in candidates
                    if page_in_expected_range(doc.metadata, constraints, ctx)
                ]

        if not candidates:
            if constraints.get("strict_title"):
                cache_docs = strict_title_cache_documents(query, constraints, k, ctx)
                if cache_docs:
                    cache_docs = annotate_docs_with_constraints(cache_docs, constraints, ctx)
                    return append_locator_backstops(cache_docs, constraints, k, ctx)
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
        docs = diversify_documents(
            candidates,
            k,
            ctx,
            max_per_source=2,
            max_per_article=1,
            min_distinct_sources=constraints.get("min_distinct_sources", 0),
        )
        docs = annotate_docs_with_constraints(docs, constraints, ctx)
        return append_locator_backstops(docs, constraints, k, ctx)

    ranked_docs = rerank_documents(query, candidates, constraints, ctx)
    if constraints.get("topic_id"):
        docs = select_topic_documents(ranked_docs, constraints, k, ctx)
    elif (not constraints and classify_query(query) == "rag_answer" and k > 5) or constraints.get("min_distinct_sources"):
        docs = diversify_documents(
            ranked_docs,
            k,
            ctx,
            min_distinct_sources=constraints.get("min_distinct_sources", 0),
        )
    else:
        docs = ranked_docs[:k]

    if classify_query(query) == "concept_explain":
        docs = enrich_concept_metadata(query, docs)

    if allow_exact_quote and is_quote_lookup_query(query):
        for doc in docs:
            doc.metadata["match_type"] = "vector_candidate"
            doc.metadata["confidence"] = 0.0

    docs = annotate_docs_with_constraints(docs, constraints, ctx)
    return append_locator_backstops(docs, constraints, k, ctx)


def refine_doc_citation_page_for_query(doc, query, ctx):
    load_ocr_page_text = _helper(ctx, "load_ocr_page_text")
    page_match_score = _helper(ctx, "page_match_score")
    infer_printed_page_from_ocr_cache = _helper(ctx, "infer_printed_page_from_ocr_cache")
    metadata = dict(doc.metadata or {})
    source = metadata.get("source")
    candidate_pages = candidate_pdf_pages_from_metadata(metadata, ctx)
    if not source or len(candidate_pages) <= 1:
        return doc

    scored_pages = []
    for pdf_page in candidate_pages:
        text = load_ocr_page_text(source, pdf_page)
        score = page_match_score(doc.page_content, text) * 2 + page_match_score(query, text)
        printed = infer_printed_page_from_ocr_cache({"source": source, "pdf_page": pdf_page})
        scored_pages.append((score, printed is not None, pdf_page, printed))

    scored_pages.sort(reverse=True)
    best_score, _has_printed, best_pdf_page, best_printed_page = scored_pages[0]
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


def refine_docs_citation_pages_for_query(docs, query, ctx):
    return [refine_doc_citation_page_for_query(doc, query, ctx) for doc in docs]


def retrieve_paragraph_documents(query, db, k, ctx):
    docs = retrieve_documents(query, db, k=k, allow_exact_quote=False, ctx=ctx)
    for doc in docs:
        doc.metadata.setdefault("retrieval_unit", "paragraph")
        doc.metadata.setdefault("match_type", "paragraph_vector_candidate")
    return docs


def retrieve_dual_documents(query, chunk_db, paragraph_db, k, ctx):
    return {
        "chunk": retrieve_documents(query, chunk_db, k=k, allow_exact_quote=True, ctx=ctx),
        "paragraph": retrieve_paragraph_documents(query, paragraph_db, k=k, ctx=ctx),
    }


def merge_prefer_paragraph_docs(paragraph_docs, chunk_docs, limit, ctx):
    clean_text = _helper(ctx, "clean_text")
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


def filter_paragraph_docs_by_text_overlap(query, docs, ctx, limit=None):
    page_match_score = _helper(ctx, "page_match_score")
    filtered = []
    for doc in docs or []:
        score = page_match_score(query, doc.page_content)
        if score <= 0:
            continue
        doc.metadata["paragraph_query_overlap_score"] = score
        filtered.append(doc)
    filtered.sort(key=lambda item: item.metadata.get("paragraph_query_overlap_score", 0), reverse=True)
    return filtered[:limit] if limit else filtered
