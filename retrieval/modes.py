import json
from pathlib import Path

from langchain_core.documents import Document

from retrieval.constraints import (
    _helper,
    candidate_pdf_pages_from_metadata,
    collection_requested,
    concept_seed_queries,
    constraints_from_query,
    controlled_multi_queries,
    is_me_source,
    metadata_matches_constraints,
    page_in_expected_range,
    source_priority,
    topic_seed_queries,
)
from retrieval.ranking import (
    annotate_docs_with_constraints,
    collapse_content_near_duplicates,
    dedupe_documents,
    diversify_documents,
    rerank_documents,
    select_topic_documents,
)


def _candidate_key(doc, ctx):
    clean_text = _helper(ctx, "clean_text")
    metadata = doc.metadata or {}
    return (
        metadata.get("source"),
        metadata.get("page"),
        metadata.get("printed_page"),
        metadata.get("citation_page"),
        clean_text(metadata.get("article") or metadata.get("section"), ""),
        clean_text(doc.page_content, "")[:120],
    )


def _content_signature(doc, ctx):
    normalize_for_match = _helper(ctx, "normalize_for_match")
    content = normalize_for_match(doc.page_content or "")
    if not content:
        return ""
    return content[:96]


def _signature_preference(doc, rank, ctx):
    is_noisy_article_title = _helper(ctx, "is_noisy_article_title")
    score_document_quality = _helper(ctx, "score_document_quality")
    clean_text = _helper(ctx, "clean_text")
    metadata = doc.metadata or {}
    article = clean_text(metadata.get("article") or metadata.get("section"), "")
    score = score_document_quality(metadata, clean_text(doc.page_content, ""))
    if not is_noisy_article_title(article):
        score += 25
    if metadata.get("page_type") not in {"toc", "title_page"}:
        score += 8
    score -= rank * 0.2
    return score


def _doc_source(doc):
    return str((doc.metadata or {}).get("source") or "").lower()


def _prefer_source_order(docs, query, ctx):
    if not docs:
        return docs
    return sorted(
        docs,
        key=lambda doc: (
            source_priority(_doc_source(doc), query, ctx),
            int((doc.metadata or {}).get("hybrid_dense_rank") or 9999),
            int((doc.metadata or {}).get("hybrid_sparse_rank") or 9999),
        ),
    )


def _sparse_allowed(doc, query, constraints, ctx):
    metadata = doc.metadata or {}
    if constraints.get("sources") and not metadata_matches_constraints(metadata, constraints):
        return False
    if collection_requested(query, ctx) == "non_me":
        return True
    source = _doc_source(doc)
    return not source or is_me_source(source)


def _has_narrow_page_ranges(constraints, max_width=3):
    ranges = []
    for source_ranges in (constraints.get("page_ranges") or {}).values():
        ranges.extend(source_ranges or [])
    if not ranges:
        return False
    return all(
        start is not None and end is not None and (end - start) <= max_width
        for start, end in ranges
    )


def _hybrid_merge_candidates(query, dense_candidates, constraints, fetch_k, ctx):
    hybrid_retrieval_enabled = _helper(ctx, "hybrid_retrieval_enabled")
    controlled_multi_queries = _helper(ctx, "controlled_multi_queries")
    sparse_retrieve_documents = _helper(ctx, "sparse_retrieve_documents")

    if not hybrid_retrieval_enabled():
        return dense_candidates

    dense_candidates = list(dense_candidates or [])
    sparse_candidates = []
    for variant in controlled_multi_queries(query, constraints, ctx):
        sparse_candidates.extend(
            sparse_retrieve_documents(variant, limit=max(fetch_k // 3, 12))
        )
    sparse_candidates = [
        doc for doc in sparse_candidates
        if _sparse_allowed(doc, query, constraints, ctx)
    ]
    if not sparse_candidates:
        return _prefer_source_order(dense_candidates, query, ctx)

    ranked = {}
    signature_to_key = {}
    signature_scores = {}
    for rank, doc in enumerate(dense_candidates, start=1):
        key = _candidate_key(doc, ctx)
        clone = Document(page_content=doc.page_content, metadata=dict(doc.metadata or {}))
        clone.metadata["hybrid_dense_rank"] = rank
        ranked[key] = clone
        signature = _content_signature(clone, ctx)
        if signature:
            preference = _signature_preference(clone, rank, ctx)
            if preference > signature_scores.get(signature, float("-inf")):
                signature_scores[signature] = preference
                signature_to_key[signature] = key

    for rank, doc in enumerate(sparse_candidates, start=1):
        key = _candidate_key(doc, ctx)
        target = ranked.get(key)
        if target is None:
            signature = _content_signature(doc, ctx)
            dense_key = signature_to_key.get(signature) if signature else None
            if dense_key is not None:
                target = ranked[dense_key]
                target.metadata["hybrid_sparse_merged"] = "content_signature"

        if target is not None:
            target.metadata["hybrid_sparse_rank"] = rank
            for key, value in (doc.metadata or {}).items():
                if key.startswith("sparse_") and value is not None:
                    target.metadata[key] = value
            target.metadata["hybrid_sparse_hit"] = True
            continue

        clone = Document(page_content=doc.page_content, metadata=dict(doc.metadata or {}))
        clone.metadata["hybrid_sparse_rank"] = rank
        clone.metadata["hybrid_sparse_hit"] = True
        ranked[key] = clone
        signature = _content_signature(clone, ctx)
        if signature and signature not in signature_to_key:
            signature_to_key[signature] = key

    merged = []
    for doc in ranked.values():
        dense_rank = doc.metadata.get("hybrid_dense_rank")
        sparse_rank = doc.metadata.get("hybrid_sparse_rank")
        rrf = 0.0
        if dense_rank:
            rrf += 1 / (60 + dense_rank)
        if sparse_rank:
            sparse_weight = 1.15 if dense_rank else 0.72
            rrf += sparse_weight / (55 + sparse_rank)
        doc.metadata["hybrid_rrf_score"] = round(rrf, 6)
        if dense_rank and sparse_rank:
            doc.metadata["hybrid_source"] = "fused"
        elif sparse_rank:
            doc.metadata["hybrid_source"] = "sparse"
        else:
            doc.metadata["hybrid_source"] = "dense"
        merged.append(doc)

    merged.sort(
        key=lambda item: (
            item.metadata.get("hybrid_rrf_score", 0),
            -int(item.metadata.get("hybrid_dense_rank") or 9999),
            item.metadata.get("sparse_score", 0) or 0,
        ),
        reverse=True,
    )
    merged = dedupe_documents(merged, ctx)
    if constraints.get("sources"):
        merged = [
            doc for doc in merged
            if metadata_matches_constraints(doc.metadata, constraints)
        ]
    if constraints.get("page_ranges"):
        ranged = [
            doc for doc in merged
            if page_in_expected_range(doc.metadata, constraints, ctx)
        ]
        if ranged:
            merged = ranged
    return _prefer_source_order(merged, query, ctx)


def _controlled_dense_candidates(query, db, constraints, fetch_k, ctx):
    controlled_multi_queries = _helper(ctx, "controlled_multi_queries")
    candidates = []
    variants = controlled_multi_queries(query, constraints, ctx)
    per_query_k = max(18, fetch_k // max(len(variants), 1))
    for variant in variants:
        candidates.extend(db.similarity_search(variant, k=per_query_k))
    return dedupe_documents(candidates, ctx)


def strict_title_cache_documents(query, constraints, limit, ctx):
    clean_text = _helper(ctx, "clean_text")
    normalize_for_match = _helper(ctx, "normalize_for_match")
    normalize_digit_text = _helper(ctx, "normalize_digit_text")
    active_concept_terms = _helper(ctx, "active_concept_terms")
    as_int = _helper(ctx, "as_int")
    find_pdf_page_by_printed_page = _helper(ctx, "find_pdf_page_by_printed_page")
    infer_printed_page_from_ocr_cache = _helper(ctx, "infer_printed_page_from_ocr_cache")
    load_ocr_page_text = _helper(ctx, "load_ocr_page_text")
    OCR_CACHE_DIR = ctx["OCR_CACHE_DIR"]
    CONCEPT_PREFERRED_MARKERS = ctx["CONCEPT_PREFERRED_MARKERS"]

    docs = []
    seen = set()
    cache_root = Path(OCR_CACHE_DIR)
    title_norm = normalize_for_match(constraints.get("title") or "")
    concept_terms = [] if constraints.get("high_precision_locator") else active_concept_terms(query)
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
            candidate_pdf_pages = []
            entry_pdf_page = as_int(entry.get("pdf_page"))
            if entry_pdf_page is not None:
                candidate_pdf_pages.append(entry_pdf_page)
            entry_pdf_end_page = as_int(entry.get("pdf_end_page"))
            if (
                entry_pdf_page is not None
                and entry_pdf_end_page is not None
                and entry_pdf_end_page >= entry_pdf_page
                and (end_printed_page - start_printed_page) <= 80
            ):
                offset_page = entry_pdf_page + (printed_page - start_printed_page)
                if entry_pdf_page <= offset_page <= entry_pdf_end_page:
                    candidate_pdf_pages.append(offset_page)
            mapped_pdf_page = find_pdf_page_by_printed_page(source, printed_page)
            if mapped_pdf_page is not None:
                candidate_pdf_pages.append(mapped_pdf_page)
            if constraints.get("explicit_volume") or constraints.get("high_precision_locator"):
                candidate_pdf_pages.append(printed_page)

            for pdf_page in dict.fromkeys(candidate_pdf_pages):
                path = cache_root / stem / f"page_{pdf_page}.json"
                page = {}
                if path.exists():
                    with path.open("r", encoding="utf-8") as f:
                        page = json.load(f)

                text = clean_text(
                    page.get("cleaned_text") if page else load_ocr_page_text(source, pdf_page),
                    "",
                )
                if not text:
                    continue
                text = normalize_digit_text(text)
                locator_quote = clean_text(entry.get("locator_quote"), "")
                if (
                    constraints.get("high_precision_locator")
                    and locator_quote
                    and normalize_for_match(locator_quote) not in normalize_for_match(text)
                ):
                    text = f"{text}\n\n引用校正：{locator_quote}"
                if concept_marker_norms:
                    page_title_norm = normalize_for_match(page.get("title_candidate") or "")
                    text_norm = normalize_for_match(text)
                    if not any(marker in page_title_norm or marker in text_norm for marker in concept_marker_norms):
                        continue

                inferred_printed_page = infer_printed_page_from_ocr_cache(
                    {"source": source, "pdf_page": pdf_page}
                )
                citation_page = inferred_printed_page if inferred_printed_page is not None else printed_page
                citation_page_type = "printed_page" if inferred_printed_page is not None else "pdf_page"

                metadata = {
                    "book": page.get("book_title") or source,
                    "article": entry.get("classic_title") or entry.get("article") or constraints.get("title"),
                    "section": entry.get("classic_title") or entry.get("article") or constraints.get("title"),
                    "page": pdf_page,
                    "printed_page": inferred_printed_page,
                    "pdf_page": page.get("page_num") or pdf_page,
                    "citation_page": citation_page,
                    "citation_page_type": citation_page_type,
                    "source": source,
                    "ocr": True,
                    "classic_title": entry.get("classic_title") or constraints.get("title"),
                    "work_title": entry.get("classic_title") or constraints.get("title"),
                    "locator_title": entry.get("classic_title") or constraints.get("title"),
                    "classic_author": entry.get("classic_author"),
                    "classic_work_type": entry.get("classic_work_type"),
                    "entry_type": entry.get("entry_type"),
                    "locator_type": entry.get("locator_type"),
                    "non_body": entry.get("non_body"),
                    "match_type": "cache_backstop",
                    "is_letter": entry.get("is_letter"),
                    "letter_title": entry.get("letter_title"),
                    "no_page_citation": entry.get("no_page_citation"),
                    "citation_mode": entry.get("citation_mode"),
                }
                key = (source, pdf_page)
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
    seen_entries = set()
    for entry in constraints.get("entries") or []:
        title = entry.get("classic_title") or entry.get("article") or constraints.get("title")
        key = (
            entry.get("source"),
            entry.get("start_page"),
            entry.get("end_page"),
            title,
        )
        if not title or key in seen_entries:
            continue
        seen_entries.add(key)
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
            "locator_type": entry.get("locator_type"),
            "non_body": entry.get("non_body"),
            "match_type": "locator_backstop",
            "is_letter": entry.get("is_letter"),
            "letter_title": entry.get("letter_title"),
            "no_page_citation": entry.get("no_page_citation"),
            "citation_mode": entry.get("citation_mode"),
        }
        content = (
            f"{title}\n"
            f"定位提示：该问题对应到《{title}》。\n"
            f"请在 PDF 第{entry.get('start_page')}-{entry.get('end_page')}页范围内核对。"
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
    existing_entries = {
        (
            doc.metadata.get("source"),
            doc.metadata.get("citation_page") or doc.metadata.get("page"),
            normalize_for_match(doc.metadata.get("classic_title") or doc.metadata.get("locator_title")),
        )
        for doc in docs
    }
    missing_backstops = []
    for doc in backstops:
        metadata = doc.metadata
        key = (
            metadata.get("source"),
            metadata.get("citation_page") or metadata.get("page"),
            normalize_for_match(metadata.get("classic_title")),
        )
        if key in existing_entries:
            continue
        missing_backstops.append(doc)
        existing_entries.add(key)

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
        candidates.extend(
            _hybrid_merge_candidates(
                seed,
                db.similarity_search(seed, k=per_seed_k),
                constraints,
                per_seed_k,
                ctx,
            )
        )

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
    is_front_matter_candidate = _helper(ctx, "is_front_matter_candidate")
    requests_derivative_material = _helper(ctx, "requests_derivative_material")
    expand_semantic_parent_docs = _helper(ctx, "expand_semantic_parent_docs")

    constraints = constraints_from_query(query, ctx)
    if (
        constraints.get("sources")
        and collection_requested(query, ctx) != "non_me"
        and not any(is_me_source(source) for source in constraints.get("sources") or [])
    ):
        constraints = dict(constraints)
        constraints.pop("sources", None)
        constraints.pop("page_ranges", None)
        constraints["strict_title"] = False
        constraints["version_scope_relaxed"] = True
    normalized_query = normalize_for_match(query)

    if constraints.get("strict_title") and "无产阶级专政" in normalized_query:
        return locator_backstop_documents(constraints, limit=k)

    if allow_exact_quote and is_quote_lookup_query(query):
        if not constraints.get("entries"):
            return []
        # Pass work_catalog constraints to scope OCR search
        exact_docs = exact_quote_lookup(query, OCR_CACHE_DIR, limit=k,
                                        constraints=constraints)
        if exact_docs:
            docs = annotate_docs_with_constraints(exact_docs, constraints, ctx)
            return append_locator_backstops(docs, constraints, k, ctx)
        if constraints.get("strict_title"):
            cache_docs = strict_title_cache_documents(query, constraints, k, ctx)
            if cache_docs:
                cache_docs = annotate_docs_with_constraints(cache_docs, constraints, ctx)
                return append_locator_backstops(cache_docs, constraints, k, ctx)
            return locator_backstop_documents(constraints, limit=k)
        return []

    if constraints.get("high_precision_locator"):
        cache_docs = strict_title_cache_documents(query, constraints, k, ctx)
        if cache_docs:
            cache_docs = annotate_docs_with_constraints(cache_docs, constraints, ctx)
            return append_locator_backstops(cache_docs, constraints, k, ctx)
        return locator_backstop_documents(constraints, limit=k)

    fetch_k = max(120 if constraints or active_concept_terms(query) else 30, k * 12)

    if constraints.get("sources"):
        if active_concept_terms(query):
            candidates = concept_constrained_candidates(query, db, constraints, fetch_k, ctx)
        elif constraints.get("topic_id"):
            candidates = topic_constrained_candidates(query, db, constraints, fetch_k, ctx)
        else:
            candidates = _controlled_dense_candidates(query, db, constraints, fetch_k, ctx)
            candidates = _hybrid_merge_candidates(query, candidates, constraints, fetch_k, ctx)
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

        if constraints.get("strict_title") and _has_narrow_page_ranges(constraints):
            candidates = dedupe_documents(
                strict_title_cache_documents(query, constraints, fetch_k, ctx) + candidates,
                ctx,
            )

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
                    doc for doc in _hybrid_merge_candidates(
                        title_query,
                        _controlled_dense_candidates(title_query, db, constraints, fetch_k, ctx),
                        constraints,
                        fetch_k,
                        ctx,
                    )
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
            candidates = _hybrid_merge_candidates(
                query,
                _controlled_dense_candidates(query, db, constraints, fetch_k, ctx),
                constraints,
                fetch_k,
                ctx,
            )
    else:
        candidates = _hybrid_merge_candidates(
            query,
            _controlled_dense_candidates(query, db, constraints, fetch_k, ctx),
            constraints,
            fetch_k,
            ctx,
        )

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
            for doc in _hybrid_merge_candidates(
                seed,
                db.similarity_search(seed, k=seed_k),
                constraints,
                seed_k,
                ctx,
            ):
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

    if constraints.get("strict_title") and not requests_derivative_material(query, constraints):
        body_candidates = [
            doc for doc in candidates
            if not is_front_matter_candidate(doc.metadata, clean_text(doc.page_content, ""), constraints)
        ]
        if body_candidates:
            candidates = body_candidates

    ranked_docs = rerank_documents(query, candidates, constraints, ctx)
    ranked_docs = collapse_content_near_duplicates(ranked_docs, ctx)
    ranked_docs = _prefer_source_order(ranked_docs, query, ctx)
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
    docs = expand_semantic_parent_docs(docs)
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
