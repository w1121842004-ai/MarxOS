import os
import sys

from marxos_retrieval_constraints import (
    metadata_matches_constraints,
    page_in_expected_range,
    topic_title_allowed,
)


def _helper(ctx, name):
    return ctx[name]


def score_source_match(metadata, constraints):
    return 100 if metadata_matches_constraints(metadata, constraints) else 0


def score_page_range(metadata, constraints, ctx):
    return 40 if page_in_expected_range(metadata, constraints, ctx) else 0


def score_topic_title_match(metadata, constraints, ctx):
    if not constraints.get("topic_id"):
        return 0
    return 45 if topic_title_allowed(metadata, constraints, ctx) else -35


def score_topic_content_match(metadata, content, constraints, ctx):
    normalize_for_match = _helper(ctx, "normalize_for_match")
    clean_text = _helper(ctx, "clean_text")
    markers = constraints.get("topic_markers") or []
    if not constraints.get("topic_id") or not markers:
        return 0

    article = normalize_for_match(clean_text(metadata.get("section") or metadata.get("article"), ""))
    lead = normalize_for_match(content[:400])
    body = normalize_for_match(content)
    score = 0
    for marker in markers:
        marker_norm = normalize_for_match(marker)
        if not marker_norm:
            continue
        if marker_norm in article:
            score += 28
        if marker_norm in lead:
            score += 24
        elif marker_norm in body:
            score += 10
    return min(score, 120)


def score_article_match(metadata, normalized_title, haystack, ctx):
    clean_text = _helper(ctx, "clean_text")
    normalize_for_match = _helper(ctx, "normalize_for_match")
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


def debug_rerank_score(index, doc, score_parts, ctx):
    RERANK_DEBUG_ENV = ctx["RERANK_DEBUG_ENV"]
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


def rerank_documents(query, docs, constraints, ctx):
    normalize_for_match = _helper(ctx, "normalize_for_match")
    clean_text = _helper(ctx, "clean_text")
    score_concept_focus = _helper(ctx, "score_concept_focus")
    score_concept_source_priority = _helper(ctx, "score_concept_source_priority")
    score_document_quality = _helper(ctx, "score_document_quality")
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
            "page_range": score_page_range(metadata, constraints, ctx),
            "topic_title": score_topic_title_match(metadata, constraints, ctx),
            "topic_content": score_topic_content_match(metadata, content, constraints, ctx),
            "article_match": score_article_match(metadata, normalized_title, haystack, ctx),
            "query_match": score_query_match(normalized_query, haystack),
            "concept_focus": score_concept_focus(query, metadata, content),
            "concept_source": score_concept_source_priority(query, metadata),
            "document_quality": score_document_quality(metadata, content),
        }
        score = sum(score_parts.values())
        debug_rerank_score(index, doc, score_parts, ctx)
        ranked.append((score, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for score, doc in ranked]


def diversify_documents(docs, k, ctx, max_per_source=2, max_per_article=1, min_distinct_sources=0):
    clean_text = _helper(ctx, "clean_text")
    normalize_for_match = _helper(ctx, "normalize_for_match")
    selected = []
    source_counts = {}
    article_counts = {}
    selected_sources = set()

    def can_select(doc):
        metadata = doc.metadata
        source = metadata.get("source") or ""
        article = clean_text(metadata.get("section") or metadata.get("article"), "")
        article_key = (source, normalize_for_match(article))
        if source_counts.get(source, 0) >= max_per_source:
            return False, source, article_key
        if article_key[1] and article_counts.get(article_key, 0) >= max_per_article:
            return False, source, article_key
        return True, source, article_key

    def record_selection(doc, source, article_key):
        selected.append(doc)
        source_counts[source] = source_counts.get(source, 0) + 1
        article_counts[article_key] = article_counts.get(article_key, 0) + 1
        if source:
            selected_sources.add(source)

    if min_distinct_sources > 0:
        for doc in docs:
            allowed, source, article_key = can_select(doc)
            if not allowed or not source or source in selected_sources:
                continue
            record_selection(doc, source, article_key)
            if len(selected) >= k or len(selected_sources) >= min_distinct_sources:
                break

    if len(selected) >= k:
        return selected[:k]

    for doc in docs:
        if doc in selected:
            continue
        allowed, source, article_key = can_select(doc)
        if not allowed:
            continue
        record_selection(doc, source, article_key)
        if len(selected) >= k:
            return selected

    for doc in docs:
        if doc in selected:
            continue
        selected.append(doc)
        if len(selected) >= k:
            break

    return selected[:k]


def annotate_docs_with_constraints(docs, constraints, ctx):
    metadata_citation_page = _helper(ctx, "metadata_citation_page")
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
            metadata.setdefault("classic_title", title)
            metadata.setdefault("work_title", title)
            metadata.setdefault("locator_title", title)

        try:
            page = int(metadata.get("page"))
        except (TypeError, ValueError):
            page = None
        citation_page = metadata_citation_page(metadata)
        tolerance = int(constraints.get("page_tolerance") or 0)

        matched_entry = None
        for entry in source_entries.get(metadata.get("source"), []):
            if citation_page is not None and (entry["start_page"] - tolerance) <= citation_page <= (entry["end_page"] + tolerance):
                matched_entry = entry
                break
            if page is None or entry["start_page"] <= page <= entry["end_page"]:
                matched_entry = entry
                break

        if matched_entry:
            entry_title = matched_entry.get("classic_title") or matched_entry.get("article") or title
            if entry_title:
                metadata["classic_title"] = entry_title
                metadata["work_title"] = entry_title
                metadata["locator_title"] = entry_title
                if not (metadata.get("raw_article") or metadata.get("raw_section")):
                    metadata["article"] = entry_title
                    metadata["section"] = entry_title
            if matched_entry.get("classic_author"):
                metadata.setdefault("classic_author", matched_entry.get("classic_author"))
            if matched_entry.get("classic_work_type"):
                metadata.setdefault("classic_work_type", matched_entry.get("classic_work_type"))
    return docs


def select_topic_documents(ranked_docs, constraints, k, ctx):
    clean_text = _helper(ctx, "clean_text")
    preferred = []
    secondary = []
    for doc in ranked_docs:
        content = clean_text(doc.page_content, "")
        title_hit = topic_title_allowed(doc.metadata, constraints, ctx)
        content_hit = score_topic_content_match(doc.metadata, content, constraints, ctx) > 0
        if title_hit and content_hit:
            preferred.append(doc)
        elif title_hit:
            secondary.append(doc)
    fallback = [doc for doc in ranked_docs if doc not in preferred]
    preferred_selected = diversify_documents(
        preferred,
        k,
        ctx,
        min_distinct_sources=constraints.get("min_distinct_sources", 0),
    )
    if len(preferred_selected) >= k:
        return preferred_selected[:k]

    combined = preferred_selected
    if len(combined) < k and secondary:
        remaining = k - len(combined)
        combined += diversify_documents(secondary, remaining, ctx)
    if len(combined) < k:
        remaining = k - len(combined)
        extra = diversify_documents([doc for doc in fallback if doc not in secondary], remaining, ctx)
        combined += extra
    return combined[:k]


def dedupe_documents(docs, ctx):
    clean_text = _helper(ctx, "clean_text")
    deduped = []
    seen = set()
    for doc in docs:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("printed_page"),
            doc.metadata.get("citation_page"),
            clean_text(doc.metadata.get("article") or doc.metadata.get("section"), ""),
            clean_text(doc.page_content, "")[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped
