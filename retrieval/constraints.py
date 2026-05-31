def _helper(ctx, name):
    return ctx[name]


def build_page_ranges(entries):
    page_ranges = {}
    for entry in entries:
        page_ranges.setdefault(entry["source"], []).append(
            (entry["start_page"], entry["end_page"])
        )
    return page_ranges


def dedupe_locator_entries(entries):
    deduped = []
    seen = set()
    for entry in entries:
        key = (
            entry.get("source"),
            entry.get("article"),
            entry.get("start_page"),
            entry.get("end_page"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def normalize_topic_entries(entries, ctx):
    normalize_topic_title = _helper(ctx, "normalize_topic_title")
    normalized_entries = []
    for entry in entries:
        normalized = dict(entry)
        article = normalize_topic_title(entry.get("article") or entry.get("classic_title"))
        if article:
            normalized["article"] = article
            normalized["classic_title"] = article
        normalized_entries.append(normalized)
    return normalized_entries


def topic_matches_query(topic, query, ctx):
    normalize_for_match = _helper(ctx, "normalize_for_match")
    query_norm = normalize_for_match(query)
    if not query_norm:
        return False

    for keyword in topic.get("keywords_any") or []:
        keyword_norm = normalize_for_match(keyword)
        if keyword_norm and keyword_norm in query_norm:
            return True

    for keyword_group in topic.get("keywords_all") or []:
        keyword_norms = [normalize_for_match(item) for item in keyword_group if normalize_for_match(item)]
        if keyword_norms and all(keyword in query_norm for keyword in keyword_norms):
            return True

    return False


def topic_entries_for_query(query, ctx):
    TOPIC_CATALOG = ctx["TOPIC_CATALOG"]
    find_toc_entries = _helper(ctx, "find_toc_entries")
    normalize_for_match = _helper(ctx, "normalize_for_match")
    clean_article_title = _helper(ctx, "clean_article_title")

    for topic in TOPIC_CATALOG:
        if not topic_matches_query(topic, query, ctx):
            continue

        entries = []
        for work in topic.get("works") or []:
            entries.extend(find_toc_entries(work))
        entries = normalize_topic_entries(dedupe_locator_entries(entries), ctx)
        if not entries:
            continue

        return {
            "topic_id": topic.get("id"),
            "topic_title": topic.get("label"),
            "section": topic.get("section") or "",
            "topic_markers": topic.get("content_markers") or [],
            "entries": entries,
            "sources": {entry["source"] for entry in entries},
            "page_ranges": build_page_ranges(entries),
            "allowed_titles": {
                normalize_for_match(clean_article_title(entry.get("article") or entry.get("classic_title")))
                for entry in entries
                if clean_article_title(entry.get("article") or entry.get("classic_title"))
            },
            "min_distinct_sources": int(topic.get("min_distinct_sources") or 0),
            "page_tolerance": int(topic.get("page_tolerance") or 0),
        }

    return {}


def topic_info_from_constraints(constraints):
    return {
        "topic_id": constraints.get("topic_id") or "",
        "topic_label": constraints.get("topic_title") or "",
        "topic_section": constraints.get("section") or "",
    }


def narrow_topic_constraints_by_query(query, constraints, ctx):
    normalize_for_match = _helper(ctx, "normalize_for_match")
    clean_article_title = _helper(ctx, "clean_article_title")
    topic_entries = constraints.get("entries") or []
    if not constraints.get("topic_id") or not topic_entries:
        return constraints

    query_norm = normalize_for_match(query)
    matched_entries = []
    for entry in topic_entries:
        title = normalize_for_match(entry.get("article") or entry.get("classic_title"))
        if title and title in query_norm:
            matched_entries.append(entry)

    if not matched_entries:
        return constraints

    return {
        **constraints,
        "entries": matched_entries,
        "sources": {entry["source"] for entry in matched_entries},
        "page_ranges": build_page_ranges(matched_entries),
        "allowed_titles": {
            normalize_for_match(clean_article_title(entry.get("article") or entry.get("classic_title")))
            for entry in matched_entries
            if clean_article_title(entry.get("article") or entry.get("classic_title"))
        },
    }


def infer_work_title_from_query(query, ctx):
    normalize_for_match = _helper(ctx, "normalize_for_match")
    WORK_TITLE_ALIASES = ctx["WORK_TITLE_ALIASES"]
    query_norm = normalize_for_match(query)
    if not query_norm:
        return None

    for title, markers in WORK_TITLE_ALIASES.items():
        title_norm = normalize_for_match(title)
        if title_norm and title_norm in query_norm:
            return title

        marker_norms = [normalize_for_match(marker) for marker in markers if normalize_for_match(marker)]
        if any(marker in query_norm for marker in marker_norms if len(marker) >= 5):
            return title
        hits = sum(1 for marker in marker_norms if marker in query_norm)
        if hits >= 2:
            return title

    return None


def concept_constraints_from_query(query, ctx):
    active_concept_terms = _helper(ctx, "active_concept_terms")
    core_classic_by_id = _helper(ctx, "core_classic_by_id")
    CONCEPT_CANONICAL_CLASSIC_IDS = ctx["CONCEPT_CANONICAL_CLASSIC_IDS"]
    entries = []
    seen = set()

    for term in active_concept_terms(query):
        classic_id = CONCEPT_CANONICAL_CLASSIC_IDS.get(term)
        if not classic_id:
            continue

        classic = core_classic_by_id(classic_id)
        if not classic:
            continue

        for entry in classic.get("entries") or []:
            key = (entry.get("source"), entry.get("start_page"), entry.get("end_page"))
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                {
                    **entry,
                    "classic_id": classic.get("id"),
                    "classic_title": classic.get("title"),
                    "classic_author": classic.get("author"),
                    "classic_work_year": classic.get("work_year"),
                    "classic_work_type": classic.get("work_type"),
                }
            )

    if not entries:
        return {}

    title = entries[0].get("classic_title")
    return {
        "title": title,
        "strict_title": True,
        "entries": entries,
        "sources": {entry["source"] for entry in entries},
        "page_ranges": build_page_ranges(entries),
    }


def constraints_from_query(query, ctx):
    extract_bibliographic_title = _helper(ctx, "extract_bibliographic_title")
    locator_entries_for_query = _helper(ctx, "locator_entries_for_query")
    classic_entries_for_query = _helper(ctx, "classic_entries_for_query")
    enrich_core_classic_entries = _helper(ctx, "enrich_core_classic_entries")
    find_toc_entries = _helper(ctx, "find_toc_entries")
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

    core_entries = classic_entries_for_query(title) if title else []
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

    concept_constraints = concept_constraints_from_query(query, ctx)
    if concept_constraints:
        return concept_constraints

    topic_constraints = narrow_topic_constraints_by_query(query, topic_entries_for_query(query, ctx), ctx)
    if topic_constraints:
        return topic_constraints

    inferred_title = infer_work_title_from_query(query, ctx)
    if inferred_title and not title:
        title = inferred_title

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
        "strict_title": True,
    }


def topic_seed_queries(query, constraints, ctx):
    normalize_for_match = _helper(ctx, "normalize_for_match")
    seeds = []
    query = str(query or "").strip()
    if query:
        seeds.append(query)

    topic_title = str(constraints.get("topic_title") or "").strip()
    if topic_title:
        seeds.append(topic_title)

    entries = constraints.get("entries") or []
    for entry in entries[:4]:
        title = str(entry.get("article") or entry.get("classic_title") or "").strip()
        if title:
            seeds.append(title)

    deduped = []
    seen = set()
    for seed in seeds:
        normalized = normalize_for_match(seed)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(str(seed).strip())
    return deduped or [query]


def concept_seed_queries(query, constraints, ctx):
    active_concept_terms = _helper(ctx, "active_concept_terms")
    normalize_for_match = _helper(ctx, "normalize_for_match")
    seeds = []
    query = str(query or "").strip()
    if query:
        seeds.append(query)

    for term in active_concept_terms(query):
        cleaned = str(term or "").strip()
        if cleaned:
            seeds.append(cleaned)

    title = str(constraints.get("title") or "").strip()
    if title:
        seeds.append(title)

    deduped = []
    seen = set()
    for seed in seeds:
        normalized = normalize_for_match(seed)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(str(seed).strip())
    return deduped or [query]


def metadata_matches_constraints(metadata, constraints):
    sources = constraints.get("sources") or set()
    if not sources:
        return True
    return (metadata.get("source") or "") in sources


def page_in_expected_range(metadata, constraints, ctx):
    metadata_citation_page = _helper(ctx, "metadata_citation_page")
    page_ranges = constraints.get("page_ranges") or {}
    ranges = page_ranges.get(metadata.get("source"))
    if not ranges:
        return False

    page = metadata_citation_page(metadata)
    if page is None:
        return False

    tolerance = int(constraints.get("page_tolerance") or 0)
    return any((start - tolerance) <= page <= (end + tolerance) for start, end in ranges)


def candidate_pdf_pages_from_metadata(metadata, ctx):
    as_int = _helper(ctx, "as_int")
    source = metadata.get("source")
    pdf_page = as_int(metadata.get("pdf_page") or metadata.get("page"))
    printed_page = as_int(metadata.get("printed_page") or metadata.get("citation_page"))
    if not source:
        return []

    candidates = []
    if pdf_page is not None:
        candidates.append(pdf_page)

    if printed_page is not None:
        find_pdf_page_by_printed_page = _helper(ctx, "find_pdf_page_by_printed_page")
        mapped_pdf_page = find_pdf_page_by_printed_page(source, printed_page)
        if mapped_pdf_page is not None:
            candidates.append(mapped_pdf_page)

    deduped = []
    seen = set()
    for page in candidates:
        if page in seen:
            continue
        seen.add(page)
        deduped.append(page)
    return deduped


def topic_title_allowed(metadata, constraints, ctx):
    clean_text = _helper(ctx, "clean_text")
    normalize_for_match = _helper(ctx, "normalize_for_match")
    allowed_titles = constraints.get("allowed_titles") or set()
    if not constraints.get("topic_id") or not allowed_titles:
        return True

    article = clean_text(metadata.get("section") or metadata.get("article"), "")
    article_norm = normalize_for_match(article)
    return any(title in article_norm for title in allowed_titles if title)
