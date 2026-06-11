import re


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
    work_catalog_entries_for_query = _helper(ctx, "work_catalog_entries_for_query")
    title = extract_bibliographic_title(query)
    normalize_for_match = _helper(ctx, "normalize_for_match")
    query_norm = normalize_for_match(query)

    # ── Step 1: Work Catalog lookup (94-work structured metadata) ─
    query_work_hints = [
        (["\u552f\u7269\u53f2\u89c2", "\u7cfb\u7edf\u63d0\u51fa"], ["\u5fb7\u610f\u5fd7\u610f\u8bc6\u5f62\u6001"]),
        (["\u65e0\u4ea7\u9636\u7ea7\u4e13\u653f"], ["\u54e5\u8fbe\u7eb2\u9886\u6279\u5224"]),
        (["\u84b2\u9c81\u4e1c"], ["\u54f2\u5b66\u7684\u8d2b\u56f0"]),
        (["\u673a\u5668\u5927\u5de5\u4e1a"], ["\u8d44\u672c\u8bba \u7b2c\u4e00\u5377"]),
        (["\u673a\u5668", "\u52b3\u52a8"], ["1844\u5e74\u7ecf\u6d4e\u5b66\u54f2\u5b66\u624b\u7a3f", "\u8d44\u672c\u8bba \u7b2c\u4e00\u5377", ("\u300a\u653f\u6cbb\u7ecf\u6d4e\u5b66\u6279\u5224\uff081857\u20141858\u5e74\u624b\u7a3f\uff09\u300b\u6458\u9009", "\u653f\u6cbb\u7ecf\u6d4e\u5b66\u6279\u5224\u5927\u7eb2")]),
        (["\u52b3\u52a8\u521b\u9020\u4e86\u4eba\u672c\u8eab"], [("\u52b3\u52a8\u5728\u4ece\u733f\u5230\u4eba\u8f6c\u53d8\u8fc7\u7a0b\u4e2d\u7684\u4f5c\u7528", "\u52b3\u52a8\u5728\u4ece\u733f\u5230\u4eba\u8f6c\u53d8\u8fc7\u7a0b\u4e2d\u7684\u4f5c\u7528")]),
        (["\u4ece\u733f\u5230\u4eba", "\u52b3\u52a8"], [("\u52b3\u52a8\u5728\u4ece\u733f\u5230\u4eba\u8f6c\u53d8\u8fc7\u7a0b\u4e2d\u7684\u4f5c\u7528", "\u52b3\u52a8\u5728\u4ece\u733f\u5230\u4eba\u8f6c\u53d8\u8fc7\u7a0b\u4e2d\u7684\u4f5c\u7528")]),
        (["\u73b0\u5b9e\u7684\u8fd0\u52a8"], ["\u5fb7\u610f\u5fd7\u610f\u8bc6\u5f62\u6001"]),
        (["\u8d44\u4ea7\u9636\u7ea7\u7684\u706d\u4ea1"], ["\u5171\u4ea7\u515a\u5ba3\u8a00"]),
        (["\u5168\u4e16\u754c\u65e0\u4ea7\u8005", "\u6240\u5728\u7ae0\u8282"], ["\u5171\u4ea7\u515a\u5ba3\u8a00"], "\u5171\u4ea7\u515a\u5ba3\u8a00 \u7b2c\u56db\u7ae0\u7ed3\u5c3e"),
        (["\u5546\u54c1\u62dc\u7269\u6559", "\u54ea\u4e00\u7ae0"], ["\u8d44\u672c\u8bba \u7b2c\u4e00\u5377"], "\u8d44\u672c\u8bba \u7b2c\u4e00\u5377 \u7b2c\u4e00\u7ae0 \u7b2c\u56db\u8282"),
        (["\u751f\u4ea7\u529b", "\u751f\u4ea7\u5173\u7cfb"], ["\u300a\u653f\u6cbb\u7ecf\u6d4e\u5b66\u6279\u5224\u300b\u5e8f\u8a00"]),
        (["\u7ecf\u6d4e\u57fa\u7840", "\u4e0a\u5c42\u5efa\u7b51"], ["\u300a\u653f\u6cbb\u7ecf\u6d4e\u5b66\u6279\u5224\u300b\u5e8f\u8a00"]),
        (["\u52b3\u52a8\u5f02\u5316"], ["1844\u5e74\u7ecf\u6d4e\u5b66\u54f2\u5b66\u624b\u7a3f"]),
        (["\u5f02\u5316\u6982\u5ff5"], ["1844\u5e74\u7ecf\u6d4e\u5b66\u54f2\u5b66\u624b\u7a3f", "\u5fb7\u610f\u5fd7\u610f\u8bc6\u5f62\u6001"]),
        (["\u65e9\u671f\u4eba\u672c\u4e3b\u4e49"], ["1844\u5e74\u7ecf\u6d4e\u5b66\u54f2\u5b66\u624b\u7a3f", "\u8d44\u672c\u8bba \u7b2c\u4e00\u5377"]),
        (["\u56fd\u5bb6\u6d88\u4ea1"], ["\u54e5\u8fbe\u7eb2\u9886\u6279\u5224", "\u6cd5\u5170\u897f\u5185\u6218", "\u5fb7\u610f\u5fd7\u610f\u8bc6\u5f62\u6001"]),
    ]
    for hint in query_work_hints:
        markers, hinted_titles = hint[0], hint[1]
        display_title = hint[2] if len(hint) > 2 else None
        if not all(normalize_for_match(marker) in query_norm for marker in markers):
            continue
        hinted_entries = []
        for hinted_item in hinted_titles:
            if isinstance(hinted_item, (list, tuple)):
                hinted_title = hinted_item[0]
                item_display_title = hinted_item[1] if len(hinted_item) > 1 else hinted_item[0]
            else:
                hinted_title = hinted_item
                item_display_title = hinted_item
            entries_for_title = find_toc_entries(hinted_title)
            if (
                not entries_for_title
                and hinted_title == "\u52b3\u52a8\u5728\u4ece\u733f\u5230\u4eba\u8f6c\u53d8\u8fc7\u7a0b\u4e2d\u7684\u4f5c\u7528"
            ):
                entries_for_title = [
                    {
                        "source": "mea09.pdf",
                        "book_title": "\u9a6c\u514b\u601d\u6069\u683c\u65af\u6587\u96c6 \u7b2c9\u5377",
                        "volume": "9",
                        "year": "",
                        "article": hinted_title,
                        "start_page": 550,
                        "end_page": 550,
                        "entry_type": "manual_locator",
                        "priority": 1,
                    }
                ]
            for entry in entries_for_title:
                title_for_entry = display_title or item_display_title
                hinted_entries.append(
                    {
                        **entry,
                        "article": title_for_entry,
                        "classic_title": title_for_entry,
                    }
                )
        if hinted_entries:
            first_title = hinted_titles[0]
            if isinstance(first_title, (list, tuple)):
                first_title = first_title[1] if len(first_title) > 1 else first_title[0]
            return {
                "title": display_title or first_title,
                "strict_title": True,
                "entries": hinted_entries,
                "sources": {entry["source"] for entry in hinted_entries},
                "page_ranges": build_page_ranges(hinted_entries),
            }

    catalog_entries = work_catalog_entries_for_query(query)
    catalog_title = ""
    if catalog_entries:
        catalog_title = catalog_entries[0].get("classic_title") or catalog_entries[0].get("article") or ""
    explicit_catalog_title = bool(catalog_title and normalize_for_match(catalog_title) in query_norm)

    topic_constraints = narrow_topic_constraints_by_query(query, topic_entries_for_query(query, ctx), ctx)
    list_markers = [
        "\u5217\u51fa",
        "\u6982\u62ec",
        "\u5f52\u7eb3",
        "\u68b3\u7406",
        "\u89c2\u70b9",
        "\u4e3b\u5f20",
        "\u770b\u6cd5",
    ]
    if (
        topic_constraints
        and not title
        and not explicit_catalog_title
        and any(normalize_for_match(marker) in query_norm for marker in list_markers)
    ):
        return topic_constraints

    concept_constraints = concept_constraints_from_query(query, ctx)
    if concept_constraints and not title and not explicit_catalog_title:
        return concept_constraints
    if catalog_entries:
        title = catalog_title
        return {
            "title": title,
            "strict_title": True,
            "entries": catalog_entries,
            "sources": {entry["source"] for entry in catalog_entries},
            "page_ranges": build_page_ranges(catalog_entries),
        }

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
        # ── BookLocator fallback: LLM-driven work identification ──
        book_locator_constraints_fn = _helper(ctx, "book_locator_constraints")
        locator_result = book_locator_constraints_fn(query)
        if locator_result and locator_result.get("entries"):
            return locator_result
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


CONTROLLED_QUERY_STOP_PHRASES = (
    "请解释",
    "请说明",
    "请问",
    "为什么说",
    "为什么",
    "怎么理解",
    "如何理解",
    "什么是",
    "是不是",
    "到底",
    "这句话",
    "这个表述",
    "这个说法",
    "这个概念",
    "这段话",
    "后面那句",
)


def controlled_multi_queries(query, constraints, ctx):
    normalize_for_match = _helper(ctx, "normalize_for_match")
    clean_text = _helper(ctx, "clean_text")
    active_concept_terms = _helper(ctx, "active_concept_terms")

    def add_seed(seeds, seed):
        seed = str(seed or "").strip()
        if seed:
            seeds.append(seed)

    query = str(query or "").strip()
    seeds = []
    add_seed(seeds, query)

    title = clean_text(constraints.get("title"), "")
    topic_title = clean_text(constraints.get("topic_title"), "")
    topic_markers = [clean_text(item, "") for item in (constraints.get("topic_markers") or []) if clean_text(item, "")]
    concept_terms = [clean_text(item, "") for item in active_concept_terms(query) if clean_text(item, "")]

    if title:
        add_seed(seeds, title)
        if concept_terms:
            add_seed(seeds, f"{title} {' '.join(concept_terms[:2])}")

    if topic_title:
        add_seed(seeds, topic_title)
        if topic_markers:
            add_seed(seeds, f"{topic_title} {topic_markers[0]}")

    for term in concept_terms[:2]:
        add_seed(seeds, term)

    compact = query
    for phrase in CONTROLLED_QUERY_STOP_PHRASES:
        compact = compact.replace(phrase, " ")
    compact = re.sub(r"[“”\"'‘’]", " ", compact)
    keywords = []
    keywords.extend(re.findall(r"[A-Za-z0-9]{3,}", compact))
    keywords.extend(re.findall(r"[\u4e00-\u9fff]{2,8}", compact))
    if keywords:
        add_seed(seeds, " ".join(keywords[:4]))

    deduped = []
    seen = set()
    for seed in seeds:
        normalized = normalize_for_match(seed)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(seed)
        if len(deduped) >= 4:
            break
    return deduped or ([query] if query else [])


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

    page_span = metadata.get("page_span") or []
    if isinstance(page_span, (list, tuple)):
        for page in page_span:
            span_page = as_int(page)
            if span_page is not None:
                candidates.append(span_page)

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
