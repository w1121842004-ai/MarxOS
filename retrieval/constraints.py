import json
import re
from functools import lru_cache
from pathlib import Path


def _helper(ctx, name):
    return ctx[name]


def build_page_ranges(entries):
    page_ranges = {}
    for entry in entries:
        page_ranges.setdefault(entry["source"], []).append(
            (entry["start_page"], entry["end_page"])
        )
    return page_ranges


ME_SOURCE_RE = re.compile(r"^me(?:\d{2}[abc]?|a\d{2}|s\d{2})\.pdf$")
ME_VOLUME_SOURCE_RE = re.compile(r"me(\d{2})([abc]?)\.pdf", re.IGNORECASE)
ME_EXPLICIT_SOURCE_RE = re.compile(r"me(\d{1,2})([abc]?)\.pdf", re.IGNORECASE)
CHINESE_VOLUME_RE = re.compile(r"(?:全集)?第\s*([0-9０-９]{1,2})\s*卷\s*([ABCabcＡＢＣａｂｃ]?)")
PAGE_HINT_RE = re.compile(r"第\s*([0-9０-９]{1,4})\s*页|([0-9０-９]{1,4})\s*页")
FULLWIDTH_DIGIT_MAP = str.maketrans("０１２３４５６７８９", "0123456789")
FULLWIDTH_LETTER_MAP = str.maketrans("ＡＢＣａｂｃ", "ABCabc")
ME_PRIORITY_TITLE_HINTS = {
    "剩余价值理论": [
        ("me26a.pdf", 1, 900, "剩余价值理论"),
        ("me26b.pdf", 1, 900, "剩余价值理论"),
        ("me26c.pdf", 1, 900, "剩余价值理论"),
    ],
    "政治经济学批判18611863年手稿": [
        ("me26a.pdf", 1, 900, "政治经济学批判（1861—1863年手稿）"),
        ("me26b.pdf", 1, 900, "政治经济学批判（1861—1863年手稿）"),
        ("me26c.pdf", 1, 900, "政治经济学批判（1861—1863年手稿）"),
    ],
    "资本论手稿": [
        ("me46a.pdf", 1, 900, "资本论手稿"),
        ("me46b.pdf", 1, 900, "资本论手稿"),
        ("me47.pdf", 1, 900, "资本论手稿"),
    ],
}
ME_HIGH_PRECISION_LOCATORS = [
    {
        "tokens_all": ["1847年6月", "共产主义者同盟中央委员会", "各国支部", "附信"],
        "title": "共产主义者同盟中央委员会附信",
        "source": "me42.pdf",
        "page": 529,
        "pdf_page": 545,
    },
    {
        "tokens_all": ["格律恩", "特利尔日报"],
        "title": "马克思致恩格斯的信",
        "source": "me27.pdf",
        "page": 54,
        "article": "第一部分卡·马克思和弗·恩格斯之间的书信",
    },
    {
        "tokens_all": ["总委员会", "极端联邦主义", "极端集中主义"],
        "title": "马克思恩格斯全集 第33卷",
        "source": "me33.pdf",
        "page": 379,
        "pdf_page": 400,
    },
    {
        "tokens_all": ["西西里岛社会党", "贺信"],
        "title": "恩格斯给西西里岛社会党的贺信",
        "source": "me22.pdf",
        "page": 761,
        "pdf_page": 772,
    },
    {
        "tokens_all": ["李卜克内西", "住宿"],
        "title": "马克思致李卜克内西相关书信",
        "source": "me34.pdf",
        "page": 270,
        "pdf_page": 287,
        "article": "第二部分卡·马克思和弗·恩格斯给其他人的信",
    },
    {
        "tokens_all": ["剩余价值理论", "地租下降", "人口增长"],
        "title": "剩余价值理论",
        "source": "me44.pdf",
        "page": 11,
        "pdf_page": 122,
    },
    {
        "tokens_all": ["乔治一世法规", "爱尔兰", "立法"],
        "title": "从美国革命到1801年合并的爱尔兰",
        "source": "me45.pdf",
        "page": 26,
    },
    {
        "tokens_all": ["恩格斯", "1857年12月7日", "危机"],
        "title": "恩格斯致马克思（1857年12月7日）",
        "source": "me29.pdf",
        "page": 228,
        "pdf_page": 230,
        "article": "恩格斯致马克思",
    },
    {
        "tokens_all": ["剩余价值理论", "机器费用", "劳动费用"],
        "title": "剩余价值理论",
        "source": "me26c.pdf",
        "page": 371,
        "pdf_page": 413,
        "article": "《剩余价值理论》",
    },
    {
        "tokens_all": ["资本从一开始就不是为了使用价值", "剩余劳动"],
        "title": "资本论手稿",
        "source": "me46b.pdf",
        "page": 97,
    },
]
ME_HIGH_PRECISION_LOCATORS_PATH = Path(__file__).resolve().parents[1] / "rag" / "me_high_precision_locators.json"
ME_ARTICLE_LOCATORS_PATH = Path(__file__).resolve().parents[1] / "rag" / "me_article_locators.json"
LETTER_DATE_RE = re.compile(r"[（(][^）)]*(?:\d{1,2}月|\d{4}年|约|初|末|左右)")


def is_letter_title(title):
    title = str(title or "")
    if not title:
        return False
    return "致" in title


def normalize_digits_letters(text):
    return str(text or "").translate(FULLWIDTH_DIGIT_MAP).translate(FULLWIDTH_LETTER_MAP)


@lru_cache(maxsize=2)
def load_me_high_precision_locators(path=None):
    path = Path(path) if path else ME_HIGH_PRECISION_LOCATORS_PATH
    locators = list(ME_HIGH_PRECISION_LOCATORS)
    if not path.exists():
        return locators
    try:
        external = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return locators
    if not isinstance(external, list):
        return locators

    seen = {
        (
            tuple(locator.get("tokens_all") or []),
            locator.get("source"),
            locator.get("page"),
        )
        for locator in locators
    }
    for locator in external:
        if not isinstance(locator, dict) or locator.get("active") is False:
            continue
        key = (
            tuple(locator.get("tokens_all") or []),
            locator.get("source"),
            locator.get("page"),
        )
        if key in seen:
            continue
        seen.add(key)
        locators.append(locator)
    return locators


@lru_cache(maxsize=2)
def load_me_article_locators(path=None):
    path = Path(path) if path else ME_ARTICLE_LOCATORS_PATH
    if not path.exists():
        return []
    try:
        locators = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(locators, list):
        return []
    return [
        locator
        for locator in locators
        if isinstance(locator, dict) and locator.get("active") is not False
    ]


def is_me_source(source):
    return bool(ME_SOURCE_RE.fullmatch(str(source or "").lower()))


def is_work_location_query(query, ctx):
    normalize_for_match = _helper(ctx, "normalize_for_match")
    query_norm = normalize_for_match(query)
    markers = [
        "出自哪里",
        "出自哪部著作",
        "出自哪本",
        "是否出自",
        "是不是出自",
        "出自马克思原著",
        "主要出自",
        "经典来源",
        "最经典出处",
        "定位原文",
        "请定位",
        "所在章节",
        "在哪一章",
        "哪一章",
        "哪部著作",
        "哪些著作",
    ]
    return any(normalize_for_match(marker) in query_norm for marker in markers)


def collection_requested(query, ctx):
    normalize_for_match = _helper(ctx, "normalize_for_match")
    query_norm = normalize_for_match(query)
    if "全集" in query_norm or re.search(r"\bme\d{1,2}[abc]?\.pdf\b", str(query or ""), re.I):
        return "me"
    if "文集" in query_norm or re.search(r"\bmea\d{1,2}\.pdf\b", str(query or ""), re.I):
        return "mea"
    if any(marker in query_norm for marker in ["选集", "選集", "書信选编", "书信选编"]) or re.search(
        r"\bmes\d{1,2}\.pdf\b", str(query or ""), re.I
    ):
        return "mes"
    return ""


def source_matches_collection(source, requested):
    source = str(source or "").lower()
    if requested == "me":
        return is_me_source(source) and not source.startswith(("mea", "mes"))
    if requested == "mea":
        return source.startswith("mea")
    if requested == "mes":
        return source.startswith("mes")
    return True


def source_priority(source, query="", ctx=None):
    source = str(source or "").lower()
    requested = collection_requested(query, ctx) if ctx is not None else ""
    if requested and source_matches_collection(source, requested):
        return 0
    if requested:
        return 3
    if source_matches_collection(source, "me"):
        return 0
    if source.startswith("mea"):
        return 1
    if source.startswith("mes"):
        return 2
    return 3


def prefer_sources_for_query(entries, query, ctx):
    if not entries:
        return entries
    requested = collection_requested(query, ctx)
    if requested:
        requested_entries = [
            entry for entry in entries if source_matches_collection(entry.get("source"), requested)
        ]
        if requested_entries:
            return sorted(
                requested_entries,
                key=lambda entry: (entry.get("priority", 99), entry.get("start_page") or 0),
            )
    return sorted(
        entries,
        key=lambda entry: (
            entry.get("priority", 99),
            source_priority(entry.get("source"), query, ctx),
            entry.get("start_page") or 0,
        ),
    )


def constraints_result(title, entries, query, ctx, strict_title=True):
    entries = prefer_sources_for_query(entries, query, ctx)
    requested = collection_requested(query, ctx)
    if requested == "me" and entries and not any(source_matches_collection(entry.get("source"), "me") for entry in entries):
        return {
            "title": title,
            "entries": entries,
            "strict_title": False,
            "version_scope_relaxed": True,
        }
    return {
        "title": title,
        "strict_title": strict_title,
        "entries": entries,
        "sources": {entry["source"] for entry in entries},
        "page_ranges": build_page_ranges(entries),
    }


def explicit_volume_constraints_from_query(query, ctx):
    query_text = normalize_digits_letters(query)
    normalize_for_match = _helper(ctx, "normalize_for_match")
    query_norm = normalize_for_match(query_text)
    source = None

    match = ME_EXPLICIT_SOURCE_RE.search(query_text)
    if match:
        volume = int(match.group(1))
        suffix = match.group(2).lower()
        source = f"me{volume:02d}{suffix}.pdf"

    if source is None:
        match = CHINESE_VOLUME_RE.search(query_text)
        if match:
            volume = int(match.group(1))
            suffix = match.group(2).lower()
            source = f"me{volume:02d}{suffix}.pdf"

    if source is None:
        return {}

    page = None
    page_match = PAGE_HINT_RE.search(query_text)
    if page_match:
        page = int(page_match.group(1) or page_match.group(2))

    title = f"马克思恩格斯全集 第{int(source[2:4])}卷"
    if source[4:5].isalpha():
        title += source[4:5].upper()
    entry = {
        "source": source,
        "book_title": title,
        "article": title,
        "classic_title": title,
        "start_page": page or 1,
        "end_page": page or 1200,
        "entry_type": "explicit_volume",
        "priority": 0,
    }
    return {
        "title": title,
        "strict_title": True,
        "entries": [entry],
        "sources": {source},
        "page_ranges": build_page_ranges([entry]),
        "page_tolerance": 2 if page else 0,
        "explicit_volume": True,
    }


def me_title_hint_constraints_from_query(query, ctx):
    normalize_for_match = _helper(ctx, "normalize_for_match")
    query_norm = normalize_for_match(query)
    requested = collection_requested(query, ctx)
    if requested in {"mea", "mes"}:
        return {}

    for marker, specs in ME_PRIORITY_TITLE_HINTS.items():
        if normalize_for_match(marker) not in query_norm:
            continue
        entries = [
            {
                "source": source,
                "book_title": article,
                "article": article,
                "classic_title": article,
                "start_page": start_page,
                "end_page": end_page,
                "entry_type": "me_title_hint",
                "priority": 0,
            }
            for source, start_page, end_page, article in specs
        ]
        return {
            "title": specs[0][3],
            "strict_title": True,
            "entries": entries,
            "sources": {entry["source"] for entry in entries},
            "page_ranges": build_page_ranges(entries),
            "page_tolerance": 3,
            "me_title_hint": True,
        }
    return {}


def high_precision_locator_constraints_from_query(query, ctx):
    normalize_for_match = _helper(ctx, "normalize_for_match")
    query_norm = normalize_for_match(query)
    requested = collection_requested(query, ctx)
    if requested in {"mea", "mes"}:
        return {}

    entries = []
    for rule in load_me_high_precision_locators():
        if not all(normalize_for_match(token) in query_norm for token in rule["tokens_all"]):
            continue
        page = rule["page"]
        title = rule["title"]
        entries.append(
            {
                "source": rule["source"],
                "book_title": rule.get("book_title", ""),
                "article": rule.get("article") or title,
                "classic_title": title,
                "start_page": page,
                "end_page": page,
                "pdf_page": rule.get("pdf_page"),
                "locator_quote": rule.get("quote"),
                "entry_type": "me_high_precision_locator",
                "priority": 0,
            }
        )

    if not entries:
        return {}

    title = entries[0]["classic_title"]
    return {
        "title": title,
        "strict_title": True,
        "entries": entries,
        "sources": {entry["source"] for entry in entries},
        "page_ranges": build_page_ranges(entries),
        "page_tolerance": 2,
        "high_precision_locator": True,
    }


def article_locator_constraints_from_query(query, title, ctx):
    normalize_for_match = _helper(ctx, "normalize_for_match")
    requested = collection_requested(query, ctx)
    if requested in {"mea", "mes"}:
        return {}

    query_norm = normalize_for_match(query)
    title_norm = normalize_for_match(title)
    entries = []

    derivative_markers = ["序言", "导言", "封面", "扉页", "第一页", "手稿的一页", "说明", "附录"]

    def article_rank(entry):
        article = entry.get("article") or ""
        article_norm = normalize_for_match(article)
        range_width = (entry.get("end_page") or 0) - (entry.get("start_page") or 0)
        derivative = any(marker in article for marker in derivative_markers)
        if title_norm and article_norm == title_norm:
            title_score = 0
        elif title_norm and title_norm in article_norm and not derivative:
            title_score = 1
        elif not derivative:
            title_score = 2
        else:
            title_score = 3
        return (
            source_priority(entry.get("source"), query, ctx),
            title_score,
            entry.get("priority", 99),
            -range_width,
            entry.get("start_page") or 9999,
        )

    for locator in load_me_article_locators():
        source = locator.get("source")
        article_title = locator.get("title") or ""
        article_norm = normalize_for_match(article_title)
        if not source or not article_norm:
            continue

        matched = False
        if len(article_norm) >= 4 and article_norm in query_norm:
            matched = True
        elif title_norm and len(title_norm) >= 4:
            matched = article_norm == title_norm or title_norm in article_norm or article_norm in title_norm
        elif len(article_norm) >= 8 and article_norm in query_norm:
            matched = True

        if not matched:
            continue

        start_page = locator.get("start_page")
        end_page = locator.get("end_page")
        if start_page is None or end_page is None:
            continue
        entry = {
            "source": source,
            "book_title": locator.get("book") or source,
            "article": article_title,
            "classic_title": article_title,
            "start_page": start_page,
            "end_page": end_page,
            "pdf_page": locator.get("pdf_start_page"),
            "pdf_end_page": locator.get("pdf_end_page"),
            "entry_type": "me_article_locator",
            "priority": 0 if locator.get("primary") else 1,
            "locator_type": locator.get("locator_type") or "article",
            "non_body": locator.get("non_body"),
        }
        if is_letter_title(article_title):
            entry["entry_type"] = "me_letter_locator"
            entry["is_letter"] = True
            entry["letter_title"] = article_title
            entry["no_page_citation"] = True
            entry["citation_mode"] = "letter_title"
        entries.append(entry)

    if not entries:
        return {}

    entries = prefer_sources_for_query(entries, query, ctx)
    used_query_exact_entries = False
    query_exact_entries = [
        entry for entry in entries
        if len(normalize_for_match(entry.get("article"))) >= 4
        and normalize_for_match(entry.get("article")) in query_norm
    ]
    if query_exact_entries:
        max_title_len = max(
            len(normalize_for_match(entry.get("article")))
            for entry in query_exact_entries
        )
        entries = [
            entry for entry in query_exact_entries
            if len(normalize_for_match(entry.get("article"))) == max_title_len
        ]
        used_query_exact_entries = True
    # Prefer exact title matches and primary article entries over derivative pages.
    exact_entries = [
        entry for entry in entries
        if normalize_for_match(entry.get("article")) == (title_norm or query_norm)
    ]
    if exact_entries:
        entries = exact_entries
    primary_entries = [entry for entry in entries if entry.get("priority", 99) == 0]
    if primary_entries and not used_query_exact_entries:
        entries = primary_entries
    entries = sorted(
        entries,
        key=article_rank,
    )[:8]

    result_title = title or entries[0].get("article")
    letter_locator = bool(entries) and all(entry.get("is_letter") for entry in entries)
    entry_title_norms = {normalize_for_match(entry.get("article")) for entry in entries if entry.get("article")}
    entry_locations = {
        (entry.get("source"), entry.get("start_page"), entry.get("end_page"))
        for entry in entries
    }
    ambiguous_locator = len(entries) > 1 and len(entry_locations) > 1 and len(entry_title_norms) <= 2
    return {
        "title": result_title,
        "strict_title": True,
        "entries": entries,
        "sources": {entry["source"] for entry in entries},
        "page_ranges": build_page_ranges(entries),
        "page_tolerance": 2,
        "article_locator": True,
        "ambiguous_locator": ambiguous_locator,
        "letter_locator": letter_locator,
        "no_page_citation": letter_locator,
        "citation_mode": "letter_title" if letter_locator else "",
    }


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

    if topic.get("id") == "peasant_cooperative" and "农民" in query_norm and is_broad_topic_query(query, ctx):
        return True

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


def is_broad_topic_query(query, ctx):
    normalize_for_match = _helper(ctx, "normalize_for_match")
    query_norm = normalize_for_match(query)
    if not query_norm:
        return False

    broad_markers = [
        "主要有哪些论述",
        "有哪些论述",
        "主要论述",
        "系统论述",
        "综合分析",
        "总体分析",
        "理论分析",
        "怎么看",
        "如何看待",
        "如何理解",
        "怎样理解",
        "主要观点",
        "观点",
        "主要看法",
        "看法",
        "主要主张",
        "主张",
        "论述",
        "主要内容",
        "梳理",
        "概括",
        "归纳",
        "总结",
    ]
    return any(normalize_for_match(marker) in query_norm for marker in broad_markers)


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
    work_catalog_title_entries_for_query = _helper(ctx, "work_catalog_title_entries_for_query")
    title = extract_bibliographic_title(query)
    normalize_for_match = _helper(ctx, "normalize_for_match")
    query_norm = normalize_for_match(query)

    if not title and is_broad_topic_query(query, ctx):
        broad_topic_constraints = narrow_topic_constraints_by_query(query, topic_entries_for_query(query, ctx), ctx)
        if broad_topic_constraints:
            broad_topic_constraints = dict(broad_topic_constraints)
            broad_topic_constraints["soft_topic"] = True
            broad_topic_constraints["strict_title"] = False
            return broad_topic_constraints

    high_precision_constraints = high_precision_locator_constraints_from_query(query, ctx)
    if high_precision_constraints:
        return high_precision_constraints

    article_locator_constraints = article_locator_constraints_from_query(query, title, ctx)
    if article_locator_constraints:
        return article_locator_constraints

    explicit_constraints = explicit_volume_constraints_from_query(query, ctx)
    if explicit_constraints:
        return explicit_constraints

    me_hint_constraints = me_title_hint_constraints_from_query(query, ctx)
    if me_hint_constraints:
        return me_hint_constraints

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
            return constraints_result(display_title or first_title, hinted_entries, query, ctx)

    title_catalog_entries = prefer_sources_for_query(work_catalog_title_entries_for_query(query), query, ctx)
    if title_catalog_entries:
        title_catalog_title = (
            title_catalog_entries[0].get("classic_title")
            or title_catalog_entries[0].get("article")
            or title
            or ""
        )
        work_concepts = []
        for entry in title_catalog_entries:
            work_concepts.extend(entry.get("classic_primary_concepts") or [])
            work_concepts.extend(entry.get("classic_concepts") or [])
        work_concepts = list(dict.fromkeys(str(item).strip() for item in work_concepts if str(item).strip()))[:8]
        work_query_seeds = []
        if work_concepts:
            work_query_seeds.append(f"{title_catalog_title} {' '.join(work_concepts[:4])}")
            work_query_seeds.append(f"{title_catalog_title} 核心观点 {' '.join(work_concepts[:3])}")
        return {
            "title": title_catalog_title,
            "strict_title": True,
            "entries": title_catalog_entries,
            "sources": {entry["source"] for entry in title_catalog_entries},
            "page_ranges": build_page_ranges(title_catalog_entries),
            "page_tolerance": 2,
            "work_catalog_title_match": True,
            "work_concepts": work_concepts,
            "work_query_seeds": work_query_seeds,
        }

    catalog_entries = prefer_sources_for_query(work_catalog_entries_for_query(query), query, ctx)
    catalog_title = ""
    if catalog_entries:
        catalog_title = catalog_entries[0].get("classic_title") or catalog_entries[0].get("article") or ""
    explicit_catalog_title = bool(catalog_title and normalize_for_match(catalog_title) in query_norm)

    topic_constraints = narrow_topic_constraints_by_query(query, topic_entries_for_query(query, ctx), ctx)
    if topic_constraints and is_broad_topic_query(query, ctx):
        topic_constraints = dict(topic_constraints)
        topic_constraints["soft_topic"] = True
        topic_constraints["strict_title"] = False
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
        return constraints_result(title, catalog_entries, query, ctx)

    locator_entries = prefer_sources_for_query(locator_entries_for_query(query), query, ctx)
    if locator_entries:
        title = locator_entries[0].get("classic_title") or locator_entries[0].get("article")
        return constraints_result(title, locator_entries, query, ctx)

    core_entries = classic_entries_for_query(title) if title else []
    if core_entries:
        entries = prefer_sources_for_query(enrich_core_classic_entries(core_entries), query, ctx)
        title = title or entries[0].get("classic_title")
        return constraints_result(title, entries, query, ctx)

    concept_constraints = concept_constraints_from_query(query, ctx)
    if concept_constraints:
        return concept_constraints

    topic_constraints = narrow_topic_constraints_by_query(query, topic_entries_for_query(query, ctx), ctx)
    if topic_constraints and is_broad_topic_query(query, ctx):
        topic_constraints = dict(topic_constraints)
        topic_constraints["soft_topic"] = True
        topic_constraints["strict_title"] = False
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

    entries = prefer_sources_for_query(find_toc_entries(title), query, ctx)
    if not entries:
        return {"title": title}

    return constraints_result(title, entries, query, ctx)


def topic_seed_queries(query, constraints, ctx):
    normalize_for_match = _helper(ctx, "normalize_for_match")
    seeds = []
    query = str(query or "").strip()
    if query:
        seeds.append(query)

    topic_title = str(constraints.get("topic_title") or "").strip()
    if topic_title:
        seeds.append(topic_title)

    for marker in (constraints.get("topic_markers") or [])[:4]:
        marker = str(marker or "").strip()
        if marker and topic_title:
            seeds.append(f"{topic_title} {marker}")
        elif marker:
            seeds.append(marker)

    entries = constraints.get("entries") or []
    entry_limit = 8 if constraints.get("soft_topic") else 4
    for entry in entries[:entry_limit]:
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
        if constraints.get("soft_topic") and len(deduped) >= 12:
            break
        if not constraints.get("soft_topic") and len(deduped) >= 6:
            break
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
        for seed in constraints.get("work_query_seeds") or []:
            add_seed(seeds, seed)
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
        limit = 8 if constraints.get("soft_topic") else 4
        if len(deduped) >= limit:
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
