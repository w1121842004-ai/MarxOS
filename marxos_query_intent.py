from __future__ import annotations

import re


def extract_quoted_title(query: str, clean_text) -> str | None:
    query = clean_text(query, "")
    match = re.search(r"《([^》]+)》", query)
    if match:
        return match.group(1).strip()
    return None


def extract_unquoted_title(query: str, clean_text) -> str | None:
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
    positions = [query.find(keyword) for keyword in keywords if keyword in query]
    if not positions:
        return None

    title = query[: min(positions)]
    title = re.sub(r"[，。：、\s\"'“”《》（）()]+$", "", title).strip()
    return title or None


def extract_bibliographic_title(query: str, clean_text) -> str | None:
    return extract_quoted_title(query, clean_text) or extract_unquoted_title(query, clean_text)


def normalize_for_match(text: str, clean_text) -> str:
    text = clean_text(text, "")
    text = re.sub(r"[《》“”\"'（）()，。；：、\s·\-.—–]", "", text)
    return text.lower()


def is_bibliographic_query(query: str, clean_text) -> bool:
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


def is_quote_lookup_query(query: str, clean_text) -> bool:
    query = clean_text(query, "")
    if extract_bibliographic_title(query, clean_text):
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


def is_analysis_query(query: str, clean_text) -> bool:
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


def is_classic_sayings_query(query: str, clean_text) -> bool:
    query = clean_text(query, "")
    saying_markers = ["经典语句", "经典名句", "名言", "名句", "语录"]
    author_markers = ["马克思", "恩格斯", "马恩", "马克思主义"]
    return any(marker in query for marker in saying_markers) and any(marker in query for marker in author_markers)
