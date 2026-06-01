from __future__ import annotations

import re


"""
MarxOS Query Intent Router — Three-way classification.

Intent types:
  quote_lookup      — user pasted a quote, wants source/page. → exact search.
  bibliographic     — user asks "where is X?" → work_catalog match.
  concept_explain   — user asks "what is X?" → constrained retrieval + concept prompt.
  theory_analysis   — user asks "how to understand/analyze X?" → broad retrieval + analysis prompt.
  rag_answer         — default catch-all → standard RAG pipeline.
"""


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
    title = re.sub(r"""[，。：、《》\s"'“”‘’（）()？?]+$""", "", title).strip()
    return title or None


def extract_bibliographic_title(query: str, clean_text) -> str | None:
    return extract_quoted_title(query, clean_text) or extract_unquoted_title(query, clean_text)


def normalize_for_match(text: str, clean_text) -> str:
    text = clean_text(text, "")
    text = re.sub(r"""[《》“”"'，。：；、\s\-—()（）？?]""", "", text)
    return text.lower()


def is_bibliographic_query(query: str, clean_text) -> bool:
    query = clean_text(query, "")
    title = extract_bibliographic_title(query, clean_text)
    if not title:
        return False

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
    if not any(keyword in query for keyword in keywords):
        return False

    if len(title) >= 24:
        return False

    analytical_markers = [
        "说明",
        "阐明",
        "概括",
        "总结",
        "反驳",
        "分析",
        "进一步",
        "基于",
        "结合",
        "讨论",
        "线索",
    ]
    return not any(marker in query for marker in analytical_markers)


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

    analytical_markers = [
        "总结",
        "概括",
        "线索",
        "系统",
        "说明",
        "阐明",
        "分析",
        "进一步",
        "基于",
        "结合",
        "讨论",
        "方法论",
        "再次注明",
    ]
    if any(marker in query for marker in analytical_markers):
        return False

    quote_keywords = ["引文", "出处", "出自", "哪一页", "哪页", "页码", "原文", "这句话", "这段话"]
    if any(keyword in query for keyword in quote_keywords):
        return True

    return (
        len(query) >= 24
        and any(marker in query for marker in ["“", "”", "\"", "《", "》"])
        and not re.search(r"[。！？!?]", query)
    )


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


def is_deep_analysis_query(query: str, clean_text) -> bool:
    """Detect queries that need LLM-powered deep analysis / paper writing.

    Triggers when the user asks for:
      - Social/theoretical analysis of contemporary phenomena
      - Marxist theoretical paper or essay
      - Multi-work synthesis on a topic
      - Application of Marxist concepts to new domains
    """
    q = clean_text(query, "")

    # Strong analysis markers → definitely deep analysis
    strong_markers = [
        "运用马克思主义", "从马克思主义", "马克思主义视角", "马克思主义分析",
        "写一篇", "撰写", "论文", "学术分析",
        "理论分析", "社会分析", "历史分析", "辩证分析",
        "当代意义", "现实意义", "当代价值", "现实启示",
    ]
    if any(m in q for m in strong_markers):
        return True

    # Compound: concept + contemporary domain → likely analysis
    concept_markers = ["异化", "阶级", "资本", "剩余价值", "生产关系", "意识形态", "辩证法",
                       "唯物史观", "商品拜物教", "原始积累", "国家理论"]
    domain_markers = ["当代", "数字", "平台", "全球", "新形式", "现代", "当今", "互联网",
                      "人工智能", "金融化", "新自由主义", "治理"]
    has_concept = any(m in q for m in concept_markers)
    has_domain = any(m in q for m in domain_markers)
    if has_concept and has_domain:
        return True

    # Long, complex questions (>50 chars) with analysis keywords
    analysis_keywords = ["分析", "论述", "阐述", "如何理解", "怎么看", "如何评价",
                         "关系", "机制", "逻辑", "根源", "本质"]
    if len(q) >= 50 and any(k in q for k in analysis_keywords):
        return True

    return False


# ── Unified Intent Router ────────────────────────────────────────

def classify_query(query: str, clean_text) -> str:
    """Six-way classification: bibliographic | quote | concept | deep_analysis | theory_analysis | rag_answer."""
    if is_bibliographic_query(query, clean_text) and extract_bibliographic_title(query, clean_text):
        return "bibliographic_lookup"
    if is_quote_lookup_query(query, clean_text):
        return "quote_lookup"

    q = clean_text(query, "")
    concept_markers = ["什么是", "何为", "概念", "定义", "解释一下", "是什么",
                       "是什么意思", "如何理解", "这个概念"]
    if any(k in q for k in concept_markers):
        return "concept_explain"

    # Deep analysis: multi-work synthesis, contemporary application, paper writing
    if is_deep_analysis_query(query, clean_text):
        return "deep_analysis"

    if is_analysis_query(query, clean_text):
        return "theory_analysis"
    return "rag_answer"


def route_query(query: str, clean_text, normalize_for_match, catalog=None):
    """Unified query router: classify intent + match work + recommend strategy.

    Returns dict:
      - intent: classified intent string
      - work_id: matched work_id from catalog (or None)
      - work_title: matched work title (or None)
      - strategy: recommended retrieval strategy
    """
    intent = classify_query(query, clean_text)
    work_id = None
    work_title = None
    strategy = "standard"

    # Try catalog match
    if catalog is not None:
        try:
            work = catalog.match_query(query, normalize_fn=normalize_for_match)
        except Exception:
            work = None

        if work:
            work_id = work.get("work_id")
            work_title = work.get("title")
            # Strategy depends on intent + match quality
            if intent == "quote_lookup":
                strategy = "exact_quote_with_work_constraint"
            elif intent == "bibliographic_lookup":
                strategy = "work_catalog_direct"
            elif intent == "concept_explain":
                strategy = "concept_constrained_retrieval"
            elif intent == "theory_analysis":
                strategy = "work_constrained_multi_source"
            else:
                strategy = "work_constrained_retrieval"
        else:
            if intent == "quote_lookup":
                strategy = "exact_quote_search"
            elif intent == "bibliographic_lookup":
                strategy = "toc_fallback"
            elif intent == "concept_explain":
                strategy = "concept_broad_retrieval"

    return {
        "intent": intent,
        "work_id": work_id,
        "work_title": work_title,
        "strategy": strategy,
    }
