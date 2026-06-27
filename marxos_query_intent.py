"""
MarxOS Query Intent Router — Scoring-based multi-way classification.

**Architecture (v2):**  Layered scoring replaces the v1 hard-priority cascade.
Each query is scored against *every* intent dimension and a probability
distribution is returned.  Confidence signals let downstream code adapt
retrieval strictness / prompt selection / CRAG thresholds.

**Intent taxonomy (7 intents):**

=============== =============================================================
Intent            Description
=============== =============================================================
bibliographic     Locate a work / volume / page range.
quote             Confirm source + page for an exact passage.
concept           Explain a concept ("what is X?").
comparison        Compare works, concepts, or authors.
deep_analysis     Multi-work synthesis, contemporary application, paper writing.
theory_analysis   Analyse through Marxist theoretical lens.
rag_answer        Default catch-all — standard hybrid retrieval.
=============== =============================================================

**Backward compatibility:**  ``classify_query()`` returns a plain string (v1
behaviour).  ``classify_query_v2()`` returns an ``IntentResult`` dataclass
whose ``__str__`` / ``__eq__`` delegate to ``.primary``, so existing
``query_intent == "concept_explain"`` comparisons still work transparently.

**Dependencies:** ``jieba`` (lightweight Chinese tokeniser, ~15 MB).

**ML classifier (v3):**  When ``data/intent_classifier.pkl`` exists,
``classify_query_v2()`` blends the rule-based scores with a lightweight
logistic-regression classifier trained on the same embedding model already
loaded by ``marxos_runtime``.  The classifier head is ~10 KB; inference
adds <1 ms.  Training script: ``scripts/build_intent_classifier.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

try:
    import jieba.posseg as pseg  # type: ignore

    _JIEBA_AVAILABLE = True
except ImportError:
    pseg = None  # type: ignore
    _JIEBA_AVAILABLE = False

# Lazy-loaded ML classifier (blends with rule scores when available)
_classifier_cache: object | None = None  # IntentClassifier or None (unloaded)
_CLASSIFIER_LOAD_ATTEMPTED = False


# ---------------------------------------------------------------------------
# Intent result dataclass (v2)
# ---------------------------------------------------------------------------


@dataclass
class IntentResult:
    """Rich intent classification result.

    ``__str__`` and ``__eq__`` delegate to ``.primary`` so this object can
    transparently replace a plain string in most call sites.
    """

    primary: str
    confidence: float
    distribution: dict[str, float] = field(default_factory=dict)
    is_ambiguous: bool = False
    question_type: str = "NONE"
    question_subtype: str = ""

    # -- transparent string compatibility --------------------------------

    def __str__(self) -> str:
        return self.primary

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.primary == other
        if isinstance(other, IntentResult):
            return self.primary == other.primary
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.primary)

    def __bool__(self) -> bool:
        return True

    # -- helpers ---------------------------------------------------------

    def is_any(self, *intents: str) -> bool:
        return self.primary in intents

    def secondary_intents(self, threshold: float = 0.2) -> list[tuple[str, float]]:
        return [
            (k, v) for k, v in sorted(self.distribution.items(), key=lambda x: -x[1])
            if v >= threshold and k != self.primary
        ]


# ---------------------------------------------------------------------------
# Question-structure analysis (Phase 2)
# ---------------------------------------------------------------------------

_Q_WORDS: dict[str, str] = {
    "什么是": "WHAT_DEFINITION",
    "何为": "WHAT_DEFINITION",
    "是什么": "WHAT_DEFINITION",
    "是什么意思": "WHAT_DEFINITION",
    "定义": "WHAT_DEFINITION",
    "概念": "WHAT_DEFINITION",
    "如何": "HOW",
    "怎么": "HOW",
    "怎样": "HOW",
    "如何理解": "HOW_UNDERSTAND",
    "怎么看": "HOW_UNDERSTAND",
    "怎么看待": "HOW_UNDERSTAND",
    "为什么": "WHY",
    "为何": "WHY",
    "原因": "WHY",
    "哪里": "WHERE",
    "哪一卷": "WHERE",
    "哪卷": "WHERE",
    "第几卷": "WHERE",
    "哪页": "WHERE",
    "出自": "WHERE",
    "出处": "WHERE",
    "收录": "WHERE",
    "在哪": "WHERE",
    "比较": "COMPARE",
    "区别": "COMPARE",
    "异同": "COMPARE",
    "对比": "COMPARE",
    "不同": "COMPARE",
    "差异": "COMPARE",
    "总结": "SUMMARIZE",
    "概括": "SUMMARIZE",
    "梳理": "SUMMARIZE",
    "归纳": "SUMMARIZE",
    "综述": "SUMMARIZE",
}

# Multi-character markers that jieba may split — we check contiguous chars in query
_COMPOUND_MARKERS: list[tuple[str, str]] = [
    ("什么是", "WHAT_DEFINITION"),
    ("是什么", "WHAT_DEFINITION"),
    ("是什么意思", "WHAT_DEFINITION"),
    ("如何理解", "HOW_UNDERSTAND"),
    ("怎么看待", "HOW_UNDERSTAND"),
    ("哪一卷", "WHERE"),
    ("第几卷", "WHERE"),
    ("出自哪里", "WHERE"),
    ("在哪一卷", "WHERE"),
]


def _detect_question_type_jieba(query: str) -> tuple[str, str]:
    """Use jieba POS tagging for finer-grained question-type detection."""
    # 1. Check compound markers first (handles jieba splitting "什么是"→"什么"+"是")
    scored: list[tuple[int, str]] = []
    for marker, qtype in _COMPOUND_MARKERS:
        if marker in query:
            scored.append((len(marker), qtype))
    if scored:
        scored.sort(key=lambda x: -x[0])
        return scored[0][1], ""

    # 2. Check single-word markers via jieba
    try:
        words = list(pseg.cut(query))
    except Exception:
        words = []

    for word, _flag in words:
        for marker, qtype in _Q_WORDS.items():
            if marker == word or (len(marker) >= 2 and marker in word):
                scored.append((len(marker), qtype))
    if scored:
        scored.sort(key=lambda x: -x[0])
        return scored[0][1], ""

    # 3. POS-based fallback
    for word, flag in words:
        if flag.startswith("r"):  # pronoun
            if word in ("什么", "啥"):
                return "WHAT", ""
            elif word in ("哪", "哪里"):
                return "WHERE", ""
            elif word in ("谁",):
                return "WHO", ""

    return "NONE", ""


def _detect_question_type_regex(query: str) -> tuple[str, str]:
    """Regex-based fallback when jieba is unavailable."""
    scored: list[tuple[int, str]] = []
    for marker, qtype in _Q_WORDS.items():
        if marker in query:
            scored.append((len(marker), qtype))
    if scored:
        scored.sort(key=lambda x: -x[0])
        return scored[0][1], ""
    return "NONE", ""


def analyse_query_structure(query: str) -> dict:
    """Extract the structural type of a user query.

    Returns::

        {
            "question_type": "WHAT_DEFINITION" | "HOW" | "WHY" | "WHERE"
                           | "COMPARE" | "SUMMARIZE" | "NONE",
            "has_book_title": bool,     # 《…》 present
            "has_quote": bool,          # ""''「」 present
            "has_analysis_verb": bool,  # 分析/论述/阐述/评价
            "has_locate_verb": bool,    # 出自/来源/收录/在哪
            "has_contemporary": bool,   # 当代/数字/平台/互联网/...
            "char_length": int,
            "is_sentence": bool,        # ends with 。！？!?
        }
    """
    qtype, subtype = (
        _detect_question_type_jieba(query)
        if _JIEBA_AVAILABLE
        else _detect_question_type_regex(query)
    )

    analysis_verbs = ["分析", "论述", "阐述", "评价", "批判", "论证"]
    locate_verbs = ["出自", "出处", "来源", "收录", "在哪", "哪卷", "哪页"]
    contemporary = [
        "当代", "数字", "平台", "全球", "互联网", "人工智能",
        "金融化", "新自由主义", "现代", "当今", "治理",
    ]

    return {
        "question_type": qtype,
        "question_subtype": subtype,
        "has_book_title": "《" in query,
        "has_quote": any(m in query for m in ('"', '"', "'", "「", "『")),
        "has_analysis_verb": any(v in query for v in analysis_verbs),
        "has_locate_verb": any(v in query for v in locate_verbs),
        "has_contemporary": any(v in query for v in contemporary),
        "char_length": len(query),
        "is_sentence": bool(re.search(r"[。！？!?]$", query.strip())),
    }


# ---------------------------------------------------------------------------
# Scoring functions — one per intent (0.0–1.0)
# ---------------------------------------------------------------------------


def _score_bibliographic(query: str, struct: dict, clean_text) -> float:
    """Score for bibliographic lookup intent."""
    score = 0.0

    # Strong signals
    if struct["has_book_title"] and struct["has_locate_verb"]:
        score += 0.45
    elif struct["has_book_title"] and struct["question_type"] == "WHERE":
        score += 0.40
    elif struct["has_book_title"]:
        score += 0.15  # 有书名但不确定是要找位置还是问内容
    elif struct["question_type"] == "WHERE":
        score += 0.20

    # Bonus: explicit bibliographic keywords
    bib_keywords = ["收录", "在哪", "哪卷", "哪页", "第几卷"]
    hits = sum(1 for kw in bib_keywords if kw in query)
    score += min(hits * 0.08, 0.24)

    # Volume references — computed once, used by bonus + penalty sections
    volume_refs = ["马恩全集", "马克思恩格斯全集", "文集", "选集", "哪一卷", "第几卷"]
    vol_ref_hits = sum(1 for v in volume_refs if v in query)

    # Volume reference + locate verb → strong bibliographic signal
    if struct["has_locate_verb"] and vol_ref_hits >= 1:
        score += 0.20 + (vol_ref_hits - 1) * 0.05
    # Without book title but with explicit volume keywords → likely bibliographic
    if not struct["has_book_title"] and vol_ref_hits >= 2:
        score += 0.15

    # Penalty: analysis/definition/summarize markers suggest not bibliographic
    if struct["has_analysis_verb"]:
        score -= 0.15
    if struct["question_type"] in ("WHAT_DEFINITION", "WHAT", "HOW", "HOW_UNDERSTAND",
                                    "SUMMARIZE", "COMPARE"):
        score -= 0.25
    # Long text + locate without book title and without volume keywords → likely quote
    if (struct["has_locate_verb"] and not struct["has_book_title"]
            and struct["char_length"] >= 10 and struct["question_type"] == "WHERE"
            and vol_ref_hits < 2):  # exception: explicit volume references → biblio
        score -= 0.25
    if struct["char_length"] > 50:
        score -= 0.05

    return max(0.0, min(1.0, score))


def _score_quote(query: str, struct: dict, clean_text) -> float:
    """Score for quote lookup intent."""
    score = 0.0

    # Strong signals: quote markers + locate intent
    if struct["has_quote"] and struct["has_locate_verb"]:
        score += 0.40
    elif struct["has_quote"] and struct["question_type"] == "WHERE":
        score += 0.35
    elif struct["has_quote"] and struct["char_length"] >= 24:
        score += 0.20

    # Long text + locate verb WITHOUT quotes (e.g. "XX出自哪里")
    # This catches quote lookups where the user pasted text without quotation marks
    if not struct["has_quote"] and struct["has_locate_verb"] and struct["char_length"] >= 10:
        score += 0.30
    if not struct["has_quote"] and struct["question_type"] == "WHERE" and struct["char_length"] >= 12:
        score += 0.20

    # Quote-locate keywords
    quote_kw = ["引文", "出处", "出自", "原文", "这句话", "这段话", "哪页"]
    kw_hits = sum(1 for kw in quote_kw if kw in query)
    score += min(kw_hits * 0.08, 0.24)

    # Long text with quotes, no question word → typical quote paste
    if struct["has_quote"] and struct["question_type"] == "NONE" and not struct["is_sentence"]:
        score += 0.25

    # Penalties
    if struct["question_type"] in ("WHAT_DEFINITION", "HOW", "HOW_UNDERSTAND", "WHY"):
        score -= 0.25
    if struct["has_analysis_verb"]:
        score -= 0.15
    if struct["has_book_title"] and not struct["has_quote"]:
        score -= 0.10  # more likely bibliographic
    # Volume reference with locate → likely bibliography, not quote
    volume_refs = ["马恩全集", "马克思恩格斯全集", "文集", "选集", "哪一卷", "第几卷"]
    if not struct["has_quote"] and any(v in query for v in volume_refs):
        score -= 0.15

    return max(0.0, min(1.0, score))


def _score_concept(query: str, struct: dict, clean_text) -> float:
    """Score for concept explanation intent."""
    score = 0.0

    # Strong signals: definition-type question
    if struct["question_type"] == "WHAT_DEFINITION":
        score += 0.45
    elif struct["question_type"] in ("WHAT", "HOW_UNDERSTAND"):
        score += 0.20

    concept_markers = ["什么是", "何为", "概念", "定义", "解释", "是什么", "是什么意思"]
    hits = sum(1 for kw in concept_markers if kw in query)
    score += min(hits * 0.08, 0.24)

    # Short query + what-type → strong concept signal
    if struct["char_length"] < 30 and struct["question_type"] in ("WHAT_DEFINITION", "WHAT"):
        score += 0.10

    # Has book title + what question → "《X》中的Y是什么"
    # Boost concept over bibliographic for this pattern
    if struct["has_book_title"] and struct["question_type"] in ("WHAT_DEFINITION", "WHAT"):
        score += 0.20  # was 0.10 — stronger boost to beat bibliographic

    # Penalties
    if struct["has_analysis_verb"] and struct["question_type"] not in ("WHAT_DEFINITION", "WHAT"):
        score -= 0.10
    if struct["question_type"] == "COMPARE":
        score -= 0.15
    # Has quote + HOW_UNDERSTAND → more likely theory_analysis than concept
    if struct["has_quote"] and struct["question_type"] in ("HOW", "HOW_UNDERSTAND"):
        score -= 0.15

    return max(0.0, min(1.0, score))


def _score_comparison(query: str, struct: dict, clean_text) -> float:
    """Score for comparison intent (NEW in v2)."""
    score = 0.0

    if struct["question_type"] == "COMPARE":
        score += 0.50

    compare_markers = ["比较", "区别", "异同", "对比", "不同", "差异", "vs", "VS", "和"]
    hits = sum(1 for kw in compare_markers if kw in query)
    if struct["question_type"] != "COMPARE" and hits >= 2:
        score += 0.30
    elif hits >= 1:
        score += 0.15

    # Entity pair detection: 《A》和《B》 / "A"和"B"
    if len(re.findall(r"[《「][^》」]+[》」]", query)) >= 2:
        score += 0.20
    if re.search(r"(?:和|与|跟).*(?:和|与|跟)", query):
        score += 0.05

    # Penalties
    if struct["question_type"] == "WHAT_DEFINITION":
        score -= 0.10

    return max(0.0, min(1.0, score))


def _score_deep_analysis(query: str, struct: dict, clean_text) -> float:
    """Score for deep analysis intent."""
    score = 0.0

    # Strong markers
    strong = [
        "运用马克思主义", "从马克思主义", "马克思主义视角", "马克思主义分析",
        "写一篇", "撰写", "论文", "学术分析",
        "当代意义", "现实意义", "当代价值", "现实启示",
    ]
    if any(m in query for m in strong):
        score += 0.40

    # Concept + contemporary domain compound
    concept_set = {
        "异化", "阶级", "资本", "剩余价值", "生产关系", "意识形态",
        "辩证法", "唯物史观", "商品拜物教", "原始积累", "国家理论",
    }
    has_concept = any(m in query for m in concept_set)
    if has_concept and struct["has_contemporary"]:
        score += 0.35

    # Long, complex queries with analysis/summarize keywords
    analysis_kw = ["分析", "论述", "阐述", "如何理解", "怎么看", "如何评价",
                    "关系", "机制", "逻辑", "根源", "本质"]
    summarize_kw = ["总结", "概括", "梳理", "归纳", "综述"]
    if struct["char_length"] >= 50 and any(k in query for k in analysis_kw):
        score += 0.25
    # Summarize + book title → deep analysis ("总结《资本论》X理论")
    if any(k in query for k in summarize_kw) and struct["has_book_title"]:
        score += 0.25

    # Communism prediction questions
    communism = [
        "共产主义是不是", "共产主义是否", "共产主义会不会",
        "共产主义能不能", "共产主义能否", "共产主义一定会实现",
        "共产主义必然实现", "共产主义会实现",
    ]
    if any(p in query for p in communism):
        score += 0.25

    return max(0.0, min(1.0, score))


def _score_theory_analysis(query: str, struct: dict, clean_text) -> float:
    """Score for theory analysis intent."""
    score = 0.0

    if struct["has_analysis_verb"]:
        score += 0.25
    if struct["question_type"] in ("HOW", "HOW_UNDERSTAND"):
        score += 0.20
    if struct["question_type"] == "WHY":
        score += 0.15

    # Quote + "如何理解" → analyzing a specific passage
    if struct["has_quote"] and struct["question_type"] in ("HOW", "HOW_UNDERSTAND"):
        score += 0.15

    analysis_kw = ["分析", "怎么看", "如何理解", "为什么", "现实", "意义", "关系", "评价"]
    hits = sum(1 for kw in analysis_kw if kw in query)
    score += min(hits * 0.05, 0.20)

    # Penalty: no Marxist content signal
    marxist_signals = [
        "马克思", "恩格斯", "资本", "剩余价值", "阶级", "异化", "意识形态",
        "商品", "生产", "社会主义", "共产主义", "唯物", "辩证法", "剥削",
    ]
    if not any(s in query for s in marxist_signals) and not struct["has_book_title"]:
        score -= 0.20  # likely out-of-domain

    if struct["has_contemporary"]:
        score -= 0.05

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Scoring orchestrator
# ---------------------------------------------------------------------------

INTENT_SCORERS = {
    "bibliographic_lookup": _score_bibliographic,
    "quote_lookup": _score_quote,
    "concept_explain": _score_concept,
    "comparison": _score_comparison,
    "deep_analysis": _score_deep_analysis,
    "theory_analysis": _score_theory_analysis,
}


def compute_intent_scores(
    query: str, clean_text, struct: dict | None = None
) -> dict[str, float]:
    """Score every intent for *query* and return raw scores (0–1)."""
    if struct is None:
        struct = analyse_query_structure(query)

    raw: dict[str, float] = {}
    for intent, scorer in INTENT_SCORERS.items():
        raw[intent] = scorer(query, struct, clean_text)

    # rag_answer always gets a floor of 0.15
    raw["rag_answer"] = 0.15

    return raw


# Tie-breaking priority for intents with equal probability.
# Higher index = wins tie.  bibliographic > quote > concept > comparison
# > deep_analysis > theory_analysis > rag_answer.
_INTENT_TIE_ORDER = [
    "rag_answer",
    "theory_analysis",
    "deep_analysis",
    "comparison",
    "concept_explain",
    "quote_lookup",
    "bibliographic_lookup",
]


def intent_distribution(raw_scores: dict[str, float]) -> dict[str, float]:
    """Normalise raw scores to a probability distribution."""
    total = sum(raw_scores.values())
    if total <= 0:
        return {k: 1.0 / len(raw_scores) for k in raw_scores}
    return {k: v / total for k, v in raw_scores.items()}


# ---------------------------------------------------------------------------
# Public API (v2 — rich result)
# ---------------------------------------------------------------------------


def _get_classifier():
    """Lazy-load the ML intent classifier (v3).  Returns None if unavailable."""
    global _classifier_cache, _CLASSIFIER_LOAD_ATTEMPTED
    if _CLASSIFIER_LOAD_ATTEMPTED:
        return _classifier_cache  # type: ignore[return-value]
    _CLASSIFIER_LOAD_ATTEMPTED = True
    try:
        from marxos_intent_classifier import IntentClassifier  # type: ignore
        _classifier_cache = IntentClassifier.load()
    except Exception:
        _classifier_cache = None
    return _classifier_cache  # type: ignore[return-value]


def classify_query_v2(query: str, clean_text) -> IntentResult:
    """Score-based intent classification with optional ML blending.

    When ``data/intent_classifier.pkl`` exists the rule-based scores are
    blended with a lightweight classifier trained on the same embedding
    model — combining the coverage of ML with the precision of rules.

    Returns ``IntentResult`` which can be used as a drop-in replacement
    for the v1 string return (``str(result)`` → primary intent).
    """
    struct = analyse_query_structure(query)
    raw = compute_intent_scores(query, clean_text, struct)

    # ── v3: blend with ML classifier when available ──
    classifier = _get_classifier()
    if classifier is not None:
        try:
            from marxos_intent_classifier import blend_predictions
            # Get embedding from the runtime (reuses already-loaded model)
            import app
            embedding_fn = getattr(app, "embed_query", None)
            if embedding_fn is not None:
                embedding = embedding_fn(query)
            else:
                embedding = None
            raw = blend_predictions(query, embedding, raw, classifier, blend_weight=0.6)
        except Exception:
            pass  # fall back to rule-only scores

    dist = intent_distribution(raw)

    # Primary = highest probability, with tie-breaking
    primary = max(dist, key=lambda k: (dist[k], _INTENT_TIE_ORDER.index(k) if k in _INTENT_TIE_ORDER else 0))
    confidence = dist[primary]

    # Ambiguity check: >=2 intents within 0.15 of the winner
    sorted_intents = sorted(dist.items(), key=lambda x: -x[1])
    is_ambiguous = (
        len(sorted_intents) >= 2
        and (sorted_intents[0][1] - sorted_intents[1][1]) < 0.15
    )

    return IntentResult(
        primary=primary,
        confidence=confidence,
        distribution=dist,
        is_ambiguous=is_ambiguous,
        question_type=struct["question_type"],
        question_subtype=struct["question_subtype"],
    )


# ---------------------------------------------------------------------------
# Public API (v1 — backward-compatible string return)
# ---------------------------------------------------------------------------

# Keep the original helper functions for backward compatibility.
# They are called from app.py:885-916.


def _clean_query(query: str, clean_text) -> str:
    return clean_text(query, "")


def extract_quoted_title(query: str, clean_text) -> str | None:
    query = clean_text(query, "")
    match = re.search(r"《([^》]+)》", query)
    if match:
        return match.group(1).strip()
    return None


def extract_unquoted_title(query: str, clean_text) -> str | None:
    query = clean_text(query, "")
    keywords = [
        "在哪一卷", "在哪卷", "哪一卷", "第几卷",
        "收录在哪里", "收在哪里", "收录在哪", "收在哪",
        "在哪里", "出自哪卷", "属于哪卷", "在哪本",
        "从哪页", "从哪一页", "从第几页", "哪一页开始",
        "从哪一页开始", "起始页", "开始页", "收录页",
    ]
    positions = [query.find(keyword) for keyword in keywords if keyword in query]
    if not positions:
        return None
    title = query[: min(positions)]
    title = re.sub(r"""[，。：《》、\s"'""'（）()？?]+$""", "", title).strip()
    return title or None


def extract_bibliographic_title(query: str, clean_text) -> str | None:
    return extract_quoted_title(query, clean_text) or extract_unquoted_title(query, clean_text)


def normalize_for_match(text: str, clean_text) -> str:
    text = clean_text(text, "")
    text = re.sub(r"""[《》""'，。：；、\s\-—()（）？?]""", "", text)
    return text.lower()


def is_bibliographic_query(query: str, clean_text) -> bool:
    query = clean_text(query, "")
    title = extract_bibliographic_title(query, clean_text)
    if not title:
        return False
    keywords = [
        "在哪一卷", "在哪卷", "哪一卷", "第几卷",
        "收录在哪里", "收在哪里", "收录在哪", "收在哪",
        "在哪里", "出自哪卷", "属于哪卷", "在哪本",
        "从哪页", "从哪一页", "从第几页", "哪一页开始",
        "从哪一页开始", "起始页", "开始页", "收录页",
    ]
    if not any(keyword in query for keyword in keywords):
        return False
    if len(title) >= 24:
        return False
    analytical_markers = [
        "说明", "阐明", "概括", "总结", "反驳", "分析",
        "进一步", "基于", "结合", "讨论", "线索",
    ]
    return not any(marker in query for marker in analytical_markers)


def is_quote_lookup_query(query: str, clean_text) -> bool:
    query = clean_text(query, "")
    if extract_bibliographic_title(query, clean_text):
        return False
    interrogative_markers = [
        "什么是", "是什么", "何为", "如何", "怎么",
        "怎样", "为什么", "本质", "意义",
    ]
    if any(marker in query for marker in interrogative_markers):
        return False
    analytical_markers = [
        "总结", "概括", "线索", "系统", "说明", "阐明",
        "分析", "进一步", "基于", "结合", "讨论", "方法论", "再次注明",
    ]
    if any(marker in query for marker in analytical_markers):
        return False
    quote_keywords = ["引文", "出处", "出自", "哪一页", "哪页", "页码", "原文", "这句话", "这段话"]
    if any(keyword in query for keyword in quote_keywords):
        return True
    return (
        len(query) >= 24
        and any(marker in query for marker in ['"', '"', "《", "》"])
        and not re.search(r"[。！？!?]", query)
    )


def is_analysis_query(query: str, clean_text) -> bool:
    query = clean_text(query, "")
    communism_patterns = [
        "共产主义是不是", "共产主义是否", "共产主义会不会",
        "共产主义能不能", "共产主义能否", "共产主义一定会实现",
        "共产主义必然实现", "共产主义会实现",
    ]
    if any(pattern in query for pattern in communism_patterns):
        return True
    return any(
        keyword in query
        for keyword in [
            "分析", "怎么看", "怎么看待", "如何理解", "为什么",
            "现实", "结合现实", "现实表现", "意义", "当代意义", "关系", "评价",
        ]
    )


def is_classic_sayings_query(query: str, clean_text) -> bool:
    query = clean_text(query, "")
    saying_markers = ["经典语句", "经典名句", "名言", "名句", "语录"]
    author_markers = ["马克思", "恩格斯", "马恩", "马克思主义"]
    return any(marker in query for marker in saying_markers) and any(
        marker in query for marker in author_markers
    )


def is_deep_analysis_query(query: str, clean_text) -> bool:
    q = clean_text(query, "")
    strong_markers = [
        "运用马克思主义", "从马克思主义", "马克思主义视角", "马克思主义分析",
        "写一篇", "撰写", "论文", "学术分析",
        "理论分析", "社会分析", "历史分析", "辩证分析",
        "当代意义", "现实意义", "当代价值", "现实启示",
    ]
    if any(m in q for m in strong_markers):
        return True
    concept_markers = [
        "异化", "阶级", "资本", "剩余价值", "生产关系", "意识形态",
        "辩证法", "唯物史观", "商品拜物教", "原始积累", "国家理论",
    ]
    domain_markers = [
        "当代", "数字", "平台", "全球", "新形式", "现代", "当今",
        "互联网", "人工智能", "金融化", "新自由主义", "治理",
    ]
    has_concept = any(m in q for m in concept_markers)
    has_domain = any(m in q for m in domain_markers)
    if has_concept and has_domain:
        return True
    analysis_keywords = [
        "分析", "论述", "阐述", "如何理解", "怎么看", "如何评价",
        "关系", "机制", "逻辑", "根源", "本质",
    ]
    if len(q) >= 50 and any(k in q for k in analysis_keywords):
        return True
    return False


# ── v1-compatible classify_query (returns plain string) ──────────────


def classify_query(query: str, clean_text) -> str:
    """Six-way classification (v1 backward-compatible).

    Returns a **plain string**.  Prefer ``classify_query_v2()`` for
    new code that can benefit from confidence / ambiguity signals.
    """
    result = classify_query_v2(query, clean_text)
    return result.primary


# ── Unified router (v1-compatible dict return) ──────────────────────


def route_query(query: str, clean_text, normalize_for_match, catalog=None):
    """Unified query router: classify intent + match work + recommend strategy.

    Returns dict with keys: intent, work_id, work_title, strategy.
    """
    result = classify_query_v2(query, clean_text)
    intent = result.primary
    work_id = None
    work_title = None
    strategy = "standard"

    if catalog is not None:
        try:
            work = catalog.match_query(query, normalize_fn=normalize_for_match)
        except Exception:
            work = None

        if work:
            work_id = work.get("work_id")
            work_title = work.get("title")
            if intent == "quote_lookup":
                strategy = "exact_quote_with_work_constraint"
            elif intent == "bibliographic_lookup":
                strategy = "work_catalog_direct"
            elif intent == "concept_explain":
                strategy = "concept_constrained_retrieval"
            elif intent in ("theory_analysis", "deep_analysis"):
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
