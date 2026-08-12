from __future__ import annotations

import re
from dataclasses import dataclass, field


EXACT_INTENTS = {"quote_lookup", "bibliographic_lookup"}
ANALYTIC_INTENTS = {"rag_answer", "deep_analysis", "theory_analysis", "concept_explain", "comparison"}
COMPARISON_INTENTS = {"comparison"}


@dataclass
class QueryPlan:
    original_query: str
    standalone_query: str
    retrieval_queries: list[str] = field(default_factory=list)
    decomposition_queries: list[str] = field(default_factory=list)
    hyde_query: str = ""
    intent: str = "rag_answer"
    mode: str = "exact"
    inherits_context: bool = False
    disabled_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "standalone_query": self.standalone_query,
            "retrieval_queries": self.retrieval_queries,
            "decomposition_queries": self.decomposition_queries,
            "hyde_query": self.hyde_query,
            "intent": self.intent,
            "mode": self.mode,
            "inherits_context": self.inherits_context,
            "disabled_reason": self.disabled_reason,
        }


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _dedupe(items: list[str], limit: int = 8) -> list[str]:
    seen = set()
    out = []
    for item in items:
        item = _norm(item)
        key = re.sub(r"[\s，。！？；：、“”‘’《》()（）\[\]]+", "", item).lower()
        if not item or not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _history_text(history: list[dict] | None, limit: int = 6) -> str:
    if not history:
        return ""
    turns = []
    for item in history[-limit:]:
        role = item.get("role") or ""
        text = item.get("text") or item.get("content") or ""
        if text:
            turns.append(f"{role}: {_norm(text)[:240]}")
    return "\n".join(turns)


def is_followup_query(query: str) -> bool:
    q = _norm(query)
    markers = [
        "这个呢", "这句呢", "这段呢", "那这个", "那这句", "那这段", "这个也是吗",
        "它呢", "上一句", "下一句", "继续", "进一步", "再分析", "再说说",
    ]
    return any(marker in q for marker in markers) or len(q) <= 12 and any(x in q for x in ["呢", "这个", "这句"])


def build_standalone_query(query: str, history: list[dict] | None, intent: str) -> tuple[str, bool]:
    query = _norm(query)
    if not history or not is_followup_query(query):
        return query, False

    previous_user = ""
    previous_bot = ""
    for item in reversed(history):
        text = _norm(item.get("text") or item.get("content") or "")
        if not text:
            continue
        role = item.get("role") or ""
        if not previous_bot and role in {"bot", "assistant"}:
            previous_bot = text
        elif not previous_user and role == "user":
            previous_user = text
        if previous_user and previous_bot:
            break

    context = previous_user or previous_bot
    if not context:
        return query, False

    if intent == "quote_lookup":
        return f"延续上一轮精确引文定位任务。上一轮问题：{context}。本轮问题：{query}", True
    return f"延续上一轮问题的主题和任务。上一轮问题：{context}。本轮问题：{query}", True


def _comparison_entities(query: str) -> list[str]:
    """Extract entities being compared from a comparison-style query.

    Detects patterns like "比较 A 和 B", "A 与 B 的区别", "A vs B".
    """
    q = _norm(query)
    entities: list[str] = []

    # Pattern 1: 《A》...《B》 (book titles)
    quoted = re.findall(r"《([^》]{2,40})》", q)
    if len(quoted) >= 2:
        entities.extend(quoted[:4])
        return entities

    # Pattern 2: Split on comparison delimiters
    comparison_delimiters = ["和", "与", "跟", "同", "以及", "对比", "比较", "vs", "VS"]
    for delim in comparison_delimiters:
        if delim in q:
            parts = q.split(delim, 1)
            if len(parts) == 2:
                left = _norm(parts[0])
                right = _norm(parts[1])
                # Remove leading comparison prefix from left
                left = re.sub(r"^(比较|对比|请分析|分析一下|谈谈|说说)\s*", "", left)
                # Remove trailing comparison suffix from right
                right = re.sub(r"\s*(的区别|的异同|的差异|的不同|的关系|的比较|的对比|方面).*$", "", right)
                if len(left) >= 2:
                    entities.append(left)
                if len(right) >= 2:
                    entities.append(right)
            break

    return entities[:4]


def decompose_query(query: str, intent: str = "") -> list[str]:
    q = _norm(query)
    pieces = []

    # ── Comparison-specific decomposition (NEW) ──────────────────────
    if intent in COMPARISON_INTENTS:
        entities = _comparison_entities(q)
        for entity in entities:
            pieces.append(f"{entity} 的核心论述")
            pieces.append(f"马克思关于{entity}的观点")
            pieces.append(f"恩格斯关于{entity}的观点")
        if len(entities) >= 2:
            pieces.append(f"{entities[0]} 和 {entities[1]} 的关系")
        return _dedupe(pieces, limit=8)

    quoted_titles = re.findall(r"《([^》]{2,80})》", q)
    for title in quoted_titles[:4]:
        pieces.append(f"《{title}》的核心论述")

    concept_markers = [
        "异化", "阶级", "国家", "资本", "剩余价值", "商品拜物教", "历史唯物主义",
        "生产关系", "生产力", "意识形态", "无产阶级专政", "革命", "劳动", "私有制",
    ]
    for marker in concept_markers:
        if marker in q:
            pieces.append(f"马克思恩格斯关于{marker}的原文论述")

    split_parts = re.split(r"[，,；;、]|以及|和|与|并且|同时|之间", q)
    for part in split_parts:
        part = _norm(part)
        if 6 <= len(part) <= 36 and any(ch in part for ch in concept_markers):
            pieces.append(part)

    if any(marker in q for marker in ["论文", "撰写", "综述", "系统分析", "综合分析"]):
        pieces.extend([
            f"{q} 的理论来源",
            f"{q} 的历史语境",
            f"{q} 的当代意义",
        ])

    return _dedupe(pieces, limit=5)


def hyde_query(query: str, decomposition: list[str]) -> str:
    q = _norm(query)
    if not q:
        return ""
    focus = "；".join(decomposition[:3]) if decomposition else q
    return (
        "用于检索的假设性摘要：马克思恩格斯在相关原文中围绕"
        f"{focus} 展开论述，涉及概念规定、历史条件、社会关系和理论意义。"
    )


def plan_query(query: str, intent: str, history: list[dict] | None = None, enable_hyde: bool = True) -> QueryPlan:
    query = _norm(query)
    standalone, inherits_context = build_standalone_query(query, history, intent)
    if intent in EXACT_INTENTS:
        return QueryPlan(
            original_query=query,
            standalone_query=standalone,
            retrieval_queries=_dedupe([standalone, query], limit=2),
            intent=intent,
            mode="context_only" if inherits_context else "exact",
            inherits_context=inherits_context,
            disabled_reason="exact_lookup",
        )

    decomposition = decompose_query(standalone, intent=intent) if intent in ANALYTIC_INTENTS else []
    hq = hyde_query(standalone, decomposition) if enable_hyde and intent in {"rag_answer", "deep_analysis", "theory_analysis"} else ""
    retrieval_queries = _dedupe([standalone, query] + decomposition + ([hq] if hq else []), limit=8)
    return QueryPlan(
        original_query=query,
        standalone_query=standalone,
        retrieval_queries=retrieval_queries,
        decomposition_queries=decomposition,
        hyde_query=hq,
        intent=intent,
        mode="expanded" if (decomposition or hq or inherits_context) else "single",
        inherits_context=inherits_context,
    )
