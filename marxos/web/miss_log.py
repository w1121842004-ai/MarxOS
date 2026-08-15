"""查询失败案例采集：自动检测 + 用户反馈，落盘 JSONL。

失败种类：
  refusal               空检索确定性拒答（无证据）
  bibliographic_miss    书目定位未确认（本地路径未收录）
  quote_unconfirmed     引文题只有未确认向量候选（无精确命中）
  low_crag              CRAG 分数低于阈值且纠正后仍不达标
  citation_issues       引文审计发现问题
  user_feedback         用户点「回答不准确」
  error                 服务异常（500）

输出 logs/query_misses.jsonl（schema v1），用 scripts/report_query_misses.py 汇总。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MISS_LOG_SCHEMA = "query-miss/v1"


def miss_log_path() -> Path:
    return Path(os.getenv("MARXOS_MISS_LOG", "logs/query_misses.jsonl"))


def log_query_miss(
    kind: str,
    query: str,
    *,
    intent: str = "",
    mode: str = "",
    path: str = "",
    crag_score: int | None = None,
    evidence_count: int = 0,
    audit_issues: int = 0,
    elapsed_ms: int = 0,
    history_turns: int = 0,
    detail: str = "",
) -> None:
    """Append one miss record; never raises (logging must not break the ask flow)."""
    try:
        record = {
            "schema_version": MISS_LOG_SCHEMA,
            "ts": int(time.time()),
            "kind": kind,
            "query": str(query or "")[:500],
            "intent": intent,
            "mode": mode,
            "path": path,
            "crag_score": crag_score,
            "evidence_count": evidence_count,
            "audit_issues": audit_issues,
            "elapsed_ms": elapsed_ms,
            "history_turns": history_turns,
            "detail": str(detail or "")[:300],
        }
        target = miss_log_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        print("miss_log write failed", file=sys.stderr, flush=True)


def detect_misses(
    query: str,
    intent: str,
    mode: str,
    path: str,
    answer: str,
    evidence: list,
    citation_audit: dict,
    crag_report: dict,
    elapsed_ms: int,
    history_turns: int,
) -> None:
    """从一次 ask 的产物中自动判定失败种类并落盘。"""
    evidence = evidence or []
    if path == "refusal":
        log_query_miss("refusal", query, intent=intent, mode=mode, path=path,
                       evidence_count=len(evidence), elapsed_ms=elapsed_ms,
                       history_turns=history_turns, detail=(answer or "")[:200])
        return
    if "未能在当前核心书目表中确认" in (answer or ""):
        log_query_miss("bibliographic_miss", query, intent=intent, mode=mode, path=path,
                       elapsed_ms=elapsed_ms, history_turns=history_turns)
        return
    match_types = {str(item.get("match_type") or "") for item in evidence}
    if intent == "quote_lookup" and match_types and "exact_quote" not in match_types:
        log_query_miss("quote_unconfirmed", query, intent=intent, mode=mode, path=path,
                       evidence_count=len(evidence), elapsed_ms=elapsed_ms,
                       history_turns=history_turns,
                       detail=",".join(sorted(match_types))[:200])
        return
    crag = crag_report or {}
    if crag.get("ok") is False or (crag.get("score") or 0) < (crag.get("threshold") or 45):
        log_query_miss("low_crag", query, intent=intent, mode=mode, path=path,
                       crag_score=crag.get("score"), evidence_count=len(evidence),
                       elapsed_ms=elapsed_ms, history_turns=history_turns)
        return
    audit = citation_audit or {}
    if audit.get("issues"):
        log_query_miss("citation_issues", query, intent=intent, mode=mode, path=path,
                       crag_score=crag.get("score"), evidence_count=len(evidence),
                       audit_issues=len(audit["issues"]), elapsed_ms=elapsed_ms,
                       history_turns=history_turns,
                       detail=",".join(str(item.get("type")) for item in audit["issues"])[:200])
