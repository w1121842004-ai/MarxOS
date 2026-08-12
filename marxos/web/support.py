from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def append_metrics_log(metrics, metrics_log_path: Path) -> None:
    try:
        metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics, ensure_ascii=False) + "\n")
    except OSError as exc:
        try:
            print(f"metrics_log_write_failed: {exc}", file=sys.stderr)
        except BrokenPipeError:
            pass


def build_ask_metrics(query, intent, history, answer, evidence, citation_audit, elapsed_ms, topic_info, crag_report, max_history_turns, extract_answer_citation_lines):
    citation_audit = citation_audit or {}
    topic_info = topic_info or {}
    crag_report = crag_report or {}
    issues = citation_audit.get("issues") or []
    return {
        "event": "api_ask",
        "ts": int(time.time()),
        "query_len": len((query or "").strip()),
        "intent": intent or "-",
        "topic_id": topic_info.get("topic_id") or "",
        "topic_label": topic_info.get("topic_label") or "",
        "topic_section": topic_info.get("topic_section") or "",
        "memory_turns": min(len(history or []), max_history_turns),
        "answer_len": len(answer or ""),
        "elapsed_ms": int(elapsed_ms or 0),
        "evidence_count": len(evidence or []),
        "citation_lines_count": len(extract_answer_citation_lines(answer or "")),
        "audit_issue_count": len(issues),
        "matched_count": len([item for item in evidence or [] if item.get("answer_citation")]),
        "fallback_used": any((item.get("answer_citation") in (None, "")) for item in (evidence or [])) and bool(evidence),
        "audit_ok": bool(citation_audit.get("ok", True)),
        "crag_path": crag_report.get("path") or "",
        "crag_score": int(crag_report.get("score") or 0),
        "crag_threshold": int(crag_report.get("threshold") or 0),
        "crag_ok": bool(crag_report.get("ok", False)),
        "crag_issue_count": len(crag_report.get("issues") or []),
        "crag_recovery_used": bool(citation_audit.get("crag_recovery_used", False)),
    }


def trim_text(text, limit):
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def build_history_summary(history, trim_text_fn, summary_max_chars):
    lines = []
    for item in history:
        role = item.get("role")
        text = trim_text_fn(item.get("text"), 180)
        if not text:
            continue
        if role == "user":
            lines.append(f"用户：{text}")
        elif role == "bot":
            lines.append(f"助手：{text}")
    if not lines:
        return ""
    summary = "\n".join(lines)
    return trim_text_fn(summary, summary_max_chars)


def build_contextual_query(query, history, max_history_turns, max_history_chars, build_history_summary_fn, trim_text_fn):
    if not history:
        return query

    recent = history[-max_history_turns * 2 :]
    older = history[: max(0, len(history) - len(recent))]

    lines = []
    older_summary = build_history_summary_fn(older)
    if older_summary:
        lines.append(f"较早对话摘要：\n{older_summary}")

    for item in recent:
        role = item.get("role")
        text = trim_text_fn(item.get("text"), 300)
        if not text:
            continue
        if role == "user":
            lines.append(f"用户：{text}")
        elif role == "bot":
            lines.append(f"助手：{text}")

    if not lines:
        return query

    contextual = (
        "以下是对话上下文，请结合上下文回答当前问题：\n"
        + "\n".join(lines)
        + f"\n\n当前问题：\n{query}\n"
        + "请优先回答当前问题，并在必要时参考上文。"
    )
    return trim_text_fn(contextual, max_history_chars)


def is_contextual_followup(query):
    markers = [
        "这个",
        "这句",
        "那句",
        "这段",
        "第一句",
        "第1句",
        "上面",
        "刚才",
        "上一条",
        "继续",
        "接着",
        "展开",
        "详细",
        "详细说明",
        "具体说",
        "具体说明",
        "再说",
        "摘下来",
        "完整段落",
    ]
    query = query or ""
    return any(marker in query for marker in markers)


def last_bot_message(history):
    for item in reversed(history or []):
        if item.get("role") == "bot" and (item.get("text") or "").strip():
            return item.get("text") or ""
    return ""


def last_bot_item(history):
    for item in reversed(history or []):
        if item.get("role") == "bot" and (item.get("text") or "").strip():
            return item
    return {}


def last_bot_topic(history):
    item = last_bot_item(history)
    topic = item.get("topic") or {}
    if not isinstance(topic, dict):
        return {}
    return topic


def topic_scoped_query(query, history, is_contextual_followup_fn):
    topic = last_bot_topic(history)
    topic_label = (topic.get("topic_label") or "").strip()
    if not topic_label:
        return query
    if topic_label in (query or ""):
        return query
    if not is_contextual_followup_fn(query):
        return query
    return f"{topic_label}：{query}"


def build_ask_response(intent, answer, evidence, citation_audit, topic_info, crag_report, elapsed_ms, history, max_history_turns, mode="", timing=None):
    return {
        "intent": intent,
        "mode": mode,
        "answer": answer,
        "evidence": evidence,
        "citation_audit": citation_audit,
        "topic": topic_info,
        "crag": crag_report or {},
        "timing": timing or {},
        "elapsed_ms": elapsed_ms,
        "memory_turns": min(len(history), max_history_turns),
    }
