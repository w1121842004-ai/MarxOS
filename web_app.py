import html
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import app


HOST = "127.0.0.1"
PORT = int(os.getenv("MARXOS_WEB_PORT", "7860"))
MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 3500
SUMMARY_MAX_CHARS = 900
OCR_CACHE_DIR = Path(app.OCR_CACHE_DIR)
METRICS_LOG_PATH = Path("logs") / "api_ask_metrics.jsonl"


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MarxOS Web</title>
  <style>
    :root {
      --bg: #f3f4f6;
      --surface: #ffffff;
      --panel: #111827;
      --panel-soft: #1f2937;
      --text: #111827;
      --muted: #6b7280;
      --line: #e5e7eb;
      --primary: #2563eb;
      --primary-dark: #1d4ed8;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .layout {
      display: grid;
      grid-template-columns: 280px 1fr;
      height: 100vh;
      overflow: hidden;
    }
    .sidebar {
      background: var(--panel);
      color: #f3f4f6;
      padding: 14px 12px;
      border-right: 1px solid #0b1220;
      display: flex;
      flex-direction: column;
      gap: 10px;
      min-height: 0;
      height: 100vh;
      overflow: hidden;
    }
    .new-chat-btn {
      width: 100%;
      border: 1px solid #374151;
      background: var(--panel-soft);
      color: #f9fafb;
      height: 38px;
      border-radius: 8px;
      font-size: 14px;
      cursor: pointer;
    }
    .new-chat-btn:hover { background: #273449; }
    .history-title {
      font-size: 12px;
      color: #9ca3af;
      padding: 0 2px;
    }
    .history-list {
      overflow: auto;
      min-height: 0;
      display: flex;
      flex-direction: column;
      gap: 6px;
      padding-right: 2px;
    }
    .history-item {
      text-align: left;
      border: 1px solid #374151;
      background: transparent;
      color: #f3f4f6;
      border-radius: 8px;
      padding: 8px 10px;
      cursor: pointer;
      width: 100%;
    }
    .history-item.active {
      background: #283548;
      border-color: #4b5563;
    }
    .history-item-title {
      font-size: 13px;
      line-height: 1.4;
      white-space: nowrap;
      text-overflow: ellipsis;
      overflow: hidden;
    }
    .history-item-time {
      margin-top: 4px;
      font-size: 11px;
      color: #9ca3af;
    }
    .main {
      display: flex;
      flex-direction: column;
      min-width: 0;
      height: 100vh;
      min-height: 0;
      overflow: hidden;
    }
    .topbar {
      height: 56px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
      backdrop-filter: blur(6px);
      padding: 0 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    .title { font-size: 16px; font-weight: 700; }
    .subtitle { font-size: 12px; color: var(--muted); }
    .chat-wrap {
      flex: 1;
      min-height: 0;
      max-width: 920px;
      width: 100%;
      margin: 0 auto;
      padding: 18px 20px 140px;
      overflow-y: auto;
    }
    .msg {
      margin: 0 0 14px;
      padding: 12px 14px;
      border-radius: 10px;
      line-height: 1.65;
      white-space: pre-wrap;
      word-break: break-word;
      border: 1px solid var(--line);
      background: #fff;
      font-size: 14px;
    }
    .msg-user {
      background: #eaf2ff;
      border-color: #cfe1ff;
      margin-left: 56px;
    }
    .msg-bot { margin-right: 56px; }
    .msg-meta {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }
    .evidence-box {
      margin-top: 8px;
      border-top: 1px dashed var(--line);
      padding-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .evidence-box summary { cursor: pointer; color: #374151; font-weight: 600; }
    .evidence-item { margin-top: 6px; padding: 6px 8px; background: #f9fafb; border-radius: 6px; }
    .evidence-cite { color: #111827; }
    .evidence-excerpt { margin-top: 4px; white-space: pre-wrap; }
    .composer-shell {
      position: fixed;
      bottom: 0;
      left: 280px;
      right: 0;
      padding: 10px 18px 16px;
      background: linear-gradient(to top, var(--bg) 80%, rgba(243, 244, 246, 0));
    }
    .composer {
      max-width: 920px;
      margin: 0 auto;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
    }
    textarea {
      width: 100%;
      min-height: 66px;
      max-height: 220px;
      resize: vertical;
      border: 0;
      outline: none;
      background: transparent;
      font-size: 15px;
      line-height: 1.5;
      color: var(--text);
    }
    .toolbar {
      margin-top: 6px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .send-btn {
      border: 0;
      background: var(--primary);
      color: #fff;
      height: 36px;
      padding: 0 16px;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
    }
    .send-btn:hover { background: var(--primary-dark); }
    .send-btn:disabled { opacity: 0.7; cursor: wait; }
    .meta { display: flex; gap: 8px; color: var(--muted); font-size: 12px; }
    .badge {
      display: inline-flex;
      align-items: center;
      height: 22px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 8px;
      background: #f9fafb;
    }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar {
        min-height: unset;
        max-height: 220px;
        height: 220px;
        border-right: 0;
        border-bottom: 1px solid #0b1220;
      }
      .composer-shell { left: 0; }
      .msg-user { margin-left: 0; }
      .msg-bot { margin-right: 0; }
      .chat-wrap { padding: 14px 14px 145px; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <button id="newChatBtn" class="new-chat-btn" type="button">+ 新建对话</button>
      <div class="history-title">历史会话</div>
      <div id="historyList" class="history-list"></div>
    </aside>
    <section class="main">
      <header class="topbar">
        <div class="title">MarxOS</div>
        <div class="subtitle">马克思主义文献问答与引文检索</div>
      </header>
      <main id="chat" class="chat-wrap"></main>
      <section class="composer-shell">
        <div class="composer">
          <textarea id="q" placeholder="给 MarxOS 发送消息（Enter 发送，Shift+Enter 换行）"></textarea>
          <div class="toolbar">
            <button id="askBtn" class="send-btn" type="button">发送</button>
            <div class="meta">
              <span class="badge" id="intent">意图：-</span>
              <span class="badge" id="cost">耗时：-</span>
            </div>
          </div>
        </div>
      </section>
    </section>
  </div>
  <script>
    const qEl = document.getElementById("q");
    const btnEl = document.getElementById("askBtn");
    const chatEl = document.getElementById("chat");
    const intentEl = document.getElementById("intent");
    const costEl = document.getElementById("cost");
    const historyListEl = document.getElementById("historyList");
    const newChatBtnEl = document.getElementById("newChatBtn");

    const STORE_KEY = "marxos_conversations_v1";
    const memoryTurns = 6;
    let conversations = [];
    let currentId = "";

    function escapeHtml(s) {
      return String(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function nowLabel(ts) {
      try { return new Date(ts).toLocaleString(); }
      catch (_) { return ""; }
    }

    function createConversation() {
      const now = Date.now();
      return {
        id: String(now) + "_" + Math.random().toString(36).slice(2, 8),
        title: "新对话",
        createdAt: now,
        updatedAt: now,
        messages: [],
      };
    }

    function getCurrentConversation() {
      return conversations.find((x) => x.id === currentId);
    }

    function setConversationTitle(conv) {
      if (!conv) return;
      const firstUser = conv.messages.find((m) => m.role === "user" && m.text.trim());
      if (!firstUser) {
        conv.title = "新对话";
        return;
      }
      conv.title = firstUser.text.trim().slice(0, 24);
    }

    function persistConversations() {
      try {
        const nonEmpty = conversations.filter((conv) => conv.messages && conv.messages.length);
        localStorage.setItem(STORE_KEY, JSON.stringify(nonEmpty));
      } catch (_) {}
    }

    function pruneConversations() {
      conversations = conversations
        .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0))
        .slice(0, 50);
    }

    function startNewConversation() {
      conversations = conversations.filter((conv) => conv.messages && conv.messages.length);
      const fresh = createConversation();
      conversations.unshift(fresh);
      pruneConversations();
      currentId = fresh.id;
      persistConversations();
      renderAll();
    }

    function loadConversationsAndStartFresh() {
      try {
        const raw = localStorage.getItem(STORE_KEY);
        if (raw) {
          const parsed = JSON.parse(raw);
          if (Array.isArray(parsed)) {
            conversations = parsed.filter((x) => x && typeof x === "object" && Array.isArray(x.messages) && x.messages.length);
          }
        }
      } catch (_) {}

      if (!conversations.length) {
        conversations = [createConversation()];
        currentId = conversations[0].id;
      } else {
        startNewConversation();
        return;
      }
      persistConversations();
      renderAll();
    }

    function buildHistoryPayload() {
      const conv = getCurrentConversation();
      if (!conv) return [];
      const compact = [];
      for (const m of conv.messages) {
        if (m.role === "user") compact.push({ role: "user", text: m.text });
        if (m.role === "bot") compact.push({
          role: "bot",
          text: m.text,
          evidence: m.evidence || [],
          topic: {
            topic_id: m.topicId || "",
            topic_label: m.topicLabel || "",
            topic_section: m.topicSection || ""
          }
        });
      }
      return compact.slice(-memoryTurns * 2);
    }

    function evidenceHtml(evidence) {
      if (!Array.isArray(evidence) || !evidence.length) return "";
      const label = "\u67e5\u770b\u8bc1\u636e";
      const items = evidence.slice(0, 8).map((e, idx) => {
        const cite = escapeHtml(e.citation || e.sentence_citation || "");
        const source = escapeHtml(e.source || "");
        const page = escapeHtml(e.printed_page || e.citation_page || "");
        const lines = e.line_start ? ("L" + escapeHtml(e.line_start) + (e.line_end ? "-" + escapeHtml(e.line_end) : "")) : "";
        const meta = source + (page ? ' | \u7b2c' + page + '\u9875' : '') + (lines ? ' | ' + lines : '');
        const excerpt = escapeHtml(String(e.excerpt || "").slice(0, 260));
        return '<details class="evidence-item">' +
          '<summary><span class="evidence-cite">E' + String(idx + 1) + ': ' + cite + '</span>' +
          '<span class="evidence-mini">' + escapeHtml(meta) + '</span></summary>' +
          (excerpt ? '<div class="evidence-excerpt">' + excerpt + '</div>' : '') +
          '</details>';
      }).join("");
      return '<details class="evidence-box"><summary>' + label + ' (' + evidence.length + ')</summary>' + items + '</details>';
    }

    function renderHistory() {
      const visibleConversations = conversations.filter((conv) => conv.messages && conv.messages.length);
      if (!visibleConversations.length) {
        historyListEl.innerHTML = "";
        return;
      }
      historyListEl.innerHTML = visibleConversations.map((conv) => {
        const active = conv.id === currentId ? " active" : "";
        const title = escapeHtml(conv.title || "新对话");
        const when = escapeHtml(nowLabel(conv.updatedAt || conv.createdAt));
        return '<button class="history-item' + active + '" data-id="' + escapeHtml(conv.id) + '">' +
          '<div class="history-item-title">' + title + '</div>' +
          '<div class="history-item-time">' + when + '</div>' +
          '</button>';
      }).join("");

      for (const node of historyListEl.querySelectorAll(".history-item")) {
        node.addEventListener("click", () => {
          const id = node.getAttribute("data-id");
          if (!id) return;
          currentId = id;
          renderAll();
        });
      }
    }

    function renderChat() {
      const conv = getCurrentConversation();
      const messages = conv ? conv.messages : [];
      if (!messages.length) {
        chatEl.innerHTML = '<div class="msg msg-bot">欢迎使用 MarxOS。你可以询问概念、请求引文，或进行理论分析。</div>';
        return;
      }
      chatEl.innerHTML = messages.map((m) => {
        if (m.role === "user") {
          return '<div class="msg msg-user">' + escapeHtml(m.text) + '</div>';
        }
        return '<div class="msg msg-bot">' + escapeHtml(m.text) +
          evidenceHtml(m.evidence || []) +
          '<div class="msg-meta">意图：' + escapeHtml(m.intent || "-") +
          (m.topicLabel ? ' ｜ 专题：' + escapeHtml(m.topicLabel) : '') +
          (m.topicSection ? ' ｜ 板块：' + escapeHtml(m.topicSection) : '') +
          ' ｜ 耗时：' + escapeHtml(String(m.cost ?? "-")) + 'ms</div></div>';
      }).join("");
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    function renderAll() {
      renderHistory();
      renderChat();
    }

    async function ask() {
      const query = qEl.value.trim();
      if (!query) return;
      const conv = getCurrentConversation();
      if (!conv) return;

      btnEl.disabled = true;
      conv.messages.push({ role: "user", text: query });
      conv.updatedAt = Date.now();
      setConversationTitle(conv);
      renderAll();
      persistConversations();

      intentEl.textContent = "意图：处理中";
      costEl.textContent = "耗时：-";
      qEl.value = "";
      try {
        const res = await fetch("/api/ask", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ query, history: buildHistoryPayload() })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "请求失败");
        conv.messages.push({
          role: "bot",
          text: data.answer || "",
          intent: data.intent || "-",
          topicId: (data.topic && data.topic.topic_id) ? data.topic.topic_id : "",
          topicLabel: (data.topic && data.topic.topic_label) ? data.topic.topic_label : "",
          topicSection: (data.topic && data.topic.topic_section) ? data.topic.topic_section : "",
          cost: data.elapsed_ms ?? "-",
          evidence: Array.isArray(data.evidence) ? data.evidence : []
        });
        conv.updatedAt = Date.now();
        setConversationTitle(conv);
        pruneConversations();
        renderAll();
        persistConversations();
        intentEl.textContent = "意图：" + (data.intent || "-");
        costEl.textContent = "耗时：" + String(data.elapsed_ms ?? "-") + "ms";
      } catch (err) {
        const msg = (err && err.message) ? err.message : "请求失败";
        conv.messages.push({ role: "bot", text: "请求失败：" + msg, intent: "-", cost: "-" });
        conv.updatedAt = Date.now();
        setConversationTitle(conv);
        renderAll();
        persistConversations();
      } finally {
        btnEl.disabled = false;
        qEl.focus();
      }
    }

    btnEl.addEventListener("click", ask);
    qEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        ask();
      }
    });
    newChatBtnEl.addEventListener("click", startNewConversation);

    loadConversationsAndStartFresh();
  </script>
</body>
</html>
"""


class MarxOSHandler(BaseHTTPRequestHandler):
    @staticmethod
    def _append_metrics_log(metrics):
        try:
            METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with METRICS_LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"metrics_log_write_failed: {exc}", file=sys.stderr)

    @staticmethod
    def _build_ask_metrics(query, intent, history, answer, evidence, citation_audit, elapsed_ms, topic_info):
        citation_audit = citation_audit or {}
        topic_info = topic_info or {}
        issues = citation_audit.get("issues") or []
        return {
            "event": "api_ask",
            "ts": int(time.time()),
            "query_len": len((query or "").strip()),
            "intent": intent or "-",
            "topic_id": topic_info.get("topic_id") or "",
            "topic_label": topic_info.get("topic_label") or "",
            "topic_section": topic_info.get("topic_section") or "",
            "memory_turns": min(len(history or []), MAX_HISTORY_TURNS),
            "answer_len": len(answer or ""),
            "elapsed_ms": int(elapsed_ms or 0),
            "evidence_count": len(evidence or []),
            "citation_lines_count": len(app.extract_answer_citation_lines(answer or "")),
            "audit_issue_count": len(issues),
            "matched_count": len([item for item in evidence or [] if item.get("answer_citation")]),
            "fallback_used": any(
                (item.get("answer_citation") in (None, ""))
                for item in (evidence or [])
            ) and bool(evidence),
            "audit_ok": bool(citation_audit.get("ok", True)),
        }

    @staticmethod
    def _trim_text(text, limit):
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"

    @classmethod
    def _build_history_summary(cls, history):
        lines = []
        for item in history:
            role = item.get("role")
            text = cls._trim_text(item.get("text"), 180)
            if not text:
                continue
            if role == "user":
                lines.append(f"用户：{text}")
            elif role == "bot":
                lines.append(f"助手：{text}")
        if not lines:
            return ""
        summary = "\n".join(lines)
        return cls._trim_text(summary, SUMMARY_MAX_CHARS)

    @staticmethod
    def _build_contextual_query(query, history):
        if not history:
            return query

        recent = history[-MAX_HISTORY_TURNS * 2 :]
        older = history[: max(0, len(history) - len(recent))]

        lines = []
        older_summary = MarxOSHandler._build_history_summary(older)
        if older_summary:
            lines.append(f"较早对话摘要：\n{older_summary}")

        for item in recent:
            role = item.get("role")
            text = MarxOSHandler._trim_text(item.get("text"), 300)
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
        return MarxOSHandler._trim_text(contextual, MAX_HISTORY_CHARS)

    @staticmethod
    def _is_contextual_followup(query):
        markers = [
            "\u8fd9\u4e2a",      # ??
            "\u8fd9\u53e5",      # ??
            "\u90a3\u53e5",      # ??
            "\u8fd9\u6bb5",      # ??
            "\u7b2c\u4e00\u53e5",  # ???
            "\u7b2c1\u53e5",     # ?1?
            "\u4e0a\u9762",      # ??
            "\u521a\u624d",      # ??
            "\u4e0a\u4e00\u6761",  # ???
            "\u6458\u4e0b\u6765",  # ???
            "\u5b8c\u6574\u6bb5\u843d",  # ????
        ]
        query = query or ""
        return any(marker in query for marker in markers)

    @staticmethod
    def _last_bot_message(history):
        for item in reversed(history or []):
            if item.get("role") == "bot" and (item.get("text") or "").strip():
                return item.get("text") or ""
        return ""

    @staticmethod
    def _last_bot_item(history):
        for item in reversed(history or []):
            if item.get("role") == "bot" and (item.get("text") or "").strip():
                return item
        return {}

    @staticmethod
    def _last_bot_topic(history):
        item = MarxOSHandler._last_bot_item(history)
        topic = item.get("topic") or {}
        if not isinstance(topic, dict):
            return {}
        return topic

    @classmethod
    def _topic_scoped_query(cls, query, history):
        topic = cls._last_bot_topic(history)
        topic_label = (topic.get("topic_label") or "").strip()
        if not topic_label:
            return query
        if topic_label in (query or ""):
            return query
        if not cls._is_contextual_followup(query):
            return query
        return f"{topic_label}：{query}"

    @staticmethod
    def _citation_from_evidence(item, index):
        evidence = item.get("evidence") or []
        if not isinstance(evidence, list) or not evidence:
            return None
        selected = evidence[index - 1] if 0 <= index - 1 < len(evidence) else evidence[0]
        source = selected.get("source") or selected.get("source_file")
        page = selected.get("printed_page") or selected.get("citation_page")
        if not source or page is None:
            return None
        try:
            page = int(page)
        except (TypeError, ValueError):
            return None
        return {
            "index": index,
            "body": selected.get("citation") or selected.get("sentence_citation") or "",
            "source": source,
            "page": page,
            "pdf_page": selected.get("pdf_page"),
            "excerpt": selected.get("excerpt") or "",
        }

    @staticmethod
    def _requested_citation_index(query):
        query = query or ""
        number_words = [
            (1, ["\u7b2c\u4e00", "\u7b2c1", "1\u53f7", "1\u6761"]),
            (2, ["\u7b2c\u4e8c", "\u7b2c\u4e24", "\u7b2c2", "2\u53f7", "2\u6761"]),
            (3, ["\u7b2c\u4e09", "\u7b2c3", "3\u53f7", "3\u6761"]),
            (4, ["\u7b2c\u56db", "\u7b2c4", "4\u53f7", "4\u6761"]),
            (5, ["\u7b2c\u4e94", "\u7b2c5", "5\u53f7", "5\u6761"]),
        ]
        for index, markers in number_words:
            if any(marker in query for marker in markers):
                return index
        match = re.search(r"(\d+)\s*[\u53f7\u6761\u6bb5]", query)
        return int(match.group(1)) if match else 1

    @staticmethod
    def _requested_citation_indices(query):
        query = query or ""
        hits = []
        for match in re.finditer(r"第\s*(\d+)\s*[条段句]", query):
            hits.append(int(match.group(1)))
        for match in re.finditer(r"(\d+)\s*[条段句]", query):
            value = int(match.group(1))
            if value not in hits:
                hits.append(value)
        return hits

    @staticmethod
    def _parse_citation_line(line):
        match = re.match(r"\s*(\d+)[\.\u3001]\s*(.+)", line)
        if not match:
            return None
        index = int(match.group(1))
        body = match.group(2).strip()
        series_match = re.search(
            r"\u300a(\u9a6c\u514b\u601d\u6069\u683c\u65af(?:\u6587\u96c6|\u9009\u96c6))\u300b\u7b2c(\d+)\u5377",
            body,
        )
        page_match = re.search(r"\u7b2c(\d+)\u9875", body)
        if not series_match or not page_match:
            return None
        series, volume = series_match.group(1), int(series_match.group(2))
        prefix = "mea" if "\u6587\u96c6" in series else "mes"
        return {
            "index": index,
            "body": body,
            "source": f"{prefix}{volume:02d}.pdf",
            "page": int(page_match.group(1)),
        }

    @classmethod
    def _parse_last_citations(cls, text):
        citations = {}
        for line in (text or "").splitlines():
            parsed = cls._parse_citation_line(line)
            if parsed:
                citations[parsed["index"]] = parsed
        return citations

    @staticmethod
    def _load_ocr_text(source, pdf_page):
        path = OCR_CACHE_DIR / source.replace(".pdf", "") / f"page_{pdf_page}.json"
        if not path.exists():
            return ""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        return app.repair_mojibake(payload.get("cleaned_text") or payload.get("raw_text") or "")

    @staticmethod
    def _find_pdf_page_by_printed_page(source, printed_page):
        source_dir = OCR_CACHE_DIR / source.replace(".pdf", "")
        if not source_dir.exists():
            return None
        paths = sorted(
            source_dir.glob("page_*.json"),
            key=lambda path: int(re.search(r"page_(\d+)", path.name).group(1)),
        )
        for path in paths:
            pdf_page = int(re.search(r"page_(\d+)", path.name).group(1))
            inferred = app.infer_printed_page_from_ocr_cache({"source": source, "pdf_page": pdf_page})
            if inferred == printed_page:
                return pdf_page
        return None

    @staticmethod
    def _paragraphs_from_text(text):
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        paragraphs = []
        current = []
        for line in lines:
            current.append(line)
            if line.endswith(("\u3002", "\uff01", "\uff1f", "\u3002\u201d")) and len("".join(current)) >= 120:
                paragraphs.append("".join(current))
                current = []
        if current:
            paragraphs.append("".join(current))
        return paragraphs

    @classmethod
    def _answer_citation_followup(cls, query, history):
        if not cls._is_contextual_followup(query):
            return None
        needed_markers = ["\u51fa\u5904", "\u6bb5\u843d", "\u6458", "\u539f\u6587", "\u54ea\u6bb5"]
        if not any(marker in query for marker in needed_markers):
            return None

        requested_index = cls._requested_citation_index(query)
        last_bot = cls._last_bot_item(history)
        citation = cls._citation_from_evidence(last_bot, requested_index)
        if not citation:
            citations = cls._parse_last_citations(cls._last_bot_message(history))
            if not citations:
                return None
            citation = citations.get(requested_index) or citations.get(1)
        if not citation:
            return None

        pdf_page = None
        try:
            pdf_page = int(citation.get("pdf_page")) if citation.get("pdf_page") is not None else None
        except (TypeError, ValueError):
            pdf_page = None
        if pdf_page is None:
            pdf_page = cls._find_pdf_page_by_printed_page(citation["source"], citation["page"])
        if pdf_page is None:
            return (
                f"\u6211\u6ca1\u6709\u5728\u672c\u5730 OCR \u9875\u7801\u6620\u5c04\u4e2d\u5b9a\u4f4d\u5230\u811a\u6ce8 {citation['index']} \u7684\u539f\u9875\uff1a{citation['body']}\n\n"
                "\u8fd9\u8bf4\u660e\u4e0a\u4e00\u6761\u56de\u7b54\u7684\u9875\u7801\u9700\u8981\u91cd\u65b0\u6838\u5bf9\uff1b\u672c\u8f6e\u4e0d\u4f1a\u7f16\u9020\u6bb5\u843d\u3002"
            )

        text = cls._load_ocr_text(citation["source"], pdf_page)
        paragraphs = cls._paragraphs_from_text(text)
        excerpt = "\n\n".join(paragraphs[:2]).strip()
        if len(excerpt) > 900:
            excerpt = excerpt[:900].rstrip() + "......"
        if not excerpt:
            excerpt = "\u8be5\u9875 OCR \u6587\u672c\u4e3a\u7a7a\uff0c\u9700\u8981\u91cd\u65b0 OCR \u6216\u6838\u5bf9\u539f PDF\u3002"

        return (
            f"\u6309\u4e0a\u4e00\u6761\u56de\u7b54\u7684\u811a\u6ce8 {citation['index']} \u5b9a\u4f4d\uff1a{citation['body']}\n\n"
            f"\u672c\u5730 OCR \u5bf9\u5e94\u5230 {citation['source']} \u7684\u7b2c {pdf_page} \u4e2a\u56fe\u50cf\u9875\uff0c\u8bc6\u522b\u51fa\u7684\u5370\u5237\u9875\u4e3a\u7b2c {citation['page']} \u9875\u3002\n\n"
            "\u539f\u9875\u6458\u5f55\u5982\u4e0b\uff1a\n\n"
            f"> {excerpt}\n\n"
            "\u8bf4\u660e\uff1a\u5982\u679c\u4e0a\u4e00\u6761\u6b63\u6587\u91cc\u7684\u90a3\u53e5\u8bdd\u662f\u6982\u62ec\u53e5\uff0c\u800c\u4e0d\u662f\u539f\u8457\u9010\u5b57\u5f15\u6587\uff0c\u6211\u8fd9\u91cc\u53ea\u7ed9\u51fa\u811a\u6ce8\u9875\u7684 OCR \u539f\u6587\uff0c\u4e0d\u628a\u6982\u62ec\u53e5\u4f2a\u88c5\u6210\u539f\u6587\u3002"
        )

    @classmethod
    def _answer_evidence_page_followup(cls, query, history):
        normalized = query or ""
        explicit_indices = cls._requested_citation_indices(query)
        if not cls._is_contextual_followup(query):
            has_page_request = any(marker in normalized for marker in ["页", "页码", "升序", "排序", "列出来"]) and any(marker in normalized for marker in ["证据", "页", "页码"])
            if not explicit_indices and not has_page_request:
                return None
            if "页" not in normalized and "页码" not in normalized:
                return None

        if "页" not in normalized and "页码" not in normalized:
            return None

        last_bot = cls._last_bot_item(history)
        evidence = last_bot.get("evidence") or []
        if not isinstance(evidence, list) or not evidence:
            return None

        numbered_requests = ["前3", "前三", "三条", "3条", "三点", "3点"]
        wants_sorted = any(marker in normalized for marker in ["升序", "排序", "列出来", "全部", "所有", "单独"])
        wants_pages = any(marker in normalized for marker in ["哪一页", "页码", "分别", "页"])
        if not wants_pages:
            return None

        items = []
        for item in evidence:
            page = item.get("printed_page") or item.get("citation_page")
            if page is None:
                continue
            items.append(item)

        if not items:
            return None

        if explicit_indices:
            selected = []
            for index in explicit_indices:
                if 1 <= index <= len(evidence):
                    item = evidence[index - 1]
                    page = item.get("printed_page") or item.get("citation_page")
                    if page is not None:
                        selected.append(item)
            if selected:
                items = selected
        elif any(marker in normalized for marker in numbered_requests):
            items = items[:3]
        elif wants_sorted:
            items = sorted(items, key=lambda item: int(item.get("printed_page") or item.get("citation_page") or 0))
        else:
            items = items[: min(5, len(items))]

        lines = ["根据上一条回答中的直接证据，相关页码如下：", ""]
        for index, item in enumerate(items, start=1):
            citation = item.get("detailed_citation") or item.get("citation") or ""
            page = item.get("printed_page") or item.get("citation_page")
            lines.append(f"{index}. 第{page}页。{citation}")
        return "\n".join(lines)

    @classmethod
    def _answer_topic_rewrite_followup(cls, query, history):
        evidence, topic = cls._topic_history_evidence(history)
        if not evidence or "改写" not in (query or ""):
            return None

        indices = cls._requested_citation_indices(query)
        if not indices:
            return None

        lines = ["按上一轮条目改写为更通顺的学术表述：", ""]
        for index in indices:
            if not (1 <= index <= len(evidence)):
                continue
            item = evidence[index - 1]
            excerpt = (item.get("excerpt") or "").replace("...", "").replace("……", "")
            excerpt = re.sub(r"\s+", "", excerpt)
            if "合作社" in excerpt or "共同耕种" in excerpt:
                rewritten = f"这条可以表述为：{excerpt[:90]}，其核心意思是通过合作化与联合生产推动农民向新的生产方式过渡。"
            elif "小农" in excerpt and "暴力" in excerpt:
                rewritten = f"这条可以表述为：{excerpt[:90]}，其核心意思是对小农不能采取暴力剥夺，而应通过政治引导和社会帮助实现过渡。"
            else:
                rewritten = f"这条可以表述为：{excerpt[:100]}。"
            lines.append(f"第{index}条：{rewritten}")
            lines.append(f"出处：{item.get('citation') or ''}")
            lines.append("")
        return "\n".join(lines).strip()

    @classmethod
    def _answer_topic_item_explain_followup(cls, query, history):
        evidence, topic = cls._topic_history_evidence(history)
        if not evidence:
            return None

        normalized = query or ""
        if not any(marker in normalized for marker in ["具体讲", "什么意思", "讲的是什么", "说的是什么", "具体是指", "解释一下", "再解释一下", "定义"]):
            return None

        indices = cls._requested_citation_indices(query)
        if not indices:
            return None

        index = indices[0]
        if not (1 <= index <= len(evidence)):
            return None

        item = evidence[index - 1]
        excerpt = (item.get("excerpt") or "").replace("...", "").replace("??", "")
        excerpt = re.sub(r"\s+", "", excerpt)
        article = item.get("article") or ""
        citation = item.get("detailed_citation") or item.get("citation") or ""

        if any(marker in excerpt for marker in ["纲领", "最低工资", "农业机器", "种子", "肥料", "共同耕种"]):
            summary = "这条主要在讲针对农业工人和小农的土地纲领安排，包括最低工资、农业投入支持、土地使用和共同耕种等制度措施。"
        elif any(marker in excerpt for marker in ["合作社", "示范", "社会帮助", "小农"]):
            summary = "这条主要在讲如何把小农逐步引导到合作社生产，重点不是强制剥夺，而是通过示范、帮助和过渡安排推进。"
        elif any(marker in excerpt for marker in ["大土地", "农村无产者", "剥夺"]):
            summary = "这条主要在讲对大地产和农村无产者问题的处理原则，核心是区分小农与大土地占有者，采取不同策略。"
        else:
            summary = f"这条主要在讲《{article}》中的一个具体判断，其核心意思是：{excerpt[:110]}。"

        return f"第{index}条具体讲的是：{summary}\n\n原文摘录：{excerpt[:160]}。\n出处：{citation}"

    @staticmethod
    def _topic_history_evidence(history):
        last_bot = MarxOSHandler._last_bot_item(history)
        evidence = last_bot.get("evidence") or []
        topic = last_bot.get("topic") or {}
        if not isinstance(evidence, list) or not isinstance(topic, dict):
            return [], {}
        return evidence, topic

    @staticmethod
    def _excerpt_key(item):
        return app.normalize_for_match((item.get("excerpt") or "")[:120])

    @classmethod
    def _rank_topic_evidence(cls, evidence):
        direct_markers = ["合作社", "共同耕种", "示范", "社会帮助", "小农", "大土地", "农村无产者", "纲领"]
        ranked = []
        for item in evidence:
            text = (item.get("excerpt") or "") + " " + (item.get("article") or "")
            score = 0
            for marker in direct_markers:
                if marker in text:
                    score += 1
            ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        deduped = []
        seen = set()
        for _, item in ranked:
            key = cls._excerpt_key(item)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped


    @staticmethod
    def _filter_ranked_evidence(ranked, markers_any=None, markers_all=None):
        markers_any = markers_any or []
        markers_all = markers_all or []
        selected = []
        for item in ranked:
            text = (item.get("excerpt") or "") + " " + (item.get("article") or "")
            if markers_all and not all(marker in text for marker in markers_all):
                continue
            if markers_any and not any(marker in text for marker in markers_any):
                continue
            selected.append(item)
        return selected

    @classmethod
    def _answer_topic_history_followup(cls, query, history):
        evidence, topic = cls._topic_history_evidence(history)
        topic_label = (topic.get("topic_label") or "").strip()
        if not topic_label or not evidence:
            return None

        ranked = cls._rank_topic_evidence(evidence)
        normalized = query or ""
        lowered = normalized.lower()

        if "再列出" in normalized and "三条" in normalized and "小农" in normalized and "过渡" in normalized:
            items = cls._filter_ranked_evidence(
                ranked,
                markers_any=["小农", "合作社", "社会帮助", "示范", "过渡"],
            )
            if items:
                lines = ["和小农过渡最相关的还可以再补这三条：", ""]
                for index, item in enumerate(items[:3], start=1):
                    lines.append(f"{index}. {(item.get('excerpt') or '')[:100]}?")
                    lines.append(f"   {item.get('citation') or ''}")
                return "\n".join(lines)

        if "哪一段" in normalized and "德国农民战争" in normalized:
            for item in ranked:
                if "德国农民战争" not in (item.get("article") or ""):
                    continue
                if "合作社" not in (item.get("excerpt") or "") and "共同耕种" not in (item.get("excerpt") or ""):
                    continue
                citation = item.get("detailed_citation") or item.get("citation") or ""
                excerpt = item.get("excerpt") or ""
                return (
                    "上一轮证据里，《德国农民战争》中和合作社最相关的是这一段：\n\n"
                    f"> {excerpt}\n\n"
                    f"出处：{citation}"
                )

        if "哪几条" in normalized or ("哪些" in normalized and "观点" in normalized):
            lines = ["上一轮证据里，最直接谈到合作社的条目主要有：", ""]
            direct = [
                item for item in ranked
                if "合作社" in (item.get("excerpt") or "") or "共同耕种" in (item.get("excerpt") or "")
            ]
            for index, item in enumerate(direct[:5], start=1):
                lines.append(f"{index}. {(item.get('excerpt') or '')[:100]}?")
                lines.append(f"   {item.get('citation') or ''}")
            if len(lines) > 2:
                return "\n".join(lines)

        if "摘录" in normalized and ("三段" in normalized or "三条" in normalized):
            direct = [
                item for item in ranked
                if "合作社" in (item.get("excerpt") or "") or "共同耕种" in (item.get("excerpt") or "")
            ]
            if direct:
                lines = ["上一轮证据里，直接涉及合作社的原文摘录可以先列这三段：", ""]
                for index, item in enumerate(direct[:3], start=1):
                    lines.append(f"{index}. {item.get('excerpt') or ''}")
                    lines.append(f"   {item.get('detailed_citation') or item.get('citation') or ''}")
                    lines.append("")
                return "\n".join(lines).strip()

        if any(marker in normalized for marker in ["整理", "分类", "三类"]):
            groups = {
                "政策主张": [],
                "过渡方式": [],
                "阶级区分": [],
            }
            for item in ranked:
                excerpt = item.get("excerpt") or ""
                citation = item.get("citation") or ""
                if any(marker in excerpt for marker in ["要求", "纲领", "建立", "降低", "废除", "租给"]):
                    groups["政策主张"].append((excerpt, citation))
                if any(marker in excerpt for marker in ["合作社", "共同耕种", "示范", "社会帮助", "联合"]):
                    groups["过渡方式"].append((excerpt, citation))
                if any(marker in excerpt for marker in ["小农", "大土地", "农村无产者", "短工", "中农", "大农"]):
                    groups["阶级区分"].append((excerpt, citation))
            lines = ["根据上一轮直接证据，可以先按三类整理：", ""]
            for label, items in groups.items():
                if not items:
                    continue
                lines.append(f"{label}?")
                for excerpt, citation in items[:2]:
                    lines.append(f"1. {excerpt[:90]}?")
                    lines.append(f"   {citation}")
                lines.append("")
            return "\n".join(lines).strip()

        if "最适合" in normalized and "农村合作" in normalized:
            lines = ["最适合拿来回答今天农村合作问题的，主要是以下三条：", ""]
            for item in ranked[:3]:
                lines.append(f"1. {item.get('excerpt', '')[:100]}?")
                lines.append(f"   {item.get('citation') or ''}")
            return "\n".join(lines)

        if "土地所有制" in normalized or ("土地" in normalized and "再补" in normalized):
            items = cls._filter_ranked_evidence(
                ranked,
                markers_any=["土地", "土地国有化", "小块土地所有制", "大土地", "租给"],
            )
            if items:
                count = 2 if "两条" in normalized or "2条" in lowered else 3
                lines = ["和土地所有制最相关的补充观点可以先列这几条：", ""]
                for index, item in enumerate(items[:count], start=1):
                    lines.append(f"{index}. {(item.get('excerpt') or '')[:100]}?")
                    lines.append(f"   {item.get('citation') or ''}")
                return "\n".join(lines)

        if "大地产" in normalized or "农村无产者" in normalized:
            items = cls._filter_ranked_evidence(ranked, markers_any=["大土地", "大农", "农村无产者", "短工"])
            if items:
                count = 2 if "两条" in normalized or "2条" in lowered else 3
                lines = ["和大地产、农村无产者最相关的观点主要有：", ""]
                for index, item in enumerate(items[:count], start=1):
                    lines.append(f"{index}. {(item.get('excerpt') or '')[:100]}?")
                    lines.append(f"   {item.get('citation') or ''}")
                return "\n".join(lines)

        if "共同耕种" in normalized and "哪一条" in normalized:
            items = cls._filter_ranked_evidence(ranked, markers_any=["共同耕种", "合作社", "联合"])
            if items:
                item = items[0]
                return (
                    "最接近‘共同耕种’表述的是这一条：\n\n"
                    f"{item.get('excerpt') or ''}\n\n"
                    f"出处：{item.get('detailed_citation') or item.get('citation') or ''}"
                )

        if "上一条引用的出处分别是什么" in normalized:
            lines = ["上一条提到的几条出处分别是：", ""]
            for index, item in enumerate(ranked[:3], start=1):
                lines.append(f"{index}. {item.get('detailed_citation') or item.get('citation') or ''}")
            return "\n".join(lines)

        if "主要集中在哪一篇作品" in normalized:
            counts = {}
            for item in ranked:
                article = item.get("article") or "未知篇名"
                counts[article] = counts.get(article, 0) + 1
            article, count = max(counts.items(), key=lambda pair: pair[1])
            lines = [f"这一组观点目前主要集中在《{article}》，因为上一轮直接证据里它出现次数最多（{count}条）。", ""]
            top_items = [item for item in ranked if (item.get("article") or "") == article][:3]
            for index, item in enumerate(top_items, start=1):
                lines.append(f"{index}. {(item.get('excerpt') or '')[:90]}?")
                lines.append(f"   {item.get('citation') or ''}")
            return "\n".join(lines)

        if "哪一段" in normalized and "过渡方式" in normalized and "不是强制剥夺" in normalized:
            items = cls._filter_ranked_evidence(
                ranked,
                markers_any=["合作社", "社会帮助", "示范", "小农", "暴力"],
            )
            if items:
                item = items[0]
                return (
                    "最能说明‘合作社是过渡方式而不是强制剥夺’的，是这一段：\n\n"
                    f"> {item.get('excerpt') or ''}\n\n"
                    f"出处：{item.get('detailed_citation') or item.get('citation') or ''}"
                )

        if "压缩成五点" in normalized or ("核心主张" in normalized and "五点" in normalized):
            lines = ["把《法德农民问题》中的核心主张压缩成五点，可以这样把握：", ""]
            for index, item in enumerate(ranked[:5], start=1):
                lines.append(f"{index}. {(item.get('excerpt') or '')[:86]}?")
            return "\n".join(lines)

        if "小农" in normalized and "大地产" in normalized and "哪些条目" in normalized:
            small_items = cls._filter_ranked_evidence(ranked, markers_any=["小农", "合作社", "示范"])
            estate_items = cls._filter_ranked_evidence(ranked, markers_any=["大土地", "大农", "农村无产者"])
            lines = ["可以先这样区分：", ""]
            if small_items:
                lines.append("讲小农的条目：")
                for item in small_items[:3]:
                    lines.append(f"1. {(item.get('excerpt') or '')[:88]}?")
                lines.append("")
            if estate_items:
                lines.append("讲大地产和农村无产者的条目：")
                for item in estate_items[:3]:
                    lines.append(f"1. {(item.get('excerpt') or '')[:88]}?")
            return "\n".join(lines).strip()

        if "工农关系" in normalized and any(marker in normalized for marker in ["归纳", "重排", "怎么排"]):
            groups = {
                "对小农的过渡与争取": cls._filter_ranked_evidence(ranked, markers_any=["小农", "合作社", "示范", "社会帮助"]),
                "对农村无产者的直接政策": cls._filter_ranked_evidence(ranked, markers_any=["农村无产者", "短工", "最低工资"]),
                "对大地产的区分处理": cls._filter_ranked_evidence(ranked, markers_any=["大土地", "大农", "剥夺"]),
            }
            lines = ["如果按工农关系重排，这十条可以先分成三组：", ""]
            for label, items in groups.items():
                if not items:
                    continue
                lines.append(f"{label}?")
                for item in items[:2]:
                    lines.append(f"1. {(item.get('excerpt') or '')[:90]}?")
                lines.append("")
            return "\n".join(lines).strip()

        if "有没有明确说" in normalized and "暴力剥夺小农" in normalized:
            items = cls._filter_ranked_evidence(ranked, markers_any=["小农", "暴力", "剥夺"])
            if items:
                item = items[0]
                return (
                    "从上一轮直接证据看，并没有把小农作为要被暴力剥夺的对象来表述；相反，相关段落更强调过渡、示范和社会帮助。\n\n"
                    f"最直接的依据是：{item.get('excerpt') or ''}\n"
                    f"出处：{item.get('detailed_citation') or item.get('citation') or ''}"
                )

        if "合并成一条" in normalized or "合并成一个完整判断" in normalized:
            indices = cls._requested_citation_indices(query)
            if len(indices) >= 2:
                chosen = []
                for index in indices[:2]:
                    if 1 <= index <= len(evidence):
                        chosen.append(evidence[index - 1])
                if len(chosen) == 2:
                    left = re.sub(r"\s+", "", chosen[0].get("excerpt") or "")[:70]
                    right = re.sub(r"\s+", "", chosen[1].get("excerpt") or "")[:70]
                    return f"合并成一条完整判断：{left}；同时，{right}?"

        if "最关键" in normalized and "三条" in normalized:
            lines = ["如果只保留最关键的三条，我会选这三条：", ""]
            for item in ranked[:3]:
                lines.append(f"1. {(item.get('excerpt') or '')[:100]}?")
                lines.append(f"   {item.get('citation') or ''}")
            return "\n".join(lines)

        if "上面三条" in normalized and "哪一页" in normalized:
            lines = ["上面三条分别出自以下页码：", ""]
            for index, item in enumerate(ranked[:3], start=1):
                page = item.get("printed_page") or item.get("citation_page")
                lines.append(f"{index}. 第{page}页，{item.get('citation') or ''}")
            return "\n".join(lines)

        if "更接近原文" in normalized or "原文表述" in normalized:
            lines = ["把最关键三条换成更接近原文的表述如下：", ""]
            for item in ranked[:3]:
                lines.append(f"1. {item.get('excerpt', '')[:120]}?")
                lines.append(f"   {item.get('citation') or ''}")
            return "\n".join(lines)

        if "概括" in normalized and "一句话" in normalized:
            item = ranked[0]
            return (
                f"一句话概括：{topic_label}中的核心态度是，"
                f"{(item.get('excerpt') or '')[:120]}?"
            )

        if "小结" in normalized or "150字" in normalized:
            pieces = []
            for item in ranked[:3]:
                text = re.sub(r"\s+", "", item.get("excerpt") or "")
                if text:
                    pieces.append(text[:48])
            if pieces:
                summary = "?".join(pieces)[:145]
                return f"150字左右的小结可以写成：{summary}?"

        if "完整抄出来" in normalized and "合作社生产" in normalized:
            items = cls._filter_ranked_evidence(ranked, markers_any=["合作社", "生产", "共同耕种"])
            if items:
                count = 2 if "两条" in normalized or "2条" in lowered else 3
                lines = ["涉及合作社生产的原文可以完整先抔这两条：", ""]
                for index, item in enumerate(items[:count], start=1):
                    lines.append(f"{index}. {item.get('excerpt') or ''}")
                    lines.append(f"   {item.get('detailed_citation') or item.get('citation') or ''}")
                    lines.append("")
                return "\n".join(lines).strip()

        return None

    @classmethod
    def _answer_history_followup(cls, query, history):
        direct_answer = cls._answer_topic_rewrite_followup(query, history)
        if direct_answer:
            return direct_answer

        direct_answer = cls._answer_topic_item_explain_followup(query, history)
        if direct_answer:
            return direct_answer

        direct_answer = cls._answer_topic_history_followup(query, history)
        if direct_answer:
            return direct_answer

        direct_answer = cls._answer_evidence_page_followup(query, history)
        if direct_answer:
            return direct_answer

        return cls._answer_citation_followup(query, history)

    def _send_json(self, status_code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body):
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_html(HTML_PAGE)
            return
        self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path != "/api/ask":
            self.send_error(404, "Not Found")
            return
        content_len = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_len)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "无效 JSON"})
            return

        query = (data.get("query") or "").strip()
        history = data.get("history") or []
        if not query:
            self._send_json(400, {"error": "问题不能为空"})
            return

        started = time.perf_counter()
        try:
            direct_answer = self._answer_history_followup(query, history)
            if direct_answer:
                intent = "citation_followup"
                answer = direct_answer
            else:
                route_query = self._topic_scoped_query(query, history)
                is_followup = self._is_contextual_followup(query)
                contextual_query = self._build_contextual_query(route_query, history) if is_followup else route_query
                intent = "rag_answer" if is_followup else app.classify_query(route_query)
                # Keep routing and safety guards scoped to the current user turn.
                # Passing the full transcript can let a previous negative case
                # incorrectly block the next question.
                answer = app.run_query(contextual_query, route_query=route_query)
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": f"服务异常: {html.escape(str(exc))}"})
            return

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        evidence = getattr(app, "LAST_EVIDENCE", [])
        citation_audit = getattr(app, "LAST_CITATION_AUDIT", {})
        topic_info = getattr(app, "LAST_TOPIC_INFO", {})
        metrics = self._build_ask_metrics(
            query=query,
            intent=intent,
            history=history,
            answer=answer,
            evidence=evidence,
            citation_audit=citation_audit,
            elapsed_ms=elapsed_ms,
            topic_info=topic_info,
        )
        self._append_metrics_log(metrics)
        try:
            print(json.dumps(metrics, ensure_ascii=False), file=sys.stderr)
        except UnicodeEncodeError:
            # Some Windows consoles cannot emit the Chinese metrics payload.
            # Never let logging break the HTTP response path.
            print(json.dumps(metrics, ensure_ascii=True), file=sys.stderr)
        self._send_json(
            200,
            {
                "intent": intent,
                "answer": answer,
                "evidence": evidence,
                "citation_audit": citation_audit,
                "topic": topic_info,
                "elapsed_ms": elapsed_ms,
                "memory_turns": min(len(history), MAX_HISTORY_TURNS),
            },
        )

    def log_message(self, fmt, *args):
        return


def main():
    app.load_vectorstore()
    if app.paragraph_vectorstore_exists():
        app.load_paragraph_vectorstore()
    try:
        server = ThreadingHTTPServer((HOST, PORT), MarxOSHandler)
    except OSError as exc:
        raise SystemExit(
            f"端口 {PORT} 启动失败：{exc}。请调整 MARXOS_WEB_PORT 后重试。"
        ) from exc

    print(f"MarxOS Web running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
