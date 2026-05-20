import html
import json
import os
import re
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
        if (m.role === "bot") compact.push({ role: "bot", text: m.text, evidence: m.evidence || [] });
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
            direct_answer = self._answer_citation_followup(query, history)
            if direct_answer:
                intent = "citation_followup"
                answer = direct_answer
            else:
                contextual_query = self._build_contextual_query(query, history)
                intent = "rag_answer" if self._is_contextual_followup(query) else app.classify_query(query)
                # Keep routing and safety guards scoped to the current user turn.
                # Passing the full transcript can let a previous negative case
                # incorrectly block the next question.
                answer = app.run_query(contextual_query, route_query=query)
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": f"服务异常: {html.escape(str(exc))}"})
            return

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self._send_json(
            200,
            {
                "intent": intent,
                "answer": answer,
                "evidence": getattr(app, "LAST_EVIDENCE", []),
                "citation_audit": getattr(app, "LAST_CITATION_AUDIT", {}),
                "elapsed_ms": elapsed_ms,
                "memory_turns": min(len(history), MAX_HISTORY_TURNS),
            },
        )

    def log_message(self, fmt, *args):
        return


def main():
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
