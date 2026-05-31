import html
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import app
import marxos_web_citations as web_citations
import marxos_web_followups as web_followups
import marxos_web_support as web_support


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
        return web_support.append_metrics_log(metrics, METRICS_LOG_PATH)

    @staticmethod
    def _build_ask_metrics(query, intent, history, answer, evidence, citation_audit, elapsed_ms, topic_info):
        return web_support.build_ask_metrics(
            query,
            intent,
            history,
            answer,
            evidence,
            citation_audit,
            elapsed_ms,
            topic_info,
            MAX_HISTORY_TURNS,
            app.extract_answer_citation_lines,
        )

    @staticmethod
    def _trim_text(text, limit):
        return web_support.trim_text(text, limit)

    @classmethod
    def _build_history_summary(cls, history):
        return web_support.build_history_summary(history, cls._trim_text, SUMMARY_MAX_CHARS)

    @classmethod
    def _build_contextual_query(cls, query, history):
        return web_support.build_contextual_query(
            query,
            history,
            MAX_HISTORY_TURNS,
            MAX_HISTORY_CHARS,
            cls._build_history_summary,
            cls._trim_text,
        )

    @staticmethod
    def _is_contextual_followup(query):
        return web_support.is_contextual_followup(query)

    @staticmethod
    def _last_bot_message(history):
        return web_support.last_bot_message(history)

    @staticmethod
    def _last_bot_item(history):
        return web_support.last_bot_item(history)

    @staticmethod
    def _last_bot_topic(history):
        return web_support.last_bot_topic(history)

    @classmethod
    def _topic_scoped_query(cls, query, history):
        return web_support.topic_scoped_query(query, history, cls._is_contextual_followup)

    @staticmethod
    def _citation_from_evidence(item, index):
        return web_citations.citation_from_evidence(item, index)

    @staticmethod
    def _requested_citation_index(query):
        return web_citations.requested_citation_index(query)

    @staticmethod
    def _requested_citation_indices(query):
        return web_citations.requested_citation_indices(query)

    @staticmethod
    def _parse_citation_line(line):
        return web_citations.parse_citation_line(line)

    @classmethod
    def _parse_last_citations(cls, text):
        return web_citations.parse_last_citations(text)

    @staticmethod
    def _load_ocr_text(source, pdf_page):
        return web_citations.load_ocr_text(source, pdf_page, OCR_CACHE_DIR, app.repair_mojibake)

    @staticmethod
    def _find_pdf_page_by_printed_page(source, printed_page):
        return web_citations.find_pdf_page_by_printed_page(
            source,
            printed_page,
            OCR_CACHE_DIR,
            app.infer_printed_page_from_ocr_cache,
        )

    @staticmethod
    def _paragraphs_from_text(text):
        return web_citations.paragraphs_from_text(text)

    @classmethod
    def _answer_citation_followup(cls, query, history):
        return web_citations.answer_citation_followup(
            query,
            history,
            cls._is_contextual_followup,
            cls._last_bot_item,
            cls._last_bot_message,
            OCR_CACHE_DIR,
            app.repair_mojibake,
            app.infer_printed_page_from_ocr_cache,
        )

    @classmethod
    def _answer_evidence_page_followup(cls, query, history):
        return web_citations.answer_evidence_page_followup(
            query,
            history,
            cls._is_contextual_followup,
            cls._last_bot_item,
        )

    @classmethod
    def _answer_topic_rewrite_followup(cls, query, history):
        return web_followups.answer_topic_rewrite_followup(
            query,
            history,
            cls._last_bot_item,
            cls._requested_citation_indices,
        )

    @classmethod
    def _answer_topic_item_explain_followup(cls, query, history):
        return web_followups.answer_topic_item_explain_followup(
            query,
            history,
            cls._last_bot_item,
            cls._requested_citation_indices,
        )

    @staticmethod
    def _topic_history_evidence(history):
        return web_followups.topic_history_evidence(history, MarxOSHandler._last_bot_item)

    @staticmethod
    def _excerpt_key(item):
        return web_followups.excerpt_key(item, app.normalize_for_match)

    @classmethod
    def _rank_topic_evidence(cls, evidence):
        return web_followups.rank_topic_evidence(evidence, app.normalize_for_match)


    @staticmethod
    def _filter_ranked_evidence(ranked, markers_any=None, markers_all=None):
        return web_followups.filter_ranked_evidence(ranked, markers_any=markers_any, markers_all=markers_all)

    @classmethod
    def _answer_topic_history_followup(cls, query, history):
        return web_followups.answer_topic_history_followup(
            query,
            history,
            cls._last_bot_item,
            cls._requested_citation_indices,
            app.normalize_for_match,
        )

    @classmethod
    def _answer_history_followup(cls, query, history):
        return web_followups.answer_history_followup(
            query,
            history,
            cls._answer_topic_rewrite_followup,
            cls._answer_topic_item_explain_followup,
            cls._answer_topic_history_followup,
            cls._answer_evidence_page_followup,
            cls._answer_citation_followup,
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
            web_support.build_ask_response(
                intent,
                answer,
                evidence,
                citation_audit,
                topic_info,
                elapsed_ms,
                history,
                MAX_HISTORY_TURNS,
            ),
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
    for line in app.phoenix.startup_status_lines():
        print(line)
    server.serve_forever()


if __name__ == "__main__":
    main()
