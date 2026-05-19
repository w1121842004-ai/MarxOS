import html
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import app


HOST = "127.0.0.1"
PORT = int(os.getenv("MARXOS_WEB_PORT", "7860"))
MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 3500
SUMMARY_MAX_CHARS = 900


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MarxOS Web</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --surface: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --border: #d1d5db;
      --primary: #2563eb;
      --primary-dark: #1d4ed8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    .app-shell {
      max-width: 980px;
      margin: 0 auto;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    .header {
      position: sticky;
      top: 0;
      z-index: 2;
      background: rgba(245, 247, 251, 0.94);
      backdrop-filter: blur(4px);
      border-bottom: 1px solid #e5e7eb;
      padding: 14px 16px;
    }
    .title {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      line-height: 1.2;
    }
    .subtitle {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 12px;
    }
    .chat-main {
      flex: 1;
      min-height: 0;
      padding: 14px 16px 120px;
    }
    .chat {
      height: 100%;
      min-height: 300px;
      max-height: calc(100vh - 180px);
      overflow: auto;
      padding: 2px 0 10px;
    }
    .msg {
      margin: 0 0 12px;
      padding: 12px 14px;
      border-radius: 8px;
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 14px;
    }
    .msg-user {
      background: #eaf2ff;
      border: 1px solid #d2e3ff;
      margin-left: 60px;
    }
    .msg-bot {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      margin-right: 60px;
    }
    .msg-meta {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }
    .composer-wrap {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(to top, #f5f7fb 82%, rgba(245, 247, 251, 0));
      padding: 8px 16px 16px;
    }
    .composer {
      max-width: 980px;
      margin: 0 auto;
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
    }
    textarea {
      width: 100%;
      min-height: 64px;
      max-height: 220px;
      resize: vertical;
      border: 0;
      outline: none;
      border-radius: 8px;
      padding: 4px 6px;
      font-size: 15px;
      line-height: 1.5;
      background: transparent;
    }
    .toolbar {
      margin-top: 6px;
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    button {
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
    button:hover { background: var(--primary-dark); }
    button:disabled { opacity: 0.7; cursor: wait; }
    .meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      height: 22px;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0 8px;
      background: #f9fafb;
    }
    @media (max-width: 768px) {
      .msg-user { margin-left: 0; }
      .msg-bot { margin-right: 0; }
    }
  </style>
</head>
<body>
  <main class="app-shell">
    <header class="header">
      <h1 class="title">MarxOS</h1>
      <p class="subtitle">连续对话模式（自动保留会话）</p>
    </header>
    <section class="chat-main">
      <div class="chat" id="chat"></div>
    </section>
    <section class="composer-wrap">
      <div class="composer">
        <textarea id="q" placeholder="给 MarxOS 发送消息（Enter 发送，Shift+Enter 换行）"></textarea>
        <div class="toolbar">
          <button id="askBtn" type="button">发送</button>
          <div class="meta">
            <span class="badge" id="intent">意图：-</span>
            <span class="badge" id="cost">耗时：-</span>
          </div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const qEl = document.getElementById("q");
    const btnEl = document.getElementById("askBtn");
    const chatEl = document.getElementById("chat");
    const intentEl = document.getElementById("intent");
    const costEl = document.getElementById("cost");

    const messages = [];
    const memoryTurns = 6;

    function buildHistoryPayload() {
      const compact = [];
      for (const m of messages) {
        if (m.role === "user") compact.push({ role: "user", text: m.text });
        if (m.role === "bot") compact.push({ role: "bot", text: m.text });
      }
      return compact.slice(-memoryTurns * 2);
    }

    function renderChat() {
      if (!messages.length) {
        chatEl.innerHTML = '<div class="msg msg-bot">欢迎使用 MarxOS 网页端。你可以连续提问，我会按聊天流展示每一轮回答。</div>';
        return;
      }
      chatEl.innerHTML = messages.map((m) => {
        if (m.role === "user") {
          return '<div class="msg msg-user">' + m.text + '</div>';
        }
        return '<div class="msg msg-bot">' + m.text + '<div class="msg-meta">意图：' + m.intent + '｜耗时：' + m.cost + 'ms</div></div>';
      }).join("");
      chatEl.scrollTop = chatEl.scrollHeight;
      try { localStorage.setItem("marxos_chat_messages", JSON.stringify(messages)); } catch (_) {}
    }

    function loadChat() {
      try {
        const raw = localStorage.getItem("marxos_chat_messages");
        if (!raw) return;
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) return;
        for (const m of parsed) {
          if (!m || typeof m !== "object") continue;
          if (m.role !== "user" && m.role !== "bot") continue;
          if (typeof m.text !== "string") continue;
          messages.push({
            role: m.role,
            text: m.text,
            intent: typeof m.intent === "string" ? m.intent : "-",
            cost: m.cost ?? "-",
          });
        }
      } catch (_) {}
    }

    async function ask() {
      const query = qEl.value.trim();
      if (!query) return;
      btnEl.disabled = true;
      messages.push({ role: "user", text: query });
      renderChat();
      intentEl.textContent = "意图：识别中";
      costEl.textContent = "耗时：-";
      qEl.value = "";
      try {
        const res = await fetch("/api/ask", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({query, history: buildHistoryPayload()})
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "请求失败");
        messages.push({ role: "bot", text: data.answer || "", intent: data.intent, cost: data.elapsed_ms });
        renderChat();
        intentEl.textContent = "意图：" + data.intent;
        costEl.textContent = "耗时：" + data.elapsed_ms + "ms";
      } catch (err) {
        messages.push({ role: "bot", text: "请求失败：" + err.message, intent: "-", cost: "-" });
        renderChat();
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
    loadChat();
    renderChat();
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
        summary = "；".join(lines)
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
            lines.append(f"早期对话摘要：{older_summary}")

        user_recent = [item for item in recent if item.get("role") == "user"]
        for item in user_recent:
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
            "【对话上下文】\n"
            + "\n".join(lines)
            + f"\n\n【当前问题】\n{query}\n"
            + "请在回答当前问题时参考上下文。"
        )
        return MarxOSHandler._trim_text(contextual, MAX_HISTORY_CHARS)

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
            contextual_query = self._build_contextual_query(query, history)
            intent = app.classify_query(query)
            answer = app.run_query(contextual_query)
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": f"服务错误: {html.escape(str(exc))}"})
            return
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self._send_json(
            200,
            {
                "intent": intent,
                "answer": answer,
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
            f"端口 {PORT} 启动失败：{exc}. 可设置环境变量 MARXOS_WEB_PORT 使用其他端口。"
        ) from exc

    print(f"MarxOS Web running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
