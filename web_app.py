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


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MarxOS — 马克思主义学术助手</title>
<style>
:root{--bg:#f5f5f0;--surface:#fff;--panel:#1a1a2e;--panel-soft:#25253e;--text:#1a1a2e;--muted:#6b7280;--line:#e5e7eb;--primary:#8b0000;--primary-dark:#5c0000;--gold:#b8860b;--green:#2d6a4f;--red:#c0392b;--amber:#d4a017}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;font-family:"Segoe UI","PingFang SC","Noto Serif SC","Microsoft YaHei",serif;background:var(--bg);color:var(--text)}
.layout{display:grid;grid-template-columns:260px 1fr;height:100vh;overflow:hidden}
.sidebar{background:var(--panel);color:#e0d6c2;padding:14px 10px;border-right:1px solid #0b0b1a;display:flex;flex-direction:column;gap:8px;min-height:0;height:100vh;overflow:hidden}
.new-chat-btn{width:100%;border:1px solid #3d3d5c;background:var(--panel-soft);color:#e0d6c2;height:36px;border-radius:6px;font-size:13px;cursor:pointer}
.new-chat-btn:hover{background:#30305a}
.history-title{font-size:11px;color:#8a8a9a;padding:0 2px;text-transform:uppercase;letter-spacing:1px}
.history-list{overflow:auto;min-height:0;display:flex;flex-direction:column;gap:4px}
.history-item{text-align:left;border:1px solid #3d3d5c;background:transparent;color:#c8bfae;border-radius:6px;padding:6px 8px;cursor:pointer;width:100%}
.history-item.active{background:#2a2a4a;border-color:var(--gold)}
.history-item-title{font-size:12px;line-height:1.3;white-space:nowrap;text-overflow:ellipsis;overflow:hidden}
.history-item-time{margin-top:2px;font-size:10px;color:#6a6a7a}
.main{display:flex;flex-direction:column;min-width:0;height:100vh;overflow:hidden}
.topbar{height:52px;border-bottom:1px solid var(--line);background:rgba(255,255,255,.95);backdrop-filter:blur(6px);padding:0 18px;display:flex;align-items:center;justify-content:space-between}
.logo{font-size:16px;font-weight:700;color:var(--primary);letter-spacing:1px}
.subtitle{font-size:11px;color:var(--muted)}
.chat-wrap{flex:1;min-height:0;max-width:860px;width:100%;margin:0 auto;padding:16px 18px 150px;overflow-y:auto}
.msg{margin:0 0 12px;padding:12px 14px;border-radius:8px;line-height:1.7;white-space:pre-wrap;word-break:break-word;border:1px solid var(--line);background:#fff;font-size:14px}
.msg-user{background:#fef9f0;border-color:#e8d5b0;margin-left:48px}
.msg-bot{margin-right:48px;border-left:3px solid var(--primary)}
.msg-meta{margin-top:6px;color:var(--muted);font-size:11px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.badge{display:inline-flex;align-items:center;height:20px;border:1px solid var(--line);border-radius:4px;padding:0 6px;background:#fafaf5;font-size:11px}
.badge-ok{border-color:var(--green);color:var(--green)}
.badge-warn{border-color:var(--amber);color:var(--amber)}
.badge-err{border-color:var(--red);color:var(--red)}
.evidence-box{margin-top:8px;border-top:1px dashed var(--line);padding-top:6px;color:var(--muted);font-size:11px}
.evidence-box summary{cursor:pointer;color:#374151;font-weight:600;font-size:12px}
.evidence-item{margin-top:4px;padding:4px 6px;background:#fafaf5;border-radius:4px}
.evidence-cite{color:#111827;font-weight:500}
.evidence-mini{color:var(--muted);font-size:10px;margin-left:8px}
.evidence-excerpt{margin-top:3px;white-space:pre-wrap;font-size:12px;color:#555}
.mode-bar{display:flex;gap:6px;margin-right:10px}
.mode-btn{border:1px solid var(--line);background:#fff;color:var(--text);height:30px;padding:0 12px;border-radius:15px;font-size:12px;cursor:pointer;transition:all .15s}
.mode-btn.active{background:var(--primary);color:#fff;border-color:var(--primary)}
.composer-shell{position:fixed;bottom:0;left:260px;right:0;padding:8px 16px 14px;background:linear-gradient(to top,var(--bg) 80%,rgba(245,245,240,0))}
.composer{max-width:860px;margin:0 auto;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:8px}
textarea{width:100%;min-height:56px;max-height:180px;resize:vertical;border:0;outline:none;background:transparent;font-size:14px;line-height:1.5;color:var(--text)}
.toolbar{margin-top:4px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
.send-btn{border:0;background:var(--primary);color:#fff;height:32px;padding:0 14px;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer}
.send-btn:hover{background:var(--primary-dark)}.send-btn:disabled{opacity:.6;cursor:wait}
.meta{display:flex;gap:6px;color:var(--muted);font-size:11px;align-items:center}
.welcome{text-align:center;color:var(--muted);padding:40px 20px}
.welcome h2{color:var(--primary);margin-bottom:8px;font-size:20px}
.welcome p{font-size:13px;line-height:1.8}
@media(max-width:860px){.layout{grid-template-columns:1fr}.sidebar{min-height:unset;max-height:180px;height:180px}.composer-shell{left:0}.msg-user{margin-left:0}.msg-bot{margin-right:0}.chat-wrap{padding:12px 12px 150px}}
</style>
</head>
<body>
<div class="layout">
<aside class="sidebar">
<button id="newChatBtn" class="new-chat-btn">+ 新建对话</button>
<div class="history-title">历史会话</div>
<div id="historyList" class="history-list"></div>
</aside>
<section class="main">
<header class="topbar"><div class="logo">MarxOS</div><div class="subtitle">马克思主义文献检索 · 概念解释 · 学术分析</div></header>
<main id="chat" class="chat-wrap"><div class="welcome"><h2>MarxOS 学术助手</h2><p>精确问答：概念解释、引文出处、篇目定位<br>深度分析：理论分析、社会批判、学术论文<br>所有回答均附可核对的原文出处。</p></div></main>
<section class="composer-shell"><div class="composer">
<textarea id="q" placeholder="输入问题…（Enter 发送，Shift+Enter 换行）"></textarea>
<div class="toolbar">
<div style="display:flex;gap:8px;align-items:center">
<div class="mode-bar">
<button class="mode-btn active" data-mode="auto" id="modeAuto">智能</button>
<button class="mode-btn" data-mode="precise" id="modePrecise">精确问答</button>
<button class="mode-btn" data-mode="deep" id="modeDeep">深度分析</button>
</div>
<button id="askBtn" class="send-btn">发送</button>
</div>
<div class="meta"><span class="badge" id="intentBadge">就绪</span><span id="costLabel">-</span></div>
</div>
</div></section>
</section>
</div>
<script>
const qEl=document.getElementById("q"),btnEl=document.getElementById("askBtn"),chatEl=document.getElementById("chat");
const intentBadge=document.getElementById("intentBadge"),costLabel=document.getElementById("costLabel");
const historyListEl=document.getElementById("historyList"),newChatBtnEl=document.getElementById("newChatBtn");
const modeAuto=document.getElementById("modeAuto"),modePrecise=document.getElementById("modePrecise"),modeDeep=document.getElementById("modeDeep");
const STORE_KEY="marxos_v2",memoryTurns=6;
let conversations=[],currentId="",currentMode="auto";
function esc(s){return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;")}
function nowLabel(ts){try{return new Date(ts).toLocaleString()}catch(_){return""}}
function createConversation(){const n=Date.now();return{id:String(n)+"_"+Math.random().toString(36).slice(2,8),title:"新对话",createdAt:n,updatedAt:n,messages:[]}}
function getConv(){return conversations.find(x=>x.id===currentId)}
function setTitle(c){if(!c)return;const u=c.messages.find(m=>m.role==="user"&&m.text.trim());c.title=u?u.text.trim().slice(0,20):"新对话"}
function persist(){try{const ne=conversations.filter(c=>c.messages&&c.messages.length);localStorage.setItem(STORE_KEY,JSON.stringify(ne))}catch(_){}}
function prune(){conversations=conversations.sort((a,b)=>(b.updatedAt||0)-(a.updatedAt||0)).slice(0,50)}
function newChat(){conversations=conversations.filter(c=>c.messages&&c.messages.length);const f=createConversation();conversations.unshift(f);prune();currentId=f.id;persist();renderAll()}
function load(){try{const raw=localStorage.getItem(STORE_KEY);if(raw){const p=JSON.parse(raw);if(Array.isArray(p))conversations=p.filter(x=>x&&Array.isArray(x.messages)&&x.messages.length)}}catch(_){}if(!conversations.length){conversations=[createConversation()];currentId=conversations[0].id}else{newChat();return}persist();renderAll()}
function buildHistory(){const c=getConv();if(!c)return[];const h=[];for(const m of c.messages){if(m.role==="user")h.push({role:"user",text:m.text});if(m.role==="bot")h.push({role:"bot",text:m.text,evidence:m.evidence||[],topic:{topic_id:m.topicId||"",topic_label:m.topicLabel||"",topic_section:m.topicSection||""}})}return h.slice(-memoryTurns*2)}
function verifyBadge(v){if(!v||!v.total)return"";const ok=v.verified||0,pa=v.partial||0,ha=v.hallucinated||0;let cls="badge-ok",label="";if(ha>0){cls="badge-err";label=ha+"条引用待确认"}else if(pa>ok){cls="badge-warn";label=pa+"条转述引用"}else{label=ok+"条引用已校验"}return'<span class="badge '+cls+'">'+label+'</span>'}
function evidenceHtml(ev){if(!Array.isArray(ev)||!ev.length)return"";const items=ev.slice(0,6).map((e,i)=>{const cite=esc(e.citation||e.sentence_citation||""),src=esc(e.source||""),pg=esc(e.printed_page||e.citation_page||""),meta=src+(pg?" | 第"+pg+"页":""),ex=esc(String(e.excerpt||"").slice(0,200));return'<details class="evidence-item"><summary><span class="evidence-cite">['+(i+1)+'] '+cite+'</span><span class="evidence-mini">'+meta+'</span></summary>'+(ex?'<div class="evidence-excerpt">'+ex+'</div>':'')+'</details>'}).join("");return'<details class="evidence-box"><summary>查看证据卡片 ('+ev.length+')</summary>'+items+'</details>'}
function renderHistory(){const vis=conversations.filter(c=>c.messages&&c.messages.length);if(!vis.length){historyListEl.innerHTML="";return}historyListEl.innerHTML=vis.map(c=>{const act=c.id===currentId?" active":"";return'<button class="history-item'+act+'" data-id="'+esc(c.id)+'"><div class="history-item-title">'+esc(c.title||"新对话")+'</div><div class="history-item-time">'+esc(nowLabel(c.updatedAt||c.createdAt))+'</div></button>'}).join("");for(const n of historyListEl.querySelectorAll(".history-item")){n.addEventListener("click",()=>{const id=n.getAttribute("data-id");if(id){currentId=id;renderAll()}})}}
function renderChat(){const c=getConv(),ms=c?c.messages:[];if(!ms.length){chatEl.innerHTML='<div class="welcome"><h2>MarxOS 学术助手</h2><p>精确问答：概念解释、引文出处、篇目定位<br>深度分析：理论分析、社会批判、学术论文<br>所有回答均附可核对的原文出处。</p></div>';return}chatEl.innerHTML=ms.map(m=>{if(m.role==="user")return'<div class="msg msg-user">'+esc(m.text)+'</div>';let meta='<div class="msg-meta">';if(m.intent)meta+='<span class="badge">'+esc(m.intent)+'</span>';if(m.mode)meta+='<span class="badge">'+esc(m.mode)+'</span>';if(m.verify)meta+=verifyBadge(m.verify);if(m.crag)meta+='<span class="badge">CRAG:'+esc(String(m.crag))+'</span>';meta+='<span>'+esc(String(m.cost||"-"))+'ms</span></div>';return'<div class="msg msg-bot">'+esc(m.text)+evidenceHtml(m.evidence||[])+meta+'</div>'}).join("");chatEl.scrollTop=chatEl.scrollHeight}
function renderAll(){renderHistory();renderChat()}
function setMode(m){currentMode=m;modeAuto.classList.toggle("active",m==="auto");modePrecise.classList.toggle("active",m==="precise");modeDeep.classList.toggle("active",m==="deep")}
modeAuto.addEventListener("click",()=>setMode("auto"));modePrecise.addEventListener("click",()=>setMode("precise"));modeDeep.addEventListener("click",()=>setMode("deep"));
async function ask(){const query=qEl.value.trim();if(!query)return;const conv=getConv();if(!conv)return;btnEl.disabled=true;conv.messages.push({role:"user",text:query});conv.updatedAt=Date.now();setTitle(conv);renderAll();persist();intentBadge.textContent="处理中...";costLabel.textContent="";qEl.value="";try{const res=await fetch("/api/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({query,history:buildHistory(),mode:currentMode})});const data=await res.json();if(!res.ok)throw new Error(data.error||"请求失败");conv.messages.push({role:"bot",text:data.answer||"",intent:data.intent||"-",mode:data.mode||"",cost:data.elapsed_ms||"-",evidence:Array.isArray(data.evidence)?data.evidence:[],verify:data.citation_audit?.content_verification||null,crag:(data.citation_audit?.crag_report?.score)||null,topicId:(data.topic?.topic_id)||"",topicLabel:(data.topic?.topic_label)||"",topicSection:(data.topic?.topic_section)||""});conv.updatedAt=Date.now();setTitle(conv);prune();renderAll();persist();const vfy=data.citation_audit?.content_verification;const vOk=vfy?(vfy.verified||0)+(vfy.partial||0):0;const vTotal=vfy?.total||0;intentBadge.textContent=data.intent||"-";costLabel.textContent=(data.elapsed_ms||"-")+"ms"+(vTotal?" | 校验:"+vOk+"/"+vTotal:"")}catch(err){const msg=err?.message||"请求失败";conv.messages.push({role:"bot",text:"请求失败："+msg,intent:"-",cost:"-"});conv.updatedAt=Date.now();setTitle(conv);renderAll();persist()}finally{btnEl.disabled=false;qEl.focus()}}
btnEl.addEventListener("click",ask);qEl.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();ask()}});newChatBtnEl.addEventListener("click",newChat);load();
</script>
</body>
</html>"""


class MarxOSHandler(BaseHTTPRequestHandler):
    @staticmethod
    def _append_metrics_log(metrics):
        return web_support.append_metrics_log(metrics, METRICS_LOG_PATH)

    @staticmethod
    def _build_ask_metrics(query, intent, history, answer, evidence, citation_audit, elapsed_ms, topic_info, crag_report):
        return web_support.build_ask_metrics(
            query, intent, history, answer, evidence, citation_audit,
            elapsed_ms, topic_info, crag_report, MAX_HISTORY_TURNS,
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
            query, history, MAX_HISTORY_TURNS, MAX_HISTORY_CHARS,
            cls._build_history_summary, cls._trim_text,
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
            source, printed_page, OCR_CACHE_DIR, app.infer_printed_page_from_ocr_cache,
        )

    @staticmethod
    def _paragraphs_from_text(text):
        return web_citations.paragraphs_from_text(text)

    @classmethod
    def _answer_citation_followup(cls, query, history):
        return web_citations.answer_citation_followup(
            query, history, cls._is_contextual_followup, cls._last_bot_item,
            cls._last_bot_message, OCR_CACHE_DIR, app.repair_mojibake,
            app.infer_printed_page_from_ocr_cache,
        )

    @classmethod
    def _answer_evidence_page_followup(cls, query, history):
        return web_citations.answer_evidence_page_followup(
            query, history, cls._is_contextual_followup, cls._last_bot_item,
        )

    @classmethod
    def _answer_topic_rewrite_followup(cls, query, history):
        return web_followups.answer_topic_rewrite_followup(
            query, history, cls._last_bot_item, cls._requested_citation_indices,
        )

    @classmethod
    def _answer_topic_item_explain_followup(cls, query, history):
        return web_followups.answer_topic_item_explain_followup(
            query, history, cls._last_bot_item, cls._requested_citation_indices,
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
            query, history, cls._last_bot_item, cls._requested_citation_indices,
            app.normalize_for_match,
        )

    @classmethod
    def _answer_history_followup(cls, query, history):
        return web_followups.answer_history_followup(
            query, history, cls._answer_topic_rewrite_followup,
            cls._answer_topic_item_explain_followup, cls._answer_topic_history_followup,
            cls._answer_evidence_page_followup, cls._answer_citation_followup,
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
        mode = data.get("mode", "auto")
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

                # Mode routing: force deep_analysis or precise based on user selection
                if mode == "deep":
                    force = "deep_analysis"
                else:
                    force = None  # auto: let run_query classify internally

                try:
                    kwargs = {"route_query": route_query, "history": history}
                    if force is not None:
                        kwargs["force_intent"] = force
                    answer = app.run_query(contextual_query, **kwargs)
                except TypeError as exc:
                    if "history" not in str(exc):
                        raise
                    if force is not None:
                        answer = app.run_query(contextual_query, route_query=route_query, force_intent=force)
                    else:
                        answer = app.run_query(contextual_query, route_query=route_query)
                intent = app.LAST_CITATION_AUDIT.get("crag_report", {}).get("intent") or force or app.classify_query(route_query)
        except Exception as exc:
            self._send_json(500, {"error": f"服务异常: {html.escape(str(exc))}"})
            return

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        evidence = getattr(app, "LAST_EVIDENCE", [])
        citation_audit = getattr(app, "LAST_CITATION_AUDIT", {})
        topic_info = getattr(app, "LAST_TOPIC_INFO", {})
        crag_report = getattr(app, "LAST_CRAG_REPORT", {})
        metrics = self._build_ask_metrics(
            query=query, intent=intent, history=history, answer=answer,
            evidence=evidence, citation_audit=citation_audit, elapsed_ms=elapsed_ms,
            topic_info=topic_info, crag_report=crag_report,
        )
        self._append_metrics_log(metrics)
        try:
            print(json.dumps(metrics, ensure_ascii=False), file=sys.stderr)
        except UnicodeEncodeError:
            print(json.dumps(metrics, ensure_ascii=True), file=sys.stderr)
        self._send_json(
            200,
            web_support.build_ask_response(
                intent, answer, evidence, citation_audit, topic_info,
                crag_report, elapsed_ms, history, MAX_HISTORY_TURNS,
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
