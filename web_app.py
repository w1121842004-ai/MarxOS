import html
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import app
from marxos.config import get_settings
from marxos.runtime_health import readiness_report, write_runtime_manifest
from marxos.web import citations as web_citations
from marxos.web import followups as web_followups
from marxos.web import miss_log
from marxos.web import support as web_support


SETTINGS = get_settings()
HOST = SETTINGS.web.host
PORT = SETTINGS.web.port
MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 3500
SUMMARY_MAX_CHARS = 900
OCR_CACHE_DIR = Path(app.OCR_CACHE_DIR)
METRICS_LOG_PATH = Path(SETTINGS.web.metrics_log_path)


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MarxOS — 马克思主义学术助手</title>
<style>
:root{--bg:#f7f7f5;--surface:#fff;--surface-soft:#f1f1ee;--panel:#20201f;--panel-hover:#2b2b29;--text:#20201f;--muted:#74746f;--line:#deded9;--line-strong:#c9c9c2;--primary:#a33a32;--primary-dark:#842e28;--green:#31735b;--red:#b8463c;--amber:#a46c18;--shadow:0 12px 36px rgba(31,31,29,.09)}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Noto Sans CJK SC","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--text);font-size:15px;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
button,textarea{font:inherit}button:focus-visible,textarea:focus-visible{outline:2px solid color-mix(in srgb,var(--primary) 55%,transparent);outline-offset:2px}
.layout{display:grid;grid-template-columns:248px minmax(0,1fr);height:100dvh;overflow:hidden}
.sidebar{background:var(--panel);color:#f1f1ed;padding:12px 10px;border-right:1px solid #161615;display:flex;flex-direction:column;gap:12px;min-height:0;height:100dvh;overflow:hidden}
.new-chat-btn{width:100%;border:1px solid #454542;background:#2a2a28;color:#f4f4ef;height:40px;border-radius:9px;font-size:13px;font-weight:600;cursor:pointer;transition:background .15s,border-color .15s}
.new-chat-btn:hover{background:#343431;border-color:#5a5a55}
.history-title{font-size:11px;color:#92928b;padding:2px 8px 0;letter-spacing:.04em}
.history-list{overflow:auto;min-height:0;display:flex;flex-direction:column;gap:4px}
.history-item{text-align:left;border:0;background:transparent;color:#c9c9c3;border-radius:8px;padding:8px 9px;cursor:pointer;width:100%;transition:background .15s,color .15s}
.history-item:hover{background:var(--panel-hover);color:#fff}.history-item.active{background:#353532;color:#fff}
.history-item-title{font-size:13px;line-height:1.35;white-space:nowrap;text-overflow:ellipsis;overflow:hidden}
.history-item-time{margin-top:3px;font-size:10px;color:#85857e}
.main{display:flex;flex-direction:column;min-width:0;height:100dvh;overflow:hidden}
.topbar{height:54px;border-bottom:1px solid var(--line);background:rgba(247,247,245,.88);backdrop-filter:blur(14px);padding:0 22px;display:flex;align-items:center;justify-content:space-between;z-index:2}
.logo{font-size:15px;font-weight:650;color:var(--text);letter-spacing:-.01em}.logo::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:9px;background:var(--primary)}
.subtitle{font-size:12px;color:var(--muted)}
.chat-wrap{flex:1;min-height:0;max-width:820px;width:100%;margin:0 auto;padding:30px 24px 178px;overflow-y:auto;scrollbar-gutter:stable}
.msg{margin:0 0 20px;padding:0;line-height:1.78;white-space:pre-wrap;word-break:break-word;background:transparent;font-size:15px;letter-spacing:.005em}
.msg-body{white-space:normal}.msg-body p{margin:0 0 12px}.msg-body p:last-child{margin-bottom:0}.msg-body strong{font-weight:650;color:#171716}.msg-body ol,.msg-body ul{margin:10px 0 14px 24px;padding:0}.msg-body li{margin:5px 0}.msg-body .cite-ref{font-size:11px;color:var(--primary);font-weight:700;vertical-align:super;margin:0 1px}.msg-body .footnotes{margin-top:18px;padding-top:12px;border-top:1px solid var(--line);font-size:13px;color:#555550}.msg-body .footnotes-title{font-weight:650;margin-bottom:6px;color:var(--text)}
.msg-user{width:fit-content;max-width:min(78%,640px);margin-left:auto;padding:10px 14px;background:#e9e9e5;border:1px solid #e0e0da;border-radius:16px 16px 4px 16px;line-height:1.58}
.msg-bot{margin-right:42px;padding-left:1px}
.msg-meta{margin-top:10px;color:var(--muted);font-size:11px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.badge{display:inline-flex;align-items:center;min-height:21px;border:1px solid var(--line);border-radius:999px;padding:1px 7px;background:var(--surface-soft);font-size:10.5px}
.badge-ok{border-color:var(--green);color:var(--green)}
.badge-warn{border-color:var(--amber);color:var(--amber)}
.badge-err{border-color:var(--red);color:var(--red)}
.fb-btn{background:none;border:1px solid var(--line);border-radius:999px;padding:1px 8px;font-size:10px;color:var(--muted);cursor:pointer;margin-left:8px}
.fb-btn:hover{color:var(--red);border-color:var(--red)}
.msg-error .msg-body{color:var(--red);border:1px solid var(--red);border-radius:8px;padding:8px 10px;background:rgba(255,0,0,.04)}
.evidence-box{margin-top:16px;border:1px solid var(--line);border-radius:10px;padding:9px 11px;color:var(--muted);font-size:11px;background:rgba(255,255,255,.55)}
.evidence-box summary{cursor:pointer;color:#4b4b47;font-weight:600;font-size:12px}
.evidence-item{margin-top:7px;padding:7px 8px;background:var(--surface-soft);border-radius:7px}
.evidence-cite{color:var(--text);font-weight:550}
.evidence-mini{color:var(--muted);font-size:10px;margin-left:8px}
.evidence-chip{display:inline-block;margin-left:8px;border-radius:999px;padding:0 6px;font-size:9.5px;border:1px solid var(--line)}
.evidence-chip-ok{color:#3f6b3f}
.evidence-chip-warn{color:#8a5a2b}
.evidence-excerpt{margin-top:3px;white-space:pre-wrap;font-size:12px;color:#555}
.mode-bar{display:flex;gap:3px;padding:3px;background:var(--surface-soft);border-radius:9px}
.mode-btn{border:0;background:transparent;color:var(--muted);height:28px;padding:0 10px;border-radius:7px;font-size:11.5px;cursor:pointer;transition:all .15s}
.mode-btn:hover{color:var(--text)}.mode-btn.active{background:#fff;color:var(--text);box-shadow:0 1px 3px rgba(0,0,0,.1)}
.composer-shell{position:fixed;bottom:0;left:248px;right:0;padding:22px 18px 18px;background:linear-gradient(to top,var(--bg) 72%,rgba(247,247,245,0))}
.composer{max-width:820px;margin:0 auto;background:var(--surface);border:1px solid var(--line-strong);border-radius:16px;padding:11px 12px 10px;box-shadow:var(--shadow);transition:border-color .15s,box-shadow .15s}.composer:focus-within{border-color:#aaa9a1;box-shadow:0 14px 40px rgba(31,31,29,.12)}
textarea{width:100%;min-height:58px;max-height:220px;resize:none;border:0;outline:none;background:transparent;font-size:15px;line-height:1.65;color:var(--text);padding:2px 4px}textarea::placeholder{color:#9a9a94}
.toolbar{margin-top:7px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
.send-btn{border:0;background:var(--text);color:#fff;height:34px;padding:0 15px;border-radius:9px;font-size:12.5px;font-weight:600;cursor:pointer}
.send-btn:hover{background:var(--primary-dark)}.send-btn:disabled{opacity:.6;cursor:wait}
.meta{display:flex;gap:6px;color:var(--muted);font-size:11px;align-items:center}
.welcome{max-width:560px;margin:14vh auto 0;color:var(--muted);padding:30px 24px;text-align:center}
.welcome h2{color:var(--text);margin:0 0 12px;font-size:26px;letter-spacing:-.035em;font-weight:650}.welcome h2::before{content:"M";display:flex;align-items:center;justify-content:center;width:44px;height:44px;margin:0 auto 20px;border-radius:13px;background:var(--text);color:#fff;font-size:19px;font-weight:700;box-shadow:0 8px 24px rgba(31,31,29,.15)}
.welcome p{font-size:14px;line-height:1.85;margin:0}
@media(max-width:860px){.layout{grid-template-columns:1fr}.sidebar{display:none}.composer-shell{left:0;padding:18px 10px 10px}.topbar{padding:0 14px}.subtitle{display:none}.chat-wrap{padding:22px 14px 170px}.msg-user{max-width:88%}.msg-bot{margin-right:0}.welcome{margin-top:8vh;padding-inline:10px}}
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
<div class="meta"><span class="badge" id="sysBadge">启动中...</span><span class="badge" id="intentBadge">就绪</span><span id="costLabel">-</span></div>
</div>
</div></section>
</section>
</div>
<script>
const qEl=document.getElementById("q"),btnEl=document.getElementById("askBtn"),chatEl=document.getElementById("chat");
const intentBadge=document.getElementById("intentBadge"),costLabel=document.getElementById("costLabel"),sysBadge=document.getElementById("sysBadge");
function checkReady(){fetch("/readyz").then(r=>r.json()).then(d=>{if(d.ready){sysBadge.textContent="系统就绪";sysBadge.className="badge badge-ok"}else{sysBadge.textContent="部分可用";sysBadge.className="badge badge-warn"}}).catch(()=>{sysBadge.textContent="服务不可用";sysBadge.className="badge badge-err"})}
const historyListEl=document.getElementById("historyList"),newChatBtnEl=document.getElementById("newChatBtn");
const modeAuto=document.getElementById("modeAuto"),modePrecise=document.getElementById("modePrecise"),modeDeep=document.getElementById("modeDeep");
const STORE_KEY="marxos_v2",memoryTurns=6;
let conversations=[],currentId="",currentMode="auto";
function esc(s){return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;")}
function inlineMd(s){return esc(s).replace(/\*\*([^*]+)\*\*/g,"<strong>$1</strong>").replace(/【(\d+)】/g,'<span class="cite-ref">[$1]</span>')}
function renderMd(s){const lines=String(s||"").replace(/\r\n/g,"\n").split("\n");let html="",list="",inFoot=false,footOpen=false;function flushList(){if(list){html+=list+"</ol>";list=""}}function closeFoot(){if(footOpen){flushList();html+="</div>";footOpen=false;inFoot=false}}for(const raw of lines){const line=raw.trim();if(!line){flushList();continue}if(/^引文注释$/.test(line)||/^\*+引文注释\*+$/.test(line)){closeFoot();html+='<div class="footnotes"><div class="footnotes-title">引文注释</div>';inFoot=true;footOpen=true;continue}const m=line.match(/^(\d+)[.、]\s+(.+)$/);if(m){if(!list)list="<ol>";list+="<li>"+inlineMd(m[2])+"</li>";continue}if(!inFoot)flushList();if(line.startsWith("- ")||line.startsWith("— ")){html+="<p>"+inlineMd(line.replace(/^[-—]\s*/,""))+"</p>";continue}if(inFoot){flushList();html+="<p>"+inlineMd(line)+"</p>"}else{html+="<p>"+inlineMd(line)+"</p>"}}flushList();closeFoot();return html}
function nowLabel(ts){try{return new Date(ts).toLocaleString()}catch(_){return""}}
function createConversation(){const n=Date.now();return{id:String(n)+"_"+Math.random().toString(36).slice(2,8),title:"新对话",createdAt:n,updatedAt:n,messages:[]}}
function getConv(){return conversations.find(x=>x.id===currentId)}
function setTitle(c){if(!c)return;const u=c.messages.find(m=>m.role==="user"&&m.text.trim());c.title=u?u.text.trim().slice(0,20):"新对话"}
function persist(){try{const ne=conversations.filter(c=>c.messages&&c.messages.length);localStorage.setItem(STORE_KEY,JSON.stringify(ne))}catch(_){}}
function prune(){conversations=conversations.sort((a,b)=>(b.updatedAt||0)-(a.updatedAt||0)).slice(0,50)}
function newChat(){conversations=conversations.filter(c=>c.messages&&c.messages.length);const f=createConversation();conversations.unshift(f);prune();currentId=f.id;persist();renderAll()}
function normalizeMessage(m){if(!m||!m.role)return null;const text=String(m.text||m.answer||m.content||"");if(!text.trim())return null;if(m.role==="user")return{role:"user",text};return{role:"bot",text,intent:m.intent||"-",mode:m.mode||"",path:m.path||"",cost:m.cost||"-",evidence:Array.isArray(m.evidence)?m.evidence:[],verify:m.verify||null,crag:m.crag||null,topicId:m.topicId||"",topicLabel:m.topicLabel||"",topicSection:m.topicSection||""}}
function load(){try{const raw=localStorage.getItem(STORE_KEY);if(raw){const p=JSON.parse(raw);if(Array.isArray(p))conversations=p.map(c=>({id:c&&c.id||("c"+Date.now()+"_"+Math.random().toString(36).slice(2,7)),title:c&&c.title||"新对话",updatedAt:c&&c.updatedAt||Date.now(),messages:Array.isArray(c&&c.messages)?c.messages.map(normalizeMessage).filter(Boolean):[]})).filter(c=>c.messages.length)}}catch(_){}if(!conversations.length){conversations=[createConversation()];currentId=conversations[0].id}else{newChat();checkReady();return}persist();renderAll();checkReady()}
function buildHistory(){const c=getConv();if(!c)return[];const h=[];for(const m of c.messages){if(m.role==="user")h.push({role:"user",text:m.text});if(m.role==="bot")h.push({role:"bot",text:m.text,intent:m.intent||"",evidence:m.evidence||[],topic:{topic_id:m.topicId||"",topic_label:m.topicLabel||"",topic_section:m.topicSection||""}})}return h.slice(-memoryTurns*2)}
function verifyBadge(v){if(!v||!v.total)return"";const ok=v.verified||0,pa=v.partial||0,ha=v.hallucinated||0;let cls="badge-ok",label="";if(ha>0){cls="badge-err";label=ha+"条引用待确认"}else if(pa>ok){cls="badge-warn";label=pa+"条转述引用"}else{label=ok+"条引用已校验"}return'<span class="badge '+cls+'">'+label+'</span>'}
function evidenceHtml(ev){if(!Array.isArray(ev)||!ev.length)return"";const labels={exact_quote:"原文核对",locator_backstop:"定位提示",cache_backstop:"页段回退",vector_candidate:"未确认候选",sparse_candidate:"稀疏候选",paragraph_vector_candidate:"段落候选"};const items=ev.slice(0,6).map((e,i)=>{const cite=esc(e.citation||e.sentence_citation||""),src=esc(e.source||""),pg=esc(e.printed_page||e.citation_page||""),meta=src+(pg?" | 第"+pg+"页":""),ex=esc(String(e.excerpt||"").slice(0,200));const mt=e.match_type||"";const chip=labels[mt]?('<span class="evidence-chip '+(mt==="exact_quote"?"evidence-chip-ok":"evidence-chip-warn")+'">'+labels[mt]+'</span>'):"";return'<details class="evidence-item"><summary><span class="evidence-cite">['+(i+1)+'] '+cite+'</span><span class="evidence-mini">'+meta+'</span>'+chip+'</summary>'+(ex?'<div class="evidence-excerpt">'+ex+'</div>':'')+'</details>'}).join("");return'<details class="evidence-box"><summary>查看证据卡片 ('+ev.length+')</summary>'+items+'</details>'}
function renderHistory(){const vis=conversations.filter(c=>c.messages&&c.messages.length);if(!vis.length){historyListEl.innerHTML="";return}historyListEl.innerHTML=vis.map(c=>{const act=c.id===currentId?" active":"";return'<button class="history-item'+act+'" data-id="'+esc(c.id)+'"><div class="history-item-title">'+esc(c.title||"新对话")+'</div><div class="history-item-time">'+esc(nowLabel(c.updatedAt||c.createdAt))+'</div></button>'}).join("");for(const n of historyListEl.querySelectorAll(".history-item")){n.addEventListener("click",()=>{const id=n.getAttribute("data-id");if(id){currentId=id;renderAll()}})}}
function renderChat(){const c=getConv(),ms=c?c.messages:[];if(!ms.length){chatEl.innerHTML='<div class="welcome"><h2>MarxOS 学术助手</h2><p>精确问答：概念解释、引文出处、篇目定位<br>深度分析：理论分析、社会批判、学术论文<br>所有回答均附可核对的原文出处。</p></div>';return}chatEl.innerHTML=ms.map((m,i)=>{if(m.role==="user")return'<div class="msg msg-user"><div class="msg-body">'+renderMd(m.text)+'</div></div>';if(m.intent==="error")return'<div class="msg msg-bot msg-error"><div class="msg-body">'+esc(m.text)+'</div><div class="msg-meta"><span class="badge badge-err">错误</span></div></div>';let meta='<div class="msg-meta">';if(m.intent)meta+='<span class="badge">'+esc(m.intent)+'</span>';if(m.mode)meta+='<span class="badge">'+esc(m.mode)+'</span>';if(m.path&&m.path!=="llm")meta+='<span class="badge">'+esc(m.path)+'</span>';if(m.verify)meta+=verifyBadge(m.verify);if(m.crag)meta+='<span class="badge">CRAG:'+esc(String(m.crag))+'</span>';meta+='<span>'+esc(String(m.cost||"-"))+'ms</span></div>';const prev=ms[i-1];const prevQ=prev&&prev.role==="user"?prev.text:"";meta+='<button class="fb-btn" data-q="'+esc(prevQ)+'" onclick="sendFeedback(this)">回答不准确</button>';return'<div class="msg msg-bot"><div class="msg-body">'+renderMd(m.text)+'</div>'+evidenceHtml(m.evidence||[])+meta+'</div>'}).join("");chatEl.scrollTop=chatEl.scrollHeight}
function sendFeedback(btn){const q=btn.dataset.q||"";if(!q)return;fetch("/api/feedback",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({query:q,message:"用户标记回答不准确"})}).then(()=>{btn.textContent="已记录";btn.disabled=true}).catch(()=>{})}
function renderAll(){renderHistory();renderChat()}
function setMode(m){currentMode=m;modeAuto.classList.toggle("active",m==="auto");modePrecise.classList.toggle("active",m==="precise");modeDeep.classList.toggle("active",m==="deep")}
modeAuto.addEventListener("click",()=>setMode("auto"));modePrecise.addEventListener("click",()=>setMode("precise"));modeDeep.addEventListener("click",()=>setMode("deep"));
function pushBotFinal(conv,data){conv.messages.push({role:"bot",text:data.answer||"",intent:data.intent||"-",mode:data.mode||"",path:data.path||"",cost:data.elapsed_ms||"-",evidence:Array.isArray(data.evidence)?data.evidence:[],verify:data.citation_audit?.content_verification||null,crag:(data.citation_audit?.crag_report?.score)||null,topicId:(data.topic?.topic_id)||"",topicLabel:(data.topic?.topic_label)||"",topicSection:(data.topic?.topic_section)||""});conv.updatedAt=Date.now();setTitle(conv);prune();renderAll();persist();checkReady();const vfy=data.citation_audit?.content_verification;const vOk=vfy?(vfy.verified||0)+(vfy.partial||0):0;const vTotal=vfy?.total||0;intentBadge.textContent=data.intent||"-";costLabel.textContent=(data.elapsed_ms||"-")+"ms"+(vTotal?" | 校验:"+vOk+"/"+vTotal:"")}
async function ask(){const query=qEl.value.trim();if(!query)return;const conv=getConv();if(!conv)return;btnEl.disabled=true;const historyPayload=buildHistory();conv.messages.push({role:"user",text:query});const pending={role:"bot",text:"正在分析问题...",intent:"stream",mode:currentMode,cost:"-"};conv.messages.push(pending);conv.updatedAt=Date.now();setTitle(conv);renderAll();persist();intentBadge.textContent="处理中...";costLabel.textContent="";qEl.value="";const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),180000);try{const res=await fetch("/api/ask_stream",{method:"POST",signal:controller.signal,headers:{"Content-Type":"application/json"},body:JSON.stringify({query,history:historyPayload,mode:currentMode})});if(!res.ok||!res.body)throw new Error("流式请求失败");const reader=res.body.getReader();const decoder=new TextDecoder("utf-8");let buf="",doneFinal=false;while(true){const {value,done}=await reader.read();if(done)break;buf+=decoder.decode(value,{stream:true});let parts=buf.split("\n\n");buf=parts.pop()||"";for(const part of parts){let ev="message",dataLine="";for(const line of part.split("\n")){if(line.startsWith("event:"))ev=line.slice(6).trim();if(line.startsWith("data:"))dataLine+=line.slice(5).trim()}if(!dataLine)continue;const data=JSON.parse(dataLine);if(ev==="status"){pending.text=data.message||"处理中...";conv.updatedAt=Date.now();renderAll();persist()}else if(ev==="final"){const idx=conv.messages.indexOf(pending);if(idx>=0)conv.messages.splice(idx,1);pushBotFinal(conv,data);doneFinal=true}else if(ev==="error"){throw new Error(data.error||"请求失败")}}}if(!doneFinal)throw new Error("流式响应未完成")}catch(err){let msg=err?.message||"请求失败";if(err?.name==="AbortError")msg="请求超时：180 秒内未收到完整回答";pending.text="请求失败："+msg;pending.intent="error";conv.updatedAt=Date.now();renderAll();persist()}finally{clearTimeout(timer);btnEl.disabled=false;qEl.focus()}}
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

    @staticmethod
    def _names_explicit_subject(query):
        # A named concept or verbatim work title means the question has its own
        # subject and must not inherit the previous topic scope.
        return bool(app.active_concept_terms(query)) or app.work_catalog_title_mentioned(query)

    @classmethod
    def _topic_scoped_query(cls, query, history):
        return web_support.topic_scoped_query(
            query,
            history,
            cls._is_contextual_followup,
            cls._names_explicit_subject,
        )

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

    @staticmethod
    def _has_explicit_work_reference(query):
        """Return True when the current turn names a catalogued work.

        A page request for a newly named work is a fresh lookup, even when the
        previous answer contains evidence with page metadata.
        """
        return bool(app.work_catalog_title_entries_for_query(query or ""))

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
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok"})
            return
        if self.path == "/readyz":
            report = readiness_report(SETTINGS, app.RUNTIME)
            self._send_json(200 if report["ready"] else 503, report)
            return
        if self.path in ("/", "/index.html"):
            self._send_html(HTML_PAGE)
            return
        self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path not in ("/api/ask", "/api/ask_stream", "/api/feedback"):
            self.send_error(404, "Not Found")
            return
        content_len = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_len)
        if self.path == "/api/feedback":
            try:
                feedback = json.loads(raw.decode("utf-8"))
                miss_log.log_query_miss(
                    "user_feedback",
                    str(feedback.get("query") or "")[:500],
                    detail=str(feedback.get("message") or "")[:300],
                )
                self._send_json(200, {"status": "ok"})
            except Exception:
                self._send_json(400, {"error": "无效反馈"})
            return
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "无效 JSON"})
            return

        if self.path == "/api/ask_stream":
            self._handle_ask_stream(data)
            return
        self._handle_ask_json(data)

    def _mode_routing(self, mode):
        if mode == "fast":
            return None, "fast"
        if mode == "standard":
            return None, "standard"
        if mode == "deep":
            return "deep_analysis", "deep"
        if mode == "precise":
            return None, "fast"
        return None, "fast"

    def _run_ask_payload(self, data, emit=None):
        query = (data.get("query") or "").strip()
        history = data.get("history") or []
        mode = data.get("mode", "auto")
        if not query:
            return 400, {"error": "问题不能为空"}

        started = time.perf_counter()
        try:
            if emit:
                emit("status", {"message": "正在分析问题..."})
            direct_answer = None
            if not self._has_explicit_work_reference(query):
                direct_answer = self._answer_history_followup(query, history)
            if direct_answer:
                last_bot = self._last_bot_item(history)
                intent = "chitchat" if (last_bot.get("intent") == "chitchat") else "citation_followup"
                answer = direct_answer
                performance = "local"
                app.LAST_ANSWER_PATH = intent
            else:
                route_query = self._topic_scoped_query(query, history)
                is_followup = self._is_contextual_followup(query)
                contextual_query = self._build_contextual_query(route_query, history) if is_followup else route_query

                force, performance = self._mode_routing(mode)
                if emit:
                    emit("status", {"message": f"正在检索证据（{performance}）..."})

                try:
                    kwargs = {"route_query": route_query, "history": history, "performance": performance}
                    if force is not None:
                        kwargs["force_intent"] = force
                    if emit:
                        emit("status", {"message": "正在生成回答..."})
                    answer = app.run_query(contextual_query, **kwargs)
                except TypeError as exc:
                    if "history" not in str(exc) and "performance" not in str(exc):
                        raise
                    if force is not None:
                        answer = app.run_query(contextual_query, route_query=route_query, force_intent=force)
                    else:
                        answer = app.run_query(contextual_query, route_query=route_query)
                intent = app.LAST_CITATION_AUDIT.get("crag_report", {}).get("intent") or force or app.classify_query(route_query)
        except Exception as exc:
            return 500, {"error": f"服务异常: {html.escape(str(exc))}"}

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        evidence = getattr(app, "LAST_EVIDENCE", [])
        citation_audit = getattr(app, "LAST_CITATION_AUDIT", {})
        topic_info = getattr(app, "LAST_TOPIC_INFO", {})
        crag_report = getattr(app, "LAST_CRAG_REPORT", {})
        timing = getattr(app, "LAST_TIMING", {})
        metrics = self._build_ask_metrics(
            query=query, intent=intent, history=history, answer=answer,
            evidence=evidence, citation_audit=citation_audit, elapsed_ms=elapsed_ms,
            topic_info=topic_info, crag_report=crag_report,
        )
        self._append_metrics_log(metrics)
        try:
            print(json.dumps(metrics, ensure_ascii=False), file=sys.stderr)
        except BrokenPipeError:
            pass
        except UnicodeEncodeError:
            try:
                print(json.dumps(metrics, ensure_ascii=True), file=sys.stderr)
            except BrokenPipeError:
                pass
        payload = web_support.build_ask_response(
                intent, answer, evidence, citation_audit, topic_info,
                crag_report, elapsed_ms, history, MAX_HISTORY_TURNS,
                mode=performance, timing=timing,
        )
        # Answer class: local_lookup / local_view / refusal / llm /
        # out_of_domain / ambiguous_locator / trace_only.
        payload["path"] = getattr(app, "LAST_ANSWER_PATH", "")
        # 失败案例自动采集（拒答/书目未收录/引文未确认/低 CRAG/引文审计问题）。
        miss_log.detect_misses(
            query=query,
            intent=payload.get("intent") or "",
            mode=performance,
            path=payload.get("path") or "",
            answer=answer or "",
            evidence=payload.get("evidence") or [],
            citation_audit=payload.get("citation_audit") or {},
            crag_report=payload.get("crag") or {},
            elapsed_ms=elapsed_ms,
            history_turns=len(history or []),
        )
        return 200, payload

    def _handle_ask_json(self, data):
        status, payload = self._run_ask_payload(data)
        self._send_json(status, payload)

    def _send_sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

    def _write_sse(self, event, payload):
        message = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        self.wfile.write(message.encode("utf-8"))
        self.wfile.flush()

    def _handle_ask_stream(self, data):
        self._send_sse_headers()

        def emit(event, payload):
            try:
                self._write_sse(event, payload)
            except (BrokenPipeError, ConnectionResetError):
                raise

        try:
            status, payload = self._run_ask_payload(data, emit=emit)
            if status == 200:
                emit("final", payload)
            else:
                emit("error", payload)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            try:
                emit("error", {"error": f"服务异常: {html.escape(str(exc))}"})
            except (BrokenPipeError, ConnectionResetError):
                return

    def log_message(self, fmt, *args):
        return


def main():
    app.load_vectorstore()
    if app.paragraph_vectorstore_exists():
        app.load_paragraph_vectorstore()
    if (
        app.hybrid_retrieval_enabled()
        and os.getenv("MARXOS_WARM_SPARSE_INDEX", "1").lower() in {"1", "true", "yes", "on"}
    ):
        threading.Thread(target=app.warm_sparse_index, daemon=True).start()
    write_runtime_manifest(
        os.getenv("MARXOS_RUNTIME_MANIFEST", "logs/runtime_manifest.json"),
        SETTINGS,
        app.RUNTIME,
    )
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
