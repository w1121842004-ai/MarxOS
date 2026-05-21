import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app
import web_app


API_URL = os.getenv("MARXOS_REGRESSION_API", "http://127.0.0.1:7860/api/ask")
REPORT_PATH = Path("logs/topic_conversation_regression.json")
MODE = os.getenv("MARXOS_REGRESSION_MODE", "http")

PROMPTS = [
    "请列出十段马克思关于农民合作社的观点",
    "把第1条和第7条改写成更通顺的学术表述",
    "哪几条最直接谈到合作社",
    "把直接涉及合作社的原文摘录三段",
    "第2条具体讲的是什么政策安排",
    "再列出三条和小农过渡有关的观点",
    "第5条里的小农定义再解释一下",
    "把第3条和第10条合并成一条完整判断",
    "如果按工农关系来归纳，怎么重排这十条",
    "这些观点主要集中在哪一篇作品里",
    "再补两条和土地所有制有关的观点",
    "哪一条最接近共同耕种的表述",
    "把涉及合作社生产的两条原文完整抄出来",
    "上一条引用的出处分别是什么",
    "德国农民战争里和合作社最相关的是哪一段",
    "把法德农民问题中的核心主张压缩成五点",
    "哪些条目是在讲小农，哪些是在讲大地产",
    "把第2条和第7条对应的页码单独列出来",
    "有没有明确说要用暴力剥夺小农",
    "把这个问题改成一个150字的小结",
    "再按政策主张、过渡方式、阶级区分三类整理",
    "其中哪些观点最适合拿来回答今天的农村合作问题",
    "如果只保留最关键的三条，你选哪三条",
    "把最关键三条换成更接近原文的表述",
    "上面三条分别出自哪一页",
    "请再给出两条关于大地产和农村无产者的观点",
    "把所有直接证据按页码升序列出来",
    "哪一段最能说明合作社是过渡方式而不是强制剥夺",
    "把这段原文再展开解释一下",
    "最后用一句话概括马克思和恩格斯在这个问题上的总体态度",
]


def call_api(query, history):
    if MODE == "local":
        direct_answer = web_app.MarxOSHandler._answer_history_followup(query, history)
        if direct_answer:
            return {
                "intent": "citation_followup",
                "answer": direct_answer,
                "evidence": app.LAST_EVIDENCE,
                "citation_audit": app.LAST_CITATION_AUDIT,
                "topic": app.LAST_TOPIC_INFO,
                "elapsed_ms": 0,
            }

        route_query = web_app.MarxOSHandler._topic_scoped_query(query, history)
        contextual_query = web_app.MarxOSHandler._build_contextual_query(route_query, history)
        intent = "rag_answer" if web_app.MarxOSHandler._is_contextual_followup(query) else app.classify_query(route_query)
        answer = app.run_query(contextual_query, route_query=route_query)
        return {
            "intent": intent,
            "answer": answer,
            "evidence": app.LAST_EVIDENCE,
            "citation_audit": app.LAST_CITATION_AUDIT,
            "topic": app.LAST_TOPIC_INFO,
            "elapsed_ms": 0,
        }

    payload = json.dumps({"query": query, "history": history}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def is_failure(prompt, data):
    answer = data.get("answer") or ""
    bad_markers = [
        "当前材料不足",
        "未包含",
        "未列入当前检索范围",
        "无法依据现有材料",
        "没有直接涉及",
    ]
    if any(marker in answer for marker in bad_markers):
        return True, "fallback_answer"
    if "农民合作社" in prompt and not (data.get("topic") or {}).get("topic_id"):
        return True, "missing_topic"
    if not answer.strip():
        return True, "empty_answer"
    return False, ""


def main():
    history = []
    results = []
    failures = []

    for index, prompt in enumerate(PROMPTS, start=1):
        try:
            data = call_api(prompt, history)
        except Exception as exc:  # noqa: BLE001
            failures.append({"turn": index, "prompt": prompt, "reason": f"request_error: {exc}"})
            break

        failed, reason = is_failure(prompt, data)
        record = {
            "turn": index,
            "prompt": prompt,
            "intent": data.get("intent"),
            "topic": data.get("topic") or {},
            "elapsed_ms": data.get("elapsed_ms"),
            "failed": failed,
            "reason": reason,
            "answer_preview": (data.get("answer") or "")[:400],
        }
        results.append(record)
        if failed:
            failures.append(record)

        history.append({"role": "user", "text": prompt})
        history.append(
            {
                "role": "bot",
                "text": data.get("answer") or "",
                "evidence": data.get("evidence") or [],
                "topic": data.get("topic") or {},
            }
        )

    report = {
        "total_turns": len(results),
        "failure_count": len(failures),
        "failures": failures,
        "results": results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"total_turns": len(results), "failure_count": len(failures), "report": str(REPORT_PATH)}, ensure_ascii=False))
    if failures:
        print(json.dumps(failures[:10], ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
