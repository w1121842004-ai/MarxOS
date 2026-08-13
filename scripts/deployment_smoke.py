#!/usr/bin/env python3
"""Deployment smoke: start the service and run the four required probe cases.

Cases (per task.md P5):
  1. health check (/healthz + /readyz)
  2. one bibliographic question
  3. one quote-lookup question
  4. one concept question
  5. one Web follow-up question (two turns with history evidence)

Exit 0 only when every case passes. Requires exclusive access to the Milvus
Lite DB (stop any running web_app.py first).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.client import HTTPConnection
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402
import web_app  # noqa: E402

REPORT_VERSION = "deployment-smoke/v1"

CASES = [
    {
        "label": "bibliographic",
        "query": "《共产党宣言》收录在哪一卷？",
        "mode": "precise",
        "must_contain": ("第", "卷"),
        "expect_evidence": False,
    },
    {
        "label": "quote",
        "query": "“全世界无产者，联合起来！”出自哪里？",
        "mode": "fast",
        "must_contain": ("共产党宣言",),
        "expect_evidence": False,
    },
    {
        "label": "concept",
        "query": "什么是剩余价值？",
        "mode": "standard",
        "must_contain": ("剩余价值",),
        "expect_evidence": True,
    },
]


def start_server() -> tuple[web_app.ThreadingHTTPServer, int, threading.Thread]:
    app.load_vectorstore()
    if app.paragraph_vectorstore_exists():
        app.load_paragraph_vectorstore()
    server = web_app.ThreadingHTTPServer(("127.0.0.1", 0), web_app.MarxOSHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1], thread


def get(port: int, path: str, timeout: int = 10) -> tuple[int, dict]:
    conn = HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        status = response.status
    finally:
        conn.close()
    return status, payload


def ask(port: int, query: str, mode: str, history: list | None = None, timeout: int = 180) -> tuple[int, dict, int]:
    started = time.perf_counter()
    conn = HTTPConnection("127.0.0.1", port, timeout=timeout)
    body = json.dumps(
        {"query": query, "mode": mode, "history": history or []},
        ensure_ascii=False,
    ).encode("utf-8")
    conn.request("POST", "/api/ask", body=body, headers={"Content-Type": "application/json"})
    try:
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        status = response.status
    finally:
        conn.close()
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return status, payload, elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(ROOT / "logs" / "deployment_smoke.json"))
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    results = []
    server, port, thread = start_server()
    try:
        # 1. health check
        health_status, health = get(port, "/healthz", timeout=5)
        results.append({
            "label": "healthz",
            "status": health_status,
            "payload": health,
            "elapsed_ms": 0,
            "error": None,
        })
        ready_status, ready = get(port, "/readyz", timeout=10)
        results.append({
            "label": "readyz",
            "status": ready_status,
            "payload": ready,
            "elapsed_ms": 0,
            "error": None,
        })

        # 2-4. bibliographic / quote / concept
        followup_history = None
        for case in CASES:
            status, payload, elapsed_ms = ask(port, case["query"], case["mode"], timeout=args.timeout)
            answer = payload.get("answer") or ""
            error = payload.get("error")
            results.append({
                "label": case["label"],
                "query": case["query"],
                "status": status,
                "answer_len": len(answer),
                "evidence_count": len(payload.get("evidence") or []),
                "path": payload.get("path"),
                "elapsed_ms": elapsed_ms,
                "error": error,
            })
            if case["label"] == "concept" and status == 200:
                followup_history = [
                    {
                        "role": "bot",
                        "text": answer,
                        "intent": payload.get("intent") or "",
                        "evidence": payload.get("evidence") or [],
                    }
                ]

        # 5. Web follow-up (two turns, uses the concept turn's evidence)
        follow_query = "上一条回答的证据在选集第几页？"
        status, payload, elapsed_ms = ask(port, follow_query, "auto", history=followup_history, timeout=args.timeout)
        answer = payload.get("answer") or ""
        results.append({
            "label": "followup",
            "query": follow_query,
            "status": status,
            "answer_len": len(answer),
            "evidence_count": len(payload.get("evidence") or []),
            "path": payload.get("path"),
            "intent": payload.get("intent"),
            "elapsed_ms": elapsed_ms,
            "error": payload.get("error"),
        })
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    failures = []
    health_ok = results[0]["status"] == 200 and results[0]["payload"].get("status") == "ok"
    if not health_ok:
        failures.append("healthz")
    ready_ok = results[1]["status"] == 200 and results[1]["payload"].get("ready") is True
    if not ready_ok:
        failures.append("readyz")
    for index, case in enumerate(CASES):
        result = results[2 + index]
        if result["status"] != 200 or result["error"] or not result["answer_len"]:
            failures.append(case["label"])
            continue
        if case["expect_evidence"] and not result["evidence_count"]:
            failures.append(f"{case['label']}:no_evidence")
    followup = results[-1]
    if followup["status"] != 200 or followup["error"] or not followup["answer_len"]:
        failures.append("followup")

    report = {
        "schema_version": REPORT_VERSION,
        "summary": {
            "cases": len(results),
            "passed": len(results) - len(set(failures)),
            "failed": len(set(failures)),
            "failures": sorted(set(failures)),
        },
        "results": results,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(
            f"[{result['label']}] status={result['status']} "
            f"answer_len={result.get('answer_len', '-')} err={result.get('error')}",
            flush=True,
        )
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
