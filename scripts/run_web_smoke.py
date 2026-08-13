from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from http.client import HTTPConnection
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app
import web_app


@dataclass
class SmokeCase:
    query: str
    mode: str
    label: str


CASES = [
    SmokeCase("什么是剩余价值？", "fast", "fast_concept"),
    SmokeCase("宗教是人民的鸦片出自哪里？", "fast", "fast_quote"),
    SmokeCase("马克思如何理解阶级斗争？", "standard", "standard_theory"),
    SmokeCase("如何理解资本主义经济危机？", "standard", "standard_crisis"),
    SmokeCase("请系统分析剩余价值理论的基本结构及其在马克思主义政治经济学中的地位。", "deep", "deep_surplus_value"),
]


def start_server() -> tuple[web_app.ThreadingHTTPServer, int, threading.Thread]:
    app.load_vectorstore()
    if app.paragraph_vectorstore_exists():
        app.load_paragraph_vectorstore()
    server = web_app.ThreadingHTTPServer(("127.0.0.1", 0), web_app.MarxOSHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1], thread


def ask(port: int, case: SmokeCase, timeout: int = 180) -> tuple[int, dict, int]:
    started = time.perf_counter()
    conn = HTTPConnection("127.0.0.1", port, timeout=timeout)
    body = json.dumps(
        {"query": case.query, "mode": case.mode, "history": []},
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
    parser = argparse.ArgumentParser(description="Run a short real web API smoke test.")
    parser.add_argument("--report", default=str(ROOT / "logs" / "web_smoke_latest.json"))
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--only-label", default="")
    args = parser.parse_args()
    cases = [case for case in CASES if not args.only_label or case.label == args.only_label]

    server, port, thread = start_server()
    results = []
    try:
        for case in cases:
            try:
                status, payload, wall_ms = ask(port, case, timeout=args.timeout)
            except Exception as exc:
                status, payload, wall_ms = 599, {"error": type(exc).__name__ + ": " + str(exc)}, args.timeout * 1000
            result = {
                "label": case.label,
                "query": case.query,
                "requested_mode": case.mode,
                "status": status,
                "response_mode": payload.get("mode"),
                "intent": payload.get("intent"),
                "elapsed_ms": payload.get("elapsed_ms"),
                "wall_ms": wall_ms,
                "answer_len": len(payload.get("answer") or ""),
                "evidence_count": len(payload.get("evidence") or []),
                "crag_path": (payload.get("crag") or {}).get("path"),
                "timing": payload.get("timing") or {},
                "error": payload.get("error"),
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "summary": {
                    "turns": len(results),
                    "ok_turns": sum(1 for item in results if item["status"] == 200 and not item["error"]),
                    "timeout_turns": sum(1 for item in results if item["status"] == 599),
                    "avg_elapsed_ms": int(
                        sum(item.get("elapsed_ms") or 0 for item in results) / max(len(results), 1)
                    ),
                    "avg_wall_ms": int(
                        sum(item.get("wall_ms") or 0 for item in results) / max(len(results), 1)
                    ),
                    "max_elapsed_ms": max((item.get("elapsed_ms") or 0 for item in results), default=0),
                    "max_wall_ms": max((item.get("wall_ms") or 0 for item in results), default=0),
                },
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 1 if any(item["status"] != 200 for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
