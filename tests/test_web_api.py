import json
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch

import web_app


class WebApiTests(unittest.TestCase):
    def test_health_and_readiness_endpoints(self):
        with patch.object(web_app, "readiness_report", return_value={"ready": False, "checks": {}}):
            server = web_app.ThreadingHTTPServer(("127.0.0.1", 0), web_app.MarxOSHandler)
            host, port = server.server_address
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                health_conn = HTTPConnection(host, port, timeout=5)
                health_conn.request("GET", "/healthz")
                health_res = health_conn.getresponse()
                health_data = json.loads(health_res.read().decode("utf-8"))
                health_conn.close()

                ready_conn = HTTPConnection(host, port, timeout=5)
                ready_conn.request("GET", "/readyz")
                ready_res = ready_conn.getresponse()
                ready_data = json.loads(ready_res.read().decode("utf-8"))
                ready_conn.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

        self.assertEqual(health_res.status, 200)
        self.assertEqual(health_data, {"status": "ok"})
        self.assertEqual(ready_res.status, 503)
        self.assertFalse(ready_data["ready"])

    def test_explicit_work_page_query_does_not_reuse_previous_evidence(self):
        history = [
            {
                "role": "bot",
                "text": "上一条回答",
                "intent": "rag_answer",
                "evidence": [
                    {
                        "citation": "《马克思恩格斯选集》第1卷，第33页",
                        "printed_page": 33,
                    }
                ],
            }
        ]

        with patch.object(web_app.app, "run_query", return_value="新检索结果") as run_query:
            handler = web_app.MarxOSHandler.__new__(web_app.MarxOSHandler)
            status, payload = handler._run_ask_payload(
                {
                    "query": "哥达纲领批判在马恩选集的哪页",
                    "history": history,
                    "mode": "auto",
                }
            )

        self.assertEqual(status, 200)
        self.assertEqual(payload["answer"], "新检索结果")
        run_query.assert_called_once()

    def test_api_ask_returns_evidence_and_writes_metrics_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "api_ask_metrics.jsonl"

            def fake_run_query(_contextual_query, route_query=None):
                web_app.app.set_last_topic_info(
                    {
                        "topic_id": "peasant_cooperative",
                        "topic_label": "农民问题与土地问题",
                        "topic_section": "科学社会主义",
                    }
                )
                web_app.app.set_last_crag_report(
                    {
                        "path": "corrective",
                        "score": 63,
                        "threshold": 52,
                        "ok": True,
                        "issues": ["locator_only"],
                    }
                )
                web_app.app.set_last_evidence(
                    [
                        {
                            "id": "E1",
                            "citation": "《马克思恩格斯选集》第1卷，第401页",
                            "source": "mes01.pdf",
                            "printed_page": 401,
                            "excerpt": "候选证据",
                        }
                    ],
                    {"ok": False, "issues": [{"type": "citation_not_in_evidence"}], "evidence_count": 1},
                )
                return "示例答案\n\n*引用注释*\n1. 《马克思恩格斯选集》第9卷，第999页。"

            with (
                patch.object(web_app, "METRICS_LOG_PATH", metrics_path),
                patch.object(web_app.MarxOSHandler, "_answer_citation_followup", return_value=None),
                patch.object(web_app.app, "classify_query", return_value="rag_answer"),
                patch.object(web_app.app, "run_query", side_effect=fake_run_query),
            ):
                server = web_app.ThreadingHTTPServer(("127.0.0.1", 0), web_app.MarxOSHandler)
                host, port = server.server_address
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    conn = HTTPConnection(host, port, timeout=5)
                    payload = {"query": "这句话出自哪里", "history": []}
                    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    conn.request("POST", "/api/ask", body=body, headers={"Content-Type": "application/json"})
                    res = conn.getresponse()
                    data = json.loads(res.read().decode("utf-8"))
                    conn.close()
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)

            self.assertEqual(res.status, 200)
            self.assertEqual(data.get("intent"), "rag_answer")
            self.assertEqual(data.get("topic", {}).get("topic_id"), "peasant_cooperative")
            self.assertEqual(data.get("crag", {}).get("path"), "corrective")
            self.assertTrue(data.get("evidence"))
            self.assertEqual(data["evidence"][0].get("source"), "mes01.pdf")

            self.assertTrue(metrics_path.exists())
            lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(lines)
            metrics = json.loads(lines[-1])
            self.assertEqual(metrics.get("event"), "api_ask")
            self.assertEqual(metrics.get("topic_id"), "peasant_cooperative")
            self.assertEqual(metrics.get("topic_label"), "农民问题与土地问题")
            self.assertEqual(metrics.get("evidence_count"), 1)
            self.assertEqual(metrics.get("citation_lines_count"), 1)
            self.assertEqual(metrics.get("matched_count"), 0)
            self.assertTrue(metrics.get("fallback_used"))
            self.assertEqual(metrics.get("crag_path"), "corrective")
            self.assertEqual(metrics.get("crag_score"), 63)

    def _post_and_collect_stream(self, host, port, payload, timeout=10):
        """POST /api/ask_stream and collect SSE events as (event, data) pairs.

        The SSE connection stays open by design (EventSource semantics), so
        read incrementally and stop once a terminal event (final/error) arrives.
        """
        import time as _time

        conn = HTTPConnection(host, port, timeout=timeout)
        events = []
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            conn.request(
                "POST",
                "/api/ask_stream",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            res = conn.getresponse()
            reader = res.fp.read1 if hasattr(res.fp, "read1") else res.fp.read
            buf = ""
            deadline = _time.perf_counter() + timeout
            while _time.perf_counter() < deadline:
                chunk = reader(4096)
                if not chunk:
                    break
                buf += chunk.decode("utf-8", "replace")
                while "\n\n" in buf:
                    part, buf = buf.split("\n\n", 1)
                    event, data_line = "", ""
                    for line in part.split("\n"):
                        if line.startswith("event:"):
                            event = line[len("event:"):].strip()
                        elif line.startswith("data:"):
                            data_line += line[len("data:"):].strip()
                    if not data_line:
                        continue
                    events.append((event, json.loads(data_line)))
                    if event in ("final", "error"):
                        return events
        finally:
            conn.close()
        return events

    def test_stream_and_json_payloads_have_same_fields(self):
        def fake_run_query(_contextual_query, route_query=None):
            web_app.app.set_last_evidence(
                [
                    {
                        "id": "E1",
                        "citation": "《马克思恩格斯选集》第1卷，第401页",
                        "detailed_citation": "《马克思恩格斯选集》第1卷，《关于费尔巴哈的提纲》，第401页",
                        "sentence_citation": "《马克思恩格斯选集》第1卷，第401页",
                        "source": "mes01.pdf",
                        "printed_page": 401,
                        "citation_page": 401,
                        "match_type": "vector_candidate",
                        "confidence": 0.0,
                        "excerpt": "候选证据",
                    }
                ],
                {"ok": True, "issues": [], "evidence_count": 1},
            )
            web_app.app.set_last_crag_report({"path": "initial", "score": 60, "threshold": 52, "ok": True})
            web_app.app.set_last_timing({"total": 1234, "mode": "deep", "intent": "concept_explain"})
            web_app.app.LAST_ANSWER_PATH = "llm"
            web_app.app.LAST_CITATION_AUDIT = {"crag_report": {"intent": "concept_explain"}}
            return "示例答案"

        with (
            patch.object(web_app.app, "run_query", side_effect=fake_run_query),
            patch.object(web_app.MarxOSHandler, "_answer_citation_followup", return_value=None),
        ):
            server = web_app.ThreadingHTTPServer(("127.0.0.1", 0), web_app.MarxOSHandler)
            host, port = server.server_address
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request_payload = {"query": "什么是剩余价值？", "history": [], "mode": "auto"}
                conn = HTTPConnection(host, port, timeout=10)
                body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
                conn.request("POST", "/api/ask", body=body, headers={"Content-Type": "application/json"})
                json_res = conn.getresponse()
                json_data = json.loads(json_res.read().decode("utf-8"))
                conn.close()

                events = self._post_and_collect_stream(host, port, request_payload)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

        final_events = [data for event, data in events if event == "final"]
        self.assertEqual(len(final_events), 1, f"events: {[e for e, _ in events]}")
        final_data = final_events[0]
        self.assertEqual(set(json_data.keys()), set(final_data.keys()))
        self.assertEqual(set(json_data["evidence"][0].keys()), set(final_data["evidence"][0].keys()))
        self.assertEqual(final_data["path"], "llm")

    def test_error_payloads_have_consistent_shape(self):
        server = web_app.ThreadingHTTPServer(("127.0.0.1", 0), web_app.MarxOSHandler)
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = HTTPConnection(host, port, timeout=5)
            body = json.dumps({"query": "", "history": [], "mode": "auto"}).encode("utf-8")
            conn.request("POST", "/api/ask", body=body, headers={"Content-Type": "application/json"})
            json_res = conn.getresponse()
            json_error = json.loads(json_res.read().decode("utf-8"))
            conn.close()

            events = self._post_and_collect_stream(host, port, {"query": "", "history": [], "mode": "auto"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(json_res.status, 400)
        error_events = [data for event, data in events if event == "error"]
        self.assertEqual(len(error_events), 1, f"events: {[e for e, _ in events]}")
        self.assertEqual(set(json_error.keys()), set(error_events[0].keys()))
        self.assertEqual(json_error, error_events[0])


if __name__ == "__main__":
    unittest.main()
