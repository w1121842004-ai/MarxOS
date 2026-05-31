import json
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch

import web_app


class WebApiTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
