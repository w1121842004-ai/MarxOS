import json
import tempfile
import unittest
from pathlib import Path

from scripts import run_web_smoke_ci


class WebSmokeRunnerTests(unittest.TestCase):
    def test_committed_dataset_has_version_and_twenty_unique_cases(self):
        dataset = run_web_smoke_ci.load_dataset(run_web_smoke_ci.DEFAULT_DATASET)

        self.assertEqual(dataset["schema_version"], 1)
        self.assertGreaterEqual(len(dataset["cases"]), 20)
        labels = [case["label"] for case in dataset["cases"]]
        self.assertEqual(len(labels), len(set(labels)))

    def test_offline_runner_validates_every_case_without_external_services(self):
        dataset = {
            "schema_version": 1,
            "cases": [
                {"label": "concept", "query": "什么是剩余价值？", "mode": "fast"},
                {"label": "deep", "query": "系统分析剩余价值理论。", "mode": "deep"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report = run_web_smoke_ci.run_dataset(dataset, report_path=report_path)

            self.assertEqual(report["summary"], {"total": 2, "passed": 2, "failed": 0})
            self.assertTrue(report_path.exists())
            persisted = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(all(item["ok"] for item in persisted["results"]))
            self.assertEqual(persisted["results"][1]["response_mode"], "deep")

    def test_invalid_case_is_reported_as_failure(self):
        dataset = {
            "schema_version": 1,
            "cases": [{"label": "empty", "query": "", "mode": "fast"}],
        }

        report = run_web_smoke_ci.run_dataset(dataset)

        self.assertEqual(report["summary"]["failed"], 1)
        self.assertEqual(report["results"][0]["status"], 400)


if __name__ == "__main__":
    unittest.main()
