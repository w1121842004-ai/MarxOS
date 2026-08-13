import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DocumentAuditCliTests(unittest.TestCase):
    def test_fixture_report_is_machine_readable_and_fails_policy(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/audit_document_pipeline.py"),
                "--input",
                str(ROOT / "tests/fixtures/document_pipeline_v1/records.jsonl"),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertEqual(report["input"]["kind"], "jsonl")
        self.assertEqual(report["summary"]["records"], 9)
        self.assertEqual(
            report["summary"]["issues_by_code"],
            {
                "ARTICLE_MISSING": 1,
                "DUPLICATE_PAGE_TEXT": 1,
                "EMPTY_RETRIEVAL_TEXT": 1,
                "FOOTNOTE_ORPHAN": 1,
                "FRONT_MATTER_LEAK": 2,
                "MOJIBAKE_REMAINS": 1,
                "PAGE_RANGE_REVERSED": 1,
            },
        )
        self.assertIn("blocking error", report["summary"]["readable"])

    def test_malformed_jsonl_returns_invalid_input_exit(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/audit_document_pipeline.py"), "--input", "-"],
            cwd=ROOT,
            input="not-json\n",
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid_jsonl", json.loads(result.stdout)["error"])

    def test_summary_only_is_concise_and_production_readable(self):
        result = subprocess.run(
            [
                sys.executable, str(ROOT / "scripts/audit_document_pipeline.py"),
                "--input", str(ROOT / "tests/fixtures/document_pipeline_v1/records.jsonl"),
                "--summary-only",
            ],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )

        self.assertEqual(result.returncode, 1)
        summary = json.loads(result.stdout)
        self.assertIn("readable", summary)
        self.assertIn("issues_by_severity", summary)
        self.assertIsInstance(summary["issues"], int)


if __name__ == "__main__":
    unittest.main()
