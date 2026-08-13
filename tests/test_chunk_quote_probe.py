import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_chunk_probe_quotes import evaluate_quote_probe, normalize_quote_text


class ChunkQuoteProbeTests(unittest.TestCase):
    def setUp(self):
        self.probe = {
            "schema_version": "rebuild-probe/v1",
            "chunk_configurations": [
                {
                    "config_id": "chars-180-overlap-40",
                    "parameters": {"chunk_size": 180, "chunk_overlap": 40},
                    "chunks": [
                        {
                            "chunk_id": "p1#c0000",
                            "source": "me01.pdf",
                            "citation_page_start": 12,
                            "citation_page_end": 12,
                            "text": "马克思指出：生产力，决定生产关系。",
                        },
                        {
                            "chunk_id": "p2#c0000",
                            "source": "me02.pdf",
                            "page_span": [20, 21],
                            "text": "这是一个只能覆盖目标引文开头的较短候选。",
                        },
                    ],
                },
                {
                    "config_id": "chars-320-overlap-64",
                    "parameters": {"chunk_size": 320, "chunk_overlap": 64},
                    "chunks": [
                        {
                            "chunk_id": "p1#c0000",
                            "source": "me01.pdf",
                            "citation_page_start": 12,
                            "citation_page_end": 12,
                            "text": "生产力决定生产关系",
                        },
                        {
                            "chunk_id": "p2#c0000",
                            "source": "me02.pdf",
                            "page_span": [20, 21],
                            "text": "这是一个只能覆盖目标引文开头，引文后半部分也在这里。",
                        },
                    ],
                },
            ],
        }
        self.dataset = [
            {
                "id": "q1",
                "expected_citations": [
                    {"source": "ME01.PDF", "citation_page": 12, "quote": "生产力决定生产关系"}
                ],
            },
            {
                "id": "q2",
                "expected_citations": [
                    {
                        "source": "me02.pdf",
                        "citation_page": 20,
                        "quote": "只能覆盖目标引文开头引文后半部分也在这里",
                    }
                ],
            },
            {
                "id": "q3",
                "expected_citations": [
                    {"source": "missing.pdf", "citation_page": 99, "quote": "不存在的引文"}
                ],
            },
        ]

    def test_normalization_removes_spacing_and_punctuation(self):
        self.assertEqual(normalize_quote_text("生产力， 决定\n生产关系。"), "生产力决定生产关系")

    def test_reports_exact_partial_and_unevaluable_cases_per_config(self):
        report = evaluate_quote_probe(self.probe, self.dataset)
        by_id = {row["config_id"]: row for row in report["configurations"]}

        small = by_id["chars-180-overlap-40"]
        self.assertEqual(small["summary"]["citation_count"], 3)
        self.assertEqual(small["summary"]["evaluable_cases"], 2)
        self.assertEqual(small["summary"]["exact_quote_containment_count"], 1)
        self.assertEqual(small["summary"]["unevaluable_cases"], 1)
        self.assertGreater(small["summary"]["mean_partial_coverage"], 0)
        self.assertLess(small["summary"]["mean_partial_coverage"], 1)
        self.assertEqual(small["failures"][0]["case_id"], "q2")

        large = by_id["chars-320-overlap-64"]
        self.assertEqual(large["summary"]["exact_quote_containment_count"], 2)
        self.assertEqual(large["summary"]["exact_quote_containment_rate"], 1.0)

    def test_cli_writes_machine_readable_report(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            probe_path = directory / "probe.json"
            eval_path = directory / "eval.json"
            output_path = directory / "report.json"
            probe_path.write_text(json.dumps(self.probe, ensure_ascii=False), encoding="utf-8")
            eval_path.write_text(json.dumps(self.dataset, ensure_ascii=False), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_chunk_probe_quotes.py"),
                    "--chunk-probe",
                    str(probe_path),
                    "--eval-dataset",
                    str(eval_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "chunk-quote-eval/v1")
            self.assertEqual(len(payload["configurations"]), 2)


if __name__ == "__main__":
    unittest.main()
