from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_rebuild_v2.py"


def _record(record_id: str, source: str, length: int, **overrides):
    record = {
        "record_id": record_id,
        "paragraph_id": record_id,
        "source": source,
        "page_type": "body",
        "retrievable": True,
        "cross_page": False,
        "paragraph_text": (record_id + "正文") * max(1, (length // (len(record_id) + 2)) + 1),
    }
    record["paragraph_text"] = record["paragraph_text"][:length]
    return {**record, **overrides}


class ProbeRebuildV2Tests(unittest.TestCase):
    def setUp(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("probe_rebuild_v2", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        assert spec.loader
        spec.loader.exec_module(self.module)

    def sample_records(self):
        return [
            _record("a-short", "a.pdf", 40),
            _record("a-medium", "a.pdf", 240),
            _record("a-long", "a.pdf", 900),
            _record("a-cross", "a.pdf", 300, cross_page=True),
            _record("b-short", "b.pdf", 55),
            _record("b-medium", "b.pdf", 260),
            _record("b-long", "b.pdf", 950),
            _record("b-cross", "b.pdf", 320, page_span=[2, 3]),
            _record("excluded", "c.pdf", 500, retrievable=False),
            _record("notes", "c.pdf", 500, page_type="notes"),
        ]

    def test_selector_is_deterministic_balanced_and_excludes_non_body(self):
        first = self.module.select_probe_records(self.sample_records(), sample_size=6, seed=23)
        second = self.module.select_probe_records(list(reversed(self.sample_records())), sample_size=6, seed=23)

        self.assertEqual([row["record_id"] for row in first], [row["record_id"] for row in second])
        self.assertEqual(6, len(first))
        self.assertEqual({"a.pdf", "b.pdf"}, {row["source"] for row in first})
        self.assertTrue(any(self.module.is_cross_page(row) for row in first))
        buckets = {self.module.length_bucket(row["paragraph_text"]) for row in first}
        self.assertEqual({"short", "medium", "long"}, buckets)
        self.assertNotIn("excluded", {row["record_id"] for row in first})
        self.assertNotIn("notes", {row["record_id"] for row in first})

    def test_chunk_metrics_have_full_coverage_valid_offsets_and_expected_duplication(self):
        records = [_record("long", "a.pdf", 260)]
        report = self.module.evaluate_chunk_config(records, chunk_size=100, chunk_overlap=20)

        self.assertEqual("chunk-structure/v1", report["schema_version"])
        self.assertEqual(3, report["summary"]["chunk_count"])
        self.assertEqual(0, report["summary"]["orphan_offset_count"])
        self.assertEqual(1.0, report["summary"]["coverage_ratio"])
        self.assertEqual(300, report["summary"]["indexed_character_count"])
        self.assertEqual(40, report["summary"]["duplicated_character_count"])
        self.assertAlmostEqual(300 / 260, report["summary"]["storage_multiplier"])
        self.assertEqual([0, 80, 160], [chunk["char_start"] for chunk in report["chunks"]])
        self.assertEqual([100, 180, 260], [chunk["char_end"] for chunk in report["chunks"]])
        self.assertTrue(all(chunk["text_sha256"] for chunk in report["chunks"]))

    def test_invalid_config_and_missing_identity_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.module.evaluate_chunk_config([_record("x", "a.pdf", 20)], 10, 10)
        with self.assertRaisesRegex(ValueError, "stable identity"):
            self.module.select_probe_records([{
                "source": "a.pdf", "page_type": "body", "retrievable": True,
                "paragraph_text": "正文",
            }], sample_size=1, seed=1)

    def test_cli_outputs_versioned_retrieval_eval_contract_without_model_dependencies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "paragraphs.jsonl"
            input_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in self.sample_records()) + "\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable, str(SCRIPT), "--input", str(input_path),
                    "--sample-size", "6", "--seed", "23",
                    "--chunk", "100:20", "--chunk", "160:24",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(0, completed.returncode, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual("rebuild-probe/v1", report["schema_version"])
        self.assertEqual("retrieval-eval-candidates/v1", report["retrieval_eval"]["schema_version"])
        self.assertEqual(6, report["selection"]["selected_count"])
        self.assertEqual(2, len(report["chunk_configurations"]))
        self.assertNotIn("embedding", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
