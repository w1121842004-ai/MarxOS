import json
import tempfile
import unittest
from pathlib import Path

from scripts.rebuild_corpus_v2 import build_artifact_manifest, build_page_records, preflight_report


class RebuildV2Tests(unittest.TestCase):
    def test_manifest_hashes_outputs_and_records_parent_build(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "artifact"
            artifact.mkdir()
            records = artifact / "records.jsonl"
            records.write_text('{"id":1}\n{"id":2}\n', encoding="utf-8")

            manifest = build_artifact_manifest(
                artifact,
                {"records": records},
                build_id="build-test",
                parent_build_id="stable-v1",
            )

        self.assertEqual(manifest["build_id"], "build-test")
        self.assertEqual(manifest["parent_build_id"], "stable-v1")
        self.assertEqual(manifest["artifacts"]["records"]["row_count"], 2)
        self.assertEqual(len(manifest["artifacts"]["records"]["sha256"]), 64)
    def test_page_records_snapshot_preserves_raw_and_normalized_hashes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "pages/book"
            source_dir.mkdir(parents=True)
            (source_dir / "page_1.json").write_text(
                json.dumps({
                    "raw_text": " 原文 ", "cleaned_text": "正文", "page_type": "body",
                    "text_source": "pdf_text_layer", "reasons": ["trim"],
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            output = root / "page_records.jsonl"

            summary = build_page_records(root / "pages", ["book.pdf"], output)
            record = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(summary["records"], 1)
        self.assertEqual(record["record_version"], "page-record/v2")
        self.assertEqual(record["page_id"], "book.pdf#pdf1")
        self.assertEqual(record["raw_text"], " 原文 ")
        self.assertEqual(record["normalized_text"], "正文")
        self.assertNotEqual(record["raw_text_sha256"], record["normalized_text_sha256"])
    def test_preflight_accepts_complete_inputs_and_new_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page_cache = root / "pages"
            (page_cache / "book").mkdir(parents=True)
            (page_cache / "book/page_1.json").write_text("{}", encoding="utf-8")
            for name in ("article.json", "work.json", "page.json"):
                (root / name).write_text("{}", encoding="utf-8")
            config = self._config(root)

            report = preflight_report(config, root=root, free_bytes=20 * 1024**3)

        self.assertTrue(report["ready"])
        self.assertEqual(report["summary"]["errors"], 0)

    def test_preflight_blocks_existing_outputs_and_missing_source_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pages").mkdir()
            for name in ("article.json", "work.json", "page.json"):
                (root / name).write_text("{}", encoding="utf-8")
            (root / "artifacts").mkdir()
            config = self._config(root)

            report = preflight_report(config, root=root, free_bytes=20 * 1024**3)

        self.assertFalse(report["ready"])
        codes = [issue["code"] for issue in report["issues"]]
        self.assertIn("SOURCE_CACHE_MISSING", codes)
        self.assertIn("OUTPUT_EXISTS", codes)

    @staticmethod
    def _config(root: Path) -> dict:
        return {
            "schema_version": "marxos-rebuild/v2",
            "scope": {"sources": ["book.pdf"]},
            "inputs": {
                "page_cache": "pages",
                "article_map": "article.json",
                "work_catalog": "work.json",
                "page_map": "page.json",
            },
            "outputs": {
                "artifact_dir": "artifacts",
                "milvus_uri": "index.db",
            },
            "contracts": {"overwrite_existing": False},
        }


if __name__ == "__main__":
    unittest.main()
