import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_bm25_chunk_configs_v2 import (
    ChunkConfig,
    evaluate_chunk_config,
    iter_chunks,
    load_evaluation_cases,
    load_retrievable_records,
)


class BM25ChunkConfigEvaluationTests(unittest.TestCase):
    def test_iter_chunks_preserves_lineage_and_uses_overlap(self):
        record = {
            "paragraph_id": "p1",
            "paragraph_text": "0123456789",
            "retrievable": True,
            "work_id": "work-a",
            "source": "book.pdf",
            "pdf_page_start": 12,
        }

        chunks = list(iter_chunks([record], ChunkConfig(6, 2)))

        self.assertEqual(["012345", "456789"], [chunk.text for chunk in chunks])
        self.assertEqual([0, 4], [chunk.char_start for chunk in chunks])
        self.assertEqual("p1", chunks[1].paragraph_id)
        self.assertEqual("work-a", chunks[1].work_id)
        self.assertEqual("book.pdf", chunks[1].source)
        self.assertEqual(12, chunks[1].page)

    def test_load_records_excludes_nonretrievable_and_invalid_rows(self):
        rows = [
            {"paragraph_id": "keep", "paragraph_text": "资本与劳动", "retrievable": True},
            {"paragraph_id": "drop", "paragraph_text": "目录", "retrievable": False},
            {"paragraph_id": "empty", "paragraph_text": "", "retrievable": True},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

            records = load_retrievable_records(path)

        self.assertEqual(["keep"], [record["paragraph_id"] for record in records])

    def test_loader_validates_evaluation_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.json"
            path.write_text(json.dumps([{"id": 1, "question": "资本"}]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected_work_id"):
                load_evaluation_cases(path)

    def test_metrics_include_recall_mrr_hard_negatives_and_coverage(self):
        records = [
            {
                "paragraph_id": "p1",
                "paragraph_text": "稀有术语 社会关系 总和",
                "retrievable": True,
                "work_id": "expected-a",
                "source": "a.pdf",
                "pdf_page_start": 1,
            },
            {
                "paragraph_id": "p2",
                "paragraph_text": "商品 资本 劳动",
                "retrievable": True,
                "work_id": "wrong-b",
                "source": "b.pdf",
                "pdf_page_start": 2,
            },
        ]
        cases = [
            {
                "id": 1,
                "question": "稀有术语 社会关系",
                "expected_work_id": "expected-a",
                "hard_negative": ["wrong-b"],
            },
            {
                "id": 2,
                "question": "不存在的问题词",
                "expected_work_id": "missing-work",
            },
        ]

        report = evaluate_chunk_config(records, cases, ChunkConfig(180, 40))

        self.assertEqual(2, report["evaluation"]["case_count"])
        self.assertEqual(1, report["evaluation"]["evaluable_case_count"])
        self.assertEqual(0.5, report["metrics"]["evaluable_coverage"])
        self.assertEqual(1.0, report["metrics"]["recall_at_1"])
        self.assertEqual(1.0, report["metrics"]["recall_at_5"])
        self.assertEqual(1.0, report["metrics"]["recall_at_8"])
        self.assertEqual(1.0, report["metrics"]["mrr_at_8"])
        self.assertEqual(0, report["metrics"]["hard_negative_hit_count_at_8"])
        self.assertEqual(2, report["corpus"]["retrievable_paragraph_count"])
        self.assertEqual(2, report["corpus"]["chunk_count"])

    def test_invalid_chunk_configuration_fails_fast(self):
        with self.assertRaises(ValueError):
            ChunkConfig(40, 40)


if __name__ == "__main__":
    unittest.main()
