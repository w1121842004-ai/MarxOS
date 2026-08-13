from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from marxos.data.retrieval_records_v2 import (
    build_semantic_child_records,
    stable_retrieval_id,
)


def _parent(**overrides):
    record = {
        "document_record_version": "document-record/v2",
        "retrieval_unit": "paragraph",
        "paragraph_id": "par_abc123",
        "parent_paragraph_id": "par_abc123",
        "paragraph_text": "天地玄黄宇宙洪荒日月盈昃辰宿列张",
        "source": "mes01.pdf",
        "book": "马克思恩格斯选集 第1卷",
        "volume": "第1卷",
        "edition_id": "mes-2012-v1",
        "publisher": "人民出版社",
        "publication_year": 2012,
        "work_id": "work-1",
        "article": "测试篇目",
        "section": "第一节",
        "pdf_page_start": 10,
        "pdf_page_end": 11,
        "printed_page_start": 8,
        "printed_page_end": 9,
        "citation_page_start": 8,
        "citation_page_end": 9,
        "citation_page_type": "printed_page",
        "page_span": [10, 11],
        "spans": [{"page_id": "mes01.pdf#pdf10", "char_start": 1, "char_end": 9}],
        "source_page_ids": ["mes01.pdf#pdf10", "mes01.pdf#pdf11"],
        "provenance": {"page_record_version": "page-record/v2"},
        "retrievable": True,
    }
    return {**record, **overrides}


class RetrievalRecordV2Tests(unittest.TestCase):
    def test_children_have_exact_roundtrip_offsets_without_clipping(self):
        parent = _parent()

        children = list(build_semantic_child_records([parent], chunk_size=6, chunk_overlap=2))

        self.assertEqual(
            [(item["parent_char_start"], item["parent_char_end"]) for item in children],
            [(0, 6), (4, 10), (8, 14), (12, 16)],
        )
        reconstructed_end = max(item["parent_char_end"] for item in children)
        self.assertEqual(reconstructed_end, len(parent["paragraph_text"]))
        for child in children:
            start, end = child["parent_char_start"], child["parent_char_end"]
            self.assertEqual(child["paragraph_text"], parent["paragraph_text"][start:end])
            self.assertFalse(child["text_was_clipped"])
            self.assertEqual(child["indexed_char_start"], start)
            self.assertEqual(child["indexed_char_end"], end)
            self.assertEqual(child["child_chunk_index"], child["child_index"])
            self.assertEqual(child["child_chunk_total"], len(children))
            self.assertEqual(child["child_char_start"], start)
            self.assertEqual(child["child_char_end"], end)

    def test_child_inherits_bibliography_pages_spans_and_provenance_immutably(self):
        parent = _parent()
        original = json.loads(json.dumps(parent, ensure_ascii=False))

        child = next(build_semantic_child_records([parent], chunk_size=99, chunk_overlap=0))

        self.assertEqual(parent, original)
        for field in (
            "source", "book", "volume", "edition_id", "publisher",
            "publication_year", "work_id", "article", "section",
            "pdf_page_start", "pdf_page_end", "printed_page_start",
            "printed_page_end", "citation_page_start", "citation_page_end",
            "citation_page_type", "page_span", "spans", "source_page_ids",
            "provenance",
        ):
            self.assertEqual(child[field], parent[field])
        self.assertIsNot(child["spans"], parent["spans"])
        self.assertEqual(child["retrieval_unit"], "semantic_child")
        self.assertEqual(child["parent_paragraph_id"], parent["paragraph_id"])

    def test_hashes_and_stable_id_are_deterministic_and_content_sensitive(self):
        parent = _parent()
        first = list(build_semantic_child_records([parent], chunk_size=8, chunk_overlap=2))
        second = list(build_semantic_child_records([parent], chunk_size=8, chunk_overlap=2))

        self.assertEqual(first, second)
        child = first[0]
        self.assertEqual(
            child["source_text_sha256"],
            hashlib.sha256(parent["paragraph_text"].encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            child["indexed_text_sha256"],
            hashlib.sha256(child["paragraph_text"].encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            child["retrieval_id"],
            stable_retrieval_id(parent["paragraph_id"], 0, 8, child["paragraph_text"]),
        )
        changed = list(build_semantic_child_records(
            [_parent(paragraph_text="地" + parent["paragraph_text"][1:])],
            chunk_size=8,
            chunk_overlap=2,
        ))
        self.assertNotEqual(first[0]["retrieval_id"], changed[0]["retrieval_id"])

    def test_non_retrievable_and_empty_records_are_excluded(self):
        records = [
            _parent(paragraph_id="excluded", retrievable=False),
            _parent(paragraph_id="empty", paragraph_text=""),
            _parent(paragraph_id="included"),
        ]

        children = list(build_semantic_child_records(records, chunk_size=99, chunk_overlap=0))

        self.assertEqual([child["parent_paragraph_id"] for child in children], ["included"])

    def test_invalid_chunk_parameters_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            list(build_semantic_child_records([_parent()], chunk_size=0, chunk_overlap=0))
        with self.assertRaisesRegex(ValueError, "chunk_overlap"):
            list(build_semantic_child_records([_parent()], chunk_size=8, chunk_overlap=8))

    def test_cli_streams_jsonl_writes_summary_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "paragraphs.jsonl"
            output_path = root / "children.jsonl"
            summary_path = root / "summary.json"
            records = [_parent(), _parent(paragraph_id="skip", retrievable=False)]
            input_path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                "scripts/build_retrieval_records_v2.py",
                "--input", str(input_path),
                "--output", str(output_path),
                "--summary", str(summary_path),
                "--chunk-size", "8",
                "--chunk-overlap", "2",
            ]

            completed = subprocess.run(command, cwd=Path(__file__).parents[1], text=True, capture_output=True)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            output_records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(output_records), 3)
            self.assertEqual(summary["schema_version"], "retrieval-build-summary/v2")
            self.assertEqual(summary["paragraph_records_read"], 2)
            self.assertEqual(summary["paragraph_records_excluded"], 1)
            self.assertEqual(summary["semantic_child_records_written"], 3)
            self.assertEqual(summary["chunk_size"], 8)
            self.assertEqual(summary["chunk_overlap"], 2)
            self.assertRegex(summary["output_sha256"], r"^[0-9a-f]{64}$")

            repeated = subprocess.run(command, cwd=Path(__file__).parents[1], text=True, capture_output=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("already exists", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
