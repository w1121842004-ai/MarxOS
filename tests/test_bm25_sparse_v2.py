import json
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import FrozenInstanceError
from pathlib import Path

from marxos.indexing.bm25_sparse_v2 import BM25Config, BM25SparseEncoderV2


class BM25SparseEncoderV2Tests(unittest.TestCase):
    def test_rare_term_receives_more_weight_than_common_term(self):
        encoder = BM25SparseEncoderV2.fit([
            "资本 劳动",
            "资本 商品",
            "资本 剩余价值",
        ])

        vector = encoder.embed_query("资本 剩余价值")

        self.assertGreater(vector[encoder.term_id("剩余")], vector[encoder.term_id("资本")])

    def test_document_weight_uses_bm25_length_normalization(self):
        encoder = BM25SparseEncoderV2.fit(["劳动 劳动", "劳动 " + "商品 " * 30])

        short, long = encoder.embed_documents(["劳动 劳动", "劳动 " + "商品 " * 30])

        term_id = encoder.term_id("劳动")
        self.assertGreater(short[term_id], long[term_id])

    def test_stats_and_config_are_immutable(self):
        encoder = BM25SparseEncoderV2.fit(["资本", "劳动"])
        with self.assertRaises(FrozenInstanceError):
            encoder.stats.document_count = 9
        with self.assertRaises(FrozenInstanceError):
            encoder.config.k1 = 2.0

    def test_unknown_and_empty_terms_are_ignored(self):
        encoder = BM25SparseEncoderV2.fit(["资本主义"])

        self.assertEqual({}, encoder.embed_query(""))
        self.assertEqual({}, encoder.embed_query("量子纠缠"))
        self.assertIsNone(encoder.term_id("不存在"))

    def test_persistence_roundtrip_is_deterministic_and_checksummed(self):
        documents = ["资本与劳动", "剩余价值与资本", "英文 Marx Marx"]
        encoder = BM25SparseEncoderV2.fit(documents, config=BM25Config(k1=1.3, b=0.7))

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "bm25.json"
            second = Path(directory) / "bm25-copy.json"
            encoder.save(first)
            loaded = BM25SparseEncoderV2.load(first)
            loaded.save(second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(encoder.embed_query("资本 劳动"), loaded.embed_query("资本 劳动"))
            payload = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual("bm25-sparse-v2", payload["format_version"])
            self.assertRegex(payload["checksum"], r"^[0-9a-f]{64}$")

            payload["stats"]["document_count"] += 1
            first.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                BM25SparseEncoderV2.load(first)

    def test_term_ids_are_stable_across_corpus_order(self):
        left = BM25SparseEncoderV2.fit(["资本 劳动", "商品"])
        right = BM25SparseEncoderV2.fit(["商品", "资本 劳动"])

        self.assertEqual(left.vocabulary, right.vocabulary)
        self.assertEqual(left.embed_query("劳动 商品"), right.embed_query("劳动 商品"))

    def test_hash_collisions_are_resolved_deterministically(self):
        with patch.object(BM25SparseEncoderV2, "_stable_term_id", return_value=7):
            left = BM25SparseEncoderV2.fit(["alpha beta gamma"])
            right = BM25SparseEncoderV2.fit(["gamma alpha beta"])

        self.assertEqual(left.vocabulary, right.vocabulary)
        self.assertEqual(len({term_id for _, term_id in left.vocabulary}), 3)


if __name__ == "__main__":
    unittest.main()
