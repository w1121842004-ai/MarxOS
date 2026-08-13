from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_milvus_v2 import BuildOptions, run_build


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(text)), 0.0, 1.0] for text in texts]


class FakeClient:
    def __init__(self, *, existing=False, fail_on_call=None):
        self.existing = existing
        self.fail_on_call = fail_on_call
        self.upsert_calls = []
        self.loaded = []

    def has_collection(self, _name):
        return self.existing

    def upsert(self, *, collection_name, data):
        call_number = len(self.upsert_calls) + 1
        if self.fail_on_call == call_number:
            raise RuntimeError("injected failure")
        self.upsert_calls.append((collection_name, data))

    def load_collection(self, name):
        self.loaded.append(name)


def write_records(path: Path, count: int = 3) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            record = {
                "retrieval_record_id": f"rr-{index:03d}",
                "paragraph_id": f"p-{index:03d}",
                "parent_paragraph_id": f"p-{index:03d}",
                "retrieval_unit": "semantic_child",
                "source": "mea01.pdf",
                "paragraph_text": f"马克思 恩格斯 文本 {index}",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class BuildMilvusV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.input = root / "retrieval.jsonl"
        self.stats = root / "bm25.json"
        self.checkpoint = root / "checkpoint.json"
        write_records(self.input)
        self.options = BuildOptions(
            input_path=self.input,
            uri=str(root / "new.db"),
            collection="v2_test",
            bm25_stats_path=self.stats,
            checkpoint_path=self.checkpoint,
            embedding_model="BAAI/bge-m3",
            dim=3,
            batch_size=2,
            device="cpu",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_fresh_build_persists_stats_rows_and_complete_checkpoint(self):
        client = FakeClient()
        schema_calls = []

        result = run_build(
            self.options,
            client_factory=lambda **_kwargs: client,
            embeddings_factory=lambda _options: FakeEmbeddings(),
            schema_factory=lambda client, collection, dim, **kwargs: schema_calls.append(
                (client, collection, dim, kwargs)
            ),
        )

        self.assertEqual(3, result["processed_count"])
        self.assertTrue(result["complete"])
        self.assertEqual([2, 1], [len(call[1]) for call in client.upsert_calls])
        self.assertEqual("rr-002", json.loads(self.checkpoint.read_text())["last_retrieval_id"])
        self.assertTrue(self.stats.is_file())
        self.assertTrue(Path(f"{self.stats}.meta.json").is_file())
        self.assertEqual(1, len(schema_calls))
        self.assertTrue(schema_calls[0][3]["enable_sparse"])

    def test_existing_collection_is_refused_without_resume(self):
        client = FakeClient(existing=True)
        with self.assertRaisesRegex(FileExistsError, "refusing existing collection"):
            run_build(
                self.options,
                client_factory=lambda **_kwargs: client,
                embeddings_factory=lambda _options: FakeEmbeddings(),
                schema_factory=lambda *_args, **_kwargs: None,
            )
        self.assertEqual([], client.upsert_calls)

    def test_existing_local_database_file_is_refused_without_resume(self):
        Path(self.options.uri).write_bytes(b"do not touch")
        called = False

        def client_factory(**_kwargs):
            nonlocal called
            called = True
            return FakeClient()

        with self.assertRaisesRegex(FileExistsError, "database already exists"):
            run_build(
                self.options,
                client_factory=client_factory,
                embeddings_factory=lambda _options: FakeEmbeddings(),
                schema_factory=lambda *_args, **_kwargs: None,
            )
        self.assertFalse(called)
        self.assertEqual(b"do not touch", Path(self.options.uri).read_bytes())

    def test_resume_continues_after_last_successful_retrieval_id(self):
        first_client = FakeClient(fail_on_call=2)
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            run_build(
                self.options,
                client_factory=lambda **_kwargs: first_client,
                embeddings_factory=lambda _options: FakeEmbeddings(),
                schema_factory=lambda *_args, **_kwargs: None,
            )
        state = json.loads(self.checkpoint.read_text())
        self.assertEqual(2, state["processed_count"])
        self.assertEqual("rr-001", state["last_retrieval_id"])

        resumed_client = FakeClient(existing=True)
        resumed = BuildOptions(**{**self.options.__dict__, "resume": True})
        result = run_build(
            resumed,
            client_factory=lambda **_kwargs: resumed_client,
            embeddings_factory=lambda _options: FakeEmbeddings(),
            schema_factory=lambda *_args, **_kwargs: self.fail("schema must not be recreated"),
        )
        self.assertEqual([1], [len(call[1]) for call in resumed_client.upsert_calls])
        self.assertEqual(3, result["processed_count"])

    def test_resume_rejects_changed_input(self):
        client = FakeClient(fail_on_call=2)
        with self.assertRaises(RuntimeError):
            run_build(
                self.options,
                client_factory=lambda **_kwargs: client,
                embeddings_factory=lambda _options: FakeEmbeddings(),
                schema_factory=lambda *_args, **_kwargs: None,
            )

    def test_resume_rejects_tampered_bm25_stats(self):
        client = FakeClient(fail_on_call=2)
        with self.assertRaises(RuntimeError):
            run_build(
                self.options,
                client_factory=lambda **_kwargs: client,
                embeddings_factory=lambda _options: FakeEmbeddings(),
                schema_factory=lambda *_args, **_kwargs: None,
            )
        self.stats.write_text("{}\n", encoding="utf-8")
        resumed = BuildOptions(**{**self.options.__dict__, "resume": True})
        with self.assertRaisesRegex(ValueError, "checksum"):
            run_build(
                resumed,
                client_factory=lambda **_kwargs: FakeClient(existing=True),
                embeddings_factory=lambda _options: FakeEmbeddings(),
                schema_factory=lambda *_args, **_kwargs: None,
            )
        with self.input.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"retrieval_record_id": "rr-new", "text": "changed"}) + "\n")

        resumed = BuildOptions(**{**self.options.__dict__, "resume": True})
        with self.assertRaisesRegex(ValueError, "checkpoint does not match"):
            run_build(
                resumed,
                client_factory=lambda **_kwargs: FakeClient(existing=True),
                embeddings_factory=lambda _options: FakeEmbeddings(),
                schema_factory=lambda *_args, **_kwargs: None,
            )

    def test_limit_builds_deterministic_probe_subset(self):
        client = FakeClient()
        limited = BuildOptions(**{**self.options.__dict__, "limit": 2})
        result = run_build(
            limited,
            client_factory=lambda **_kwargs: client,
            embeddings_factory=lambda _options: FakeEmbeddings(),
            schema_factory=lambda *_args, **_kwargs: None,
        )
        self.assertEqual(2, result["selected_count"])
        self.assertEqual(2, result["processed_count"])

    def test_duplicate_retrieval_ids_are_rejected_before_writes(self):
        duplicate = json.loads(self.input.read_text(encoding="utf-8").splitlines()[0])
        with self.input.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(duplicate, ensure_ascii=False) + "\n")
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "duplicate retrieval_record_id"):
            run_build(
                self.options,
                client_factory=lambda **_kwargs: client,
                embeddings_factory=lambda _options: FakeEmbeddings(),
                schema_factory=lambda *_args, **_kwargs: None,
            )
        self.assertEqual([], client.upsert_calls)

    def test_authoritative_retrieval_id_field_is_accepted_and_normalized(self):
        record = {
            "retrieval_id": "semantic-child-authoritative-id",
            "id": "same-id",
            "paragraph_id": "p-001",
            "paragraph_text": "用于检索的权威子块文本",
        }
        self.input.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        client = FakeClient()
        result = run_build(
            self.options,
            client_factory=lambda **_kwargs: client,
            embeddings_factory=lambda _options: FakeEmbeddings(),
            schema_factory=lambda *_args, **_kwargs: None,
        )
        self.assertEqual("semantic-child-authoritative-id", result["last_retrieval_id"])
        self.assertEqual(
            "semantic-child-authoritative-id",
            client.upsert_calls[0][1][0]["record_id"],
        )


if __name__ == "__main__":
    unittest.main()
