import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from marxos.config.settings import get_settings
from marxos.runtime import RuntimeState
from marxos.runtime_health import build_runtime_manifest, readiness_report, write_runtime_manifest


class RuntimeHealthTests(unittest.TestCase):
    def tearDown(self):
        get_settings.cache_clear()

    def test_default_profile_uses_bm25_sparse_encoder(self):
        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            settings = get_settings()

        self.assertEqual(settings.profiles.active_retrieval_profile, "milvus_bgem3_v2")
        self.assertEqual(settings.index.milvus_sparse_provider, "bm25")
        self.assertTrue(settings.index.milvus_hybrid_search)

    def test_manifest_is_a_secret_free_config_snapshot(self):
        settings = get_settings()
        runtime = Mock()
        runtime.vector_backend.return_value = "milvus"
        runtime.milvus_vectorstore_instance = object()
        runtime.milvus_client_instance = Mock()
        runtime.milvus_client_instance.describe_collection.return_value = {
            "fields": [
                {"name": "id"}, {"name": "text"}, {"name": "embedding"},
                {"name": "sparse_embedding"},
            ]
        }
        runtime.embeddings_instance = object()
        runtime.sparse_embeddings_instance = None

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "must-not-leak"}):
            manifest = build_runtime_manifest(settings, runtime)

        encoded = json.dumps(manifest, ensure_ascii=False)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["retrieval"]["sparse_provider"], settings.index.milvus_sparse_provider)
        self.assertTrue(manifest["llm"]["api_key_configured"])
        self.assertNotIn("must-not-leak", encoded)

    def test_readiness_does_not_load_models(self):
        settings = get_settings()
        runtime = Mock()
        runtime.vector_backend.return_value = "milvus"
        runtime.milvus_vectorstore_instance = object()
        runtime.milvus_client_instance = Mock()
        runtime.milvus_client_instance.describe_collection.return_value = {
            "fields": [
                {"name": "id"}, {"name": "text"}, {"name": "embedding"},
                {"name": "sparse_embedding"},
            ]
        }
        runtime.embeddings_instance = object()
        runtime.sparse_embeddings_instance = None

        with patch("marxos.runtime_health.Path.exists", return_value=True), patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "configured"}
        ):
            report = readiness_report(settings, runtime)

        self.assertTrue(report["ready"])
        runtime.load_embeddings.assert_not_called()
        runtime.load_sparse_embeddings.assert_not_called()
        runtime.load_vectorstore.assert_not_called()

    def test_readiness_reports_all_missing_dependencies(self):
        settings = get_settings()
        runtime = Mock()
        runtime.vector_backend.return_value = "milvus"
        runtime.milvus_vectorstore_instance = None
        runtime.milvus_client_instance = None

        with patch("marxos.runtime_health.Path.exists", return_value=False), patch.dict(
            os.environ, {}, clear=True
        ):
            report = readiness_report(settings, runtime)

        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["index_path"]["ok"])
        self.assertFalse(report["checks"]["collection_loaded"]["ok"])
        self.assertFalse(report["checks"]["llm_api_key"]["ok"])

    def test_manifest_write_is_valid_json(self):
        settings = get_settings()
        runtime = Mock()
        runtime.vector_backend.return_value = "milvus"
        runtime.milvus_vectorstore_instance = None
        runtime.milvus_client_instance = None
        runtime.embeddings_instance = None
        runtime.sparse_embeddings_instance = None

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "runtime.json"
            written = write_runtime_manifest(target, settings, runtime)
            parsed = json.loads(target.read_text(encoding="utf-8"))

        self.assertEqual(written, target)
        self.assertEqual(parsed["schema_version"], 1)

    def test_milvus_encoders_load_before_client_threads_start(self):
        runtime = RuntimeState(
            embedding_model="BAAI/bge-m3",
            vectorstore_dir="unused",
            paragraph_vectorstore_dir="unused",
            dev_mode_env="DEV",
            dev_token_env="TOKEN",
            dev_token_input_env="TOKEN_INPUT",
            trace_env="TRACE",
            trace_only_env="TRACE_ONLY",
            dual_retrieval_env="DUAL",
        )
        events = []
        dense = object()
        sparse = object()
        client = Mock()
        client_class = Mock(return_value=client)

        with (
            patch.object(runtime, "load_embeddings", side_effect=lambda: events.append("dense") or dense),
            patch.object(runtime, "load_sparse_embeddings", side_effect=lambda: events.append("sparse") or sparse),
            patch.object(runtime, "_import_milvus_client", side_effect=lambda: events.append("client_import") or client_class),
            patch("marxos.runtime.MilvusVectorBackend") as backend_class,
            patch.dict(os.environ, {"MILVUS_PREWARM_QUERY_ENCODER": "0"}),
        ):
            runtime.load_milvus_vectorstore()

        self.assertEqual(events, ["dense", "sparse", "client_import"])
        backend_class.assert_called_once_with(
            client=client,
            collection_name=runtime.milvus_collection,
            embedding_model=dense,
            sparse_embedding_model=sparse,
            collection_loaded=True,
        )


if __name__ == "__main__":
    unittest.main()
