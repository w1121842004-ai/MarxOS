"""Side-effect-free runtime diagnostics and a sanitized configuration snapshot."""

from __future__ import annotations

import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from marxos.config.settings import AppSettings


def _configured_env(*names: str) -> bool:
    return any(bool(os.getenv(name, "").strip()) for name in names)


def build_runtime_manifest(settings: AppSettings, runtime: Any) -> dict[str, Any]:
    """Return a serializable snapshot; secret values are deliberately excluded."""
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "application": "MarxOS",
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "profiles": {
            "corpus": settings.profiles.active_corpus_profile,
            "retrieval": settings.profiles.active_retrieval_profile,
            "answer": settings.profiles.active_answer_profile,
        },
        "corpus": {
            "paragraph_cache": settings.corpus.paragraph_cache_path,
            "semantic_parent_cache": settings.corpus.semantic_parent_cache_path,
        },
        "retrieval": {
            "backend": runtime.vector_backend(),
            "embedding_model": settings.models.embedding_model,
            "embedding_device": settings.models.embedding_device,
            "milvus_uri": settings.index.milvus_uri,
            "collection": settings.index.milvus_collection,
            "dimension": settings.index.milvus_dim,
            "sparse_provider": settings.index.milvus_sparse_provider,
            "hybrid_search": settings.index.milvus_hybrid_search,
            "retrieval_unit": settings.index.milvus_retrieval_unit,
        },
        "llm": {
            "base_url": settings.models.deepseek_base_url,
            "model": settings.models.deepseek_model,
            "flash_model": settings.models.deepseek_flash_model,
            "api_key_configured": _configured_env("DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
        },
        "runtime": {
            "vectorstore_loaded": runtime.milvus_vectorstore_instance is not None
            if runtime.vector_backend() == "milvus"
            else runtime.vectorstore_instance is not None,
            "dense_encoder_loaded": runtime.embeddings_instance is not None,
            "sparse_encoder_loaded": runtime.sparse_embeddings_instance is not None,
        },
    }


def readiness_report(settings: AppSettings, runtime: Any) -> dict[str, Any]:
    """Inspect runtime state without opening an index or loading either encoder."""
    backend = runtime.vector_backend()
    index_path = settings.index.milvus_uri if backend == "milvus" else settings.index.vectorstore_dir
    index_exists = Path(index_path).exists()
    collection_loaded = (
        runtime.milvus_vectorstore_instance is not None
        if backend == "milvus"
        else runtime.vectorstore_instance is not None
    )
    schema_ok = True
    schema_detail = "not_applicable"
    if backend == "milvus":
        schema_ok = False
        schema_detail = "collection_not_loaded"
        client = runtime.milvus_client_instance
        if collection_loaded and client is not None:
            try:
                description = client.describe_collection(settings.index.milvus_collection)
                fields = description.get("fields", []) if isinstance(description, dict) else []
                field_names = {
                    field.get("name")
                    for field in fields
                    if isinstance(field, dict) and field.get("name")
                }
                required = {"id", "text", "embedding"}
                if settings.index.milvus_hybrid_search:
                    required.add("sparse_embedding")
                missing = sorted(required - field_names)
                schema_ok = not missing
                schema_detail = {"required": sorted(required), "missing": missing}
            except Exception as exc:
                schema_detail = f"schema_probe_failed: {type(exc).__name__}"
    checks = {
        "index_path": {"ok": index_exists, "path": index_path},
        "collection_loaded": {
            "ok": collection_loaded,
            "collection": settings.index.milvus_collection if backend == "milvus" else None,
        },
        "schema": {"ok": schema_ok, "detail": schema_detail},
        "embedding_config": {
            "ok": bool(settings.models.embedding_model.strip()),
            "model": settings.models.embedding_model,
            "device": settings.models.embedding_device,
        },
        "llm_api_key": {
            "ok": _configured_env("DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
            "provider": "deepseek",
        },
    }
    return {
        "ready": all(check["ok"] for check in checks.values()),
        "status": "ready" if all(check["ok"] for check in checks.values()) else "not_ready",
        "checks": checks,
    }


def write_runtime_manifest(path: str | Path, settings: AppSettings, runtime: Any) -> Path:
    """Atomically write the sanitized snapshot without replacing a valid file early."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_runtime_manifest(settings, runtime), ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(target)
    return target
