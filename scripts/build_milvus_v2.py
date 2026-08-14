#!/usr/bin/env python3
"""Build a side-by-side Milvus v2 collection from RetrievalRecord JSONL.

The builder is intentionally conservative: it never drops a collection,
refuses an existing target unless ``--resume`` is supplied, and binds every
checkpoint to the exact input, BM25 statistics, model, and collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marxos.embeddings import (  # noqa: E402
    HuggingFaceEmbeddings,
    embedding_encode_kwargs,
    resolve_cached_model_path,
)
from marxos.indexing.bm25_sparse_v2 import BM25SparseEncoderV2  # noqa: E402
from marxos.indexing.milvus_contract_v2 import (  # noqa: E402
    create_v2_schema,
    row_from_record_v2,
)


CHECKPOINT_VERSION = "milvus-v2-build-checkpoint/v1"
BM25_META_VERSION = "bm25-v2-corpus-binding/v1"


@dataclass(frozen=True)
class BuildOptions:
    input_path: Path
    uri: str
    collection: str
    bm25_stats_path: Path
    checkpoint_path: Path
    embedding_model: str = "BAAI/bge-m3"
    dim: int = 1024
    batch_size: int = 8
    device: str = "cpu"
    token: str = ""
    limit: int = 0
    resume: bool = False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            retrieval_id = str(
                record.get("retrieval_id")
                or record.get("retrieval_record_id")
                or record.get("record_id")
                or record.get("id")
                or ""
            ).strip()
            if not retrieval_id:
                raise ValueError(f"missing retrieval_record_id at {path}:{line_number}")
            if retrieval_id in seen:
                raise ValueError(f"duplicate retrieval_record_id: {retrieval_id}")
            seen.add(retrieval_id)
            records.append({**record, "retrieval_record_id": retrieval_id})
            if limit and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"no retrieval records selected from {path}")
    return records


def _indexed_text(record: dict[str, Any]) -> str:
    text = str(
        record.get("indexed_text")
        or record.get("paragraph_text")
        or record.get("text")
        or record.get("source_text")
        or ""
    )
    if not text.strip():
        raise ValueError(f"empty indexed text for {record['retrieval_record_id']}")
    return text


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _signature(options: BuildOptions, input_sha256: str, selected_count: int) -> dict[str, Any]:
    return {
        "input_path": str(options.input_path.resolve()),
        "input_sha256": input_sha256,
        "selected_count": selected_count,
        "limit": options.limit,
        "uri": options.uri,
        "collection": options.collection,
        "embedding_model": options.embedding_model,
        "dim": options.dim,
        "batch_size": options.batch_size,
        "device": options.device,
        "bm25_stats_path": str(options.bm25_stats_path.resolve()),
    }


def _validate_checkpoint(state: dict[str, Any], signature: dict[str, Any]) -> None:
    expected = {"version": CHECKPOINT_VERSION, **signature}
    mismatches = [key for key, value in expected.items() if state.get(key) != value]
    if mismatches:
        raise ValueError(f"checkpoint does not match current build: {', '.join(mismatches)}")


def _bm25_meta_path(stats_path: Path) -> Path:
    return Path(f"{stats_path}.meta.json")


def _prepare_bm25(
    options: BuildOptions,
    texts: list[str],
    input_sha256: str,
) -> tuple[BM25SparseEncoderV2, str]:
    stats_path = options.bm25_stats_path
    meta_path = _bm25_meta_path(stats_path)
    expected_meta = {
        "version": BM25_META_VERSION,
        "input_sha256": input_sha256,
        "limit": options.limit,
        "document_count": len(texts),
    }
    if stats_path.exists() or meta_path.exists():
        if not stats_path.is_file() or not meta_path.is_file():
            raise ValueError("BM25 stats and corpus-binding metadata must both exist")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if any(meta.get(key) != value for key, value in expected_meta.items()):
            raise ValueError("BM25 stats do not match selected retrieval corpus")
        encoder = BM25SparseEncoderV2.load(stats_path)
        if encoder.stats.document_count != len(texts):
            raise ValueError("BM25 document count does not match selected retrieval corpus")
        return encoder, _sha256_file(stats_path)

    encoder = BM25SparseEncoderV2.fit(texts)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = stats_path.with_name(f".{stats_path.name}.tmp")
    encoder.save(temporary)
    os.replace(temporary, stats_path)
    stats_sha256 = _sha256_file(stats_path)
    _atomic_json(meta_path, {**expected_meta, "stats_sha256": stats_sha256})
    return encoder, stats_sha256


def _default_embeddings_factory(options: BuildOptions):
    model_path = resolve_cached_model_path(options.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": options.device},
        encode_kwargs={
            **embedding_encode_kwargs(options.embedding_model),
            "batch_size": options.batch_size,
        },
    )


def _default_client_factory(**kwargs):
    env_uri = os.getenv("MILVUS_URI")
    clear_env_uri = bool(env_uri and "://" not in env_uri)
    if clear_env_uri:
        os.environ.pop("MILVUS_URI", None)
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:  # pragma: no cover - integration boundary
        raise RuntimeError("pymilvus is required to build the v2 index") from exc
    finally:
        if clear_env_uri and env_uri is not None:
            os.environ["MILVUS_URI"] = env_uri
    return MilvusClient(**kwargs)


def _batches(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _local_database_path(uri: str) -> Path | None:
    value = str(uri or "").strip()
    return None if "://" in value else Path(value)


def run_build(
    options: BuildOptions,
    *,
    client_factory: Callable[..., Any] | None = None,
    embeddings_factory: Callable[[BuildOptions], Any] | None = None,
    schema_factory: Callable[..., Any] = create_v2_schema,
) -> dict[str, Any]:
    if not options.input_path.is_file():
        raise FileNotFoundError(options.input_path)
    if options.dim <= 0 or options.batch_size <= 0 or options.limit < 0:
        raise ValueError("dim and batch-size must be positive; limit cannot be negative")
    if not options.collection.strip():
        raise ValueError("collection cannot be empty")
    local_database = _local_database_path(options.uri)
    if local_database is not None and local_database.exists() and not options.resume:
        raise FileExistsError(f"refusing because local database already exists: {local_database}")

    records = _read_records(options.input_path, options.limit)
    texts = [_indexed_text(record) for record in records]
    input_sha256 = _sha256_file(options.input_path)
    signature = _signature(options, input_sha256, len(records))

    state: dict[str, Any] | None = None
    if options.checkpoint_path.exists():
        if not options.resume:
            raise FileExistsError(
                f"checkpoint already exists; pass --resume: {options.checkpoint_path}"
            )
        state = json.loads(options.checkpoint_path.read_text(encoding="utf-8"))
        _validate_checkpoint(state, signature)
    elif options.resume:
        raise FileNotFoundError(f"resume checkpoint does not exist: {options.checkpoint_path}")

    # 本地 DB 的父目录不存在时 MilvusClient 会直接失败，先建目录。
    local_database = _local_database_path(options.uri)
    if local_database is not None:
        local_database.parent.mkdir(parents=True, exist_ok=True)

    # Load the native embedding model before starting embedded Milvus Lite.
    embeddings = (embeddings_factory or _default_embeddings_factory)(options)
    kwargs = {"uri": options.uri}
    if options.token:
        kwargs["token"] = options.token
    client = (client_factory or _default_client_factory)(**kwargs)
    exists = bool(client.has_collection(options.collection))
    if exists and not options.resume:
        raise FileExistsError(f"refusing existing collection: {options.collection}")

    sparse_encoder, stats_sha256 = _prepare_bm25(options, texts, input_sha256)
    if state is not None and state.get("bm25_stats_sha256") != stats_sha256:
        raise ValueError("checkpoint does not match current BM25 stats")

    if state is None:
        state = {
            "version": CHECKPOINT_VERSION,
            **signature,
            "bm25_stats_sha256": stats_sha256,
            "processed_count": 0,
            "last_retrieval_id": None,
            "complete": False,
        }
        _atomic_json(options.checkpoint_path, state)

    if not exists:
        schema_factory(
            client,
            options.collection,
            options.dim,
            drop_existing=False,
            enable_sparse=True,
        )

    processed = int(state.get("processed_count") or 0)
    if not 0 <= processed <= len(records):
        raise ValueError("checkpoint processed_count is outside selected corpus")
    if processed:
        expected_last = records[processed - 1]["retrieval_record_id"]
        if state.get("last_retrieval_id") != expected_last:
            raise ValueError("checkpoint last_retrieval_id does not match deterministic input order")

    for batch_records in _batches(records[processed:], options.batch_size):
        batch_texts = [_indexed_text(record) for record in batch_records]
        dense_vectors = embeddings.embed_documents(batch_texts)
        sparse_vectors = sparse_encoder.embed_documents(batch_texts)
        if len(dense_vectors) != len(batch_records) or len(sparse_vectors) != len(batch_records):
            raise ValueError("encoder output count does not match input batch")
        if any(len(vector) != options.dim for vector in dense_vectors):
            raise ValueError("dense embedding dimension does not match --dim")
        rows = [
            row_from_record_v2(record, dense, text, sparse)
            for record, dense, text, sparse in zip(
                batch_records, dense_vectors, batch_texts, sparse_vectors
            )
        ]
        client.upsert(collection_name=options.collection, data=rows)
        processed += len(batch_records)
        state = {
            **state,
            "processed_count": processed,
            "last_retrieval_id": batch_records[-1]["retrieval_record_id"],
            "complete": False,
        }
        _atomic_json(options.checkpoint_path, state)

    # macOS ARM: loading the collection runs a FAISS-backed HNSW parallel region.
    # With torch's libomp already initialized for multi-threaded embedding, worker
    # suspend segfaults (SIGSEGV). Single-thread the final load; embedding stays
    # multi-threaded above. Re-runs are safe: --resume replays only this step.
    try:
        import faiss

        faiss.omp_set_num_threads(1)
    except Exception:
        pass
    client.load_collection(options.collection)
    state = {**state, "processed_count": processed, "complete": True}
    _atomic_json(options.checkpoint_path, state)
    return state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--uri", required=True, help="new Milvus Lite DB path or server URI")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--bm25-stats", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--token", default=os.getenv("MILVUS_TOKEN", ""))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    options = BuildOptions(
        input_path=args.input,
        uri=args.uri,
        collection=args.collection,
        bm25_stats_path=args.bm25_stats,
        checkpoint_path=args.checkpoint,
        embedding_model=args.embedding_model,
        dim=args.dim,
        batch_size=args.batch_size,
        device=args.device,
        token=args.token,
        limit=args.limit,
        resume=args.resume,
    )
    try:
        result = run_build(options)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
