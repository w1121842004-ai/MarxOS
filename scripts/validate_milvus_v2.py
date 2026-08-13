#!/usr/bin/env python3
"""Validate the side-by-side corpus-v2 Milvus collection before promotion.

Checks are read-only and conservative:
  - collection exists and its schema covers the document-record/v2 contract
  - row count matches the authoritative semantic child JSONL
  - sampled rows round-trip: id, text, dual hashes, lineage, page fields
  - lineage resolves against the enriched paragraph cache
  - front matter never leaks into retrievable rows
  - optional hybrid retrieval probe using the production encoders

Exit code 0 means every blocking check passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

# Same macOS ARM libomp guard as app.py: the hybrid probe below loads BGE-M3
# (torch) and searches the FAISS-backed Milvus Lite index.
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marxos.indexing.milvus_contract_v2 import (  # noqa: E402
    BOOL_FIELDS,
    INT_FIELDS,
    STRING_FIELDS,
    from_milvus_int,
    text_sha256,
)

REPORT_VERSION = "milvus-v2-validate/v1"
FRONT_MATTER_PAGE_TYPES = {
    "toc", "title_page", "preface", "preface_editorial", "editorial_note",
    "publication_note", "index", "footnote_region", "unknown",
}
DEFAULT_PROBE_QUERIES = [
    "什么是剩余价值？",
    "宗教是人民的鸦片出自哪里？",
    "哥达纲领批判在马恩选集哪页？",
    "商品的二因素是什么？",
    "1844年经济学哲学手稿在马恩选集第几卷？",
]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            records.append(value)
    return records


def _client_factory(uri: str):
    env_uri = os.getenv("MILVUS_URI")
    clear_env_uri = bool(env_uri and "://" not in env_uri)
    if clear_env_uri:
        os.environ.pop("MILVUS_URI", None)
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise RuntimeError("pymilvus is required to validate the v2 index") from exc
    finally:
        if clear_env_uri and env_uri is not None:
            os.environ["MILVUS_URI"] = env_uri
    return MilvusClient(uri=uri)


def check_schema(client: Any, collection: str, description: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    fields = description.get("fields", []) if isinstance(description, dict) else []
    names = {field.get("name") for field in fields if isinstance(field, dict)}
    expected = set(STRING_FIELDS) | set(INT_FIELDS) | set(BOOL_FIELDS) | {"embedding", "sparse_embedding"}
    missing = sorted(expected - names)
    if missing:
        issues.append(f"schema missing contract fields: {', '.join(missing)}")
    dynamic = description.get("enable_dynamic_field", False) if isinstance(description, dict) else False
    if dynamic:
        issues.append("dynamic fields are enabled; the v2 contract requires enable_dynamic_field=false")
    return not issues, issues


def row_count(client: Any, collection: str, expected: int) -> tuple[int | None, list[str]]:
    issues: list[str] = []
    actual: int | None = None
    try:
        result = client.query(collection_name=collection, filter="", output_fields=["count(*)"], limit=1)
        value = (result or [{}])[0].get("count(*)")
        if value is not None:
            actual = int(value)
    except Exception:
        actual = None
    if actual is None:
        description = client.describe_collection(collection) or {}
        actual = description.get("num_entities") or 0
        actual = int(actual)
    if actual != expected:
        issues.append(f"row count mismatch: expected={expected} actual={actual}")
    return actual, issues


def _child_source_text(record: dict[str, Any]) -> str:
    return str(record.get("source_text") or record.get("paragraph_text") or record.get("text") or "")


def _child_indexed_text(record: dict[str, Any]) -> str:
    return str(record.get("indexed_text") or record.get("paragraph_text") or record.get("text") or "")


def check_sampled_rows(
    client: Any,
    collection: str,
    records: list[dict[str, Any]],
    stride: int,
    parent_ids: set[str],
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    sampled = [
        record for index, record in enumerate(records)
        if index % stride == 0 or index == len(records) - 1
    ]
    ids = [str(record.get("retrieval_id") or "") for record in sampled]
    fetched: dict[str, dict[str, Any]] = {}
    try:
        rows = client.get(collection_name=collection, ids=ids) or []
        fetched = {str(row.get("id")): row for row in rows if row}
    except Exception as exc:
        issues.append(f"client.get sampling failed: {exc}")
        return {"sampled": len(sampled), "checked": 0}, issues

    missing = [record_id for record_id in ids if record_id not in fetched]
    if missing:
        issues.append(f"{len(missing)} sampled rows missing from Milvus (first: {missing[:3]})")

    hash_mismatches = 0
    text_mismatches = 0
    lineage_missing = 0
    front_matter_leaks = 0
    bad_page_ranges = 0
    for record in sampled:
        row = fetched.get(str(record.get("retrieval_id") or ""))
        if not row:
            continue
        expected_text = _child_indexed_text(record)
        if str(row.get("text") or "") != expected_text:
            text_mismatches += 1
        if row.get("indexed_text_hash") != _sha256_text(expected_text):
            hash_mismatches += 1
        expected_source = _child_source_text(record)
        if row.get("source_text_hash") != text_sha256(expected_source):
            hash_mismatches += 1
        parent = str(row.get("parent_paragraph_id") or "")
        if not parent or parent not in parent_ids:
            lineage_missing += 1
        page_type = str(row.get("page_type") or "unknown")
        if bool(row.get("retrievable")) and page_type in FRONT_MATTER_PAGE_TYPES:
            front_matter_leaks += 1
        pdf_start = from_milvus_int(row.get("pdf_page_start"))
        pdf_end = from_milvus_int(row.get("pdf_page_end"))
        if pdf_start is None or pdf_end is None or pdf_start < 1 or pdf_end < pdf_start:
            bad_page_ranges += 1

    if text_mismatches:
        issues.append(f"text mismatch on {text_mismatches} sampled rows")
    if hash_mismatches:
        issues.append(f"hash mismatch on {hash_mismatches} sampled rows")
    if lineage_missing:
        issues.append(f"lineage unresolved on {lineage_missing} sampled rows")
    if front_matter_leaks:
        issues.append(f"front matter leaked into {front_matter_leaks} retrievable rows")
    if bad_page_ranges:
        issues.append(f"bad pdf page ranges on {bad_page_ranges} sampled rows")
    return {"sampled": len(sampled), "checked": len(sampled) - len(missing)}, issues


def check_lineage_index(parents: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    missing_index = 0
    missing_text = 0
    for record in parents:
        if record.get("paragraph_index") is None:
            missing_index += 1
        if not str(record.get("paragraph_text") or "").strip():
            missing_text += 1
    if missing_index:
        issues.append(f"{missing_index} parent paragraphs lack paragraph_index")
    if missing_text:
        issues.append(f"{missing_text} parent paragraphs lack paragraph_text")
    return {"parents": len(parents)}, issues


def hybrid_probe(
    client: Any,
    collection: str,
    queries: list[str],
    *,
    embedding_model: str,
    bm25_stats_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    from marxos.embeddings import HuggingFaceEmbeddings, embedding_encode_kwargs, resolve_cached_model_path
    from marxos.indexing.bm25_sparse_v2 import BM25SparseEncoderV2

    try:
        encoder = BM25SparseEncoderV2.load(bm25_stats_path)
    except Exception as exc:
        return {"probe": "skipped"}, [f"BM25 stats load failed: {exc}"]
    model_path = resolve_cached_model_path(embedding_model)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={**embedding_encode_kwargs(embedding_model), "batch_size": 8},
    )
    try:
        from pymilvus import AnnSearchRequest, RRFRanker
    except ImportError as exc:
        return {"probe": "skipped"}, [f"pymilvus import failed: {exc}"]

    probes = []
    try:
        client.load_collection(collection)
    except Exception as exc:
        issues.append(f"load_collection failed: {exc}")
        return {"probe": "failed"}, issues
    for query in queries:
        dense = embeddings.embed_query(query)
        sparse = encoder.embed_query(query)
        dense_request = AnnSearchRequest(
            data=[dense],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=8,
        )
        sparse_request = AnnSearchRequest(
            data=[sparse],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {}},
            limit=8,
        )
        hits = client.hybrid_search(
            collection_name=collection,
            reqs=[dense_request, sparse_request],
            ranker=RRFRanker(),
            limit=8,
            output_fields=["id", "source", "article", "work_id", "citation_page_start", "page_type"],
        )
        top = []
        for hit in (hits[0] if hits else []):
            entity = hit.get("entity") or {}
            top.append({
                "id": entity.get("id"),
                "source": entity.get("source"),
                "article": entity.get("article"),
                "work_id": entity.get("work_id"),
                "page": entity.get("citation_page_start"),
            })
        probes.append({"query": query, "top": top})
        if not top:
            issues.append(f"hybrid probe returned no hits for: {query}")
    return {"probe": "ok", "queries": len(probes), "results": probes}, issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-records", type=Path,
                        default=ROOT / "data/artifacts/corpus_v2/semantic_child_records_v2_1.jsonl")
    parser.add_argument("--parent-records", type=Path,
                        default=ROOT / "data/artifacts/corpus_v2/paragraph_records_enriched_v2_1.jsonl")
    parser.add_argument("--uri", default=str(ROOT / "data/milvus_lite/marxos_corpus_v2.db"))
    parser.add_argument("--collection", default="marxos_passages_v2")
    parser.add_argument("--bm25-stats", type=Path,
                        default=ROOT / "data/artifacts/corpus_v2/bm25_stats_v2_1.json")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--sample-stride", type=int, default=90)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "logs/milvus_v2_validate.json")
    args = parser.parse_args()

    checks: dict[str, Any] = {}
    issues: list[str] = []
    try:
        records = _load_jsonl(args.child_records)
        parents = _load_jsonl(args.parent_records)
        parent_ids = {str(record.get("paragraph_id") or "") for record in parents}
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    client = _client_factory(args.uri)
    if not client.has_collection(args.collection):
        print(f"error: collection {args.collection} does not exist", file=sys.stderr)
        return 2
    try:
        client.load_collection(args.collection)
    except Exception as exc:
        print(f"error: load_collection failed: {exc}", file=sys.stderr)
        return 2

    description = client.describe_collection(args.collection) or {}
    ok, schema_issues = check_schema(client, args.collection, description)
    checks["schema"] = {"ok": ok, "issues": schema_issues}
    issues.extend(schema_issues)

    actual, count_issues = row_count(client, args.collection, len(records))
    checks["row_count"] = {"ok": not count_issues, "expected": len(records), "actual": actual,
                           "issues": count_issues}
    issues.extend(count_issues)

    sample_summary, sample_issues = check_sampled_rows(
        client, args.collection, records, max(args.sample_stride, 1), parent_ids
    )
    checks["sampled_rows"] = {"ok": not sample_issues, **sample_summary, "issues": sample_issues}
    issues.extend(sample_issues)

    lineage_summary, lineage_issues = check_lineage_index(parents)
    checks["lineage_parents"] = {"ok": not lineage_issues, **lineage_summary, "issues": lineage_issues}
    issues.extend(lineage_issues)

    if not args.skip_probe:
        probe_summary, probe_issues = hybrid_probe(
            client, args.collection, DEFAULT_PROBE_QUERIES,
            embedding_model=args.embedding_model, bm25_stats_path=args.bm25_stats,
        )
        checks["hybrid_probe"] = {"ok": not probe_issues, **probe_summary, "issues": probe_issues}
        issues.extend(probe_issues)
    else:
        checks["hybrid_probe"] = {"ok": True, "probe": "skipped", "issues": []}

    report = {
        "schema_version": REPORT_VERSION,
        "uri": args.uri,
        "collection": args.collection,
        "ready": not issues,
        "checks": checks,
        "issues": issues,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "checks"},
                     ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
