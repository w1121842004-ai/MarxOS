from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import app  # noqa: E402


DEFAULT_QUERIES = [
    "商品拜物教到底拜的是什么？",
    "为什么同一劳动会有两种规定？",
    "剩余价值理论的基本结构是什么？",
    "为什么说旧国家机器不能直接拿来用？",
    "从资本论和工资价格利润说明马克思如何理解剩余价值。",
    "异化劳动、实践和人的本质之间是什么关系？",
]


def doc_key(doc) -> tuple:
    metadata = doc.metadata or {}
    return (
        metadata.get("source"),
        metadata.get("parent_paragraph_id") or metadata.get("paragraph_id"),
        metadata.get("chunk_id"),
        metadata.get("citation_page") or metadata.get("printed_page") or metadata.get("page"),
        str(doc.page_content or "")[:80],
    )


def dedupe(docs):
    merged = []
    seen = set()
    for doc in docs or []:
        key = doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
    return merged


def timed(fn):
    started = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - started) * 1000


def raw_single(db, query, fetch_k):
    docs, elapsed_ms = timed(lambda: db.similarity_search(query, k=fetch_k))
    return {
        "mode": "single_raw",
        "elapsed_ms": elapsed_ms,
        "queries": [query],
        "candidate_count": len(docs or []),
        "top_sources": top_sources(docs),
    }


def raw_multi(db, query, fetch_k, max_variants):
    constraints = app.constraints_from_query(query)
    variants = app.controlled_multi_queries(query, constraints)[:max_variants]
    per_query_k = max(18, fetch_k // max(len(variants), 1))

    def run():
        docs = []
        per_variant = []
        for variant in variants:
            variant_docs, variant_ms = timed(lambda v=variant: db.similarity_search(v, k=per_query_k))
            docs.extend(variant_docs or [])
            per_variant.append(
                {
                    "query": variant,
                    "elapsed_ms": variant_ms,
                    "candidate_count": len(variant_docs or []),
                }
            )
        return dedupe(docs), per_variant

    (docs, per_variant), elapsed_ms = timed(run)
    return {
        "mode": "multi_raw",
        "elapsed_ms": elapsed_ms,
        "queries": variants,
        "per_query_k": per_query_k,
        "candidate_count": len(docs or []),
        "top_sources": top_sources(docs),
        "per_variant": per_variant,
    }


def full_app(db, query, k):
    docs, elapsed_ms = timed(lambda: app.retrieve_documents(query, db, k=k, allow_exact_quote=False))
    return {
        "mode": "full_app",
        "elapsed_ms": elapsed_ms,
        "queries": [query],
        "candidate_count": len(docs or []),
        "top_sources": top_sources(docs),
    }


def top_sources(docs, limit=5):
    items = []
    for doc in (docs or [])[:limit]:
        metadata = doc.metadata or {}
        items.append(
            {
                "source": metadata.get("source"),
                "article": metadata.get("article") or metadata.get("section"),
                "retrieval_unit": metadata.get("retrieval_unit"),
                "parent": metadata.get("parent_paragraph_id"),
            }
        )
    return items


def summarize(rows):
    by_mode = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row["elapsed_ms"])
    return {
        mode: {
            "count": len(values),
            "avg_ms": round(statistics.mean(values), 1),
            "median_ms": round(statistics.median(values), 1),
            "min_ms": round(min(values), 1),
            "max_ms": round(max(values), 1),
        }
        for mode, values in by_mode.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark MarxOS Milvus retrieval latency.")
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--queries-file")
    parser.add_argument("--fetch-k", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-variants", type=int, default=4)
    parser.add_argument("--include-full-app", action="store_true")
    parser.add_argument("--report", default="logs/milvus_retrieval_time_benchmark.json")
    args = parser.parse_args()

    queries = list(args.query)
    if args.queries_file:
        queries.extend(
            line.strip()
            for line in Path(args.queries_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not queries:
        queries = DEFAULT_QUERIES

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    db = app.load_vectorstore()

    rows = []
    for iteration in range(args.iterations):
        for query in queries:
            for runner in (
                lambda q=query: raw_single(db, q, args.fetch_k),
                lambda q=query: raw_multi(db, q, args.fetch_k, args.max_variants),
            ):
                row = runner()
                row["iteration"] = iteration + 1
                row["query"] = query
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if args.include_full_app:
                row = full_app(db, query, args.top_k)
                row["iteration"] = iteration + 1
                row["query"] = query
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

    report = {
        "config": {
            "fetch_k": args.fetch_k,
            "top_k": args.top_k,
            "iterations": args.iterations,
            "max_variants": args.max_variants,
            "milvus_uri": os.getenv("MILVUS_URI"),
            "milvus_hybrid_search": os.getenv("MILVUS_HYBRID_SEARCH"),
            "milvus_sparse_provider": os.getenv("MILVUS_SPARSE_PROVIDER"),
            "vector_backend": os.getenv("MARXOS_VECTOR_BACKEND"),
        },
        "summary": summarize(rows),
        "results": rows,
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(report["summary"], ensure_ascii=False), flush=True)
    print(f"REPORT {args.report}", flush=True)


if __name__ == "__main__":
    main()
