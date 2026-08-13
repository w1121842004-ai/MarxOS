#!/usr/bin/env python3
"""Measure the three retrieval channels separately (no longer one "hybrid").

Channels:
  1. dense          — Milvus ANN search on BGE-M3 vectors only
  2. hybrid         — Milvus dense + corpus-aware BM25 sparse (RRF), default path
  3. local_bm25     — local paragraph-level BM25 index (sparse_retrieve_documents)

For each concept question in the eval set, report top-k sources per channel and
whether an expected source lands in the top-3. Exit 0 regardless; the report is
the deliverable (used to justify routing decisions in P2).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402
from marxos.vector_backend import MilvusVectorBackend  # noqa: E402
from scripts.evaluate_retrieval import CONCEPT_QUESTIONS  # noqa: E402

REPORT_VERSION = "retrieval-paths-compare/v1"
TOP_K = 8
HIT_K = 3


def top_sources(docs, limit=TOP_K):
    sources = []
    seen = set()
    for doc in docs or []:
        # backend.search returns VectorSearchResult; other channels return Documents.
        document = getattr(doc, "document", doc)
        source = (document.metadata or {}).get("source")
        if source and source not in seen:
            seen.add(source)
            sources.append(source)
        if len(sources) >= limit:
            break
    return sources


def run_channel(name, questions, fn):
    results = []
    started = time.perf_counter()
    for index, item in enumerate(questions, start=1):
        query = item.question
        docs = fn(query)
        sources = top_sources(docs)
        expected = set(item.expected_sources or [])
        hit = any(source in expected for source in sources[:HIT_K])
        results.append(
            {
                "index": index,
                "query": query,
                "expected_sources": list(item.expected_sources or []),
                "top_sources": sources[:TOP_K],
                "hit_top3": hit,
            }
        )
        print(f"[{name}] {index}. {query[:24]} hit={hit} top={sources[:4]}", flush=True)
    elapsed = time.perf_counter() - started
    hits = sum(1 for result in results if result["hit_top3"])
    return {"hits": hits, "total": len(results), "elapsed_s": round(elapsed, 1), "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "logs" / "retrieval_paths_compare.json")
    args = parser.parse_args()

    questions = list(CONCEPT_QUESTIONS)
    runtime = app.RUNTIME
    hybrid_backend = runtime.load_milvus_vectorstore()

    channels = {}

    # 1. dense only
    dense_backend = MilvusVectorBackend(
        client=runtime.milvus_client_instance,
        collection_name=app.MILVUS_COLLECTION,
        embedding_model=runtime.load_embeddings(),
        sparse_embedding_model=None,
        hybrid_enabled=False,
        collection_loaded=True,
    )
    channels["dense"] = run_channel(
        "dense", questions, lambda query, db=dense_backend: db.search(query, k=TOP_K)
    )

    # 2. hybrid (dense + corpus BM25 sparse)
    channels["hybrid"] = run_channel(
        "hybrid", questions, lambda query, db=hybrid_backend: db.search(query, k=TOP_K)
    )

    # 3. local paragraph BM25
    app.warm_sparse_index()
    channels["local_bm25"] = run_channel(
        "local_bm25", questions, lambda query: app.sparse_retrieve_documents(query, limit=TOP_K)
    )

    report = {
        "schema_version": REPORT_VERSION,
        "collection": app.MILVUS_COLLECTION,
        "top_k": TOP_K,
        "hit_k": HIT_K,
        "channels": channels,
        "summary": {
            name: {"hits": channel["hits"], "total": channel["total"],
                   "elapsed_s": channel["elapsed_s"]}
            for name, channel in channels.items()
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n=== SUMMARY ===")
    for name, channel in channels.items():
        print(f"{name}: {channel['hits']}/{channel['total']} top-3 source hits ({channel['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
