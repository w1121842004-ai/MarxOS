# Milvus Stabilization Plan

This document originally tracked the first Milvus migration. The codebase has
now moved past the initial migration: Milvus Lite is the default retrieval
backend in the current configuration. The remaining work is stabilization,
health checks, evaluation, and deployment readiness.

Current corpus scope:

- 马克思恩格斯全集
- 马克思恩格斯文集
- 马克思恩格斯选集

Lenin, Mao, and other corpora are intentionally out of scope for this phase.

## Goals

1. Treat Milvus Lite as the default local retrieval backend.
2. Use `BAAI/bge-m3` as the default dense embedding model.
3. Keep sparse/hybrid retrieval explicit through configuration.
4. Keep citation rendering deterministic: the model cites `[E1]`, while MarxOS renders the final citation from metadata.
5. Preserve FAISS as a fallback and regression comparison path.
6. Add enough health checks and smoke tests for initial deployment.

## Collection

Promoted default collection (P1, 2026-08-13 — corpus-v2 side-by-side rebuild):

```text
marxos_passages_v2
```

Default local URI:

```text
./data/milvus_lite/marxos_corpus_v2.db
```

Retrieval unit / chunking / sparse:

```text
semantic_child
chunk 320 / overlap 64
corpus-aware BM25 sparse (data/artifacts/corpus_v2/bm25_stats_v2_1.json)
```

The previous P0 collection remains available as the rollback entry:

```text
marxos_text_layer_bgem3 @ ./data/milvus_lite/marxos_text_layer_bgem3.db
```

Default vector field:

```text
embedding
```

Default embedding:

```text
BAAI/bge-m3
```

Vector dimension:

```text
1024
```

Metric:

```text
COSINE
```

Index recommendation:

```text
HNSW
M = 16
efConstruction = 200
```

## Schema

Milvus scalar fields:

```text
id                  VarChar primary key
corpus              VarChar, e.g. marx_engels
author              VarChar, e.g. 马克思恩格斯
series              VarChar, 全集 / 文集 / 选集 normalized as title
book                VarChar, source volume title
volume              VarChar, 第1卷 / 第1卷A / 第26卷(上)
article             VarChar
section             VarChar
source              VarChar, e.g. me01.pdf
source_file         VarChar
paragraph_id        VarChar
parent_paragraph_id VarChar
chunk_id            VarChar
retrieval_unit      VarChar
pdf_page_start      Int64
pdf_page_end        Int64
printed_page_start  Int64
printed_page_end    Int64
citation_page_start Int64
citation_page_end   Int64
citation_page_type  VarChar
page_type           VarChar
is_letter           Bool
citation_mode       VarChar
text_hash           VarChar
text                VarChar
embedding           FloatVector(1024)
sparse_embedding    SparseFloatVector, optional
```

The text field is stored so result cards can be rendered without a second lookup.

## Source Data

The build script defaults to the first existing cache selected by
`marxos.config.settings`:

```text
data/semantic_parent_cache.jsonl
data/paragraph_cache.jsonl
data/semantic_parent_cache_core.jsonl
data/paragraph_cache_core.jsonl
```

The active retrieval unit is controlled by `MILVUS_RETRIEVAL_UNIT` and the
retrieval profile. The current hybrid profile uses semantic child retrieval;
paragraph-level records remain important for metadata preservation and fallback
testing.

## Runtime Path

```text
app.load_vectorstore()
-> marxos.runtime.RuntimeState.load_vectorstore()
-> RuntimeState.load_milvus_vectorstore()
-> pymilvus.MilvusClient(uri=MILVUS_URI)
-> client.load_collection(MILVUS_COLLECTION)
-> marxos.vector_backend.MilvusVectorBackend
```

Relevant environment variables:

```text
MARXOS_VECTOR_BACKEND=milvus
MILVUS_URI=./data/milvus_lite/marxos_corpus_v2.db
MILVUS_COLLECTION=marxos_passages_v2
MARXOS_EMBEDDING_MODEL=BAAI/bge-m3
MARXOS_EMBEDDING_DEVICE=cpu
MILVUS_SPARSE_PROVIDER=bm25
MARXOS_BM25_STATS_PATH=data/artifacts/corpus_v2/bm25_stats_v2_1.json
MILVUS_HYBRID_SEARCH=1
MILVUS_PREWARM_QUERY_ENCODER=1
MILVUS_PREWARM_SEARCH=0
OMP_NUM_THREADS=1
```

Rollback (old P0 baseline):

```text
MARXOS_CORPUS_PROFILE=me_full
MARXOS_RETRIEVAL_PROFILE=milvus_bgem3_stable
MILVUS_URI=./data/milvus_lite/marxos_text_layer_bgem3.db
MILVUS_COLLECTION=marxos_text_layer_bgem3
MILVUS_SPARSE_PROVIDER=lexical
```

Milvus Lite does not require a separate Milvus server, but it still opens the
local `.db`, loads the collection, loads BGE-M3, optionally loads a sparse
encoder, and may prewarm query encoding on each Python process start.

## Citation Rule

The LLM should cite evidence ids only:

```text
[E1] [E2]
```

The backend renders:

```text
《马克思恩格斯全集》第X卷，《篇名》，北京：人民出版社，第Y页。
```

This keeps page numbers out of the model's control.

## Build / Rebuild Steps

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Build or rebuild the local Milvus Lite DB:

   ```bash
   MARXOS_EMBEDDING_MODEL=BAAI/bge-m3 \
   MARXOS_EMBEDDING_DEVICE=cpu \
   MILVUS_URI=./data/milvus_lite/marxos_bgem3_sparse.db \
   .venv/bin/python scripts/build_milvus_collection.py
   ```

3. Keep FAISS assets available until Milvus quality and latency are stable.

4. Switch web retrieval explicitly when needed:

```bash
MARXOS_VECTOR_BACKEND=milvus
```

## Validation

Run existing retrieval probes:

```bash
.venv/bin/python scripts/test.py web
.venv/bin/python scripts/check.py --mode quick
.venv/bin/python scripts/evaluate_hybrid_retrieval.py
.venv/bin/python scripts/run_web_smoke.py --only-label standard_theory
```

Compare:

```text
latency
top-k source/article match
citation metadata completeness
answer quality
```

Deployment-readiness checks should additionally confirm:

- Milvus DB path exists
- collection exists
- collection schema contains dense and expected sparse fields
- embedding model is available locally or can be downloaded in the target environment
- `/api/ask` and `/api/ask_stream` return compatible payloads

## Non-Goals

- Do not import Lenin/Mao corpora in this phase.
- Do not remove FAISS until Milvus quality and latency are stable.
- Do not optimize startup latency before health checks and deterministic smoke tests are in place.
