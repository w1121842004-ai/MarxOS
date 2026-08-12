# Milvus Migration Plan

This plan covers the first Milvus migration scope only:

- 马克思恩格斯全集
- 马克思恩格斯文集
- 马克思恩格斯选集

Lenin, Mao, and other corpora are intentionally out of scope for this phase.

## Goals

1. Move the long-lived retrieval index from local FAISS files to Milvus.
2. Use `BAAI/bge-m3` as the default dense embedding model.
3. Keep citation rendering deterministic: the model cites `[E1]`, while MarxOS renders the final citation from metadata.
4. Preserve FAISS as a fallback while Milvus is introduced.

## Collection

Default collection name:

```text
marxos_me_passages
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
chunk_id            VarChar
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
```

The text field is stored so result cards can be rendered without a second lookup.

## Source Data

The first import source is:

```text
data/paragraph_cache_core.jsonl
```

Each record maps to one Milvus row. Later we can add a child-chunk collection, but paragraph-level rows are the best first step because they preserve citation metadata and avoid huge prompt contexts.

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

## Migration Steps

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start Milvus locally or point to an existing server.

3. Build the collection:

```bash
MARXOS_EMBEDDING_MODEL=BAAI/bge-m3 \
MILVUS_URI=http://localhost:19530 \
.venv/bin/python scripts/build_milvus_collection.py
```

4. Keep FAISS enabled until Milvus search is validated.

5. Add a Milvus backend behind `marxos/vector_backend.py`.

6. Switch web retrieval via environment variable:

```bash
MARXOS_VECTOR_BACKEND=milvus
```

## Validation

Run existing retrieval probes against both backends:

```bash
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

## Non-Goals

- Do not import Lenin/Mao corpora in this phase.
- Do not enable bge-m3 sparse or multi-vector retrieval yet.
- Do not remove FAISS until Milvus quality and latency are stable.
