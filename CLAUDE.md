# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MarxOS is a local RAG system for Chinese-language Marxist classical texts. It processes PDFs of Marx/Engels works (OCR → paragraph cache → Milvus Lite BGE-M3 dense+sparse retrieval → DeepSeek generation → citation audit), outputting academic-style answers with verifiable source citations. FAISS vectorstores are retained as fallback/offline indexes.

## Setup & Run

```bash
# First time (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then add DEEPSEEK_API_KEY

# CLI
.venv/bin/python app.py

# Web (http://127.0.0.1:7860)
.venv/bin/python web_app.py

# Dev mode (no LLM call, trace only)
MARXOS_DEV_MODE=1 MARXOS_TRACE_ONLY=1 .venv/bin/python app.py
```

```powershell
# First time (Windows)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then add DEEPSEEK_API_KEY

# CLI
venv\Scripts\python.exe app.py

# Web (http://127.0.0.1:7860)
venv\Scripts\python.exe web_app.py

# Dev mode (no LLM call, trace only)
$env:MARXOS_DEV_MODE="1"; $env:MARXOS_TRACE_ONLY="1"
venv\Scripts\python.exe app.py
```

## Test & Verify

```bash
# Quick regression (smoke + unit tests, ~4 min)
.venv/bin/python scripts/check.py --mode quick

# Full evaluation (+ retrieval + citation + e2e, ~30 min)
.venv/bin/python scripts/check.py --mode full

# Single test file
.venv/bin/python -m unittest discover -s tests -p test_web_api.py

# Metadata evaluation (fast, no LLM calls)
.venv/bin/python -X utf8 scripts/evaluate_e2e.py --dataset eval_dataset_v2.json

# Audit entry point
.venv/bin/python scripts/audit.py list
```

## Architecture

### Query Pipeline

```
User Query
  → classify_query()                    (marxos_query_intent.py / app.py)
  → constraints_from_query()            (retrieval/constraints.py)
      → WorkCatalog.match_query()       (marxos_work_catalog.py)
      → fallback: locator/classic/topic rules
  → retrieve_documents()                (retrieval/modes.py)
      → hybrid merge (dense + BM25)
      → CRAG quality check → corrective retrieval if needed
  → rerank / diversify / dedup          (retrieval/ranking.py)
  → citation page refinement
  → build_context() → evidence cards
  → build_prompt(intent) → DeepSeek
  → audit_answer_citations() → final answer
```

### The ctx Pattern

`app.py` defines `_retrieval_ctx()` — a dict bundling helper functions and config values. All `retrieval/*` modules receive this ctx and extract needed helpers via `_helper(ctx, "name")`. The retrieval package has zero direct imports from `app.py`.

When adding a new capability to retrieval, add the function to `_retrieval_ctx()` in `app.py`, then extract it in the appropriate `retrieval/*` module.

### Retrieval Package (`retrieval/`)

- `__init__.py` — public API facade (41 symbols); `app.py` imports as `retrieval_utils`
- `constraints.py` — `constraints_from_query()` builds `{title, sources, page_ranges, entries}`. First tries WorkCatalog, then falls through locator→classic→concept→topic rules
- `modes.py` — `retrieve_documents()` main retrieval with hybrid merge, CRAG, backstop
- `ranking.py` — `rerank_documents()`, `diversify_documents()`, `dedupe_documents()`

### Work Catalog (`rag/work_catalog.json`)

89 works with structured metadata: `work_id`, `title`, `aliases`, `author`, `discipline`, `concepts`, `primary_concepts`, `quotes`, and cross-edition page references (`editions.wenji_vN` / `editions.xuanji_vN`).

`marxos_work_catalog.py` loads this and provides `match_query()` (title→alias→quote→concept matching with primary_concept scoring) and `get_entries()` / `get_constraints()`.

Two PDF editions are indexed:
- `mea01-10.pdf` = 《马克思恩格斯文集》10卷 (2009, 人民出版社)
- `mes01-04.pdf` = 《马克思恩格斯选集》第3版 4卷 (2012, 人民出版社)

### Core Modules

| Module | Responsibility |
|--------|---------------|
| `app.py` | CLI entry, `run_query()`, global config, `_retrieval_ctx()` |
| `marxos_orchestration.py` | Query pipeline glue: prepare→lookup→retrieve→CRAG→answer |
| `marxos_query_intent.py` | `classify_query()`: bibliographic/quote/concept/analysis/rag |
| `marxos_citations.py` | Citation formatting, evidence cards, citation audit |
| `marxos_answers.py` | Local rule-based answers (no LLM), reject rules |
| `marxos_prompts.py` | Prompt builders per intent type |
| `marxos_runtime.py` | Runtime state, vectorstore/embedding loading, env flags |

### RAG/OCR Pipeline (`rag/`)

- **V2 build pipeline (use this)**:
  1. `ocr_to_cache.py` → `data/ocr_cache/` (PDF→OCR text, per-page JSON+TXT)
  2. `paragraph_cache.py` → `data/paragraph_cache_core.jsonl` (paragraph detection)
  3. `scripts/build_semantic_child_vectorstore.py` → `vectorstore/marx_reader_core/` (180-char child chunks with `parent_paragraph_id`)
  4. `scripts/build_paragraph_vectorstore.py` → `vectorstore/marx_reader_paragraph/` (full paragraphs)
- **V1 build script** (`build_vectorstore_from_cache.py`): DEPRECATED — produces chunks without `parent_paragraph_id`, which cannot be expanded to paragraph windows. Its utility functions (`document_from_cache`, `BOOK_MAPPING`, `is_me_volume`, etc.) remain in active use.
- `semantic_retrieval.py` — `expand_semantic_parent_docs()` (child→paragraph window expansion), BM25 sparse retrieval
- `exact_quote_lookup.py` — OCR cache full-text search (bypasses vector retrieval)

## Where to Change What

- **Retrieval accuracy**: `retrieval/constraints.py` (constraint building) or `retrieval/ranking.py` (rerank weights)
- **Work matching**: `marxos_work_catalog.py` (`match_query` logic) or `rag/work_catalog.json` (aliases, primary_concepts)
- **Answer style**: `marxos_prompts.py` (prompt templates)
- **Citation format**: `marxos_citations.py`
- **Reject rules / local answers**: `marxos_answers.py`
- **Web API/UI**: `web_app.py` + `marxos_web_support.py`
- **OCR/vectorstore build**: V2 pipeline — `scripts/build_paragraph_cache.py` → `scripts/build_semantic_child_vectorstore.py` → `scripts/build_paragraph_vectorstore.py`

## Key Data Files

- `rag/work_catalog.json` — 89-work structured metadata (primary source of truth)
- `eval_dataset_v2.json` — 120-question evaluation set with work_id/source/discipline annotations
- `rag/article_map_core.json` — OCR-generated TOC entries (noisy, use work_catalog instead)
- `data/page_map.json` — PDF page ↔ printed page mapping
- `data/ocr_cache/` — Per-page OCR text (JSON+TXT per PDF page)
- `data/paragraph_cache_core.jsonl` — Paragraph records (JSONL), feeds both vectorstores and `expand_semantic_parent_docs()`
- `vectorstore/marx_reader_core/` — Child-chunk FAISS index (298K vectors, 180-char, with `parent_paragraph_id`)
- `vectorstore/marx_reader_paragraph/` — Paragraph FAISS index (43K vectors, dual-retrieval supplement)

## Fragile Points

1. **Constraint building order** in `constraints_from_query()` — work_catalog must run first, before locator/classic fallbacks. Rearranging these steps can break retrieval precision.
2. **ctx dict keys** — retrieval modules access helpers by string name. Renaming a function in `app.py` without updating `_retrieval_ctx()` causes silent runtime failures.
3. **Page numbers** — printed page ranges in `work_catalog.json` must match `page_map.json`. Verify with `scripts/audit.py page-metadata` after any page number edits.
4. **OCR quality** determines citation accuracy — LLM cannot fix garbled OCR text. Always verify with `scripts/audit.py exact-quote-top1` after reprocessing PDFs.
5. **Legacy `core_classics.json`** has 15 works with some page errors (e.g., german_ideology p499 should be p507). `work_catalog.json` is the corrected source.
6. **V1 vs V2 vectorstore** — Always use V2 build pipeline (`build_paragraph_cache.py` → `build_semantic_child_vectorstore.py` → `build_paragraph_vectorstore.py`). V1 chunks lack `parent_paragraph_id` and silently break paragraph window expansion.
7. **LLM retry configuration** — All three LLM call sites use `max_retries=2-3` and `timeout=30-120s` via OpenAI client config. If requests fail transiently, the SDK handles retries automatically; persistent failures still propagate as exceptions.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs two jobs on push/PR:
- `quick-checks`: `scripts/check.py --mode quick` (smoke + unit tests)
- `metadata-eval`: `scripts/evaluate_e2e.py --ci --threshold 65` (work_correct must stay ≥65%)
