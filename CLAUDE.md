# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MarxOS is a local RAG system for Chinese-language Marxist classical texts. It processes PDFs of Marx/Engels works (OCR → FAISS vectorstore → hybrid retrieval → DeepSeek generation → citation audit), outputting academic-style answers with verifiable source citations.

## Setup & Run

```powershell
# First time
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

```powershell
# Quick regression (smoke + unit tests, ~4 min)
venv\Scripts\python.exe scripts/check.py --mode quick

# Full evaluation (+ retrieval + citation + e2e, ~30 min)
venv\Scripts\python.exe scripts/check.py --mode full

# Single test file
venv\Scripts\python.exe -m unittest discover -s tests -p test_web_api.py

# Metadata evaluation (fast, no LLM calls)
venv\Scripts\python.exe -X utf8 scripts/evaluate_e2e.py --dataset eval_dataset_v2.json

# Audit entry point
venv\Scripts\python.exe scripts/audit.py list
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

- `ocr_to_cache.py` → `clean_ocr_text.py` → `build_vectorstore_from_cache.py`
- `paragraph_cache.py` → `semantic_retrieval.py` (BM25 sparse + parent window)
- `exact_quote_lookup.py` — OCR cache full-text search (bypasses vector retrieval)
- `core_classics.json` → 15 legacy works (superseded by `work_catalog.json`)

## Where to Change What

- **Retrieval accuracy**: `retrieval/constraints.py` (constraint building) or `retrieval/ranking.py` (rerank weights)
- **Work matching**: `marxos_work_catalog.py` (`match_query` logic) or `rag/work_catalog.json` (aliases, primary_concepts)
- **Answer style**: `marxos_prompts.py` (prompt templates)
- **Citation format**: `marxos_citations.py`
- **Reject rules / local answers**: `marxos_answers.py`
- **Web API/UI**: `web_app.py` + `marxos_web_support.py`
- **OCR/vectorstore build**: `rag/build_vectorstore_from_cache.py`

## Key Data Files

- `rag/work_catalog.json` — 89-work structured metadata (primary source of truth)
- `eval_dataset_v2.json` — 120-question evaluation set with work_id/source/discipline annotations
- `rag/article_map_core.json` — OCR-generated TOC entries (noisy, use work_catalog instead)
- `data/page_map.json` — PDF page ↔ printed page mapping
- `data/ocr_cache/` — Per-page OCR text (JSON per PDF page)
- `vectorstore/marx_reader_core/` — Primary FAISS index

## Fragile Points

1. **Constraint building order** in `constraints_from_query()` — work_catalog must run first, before locator/classic fallbacks. Rearranging these steps can break retrieval precision.
2. **ctx dict keys** — retrieval modules access helpers by string name. Renaming a function in `app.py` without updating `_retrieval_ctx()` causes silent runtime failures.
3. **Page numbers** — printed page ranges in `work_catalog.json` must match `page_map.json`. Verify with `scripts/audit.py page-metadata` after any page number edits.
4. **OCR quality** determines citation accuracy — LLM cannot fix garbled OCR text. Always verify with `scripts/audit.py exact-quote-top1` after reprocessing PDFs.
5. **Legacy `core_classics.json`** has 15 works with some page errors (e.g., german_ideology p499 should be p507). `work_catalog.json` is the corrected source.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs two jobs on push/PR:
- `quick-checks`: `scripts/check.py --mode quick` (smoke + unit tests)
- `metadata-eval`: `scripts/evaluate_e2e.py --ci --threshold 65` (work_correct must stay ≥65%)
