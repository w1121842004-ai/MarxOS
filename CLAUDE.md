# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MarxOS is a local RAG system for Chinese-language Marxist classical texts. It processes PDFs of Marx/Engels works (OCR → paragraph cache → Milvus Lite BGE-M3 dense+sparse retrieval → DeepSeek generation → citation audit), outputting academic-style answers with verifiable source citations. FAISS vectorstores are retained as fallback/offline indexes.

Three PDF editions are indexed:
- `mea01-10.pdf` = 《马克思恩格斯文集》10卷 (2009, 人民出版社)
- `mes01-04.pdf` = 《马克思恩格斯选集》第3版 4卷 (2012, 人民出版社)
- `me01-50.pdf` = 《马克思恩格斯全集》 ~50卷

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

### Package Layout

```
app.py                          # CLI entry, run_query(), all helpers, _retrieval_ctx(), performance presets
web_app.py                      # Web UI (HTML SPA) + SSE streaming HTTP server
marxos/                         # Core library (refactored from flat marxos_*.py)
├── __init__.py
├── app/
│   ├── orchestration.py        # Query pipeline: prepare→lookup→retrieve→CRAG→answer
│   └── runtime.py              # Re-exports RuntimeState from marxos.runtime
├── config/
│   └── settings.py             # Profile-based AppSettings (corpus/retrieval/answer profiles)
├── data/
│   ├── loaders.py              # Article map, topic catalog loading
│   ├── splitters.py            # Text splitters for indexing
│   └── paragraph_cache.py      # Paragraph cache I/O
├── generation/
│   ├── answers.py              # Local rule-based answers, reject rules, view-list formatting
│   ├── citations.py            # Citation formatting, evidence cards, citation audit
│   ├── citation_audit.py       # CitationVerifier: content-level OCR verification
│   ├── citation_verifier.py    # Additional verification logic
│   ├── prompts.py              # Prompt builders per intent + citation style rules
│   └── llm_client.py           # DeepSeek OpenAI-compatible client factory
├── indexing/
│   ├── faiss_builder.py        # FAISS vectorstore construction
│   └── milvus_builder.py       # Milvus Lite collection construction
├── web/
│   ├── citations.py            # Web citation follow-up handlers
│   ├── followups.py            # Web topic/history follow-up handlers
│   └── support.py              # Web metrics, history trimming, contextual query building
├── ambiguous.py                # Ambiguous locator answer formatting
├── book_locator.py             # LLM-driven BookLocator agent (DeepSeek fallback)
├── embeddings.py               # Embedding model wrapper (BGE-M3)
├── intent_classifier.py        # ML intent classifier (logistic regression head, optional)
├── phoenix.py                  # Phoenix observability (OpenInference tracing)
├── query_intent.py             # classify_query_v2(): layered scoring intent router (v2)
├── query_planner.py            # QueryPlan: multi-query decomposition for retrieval
├── runtime.py                  # RuntimeState: vectorstore/embedding loading, env gating
├── trace.py                    # Trace output formatting (dev mode)
├── vector_backend.py           # VectorBackend protocol: FAISS + Milvus adapters
└── work_catalog.py             # WorkCatalog: 94-work structured metadata matching
retrieval/                      # Retrieval orchestration package
├── __init__.py                 # Public API facade (41 exported symbols)
├── constraints.py              # constraints_from_query(): multi-source constraint building
├── modes.py                    # retrieve_documents(): hybrid merge, CRAG, backstops
└── ranking.py                  # rerank/diversify/dedupe/collapse, topic selection
rag/                            # RAG/OCR pipeline + reference data
├── work_catalog.json           # 94 works with structured metadata (primary source of truth)
├── core_classics.json          # 15 classic works (legacy, some page errors → prefer work_catalog)
├── core_classics.py            # classic_entries_for_query(), load_core_classics()
├── topic_catalog.json          # Topic-based retrieval catalog
├── article_map*.json           # OCR-generated TOC entries (noisy)
├── exact_quote_lookup.py       # OCR cache full-text search (bypasses vector retrieval)
├── semantic_retrieval.py       # expand_semantic_parent_docs(), BM25 sparse retrieval
├── ocr_to_cache.py             # PDF → OCR text (per-page JSON+TXT)
├── paragraph_cache.py          # Paragraph detection from OCR cache
├── build_vectorstore_from_cache.py  # V1 build script (DEPRECATED but utilities still used)
├── me_high_precision_locators.json  # Token→exact page locators
├── me_article_locators.json    # Token→page range article locators
└── me_letter_locators.json     # Letter locators
scripts/                        # Build, evaluation, audit scripts (50+ files)
├── check.py                    # CI entry: quick (smoke+tests) / full (+retrieval eval)
├── audit.py                    # Audit command hub
├── build_paragraph_cache.py     # V2 build step 2: OCR→paragraph detection
├── build_semantic_child_vectorstore.py  # V2 build step 3: 180-char child chunks→FAISS
├── build_paragraph_vectorstore.py       # V2 build step 4: full paragraphs→FAISS
├── build_milvus_collection.py   # Milvus Lite collection builder
├── build_intent_classifier.py   # ML classifier training
├── evaluate_e2e.py              # End-to-end metadata evaluation (work_correct≥65%)
├── evaluate_retrieval.py        # Retrieval quality evaluation
├── evaluate_ragas.py            # RAGAS evaluation
└── evaluate_answer_quality.py   # Answer quality LLM-judge evaluation
tests/                          # Unit and regression tests (8 files)
```

### Query Pipeline (end-to-end)

```
User Query
  → classify_query()                           (marxos/query_intent.py, v2 layered scoring)
  → query_planner.plan_query()                 (marxos/query_planner.py, multi-query decomposer)
  → constraints_from_query()                   (retrieval/constraints.py)
      → WorkCatalog.match_query()              (marxos/work_catalog.py, 94-work metadata)
      → high_precision_locators               (me_high_precision_locators.json)
      → article_locators                      (me_article_locators.json)
      → classic_entries_for_query()           (app.py CLASSIC_LOCATOR_RULES + core_classics)
      → concept_constraints_from_query()       (CONCEPT_CANONICAL_CLASSIC_IDS mapping)
      → topic_entries_for_query()             (topic_catalog.json)
      → BookLocator LLM fallback               (marxos/book_locator.py)
  → retrieve_documents()                       (retrieval/modes.py)
      → controlled multi-query variants (dense retrieval)
      → hybrid merge (dense + BM25 sparse, RRF fusion)
      → OCR cache backstop (strict_title_cache_documents)
      → expand_semantic_parent_docs()          (child chunk → parent paragraph window)
  → CRAG quality assessment                    (marxos/app/orchestration.py)
      → corrective retrieval if score < threshold
  → rerank / diversify / dedup / collapse       (retrieval/ranking.py)
  → annotate_docs_with_constraints()           (enrich metadata)
  → build_context() → EVIDENCE-CARD format
  → build_prompt(intent, mode) → DeepSeek      (marxos/generation/prompts.py)
  → repair_answer_citations()                  (citation format repair)
  → filter_evidence_to_answer()                (evidence matching)
  → audit_answer_citations()                   (citation audit)
  → verify_citations()                         (OCR content verification, deep mode only)
  → recovery rounds (up to 2)                  (re-retrieve + regenerate if audit fails)
  → final answer
```

### Performance Modes

Three tiers configured in `performance_settings()` in `app.py`:

| Knob | fast | standard | deep |
|------|------|----------|------|
| retrieve_k | 3 | 4 | 5 |
| rag_retrieve_k | 5 | 8 | 12 |
| paragraph_retrieval | no | no | yes |
| corrective_retrieval | no | yes | yes |
| planner_multi_query | no | no | yes |
| hybrid_retrieval | no | no | yes |
| content_verification | no | no | yes |
| max_recovery_rounds | 0 | 0 | 2 |
| context_doc_char_limit | 800 | 1500 | 4000 |
| context_total_char_limit | 2500 | 6000 | 16000 |
| llm_timeout | 35s | 60s | 120s |

### Intent Taxonomy (v2)

`classify_query_v2()` in `marxos/query_intent.py` uses layered scoring across 7 intents, returning an `IntentResult` dataclass with `.primary`, `.confidence`, `.distribution`, `.is_ambiguous`. The object compares equal to its `.primary` string for backward compatibility.

| Intent | Description | Typical path |
|--------|-------------|-------------|
| `bibliographic_lookup` | Locate a work/volume/page range | Local article map → core classics |
| `quote_lookup` | Confirm source+page for exact passage | Exact OCR quote lookup → vector candidates |
| `concept_explain` | Explain a concept ("what is X?") | Concept-constrained retrieval → LLM |
| `comparison` | Compare works, concepts, or authors | RAG → LLM |
| `deep_analysis` | Multi-work synthesis, paper writing | Full retrieval pipeline → LLM |
| `theory_analysis` | Analyse through Marxist theoretical lens | RAG → LLM |
| `rag_answer` | Default catch-all | Standard hybrid retrieval → LLM |
| `chitchat` | Greetings, identity queries | Local answer, no retrieval |

### Profile-Based Configuration

`marxos/config/settings.py` defines frozen dataclass-based config with env var overrides:

- **Corpus profiles**: `me_full` (default, all editions), `core_test` (wenji+xuanji only)
- **Retrieval profiles**: `milvus_bgem3_hybrid` (default, dense+sparse), `milvus_bgem3_fast` (dense only), `faiss_semantic`
- **Answer profiles**: `deepseek_default` (deep mode), `deepseek_fast`, `deepseek_standard`

Access via `get_settings()` (LRU-cached singleton → restart needed for env var changes to take effect). Settings propagate through `app.py` module-level variables initialized from the singleton.

### The ctx Pattern

`app.py` defines `_retrieval_ctx()` — a dict bundling ~50 helper functions and config values. All `retrieval/*` modules receive this ctx and extract needed helpers via `_helper(ctx, "name")`. The retrieval package has zero direct imports from `app.py`.

When adding a new capability to retrieval, add the function to `_retrieval_ctx()` in `app.py`, then extract it via `_helper(ctx, "name")` in the appropriate `retrieval/*` module.

### Retrieval Package (`retrieval/`)

- `__init__.py` — public API facade (41 symbols); `app.py` imports as `retrieval_utils`
- `constraints.py` — `constraints_from_query()` builds `{title, sources, page_ranges, entries}`. Priority order: high_precision_locator → article_locator → explicit_volume → me_title_hint → work_catalog → topic → concept → locator_rules → classic_entries → BookLocator LLM fallback. Also: `topic_seed_queries()`, `concept_seed_queries()`, `controlled_multi_queries()`
- `modes.py` — `retrieve_documents()`: controlled multi-query dense → hybrid merge (dense+sparse RRF) → OCR cache backstop → expand_semantic_parent → annotate → append backstops. Also: `retrieve_paragraph_documents()`, `refine_docs_citation_pages_for_query()`, `filter_paragraph_docs_by_text_overlap()`, `merge_prefer_paragraph_docs()`
- `ranking.py` — `rerank_documents()` (10 scoring dimensions: source, page_range, topic_title, topic_content, article, query, hybrid_signal, concept_focus, concept_source, document_quality), `diversify_documents()`, `dedupe_documents()`, `collapse_content_near_duplicates()`, `select_topic_documents()`, `annotate_docs_with_constraints()`

### Work Catalog (`rag/work_catalog.json`)

94 works with structured metadata: `work_id`, `title`, `aliases`, `author`, `discipline`, `concepts`, `primary_concepts`, `quotes`, and cross-edition page references (`editions.wenji_vN` / `editions.xuanji_vN`).

`marxos/work_catalog.py` loads this and provides:
- `match_query()` — title→alias→quote→concept matching with primary_concept scoring
- `match_title_query()` — explicit title/alias mention detection
- `match_by_concepts()` — concept→work reverse lookup
- `get_entries()` / `get_constraints()` — structured constraint generation (source + page_range per edition)

### Core Modules — Detailed

| Module | Responsibility |
|--------|---------------|
| `app.py` | CLI entry, `run_query()`, all utility/helper functions (~3400 lines), `_retrieval_ctx()`, `performance_settings()`, `UNSUPPORTED_CLAIM_RULES`, global state (`LAST_EVIDENCE`, etc.) |
| `marxos/app/orchestration.py` | Pipeline glue: `prepare_query_request()`, `maybe_answer_local_lookup()`, `collect_retrieval_materials()`, `assess_retrieval_quality()`, `maybe_answer_local_view_query()` |
| `marxos/query_intent.py` | `classify_query_v2()`: layered scoring router (v2), `IntentResult` dataclass, jieba POS tagging, optional ML classifier blending |
| `marxos/query_planner.py` | `plan_query()`: multi-query decomposition (standalone_query + retrieval_queries) |
| `marxos/generation/answers.py` | Local answers: `answer_unsupported_claim()`, `is_view_list_query()`, `build_topic_view_list_answer()`, `build_strict_title_view_list_answer()` |
| `marxos/generation/prompts.py` | Prompt builders: `build_prompt(intent, query, context, mode)`, `final_answer_style_rules()`, `footnote_citation_rules()`, `compact_citation_rules()`, `task_boundary_rules()` |
| `marxos/generation/citations.py` | Citation operations: `format_citation()`, `evidence_from_docs()`, `audit_answer_citations()`, `repair_answer_citations()`, `filter_evidence_to_answer()` |
| `marxos/generation/citation_audit.py` | `CitationVerifier`: content-level verification against OCR cache text |
| `marxos/generation/llm_client.py` | `create_deepseek_client()`: OpenAI-compatible client factory with retry/timeout config |
| `marxos/runtime.py` | `RuntimeState`: lazy loading of embeddings, vectorstores (FAISS/Milvus), env flag gating |
| `marxos/vector_backend.py` | `VectorBackend` protocol + `MilvusVectorBackend` (dense+sparse hybrid) + FAISS adapter |
| `marxos/work_catalog.py` | `WorkCatalog`: loads `work_catalog.json`, title/concept matching, constraint generation |
| `marxos/book_locator.py` | `BookLocator`: LLM-driven work identification (DeepSeek fallback for unmatched queries) |
| `marxos/phoenix.py` | Phoenix observability: span management, doc/evidence/constraint summarization for tracing |
| `marxos/trace.py` | Dev-mode trace output: query trace, doc trace, prompt trace, constraints trace |
| `marxos/ambiguous.py` | Ambiguous locator answer: when multiple works match, format a clarifying response |
| `marxos/embeddings.py` | BGE-M3 embedding model wrapper (HuggingFace) |
| `marxos/intent_classifier.py` | Optional ML classifier: logistic regression head (~10 KB), blends with rule scores |
| `marxos/config/settings.py` | `AppSettings` dataclass tree: corpus/retrieval/answer/index/model/web settings with env overrides |
| `marxos/data/loaders.py` | Load merged article maps, topic catalogs from JSON |
| `marxos/data/splitters.py` | Text splitters for vectorstore indexing |
| `web_app.py` | Web UI (HTML SPA, ~470 lines) + SSE streaming (`/api/ask_stream`) + `MarxOSHandler` HTTP server |
| `marxos/web/citations.py` | Web citation follow-up: page detail lookup, OCR text loading, paragraph extraction |
| `marxos/web/followups.py` | Web follow-up handlers: topic scoping, evidence ranking, excerpt dedup, citation indices |
| `marxos/web/support.py` | Web utilities: metrics log, history trimming, contextual query building, trim/summarize |

### RAG/OCR Pipeline (`rag/`)

- **V2 build pipeline (always use this)**:
  1. `ocr_to_cache.py` → `data/ocr_cache/` (PDF→OCR text, per-page JSON+TXT)
  2. `scripts/build_paragraph_cache.py` → `data/paragraph_cache_core.jsonl` (paragraph detection)
  3. `scripts/build_semantic_child_vectorstore.py` → `vectorstore/marx_reader_core/` (180-char child chunks with `parent_paragraph_id`)
  4. `scripts/build_paragraph_vectorstore.py` → `vectorstore/marx_reader_paragraph/` (full paragraphs)
- **Milvus build**: `scripts/build_milvus_collection.py` → `data/milvus_lite/marxos_bgem3_sparse.db`
- **V1 build script** (`build_vectorstore_from_cache.py`): DEPRECATED — chunks lack `parent_paragraph_id`, silently break paragraph expansion. Its utility functions (`document_from_cache`, `BOOK_MAPPING`, `is_me_volume`, etc.) remain in active use.
- `semantic_retrieval.py` — `expand_semantic_parent_docs()` (child→parent window expansion), BM25 sparse retrieval
- `exact_quote_lookup.py` — OCR cache full-text search (bypasses vector retrieval, used for quote_lookup intent)

## Where to Change What

- **Retrieval accuracy**: `retrieval/constraints.py` (constraint building order) or `retrieval/ranking.py` (rerank weights)
- **Work matching**: `marxos/work_catalog.py` (`match_query` logic) or `rag/work_catalog.json` (aliases, primary_concepts, page ranges)
- **Answer style**: `marxos/generation/prompts.py` (prompt templates, style rules, citation format rules)
- **Citation format**: `marxos/generation/citations.py`
- **Reject rules / local answers**: `marxos/generation/answers.py` + `app.py` (`UNSUPPORTED_CLAIM_RULES`)
- **Performance knobs**: `performance_settings()` in `app.py` (three presets: fast/standard/deep)
- **Web API/UI**: `web_app.py` + `marxos/web/support.py` + `marxos/web/citations.py` + `marxos/web/followups.py`
- **OCR/vectorstore build**: V2 pipeline — `scripts/build_paragraph_cache.py` → `scripts/build_semantic_child_vectorstore.py` → `scripts/build_paragraph_vectorstore.py`
- **Intent classification**: `marxos/query_intent.py` (scoring weights) or `scripts/build_intent_classifier.py` (ML classifier training)
- **Configuration defaults**: `marxos/config/settings.py` (profile definitions, env var mappings)
- **Query planning**: `marxos/query_planner.py` (multi-query decomposition strategy)
- **Milvus integration**: `marxos/vector_backend.py` + `marxos/indexing/milvus_builder.py`
- **Phoenix tracing**: `marxos/phoenix.py` (span creation, attribute setting)

## Key Data Files

- `rag/work_catalog.json` — 94-work structured metadata (primary source of truth)
- `eval_dataset_v2.json` — 120-question evaluation set with work_id/source/discipline annotations
- `rag/article_map.json` + `rag/article_map_core.json` — OCR-generated TOC entries (noisy, prefer work_catalog)
- `rag/topic_catalog.json` — Topic-based retrieval catalog with keyword→work mappings
- `rag/core_classics.json` — 15 classic works (legacy; some page errors → prefer work_catalog)
- `rag/me_high_precision_locators.json` — High-precision locators (token→exact page)
- `rag/me_article_locators.json` — Article-level locators (token→page range)
- `rag/me_letter_locators.json` — Letter locators
- `data/page_map.json` — PDF page ↔ printed page mapping
- `data/ocr_cache/` — Per-page OCR text (JSON+TXT per PDF page)
- `data/paragraph_cache_core.jsonl` — Paragraph records (JSONL), feeds vectorstores + `expand_semantic_parent_docs()`
- `data/semantic_parent_cache_core.jsonl` — Semantic parent cache (paragraph windows)
- `data/intent_classifier.pkl` — Optional ML intent classifier (~10 KB)
- `vectorstore/marx_reader_core/` — Child-chunk FAISS index (298K vectors, 180-char, with `parent_paragraph_id`)
- `vectorstore/marx_reader_paragraph/` — Paragraph FAISS index (43K vectors, dual-retrieval supplement)
- `data/milvus_lite/marxos_bgem3_sparse.db` — Milvus Lite collection (BGE-M3 dense+sparse hybrid)
- `logs/api_ask_metrics.jsonl` — Web API metrics log (JSONL)

## Fragile Points

1. **Constraint building order** in `constraints_from_query()` — must follow: high_precision_locator → article_locator → explicit_volume → me_title_hint → work_catalog → topic → concept → locator_rules → classic_entries → BookLocator LLM. Rearranging these steps can break retrieval precision.
2. **ctx dict keys** — retrieval modules access helpers by string name via `_helper(ctx, "name")`. Renaming a function in `app.py` without updating `_retrieval_ctx()` causes silent runtime failures (no ImportError — just missing dict keys).
3. **Page numbers** — printed page ranges in `work_catalog.json` must match `page_map.json`. Verify with `scripts/audit.py page-metadata` after any page number edits.
4. **OCR quality** determines citation accuracy — LLM cannot fix garbled OCR text. Always verify with `scripts/audit.py exact-quote-top1` after reprocessing PDFs.
5. **Legacy `core_classics.json`** has 15 works with known page errors (e.g., german_ideology p499 should be p507). `work_catalog.json` is the corrected source — always prefer it.
6. **V1 vs V2 vectorstore** — Always use V2 build pipeline (`build_paragraph_cache.py` → `build_semantic_child_vectorstore.py` → `build_paragraph_vectorstore.py`). V1 chunks lack `parent_paragraph_id` and silently break paragraph window expansion in `expand_semantic_parent_docs()`.
7. **LLM retry configuration** — All three LLM call sites use `max_retries=2-3` and `timeout=30-120s` via OpenAI client config. Retries are automatic; persistent failures still propagate.
8. **Settings singleton** — `get_settings()` is LRU-cached. Modifying env vars after first import requires a process restart to take effect.
9. **Hybrid retrieval cold start** — BM25 sparse index builds lazily on first use. If `SEMANTIC_SPARSE_COLD_START=skip` and the index isn't warm, hybrid merge falls back to dense-only silently. Pre-warm with `MARXOS_WARM_SPARSE_INDEX=1`.
10. **Performance mode propagation** — `run_query()` passes `performance` dict to `retrieve_documents()`, `build_context()`, `build_prompt()`, and LLM calls. Missing keys silently fall back to deep-mode behavior.
11. **Module re-exports** — `marxos/app/runtime.py` re-exports `RuntimeState` from `marxos.runtime`. This indirection prevents circular imports — don't add direct `marxos.runtime` → `app.py` imports.
12. **Citation recovery rounds** — Up to 2 recovery rounds (citation format + content verification). Each re-runs the full retrieval + generation pipeline. Gated by `max_recovery_rounds` in the performance preset.
13. **OCR cache path resolution** — `load_ocr_page_text()` strips `.pdf` from source names and looks up `data/ocr_cache/{stem}/page_{N}.json`. Source names must match OCR cache directory names exactly.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs two jobs on push/PR:
- `quick-checks`: `scripts/check.py --mode quick` (validate_maps → regression_smoke → tests/app)
- `metadata-eval`: `scripts/evaluate_e2e.py --ci --threshold 65` (work_correct must stay ≥65%)
