# MarxOS Codebase Inventory

这份文档说明当前代码库里主要代码文件的职责、上下游关系，以及已经识别出的可继续合并或收口的地方。

## 1. 当前主线

项目主线已经比较清楚：

```text
PDF
-> OCR cache
-> text cleaning / page metadata
-> chunk / paragraph vectorstore
-> retrieval / rerank / citation refine
-> prompt / LLM
-> answer / evidence / citation audit
-> web / CLI
```

对外真正的产品骨架是：

- `app.py`
- `web_app.py`
- `rag/`
- `scripts/check.py`
- `tests/*.py`

## 2. 顶层入口

| 文件 | 作用 | 关联 |
| --- | --- | --- |
| `app.py` | CLI 入口与总控编排。负责把 intent、retrieval、prompts、citations、answers、LLM 调用串起来。 | 所有核心模块 |
| `web_app.py` | Web API 与前端页面，封装 `app.run_query()`。 | `app.py`、`logs/` |
| `marxos_web_support.py` | Web 层公共辅助函数，负责 metrics、history/context 和响应 payload 拼装。 | `web_app.py` |
| `marxos_web_followups.py` | Web 层专题 follow-up 规则，负责专题追问的改写、解释和历史证据整理。 | `web_app.py` |
| `marxos_web_citations.py` | Web 层引文与页码 follow-up 规则，负责脚注解析、OCR 页定位和页码追问。 | `web_app.py` |

## 3. 核心模块

| 文件 | 核心功能 | 关联 |
| --- | --- | --- |
| `marxos_runtime.py` | 运行时状态、embedding/vectorstore 缓存、开发模式开关 | `app.py`、`web_app.py` |
| `marxos_embeddings.py` | HuggingFace embeddings 兼容导入层，压平旧 LangChain 弃用 warning | `marxos_runtime.py`、构建与评测脚本 |
| `marxos_query_intent.py` | query intent 分类、路由辅助 | `app.py` |
| `marxos_citations.py` | citation 格式化、证据生成、回答引文审计 | `app.py`、`web_app.py` |
| `marxos_answers.py` | 本地直接回答的规则分支，包括专题列表回答、拒答规则、摘录清洗 | `app.py` |
| `marxos_prompts.py` | prompt builders 与回答风格规则 | `app.py` |
| `marxos_trace.py` | trace 输出、TRACE_ONLY 模式辅助 | `app.py` |

## 4. 检索分层

检索现在已经从 `app.py` 中拆出，并继续细分：

| 文件 | 核心功能 | 关联 |
| --- | --- | --- |
| `marxos_retrieval.py` | 对外 facade，保留统一调用面 | `app.py` |
| `marxos_retrieval_constraints.py` | title/topic/source/page range constraints，seed queries，candidate pdf pages | `marxos_retrieval.py` |
| `marxos_retrieval_ranking.py` | rerank、diversify、constraint annotation、topic selection | `marxos_retrieval.py` |
| `marxos_retrieval_modes.py` | 实际 retrieval 执行、strict-title backstop、paragraph/dual retrieval、citation-page refinement | `marxos_retrieval.py` |

这组模块共同替代了原先 `app.py` 中那一大段 retrieval 实现。

## 5. OCR / RAG / 语料处理

| 文件 | 核心功能 | 关联 |
| --- | --- | --- |
| `rag/ocr_to_cache.py` | 从 PDF 提取文本层或跑 OCR，写入 OCR cache | `data/ocr_cache/`、`rag/clean_ocr_text.py`、`rag/page_number_detection.py` |
| `rag/clean_ocr_text.py` | OCR 文本清洗 | `rag/ocr_to_cache.py`、`rag/build_vectorstore_from_cache.py` |
| `rag/page_number_detection.py` | 页码、页眉页脚、版心信号判断 | `rag/ocr_to_cache.py`、`rag/build_vectorstore_from_cache.py` |
| `rag/build_vectorstore_from_cache.py` | 从 OCR cache 构建 chunk 向量库 | `vectorstore/marx_reader_core/` |
| `rag/paragraph_cache.py` | 段落 cache 构建与处理 | `scripts/build_paragraph_cache.py`、`scripts/build_paragraph_vectorstore.py` |
| `rag/exact_quote_lookup.py` | 精确引文检索，优先绕过向量检索 | `data/ocr_cache/`、`rag/core_classics.py` |
| `rag/core_classics.py` | 核心经典目录、别名、书目分组加载 | `rag/core_classics.json`、`rag/core_bibliography_catalog.json` |
| `rag/generate_article_map.py` | 生成篇目映射 | `rag/article_map.json`、`rag/article_map_core.json` |
| `rag/repair_vectorstore_metadata.py` | 修复历史向量库 metadata | `rag/build_vectorstore_from_cache.py` |

## 6. 数据与配置文件

| 文件 | 作用 | 关联 |
| --- | --- | --- |
| `rag/core_classics.json` | 核心经典目录、别名、条目范围 | `rag/core_classics.py`、检索评测 |
| `rag/core_bibliography_catalog.json` | 核心书目目录 | `rag/core_classics.py` |
| `rag/article_map.json` | 全量篇目映射 | 语料构建脚本 |
| `rag/article_map_core.json` | 核心篇目映射 | `app.py`、`rag/build_vectorstore_from_cache.py` |
| `rag/topic_catalog.json` | 专题检索目录 | `app.py`、`marxos_retrieval_*` |
| `eval_dataset.json` | 端到端评测集 | `scripts/evaluate_eval_dataset.py` |
| `data/page_map.json` | PDF 页与印刷页映射 | `app.py` |

## 7. 构建、评测与审计脚本

### 7.1 正式构建 / 检查

| 文件 | 作用 | 备注 |
| --- | --- | --- |
| `scripts/check.py` | `quick/full` 两档本地检查入口 | 正式入口 |
| `scripts/regression_smoke.py` | 轻量烟测 | `quick` |
| `scripts/evaluate_retrieval.py` | 检索评测 | `full` |
| `scripts/evaluate_citation_pages.py` | 引文页码评测 | `full` |
| `scripts/evaluate_eval_dataset.py` | 端到端评测 | `full` |
| `scripts/build_page_map.py` | 预生成 `data/page_map.json` | 可选辅助 |
| `scripts/build_paragraph_cache.py` | 构建 paragraph cache | 构建脚本 |
| `scripts/build_paragraph_vectorstore.py` | 构建 paragraph vectorstore | 构建脚本 |

### 7.2 审计与诊断

| 文件 | 作用 | 备注 |
| --- | --- | --- |
| `scripts/audit.py` | 审计统一入口 | 建议优先使用 |
| `scripts/audit_cache_page_sequence.py` | OCR cache 页序检查 | 诊断脚本 |
| `scripts/audit_concept_metadata.py` | 概念题 metadata 检查 | 诊断脚本 |
| `scripts/audit_exact_quote_top1.py` | 精确引文 top1 检查 | 诊断脚本 |
| `scripts/audit_ocr_printed_pages.py` | OCR printed page 检查 | 诊断脚本 |
| `scripts/audit_page_candidates.py` | 页码候选检查 | 诊断脚本 |
| `scripts/audit_page_metadata.py` | 向量库页码 metadata 检查 | 诊断脚本 |
| `scripts/audit_paragraph_cache.py` | paragraph cache 检查 | 诊断脚本 |
| `scripts/compare_dual_retrieval.py` | dual retrieval 对比 | 调参脚本 |
| `scripts/inspect_article_map.py` | article map 检查 | 工具脚本 |
| `scripts/report_api_ask_metrics.py` | API 指标汇总 | 运维观察 |
| `scripts/topic_conversation_regression.py` | 话题式 API 回归 | 专项脚本 |

## 8. 测试

| 文件 | 作用 |
| --- | --- |
| `tests/test_run_query_regressions.py` | `run_query()` 出口行为回归 |
| `tests/test_web_api.py` | `/api/ask` evidence / metrics 回归 |
| `tests/test_app_local_paths.py` | 本地路径、离线路径、引文路径大套件 |
| `tests/test_core_bibliography.py` | 核心书目逻辑 |
| `tests/test_page_metadata_inference.py` | 页码 metadata 推断 |
| `tests/test_paragraph_cache.py` | paragraph cache 逻辑 |

## 9. 已经清掉的历史残留

这些文件已经移除，不再作为主线入口：

- `ocr/pdf_to_text.py`
- `rag/build_knowledge_base.py`

## 10. 仍可继续收口的地方

### A. `app.py` 仍然偏大

虽然已经从超大实现文件瘦到总控入口，但 `run_query()` 周边仍有不少 orchestration 和 glue code。后续可以继续收：

- topic info / last evidence 状态管理
- OpenAI / DeepSeek client 调用包装
- CLI `main()` 与交互层

### B. retrieval facade 还可以再薄

`marxos_retrieval.py` 现在只是 facade，这已经比之前好很多，但后面可以考虑：

- 明确公共 API
- 把内部 helper 只保留在子模块，不再全部从 facade 导出

### C. 文档和中文编码

本次已经重写 `README.md` 和本文件，但历史 `docs/dev_logs/` 里仍可能有旧编码遗留。它们不影响运行，只影响阅读体验。

### D. 依赖 warning

项目自己的 `HuggingFaceEmbeddings` 弃用 warning 已做兼容处理。当前仍可能看到的是：

- `faiss`
- `setuptools._distutils`

这类 warning 来自第三方依赖，不是项目主逻辑问题。

## 11. 推荐维护顺序

如果继续做下一轮整理，建议顺序是：

1. 继续压薄 `app.py`
2. 收紧 `marxos_retrieval.py` facade 的公共导出
3. 视需要整理 `docs/dev_logs/`
4. 最后再考虑更大规模的目录重组

## 12. 一句话总结

这个仓库现在已经不是“功能散乱但能跑”的状态，而是“主线清楚、模块分层基本成形、测试和评测都有护栏”的状态。后续最值得做的，不是推倒重来，而是继续温和地收口入口文件和文档边界。

## 13. 维护入口

如果目标是“快速判断某类改动该改哪里”，优先看：

- `docs/maintenance_guide.md`
