# MarxOS Engineering Refactor Plan

本文档记录 MarxOS 后续工程化分层重构方案，方便中断后继续执行。目标不是简单搬文件，而是把配置、数据处理、索引、检索、生成、Web 编排拆成清晰层次，降低后续切换语料、向量库、模型、检索策略时的修改成本。

## 1. 重构目标

当前项目已经具备 RAG、Milvus、BGE-M3、意图识别、引用审计、RAGAS 评测等能力，但部分配置和运行逻辑仍散落在 `app.py`、`scripts/*`、`rag/*`、`retrieval/*` 中。

最近“文集/选集测试库切回马恩全集 OCR”的问题说明：语料范围、路径、向量库、检索优先级这类设置本应只改配置层，但实际需要修改多个文件。因此需要建立稳定的工程分层。

目标分层：

```text
配置层 config
↓
数据处理层 loader / parser / splitter
↓
索引层 index builder / vectorstore
↓
检索层 retriever / fusion / reranker
↓
生成层 prompt / LLM / citation
↓
应用层 orchestration / web
```

## 2. 目标目录结构

建议逐步形成如下结构：

```text
marxos/
  config/
    settings.py
    corpus.py
    retrieval.py
    models.py

  data/
    loaders.py
    parsers.py
    paragraph_cache.py
    splitters.py

  indexing/
    milvus_builder.py
    milvus_schema.py
    faiss_builder.py
    sparse_builder.py
    healthcheck.py

  retrieval/
    constraints.py
    retriever.py
    fusion.py
    reranker.py
    ranking.py
    modes.py

  generation/
    prompts.py
    answers.py
    citations.py
    citation_audit.py
    llm_client.py

  app/
    orchestration.py
    runtime.py
    web.py
```

当前根目录中的 `app.py`、`web_app.py`、`marxos_*.py` 可以先保留为兼容入口，不要一次性删除。

## 3. 分阶段执行计划

### 阶段 1：建立配置层

目标：先解决“路径、模型、语料范围、检索开关散落各处”的问题。

新增：

```text
marxos/config/__init__.py
marxos/config/settings.py
marxos/config/corpus.py
marxos/config/retrieval.py
marxos/config/models.py
```

收口配置：

- `ARTICLE_MAP_PATH`
- `ARTICLE_MAP_EXTRA_PATHS`
- `PARAGRAPH_CACHE_PATH`
- `SEMANTIC_PARENT_CACHE_PATH`
- `VECTORSTORE_DIR`
- `PARAGRAPH_VECTORSTORE_DIR`
- `MILVUS_URI`
- `MILVUS_COLLECTION`
- `MARXOS_EMBEDDING_MODEL`
- `MARXOS_EMBEDDING_DEVICE`
- `MILVUS_HYBRID_SEARCH`
- `MILVUS_SPARSE_PROVIDER`
- `SEMANTIC_CHILD_CHUNK_SIZE`
- `SEMANTIC_CHILD_CHUNK_OVERLAP`
- `MARXOS_PREFERRED_EDITIONS`

原则：

- 默认语料优先级：`me` 全集 > `wenji` 文集 > `xuanji` 选集。
- `*_core` 文件只作为早期小范围测试兼容回退。
- 运行时代码禁止直接写死 `article_map_core`、`paragraph_cache_core`、`marx_reader_core`。
- 所有脚本尽量从配置层读取默认值，CLI 参数仍可覆盖。

阶段 1 验证：

```bash
.venv/bin/python -m py_compile app.py marxos/work_catalog.py rag/semantic_retrieval.py scripts/build_milvus_collection.py

.venv/bin/python -c "import app; print(app.ARTICLE_MAP_PATH); print(len(app.ARTICLE_MAP)); print('me01.pdf' in app.ARTICLE_MAP)"
```

期望：

- `ARTICLE_MAP_PATH` 默认是 `rag/article_map.json`。
- `ARTICLE_MAP` 同时包含 `meXX.pdf` 和 `meaXX/mesXX.pdf`。
- 现有网页服务仍可启动。

### 阶段 2：迁移数据处理层

目标：把 OCR/page map/article map/paragraph cache/semantic split 相关代码从 `app.py`、`rag/*`、`scripts/*` 中收拢。

迁移对象：

- `rag/paragraph_cache.py`
- `rag/semantic_retrieval.py` 中与 child chunk 构建相关的部分
- `scripts/build_paragraph_cache.py`
- `scripts/build_semantic_parent_cache.py`

目标模块：

```text
marxos/data/loaders.py
marxos/data/parsers.py
marxos/data/paragraph_cache.py
marxos/data/splitters.py
```

原则：

- `scripts/*` 只保留 CLI 入口。
- 数据层不直接依赖 LLM。
- 数据层不直接依赖 Web。
- splitter 只负责切分和元数据继承，不负责检索排序。

阶段 2 验证：

```bash
.venv/bin/python scripts/build_paragraph_cache.py --help
.venv/bin/python scripts/build_semantic_parent_cache.py --help
.venv/bin/python -m py_compile marxos/data/*.py
```

### 阶段 3：迁移索引层

目标：把 Milvus/FAISS/sparse index 构建逻辑从脚本中拆到可复用模块。

迁移对象：

- `scripts/build_milvus_collection.py`
- `scripts/build_semantic_child_vectorstore.py`
- `scripts/build_paragraph_vectorstore.py`
- sparse index 构建相关逻辑

目标模块：

```text
marxos/indexing/milvus_builder.py
marxos/indexing/milvus_schema.py
marxos/indexing/faiss_builder.py
marxos/indexing/sparse_builder.py
marxos/indexing/healthcheck.py
```

原则：

- schema 定义和 upsert 逻辑分开。
- index builder 不参与回答生成。
- Milvus Lite / Standalone 尽量通过配置切换。
- BGE-M3 dense/sparse 是否启用通过配置控制。

阶段 3 验证：

```bash
.venv/bin/python scripts/build_milvus_collection.py --limit 20 --drop-existing --uri ./data/milvus_lite/probe.db
.venv/bin/python -c "from pymilvus import MilvusClient; c=MilvusClient(uri='./data/milvus_lite/probe.db'); print(c.list_collections())"
```

### 阶段 4：整理检索层

目标：让检索逻辑完全聚合在 retrieval 层，不再让 `app.py` 承担大量检索职责。

当前已有：

```text
retrieval/constraints.py
retrieval/modes.py
retrieval/ranking.py
```

建议补充：

```text
retrieval/retriever.py
retrieval/fusion.py
retrieval/reranker.py
```

迁移对象：

- `app.py` 中的 `retrieve_documents`
- 多问题 query expansion
- topic constrained candidates
- dense/sparse/hybrid merge
- strict title fallback
- locator backstop

原则：

- `constraints.py` 只做约束识别。
- `retriever.py` 只做召回入口。
- `fusion.py` 负责多路召回合并、去重、RRF。
- `reranker.py` 负责 rerank 模型或规则重排。
- `ranking.py` 保留规则打分函数。
- `modes.py` 管 fast/standard/deep 的策略差异。

阶段 4 验证：

```bash
.venv/bin/python scripts/evaluate_eval_dataset.py --dataset eval_dataset.json --top-k 8 --report logs/eval_dataset_refactor_probe.json
```

### 阶段 5：迁移生成层

目标：把 prompt、答案模板、引用格式化、引用审计集中到 generation 层。

迁移对象：

- `marxos/generation/prompts.py`
- `marxos/generation/answers.py`
- `marxos/generation/citations.py`
- `marxos/generation/citation_verifier.py`
- `marxos/app/orchestration.py` 中生成相关部分

目标模块：

```text
marxos/generation/prompts.py
marxos/generation/answers.py
marxos/generation/citations.py
marxos/generation/citation_audit.py
marxos/generation/llm_client.py
```

原则：

- “列原文/摘录/引文”与“理论分析/解释/总结”边界集中在 prompt/answer builder 层。
- citation formatter 不负责召回。
- citation audit 是可开关的后端步骤，不应绑死在 RAG 主流程。
- LLM client 统一封装 DeepSeek/OpenAI/Claude 等后续扩展。

阶段 5 验证：

```bash
.venv/bin/python -m py_compile marxos/generation/*.py
.venv/bin/python scripts/regression_smoke.py
```

### 阶段 6：瘦身应用层

目标：让 `app.py` 和 `web_app.py` 只做入口，不再承载核心业务。

目标模块：

```text
marxos/app/runtime.py
marxos/app/orchestration.py
marxos/app/web.py
```

目标请求流程：

```text
query
→ intent / planner
→ constraints
→ retrieval
→ evidence selection
→ generation
→ citation audit
→ response
```

原则：

- `app.py` 保留旧 import 兼容。
- `web_app.py` 只负责 Web UI/API。
- orchestration 层只编排，不写具体检索/生成算法。

阶段 6 验证：

```bash
.venv/bin/python web_app.py
```

网页端测试：

- `列出10条马克思恩格斯 关于农民的论述`
- `马克思关于农民问题主要有哪些论述`
- `《共产党宣言》收录在哪里？`
- `宗教是人民的鸦片如何理解？`

### 阶段 7：测试与评测收口

目标：把不同测试类型区分清楚。

建议分类：

```text
tests/
  unit/
  integration/
  regression/
  fixtures/
```

评测脚本保留在 `scripts/`：

- `evaluate_eval_dataset.py`
- `evaluate_ragas.py`
- `evaluate_retrieval.py`
- `regression_smoke.py`

至少保留这些回归指标：

- 检索 top-k 命中率
- 引用页码命中率
- 原文摘录任务格式正确率
- 文集/选集/全集源优先级正确率
- RAGAS faithfulness / context recall
- MarxOS deterministic 指标

### 阶段 8：清理废弃文件

目标：确认新结构稳定后，再删除或归档废弃入口。

可清理对象：

- 明确 deprecated 的 FAISS v1 构建入口
- 已不用的 old eval 脚本
- 临时 probe 脚本
- 旧 `_core` 默认逻辑

原则：

- 先标记 deprecated。
- 至少一个版本周期后再删除。
- 删除前用 `rg` 确认没有运行时引用。

## 4. 中断恢复方式

每次恢复时先执行：

```bash
git status --short
rg -n "article_map_core|paragraph_cache_core|semantic_parent_cache_core|marx_reader_core" app.py rag scripts retrieval marxos/work_catalog.py
```

再看当前阶段：

- 如果 `marxos/config/` 还不存在，从阶段 1 开始。
- 如果配置层已存在，但脚本仍有硬编码路径，继续阶段 1。
- 如果脚本已薄入口化，继续阶段 3。
- 如果 `app.py` 仍有大量检索函数，继续阶段 4。
- 如果 prompt/answer/citation 仍在根目录，继续阶段 5。

每完成一个阶段，至少运行：

```bash
.venv/bin/python -m py_compile app.py web_app.py
.venv/bin/python -m py_compile $(find marxos retrieval rag scripts -name "*.py" -maxdepth 3)
```

如果 `find ...` 在当前 shell 中不可用或输出过长，可改用明确文件列表。

## 5. 当前注意事项

- 当前 Milvus DB 仍可能来自早期 core 缓存，代码层改完不等于语料层已经完整切到全集 OCR。
- 要真正让原文输出全面来自《马克思恩格斯全集》，仍需重建：

```text
OCR/page map
→ data/paragraph_cache.jsonl
→ data/semantic_parent_cache.jsonl
→ Milvus collection
```

- 在全集 OCR 库完成前，`*_core` 回退不要删除，否则现有网页测试可能断。
- 查询明确要求“文集/选集”时，应尊重用户指定版本；没有指定时默认全集优先。

## 6. 下一步建议

下一次继续时，优先执行阶段 1：

1. 新建 `marxos/config/`。
2. 把路径、模型、Milvus、检索开关全部集中。
3. 修改 `app.py`、`rag/semantic_retrieval.py`、`scripts/build_milvus_collection.py` 等读取配置层。
4. 保持行为不变，先通过 smoke test。

## 7. 当前执行进度

更新时间：2026-06-27

已完成第一轮工程化收口：

- 新增 `marxos/config/`。
- 新增 `marxos/config/settings.py`，集中管理 corpus、model、index、retrieval、web 配置。
- `app.py` 已改为从配置层读取：
  - article map
  - paragraph cache
  - semantic parent cache
  - vectorstore
  - Milvus
  - embedding model/device
  - trace/dev/retrieval env key
- `web_app.py` 已改为从配置层读取 host、port、metrics log。
- `marxos/work_catalog.py` 已改为从配置层读取版本优先级。

已完成第二轮边界收口：

- 新增 `marxos/data/`。
- 新增 `marxos/data/loaders.py`，集中管理 JSON/JSONL、article map 合并、topic catalog 加载。
- 新增 `marxos/data/paragraph_cache.py`，作为 paragraph cache 的数据层入口。
- 新增 `marxos/data/splitters.py`，集中 semantic parent / child chunk 切分入口。
- `scripts/build_semantic_parent_cache.py` 已改为调用 `marxos.data.splitters`。
- 新增 `marxos/indexing/`。
- 新增 `marxos/indexing/milvus_builder.py` 和 `marxos/indexing/faiss_builder.py`，作为索引层兼容入口。

已完成第三轮生成层收口：

- 新增 `marxos/generation/`。
- 新增 prompt、answer、citation、citation audit wrapper。
- 新增 `marxos/generation/llm_client.py`。
- `app.py` 中 BookLocator、CitationVerifier、主回答生成已改为通过 `generation.llm_client` 创建 DeepSeek client。
- `scripts/build_me_qa_eval_dataset.py` 已改为使用统一 LLM client 和模型配置。

已完成第四轮应用层入口收口：

- 新增 `marxos/app/`。
- 新增 `marxos/app/runtime.py`、`marxos/app/orchestration.py`、`marxos/app/web.py`。
- `app.py` 已改为从 `marxos.app.runtime` 和 `marxos.app.orchestration` 引入应用层能力。

已完成第五轮验证：

- 核心文件 `py_compile` 通过。
- `load_merged_article_map()` 验证返回 74 个 source，包含 `me01.pdf` 和 `mea01.pdf`。
- `app` 配置加载验证通过：
  - article map 默认 `rag/article_map.json`
  - 当前 paragraph cache 按兼容策略回退到 `data/paragraph_cache_core.jsonl`
  - DeepSeek model 通过配置层读取

当前仍保留的兼容项：

- `*_core` 仍保留在配置层作为回退，不应删除。
- `scripts/build_work_metadata.py` 仍专门面向文集/选集 work catalog，后续如果要让 work catalog 也支持全集，需要单独扩展元数据结构。
- `scripts/evaluate_ragas.py` 仍保留 DeepSeek/OpenAI 评测环境参数，因为它是外部 judge runner。

## 8. Profile 化配置

当前配置层已经支持三类 active profile：

- `MARXOS_CORPUS_PROFILE`
- `MARXOS_RETRIEVAL_PROFILE`
- `MARXOS_ANSWER_PROFILE`

当前内置 profile：

- corpus
  - `me_full`
  - `core_test`
- retrieval
  - `milvus_bgem3_hybrid`
  - `milvus_bgem3_fast`
  - `faiss_semantic`
- answer
  - `deepseek_default`
  - `deepseek_fast`
  - `deepseek_standard`

示例：

```bash
MARXOS_CORPUS_PROFILE=me_full \
MARXOS_RETRIEVAL_PROFILE=milvus_bgem3_hybrid \
MARXOS_ANSWER_PROFILE=deepseek_default \
.venv/bin/python web_app.py
```

切到 core 测试语料 + 快速检索 + 快速回答：

```bash
MARXOS_CORPUS_PROFILE=core_test \
MARXOS_RETRIEVAL_PROFILE=milvus_bgem3_fast \
MARXOS_ANSWER_PROFILE=deepseek_fast \
.venv/bin/python web_app.py
```

环境变量仍然可以覆盖 profile 内的单项值。例如：

- `ARTICLE_MAP_PATH`
- `PARAGRAPH_CACHE_PATH`
- `MILVUS_URI`
- `MILVUS_HYBRID_SEARCH`
- `SEMANTIC_CHILD_CHUNK_SIZE`
- `DEEPSEEK_MODEL`

因此现在的策略是：

- 日常切方案：优先切 profile
- 精细调试单项参数：再用具体环境变量覆盖

下一步建议：

1. 继续把 `app.py` 内部的大函数迁到 `marxos/app/orchestration.py` 或更细的 service 模块。
2. 把 `scripts/build_milvus_collection.py` 主实现迁到 `marxos/indexing/milvus_builder.py`，脚本只保留 argparse。
3. 继续压缩 `app.py` 内部的大函数，把更多实现迁到 `marxos/generation/` 与 `marxos/app/` 的细分模块。
4. 当全集 OCR cache 完成后，删除配置层之外的 core 回退依赖。
