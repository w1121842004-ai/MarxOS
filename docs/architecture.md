# 架构概览

这份文档不是文件清单，而是主链路说明。它回答的是三个问题：

1. 用户问题是怎么流过系统的
2. 哪些模块在主线上
3. 哪些地方最容易改坏，需要优先回归

## 1. 主流程

整体链路可以概括成：

```text
用户问题
-> Web 本地 follow-up 判断
-> query intent 判断
-> query planner
-> retrieval constraints
-> Milvus/FAISS/BM25 检索
-> rerank / citation refine
-> evidence cards
-> prompt builder
-> DeepSeek
-> citation audit / final answer
```

如果命中本地快速路径，链路会更短：

```text
用户问题
-> query intent 判断
-> 本地规则回答
-> evidence 过滤
-> citation audit
-> final answer
```

## 2. 主线模块

### 入口层

- [app.py](../app.py)
  - 对外入口
  - 保留 `run_query(...)`、CLI 入口和兼容包装函数
  - 负责把各子模块串起来

- [web_app.py](../web_app.py)
  - Web API / UI 入口
  - 调用 `app.run_query(...)`
  - 在进入主 RAG 前会先处理部分本地 follow-up，例如上一轮证据页码、专题条目解释、引文原页摘录

- [marxos/web/support.py](../marxos/web/support.py)
  - Web 层公共辅助函数
  - 包括 metrics、history 压缩、follow-up 判断和响应 payload 拼装

- [marxos/web/followups.py](../marxos/web/followups.py)
  - Web 层专题 follow-up 规则
  - 包括专题追问改写、条目解释、专题历史追问整理

- [marxos/web/citations.py](../marxos/web/citations.py)
  - Web 层引文与页码 follow-up 规则
  - 包括脚注解析、OCR 页定位、页码追问和原页摘录

### 流程编排

- [marxos/app/orchestration.py](../marxos/app/orchestration.py)
  - `run_query(...)` 的流程胶水
  - 包括输入预处理、本地查答分支、检索材料收集、本地列表题分支

### 运行时与依赖

- [marxos/runtime.py](../marxos/runtime.py)
  - Milvus / FAISS vectorstore 与 paragraph vectorstore 加载
  - dev/trace/dual retrieval 开关
  - Milvus Lite collection 加载、embedding 与 sparse encoder 复用、可选预热

- [marxos/embeddings.py](../marxos/embeddings.py)
  - 统一 embedding 导入和兼容层

### 路由与回答

- [marxos/query_intent.py](../marxos/query_intent.py)
  - bibliographic / quote / concept / analysis / rag 路由判断

- [marxos/generation/answers.py](../marxos/generation/answers.py)
  - 本地回答拼装
  - 列表题、专题题、拒答规则、摘录整理

- [marxos/generation/prompts.py](../marxos/generation/prompts.py)
  - prompt builders

### 检索与引文

- [retrieval/__init__.py](../retrieval/__init__.py)
  - facade 层，对外暴露检索公共 API（41 符号）

- [retrieval/constraints.py](../retrieval/constraints.py)
  - 标题/专题/概念约束、seed queries

- [retrieval/ranking.py](../retrieval/ranking.py)
  - rerank、diversify、dedup、topic selection

- [retrieval/modes.py](../retrieval/modes.py)
  - exact quote、sparse-first、dense/hybrid retrieval、strict-title backstop、paragraph/dual retrieval、citation refinement

- [marxos/vector_backend.py](../marxos/vector_backend.py)
  - `MilvusVectorBackend`
  - 封装 Milvus dense search / hybrid search / filter / query vector cache

- [marxos/generation/citations.py](../marxos/generation/citations.py)
  - 引文格式
  - evidence cards
  - final answer citation audit

- [marxos/trace.py](../marxos/trace.py)
  - trace 输出和 TRACE_ONLY 调试

## 3. 数据与构建链路

### OCR / cache / vectorstore

- `data/ocr_cache/`
  - OCR 后的文本缓存

- `data/milvus_lite/marxos_text_layer_bgem3.db`
  - 当前默认 Milvus Lite 数据库
  - collection 默认是 `marxos_text_layer_bgem3`
  - 默认 embedding 是 `BAAI/bge-m3`
  - 默认 sparse provider 是 `lexical`

- [rag/ocr_to_cache.py](../rag/ocr_to_cache.py)
  - PDF 文本层提取和扫描页 OCR

- [rag/build_vectorstore_from_cache.py](../rag/build_vectorstore_from_cache.py)
  - 从 cache 构建 chunk 向量库

- [scripts/build_paragraph_cache.py](../scripts/build_paragraph_cache.py)
  - 段落级 cache

- [scripts/build_paragraph_vectorstore.py](../scripts/build_paragraph_vectorstore.py)
  - 段落级向量库

- [scripts/build_milvus_collection.py](../scripts/build_milvus_collection.py)
  - 从 paragraph cache 或 semantic parent cache 构建 Milvus collection

## 3.1 当前默认运行基线

当前默认配置来自 [marxos/config/settings.py](../marxos/config/settings.py)：

```text
MARXOS_CORPUS_PROFILE=me_full_v2
MARXOS_RETRIEVAL_PROFILE=milvus_bgem3_v2
MARXOS_ANSWER_PROFILE=deepseek_default
MILVUS_URI=./data/milvus_lite/marxos_corpus_v2.db
MILVUS_COLLECTION=marxos_passages_v2
MARXOS_EMBEDDING_MODEL=BAAI/bge-m3
MILVUS_SPARSE_PROVIDER=bm25
MARXOS_BM25_STATS_PATH=data/artifacts/corpus_v2/bm25_stats_v2_1.json
MILVUS_HYBRID_SEARCH=1
OMP_NUM_THREADS=1
```

回滚入口（旧 P0 基线）：`MARXOS_CORPUS_PROFILE=me_full` + `MARXOS_RETRIEVAL_PROFILE=milvus_bgem3_stable` + 对应旧 `MILVUS_URI`/`MILVUS_COLLECTION`/`PARAGRAPH_CACHE_PATH`（见 `.env.example` 注释）。

Milvus Lite 是本地内嵌数据库，不需要单独部署 Milvus server；但每个 Python 进程启动后仍需要打开 `.db`、加载 collection、加载 embedding/sparse encoder，并可按 `MILVUS_PREWARM_QUERY_ENCODER` 做预热。macOS ARM 上必须保持 `OMP_NUM_THREADS=1`（torch 与 Milvus Lite 的 FAISS 索引共用 libomp，多线程检索段错误）。

## 4. 三类脚本

### 构建类

- `build_*`
- `detect_printed_page_start.py`

### 评测类

- `evaluate_*`
- `regression_smoke.py`
- `topic_conversation_regression.py`

### 审计类

- `audit.py`
- `audit_*`
- `report_api_ask_metrics.py`

## 5. 最容易改坏的点

### retrieval 侧

- `strict_title` 约束
- concept rerank
- paragraph/chunk 合并顺序
- citation refine

这类改动后，至少跑：

```powershell
venv\Scripts\python.exe scripts\check.py --mode full
```

### answer 侧

- prompt builders
- local answer 拼装
- citation filtering / audit

这类改动后，至少跑：

```powershell
venv\Scripts\python.exe scripts\check.py --mode quick
```

如果改动涉及 `web_app.py` 的多轮追问、payload、前端显示，还必须额外跑：

```powershell
venv\Scripts\python.exe scripts\test.py web
```

### OCR / 数据侧

- OCR 清洗
- printed page 推断
- article map / topic catalog

这类改动除了跑 `full`，还建议配合：

```powershell
venv\Scripts\python.exe scripts\audit.py list
```

## 6. 推荐阅读顺序

第一次接手项目时，建议按这个顺序看：

1. [README.md](../README.md)
2. [docs/codebase_inventory.md](./codebase_inventory.md)
3. [docs/architecture.md](./architecture.md)
4. [docs/eval_questions.md](./eval_questions.md)
5. [app.py](../app.py)
6. `marxos_*` 主线模块
7. [task.md](../task.md)
## 7. Retrieval Package Status

retrieval 层已经开始从顶层平铺模块收口到 `retrieval/` 包：

- `retrieval/__init__.py`
  - 对外 facade，供 `app.py` 统一导入
- `retrieval/constraints.py`
  - title/topic/source/page range constraints 与 seed queries
- `retrieval/ranking.py`
  - rerank、diversify、constraint annotation、topic document selection
- `retrieval/modes.py`
  - 实际 retrieval 执行、strict-title backstop、paragraph/dual retrieval、citation-page refinement

检索逻辑已全部迁入 `retrieval/` 包，旧的 `marxos_retrieval*.py` 兼容层已移除。

## 8. Refactor Triggers

出现下面任一信号时，默认认为“可以开始重构”，不再继续靠维护文档和口头约定硬撑：

1. 新增一个检索策略需要同时修改超过 2 个 `retrieval/*` 子模块，或还要回头补 `app.py` glue code。
2. `app.py` 和 `web_app.py` 出现超过 20 行的重复逻辑，且重复逻辑已经影响修复或测试节奏。
3. 维护者要解释的“例外规则”已经超出 `docs/maintenance_guide.md` 的既有导航，需要额外口头补充才能安全改动。
4. `scripts/check.py --mode quick` 已经不足以覆盖高频回归，开始频繁依赖人工挑选脚本组合来判断是否可合并。

这些信号的意义不是说明当前实现“失败了”，而是说明之前为速度做的透支已经到了该还款的时候。
