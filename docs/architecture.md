# 架构概览

这份文档不是文件清单，而是主链路说明。它回答的是三个问题：

1. 用户问题是怎么流过系统的
2. 哪些模块在主线上
3. 哪些地方最容易改坏，需要优先回归

## 1. 主流程

整体链路可以概括成：

```text
用户问题
-> query intent 判断
-> retrieval constraints
-> chunk / paragraph 检索
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

- [marxos_web_support.py](../marxos_web_support.py)
  - Web 层公共辅助函数
  - 包括 metrics、history 压缩、follow-up 判断和响应 payload 拼装

- [marxos_web_followups.py](../marxos_web_followups.py)
  - Web 层专题 follow-up 规则
  - 包括专题追问改写、条目解释、专题历史追问整理

- [marxos_web_citations.py](../marxos_web_citations.py)
  - Web 层引文与页码 follow-up 规则
  - 包括脚注解析、OCR 页定位、页码追问和原页摘录

### 流程编排

- [marxos_orchestration.py](../marxos_orchestration.py)
  - `run_query(...)` 的流程胶水
  - 包括输入预处理、本地查答分支、检索材料收集、本地列表题分支

### 运行时与依赖

- [marxos_runtime.py](../marxos_runtime.py)
  - vectorstore / paragraph vectorstore 加载
  - dev/trace/dual retrieval 开关

- [marxos_embeddings.py](../marxos_embeddings.py)
  - 统一 embedding 导入和兼容层

### 路由与回答

- [marxos_query_intent.py](../marxos_query_intent.py)
  - bibliographic / quote / concept / analysis / rag 路由判断

- [marxos_answers.py](../marxos_answers.py)
  - 本地回答拼装
  - 列表题、专题题、拒答规则、摘录整理

- [marxos_prompts.py](../marxos_prompts.py)
  - prompt builders

### 检索与引文

- [retrieval/__init__.py](../retrieval/__init__.py)
  - facade 层，对外暴露检索公共 API（41 符号）

- [retrieval/constraints.py](../retrieval/constraints.py)
  - 标题/专题/概念约束、seed queries

- [retrieval/ranking.py](../retrieval/ranking.py)
  - rerank、diversify、dedup、topic selection

- [retrieval/modes.py](../retrieval/modes.py)
  - hybrid (dense+BM25) retrieval、strict-title backstop、paragraph/dual retrieval、citation refinement

- [marxos_citations.py](../marxos_citations.py)
  - 引文格式
  - evidence cards
  - final answer citation audit

- [marxos_trace.py](../marxos_trace.py)
  - trace 输出和 TRACE_ONLY 调试

## 3. 数据与构建链路

### OCR / cache / vectorstore

- `data/ocr_cache/`
  - OCR 后的文本缓存

- [rag/ocr_to_cache.py](../rag/ocr_to_cache.py)
  - PDF 文本层提取和扫描页 OCR

- [rag/build_vectorstore_from_cache.py](../rag/build_vectorstore_from_cache.py)
  - 从 cache 构建 chunk 向量库

- [scripts/build_paragraph_cache.py](../scripts/build_paragraph_cache.py)
  - 段落级 cache

- [scripts/build_paragraph_vectorstore.py](../scripts/build_paragraph_vectorstore.py)
  - 段落级向量库

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
