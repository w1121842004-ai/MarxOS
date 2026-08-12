# MarxOS Maintenance Guide

这份文档面向后续维护者，重点回答一个问题：
“我要改某类行为时，应该先看哪里？”

## 1. 先判断改动属于哪一层

### CLI / 主流程编排

优先看：

- `app.py`
- `marxos/app/orchestration.py`
- `marxos/runtime.py`

适用场景：

- 调整 `run_query(...)` 主链路
- 改开发模式、trace-only、离线路径
- 调整 CLI 入口行为

### 检索策略

优先看：

- `retrieval/__init__.py`
- `retrieval/constraints.py`
- `retrieval/ranking.py`
- `retrieval/modes.py`

适用场景：

- 改标题约束、专题约束、来源约束
- 调整召回、rerank、diversify
- 调整 strict-title backstop
- 调整 paragraph / dual retrieval / citation-page refinement

### 回答生成与拒答

优先看：

- `marxos/generation/answers.py`
- `marxos/generation/prompts.py`
- `marxos/generation/citations.py`

适用场景：

- 调整“无答案时不乱答”的规则
- 改回答结构、语气、引用注释格式
- 改证据抽取、引文校验、citation audit

### Web API / 前端交互

优先看：

- `web_app.py`
- `marxos/web/support.py`
- `marxos/web/followups.py`
- `marxos/web/citations.py`

适用场景：

- 调整 `/api/ask` 返回 payload
- 改 topic follow-up、多轮历史整理
- 改 citation/page follow-up
- 改 metrics 记录与 evidence 展示

### OCR / 语料 / 向量库

优先看：

- `rag/ocr_to_cache.py`
- `rag/clean_ocr_text.py`
- `rag/page_number_detection.py`
- `rag/build_vectorstore_from_cache.py`
- `rag/paragraph_cache.py`

适用场景：

- OCR 文本清洗异常
- printed page / pdf page 映射异常
- chunk 或 paragraph cache 构建异常
- 向量库 metadata 缺失或不稳定

## 2. 常见修改任务对照表

### 想改“某类问题应该走哪条路由”

先看：

- `marxos/query_intent.py`
- `app.py`

补充检查：

- `tests/test_run_query_regressions.py`
- `tests/test_web_api.py`

### 想改“检索结果不准/召回不稳”

先看：

- `retrieval/constraints.py`
- `retrieval/ranking.py`
- `retrieval/modes.py`

建议验证：

- `venv\Scripts\python.exe scripts\evaluate_retrieval.py`
- `venv\Scripts\python.exe scripts\audit.py exact-quote-top1`

### 想改“回答格式或拒答规则”

先看：

- `marxos/generation/answers.py`
- `marxos/generation/prompts.py`
- `marxos/generation/citations.py`

建议验证：

- `venv\Scripts\python.exe scripts\check.py --mode quick`
- `venv\Scripts\python.exe scripts\evaluate_eval_dataset.py`

### 想改“网页端返回字段或多轮问答行为”

先看：

- `web_app.py`
- `marxos/web/support.py`
- `marxos/web/followups.py`
- `marxos/web/citations.py`

建议验证：

- `venv\Scripts\python.exe -m unittest discover -s tests -p test_web_api.py`

### 想改“页码、原页摘录、引文定位”

先看：

- `marxos/generation/citations.py`
- `marxos/web/citations.py`
- `rag/page_number_detection.py`
- `data/page_map.json`

建议验证：

- `venv\Scripts\python.exe scripts\evaluate_citation_pages.py`
- `venv\Scripts\python.exe scripts\audit.py page-metadata`

## 3. 推荐的维护顺序

做改动时，尽量按下面顺序推进：

1. 先确认问题属于哪一层，不要一上来就在 `app.py` 和 `web_app.py` 里直接堆逻辑。
2. 优先改对应职责模块，只把必要的 glue code 留在入口层。
3. 至少补一条能覆盖该行为的测试或回归脚本验证。
4. 如果改动影响检索或引文质量，再跑专项评测或审计脚本。

## 4. 最小验证建议

### 日常小改

```powershell
venv\Scripts\python.exe scripts\check.py --mode quick
```

### Web 层改动

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -p test_web_api.py
```

### 检索 / 引文改动

```powershell
venv\Scripts\python.exe scripts\check.py --mode full
```

### 只想看有哪些专项审计

```powershell
venv\Scripts\python.exe scripts\audit.py list
```

## 5. 维护时的几个边界

- `app.py` 和 `web_app.py` 现在已经更偏“入口层”，后续尽量不要把大段业务逻辑重新塞回去。
- `retrieval/__init__.py` 是 facade；新的检索细节优先落到 `retrieval/constraints / ranking / modes` 子模块。
- 语料、OCR、页码问题通常不是 LLM 能自动补救的，涉及引用质量时优先先查 `rag/` 和 `data/`。
- 历史 `docs/dev_logs/` 更适合追溯背景，不适合作为当前行为规范；当前规范以 `README.md`、`docs/architecture.md`、本文件为准。
