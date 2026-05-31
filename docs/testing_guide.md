# MarxOS Testing Guide

这份文档用于说明现在项目里的测试入口应该怎么选。

## 1. 快速判断跑哪组

- 改 `app.py`、问答主链路、拒答、引文格式：跑 `app`
- 改 `web_app.py`、`/api/ask`、多轮问答、metrics：跑 `web`
- 改 `rag/`、书目、页码推断、paragraph cache：跑 `rag`
- 想整体过一遍 unittest：跑 `all`

## 2. 分组 unittest 入口

```powershell
venv\Scripts\python.exe scripts\test.py list
venv\Scripts\python.exe scripts\test.py app
venv\Scripts\python.exe scripts\test.py web
venv\Scripts\python.exe scripts\test.py rag
venv\Scripts\python.exe scripts\test.py all
```

## 3. 和 `check.py` 的关系

- `scripts\test.py` 用于按职责跑单元测试。
- `scripts\check.py --mode quick` 用于日常快速回归。
- `scripts\check.py --mode full` 会在 `quick` 基础上继续跑检索、页码、数据集评测。

## 4. 当前测试分组

### `app`

- `tests/test_run_query_regressions.py`
- `tests/test_app_local_paths.py`

### `web`

- `tests/test_web_api.py`

### `rag`

- `tests/test_core_bibliography.py`
- `tests/test_page_metadata_inference.py`
- `tests/test_paragraph_cache.py`

## 5. 当前已知特点

- `app` 组不是秒级测试，当前完整跑完大约需要几分钟。
- 某些测试会触发本地模型或向量库加载，因此第一次运行通常比后续更慢。
- 现在已经把 `TRACE_ONLY` 测试里的默认 stderr 噪声压下去了，`quick` 输出会更干净。
