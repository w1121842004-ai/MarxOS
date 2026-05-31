# tests 目录说明

当前测试更适合按职责来理解。

## `app`

覆盖主问答入口、本地直答、拒答、引文和检索回归：

- `test_run_query_regressions.py`
- `test_app_local_paths.py`

运行：

```powershell
venv\Scripts\python.exe scripts\test.py app
```

## `web`

覆盖 `/api/ask` 的 payload、evidence 和 metrics 行为：

- `test_web_api.py`

运行：

```powershell
venv\Scripts\python.exe scripts\test.py web
```

## `rag`

覆盖书目目录、页码 metadata 推断、paragraph cache 等基础逻辑：

- `test_core_bibliography.py`
- `test_page_metadata_inference.py`
- `test_paragraph_cache.py`

运行：

```powershell
venv\Scripts\python.exe scripts\test.py rag
```

## `all`

运行全部 unittest：

```powershell
venv\Scripts\python.exe scripts\test.py all
```

## 和 `check.py` 的关系

- `scripts\test.py` 适合按模块做定向回归。
- `scripts\check.py --mode quick` 适合日常小改后的快速检查。
- `scripts\check.py --mode full` 适合检索、引文、评测相关改动后的完整检查。
