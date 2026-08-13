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

CI 还会运行版本化的离线 Web 契约烟测：

```powershell
venv\Scripts\python.exe scripts\run_web_smoke_ci.py --report logs\web_smoke_ci.json
```

数据集固定在 `tests/fixtures/web_smoke_v1.json`。该入口 mock 掉 LLM 调用，不加载模型、
Milvus 或机器私有索引，适合 Windows/macOS CI。`scripts/run_web_smoke.py` 保留为部署环境的
真实端到端烟测，二者不可互相替代。

## 3. 和 `check.py` 的关系

- `scripts\test.py` 用于按职责跑单元测试。
- `scripts\check.py --mode quick` 用于日常快速回归；当前实际包含 `validate_maps`、`regression_smoke`、`scripts/test.py app`。
- `scripts\check.py --mode full` 会在 `quick` 基础上继续跑检索、页码、数据集评测。
- `quick` 当前不包含 `web` 组；改 `web_app.py`、多轮追问、前端 payload 或 evidence 显示时，必须额外跑 `scripts/test.py web`。

## 4. 当前测试分组

### `app`

- `tests/test_run_query_regressions.py`
- `tests/test_app_local_paths.py`

### `web`

- `tests/test_web_api.py`
- `tests/test_web_smoke_runner.py`

### `rag`

- `tests/test_core_bibliography.py`
- `tests/test_page_metadata_inference.py`
- `tests/test_paragraph_cache.py`

## 5. 当前已知特点

- `app` 组不是秒级测试，当前完整跑完大约需要几分钟。
- 某些测试会触发本地模型或向量库加载，因此第一次运行通常比后续更慢。
- 现在已经把 `TRACE_ONLY` 测试里的默认 stderr 噪声压下去了，`quick` 输出会更干净。
- 当前默认后端是 Milvus Lite + BGE-M3 hybrid；如果本地 `data/milvus_lite/*.db` 或 HuggingFace 模型缓存缺失，相关测试和评测会先失败在加载阶段。
- 不建议直接把 `python -m unittest discover -s tests` 当作日常快速基线；当前全量 discover 可能触发真实 LLM / citation verifier 调用。
- 在受限沙箱中，`test_web_api.py` 里真实绑定 `ThreadingHTTPServer(("127.0.0.1", 0), ...)` 的用例可能报 `PermissionError: [Errno 1] Operation not permitted`。这种失败说明当前环境禁止 socket bind，不等同于 Web API 逻辑失败。

## 6. 稳定化阶段推荐组合

文档或纯前端样式改动：

```powershell
venv\Scripts\python.exe scripts\test.py web
```

Web API、多轮追问、证据卡显示改动：

```powershell
venv\Scripts\python.exe scripts\test.py web
venv\Scripts\python.exe scripts\check.py --mode quick
```

检索、引文、页码、语料 metadata 改动：

```powershell
venv\Scripts\python.exe scripts\check.py --mode full
```

初步部署前：

```powershell
venv\Scripts\python.exe scripts\test.py all
venv\Scripts\python.exe scripts\check.py --mode full
```

## 7. 当前待修复的测试基础设施问题

- `scripts/validate_maps.py` 直接运行时需要保证项目根目录在 `sys.path` 中，否则会在导入 `marxos.config` 时失败。
- 全量 unittest discover 需要避免默认触发真实 LLM 调用；外部 API 相关用例应要求显式环境变量。
- Web socket 绑定测试需要在受限环境下有替代路径或 skip 机制。

## 8. 2026-08-12 基线

- `tests/test_run_query_regressions.py`：8/8 通过。
- `tests/test_web_api.py`：2/2 通过（需要允许本机临时端口）。
- `scripts/test.py all`：未通过；在 `test_app_local_paths.py` 报告 15 failures / 2 errors（该文件共运行 69 tests）后停止。
- 在清零这些失败前，仓库不满足初步部署门禁；不得把聚焦测试通过描述为“全量测试通过”。

## 9. 2026-08-13 基线（P2 修复后）

- 默认 profile：`me_full_v2` + `milvus_bgem3_v2`。
- `scripts/test.py all`：全绿（app/web/rag 三组；`test_app_local_paths.py` 69/69）。
- `scripts/regression_smoke.py`：8/8。
- `scripts/evaluate_retrieval.py`：92/95（基线 91/95，无回退）。
- `scripts/run_web_smoke_ci.py`：20/20；`scripts/run_web_smoke.py`：5/5。
- `scripts/check.py --mode quick`：全绿（validate_index_manifest、validate_maps、regression_smoke、test.py p0）。
- 运行 macOS 本地进程时保持 `OMP_NUM_THREADS=1`（已内置到入口与启动脚本），否则 torch + Milvus Lite 检索会段错误。
- 检索约束/模式优先级与三路检索分开度量见 `docs/retrieval_priority.md`。
