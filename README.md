# MarxOS

MarxOS 是一个面向马克思主义经典文本的本地检索问答项目。它把 PDF 原文先做 OCR 和清洗，再构建向量库，最后结合规则检索、引文校对和 LLM 生成，输出带出处的学术型回答。

## 项目目标

- 面向马克思、恩格斯经典著作的中文问答
- 尽量给出可核对的篇名、页码和引文出处
- 优先支持本地语料、本地向量库和离线式工作流
- 把检索质量、引文质量和回答质量都纳入回归检查

## 核心流程

```text
PDF
-> OCR cache
-> 文本清洗与页码识别
-> chunk / paragraph cache
-> embeddings
-> FAISS vectorstore
-> 检索 / rerank / citation refine
-> DeepSeek
-> 带引文回答
```

## 环境要求

- Python 3.10
- Windows PowerShell
- 已安装 `requirements.txt` 中依赖
- 如需完整 OCR，需本地 PaddleOCR / PaddlePaddle 环境可用

## 快速开始

1. 创建并激活虚拟环境

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. 配置 `.env`

```powershell
copy .env.example .env
```

最少需要配置：

```text
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

3. 启动命令行问答

```powershell
venv\Scripts\python.exe app.py
```

4. 启动网页端

```powershell
venv\Scripts\python.exe web_app.py
```

默认地址：

```text
http://127.0.0.1:7860
```

如端口冲突，可改端口：

```powershell
$env:MARXOS_WEB_PORT="7861"
venv\Scripts\python.exe web_app.py
```

## 开发调试

开启开发模式与 trace：

```powershell
$env:MARXOS_DEV_MODE="1"
$env:MARXOS_TRACE="1"
venv\Scripts\python.exe app.py
```

只跑检索与 prompt，不调用 DeepSeek：

```powershell
$env:MARXOS_DEV_MODE="1"
$env:MARXOS_TRACE_ONLY="1"
venv\Scripts\python.exe app.py
```

## 构建知识库

从 OCR cache 构建核心向量库：

```powershell
venv\Scripts\python.exe rag\build_vectorstore_from_cache.py
```

常用环境变量示例：

```powershell
$env:ME_VOLUMES_ONLY="1"
$env:SKIP_PDFS="capital.pdf"
$env:BATCH_SIZE="1024"
venv\Scripts\python.exe rag\build_vectorstore_from_cache.py
```

生成核心 article map 并构建核心库：

```powershell
$targets="mea01.pdf,mea02.pdf,mea03.pdf,mea04.pdf,mea05.pdf,mea06.pdf,mea07.pdf,mea08.pdf,mea09.pdf,mea10.pdf,mes01.pdf,mes02.pdf,mes03.pdf,mes04.pdf"
$env:TARGET_PDFS=$targets
$env:SKIP_PDFS=""
$env:ARTICLE_MAP_PATH="rag/article_map_core.json"
venv\Scripts\python.exe rag\generate_article_map.py

$env:VECTORSTORE_DIR="vectorstore/marx_reader_core"
$env:BATCH_SIZE="1024"
venv\Scripts\python.exe -u rag\build_vectorstore_from_cache.py
```

## 本地验证

快速回归：

```powershell
venv\Scripts\python.exe scripts\check.py --mode quick
```

`quick` 当前包含：

- `scripts/regression_smoke.py`
- `tests/test_run_query_regressions.py`

完整检查：

```powershell
venv\Scripts\python.exe scripts\check.py --mode full
```

`full` 在 `quick` 基础上增加：

- `scripts/evaluate_retrieval.py`
- `scripts/evaluate_citation_pages.py`
- `scripts/evaluate_eval_dataset.py`

也可以单独跑：

```powershell
$env:PYTHONIOENCODING="utf-8"
venv\Scripts\python.exe scripts\evaluate_retrieval.py
venv\Scripts\python.exe scripts\evaluate_eval_dataset.py
```

统一审计入口：

```powershell
venv\Scripts\python.exe scripts\audit.py list
venv\Scripts\python.exe scripts\audit.py page-metadata
venv\Scripts\python.exe scripts\audit.py exact-quote-top1
```

## 当前代码结构

主入口：

- `app.py`: CLI 主入口与总控编排
- `web_app.py`: Web API 与内嵌前端

核心模块：

- `marxos_runtime.py`: 运行时状态、向量库加载、开发模式开关
- `marxos_embeddings.py`: HuggingFace embeddings 兼容导入层
- `marxos_query_intent.py`: query intent 分类与路由辅助
- `marxos_citations.py`: 引文格式、证据抽取、citation audit
- `marxos_answers.py`: 本地回答分支、专题列表回答、拒答规则
- `marxos_prompts.py`: prompt builders 与回答风格规则
- `marxos_trace.py`: trace/debug/TRACE_ONLY

检索分层：

- `marxos_retrieval.py`: 对外 facade
- `marxos_retrieval_constraints.py`: title/topic/source constraints 与 seed queries
- `marxos_retrieval_ranking.py`: rerank、diversify、constraint annotation
- `marxos_retrieval_modes.py`: 实际检索执行、backstop、paragraph/dual retrieval、页码 refinement

RAG / OCR：

- `rag/ocr_to_cache.py`
- `rag/build_vectorstore_from_cache.py`
- `rag/paragraph_cache.py`
- `rag/exact_quote_lookup.py`
- `rag/core_classics.py`
- `rag/page_number_detection.py`
- `rag/clean_ocr_text.py`

测试与脚本：

- `scripts/check.py`: quick/full 统一入口
- `scripts/evaluate_*.py`: 评测脚本
- `scripts/audit.py`: 审计脚本统一入口
- `tests/*.py`: 单元测试与回归测试

## 重要数据路径

- `data/ocr_cache/`: OCR 缓存
- `vectorstore/marx_reader_core/`: 核心向量库
- `vectorstore/marx_reader_paragraph/`: 段落向量库
- `rag/article_map_core.json`: 核心篇目映射
- `rag/core_classics.json`: 核心经典目录
- `rag/topic_catalog.json`: 专题目录
- `eval_dataset.json`: 端到端评测集

`data/`、`vectorstore/`、日志与缓存目录一般不应直接提交到 Git。

## 已完成的结构整理

- 删除了历史残留脚本：
  - `ocr/pdf_to_text.py`
  - `rag/build_knowledge_base.py`
- `app.py` 已从超大实现文件收缩为总控入口
- prompt、trace、citations、answers、retrieval、runtime 已拆出独立模块
- retrieval 已继续细分为 `constraints / ranking / modes`
- 审计脚本已有统一入口 `scripts/audit.py`

## 当前已知事项

- 第三方依赖 `faiss` / `setuptools` 仍会输出少量弃用 warning，这不是项目主逻辑错误
- OCR、页码推断和 article map 仍然决定最终引文质量，语料侧问题不会被 LLM 自动修复
- `data/`、`vectorstore/`、本地日志体积较大，清理和备份应单独管理

## 推荐阅读

- [docs/codebase_inventory.md](C:/Users/Administrator/Desktop/MarxOS/docs/codebase_inventory.md)
- [docs/architecture.md](C:/Users/Administrator/Desktop/MarxOS/docs/architecture.md)
- [docs/maintenance_guide.md](C:/Users/Administrator/Desktop/MarxOS/docs/maintenance_guide.md)
- [docs/eval_questions.md](C:/Users/Administrator/Desktop/MarxOS/docs/eval_questions.md)
- [docs/dev_logs/README.md](C:/Users/Administrator/Desktop/MarxOS/docs/dev_logs/README.md)
- [scripts/README.md](C:/Users/Administrator/Desktop/MarxOS/scripts/README.md)

## PowerShell 编码

如 PowerShell 中文显示异常，可先执行：

```powershell
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::UTF8
```

## Grouped Test Entry

Use `scripts/test.py` when you want to run unittest suites by responsibility:

```powershell
venv\Scripts\python.exe scripts\test.py list
venv\Scripts\python.exe scripts\test.py app
venv\Scripts\python.exe scripts\test.py web
venv\Scripts\python.exe scripts\test.py rag
venv\Scripts\python.exe scripts\test.py all
```

See also: [docs/testing_guide.md](C:/Users/Administrator/Desktop/MarxOS/docs/testing_guide.md)

## Optional Arize Phoenix Tracing

MarxOS supports optional Arize Phoenix tracing for the main `run_query(...)`
pipeline. The integration is disabled by default and becomes active only when
`MARXOS_PHOENIX_ENABLED=1`.

Install the optional tracing dependencies with:

```powershell
venv\Scripts\pip.exe install -r requirements-phoenix.txt
```

If you want a local Phoenix server, a common setup is:

```powershell
venv\Scripts\python.exe -m phoenix.server.main serve
```

Recommended environment variables:

```powershell
$env:MARXOS_PHOENIX_ENABLED="1"
$env:MARXOS_PHOENIX_AUTO_INSTRUMENT="1"
$env:MARXOS_PHOENIX_PROJECT_NAME="MarxOS"
$env:MARXOS_PHOENIX_SERVICE_NAME="marxos"
$env:PHOENIX_COLLECTOR_ENDPOINT="http://127.0.0.1:6006/v1/traces"
```

If you use Phoenix Cloud, also set:

```powershell
$env:PHOENIX_API_KEY="your_api_key"
```

Then start MarxOS in the same shell:

```powershell
$env:MARXOS_PHOENIX_ENABLED="1"
venv\Scripts\python.exe app.py
```

The current tracing hooks cover:

- request preparation and routing
- local lookup and local view answer branches
- retrieval constraints and top retrieved docs
- prompt construction
- DeepSeek generation through the OpenAI-compatible client
- citation audit and filtered evidence

If the OpenTelemetry or OpenInference packages are not installed, MarxOS will
continue to run normally and skip exporting traces.
