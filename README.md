# MarxOS

MarxOS 是一个基于 OCR、RAG 和 Academic Retrieval 的马克思主义学术智能体，目标是构建可检索、可引用、可扩展的马克思主义原著知识系统。

## Project Overview

MarxOS 希望支持：

- 阅读扫描版马克思主义原著 PDF
- 将 OCR 文本转换为可检索知识库
- 基于原著进行学术问答
- 提供页码级引用和篇目定位
- 支持概念解释、引文查找和理论分析

## Core Pipeline

```text
PDF
-> OCR cache
-> Text cleaning
-> Document(metadata)
-> Chunking
-> Embedding
-> FAISS
-> Retriever
-> DeepSeek
-> Academic answer
```

## Current Progress

当前已经完成：

- OCR Pipeline
- OCR cache 复用
- FAISS Knowledge Base
- Multi-Document RAG
- Metadata Engineering
- Citation RAG
- Article / catalog mapping
- Query intent classification
- Academic citation formatting

本地已生成的核心资产包括：

```text
data/ocr_cache/
vectorstore/marx_knowledge_base/
rag/article_map.json
```

注意：`data/` 和 `vectorstore/` 体积较大，默认不会提交到 Git。

## Quick Start

建议使用 Python 3.10。PaddleOCR / PaddlePaddle 在较新的 Python 版本上容易出现 ABI 或 DLL 兼容问题。

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

复制环境变量示例：

```powershell
copy .env.example .env
```

然后在 `.env` 中填写：

```text
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

运行问答入口：

```powershell
venv\Scripts\python.exe app.py
```

如需查看内部检索来源：

```powershell
$env:MARXOS_DEBUG_SOURCES="1"
venv\Scripts\python.exe app.py
```

## Build Knowledge Base

如果已经有 OCR 缓存，可以直接从缓存构建向量库：

```powershell
venv\Scripts\python.exe rag\build_vectorstore_from_cache.py
```

常用过滤参数：

```powershell
$env:ME_VOLUMES_ONLY="1"
$env:SKIP_PDFS="capital.pdf"
$env:BATCH_SIZE="1024"
venv\Scripts\python.exe rag\build_vectorstore_from_cache.py
```

构建结果会写入：

```text
vectorstore/marx_knowledge_base/
```

## Project Structure

```text
MarxOS/
├── app.py
├── requirements.txt
├── crawler/
│   └── crawl_marxists.py
├── ocr/
│   └── pdf_to_text.py
├── rag/
│   ├── article_map.json
│   ├── build_vectorstore_from_cache.py
│   ├── generate_article_map.py
│   ├── ocr_to_cache.py
│   └── repair_vectorstore_metadata.py
├── docs/
│   ├── eval_questions.md
│   └── dev_logs/
├── data/              # local only
└── vectorstore/       # local only
```

## Evaluation

固定评测问题见：

```text
docs/eval_questions.md
```

每次修改检索、metadata、prompt、reranker 或引用格式后，建议用这组问题做一次回归检查。

## Known Issues

- `app.py` 目前聚合了检索、意图识别、引用格式、prompt 构建和 LLM 调用，后续应拆分模块。
- `article_map.json` 仍依赖启发式目录抽取，需要对重要著作做人工抽样校验。
- 页码 metadata 受 OCR 质量影响，部分印刷页码仍需复核。
- `data/`、`vectorstore/` 不提交到 Git，换机器时需要重新生成或从本地备份恢复。

## Roadmap

### Stage 3: Retrieval Engineering

- BM25 Hybrid Search
- Reranker
- Query Rewrite
- Citation Grounding
- Fixed evaluation workflow

### Stage 4: Knowledge Engineering

- 自动章节识别
- 理论主题 metadata
- 重点著作人工校验表
- 学术知识图谱

### Stage 5: Productization

- Web UI
- Academic Research Assistant
- 多用户系统
- 学术论文辅助

## PowerShell Encoding

如果 PowerShell 中看到 README 或日志中文乱码，优先设置 UTF-8 输出：

```powershell
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::UTF8
```

文件本身应以 UTF-8 保存。
