````markdown
# MarxOS

> MarxOS 是一个基于 OCR + RAG + Academic Retrieval 的马克思主义学术智能体。

---

# Project Overview

MarxOS 的目标是：

构建一个能够：

- 阅读马克思主义原著
- 理解理论内容
- 支持学术检索
- 提供页码级引用
- 进行理论解释与生成

的：

# Academic AI Knowledge System

---

# Core Features

## 1. OCR PDF Recognition

支持：

```text
扫描版 PDF
↓
OCR
↓
可检索文本
````

核心技术：

* PaddleOCR
* pdf2image
* Poppler

---

## 2. Academic RAG

支持：

```text
原著检索
+
LLM 学术生成
```

实现：

* 多文档检索
* 理论问答
* 学术解释
* Citation RAG

---

## 3. Metadata Engineering

当前支持：

```json
{
  "book": "马克思恩格斯全集 第49卷",
  "page": 3,
  "source": "me49.pdf",
  "ocr": true
}
```

支持：

* 页码级引用
* Academic Citation
* 多文档来源追踪

---

## 4. Multi-Document Knowledge Base

当前知识库已支持：

* 《马克思恩格斯全集》
* 《德意志意识形态》
* 《1857-1858年经济学手稿》

等文献联合检索。

---

# System Architecture

当前 MarxOS Pipeline：

```text
PDF
↓
OCR (PaddleOCR)
↓
Text Cleaning
↓
Document(metadata)
↓
Chunking
↓
Embedding
↓
FAISS
↓
Retriever
↓
DeepSeek
↓
Academic Answer
```

---

# Tech Stack

## AI / RAG

* LangChain
* FAISS
* Sentence Transformers
* DeepSeek API

---

## OCR

* PaddleOCR
* PaddlePaddle
* pdf2image

---

## Vector Database

* FAISS

---

## Environment

* Python 3.10

---

# Project Structure

```text
MarxOS/

├── app.py
├── README.md
├── requirements.txt
├── .env

├── crawler/
│   └── crawl_marxists.py

├── ocr/
│   └── pdf_to_text.py

├── rag/
│   ├── build_vectorstore.py
│   └── build_knowledge_base.py

├── vectorstore/
│   └── marx_knowledge_base/

├── data/
│   └── marx_engels全集/

├── docs/
│   └── dev_logs/
```

---

# Current Progress

当前已完成：

* OCR Pipeline
* Metadata System
* FAISS Knowledge Base
* Multi-Document RAG
* Academic Retrieval
* Citation RAG
* Chunk Engineering

当前知识库 chunk 数量：

```text
1421+
```

---

# Key Engineering Problems

## 1. Python 3.13/3.14 Compatibility

问题：

```text
PaddleOCR 与 Python 3.13+ 不兼容
```

解决：

```text
统一切换 Python 3.10
```

---

## 2. numpy / opencv ABI Conflict

问题：

```text
numpy.core.multiarray failed to import
```

解决：

```text
numpy==1.23.5
opencv-python==4.6.0.66
```

---

## 3. OCR Text Noise

问题：

OCR 输出存在：

* 换行混乱
* 空格异常
* 句子断裂

解决：

```python
clean_text()
```

进行文本清洗。

---

## 4. Chunk Semantic Pollution

问题：

```text
chunk_size=500
```

导致多个理论主题混入同一 chunk。

解决：

```python
chunk_size=300
chunk_overlap=50
```

并增加：

```python
separators=["。","？","！"]
```

实现中文语义切块。

---

# Roadmap

## Stage 3 — Retrieval Engineering

计划：

* Reranker
* BM25 Hybrid Search
* Query Rewrite
* Citation Grounding

---

## Stage 4 — Knowledge Engineering

计划：

* 自动章节识别
* 理论主题 metadata
* 学术知识图谱

---

## Stage 5 — Productization

计划：

* Streamlit Web UI
* 多用户系统
* Academic Research Assistant
* AI 学术论文辅助

---

# Engineering Reflection

MarxOS 当前最大的工程收获：

```text
AI 系统真正核心
不是模型本身
而是：

- 数据工程
- Knowledge Engineering
- Retrieval Engineering
```

---

# Future Vision

MarxOS 的长期目标：

```text
构建真正的：
马克思主义 AI 学术研究平台
```

未来将逐步支持：

* 原著研究
* 理论分析
* 学术论文辅助
* 知识图谱
* AI 学术助手
* Marxism Research Agent

```
```
