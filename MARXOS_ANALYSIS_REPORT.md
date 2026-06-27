# MarxOS 系统分析报告

> 重写日期：2026-06-23  
> 项目路径：`/Users/HONOR/Desktop/AIproject/MarxOS`  
> 分析对象：本地代码库、README、架构文档、测试与脚本入口

---

## 1. 执行摘要

MarxOS 是一个面向中文马克思主义经典文本的本地检索增强生成系统。系统目标不是做通用聊天，而是围绕马克思、恩格斯原著提供可核验出处的学术问答：用户提出概念、篇目、引文或理论分析问题，系统从本地 OCR 语料和向量库中检索证据，再由 DeepSeek 等 OpenAI 兼容接口生成回答，并通过引文审计减少“有回答、无依据”的风险。

当前项目已经形成较清晰的工程主线：

```text
PDF 原文
-> OCR 缓存
-> 文本清洗 / 页码识别
-> 段落缓存
-> 子块向量库 / 段落向量库 / 可选 Milvus 集合
-> 约束检索 / 混合检索 / 重排序
-> Evidence Cards
-> Prompt + LLM
-> 引文修复与审计
-> CLI / Web 输出
```

从代码结构看，项目已经从早期的“大入口文件 + 分散脚本”演进为分层系统：`app.py` 仍是核心总控，但运行时、Prompt、引文、检索、Web 支持、OCR/RAG 构建和评测脚本已经拆分为独立模块。系统的主要技术亮点是“小块检索、大块召回”、约束感知检索、精确引文路径、CRAG 式纠正性检索和回答后的引文审计。

系统当前也存在几个需要继续收口的问题：`app.py` 仍偏大；FAISS、Milvus、BM25、段落向量库之间的运行主线需要在文档和配置层进一步明确；语料质量、页码映射、篇目定位仍是最终回答质量的瓶颈；测试覆盖聚焦核心路径，但对完整数据构建、Web 端交互和真实 LLM 调用的自动化覆盖仍有限。

---

## 2. 项目定位

### 2.1 系统定义

MarxOS 可以定义为：

**一个面向马克思主义经典文本的本地知识检索与学术问答系统。**

它和普通 RAG 项目的关键差异在于：

| 维度 | MarxOS 的侧重点 |
| --- | --- |
| 语料 | 以马克思、恩格斯中文经典文本为主，依赖本地 OCR 与目录元数据 |
| 输出 | 强调篇名、卷册、页码、证据卡和引文可核验性 |
| 检索 | 同时使用语义向量、BM25、篇目约束、页码约束和标题定位 |
| 质量 | 检索质量门、引文审计、回答合同评估共同约束输出 |
| 部署 | 优先本地运行，Web/CLI 双入口，可选接入 Milvus 与 Phoenix |

### 2.2 主要用户场景

系统最适合的场景包括：

- 查找某篇著作所在卷册、篇名和页码范围。
- 核验某句话是否出自马克思、恩格斯原著。
- 解释“异化劳动”“商品拜物教”“唯物史观”等核心概念。
- 围绕某一主题汇总多篇原著证据。
- 生成带出处的学术型分析回答。

它不适合作为无依据的开放式政治评论工具，也不应在证据不足时强行输出确定结论。当前 Prompt 和本地回答逻辑已经在这方面加入了拒答、材料不足提示和引文约束。

---

## 3. 总体架构

### 3.1 分层视图

```text
┌────────────────────────────────────────────┐
│  用户入口层                                 │
│  CLI: app.py                                │
│  Web: web_app.py + /api/ask                 │
└──────────────────────┬─────────────────────┘
                       │
┌──────────────────────▼─────────────────────┐
│  编排层                                     │
│  run_query / intent / planner / CRAG / LLM  │
│  app.py + marxos_orchestration.py           │
└──────────────────────┬─────────────────────┘
                       │
┌──────────────────────▼─────────────────────┐
│  检索层                                     │
│  constraints / modes / ranking              │
│  Dense + BM25 + Hybrid + backstop           │
└──────────────────────┬─────────────────────┘
                       │
┌──────────────────────▼─────────────────────┐
│  数据与索引层                               │
│  OCR cache / paragraph cache / FAISS / Milvus│
└──────────────────────┬─────────────────────┘
                       │
┌──────────────────────▼─────────────────────┐
│  语料构建与质量层                           │
│  OCR、段落构建、向量构建、评测、审计脚本       │
└────────────────────────────────────────────┘
```

### 3.2 核心模块职责

| 模块 | 职责 | 评价 |
| --- | --- | --- |
| `app.py` | CLI 入口、主查询链路、上下文构建、运行期 glue code | 功能完整，但仍是最大的复杂度集中点 |
| `web_app.py` | 自定义 HTTP 服务、内嵌前端、`/api/ask` | 轻量直接，适合本地产品形态 |
| `marxos_runtime.py` | 向量库与运行状态缓存、开发模式开关 | 让入口文件不直接承担所有加载逻辑 |
| `marxos_query_intent.py` | 查询意图分类和路由辅助 | 规则主导，响应快，可解释 |
| `marxos_query_planner.py` | 查询分解、多查询、HyDE 等规划能力 | 适合复杂问题扩展 |
| `retrieval/constraints.py` | 标题、篇目、主题、页码等约束构建 | 检索准确性的关键模块 |
| `retrieval/modes.py` | Dense、BM25、Hybrid、backstop、引文页 refinement | 实际检索执行核心 |
| `retrieval/ranking.py` | 重排序、去重、多样化、约束标注 | 控制上下文质量和证据分布 |
| `marxos_citations.py` | 证据卡、引文格式、回答审计和修复 | 系统可信度的关键模块 |
| `marxos_answers.py` | 本地规则回答、拒答、专题列表回答 | 减少不必要的 LLM 调用 |
| `rag/` | OCR、文本清洗、段落缓存、精确引文、语义召回 | 数据资产和 RAG 能力的底座 |
| `scripts/` | 构建、评测、审计、质量门禁 | 工程化程度较高 |

---

## 4. 数据与知识库管线

### 4.1 数据资产

当前项目围绕三类核心数据展开：

| 类型 | 路径 | 作用 |
| --- | --- | --- |
| OCR 缓存 | `data/ocr_cache/` | 每页 OCR 文本、页码和基础元数据 |
| 段落缓存 | `data/paragraph_cache_core.jsonl` | 自然段落级记录，是子块和段落索引的共同来源 |
| 向量库 | `data/milvus_lite/marxos_bgem3_sparse.db` | 默认 Milvus Lite 索引，包含 BGE-M3 dense 与 sparse 向量 |
| FAISS fallback | `vectorstore/marx_reader_core/`、`vectorstore/marx_reader_paragraph/` | 子块级和段落级 FAISS 索引，保留作离线回退 |
| 结构化目录 | `rag/core_classics.json`、`rag/article_map*.json`、`rag/work_catalog.json` | 书目、篇目、主题和著作元数据 |
| 页码映射 | `data/page_map.json` | PDF 页与印刷页、引用页之间的桥接 |
| 评测集 | `eval_dataset*.json` | 检索、元数据、端到端质量评估 |

### 4.2 V2 构建主线

项目 README 明确推荐 V2 管线：

```text
Step 1: OCR cache -> paragraph cache
  scripts/build_paragraph_cache.py

Step 2: paragraph cache -> semantic child vectorstore
  scripts/build_semantic_child_vectorstore.py
  默认子块大小 180 chars，overlap 40

Step 3: paragraph cache -> paragraph vectorstore
  scripts/build_paragraph_vectorstore.py
```

V1 脚本 `rag/build_vectorstore_from_cache.py` 仍保留工具函数，但作为直接从 OCR 页切块建库的主流程已经废弃。废弃原因很明确：旧 chunk 缺少 `parent_paragraph_id`，无法通过父段落窗口扩展拿回完整上下文。

### 4.3 小块检索、大块召回

这是当前系统最重要的 RAG 设计：

```text
段落
-> 切成 180 字左右 semantic child chunks
-> child chunk 进入向量库
-> 查询命中 child chunk
-> 根据 parent_paragraph_id 找回所属段落
-> 扩展前后段落窗口
-> 把完整上下文送入 Evidence Cards 和 LLM
```

该方案同时解决两个常见问题：

- 如果直接用大块 embedding，语义会被稀释，精确概念或短引文命中率下降。
- 如果只把小块交给 LLM，回答容易断章取义，引文上下文不足。

MarxOS 通过“检索阶段小、生成阶段大”的策略，在召回精度和上下文完整性之间取得较好平衡。

### 4.4 Milvus 演进线

代码库已经加入 `marxos_vector_backend.py` 和 `scripts/build_milvus_collection.py`，并提供 `docker-compose.milvus.yml`。当前默认检索主线已经切换到 Milvus Lite + BGE-M3 dense/sparse hybrid。

当前应将 Milvus 理解为“默认主检索/扩展后端”，FAISS 保留为可再生成的离线 fallback：

| 后端 | 当前角色 |
| --- | --- |
| Milvus Lite | 默认向量后端，使用 BGE-M3 dense+sparse 与 RRF hybrid |
| FAISS | 离线 fallback，适合回归对照和极简本地运行 |
| BM25 / local sparse | 旧本地稀疏补充路径，默认不与 Milvus hybrid 叠加 |

---

## 5. 查询处理链路

### 5.1 主流程

```text
用户问题
-> 意图识别
-> 查询规划 / 多查询扩展
-> 约束构建
-> 检索与候选扩展
-> 重排序、去重、多样化
-> CRAG 检索质量评估
-> Evidence Cards
-> Prompt 构建
-> DeepSeek / OpenAI-compatible LLM
-> 引文审计与修复
-> CLI 或 Web 响应
```

### 5.2 意图识别

系统支持的核心意图包括书目查找、引文定位、概念解释、比较分析、理论分析、深度分析和通用 RAG 问答。意图识别主要由规则完成，并辅以训练脚本和轻量分类器相关文件。

这种方式符合项目当前需求：学术查询的意图模式相对稳定，规则分类可解释、低成本、离线可用，也便于在测试中固定行为。

### 5.3 约束构建

约束构建是 MarxOS 检索效果的关键。系统不只把用户问题丢给向量库，而是尝试从问题中抽取：

- 具体著作标题或别名。
- 篇目、章节、卷册、页码范围。
- 主题或概念。
- 经典著作目录中的条目。
- 文章定位器、书信定位器和高精度定位器中的候选。

这些约束会影响检索过滤、候选加权、重排序和后续引用页修正。对马克思主义经典文本这种“标题相近、篇目众多、卷册复杂”的语料来说，结构化约束比单纯向量相似度更重要。

### 5.4 检索与排序

当前检索层由 `retrieval/` 包承担，主要策略包括：

| 策略 | 作用 |
| --- | --- |
| Dense 检索 | 从向量库中找语义相近段落或子块 |
| BM25 检索 | 捕捉标题、人名、短语和精确术语 |
| Hybrid 融合 | 通过融合策略合并 Dense 与 Sparse 候选 |
| Strict-title backstop | 当明确标题查询失效时，用标题约束兜底 |
| 段落窗口扩展 | 从 child chunk 回捞完整段落上下文 |
| 多样化 | 限制同一来源、同一篇目过度占据上下文 |
| citation refinement | 对引用页、印刷页、PDF 页进行修正 |

### 5.5 CRAG 纠正性检索

系统引入了检索质量评分和纠正性检索循环。当初始证据质量不足时，会尝试更宽的查询、更大的候选池、段落向量补充或其他 backstop 方案，再重新评分。

这对中文经典文本问答很实用，因为许多问题会同时包含概念词、历史语境和著作标题，第一次检索可能被某个关键词带偏。CRAG 循环提供了有限的自修复能力。

---

## 6. 生成、证据与引文审计

### 6.1 Prompt 体系

`marxos_prompts.py` 按不同意图构建 Prompt，包括引文、概念、理论分析、深度分析和默认 RAG。Prompt 共同强调：

- 必须基于提供的材料回答。
- 引文需要对应证据。
- 不能编造卷册、页码和篇名。
- 材料不足时要说明不足。
- 信件和低置信页码场景需要特殊处理。

### 6.2 Evidence Cards

在进入 LLM 之前，系统会把检索文档转成证据卡。证据卡承担两个角色：

- 给 LLM 提供可读上下文和出处。
- 给回答后的引文审计提供可匹配的证据编号、篇名和页码。

这比简单拼接文档片段更适合学术问答，因为回答需要同时满足“内容相关”和“出处可核验”。

### 6.3 引文审计与修复

`marxos_citations.py` 和 `marxos_citation_verifier.py` 构成回答后的质量控制层。主要检查包括：

- 回答中的证据编号是否存在。
- 行内引文是否能匹配证据卡。
- 页码标签是否可信。
- 引文格式是否符合项目约定。
- 必要时调用 LLM 对照 OCR 原文验证引用内容。

审计失败时，系统可以尝试修复回答引用或触发恢复流程。这个机制是 MarxOS 区别于普通问答 Demo 的核心可信度设计。

---

## 7. Web 与 CLI 应用

### 7.1 CLI

`app.py` 是命令行入口，也暴露 `run_query()` 供 Web 层调用。CLI 适合开发调试、批量评估和本地直接问答。

常用模式包括：

| 模式 | 环境变量 |
| --- | --- |
| 开发模式 | `MARXOS_DEV_MODE=1` |
| Trace | `MARXOS_TRACE=1` |
| 只跑检索和 Prompt，不调 LLM | `MARXOS_TRACE_ONLY=1` |
| 混合检索 | `MARXOS_HYBRID_RETRIEVAL=1` |
| 双向量检索 | `MARXOS_DUAL_RETRIEVAL=1` |

### 7.2 Web 应用

`web_app.py` 使用标准库 `http.server.ThreadingHTTPServer` 实现本地 Web 服务，而不是 Gradio 或 FastAPI。它提供：

- `GET /`：返回内嵌 HTML/CSS/JS 页面。
- `POST /api/ask`：接收 `{query, history, mode}`，返回回答、意图、证据和指标。
- `MARXOS_WEB_PORT`：默认端口 `7860`。
- `logs/api_ask_metrics.jsonl`：记录 API 指标。

这种实现依赖少、便于本地运行，但如果未来要多人使用、权限控制、流式输出、并发管理或部署到服务器，建议迁移到更标准的 Web 框架或至少拆分前端资源。

---

## 8. 质量保证体系

### 8.1 测试现状

当前 Python 代码约 29K 行，`tests/` 下有 8 个测试文件，约 91 个 `unittest` 测试方法。测试重点集中在：

- `run_query()` 出口行为。
- 精确引文路径。
- 检索约束与排序。
- 页码 metadata 推断。
- 段落缓存逻辑。
- Web API 响应和指标记录。
- Phoenix 开关状态。
- 前置页、目录页降权。

测试不是全面覆盖所有路径，但已经覆盖了系统最容易回归的核心行为。

### 8.2 本地检查入口

`scripts/check.py` 是当前推荐检查入口：

| 模式 | 内容 |
| --- | --- |
| `quick` | `validate_maps`、`regression_smoke`、`scripts/test.py app` |
| `full` | `quick` + 检索评估 + 引文页评估 + eval dataset 评估 |

`scripts/test.py` 提供按职责分组的 unittest：

| 分组 | 覆盖 |
| --- | --- |
| `app` | 主问答链路、本地回答、引文和检索回归 |
| `web` | Web API payload、evidence、metrics |
| `rag` | 书目、页码推断、paragraph cache |
| `all` | 运行上述分组 |

### 8.3 评测与审计脚本

项目有较完整的离线评测与审计脚本，包括：

- `scripts/evaluate_retrieval.py`
- `scripts/evaluate_citation_pages.py`
- `scripts/evaluate_eval_dataset.py`
- `scripts/evaluate_e2e.py`
- `scripts/evaluate_hybrid_retrieval.py`
- `scripts/evaluate_answer_quality.py`
- `scripts/audit.py`
- `scripts/run_me_retrieval_quality_gate.py`
- `scripts/run_me_enterprise_retrieval_gate.py`

这些脚本体现了项目已经把“检索命中、页码准确、回答格式、引文合同”纳入工程质量体系，而不只是手工试问。

### 8.4 CI

`.github/workflows/ci.yml` 当前在 Windows 和 macOS 两个平台运行：

- `quick-checks`：安装 `requirements-ci.txt`，执行 `python scripts/check.py --mode quick`。
- `metadata-eval`：开启开发/trace-only 模式，执行 `evaluate_e2e.py`，要求 work 分数和 citation 分数达到阈值。

CI 已经覆盖跨平台基础验证，这是一个明显优点。仍可补充的方向包括 lint、类型检查、Web smoke、Milvus 构建验证和更多数据完整性检查。

---

## 9. 配置与依赖

### 9.1 关键配置

| 变量 | 默认值 / 作用 |
| --- | --- |
| `DEEPSEEK_API_KEY` | LLM API Key |
| `DEEPSEEK_BASE_URL` | 默认 `https://api.deepseek.com` |
| `MARXOS_EMBEDDING_MODEL` | 默认 `BAAI/bge-m3` |
| `MARXOS_EMBEDDING_DEVICE` | 默认 `cpu` |
| `MARXOS_VECTOR_BACKEND` | 默认可用 `milvus`；未显式配置时优先使用已存在的 Milvus Lite 库 |
| `MILVUS_URI` | 默认 `./data/milvus_lite/marxos_bgem3_sparse.db` |
| `MILVUS_HYBRID_SEARCH` | Milvus dense+sparse hybrid 开关 |
| `MILVUS_SPARSE_PROVIDER` | 默认建议 `bge-m3` |
| `VECTORSTORE_DIR` | FAISS fallback，默认 `vectorstore/marx_reader_core` |
| `PARAGRAPH_VECTORSTORE_DIR` | FAISS fallback，默认 `vectorstore/marx_reader_paragraph` |
| `PARAGRAPH_CACHE_PATH` | 默认 `data/paragraph_cache_core.jsonl` |
| `OCR_CACHE_DIR` | 默认 `data/ocr_cache` |
| `SEMANTIC_PARENT_WINDOW` | 默认 `1` |
| `SEMANTIC_CHILD_CHUNK_SIZE` | 默认 `180` |
| `SEMANTIC_CHILD_CHUNK_OVERLAP` | 默认 `40` |
| `SEMANTIC_SPARSE_TOP_K` | 默认 `24` |
| `MARXOS_WEB_PORT` | 默认 `7860` |
| `MARXOS_PHOENIX_ENABLED` | Phoenix/OTEL 可观测性开关 |

### 9.2 依赖分层

| 文件 | 用途 |
| --- | --- |
| `requirements-ci.txt` | CI 与核心推理/测试依赖 |
| `requirements.txt` | 完整本地运行依赖，包含 OCR、PDF、图像处理等 |
| `requirements-phoenix.txt` | Phoenix 与 OpenTelemetry 可观测性依赖 |
| `docker-compose.milvus.yml` | Milvus 本地服务 |

系统默认依赖较重，主要重在 embedding、FAISS、PaddleOCR 和 PDF/OCR 工具链。建议继续保持 CI 依赖和完整 OCR 依赖分离，避免每次验证都安装全量 OCR 栈。

---

## 10. 当前优势

1. **问题域明确。** 系统围绕马克思主义经典文本的“可核验问答”设计，不是泛化 RAG 玩具。
2. **检索策略扎实。** 小块检索、大块召回、BM25 补充、结构化约束、重排序和 backstop 共同工作。
3. **引文意识强。** 证据卡、页码修正、引文审计和回答合同让系统输出更接近学术工具。
4. **工程入口清晰。** README、`scripts/check.py`、`scripts/test.py`、`docs/codebase_inventory.md` 已经形成维护路径。
5. **质量脚本丰富。** 检索、页码、端到端、回答质量、企业级门禁都有专门脚本。
6. **本地优先。** 支持本地语料、本地向量库和开发模式，对私有语料与离线调试友好。
7. **可演进。** Milvus、Phoenix、ML intent classifier 等扩展点已经出现，后续可以逐步服务化和可观测化。

---

## 11. 主要风险与问题

### 11.1 `app.py` 仍是复杂度中心

虽然已有大量拆分，`app.py` 仍超过 3K 行，承担了主链路、兼容 wrapper、上下文构建、CLI 和部分恢复逻辑。长期看，这会让新功能容易继续堆回入口文件。

建议继续拆出：

- LLM client 与调用重试。
- Evidence/context builder。
- answer recovery workflow。
- CLI 交互层。
- retrieval ctx 注入 wrapper。

### 11.2 检索后端状态需要更明确

代码中同时存在 FAISS、段落 FAISS、BM25、Milvus backend、BGE-M3 配置和历史 FAISS 说明。报告和 README 需要持续同步“默认路径”和“实验路径”，否则使用者容易不知道当前该构建哪个索引、设置哪个环境变量。

### 11.3 数据质量仍决定上限

OCR 错误、印刷页识别、PDF 页映射、目录页混入、篇目定位不准，都会直接影响回答可信度。LLM 不能从根本上修复语料错误，只能在已有证据上组织语言。

建议把数据质量继续当作一等工程对象，而不是后处理问题。

### 11.4 Web 端仍偏本地工具形态

当前 Web 实现简洁，但没有用户体系、权限控制、流式输出、前端资源拆分、服务化部署结构。若目标只是本机研究助手，问题不大；若要多人试用，需要提前重构。

### 11.5 测试覆盖仍有盲区

当前测试对核心逻辑有价值，但以下路径仍需要更强自动化：

- 完整 OCR 到向量库构建的集成验证。
- Milvus 后端的最小 smoke。
- Web 端真实浏览器交互。
- 真实 LLM 调用的隔离评估。
- 大数据资产缺失或损坏时的降级行为。

---

## 12. 推荐改进路线

### 短期：稳定主线

1. 在 README 和本报告中明确“默认 FAISS 路径”和“可选 Milvus 路径”的边界。
2. 继续压薄 `app.py`，优先拆出 evidence/context builder 和 LLM client。
3. 给 `MARXOS_VECTOR_BACKEND`、Milvus collection、FAISS fallback 写一份短配置说明。
4. 将 `scripts/test.py all` 是否应纳入 `check.py quick` 做一次取舍。
5. 增加一个 Web smoke，验证 `/api/ask` 在开发模式下可用。

### 中期：提升可信度

1. 建立数据资产版本清单：OCR 缓存、段落缓存、向量库、page_map、article_map 的生成时间和来源。
2. 把页码、篇目定位、目录页过滤的评测结果沉淀为固定报告。
3. 扩展 eval dataset，覆盖更多概念、书信、歧义标题、跨卷主题。
4. 对回答质量评估引入更明确的 rubric：证据相关性、引文正确性、分析完整性、拒答合理性。

### 长期：服务化与产品化

1. 如果需要多人使用，将 Web 层迁移到 FastAPI 或类似框架，并拆分前端资源。
2. 将 Milvus 作为标准后端之一，提供一键构建和健康检查。
3. 引入后台任务管理，用于 OCR、建库、评测和索引刷新。
4. 加入可观测 dashboard，沉淀检索命中、审计失败、LLM 调用耗时和用户问题类型。

---

## 13. 总体结论

MarxOS 当前已经具备一个严肃本地学术 RAG 系统的雏形：它不是只把 PDF 切块后接 LLM，而是围绕经典文本问答的真实难点，建立了 OCR、页码、篇目、检索、证据、引文和评测的完整链路。

项目下一阶段最值得投入的方向，不是增加更多炫技型功能，而是继续把主线收紧：明确默认后端，稳定数据资产版本，压薄入口文件，强化数据质量评测，并把 Web 端从本地演示逐步推向更可靠的研究工具形态。

一句话评价：

**MarxOS 已经从“能回答问题的 RAG 原型”进入“有证据意识和质量闭环的专业文本研究系统”阶段；后续的核心任务是工程收口和可信度持续提升。**
