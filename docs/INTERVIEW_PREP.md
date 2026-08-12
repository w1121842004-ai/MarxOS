# MarxOS 面试项目全景回顾

> 本文档面向面试准备，按"项目概述 → 架构演进 → 核心挑战 → 优化决策 → 工程体系"展开。
> 每个技术点都配有"遇到的问题 → 解决方案 → 量化效果"，方便面试时讲述。

---

## 一、一句话概括

**MarxOS** 是一个面向《马克思恩格斯全集/文集/选集》中文经典文本的**本地 RAG 智能问答系统**，支持概念解释、引文定位、书目查找、理论分析等 7 类查询意图，所有回答附带可核对的原文出处。

**技术关键词**：RAG、混合检索（Dense + BM25）、CRAG 自修复、意图路由、引用审计、OCR、Milvus、FAISS、BGE-M3、DeepSeek

---

## 二、项目规模（量化数据）

| 维度 | 数据 |
|------|------|
| 代码量 | ~8,300 行 Python（app.py 3,438 行，其余分布在 35+ 模块） |
| 语料规模 | 75 卷 PDF → 5.7 万页 OCR → 4.3 万段落 → 29.8 万子块向量 |
| 知识库 | 94 部著作的结构化元数据（标题/别名/概念/引文/跨版本页码） |
| 测试 | 8 个测试文件，91 个测试方法 |
| 评估数据集 | 3 套，共 400+ 道人工标注题目 |
| 工具脚本 | 50+ 个（构建/评测/审计/数据集生成） |
| 模块数 | 35+ Python 模块，分布在 7 个子包中 |

---

## 三、架构演进（体现工程能力的核心叙事）

### 3.1 V1：单体脚本期（初始提交 → 约 4 个月）

**状态**：所有逻辑堆在 app.py 单文件中，超过 2000 行。

**问题**：
- 改一个检索参数需要在整个文件中搜索
- 配置路径散落在各处（`article_map_core`、`paragraph_cache_core` 硬编码）
- 无法独立测试任何子模块
- 切换语料库（文集 → 全集）需要改 5-6 个文件

### 3.2 V2：扁平模块期

**改动**：将 `app.py` 中的函数按职责拆分到 `marxos_*.py` 扁平文件：
- `marxos_query_intent.py` — 意图识别
- `marxos_citations.py` — 引用格式化
- `marxos_prompts.py` — Prompt 模板
- `marxos_runtime.py` — 运行时状态
- `retrieval/` 子包 — 检索约束、模式、排序

**收益**：模块职责第一次清晰了，`app.py` 从 2000+ 行降到 ~1900 行。

**遗留问题**：
- 20 个 `marxos_*.py` 平铺在根目录，依然杂乱
- 配置仍然散落，切换语料仍需改多文件
- 没有统一的配置入口

### 3.3 V3：分层包架构（当前状态）

**改动**：将所有 `marxos_*.py` 按功能域移入 `marxos/` 子包：

```
marxos/
├── config/          # 配置层（Profile-based，env var 可覆盖）
├── data/            # 数据加载层
├── indexing/        # 索引构建层
├── generation/      # 生成层（prompt/答案/引用/LLM客户端）
├── app/             # 应用编排层
└── web/             # Web 服务层
```

**关键创新 —— Profile 化配置系统**（`marxos/config/settings.py`）：

面试时可以重点讲这个：

```python
# 通过环境变量一键切换整套方案，而非逐个改参数
MARXOS_CORPUS_PROFILE=me_full \        # 语料：全部三套
MARXOS_RETRIEVAL_PROFILE=milvus_bgem3_hybrid \  # 检索：混合模式
MARXOS_ANSWER_PROFILE=deepseek_default \        # 回答：深度模式
.venv/bin/python web_app.py
```

核心设计：frozen dataclass + LRU 缓存 + 环境变量覆盖。每个 profile 是一组预设值，但每个单项仍可通过环境变量单独覆盖。

**收益**：
- 切换语料/检索/回答方案：一行命令
- 新增方案：加一个 profile dict 条目
- 所有模块通过 `get_settings()` 统一获取配置，杜绝硬编码

---

## 四、核心工程挑战与解决方案（面试重点）

### 挑战 1：OCR 乱码（Mojibake）

**问题**：PaddleOCR 输出的中文文本出现系统性编码损坏，典型模式是 UTF-8 字节被误解释为 Latin-1：
- "马克思" → "é©¬å…‹æ€"
- "资本论" → "èµ„æœ¬è®º"

**影响面**：直接影响 5.7 万页 OCR 缓存质量，进而影响检索召回率和引用准确度。

**解决方案**（`app.py:repair_mojibake()`）：

```python
def repair_mojibake(text):
    markers = ("Ã", "Â", "ã", "å", "æ", "ç", "è", "é", "ï", "ä")
    if not any(marker in text for marker in markers):
        return text  # 快速路径：无乱码直接返回
    
    # 对疑似乱码片段做 Latin-1 → UTF-8 反向解码
    def decode_run(match):
        run = match.group(0)
        try:
            return run.encode("latin1").decode("utf-8")
        except UnicodeError:
            return run  # 解码失败保留原文
    
    return re.sub(r"[\x00-\xff]+", decode_run, text)
```

**设计要点**：
1. 先做**标记检测**（O(n) 字符串扫描），不做全量编解码——99% 的正常文本走快速路径
2. 只对疑似片段做 Latin-1 → UTF-8 反向解码
3. 解码失败保留原文，不引入新错误

### 挑战 2：PDF 页码 ≠ 印刷页码

**问题**：PDF 文件的页码（pdf_page）和书籍的印刷页码（printed_page）之间存在偏移。例如《全集》第 1 卷的 PDF 第 1 页是封面，第 20 页才是印刷第 1 页。

**影响面**：引用页码错误是学术问答的致命伤——用户看到"第 1 页"实际是封面。

**解决方案**（三层页码解析，`app.py`）：

```
Layer 1: page_map.json（人工校准的映射表）
  → PDF 页码 → 印刷页码的直接映射

Layer 2: OCR 缓存边缘检测（自动推断）
  → 读取每页 OCR 文本的前 3 行 + 后 3 行
  → 正则匹配独立的数字行（如 "501"）
  → 验证：-5 ≤ (pdf_page - printed_page) ≤ 180

Layer 3: 元数据归一化（normalize_metadata）
  → 统一 citation_page / printed_page / pdf_page
  → 引用输出永远优先用印刷页码
```

**关键函数**：`infer_printed_page_from_ocr_cache()` — 不依赖人工映射表，自动从 OCR 文本边缘推断印刷页码。

### 挑战 3：小块检索 vs 大块上下文的矛盾

**问题**：这是 RAG 领域的经典 trade-off——
- 小块（180 字符）：Embedding 语义精准，但上下文不足，LLM 看不懂
- 大块（完整段落 500-2000 字符）：上下文完整，但 Embedding 语义稀释

**解决方案**：**小块索引 + 父段落窗口扩展**

```
索引侧（build 阶段）：
  完整段落 ──→ 切成 180 字符子块 ──→ 每个子块嵌入 BGE-M3
            ──→ 子块记录 parent_paragraph_id
            ──→ 子块存入 FAISS/Milvus

检索侧（query 阶段）：
  用户 Query ──→ Dense 检索 ──→ 命中 top-K 子块
            ──→ 查 parent_paragraph_id
            ──→ 展开：前 1 段 + 本段 + 后 1 段
            ──→ 500-2000 字符的完整上下文窗口
```

**效果**：小块 Embedding 语义精准（180 char）+ LLM 上下文完整（3 段窗口）。

### 挑战 4：V1 向量库的 parent_paragraph_id 缺失

**问题**：早期的 FAISS 构建脚本（`build_vectorstore_from_cache.py`）产出的向量块**没有** `parent_paragraph_id` 字段。当系统尝试做父段落展开时，这些块**静默失败**——展开函数返回空上下文。

**发现方式**：不是报错，而是回答质量下降。通过 trace 输出发现 `expand_semantic_parent_docs` 总是返回原始小块。

**解决方案**：
1. 标记 V1 构建脚本为 DEPRECATED
2. 重写 V2 构建管线：`build_paragraph_cache.py` → `build_semantic_child_vectorstore.py` → `build_paragraph_vectorstore.py`
3. V1 脚本的工具函数（`document_from_cache` 等）保留，因为 40+ 个脚本依赖它们

**教训**：元数据字段缺失不会报错，但会静默降级。应该在构建阶段做 schema 校验。

### 挑战 5：多版本语料的优先级混乱

**问题**：三套 PDF（全集 me / 文集 mea / 选集 mes）存在大量重复篇目，同一篇《关于费尔巴哈的提纲》在三个版本中页码不同：
- 全集 me03.pdf：第 3-9 页
- 文集 mea01.pdf：第 499-506 页
- 选集 mes01.pdf：第 133-140 页

**用户期望**：默认优先返回《全集》（最权威），但明确指定时尊重用户选择。

**解决方案**（`retrieval/constraints.py`）：

```python
# 版本优先级（可配置）
_PREFERRED_EDITIONS = ("me", "wenji", "xuanji")

def source_priority(source, query, ctx):
    # 如果用户明确说了"文集"或"选集"，尊重用户选择
    requested = collection_requested(query, ctx)
    if requested and source_matches_collection(source, requested):
        return 0  # 最高优先级
    # 否则按默认优先级排序
    ...
```

同时 `work_catalog.json` 为每部著作的每个版本存储独立条目：

```json
{
  "work_id": "theses-feuerbach",
  "title": "关于费尔巴哈的提纲",
  "editions": {
    "wenji_v1": {"source": "mea01.pdf", "start_page": 499, "end_page": 506},
    "xuanji_v1": {"source": "mes01.pdf", "start_page": 133, "end_page": 140}
  }
}
```

### 挑战 6：意图识别的演进

**V1（硬优先级级联）**：
```
if is_bibliographic: return "bibliographic"
elif is_quote: return "quote"  
elif is_concept: return "concept"
...
```
**问题**：一旦匹配就返回，无法处理"既是引文又是概念"的歧义，且评分不可解释。

**V2（并行打分 + 概率分布）**（当前主线）：
```
6 个 scorer 并行打分 → 归一化为概率分布 → 取最高分
```
- 每个 scorer 输出 0.0-1.0 浮点数
- 歧义检测：top1 和 top2 分差 < 0.15 → 标记为 ambiguous
- `IntentResult` 数据类同时给出 primary + confidence + distribution

**V3（规则 + ML 混合，可选）**：
- LogisticRegression 分类器（~10 KB）在 BGE-M3 嵌入上训练
- 混合策略：`0.6 * ML_prob + 0.4 * rule_score`
- 模型文件不存在时自动降级为纯规则（零影响）

**训练数据构造**：
- 人工标注：2 套 eval dataset（~320 题）
- 模板合成：94 部著作 × 7 组模板（词条/概念/引文槽位填充）→ ~560 条

### 挑战 7：RAG 质量自愈（CRAG）

**问题**：检索结果质量参差不齐，有些 query 返回的全是无页码锚点的"定位提示"型文档。

**解决方案**：**CRAG 评分 + 纠正检索 + 恢复循环**

```
检索结果 ──→ assess_retrieval_quality()
  ├─ 评分维度：文档数量、页码锚点、match_type、来源多样性、严格标题绑定
  ├─ 阈值：45（常规）/ 52（严格标题）/ 55（引文）
  │
  ├─ score ≥ threshold → ✅ 直接生成
  └─ score < threshold → 🔄 纠正性检索
       ├─ 用 route_query 重新检索（更大的 k）
       ├─ 段落级 text overlap 过滤
       ├─ 重新 CRAG 评分
       ├─ 取分数更高的结果
       │
       └─ 生成后 ──→ 引文审计
            ├─ 格式检查（是否有编号/出处）
            └─ OCR 内容验证（deep 模式）
                 ├─ 通过 → 返回
                 └─ 幻觉率 > 50% → 🔄 第 2 轮恢复
```

**最多 2 轮恢复**，被 `max_recovery_rounds` 控制。

### 挑战 8：BM25 冷启动

**问题**：BM25 稀疏索引构建在首次使用时触发，需要遍历 4.3 万段落，耗时 5-8 秒。如果在用户 query 时触发，首次请求延迟爆炸。

**解决方案**：
1. 环境变量 `SEMANTIC_SPARSE_COLD_START=skip` 时冷启动阶段跳过 BM25（静默降级到 dense-only）
2. `MARXOS_WARM_SPARSE_INDEX=1` 时在 web 服务启动时后台线程预构建
3. `hybrid_retrieval_enabled()` 在冷启动阶段返回 False

```python
# web_app.py 启动逻辑
if hybrid_retrieval_enabled() and warm_sparse:
    threading.Thread(target=warm_sparse_index, daemon=True).start()
```

---

## 五、性能优化体系（三层 Preset）

面试时可以强调：这不是简单的"快中慢"三档，而是**系统性地控制整个 RAG 管线的每个环节**。

| 环节 | fast | standard | deep |
|------|------|----------|------|
| 检索数量 | 3-5 | 4-8 | 5-12 |
| 段落双路检索 | ❌ | ❌ | ✅ |
| 混合检索 (BM25) | ❌ | ❌ | ✅ |
| 多查询分解 | ❌ | ❌ | ✅ |
| CRAG 纠正检索 | ❌ | ✅ | ✅ |
| 引用 OCR 验证 | ❌ | ❌ | ✅ |
| 恢复循环 | 0 轮 | 0 轮 | 2 轮 |
| 上下文预算 | 2,500 字符 | 6,000 字符 | 16,000 字符 |
| LLM timeout | 35s | 60s | 120s |

**设计要点**：每个环节的 on/off 和参数独立控制，通过 `performance` dict 从 `run_query()` 向下游所有函数传播。

---

## 六、数据管线（面试时可以画图）

```
PDF 原著 (75 卷)
  │  PaddleOCR
  ▼
OCR 缓存 (data/ocr_cache/, 5.7 万页, 每页 JSON + TXT)
  │  段落检测 (正则 + 页码推断)
  ▼
段落缓存 (paragraph_cache_core.jsonl, 43,212 条)
  │  ├─ 路径 1: 子块切分 (180 char) → BGE-M3 → FAISS (298K 向量)
  │  ├─ 路径 2: 完整段落 → BGE-M3 → FAISS (43K 向量)
  │  └─ 路径 3: BGE-M3 dense + sparse → Milvus Lite (混合)
  ▼
向量库 (3 个索引, 可切换)
  │
  ▼
检索 → 排序 → 上下文构建 → DeepSeek → 引用审计 → 最终回答
```

---

## 七、引文质量控制体系

这是 MarxOS 区别于普通 RAG 系统的核心能力：

1. **引用格式修复**（`repair_answer_citations`）：LLM 输出的序号/书名/页码可能格式不一致，用正则 + 证据卡匹配做规范化
2. **引用匹配过滤**（`filter_evidence_to_answer`）：只保留在回答中实际使用的证据卡
3. **引用审计**（`audit_answer_citations`）：检查每条引用是否有对应的证据卡
4. **OCR 内容验证**（`verify_citations`，deep 模式）：用第二个 LLM 调用来验证引用内容是否在 OCR 原文中确实存在

**为什么需要内容验证？**
- LLM 有时会"幻觉"出一个看起来合理的页码
- 即使约束了证据卡，LLM 也可能把不同证据卡的信息糅合成一个不存在的引用
- 第二个 LLM 调用做独立验证，不以第一个 LLM 的判断为准

---

## 八、工程踩坑记录（面试加分项）

| 坑 | 症状 | 根因 | 解决方案 |
|----|------|------|----------|
| ctx dict 键名变更 | 静默失败，检索返回空 | `_helper(ctx, "name")` 中字符串键名与 `_retrieval_ctx()` 中定义不一致 | 编码规范：ctx 键名和 helper 提取名必须一致，考虑用 dataclass 替代 dict |
| OCR 路径匹配 | 部分 PDF 的 OCR 文本读不到 | `load_ocr_page_text()` 用 `source_stem()` 去掉 `.pdf` 后缀，但部分 source 字段带了额外路径 | 统一 source 字段格式 |
| Settings 单例陈旧 | 改了环境变量不生效 | `get_settings()` 用了 `@lru_cache`，首次调用后缓存 | 文档化：改 env var 需要重启进程 |
| parent_paragraph_id 缺失 | 段落扩展静默返回空 | V1 向量库无此字段 | V2 重建 + legacy 检测 |
| 混合检索静默降级 | BM25 不生效但无报错 | 冷启动时 sparse index 未就绪 | 显式 `hybrid_retrieval_enabled()` 检查 |
| RRF 融合时 dense rank 丢失 | 稀疏结果覆盖了稠密结果 | 合并逻辑未保留原始 rank | RRF 公式中同时记录 dense_rank 和 sparse_rank |

---

## 九、测试与质量保障

```
quick check (CI, ~4 min)
├── validate_maps.py         # 验证 work_catalog + article_map + page_map 一致性
├── regression_smoke.py       # 关键查询回归测试
└── test.py app               # 单元测试 (8 文件, 91 方法)

full check (~30 min)
├── 上述 quick checks
├── evaluate_retrieval.py     # 检索 top-k 命中率
├── evaluate_citation_pages.py # 引用页码命中率
└── evaluate_eval_dataset.py   # 端到端 work_correct 指标

CI gate (GitHub Actions)
├── metadata-eval: work_correct ≥ 65%
└── quick-checks: 全部通过
```

---

## 十、技术栈总结

| 层 | 选型 | 理由 |
|----|------|------|
| LLM | DeepSeek Chat (OpenAI 兼容 API) | 中文能力强、成本低 |
| Embedding | BGE-M3 (1024-dim) | 中文优化、支持 dense+sparse 双路 |
| 向量库(主) | Milvus Lite | HNSW + COSINE、支持混合检索 schema |
| 向量库(备) | FAISS | 离线可用、零依赖 |
| 稀疏检索 | BM25 (rank-bm25) | 关键词精确匹配，补充语义检索盲区 |
| OCR | PaddleOCR | 中文识别精度高 |
| 分词 | jieba | 轻量、中文 POS 标注 |
| ML 分类器 | sklearn LogisticRegression | ~10 KB、<1ms 推理 |
| Web | Python stdlib http.server | 零外部依赖、SPA 内嵌 |
| 追踪 | Phoenix / OpenInference | LLM 调用链路追踪 |

---

## 十一、后续方向（体现技术视野）

1. **全集 OCR 完成**：当前默认语料仍以文集/选集为主，全集 OCR 完成后检索覆盖度将大幅提升
2. **Reranker 模型**：当前用规则打分（10 维），可以换 BGE-Reranker 或 Cohere Rerank 做 Cross-Encoder 重排
3. **HyDE 查询扩展**：当前 `query_planner` 做的是关键词变体，可以加入 HyDE（生成假设文档再检索）
4. **多轮对话状态管理**：当前 Web 端有简单历史记录，但没有真正的对话状态追踪
5. **流式生成**：当前 SSE 只传输状态，可以做到 token 级流式输出
