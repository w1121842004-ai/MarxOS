# 检索约束与模式优先级

本文档记录 `retrieval/constraints.py` 的约束构建顺序和 `retrieval/modes.py` 的检索执行顺序（P2 梳理后的当前行为）。修改前先读 `docs/maintenance_guide.md` 的验证要求。

## 1. constraints_from_query 构建顺序

```text
1. 定义型概念题门禁（definition_concept_query）
   WHAT_DEFINITION（是什么/什么是）+ 无《》书名 + 命中概念词
   → 跳过 high_precision_locator、article_locator、work-catalog 严格标题分支
   → 概念题按概念约束检索，不被 locator/页段回退锁死
   （显式书名如《1844年经济学哲学手稿》是什么 不受影响）
2. 宽泛专题分支（is_broad_topic_query）
   → 仅当查询中没有逐字出现的著作标题（work_catalog_title_mentioned）
   → 产出 soft_topic 约束
3. high_precision_locator（token→精确页）
4. article_locator（篇名→全集页范围）
   → page_ranges / sources 合并 core classic 跨版本范围（选集/文集候选可通过过滤）
5. explicit_volume（明确指定某卷）
6. me_title_hint（全集篇名提示）
7. query_work_hints（硬编码标记→篇名表）
8. work_catalog 严格标题（逐字标题/别名）
9. work_catalog 概念匹配（match_by_concepts）
   → 定义型概念题下 strict_title=False（排序焦点，不硬锁）
10. topic 专题（narrow_topic_constraints_by_query）
11. concept 概念（CONCEPT_CANONICAL_CLASSIC_IDS）
    → 始终 strict_title=False；classic entries 仅作排序焦点
12. locator_rules / classic_entries（核心经典目录）
13. BookLocator LLM 兜底
    → planner 变体检索（variant_retrieval=True）跳过，避免无谓 LLM 调用
```

要点：

- **strict_title 的语义**：只有「用户明确指向某篇著作」才成立（locator / 逐字标题匹配）。概念约束和定义型概念题一律非 strict。
- **page_ranges 按 source 分别校验**（`page_in_expected_range`），跨版本条目合并到 `page_ranges`/`sources`，不改变 `entries` 的严格性。
- **版本优先级**：查询未指定版本时 me（全集）> mea（文集）> mes（选集）（`prefer_sources_for_query` / `source_priority`，由 `MARXOS_PREFERRED_EDITIONS` 配置）。

## 2. retrieve_documents 执行顺序（modes.py）

```text
A. exact quote（quote_lookup，非著作定位题）
   1. OCR 精确引文（exact_quote_lookup）→ 命中即返回
   2. strict_title 时 OCR 页段回退（strict_title_cache_documents）
   3. sparse_first 策略时本地 BM25 先行
   4. 未命中 → 落入 C（不再直接返回空；候选标注为未确认）
B. high_precision_locator → OCR 页段 / locator 回退
C. 主检索
   - 有概念词 → concept_constrained_candidates（seeds 扩展 + dense/hybrid）
   - 有 topic_id → topic_constrained_candidates
   - 否则 → controlled dense + hybrid merge + source/page 过滤
   - strict_title + 窄页范围 → 合并 OCR 页段回退
D. 重排（rerank_documents 10 维打分）→ diversify / dedupe / topic select
E. 标注（annotate_docs_with_constraints）→ expand_semantic_parent → append_locator_backstops
```

quote 查询的候选在下游统一标注 `match_type=vector_candidate, confidence=0.0`，prompt 中明确警告「No exact quote match was found；vector candidates only」。

## 3. 三路检索的分开度量

P2 起不再把三者笼统称为「hybrid」，各自独立测量：

| 通道 | 实现 | 概念题 top-3 source 命中（2026-08-13） |
| --- | --- | --- |
| dense | Milvus ANN（BGE-M3 1024 维） | 9/9（0.5s / 9 题） |
| hybrid | dense + corpus-aware BM25 sparse（RRF） | 9/9（7.2s） |
| local_bm25 | 本地段落级 BM25 索引（sparse_retrieve_documents） | 8/9（0.1s） |

测量脚本：`scripts/compare_retrieval_paths.py`（报告 `logs/retrieval_paths_compare.json`）。结论：概念题上 dense 单独已覆盖全部命中；hybrid 的增量价值在引文/稀疏匹配场景，延迟成本主要来自 BM25 查询编码。

## 4. CRAG 与 evidence 展示一致性

- CRAG 评分（`assess_retrieval_quality`）与 evidence 卡片同源：卡片携带 `match_type` / `confidence`。
- 前端证据卡按 match_type 标注（原文核对 / 定位提示 / 页段回退 / 未确认候选 / 稀疏候选 / 段落候选），未确认证据不再与原文核对证据同外观。
- locator-only（全部为定位回退）在 CRAG 中扣 35 分并标记 `locator_only`，不通过即触发纠正性检索。
- 专题追问 scoping：查询中显式命名新概念或著作标题时不继承上一轮专题前缀（`topic_scoped_query` 的 `is_explicit_subject_fn`）。
