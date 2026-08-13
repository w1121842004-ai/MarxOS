# Corpus v2 旁路重建运行手册

本次重建只覆盖当前稳定基线中的《文集》10 卷和《选集》4 卷，不在重建过程中扩展语料范围。旧 paragraph cache、旧 Milvus Lite DB 和旧 profile 始终保留，直到 v2 完成晋级验收。

## 不可变原则

1. PDF 和页面 cache 是输入；PageRecord 和 ParagraphRecord 是权威数据产物。
2. semantic parent、semantic child 和 Milvus 均为可重复生成的派生产物。
3. 每次构建写入新的空目录或空 DB，不原地覆盖。
4. 未通过数据门禁不得运行全量 embedding。
5. 未通过新旧检索对比不得切换默认 profile。

## 构建阶段

1. `preflight`：校验输入、磁盘空间、配置、checksum 和工具版本。
2. `pages`：把页面 cache 规范化为版本化 PageRecord。
3. `paragraphs`：生成稳定 ID 的 ParagraphRecord，并运行全量质量审计。
4. `retrieval-units`：分别生成 semantic parent 和候选 semantic child。
5. `probe`：对 chunk 和 sparse 候选运行分层 eval，冻结胜出参数。
6. `index`：写入新的 `marxos_corpus_v2.db`，支持 checkpoint 和幂等重试。
7. `validate`：校验 schema、row count、hash、lineage round-trip、检索质量和启动稳定性。
8. `promote`：新增并切换 v2 profile；旧 profile 保留为即时回滚入口。

配置入口为 `config/rebuild_v2.json`。任何影响文本、ID、页码、篇名、chunk、embedding 或 sparse 向量的修改都必须改变 manifest 构建指纹，不能静默复用旧产物。

书目富化采用保守策略：版本、出版社和卷次可由 source 确定性继承；`work_id` 只有在来源、印刷页范围和规范篇名同时匹配时才写入。覆盖不足允许留空，错误挂载作品 ID 不允许通过门禁。

## 当前冻结结果（2026-08-12）

- 文档构建版本：`corpus-v2-efa06171f7698079`。
- ParagraphRecord：41,613 条；默认正文 34,260 条；隔离 7,353 条；审计阻断错误 0。
- `编辑说明` 等编辑性前置材料已归类为 `preface_editorial`，默认检索泄漏为 0。
- RetrievalRecord：`semantic_child`，chunk `320/64`，45,875 条；ID、hash、offset 与 parent lineage 全量一致。
- sparse：真实 corpus-aware BM25；完整词表的 32 位 term ID 碰撞采用确定性线性探测消解。
- Milvus v2 probe：BGE-M3 dense + BM25 sparse，50/50 写入并回读通过。

## 晋级完成记录（2026-08-13）

- 全量 embedding 由 checkpoint 续建完成：`data/milvus_lite/marxos_corpus_v2.db` / `marxos_passages_v2` 共 45,875 行，checkpoint `complete=true`。
- validate：`scripts/validate_milvus_v2.py` 全绿（schema 契约、行数、511 抽样双 hash + lineage + 页码 round-trip、41,613 父段落、5 问 hybrid probe）。
- 新旧检索对比：91/95 = 91/95，零回退。
- Web smoke：离线 20/20；真实端到端 5/5。启动稳定性 10/10。旧 profile 回滚启动 + 检索通过。
- 默认 profile 已切换：`me_full_v2` + `milvus_bgem3_v2`。回滚入口：`MARXOS_CORPUS_PROFILE=me_full` + `MARXOS_RETRIEVAL_PROFILE=milvus_bgem3_stable`（及对应旧 MILVUS_URI/COLLECTION/PARAGRAPH_CACHE_PATH 环境变量，见 `.env.example` 注释）。
- 索引 manifest 已更新：`config/index_manifest_v1.json`（row 45,875，320/64，bm25 sparse），`scripts/validate_index_manifest.py` 校验通过。

## 历史阻断项

- 旧 cache 中至少 6,643 条记录处于注释或各类索引区域，必须完成页面类型重分类。
- 旧 cache 中存在被 OCR 文本污染的篇名，不能把“非空 article”视为篇名有效。
- 旧索引会把超过 2,000 字的段落裁剪后再计算唯一 `text_hash`；v2 必须区分源文本和索引文本 hash。
- chunk 大小与 sparse provider 尚未通过 probe，不得提前冻结全量索引参数。
- macOS ARM：torch 与 Milvus Lite（其 HNSW 索引由 FAISS 实现）共用 libomp，多线程检索会在 `__kmp_suspend_initialize_thread` 处 SIGSEGV。运行与验收均需 `OMP_NUM_THREADS=1`（app.py 入口、启动脚本、`.env.example` 已内置兜底）；builder 收尾的 `load_collection` 前已内置 `faiss.omp_set_num_threads(1)`。全量重建时若收尾崩溃，用 `OMP_NUM_THREADS=1` 加 `--resume` 重跑即可安全续建。
