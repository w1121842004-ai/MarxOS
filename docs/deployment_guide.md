# MarxOS 初步部署指南

> 目标形态：**单机/局域网单进程**。不追求容器化或远程多实例；数据资产直接拷贝，不在目标机器重建。

## 1. 部署包清单

### 运行时必需（必须拷贝）

| 资产 | 路径 | 体积 | 用途 |
| --- | --- | --- | --- |
| Milvus Lite 向量库 | `data/milvus_lite/marxos_corpus_v2.db/` | ~476M | 主检索（dense + BM25 sparse hybrid） |
| 语料段落记录 | `data/artifacts/corpus_v2/paragraph_records_enriched_v2_1.jsonl` | ~105M | 子块→父段落窗口扩展、书目元数据 |
| BM25 词表统计 | `data/artifacts/corpus_v2/bm25_stats_v2_1.json` | ~119M | 运行时 hybrid 查询的 sparse 编码 |
| OCR 文本缓存 | `data/ocr_cache_text_layer/` | ~963M | 精确引文查找、引文内容验证 |
| 结构化目录 | `rag/*.json`（article_map / work_catalog / topic_catalog / locators…） | ~8.4M | 约束构建与书目定位 |
| 索引 manifest | `config/index_manifest_v1.json` | 1.7K | 校验与追溯 |
| 代码 | 仓库全部 `.py`、`marxos/`、`retrieval/`、`web_app.py`、`app.py`、启动脚本、`.env.example` | — | — |

### 构建中间产物（不必拷贝）

`data/artifacts/corpus_v2/` 下的 `page_records.jsonl`、`paragraph_records*.jsonl`（v2/v2_1 未富化版）、`semantic_child_records*.jsonl`、各 probe/checkpoint 文件仅用于重建管线；运行时不需要。`data/milvus_lite/` 下其他 `*probe*.db`、旧 text_layer DB 除非需要回滚入口。

### 模型缓存

BGE-M3 默认从 `~/.cache/huggingface/hub/models--BAAI--bge-m3` 读取；离线机器需整目录拷贝该缓存（约 2.3GB），或首次启动联网下载。`TRANSFORMERS_OFFLINE=1` 已在 app.py 内置，缓存就位后无需联网。

## 2. 新终端部署步骤

```bash
# 1. 代码 + 数据资产拷贝（见上表）
# 2. 虚拟环境
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. 配置
cp .env.example .env
#    编辑 .env：DEEPSEEK_API_KEY=...

# 4. 索引 manifest 校验（可选但推荐）
.venv/bin/python scripts/validate_index_manifest.py

# 5. 启动
./启动MarxOS网页端.command        # macOS 双击或终端执行
# 或 .venv/bin/python web_app.py   # 端口冲突：MARXOS_WEB_PORT=7861

# 6. 部署 smoke
.venv/bin/python scripts/deployment_smoke.py   # 6 项全过 = 部署可用
```

注意：`deployment_smoke.py` 会启动自己的服务实例，需要 Milvus DB 独占访问——先停掉已运行的 web_app.py。

## 3. 部署 smoke 覆盖（task.md P5 契约）

| 项 | 内容 |
| --- | --- |
| health | `/healthz` 200 + `{"status":"ok"}` |
| readiness | `/readyz` 200 且 `ready=true`（索引/collection/schema/模型/API key） |
| 书目题 | 《共产党宣言》收录在哪一卷？ |
| 引文题 | “全世界无产者，联合起来！”出自哪里？ |
| 概念题 | 什么是剩余价值？（走 flash 模型 + 证据卡） |
| Web follow-up | 以上一轮概念题 evidence 追问页码 |

## 4. 启动性能（实测 2026-08-13）

| 阶段 | 耗时 |
| --- | --- |
| 进程启动至向量库就绪（BGE-M3 + 119M BM25 统计 + collection） | ~4.8s |
| 首次 hybrid 检索（含查询编码预热） | ~6.4s |
| **冷启动到可服务** | **~11s** |

决策：**暂不做懒加载/后台加载**。首屏等待约 11s 在本地应用场景可接受；懒加载会把「启动失败」延迟成「首个请求失败」，与“先保证确定性”的目标冲突。后续如需要，从 `RuntimeState.load_milvus_vectorstore` 的调用点入手。

## 5. 故障定位表

| 症状 | 类别 | 定位方法 |
| --- | --- | --- |
| 启动即退出，报缺 `.db`/collection | 索引 | `ls data/milvus_lite/marxos_corpus_v2.db`；核对 `config/index_manifest_v1.json` 与 `scripts/validate_index_manifest.py` |
| `/readyz` `index_path: false` | 索引 | 检查 `MILVUS_URI`/`MILVUS_COLLECTION`（默认 profile 已内置，勿在 `.env` 误覆盖） |
| `/readyz` `llm_api_key: false` | 配置 | `.env` 的 `DEEPSEEK_API_KEY` 缺失或为空 |
| 回答为空/报 `connection`/`401`/`429` | LLM | 密钥、网络、余额；`logs/api_ask_metrics.jsonl` 查看失败轮次 |
| 书目/引文题报“未检索到” | 语料 | OCR 缓存目录缺失：`ls data/ocr_cache_text_layer/`；paragraph 记录缺失：检查 `data/artifacts/corpus_v2/paragraph_records_enriched_v2_1.jsonl` |
| 端口占用报错 | 端口 | `MARXOS_WEB_PORT=7861` 换端口；macOS `lsof -i :7860` 找占用进程 |
| 启动段错误（macOS ARM） | 配置 | 确认 `OMP_NUM_THREADS=1`（app.py/启动脚本已内置，勿在外部脚本里覆盖为多线程） |
| 检索明显变差 | 索引 | manifest 校验是否通过；是否误用了旧 text_layer DB（回滚入口见 `.env.example` 注释） |

## 6. 回滚

切回旧 P0 基线：`.env` 中取消底部注释三组变量（`MARXOS_CORPUS_PROFILE=me_full`、`MARXOS_RETRIEVAL_PROFILE=milvus_bgem3_stable`、旧 `MILVUS_URI`/`MILVUS_COLLECTION`/`PARAGRAPH_CACHE_PATH`），重启即可。旧库 `marxos_text_layer_bgem3.db` 需在部署包中保留。

## 7. 暂不做的（task.md P5 边界）

- 容器化（Docker/k8s）
- Milvus Standalone 服务化
- 目标机器重建索引（DB 作为资产直接分发）
- 生产模式过滤无关问题（域外问答暂走 flash 模型通用回答）
