# scripts 目录说明

这个目录里的脚本现在按用途大致分成三类。

## 1. 构建类

- `build_page_map.py`
- `build_paragraph_cache.py`
- `build_semantic_child_vectorstore.py`
- `build_paragraph_vectorstore.py`
- `build_milvus_collection.py`
- `build_intent_classifier.py`
- `build_intent_dataset.py`
- `build_ai_assisted_intent_dataset.py`
- `build_intent_generalization_eval.py`
- `detect_printed_page_start.py`

用途：

- 处理 OCR/cache/page map
- 构建“小块召回、大块返回”检索用的 semantic child / paragraph 向量库
- 构建 Milvus 检索 collection（当前范围：马恩全集、文集、选集）
- 构建、训练、评测 intent 小模型相关数据

默认构建目标现在使用全集优先的非 core 路径，例如
`data/paragraph_cache.jsonl`、`data/semantic_parent_cache.jsonl`、
`vectorstore/marx_reader`。`*_core` 文件和目录只作为早期文集/选集小范围测试的
兼容回退；如果确实要复现实验，需要显式设置对应环境变量。

## 2. 评测类

- `check.py`
- `evaluate_retrieval.py`
- `evaluate_citation_pages.py`
- `evaluate_eval_dataset.py`
- `evaluate_ragas.py`
- `regression_smoke.py`
- `topic_conversation_regression.py`

用途：

- 跑快速回归
- 跑完整检索/回答/页码评测
- 跑 RAGAS faithfulness / relevancy / context recall 等回答质量评测

推荐入口：

```powershell
venv\Scripts\python.exe scripts\check.py --mode quick
venv\Scripts\python.exe scripts\check.py --mode full
```

RAGAS 建议分两步跑，避免它的新依赖升级主运行环境里的 LangChain：

```bash
# 1) 用 MarxOS 主环境生成 RAGAS samples；RAGAS 回答质量指标需要 response，
#    所以正式跑 RAGAS 时要加 --generate-answers 调 DeepSeek 生成答案。
.venv/bin/python scripts/evaluate_ragas.py \
  --limit 20 \
  --top-k 8 \
  --generate-answers \
  --answer-mode fast \
  --prepare-only \
  --samples-out logs/ragas_samples.jsonl

# 2) 单独创建 RAGAS 评测环境并跑指标
python3 -m venv .venv-ragas
.venv-ragas/bin/pip install -r requirements-ragas.txt
.venv-ragas/bin/python scripts/evaluate_ragas.py \
  --input-samples logs/ragas_samples.jsonl \
  --metrics faithfulness context_recall factual_correctness \
  --judge-model deepseek-chat \
  --judge-base-url https://api.deepseek.com \
  --report logs/ragas_report_deepseek_only.json
```

如果只想先检查检索上下文，不想调用 DeepSeek 生成答案，可以去掉
`--generate-answers` 并保留 `--prepare-only`；这样只会生成 samples，不会运行
RAGAS 指标。

`answer_relevancy` 会默认调用 OpenAI embedding；如果没有 OpenAI key，结果会是
`nan`。当前推荐先跑上面的 DeepSeek-only 指标。

日常快速看检索和引用是否靠谱，可以先跑 MarxOS 确定性指标，不调用 judge：

```bash
.venv-ragas/bin/python scripts/evaluate_ragas.py \
  --input-samples logs/ragas_samples.jsonl \
  --marxos-only \
  --report logs/ragas_report_marxos_only.json
```

报告里的 `marxos_summary` 会给出预期篇目命中率、作者命中率、硬负例污染率、
答案引用标记率等项目内指标，适合和 RAGAS 分数一起看。

## 3. 审计类

- `audit.py`
- `audit_*`
- `report_api_ask_metrics.py`
- `compare_dual_retrieval.py`
- `inspect_article_map.py`

用途：

- 对某个问题点做专项排查
- 比如页码、概念 metadata、exact quote top1、段落缓存质量

推荐入口：

```powershell
venv\Scripts\python.exe scripts\audit.py list
```

## 4. 当前命名约定

- `build_*`: 生成缓存、索引、向量库
- `evaluate_*`: 正式评测
- `audit_*`: 专项审计
- `report_*`: 汇总报告

如果后面再加脚本，优先沿用这套前缀，不再混用临时命名。

## 5. Intent 数据集

意图识别相关数据集已经单独整理到：

```text
docs/intent_datasets.md
```

常用入口：

```powershell
venv\Scripts\python.exe scripts\build_ai_assisted_intent_dataset.py
venv\Scripts\python.exe scripts\build_intent_generalization_eval.py
venv\Scripts\python.exe scripts\build_intent_classifier.py --intent-dataset-dir data\intent_dataset_ai_assisted_2000 --include-manual-labels --output data\intent_classifier.pkl
```
