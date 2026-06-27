# Intent 数据集说明

这份文档只说明“意图识别”相关数据。RAG 检索、页码、端到端回答评测仍看 `docs/eval_questions.md`。

## 推荐结论

当前 intent 小模型建议采用：

- 训练主集：`data/intent_dataset_ai_assisted_2000/intent_train.json`
- 训练校准：可混入 `eval_dataset_v2.json` 和 `eval_dataset_me_200.json` 映射后的 intent 标签
- 调参验证：`data/intent_dataset_ai_assisted_2000/intent_validation.json`
- 常规测试：`data/intent_dataset_ai_assisted_2000/intent_test.json`
- 泛化测试：`data/intent_generalization_400/intent_generalization_400.json`

不建议再用 `data/intent_dataset_10000/` 作为最终效果指标。它适合预训练或蒸馏暖启动，但模板分布较强，测试分数容易虚高。

## 数据集分层

| 数据集 | 数量 | 主要用途 | 是否可训练 | 是否可做最终评测 | 备注 |
|---|---:|---|---|---|---|
| `data/intent_dataset_ai_assisted_2000/` | 2000 | intent 银标训练、验证、测试 | 是 | 可作常规测试 | 有统一标注标准、置信度、边界样本、复核标记 |
| `data/intent_generalization_400/` | 400 | held-out 泛化测试 | 否 | 是 | 与当前训练/验证/测试/旧标注 exact query 不重复 |
| `data/intent_dataset_10000/` | 10000 | 预训练、蒸馏暖启动 | 可选 | 不建议 | 模板化较强，容易高估泛化 |
| `eval_dataset_v2.json` | 200 | RAG/work-level 评测；可映射 intent 做校准 | 可选校准 | 不建议单独做 intent 最终评测 | 早期标签边界和当前 intent 标准不完全一致 |
| `eval_dataset_me_200.json` | 200 | 马恩全集 QA/引用/端到端评测；可映射 intent 做校准 | 可选校准 | 不建议单独做 intent 最终评测 | 由 `scripts/build_me_qa_eval_dataset.py` 随机抽材料生成 |

## 当前推荐命令

训练当前轻量 intent classifier：

```bash
.venv/bin/python scripts/build_intent_classifier.py \
  --intent-dataset-dir data/intent_dataset_ai_assisted_2000 \
  --include-manual-labels \
  --output data/intent_classifier.pkl
```

重新生成 2000 条 AI 辅助银标：

```bash
.venv/bin/python scripts/build_ai_assisted_intent_dataset.py
```

重新生成 400 条 held-out 泛化测试：

```bash
.venv/bin/python scripts/build_intent_generalization_eval.py
```

生成 10000 条模板数据：

```bash
.venv/bin/python scripts/build_intent_dataset.py
```

## 标注来源

### `ai_assisted_label_v1`

位置：`data/intent_dataset_ai_assisted_2000/`

特点：

- 有 `LABELING_GUIDE.md`
- 每条有 `confidence`
- 每条有 `label_reason`
- 每条有 `boundary_case`
- 低置信边界题有 `needs_human_review`

这是当前最适合作为训练主集的 intent 数据。

### `ai_assisted_generalization_eval_v1`

位置：`data/intent_generalization_400/`

特点：

- 只有 test split
- exact query 不重合当前已知训练/验证/测试/旧标注数据
- 用于看真实一点的泛化能力

这份数据不要混入训练。

### `synthetic_intent_v1`

位置：`data/intent_dataset_10000/`

特点：

- 数量大
- 类别均衡
- 模板生成痕迹较强

适合预训练或蒸馏暖启动，不适合作为最终泛化指标。

## 最近一次结果口径

当前 `data/intent_classifier.pkl` 使用 `data/intent_dataset_ai_assisted_2000/` 加旧 400 条校准训练。

在 `data/intent_dataset_ai_assisted_2000/intent_test.json` 上：

- accuracy: 95.0%
- macro F1: 95.1%

在 `data/intent_generalization_400/intent_generalization_400.json` 上：

- 纯规则 accuracy: 43.8%
- 纯小模型 accuracy: 80.2%
- 当前运行时混合 accuracy: 83.0%
- clean 子集 accuracy: 87.3%
- hard boundary 子集 accuracy: 76.1%

这些数字不要和 `data/intent_dataset_10000/` 的模板测试分数直接比较。

## 命名约定

后续 intent 数据建议放在 `data/intent_*` 下：

- `intent_dataset_*`: 可训练或可切分数据集
- `intent_generalization_*`: 不进入训练的泛化评测集
- `intent_*_review`: 人工复核候选

字段约定：

- `query`: 用户问题
- `intent`: 7 类标准 intent
- `split`: `train` / `validation` / `test`
- `source`: 标签来源
- `confidence`: 标注置信度
- `boundary_case`: 是否边界题
- `needs_human_review`: 是否优先人工复核
- `label_reason`: 标注理由

## 注意事项

- 不要把 `intent_generalization_400` 混入训练。
- 不要把 `intent_dataset_10000` 的 test 分数当最终效果。
- `eval_dataset_v2.json` 和 `eval_dataset_me_200.json` 主要仍服务 RAG/端到端评测；映射到 intent 时要接受标签边界不完全一致的问题。
- 真正要提升 intent 泛化，优先复核 `needs_human_review=true` 的样本，而不是继续堆模板数量。
