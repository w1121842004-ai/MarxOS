# 评测说明

这份文档说明仓库里几套常用评测各自覆盖什么、该在什么场景下运行，以及结果应该怎么理解。

如果只关心“小模型意图识别”的训练集、验证集、泛化测试集，请看 [intent_datasets.md](intent_datasets.md)。本页主要说明 RAG 检索、引文页码和端到端回答评测。

## 1. 评测分层

当前主线有三层检查：

1. `scripts/check.py`
   - 日常快速回归入口
   - `--mode quick` 适合改代码后先做一轮冒烟
   - `--mode full` 会串起更完整的评测

2. `scripts/evaluate_retrieval.py`
   - 主要检查检索层
   - 关注“问题能不能命中正确文献、正确篇目、正确概念来源”

3. `scripts/evaluate_eval_dataset.py`
   - 主要检查端到端回答层
   - 关注“最终回答是否落在正确文献范围内，是否出现明显误引或幻觉”

另外还有：

- `scripts/evaluate_citation_pages.py`
  - 检查引文页码细化和出处页标注

## 2. 检索评测题型

`scripts/evaluate_retrieval.py` 里的主问题集目前分成四组：

### `core_title`

检查“篇名/著作名定位”。

适合发现的问题：

- 标题约束没有生效
- 检索命中了相关材料，但篇目错了
- rerank 把正确篇目压到了后面

### `core_quote`

检查“经典引文精确命中”。

适合发现的问题：

- 引文归一化不稳
- exact quote / backstop 没接住
- 正确出处在候选里，但 top1 不是它

### `concept`

检查“概念解释类检索”。

适合发现的问题：

- 概念词重排不够强
- 定义句排序不稳定
- 同主题不同篇目之间的优先级不合理

### `negative`

检查“该拒答时能不能拒答”。

适合发现的问题：

- 检索被伪经典句骗走
- 明显不存在的出处被硬匹配到材料
- 约束不足导致误命中

## 3. 端到端评测关注什么

`scripts/evaluate_eval_dataset.py` 读取根目录下的 `eval_dataset.json`，逐条运行完整问答流程，再检查：

- 回答是否命中预期文献
- 负例是否保持拒答
- 是否出现明显幻觉信号
- 最终输出的出处信息是否在合理范围内

这套评测更接近真实用户体验，所以当 retrieval 评测是绿的，但这里还有失败时，通常说明问题在：

- 回答拼装
- 引文格式化
- 最终审计逻辑
- 某些 query intent 分支

## 4. 常用运行方式

### 快速回归

```powershell
venv\Scripts\python.exe scripts\check.py --mode quick
```

### 完整回归

```powershell
venv\Scripts\python.exe scripts\check.py --mode full
```

当前 `full` 会覆盖：

- `scripts/regression_smoke.py`
- `tests/test_run_query_regressions.py`
- `scripts/evaluate_retrieval.py`
- `scripts/evaluate_citation_pages.py`
- `scripts/evaluate_eval_dataset.py`

### 单独看检索层

```powershell
venv\Scripts\python.exe scripts/evaluate_retrieval.py
```

### 单独看端到端数据集

```powershell
venv\Scripts\python.exe scripts/evaluate_eval_dataset.py
```

### 单独看引文页码

```powershell
venv\Scripts\python.exe scripts/evaluate_citation_pages.py
```

## 5. 什么时候该跑哪一套

- 只改了文档、README、纯注释：
  - 一般不用跑重评测

- 改了 prompt、intent、答案拼装、trace：
  - 先跑 `check.py --mode quick`

- 改了 retrieval、rerank、chunk、citation refine、页码逻辑：
  - 直接跑 `check.py --mode full`

- 改了 OCR、cache、vectorstore 构建链路：
  - 跑 `full`
  - 必要时配合 `scripts/audit.py` 做专项抽查

## 6. 结果怎么读

一个比较稳的顺序是：

1. 先看 `quick` 是否通过  
   如果这里就红了，先不要急着跑更重的评测。

2. 再看 `evaluate_retrieval.py`  
   如果失败集中在 `concept`，优先看概念重排和定义句排序。  
   如果失败集中在 `core_quote`，优先看 exact quote / alias / backstop。

3. 最后看 `evaluate_eval_dataset.py`  
   如果 retrieval 是绿的，但端到端不是，通常问题不在底层召回，而在回答出口。

## 7. 题目来源

- 检索评测主问题集定义在 [scripts/evaluate_retrieval.py](../scripts/evaluate_retrieval.py)
- 端到端数据集定义在 [eval_dataset.json](../eval_dataset.json)

如果后面要补新题，建议优先补：

- 经典引文的同义问法
- 概念解释的口语化问法
- 会误提到别的书名的干扰题
- 拒答边界题
