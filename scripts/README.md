# scripts 目录说明

这个目录里的脚本现在按用途大致分成三类。

## 1. 构建类

- `build_page_map.py`
- `build_paragraph_cache.py`
- `build_paragraph_vectorstore.py`
- `detect_printed_page_start.py`

用途：

- 处理 OCR/cache/page map
- 构建 chunk / paragraph 向量库

## 2. 评测类

- `check.py`
- `evaluate_retrieval.py`
- `evaluate_citation_pages.py`
- `evaluate_eval_dataset.py`
- `regression_smoke.py`
- `topic_conversation_regression.py`

用途：

- 跑快速回归
- 跑完整检索/回答/页码评测

推荐入口：

```powershell
venv\Scripts\python.exe scripts\check.py --mode quick
venv\Scripts\python.exe scripts\check.py --mode full
```

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
