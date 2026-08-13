# MarxOS 文档数据契约 v1

当前契约版本为 `document-record/v2`，实现位于 `marxos/data/document_contract.py`。新生成的数据必须显式携带版本；旧 cache 可读取，但审计报告会显示其契约字段覆盖率。

## 固定处理链路

```text
PDF 文本层 / OCR
→ 页面清洗（页眉页脚、乱码、页面类型）
→ paragraph record
→ semantic parent
→ semantic child
→ Milvus row
```

清洗只能发生在页面进入 paragraph 之前。parent、child 和 Milvus 构建只能继承、组合或裁剪已清洗文本，不得各自重新解释 OCR。

## 三类检索单元

| retrieval_unit | 文本来源 | 主 ID | 父级关系 |
| --- | --- | --- | --- |
| `paragraph` | 相邻页面行合并后的自然段 | `paragraph_id` | `parent_paragraph_id` 指向自身 |
| `semantic_parent` | 同来源、书目、篇名和章节下的连续 paragraph | `paragraph_id` / `semantic_parent_id` | `child_source_paragraph_ids` 保存全部来源段落 |
| `semantic_child` | paragraph 或 semantic parent 的定长子块 | 子块 ID | `parent_paragraph_id` 指向被切分记录 |

`paragraph_child` 和 `milvus_passage` 仅作为旧索引兼容输入；新构建数据统一写 `semantic_child`。

## 必须继承的字段

- 书目：`source`、`book`、`volume`、`article`、`section`
- 页码：`pdf_page_start/end`、`printed_page_start/end`、`citation_page_start/end`、`citation_page_type`
- lineage：`paragraph_id`、`parent_paragraph_id`、`page_span`、`source_page_ids`
- provenance：`text_source`、`page_number_source`、`cleaning_reasons`
- 契约：`document_record_version`、`retrieval_unit`

内部缺失页码使用 `None`。Milvus 边界如必须使用 `-1`，读取后应恢复成缺失值。通用 `page` 表示引用页，PDF 页只使用 `pdf_page`。

## 质量门禁

运行：

```bash
.venv/bin/python scripts/audit_document_pipeline.py \
  --input data/paragraph_cache_text_layer.jsonl \
  --report logs/document_pipeline_audit.json
```

报告采用 `document-audit/v1`，稳定输出 issue code、记录 ID、来源和页码。错误级问题返回状态码 1；输入损坏返回 2。审计完全离线，不加载 LLM、embedding 或 Milvus。

质量代码包括 `PAGE_REQUIRED`、`PAGE_RANGE_REVERSED`、`FRONT_MATTER_LEAK`、`ARTICLE_MISSING`、`EMPTY_RETRIEVAL_TEXT`、`MOJIBAKE_REMAINS`、`DUPLICATE_TEXT` 和 `FOOTNOTE_ORPHAN`。重复文本与孤立脚注先作为 warning 供人工复核，其余为阻断错误。
