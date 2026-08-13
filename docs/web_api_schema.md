# Web API 响应 Schema

`/api/ask`（JSON）与 `/api/ask_stream`（SSE）由同一函数 `MarxOSHandler._run_ask_payload` 构建，返回**同一字段集合**。流式端点只是额外发送 `status` 进度事件，最终以 `final` 事件携带完整 payload。

## 请求

```json
{
  "query": "什么是剩余价值？",
  "mode": "auto | fast | precise | standard | deep",
  "history": [{"role": "user", "text": "..."}, {"role": "bot", "text": "...", "intent": "...", "evidence": [...], "topic": {...}}]
}
```

## 成功响应（200）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `intent` | string | 意图分类结果 |
| `mode` | string | 实际执行模式（fast/standard/deep/local/precise） |
| `answer` | string | 最终回答文本（本地/LLM/拒答） |
| `path` | string | 回答类别：`local_lookup` / `local_view` / `llm` / `refusal` / `out_of_domain` / `ambiguous_locator` / `trace_only` |
| `evidence` | array | 证据卡（见下） |
| `citation_audit` | object | `{ok, issues, evidence_count, answer, mode, content_verification?, crag_report?}` |
| `topic` | object | `{topic_id, topic_label, topic_section}` |
| `crag` | object | CRAG 报告 `{ok, score, path, threshold, issues?}` |
| `timing` | object | 各阶段耗时（ms）与 `total`、`mode`、`intent` |
| `elapsed_ms` | int | 服务端耗时 |
| `memory_turns` | int | 参与本次请求的历史轮数 |

### evidence 条目字段

| 字段 | 说明 |
| --- | --- |
| `id` / `citation` / `detailed_citation` / `sentence_citation` | 证据编号与三种出处渲染（短/详细/句级） |
| `source` / `source_file` / `series` / `volume` / `article` / `section` | 出处与篇目 |
| `printed_page` / `citation_page` / `pdf_page` | 印刷页/引用页/PDF 页 |
| `paragraph_id` / `line_start` / `line_end` / `char_start` / `char_end` | 定位字段 |
| `match_type` | `exact_quote` / `locator_backstop` / `cache_backstop` / `vector_candidate` / `sparse_candidate` / `paragraph_vector_candidate` |
| `confidence` | 候选置信度（vector_candidate 为 0.0） |
| `is_letter` / `letter_title` / `no_page_citation` / `citation_mode` | 书信/无页码引用模式 |
| `excerpt` | 原文片段（≤240 字） |

前端证据卡按 `match_type` 标注：原文核对 / 定位提示 / 页段回退 / 未确认候选 / 稀疏候选 / 段落候选。

## 错误响应

| 状态码 | 字段 |
| --- | --- |
| 400 | `{"error": "问题不能为空"}` / `{"error": "无效 JSON"}` |
| 404 | 标准 send_error |
| 500 | `{"error": "服务异常: ..."}` |

流式端点以 `error` 事件发送同样的 `{"error": ...}`；非流式以状态码 + body 返回。前端将错误渲染为红色错误气泡（不当作正常回答），并附「错误」badge。

## 流式事件序列

```text
event: status   data: {"message": "正在分析问题..."}
event: status   data: {"message": "正在检索证据（deep）..."}
event: status   data: {"message": "正在生成回答..."}
event: final    data: <200 payload 全量>
（失败时：event: error data: {"error": "..."}）
```

前端超时保护：180 秒未收到 `final` 自动中止并显示「请求超时」。页面头部 `/readyz` 轮询徽标：系统就绪（绿）/ 部分可用（黄）/ 服务不可用（红）。

## 回归测试

- `tests/test_web_api.py::test_stream_and_json_payloads_have_same_fields`：流式 final 与非流式 200 的顶层字段集合一致，evidence 条目字段集合一致。
- `tests/test_web_api.py::test_error_payloads_have_consistent_shape`：空查询在两端点返回相同的 `{"error": ...}` 形状。
- `tests/test_web_api.py::test_explicit_work_page_query_does_not_reuse_previous_evidence`：多轮追问页码不复用回归。
- `tests/test_run_query_regressions.py::test_topic_followup_does_not_leak_into_unrelated_question`：专题追问不越界回归。
