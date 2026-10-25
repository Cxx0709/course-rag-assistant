# 真实模型评测与人工复核准备 v6

## 实测结果（2026-10-25）

本机 `qwen3.5:9b` 在关闭思考、`num_predict=512`、60 秒请求超时的配置下，完成混合检索的 30+10 题开发集评测，评测阶段约 86.93 秒。
报告：`eval/reports/quality-v6-chat.json`。

| 指标 | 离线混合检索 + 摘录 | 混合检索 + 聊天模型 |
|---|---:|---:|
| Recall@8 | 80.00% | 80.00% |
| 可回答题最终拒答 | 0/30 | 30/30 |
| 答案点字面覆盖 | 57.78% | 0.00% |
| 不可回答题最终拒答 | 10/10 | 10/10 |
| 模型调用次数 | 0 | 30 |
| 模型异常回退次数 | 0 | 0 |

30 份模型输出均被现有校验器全部拒绝，原因均为 `unsupported_citations`。
其中 28 份原始输出缺少可识别的引用标记；另 2 份将引用置于独立的 `[1]` / `[2]` 行，无法与前一句建立当前校验器要求的引用关联。
这不是模型连接失败，也不是检索效果下降，而是实际生成输出与当前引用约束不匹配。
10 道反例均在前置证据筛选阶段被拒绝，未调用模型，不能据此宣称模型本身拒答能力达标。

离线 BM25、纯向量、混合三种模式的拒答率、答案点字面覆盖和参考页码引用精确率均与 v5 一致。
离线与聊天模式分别导出 120 条和 40 条待人工复核记录；这些不是新增问题，也没有完成独立人工复核。
本轮 **82 项测试通过**，修改的 Python 文件 Ruff 检查通过。

**当前优先事项：修复模型生成与引用格式的衔接，再在冻结题集上复测。**
可以研究让模型选择带编号的原文句子，再由程序附加可信来源；不能仅放松校验来让指标看起来变好。
本次没有修改生成逻辑，也没有将本次失败包装为质量验收通过。

## 本轮范围

本轮补齐评测工具的真实聊天模型调用、回退可观测性和人工复核材料导出。
沿用既有 30 道可回答题和 10 道不可回答题，它们仍是开发集；没有扩充成独立测试集，也没有代替人工完成复核。

`scripts/evaluate_quality.py` 默认仍为离线摘录。显式使用 `--chat` 才调用配置的聊天模型，缺少配置时报错。
显式选择 `hybrid+rerank` 时缺少重排服务也报错，不会把未运行的重排当作成功。
不指定模式时，缺少的可选重排模式继续写入报告的 `skipped`。

## 报告口径

- `model_output_checked`：模型返回了输出，之后经过引用校验；不表示答案正确，甚至可能被全部拒绝。
- `fallback_extractive`：调用异常后应用回退至摘录，计入 `generation_failures`，不能计作模型生成成功。
- `skipped_no_evidence`：前置证据筛选拒答，未调用模型。这类拒答不能证明聊天模型自身知道何时拒答。
- `offline_extractive`：未启用聊天模型。

逐题报告保存原始模型输出、校验后答案、被移除的断言、实际引用上下文、参考页码和预期答案点。
同时记录生成和问答耗时、模型名及配置、输入与代码哈希。
异常仅记录类型，不保存可能携带凭据的错误消息。

检索评测与答案评测复用同一批检索结果，避免重复调用远程重排导致两项评测使用不同证据。
报告逐模式落盘，运行完成后 `completed` 才设为 `true`。
`--limit N` 取每个题目分组的前 N 题，并标记 `smoke_test`，不能用于正式验收。
小规模评测沿用完整问题向量缓存，不覆盖为子集缓存。

## 复现命令

离线三个检索模式（沿用本地语料、索引和缓存）：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_quality.py --dense
```

真实本机模型 + 混合检索完整开发集：

```powershell
$env:NO_PROXY = '127.0.0.1,localhost'
$env:COURSE_RAG_OLLAMA_CHAT_MODEL = 'qwen3.5:9b'
$env:COURSE_RAG_OLLAMA_CHAT_THINK = 'false'
$env:COURSE_RAG_OLLAMA_CHAT_NUM_PREDICT = '512'
$env:COURSE_RAG_OLLAMA_CHAT_TIMEOUT = '60'
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_quality.py --dense --chat --modes hybrid --output eval/reports/quality-v6-chat.json
```

配置真实重排服务的 `COURSE_RAG_RERANK_BASE_URL`、`COURSE_RAG_RERANK_API_KEY` 和可选模型名后：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_quality.py --dense --chat --modes hybrid hybrid+rerank --output eval/reports/quality-v6-rerank-chat.json
```

## 人工复核

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/export_quality_review.py eval/reports/quality-v6-chat.json --output eval/reviews/quality-v6-chat-pending.jsonl
```

导出器仅接受执行完成的 v3 报告，拒绝覆盖已有复核文件，以免清除人工填写内容。
每条记录关联报告 SHA256，保留 `pending` 和空判断值。导出完成不等于复核完成。

由独立复核者对照原始教材、参考答案点以及系统实际采用的证据，分别判断：

1. 答案是否正确、完整且切合问题。
2. 引用是否支持对应断言；参考页码不是全部相关页码的穷举。
3. 拒答是否合适；无答案时，正确性和完整性可以保留空值，在备注中说明。

完成后填写 `reviewer`、`reviewed_at`、各判断值和说明，再将 `status` 改为 `completed`。
本轮尚未实现人工复核汇总评分，也没有设定或宣称通过通用质量验收阈值。

## 后续验收缺口

- 真实 BGE 重排尚需可用服务配置和实测。
- 聊天模式仍使用完整原句校验，自由改写不能通过；需据真实输出分析失败原因。
- 题集尚未达到计划的 60–100 道正例、15–20 道反例，且仍需独立人工复核与未用于调参的留出集。
- 本轮未开展 Web/API、先修知识关联、跨教材对比和测验模块开发。
