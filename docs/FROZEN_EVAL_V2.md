# FinResearch Frozen Eval V2

## 冻结状态

V2 已完成工程冻结：Retrieval 128 条（SHA-256 `2a31b7fc...bc6c58`），Agentic RAG 74 条（SHA-256 `b00e5368...b19b1`）。题目、语义 gold 和可执行 schema 已校验，并已完成第二审阅者独立复核（由项目所有者确认，不在仓库保存复核者身份或私人笔记）。仍需保留的数据边界是：

1. robustness 改写共享 V1 语义 gold，128 条不等于 128 个独立事实；
2. 这是项目自建集，不是公开 Benchmark，不宣称行业 SOTA。

Retrieval 类型分布：cross-company 28、same-section 8、single-fact 43、table/cross-chunk 49。Agent 集包含 cross-company 16、cross-section 2、out-of-scope 8、single-fact 30、unanswerable 18。

## R0—R2 真实检索消融

固定 Top-5，语料为比亚迪与宁德时代 2024 年报 881 个 chunks。R1 使用 BM25 Top-20 + `bge-reranker-base`；R2 在不改变 R1 排序的情况下加入有界邻块/表头扩展。

| 组别 | Recall@5 | MRR | NDCG@5 | Slot Coverage | Complete Evidence | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| R0 BM25 | 79.95% | 65.56% | 68.20% | 78.12% | 75.00% | 4.007 ms |
| R1 + Cross-Encoder | 86.46% | 72.15% | 74.04% | 85.55% | 82.81% | 1,866.464 ms |
| R2 + Neighbor/Header | 89.97% | 72.50% | 75.39% | 89.84% | 88.28% | 1,866.928 ms |

Reranker 带来了主要质量收益，也是延迟的主要来源。邻块扩展几乎不增加计算成本，将 Complete Evidence 从 82.81% 提高到 88.28%，但平均上下文从 5 块增至 7 块。结果文件 SHA-256 为 `1a579ac8675be7d98d0020a50624c88c8de2726ba4bfdb234fc0a8a9df0abbc5`。

## R3—R5 DeepSeek 冻结消融

在用户一次性明确授权后，使用 `deepseek-v4-flash`、温度 0、同一 74 条冻结集和同一 R2 检索基座完成盲测。只发送评测问题和每题经过 Top-K 与 Token 预算裁剪的 evidence blocks，未上传 PDF、密钥或其他私人数据。

| 组 | 方法 | Route Acc. | Strict Success | Citation Hit | Claim Support | Abstention F1 |
|---|---|---:|---:|---:|---:|---:|
| R3 | R2 + decomposition/slots | 54.05% | 25.68% | 47.30% | N/A | 56.18% |
| R4 | R3 + critic/二次检索 | 54.05% | 24.32% | 45.95% | N/A | 56.18% |
| R5 | R4 + claim/answer critic | 54.05% | 18.92% | 39.19% | 33.33% | 53.61% |

| 组 | 二轮增量召回 | 无效二轮 | Mean/P95 | Calls | Tokens |
|---|---:|---:|---:|---:|---:|
| R3 | 0.00% | 0.00% | 4.44 / 7.15 s | 151 | 329,303 |
| R4 | 1.35% | 72.97% | 7.48 / 12.80 s | 189 | 500,944 |
| R5 | 1.35% | 74.32% | 7.82 / 13.52 s | 215 | 621,566 |

结论不是“环节越多越好”：

- R4 相比 R3 多消耗 171,641 tokens，平均多 3.03 秒，但二轮增量召回仅 1.35%，Strict Success 下降 1.35 个百分点。当前 follow-up 改写大多无效，应在候选集饱和时跳过。
- R5 将拒答 Recall 提到 100%，但只回答 3/74 题，出现明显过度拒答；Strict Success 进一步下降 5.41 个百分点。需要分开校准确定性 critic 与模型 critic，而不是放宽安全门禁。
- Route Accuracy 在三组中均为 54.05%，是共同上限瓶颈；应先修复 `single_fact`/`same_section_multi_slot`/`unanswerable` 的路由 hard negatives。

三组共 555 次调用、1,451,813 tokens，0 个运行错误。按 2026-10-11 [DeepSeek Flash 官方 off-peak 价格](https://api-docs.deepseek.com/quick_start/pricing/)，由于当前客户端没有保存 cache hit/miss 拆分，估算费用范围为 0.0662—0.2644 美元，最终以 API 账单为准。

## Bad Case

`results/bad_cases_frozen_v2.json` 保留 12 个运行中的真实失败：6 个 R2 仍未解决，6 个由 R0 失败但被邻块扩展修复。根因包括：

- 表格目标未进入 Top-20 candidate，邻块扩展无法补救；
- Cross-Encoder 将原本命中的稀疏检索结果排出 Top-5；
- 跨公司问题只召回一端，相关不等于证据完整；
- 表头与数据行分块后，需要邻块关系传播槽位覆盖。

Bad Case 文件 SHA-256 为 `543d5db334d78a17f85bdd47e5f3ba510f7e8ba59b126b8e9e541faeee8016a4`。

`results/agent_bad_cases_frozen_v2.json` 另保留 12 个不含年报原文的 Agent 失败摘要，根因包括路由错配、无增量二次检索、过度拒答和 claim-citation 绑定失败。

## 复现与边界

```bash
./.venv/bin/python -m finresearch.v2_data
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  -m finresearch.retrieval_v2_cli \
  --documents data/processed/byd_2024_chunks_v3.jsonl \
              data/processed/catl_2024_chunks_v3.jsonl \
  --cases data/eval/frozen_v2/retrieval_frozen_v2.jsonl \
  --output results/retrieval_ablation_frozen_v2.json
./.venv/bin/python -m finresearch.failure_report
./.venv/bin/python -m finresearch.agent_report_v2
```

公开仓库只保留评测 schema、doc id、汇总指标和根因。包含短引用的逐题 R3—R5 原始记录仍由 `.gitignore` 排除；不分发年报 PDF 或完整解析文本。
