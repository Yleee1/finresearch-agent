# FinResearch 冻结评测 V1

## 评测协议

- 语料：比亚迪与宁德时代 2024 年报，519 页、881 chunks。
- 检索冻结集：40 条，覆盖财务、经营、产品技术、人员与 ESG 信息。
- Agent 冻结集：24 条，其中 20 条可回答、4 条应拒答，包含 4 条跨公司比较。
- 隔离规则：冻结集与 seed 开发集没有相同问题，也没有重合的 gold chunk。
- 防篡改：题集与语料的 SHA-256、条数和类别分布记录在 `data/eval/frozen_v1_manifest.json`，并由 `tests/test_frozen_eval.py` 自动校验。
- 运行规则：题集先冻结、后盲测；首轮结果不用于修改 prompt、检索参数、gold label 或 checkpoint。

## 检索结果

Top-K 固定为 5，Reranker 的候选集固定为 Top-20。延迟在 Apple M5 16GB、CPU 环境测得，不包含离线模型首次下载。

| 方法 | Hit@5 | Recall@5 | MRR | NDCG@5 | 平均查询延迟 |
|---|---:|---:|---:|---:|---:|
| BM25 | 87.50% | 86.25% | 71.33% | 74.14% | 4.02 ms |
| BGE Dense | 57.50% | 55.00% | 45.50% | 46.59% | 3.01 ms |
| 等权 RRF Hybrid | 82.50% | 82.50% | 61.54% | 66.38% | 7.31 ms |
| BM25 + BGE Reranker | **95.00%** | **95.00%** | **75.92%** | **80.90%** | 2,567.34 ms |
| Hybrid + BGE Reranker | **95.00%** | **95.00%** | **75.92%** | **80.90%** | 2,339.56 ms |

BM25 + Reranker 相比 BM25 将 Recall@5 提高 8.75 个百分点、NDCG@5 提高 6.76 个百分点，但 CPU 平均查询延迟从 4.02 ms 增至 2.57 s。Hybrid 没有带来精度收益，还增加了 Dense 建库与维护成本，因此当前默认配置保持 BM25 + Reranker。

## Agent 结果

Agent 使用 BM25 + Reranker、Top-5、候选 Top-20、最多两轮检索和 DeepSeek。严格成功要求 route、status、事实覆盖和引用命中全部正确。

| 指标 | 首轮冻结集结果 |
|---|---:|
| Route Accuracy | 75.00% |
| Status Accuracy | **100.00%** |
| Mean Fact Coverage | **97.92%** |
| Citation Hit Rate | **91.67%** |
| Strict Success Rate | 62.50% |
| Grounded Task Success（诊断项，不要求 route） | 87.50% |
| 平均端到端延迟 | 7.57 s |
| API 调用 / Token | 92 / 225,852 |

`Grounded Task Success` 是运行后补充的诊断指标，用于把答案质量与路由分类错误拆开；它不是预注册主指标。原始严格分数保持不变，不用人工复核结果覆盖机器指标。

## Bad Case 与根因

| Case | 现象 | 根因与处理方向 |
|---|---|---|
| fr010 | Top-5 找到了分部报告中的境内/境外金额，但没有召回直接披露占比的管理层表格 | 重复披露口径不同；增加“金额/占比”槽位完整性排序 |
| fr028 | Top-5 找到动力电池收入及同比，但没有同时召回含毛利率的后续表格块 | 表头与数值跨块、证据不完整；检索后合并相邻 chunk |
| fa002 | 同一段落中的纳税额和捐资额被路由成 multi | 多属性过度拆解；路由器需要区分“同段多槽位”和“多来源” |
| fa004 | 回答将 159,608,576 千元合理取整为 159,609 百万元，Exact Match 只计中一个事实 | 数值单位与舍入归一化不足；增加 Decimal 与单位换算评分器 |
| fa008 | 回答引用 p59/p220 的等价利润分配披露，gold 只标 p3，机器判定 Citation Miss | 年报重复披露导致 gold 不完备；下一版预注册等价证据集合 |
| fa009 | 同一 ESG 段落的三个属性被路由成 multi，答案和引用正确 | 路由分类过度敏感，不应按槽位数量机械拆解 |
| fa010 | 同一回收能力段落的三个数值被路由成 multi | 同段多属性误判；加入路由 hard negative |
| fa011 | 同一销量概述中的总量与两个分项被路由成 multi | 层级指标误判为多来源任务 |
| fa015 | 模型选择 multi，原 gold 标 single；答案、事实和引用均正确 | 冻结标签与既有“多个原因可 multi”规则存在歧义；保留原分，V2 明确判定规范 |
| fa016 | 跨 2024/2023 两个年份被模型路由为 multi，原 gold 标 single | gold 与既有“多个年份为 multi”规则冲突；保留原分并在 V2 修订标注规范 |
| fa019 | 两家公司均引用年报中的等价分红披露，但不是冻结 gold 指定页 | 多公司重复披露造成 Citation Hit 假阴性；不修改 V1，V2 使用证据集合判分 |

## 可复现命令

```bash
cd finresearch-agent
PYTHONPATH=src .venv/bin/python -m unittest tests.test_frozen_eval -v

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m finresearch.cli \
  --documents data/processed/byd_2024_chunks_v3.jsonl data/processed/catl_2024_chunks_v3.jsonl \
  --queries data/eval/retrieval_frozen_v1.jsonl \
  --retrievers bm25 dense hybrid --top-k 5 --candidate-k 20 --device cpu \
  --output results/retrieval_frozen_v1.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m finresearch.rerank_cli \
  --documents data/processed/byd_2024_chunks_v3.jsonl data/processed/catl_2024_chunks_v3.jsonl \
  --queries data/eval/retrieval_frozen_v1.jsonl \
  --base bm25 --candidate-k 20 --top-k 5 --device cpu \
  --output results/rerank_bm25_frozen_v1.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m finresearch.agent_eval_cli \
  --documents data/processed/byd_2024_chunks_v3.jsonl data/processed/catl_2024_chunks_v3.jsonl \
  --cases data/eval/agent_frozen_v1.jsonl \
  --output results/agent_eval_frozen_v1.jsonl \
  --summary results/agent_eval_frozen_v1_summary.json \
  --retriever bm25-rerank --top-k 5 --candidate-k 20 --max-iterations 2 --device cpu
```

## 简历可用结论

> 构建与开发集零问题、零 gold chunk 重合的 40 条冻结检索集，BM25 + Cross-Encoder Reranker 取得 Recall@5 95.00%、NDCG@5 80.90%；在 24 条冻结 Agent 测试上实现 Status Accuracy 100%、Fact Coverage 97.92% 和 Citation Hit 91.67%，并对路由误判、重复披露与数值归一化 Bad Case 完成归因。

这里不写“Agent 成功率 100%”。V1 的严格成功率是 62.50%，诊断口径的 Grounded Task Success 是 87.50%。
