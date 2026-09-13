# FinResearch Agent

面向金融研报和公司公告的多跳检索与事实核查系统。项目以可复现评测为主线，逐步从稀疏检索基线扩展到 Dense Retrieval、Reranker 和迭代式 Agentic Retrieval。

## 当前阶段

当前已完成真实 PDF 到检索精排的第一条可评测链路：

- 比亚迪、宁德时代 2024 年报共 519 页、881 个页码感知 chunks
- 中文软换行修复、重复页眉页脚清理与表格行保护
- 可解释 BM25、`bge-small-zh-v1.5` Dense Retrieval、RRF Hybrid
- `bge-reranker-base` Cross-Encoder 精排
- Hit@K、Recall@K、MRR、NDCG@K 与 CPU 延迟
- 16 条 seed 查询及 12 个已归因 Bad Cases

`data/demo` 仅用于验证代码，其指标不能写入简历。

## 数据来源与版权

本仓库不分发比亚迪、宁德时代年报 PDF，也不提交由年报解析得到的完整文本块。年报版权归原发布主体所有，公开数据仅用于个人学习、研究与可复现实验：

- 比亚迪股份有限公司 2024 年年度报告：巨潮资讯网公告文件 `1222881496.PDF`。
- 宁德时代新能源科技股份有限公司 2024 年年度报告：巨潮资讯网公告文件 `1222806982.PDF`。
- 完整 URL、页数和文件 SHA-256 记录在 [`data/sources.json`](data/sources.json)。

在遵守来源网站条款和当地法律的前提下，可运行以下脚本下载到本地。脚本会校验 SHA-256；`data/raw/` 和 `data/processed/` 已被 `.gitignore` 排除：

```bash
./scripts/download_reports.sh
```

公开仓库只保留 [`data/demo/documents.jsonl`](data/demo/documents.jsonl) 中 4 个完全合成、已去除真实公司名称和真实数值的示例块，用于验证数据格式和运行链路；它们不是年报原文，也不能用于报告项目指标。

第一次接触项目时，先阅读[完整项目蓝图](docs/PROJECT_BLUEPRINT.md)，理解最终产品、端到端数据流、技术栈和面试知识地图，再继续增加功能。

当前学习进度：[Day 1 检索评测](docs/day01.md) → [Day 2 真实 PDF 解析](docs/day02.md) → [Day 3 Dense 与精排](docs/day03.md) → [Day 4 Agent 闭环](docs/day04.md)。

## 快速运行

```bash
cd finresearch-agent
PYTHONPATH=src python3 -m finresearch.cli \
  --documents data/demo/documents.jsonl \
  --queries data/demo/queries.jsonl \
  --top-k 3
```

运行测试：

```bash
cd finresearch-agent
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

真实 BM25/Dense/Hybrid 对照（首次下载模型后可离线运行）：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m finresearch.cli \
  --documents data/processed/byd_2024_chunks_v3.jsonl data/processed/catl_2024_chunks_v3.jsonl \
  --queries data/eval/retrieval_seed_v1.jsonl \
  --retrievers bm25 dense hybrid \
  --top-k 5 --candidate-k 20 --device cpu \
  --output results/retrieval_seed_v1.json
```

BM25 Top-20 后接 Cross-Encoder：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m finresearch.rerank_cli \
  --documents data/processed/byd_2024_chunks_v3.jsonl data/processed/catl_2024_chunks_v3.jsonl \
  --queries data/eval/retrieval_seed_v1.jsonl \
  --base bm25 --candidate-k 20 --top-k 5 --device cpu \
  --output results/rerank_bm25_seed_v1.json
```

当前 seed 集最佳 Hit@5 为 93.75%（BM25 + Reranker）。该集合仍小，只用于开发迭代；最终简历指标需在扩充并冻结的隔离测试集上重新报告。

Agent 实现包含问题拆解、最多两轮检索、Evidence Verifier、低证据拒答和逐字引用校验。真实 DeepSeek 冒烟运行方式见 `docs/day04.md`；`.env` 不得提交。

真实 Agent smoke 已验证：证据完整问题可返回带页码引用；口径不一致问题在两轮检索后拒答；`single/multi` 路由将简单问题延迟从约 19.07 秒降至 9.17 秒。详见 `results/agent_smoke_report.md`。

12 条 Agent seed 开发集的同版本 V2 重跑得到 100% Route/Status/Fact/Citation/Success，平均端到端 7.61 秒；由于集合很小且参与过 prompt 调整，该数字明确标记为开发指标，不能直接写成最终泛化性能。

## 冻结评测 V1

已在调参完成后冻结一套独立评测：40 条检索问题与 24 条 Agent 问题，与 seed 开发集保持零问题、零 gold chunk 重合，并保存 SHA-256 清单。首次盲测中，BM25 + Reranker 达到 Recall@5 95.00%、NDCG@5 80.90%；Agent 的 Status Accuracy 为 100%、Mean Fact Coverage 为 97.92%、Citation Hit Rate 为 91.67%，严格成功率为 62.50%。

完整协议、全部方法对照、11 个 Bad Case 和简历可用表述见 [`docs/FROZEN_EVAL_V1.md`](docs/FROZEN_EVAL_V1.md)。原始结果保存在 `results/*frozen_v1*`，失败的首次沙箱网络运行也单独保留，没有覆盖。

## 项目路线

1. BM25 与评测基线
2. 真实 PDF 解析、版面感知分块和表格保护
3. Dense Retrieval 与 BM25 + Dense RRF
4. Cross-Encoder Reranker 与 Hard Negative 实验
5. Query Decomposition 与最多两轮的迭代检索
6. Evidence Verifier、引用校验和低置信度拒答
7. FastAPI/Gradio Demo、延迟与成本报告

## 开源归因

项目会参考 RAGFlow DeepDoc 的复杂文档解析思路、BEIR 的检索评测方式，以及开源 Embedding/Reranker 模型。本仓库会保留依赖许可和出处，并明确标注自行实现的数据、评测与 Agent 模块。
