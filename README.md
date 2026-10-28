# FinResearch Agent

面向金融研报和公司公告的反思式 Agentic RAG。项目以可复现评测为主线，从稀疏检索、Dense Retrieval 和 Cross-Encoder 扩展到证据槽位、缺失证据驱动的二次检索、claim ledger、引用审计和预算内拒答。

## 当前阶段

V3 核心 Harness 已完成本地实现：结构化 `ResearchPlan`、增量检索 Gate、程序性 SQLite Memory、脱敏 JSONL/Langfuse v4 Sink 和受保护的 Harness Ledger。全量测试 65 项通过，11 条确定性升级验收全部通过。F5 现使用固定 commit 的官方 Evo-Harness `AgentWorkspace`/`Observer`，加自定义 Context-to-Harness 编译器与 dev→held-out→frozen 门禁真实运行；在同一 Qwen3-1.7B 本地求解器条件下，12 条 frozen 的 Route Accuracy 从 F4 的 8.33% 提升到 75.00%，Strict Success 从 0 提升到 8.33%。回答生成仍是主要瓶颈，且该 8/8/12 作者构造小集不能替代 V2 主指标；F0–F4 的 DeepSeek 数字也不能与该本地 Qwen 对照混算。详见 [`docs/UPGRADE_IMPLEMENTATION_V3.md`](docs/UPGRADE_IMPLEMENTATION_V3.md)。

V2 代码链路已经实现：

- `single_fact`、同段多槽位、跨章节多跳、跨公司比较与越界路由；
- 结构化 Evidence Pack、来源查询/轮次追踪、重复披露分组和有界邻块扩展；
- 显式 evidence slot coverage，第二轮只面向缺失槽位，无新增证据立即停止；
- 原子 claim ledger，以及公司、年份、指标、数值、单位和逐字引用审计；
- 检索轮数、子查询、证据块、上下文 Token、LLM 调用和答案修复硬预算；
- 支持 `acceptable_evidence_sets` 的 V2 评分 schema。

完整设计见 [`docs/AGENTIC_RAG_V2.md`](docs/AGENTIC_RAG_V2.md)。V2 已完成 128 条 Retrieval 和 74 条 Agent 冻结集、第二审阅者独立复核，以及 R0—R5 真实消融。复核完成状态由项目所有者确认，仓库不保存复核者身份。状态与真实数字见 [`docs/FROZEN_EVAL_V2.md`](docs/FROZEN_EVAL_V2.md)。V1 冻结结果保持不变。

V2 检索消融（Top-5）：

| 系统 | Recall | MRR | NDCG | Slot Coverage | Complete Evidence | 平均延迟 |
|---|---:|---:|---:|---:|---:|---:|
| R0 BM25 | 79.95% | 65.56% | 68.20% | 78.12% | 75.00% | 4.01 ms |
| R1 + Cross-Encoder | 86.46% | 72.15% | 74.04% | 85.55% | 82.81% | 1.87 s |
| R2 + 邻块/表头 | 89.97% | 72.50% | 75.39% | 89.84% | 88.28% | 1.87 s |

在同一冻结 Agent 集上，R3/R4/R5 的 Strict Success 分别为 25.68%/24.32%/18.92%。R4 的二轮增量召回仅 1.35%，无效二轮率 72.97%；R5 将拒答 Recall 提到 100%，但出现明显过度拒答。这些负结果被完整保留，不用 V1 或开发集数字替换。

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
./.venv/bin/python -m pytest -q
```

生成结构感知候选语料。该模式将正文按语义分块，将可解析表格按完整行组切分，
在每个表格子块中重复表头，并写入 `table_id`、页码和行范围。输出使用新的
chunk ID，不能直接套用 V3 冻结集的 gold ID：

```bash
./.venv/bin/finresearch-ingest \
  --pdf data/raw/byd_2024_annual_report.pdf \
  --output data/processed/byd_2024_chunks_v4.jsonl \
  --document-id byd-2024-annual --title 比亚迪2024年报 \
  --company 比亚迪 --year 2024 \
  --strategy semantic --clean-margins --structure-aware-tables
```

生成候选 gold ID 迁移文件并验证每个新 ID 确实存在：

```bash
PYTHONPATH=src ./.venv/bin/python -m finresearch.v4_data \
  --documents data/processed/byd_2024_chunks_v4.jsonl \
              data/processed/catl_2024_chunks_v4.jsonl \
  --output-dir data/eval/v4_candidate
```

V3/V4 配对检索实验的配置、真实指标、回归样本和结论见
[`docs/V4_TABLE_CHUNKING_EVAL.md`](docs/V4_TABLE_CHUNKING_EVAL.md)。V4 当前是
`candidate_pending_independent_review`，未覆盖 V3 冻结主线。

V3 核心升级验收（无模型、无网络）：

```bash
PYTHONPATH=src ./.venv/bin/python -m finresearch.upgrade_eval_cli \
  --output results/v3/upgrade_component_eval.json
```

V3 F0–F4 使用同一数据、模型、检索器和预算，通过 `agent_eval_cli --ablation F0...F4` 运行；数据清单与原始结果分别位于 `data/eval/v3/manifest.json` 和 `results/v3/`。Frozen 只允许按报告中的选定配置运行一次，不应把 quick-start 当作重复调参入口。

运行 F5 离线自演化。编译器和求解器均使用本地 MLX Qwen，候选只允许修改 Skill、两个运行开关和白名单路由规则；只有 dev 晋级候选才读取 held-out，frozen 不参与选择：

```bash
./.venv/bin/pip install -e '.[evo]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python \
  -m finresearch.evo_f5 \
  --documents data/processed/byd_2024_chunks_v3.jsonl \
              data/processed/catl_2024_chunks_v3.jsonl \
  --run-tag v3 --max-cycles 4
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

V2 Retrieval 同集消融：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  -m finresearch.retrieval_v2_cli \
  --documents data/processed/byd_2024_chunks_v3.jsonl \
              data/processed/catl_2024_chunks_v3.jsonl \
  --cases data/eval/frozen_v2/retrieval_frozen_v2.jsonl \
  --output results/retrieval_ablation_frozen_v2.json
```

Agent 实现包含问题拆解、最多两轮检索、Evidence Verifier、低证据拒答和逐字引用校验。真实 DeepSeek 冒烟运行方式见 `docs/day04.md`；`.env` 不得提交。

V2 Agent 可显式设置全部硬预算：

```bash
PYTHONPATH=src .venv/bin/python -m finresearch.agent_cli \
  "比较比亚迪和宁德时代2024年营业收入" \
  --documents data/processed/byd_2024_chunks_v3.jsonl \
              data/processed/catl_2024_chunks_v3.jsonl \
  --retriever bm25-rerank --max-iterations 2 \
  --max-sub-queries 4 --max-evidence-chunks 12 \
  --max-context-tokens 8000 --max-llm-calls 8 \
  --max-neighbor-chunks 2
```

真实 Agent smoke 已验证：证据完整问题可返回带页码引用；口径不一致问题在两轮检索后拒答；`single/multi` 路由将简单问题延迟从约 19.07 秒降至 9.17 秒。详见 `results/agent_smoke_report.md`。

12 条 Agent seed 开发集的同版本 V2 重跑得到 100% Route/Status/Fact/Citation/Success，平均端到端 7.61 秒；由于集合很小且参与过 prompt 调整，该数字明确标记为开发指标，不能直接写成最终泛化性能。

## Web 演示

无需额外前端依赖，默认使用 4 条完全合成的 demo 文档：

```bash
./.venv/bin/python -m finresearch.web_demo --port 8766
```

打开 `http://127.0.0.1:8766`。页面展示 route、sub-query、evidence slot、页码、来源查询、延迟、Token 估算和不可信文档指令告警。传入本地 `data/processed/*chunks_v3.jsonl` 可切换到真实年报语料，数据不会离开本机。

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
