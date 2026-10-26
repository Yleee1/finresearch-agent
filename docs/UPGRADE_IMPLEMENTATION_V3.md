# FinResearch Eval-Driven Harness 实现报告

## 已落地

- `ResearchPlan` 结构化任务合同，显式包含 answerability、route、entity、period、evidence slot 和 dependency；非法 Schema fail-closed。
- `RetrievalValueGate` 在第二轮前检查缺失槽位、近似重复 query、候选集饱和与剩余轮次，并将 stop reason 写入 `AgentTrace`。
- 本地 SQLite 程序性记忆，仅允许 `procedural`/`negative_experience`；两条独立 trace 才激活，过期/争议记忆不返回，禁止冻结答案、评分器细节、未绑定财务数值和指令性文本。
- 本地 JSONL `TraceEvent` Sink，记录 planner、retrieval、gate 和 result；摘要强制限长，只传 doc ID/槽位/分数等元数据。
- Harness 配置和 JSONL Ledger；数据、gold、scorer、权限与文档哈希校验被列为不可修改组件。
- Demo 展示 harness version 和 Retry Gate 的继续/停止原因。

## 真实验证

```bash
./.venv/bin/ruff check src tests
./.venv/bin/pytest -q
PYTHONPATH=src ./.venv/bin/python -m finresearch.upgrade_eval_cli
```

- 全量自动化测试：65 passed。
- V3 确定性组件验收：11/11 passed，包含 4 条 Planner、3 条 Gate、2 条 Memory、1 条 Harness 保护和 1 条 Trace 脱敏。
- 原始记录：`results/v3/upgrade_component_eval.json`。

这个 11/11 只是确定性组件验收，不是端到端 Strict Task Success，不能与 V1/V2 冻结指标混写。

## V3 隔离评测与负结果

- 新建 dev/held-out/frozen 8/8/12 条 Agent 集；V3 gold doc ID 与 V2 frozen 重合为 0，SHA-256 见 `data/eval/v3/manifest.json`。该集合为作者构造，尚未获得第二审阅者签字。
- Dev Strict Success：F0/F1/F2/F3/F4 均为 37.50%。F1 将 Route Accuracy 从 62.50% 提到 87.50%，但没有转化为端到端收益。
- Held-out Strict Success：F0/F1 为 12.50%，F2/F3/F4 为 0；F2 无效二轮率 87.50%，F3/F4 为 75.00%。程序性记忆虽命中，但当前只记录 provenance，未影响计划/检索，因此 F3 没有净收益。
- 按“质量相同取低成本”选择 F0，只执行一次 frozen：Strict Success 16.67%、Route Accuracy 66.67%、Abstention Recall 100%。10 个可回答问题全部失败，根因见 `results/v3/bad_cases.json`。
- `OptionalLangfuseSink` 已按 Langfuse Python SDK v4 的 observation context API 完成真实依赖契约测试；本机未配置 Langfuse 服务端凭据，因此不声称远端 trace 上传验收。
- F5 已真实执行：官方 Evo-Harness 固定在 commit `3c7c7b8c62a3f07da1d4407aa03f0ca079954bcf`，使用其 `AgentWorkspace`/`Observer` 数据合同，并实现 FinResearch 专用 Context-to-Harness pipeline。编译器与求解器均为本地 MLX Qwen3-1.7B-4bit，不向外部 API 发送评测数据。
- 候选只能修改 Skill、`allow_reflection`/`use_answer_critic` 和白名单路由规则；gold、scorer、文档、权限与 frozen 被保护。第 1 个候选在 dev 将 Strict Success 12.50%→25.00%、无效二轮率 50.00%→12.50%，晋级后在 held-out 将 Strict Success 0→25.00%，且 Abstention Recall 保持 100%，因此被接受。
- 接受后才做 frozen 最终比较：同一 Qwen 求解器下，F4→F5 的 Route Accuracy 8.33%→75.00%、Strict Success 0→8.33%、无效二轮率 16.67%→8.33%、总 Token 128,966→60,128。F5 对 10 个可回答问题仍全部失败，说明演化修复了路由与部分成本，没有修复答案生成。
- 早期编译器候选未通过 dev Gate 的负结果保留在 `results/v3/F5_evolution.json`；最终审计记录见 `results/v3/F5_evolution_v3.json`。Frozen 未用于候选选择。

## 结果边界

- 这是基于官方 workspace/observer 合同实现的项目专用 pipeline，不是 Evo-Harness 上游发布的五条 benchmark pipeline，也不是模型权重自训练或 Agentic RL。
- F5 与 F4 的有效对照仅限同一 local-Qwen 条件；不能把此前 DeepSeek F0–F4 数字与 F5 直接比较。
- V3 frozen 样本量只有 12 且未做第二审阅；F5 的 8.33% Strict Success 不替代 V2 简历主指标。
