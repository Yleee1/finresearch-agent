# FinResearch 基线保护快照（2026-10-10）

- Python 3.14.7
- finresearch-agent 0.1.0
- NumPy 2.5.3
- sentence-transformers 5.7.0
- pytest 9.1.1
- V2 改造后全量测试：`43 passed`

V1 保护对象 SHA-256：

```text
8d4da80956d7eaba10cf0e390717f3c50a90e509d7a5bef06d4e5953de0e771e  data/eval/retrieval_frozen_v1.jsonl
12ef6731fd09e34190502e31ba52db95b8c5d7f02a7187b2f4060d2cd4c077da  data/eval/agent_frozen_v1.jsonl
f8de855180448b0753d5205be77052462a3d28c6020de57813f714007caf8e04  results/frozen_eval_v1_summary.json
```

验证命令：

```bash
.venv/bin/pip install -e '.[dev,retrieval,ingest]'
.venv/bin/python -m pytest -q
```
