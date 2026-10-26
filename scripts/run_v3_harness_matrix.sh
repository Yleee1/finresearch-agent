#!/usr/bin/env bash
set -euo pipefail

split="${1:-dev}"
if [[ "$split" != "dev" && "$split" != "heldout" ]]; then
  echo "usage: $0 [dev|heldout]" >&2
  echo "frozen is intentionally excluded from repeatable matrix runs" >&2
  exit 2
fi

for group in F0 F1 F2 F3 F4; do
  ./.venv/bin/python -m finresearch.agent_eval_cli \
    --documents data/processed/byd_2024_chunks_v3.jsonl data/processed/catl_2024_chunks_v3.jsonl \
    --cases "data/eval/v3/${split}_v3.jsonl" \
    --output "results/v3/${group}_${split}.jsonl" \
    --summary "results/v3/${group}_${split}_summary.json" \
    --ablation "$group" --retriever bm25-rerank --top-k 5 --candidate-k 20 \
    --max-iterations 2 --device cpu --dataset-split "${split}_v3" \
    --harness-version "${group}-v3"
done
