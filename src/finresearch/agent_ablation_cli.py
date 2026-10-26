from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FinResearch R0-R5 agent matrix")
    parser.add_argument("--documents", nargs="+", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output-dir", default="results/agent_ablation_v2")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    retrievers = {
        "R0": "bm25",
        "R1": "bm25-rerank",
        "R2": "bm25-rerank",
        "R3": "bm25-rerank",
        "R4": "bm25-rerank",
        "R5": "bm25-rerank",
    }
    environment = dict(os.environ)
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    summaries = {}
    for variant, retriever in retrievers.items():
        output = output_dir / f"{variant.lower()}_records.jsonl"
        summary = output_dir / f"{variant.lower()}_summary.json"
        command = [
            sys.executable,
            "-m",
            "finresearch.agent_eval_cli",
            "--documents",
            *args.documents,
            "--cases",
            args.cases,
            "--output",
            str(output),
            "--summary",
            str(summary),
            "--retriever",
            retriever,
            "--ablation",
            variant,
            "--device",
            args.device,
            "--env-file",
            args.env_file,
        ]
        if args.limit is not None:
            command.extend(("--limit", str(args.limit)))
        if args.resume:
            command.append("--resume")
        subprocess.run(command, check=True, env=environment)
        summaries[variant] = json.loads(summary.read_text(encoding="utf-8"))
        (output_dir / "matrix_summary.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
