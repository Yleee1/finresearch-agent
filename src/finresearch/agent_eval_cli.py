from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
from time import perf_counter

from .agent import ResearchAgent
from .agent_evaluation import (
    aggregate_agent_scores,
    load_agent_cases,
    score_agent_result,
)
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .io import load_documents
from .llm_client import DeepSeekResearchLLM, OpenAICompatibleClient
from .reranker import RerankRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated FinResearch Agent evaluation")
    parser.add_argument("--documents", nargs="+", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retriever", choices=("bm25-rerank", "hybrid-rerank"), default="bm25-rerank")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--env-file", default=".env")
    return parser


def _load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    deduplicated = {record["case_id"]: record for record in records}
    return list(deduplicated.values())


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = Path(f"{output_path}.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another evaluation is writing {output_path}") from exc
    existing = _load_existing(output_path)
    if existing and not args.resume:
        raise FileExistsError(f"output already exists: {output_path}; use --resume")
    completed_ids = {item["case_id"] for item in existing}

    documents = [
        document
        for path in args.documents
        for document in load_documents(path)
    ]
    bm25 = BM25Retriever(documents)
    if args.retriever == "hybrid-rerank":
        dense = DenseRetriever(documents, device=args.device)
        base = HybridRetriever(bm25, dense, candidate_k=args.candidate_k)
    else:
        base = bm25
    retriever = RerankRetriever(
        base,
        device=args.device,
        candidate_k=args.candidate_k,
    )
    client = OpenAICompatibleClient.from_env(args.env_file)
    agent = ResearchAgent(
        retriever,
        DeepSeekResearchLLM(client),
        top_k=args.top_k,
        max_iterations=args.max_iterations,
    )

    cases = [case for case in load_agent_cases(args.cases) if case.case_id not in completed_ids]
    if args.limit is not None:
        cases = cases[: args.limit]
    mode = "a" if existing else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for case in cases:
            before = dict(client.usage)
            started = perf_counter()
            try:
                result = agent.run(case.question)
                compact = {
                    "case_id": case.case_id,
                    "question": case.question,
                    "route": result.route,
                    "status": result.status,
                    "answer": result.answer,
                    "citations": [asdict(item) for item in result.citations],
                    "iterations": result.iterations,
                    "traces": [asdict(item) for item in result.traces],
                    "refusal_reason": result.refusal_reason,
                }
            except Exception as exc:
                compact = {
                    "case_id": case.case_id,
                    "question": case.question,
                    "route": "",
                    "status": "error",
                    "answer": "",
                    "citations": [],
                    "iterations": 0,
                    "traces": [],
                    "refusal_reason": "",
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
            seconds = perf_counter() - started
            score = score_agent_result(case, compact)
            usage = {
                field: client.usage[field] - before[field]
                for field in client.usage
            }
            record = {
                **compact,
                "score": score,
                "usage": usage,
                "end_to_end_seconds": round(seconds, 4),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"case_id": case.case_id, **score}, ensure_ascii=False))

    all_records = _load_existing(output_path)
    case_by_id = {case.case_id: case for case in load_agent_cases(args.cases)}
    scores = [
        score_agent_result(case_by_id[record["case_id"]], record)
        for record in all_records
    ]
    summary = aggregate_agent_scores(scores)
    summary.update(
        {
            "retriever": args.retriever,
            "total_llm_calls": sum(record["usage"]["calls"] for record in all_records),
            "total_tokens": sum(record["usage"]["total_tokens"] for record in all_records),
            "mean_end_to_end_seconds": sum(record["end_to_end_seconds"] for record in all_records)
            / len(all_records),
        }
    )
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
