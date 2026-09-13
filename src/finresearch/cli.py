from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .evaluation import evaluate_retriever
from .hybrid import HybridRetriever
from .io import load_documents, load_queries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark FinResearch retrieval")
    parser.add_argument(
        "--documents",
        required=True,
        nargs="+",
        help="One or more document JSONL paths",
    )
    parser.add_argument("--queries", required=True, help="Evaluation query JSONL path")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--retrievers",
        nargs="+",
        choices=("bm25", "dense", "hybrid"),
        default=("bm25",),
    )
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--output", help="Optional JSON report path")
    return parser


def _benchmark(retriever, queries, top_k: int, build_seconds: float) -> dict:
    started = perf_counter()
    report = evaluate_retriever(retriever, queries, top_k=top_k)
    query_seconds = perf_counter() - started
    result = asdict(report)
    result.update(
        {
            "build_seconds": round(build_seconds, 4),
            "query_seconds": round(query_seconds, 4),
            "mean_query_ms": round(query_seconds * 1000 / len(queries), 2),
        }
    )
    return result


def main() -> None:
    args = build_parser().parse_args()
    documents = [
        document
        for path in args.documents
        for document in load_documents(path)
    ]
    queries = load_queries(args.queries)
    if not queries:
        raise ValueError("queries must not be empty")

    requested = set(args.retrievers)
    retrievers: dict[str, object] = {}
    build_times: dict[str, float] = {}

    if requested & {"bm25", "hybrid"}:
        started = perf_counter()
        retrievers["bm25"] = BM25Retriever(documents)
        build_times["bm25"] = perf_counter() - started

    if requested & {"dense", "hybrid"}:
        started = perf_counter()
        retrievers["dense"] = DenseRetriever(
            documents,
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
        )
        build_times["dense"] = perf_counter() - started

    if "hybrid" in requested:
        retrievers["hybrid"] = HybridRetriever(
            retrievers["bm25"],
            retrievers["dense"],
            candidate_k=args.candidate_k,
        )
        build_times["hybrid"] = build_times["bm25"] + build_times["dense"]

    payload = {
        "corpus_size": len(documents),
        "query_count": len(queries),
        "model": args.model if requested & {"dense", "hybrid"} else None,
        "device": args.device if requested & {"dense", "hybrid"} else None,
        "results": {
            name: _benchmark(retrievers[name], queries, args.top_k, build_times[name])
            for name in args.retrievers
        },
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
