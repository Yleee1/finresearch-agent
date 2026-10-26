from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .evaluation import evaluate_retriever
from .hybrid import HybridRetriever
from .io import load_documents, load_queries
from .reranker import RerankRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Cross-Encoder reranking")
    parser.add_argument("--documents", nargs="+", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--base", choices=("bm25", "hybrid"), default="bm25")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output")
    return parser


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

    started = perf_counter()
    bm25 = BM25Retriever(documents)
    if args.base == "hybrid":
        dense = DenseRetriever(
            documents,
            model_name=args.embedding_model,
            device=args.device,
            batch_size=args.batch_size,
        )
        base = HybridRetriever(bm25, dense, candidate_k=args.candidate_k)
    else:
        base = bm25
    retriever = RerankRetriever(
        base,
        model_name=args.reranker_model,
        device=args.device,
        candidate_k=args.candidate_k,
        batch_size=args.batch_size,
    )
    build_seconds = perf_counter() - started

    started = perf_counter()
    report = evaluate_retriever(retriever, queries, top_k=args.top_k)
    query_seconds = perf_counter() - started
    payload = {
        "corpus_size": len(documents),
        "query_count": len(queries),
        "base": args.base,
        "candidate_k": args.candidate_k,
        "embedding_model": args.embedding_model if args.base == "hybrid" else None,
        "reranker_model": args.reranker_model,
        "device": args.device,
        "result": {
            **asdict(report),
            "build_seconds": round(build_seconds, 4),
            "query_seconds": round(query_seconds, 4),
            "mean_query_ms": round(query_seconds * 1000 / len(queries), 2),
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
