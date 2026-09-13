from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .io import load_documents, load_queries
from .metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from .reranker import RerankRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write case-level retrieval diagnostics")
    parser.add_argument("--documents", nargs="+", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--retriever",
        choices=("bm25", "dense", "hybrid", "bm25-rerank", "hybrid-rerank"),
        default="bm25-rerank",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    documents = [
        document
        for path in args.documents
        for document in load_documents(path)
    ]
    cases = load_queries(args.queries)
    bm25 = BM25Retriever(documents)

    if args.retriever == "bm25":
        retriever = bm25
    elif args.retriever == "dense":
        retriever = DenseRetriever(documents, device=args.device)
    elif args.retriever == "bm25-rerank":
        retriever = RerankRetriever(
            bm25,
            device=args.device,
            candidate_k=args.candidate_k,
        )
    else:
        dense = DenseRetriever(documents, device=args.device)
        hybrid = HybridRetriever(bm25, dense, candidate_k=args.candidate_k)
        if args.retriever == "hybrid":
            retriever = hybrid
        else:
            retriever = RerankRetriever(
                hybrid,
                device=args.device,
                candidate_k=args.candidate_k,
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            results = retriever.search(case.query, top_k=args.top_k)
            ranked_ids = [result.doc_id for result in results]
            relevant_ids = set(case.relevant_doc_ids)
            record = {
                "query_id": case.query_id,
                "query": case.query,
                "relevant_doc_ids": list(case.relevant_doc_ids),
                "ranked_doc_ids": ranked_ids,
                "hit_at_k": hit_at_k(ranked_ids, relevant_ids, args.top_k),
                "recall_at_k": recall_at_k(ranked_ids, relevant_ids, args.top_k),
                "reciprocal_rank": reciprocal_rank(ranked_ids, relevant_ids),
                "ndcg_at_k": ndcg_at_k(ranked_ids, relevant_ids, args.top_k),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
