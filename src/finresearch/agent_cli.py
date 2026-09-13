from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from .agent import ResearchAgent
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .io import load_documents
from .llm_client import DeepSeekResearchLLM, OpenAICompatibleClient
from .reranker import RerankRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask the bounded FinResearch Agent")
    parser.add_argument("question")
    parser.add_argument("--documents", nargs="+", required=True)
    parser.add_argument(
        "--retriever",
        choices=("bm25", "bm25-rerank", "hybrid-rerank"),
        default="bm25-rerank",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    documents = [
        document
        for path in args.documents
        for document in load_documents(path)
    ]
    bm25 = BM25Retriever(documents)
    if args.retriever == "bm25":
        retriever = bm25
    elif args.retriever == "bm25-rerank":
        retriever = RerankRetriever(
            bm25,
            device=args.device,
            candidate_k=args.candidate_k,
        )
    else:
        dense = DenseRetriever(documents, device=args.device)
        hybrid = HybridRetriever(bm25, dense, candidate_k=args.candidate_k)
        retriever = RerankRetriever(
            hybrid,
            device=args.device,
            candidate_k=args.candidate_k,
        )

    client = OpenAICompatibleClient.from_env(args.env_file)
    llm = DeepSeekResearchLLM(client)
    started = perf_counter()
    result = ResearchAgent(
        retriever,
        llm,
        top_k=args.top_k,
        max_iterations=args.max_iterations,
    ).run(args.question)
    payload = asdict(result)
    payload["llm_usage"] = {
        **client.usage,
        "api_seconds": round(float(client.usage["api_seconds"]), 4),
    }
    payload["end_to_end_seconds"] = round(perf_counter() - started, 4)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
