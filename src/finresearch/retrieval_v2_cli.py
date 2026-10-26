from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from .bm25 import BM25Retriever
from .budgets import ResearchBudget
from .evaluation_v2 import load_retrieval_cases_v2, score_evidence_slots
from .evidence import EvidencePackBuilder
from .io import load_documents
from .metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank
from .reranker import RerankRetriever


def _relevant_ids(case) -> set[str]:
    return {
        doc_id
        for slot in case.evidence_slots
        for group in slot.acceptable_evidence_sets
        for doc_id in group
    }


def _evaluate(name: str, cases, search, *, top_k: int) -> dict:
    rows = []
    started = perf_counter()
    for case in cases:
        ids = search(case)
        relevant = _relevant_ids(case)
        slot_score = score_evidence_slots(case, ids)
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "retrieved_doc_ids": ids,
                "recall": recall_at_k(ids, relevant, len(ids)),
                "mrr": reciprocal_rank(ids, relevant),
                "ndcg": ndcg_at_k(ids, relevant, len(ids)),
                **slot_score,
            }
        )
    seconds = perf_counter() - started
    return {
        "variant": name,
        "case_count": len(rows),
        "top_k": top_k,
        "mean_context_chunks": round(mean([len(row["retrieved_doc_ids"]) for row in rows]), 4),
        "recall": round(mean([row["recall"] for row in rows]), 4),
        "mrr": round(mean([row["mrr"] for row in rows]), 4),
        "ndcg": round(mean([row["ndcg"] for row in rows]), 4),
        "evidence_slot_coverage": round(
            mean([row["evidence_slot_coverage"] for row in rows]), 4
        ),
        "multi_hop_complete_evidence_recall": round(
            mean([row["multi_hop_complete_evidence_recall"] for row in rows]), 4
        ),
        "elapsed_seconds": round(seconds, 4),
        "mean_query_ms": round(seconds * 1000 / len(rows), 3) if rows else 0.0,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinResearch retrieval V2 R0-R2")
    parser.add_argument("--documents", nargs="+", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--max-neighbor-chunks", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reuse-rerank-results",
        action="store_true",
        help="Reuse per-case R1 ids from an existing output file when only expansion logic changed.",
    )
    args = parser.parse_args()

    documents = [doc for path in args.documents for doc in load_documents(path)]
    cases = load_retrieval_cases_v2(args.cases)
    bm25 = BM25Retriever(documents)
    previous = {}
    if args.reuse_rerank_results and Path(args.output).exists():
        previous = json.loads(Path(args.output).read_text(encoding="utf-8"))
    reused_ids = {
        row["case_id"]: row["retrieved_doc_ids"]
        for row in previous.get("R1", {}).get("cases", [])
    }
    reranker = None
    if len(reused_ids) != len(cases):
        reranker = RerankRetriever(
            bm25, candidate_k=args.candidate_k, device=args.device
        )
    pack_builder = EvidencePackBuilder(documents)
    budget = ResearchBudget(
        max_evidence_chunks=args.top_k + args.max_neighbor_chunks,
        max_neighbor_chunks=args.max_neighbor_chunks,
    )
    bm25_cache = {}
    rerank_cache = {}

    def search_bm25(case):
        if case.case_id not in bm25_cache:
            bm25_cache[case.case_id] = bm25.search(case.question, top_k=args.top_k)
        return [item.doc_id for item in bm25_cache[case.case_id]]

    def search_rerank(case):
        if case.case_id in reused_ids:
            return list(reused_ids[case.case_id])
        if case.case_id not in rerank_cache:
            rerank_cache[case.case_id] = reranker.search(case.question, top_k=args.top_k)
        return [item.doc_id for item in rerank_cache[case.case_id]]

    def search_neighbors(case):
        seed_ids = search_rerank(case)
        expanded = list(seed_ids)
        for seed_id in seed_ids:
            source = pack_builder.documents.get(seed_id)
            if source is None:
                continue
            for neighbor_id in (source.prev_chunk_id, source.next_chunk_id):
                if neighbor_id and neighbor_id not in expanded:
                    expanded.append(neighbor_id)
                    if len(expanded) >= budget.max_evidence_chunks:
                        return expanded
        return expanded

    reports = {
        "R0": _evaluate("R0_BM25", cases, search_bm25, top_k=args.top_k),
        "R1": _evaluate("R1_BM25_Reranker", cases, search_rerank, top_k=args.top_k),
        "R2": _evaluate("R2_R1_NeighborExpansion", cases, search_neighbors, top_k=args.top_k),
    }
    if reused_ids and previous.get("R1"):
        reports["R1"] = previous["R1"]
        reports["R2"]["elapsed_seconds"] = round(
            float(previous["R1"]["elapsed_seconds"])
            + float(reports["R2"]["elapsed_seconds"]),
            4,
        )
        reports["R2"]["mean_query_ms"] = round(
            float(previous["R1"]["mean_query_ms"])
            + float(reports["R2"]["mean_query_ms"]),
            3,
        )
        reports["R2"]["latency_note"] = "R1 rerank latency plus cached neighbor expansion"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {name: {key: value for key, value in report.items() if key != "cases"}
             for name, report in reports.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
