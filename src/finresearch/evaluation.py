from __future__ import annotations

from dataclasses import dataclass

from .metrics import hit_at_k, mean, ndcg_at_k, recall_at_k, reciprocal_rank
from .schemas import QueryCase, SearchResult


@dataclass(frozen=True)
class EvaluationReport:
    query_count: int
    hit_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    top_k: int


def evaluate_retriever(retriever, cases: list[QueryCase], top_k: int) -> EvaluationReport:
    hits: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for case in cases:
        results: list[SearchResult] = retriever.search(case.query, top_k=top_k)
        ranked_ids = [result.doc_id for result in results]
        relevant_ids = set(case.relevant_doc_ids)
        hits.append(hit_at_k(ranked_ids, relevant_ids, top_k))
        recalls.append(recall_at_k(ranked_ids, relevant_ids, top_k))
        reciprocal_ranks.append(reciprocal_rank(ranked_ids, relevant_ids))
        ndcgs.append(ndcg_at_k(ranked_ids, relevant_ids, top_k))
    return EvaluationReport(
        query_count=len(cases),
        hit_at_k=mean(hits),
        recall_at_k=mean(recalls),
        mrr=mean(reciprocal_ranks),
        ndcg_at_k=mean(ndcgs),
        top_k=top_k,
    )
