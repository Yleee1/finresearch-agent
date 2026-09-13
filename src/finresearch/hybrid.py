from __future__ import annotations

from .fusion import reciprocal_rank_fusion
from .schemas import SearchResult


class HybridRetriever:
    def __init__(self, sparse_retriever, dense_retriever, *, candidate_k: int = 20) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.candidate_k = candidate_k

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        sparse = self.sparse_retriever.search(query, top_k=self.candidate_k)
        dense = self.dense_retriever.search(query, top_k=self.candidate_k)
        return reciprocal_rank_fusion([sparse, dense], top_k=top_k)
