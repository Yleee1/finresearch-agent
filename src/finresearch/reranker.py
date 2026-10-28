from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .schemas import SearchResult


class PairScorer(Protocol):
    def predict(self, sentences, **kwargs): ...


class RerankRetriever:
    """Retrieve a broad candidate set, then score query-passage pairs."""

    def __init__(
        self,
        base_retriever,
        *,
        model_name: str = "BAAI/bge-reranker-base",
        scorer: PairScorer | None = None,
        device: str | None = None,
        candidate_k: int = 20,
        batch_size: int = 16,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if scorer is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "Reranking requires: pip install -e '.[retrieval]'"
                ) from exc
            scorer = CrossEncoder(model_name, device=device)

        self.base_retriever = base_retriever
        self.scorer = scorer
        self.candidate_k = candidate_k
        self.batch_size = batch_size

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        candidates = self.base_retriever.search(query, top_k=self.candidate_k)
        if not candidates:
            return []

        pairs: Sequence[tuple[str, str]] = [
            (query, f"{candidate.title}\n{candidate.text}")
            for candidate in candidates
        ]
        scores = np.asarray(
            self.scorer.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        ).reshape(-1)
        if scores.shape[0] != len(candidates):
            raise ValueError("scorer returned an invalid score shape")

        ordered = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].doc_id),
        )[:top_k]
        return [
            SearchResult(
                doc_id=candidate.doc_id,
                score=float(score),
                rank=rank,
                title=candidate.title,
                text=candidate.text,
                source=candidate.source,
                document_id=candidate.document_id,
                company=candidate.company,
                report_year=candidate.report_year,
                page_start=candidate.page_start,
                page_end=candidate.page_end,
                section_path=candidate.section_path,
                prev_chunk_id=candidate.prev_chunk_id,
                next_chunk_id=candidate.next_chunk_id,
                chunk_type=candidate.chunk_type,
                table_id=candidate.table_id,
                table_header=candidate.table_header,
                row_start=candidate.row_start,
                row_end=candidate.row_end,
            )
            for rank, (candidate, score) in enumerate(ordered, start=1)
        ]
