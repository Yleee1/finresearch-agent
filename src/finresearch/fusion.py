from __future__ import annotations

from collections import defaultdict

from .schemas import SearchResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    *,
    top_k: int = 5,
    rank_constant: int = 60,
) -> list[SearchResult]:
    """Fuse rankings without assuming their raw scores are calibrated."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if rank_constant < 0:
        raise ValueError("rank_constant must be non-negative")

    fused_scores: dict[str, float] = defaultdict(float)
    representatives: dict[str, SearchResult] = {}
    for ranked_list in ranked_lists:
        for result in ranked_list:
            fused_scores[result.doc_id] += 1.0 / (rank_constant + result.rank)
            representatives.setdefault(result.doc_id, result)

    ordered_ids = sorted(fused_scores, key=lambda doc_id: (-fused_scores[doc_id], doc_id))
    return [
        SearchResult(
            doc_id=doc_id,
            score=fused_scores[doc_id],
            rank=rank,
            title=representatives[doc_id].title,
            text=representatives[doc_id].text,
            source=representatives[doc_id].source,
            document_id=representatives[doc_id].document_id,
            company=representatives[doc_id].company,
            report_year=representatives[doc_id].report_year,
            page_start=representatives[doc_id].page_start,
            page_end=representatives[doc_id].page_end,
            section_path=representatives[doc_id].section_path,
            prev_chunk_id=representatives[doc_id].prev_chunk_id,
            next_chunk_id=representatives[doc_id].next_chunk_id,
        )
        for rank, doc_id in enumerate(ordered_ids[:top_k], start=1)
    ]
