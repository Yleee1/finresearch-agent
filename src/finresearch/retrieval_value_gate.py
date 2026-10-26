from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower()))


@dataclass(frozen=True)
class RetrievalGateDecision:
    should_retry: bool
    expected_missing_slots: tuple[str, ...] = ()
    query_delta: str = ""
    follow_up_queries: tuple[str, ...] = ()
    stop_reason: str | None = None


class RetrievalValueGate:
    """Deterministic guard against repeated, slot-free second retrieval rounds."""

    def __init__(self, *, duplicate_threshold: float = 0.86) -> None:
        if not 0 <= duplicate_threshold <= 1:
            raise ValueError("duplicate_threshold must be between 0 and 1")
        self.duplicate_threshold = duplicate_threshold

    def decide(
        self,
        *,
        missing_slots: tuple[str, ...],
        proposed_queries: tuple[str, ...],
        used_queries: tuple[str, ...],
        remaining_rounds: int,
        previous_doc_ids: tuple[str, ...] = (),
        current_doc_ids: tuple[str, ...] = (),
    ) -> RetrievalGateDecision:
        if remaining_rounds <= 0:
            return RetrievalGateDecision(False, stop_reason="retrieval_budget_exhausted")
        if not missing_slots:
            return RetrievalGateDecision(False, stop_reason="no_missing_slots")
        candidates = tuple(query.strip() for query in proposed_queries if query.strip())
        if not candidates:
            return RetrievalGateDecision(False, missing_slots, stop_reason="no_slot_bound_query")
        if previous_doc_ids and set(previous_doc_ids) == set(current_doc_ids):
            return RetrievalGateDecision(False, missing_slots, stop_reason="candidate_set_saturated")
        accepted = []
        for query in candidates:
            normalized = _normalize(query)
            if not normalized:
                continue
            duplicate = any(
                SequenceMatcher(None, normalized, _normalize(old)).ratio()
                >= self.duplicate_threshold
                for old in used_queries
            )
            if not duplicate:
                accepted.append(query)
        if not accepted:
            return RetrievalGateDecision(False, missing_slots, stop_reason="near_duplicate_query")
        return RetrievalGateDecision(
            True,
            missing_slots,
            query_delta="missing-slot constrained follow-up",
            follow_up_queries=tuple(dict.fromkeys(accepted)),
        )
