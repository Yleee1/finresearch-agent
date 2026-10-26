from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchBudget:
    """Hard limits for a single research run.

    Token counts are deterministic estimates (roughly four Latin characters or
    two CJK characters per token), so the same run is reproducible without a
    tokenizer dependency.
    """

    max_retrieval_rounds: int = 2
    max_sub_queries: int = 4
    max_evidence_chunks: int = 12
    max_context_tokens: int = 8_000
    max_llm_calls: int = 8
    max_neighbor_chunks: int = 2
    max_answer_repairs: int = 1

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value < 0 or (name != "max_answer_repairs" and value == 0):
                raise ValueError(f"{name} must be positive")


def estimate_tokens(text: str) -> int:
    cjk = sum("\u3400" <= char <= "\u9fff" for char in text)
    other = max(0, len(text) - cjk)
    return max(1, (cjk + 1) // 2 + (other + 3) // 4)
