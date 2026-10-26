from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

ALLOWED_COMPONENTS = {
    "planner_prompt", "route_rules", "answerability_rules", "tool_guidance",
    "memory_policy", "retrieval_gate_threshold", "critic_parameters", "budgets",
}
PROTECTED_COMPONENTS = {
    "dataset", "gold_evidence", "scorer", "permissions", "security_boundary",
    "document_hash_validation", "download_logic",
}


@dataclass(frozen=True)
class HarnessConfig:
    version: str = "v3-dev-0"
    planner_prompt_hash: str = ""
    retrieval_gate_threshold: float = 0.86
    critic_parameters: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, int] = field(default_factory=lambda: {"retrieval_rounds": 2, "llm_calls": 6})

    def apply_candidate(self, component: str, patch: dict[str, Any], *, next_version: str) -> HarnessConfig:
        if component in PROTECTED_COMPONENTS or component not in ALLOWED_COMPONENTS:
            raise ValueError(f"harness candidate cannot modify protected/unknown component: {component}")
        if component == "retrieval_gate_threshold":
            value = float(patch["value"])
            if not 0 <= value <= 1:
                raise ValueError("gate threshold must be between 0 and 1")
            return replace(self, version=next_version, retrieval_gate_threshold=value)
        if component == "critic_parameters":
            return replace(self, version=next_version, critic_parameters=dict(patch))
        if component == "budgets":
            values = {str(key): int(value) for key, value in patch.items()}
            if any(value <= 0 for value in values.values()):
                raise ValueError("all budgets must be positive")
            return replace(self, version=next_version, budgets=values)
        return replace(self, version=next_version)
