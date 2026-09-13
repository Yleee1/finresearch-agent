from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path


@dataclass(frozen=True)
class AgentEvalCase:
    case_id: str
    question: str
    expected_route: str
    expected_status: str
    expected_facts: tuple[str, ...]
    relevant_doc_ids: tuple[str, ...]


def load_agent_cases(path: str | Path) -> list[AgentEvalCase]:
    cases = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                cases.append(
                    AgentEvalCase(
                        case_id=value["case_id"],
                        question=value["question"],
                        expected_route=value["expected_route"],
                        expected_status=value["expected_status"],
                        expected_facts=tuple(value["expected_facts"]),
                        relevant_doc_ids=tuple(value["relevant_doc_ids"]),
                    )
                )
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid agent case at {path}:{line_number}") from exc
    return cases


def _normalize(text: str) -> str:
    return re.sub(r"[\s,，]", "", text).lower()


def score_agent_result(case: AgentEvalCase, result: dict) -> dict:
    answer = _normalize(str(result.get("answer", "")))
    fact_matches = [
        _normalize(fact) in answer
        for fact in case.expected_facts
    ]
    fact_coverage = (
        sum(fact_matches) / len(fact_matches)
        if fact_matches
        else 1.0
    )
    cited_ids = {
        item.get("doc_id")
        for item in result.get("citations", [])
        if isinstance(item, dict)
    }
    citation_hit = (
        bool(cited_ids & set(case.relevant_doc_ids))
        if case.expected_status == "answered"
        else not cited_ids
    )
    route_correct = result.get("route") == case.expected_route
    status_correct = result.get("status") == case.expected_status
    return {
        "case_id": case.case_id,
        "route_correct": route_correct,
        "status_correct": status_correct,
        "fact_coverage": fact_coverage,
        "citation_hit": citation_hit,
        "success": route_correct and status_correct and fact_coverage == 1.0 and citation_hit,
    }


def aggregate_agent_scores(scores: list[dict]) -> dict:
    if not scores:
        return {"case_count": 0}
    count = len(scores)
    mean = lambda field: sum(float(item[field]) for item in scores) / count
    return {
        "case_count": count,
        "route_accuracy": mean("route_correct"),
        "status_accuracy": mean("status_correct"),
        "mean_fact_coverage": mean("fact_coverage"),
        "citation_hit_rate": mean("citation_hit"),
        "success_rate": mean("success"),
    }
