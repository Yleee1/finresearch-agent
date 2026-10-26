from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentEvalCase:
    case_id: str
    question: str
    expected_route: str
    expected_status: str
    expected_facts: tuple[str, ...]
    relevant_doc_ids: tuple[str, ...]
    category: str = ""


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
                        category=str(value.get("category", "")),
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
    relevant_ids = set(case.relevant_doc_ids)
    correct_citations = cited_ids & relevant_ids
    citation_precision = (
        len(correct_citations) / len(cited_ids)
        if cited_ids
        else float(not relevant_ids)
    )
    citation_recall = (
        len(correct_citations) / len(relevant_ids)
        if relevant_ids
        else float(not cited_ids)
    )
    claims = result.get("claim_ledger", [])
    claim_support_rate = (
        sum(bool(item.get("supported")) for item in claims) / len(claims)
        if claims
        else None
    )
    traces = result.get("traces", [])
    first_ids = set(traces[0].get("retrieved_doc_ids", [])) if traces else set()
    final_ids = set(traces[-1].get("retrieved_doc_ids", [])) if traces else set()
    first_recall = len(first_ids & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0
    final_recall = len(final_ids & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0
    second_round_incremental_recall = max(0.0, final_recall - first_recall)
    invalid_second_round = float(len(traces) > 1 and second_round_incremental_recall == 0)
    return {
        "case_id": case.case_id,
        "expected_route": case.expected_route,
        "predicted_route": str(result.get("route", "")),
        "route_correct": route_correct,
        "status_correct": status_correct,
        "fact_coverage": fact_coverage,
        "citation_hit": citation_hit,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "claim_support_rate": claim_support_rate,
        "expected_abstain": case.expected_status == "abstained",
        "predicted_abstain": result.get("status") == "abstained",
        "second_round_incremental_recall": second_round_incremental_recall,
        "invalid_second_round": invalid_second_round,
        "success": route_correct and status_correct and fact_coverage == 1.0 and citation_hit,
    }


def aggregate_agent_scores(scores: list[dict]) -> dict:
    if not scores:
        return {"case_count": 0}
    count = len(scores)
    mean = lambda field: sum(float(item[field]) for item in scores) / count
    expected_abstain = [bool(item.get("expected_abstain")) for item in scores]
    predicted_abstain = [bool(item.get("predicted_abstain")) for item in scores]
    tp = sum(gold and pred for gold, pred in zip(expected_abstain, predicted_abstain))
    fp = sum(not gold and pred for gold, pred in zip(expected_abstain, predicted_abstain))
    fn = sum(gold and not pred for gold, pred in zip(expected_abstain, predicted_abstain))
    abstention_precision = tp / (tp + fp) if tp + fp else 0.0
    abstention_recall = tp / (tp + fn) if tp + fn else 0.0
    abstention_f1 = (
        2 * abstention_precision * abstention_recall
        / (abstention_precision + abstention_recall)
        if abstention_precision + abstention_recall
        else 0.0
    )
    claim_scores = [
        float(item["claim_support_rate"])
        for item in scores
        if item.get("claim_support_rate") is not None
    ]
    route_labels = sorted(
        {
            str(item.get(field, ""))
            for item in scores
            for field in ("expected_route", "predicted_route")
            if item.get(field, "")
        }
    )
    route_f1_values = []
    for label in route_labels:
        tp_route = sum(
            item.get("expected_route") == label and item.get("predicted_route") == label
            for item in scores
        )
        fp_route = sum(
            item.get("expected_route") != label and item.get("predicted_route") == label
            for item in scores
        )
        fn_route = sum(
            item.get("expected_route") == label and item.get("predicted_route") != label
            for item in scores
        )
        denominator = 2 * tp_route + fp_route + fn_route
        route_f1_values.append(2 * tp_route / denominator if denominator else 0.0)
    return {
        "case_count": count,
        "route_accuracy": mean("route_correct"),
        "route_macro_f1": sum(route_f1_values) / len(route_f1_values)
        if route_f1_values
        else 0.0,
        "status_accuracy": mean("status_correct"),
        "mean_fact_coverage": mean("fact_coverage"),
        "citation_hit_rate": mean("citation_hit"),
        "citation_precision": sum(float(item.get("citation_precision", 0)) for item in scores)
        / count,
        "citation_recall": sum(float(item.get("citation_recall", 0)) for item in scores)
        / count,
        "claim_support_rate": sum(claim_scores) / len(claim_scores) if claim_scores else None,
        "abstention_precision": abstention_precision,
        "abstention_recall": abstention_recall,
        "abstention_f1": abstention_f1,
        "mean_second_round_incremental_recall": sum(
            float(item.get("second_round_incremental_recall", 0)) for item in scores
        )
        / count,
        "invalid_second_round_rate": sum(
            float(item.get("invalid_second_round", 0)) for item in scores
        )
        / count,
        "success_rate": mean("success"),
    }
