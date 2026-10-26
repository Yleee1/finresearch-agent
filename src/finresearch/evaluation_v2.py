from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvidenceSlotGold:
    slot_id: str
    acceptable_evidence_sets: tuple[frozenset[str], ...]


@dataclass(frozen=True)
class RetrievalCaseV2:
    case_id: str
    question: str
    category: str
    evidence_slots: tuple[EvidenceSlotGold, ...]


def load_retrieval_cases_v2(path: str | Path) -> list[RetrievalCaseV2]:
    cases: list[RetrievalCaseV2] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                slots = tuple(
                    EvidenceSlotGold(
                        slot_id=item["slot_id"],
                        acceptable_evidence_sets=tuple(
                            frozenset(group) for group in item["acceptable_evidence_sets"]
                        ),
                    )
                    for item in value["evidence_slots"]
                )
                cases.append(
                    RetrievalCaseV2(
                        case_id=value["case_id"],
                        question=value["question"],
                        category=value["category"],
                        evidence_slots=slots,
                    )
                )
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid V2 case at {path}:{line_number}") from exc
    return cases


def score_evidence_slots(case: RetrievalCaseV2, retrieved_ids: list[str]) -> dict:
    retrieved = set(retrieved_ids)
    slot_hits = {
        slot.slot_id: any(group.issubset(retrieved) for group in slot.acceptable_evidence_sets)
        for slot in case.evidence_slots
    }
    coverage = sum(slot_hits.values()) / len(slot_hits) if slot_hits else 1.0
    return {
        "case_id": case.case_id,
        "slot_hits": slot_hits,
        "evidence_slot_coverage": coverage,
        "multi_hop_complete_evidence_recall": float(all(slot_hits.values())),
    }


def score_retrieval_rounds(
    case: RetrievalCaseV2,
    retrieval_rounds: list[list[str]],
) -> dict:
    accumulated: list[str] = []
    coverage_by_round: list[float] = []
    complete_by_round: list[float] = []
    for round_ids in retrieval_rounds:
        accumulated.extend(round_ids)
        score = score_evidence_slots(case, accumulated)
        coverage_by_round.append(float(score["evidence_slot_coverage"]))
        complete_by_round.append(float(score["multi_hop_complete_evidence_recall"]))
    first = coverage_by_round[0] if coverage_by_round else 0.0
    final = coverage_by_round[-1] if coverage_by_round else 0.0
    second_increment = coverage_by_round[1] - first if len(coverage_by_round) > 1 else 0.0
    return {
        "case_id": case.case_id,
        "coverage_by_round": coverage_by_round,
        "complete_by_round": complete_by_round,
        "evidence_slot_coverage": final,
        "multi_hop_complete_evidence_recall": complete_by_round[-1]
        if complete_by_round
        else 0.0,
        "second_round_incremental_recall": max(0.0, second_increment),
        "invalid_second_round": float(len(retrieval_rounds) > 1 and second_increment <= 0),
    }


def classification_report(expected: list[str], predicted: list[str]) -> dict:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    labels = sorted(set(expected) | set(predicted))
    per_label: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for label in labels:
        tp = sum(gold == label and pred == label for gold, pred in zip(expected, predicted))
        fp = sum(gold != label and pred == label for gold, pred in zip(expected, predicted))
        fn = sum(gold == label and pred != label for gold, pred in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "support": sum(item == label for item in expected),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    count = len(expected)
    return {
        "count": count,
        "accuracy": round(sum(a == b for a, b in zip(expected, predicted)) / count, 4)
        if count
        else 0.0,
        "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
        "per_label": per_label,
    }


def score_decomposition(expected_slots: list[str], predicted_slots: list[str]) -> dict:
    gold = set(expected_slots)
    predicted = set(predicted_slots)
    return {
        "decomposition_exact": float(gold == predicted),
        "decomposition_slot_coverage": len(gold & predicted) / len(gold) if gold else 1.0,
        "decomposition_slot_precision": len(gold & predicted) / len(predicted)
        if predicted
        else float(not gold),
    }


def score_citations(
    acceptable_doc_ids: set[str],
    citation_doc_ids: list[str],
) -> dict:
    predicted = set(citation_doc_ids)
    correct = acceptable_doc_ids & predicted
    precision = len(correct) / len(predicted) if predicted else float(not acceptable_doc_ids)
    recall = len(correct) / len(acceptable_doc_ids) if acceptable_doc_ids else float(not predicted)
    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_hit": float(bool(correct)) if acceptable_doc_ids else float(not predicted),
    }


def score_claim_support(claims: list[dict]) -> dict:
    supported = sum(bool(item.get("supported")) for item in claims)
    return {
        "claim_count": len(claims),
        "supported_claim_count": supported,
        "claim_support_rate": supported / len(claims) if claims else 1.0,
    }


def score_abstention(expected_abstain: list[bool], predicted_abstain: list[bool]) -> dict:
    if len(expected_abstain) != len(predicted_abstain):
        raise ValueError("abstention labels must have the same length")
    tp = sum(gold and pred for gold, pred in zip(expected_abstain, predicted_abstain))
    fp = sum(not gold and pred for gold, pred in zip(expected_abstain, predicted_abstain))
    fn = sum(gold and not pred for gold, pred in zip(expected_abstain, predicted_abstain))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "abstention_precision": round(precision, 4),
        "abstention_recall": round(recall, 4),
        "abstention_f1": round(f1, 4),
    }


def summarize_costs(records: list[dict]) -> dict:
    def percentile95(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
        return ordered[index]

    latencies = [float(item.get("end_to_end_seconds", 0.0)) for item in records]
    count = len(records)
    return {
        "mean_latency_seconds": round(sum(latencies) / count, 4) if count else 0.0,
        "p95_latency_seconds": round(percentile95(latencies), 4),
        "mean_retrieval_rounds": round(
            sum(float(item.get("iterations", 0)) for item in records) / count, 4
        )
        if count
        else 0.0,
        "llm_calls": sum(int(item.get("llm_calls", 0)) for item in records),
        "prompt_tokens": sum(int(item.get("prompt_tokens", 0)) for item in records),
        "completion_tokens": sum(int(item.get("completion_tokens", 0)) for item in records),
    }
def write_frozen_manifest(path: str | Path, *, source_path: str | Path) -> None:
    source = Path(source_path)
    content = source.read_bytes()
    cases = load_retrieval_cases_v2(source)
    categories: dict[str, int] = {}
    for case in cases:
        categories[case.category] = categories.get(case.category, 0) + 1
    payload = {
        "file": source.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "case_count": len(cases),
        "category_distribution": dict(sorted(categories.items())),
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
