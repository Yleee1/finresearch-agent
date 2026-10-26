from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .agent_schemas import ResearchPlan


def planner_metrics(pairs: Iterable[tuple[ResearchPlan, ResearchPlan]]) -> dict[str, float]:
    rows = list(pairs)
    if not rows:
        return {"case_count": 0}
    totals = Counter()
    for expected, actual in rows:
        expected_entities, actual_entities = set(expected.entities), set(actual.entities)
        totals["entity_tp"] += len(expected_entities & actual_entities)
        totals["entity_fp"] += len(actual_entities - expected_entities)
        totals["entity_fn"] += len(expected_entities - actual_entities)
        totals["period"] += expected.periods == actual.periods
        totals["route"] += expected.route == actual.route
        totals["answerability"] += expected.answerability == actual.answerability
        expected_slots = {slot.slot_id for slot in expected.evidence_slots}
        actual_slots = {slot.slot_id for slot in actual.evidence_slots}
        totals["slot_sum"] += len(expected_slots & actual_slots) / max(1, len(expected_slots))
        totals["dependency"] += expected.dependencies == actual.dependencies
    precision = totals["entity_tp"] / max(1, totals["entity_tp"] + totals["entity_fp"])
    recall = totals["entity_tp"] / max(1, totals["entity_tp"] + totals["entity_fn"])
    count = len(rows)
    return {
        "case_count": count,
        "entity_precision": precision,
        "entity_recall": recall,
        "period_accuracy": totals["period"] / count,
        "route_accuracy": totals["route"] / count,
        "answerability_accuracy": totals["answerability"] / count,
        "evidence_slot_coverage": totals["slot_sum"] / count,
        "dependency_accuracy": totals["dependency"] / count,
    }
