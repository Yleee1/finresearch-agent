from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .agent_schemas import EvidenceSlot, QueryPlan, ResearchPlan
from .query_analysis import QueryAnalyzer

ALLOWED_ANSWERABILITY = {"answerable", "unanswerable", "unknown"}
ALLOWED_ROUTES = {
    "single",
    "multi",
    "single_fact",
    "same_section_multi_slot",
    "cross_company_compare",
    "cross_section_multi_hop",
    "cross_period_compare",
    "derived_calculation",
    "unanswerable",
    "out_of_scope",
}


class PlannerValidationError(ValueError):
    pass


def _slot(value: EvidenceSlot | dict[str, Any]) -> EvidenceSlot:
    if isinstance(value, EvidenceSlot):
        return value
    if not isinstance(value, dict):
        raise PlannerValidationError("evidence slot must be an object")
    slot_id = str(value.get("slot_id", "")).strip()
    if not slot_id:
        raise PlannerValidationError("evidence slot requires slot_id")
    period = value.get("period", value.get("year"))
    year = int(period) if str(period or "").isdigit() else None
    depends_on = value.get("depends_on", ())
    return EvidenceSlot(
        slot_id=slot_id,
        description=str(value.get("description") or " ".join(
            str(value.get(key, "")) for key in ("company", "metric", "period")
        )).strip(),
        company=str(value.get("company", "")).strip(),
        year=year,
        metric=str(value.get("metric", "")).strip(),
        depends_on=tuple(str(item) for item in depends_on or ()),
    )


def validate_research_plan(value: ResearchPlan | QueryPlan | dict[str, Any], question: str) -> ResearchPlan:
    """Coerce a planner response and fail closed on malformed structured fields."""

    if isinstance(value, ResearchPlan):
        plan = value
    elif isinstance(value, QueryPlan):
        entities = tuple(value.company_filters)
        periods = tuple(str(item) for item in value.year_filters)
        plan = ResearchPlan(
            answerability=(
                "unanswerable"
                if value.route in {"out_of_scope", "unanswerable"}
                else "unknown"
            ),
            route=value.route,
            entities=entities,
            periods=periods,
            evidence_slots=value.evidence_slots,
            dependencies=value.dependencies,
            retrieval_queries=value.sub_queries or (question,),
            reasoning=value.reasoning,
        )
    elif isinstance(value, dict):
        dependencies = []
        for item in value.get("dependencies", ()) or ():
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise PlannerValidationError("each dependency must contain two slot ids")
            dependencies.append((str(item[0]), str(item[1])))
        plan = ResearchPlan(
            answerability=str(value.get("answerability", "unknown")),
            route=str(value.get("route", "")),
            entities=tuple(str(item) for item in value.get("entities", ()) or ()),
            periods=tuple(str(item) for item in value.get("periods", ()) or ()),
            evidence_slots=tuple(_slot(item) for item in value.get("evidence_slots", ()) or ()),
            dependencies=tuple(dependencies),
            retrieval_queries=tuple(
                str(item).strip()
                for item in value.get("retrieval_queries", ()) or ()
                if str(item).strip()
            ),
            reasoning=str(value.get("reasoning", "")),
        )
    else:
        raise PlannerValidationError("planner output must be ResearchPlan, QueryPlan, or dict")

    if plan.answerability not in ALLOWED_ANSWERABILITY:
        raise PlannerValidationError(f"invalid answerability: {plan.answerability}")
    if plan.route not in ALLOWED_ROUTES:
        raise PlannerValidationError(f"invalid route: {plan.route}")
    slot_ids = [slot.slot_id for slot in plan.evidence_slots]
    if len(slot_ids) != len(set(slot_ids)):
        raise PlannerValidationError("evidence slot ids must be unique")
    unknown = {
        item for edge in plan.dependencies for item in edge if item not in set(slot_ids)
    }
    if unknown:
        raise PlannerValidationError(f"dependency references unknown slots: {sorted(unknown)}")
    if (
        plan.answerability != "unanswerable"
        and plan.route != "out_of_scope"
        and not plan.retrieval_queries
    ):
        raise PlannerValidationError("answerable/unknown plan requires retrieval_queries")
    return plan


class StructuredPlanner:
    """Adapter around model planning with an optional deterministic fallback."""

    def __init__(self, planner=None, *, fallback: bool = False) -> None:
        self.planner = planner
        self.fallback = fallback

    def plan(self, question: str) -> ResearchPlan:
        try:
            if self.planner is None:
                raw = QueryAnalyzer().analyze(question)
            elif hasattr(self.planner, "plan_structured"):
                raw = self.planner.plan_structured(question)
            else:
                raw = self.planner.plan(question)
            return validate_research_plan(raw, question)
        except (PlannerValidationError, TypeError, ValueError):
            if not self.fallback:
                raise
            return validate_research_plan(QueryAnalyzer().analyze(question), question)

    @staticmethod
    def public_dict(plan: ResearchPlan) -> dict[str, Any]:
        return asdict(plan)
