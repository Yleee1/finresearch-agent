from __future__ import annotations

import re

from .agent_schemas import EvidenceSlot, QueryPlan

_YEAR = re.compile(r"(?:19|20)\d{2}")
_KNOWN_COMPANIES = ("比亚迪", "宁德时代", "BYD", "CATL")
_METRICS = (
    "营业收入",
    "研发投入",
    "研发费用",
    "净利润",
    "毛利率",
    "销量",
    "产能",
    "员工人数",
    "前五名客户",
    "前五名供应商",
    "销售费用",
    "现金分红",
    "每10股",
)


class QueryAnalyzer:
    """Deterministic fallback planner used in tests and fail-closed paths."""

    def analyze(self, question: str) -> QueryPlan:
        text = question.strip()
        if not text:
            raise ValueError("question must not be empty")
        companies = tuple(name for name in _KNOWN_COMPANIES if name.lower() in text.lower())
        years = tuple(dict.fromkeys(int(value) for value in _YEAR.findall(text)))
        metrics = tuple(metric for metric in _METRICS if metric in text)

        if any(word in text.lower() for word in ("写代码", "天气", "股票推荐", "prompt injection")):
            route = "out_of_scope"
        elif len(companies) > 1:
            route = "cross_company_compare"
        elif len(years) > 1 or any(word in text for word in ("跨章节", "推导", "分别查找")):
            route = "cross_section_multi_hop"
        elif len(metrics) > 1:
            route = "same_section_multi_slot"
        else:
            route = "single_fact"

        subjects = companies or ("",)
        slot_metrics = metrics or ("目标事实",)
        slots: list[EvidenceSlot] = []
        for company in subjects:
            for metric in slot_metrics:
                slot_id = "_".join(
                    part for part in (company or "fact", metric, str(years[0]) if years else "") if part
                )
                slots.append(
                    EvidenceSlot(
                        slot_id=slot_id,
                        description=" ".join(
                            part for part in (company, str(years[0]) if years else "", metric) if part
                        ),
                        company=company,
                        year=years[0] if len(years) == 1 else None,
                        metric=metric if metric != "目标事实" else "",
                    )
                )

        if route == "out_of_scope":
            sub_questions: tuple[str, ...] = ()
        elif route in {"cross_company_compare", "cross_section_multi_hop"}:
            sub_questions = tuple(slot.description for slot in slots)
        else:
            # Multiple fields expected in the same section stay one retrieval query.
            sub_questions = (text,)
        return QueryPlan(
            sub_queries=sub_questions or (text,),
            route=route,
            reasoning="rule-based V2 routing",
            sub_questions=sub_questions or (text,),
            evidence_slots=tuple(slots),
            dependencies=(),
            company_filters=companies,
            year_filters=years,
            metric_filters=metrics,
        )
