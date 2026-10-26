from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .harness_config import HarnessConfig
from .planner import validate_research_plan
from .procedural_memory import ProceduralMemoryStore
from .query_analysis import QueryAnalyzer
from .retrieval_value_gate import RetrievalValueGate
from .trace_events import JSONLTraceSink, TraceEvent


def run_component_eval() -> dict:
    records: list[dict] = []

    planner_cases = (
        ("比亚迪2024年营业收入是多少？", "single_fact", 1),
        ("比亚迪2024年营业收入和净利润分别是多少？", "same_section_multi_slot", 2),
        ("比较比亚迪和宁德时代2024年营业收入", "cross_company_compare", 2),
        ("写代码查天气", "out_of_scope", 1),
    )
    for index, (question, route, slots) in enumerate(planner_cases, 1):
        plan = validate_research_plan(QueryAnalyzer().analyze(question), question)
        passed = plan.route == route and len(plan.evidence_slots) == slots
        records.append({"case_id": f"planner_{index}", "component": "planner", "passed": passed})

    gate = RetrievalValueGate()
    gate_cases = (
        ({"missing_slots": ("b",), "proposed_queries": ("CATL 2024 revenue",), "used_queries": ("BYD 2024 revenue",), "remaining_rounds": 1}, True),
        ({"missing_slots": ("b",), "proposed_queries": ("CATL 2024 revenue",), "used_queries": ("CATL 2024 revenue",), "remaining_rounds": 1}, False),
        ({"missing_slots": (), "proposed_queries": ("q",), "used_queries": (), "remaining_rounds": 1}, False),
    )
    for index, (kwargs, expected) in enumerate(gate_cases, 1):
        actual = gate.decide(**kwargs)
        records.append({
            "case_id": f"gate_{index}", "component": "retrieval_gate",
            "passed": actual.should_retry is expected, "stop_reason": actual.stop_reason,
        })

    memory = ProceduralMemoryStore(activation_evidence=2)
    try:
        first = memory.add_candidate(
            memory_type="procedural", task_signature="cross_company|revenue",
            content="跨公司对比需要为每家公司建立独立证据槽位",
            applies_to="cross_company", source_trace_id="trace-1",
        )
        second = memory.add_candidate(
            memory_type="procedural", task_signature="cross_company|revenue",
            content="跨公司对比需要为每家公司建立独立证据槽位",
            applies_to="cross_company", source_trace_id="trace-2",
        )
        records.extend((
            {"case_id": "memory_activation_gate", "component": "memory", "passed": first.status == "disputed" and second.status == "active"},
            {"case_id": "memory_search", "component": "memory", "passed": len(memory.search("cross_company|revenue")) == 1},
        ))
    finally:
        memory.close()

    protected = False
    try:
        HarnessConfig().apply_candidate("scorer", {"value": 1}, next_version="bad")
    except ValueError:
        protected = True
    records.append({"case_id": "protected_scorer", "component": "harness", "passed": protected})

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.jsonl"
        JSONLTraceSink(path).emit(TraceEvent(
            trace_id="trace", run_id="run", dataset_split="dev_v3",
            harness_version="v3-dev-0", stage="retrieval",
            output_summary={"raw": "x" * 1000},
        ))
        raw_length = len(json.loads(path.read_text())["output_summary"]["raw"])
    records.append({"case_id": "trace_redaction", "component": "trace", "passed": raw_length <= 280})

    passed = sum(bool(item["passed"]) for item in records)
    by_component = {}
    for component in sorted({item["component"] for item in records}):
        rows = [item for item in records if item["component"] == component]
        by_component[component] = {
            "case_count": len(rows),
            "success_rate": sum(bool(item["passed"]) for item in rows) / len(rows),
        }
    return {
        "suite": "v3_upgrade_component_eval",
        "scope": "deterministic component checks; not end-to-end frozen task success",
        "case_count": len(records),
        "success_rate": passed / len(records),
        "by_component": by_component,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic FinResearch V3 upgrade checks")
    parser.add_argument("--output", default="results/v3/upgrade_component_eval.json")
    args = parser.parse_args()
    result = run_component_eval()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("suite", "case_count", "success_rate")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
