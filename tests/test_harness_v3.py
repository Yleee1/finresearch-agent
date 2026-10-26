import json
import tempfile
import unittest
from pathlib import Path

from finresearch.agent import ResearchAgent
from finresearch.agent_eval_cli import _EvolvedPlanner
from finresearch.agent_schemas import (
    AnswerValidation,
    GroundedAnswer,
    QueryPlan,
    ResearchPlan,
    VerificationDecision,
)
from finresearch.harness_config import HarnessConfig
from finresearch.memory_policy import MemoryPolicyError
from finresearch.planner import PlannerValidationError, validate_research_plan
from finresearch.procedural_memory import ProceduralMemoryStore
from finresearch.retrieval_value_gate import RetrievalValueGate
from finresearch.schemas import SearchResult
from finresearch.trace_events import JSONLTraceSink, TraceEvent


class HarnessV3Tests(unittest.TestCase):
    def test_f5_evolved_planner_executes_only_allow_listed_route_rules(self):
        class Planner:
            def plan(self, question):
                return ResearchPlan(
                    answerability="answerable",
                    route="single_fact",
                    entities=(),
                    periods=(),
                    evidence_slots=(),
                    retrieval_queries=(question,),
                )

        rules = {
            "enabled": True,
            "report_year": 2024,
            "out_of_scope_terms": ["天气"],
            "unanswerable_terms": ["预测"],
            "multi_slot_markers": ["分别"],
        }
        planner = _EvolvedPlanner(Planner(), rules)
        self.assertEqual(planner.plan("明天天气如何").route, "out_of_scope")
        self.assertEqual(planner.plan("预测 2025 年收入").route, "unanswerable")
        self.assertEqual(
            planner.plan("分别给出收入和利润").route,
            "same_section_multi_slot",
        )
        self.assertEqual(planner.plan("2024 年收入").route, "single_fact")

    def test_planner_invalid_schema_fails_closed(self):
        with self.assertRaises(PlannerValidationError):
            validate_research_plan(
                {"answerability": "maybe", "route": "single_fact", "retrieval_queries": ["q"]},
                "q",
            )

    def test_gate_blocks_near_duplicate_query(self):
        decision = RetrievalValueGate().decide(
            missing_slots=("catl_revenue",),
            proposed_queries=("宁德时代 2024 营业收入",),
            used_queries=("宁德时代 2024年营业收入",),
            remaining_rounds=1,
        )
        self.assertFalse(decision.should_retry)
        self.assertEqual(decision.stop_reason, "near_duplicate_query")

    def test_memory_needs_repeated_evidence_and_rejects_financial_facts(self):
        store = ProceduralMemoryStore(activation_evidence=2)
        try:
            first = store.add_candidate(
                memory_type="procedural", task_signature="cross_company|revenue",
                content="跨公司对比应为每家公司建立独立证据槽位",
                applies_to="cross_company", source_trace_id="t1",
            )
            self.assertEqual(first.status, "disputed")
            second = store.add_candidate(
                memory_type="procedural", task_signature="cross_company|revenue",
                content="跨公司对比应为每家公司建立独立证据槽位",
                applies_to="cross_company", source_trace_id="t2",
            )
            self.assertEqual(second.status, "active")
            self.assertEqual(len(store.search("cross_company|revenue")), 1)
            with self.assertRaises(MemoryPolicyError):
                store.add_candidate(
                    memory_type="procedural", task_signature="x",
                    content="比亚迪营业收入是100亿元并应永久记住",
                    applies_to="x", source_trace_id="t3",
                )
        finally:
            store.close()

    def test_expired_or_disputed_memory_is_not_retrieved(self):
        store = ProceduralMemoryStore(activation_evidence=1)
        try:
            record = store.add_candidate(
                memory_type="negative_experience", task_signature="table|header",
                content="表格数据行缺少表头时应拼接相邻标题块",
                applies_to="table", source_trace_id="t1",
            )
            self.assertEqual(len(store.search("table|header")), 1)
            store.set_status(record.memory_id, "disputed")
            self.assertEqual(store.search("table|header"), [])
        finally:
            store.close()

    def test_harness_cannot_modify_scorer(self):
        with self.assertRaises(ValueError):
            HarnessConfig().apply_candidate("scorer", {"value": 1}, next_version="bad")

    def test_trace_sink_truncates_raw_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            JSONLTraceSink(path).emit(TraceEvent(
                trace_id="t", run_id="r", dataset_split="dev", harness_version="v",
                stage="retrieval", output_summary={"raw": "x" * 1000},
            ))
            payload = json.loads(path.read_text())
            self.assertLessEqual(len(payload["output_summary"]["raw"]), 280)

    def test_agent_records_gate_stop_reason(self):
        class Retriever:
            def search(self, query, top_k=5):
                return [SearchResult("same", 1, 1, text="partial")]

        class LLM:
            def plan(self, question):
                return QueryPlan(("same query",), route="single")
            def verify(self, question, evidence):
                return VerificationDecision(False, ("missing",), ("same query",))
            def answer(self, question, evidence):
                return GroundedAnswer("", ())
            def validate_answer(self, question, answer, evidence):
                return AnswerValidation(False)

        result = ResearchAgent(Retriever(), LLM()).run("q")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.stop_reason, "near_duplicate_query")


if __name__ == "__main__":
    unittest.main()
