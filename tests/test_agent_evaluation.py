import unittest

from finresearch.agent_evaluation import (
    AgentEvalCase,
    aggregate_agent_scores,
    score_agent_result,
)


class AgentEvaluationTests(unittest.TestCase):
    def test_scores_answered_case(self):
        case = AgentEvalCase(
            "a1", "问题", "single", "answered", ("110,335,509", "30.48%"), ("d1",)
        )
        score = score_agent_result(
            case,
            {
                "route": "single",
                "status": "answered",
                "answer": "收入110335509千元，占比30.48%。",
                "citations": [{"doc_id": "d1"}],
            },
        )
        self.assertTrue(score["success"])

    def test_scores_correct_abstention_without_citation(self):
        case = AgentEvalCase("u1", "问题", "single", "abstained", (), ())
        score = score_agent_result(
            case,
            {"route": "single", "status": "abstained", "answer": "证据不足", "citations": []},
        )
        self.assertTrue(score["success"])

    def test_aggregates_metrics(self):
        report = aggregate_agent_scores(
            [
                {"route_correct": True, "status_correct": True, "fact_coverage": 1.0, "citation_hit": True, "success": True},
                {"route_correct": False, "status_correct": True, "fact_coverage": 0.5, "citation_hit": False, "success": False},
            ]
        )
        self.assertEqual(report["route_accuracy"], 0.5)
        self.assertEqual(report["success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
