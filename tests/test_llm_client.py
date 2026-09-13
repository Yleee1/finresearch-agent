import unittest

from finresearch.llm_client import DeepSeekResearchLLM, _extract_json
from finresearch.schemas import SearchResult


class QueueClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat_json(self, system, user):
        return self.responses.pop(0)


class JsonParsingTests(unittest.TestCase):
    def test_extracts_fenced_json(self):
        self.assertEqual(_extract_json('```json\n{"ok": true}\n```'), {"ok": True})

    def test_rejects_non_object(self):
        with self.assertRaises(ValueError):
            _extract_json("[1, 2]")


class DeepSeekResearchLLMTests(unittest.TestCase):
    def test_parses_all_three_structured_stages(self):
        client = QueueClient(
            [
                {"route": "multi", "sub_queries": ["研发金额", "增长原因"], "reasoning": "多跳"},
                {
                    "sufficient": True,
                    "missing_evidence": [],
                    "follow_up_queries": [],
                },
                {
                    "answer": "研发投入增长。",
                    "citations": [{"doc_id": "d1", "quote": "研发投入增长"}],
                },
                {"valid": True, "issues": []},
            ]
        )
        llm = DeepSeekResearchLLM(client)
        evidence = [SearchResult("d1", 1.0, 1, text="研发投入增长35%")]
        self.assertEqual(llm.plan("研发情况").sub_queries, ("研发金额", "增长原因"))
        self.assertTrue(llm.verify("研发情况", evidence).sufficient)
        answer = llm.answer("研发情况", evidence)
        self.assertEqual(answer.citations[0].doc_id, "d1")
        self.assertTrue(llm.validate_answer("研发情况", answer, evidence).valid)

    def test_single_route_keeps_only_one_query(self):
        llm = DeepSeekResearchLLM(
            QueueClient(
                [{"route": "single", "sub_queries": ["产能利用率", "不应执行的扩展"]}]
            )
        )
        plan = llm.plan("产能利用率是多少")
        self.assertEqual(plan.route, "single")
        self.assertEqual(plan.sub_queries, ("产能利用率",))


if __name__ == "__main__":
    unittest.main()
