import unittest

from finresearch.agent import ResearchAgent
from finresearch.agent_schemas import (
    AnswerValidation,
    Citation,
    GroundedAnswer,
    QueryPlan,
    VerificationDecision,
)
from finresearch.schemas import SearchResult


class FakeRetriever:
    def search(self, query, top_k=5):
        if "原因" in query:
            return [SearchResult("reason", 0.8, 1, text="增加原因为职工薪酬上升")]
        return [SearchResult("amount", 0.9, 1, text="研发费用为100亿元")]


class TwoRoundLLM:
    def __init__(self):
        self.verify_calls = 0

    def plan(self, question):
        return QueryPlan(("研发费用金额",))

    def verify(self, question, evidence):
        self.verify_calls += 1
        if not any(item.doc_id == "reason" for item in evidence):
            return VerificationDecision(
                False,
                missing_evidence=("增长原因",),
                follow_up_queries=("研发费用增加原因",),
            )
        return VerificationDecision(True)

    def answer(self, question, evidence):
        return GroundedAnswer(
            "研发费用为100亿元，增加原因为职工薪酬上升。",
            (
                Citation("amount", "研发费用为100亿元"),
                Citation("reason", "增加原因为职工薪酬上升"),
            ),
        )

    def validate_answer(self, question, answer, evidence):
        return AnswerValidation(True)


class NeverEnoughLLM(TwoRoundLLM):
    def verify(self, question, evidence):
        return VerificationDecision(False, missing_evidence=("缺失指标",))


class BadCitationLLM(TwoRoundLLM):
    def verify(self, question, evidence):
        return VerificationDecision(True)

    def answer(self, question, evidence):
        return GroundedAnswer("虚构答案", (Citation("unknown", "不存在的原文"),))


class UnsupportedClaimLLM(TwoRoundLLM):
    def verify(self, question, evidence):
        return VerificationDecision(True)

    def answer(self, question, evidence):
        return GroundedAnswer(
            "研发投入增长是因为工资增加。",
            (Citation("amount", "研发费用为100亿元"),),
        )

    def validate_answer(self, question, answer, evidence):
        return AnswerValidation(False, ("引用没有支持因果关系",))


class ResearchAgentTests(unittest.TestCase):
    def test_second_round_retrieval_completes_evidence(self):
        result = ResearchAgent(FakeRetriever(), TwoRoundLLM()).run("研发费用如何变化？")
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.iterations, 2)
        self.assertEqual({item.doc_id for item in result.evidence}, {"amount", "reason"})
        self.assertEqual(len(result.citations), 2)

    def test_abstains_after_iteration_limit(self):
        result = ResearchAgent(FakeRetriever(), NeverEnoughLLM()).run("未知指标")
        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.refusal_reason, "insufficient_evidence")
        self.assertEqual(result.iterations, 2)

    def test_rejects_hallucinated_citation(self):
        result = ResearchAgent(FakeRetriever(), BadCitationLLM()).run("研发费用")
        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.refusal_reason, "citation_validation_failed")

    def test_rejects_semantically_unsupported_claim(self):
        result = ResearchAgent(FakeRetriever(), UnsupportedClaimLLM()).run("研发投入为何增长")
        self.assertEqual(result.status, "abstained")
        self.assertIn("answer_grounding_failed", result.refusal_reason)

    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            ResearchAgent(FakeRetriever(), TwoRoundLLM()).run("  ")


if __name__ == "__main__":
    unittest.main()
