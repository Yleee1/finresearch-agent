import unittest

import numpy as np

from finresearch.reranker import RerankRetriever
from finresearch.schemas import SearchResult


class FakeRetriever:
    def search(self, query, top_k=5):
        candidates = [
            SearchResult("wrong", 2.0, 1, "财务指标", "2023年研发费用"),
            SearchResult("right", 1.0, 2, "财务指标", "2024年研发费用为100亿元"),
        ]
        return candidates[:top_k]


class FakeScorer:
    def predict(self, pairs, **kwargs):
        return np.asarray([0.1 if "2023年" in passage else 0.9 for _, passage in pairs])


class RerankRetrieverTests(unittest.TestCase):
    def test_cross_encoder_can_promote_relevant_candidate(self):
        retriever = RerankRetriever(FakeRetriever(), scorer=FakeScorer(), candidate_k=2)
        results = retriever.search("2024年研发费用是多少", top_k=2)
        self.assertEqual(results[0].doc_id, "right")
        self.assertEqual([result.rank for result in results], [1, 2])

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            RerankRetriever(FakeRetriever(), scorer=FakeScorer(), candidate_k=0)


if __name__ == "__main__":
    unittest.main()
