import unittest

from finresearch.bm25 import BM25Retriever
from finresearch.fusion import reciprocal_rank_fusion
from finresearch.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from finresearch.schemas import Document, SearchResult
from finresearch.tokenization import tokenize


class TokenizationTests(unittest.TestCase):
    def test_mixed_chinese_and_ascii(self):
        tokens = tokenize("比亚迪 Q3 毛利率22.3%")
        self.assertIn("比亚", tokens)
        self.assertIn("q3", tokens)
        self.assertIn("22.3%", tokens)


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document("rnd", "研发投入", "公司研发投入同比增长36%"),
            Document("margin", "毛利率", "汽车业务毛利率为22.3%"),
            Document("policy", "补贴政策", "以旧换新和充电设施"),
        ]

    def test_relevant_document_ranks_first(self):
        results = BM25Retriever(self.documents).search("汽车业务毛利率", top_k=3)
        self.assertEqual(results[0].doc_id, "margin")
        self.assertGreater(results[0].score, results[1].score)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            BM25Retriever([])


class FusionTests(unittest.TestCase):
    def test_rrf_rewards_consistent_ranking(self):
        first = [SearchResult("a", 8.0, 1), SearchResult("b", 7.0, 2)]
        second = [SearchResult("b", 0.9, 1), SearchResult("c", 0.8, 2)]
        fused = reciprocal_rank_fusion([first, second], top_k=3, rank_constant=60)
        self.assertEqual(fused[0].doc_id, "b")


class MetricTests(unittest.TestCase):
    def test_binary_relevance_metrics(self):
        ranked = ["x", "a", "b"]
        relevant = {"a", "b"}
        self.assertEqual(recall_at_k(ranked, relevant, 2), 0.5)
        self.assertEqual(hit_at_k(ranked, relevant, 2), 1.0)
        self.assertEqual(hit_at_k(["x"], relevant, 1), 0.0)
        self.assertEqual(reciprocal_rank(ranked, relevant), 0.5)
        self.assertGreater(ndcg_at_k(ranked, relevant, 3), 0.0)


if __name__ == "__main__":
    unittest.main()
