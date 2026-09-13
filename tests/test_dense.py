import unittest

import numpy as np

from finresearch.bm25 import BM25Retriever
from finresearch.dense import DenseRetriever
from finresearch.hybrid import HybridRetriever
from finresearch.schemas import Document


class FakeEncoder:
    vectors = {
        "汽车毛利率": [1.0, 0.0],
        "利润表现": [1.0, 0.0],
        "电池产能": [0.0, 1.0],
        "产能情况": [0.0, 1.0],
    }

    def encode(self, sentences, **kwargs):
        rows = []
        for sentence in sentences:
            key = next((key for key in self.vectors if key in sentence), None)
            rows.append(self.vectors.get(key, [0.5, 0.5]))
        array = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / norms


class DenseRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document("margin", "汽车毛利率", "利润表现"),
            Document("capacity", "电池产能", "产能情况"),
        ]

    def test_semantic_ranking(self):
        retriever = DenseRetriever(self.documents, encoder=FakeEncoder())
        results = retriever.search("利润表现", top_k=2)
        self.assertEqual(results[0].doc_id, "margin")
        self.assertGreater(results[0].score, results[1].score)

    def test_hybrid_retriever_returns_ranked_results(self):
        dense = DenseRetriever(self.documents, encoder=FakeEncoder())
        hybrid = HybridRetriever(BM25Retriever(self.documents), dense, candidate_k=2)
        results = hybrid.search("汽车毛利率", top_k=2)
        self.assertEqual(results[0].doc_id, "margin")

    def test_rejects_empty_documents(self):
        with self.assertRaises(ValueError):
            DenseRetriever([], encoder=FakeEncoder())


if __name__ == "__main__":
    unittest.main()
