from __future__ import annotations

import math
from collections import Counter

from .schemas import Document, SearchResult
from .tokenization import tokenize


class BM25Retriever:
    def __init__(
        self,
        documents: list[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("documents must not be empty")
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.documents = documents
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(doc.title + " " + doc.text)) for doc in documents]
        self.lengths = [sum(counter.values()) for counter in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths)
        self.document_frequency: Counter[str] = Counter()
        for counter in self.term_frequencies:
            self.document_frequency.update(counter.keys())

    def _idf(self, term: str) -> float:
        document_count = len(self.documents)
        frequency = self.document_frequency.get(term, 0)
        return math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))

    def _score(self, query_tokens: list[str], document_index: int) -> float:
        frequencies = self.term_frequencies[document_index]
        document_length = self.lengths[document_index]
        length_norm = 1 - self.b + self.b * document_length / self.average_length
        score = 0.0
        for term in set(query_tokens):
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            numerator = term_frequency * (self.k1 + 1)
            denominator = term_frequency + self.k1 * length_norm
            score += self._idf(term) * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_tokens = tokenize(query)
        scored = [
            (self._score(query_tokens, index), document)
            for index, document in enumerate(self.documents)
        ]
        scored.sort(key=lambda item: (-item[0], item[1].doc_id))
        results: list[SearchResult] = []
        for rank, (score, document) in enumerate(scored[:top_k], start=1):
            results.append(
                SearchResult(
                    doc_id=document.doc_id,
                    score=score,
                    rank=rank,
                    title=document.title,
                    text=document.text,
                    source=document.source,
                    document_id=document.document_id,
                    company=document.company,
                    report_year=document.report_year,
                    page_start=document.page_start,
                    page_end=document.page_end,
                    section_path=document.section_path,
                    prev_chunk_id=document.prev_chunk_id,
                    next_chunk_id=document.next_chunk_id,
                )
            )
        return results
