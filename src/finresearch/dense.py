from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .schemas import Document, SearchResult


class TextEncoder(Protocol):
    def encode(self, sentences, **kwargs): ...


class DenseRetriever:
    """Bi-encoder retrieval with normalized embeddings and cosine similarity."""

    def __init__(
        self,
        documents: list[Document],
        *,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        encoder: TextEncoder | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if not documents:
            raise ValueError("documents must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Dense retrieval requires: pip install -e '.[retrieval]'"
                ) from exc
            encoder = SentenceTransformer(model_name, device=device)

        self.documents = documents
        self.encoder = encoder
        self.batch_size = batch_size
        passages = [f"{document.title}\n{document.text}" for document in documents]
        self.document_embeddings = self._encode(passages)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self.encoder.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(texts):
            raise ValueError("encoder returned an invalid embedding shape")
        return array

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_embedding = self._encode([query])[0]
        scores = self.document_embeddings @ query_embedding
        indices = sorted(
            range(len(self.documents)),
            key=lambda index: (-float(scores[index]), self.documents[index].doc_id),
        )[:top_k]
        return [
            SearchResult(
                doc_id=self.documents[index].doc_id,
                score=float(scores[index]),
                rank=rank,
                title=self.documents[index].title,
                text=self.documents[index].text,
                source=self.documents[index].source,
                document_id=self.documents[index].document_id,
                company=self.documents[index].company,
                report_year=self.documents[index].report_year,
                page_start=self.documents[index].page_start,
                page_end=self.documents[index].page_end,
                section_path=self.documents[index].section_path,
                prev_chunk_id=self.documents[index].prev_chunk_id,
                next_chunk_id=self.documents[index].next_chunk_id,
                chunk_type=self.documents[index].chunk_type,
                table_id=self.documents[index].table_id,
                table_header=self.documents[index].table_header,
                row_start=self.documents[index].row_start,
                row_end=self.documents[index].row_end,
            )
            for rank, index in enumerate(indices, start=1)
        ]
