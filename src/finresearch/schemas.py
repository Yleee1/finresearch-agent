from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    source: str = ""
    document_id: str = ""
    company: str = ""
    report_year: int | None = None
    section_path: str = ""
    page_start: int | None = None
    page_end: int | None = None
    prev_chunk_id: str = ""
    next_chunk_id: str = ""
    chunk_type: str = "text"
    table_id: str = ""
    table_header: tuple[str, ...] = ()
    row_start: int | None = None
    row_end: int | None = None


@dataclass(frozen=True)
class QueryCase:
    query_id: str
    query: str
    relevant_doc_ids: tuple[str, ...]


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    score: float
    rank: int
    title: str = ""
    text: str = ""
    source: str = ""
    document_id: str = ""
    company: str = ""
    report_year: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_path: str = ""
    prev_chunk_id: str = ""
    next_chunk_id: str = ""
    chunk_type: str = "text"
    table_id: str = ""
    table_header: tuple[str, ...] = ()
    row_start: int | None = None
    row_end: int | None = None
