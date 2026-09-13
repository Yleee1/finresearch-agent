from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schemas import Document, QueryCase


def _read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def load_documents(path: str | Path) -> list[Document]:
    return [Document(**record) for record in _read_jsonl(path)]


def load_queries(path: str | Path) -> list[QueryCase]:
    cases: list[QueryCase] = []
    for record in _read_jsonl(path):
        cases.append(
            QueryCase(
                query_id=record["query_id"],
                query=record["query"],
                relevant_doc_ids=tuple(record["relevant_doc_ids"]),
            )
        )
    return cases


def write_documents(path: str | Path, documents: list[Document]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")
