from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .memory_policy import validate_procedural_content


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2,}", text.lower()))


@dataclass(frozen=True)
class ResearchMemory:
    memory_id: str
    memory_type: str
    task_signature: str
    content: str
    applies_to: str
    source_trace_ids: tuple[str, ...]
    evidence_count: int
    confidence: float
    created_at: str
    valid_from: str
    valid_to: str | None = None
    supersedes: tuple[str, ...] = ()
    status: str = "active"


class ProceduralMemoryStore:
    """Auditable local memory for strategies, never answer facts."""

    def __init__(self, path: str | Path = ":memory:", *, activation_evidence: int = 2) -> None:
        self.path = str(path)
        self.activation_evidence = activation_evidence
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS research_memory (
            memory_id TEXT PRIMARY KEY, memory_type TEXT NOT NULL, task_signature TEXT NOT NULL,
            content TEXT NOT NULL, applies_to TEXT NOT NULL, source_trace_ids TEXT NOT NULL,
            evidence_count INTEGER NOT NULL, confidence REAL NOT NULL, created_at TEXT NOT NULL,
            valid_from TEXT NOT NULL, valid_to TEXT, supersedes TEXT NOT NULL, status TEXT NOT NULL,
            UNIQUE(task_signature, content)
            )"""
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    @staticmethod
    def _record(row: sqlite3.Row) -> ResearchMemory:
        return ResearchMemory(
            memory_id=row["memory_id"], memory_type=row["memory_type"],
            task_signature=row["task_signature"], content=row["content"],
            applies_to=row["applies_to"], source_trace_ids=tuple(json.loads(row["source_trace_ids"])),
            evidence_count=int(row["evidence_count"]), confidence=float(row["confidence"]),
            created_at=row["created_at"], valid_from=row["valid_from"], valid_to=row["valid_to"],
            supersedes=tuple(json.loads(row["supersedes"])), status=row["status"],
        )

    def add_candidate(
        self,
        *,
        memory_type: str,
        task_signature: str,
        content: str,
        applies_to: str,
        source_trace_id: str,
        confidence: float = 0.5,
    ) -> ResearchMemory:
        if memory_type not in {"procedural", "negative_experience"}:
            raise ValueError("only procedural and negative_experience memories are allowed")
        clean = validate_procedural_content(content)
        existing = self.connection.execute(
            "SELECT * FROM research_memory WHERE task_signature=? AND content=?",
            (task_signature, clean),
        ).fetchone()
        if existing:
            sources = list(dict.fromkeys(json.loads(existing["source_trace_ids"]) + [source_trace_id]))
            count = len(sources)
            status = "active" if count >= self.activation_evidence else "disputed"
            self.connection.execute(
                "UPDATE research_memory SET source_trace_ids=?, evidence_count=?, confidence=?, status=? WHERE memory_id=?",
                (json.dumps(sources), count, max(float(existing["confidence"]), confidence), status, existing["memory_id"]),
            )
            self.connection.commit()
            row = self.connection.execute("SELECT * FROM research_memory WHERE memory_id=?", (existing["memory_id"],)).fetchone()
            return self._record(row)
        created = _now()
        memory_id = f"rm-{uuid4().hex}"
        status = "active" if self.activation_evidence <= 1 else "disputed"
        self.connection.execute(
            "INSERT INTO research_memory VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (memory_id, memory_type, task_signature, clean, applies_to,
             json.dumps([source_trace_id]), 1, confidence, created, created, None, "[]", status),
        )
        self.connection.commit()
        row = self.connection.execute("SELECT * FROM research_memory WHERE memory_id=?", (memory_id,)).fetchone()
        return self._record(row)

    def search(self, task_signature: str, *, limit: int = 3, at: str | None = None) -> list[ResearchMemory]:
        instant = at or _now()
        rows = self.connection.execute(
            "SELECT * FROM research_memory WHERE status='active' AND valid_from<=? AND (valid_to IS NULL OR valid_to>?)",
            (instant, instant),
        ).fetchall()
        query_tokens = _tokens(task_signature)
        ranked = []
        for row in rows:
            record = self._record(row)
            candidate = _tokens(record.task_signature + " " + record.applies_to)
            overlap = len(query_tokens & candidate) / max(1, len(query_tokens | candidate))
            exact = 1.0 if record.task_signature == task_signature else 0.0
            ranked.append((exact + overlap + record.confidence * 0.1, record))
        ranked.sort(key=lambda item: (-item[0], item[1].memory_id))
        return [record for score, record in ranked[:limit] if score > 0]

    def set_status(self, memory_id: str, status: str, *, valid_to: str | None = None) -> None:
        if status not in {"active", "disputed", "expired"}:
            raise ValueError("invalid memory status")
        self.connection.execute(
            "UPDATE research_memory SET status=?, valid_to=? WHERE memory_id=?",
            (status, valid_to, memory_id),
        )
        self.connection.commit()

    def export(self) -> list[dict]:
        return [asdict(self._record(row)) for row in self.connection.execute("SELECT * FROM research_memory ORDER BY created_at, memory_id")]
