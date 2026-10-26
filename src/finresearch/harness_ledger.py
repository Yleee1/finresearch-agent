from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import time


@dataclass(frozen=True)
class HarnessCandidateRecord:
    candidate_id: str
    parent_version: str
    component: str
    hypothesis: str
    diff_summary: str
    dev_delta: float
    heldout_delta: float | None
    latency_delta: float
    token_delta: float
    regressions: tuple[str, ...] = field(default_factory=tuple)
    decision: str = "pending"
    created_at: float = field(default_factory=time)


class HarnessLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: HarnessCandidateRecord) -> None:
        if record.decision not in {"pending", "accepted", "rejected", "rolled_back"}:
            raise ValueError("invalid candidate decision")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
