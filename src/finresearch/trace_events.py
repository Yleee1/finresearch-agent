from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import time
from typing import Any, Protocol


@dataclass(frozen=True)
class TraceEvent:
    trace_id: str
    run_id: str
    dataset_split: str
    harness_version: str
    stage: str
    input_refs: tuple[str, ...] = ()
    output_summary: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    error_code: str = ""
    evaluation_scores: dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time)


class TraceSink(Protocol):
    def emit(self, event: TraceEvent) -> None: ...


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= 280 else value[:277] + "..."
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)


class JSONLTraceSink:
    """Local replayable sink; summaries are length-bounded to avoid full-text leaks."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: TraceEvent) -> None:
        payload = _safe(asdict(event))
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")


class NullTraceSink:
    def emit(self, event: TraceEvent) -> None:
        return None
