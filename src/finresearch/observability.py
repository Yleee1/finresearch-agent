from __future__ import annotations

from .trace_events import TraceEvent


class OptionalLangfuseSink:
    """Langfuse v4 sink; raw annual-report text is never sent.

    The client is injected so tests can use ``Langfuse(tracing_enabled=False)`` and
    production can use ``langfuse.get_client()`` with server-side credentials.
    """

    def __init__(self, client=None, *, enabled: bool = False) -> None:
        self.client = client
        self.enabled = enabled

    def emit(self, event: TraceEvent) -> None:
        if not self.enabled or self.client is None:
            return
        start = getattr(self.client, "start_as_current_observation", None)
        if start is None:
            raise RuntimeError("Langfuse v4 client with start_as_current_observation is required")
        metadata = {
                "trace_id": event.trace_id,
                "run_id": event.run_id,
                "dataset_split": event.dataset_split,
                "harness_version": event.harness_version,
                "input_refs": list(event.input_refs),
                "latency_ms": event.latency_ms,
                "error_code": event.error_code,
                "token_usage": event.token_usage,
                "evaluation_scores": event.evaluation_scores,
            }
        with start(
            name=event.stage,
            as_type="span",
            input={"input_refs": list(event.input_refs)},
            output=event.output_summary,
            metadata=metadata,
            version=event.harness_version,
            level="ERROR" if event.error_code else "DEFAULT",
            status_message=event.error_code or None,
        ):
            pass

    def flush(self) -> None:
        if not self.enabled or self.client is None:
            return
        flush = getattr(self.client, "flush", None)
        if flush is not None:
            flush()
