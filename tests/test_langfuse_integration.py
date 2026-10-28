import unittest
from contextlib import nullcontext

from finresearch.observability import OptionalLangfuseSink
from finresearch.trace_events import TraceEvent


class LangfuseIntegrationTests(unittest.TestCase):
    def test_offline_sdk_contract_accepts_redacted_trace(self):
        from langfuse import Langfuse

        client = Langfuse(tracing_enabled=False)
        sink = OptionalLangfuseSink(client, enabled=True)
        sink.emit(TraceEvent(
            trace_id="trace-1",
            run_id="run-1",
            dataset_split="dev_v3",
            harness_version="v3-dev-0",
            stage="retrieval",
            input_refs=("doc-1:p12",),
            output_summary={"slot_coverage": 1.0, "raw_text_sent": False},
            token_usage={"prompt_tokens": 0},
        ))
        sink.flush()

    def test_old_client_shape_is_rejected(self):
        with self.assertRaises(RuntimeError):
            OptionalLangfuseSink(object(), enabled=True).emit(TraceEvent(
                trace_id="t", run_id="r", dataset_split="dev_v3",
                harness_version="v", stage="planner",
            ))

    def test_stages_from_one_run_share_native_langfuse_trace(self):
        class RecordingClient:
            def __init__(self):
                self.calls = []

            @staticmethod
            def create_trace_id(*, seed):
                return f"native:{seed}"

            def start_as_current_observation(self, **kwargs):
                self.calls.append(kwargs)
                return nullcontext()

        client = RecordingClient()
        sink = OptionalLangfuseSink(client, enabled=True)
        for stage in ("planner", "retrieval", "result"):
            sink.emit(TraceEvent(
                trace_id="trace-one-run",
                run_id="run-1",
                dataset_split="interactive",
                harness_version="v3-dev-0",
                stage=stage,
            ))

        trace_ids = {
            call["trace_context"]["trace_id"]
            for call in client.calls
        }
        self.assertEqual(trace_ids, {"native:trace-one-run"})


if __name__ == "__main__":
    unittest.main()
