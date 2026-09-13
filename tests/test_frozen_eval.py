from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class FrozenEvaluationIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(
            (ROOT / "data/eval/frozen_v1_manifest.json").read_text(encoding="utf-8")
        )

    def test_hashes_and_counts_match_manifest(self) -> None:
        for split in ("retrieval", "agent"):
            metadata = self.manifest[split]
            path = ROOT / metadata["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(metadata["sha256"], digest)
            self.assertEqual(metadata["case_count"], len(load_jsonl(path)))

        for metadata in self.manifest["corpus"]:
            path = ROOT / metadata["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(metadata["sha256"], digest)
            self.assertEqual(metadata["chunk_count"], len(load_jsonl(path)))

    def test_frozen_questions_and_relevant_chunks_are_isolated(self) -> None:
        retrieval_seed = load_jsonl(ROOT / "data/eval/retrieval_seed_v1.jsonl")
        retrieval_frozen = load_jsonl(ROOT / "data/eval/retrieval_frozen_v1.jsonl")
        agent_seed = load_jsonl(ROOT / "data/eval/agent_seed_v1.jsonl")
        agent_frozen = load_jsonl(ROOT / "data/eval/agent_frozen_v1.jsonl")

        seed_questions = {case["query"] for case in retrieval_seed} | {
            case["question"] for case in agent_seed
        }
        frozen_questions = {case["query"] for case in retrieval_frozen} | {
            case["question"] for case in agent_frozen
        }
        self.assertFalse(seed_questions & frozen_questions)

        seed_chunks = {
            doc_id
            for case in retrieval_seed + agent_seed
            for doc_id in case["relevant_doc_ids"]
        }
        frozen_chunks = {
            doc_id
            for case in retrieval_frozen + agent_frozen
            for doc_id in case["relevant_doc_ids"]
        }
        self.assertFalse(seed_chunks & frozen_chunks)

    def test_agent_class_balance_matches_manifest(self) -> None:
        cases = load_jsonl(ROOT / "data/eval/agent_frozen_v1.jsonl")
        answered = sum(case["expected_status"] == "answered" for case in cases)
        abstained = sum(case["expected_status"] == "abstained" for case in cases)
        self.assertEqual(self.manifest["agent"]["answered_count"], answered)
        self.assertEqual(self.manifest["agent"]["abstained_count"], abstained)


if __name__ == "__main__":
    unittest.main()
