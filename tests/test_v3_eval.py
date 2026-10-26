import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class V3EvaluationDataTests(unittest.TestCase):
    def test_manifest_hashes_and_exact_question_isolation(self):
        manifest = json.loads((ROOT / "data/eval/v3/manifest.json").read_text())
        questions = {}
        for split, metadata in manifest["splits"].items():
            path = ROOT / metadata["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), metadata["sha256"])
            rows = _rows(path)
            self.assertEqual(len(rows), metadata["count"])
            questions[split] = {row["question"] for row in rows}
        self.assertFalse(questions["dev"] & questions["heldout"])
        self.assertFalse(questions["dev"] & questions["frozen"])
        self.assertFalse(questions["heldout"] & questions["frozen"])

    def test_v3_gold_chunks_do_not_overlap_v2_agent_gold(self):
        v2 = _rows(ROOT / "data/eval/frozen_v2/agent_frozen_v2.jsonl")
        v3 = [
            row
            for name in ("dev_v3.jsonl", "heldout_v3.jsonl", "frozen_v3.jsonl")
            for row in _rows(ROOT / "data/eval/v3" / name)
        ]
        v2_ids = {item for row in v2 for item in row.get("relevant_doc_ids", [])}
        v3_ids = {item for row in v3 for item in row.get("relevant_doc_ids", [])}
        self.assertFalse(v2_ids & v3_ids)


if __name__ == "__main__":
    unittest.main()
