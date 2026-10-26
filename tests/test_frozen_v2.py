from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from finresearch.evaluation_v2 import load_retrieval_cases_v2

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data/eval/frozen_v2"


class FrozenV2IntegrityTests(unittest.TestCase):
    def test_hashes_counts_and_schema_match_manifest(self):
        manifest = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
        for split in ("retrieval", "agent"):
            metadata = manifest[split]
            path = V2 / metadata["file"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), metadata["sha256"])
            rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(rows), metadata["case_count"])
        self.assertEqual(len(load_retrieval_cases_v2(V2 / manifest["retrieval"]["file"])), 128)

    def test_required_challenge_mix_is_present(self):
        manifest = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
        retrieval = manifest["retrieval"]
        agent = manifest["agent"]
        self.assertGreaterEqual(retrieval["category_distribution"]["cross_company_compare"], 25)
        self.assertGreaterEqual(retrieval["category_distribution"]["table_cross_chunk"], 19)
        abstention_count = (
            agent["category_distribution"]["unanswerable"]
            + agent["category_distribution"]["out_of_scope"]
        )
        self.assertGreaterEqual(abstention_count / agent["case_count"], 0.2)

    def test_review_attestation_and_public_agent_matrix(self):
        manifest = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
        for split in ("retrieval", "agent"):
            self.assertEqual(
                manifest[split]["review_status"],
                "independent_second_review_completed_user_attested",
            )
        matrix = json.loads(
            (ROOT / "results/agent_matrix_frozen_v2.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(matrix["variants"]), {"R3", "R4", "R5"})
        self.assertEqual(matrix["aggregate_usage"]["calls"], 555)
        self.assertTrue(all(row["case_count"] == 74 for row in matrix["variants"].values()))
        self.assertTrue(
            all(row["runtime_error_count"] == 0 for row in matrix["variants"].values())
        )
        bad_cases = json.loads(
            (ROOT / "results/agent_bad_cases_frozen_v2.json").read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(bad_cases["bad_case_count"], 10)
        self.assertFalse(bad_cases["contains_report_text"])


if __name__ == "__main__":
    unittest.main()
