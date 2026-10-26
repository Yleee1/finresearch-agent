from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_key(case_id: str) -> str:
    match = re.match(r"av2-(fa\d+|fu\d+|u\d+|o\d+)", case_id)
    return match.group(1) if match else case_id


def _root_cause(r3: dict, r4: dict, r5: dict) -> tuple[str, str]:
    s3, s4, s5 = r3["score"], r4["score"], r5["score"]
    if s4["invalid_second_round"]:
        return (
            "reflection_no_incremental_evidence",
            "Tighten follow-up queries or skip reflection when the first-pass candidate set is saturated.",
        )
    if s4["success"] and not s5["success"]:
        return (
            "answer_critic_over_abstention",
            "Calibrate deterministic and model critics separately; retain the rejected claim ledger.",
        )
    if not s5["route_correct"]:
        return (
            "planner_route_mismatch",
            "Add route-specific hard negatives and score planning independently from answer quality.",
        )
    if not s5["expected_abstain"] and s5["predicted_abstain"]:
        return (
            "false_abstention",
            "Inspect slot matching and verifier thresholds before increasing retrieval rounds.",
        )
    if s5["claim_support_rate"] is not None and s5["claim_support_rate"] < 1:
        return (
            "claim_citation_mismatch",
            "Require one atomic claim per value and an exact citation binding before generation completes.",
        )
    if s5["citation_recall"] < 1:
        return (
            "citation_coverage_gap",
            "Preserve all evidence sides needed by comparison claims and audit each citation id.",
        )
    if s3["success"] and not s4["success"]:
        return (
            "reflection_regression",
            "Gate the second round on measurable missing slots instead of model uncertainty alone.",
        )
    return (
        "unresolved_end_to_end_failure",
        "Inspect the saved per-round trace and assign the earliest failing module.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the public FinResearch R3-R5 report")
    parser.add_argument("--input-dir", default="results/agent_ablation_frozen_v2_deepseek")
    parser.add_argument("--matrix-output", default="results/agent_matrix_frozen_v2.json")
    parser.add_argument("--bad-cases-output", default="results/agent_bad_cases_frozen_v2.json")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    source = Path(args.input_dir)
    variants: dict[str, dict] = {}
    records: dict[str, dict[str, dict]] = {}
    for variant in ("r3", "r4", "r5"):
        summary_path = source / f"{variant}_summary.json"
        records_path = source / f"{variant}_records.jsonl"
        rows = _load_jsonl(records_path)
        summary = _load_json(summary_path)
        if len(rows) != summary["case_count"]:
            raise ValueError(f"{variant} record count does not match summary")
        latencies = sorted(float(row["end_to_end_seconds"]) for row in rows)
        variants[variant.upper()] = {
            **summary,
            "mean_iterations": sum(int(row["iterations"]) for row in rows) / len(rows),
            "p95_end_to_end_seconds": latencies[math.ceil(0.95 * len(latencies)) - 1],
            "mean_llm_calls": summary["total_llm_calls"] / len(rows),
            "mean_tokens": summary["total_tokens"] / len(rows),
            "answered_count": sum(row["status"] == "answered" for row in rows),
            "abstained_count": sum(row["status"] == "abstained" for row in rows),
            "records_sha256": _sha256(records_path),
            "summary_sha256": _sha256(summary_path),
            "runtime_error_count": sum(row.get("status") == "error" for row in rows),
        }
        records[variant.upper()] = {row["case_id"]: row for row in rows}

    r3, r4, r5 = (variants[name] for name in ("R3", "R4", "R5"))
    matrix = {
        "protocol": {
            "cases": "data/eval/frozen_v2/agent_frozen_v2.jsonl",
            "case_count": 74,
            "retriever": "BM25 Top-20 + Cross-Encoder Top-5 + bounded neighbor expansion",
            "model": "deepseek-v4-flash",
            "temperature": 0,
            "max_retrieval_rounds": 2,
            "review_status": "independent_second_review_completed_user_attested",
            "authorization_scope": "One-time DeepSeek R3-R5 frozen evaluation; completed and not reusable for later runs.",
            "privacy": "Only evaluation questions and bounded retrieved evidence blocks were sent; PDFs, credentials, and private data were not uploaded.",
        },
        "variants": variants,
        "deltas": {
            "R4_minus_R3": {
                "success_rate": r4["success_rate"] - r3["success_rate"],
                "citation_hit_rate": r4["citation_hit_rate"] - r3["citation_hit_rate"],
                "total_tokens": r4["total_tokens"] - r3["total_tokens"],
                "mean_end_to_end_seconds": r4["mean_end_to_end_seconds"]
                - r3["mean_end_to_end_seconds"],
            },
            "R5_minus_R4": {
                "success_rate": r5["success_rate"] - r4["success_rate"],
                "citation_hit_rate": r5["citation_hit_rate"] - r4["citation_hit_rate"],
                "abstention_recall": r5["abstention_recall"] - r4["abstention_recall"],
                "total_tokens": r5["total_tokens"] - r4["total_tokens"],
                "mean_end_to_end_seconds": r5["mean_end_to_end_seconds"]
                - r4["mean_end_to_end_seconds"],
            },
        },
        "aggregate_usage": {
            "calls": sum(variants[name]["total_llm_calls"] for name in variants),
            "tokens": sum(variants[name]["total_tokens"] for name in variants),
        },
        "pricing_note": {
            "observed_prompt_tokens": 1_348_266,
            "observed_completion_tokens": 103_547,
            "estimated_usd_off_peak_range": [0.0662, 0.2644],
            "assumption": "2026-10-11 DeepSeek Flash off-peak rates; range spans all input cache-hit to all cache-miss. Actual invoice controls.",
            "source": "https://api-docs.deepseek.com/quick_start/pricing/",
        },
    }
    matrix_path = Path(args.matrix_output)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    candidates = []
    for case_id, row5 in records["R5"].items():
        row3, row4 = records["R3"][case_id], records["R4"][case_id]
        if row5["score"]["success"]:
            continue
        cause, recommendation = _root_cause(row3, row4, row5)
        priority = (
            0 if row4["score"]["success"] and not row5["score"]["success"] else
            1 if row4["score"]["invalid_second_round"] else
            2 if not row5["score"]["route_correct"] else 3
        )
        candidates.append((priority, case_id, cause, recommendation))
    selected = []
    semantic_keys = set()
    causes = set()
    for _, case_id, cause, recommendation in sorted(candidates):
        key = _semantic_key(case_id)
        if key in semantic_keys:
            continue
        # Prefer cause diversity for the first pass, then fill remaining slots.
        if cause in causes and len(selected) < 5:
            continue
        semantic_keys.add(key)
        causes.add(cause)
        selected.append((case_id, cause, recommendation))
        if len(selected) >= args.limit:
            break
    if len(selected) < args.limit:
        for _, case_id, cause, recommendation in sorted(candidates):
            key = _semantic_key(case_id)
            if key in semantic_keys:
                continue
            semantic_keys.add(key)
            selected.append((case_id, cause, recommendation))
            if len(selected) >= args.limit:
                break

    bad_cases = []
    for case_id, cause, recommendation in selected:
        rows = {name: records[name][case_id] for name in ("R3", "R4", "R5")}
        bad_cases.append(
            {
                "case_id": case_id,
                "root_cause": cause,
                "recommendation": recommendation,
                "outcomes": {
                    name: {
                        "route": row["route"],
                        "status": row["status"],
                        "iterations": row["iterations"],
                        "success": row["score"]["success"],
                        "route_correct": row["score"]["route_correct"],
                        "citation_recall": row["score"]["citation_recall"],
                        "claim_support_rate": row["score"]["claim_support_rate"],
                        "invalid_second_round": row["score"]["invalid_second_round"],
                    }
                    for name, row in rows.items()
                },
            }
        )
    bad_payload = {
        "source": str(matrix_path),
        "bad_case_count": len(bad_cases),
        "contains_report_text": False,
        "cases": bad_cases,
    }
    bad_path = Path(args.bad_cases_output)
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text(
        json.dumps(bad_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"matrix": str(matrix_path), "bad_cases": len(bad_cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
