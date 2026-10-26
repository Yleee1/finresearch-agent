from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def classify(r0: dict, r1: dict, r2: dict) -> tuple[str, str]:
    r0_score = float(r0["multi_hop_complete_evidence_recall"])
    r1_score = float(r1["multi_hop_complete_evidence_recall"])
    r2_score = float(r2["multi_hop_complete_evidence_recall"])
    if r0_score < r1_score:
        return "ranking_error", "BM25 candidate contains useful evidence but lexical rank is too low; reranker fixes it."
    if r1_score < r2_score:
        return "cross_chunk_context", "The hit lacks an adjacent table/header chunk; bounded neighbor expansion fixes it."
    if r2["category"] == "cross_company_compare":
        return "cross_company_incomplete", "One company remains missing after a single retrieval pass."
    if r2["category"] == "table_cross_chunk":
        return "table_candidate_miss", "The required table evidence is absent from the reranked candidate neighborhood."
    return "candidate_generation_miss", "The gold evidence is absent from the final bounded context."


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract evidence-backed FinResearch V2 bad cases")
    parser.add_argument("--results", default="results/retrieval_ablation_frozen_v2.json")
    parser.add_argument("--output", default="results/bad_cases_frozen_v2.json")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    report = json.loads(Path(args.results).read_text(encoding="utf-8"))
    by_variant = {
        variant: {row["case_id"]: row for row in report[variant]["cases"]}
        for variant in ("R0", "R1", "R2")
    }
    unresolved = [
        case_id
        for case_id, row in by_variant["R2"].items()
        if not row["multi_hop_complete_evidence_recall"]
    ]
    fixed = [
        case_id
        for case_id, row in by_variant["R0"].items()
        if not row["multi_hop_complete_evidence_recall"]
        and by_variant["R2"][case_id]["multi_hop_complete_evidence_recall"]
    ]
    selected = []
    semantic_keys = set()
    for case_id in unresolved + fixed:
        match = re.match(r"rv2-(fr\d+|cross-[a-z_]+)", case_id)
        semantic_key = match.group(1) if match else case_id
        if semantic_key in semantic_keys:
            continue
        semantic_keys.add(semantic_key)
        selected.append(case_id)
        if len(selected) >= args.limit:
            break
    cases = []
    for case_id in selected:
        r0 = by_variant["R0"][case_id]
        r1 = by_variant["R1"][case_id]
        r2 = by_variant["R2"][case_id]
        root_cause, recommendation = classify(r0, r1, r2)
        cases.append(
            {
                "case_id": case_id,
                "category": r2["category"],
                "final_variant_passed": bool(r2["multi_hop_complete_evidence_recall"]),
                "r0_slot_coverage": r0["evidence_slot_coverage"],
                "r1_slot_coverage": r1["evidence_slot_coverage"],
                "r2_slot_coverage": r2["evidence_slot_coverage"],
                "r2_retrieved_doc_ids": r2["retrieved_doc_ids"],
                "root_cause": root_cause,
                "recommendation": recommendation,
                "regression_status": "unresolved" if not r2["multi_hop_complete_evidence_recall"] else "fixed_by_r2",
            }
        )
    payload = {
        "source": args.results,
        "selection": "final unresolved cases first, followed by R0 failures fixed by R2",
        "bad_case_count": len(cases),
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"bad_case_count": len(cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
