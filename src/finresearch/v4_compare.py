from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

METRICS = (
    "recall",
    "mrr",
    "ndcg",
    "evidence_slot_coverage",
    "multi_hop_complete_evidence_recall",
)


def _category_metrics(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    return {
        category: {
            "case_count": len(items),
            **{
                metric: round(sum(float(item[metric]) for item in items) / len(items), 4)
                for metric in METRICS
            },
        }
        for category, items in sorted(grouped.items())
    }


def build_comparison(baseline: dict, candidate: dict) -> dict:
    comparison: dict[str, dict] = {}
    for variant in ("R0", "R1", "R2"):
        before = baseline[variant]
        after = candidate[variant]
        before_cases = {row["case_id"]: row for row in before["cases"]}
        after_cases = {row["case_id"]: row for row in after["cases"]}
        if before_cases.keys() != after_cases.keys():
            raise ValueError(f"{variant}: baseline and candidate case ids differ")

        improved: list[str] = []
        regressed: list[str] = []
        for case_id, old in before_cases.items():
            delta = float(after_cases[case_id]["multi_hop_complete_evidence_recall"]) - float(
                old["multi_hop_complete_evidence_recall"]
            )
            if delta > 0:
                improved.append(case_id)
            elif delta < 0:
                regressed.append(case_id)

        baseline_categories = _category_metrics(before["cases"])
        candidate_categories = _category_metrics(after["cases"])
        category_comparison = {}
        for category in baseline_categories:
            old = baseline_categories[category]
            new = candidate_categories[category]
            category_comparison[category] = {
                "case_count": old["case_count"],
                "baseline": {metric: old[metric] for metric in METRICS},
                "candidate": {metric: new[metric] for metric in METRICS},
                "delta": {
                    metric: round(float(new[metric]) - float(old[metric]), 4)
                    for metric in METRICS
                },
            }

        comparison[variant] = {
            "baseline": {metric: before[metric] for metric in METRICS},
            "candidate": {metric: after[metric] for metric in METRICS},
            "delta": {
                metric: round(float(after[metric]) - float(before[metric]), 4)
                for metric in METRICS
            },
            "mean_query_ms": {
                "baseline": before["mean_query_ms"],
                "candidate": after["mean_query_ms"],
                "delta": round(
                    float(after["mean_query_ms"]) - float(before["mean_query_ms"]), 3
                ),
            },
            "paired_complete_recall_changes": {
                "improved_count": len(improved),
                "regressed_count": len(regressed),
                "improved_case_ids": improved,
                "regressed_case_ids": regressed,
            },
            "categories": category_comparison,
        }
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare V3 and V4 retrieval reports")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    report = build_comparison(baseline, candidate)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
