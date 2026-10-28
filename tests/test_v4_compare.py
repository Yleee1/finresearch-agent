from finresearch.v4_compare import build_comparison


def _report(hit: float) -> dict:
    row = {
        "case_id": "case-1",
        "category": "table_cross_chunk",
        "recall": hit,
        "mrr": hit,
        "ndcg": hit,
        "evidence_slot_coverage": hit,
        "multi_hop_complete_evidence_recall": hit,
    }
    variant = {
        "recall": hit,
        "mrr": hit,
        "ndcg": hit,
        "evidence_slot_coverage": hit,
        "multi_hop_complete_evidence_recall": hit,
        "mean_query_ms": 1.0,
        "cases": [row],
    }
    return {name: variant for name in ("R0", "R1", "R2")}


def test_build_comparison_reports_paired_improvement() -> None:
    comparison = build_comparison(_report(0.0), _report(1.0))

    assert comparison["R1"]["delta"]["multi_hop_complete_evidence_recall"] == 1.0
    assert comparison["R1"]["paired_complete_recall_changes"]["improved_case_ids"] == [
        "case-1"
    ]
    assert (
        comparison["R1"]["categories"]["table_cross_chunk"]["candidate"][
            "evidence_slot_coverage"
        ]
        == 1.0
    )
