from finresearch.v4_data import migrate_agent_cases, migrate_retrieval_cases


def test_migrate_retrieval_keeps_slots_separate() -> None:
    rows = [
        {
            "case_id": "cross",
            "source_v1": ["left", "right"],
            "evidence_slots": [
                {"slot_id": "left", "acceptable_evidence_sets": [["old-left"]]},
                {"slot_id": "right", "acceptable_evidence_sets": [["old-right"]]},
            ],
        }
    ]

    migrated = migrate_retrieval_cases(
        rows, {"left": ["new-left"], "right": ["new-right-a", "new-right-b"]}
    )

    assert migrated[0]["evidence_slots"][0]["acceptable_evidence_sets"] == [["new-left"]]
    assert migrated[0]["evidence_slots"][1]["acceptable_evidence_sets"] == [
        ["new-right-a", "new-right-b"]
    ]
    assert rows[0]["evidence_slots"][0]["acceptable_evidence_sets"] == [["old-left"]]


def test_migrate_agent_variants_share_base_gold() -> None:
    rows = [
        {"case_id": "av2-fa001-o", "relevant_doc_ids": ["old"]},
        {"case_id": "av2-fa001-x1", "relevant_doc_ids": ["old"]},
        {"case_id": "av2-u01", "relevant_doc_ids": []},
    ]

    migrated = migrate_agent_cases(rows, {"fa001": ["new-a", "new-b"]})

    assert migrated[0]["relevant_doc_ids"] == ["new-a", "new-b"]
    assert migrated[1]["relevant_doc_ids"] == ["new-a", "new-b"]
    assert migrated[2]["relevant_doc_ids"] == []
