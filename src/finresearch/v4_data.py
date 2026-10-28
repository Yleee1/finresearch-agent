from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _render_jsonl(rows: list[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    rendered = _render_jsonl(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return _sha256_bytes(rendered.encode("utf-8"))


def _agent_source_id(case_id: str) -> str | None:
    match = re.match(r"^av2-(fa\d{3})(?:-|$)", case_id)
    return match.group(1) if match else None


def migrate_retrieval_cases(rows: list[dict], mapping: dict[str, list[str]]) -> list[dict]:
    migrated: list[dict] = []
    for original in rows:
        row = json.loads(json.dumps(original, ensure_ascii=False))
        sources = row.get("source_v1")
        source_ids = [sources] if isinstance(sources, str) else list(sources or [])
        slots = row["evidence_slots"]
        if len(source_ids) != len(slots):
            raise ValueError(
                f"{row['case_id']}: {len(source_ids)} source ids for {len(slots)} evidence slots"
            )
        for source_id, slot in zip(source_ids, slots):
            if source_id not in mapping:
                raise KeyError(f"missing retrieval mapping for {source_id}")
            slot["acceptable_evidence_sets"] = [list(mapping[source_id])]
        migrated.append(row)
    return migrated


def migrate_agent_cases(rows: list[dict], mapping: dict[str, list[str]]) -> list[dict]:
    migrated: list[dict] = []
    for original in rows:
        row = json.loads(json.dumps(original, ensure_ascii=False))
        source_id = _agent_source_id(row["case_id"])
        if source_id is not None:
            if source_id not in mapping:
                raise KeyError(f"missing agent mapping for {source_id}")
            row["relevant_doc_ids"] = list(mapping[source_id])
        elif row.get("relevant_doc_ids"):
            raise ValueError(f"unmapped non-empty agent gold in {row['case_id']}")
        migrated.append(row)
    return migrated


def _load_documents(paths: list[str | Path]) -> dict[str, dict]:
    documents: dict[str, dict] = {}
    for path in paths:
        for row in _read_jsonl(path):
            doc_id = row["doc_id"]
            if doc_id in documents:
                raise ValueError(f"duplicate document id: {doc_id}")
            documents[doc_id] = row
    return documents


def _all_mapped_ids(mapping: dict) -> set[str]:
    return {
        doc_id
        for section in ("retrieval", "agent")
        for ids in mapping[section].values()
        for doc_id in ids
    }


def _audit_rows(
    mapping: dict,
    retrieval_rows: list[dict],
    agent_rows: list[dict],
    documents: dict[str, dict],
) -> list[dict]:
    questions: dict[tuple[str, str], str] = {}
    old_ids: dict[tuple[str, str], list[str]] = {}
    for row in retrieval_rows:
        source = row.get("source_v1")
        if row.get("variant") != "original" or not isinstance(source, str):
            continue
        questions[("retrieval", source)] = row["question"]
        old_ids[("retrieval", source)] = [
            doc_id
            for slot in row["evidence_slots"]
            for group in slot["acceptable_evidence_sets"]
            for doc_id in group
        ]
    for row in agent_rows:
        source = _agent_source_id(row["case_id"])
        if source and row.get("variant") == "original":
            questions[("agent", source)] = row["question"]
            old_ids[("agent", source)] = list(row["relevant_doc_ids"])

    audit: list[dict] = []
    for section in ("retrieval", "agent"):
        for source_id, new_ids in mapping[section].items():
            evidence = []
            for doc_id in new_ids:
                document = documents[doc_id]
                evidence.append(
                    {
                        "doc_id": doc_id,
                        "chunk_type": document.get("chunk_type", "text"),
                        "page": document.get("page") or document.get("page_start"),
                        "table_id": document.get("table_id"),
                        "excerpt": document.get("text", "")[:240],
                    }
                )
            audit.append(
                {
                    "section": section,
                    "source_id": source_id,
                    "question": questions.get((section, source_id), ""),
                    "old_doc_ids": old_ids.get((section, source_id), []),
                    "new_doc_ids": new_ids,
                    "new_evidence": evidence,
                }
            )
    return audit


def _manifest_entry(path: Path, digest: str, rows: list[dict]) -> dict:
    return {
        "file": path.name,
        "sha256": digest,
        "case_count": len(rows),
        "category_distribution": dict(
            sorted(Counter(row["category"] for row in rows).items())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate V2 evaluation gold ids to the structure-aware V4 corpus"
    )
    parser.add_argument(
        "--retrieval-cases", default="data/eval/frozen_v2/retrieval_frozen_v2.jsonl"
    )
    parser.add_argument(
        "--agent-cases", default="data/eval/frozen_v2/agent_frozen_v2.jsonl"
    )
    parser.add_argument(
        "--mapping", default="data/eval/v4_candidate/gold_mapping_v4.json"
    )
    parser.add_argument("--documents", nargs="+", required=True)
    parser.add_argument("--output-dir", default="data/eval/v4_candidate")
    args = parser.parse_args()

    mapping = _read_json(args.mapping)
    documents = _load_documents(args.documents)
    missing = sorted(_all_mapped_ids(mapping) - set(documents))
    if missing:
        raise ValueError(f"mapped V4 document ids do not exist: {missing}")

    old_retrieval = _read_jsonl(args.retrieval_cases)
    old_agent = _read_jsonl(args.agent_cases)
    retrieval = migrate_retrieval_cases(old_retrieval, mapping["retrieval"])
    agent = migrate_agent_cases(old_agent, mapping["agent"])

    output = Path(args.output_dir)
    retrieval_path = output / "retrieval_v4_candidate.jsonl"
    agent_path = output / "agent_v4_candidate.jsonl"
    audit_path = output / "migration_audit.jsonl"
    retrieval_hash = _write_jsonl(retrieval_path, retrieval)
    agent_hash = _write_jsonl(agent_path, agent)
    audit_hash = _write_jsonl(
        audit_path, _audit_rows(mapping, old_retrieval, old_agent, documents)
    )

    manifest = {
        "status": "candidate_pending_independent_review",
        "note": (
            "Gold semantics and questions are inherited from frozen V2; only document ids "
            "were manually migrated to the V4 corpus. This is a paired development comparison, "
            "not a new independent frozen benchmark."
        ),
        "mapping": {
            "file": Path(args.mapping).name,
            "sha256": _sha256_bytes(Path(args.mapping).read_bytes()),
        },
        "source_cases": {
            "retrieval": {
                "file": str(args.retrieval_cases),
                "sha256": _sha256_bytes(Path(args.retrieval_cases).read_bytes()),
            },
            "agent": {
                "file": str(args.agent_cases),
                "sha256": _sha256_bytes(Path(args.agent_cases).read_bytes()),
            },
        },
        "v4_corpora": [
            {
                "file": str(path),
                "sha256": _sha256_bytes(Path(path).read_bytes()),
            }
            for path in args.documents
        ],
        "outputs": {
            "retrieval": _manifest_entry(retrieval_path, retrieval_hash, retrieval),
            "agent": _manifest_entry(agent_path, agent_hash, agent),
            "audit": {
                "file": audit_path.name,
                "sha256": audit_hash,
                "row_count": len(mapping["retrieval"]) + len(mapping["agent"]),
            },
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
