from __future__ import annotations

import argparse
import fcntl
import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from .agent import ResearchAgent
from .agent_evaluation import (
    aggregate_agent_scores,
    load_agent_cases,
    score_agent_result,
)
from .bm25 import BM25Retriever
from .budgets import ResearchBudget
from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .io import load_documents
from .llm_client import DeepSeekResearchLLM, OpenAICompatibleClient
from .planner import StructuredPlanner
from .procedural_memory import ProceduralMemoryStore
from .reranker import RerankRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated FinResearch Agent evaluation")
    parser.add_argument("--documents", nargs="+", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retriever",
        choices=("bm25", "bm25-rerank", "hybrid-rerank"),
        default="bm25-rerank",
    )
    parser.add_argument(
        "--ablation",
        choices=(
            "R0", "R1", "R2", "R3", "R4", "R5",
            "F0", "F1", "F2", "F3", "F4", "F5",
        ),
        default="R5",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--max-sub-queries", type=int, default=4)
    parser.add_argument("--max-evidence-chunks", type=int, default=12)
    parser.add_argument("--max-context-tokens", type=int, default=8000)
    parser.add_argument("--max-llm-calls", type=int, default=8)
    parser.add_argument("--max-neighbor-chunks", type=int, default=2)
    parser.add_argument("--no-neighbor-expansion", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--caller", choices=("deepseek", "mlx"), default="deepseek")
    parser.add_argument("--model")
    parser.add_argument("--dataset-split", default="interactive")
    parser.add_argument("--harness-version", default="v2-compat")
    parser.add_argument("--harness-workspace")
    return parser


class _LegacyPlanner:
    def __init__(self, llm) -> None:
        self.llm = llm

    def plan(self, question: str):
        return self.llm.plan(question)


class _EvolvedPlanner:
    """Execute the allow-listed route rules compiled into an F5 workspace."""

    def __init__(self, planner: StructuredPlanner, rules: dict) -> None:
        self.planner = planner
        self.rules = rules

    def plan(self, question: str):
        plan = self.planner.plan(question)
        if not self.rules.get("enabled"):
            return plan
        lowered = question.lower()
        if any(str(term).lower() in lowered for term in self.rules["out_of_scope_terms"]):
            return replace(
                plan,
                answerability="unanswerable",
                route="out_of_scope",
                retrieval_queries=(),
            )
        years = [int(value) for value in re.findall(r"(?:19|20)\d{2}", question)]
        future_year = any(year > int(self.rules["report_year"]) for year in years)
        if future_year or any(
            str(term).lower() in lowered for term in self.rules["unanswerable_terms"]
        ):
            return replace(
                plan,
                answerability="unanswerable",
                route="unanswerable",
                retrieval_queries=(),
            )
        if any(str(term) in question for term in self.rules["multi_slot_markers"]):
            return replace(plan, route="same_section_multi_slot")
        return plan


def _load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    deduplicated = {record["case_id"]: record for record in records}
    return list(deduplicated.values())


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = Path(f"{output_path}.lock").open("a+", encoding="utf-8")  # noqa: SIM115
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another evaluation is writing {output_path}") from exc
    existing = _load_existing(output_path)
    if existing and not args.resume:
        raise FileExistsError(f"output already exists: {output_path}; use --resume")
    completed_ids = {item["case_id"] for item in existing}

    documents = [
        document
        for path in args.documents
        for document in load_documents(path)
    ]
    bm25 = BM25Retriever(documents)
    if args.retriever == "hybrid-rerank":
        dense = DenseRetriever(documents, device=args.device)
        base = HybridRetriever(bm25, dense, candidate_k=args.candidate_k)
    else:
        base = bm25
    retriever = (
        base
        if args.retriever == "bm25"
        else RerankRetriever(base, device=args.device, candidate_k=args.candidate_k)
    )
    ablation = {
        "R0": {
            "decompose_queries": False,
            "use_evidence_slots": False,
            "require_verification": False,
            "use_retrieval_critic": False,
            "allow_reflection": False,
            "use_answer_critic": False,
            "expand_neighbors": False,
        },
        "R1": {
            "decompose_queries": False,
            "use_evidence_slots": False,
            "require_verification": False,
            "use_retrieval_critic": False,
            "allow_reflection": False,
            "use_answer_critic": False,
            "expand_neighbors": False,
        },
        "R2": {
            "decompose_queries": False,
            "use_evidence_slots": False,
            "require_verification": False,
            "use_retrieval_critic": False,
            "allow_reflection": False,
            "use_answer_critic": False,
            "expand_neighbors": True,
        },
        "R3": {
            "decompose_queries": True,
            "use_evidence_slots": True,
            "require_verification": True,
            "use_retrieval_critic": True,
            "allow_reflection": False,
            "use_answer_critic": False,
            "expand_neighbors": True,
        },
        "R4": {
            "decompose_queries": True,
            "use_evidence_slots": True,
            "require_verification": True,
            "use_retrieval_critic": True,
            "allow_reflection": True,
            "use_answer_critic": False,
            "expand_neighbors": True,
        },
        "R5": {
            "decompose_queries": True,
            "use_evidence_slots": True,
            "require_verification": True,
            "use_retrieval_critic": True,
            "allow_reflection": True,
            "use_answer_critic": True,
            "expand_neighbors": True,
        },
        "F0": {
            "decompose_queries": True,
            "use_evidence_slots": True,
            "require_verification": True,
            "use_retrieval_critic": True,
            "allow_reflection": False,
            "use_answer_critic": False,
            "expand_neighbors": True,
        },
        "F1": {
            "decompose_queries": True,
            "use_evidence_slots": True,
            "require_verification": True,
            "use_retrieval_critic": True,
            "allow_reflection": False,
            "use_answer_critic": False,
            "expand_neighbors": True,
        },
        "F2": {
            "decompose_queries": True,
            "use_evidence_slots": True,
            "require_verification": True,
            "use_retrieval_critic": True,
            "allow_reflection": True,
            "use_answer_critic": False,
            "expand_neighbors": True,
        },
        "F3": {
            "decompose_queries": True,
            "use_evidence_slots": True,
            "require_verification": True,
            "use_retrieval_critic": True,
            "allow_reflection": True,
            "use_answer_critic": False,
            "expand_neighbors": True,
        },
        "F4": {
            "decompose_queries": True,
            "use_evidence_slots": True,
            "require_verification": True,
            "use_retrieval_critic": True,
            "allow_reflection": True,
            "use_answer_critic": True,
            "expand_neighbors": True,
        },
        "F5": {
            "decompose_queries": True,
            "use_evidence_slots": True,
            "require_verification": True,
            "use_retrieval_critic": True,
            "allow_reflection": True,
            "use_answer_critic": True,
            "expand_neighbors": True,
        },
    }[args.ablation]
    harness_guidance = ""
    harness_route_rules = None
    if args.ablation == "F5":
        if not args.harness_workspace:
            raise ValueError("F5 requires --harness-workspace")
        skill_dir = Path(args.harness_workspace) / "skills" / "finresearch-v3"
        policy_path = skill_dir / "policy.json"
        skill_path = skill_dir / "SKILL.md"
        if not policy_path.exists() or not skill_path.exists():
            raise FileNotFoundError("F5 workspace requires policy.json and SKILL.md")
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        allowed_overrides = {"allow_reflection", "use_answer_critic"}
        unknown = set(policy.get("overrides", {})) - allowed_overrides
        if unknown:
            raise ValueError(f"F5 policy contains protected/unknown overrides: {sorted(unknown)}")
        for key, value in policy.get("overrides", {}).items():
            if not isinstance(value, bool):
                raise TypeError(f"F5 override {key} must be boolean")
            ablation[key] = value
        harness_route_rules = policy.get("route_rules")
        if not isinstance(harness_route_rules, dict):
            raise TypeError("F5 policy requires route_rules")
        required_route_fields = {
            "enabled", "report_year", "out_of_scope_terms",
            "unanswerable_terms", "multi_slot_markers",
        }
        if set(harness_route_rules) != required_route_fields:
            raise ValueError("F5 route_rules has protected/unknown fields")
        allowed_terms = {
            "out_of_scope_terms": {"写代码", "天气", "推荐", "预订", "机票", "电影"},
            "unanswerable_terms": {"预测", "一定", "能否证明"},
            "multi_slot_markers": {"分别", "以及", "及", "和", "哪两项"},
        }
        if harness_route_rules.get("report_year") != 2024:
            raise ValueError("F5 route_rules cannot change the corpus report year")
        for field, allowed in allowed_terms.items():
            values = harness_route_rules.get(field)
            if not isinstance(values, list) or not set(values) <= allowed:
                raise ValueError(f"F5 route_rules contains non-allow-listed {field}")
        harness_guidance = skill_path.read_text(encoding="utf-8")
    if args.caller == "mlx":
        if not args.model:
            raise ValueError("--caller mlx requires --model")
        from .mlx_client import MLXJSONClient

        client = MLXJSONClient(args.model)
    else:
        client = OpenAICompatibleClient.from_env(args.env_file)
    research_llm = DeepSeekResearchLLM(client, harness_guidance=harness_guidance)
    procedural_memory = None
    if args.ablation in {"F3", "F4", "F5"}:
        procedural_memory = ProceduralMemoryStore()
        for trace_id in ("dev-pattern-1", "dev-pattern-2"):
            procedural_memory.add_candidate(
                memory_type="procedural",
                task_signature="same_section_multi_slot",
                content="同一表格多指标应保持一次查询，并分别建立证据槽位。",
                applies_to="same_section_multi_slot",
                source_trace_id=trace_id,
            )
    agent = ResearchAgent(
        retriever,
        research_llm,
        top_k=args.top_k,
        max_iterations=args.max_iterations,
        budget=ResearchBudget(
            max_retrieval_rounds=args.max_iterations,
            max_sub_queries=args.max_sub_queries,
            max_evidence_chunks=args.max_evidence_chunks,
            max_context_tokens=args.max_context_tokens,
            max_llm_calls=args.max_llm_calls,
            max_neighbor_chunks=args.max_neighbor_chunks,
        ),
        **{
            **ablation,
            "expand_neighbors": ablation["expand_neighbors"]
            and not args.no_neighbor_expansion,
        },
        structured_planner=(
            StructuredPlanner(_LegacyPlanner(research_llm))
            if args.ablation == "F0"
            else _EvolvedPlanner(StructuredPlanner(research_llm), harness_route_rules)
            if args.ablation == "F5"
            else None
        ),
        procedural_memory=procedural_memory,
        harness_version=args.harness_version,
        dataset_split=args.dataset_split,
    )

    cases = [case for case in load_agent_cases(args.cases) if case.case_id not in completed_ids]
    if args.limit is not None:
        cases = cases[: args.limit]
    mode = "a" if existing else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for case in cases:
            before = dict(client.usage)
            started = perf_counter()
            try:
                result = agent.run(case.question)
                compact = {
                    "case_id": case.case_id,
                    "question": case.question,
                    "route": result.route,
                    "status": result.status,
                    "answer": result.answer,
                    "citations": [asdict(item) for item in result.citations],
                    "iterations": result.iterations,
                    "traces": [asdict(item) for item in result.traces],
                    "refusal_reason": result.refusal_reason,
                    "claim_ledger": [asdict(item) for item in result.claim_ledger],
                    "llm_calls": result.llm_calls,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "harness_version": result.harness_version,
                    "memory_hit_ids": list(result.memory_hit_ids),
                    "stop_reason": result.stop_reason,
                }
            except Exception as exc:  # noqa: BLE001 - preserve the rest of a paid batch
                compact = {
                    "case_id": case.case_id,
                    "question": case.question,
                    "route": "",
                    "status": "error",
                    "answer": "",
                    "citations": [],
                    "iterations": 0,
                    "traces": [],
                    "refusal_reason": "",
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
            seconds = perf_counter() - started
            score = score_agent_result(case, compact)
            usage = {
                field: client.usage[field] - before[field]
                for field in client.usage
            }
            record = {
                **compact,
                "score": score,
                "usage": usage,
                "end_to_end_seconds": round(seconds, 4),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"case_id": case.case_id, **score}, ensure_ascii=False))

    all_records = _load_existing(output_path)
    case_by_id = {case.case_id: case for case in load_agent_cases(args.cases)}
    scores = [
        score_agent_result(case_by_id[record["case_id"]], record)
        for record in all_records
    ]
    summary = aggregate_agent_scores(scores)
    summary.update(
        {
            "retriever": args.retriever,
            "ablation": args.ablation,
            "total_llm_calls": sum(record["usage"]["calls"] for record in all_records),
            "total_tokens": sum(record["usage"]["total_tokens"] for record in all_records),
            "mean_end_to_end_seconds": sum(record["end_to_end_seconds"] for record in all_records)
            / len(all_records),
        }
    )
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if procedural_memory is not None:
        procedural_memory.close()


if __name__ == "__main__":
    main()
