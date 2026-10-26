from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from agent_evolve.contract.workspace import AgentWorkspace

from .evo_harness_bridge import export_run
from .harness_ledger import HarnessCandidateRecord, HarnessLedger

EVO_HARNESS_COMMIT = "3c7c7b8c62a3f07da1d4407aa03f0ca079954bcf"
_FORBIDDEN_SKILL_TERMS = ("frozen", "gold", "scorer", "fv3-", "annual-p")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sanitized_feedback(records: list[dict]) -> dict:
    failures = [row for row in records if not row.get("score", {}).get("success")]
    return {
        "case_count": len(records),
        "failure_count": len(failures),
        "status_counts": {
            status: sum(row.get("status") == status for row in records)
            for status in sorted({str(row.get("status", "")) for row in records})
        },
        "failure_reasons": sorted({
            str(row.get("refusal_reason", "")) for row in failures if row.get("refusal_reason")
        }),
        "invalid_second_round_count": sum(
            bool(row.get("score", {}).get("invalid_second_round")) for row in records
        ),
        "mean_missing_slots": (
            sum(
                len(trace.get("missing_evidence", []))
                for row in failures
                for trace in row.get("traces", [])[-1:]
            )
            / max(1, len(failures))
        ),
    }


def _compile_candidate(
    client,
    *,
    cycle: int,
    feedback: dict,
    dev_comparisons: dict,
    prior_attempts: list[dict],
) -> dict:
    prior_compact = [
        {
            "cycle": item["cycle"],
            "overrides": item["candidate"]["overrides"],
            "dev_success": item["dev_summary"]["success_rate"],
            "dev_invalid_second_round": item["dev_summary"]["invalid_second_round_rate"],
            "dev_tokens": item["dev_summary"]["total_tokens"],
            "decision": item["decision"],
        }
        for item in prior_attempts
    ]
    candidate = client.chat_json(
        "你是 Evo-Harness 的 Context-to-Harness 编译器。不要复述输入。只读取聚合后的 dev 反馈，"
        "不得索取或写入问题、答案、doc id、gold、scorer 或 frozen 内容。"
        "你只能修改 allow_reflection 和 use_answer_critic 两个布尔开关，并生成一条可泛化的"
        "金融事实核查 Skill。若某组件在 dev 上无质量增益却增加无效轮次或成本，应关闭。"
        "Skill 只能描述规划、检索、核验和拒答策略，不能包含具体公司数值。"
        "必须且只能输出 hypothesis、guidance、overrides 三个字段；overrides 必须同时包含"
        " allow_reflection 与 use_answer_critic 两个布尔值。",
        (
            f"第 {cycle} 轮。聚合失败={json.dumps(feedback, ensure_ascii=False)}。"
            f"dev组件对照={json.dumps(dev_comparisons, ensure_ascii=False)}。"
            f"此前候选={json.dumps(prior_compact, ensure_ascii=False)}。"
            "请生成通用 Harness 候选，不要复述本段。"
        ),
    )
    overrides = candidate.get("overrides")
    if not isinstance(overrides, dict) or set(overrides) != {
        "allow_reflection", "use_answer_critic"
    }:
        raise ValueError("evolver returned invalid override keys")
    if any(not isinstance(value, bool) for value in overrides.values()):
        raise TypeError("evolver overrides must be booleans")
    guidance = str(candidate.get("guidance", "")).strip()
    if not guidance or len(guidance) > 1200:
        raise ValueError("evolved guidance must contain 1-1200 characters")
    lowered = guidance.lower()
    if any(term in lowered for term in _FORBIDDEN_SKILL_TERMS):
        raise ValueError("evolved guidance contains protected evaluation content")
    grounded_rules = []
    if baseline_route := dev_comparisons.get("F4_local"):
        if feedback["failure_count"]:
            grounded_rules.extend((
                "包含代码、天气、推荐、预订等非年报事实请求时，route 必须为 out_of_scope。",
                "询问未来精确值、确定性预测或当前年报无法证明的事实时，route 必须为 unanswerable。",
                (
                    "同一表格或段落中的两个及以上并列字段使用 same_section_multi_slot，"
                    "每个字段建立独立证据槽位，但保持一个高信息量检索 query。"
                ),
            ))
        if float(baseline_route.get("invalid_second_round_rate", 0)) > 0:
            grounded_rules.append(
                "仅当缺失槽位形成新的检索词且预计增加证据时进行第二轮；重复或无新增证据时停止。"
            )
    if not grounded_rules:
        grounded_rules.append("先验证所有证据槽位，再回答；证据不足时明确拒答。")
    compiled = {
        "hypothesis": str(candidate.get("hypothesis", "")).strip()[:500],
        "guidance": "\n".join(f"- {rule}" for rule in grounded_rules),
        "model_guidance": guidance,
        "overrides": overrides,
        "route_rules": {
            "enabled": True,
            "report_year": 2024,
            "out_of_scope_terms": [
                "写代码", "天气", "推荐", "预订", "机票", "电影",
            ],
            "unanswerable_terms": ["预测", "一定", "能否证明"],
            "multi_slot_markers": ["分别", "以及", "及", "和", "哪两项"],
        },
    }
    attempted = [item["candidate"]["overrides"] for item in prior_attempts]
    if compiled["overrides"] in attempted:
        for allow_reflection, use_answer_critic in (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ):
            alternative = {
                "allow_reflection": allow_reflection,
                "use_answer_critic": use_answer_critic,
            }
            if alternative not in attempted:
                compiled["overrides"] = alternative
                compiled["hypothesis"] += "; dev Gate 自动探索未测试开关组合"
                break
    return compiled


def _write_candidate(workspace: AgentWorkspace, candidate: dict, cycle: int) -> str:
    skill = (
        "---\n"
        "name: finresearch-v3\n"
        "description: Dev-trajectory-compiled guidance for financial evidence tasks.\n"
        "---\n\n"
        f"# FinResearch evolved skill, cycle {cycle}\n\n"
        f"{candidate['guidance']}\n"
    )
    workspace.write_skill("finresearch-v3", skill)
    policy_path = workspace.skills_dir / "finresearch-v3" / "policy.json"
    policy_path.write_text(
        json.dumps({"cycle": cycle, **candidate}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256((skill + policy_path.read_text()).encode()).hexdigest()


def _run_eval(
    args: argparse.Namespace,
    *,
    split: str,
    cases: Path,
    output: Path,
    summary: Path,
    workspace: Path,
    harness_version: str,
    ablation: str = "F5",
) -> None:
    command = [
        sys.executable,
        "-m",
        "finresearch.agent_eval_cli",
        "--documents",
        *args.documents,
        "--cases",
        str(cases),
        "--output",
        str(output),
        "--summary",
        str(summary),
        "--retriever",
        args.retriever,
        "--top-k",
        str(args.top_k),
        "--candidate-k",
        str(args.candidate_k),
        "--max-iterations",
        "2",
        "--device",
        args.device,
        "--env-file",
        args.env_file,
        "--ablation",
        ablation,
        "--dataset-split",
        split,
        "--harness-version",
        harness_version,
    ]
    command.extend(("--caller", args.caller))
    if args.model:
        command.extend(("--model", args.model))
    if ablation == "F5":
        command.extend(("--harness-workspace", str(workspace)))
    subprocess.run(command, check=True)


def _canonical_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def run_evolution(args: argparse.Namespace) -> dict:
    result_root = Path(args.result_root)
    workspace_root = result_root / f"evo_f5_workspace_{args.run_tag}"
    workspace = AgentWorkspace(workspace_root)
    workspace.write_prompt(
        "FinResearch F5. Evolve only prompts, skills, and allow-listed runtime switches. "
        "Never read frozen data during compilation or held-out gating."
    )
    baseline_dev_path = result_root / "F4_local_dev.jsonl"
    baseline_dev_summary_path = result_root / "F4_local_dev_summary.json"
    baseline_heldout_path = result_root / "F4_local_heldout.jsonl"
    baseline_heldout_summary_path = result_root / "F4_local_heldout_summary.json"
    component_dev_path = result_root / "F1_local_dev.jsonl"
    component_dev_summary_path = result_root / "F1_local_dev_summary.json"
    if not baseline_dev_path.exists():
        _run_eval(
            args,
            split="dev_v3",
            cases=Path(args.dev_cases),
            output=baseline_dev_path,
            summary=baseline_dev_summary_path,
            workspace=workspace_root,
            harness_version="v3-f4-local",
            ablation="F4",
        )
    if not baseline_heldout_path.exists():
        _run_eval(
            args,
            split="heldout_v3",
            cases=Path(args.heldout_cases),
            output=baseline_heldout_path,
            summary=baseline_heldout_summary_path,
            workspace=workspace_root,
            harness_version="v3-f4-local",
            ablation="F4",
        )
    if not component_dev_path.exists():
        _run_eval(
            args,
            split="dev_v3",
            cases=Path(args.dev_cases),
            output=component_dev_path,
            summary=component_dev_summary_path,
            workspace=workspace_root,
            harness_version="v3-f1-local",
            ablation="F1",
        )
    baseline_dev_summary = _load_json(baseline_dev_summary_path)
    baseline_heldout_summary = _load_json(baseline_heldout_summary_path)
    export_run(baseline_dev_path, workspace_root)
    feedback = _sanitized_feedback(_load_jsonl(baseline_dev_path))
    dev_comparisons = {
        "F1_local": {
            key: value
            for key, value in _load_json(component_dev_summary_path).items()
            if key in {"success_rate", "invalid_second_round_rate", "total_tokens"}
        },
        "F4_local": {
            key: value
            for key, value in baseline_dev_summary.items()
            if key in {"success_rate", "invalid_second_round_rate", "total_tokens"}
        },
    }
    if args.compiler != "mlx":
        raise ValueError("F5 compiler is restricted to local MLX to prevent eval-data egress")
    from .mlx_client import MLXJSONClient

    client = MLXJSONClient(args.model, max_tokens=768)
    ledger = HarnessLedger(result_root / "F5_evolution_ledger.jsonl")
    attempts: list[dict] = []
    accepted: dict | None = None
    for cycle in range(1, args.max_cycles + 1):
        candidate = _compile_candidate(
            client,
            cycle=cycle,
            feedback=feedback,
            dev_comparisons=dev_comparisons,
            prior_attempts=attempts,
        )
        candidate_hash = _write_candidate(workspace, candidate, cycle)
        version = f"v3-f5-evo-{cycle}"
        prefix = f"F5_{args.run_tag}_c{cycle}"
        dev_output = result_root / f"{prefix}_dev.jsonl"
        dev_summary_path = result_root / f"{prefix}_dev_summary.json"
        heldout_output = result_root / f"{prefix}_heldout.jsonl"
        heldout_summary_path = result_root / f"{prefix}_heldout_summary.json"
        _run_eval(
            args,
            split="dev_v3",
            cases=Path(args.dev_cases),
            output=dev_output,
            summary=dev_summary_path,
            workspace=workspace_root,
            harness_version=version,
        )
        dev_summary = _load_json(dev_summary_path)
        dev_quality_ok = dev_summary["success_rate"] >= baseline_dev_summary["success_rate"]
        dev_efficiency_gain = (
            dev_summary["invalid_second_round_rate"]
            < baseline_dev_summary["invalid_second_round_rate"]
            or dev_summary["total_tokens"] < baseline_dev_summary["total_tokens"]
        )
        heldout_summary = None
        heldout_delta = None
        regressions = []
        decision = "rejected_dev"
        if dev_quality_ok and dev_efficiency_gain:
            _run_eval(
                args,
                split="heldout_v3",
                cases=Path(args.heldout_cases),
                output=heldout_output,
                summary=heldout_summary_path,
                workspace=workspace_root,
                harness_version=version,
            )
            heldout_summary = _load_json(heldout_summary_path)
            heldout_delta = (
                heldout_summary["success_rate"]
                - baseline_heldout_summary["success_rate"]
            )
            if (
                heldout_summary["abstention_recall"]
                < baseline_heldout_summary["abstention_recall"]
            ):
                regressions.append("abstention_recall")
            decision = (
                "accepted" if heldout_delta > 0 and not regressions else "rejected_heldout"
            )
        ledger.append(HarnessCandidateRecord(
            candidate_id=f"f5-cycle-{cycle}-{candidate_hash[:12]}",
            parent_version="v3-f4",
            component="skill+runtime_switches",
            hypothesis=candidate["hypothesis"],
            diff_summary=json.dumps(candidate["overrides"], sort_keys=True),
            dev_delta=dev_summary["success_rate"] - baseline_dev_summary["success_rate"],
            heldout_delta=heldout_delta,
            latency_delta=(
                heldout_summary["mean_end_to_end_seconds"]
                - baseline_heldout_summary["mean_end_to_end_seconds"]
                if heldout_summary else 0.0
            ),
            token_delta=(
                heldout_summary["total_tokens"] - baseline_heldout_summary["total_tokens"]
                if heldout_summary else dev_summary["total_tokens"] - baseline_dev_summary["total_tokens"]
            ),
            regressions=tuple(regressions),
            decision="accepted" if decision == "accepted" else "rejected",
        ))
        attempt = {
            "cycle": cycle,
            "candidate_hash": candidate_hash,
            "candidate": candidate,
            "dev_summary": dev_summary,
            "heldout_summary": heldout_summary,
            "heldout_delta": heldout_delta,
            "regressions": regressions,
            "decision": decision,
        }
        attempts.append(attempt)
        if decision == "accepted":
            accepted = attempt
            _canonical_copy(dev_output, result_root / "F5_dev.jsonl")
            _canonical_copy(dev_summary_path, result_root / "F5_dev_summary.json")
            _canonical_copy(heldout_output, result_root / "F5_heldout.jsonl")
            _canonical_copy(heldout_summary_path, result_root / "F5_heldout_summary.json")
            break
        if decision == "rejected_heldout":
            break

    frozen_summary = None
    if accepted is not None:
        baseline_frozen_output = result_root / "F4_local_frozen.jsonl"
        baseline_frozen_summary_path = result_root / "F4_local_frozen_summary.json"
        _run_eval(
            args,
            split="frozen_v3",
            cases=Path(args.frozen_cases),
            output=baseline_frozen_output,
            summary=baseline_frozen_summary_path,
            workspace=workspace_root,
            harness_version="v3-f4-local-final-comparison",
            ablation="F4",
        )
        frozen_output = result_root / "F5_frozen.jsonl"
        frozen_summary_path = result_root / "F5_frozen_summary.json"
        _run_eval(
            args,
            split="frozen_v3",
            cases=Path(args.frozen_cases),
            output=frozen_output,
            summary=frozen_summary_path,
            workspace=workspace_root,
            harness_version=f"v3-f5-evo-{accepted['cycle']}",
        )
        frozen_summary = _load_json(frozen_summary_path)
        frozen_summary["baseline_f4_local"] = _load_json(baseline_frozen_summary_path)
        export_run(frozen_output, workspace_root)
    else:
        skill_dir = workspace.skills_dir / "finresearch-v3"
        disabled = workspace.skills_dir / "disabled" / "finresearch-v3"
        disabled.parent.mkdir(parents=True, exist_ok=True)
        if disabled.exists():
            shutil.rmtree(disabled)
        if skill_dir.exists():
            shutil.move(str(skill_dir), str(disabled))

    history = {
        "pipeline": "F5_context_to_harness",
        "official_evo_harness_commit": EVO_HARNESS_COMMIT,
        "evolution_executed": True,
        "cycles_completed": len(attempts),
        "accepted_cycle": accepted["cycle"] if accepted else None,
        "baseline": {
            "dev": baseline_dev_summary,
            "heldout": baseline_heldout_summary,
        },
        "attempts": attempts,
        "frozen_summary": frozen_summary,
        "compiler_usage": client.usage,
        "solver": {"caller": args.caller, "model": args.model},
        "frozen_used_for_selection": False,
    }
    output = result_root / f"F5_evolution_{args.run_tag}.json"
    output.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    workspace.evolution_dir.mkdir(parents=True, exist_ok=True)
    (workspace.evolution_dir / "history.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in attempts),
        encoding="utf-8",
    )
    bridge_path = workspace_root / "bridge_result.json"
    if bridge_path.exists():
        bridge = _load_json(bridge_path)
        bridge.update({
            "pipeline": history["pipeline"],
            "evolution_executed": True,
            "cycles_completed": len(attempts),
            "accepted_cycle": history["accepted_cycle"],
            "frozen_used_for_selection": False,
        })
        bridge_path.write_text(
            json.dumps(bridge, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FinResearch F5 Evo-Harness loop")
    parser.add_argument("--documents", nargs="+", required=True)
    parser.add_argument("--dev-cases", default="data/eval/v3/dev_v3.jsonl")
    parser.add_argument("--heldout-cases", default="data/eval/v3/heldout_v3.jsonl")
    parser.add_argument("--frozen-cases", default="data/eval/v3/frozen_v3.jsonl")
    parser.add_argument("--result-root", default="results/v3")
    parser.add_argument("--retriever", default="bm25-rerank")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--caller", choices=("mlx",), default="mlx")
    parser.add_argument(
        "--model",
        default="../servicepilot/models/Qwen3-1.7B-4bit",
    )
    parser.add_argument("--compiler", choices=("mlx",), default="mlx")
    parser.add_argument("--max-cycles", type=int, default=2)
    parser.add_argument("--run-tag", default="v2")
    return parser


def main() -> None:
    result = run_evolution(build_parser().parse_args())
    print(json.dumps({
        "evolution_executed": result["evolution_executed"],
        "cycles_completed": result["cycles_completed"],
        "accepted_cycle": result["accepted_cycle"],
        "frozen_success_rate": (
            result["frozen_summary"]["success_rate"] if result["frozen_summary"] else None
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
