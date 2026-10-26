from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_run(input_path: Path, workspace_root: Path) -> dict:
    """Export FinResearch trajectories through official Evo-Harness contracts."""

    try:
        from agent_evolve.contract.workspace import AgentWorkspace
        from agent_evolve.engine.observer import Observer
        from agent_evolve.types import Feedback, Observation, Task, Trajectory
    except ImportError as exc:
        raise RuntimeError(
            "Install optional dependency from A-EVO-Lab/a-evolve release/evo-harness"
        ) from exc

    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line]
    workspace = AgentWorkspace(workspace_root)
    workspace.write_prompt(
        "FinResearch solver harness. Skills may guide planning, retrieval, verification, and recovery; "
        "they must never contain frozen answers, gold evidence, or scorer details."
    )
    observations = []
    for row in rows:
        score = dict(row.get("score") or {})
        steps = list(row.get("traces") or [])
        observations.append(Observation(
            task=Task(
                id=str(row["case_id"]),
                input=str(row.get("question", "")),
                metadata={"dataset_split": "v3", "harness_version": row.get("harness_version")},
            ),
            trajectory=Trajectory(
                task_id=str(row["case_id"]),
                output=str(row.get("answer", "")),
                steps=steps,
            ),
            feedback=Feedback(
                success=bool(score.get("success")),
                score=float(bool(score.get("success"))),
                detail=str(row.get("refusal_reason") or "passed"),
                raw={"metrics": score},
            ),
        ))
    batch_path = Observer(workspace.evolution_dir).collect(observations)
    result = {
        "contract": "official evo-harness AgentWorkspace/Observer",
        "observation_count": len(observations),
        "batch_path": str(batch_path),
        "workspace": str(workspace.root),
        "evolution_executed": False,
    }
    (workspace_root / "bridge_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FinResearch traces to Evo-Harness format")
    parser.add_argument("--input", required=True)
    parser.add_argument("--workspace", default="results/v3/evo_harness_workspace")
    args = parser.parse_args()
    print(json.dumps(export_run(Path(args.input), Path(args.workspace)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
