"""Evaluate the ordinary calibrated E04 gate repair on the three failed scenes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from terminal_marl.training import diagnose_p3_bounded_gate_v1 as bounded
from terminal_marl.training import student_rollout_v4_2 as student_rollout


REQUESTS = 24
RUN_INDICES = (0, 1, 2)
STAGE = "P3_E04_GATE_COVERAGE_REPAIR_SCENE_COMPARISON_V1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    return bounded._jsonable(value)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def evaluate(*, checkpoint: Path, run_index: int, output: Path, max_steps: int) -> dict[str, Any]:
    if int(run_index) not in RUN_INDICES:
        raise ValueError(f"run_index must be one of {RUN_INDICES}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if isinstance(max_steps, bool) or int(max_steps) < 1:
        raise ValueError("max_steps must be positive")

    config = bounded.build_config()
    binding = bounded.source_binding(int(run_index))
    source_checkpoint = Path(binding["source_checkpoint"])
    policy, payload = bounded._load_calibrated_source(checkpoint, config)
    weight_before = bounded._state_digest(policy.model.state_dict())
    rollout_path = output / "rollout"
    steps: list[Any] = []
    rollout_summary: dict[str, Any] = {}
    exception: dict[str, str] | None = None
    try:
        steps, rollout_summary = student_rollout.rollout(
            policy,
            config,
            seed=int(binding["failure_seed"]),
            max_steps=int(max_steps),
            output=rollout_path,
            deterministic=False,
            collect_labels=False,
        )
    except Exception as exc:
        exception = {"type": type(exc).__name__, "message": str(exc)}
    weight_after = bounded._state_digest(policy.model.state_dict())
    outcomes = bounded._outcome_summary(rollout_summary, REQUESTS)
    result = {
        "stage": STAGE,
        "status": "failed" if exception is not None else "completed",
        "source": "E04 gate coverage repair",
        "comparison_scope": "same_run_failed_scenes_only",
        "policy_role": "coverage_repair_not_old_run_final_policy",
        "run_index": int(run_index),
        "scenario": "dense24",
        "seed": int(binding["failure_seed"]),
        "max_steps": int(max_steps),
        "steps": int(len(steps)),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_failed_run_checkpoint": str(source_checkpoint),
        "source_failed_run_checkpoint_sha256": str(binding["source_checkpoint_sha256"]),
        "source_manifest": str(binding["manifest"]),
        "source_payload_stage": payload.get("stage"),
        "loader": {
            "class": "TemperatureCalibratedGateResidualAdapterStudentPPOV42",
            "optimizer_restored": False,
            "probability_floor": None,
            "bounded_adapter": False,
        },
        "config": {
            "v4_2_rejected_release_hold_mode": str(config.v4_2_rejected_release_hold_mode),
            "action_schema_version": str(config.action_schema_version),
            "observation_schema_version": str(config.observation_schema_version),
            "teacher_enabled": bool(config.teacher_enabled),
            "runtime_proposal_shortlist_k": int(config.runtime_proposal_shortlist_k),
        },
        "rollout": {
            "action_tape": rollout_summary.get("action_tape"),
            "terminal": bool(rollout_summary.get("terminal", False)),
            "safety_error": rollout_summary.get("safety_error"),
            "summary_safety": rollout_summary.get("safety", {}),
            "training_called": False,
            "ppo_called": False,
            "teacher_used": False,
        },
        "outcomes": outcomes,
        "exception": exception,
        "weight_digest_before": weight_before,
        "weight_digest_after": weight_after,
        "weights_unchanged": bool(weight_before == weight_after),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "summary.json", result)
    return result


def main() -> int:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-index", type=int, choices=RUN_INDICES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1800)
    args = parser.parse_args()
    result = evaluate(
        checkpoint=args.checkpoint,
        run_index=args.run_index,
        output=args.output,
        max_steps=args.max_steps,
    )
    print(json.dumps({
        "status": result["status"],
        "run_index": result["run_index"],
        "steps": result["steps"],
        "outcomes": result["outcomes"],
        "exception": result["exception"],
        "weights_unchanged": result["weights_unchanged"],
    }, sort_keys=True))
    return 0 if result["exception"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
