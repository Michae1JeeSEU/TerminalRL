"""Paired development evaluation of one repaired policy under two arbiters."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.training import evaluate_random_soc_gate_repair_v1 as base
from terminal_marl.training import student_rollout_v4_2 as student


STAGE = "RANDOM_SOC_ARBITRATION_REPAIR_DEVELOPMENT_V1"
SCENARIO = "dense24"
DEVELOPMENT_PROTOCOL = Path(
    "docs/evidence/v4_2/random_soc_arbitration_repair_v1_20260925.md"
)
CANDIDATE_CHECKPOINT_SHA256 = (
    "e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353"
)
SMOKE_SEED = 847000
FULL_SEEDS = (847101, 847102)
SMOKE_MAX_STEPS = 2
OBSERVATION_DIM = 88


build_scenario_config = base.build_scenario_config
_build_config = base._build_config
_load_yaml_protocol = base._load_yaml_protocol
_evaluate_arm = base._evaluate_arm
_new_arm_record = base._new_arm_record
_not_run_arm = base._not_run_arm
_loader_record = base._loader_record
_outcome_fields = base._outcome_fields
_config_snapshot = base._config_snapshot
_aggregate = base._aggregate
_dev_selection_gates = base._dev_selection_gates
_jsonable = base._jsonable
_write_json = base._write_json
load_repaired_checkpoint = base.load_repaired_checkpoint
_policy_weight_digest = base._policy_weight_digest


def _validate_binding(seed: int, checkpoint_sha256: str, *, smoke: bool) -> None:
    allowed = (SMOKE_SEED,) if smoke else FULL_SEEDS
    if int(seed) not in allowed:
        raise ValueError(
            f"seed must be one of {allowed} for {'smoke' if smoke else 'full'} evaluation"
        )
    if str(checkpoint_sha256) != CANDIDATE_CHECKPOINT_SHA256:
        raise ValueError("candidate checkpoint SHA256 mismatch")


def _config_diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    keys = sorted(set(left) | set(right))
    return {
        key: {"baseline": left.get(key), "soc_priority": right.get(key)}
        for key in keys
        if left.get(key) != right.get(key)
    }


def _paired_configs(protocol: Mapping[str, Any]):
    current = _build_config(SCENARIO, protocol)
    baseline = replace(
        current,
        release_arbitration_mode="downstream_priority",
    )
    soc_priority = replace(
        baseline,
        release_arbitration_mode="downstream_soc_priority",
    )
    baseline_snapshot = _config_snapshot(baseline)
    soc_snapshot = _config_snapshot(soc_priority)
    diff = _config_diff(baseline_snapshot, soc_snapshot)
    expected = {
        "release_arbitration_mode": {
            "baseline": "downstream_priority",
            "soc_priority": "downstream_soc_priority",
        }
    }
    if diff != expected:
        raise ValueError(f"arbitration pair changed config fields: {sorted(diff)}")
    if bool(getattr(baseline, "reconcile_late_merge_slots_enabled", False)):
        raise ValueError("arbitration pair requires late-merge reconciliation disabled")
    if bool(getattr(baseline, "teacher_enabled", False)):
        raise ValueError("arbitration pair requires Teacher disabled")
    if str(getattr(baseline, "scheduler_mode", "execution")) != "execution":
        raise ValueError("arbitration pair requires execution scheduler mode")
    return baseline, soc_priority, baseline_snapshot, soc_snapshot, diff


def _smoke_tape(path: Any, expected_steps: int) -> dict[str, Any]:
    if not isinstance(path, str) or not path:
        raise ValueError("smoke rollout returned no action tape")
    tape_path = Path(path)
    if not tape_path.is_file():
        raise ValueError(f"smoke action tape not found: {tape_path}")
    payload = json.loads(tape_path.read_text(encoding="utf-8"))
    rows = payload.get("action_tape", payload.get("records", []))
    if not isinstance(rows, list):
        raise ValueError("smoke action tape rows must be a list")
    if len(rows) != int(expected_steps):
        raise ValueError(
            f"smoke action tape length {len(rows)} does not match steps {expected_steps}"
        )
    return {"parsed": True, "rows": len(rows), "schema_version": payload.get("schema_version")}


def _smoke_arm(
    *,
    label: str,
    checkpoint: Path,
    candidate_sha256: str,
    config: Any,
    seed: int,
    output: Path,
    checkpoint_loader: Any = None,
) -> tuple[dict[str, Any], bool, str | None]:
    record = _new_arm_record(label, checkpoint, config, SCENARIO, seed)
    policy = None
    stop = False
    stop_reason: str | None = None
    try:
        if checkpoint_loader is not None:
            policy, metadata = checkpoint_loader(checkpoint, config)
        else:
            policy, metadata = load_repaired_checkpoint(
                checkpoint, config, expected_sha256=candidate_sha256,
            )
        record["loader"] = _loader_record(metadata)
        record["weight_digest_before"] = _policy_weight_digest(policy)
        steps, summary = student.rollout(
            policy,
            config,
            seed=int(seed),
            max_steps=SMOKE_MAX_STEPS,
            output=output,
            deterministic=True,
            collect_labels=False,
        )
        if not isinstance(summary, Mapping):
            raise TypeError("smoke rollout summary must be a mapping")
        record["summary"] = summary
        if base._summary_has_safety_error(summary):
            raise SafetyInvariantError(
                str(summary.get("safety_error") or "SafetyInvariantError in smoke summary")
            )
        safety = summary.get("safety") or {}
        if isinstance(safety, Mapping) and any(bool(value) for value in safety.values()):
            raise SafetyInvariantError("nonzero safety counter in smoke summary")
        if not isinstance(steps, list) or len(steps) != SMOKE_MAX_STEPS:
            raise ValueError("smoke rollout must return exactly two steps")
        for step in steps:
            world = np.asarray(getattr(step, "world", ()), dtype=np.float32)
            if world.ndim != 2 or world.shape[1] != OBSERVATION_DIM:
                raise ValueError("smoke rollout observation schema is not finite 88D")
            if not np.isfinite(world).all():
                raise ValueError("smoke rollout observation schema is not finite 88D")
            if bool(getattr(step, "teacher_used", False)):
                raise ValueError("smoke rollout used Teacher behavior")
        tape = _smoke_tape(summary.get("action_tape"), len(steps))
        record.update({
            "tape": summary.get("action_tape"),
            "action_tape": summary.get("action_tape"),
            "completed": int(summary.get("completed", 0)),
            **_outcome_fields(summary, int(config.total_requested_uavs)),
            "label_rows": int(summary.get("label_rows", 0)),
            "PPO_ALLOWED": bool(summary.get("PPO_ALLOWED", False)),
            "teacher_enabled": bool(getattr(config, "teacher_enabled", False)),
            "smoke_partial": True,
            "hard_gate_pass": False,
            "gates": {
                "smoke_steps": len(steps) == SMOKE_MAX_STEPS,
                "finite_88d_observation_schema": True,
                "tape_output_parsed": True,
                "teacher_disabled": not bool(getattr(config, "teacher_enabled", False)),
                "no_labels": int(summary.get("label_rows", 0)) == 0,
                "PPO_ALLOWED_false": not bool(summary.get("PPO_ALLOWED", False)),
                "terminal_required": False,
            },
            "diagnostics": {
                "smoke": True,
                "max_steps": SMOKE_MAX_STEPS,
                "terminal_required": False,
                "tape": tape,
            },
            "lifecycle": {
                "replay_exact": None,
                "rows": [],
                "smoke_partial": True,
            },
            "PASS": False,
            "status": "completed",
        })
    except SafetyInvariantError as exc:
        record.update({
            "status": "failed",
            "safety_invariant_error": True,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        })
        stop = True
        stop_reason = "SafetyInvariantError raised during smoke arm"
    except Exception as exc:
        record.update({
            "status": "failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        })
        stop = True
        stop_reason = "unexpected smoke arm error"
    finally:
        if policy is not None and record.get("weight_digest_before") is not None:
            try:
                record["weight_digest_after"] = _policy_weight_digest(policy)
                record["weights_unchanged"] = bool(
                    record["weight_digest_before"] == record["weight_digest_after"]
                )
            except Exception as exc:
                record["weights_unchanged"] = False
                record.setdefault("error", {"type": type(exc).__name__, "message": str(exc)})
                record["status"] = "failed"
                stop = True
                stop_reason = "unexpected policy digest error"
        record["invariants"] = {
            "weights_unchanged": record.get("weights_unchanged") is True,
            "teacher_disabled": record.get("teacher_enabled") is False,
            "label_rows_zero": int(record.get("label_rows", 0)) == 0,
            "PPO_ALLOWED_false": record.get("PPO_ALLOWED") is False,
            "smoke_partial": record.get("smoke_partial") is True,
        }
        required = (
            record.get("weights_unchanged") is True
            and record.get("teacher_enabled") is False
            and int(record.get("label_rows", 0)) == 0
            and record.get("PPO_ALLOWED") is False
        )
        if record.get("status") == "completed" and not required:
            record["status"] = "failed"
            record["error"] = {
                "type": "SmokeInvariantError",
                "message": "smoke student-only invariants failed",
            }
            stop = True
            stop_reason = "smoke student-only invariant failure"
        record["PASS"] = False
    return record, stop, stop_reason


def _full_arm(
    *,
    label: str,
    checkpoint: Path,
    candidate_sha256: str,
    config: Any,
    protocol: Mapping[str, Any],
    seed: int,
    output: Path,
) -> tuple[dict[str, Any], bool, str | None]:
    # Labels deliberately avoid "source" so the reused helper selects the
    # strict repaired-checkpoint loader for both arbitration arms.
    return _evaluate_arm(
        label=label,
        checkpoint=checkpoint,
        candidate_sha256=candidate_sha256,
        config=config,
        protocol=protocol,
        scenario=SCENARIO,
        seed=int(seed),
        output=output,
    )


def _not_run(*, label: str, checkpoint: Path, config: Any, seed: int, reason: str):
    return _not_run_arm(
        label=label,
        checkpoint=checkpoint,
        config=config,
        scenario=SCENARIO,
        seed=int(seed),
        reason=reason,
    )


def run(
    *,
    checkpoint: str | Path,
    checkpoint_sha256: str,
    output: str | Path,
    seed: int,
    smoke: bool = False,
) -> dict[str, Any]:
    _validate_binding(int(seed), checkpoint_sha256, smoke=bool(smoke))
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"pair output must be fresh: {output_path}")
    protocol = _load_yaml_protocol()
    baseline, soc_priority, baseline_snapshot, soc_snapshot, diff = _paired_configs(protocol)
    checkpoint_path = Path(checkpoint)
    output_path.mkdir(parents=True, exist_ok=False)

    arms: list[dict[str, Any]] = []
    specs = (
        ("baseline", baseline),
        ("soc_priority", soc_priority),
    )
    stopped = False
    stopped_after_baseline = False
    stop_reason: str | None = None
    for label, config in specs:
        if stopped:
            record = _not_run(
                label=label,
                checkpoint=checkpoint_path,
                config=config,
                seed=int(seed),
                reason=stop_reason or "previous arm stopped subsequent arm",
            )
            record.update({
                "candidate_sha256": str(checkpoint_sha256),
                "arbitration_mode": str(config.release_arbitration_mode),
                "development_only": True,
                "P3_QUALIFIED": False,
                "PPO_ALLOWED": False,
                "P4_ALLOWED": False,
                "promotable": False,
                "checkpoint_promotable": False,
            })
            arms.append(record)
            continue
        if smoke:
            record, stopped, stop_reason = _smoke_arm(
                label=label,
                checkpoint=checkpoint_path,
                candidate_sha256=str(checkpoint_sha256),
                config=config,
                seed=int(seed),
                output=output_path / label,
            )
        else:
            record, stopped, stop_reason = _full_arm(
                label=label,
                checkpoint=checkpoint_path,
                candidate_sha256=str(checkpoint_sha256),
                config=config,
                protocol=protocol,
                seed=int(seed),
                output=output_path / label,
            )
        record["candidate_sha256"] = str(checkpoint_sha256)
        record["arbitration_mode"] = str(config.release_arbitration_mode)
        record.update({
            "development_only": True,
            "P3_QUALIFIED": False,
            "PPO_ALLOWED": False,
            "P4_ALLOWED": False,
            "promotable": False,
            "checkpoint_promotable": False,
        })
        arms.append(record)
        if len(arms) == 1:
            stopped_after_baseline = bool(stopped)

    errors = [
        record.get("error") or {
            "arm": record.get("label"),
            "reason": record.get("reason", "arm failed"),
        }
        for record in arms
        if record.get("status") in {"failed", "not_run"}
    ]
    safety_failure = any(bool(record.get("safety_invariant_error")) for record in arms)
    if smoke:
        aggregate = {
            "baseline": {"smoke_partial": True},
            "soc_priority": {"smoke_partial": True},
        }
        selection = {
            "development_only": True,
            "selection_report_only": True,
            "smoke_partial": True,
            "hard_gate_pass": False,
            "gates": {"smoke_partial": True, "hard_gate_pass": False},
            "PASS": False,
        }
        status = "SMOKE_FAILED" if errors or safety_failure else "SMOKE_COMPLETED"
    else:
        baseline_metrics, baseline_error = _aggregate(arms[0])
        soc_metrics, soc_error = _aggregate(arms[1])
        gates = _dev_selection_gates(
            baseline_metrics if baseline_metrics is not None else {},
            soc_metrics if soc_metrics is not None else {},
        )
        aggregate = {
            "baseline": baseline_metrics,
            "soc_priority": soc_metrics,
            "baseline_error": baseline_error,
            "soc_priority_error": soc_error,
        }
        selection = {
            "development_only": True,
            "selection_report_only": True,
            "baseline_metrics": baseline_metrics,
            "soc_priority_metrics": soc_metrics,
            "baseline_aggregate_error": baseline_error,
            "soc_priority_aggregate_error": soc_error,
            "gates": gates,
            "PASS": bool(all(gates.values())),
        }
        status = "FAILED" if errors or safety_failure else "COMPLETED"

    result = {
        "stage": STAGE,
        "status": status,
        "smoke": bool(smoke),
        "development_only": True,
        "P3_QUALIFIED": False,
        "PPO_ALLOWED": False,
        "P4_ALLOWED": False,
        "promotable": False,
        "checkpoint_promotable": False,
        "scenario": SCENARIO,
        "seed": int(seed),
        "candidate_sha256": str(checkpoint_sha256),
        "checkpoint_sha256": str(checkpoint_sha256),
        "checkpoint": str(checkpoint_path),
        "development_protocol": str(DEVELOPMENT_PROTOCOL),
        "gate_reference_yaml": str(base.PROTOCOL),
        "horizon_steps": int(round(float(baseline.max_simulation_seconds) / float(baseline.dt))),
        "smoke_max_steps": SMOKE_MAX_STEPS if smoke else None,
        "stopped_after_baseline": stopped_after_baseline,
        "actual_config_snapshots": {
            "baseline": baseline_snapshot,
            "soc_priority": soc_snapshot,
        },
        "config_diff": diff,
        "unique_config_diff": diff,
        "aggregate": aggregate,
        "selection": selection,
        "selection_gates": selection["gates"],
        "records": arms,
        "arms": arms,
        "failure": {
            "safety_invariant_error": safety_failure,
            "arm_errors": errors,
        },
    }
    _write_json(output_path / "pair_summary.json", result)
    return result


evaluate_pair = run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        checkpoint=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        output=args.output,
        seed=args.seed,
        smoke=bool(args.smoke),
    )
    print(json.dumps(_jsonable(result), sort_keys=True))
    return 0 if result.get("status") in {"COMPLETED", "SMOKE_COMPLETED"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
