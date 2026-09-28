"""Evaluate fixed task-neutral checkpoints on four fresh dense24 scenes.

This runner is deliberately separate from the registered six-cell evaluator.
It performs no training, Teacher call, checkpoint update, or performance gate.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval
from terminal_marl.training import run_p3_task_neutral_pilot as pilot


STAGE = "P3_TASK_NEUTRAL_FRESH_PHYSICAL_V1"
SCENARIO = "dense24"
ARMS = ("before", "after")
FRESH_CELLS = (
    {"seed": 874601, "scenario": SCENARIO, "split": "new_physical"},
    {"seed": 874602, "scenario": SCENARIO, "split": "new_physical"},
    {"seed": 874603, "scenario": SCENARIO, "split": "new_physical"},
    {"seed": 874604, "scenario": SCENARIO, "split": "new_physical"},
)
POLICY_SEEDS = {
    874601: 874701,
    874602: 874702,
    874603: 874703,
    874604: 874704,
}
BEFORE_CHECKPOINT = neutral_eval.CANDIDATE_CHECKPOINT
BEFORE_CHECKPOINT_SHA256 = neutral_eval.CANDIDATE_CHECKPOINT_SHA256
AFTER_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_critic_warmup_20260928/pilot/pilot.pt"
)
AFTER_CHECKPOINT_SHA256 = (
    "fa0e4218893f51934772e9629e507b86d51024d9f7b4664e6831e31ac135c289"
)
REQUESTS_PER_SCENE = int(neutral_eval.EXPECTED_REQUESTS[SCENARIO])
SCENES_EXPECTED = len(FRESH_CELLS)
MATRIX_EXPECTED = SCENES_EXPECTED * len(ARMS)
REQUESTS_PER_ARM_EXPECTED = SCENES_EXPECTED * REQUESTS_PER_SCENE


class FreshEvaluationError(ValueError):
    """Raised for a fresh-scene binding or runtime contract violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FreshEvaluationError(message)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    pilot._write_json(path, value)


def _sha256(path: str | Path) -> str:
    return pilot._sha256(path)


def _config_snapshot(config: Any) -> dict[str, Any]:
    return neutral_eval._config_snapshot(config)


def _fresh_cells() -> list[dict[str, Any]]:
    return [dict(cell) for cell in FRESH_CELLS]


def _planned_matrix() -> list[dict[str, Any]]:
    return [
        {
            "arm": arm,
            "seed": int(cell["seed"]),
            "scenario": str(cell["scenario"]),
            "split": str(cell["split"]),
            "policy_seed": int(POLICY_SEEDS[int(cell["seed"])]),
        }
        for arm in ARMS
        for cell in FRESH_CELLS
    ]


def _profile_path(arm: str) -> tuple[Path, str]:
    if arm == "before":
        return Path(BEFORE_CHECKPOINT), BEFORE_CHECKPOINT_SHA256
    if arm == "after":
        return AFTER_CHECKPOINT, AFTER_CHECKPOINT_SHA256
    raise FreshEvaluationError(f"unknown fresh evaluation arm: {arm}")


def _load_profile(arm: str, config: Any) -> tuple[Any, dict[str, Any]]:
    """Load and bind one fixed checkpoint against the dense24 config."""

    declared, expected_sha = _profile_path(arm)
    if arm == "before":
        resolved, locator_report = neutral_eval._checkpoint_from_report(
            declared, expected_sha,
        )
    else:
        resolved = neutral_eval._synced_path(declared)
        _require(resolved.is_file(), f"fresh after checkpoint missing: {declared}")
        locator_report = None
    actual_sha = _sha256(resolved)
    _require(actual_sha == expected_sha,
             f"{arm} checkpoint SHA mismatch: {actual_sha} != {expected_sha}")
    policy, payload = rank_eval._load_policy(resolved, config)
    payload_check: dict[str, Any] | None = None
    if arm == "after":
        _require(isinstance(payload, Mapping), "after pilot payload must be a mapping")
        payload_check = pilot._validate_pilot_payload(payload, config)
    profile = {
        "arm": arm,
        "declared_checkpoint": str(declared),
        "resolved_checkpoint": str(resolved),
        "checkpoint_sha256": actual_sha,
        "expected_checkpoint_sha256": expected_sha,
        "locator_report": locator_report,
        "model_digest": rank_eval._model_digest(policy),
        "payload_keys": sorted(payload) if isinstance(payload, Mapping) else [],
        "pilot_payload_check": payload_check,
        "config_profile": "dense24_neutral",
    }
    return policy, profile


def _record_annotations(
    record: Mapping[str, Any], *, arm: str, cell: Mapping[str, Any], config: Any,
) -> dict[str, Any]:
    value = dict(record)
    metrics = neutral_eval._record_metrics(value, smoke=False)
    seed = int(cell["seed"])
    value.update({
        "arm": arm,
        "scenario": str(cell["scenario"]),
        "seed": seed,
        "policy_seed": int(POLICY_SEEDS[seed]),
        "split": str(cell["split"]),
        "fresh_physical_scene": True,
        "old_829xxx_heldout_consumed": False,
        "checkpoint_profile": "before_candidate" if arm == "before" else "after_pilot",
        "checkpoint_sha256_expected": _profile_path(arm)[1],
        "config": _config_snapshot(config),
        "task_priority_enabled": False,
        "low_level_priority_time_weight": 0.0,
        "model_unchanged": value.get("weights_unchanged"),
        "safety": dict(metrics["safety"] or {}),
        "safety_nonzero": bool(metrics["safety_nonzero"]),
        "reward": dict(metrics["reward"]),
        "wait": {
            key: metrics[key]
            for key in ("mean_wait", "p95_wait", "max_wait", "lowSOC_wait", "priority_wait")
        },
        "technical_ok": metrics["technical_ok"],
    })
    return value


def _empty_arm_summary() -> dict[str, Any]:
    return {
        "records": 0,
        "requests": 0,
        "normal": 0,
        "emergency": 0,
        "failed": 0,
        "failed_energy": 0,
        "unresolved": 0,
        "technical_ok": None,
        "full_request_complete": None,
    }


def _compact_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = neutral_eval._record_metrics(record, smoke=False)
    return {
        key: metrics[key]
        for key in (
            "requests", "normal", "emergency", "failed_energy", "failed", "other",
            "unresolved", "mean_wait", "p95_wait", "max_wait", "lowSOC_wait",
            "priority_wait", "reward", "safety_nonzero", "weights_unchanged",
            "technical_ok", "lifecycle_valid",
        )
    }


def _metrics_delta(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> dict[str, float | None] | None:
    if before is None or after is None:
        return None
    delta: dict[str, float | None] = {}
    for key in (
        "normal", "emergency", "failed_energy", "failed", "unresolved",
        "mean_wait", "p95_wait", "max_wait", "lowSOC_wait", "priority_wait",
    ):
        left, right = before.get(key), after.get(key)
        if isinstance(left, (int, float)) and not isinstance(left, bool) and \
                isinstance(right, (int, float)) and not isinstance(right, bool):
            delta[key] = float(right) - float(left)
        else:
            delta[key] = None
    before_reward = before.get("reward")
    after_reward = after.get("reward")
    left = before_reward.get("total") if isinstance(before_reward, Mapping) else None
    right = after_reward.get("total") if isinstance(after_reward, Mapping) else None
    delta["reward_total"] = (
        float(right) - float(left)
        if isinstance(left, (int, float)) and not isinstance(left, bool)
        and isinstance(right, (int, float)) and not isinstance(right, bool)
        else None
    )
    return delta


def _paired_cells(
    before_records: Sequence[Mapping[str, Any]],
    after_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    def index(records: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], Mapping[str, Any]]:
        return {(int(row["seed"]), str(row["scenario"])): row for row in records}

    before = index(before_records)
    after = index(after_records)
    rows: list[dict[str, Any]] = []
    for cell in FRESH_CELLS:
        key = (int(cell["seed"]), str(cell["scenario"]))
        before_metrics = _compact_metrics(before[key]) if key in before else None
        after_metrics = _compact_metrics(after[key]) if key in after else None
        rows.append({
            "seed": key[0],
            "scenario": key[1],
            "split": str(cell["split"]),
            "policy_seed": int(POLICY_SEEDS[key[0]]),
            "before": before_metrics,
            "after": after_metrics,
            "delta_after_minus_before": _metrics_delta(before_metrics, after_metrics),
        })
    return rows


def _arm_summary(
    records: Sequence[Mapping[str, Any]], profile: Mapping[str, Any], policy: Any,
    record_paths: Sequence[str],
) -> dict[str, Any]:
    digest_after = rank_eval._model_digest(policy)
    summary = neutral_eval._aggregate_arm(records, smoke=False) if records else _empty_arm_summary()
    return {
        "records": len(records),
        "record_paths": list(record_paths),
        "summary": summary,
        "checkpoint_profile": dict(profile),
        "model_digest_before_rollouts": profile.get("model_digest"),
        "model_digest_after_rollouts": digest_after,
        "model_unchanged": bool(digest_after == profile.get("model_digest")),
        "weights_unchanged": bool(records) and all(
            row.get("weights_unchanged") is True for row in records
        ) if records else None,
        "safety_nonzero_cells": sum(bool(row.get("safety_nonzero")) for row in records),
    }


def _progress(
    output: Path, *, status: str, processed: int, error: str | None = None,
    check_only: bool,
) -> None:
    value: dict[str, Any] = {
        "stage": STAGE,
        "status": status,
        "phase": "preflight" if check_only else ("complete" if status == "COMPLETED" else "evaluation"),
        "records_processed": int(processed),
        "records_expected": MATRIX_EXPECTED,
        "scenes_expected": SCENES_EXPECTED,
        "arms_expected": len(ARMS),
        "check_only": bool(check_only),
        "full_evaluation": bool(not check_only and status == "COMPLETED"),
        "runtime_rollout_calls": 0 if check_only else int(processed),
        "PPO_ALLOWED": False,
        "teacher_called": False,
        "training_or_update_called": False,
        "heldout_consumed": False,
    }
    if error:
        value["error"] = str(error)
    _write_json(output / "progress.json", value)


def _base_report(output: Path, *, check_only: bool) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "status": "STARTING",
        "check_only": bool(check_only),
        "full_evaluation": False,
        "PPO_ALLOWED": False,
        "teacher_called": False,
        "heldout_consumed": False,
        "old_829xxx_heldout_consumed": False,
        "actions": {"training": False, "teacher": False, "checkpoint_update": False},
        "artifacts": {
            "output_dir": str(output),
            "records": str(output / "records"),
            "progress": str(output / "progress.json"),
            "report": str(output / "report.json"),
        },
        "binding": {
            "evaluation_kind": "new_physical_scenes_fixed_checkpoint_eval",
            "cells": _fresh_cells(),
            "matrix": _planned_matrix(),
            "policy_seeds": {str(k): int(v) for k, v in POLICY_SEEDS.items()},
            "matrix_expected": MATRIX_EXPECTED,
            "requests_per_scene": REQUESTS_PER_SCENE,
            "requests_per_arm_expected": REQUESTS_PER_ARM_EXPECTED,
            "old_829xxx_heldout_touched": False,
        },
        "checks": {
            "matrix_expected": MATRIX_EXPECTED,
            "records_processed": 0,
            "checkpoint_profiles_valid": False,
            "config_neutralized": False,
            "safety_nonzero_cells": 0,
            "models_frozen": False,
            "training_or_update_called": False,
            "check_only_zero_runtime": bool(check_only),
        },
        "arms": {},
        "paired_cells": [],
        "records_processed": 0,
        "records_expected": MATRIX_EXPECTED,
        "deviations": [],
        "unresolved": [],
    }


def run(*, output_dir: str | Path, check_only: bool = False) -> dict[str, Any]:
    output = Path(output_dir)
    _require(not output.exists(), f"fresh output required: {output}")
    output.mkdir(parents=True, exist_ok=False)
    _progress(output, status="STARTING", processed=0, check_only=check_only)
    report = _base_report(output, check_only=check_only)
    before_records: list[dict[str, Any]] = []
    after_records: list[dict[str, Any]] = []
    profiles: dict[str, dict[str, Any]] = {}
    policies: dict[str, Any] = {}
    stop_reason: str | None = None
    processed = 0
    try:
        _configs, neutral_configs, _runtime = neutral_eval._build_neutral_inputs()
        config = neutral_configs[SCENARIO]
        _require(getattr(config, "task_priority_enabled", None) is False,
                 "dense24 config task_priority_enabled is not false")
        _require(float(getattr(config, "low_level_priority_time_weight", 1.0)) == 0.0,
                 "dense24 config priority weight is not zero")
        for arm in ARMS:
            policies[arm], profiles[arm] = _load_profile(arm, config)
        report["binding"].update({
            "config": _config_snapshot(config),
            "checkpoint_profiles": profiles,
        })
        report["checks"]["checkpoint_profiles_valid"] = True
        report["checks"]["config_neutralized"] = True
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        stop_reason = str(exc)

    if stop_reason is not None:
        report.update({"status": "FAILED", "stop_reason": stop_reason})
        report["unresolved"] = [stop_reason]
        _write_json(output / "report.json", report)
        _progress(output, status="FAILED", processed=0, error=stop_reason, check_only=check_only)
        return report

    if check_only:
        report.update({
            "status": "CHECK_ONLY_OK",
            "full_evaluation": False,
            "technical_ok": None,
            "arms": {arm: {"records": 0, "summary": _empty_arm_summary()} for arm in ARMS},
            "paired_cells": _paired_cells([], []),
        })
        _write_json(output / "report.json", report)
        _progress(output, status="CHECK_ONLY_OK", processed=0, check_only=True)
        return report

    record_lists = {"before": before_records, "after": after_records}
    record_paths: dict[str, list[str]] = {"before": [], "after": []}
    for arm in ARMS:
        for cell in FRESH_CELLS:
            seed = int(cell["seed"])
            scenario = str(cell["scenario"])
            policy_seed = int(POLICY_SEEDS[seed])
            rollout_output = output / "rollouts" / arm / f"{scenario}_{seed}"
            try:
                record = rank_eval._runtime_record(
                    policy=policies[arm],
                    checkpoint=Path(profiles[arm]["resolved_checkpoint"]),
                    config=neutral_configs[SCENARIO],
                    scenario=scenario,
                    seed=seed,
                    output=rollout_output,
                    smoke=False,
                    policy_seed=policy_seed,
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                record = rank_eval._failure_record(
                    scenario=scenario,
                    seed=seed,
                    checkpoint=Path(profiles[arm]["resolved_checkpoint"]),
                    requests=REQUESTS_PER_SCENE,
                    error=exc,
                    policy_seed=policy_seed,
                )
                record = rank_eval._normalize_runtime_record(
                    record, scenario=scenario, seed=seed, requests=REQUESTS_PER_SCENE,
                )
            record = _record_annotations(
                record, arm=arm, cell=cell, config=neutral_configs[SCENARIO],
            )
            record_path = output / "records" / arm / f"{scenario}_{seed}.json"
            _write_json(record_path, record)
            record_lists[arm].append(record)
            record_paths[arm].append(str(record_path))
            processed += 1
            reason = neutral_eval._record_fatal_reason(record, smoke=False)
            _progress(
                output,
                status="RUNNING" if reason is None else "FAILED",
                processed=processed,
                error=reason,
                check_only=False,
            )
            if reason is not None:
                stop_reason = f"{arm}/{scenario}/{seed}: {reason}"
                break
        if stop_reason is not None:
            break

    arm_reports = {
        arm: _arm_summary(record_lists[arm], profiles[arm], policies[arm], record_paths[arm])
        for arm in ARMS
    }
    technical_ok = (
        stop_reason is None
        and processed == MATRIX_EXPECTED
        and all(arm_reports[arm]["summary"].get("technical_ok") is True for arm in ARMS)
    )
    models_frozen = processed == MATRIX_EXPECTED and all(
        arm_reports[arm]["model_unchanged"] is True
        and arm_reports[arm]["weights_unchanged"] is True
        for arm in ARMS
    )
    report.update({
        "status": "FAILED" if stop_reason else "COMPLETED",
        "full_evaluation": bool(stop_reason is None),
        "technical_ok": bool(technical_ok),
        "arms": arm_reports,
        "paired_cells": _paired_cells(before_records, after_records),
        "records_processed": processed,
        "stop_reason": stop_reason,
        "unresolved": [] if stop_reason is None else [stop_reason],
    })
    report["checks"].update({
        "records_processed": processed,
        "models_frozen": bool(models_frozen),
        "safety_nonzero_cells": sum(
            int(arm_reports[arm]["safety_nonzero_cells"]) for arm in ARMS
        ),
    })
    _write_json(output / "report.json", report)
    _progress(
        output,
        status=report["status"],
        processed=processed,
        error=stop_reason,
        check_only=False,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(output_dir=args.output_dir, check_only=args.check_only)
    except (FreshEvaluationError, FileNotFoundError, ValueError, OSError) as exc:
        print(json.dumps({"stage": STAGE, "status": "ERROR", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({
        "stage": STAGE,
        "status": result.get("status"),
        "records_processed": result.get("records_processed", 0),
        "technical_ok": result.get("technical_ok"),
        "PPO_ALLOWED": False,
    }, sort_keys=True))
    return 0 if result.get("status") in {"CHECK_ONLY_OK", "COMPLETED"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
