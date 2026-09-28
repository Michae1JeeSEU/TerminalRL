"""Fixed policy-seed repeats for the task-neutral dense development cells."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval
from terminal_marl.training import run_p3_task_neutral_pilot as pilot_eval

STAGE = "P3_TASK_NEUTRAL_POLICY_SEED_REPEATS_V1"
ARMS = ("before", "after")
ENV_SEEDS = (873103, 873106)
POLICY_SEEDS = (874301, 874302, 874303, 874304)
BEFORE_CHECKPOINT = neutral_eval.CANDIDATE_CHECKPOINT
BEFORE_CHECKPOINT_SHA256 = neutral_eval.CANDIDATE_CHECKPOINT_SHA256
AFTER_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_ppo_pilot_20260928/full/pilot.pt"
)
AFTER_CHECKPOINT_SHA256 = "1f7ca8ad1fa4a041b3e62017c56dc47a7c418865613a6241c86fa18763d0fb81"


class TaskNeutralRepeatsError(ValueError):
    pass


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise TaskNeutralRepeatsError(message)


def repeat_cells() -> list[dict[str, Any]]:
    cells = [
        dict(cell) for cell in neutral_eval._evaluation_cells()
        if str(cell["scenario"]) == "dense24" and int(cell["seed"]) in ENV_SEEDS
    ]
    cells.sort(key=lambda cell: ENV_SEEDS.index(int(cell["seed"])))
    _require(
        [(int(cell["seed"]), str(cell["scenario"])) for cell in cells]
        == [(873103, "dense24"), (873106, "dense24")],
        "registered dense repeat cells changed",
    )
    return cells


def repeat_matrix(cells: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    selected = repeat_cells() if cells is None else [dict(cell) for cell in cells]
    matrix = [
        {"arm": arm, "scenario": str(cell["scenario"]), "env_seed": int(cell["seed"]),
         "policy_seed": int(policy_seed)}
        for arm in ARMS for cell in selected for policy_seed in POLICY_SEEDS
    ]
    keys = {(row["arm"], row["env_seed"], row["policy_seed"]) for row in matrix}
    _require(len(matrix) == 16 and len(keys) == 16, "repeat matrix must contain 16 unique rollouts")
    return matrix


def _metric_view(record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = neutral_eval._record_metrics(record)
    value = {
        key: metrics[key]
        for key in (
            "requests", "normal", "emergency", "failed", "failed_energy", "unresolved",
            "mean_wait", "p95_wait", "max_wait", "lowSOC_wait", "reward", "safety",
            "technical_ok",
        )
    }
    value["reward_total"] = value["reward"].get("total")
    return value


def _binding(checkpoint: Path, expected_sha: str, config: Any, label: str,
             *, validate_pilot_payload: bool = False) -> dict[str, Any]:
    resolved, _ = neutral_eval._checkpoint_from_report(checkpoint, expected_sha)
    policy, payload = rank_eval._load_policy(resolved, config)
    value = {
        "label": label, "declared_checkpoint": str(checkpoint),
        "resolved_checkpoint": str(resolved), "checkpoint_sha256": neutral_eval._sha256(resolved),
        "expected_checkpoint_sha256": expected_sha,
        "model_digest": rank_eval._model_digest(policy),
    }
    if validate_pilot_payload:
        _require(isinstance(payload, Mapping), "after checkpoint payload metadata is missing")
        value["pilot_payload_check"] = pilot_eval._validate_pilot_payload(payload, config)
    return value


def _decorate(record: Mapping[str, Any], item: Mapping[str, Any], checkpoint: Path,
              expected_sha: str, config: Any, record_path: Path) -> dict[str, Any]:
    value = dict(record)
    metrics = _metric_view(value)
    value.update({
        "arm": str(item["arm"]), "scenario": str(item["scenario"]),
        "seed": int(item["env_seed"]), "env_seed": int(item["env_seed"]),
        "policy_seed": int(item["policy_seed"]), "checkpoint_declared": str(checkpoint),
        "checkpoint_sha256_expected": expected_sha, "config": neutral_eval._config_snapshot(config),
        "task_priority_enabled": False, "low_level_priority_time_weight": 0.0,
        "record_path": str(record_path), "model_unchanged": value.get("weights_unchanged"),
        "normal": metrics["normal"], "emergency": metrics["emergency"],
        "failed": metrics["failed"], "failed_energy": metrics["failed_energy"],
        "unresolved": metrics["unresolved"], "safety": metrics["safety"],
        "reward": metrics["reward"],
        "wait": {key: metrics[key] for key in ("mean_wait", "p95_wait", "max_wait", "lowSOC_wait")},
        "technical_ok": metrics["technical_ok"],
    })
    return value


def _fatal(record: Mapping[str, Any], expected_requests: int) -> str | None:
    if int(record.get("requests", 0) or 0) != expected_requests:
        return f"expected {expected_requests} requests, got {record.get('requests')}"
    if record.get("status") != "completed":
        return f"runtime status is {record.get('status')!r}"
    return neutral_eval._record_fatal_reason(record, smoke=False)


def _reuse_records(root: Path, config: Any, output: Path | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate and copy the eight existing before records without replaying them."""

    records: list[dict[str, Any]] = []
    source_paths: list[str] = []
    seen: set[tuple[int, int]] = set()
    for item in repeat_matrix():
        if item["arm"] != "before":
            continue
        key = (int(item["env_seed"]), int(item["policy_seed"]))
        _require(key not in seen, f"duplicate reused before identity: {key}")
        seen.add(key)
        source = neutral_eval._synced_path(
            root / "records" / "before" /
            f"{item['scenario']}_env{item['env_seed']}_policy{item['policy_seed']}.json"
        )
        record = neutral_eval._read_json(source)
        _require(record.get("arm") == "before", f"reused record arm mismatch: {source}")
        _require(str(record.get("scenario")) == "dense24", f"reused record scenario mismatch: {source}")
        _require(int(record.get("seed")) == key[0] and int(record.get("env_seed")) == key[0],
                 f"reused record environment seed mismatch: {source}")
        _require(int(record.get("policy_seed")) == key[1], f"reused record policy seed mismatch: {source}")
        expected_requests = int(neutral_eval.EXPECTED_REQUESTS["dense24"])
        _require(int(record.get("requests", 0)) == expected_requests,
                 f"reused record request count mismatch: {source}")
        checkpoint_sha = record.get("checkpoint_sha256") or record.get("checkpoint_sha256_expected")
        _require(checkpoint_sha == BEFORE_CHECKPOINT_SHA256,
                 f"reused record checkpoint SHA mismatch: {source}")
        _require(record.get("status") == "completed" and record.get("lifecycle_valid") is True,
                 f"reused record status/lifecycle incomplete: {source}")
        _require(record.get("weights_unchanged") is True and record.get("model_unchanged") is True,
                 f"reused record model state is not frozen: {source}")
        _require(not record.get("error") and record.get("technical_ok") is True,
                 f"reused record technical state is incomplete: {source}")
        _require(record.get("training_or_update_called") is False and record.get("teacher_called") is False,
                 f"reused record contains update/Teacher state: {source}")
        _require(record.get("task_priority_enabled") is False
                 and float(record.get("low_level_priority_time_weight", 1.0)) == 0.0,
                 f"reused record task-neutral config flags mismatch: {source}")
        _require(record.get("config") == neutral_eval._config_snapshot(config),
                 f"reused record config mismatch: {source}")
        metrics = _metric_view(record)
        _require(metrics["technical_ok"] is True
                 and not any(bool(value) for value in metrics["safety"].values()),
                 f"reused record safety/technical metrics mismatch: {source}")
        summary = record.get("summary")
        _require(isinstance(summary, Mapping)
                 and int(summary.get("seed")) == key[0]
                 and int(summary.get("policy_seed")) == key[1],
                 f"reused record rollout seed metadata mismatch: {source}")
        destination = (output / "records" / "before" / (
            f"{item['scenario']}_env{item['env_seed']}_policy{item['policy_seed']}.json"
        )) if output is not None else source
        copied = dict(record)
        copied.update({
            "record_path": str(destination),
            "source_record_path": str(source),
            "source_provenance": {"kind": "reused_before_record", "source_path": str(source)},
        })
        records.append(copied)
        source_paths.append(str(source))
        if output is not None:
            neutral_eval._write_json(destination, copied)
    _require(len(seen) == 8 and len(records) == 8, "reuse root must contain exactly eight before records")
    return records, source_paths


def _run_one(item: Mapping[str, Any], *, output: Path, config: Any,
             checkpoint: Path, expected_sha: str) -> tuple[dict[str, Any], str | None]:
    arm = str(item["arm"])
    record_path = output / "records" / arm / (
        f"{item['scenario']}_env{item['env_seed']}_policy{item['policy_seed']}.json"
    )
    rollout_output = output / "rollouts" / arm / record_path.stem
    resolved, _ = neutral_eval._checkpoint_from_report(checkpoint, expected_sha)
    policy, _ = rank_eval._load_policy(resolved, config)
    record = rank_eval._runtime_record(
        policy=policy, checkpoint=resolved, config=config, scenario=str(item["scenario"]),
        seed=int(item["env_seed"]), policy_seed=int(item["policy_seed"]),
        output=rollout_output, smoke=False,
    )
    value = _decorate(record, item, checkpoint, expected_sha, config, record_path)
    return value, _fatal(value, int(neutral_eval.EXPECTED_REQUESTS[str(item["scenario"])]))


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    value = neutral_eval._aggregate_arm(records)
    return {key: value[key] for key in (
        "records", "requests", "normal", "emergency", "failed", "unresolved", "mean_wait",
        "p95_wait", "max_wait", "lowSOC_wait", "reward_total", "safety_nonzero_cells", "technical_ok",
    )}


def _pairs(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (str(row.get("arm")), int(row["env_seed"]), int(row["policy_seed"])): row
        for row in records if "env_seed" in row and "policy_seed" in row
    }
    rows = []
    for env_seed in ENV_SEEDS:
        for policy_seed in POLICY_SEEDS:
            left = index.get(("before", env_seed, policy_seed))
            right = index.get(("after", env_seed, policy_seed))
            before_metrics = None if left is None else _metric_view(left)
            after_metrics = None if right is None else _metric_view(right)
            keys = ("normal", "emergency", "failed", "unresolved", "mean_wait", "p95_wait",
                    "max_wait", "lowSOC_wait", "reward_total")
            before = None if before_metrics is None else {key: before_metrics[key] for key in keys}
            after = None if after_metrics is None else {key: after_metrics[key] for key in keys}
            diff = {
                key: None if before is None or after is None or before[key] is None or after[key] is None
                else after[key] - before[key]
                for key in keys
            }
            rows.append({"scenario": "dense24", "env_seed": env_seed, "policy_seed": policy_seed,
                         "before": before, "after": after,
                         "difference_after_minus_before": diff})
    return rows


def _progress(output: Path, status: str, processed: int, expected: int,
              check_only: bool, error: str | None = None,
              last_record: Mapping[str, Any] | None = None) -> None:
    value: dict[str, Any] = {
        "stage": STAGE, "status": status, "processed": processed, "expected": expected,
        "matrix_size": expected, "check_only": check_only, "full_evaluation": not check_only,
        "training_or_update_called": False, "PPO_ALLOWED": False,
    }
    if error:
        value["error"] = error
    if last_record is not None:
        value["last_record"] = {
            "arm": str(last_record["arm"]), "scenario": str(last_record["scenario"]),
            "env_seed": int(last_record["env_seed"]), "policy_seed": int(last_record["policy_seed"]),
        }
    neutral_eval._write_json(output / "progress.json", value)


def _report(status: str, check_only: bool, matrix: Sequence[Mapping[str, Any]],
            binding: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
            record_paths: Sequence[str], stop_reason: str | None = None,
            *, planned_reused: int = 0, planned_new: int | None = None,
            reused_source_paths: Sequence[str] = ()) -> dict[str, Any]:
    records_reused = sum(bool(row.get("source_provenance")) for row in records)
    result: dict[str, Any] = {
        "stage": STAGE, "status": status, "check_only": check_only,
        "full_evaluation": not check_only, "records_expected": len(matrix),
        "records_processed": len(records), "matrix": [dict(row) for row in matrix],
        "record_paths": list(record_paths), "binding": dict(binding),
        "records_reused": records_reused, "new_rollouts": len(records) - records_reused,
        "planned_reused_records": int(planned_reused),
        "planned_new_rollouts": len(matrix) - int(planned_reused) if planned_new is None else int(planned_new),
        "reused_source_paths": list(reused_source_paths),
        "arms": {arm: _summary([row for row in records if row.get("arm") == arm]) for arm in ARMS},
        "paired_differences": _pairs(records),
        "scope": {"policy_seed_repeats": True, "random_action_repeats": True,
                  "independent_environment_generalization": False},
        "technical_ok": None if check_only else bool(
            len(records) == len(matrix) and all(row.get("technical_ok") is True for row in records)
        ),
        "training_or_update_called": False, "teacher_called": False, "PPO_ALLOWED": False,
    }
    if stop_reason:
        result["stop_reason"] = stop_reason
    return result


def run(*, output_dir: str | Path, check_only: bool = False,
        after_checkpoint: str | Path | None = None,
        after_checkpoint_sha256: str | None = None,
        reuse_before_root: str | Path | None = None) -> dict[str, Any]:
    custom_after = after_checkpoint is not None or after_checkpoint_sha256 is not None
    _require(
        not custom_after or (after_checkpoint is not None and after_checkpoint_sha256 is not None),
        "--after-checkpoint and --after-checkpoint-sha256 must be supplied together",
    )
    after_path = Path(after_checkpoint) if after_checkpoint is not None else AFTER_CHECKPOINT
    after_sha = str(after_checkpoint_sha256) if after_checkpoint_sha256 is not None else AFTER_CHECKPOINT_SHA256
    reuse_root = Path(reuse_before_root) if reuse_before_root is not None else None
    output = Path(output_dir)
    _require(not output.exists(), f"repeat output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    cells, matrix = repeat_cells(), repeat_matrix()
    expected = len(matrix)
    binding: dict[str, Any] = {
        "stage": STAGE, "before_checkpoint_sha256": BEFORE_CHECKPOINT_SHA256,
        "after_checkpoint_sha256": after_sha, "cells": cells,
        "policy_seeds": list(POLICY_SEEDS), "training_or_update_called": False,
        "PPO_ALLOWED": False, "reuse_before_root": None if reuse_root is None else str(reuse_root),
    }
    _progress(output, "STARTING", 0, expected, check_only)
    records: list[dict[str, Any]] = []
    record_paths: list[str] = []
    reused_source_paths: list[str] = []
    try:
        configs, neutral_configs, _ = neutral_eval._build_neutral_inputs()
        _require("dense24" in configs and "dense24" in neutral_configs, "dense24 config is missing")
        config = neutral_configs["dense24"]
        _require(getattr(config, "task_priority_enabled") is False, "task-neutral flag is not false")
        _require(float(getattr(config, "low_level_priority_time_weight")) == 0.0,
                 "task-neutral priority reward weight is not zero")
        binding["config"] = neutral_eval._config_snapshot(config)
        binding["arms"] = {
            "before": _binding(BEFORE_CHECKPOINT, BEFORE_CHECKPOINT_SHA256, config, "before"),
            "after": _binding(
                after_path, after_sha, config, "after",
                validate_pilot_payload=custom_after,
            ),
        }
        if reuse_root is not None:
            _, reused_source_paths = _reuse_records(reuse_root, config)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        result = _report(
            "FAILED", check_only, matrix, {**binding, "preflight_error": str(exc)},
            records, record_paths, str(exc), planned_reused=8 if reuse_root else 0,
            planned_new=8 if reuse_root else 16, reused_source_paths=reused_source_paths,
        )
        neutral_eval._write_json(output / "report.json", result)
        _progress(output, "FAILED", 0, expected, check_only, str(exc))
        return result
    if check_only:
        result = _report(
            "CHECK_ONLY_OK", True, matrix, binding, records, record_paths,
            planned_reused=8 if reuse_root else 0,
            planned_new=8 if reuse_root else 16, reused_source_paths=reused_source_paths,
        )
        neutral_eval._write_json(output / "report.json", result)
        _progress(output, "CHECK_ONLY_OK", 0, expected, True)
        return result

    if reuse_root is not None:
        reused_records, reused_source_paths = _reuse_records(reuse_root, config, output)
        records.extend(reused_records)
        record_paths.extend(str(record["record_path"]) for record in reused_records)
        _progress(output, "RUNNING", len(records), expected, False,
                  last_record=reused_records[-1] if reused_records else None)

    stop_reason = None
    for item in matrix:
        arm = str(item["arm"])
        if reuse_root is not None and arm == "before":
            continue
        checkpoint = BEFORE_CHECKPOINT if arm == "before" else after_path
        expected_sha = BEFORE_CHECKPOINT_SHA256 if arm == "before" else after_sha
        try:
            record, reason = _run_one(item, output=output, config=config,
                                      checkpoint=checkpoint, expected_sha=expected_sha)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            record_path = output / "records" / arm / (
                f"{item['scenario']}_env{item['env_seed']}_policy{item['policy_seed']}.json"
            )
            record = rank_eval._failure_record(
                scenario=str(item["scenario"]), seed=int(item["env_seed"]),
                checkpoint=neutral_eval._synced_path(checkpoint),
                requests=int(neutral_eval.EXPECTED_REQUESTS[str(item["scenario"])]),
                error=exc, policy_seed=int(item["policy_seed"]),
            )
            record = _decorate(record, item, checkpoint, expected_sha, config, record_path)
            reason = str(exc)
        record_path = Path(str(record["record_path"]))
        neutral_eval._write_json(record_path, record)
        records.append(record)
        record_paths.append(str(record_path))
        _progress(output, "RUNNING" if reason is None else "FAILED", len(records), expected,
                  False, reason, item)
        if reason is not None:
            stop_reason = (
                f"{arm}/dense24/{item['env_seed']}/policy{item['policy_seed']}: {reason}"
            )
            break
    status = "FAILED" if stop_reason else "COMPLETED"
    result = _report(
        status, False, matrix, binding, records, record_paths, stop_reason,
        planned_reused=8 if reuse_root else 0, planned_new=8 if reuse_root else 16,
        reused_source_paths=reused_source_paths,
    )
    neutral_eval._write_json(output / "report.json", result)
    last_item = matrix[len(records) - 1] if records else None
    _progress(output, status, len(records), expected, False, stop_reason, last_item)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--after-checkpoint", type=Path)
    parser.add_argument("--after-checkpoint-sha256")
    parser.add_argument("--reuse-before-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.after_checkpoint is None) != (args.after_checkpoint_sha256 is None):
        print(json.dumps({
            "stage": STAGE, "status": "ERROR",
            "error": "--after-checkpoint and --after-checkpoint-sha256 must be supplied together",
        }, sort_keys=True), flush=True)
        return 2
    try:
        result = run(
            output_dir=args.output_dir, check_only=args.check_only,
            after_checkpoint=args.after_checkpoint,
            after_checkpoint_sha256=args.after_checkpoint_sha256,
            reuse_before_root=args.reuse_before_root,
        )
    except (TaskNeutralRepeatsError, FileNotFoundError, ValueError, OSError) as exc:
        print(json.dumps({"stage": STAGE, "status": "ERROR", "error": str(exc)}, sort_keys=True), flush=True)
        return 2
    print(json.dumps({"stage": STAGE, "status": result.get("status"),
                      "records_processed": result.get("records_processed", 0),
                      "technical_ok": result.get("technical_ok"), "PPO_ALLOWED": False}, sort_keys=True),
          flush=True)
    return 0 if result.get("status") in {"CHECK_ONLY_OK", "COMPLETED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
