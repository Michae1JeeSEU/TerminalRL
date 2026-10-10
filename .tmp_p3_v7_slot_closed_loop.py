"""Bounded inference-only V7 late-slot closed-loop comparison helper.

Default execution is check-only.  ``--smoke`` runs four independent two-step
cells into a separate smoke root.  ``--run-full`` is reserved for an
explicitly authorized four-cell run and is never implied by import.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

from terminal_marl.core.safety_trace import SafetyTraceRecorder, _digest
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import evaluate_p3_adaptive_kl_v7 as evaluator


STAGE = "P3_V7_RUN0_SLOT_CLOSED_LOOP_V1"
SEED = 873106
SCENARIO = "dense24"
SMOKE_STEPS = 2
EXPECTED_ORIGINAL_STEPS = 897
FORMAL_ROOT = Path("/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926")
DEFAULT_OUTPUT = Path("/home/user/training_outputs/terminal/p3_adaptive_kl_run0_slot_closed_loop_20260926")
DEFAULT_SMOKE_OUTPUT = Path("/home/user/training_outputs/terminal/p3_adaptive_kl_run0_slot_closed_loop_20260926_smoke")
TAPE = FORMAL_ROOT / "run0" / "dev_after_0080" / "dense24_873106" / "action_tapes" / "safety_trace_seed_873106_step_896.json"
RUN0_CHECKPOINT = FORMAL_ROOT / "run0" / "checkpoints" / "checkpoint_episode_0080.pt"


class ClosedLoopInputError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClosedLoopInputError(message)


def _jsonable(value: Any) -> Any:
    return evaluator._jsonable(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON input missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _cell_name(label: str, reconcile: bool) -> str:
    return f"{label}_{'late_on' if reconcile else 'late_off'}_{SCENARIO}_{SEED}"


def _config_diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    return sorted({str(key) for key in set(left) | set(right) if left.get(key) != right.get(key)})


def _replay_outcomes(env: TerminalParallelEnv) -> dict[str, Any]:
    return {str(key): getattr(value, "value", value) for key, value in env.ledger.outcomes.items()}


def _cell_replay_check(config: Any, record: Mapping[str, Any]) -> dict[str, Any]:
    tape_value = record.get("action_tape")
    _require(isinstance(tape_value, str), "cell record has no action tape")
    tape = Path(tape_value)
    _require(tape.is_file(), f"cell action tape missing: {tape}")
    replay_env = TerminalParallelEnv(config)
    replay = SafetyTraceRecorder.replay_action_tape(replay_env, tape)
    summary = record.get("summary") if isinstance(record.get("summary"), Mapping) else {}
    original_outcomes = summary.get("outcomes") if isinstance(summary, Mapping) else {}
    original_outcomes = {str(key): value for key, value in (original_outcomes or {}).items()}
    replay_outcomes = _replay_outcomes(replay_env)
    return {
        "exact": bool(replay.get("exact")),
        "steps_expected": replay.get("steps_expected"),
        "steps_replayed": replay.get("steps_replayed"),
        "outcomes_match": original_outcomes == replay_outcomes,
        "original_outcomes": original_outcomes,
        "replay_outcomes": replay_outcomes,
    }


_NORMALIZED_ACTION_FIELDS = (
    "a_requested",
    "a_requested_v3",
    "joint_action_index",
    "requested_gate",
    "requested_proposal",
)


def _normalized_action_sequence(tape_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = tape_payload.get("action_tape")
    _require(isinstance(rows, list), "action tape list required")
    return [
        {field: _jsonable(row.get(field)) for field in _NORMALIZED_ACTION_FIELDS}
        for row in rows
    ]


def _original_off_equivalence(record: Mapping[str, Any]) -> dict[str, Any]:
    original_manifest = _read_json(FORMAL_ROOT / "run0" / "manifest.json")
    original_record = next(
        row for row in (original_manifest.get("dev_after", {}).get("records") or [])
        if row.get("scenario") == SCENARIO and int(row.get("seed", -1)) == SEED
    )
    original_tape = _read_json(TAPE)
    current_tape_path = Path(str(record.get("action_tape")))
    current_tape = _read_json(current_tape_path)
    original_outcomes = original_record.get("outcomes") or {}
    current_outcomes = record.get("outcomes") or (record.get("summary") or {}).get("outcomes") or {}
    checks = {
        "action_sequence_equal": _normalized_action_sequence(original_tape) == _normalized_action_sequence(current_tape),
        "action_steps_equal": len(original_tape.get("action_tape", [])) == len(current_tape.get("action_tape", [])),
        "initial_reset_state_hash_equal": original_tape.get("reset_state_hash") == current_tape.get("reset_state_hash"),
        "initial_reset_recipe_hash_equal": original_tape.get("reset_recipe_hash") == current_tape.get("reset_recipe_hash"),
        "environment_config_hash_equal": original_tape.get("environment_config_hash") == current_tape.get("environment_config_hash"),
        "outcomes_equal": _jsonable(original_outcomes) == _jsonable(current_outcomes),
    }
    return {
        "checked": True,
        "original_tape": str(TAPE),
        "current_tape": str(current_tape_path),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "original_steps": len(original_tape.get("action_tape", [])),
        "current_steps": len(current_tape.get("action_tape", [])),
    }


def _load_bindings(protocol: Mapping[str, Any], config: Any, run0_checkpoint: Path) -> dict[str, Any]:
    protocol_sha = evaluator._sha256(evaluator.PROTOCOL)
    optimizer = protocol.get("optimizer") or {}
    learning_rate = float(optimizer.get("learning_rate", 3e-4))
    source_path = Path(str((protocol.get("lineage") or {}).get("source_checkpoint_path")))
    _require(source_path.is_file(), f"source checkpoint missing: {source_path}")
    _require(run0_checkpoint.is_file(), f"run0 checkpoint missing: {run0_checkpoint}")
    source_policy, source_payload, source_sha = evaluator._load_eval_policy(
        source_path, config=config, expected_checkpoint_sha256=evaluator.SOURCE_CHECKPOINT_SHA256,
        protocol_sha256=protocol_sha, final_checkpoint=False, learning_rate=learning_rate,
    )
    run0_sha = _sha256(run0_checkpoint)
    run0_policy, run0_payload, _ = evaluator._load_eval_policy(
        run0_checkpoint, config=config, expected_checkpoint_sha256=run0_sha,
        protocol_sha256=protocol_sha, final_checkpoint=True, learning_rate=learning_rate,
    )
    return {
        "protocol_sha256": protocol_sha,
        "source": {
            "label": "source_03c0", "path": str(source_path), "sha256": source_sha,
            "policy": source_policy, "payload": source_payload,
            "model_digest_before": evaluator._model_digest(source_policy),
            "admission_weight_digest_before": evaluator.admission._policy_weight_digest(source_policy),
            "optimizer_state_empty": len(source_policy.optimizer.state) == 0,
        },
        "run0": {
            "label": "run0_checkpoint_episode_0080", "path": str(run0_checkpoint), "sha256": run0_sha,
            "policy": run0_policy, "payload": run0_payload,
            "model_digest_before": evaluator._model_digest(run0_policy),
            "admission_weight_digest_before": evaluator.admission._policy_weight_digest(run0_policy),
            "optimizer_state_empty": len(run0_policy.optimizer.state) == 0,
        },
    }


def _original_off_preflight(
    *,
    protocol: Mapping[str, Any],
    config: Any,
    run0_checkpoint: Path,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = FORMAL_ROOT / "run0" / "manifest.json"
    manifest = _read_json(manifest_path)
    snapshots = manifest.get("actual_config_snapshots") or {}
    actual_snapshot = snapshots.get(SCENARIO)
    expected_snapshot = evaluator._config_snapshot(config)
    record = next(
        row for row in (manifest.get("dev_after", {}).get("records") or [])
        if row.get("scenario") == SCENARIO and int(row.get("seed", -1)) == SEED
    )
    tape = Path(str(record.get("action_tape")))
    tape_payload = _read_json(tape)
    checks = {
        "manifest_status_observed": manifest.get("status"),
        "original_record_completed": record.get("status") == "completed",
        "original_record_checkpoint_matches": str(record.get("checkpoint")) == str(run0_checkpoint),
        "original_record_weights_unchanged": record.get("weights_unchanged") is True,
        "original_record_teacher_disabled": record.get("teacher_enabled") is False,
        "original_record_PPO_ALLOWED_false": record.get("PPO_ALLOWED") is False,
        "original_record_replay_exact": (record.get("diagnostics") or {}).get("replay_exact") is True,
        "original_record_action_tape_present": tape.is_file(),
        "original_config_snapshot_matches": actual_snapshot == expected_snapshot,
        "original_config_reconcile_off": isinstance(actual_snapshot, Mapping) and not bool(actual_snapshot.get("reconcile_late_merge_slots_enabled")),
        "original_tape_seed_matches": tape_payload.get("episode_seed") == SEED,
        "original_tape_steps_897": len(tape_payload.get("action_tape", [])) == EXPECTED_ORIGINAL_STEPS,
        "original_tape_environment_config_hash_matches": tape_payload.get("environment_config_hash") == _digest(vars(config)),
        "original_tape_reset_state_hash_present": isinstance(tape_payload.get("reset_state_hash"), str),
        "original_tape_arrival_rng_hash_present": isinstance(tape_payload.get("arrival_rng_state_hash"), str),
        "original_tape_policy_rng_present": isinstance(tape_payload.get("policy_rng_state_b64"), str),
        "original_record_model_unchanged": (record.get("weight_digest_before") == record.get("weight_digest_after")),
        "run0_loaded_admission_digest_matches_original_record": (
            bindings["run0"]["admission_weight_digest_before"] == record.get("weight_digest_before")
        ),
        "run0_tensor_digest_comparison": "not_comparable",
        "source_checkpoint_sha256_bound": bindings["source"]["sha256"] == evaluator.SOURCE_CHECKPOINT_SHA256,
        "run0_optimizer_state_empty": bindings["run0"]["optimizer_state_empty"],
        "source_optimizer_state_empty": bindings["source"]["optimizer_state_empty"],
    }
    return {
        "manifest": str(manifest_path),
        "original_record": {
            "scenario": SCENARIO, "seed": SEED, "status": record.get("status"),
            "checkpoint": record.get("checkpoint"), "action_tape": str(tape),
            "steps": record.get("steps_recorded"), "completed": record.get("completed"),
            "weights_unchanged": record.get("weights_unchanged"),
            "weight_digest_before": record.get("weight_digest_before"),
            "weight_digest_after": record.get("weight_digest_after"),
            "loaded_run0_admission_weight_digest": bindings["run0"]["admission_weight_digest_before"],
            "loaded_run0_tensor_digest": bindings["run0"]["model_digest_before"],
        },
        "checks": checks,
        "all_checks_pass": all(
            value is True
            for key, value in checks.items()
            if key not in {"manifest_status_observed", "run0_tensor_digest_comparison"}
        ),
        "original_off_reuse_equivalent": all(
            value is True
            for key, value in checks.items()
            if key not in {"manifest_status_observed", "run0_tensor_digest_comparison"}
        ),
        "config_snapshot_sha256": hashlib.sha256(json.dumps(expected_snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "protocol_sha256": bindings["protocol_sha256"],
        "source_checkpoint_sha256": bindings["source"]["sha256"],
        "run0_checkpoint_sha256": bindings["run0"]["sha256"],
    }


def _preflight() -> dict[str, Any]:
    protocol = evaluator._protocol()
    configs = evaluator._build_configs(protocol)
    config_off = configs[SCENARIO]
    _require(not bool(config_off.reconcile_late_merge_slots_enabled), "V7 base config is not late-slot off")
    config_on = replace(config_off, reconcile_late_merge_slots_enabled=True)
    off_snapshot = evaluator._config_snapshot(config_off)
    on_snapshot = evaluator._config_snapshot(config_on)
    bindings = _load_bindings(protocol, config_off, RUN0_CHECKPOINT)
    original = _original_off_preflight(
        protocol=protocol, config=config_off, run0_checkpoint=RUN0_CHECKPOINT, bindings=bindings,
    )
    return {
        "stage": STAGE, "status": "CHECK_ONLY_OK" if original["all_checks_pass"] else "CHECK_ONLY_FAILED",
        "protocol_sha256": bindings["protocol_sha256"],
        "protocol_digest": evaluator._protocol_digest(protocol),
        "scenario": SCENARIO, "seed": SEED,
        "tape": str(TAPE), "run0_checkpoint": str(RUN0_CHECKPOINT),
        "source_checkpoint": bindings["source"]["path"],
        "binding": {
            "source_sha256": bindings["source"]["sha256"],
            "run0_sha256": bindings["run0"]["sha256"],
            "source_model_digest": bindings["source"]["model_digest_before"],
            "run0_model_digest": bindings["run0"]["model_digest_before"],
            "model_digest_algorithm": "v6._tensor_digest; not_comparable_to_original_record_weight_digest",
            "source_admission_weight_digest": bindings["source"]["admission_weight_digest_before"],
            "run0_admission_weight_digest": bindings["run0"]["admission_weight_digest_before"],
            "source_optimizer_state_empty": bindings["source"]["optimizer_state_empty"],
            "run0_optimizer_state_empty": bindings["run0"]["optimizer_state_empty"],
        },
        "config_diff_off_on": _config_diff(off_snapshot, on_snapshot),
        "config_off": off_snapshot,
        "config_on": on_snapshot,
        "original_off_preflight": original,
        "cells": [
            {"label": label, "reconcile_late_merge_slots_enabled": flag, "config_snapshot": evaluator._config_snapshot(config_on if flag else config_off)}
            for label, policy_key in (("source_03c0", "source"), ("run0_checkpoint_episode_0080", "run0"))
            for flag in (False, True)
        ],
        "output_created": False,
    }


def _run_cells(*, mode: str, output: Path) -> dict[str, Any]:
    preflight = _preflight()
    _require(preflight["status"] == "CHECK_ONLY_OK", "preflight checks failed")
    _require(not output.exists(), f"output root already exists: {output}")
    protocol = evaluator._protocol()
    runtime_protocol = evaluator.v6._runtime_protocol(protocol)
    configs = evaluator._build_configs(protocol)
    config_off = configs[SCENARIO]
    config_on = replace(config_off, reconcile_late_merge_slots_enabled=True)
    bindings = _load_bindings(protocol, config_off, RUN0_CHECKPOINT)
    output.mkdir(parents=True, exist_ok=False)
    cells: list[dict[str, Any]] = []
    policy_specs = (("source_03c0", bindings["source"]), ("run0_checkpoint_episode_0080", bindings["run0"]))
    try:
        for label, binding in policy_specs:
            for reconcile in (False, True):
                config = config_on if reconcile else config_off
                cell_dir = output / _cell_name(label, reconcile)
                model_before = evaluator._model_digest(binding["policy"])
                record = evaluator._episode_record(
                    policy=binding["policy"], label=label, checkpoint=Path(binding["path"]),
                    config=config, protocol=protocol, runtime_protocol=runtime_protocol,
                    scenario=SCENARIO, seed=SEED, output=cell_dir,
                    max_steps=SMOKE_STEPS if mode == "smoke" else None,
                    require_full_resolution=False if mode == "smoke" else True,
                )
                model_after = evaluator._model_digest(binding["policy"])
                replay = _cell_replay_check(config, record) if record.get("action_tape") else {
                    "exact": False, "outcomes_match": False, "missing_tape": True,
                }
                original_off_equivalence = (
                    _original_off_equivalence(record)
                    if mode == "full" and label == "run0_checkpoint_episode_0080" and not reconcile
                    else {"checked": False, "all_checks_pass": True, "checks": {}}
                )
                threshold = float((protocol.get("reward") or {}).get("low_soc_urgency_threshold", .65))
                lifecycle_summary = evaluator._lifecycle_summary([record], threshold=threshold)
                technical_checks_pass = all(
                    (
                        record.get("status") == "completed",
                        model_before == model_after,
                        record.get("weights_unchanged") is True,
                        record.get("teacher_enabled") is False,
                        record.get("PPO_ALLOWED") is False,
                        replay.get("exact") is True,
                        replay.get("outcomes_match") is True,
                        original_off_equivalence.get("all_checks_pass") is True,
                    )
                )
                cell = {
                    "stage": f"{STAGE}_{mode.upper()}", "status": "completed" if technical_checks_pass else "failed",
                    "label": label, "scenario": SCENARIO, "seed": SEED,
                    "reconcile_late_merge_slots_enabled": reconcile,
                    "config_snapshot": evaluator._config_snapshot(config),
                    "checkpoint": binding["path"], "checkpoint_sha256": binding["sha256"],
                    "model_weight_digest_before": model_before,
                    "model_weight_digest_after": model_after,
                    "weights_unchanged": bool(model_before == model_after and record.get("weights_unchanged") is True),
                    "teacher_disabled": record.get("teacher_enabled") is False,
                    "PPO_ALLOWED": False,
                    "technical_checks_pass": technical_checks_pass,
                    "record": record,
                    "replay": replay,
                    "original_off_equivalence": original_off_equivalence,
                    "lifecycle": lifecycle_summary,
                    "output": str(cell_dir),
                }
                _write_json(cell_dir / "cell_manifest.json", cell)
                cells.append(cell)
                _write_json(output / "progress.json", {
                    "stage": STAGE, "mode": mode, "processed_cells": len(cells),
                    "expected_cells": 4, "cells": cells,
                })
                if cell["status"] != "completed":
                    raise ClosedLoopInputError(f"cell failed: {label} reconcile={reconcile}")
        status = "SMOKE_COMPLETED" if mode == "smoke" else "FULL_COMPLETED"
    except BaseException as exc:
        status = "FAILED"
        _write_json(output / "manifest.json", {
            "stage": STAGE, "mode": mode, "status": status,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "preflight": preflight, "cells": cells,
            "output_created": True, "P3_QUALIFIED": False, "P4_ALLOWED": False, "PPO_ALLOWED": False,
        })
        raise
    result = {
        "stage": STAGE, "mode": mode, "status": status,
        "preflight": preflight, "cells": cells,
        "output": str(output), "output_created": True,
        "all_cells_weights_unchanged": all(cell["weights_unchanged"] for cell in cells),
        "all_cells_teacher_disabled": all(cell["teacher_disabled"] for cell in cells),
        "all_cells_PPO_ALLOWED_false": all(not cell["PPO_ALLOWED"] for cell in cells),
        "all_cells_replay_exact": all(cell["replay"].get("exact") is True for cell in cells),
        "all_cells_outcomes_match": all(cell["replay"].get("outcomes_match") is True for cell in cells),
        "all_cells_technical_checks_pass": all(cell.get("technical_checks_pass") is True for cell in cells),
        "P3_QUALIFIED": False, "P4_ALLOWED": False, "PPO_ALLOWED": False,
    }
    _write_json(output / "manifest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--run-full", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-output", type=Path, default=DEFAULT_SMOKE_OUTPUT)
    args = parser.parse_args()
    if not args.smoke and not args.run_full:
        result = _preflight()
    elif args.smoke:
        result = _run_cells(mode="smoke", output=args.smoke_output)
    else:
        result = _run_cells(mode="full", output=args.output)
    print(json.dumps(_jsonable(result), sort_keys=True), flush=True)
    return 0 if result.get("status") in {"CHECK_ONLY_OK", "SMOKE_COMPLETED", "FULL_COMPLETED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
