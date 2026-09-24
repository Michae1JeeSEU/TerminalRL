"""Controlled formal P3 V5 runner after an explicit bounded-pilot admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.core.checkpoints import save_checkpoint
from terminal_marl.training import (
    evaluate_random_soc_gate_repair_v1 as base,
)
from terminal_marl.training import (
    run_v4_2_p3_controlled_emergency_admission_v5 as admission,
)
from terminal_marl.training import (
    run_v4_2_p3_mixed_priority_soc_training_v3 as p3_legacy,
)


STAGE = "P3_FORMAL_CONTROLLED_EMERGENCY_V5"
RUNNER_MODULE = "terminal_marl.training.run_v4_2_p3_controlled_emergency_formal_v5"
FORMAL_PROTOCOL = Path("terminal_marl/configs/low_v4_2/p3_controlled_emergency_formal_v5.yaml")
ADMISSION_REPORT_STAGE = "P3_FORMAL_ADMISSION_DECISION_V5"
CHECKPOINT_KIND = "repair"
CHECKPOINT_SHA256 = admission.REPAIR_CANDIDATE_CHECKPOINT_SHA256
RUN_INDICES = (0, 1, 2)
EPISODES_PER_RUN = 240
EPISODE_PATTERN = ("flow6", "dense24", "flow12", "dense24")
DEV_SEEDS = (853101, 853102, 853103, 853104, 853105, 853106)
DEV_SCENARIOS = ("flow6", "flow12", "dense24", "flow6", "flow12", "dense24")
DEV_CHECKPOINTS = (40, 80, 120, 160, 200, 240)
HELDOUT_SEEDS = set(range(854001, 854013))


class FormalInputError(ValueError):
    """Raised when formal authorization or a frozen binding is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalInputError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    return admission._jsonable(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON input missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _formal_protocol() -> dict[str, Any]:
    _require(FORMAL_PROTOCOL.is_file(), f"formal protocol missing: {FORMAL_PROTOCOL}")
    value = yaml.safe_load(FORMAL_PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, Mapping), "formal protocol must be a mapping")
    value = dict(value)
    _require(value.get("protocol") == "V4_2_P3_CONTROLLED_EMERGENCY_FORMAL_V5",
             "formal protocol id changed")
    _require(value.get("source_admission_config") == admission.PROTOCOL.as_posix(),
             "formal admission source binding changed")
    implementation = value.get("implementation") or {}
    _require(implementation.get("runner_module") == RUNNER_MODULE,
             "formal runner module binding changed")
    lineage = value.get("lineage") or {}
    _require(lineage.get("selected_initial_checkpoint_kind") == "repair_candidate",
             "formal checkpoint kind changed")
    _require(lineage.get("selected_initial_checkpoint_sha256") == CHECKPOINT_SHA256,
             "formal checkpoint SHA binding changed")
    _require(lineage.get("same_initial_checkpoint_for_all_runs") is True,
             "formal runs must share the initial checkpoint")
    _require(lineage.get("fresh_optimizer_for_each_run") is True,
             "formal runs must use fresh optimizers")
    _require(lineage.get("no_cross_run_optimizer_state") is True,
             "formal runs must not share optimizer state")
    bindings = value.get("bindings") or {}
    for key, expected in {
        "action_schema_version": "low_action_schema_v4_2_hierarchical",
        "observation_schema_version": "phaseA_low_obs_v2_1_retry_state",
        "observation_dimension": 88,
        "vehicle_profile_version": "eh216s_research_v2",
        "runtime_shortlist_k": 3,
        "reconcile_late_merge_slots_enabled": False,
        "teacher": False,
        "scheduler_mode": "execution",
        "high_level_enabled": False,
        "shield_enabled": True,
        "algorithm": "PPO",
        "new_algorithm": False,
    }.items():
        _require(bindings.get(key) == expected, f"formal binding changed: {key}")
    environment = value.get("environment") or {}
    horizons = environment.get("full_scenario_horizon_seconds")
    _require(horizons == {"flow6": 1210, "flow12": 1570, "dense24": 1800},
             "formal scenario horizons changed")
    _require(environment.get("scenarios") == ["flow6", "dense24", "flow12"],
             "formal scenarios changed")
    schedule = value.get("formal_schedule") or {}
    _require(schedule.get("independent_runs") == 3, "formal run count changed")
    _require(schedule.get("episodes_per_run") == EPISODES_PER_RUN,
             "formal episode count changed")
    _require(schedule.get("total_episodes") == 720, "formal total episode count changed")
    _require(schedule.get("run_indices") == list(RUN_INDICES), "formal run indices changed")
    _require(schedule.get("episode_pattern") == list(EPISODE_PATTERN),
             "formal episode pattern changed")
    _require(schedule.get("pattern_repetitions") == 60, "formal pattern repetitions changed")
    _require(schedule.get("scenario_order_is_registered") is True,
             "formal scenario order binding changed")
    seeds = schedule.get("train_seeds") or {}
    expected_seeds = {index: list(range(850001 + index * 1000, 850241 + index * 1000))
                      for index in RUN_INDICES}
    _require({int(key): list(values) for key, values in seeds.items()} == expected_seeds,
             "formal train seed schedule changed")
    development = value.get("development") or {}
    _require(tuple(development.get("seeds", ())) == DEV_SEEDS,
             "formal development seeds changed")
    _require(tuple(development.get("scenarios", ())) == DEV_SCENARIOS,
             "formal development scenarios changed")
    _require(tuple(development.get("compare_after_updates", ())) == DEV_CHECKPOINTS,
             "formal development checkpoints changed")
    _require(development.get("baseline_before_training") is True and
             development.get("dev_runs_before_first_update") is True,
             "formal baseline binding changed")
    _require(development.get("full_resolution_required") is True,
             "formal development must require full resolution")
    _require(development.get("heldout") is False, "formal development heldout binding changed")
    heldout = value.get("heldout_reservation") or {}
    _require(heldout.get("run_in_this_task") is False and
             heldout.get("consume_or_reset") is False,
             "formal heldout reservation changed")
    checkpointing = value.get("checkpointing") or {}
    _require(checkpointing.get("save_after_each_accepted_episode") is True,
             "formal checkpoint cadence changed")
    return value


def _expected_train_seeds(formal: Mapping[str, Any], run_index: int) -> list[int]:
    seeds = (formal["formal_schedule"]["train_seeds"] or {})
    values = seeds.get(run_index, seeds.get(str(run_index)))
    _require(isinstance(values, list), f"missing train seeds for run {run_index}")
    return [int(value) for value in values]


def _validate_schedule_disjoint(formal: Mapping[str, Any]) -> None:
    train_sets = [set(_expected_train_seeds(formal, index)) for index in RUN_INDICES]
    _require(all(len(values) == EPISODES_PER_RUN for values in train_sets),
             "formal train seed count changed")
    _require(not (train_sets[0] & train_sets[1] or train_sets[0] & train_sets[2] or
                  train_sets[1] & train_sets[2]), "formal train seeds overlap")
    dev = set(DEV_SEEDS)
    _require(not any(dev & values for values in train_sets), "formal dev/train seeds overlap")
    _require(not any(HELDOUT_SEEDS & values for values in train_sets),
             "formal heldout/train seeds overlap")
    _require(not (HELDOUT_SEEDS & dev), "formal heldout/dev seeds overlap")


def _validate_admission_report(path: Path, formal: Mapping[str, Any]) -> dict[str, Any]:
    report = _read_json(path)
    _require(report.get("stage") == ADMISSION_REPORT_STAGE,
             "admission report stage mismatch")
    _require(report.get("FORMAL_P3_ALLOWED") is True,
             "formal admission authorization is false")
    _require(report.get("initial_checkpoint_sha256") == CHECKPOINT_SHA256,
             "admission initial checkpoint SHA mismatch")
    _require(report.get("admission_protocol_sha256") == _sha256(admission.PROTOCOL),
             "admission protocol SHA mismatch")
    _require(report.get("formal_protocol_sha256") == _sha256(FORMAL_PROTOCOL),
             "formal protocol SHA mismatch")
    pilot_path = Path(str(report.get("pilot_manifest_path", "")))
    _require(pilot_path.is_file(), f"pilot manifest missing: {pilot_path}")
    _require(report.get("pilot_manifest_sha256") == _sha256(pilot_path),
             "pilot manifest SHA mismatch")
    pilot = _read_json(pilot_path)
    _require(pilot.get("PILOT_TECHNICAL_PASS") is True,
             "pilot manifest technical pass is false")
    _require(pilot.get("selected_checkpoint_sha256") == CHECKPOINT_SHA256,
             "pilot selected checkpoint SHA mismatch")
    return {
        "report": report,
        "pilot_manifest": pilot,
        "pilot_manifest_path": str(pilot_path),
    }


def _validate_runtime_bindings(report: Mapping[str, Any], configs: Mapping[str, Any],
                               snapshots: Mapping[str, Any],
                               admission_protocol: Mapping[str, Any]) -> None:
    _require(report.get("actual_config_snapshots") == _jsonable(snapshots),
             "admission actual configuration snapshots mismatch")
    _require(report.get("optimizer_boundary") ==
             _jsonable(admission_protocol.get("optimizer", {})),
             "admission optimizer boundary mismatch")
    semantic_digest = admission._protocol_digest(admission_protocol)
    for key in ("protocol_sha256", "pilot_protocol_sha256"):
        if key in report:
            _require(report.get(key) == semantic_digest,
                     f"admission semantic protocol digest mismatch: {key}")
    schema_binding = report.get("schema_binding")
    if schema_binding is not None:
        expected = admission._schema_binding(configs["flow6"])
        _require(schema_binding == _jsonable(expected),
                 "admission schema binding mismatch")


def _config_context(admission_protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    base_protocol = base._load_yaml_protocol()
    configs = {
        scenario: admission._build_config(scenario, base_protocol)
        for scenario in ("flow6", "flow12", "dense24")
    }
    for scenario, config in configs.items():
        _require(not bool(getattr(config, "reconcile_late_merge_slots_enabled", False)),
                 f"late-slot reconciliation enabled for {scenario}")
        _require(str(getattr(config, "action_schema_version", "")) ==
                 "low_action_schema_v4_2_hierarchical", "action schema changed")
        _require(int(getattr(config, "observation_dimension", 88)) == 88,
                 "observation dimension changed")
        _require(not bool(getattr(config, "teacher_enabled", False)), "Teacher enabled")
    snapshots = {scenario: base._config_snapshot(config) for scenario, config in configs.items()}
    _require(float(configs["flow6"].max_simulation_seconds) == 1210.0,
             "flow6 horizon mismatch")
    _require(float(configs["flow12"].max_simulation_seconds) == 1570.0,
             "flow12 horizon mismatch")
    _require(float(configs["dense24"].max_simulation_seconds) == 1800.0,
             "dense24 horizon mismatch")
    return configs, snapshots


def _public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(record)
    value.pop("_steps", None)
    return value


def _training_hard_pass(record: Mapping[str, Any]) -> bool:
    gates = record.get("gates")
    reward = record.get("reward_telemetry") or {}
    return bool(
        record.get("status") == "completed"
        and record.get("hard_gate_pass") is True
        and record.get("weights_unchanged") is True
        and record.get("failed_energy", 0) == 0
        and record.get("other_failure_count", 0) == 0
        and record.get("safety_invariant_error") is not True
        and reward.get("available") is True
        and reward.get("components_consistent") is True
        and isinstance(gates, Mapping)
        and all(bool(value) for key, value in gates.items() if key != "full_resolution")
    )


def _dev_hard_pass(record: Mapping[str, Any]) -> bool:
    return bool(_training_hard_pass(record) and record.get("full_resolution") is True)


def _rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _save_and_restore(policy: Any, checkpoint: Path, config: Any,
                      admission_protocol: Mapping[str, Any], formal_sha: str,
                      *, episode_index: int, timestep: int, trainable: tuple[str, ...]) -> tuple[Any, bool]:
    save_checkpoint(
        checkpoint, policy, config, stage=STAGE, formal=True, development_only=True,
        ppo_called=True,
        protocol_sha256=formal_sha, formal_protocol_sha256=formal_sha,
        protocol_digest=admission._protocol_digest(admission_protocol),
        admission_protocol_sha256=_sha256(admission.PROTOCOL),
        source_checkpoint_sha256=CHECKPOINT_SHA256,
        initial_checkpoint_sha256=CHECKPOINT_SHA256,
        checkpoint_kind=CHECKPOINT_KIND, checkpoint_promotable=False,
        P3_QUALIFIED=False, P4_ALLOWED=False, PPO_ALLOWED=False,
        episode_index=int(episode_index), timestep=int(timestep),
        trainable_parameters=list(trainable), rng_state=_rng_state(),
    )
    restored = p3_legacy._restored_policy(checkpoint, config, admission_protocol)
    exact = bool(
        admission._state_equal(policy.model.state_dict(), restored.model.state_dict())
        and admission._state_equal(policy.optimizer.state_dict(), restored.optimizer.state_dict())
    )
    return restored, exact


def _dev_evaluation(*, policy: Any, configs: Mapping[str, Any], admission_protocol: Mapping[str, Any],
                    output: Path, accepted_episodes: int, checkpoint: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    fatal = False
    for seed, scenario in zip(DEV_SEEDS, DEV_SCENARIOS):
        config = configs[scenario]
        horizon = int(round(float(config.max_simulation_seconds) / float(config.dt)))
        record = admission._record_rollout(
            policy=policy, config=config, p=admission_protocol,
            scenario=scenario, seed=int(seed), output=output / f"{scenario}_{seed}",
            label=f"dev_{accepted_episodes:04d}", deterministic=True,
            max_steps=horizon, require_full_resolution=True,
            checkpoint=checkpoint,
        )
        public = _public_record(record)
        records.append(public)
        if not _dev_hard_pass(record):
            fatal = True
            break
    result = {
        "stage": "P3_FORMAL_DEV",
        "accepted_episodes": int(accepted_episodes),
        "records": records,
        "complete": len(records) == len(DEV_SEEDS),
        "full_resolution": bool(not fatal and len(records) == len(DEV_SEEDS)),
        "hard_failure": bool(fatal),
        "same_initial_states": True,
        "deterministic": True,
        "heldout": False,
    }
    _write_json(output / "manifest.json", result)
    return result


def _development_comparison(baseline: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    try:
        comparison = admission._performance_comparison(baseline, current)
        rows = comparison.get("paired_records", [])
        return {
            "accepted_episodes": int(current.get("accepted_episodes", 0)),
            "paired_records": rows,
            "priority_weighted_wait_deltas": [
                row.get("priority_weighted_wait_delta_seconds")
                for row in rows if isinstance(row, Mapping)
            ],
            "low_soc_wait_deltas": [
                row.get("low_soc_mean_wait_delta_seconds")
                for row in rows if isinstance(row, Mapping)
            ],
            "pooled_emergency": comparison.get("pooled_emergency"),
            "report_only": True,
            "review_required": True,
        }
    except Exception as exc:
        return {
            "accepted_episodes": int(current.get("accepted_episodes", 0)),
            "paired_records": [],
            "priority_weighted_wait_deltas": [],
            "low_soc_wait_deltas": [],
            "report_only": True,
            "review_required": True,
            "available": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def _preflight(*, run_index: int, admission_report: Path, checkpoint: Path) -> dict[str, Any]:
    formal = _formal_protocol()
    _validate_schedule_disjoint(formal)
    admission_protocol = admission.protocol()
    authorization = _validate_admission_report(admission_report, formal)
    _require(checkpoint.is_file(), f"initial checkpoint missing: {checkpoint}")
    _require(_sha256(checkpoint) == CHECKPOINT_SHA256, "initial checkpoint SHA mismatch")
    configs, snapshots = _config_context(admission_protocol)
    _validate_runtime_bindings(
        authorization["report"], configs, snapshots, admission_protocol,
    )
    policy, metadata, trainable, loaded_sha = admission._load_source(
        checkpoint, CHECKPOINT_KIND, configs["flow6"], admission_protocol,
    )
    _require(loaded_sha == CHECKPOINT_SHA256, "strict loader returned unexpected SHA")
    _require(len(policy.optimizer.state) == 0, "initial optimizer is not fresh")
    return {
        "formal": formal,
        "formal_protocol_sha256": _sha256(FORMAL_PROTOCOL),
        "admission_protocol": admission_protocol,
        "admission_protocol_sha256": _sha256(admission.PROTOCOL),
        "authorization": authorization,
        "configs": configs,
        "config_snapshots": snapshots,
        "policy": policy,
        "metadata": metadata,
        "trainable": tuple(trainable),
        "loaded_sha": loaded_sha,
        "run_index": int(run_index),
        "checkpoint": checkpoint,
    }


def _check_only_payload(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "status": "CHECK_ONLY_OK",
        "check_only": True,
        "run_index": context["run_index"],
        "formal_protocol_sha256": context["formal_protocol_sha256"],
        "admission_protocol_sha256": context["admission_protocol_sha256"],
        "initial_checkpoint_sha256": context["loaded_sha"],
        "checkpoint_kind": CHECKPOINT_KIND,
        "strict_load": True,
        "fresh_optimizer": True,
        "actual_config_snapshots": context["config_snapshots"],
        "trainable_parameter_count": len(context["trainable"]),
        "heldout_executed": False,
        "output_created": False,
        "FORMAL_P3_ALLOWED": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
        "PPO_ALLOWED": False,
    }


def run(*, run_index: int, admission_report: str | Path, checkpoint: str | Path,
        output: str | Path, check_only: bool = False) -> dict[str, Any]:
    _require(int(run_index) in RUN_INDICES, "run-index must be 0, 1, or 2")
    output_path = Path(output)
    context = _preflight(
        run_index=int(run_index), admission_report=Path(admission_report),
        checkpoint=Path(checkpoint),
    )
    if check_only:
        return _check_only_payload(context)
    _require(not output_path.exists(), f"formal output must be fresh: {output_path}")
    output_path.mkdir(parents=True, exist_ok=False)
    formal = context["formal"]
    admission_protocol = context["admission_protocol"]
    configs = context["configs"]
    policy = context["policy"]
    trainable = context["trainable"]
    train_seeds = _expected_train_seeds(formal, int(run_index))
    records: list[dict[str, Any]] = []
    dev_evaluations: list[dict[str, Any]] = []
    accepted = 0
    timestep = 0
    stopped = False
    stop_reason: str | None = None
    checkpoint_path: str | None = None
    current_checkpoint = Path(context["checkpoint"])
    ppo_called = False
    baseline: dict[str, Any] | None = None
    try:
        baseline = _dev_evaluation(
            policy=policy, configs=configs, admission_protocol=admission_protocol,
            output=output_path / "dev_before_training", accepted_episodes=0,
            checkpoint=current_checkpoint,
        )
        dev_evaluations.append(baseline)
        if baseline["hard_failure"] or not baseline["full_resolution"]:
            stopped = True
            stop_reason = "baseline development full-resolution or hard failure"
        if not stopped:
            for episode_index, (seed, scenario) in enumerate(
                zip(train_seeds, (EPISODE_PATTERN * 60)), start=1,
            ):
                config = configs[scenario]
                horizon = int(round(float(config.max_simulation_seconds) / float(config.dt)))
                record = admission._record_rollout(
                    policy=policy, config=config, p=admission_protocol,
                    scenario=scenario, seed=int(seed),
                    output=output_path / "episodes" / f"episode_{episode_index:04d}_{scenario}_{seed}",
                    label=f"formal_{run_index}", deterministic=False, max_steps=horizon,
                    require_full_resolution=False, checkpoint=current_checkpoint,
                )
                record["checkpoint_input_sha256"] = (
                    _sha256(current_checkpoint) if current_checkpoint.is_file() else CHECKPOINT_SHA256
                )
                record["checkpoint_sha256"] = record["checkpoint_input_sha256"]
                record["initial_checkpoint_sha256"] = CHECKPOINT_SHA256
                hard_pass = _training_hard_pass(record)
                if hard_pass:
                    ppo_called = True
                    update, update_gates = p3_legacy._transactional_update(
                        policy, record.get("_steps", []), config,
                        str(record.get("action_tape")), admission_protocol,
                    )
                    record["update"] = update
                    record["update_gates"] = update_gates
                    record["ppo_called"] = True
                    update_pass = bool(isinstance(update_gates, Mapping) and
                                       update_gates and all(bool(value) for value in update_gates.values()) and
                                       isinstance(update, Mapping) and
                                       int(update.get("accepted_updates", 0)) >= 1)
                else:
                    update_pass = False
                    record["update"] = None
                    record["update_gates"] = {"rollout_hard_gates": False}
                if hard_pass and update_pass:
                    next_timestep = timestep + len(record.get("_steps", []))
                    checkpoint_file = output_path / "checkpoints" / f"checkpoint_episode_{episode_index:04d}.pt"
                    try:
                        restored, restore_exact = _save_and_restore(
                            policy, checkpoint_file, config, admission_protocol,
                            context["formal_protocol_sha256"], episode_index=episode_index,
                            timestep=next_timestep, trainable=trainable,
                        )
                    except Exception as exc:
                        restore_exact = False
                        record["checkpoint_error"] = {"type": type(exc).__name__, "message": str(exc)}
                        restored = policy
                    record["checkpoint_restore_exact"] = bool(restore_exact)
                    record["checkpoint_after"] = str(checkpoint_file)
                    record["checkpoint_after_sha256"] = (
                        _sha256(checkpoint_file) if checkpoint_file.is_file() else None
                    )
                    if restore_exact:
                        accepted += 1
                        timestep = next_timestep
                        policy = restored
                        current_checkpoint = checkpoint_file
                        checkpoint_path = str(checkpoint_file)
                    else:
                        stopped = True
                        stop_reason = "checkpoint restore integrity failure"
                else:
                    stopped = True
                    stop_reason = "rollout or PPO update hard failure"
                record["accepted_episode"] = bool(hard_pass and update_pass and record.get("checkpoint_restore_exact") is True)
                records.append(_public_record(record))
                _write_json(output_path / "progress.json", {
                    "stage": STAGE, "run_index": int(run_index),
                    "accepted_episodes": accepted, "episode_index": episode_index,
                    "records": records, "dev_evaluations": dev_evaluations,
                    "stopped": stopped, "stop_reason": stop_reason,
                    "no_resume_claim": True, "heldout_executed": False,
                })
                if stopped:
                    break
                if accepted in DEV_CHECKPOINTS:
                    dev = _dev_evaluation(
                        policy=policy, configs=configs, admission_protocol=admission_protocol,
                        output=output_path / f"dev_after_{accepted:04d}",
                        accepted_episodes=accepted, checkpoint=current_checkpoint,
                    )
                    dev_evaluations.append(dev)
                    if dev["hard_failure"] or not dev["full_resolution"]:
                        stopped = True
                        stop_reason = "development full-resolution or hard failure"
                        break
        if stopped and stop_reason and stop_reason.startswith("development"):
            status = "BLOCKED_FOR_REVIEW"
        elif stopped:
            status = "FAILED"
        else:
            status = "COMPLETED"
    except KeyboardInterrupt:
        stopped = True
        status = "BLOCKED_FOR_REVIEW"
        stop_reason = "interrupted; partial outputs preserved for primary review"
    except Exception as exc:
        stopped = True
        status = "FAILED"
        stop_reason = f"formal run exception: {type(exc).__name__}: {exc}"
        if baseline is None:
            baseline = {
                "stage": "P3_FORMAL_DEV", "accepted_episodes": 0,
                "records": [], "complete": False, "full_resolution": False,
                "hard_failure": True, "same_initial_states": True,
                "deterministic": True, "heldout": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
    result = {
        "stage": STAGE, "status": status, "run_index": int(run_index),
        "episodes_requested": EPISODES_PER_RUN, "episodes_recorded": len(records),
        "accepted_episodes": accepted, "records": records,
        "dev_baseline": baseline, "dev_evaluations": dev_evaluations,
        "development_comparisons": [
            _development_comparison(baseline, evaluation)
            for evaluation in dev_evaluations[1:]
        ] if baseline is not None else [],
        "stopped": stopped, "stop_reason": stop_reason,
        "selected_checkpoint": str(context["checkpoint"]),
        "selected_checkpoint_kind": CHECKPOINT_KIND,
        "initial_checkpoint_sha256": CHECKPOINT_SHA256,
        "final_checkpoint": checkpoint_path,
        "formal_protocol_sha256": context["formal_protocol_sha256"],
        "admission_protocol_sha256": context["admission_protocol_sha256"],
        "actual_config_snapshots": context["config_snapshots"],
        "train_seeds": train_seeds, "episode_pattern": list(EPISODE_PATTERN),
        "dev_seeds": list(DEV_SEEDS), "dev_scenarios": list(DEV_SCENARIOS),
        "dev_checkpoints": list(DEV_CHECKPOINTS),
        "heldout_executed": False, "no_resume_claim": True,
        "development_only": True, "performance_review_required": True,
        "FORMAL_P3_ALLOWED": False, "P3_QUALIFIED": False,
        "P4_ALLOWED": False, "checkpoint_promotable": False,
        "PPO_ALLOWED": False, "ppo_called": bool(ppo_called),
    }
    _write_json(output_path / "manifest.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--admission-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        run_index=args.run_index, admission_report=args.admission_report,
        checkpoint=args.checkpoint, output=args.output, check_only=args.check_only,
    )
    print(json.dumps(_jsonable(result), sort_keys=True), flush=True)
    return 0 if result.get("status") in {"COMPLETED", "CHECK_ONLY_OK"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
