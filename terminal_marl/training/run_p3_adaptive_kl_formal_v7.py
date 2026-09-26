"""Prepared V7 formal runner with an optional adaptive reference-KL penalty.

The runner is intentionally formal-only.  It reuses the reviewed V6 rollout,
configuration, checkpoint, development, and recovery helpers, while binding a
new protocol and a primary-supplied V7 admission report.  No launch is
performed by importing this module.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.algo.gate_reference_v4_2 import GateReference
from terminal_marl.core.checkpoints import save_checkpoint
from terminal_marl.training import p3_kl_recovery_v1 as recovery
from terminal_marl.training import run_v4_2_p3_gate_anchored_v6 as v6


STAGE = "P3_ADAPTIVE_KL_FORMAL_V7"
RUNNER_MODULE = "terminal_marl.training.run_p3_adaptive_kl_formal_v7"
PROTOCOL = Path("terminal_marl/configs/low_v4_2/p3_adaptive_kl_v7.yaml")
SOURCE_CHECKPOINT_SHA256 = v6.SOURCE_CHECKPOINT_SHA256
SOURCE_CHECKPOINT_KIND = v6.SOURCE_CHECKPOINT_KIND
OBSERVATION_DIM = 88
RUN_INDICES = (0, 1, 2)
EPISODE_PATTERN = ("flow6", "dense24", "flow12", "dense24")
RECOVERY_STATUSES = frozenset({"UPDATED", "SKIPPED_KL", "HARD_FAILED"})
SKIP_STALL_LIMIT = 10


class V7InputError(ValueError):
    """Raised when a V7 protocol or admission binding is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V7InputError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    _require(target.is_file(), f"JSON input missing: {target}")
    value = json.loads(target.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {target}")
    return value


def _protocol() -> dict[str, Any]:
    _require(PROTOCOL.is_file(), f"V7 protocol missing: {PROTOCOL}")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, Mapping), "V7 protocol must be a mapping")
    value = dict(value)
    _require(value.get("protocol") == "V4_2_P3_ADAPTIVE_KL_V7", "V7 protocol id changed")
    _require(value.get("status") in {"IMPLEMENTED_PENDING_VALIDATION", "READY_FOR_FORMAL_LAUNCH"},
             "V7 protocol is not in a preparatory state")
    lineage = value.get("lineage") or {}
    _require(lineage.get("source_checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256,
             "V7 source checkpoint SHA changed")
    bindings = value.get("bindings") or {}
    expected_bindings = {
        "action_schema_version": "low_action_schema_v4_2_hierarchical",
        "observation_schema_version": "phaseA_low_obs_v2_1_retry_state",
        "observation_dimension": OBSERVATION_DIM,
        "vehicle_profile_version": "eh216s_research_v2",
        "runtime_shortlist_k": 3,
        "reconcile_late_merge_slots_enabled": False,
        "v4_2_rejected_release_hold_mode": "canonical_wait",
        "teacher": False,
        "scheduler_mode": "execution",
        "high_level_enabled": False,
        "shield_enabled": True,
        "algorithm_family": "PPO",
        "architecture_change": False,
        "probability_floor": False,
        "online_teacher": False,
        "uid_or_soc_override": False,
    }
    for key, expected in expected_bindings.items():
        _require(bindings.get(key) == expected, f"V7 binding changed: {key}")
    environment = value.get("environment") or {}
    _require(environment.get("scenarios") == ["flow6", "dense24", "flow12"],
             "V7 scenario order changed")
    _require(environment.get("full_scenario_horizon_seconds") ==
             {"flow6": 1210, "flow12": 1570, "dense24": 1800},
             "V7 scenario horizons changed")
    optimizer = value.get("optimizer") or {}
    expected_optimizer = {
        "trainable_scope": ["gate_adapter", "critic_encoder", "critic_head"],
        "frozen_scope": ["actor_encoder", "gate_head", "proposal_head", "gate_temperature"],
        "learning_rate": 3e-5, "fresh_adam": True, "update_epochs": 2,
        "minibatch_size": 32, "gamma": .99, "gae_lambda": .95, "clip_ratio": .1,
        "value_coef": .5, "entropy_coef": .001, "target_rollout_kl": .01,
        "maximum_approx_kl": .05, "reference_gate_kl_coefficient": .1,
        "reference_gate_kl_soft_target": .02, "reference_gate_kl_soft_gain": 400.0,
        "cumulative_reference_weighted_kl_hard_cap": .05,
        "reference_gate_kl_direction": "q_source_to_p_current",
        "reference_gate_kl_scope": "gate_distribution_only",
        "reference_gate_is_fixed": True,
    }
    for key, expected in expected_optimizer.items():
        actual = optimizer.get(key)
        if isinstance(expected, float):
            _require(float(actual) == expected, f"V7 optimizer binding changed: {key}")
        else:
            _require(actual == expected, f"V7 optimizer binding changed: {key}")
    regularization = value.get("reference_gate_regularization") or {}
    _require(regularization.get("enabled") is True, "V7 reference regularization is disabled")
    _require(float(regularization.get("soft_target")) == .02 and
             float(regularization.get("soft_gain")) == 400.0,
             "V7 soft penalty binding changed")
    formal = value.get("formal_schedule") or {}
    _require(formal.get("independent_runs") == 3 and formal.get("episodes_per_run") == 240,
             "V7 formal schedule changed")
    _require(formal.get("episode_pattern") == list(EPISODE_PATTERN),
             "V7 episode pattern changed")
    _require(formal.get("pattern_repetitions") == 60, "V7 pattern repetitions changed")
    _require(formal.get("development", {}).get("seeds") ==
             [873101, 873102, 873103, 873104, 873105, 873106],
             "V7 development seeds changed")
    _require((value.get("heldout_reservation") or {}).get("run_in_this_task") is False,
             "V7 heldout reservation changed")
    _require((value.get("post_training_evaluation") or {}).get("run_in_this_task") is True,
             "V7 post-training evaluation plan must be enabled after formal review")
    adaptive = value.get("adaptive_probe") or {}
    _require(adaptive.get("primary_admission_required") is True,
             "V7 adaptive probe admission binding missing")
    return value


def _protocol_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _runtime_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return v6._runtime_protocol(protocol)


def _build_configs(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return v6._build_configs(protocol)


def _load_anchor_data(protocol: Mapping[str, Any]):
    return v6._load_anchor_data(protocol)


def _reference(policy: Any, old_rows: Sequence[Mapping[str, Any]],
               coverage_rows: Sequence[Mapping[str, Any]], weights: np.ndarray,
               protocol: Mapping[str, Any]) -> GateReference:
    rows = [*old_rows, *coverage_rows]
    observations, _ = v6.coverage._rows_to_tensors(rows)
    optimizer = protocol["optimizer"]
    reference = GateReference.from_policy(
        policy, observations, weights,
        coefficient=float(optimizer["reference_gate_kl_coefficient"]),
        max_kl=float(optimizer["cumulative_reference_weighted_kl_hard_cap"]),
        soft_kl_target=float(optimizer["reference_gate_kl_soft_target"]),
        soft_kl_gain=float(optimizer["reference_gate_kl_soft_gain"]),
    )
    _require(len(reference.observations) == len(rows), "V7 reference row count changed")
    return reference


def _schedule(protocol: Mapping[str, Any], run_index: int) -> tuple[list[int], list[str]]:
    _require(run_index in RUN_INDICES, "run-index must be 0, 1, or 2")
    raw = protocol["formal_schedule"]["train_seeds"].get(run_index)
    if raw is None:
        raw = protocol["formal_schedule"]["train_seeds"].get(str(run_index))
    _require(isinstance(raw, Mapping), f"V7 train seeds missing for run {run_index}")
    seeds = list(range(int(raw["start"]), int(raw["end"]) + 1))
    scenarios = list(EPISODE_PATTERN) * 60
    _require(len(seeds) == len(scenarios) == 240, "V7 schedule length changed")
    return seeds, scenarios


def _validate_schedule(protocol: Mapping[str, Any]) -> None:
    schedules = [_schedule(protocol, index)[0] for index in RUN_INDICES]
    _require(schedules == [list(range(870001, 870241)), list(range(871001, 871241)),
                           list(range(872001, 872241))], "V7 training seed schedule changed")
    _require(not (set(schedules[0]) & set(schedules[1]) or
                  set(schedules[0]) & set(schedules[2]) or
                  set(schedules[1]) & set(schedules[2])), "V7 training seeds overlap")
    development = set(int(seed) for seed in protocol["formal_schedule"]["development"]["seeds"])
    _require(not any(development.intersection(seeds) for seeds in schedules),
             "V7 development/training seeds overlap")
    heldout = set(protocol["heldout_reservation"]["existing_heldout_v5"])
    heldout.update(protocol["heldout_reservation"]["historical_heldout"])
    _require(not any(heldout.intersection(seeds) for seeds in schedules),
             "V7 heldout/training seeds overlap")
    _require(not heldout.intersection(development), "V7 heldout/development seeds overlap")


def _validate_admission(path: Path, protocol_sha256: str) -> dict[str, Any]:
    report = _read_json(path)
    _require(report.get("stage") == "P3_ADAPTIVE_KL_FORMAL_ADMISSION_V7",
             "admission stage is not V7")
    _require(report.get("FORMAL_P3_ALLOWED") is True, "V7 formal admission is false")
    source_sha = report.get("initial_checkpoint_sha256", report.get("source_checkpoint_sha256"))
    _require(source_sha == SOURCE_CHECKPOINT_SHA256, "V7 admission source SHA mismatch")
    _require(report.get("protocol_sha256") == protocol_sha256,
             "V7 admission protocol SHA mismatch")
    manifests = report.get("adaptive_probe_manifests")
    _require(isinstance(manifests, list) and len(manifests) == 2,
             "V7 admission must bind two adaptive probe manifests")
    checked: list[dict[str, Any]] = []
    run_indices: list[int] = []
    for entry in manifests:
        _require(isinstance(entry, Mapping), "adaptive probe manifest binding must be an object")
        path_value = entry.get("path", entry.get("manifest_path"))
        expected_sha = entry.get("sha256", entry.get("manifest_sha256"))
        _require(path_value and expected_sha, "adaptive probe path/SHA is missing")
        manifest_path = Path(str(path_value))
        _require(manifest_path.is_file(), f"adaptive probe manifest missing: {manifest_path}")
        actual_sha = _sha256(manifest_path)
        _require(actual_sha == expected_sha, "adaptive probe manifest SHA mismatch")
        manifest = _read_json(manifest_path)
        _require(manifest.get("stage") == "P3_ADAPTIVE_KL_PROBE_V1",
                 "adaptive probe manifest stage is not V1")
        try:
            probe_run_index = int(manifest.get("run_index"))
        except (TypeError, ValueError) as exc:
            raise V7InputError("adaptive probe manifest run_index is missing") from exc
        _require(probe_run_index in {0, 2},
                 "adaptive probe manifest run_index must be 0 or 2")
        run_indices.append(probe_run_index)
        probe_source_sha = (manifest.get("source_checkpoint_sha256") or
                            manifest.get("initial_checkpoint_sha256"))
        _require(probe_source_sha == SOURCE_CHECKPOINT_SHA256,
                 "adaptive probe source SHA mismatch")
        soft = manifest.get("reference_soft_settings") or {}
        options = manifest.get("reference_options") or {}
        soft_target = soft.get("soft_kl_target", options.get("soft_kl_target"))
        soft_gain = soft.get("soft_kl_gain", options.get("soft_kl_gain"))
        try:
            soft_matches = float(soft_target) == .02 and float(soft_gain) == 400.0
            coefficient = float(manifest.get("reference_gate_coefficient", .1))
            max_kl = float(manifest.get("reference_gate_max_kl", .05))
        except (TypeError, ValueError) as exc:
            raise V7InputError("adaptive probe soft KL settings are invalid") from exc
        _require(soft_matches, "adaptive probe soft KL settings mismatch")
        _require(coefficient == .1, "adaptive probe reference coefficient mismatch")
        _require(max_kl == .05, "adaptive probe reference cap mismatch")
        _require(manifest.get("status") == "COMPLETED", "adaptive probe is not completed")
        checked.append({"path": str(manifest_path), "sha256": actual_sha,
                        "status": manifest.get("status"), "stage": manifest.get("stage"),
                        "run_index": probe_run_index,
                        "source_checkpoint_sha256": probe_source_sha,
                        "reference_soft_settings": {
                            "soft_kl_target": float(soft_target),
                            "soft_kl_gain": float(soft_gain),
                        },
                        "reference_gate_coefficient": coefficient,
                        "reference_gate_max_kl": max_kl})
    _require(set(run_indices) == {0, 2} and len(run_indices) == 2,
             "adaptive probe manifests must cover run indices 0 and 2 exactly once")
    return {"decision": report, "adaptive_probe_manifests": checked}


def _preflight(*, checkpoint: Path, output: Path, run_index: int,
               admission_report: Path) -> dict[str, Any]:
    protocol = _protocol()
    _validate_schedule(protocol)
    protocol_sha = _sha256(PROTOCOL)
    authorization = _validate_admission(admission_report, protocol_sha)
    configs, snapshots = _build_configs(protocol)
    policy, payload, trainable, loaded_sha = v6._load_source(checkpoint, configs["dense24"], protocol)
    old_rows, coverage_rows, weights, weight_report = _load_anchor_data(protocol)
    reference = _reference(policy, old_rows, coverage_rows, weights, protocol)
    policy.gate_reference = reference
    return {
        "protocol": protocol, "protocol_sha256": protocol_sha,
        "protocol_digest": _protocol_digest(protocol), "runtime_protocol": _runtime_protocol(protocol),
        "configs": configs, "config_snapshots": snapshots, "checkpoint": checkpoint,
        "run_index": run_index, "policy": policy, "source_payload": payload,
        "trainable": trainable, "loaded_sha": loaded_sha, "old_rows": old_rows,
        "coverage_rows": coverage_rows, "weights": weights, "weight_report": weight_report,
        "reference": reference, "reference_digest": v6._reference_digest(reference),
        "authorization": authorization, "output": output,
    }


def _check_only_payload(context: Mapping[str, Any]) -> dict[str, Any]:
    reference = context["reference"]
    return {
        "stage": STAGE, "status": "CHECK_ONLY_OK", "check_only": True,
        "run_index": context["run_index"], "output_created": False,
        "output_existed": Path(context["output"]).exists(),
        "protocol_sha256": context["protocol_sha256"],
        "protocol_digest": context["protocol_digest"],
        "source_checkpoint_sha256": context["loaded_sha"],
        "source_checkpoint_kind": SOURCE_CHECKPOINT_KIND,
        "reference_digest": context["reference_digest"],
        "reference_gate_coefficient": reference.coefficient,
        "reference_gate_soft_target": reference.soft_kl_target,
        "reference_gate_soft_gain": reference.soft_kl_gain,
        "reference_gate_max_kl": reference.max_kl,
        "trainable_parameters": list(context["trainable"]),
        "fresh_adam": len(context["policy"].optimizer.state) == 0,
        "anchor_rows": len(context["old_rows"]) + len(context["coverage_rows"]),
        "adaptive_probe_manifests": context["authorization"]["adaptive_probe_manifests"],
        "heldout_executed": False, "P3_QUALIFIED": False, "P4_ALLOWED": False,
        "PPO_ALLOWED": False, "formal_launch_authorized": True,
    }


def _gate_adapter_l2_delta(policy: Any, source_payload: Mapping[str, Any]) -> float:
    source_state = source_payload.get("model")
    _require(isinstance(source_state, Mapping), "source model state is missing")
    squared = torch.zeros((), dtype=torch.float64)
    for name, current in policy.model.state_dict().items():
        if name == "gate_adapter" or name.startswith("gate_adapter."):
            _require(name in source_state, f"source gate adapter tensor missing: {name}")
            source = torch.as_tensor(source_state[name], dtype=current.dtype, device=current.device)
            squared = squared + (current.detach().to(torch.float64) - source.detach().to(torch.float64)).square().sum()
    return float(torch.sqrt(squared).cpu())


def _gate_adapter_state(policy: Any) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in policy.model.state_dict().items()
        if name == "gate_adapter" or name.startswith("gate_adapter.")
    }


def _gate_adapter_delta(before: Mapping[str, torch.Tensor], policy: Any) -> float:
    after = _gate_adapter_state(policy)
    _require(set(before) == set(after), "gate adapter state names changed")
    squared = torch.zeros((), dtype=torch.float64)
    for name in before:
        squared = squared + (after[name].to(torch.float64) - before[name].to(torch.float64)).square().sum()
    return float(torch.sqrt(squared))


def _rng_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    def equal(a: Any, b: Any) -> bool:
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, np.ndarray):
            return isinstance(b, np.ndarray) and np.array_equal(a, b)
        if isinstance(a, Mapping):
            return (isinstance(b, Mapping) and set(a) == set(b)
                    and all(equal(a[key], b[key]) for key in a))
        if isinstance(a, (tuple, list)):
            return (type(a) is type(b) and len(a) == len(b)
                    and all(equal(x, y) for x, y in zip(a, b)))
        return bool(a == b)

    return equal(left, right)


def _save_and_restore(context: Mapping[str, Any], checkpoint: Path, config: Any,
                      *, episode_index: int, timestep: int, ppo_called: bool,
                      counters: Mapping[str, Any] | None = None,
                      gate_adapter_parameter_delta_norm: float | None = None,
                      ) -> tuple[Any, bool, bool]:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    reference = context["reference"]
    saved_rng = v6._rng_state()
    save_checkpoint(
        checkpoint, context["policy"], config, stage=STAGE, formal=True,
        development_only=True, ppo_called=bool(ppo_called),
        protocol_sha256=context["protocol_sha256"], protocol_digest=context["protocol_digest"],
        source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
        initial_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
        checkpoint_kind=SOURCE_CHECKPOINT_KIND, checkpoint_promotable=False,
        P3_QUALIFIED=False, P4_ALLOWED=False, PPO_ALLOWED=False,
        episode_index=int(episode_index), timestep=int(timestep),
        trainable_parameters=list(context["trainable"]), rng_state=saved_rng,
        source_reference_digest=context["reference_digest"],
        reference_gate_digest=context["reference_digest"],
        reference_gate_coefficient=float(reference.coefficient),
        reference_gate_soft_target=float(reference.soft_kl_target),
        reference_gate_soft_gain=float(reference.soft_kl_gain),
        reference_gate_max_kl=float(reference.max_kl),
        processed_updated_skipped_counters=dict(counters or {}),
        gate_adapter_parameter_delta_norm=gate_adapter_parameter_delta_norm,
    )
    restored = v6._restore_policy(checkpoint, config, context["protocol"], reference)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _require(payload.get("protocol_sha256") == context["protocol_sha256"],
             "checkpoint protocol SHA changed")
    _require(payload.get("source_reference_digest") == context["reference_digest"],
             "checkpoint source reference digest changed")
    _require(float(payload.get("reference_gate_soft_target")) == float(reference.soft_kl_target),
             "checkpoint soft target changed")
    _require(float(payload.get("reference_gate_soft_gain")) == float(reference.soft_kl_gain),
             "checkpoint soft gain changed")
    _require(payload.get("rng_state") is not None, "checkpoint RNG state is missing")
    restored_rng = v6._rng_state()
    rng_exact = _rng_equal(saved_rng, restored_rng)
    exact = bool(
        v6.admission._state_equal(context["policy"].model.state_dict(), restored.model.state_dict())
        and v6.admission._state_equal(context["policy"].optimizer.state_dict(), restored.optimizer.state_dict())
        and v6._reference_digest(restored.gate_reference) == context["reference_digest"]
        and rng_exact
    )
    return restored, exact, rng_exact


def _dev_evaluate(context: Mapping[str, Any], output: Path, checkpoint: Path,
                  *, accepted_updates: int, processed_episodes: int,
                  label: str) -> dict[str, Any]:
    dev = context["protocol"]["formal_schedule"]["development"]
    seeds = [int(value) for value in dev["seeds"]]
    scenarios = [str(value) for value in dev["scenarios"]]
    result = v6._dev_evaluate(
        policy=context["policy"], configs=context["configs"],
        runtime_protocol=context["runtime_protocol"], v6=context["protocol"],
        output=output, accepted_episodes=accepted_updates, checkpoint=checkpoint,
        seeds=seeds, scenarios=scenarios, protocol_sha256=context["protocol_sha256"],
    )
    result["label"] = label
    result["processed_episodes"] = int(processed_episodes)
    result["updated_episodes"] = int(accepted_updates)
    result["stage"] = f"{STAGE}_DEV"
    return result


def _dev_evaluate_preserving_training_rng(
    context: Mapping[str, Any], output: Path, checkpoint: Path, *,
    accepted_updates: int, processed_episodes: int, label: str,
) -> dict[str, Any]:
    training_rng = v6._rng_state()
    try:
        return _dev_evaluate(
            context, output, checkpoint,
            accepted_updates=accepted_updates,
            processed_episodes=processed_episodes,
            label=label,
        )
    finally:
        v6._restore_rng_state(training_rng)


def _recovery(context: Mapping[str, Any], record: Mapping[str, Any], config: Any) -> dict[str, Any]:
    result = recovery.recover_update(
        context["policy"], record.get("_steps", []), config,
        str(record.get("action_tape")), context["runtime_protocol"], context["reference"],
        context["old_rows"], context["coverage_rows"],
    )
    status = str(result.get("status"))
    _require(status in RECOVERY_STATUSES, f"unknown recovery status: {status!r}")
    return result


def _training_run(context: dict[str, Any], output: Path) -> dict[str, Any]:
    run_index = int(context["run_index"])
    seeds, scenarios = _schedule(context["protocol"], run_index)
    started_wall = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    dev_evaluations: list[dict[str, Any]] = []
    current_checkpoint = Path(context["checkpoint"])
    checkpoints: list[str] = []
    timestep = 0
    processed = updated = skipped = hard_failed = 0
    accepted_minibatches = optimizer_steps = 0
    consecutive_skips = 0
    stop_reason: str | None = None
    initial_seed = int(seeds[0])
    torch.manual_seed(initial_seed)
    np.random.seed(initial_seed)
    random.seed(initial_seed)
    before = _dev_evaluate_preserving_training_rng(
        context, output / "dev_before_training", current_checkpoint,
        accepted_updates=0, processed_episodes=0, label="before",
    )
    dev_evaluations.append(before)
    if before.get("hard_failure") or not before.get("full_resolution"):
        stop_reason = "baseline development full-resolution or hard failure"
    for episode_index, (seed, scenario) in enumerate(zip(seeds, scenarios), start=1):
        if stop_reason is not None:
            break
        config = context["configs"][scenario]
        horizon = int(round(float(config.max_simulation_seconds) / float(config.dt)))
        episode_started = time.perf_counter()
        record = v6.admission._record_rollout(
            policy=context["policy"], config=config, p=context["runtime_protocol"],
            scenario=scenario, seed=int(seed),
            output=output / "episodes" / f"episode_{episode_index:04d}_{scenario}_{seed}",
            label=f"formal_v7_{run_index}", deterministic=False, max_steps=horizon,
            require_full_resolution=False, checkpoint=current_checkpoint,
        )
        record.update({
            "run_index": run_index, "episode_index": episode_index, "seed": int(seed),
            "scenario": scenario, "source_checkpoint_sha256": context["loaded_sha"],
            "source_reference_digest": context["reference_digest"],
            "protocol_sha256": context["protocol_sha256"],
            "timing_rollout_wall_seconds": float(time.perf_counter() - episode_started),
        })
        processed += 1
        hard_pass = bool(v6._training_pass(record))
        record["rollout_hard_pass"] = hard_pass
        record["ppo_called"] = bool(hard_pass)
        recovery_result: dict[str, Any]
        if not hard_pass:
            hard_failed += 1
            consecutive_skips = 0
            recovery_result = {"status": "HARD_FAILED", "reason": "rollout hard gate"}
            stop_reason = "rollout hard failure"
        else:
            gate_before = _gate_adapter_state(context["policy"])
            try:
                recovery_result = _recovery(context, record, config)
            except Exception as exc:
                recovery_result = {"status": "HARD_FAILED", "exception": type(exc).__name__,
                                   "message": str(exc)}
            status = str(recovery_result.get("status"))
            if status not in {"UPDATED", "SKIPPED_KL"}:
                hard_failed += 1
                consecutive_skips = 0
                stop_reason = "recovery hard failure"
            else:
                gate_delta = _gate_adapter_delta(gate_before, context["policy"])
                record["gate_adapter_parameter_delta_norm"] = gate_delta
                next_updated = updated + int(status == "UPDATED")
                next_skipped = skipped + int(status == "SKIPPED_KL")
                next_counters = {
                    "processed": processed,
                    "updated": next_updated,
                    "skipped": next_skipped,
                }
                next_timestep = timestep + len(record.get("_steps", []))
                checkpoint_file = output / "checkpoints" / f"checkpoint_episode_{episode_index:04d}.pt"
                try:
                    restored, exact, rng_exact = _save_and_restore(
                        context, checkpoint_file, config, episode_index=episode_index,
                        timestep=next_timestep, ppo_called=True,
                        counters=next_counters,
                        gate_adapter_parameter_delta_norm=gate_delta,
                    )
                    record["checkpoint_after"] = str(checkpoint_file)
                    record["checkpoint_after_sha256"] = _sha256(checkpoint_file)
                    record["checkpoint_restore_exact"] = bool(exact)
                    record["checkpoint_rng_restore_exact"] = bool(rng_exact)
                except Exception as exc:
                    restored, exact, rng_exact = context["policy"], False, False
                    record["checkpoint_restore_exact"] = False
                    record["checkpoint_rng_restore_exact"] = False
                    record["checkpoint_error"] = {"type": type(exc).__name__, "message": str(exc)}
                if exact:
                    context["policy"] = restored
                    current_checkpoint = checkpoint_file
                    timestep = next_timestep
                    checkpoints.append(str(checkpoint_file))
                    update = recovery_result.get("update") or {}
                    accepted_minibatches += int(update.get("accepted_updates", 0) or 0)
                    optimizer_steps += int(update.get("updates", update.get("accepted_updates", 0)) or 0)
                    gate_delta = _gate_adapter_l2_delta(context["policy"], context["source_payload"])
                    record["gate_adapter_l2_delta_from_source"] = gate_delta
                    if status == "UPDATED":
                        updated += 1
                        consecutive_skips = 0
                    else:
                        skipped += 1
                        consecutive_skips += 1
                        if consecutive_skips >= SKIP_STALL_LIMIT:
                            stop_reason = "learning stalled after 10 consecutive safe skips"
                else:
                    hard_failed += 1
                    stop_reason = "checkpoint save or restore failure"
        record["recovery"] = recovery_result
        record["recovery_status"] = str(recovery_result.get("status"))
        record["processed_updated_skipped"] = {
            "processed": processed, "updated": updated, "skipped": skipped,
        }
        record["timing_episode_wall_seconds"] = float(time.perf_counter() - episode_started)
        record["accepted_episode"] = bool(record.get("checkpoint_restore_exact") is True and
                                            record["recovery_status"] == "UPDATED")
        public = v6.formal_v5._public_record(record)
        records.append(public)
        _write_json(output / "progress.json", {
            "stage": STAGE, "run_index": run_index, "episodes_requested": len(seeds),
            "processed_episodes": processed, "updated_episodes": updated,
            "skipped_kl_episodes": skipped, "hard_failed_episodes": hard_failed,
            "accepted_minibatches": accepted_minibatches, "optimizer_steps": optimizer_steps,
            "records": records, "dev_evaluations": dev_evaluations,
            "stopped": stop_reason is not None, "stop_reason": stop_reason,
            "heldout_executed": False,
        })
        if stop_reason is not None:
            break
        if processed in {40, 80, 120, 160, 200, 240}:
            dev = _dev_evaluate_preserving_training_rng(
                context, output / f"dev_after_{processed:04d}", current_checkpoint,
                accepted_updates=updated, processed_episodes=processed,
                label=f"after_{processed:04d}",
            )
            dev_evaluations.append(dev)
            if dev.get("hard_failure") or not dev.get("full_resolution"):
                stop_reason = "development full-resolution or hard failure"
                break
    if stop_reason is None and processed == len(seeds):
        status = "COMPLETED"
    elif stop_reason and stop_reason.startswith("learning stalled"):
        status = "LEARNING_STALLED"
    else:
        status = "FAILED"
    final_checkpoint = checkpoints[-1] if checkpoints else None
    final_checkpoint_sha256 = _sha256(final_checkpoint) if final_checkpoint else None
    manifest = {
        "stage": STAGE, "status": status, "run_index": run_index,
        "output": str(output), "protocol_sha256": context["protocol_sha256"],
        "protocol_digest": context["protocol_digest"],
        "source_checkpoint_sha256": context["loaded_sha"],
        "source_reference_digest": context["reference_digest"],
        "reference_gate_coefficient": context["reference"].coefficient,
        "reference_gate_soft_target": context["reference"].soft_kl_target,
        "reference_gate_soft_gain": context["reference"].soft_kl_gain,
        "reference_gate_max_kl": context["reference"].max_kl,
        "train_seeds": seeds, "episode_pattern": scenarios,
        "episodes_requested": len(seeds), "processed_episodes": processed,
        "updated_episodes": updated, "skipped_kl_episodes": skipped,
        "hard_failed_episodes": hard_failed, "accepted_episodes": updated,
        "accepted_minibatches": accepted_minibatches, "optimizer_steps": optimizer_steps,
        "effective_learning": {
            "updated_ratio": float(updated / processed) if processed else 0.0,
            "zero_gate_changes": int(sum(
                row.get("recovery_status") in {"UPDATED", "SKIPPED_KL"}
                and row.get("checkpoint_restore_exact") is True
                and "gate_adapter_parameter_delta_norm" in row
                and float(row["gate_adapter_parameter_delta_norm"]) == 0.0
                for row in records
            )),
            "accepted_minibatches": accepted_minibatches,
            "optimizer_steps": optimizer_steps,
            "processed": processed, "updated": updated, "skipped": skipped,
        },
        "records": records, "dev_evaluations": dev_evaluations,
        "dev_before": before, "dev_after": dev_evaluations[-1] if len(dev_evaluations) > 1 else None,
        "checkpoints": checkpoints, "final_checkpoint": final_checkpoint,
        "final_checkpoint_sha256": final_checkpoint_sha256,
        "actual_config_snapshots": context["config_snapshots"],
        "stop_reason": stop_reason, "heldout_executed": False,
        "P3_QUALIFIED": False, "P4_ALLOWED": False, "PPO_ALLOWED": False,
        "checkpoint_promotable": False, "automatic_restart": False,
        "timing_wall_seconds": float(time.perf_counter() - started_wall),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def run(*, run_index: int, output: str | Path, checkpoint: str | Path,
        admission_report: str | Path, check_only: bool = False) -> dict[str, Any]:
    _require(run_index in RUN_INDICES, "run-index must be 0, 1, or 2")
    output_path = Path(output)
    if not check_only:
        _require(not output_path.exists(), f"V7 output must be fresh: {output_path}")
    context = _preflight(
        checkpoint=Path(checkpoint), output=output_path, run_index=run_index,
        admission_report=Path(admission_report),
    )
    if check_only:
        return _check_only_payload(context)
    return _training_run(context, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=int, choices=RUN_INDICES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--admission-report", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(
            run_index=args.run_index, output=args.output, checkpoint=args.checkpoint,
            admission_report=args.admission_report, check_only=args.check_only,
        )
    except (V7InputError, FileNotFoundError, ValueError) as exc:
        print(json.dumps({"stage": STAGE, "status": "ERROR", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True), flush=True)
    return 0 if result.get("status") in {"CHECK_ONLY_OK", "COMPLETED"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
