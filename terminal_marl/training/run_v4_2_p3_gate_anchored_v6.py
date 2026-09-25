"""Bounded V6 PPO runner with a fixed source gate reference distribution.

The runner is deliberately small and keeps the existing V4.2 rollout, replay,
likelihood, safety and checkpoint helpers as the execution boundary.  The V6
YAML owns the schedule and optimizer values; the older admission YAML is used
only by the reviewed rollout helper, which expects its lifecycle and hard-gate
field names.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATED_ARCHITECTURE_ID,
    CALIBRATION_TEMPERATURE,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.algo.gate_reference_v4_2 import GateReference
from terminal_marl.core.checkpoints import load_checkpoint, save_checkpoint
from terminal_marl.training import (
    evaluate_random_soc_gate_repair_v1 as base,
)
from terminal_marl.training import (
    run_v4_2_p3_controlled_emergency_admission_v5 as admission,
)
from terminal_marl.training import (
    run_v4_2_p3_controlled_emergency_formal_v5 as formal_v5,
)
from terminal_marl.training import (
    run_v4_2_p3_mixed_priority_soc_training_v3 as p3_legacy,
)
from terminal_marl.training import (
    train_p3_gate_coverage_repair_v2 as coverage,
)


STAGE = "P3_GATE_ANCHORED_V6"
RUNNER_MODULE = "terminal_marl.training.run_v4_2_p3_gate_anchored_v6"
PROTOCOL = Path("terminal_marl/configs/low_v4_2/p3_gate_anchored_v6.yaml")
SOURCE_CHECKPOINT_SHA256 = (
    "03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba"
)
SOURCE_CHECKPOINT_KIND = "coverage_gate_adapter_fit"
OBSERVATION_DIM = 88
RUN_INDICES = (0, 1, 2)
EPISODE_PATTERN = ("flow6", "dense24", "flow12", "dense24")
TRAINABLE_PREFIXES = ("gate_adapter.", "critic_encoder.", "critic_head.")
FROZEN_PREFIXES = ("actor_encoder.", "gate_head.", "proposal_head.")
ALLOWED_OUTCOMES = frozenset(
    {"completed", "emergency_landed", "failed_energy", "unresolved"}
)


class V6InputError(ValueError):
    """Raised when a V6 input violates a frozen binding."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V6InputError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return _jsonable(value.value)
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
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    _require(target.is_file(), f"JSON input missing: {target}")
    value = json.loads(target.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {target}")
    return value


def _protocol() -> dict[str, Any]:
    """Read the V6 YAML and validate numeric/schedule bindings."""

    _require(PROTOCOL.is_file(), f"V6 protocol missing: {PROTOCOL}")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, Mapping), "V6 protocol must be a mapping")
    value = dict(value)
    _require(value.get("protocol") == "V4_2_P3_GATE_ANCHORED_V6", "V6 protocol id changed")
    lineage = value.get("lineage") or {}
    _require(lineage.get("source_checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256,
             "V6 source checkpoint SHA changed")
    _require(lineage.get("source_checkpoint_kind") == SOURCE_CHECKPOINT_KIND,
             "V6 source checkpoint kind changed")
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
        _require(bindings.get(key) == expected, f"V6 binding changed: {key}")
    environment = value.get("environment") or {}
    _require(environment.get("scenarios") == ["flow6", "dense24", "flow12"],
             "V6 scenario order changed")
    _require(environment.get("full_scenario_horizon_seconds") ==
             {"flow6": 1210, "flow12": 1570, "dense24": 1800},
             "V6 scenario horizons changed")
    optimizer = value.get("optimizer") or {}
    for key, expected in {
        "trainable_scope": ["gate_adapter", "critic_encoder", "critic_head"],
        "frozen_scope": ["actor_encoder", "gate_head", "proposal_head", "gate_temperature"],
        "learning_rate": 3e-5,
        "fresh_adam": True,
        "update_epochs": 2,
        "minibatch_size": 32,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.1,
        "value_coef": 0.5,
        "entropy_coef": 0.001,
        "target_rollout_kl": 0.01,
        "maximum_approx_kl": 0.05,
        "reference_gate_kl_coefficient": 0.1,
        "cumulative_reference_weighted_kl_hard_cap": 0.05,
        "reference_gate_kl_direction": "q_source_to_p_current",
        "reference_gate_kl_scope": "gate_distribution_only",
        "reference_gate_is_fixed": True,
    }.items():
        actual = optimizer.get(key)
        if isinstance(expected, float):
            _require(float(actual) == expected, f"V6 optimizer binding changed: {key}")
        else:
            _require(actual == expected, f"V6 optimizer binding changed: {key}")
    _require((value.get("reference_gate_regularization") or {}).get("enabled") is True,
             "V6 reference regularization is disabled")
    formal = value.get("formal_schedule") or {}
    _require(formal.get("independent_runs") == 3 and formal.get("episodes_per_run") == 240,
             "V6 formal run count changed")
    _require(formal.get("run_indices") == [0, 1, 2], "V6 formal run indices changed")
    _require(formal.get("episode_pattern") == list(EPISODE_PATTERN),
             "V6 formal episode pattern changed")
    _require(formal.get("pattern_repetitions") == 60, "V6 formal repetitions changed")
    _require((value.get("checkpointing") or {}).get("save_after_each_accepted_episode") is True,
             "V6 checkpoint cadence changed")
    _require((value.get("heldout_reservation") or {}).get("run_in_this_task") is False,
             "V6 heldout reservation changed")
    return value


def _protocol_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _runtime_protocol(v6: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt only the optimizer aliases needed by the reviewed V5 helper."""

    runtime = copy.deepcopy(admission.protocol())
    optimizer = dict(runtime.get("optimizer") or {})
    source_optimizer = v6["optimizer"]
    optimizer.update({
        "learning_rate": float(source_optimizer["learning_rate"]),
        "update_epochs": int(source_optimizer["update_epochs"]),
        "minibatch_size": int(source_optimizer["minibatch_size"]),
        "gamma": float(source_optimizer["gamma"]),
        "gae_lambda": float(source_optimizer["gae_lambda"]),
        "clip_ratio": float(source_optimizer["clip_ratio"]),
        "value_coef": float(source_optimizer["value_coef"]),
        "entropy_coef": float(source_optimizer["entropy_coef"]),
        "target_kl": float(source_optimizer["target_rollout_kl"]),
        "maximum_approx_kl": float(source_optimizer["maximum_approx_kl"]),
        # These are the existing update-gate boundary used by the V5/V3 helper.
        "minimum_accepted_updates": 1,
        "maximum_rejected_updates": 1,
    })
    runtime["optimizer"] = optimizer
    return runtime


def _build_configs(v6: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    base_protocol = base._load_yaml_protocol()
    configs: dict[str, Any] = {}
    for scenario in ("flow6", "flow12", "dense24"):
        config = admission._build_config(scenario, base_protocol)
        _require(hasattr(config, "v4_2_rejected_release_hold_mode"),
                 "execution config lacks canonical WAIT field")
        config = replace(config, v4_2_rejected_release_hold_mode="canonical_wait")
        _require(str(config.v4_2_rejected_release_hold_mode) == "canonical_wait",
                 "canonical WAIT mode was not applied")
        _require(not bool(getattr(config, "teacher_enabled", False)), "Teacher enabled")
        _require(str(getattr(config, "scheduler_mode", "")) == "execution",
                 "execution scheduler changed")
        _require(not bool(getattr(config, "reconcile_late_merge_slots_enabled", False)),
                 "late-slot reconciliation enabled")
        expected_horizon = float(v6["environment"]["full_scenario_horizon_seconds"][scenario])
        _require(float(config.max_simulation_seconds) == expected_horizon,
                 f"{scenario} horizon changed")
        _require(int(getattr(config, "observation_dimension", OBSERVATION_DIM)) == OBSERVATION_DIM,
                 "observation dimension changed")
        configs[scenario] = config
    snapshots = {name: base._config_snapshot(config) for name, config in configs.items()}
    return configs, snapshots


def _trainable_names(policy: Any) -> tuple[str, ...]:
    names: list[str] = []
    for name, parameter in policy.model.named_parameters():
        enabled = name.startswith(TRAINABLE_PREFIXES)
        parameter.requires_grad_(enabled)
        if enabled:
            names.append(name)
    _require(bool(names), "V6 trainable parameter scope is empty")
    _require(all(name.startswith(TRAINABLE_PREFIXES) for name in names),
             "V6 trainable parameter scope changed")
    return tuple(names)


def _load_source(checkpoint: str | Path, config: Any, v6: Mapping[str, Any]):
    path = Path(checkpoint)
    _require(path.is_file(), f"source checkpoint missing: {path}")
    actual_sha = _sha256(path)
    _require(actual_sha == SOURCE_CHECKPOINT_SHA256,
             f"source checkpoint SHA mismatch: {actual_sha}")
    optimizer = v6["optimizer"]
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM,
        temperature=CALIBRATION_TEMPERATURE,
        learning_rate=float(optimizer["learning_rate"]),
    )
    payload = load_checkpoint(path, policy, config, load_optimizer=False)
    source_state = payload.get("model") if isinstance(payload, Mapping) else None
    _require(isinstance(source_state, Mapping), "source model state is missing")
    source_digest = _tensor_digest(source_state)
    _require(getattr(policy.model, "architecture_id", None) == CALIBRATED_ARCHITECTURE_ID,
             "source architecture changed")
    _require(float(policy.model.gate_temperature.detach().cpu().item()) ==
             float(CALIBRATION_TEMPERATURE), "source gate temperature changed")
    _require(len(policy.optimizer.state) == 0, "source optimizer state was restored")
    _require(_tensor_digest(policy.model.state_dict()) == source_digest,
             "source model weights do not match checkpoint payload")
    names = _trainable_names(policy)
    policy.optimizer = torch.optim.Adam(
        [parameter for parameter in policy.model.parameters() if parameter.requires_grad],
        lr=float(optimizer["learning_rate"]),
    )
    _require(len(policy.optimizer.state) == 0, "fresh Adam has state before update")
    policy.model.eval()
    return policy, payload, names, actual_sha


def _path_list(group: Mapping[str, Any]) -> list[str | Path]:
    files = group.get("files") or group.get("paths") or []
    result: list[str | Path] = []
    for item in files:
        if isinstance(item, Mapping):
            item = item.get("path")
        if item:
            result.append(item)
    return result


def _load_anchor_data(v6: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    groups = v6.get("anchor_data", {}).get("groups", {})
    old_group = groups.get("old_train_anchor") or {}
    coverage_group = groups.get("coverage_original_bounded_run0_run1") or {}
    old_paths = _path_list(old_group)
    coverage_paths = _path_list(coverage_group)
    _require(old_paths and coverage_paths, "V6 anchor label paths are missing")
    old_rows = coverage._load_rows(old_paths, split="train")
    coverage_rows = coverage._load_rows(coverage_paths, split="train")
    if "combined_rows" in old_group:
        _require(len(old_rows) == int(old_group["combined_rows"]),
                 "V6 old anchor row count changed")
    if "combined_rows" in coverage_group:
        _require(len(coverage_rows) == int(coverage_group["combined_rows"]),
                 "V6 coverage anchor row count changed")
    weights, report = coverage.build_sample_weights(old_rows, coverage_rows)
    _require(len(weights) == len(old_rows) + len(coverage_rows),
             "V6 sample weights do not match anchor rows")
    return old_rows, coverage_rows, np.asarray(weights, dtype=np.float64), report


def _reference(policy: Any, old_rows: Sequence[Mapping[str, Any]],
               coverage_rows: Sequence[Mapping[str, Any]], weights: np.ndarray,
               v6: Mapping[str, Any]) -> GateReference:
    rows = [*old_rows, *coverage_rows]
    observations, _ = coverage._rows_to_tensors(rows)
    optimizer = v6["optimizer"]
    reference = GateReference.from_policy(
        policy, observations, weights,
        coefficient=float(optimizer["reference_gate_kl_coefficient"]),
        max_kl=float(optimizer["cumulative_reference_weighted_kl_hard_cap"]),
    )
    _require(len(reference.observations) == len(rows), "reference row count changed")
    return reference


def _tensor_digest(value: Any) -> str:
    digest = hashlib.sha256()
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            digest.update(str(key).encode("utf-8"))
            digest.update(_tensor_digest(value[key]).encode("ascii"))
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    else:
        digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _reference_digest(reference: GateReference) -> str:
    return _tensor_digest({
        "observations": reference.observations,
        "reference_logits": reference.reference_logits,
        "reference_probs": reference.reference_probs,
        "weights": reference.weights,
        "coefficient": reference.coefficient,
        "max_kl": reference.max_kl,
    })


def _rng_state() -> dict[str, Any]:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}


def _restore_rng_state(value: Mapping[str, Any] | None) -> None:
    if not isinstance(value, Mapping):
        return
    if value.get("torch") is not None:
        torch.set_rng_state(torch.as_tensor(value["torch"], dtype=torch.uint8).cpu())
    if value.get("numpy") is not None:
        numpy_state = tuple(value["numpy"])
        if len(numpy_state) >= 2 and not isinstance(numpy_state[1], np.ndarray):
            numpy_state = (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
        np.random.set_state(numpy_state)
    if value.get("python") is not None:
        random.setstate(tuple(value["python"]))


def _restore_policy(checkpoint: Path, config: Any, v6: Mapping[str, Any],
                    reference: GateReference):
    optimizer = v6["optimizer"]
    restored = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM,
        temperature=CALIBRATION_TEMPERATURE,
        learning_rate=float(optimizer["learning_rate"]),
    )
    _trainable_names(restored)
    restored.optimizer = torch.optim.Adam(
        [parameter for parameter in restored.model.parameters() if parameter.requires_grad],
        lr=float(optimizer["learning_rate"]),
    )
    payload = load_checkpoint(checkpoint, restored, config, load_optimizer=True)
    restored.gate_reference = reference
    _restore_rng_state(payload.get("rng_state") if isinstance(payload, Mapping) else None)
    restored.model.eval()
    return restored


def _save_and_restore(policy: Any, checkpoint: Path, config: Any,
                      v6: Mapping[str, Any], reference: GateReference,
                      *, episode_index: int, timestep: int,
                      trainable: Sequence[str], protocol_sha256: str,
                      formal: bool = True, ppo_called: bool = True) -> tuple[Any, bool]:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    source_digest = _reference_digest(reference)
    save_checkpoint(
        checkpoint, policy, config,
        stage=STAGE, formal=bool(formal), development_only=True,
        ppo_called=bool(ppo_called),
        protocol_sha256=protocol_sha256, source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
        initial_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
        checkpoint_kind=SOURCE_CHECKPOINT_KIND, checkpoint_promotable=False,
        P3_QUALIFIED=False, P4_ALLOWED=False, PPO_ALLOWED=False,
        episode_index=int(episode_index), timestep=int(timestep),
        trainable_parameters=list(trainable), rng_state=_rng_state(),
        reference_gate_digest=source_digest,
        reference_gate_coefficient=float(reference.coefficient),
        reference_gate_max_kl=float(reference.max_kl),
    )
    restored = _restore_policy(checkpoint, config, v6, reference)
    _require(
        bool(policy.optimizer.state) == bool(restored.optimizer.state),
        "restored Adam state presence changed",
    )
    exact = bool(
        admission._state_equal(policy.model.state_dict(), restored.model.state_dict())
        and admission._state_equal(policy.optimizer.state_dict(), restored.optimizer.state_dict())
        and _reference_digest(restored.gate_reference) == source_digest
    )
    return restored, exact


def _anchor_metrics(policy: Any, old_rows: Sequence[Mapping[str, Any]],
                    coverage_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "old_train_anchor": coverage._metrics(policy, old_rows),
        "coverage_original_bounded_run0_run1": coverage._metrics(policy, coverage_rows),
    }


def _update(policy: Any, steps: list[Any], config: Any, tape_path: str,
            runtime_protocol: Mapping[str, Any], reference: GateReference,
            old_rows: Sequence[Mapping[str, Any]],
            coverage_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, bool], dict[str, Any]]:
    before_model = copy.deepcopy(policy.model.state_dict())
    before_optimizer = copy.deepcopy(policy.optimizer.state_dict())
    before_reference = _reference_digest(reference)
    before_anchor = _anchor_metrics(policy, old_rows, coverage_rows)
    before_reference_metrics = reference.metrics(policy)
    policy.gate_reference = reference
    update, gates = p3_legacy._transactional_update(
        policy, steps, config, tape_path, runtime_protocol,
    )
    update = dict(update) if isinstance(update, Mapping) else {"value": update}
    gates = {str(key): bool(value) for key, value in (gates or {}).items()}
    after_reference_metrics = reference.metrics(policy)
    after_reference = _reference_digest(reference)
    gates["reference_source_fixed"] = bool(
        policy.gate_reference is reference and after_reference == before_reference
    )
    gates["reference_gate_kl_cap"] = bool(
        np.isfinite(float(after_reference_metrics.get("reference_kl", np.nan)))
        and float(after_reference_metrics.get("reference_kl", np.inf)) <= float(reference.max_kl)
    )
    gates["proposal_temperature_frozen"] = bool(
        p3_legacy._frozen_state_equal(before_model, policy.model.state_dict())
    )
    update["reference_gate_before"] = before_reference_metrics
    update["reference_gate_after"] = after_reference_metrics
    update["anchor_metrics_before"] = before_anchor
    update["anchor_metrics_after"] = _anchor_metrics(policy, old_rows, coverage_rows)
    if not all(gates.values()):
        policy.model.load_state_dict(before_model)
        policy.optimizer.load_state_dict(before_optimizer)
    return update, gates, {
        "reference_gate_before": before_reference_metrics,
        "reference_gate_after": after_reference_metrics,
        "anchor_metrics_before": before_anchor,
        "anchor_metrics_after": update["anchor_metrics_after"],
    }


def _training_pass(record: Mapping[str, Any]) -> bool:
    return bool(formal_v5._training_hard_pass(record))


def _dev_pass(record: Mapping[str, Any]) -> bool:
    return bool(formal_v5._dev_hard_pass(record))


def _dev_evaluate(*, policy: Any, configs: Mapping[str, Any], runtime_protocol: Mapping[str, Any],
                  v6: Mapping[str, Any], output: Path, accepted_episodes: int,
                  checkpoint: Path, seeds: Sequence[int], scenarios: Sequence[str],
                  protocol_sha256: str) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed, scenario in zip(seeds, scenarios):
        config = configs[str(scenario)]
        horizon = int(round(float(config.max_simulation_seconds) / float(config.dt)))
        record = admission._record_rollout(
            policy=policy, config=config, p=runtime_protocol, scenario=str(scenario),
            seed=int(seed), output=output / f"{scenario}_{int(seed)}",
            label=f"dev_{accepted_episodes:04d}", deterministic=False,
            max_steps=horizon, require_full_resolution=True, checkpoint=checkpoint,
        )
        records.append(formal_v5._public_record(record))
        if not _dev_pass(record):
            break
    complete = len(records) == len(tuple(seeds))
    result = {
        "stage": "P3_GATE_ANCHORED_V6_DEV",
        "accepted_episodes": int(accepted_episodes),
        "records": records,
        "complete": bool(complete),
        "full_resolution": bool(complete and all(_dev_pass(row) for row in records)),
        "hard_failure": bool(not complete or any(not _dev_pass(row) for row in records)),
        "same_initial_states": True,
        "stochastic": True,
        "heldout": False,
        "wall_seconds": float(time.perf_counter() - started),
    }
    _write_json(output / "manifest.json", result)
    return result


def _comparison(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = admission._performance_comparison(before, after)
        result["report_only"] = True
        result["review_required"] = True
        return result
    except Exception as exc:
        return {
            "paired_records": [], "report_only": True, "review_required": True,
            "available": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def _smoke_checks(record: Mapping[str, Any], steps: Sequence[Any], expected_steps: int) -> dict[str, bool]:
    checks = admission._smoke_contract(record, list(steps), expected_steps=expected_steps)
    # This task's smoke is intentionally partial; no full-resolution gate is used.
    return {str(key): bool(value) for key, value in checks.items()}


def _schedule(v6: Mapping[str, Any], run_index: int | None = None) -> tuple[list[int], list[str]]:
    if run_index is None:
        section = v6["pilot"]
        seeds_spec = section["train_seeds"]
        seeds = list(range(int(seeds_spec["start"]), int(seeds_spec["end"]) + 1))
        scenarios = list(section["episode_pattern"]) * int(section["pattern_repetitions"])
    else:
        section = v6["formal_schedule"]
        raw = section["train_seeds"].get(run_index, section["train_seeds"].get(str(run_index)))
        _require(isinstance(raw, Mapping), f"formal train seeds missing for run {run_index}")
        seeds = list(range(int(raw["start"]), int(raw["end"]) + 1))
        scenarios = list(section["episode_pattern"]) * int(section["pattern_repetitions"])
    _require(len(seeds) == len(scenarios), "V6 seed and scenario schedule lengths differ")
    return seeds, scenarios


def _validate_schedule(v6: Mapping[str, Any]) -> None:
    pilot_seeds, pilot_scenarios = _schedule(v6)
    _require(pilot_seeds == list(range(859201, 859213)), "V6 pilot seeds changed")
    _require(pilot_scenarios == list(EPISODE_PATTERN) * 3, "V6 pilot pattern changed")
    formal_sets = [set(_schedule(v6, index)[0]) for index in RUN_INDICES]
    _require(all(len(values) == 240 for values in formal_sets), "V6 formal seed count changed")
    _require(not (formal_sets[0] & formal_sets[1] or formal_sets[0] & formal_sets[2] or
                  formal_sets[1] & formal_sets[2]), "V6 formal seeds overlap")
    dev = set(int(value) for value in v6["formal_schedule"]["development"]["seeds"])
    _require(not any(dev & values for values in formal_sets), "V6 formal/dev seeds overlap")
    heldout = set((v6.get("heldout_reservation") or {}).get("existing_heldout_v5", ()))
    heldout |= set((v6.get("heldout_reservation") or {}).get("historical_heldout", ()))
    _require(not any(heldout & values for values in formal_sets), "V6 heldout/train seeds overlap")
    _require(not heldout.intersection(dev), "V6 heldout/dev seeds overlap")


def _validate_admission(path: Path, v6: Mapping[str, Any], protocol_sha256: str) -> dict[str, Any]:
    report = _read_json(path)
    _require(report.get("FORMAL_P3_ALLOWED") is True,
             "formal admission authorization is false")
    source_sha = report.get("initial_checkpoint_sha256", report.get("source_checkpoint_sha256"))
    _require(source_sha == SOURCE_CHECKPOINT_SHA256, "admission source checkpoint SHA mismatch")
    _require(report.get("protocol_sha256") == protocol_sha256,
             "admission protocol SHA mismatch: protocol_sha256")
    pilot_path_value = report.get("pilot_manifest_path")
    _require(pilot_path_value, "pilot manifest path missing from admission")
    pilot_path = Path(str(pilot_path_value))
    _require(pilot_path.is_file(), f"pilot manifest missing: {pilot_path}")
    pilot_sha = _sha256(pilot_path)
    _require(report.get("pilot_manifest_sha256") == pilot_sha,
             "pilot manifest SHA mismatch")
    pilot = _read_json(pilot_path)
    _require(pilot.get("stage") == f"{STAGE}_PILOT", "pilot manifest stage mismatch")
    _require(pilot.get("status") == "COMPLETED", "pilot manifest status is not completed")
    _require(pilot.get("PILOT_TECHNICAL_PASS") is True,
             "pilot technical pass is false")
    _require(pilot.get("protocol_sha256") == protocol_sha256,
             "pilot protocol SHA mismatch")
    pilot_source = pilot.get("selected_checkpoint_sha256", pilot.get("source_checkpoint_sha256"))
    _require(pilot_source == SOURCE_CHECKPOINT_SHA256, "pilot source checkpoint SHA mismatch")
    count = pilot.get("accepted_episodes", pilot.get("episodes_recorded", pilot.get("episodes_completed")))
    if count is None and isinstance(pilot.get("records"), list):
        count = len(pilot["records"])
    _require(count is not None, "pilot episode count is missing")
    _require(int(count) == 12, "pilot does not contain all 12 episodes")
    return {"decision": report, "pilot": pilot, "pilot_manifest_path": str(pilot_path)}


def _preflight(*, stage: str, checkpoint: Path, output: Path,
               run_index: int | None, admission_report: Path | None) -> dict[str, Any]:
    v6 = _protocol()
    _validate_schedule(v6)
    _require(stage in {"smoke", "pilot", "formal"}, "stage must be smoke, pilot, or formal")
    if stage == "formal":
        _require(run_index in RUN_INDICES, "formal run-index must be 0, 1, or 2")
        _require(admission_report is not None, "formal stage requires --admission-report")
    else:
        _require(run_index is None, "--run-index is only valid for formal stage")
        _require(admission_report is None, "--admission-report is only valid for formal stage")
    protocol_sha = _sha256(PROTOCOL)
    authorization = None
    if stage == "formal":
        authorization = _validate_admission(admission_report, v6, protocol_sha)
    configs, snapshots = _build_configs(v6)
    policy, payload, trainable, loaded_sha = _load_source(checkpoint, configs["dense24"], v6)
    old_rows, coverage_rows, weights, weight_report = _load_anchor_data(v6)
    reference = _reference(policy, old_rows, coverage_rows, weights, v6)
    policy.gate_reference = reference
    return {
        "stage": stage, "v6": v6, "runtime_protocol": _runtime_protocol(v6),
        "protocol_sha256": protocol_sha, "configs": configs, "config_snapshots": snapshots,
        "checkpoint": checkpoint,
        "run_index": run_index,
        "policy": policy, "source_payload": payload, "trainable": trainable,
        "loaded_sha": loaded_sha, "old_rows": old_rows, "coverage_rows": coverage_rows,
        "weights": weights, "weight_report": weight_report, "reference": reference,
        "reference_digest": _reference_digest(reference), "authorization": authorization,
        "output": output,
    }


def _check_only_payload(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": f"{STAGE}_{str(context['stage']).upper()}",
        "status": "CHECK_ONLY_OK",
        "check_only": True,
        "run_index": context.get("run_index"),
        "output_created": False,
        "output_existed": bool(Path(context["output"]).exists()),
        "protocol_sha256": context["protocol_sha256"],
        "source_checkpoint_sha256": context["loaded_sha"],
        "source_checkpoint_kind": SOURCE_CHECKPOINT_KIND,
        "trainable_parameters": list(context["trainable"]),
        "fresh_adam": len(context["policy"].optimizer.state) == 0,
        "anchor_rows": len(context["old_rows"]) + len(context["coverage_rows"]),
        "reference_digest": context["reference_digest"],
        "reference_gate_coefficient": context["reference"].coefficient,
        "reference_gate_max_kl": context["reference"].max_kl,
        "actual_config_snapshots": context["config_snapshots"],
        "heldout_executed": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "PPO_ALLOWED": False,
        "checkpoint_promotable": False,
    }


def _smoke_run(context: Mapping[str, Any], output: Path) -> dict[str, Any]:
    v6 = context["v6"]
    config = context["configs"]["dense24"]
    smoke = v6["smoke"]
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = admission._record_rollout(
        policy=context["policy"], config=config, p=context["runtime_protocol"],
        scenario="dense24", seed=int(smoke["seed"]), output=output / "dense24_859000",
        label="smoke", deterministic=False, max_steps=int(smoke["max_steps"]),
        require_full_resolution=False, checkpoint=Path(context["output"]),
    )
    steps = record.pop("_steps", [])
    checks = _smoke_checks(record, steps, int(smoke["max_steps"]))
    checkpoint = output / "checkpoints" / "smoke_roundtrip.pt"
    restored, restore_exact = _save_and_restore(
        context["policy"], checkpoint, config, v6, context["reference"],
        episode_index=0, timestep=len(steps), trainable=context["trainable"],
        protocol_sha256=context["protocol_sha256"], formal=False, ppo_called=False,
    )
    checks["checkpoint_restore_exact"] = bool(restore_exact)
    checks["fresh_optimizer_before_smoke"] = len(context["policy"].optimizer.state) == 0
    result = {
        "stage": f"{STAGE}_SMOKE", "status": "SMOKE_COMPLETED" if all(checks.values()) else "SMOKE_FAILED",
        "PASS": bool(all(checks.values())), "smoke_partial": True, "records": [record],
        "smoke_checks": checks, "selected_checkpoint_sha256": context["loaded_sha"],
        "source_checkpoint_kind": SOURCE_CHECKPOINT_KIND, "reference_digest": context["reference_digest"],
        "anchor_rows": len(context["old_rows"]) + len(context["coverage_rows"]),
        "checkpoint": str(checkpoint), "checkpoint_restore_exact": bool(restore_exact),
        "config": base._config_snapshot(config), "horizon_steps": int(round(
            float(config.max_simulation_seconds) / float(config.dt)
        )), "smoke_max_steps": int(smoke["max_steps"]),
        "trainable_parameters": list(context["trainable"]), "fresh_optimizer": True,
        "wall_seconds": float(time.perf_counter() - started), "heldout_executed": False,
        "P3_QUALIFIED": False, "P4_ALLOWED": False, "PPO_ALLOWED": False,
        "checkpoint_promotable": False,
    }
    _write_json(output / "progress.json", {"stage": result["stage"], "records": [record], "smoke_checks": checks})
    _write_json(output / "manifest.json", result)
    return result


def _training_run(context: Mapping[str, Any], output: Path, *, run_index: int | None) -> dict[str, Any]:
    v6 = context["v6"]
    runtime = context["runtime_protocol"]
    configs = context["configs"]
    formal = run_index is not None
    seeds, scenarios = _schedule(v6, run_index if formal else None)
    if formal:
        dev_section = v6["formal_schedule"]["development"]
    else:
        dev_section = v6["pilot"]["development"]
    dev_seeds = [int(value) for value in dev_section["seeds"]]
    dev_scenarios = [str(value) for value in dev_section["scenarios"]]
    if not formal:
        dev_scenarios = [str(value) for value in ("flow6", "flow12", "dense24")]
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    dev_evaluations: list[dict[str, Any]] = []
    accepted = 0
    timestep = 0
    stopped = False
    stop_reason: str | None = None
    current_checkpoint = Path(context["output"]) / "source_checkpoint.pt"
    # This path is only a report binding; no source copy is written.
    current_checkpoint = Path(str(context.get("checkpoint", context["output"])))
    ppo_called = False
    before_dev: dict[str, Any] | None = None
    final_checkpoint: str | None = None
    checkpoints = []
    try:
        before_dev = _dev_evaluate(
            policy=context["policy"], configs=configs, runtime_protocol=runtime,
            v6=v6, output=output / "dev_before_training", accepted_episodes=0,
            checkpoint=current_checkpoint, seeds=dev_seeds, scenarios=dev_scenarios,
            protocol_sha256=context["protocol_sha256"],
        )
        dev_evaluations.append(before_dev)
        if before_dev["hard_failure"] or not before_dev["full_resolution"]:
            stopped = True
            stop_reason = "baseline development full-resolution or hard failure"
        for episode_index, (seed, scenario) in enumerate(zip(seeds, scenarios), start=1):
            if stopped:
                break
            config = configs[str(scenario)]
            horizon = int(round(float(config.max_simulation_seconds) / float(config.dt)))
            episode_started = time.perf_counter()
            rollout_started = time.perf_counter()
            record = admission._record_rollout(
                policy=context["policy"], config=config, p=runtime,
                scenario=str(scenario), seed=int(seed),
                output=output / "episodes" / f"episode_{episode_index:04d}_{scenario}_{seed}",
                label=f"{('formal' if formal else 'pilot')}_{run_index if formal else 0}",
                deterministic=False, max_steps=horizon,
                require_full_resolution=False, checkpoint=current_checkpoint,
            )
            record["timing_rollout_wall_seconds"] = float(time.perf_counter() - rollout_started)
            record["initial_checkpoint_sha256"] = SOURCE_CHECKPOINT_SHA256
            record["checkpoint_input_sha256"] = SOURCE_CHECKPOINT_SHA256 if not Path(current_checkpoint).is_file() else _sha256(current_checkpoint)
            hard_pass = _training_pass(record)
            update = None
            update_gates: dict[str, bool] = {"rollout_hard_gates": False}
            update_metrics: dict[str, Any] = {}
            update_started = time.perf_counter()
            if hard_pass:
                ppo_called = True
                try:
                    update, update_gates, update_metrics = _update(
                        context["policy"], record.get("_steps", []), config,
                        str(record.get("action_tape")), runtime, context["reference"],
                        context["old_rows"], context["coverage_rows"],
                    )
                    update_gates = dict(update_gates)
                    update_pass = bool(
                        update_gates and all(update_gates.values()) and
                        int(update.get("accepted_updates", 0)) >= 1
                    )
                except Exception as exc:
                    update = {"exception": type(exc).__name__, "message": str(exc)}
                    update_gates = {"update_transaction": False}
                    update_metrics = {}
                    update_pass = False
            else:
                update_pass = False
            record["timing_update_wall_seconds"] = float(time.perf_counter() - update_started)
            record["timing_episode_wall_seconds"] = float(time.perf_counter() - episode_started)
            record["update"] = update
            record["update_gates"] = update_gates
            record["anchor_metrics"] = update_metrics
            record["ppo_called"] = bool(hard_pass)
            if hard_pass and update_pass:
                next_timestep = timestep + len(record.get("_steps", []))
                checkpoint_file = output / "checkpoints" / f"checkpoint_episode_{episode_index:04d}.pt"
                try:
                    restored, restore_exact = _save_and_restore(
                        context["policy"], checkpoint_file, config, v6, context["reference"],
                        episode_index=episode_index, timestep=next_timestep,
                        trainable=context["trainable"], protocol_sha256=context["protocol_sha256"],
                    )
                except Exception as exc:
                    restored, restore_exact = context["policy"], False
                    record["checkpoint_error"] = {"type": type(exc).__name__, "message": str(exc)}
                record["checkpoint_restore_exact"] = bool(restore_exact)
                record["checkpoint_after"] = str(checkpoint_file)
                record["checkpoint_after_sha256"] = _sha256(checkpoint_file) if checkpoint_file.is_file() else None
                if restore_exact:
                    accepted += 1
                    timestep = next_timestep
                    context["policy"].gate_reference = context["reference"]
                    context["policy"] = restored
                    current_checkpoint = checkpoint_file
                    final_checkpoint = str(checkpoint_file)
                    checkpoints.append(str(checkpoint_file))
                else:
                    stopped = True
                    stop_reason = "checkpoint restore integrity failure"
            else:
                stopped = True
                stop_reason = "rollout or PPO update hard failure"
            record["accepted_episode"] = bool(hard_pass and update_pass and record.get("checkpoint_restore_exact") is True)
            public = formal_v5._public_record(record)
            records.append(public)
            _write_json(output / "progress.json", {
                "stage": STAGE, "run_index": run_index, "episodes_requested": len(seeds),
                "episode_index": episode_index, "accepted_episodes": accepted,
                "records": records, "dev_evaluations": dev_evaluations,
                "stopped": stopped, "stop_reason": stop_reason,
                "heldout_executed": False, "no_resume_claim": True,
            })
            if stopped:
                break
            checkpoints_after = v6["formal_schedule"]["development"].get("compare_after_updates", []) if formal else []
            if accepted in [int(x) for x in checkpoints_after]:
                dev = _dev_evaluate(
                    policy=context["policy"], configs=configs, runtime_protocol=runtime,
                    v6=v6, output=output / f"dev_after_{accepted:04d}", accepted_episodes=accepted,
                    checkpoint=current_checkpoint, seeds=dev_seeds, scenarios=dev_scenarios,
                    protocol_sha256=context["protocol_sha256"],
                )
                dev_evaluations.append(dev)
                if dev["hard_failure"] or not dev["full_resolution"]:
                    stopped = True
                    stop_reason = "development full-resolution or hard failure"
                    break
        if stopped:
            status = "BLOCKED_FOR_REVIEW" if stop_reason and stop_reason.startswith("development") else "FAILED"
        else:
            status = "COMPLETED"
    except KeyboardInterrupt:
        stopped = True
        status = "BLOCKED_FOR_REVIEW"
        stop_reason = "interrupted; partial outputs preserved"
    except Exception as exc:
        stopped = True
        status = "FAILED"
        stop_reason = f"runner exception: {type(exc).__name__}: {exc}"
    after_dev = None
    if not stopped and not formal:
        after_dev = _dev_evaluate(
            policy=context["policy"], configs=configs, runtime_protocol=runtime,
            v6=v6, output=output / "dev_after_training", accepted_episodes=accepted,
            checkpoint=current_checkpoint, seeds=dev_seeds, scenarios=dev_scenarios,
            protocol_sha256=context["protocol_sha256"],
        )
        dev_evaluations.append(after_dev)
        if after_dev["hard_failure"] or not after_dev["full_resolution"]:
            status = "FAILED"
            stopped = True
            stop_reason = "development full-resolution or hard failure"
    pilot_technical_pass = bool(
        not formal
        and status == "COMPLETED"
        and len(records) == len(seeds) == 12
        and accepted == 12
        and before_dev is not None
        and before_dev.get("full_resolution") is True
        and after_dev is not None
        and after_dev.get("full_resolution") is True
        and all(
            row.get("accepted_episode") is True
            and isinstance(row.get("update_gates"), Mapping)
            and all(bool(value) for value in row["update_gates"].values())
            for row in records
        )
    )
    result = {
        "stage": f"{STAGE}_{'FORMAL' if formal else 'PILOT'}",
        "status": status, "PASS": bool(status == "COMPLETED"),
        "run_index": run_index, "episodes_requested": len(seeds),
        "episodes_recorded": len(records), "accepted_episodes": accepted,
        "records": records, "dev_before": before_dev, "dev_after": after_dev,
        "dev_evaluations": dev_evaluations,
        "performance_comparison": _comparison(before_dev or {}, after_dev or {}) if after_dev else None,
        "selected_checkpoint_sha256": context["loaded_sha"],
        "selected_checkpoint_kind": SOURCE_CHECKPOINT_KIND,
        "final_checkpoint": final_checkpoint, "checkpoints": checkpoints,
        "protocol_sha256": context["protocol_sha256"],
        "actual_config_snapshots": context["config_snapshots"],
        "train_seeds": seeds, "episode_pattern": scenarios,
        "dev_seeds": dev_seeds, "dev_scenarios": dev_scenarios,
        "heldout_executed": False, "no_resume_claim": True,
        "source_reference_digest": context["reference_digest"],
        "anchor_rows": len(context["old_rows"]) + len(context["coverage_rows"]),
        "stopped": stopped, "stop_reason": stop_reason, "ppo_called": bool(ppo_called),
        "PILOT_TECHNICAL_PASS": pilot_technical_pass,
        "development_only": True, "FORMAL_P3_ALLOWED": False,
        "P3_QUALIFIED": False, "P4_ALLOWED": False,
        "checkpoint_promotable": False, "PPO_ALLOWED": False,
    }
    _write_json(output / "manifest.json", result)
    return result


def run(*, stage: str, checkpoint: str | Path, output: str | Path,
        run_index: int | None = None, admission_report: str | Path | None = None,
        check_only: bool = False) -> dict[str, Any]:
    stage = str(stage)
    output_path = Path(output)
    checkpoint_path = Path(checkpoint)
    if not check_only:
        _require(not output_path.exists(), f"V6 output must be fresh: {output_path}")
    context = _preflight(
        stage=stage, checkpoint=checkpoint_path, output=output_path,
        run_index=run_index, admission_report=Path(admission_report) if admission_report else None,
    )
    if check_only:
        return _check_only_payload(context)
    context["output"] = checkpoint_path
    if stage == "smoke":
        return _smoke_run(context, output_path)
    return _training_run(context, output_path, run_index=run_index if stage == "formal" else None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-index", type=int)
    parser.add_argument("--admission-report", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    result = run(
        stage=args.stage, checkpoint=args.checkpoint, output=args.output,
        run_index=args.run_index, admission_report=args.admission_report,
        check_only=args.check_only,
    )
    print(json.dumps(_jsonable(result), sort_keys=True), flush=True)
    return 0 if result.get("status") in {"COMPLETED", "SMOKE_COMPLETED", "CHECK_ONLY_OK"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
