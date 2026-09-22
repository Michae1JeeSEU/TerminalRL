"""Read-only paired evaluation for the frozen current-MDP P3 baseline.

The evaluator consumes the preregistered current-MDP YAML, runs the nominal
rule controller beside the frozen original K=3 checkpoint, and writes
descriptive evidence only.  It has no optimizer, update, Teacher, or
checkpoint-promotion path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.training import (
    evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness,
)
from terminal_marl.training import (
    evaluate_v4_2_p2_gate_runtime_shortlist_v1 as runtime_shortlist,
)
from terminal_marl.training import (
    diagnose_v4_2_p3_safety_energy_sensitivity_v1 as safety_energy,
)
from terminal_marl.training import student_rollout_v4_2
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry


STAGE = "V4_2_P3_CURRENT_MDP_BASELINE_V1"
PROTOCOL = Path("terminal_marl/configs/low_v4_2/p3_current_mdp_baseline_v1.yaml")
PROTOCOL_SHA = "5b6bec77c7fb77d3385cc0f983971dbc5db96c1746a2513089a47be365674432"
ORIGINAL_K3_SHA256 = "4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c"
SCENARIO = "dense24"
CONTROLLERS = ("rule", "original_k3")
COHORTS = ("high_soc_qualification", "low_soc_stress")
EXPECTED_REQUESTS = 24
SAFE_DISTANCE_M = 80.0
PROTECTED_FLOOR_M = 40.0
COLLISION_DISTANCE_M = 10.0
ENERGY_CAPACITY = 1500.0
TOUCHDOWN_SOC_MIN = 0.25
FULL_HORIZON_SECONDS = 1800.0
RUNTIME_SHORTLIST_K = 3
OBSERVATION_DIM = 88
OBSERVATION_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
ACTION_SCHEMA = "low_action_schema_v4_2_hierarchical"
SMOKE_SEEDS = {
    "high_soc_qualification": (835000,),
    "low_soc_stress": (835100,),
}
SAFETY_COUNTER_KEYS = (
    "SafetyInvariantError",
    "collisions",
    "nmac",
    "hard_separation_violations",
    "uncertified_execution",
)
SOURCE_FILES = (
    Path(__file__),
    Path(effectiveness.__file__),
    Path(effectiveness.generic.__file__),
    Path(student_rollout_v4_2.__file__),
    Path("terminal_marl/training/run_v4_2_curriculum_pilot.py"),
    Path("terminal_marl/training/run_v4_2_ppo_p2_resume_v4.py"),
)


class BaselineInputError(ValueError):
    """Raised when a frozen baseline input binding is not satisfied."""


class BaselineEpisodeFailure(RuntimeError):
    """Raised to stop the cohort after a safety or replay failure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineInputError(message)


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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _jsonable(value.value)
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaselineEpisodeFailure(f"cannot read JSON summary: {path}") from exc
    if not isinstance(value, dict):
        raise BaselineEpisodeFailure(f"JSON summary is not an object: {path}")
    return value


def load_protocol(path: str | Path = PROTOCOL) -> dict[str, Any]:
    """Load and validate the frozen YAML before any simulator reset."""
    source = Path(path)
    _require(source.is_file(), f"protocol not found: {source}")
    _require(_sha256(source) == PROTOCOL_SHA, "current-MDP protocol SHA mismatch")
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise BaselineInputError("cannot load current-MDP protocol") from exc
    _require(isinstance(value, dict), "current-MDP protocol must be a mapping")
    _require(value.get("protocol") == STAGE, "current-MDP protocol id changed")
    _require(value.get("status") == "PREREGISTERED_CURRENT_MDP_BASELINE", "protocol status changed")

    freeze = value.get("freeze")
    _require(isinstance(freeze, Mapping), "protocol freeze section missing")
    _require(freeze.get("freeze_before_first_reset") is True, "reset freeze binding changed")
    _require(freeze.get("no_seed_evaluated_before_protocol_freeze") is True, "seed freeze binding changed")
    _require(freeze.get("consumed_seed_policy") == "no_rerun_no_reset_no_tuning", "consumed-seed policy changed")

    bindings = value.get("bindings")
    _require(isinstance(bindings, Mapping), "protocol bindings section missing")
    expected_bindings = {
        "candidate_controller": "original_k3",
        "candidate_checkpoint_sha256": ORIGINAL_K3_SHA256,
        "paired_controller": "rule",
        "observation_dimension": OBSERVATION_DIM,
        "observation_schema": OBSERVATION_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "runtime_proposal_shortlist_k": RUNTIME_SHORTLIST_K,
        "fixed_geometry": True,
        "no_teacher": True,
        "no_labels": True,
        "no_updates": True,
    }
    for key, expected in expected_bindings.items():
        _require(bindings.get(key) == expected, f"protocol binding changed: {key}")

    safety = value.get("safety")
    _require(isinstance(safety, Mapping), "protocol safety section missing")
    for key, expected in {
        "preferred_separation_m": SAFE_DISTANCE_M,
        "protected_separation_m": PROTECTED_FLOOR_M,
        "collision_event_rule": "distance_m_lt_10",
        "collision_event_threshold_m": COLLISION_DISTANCE_M,
        "buffer_takeover_requires_swept_and_recursive_certificate": True,
    }.items():
        _require(safety.get(key) == expected, f"protocol safety binding changed: {key}")

    energy = value.get("energy")
    _require(isinstance(energy, Mapping), "protocol energy section missing")
    _require(energy.get("capacity_units") == ENERGY_CAPACITY, "energy capacity binding changed")
    _require(energy.get("touchdown_soc_min") == TOUCHDOWN_SOC_MIN, "touchdown SOC binding changed")

    attributes = value.get("attributes")
    _require(isinstance(attributes, Mapping), "protocol attributes section missing")
    for key, expected in {
        "priority_mode": "random_135",
        "priority_values": [1, 3, 5],
        "mission_probabilities": [0.45, 0.25, 0.10, 0.20],
        "low_soc_analysis_threshold": 0.65,
    }.items():
        _require(attributes.get(key) == expected, f"protocol attribute binding changed: {key}")

    reward = value.get("reward")
    _require(isinstance(reward, Mapping), "protocol reward section missing")
    for key, expected in {
        "low_level_priority_time_weight": 1.0,
        "low_level_soc_urgency_time_weight": 2.0,
        "low_level_soc_urgency_threshold": 0.65,
        "urgency_reward_config_exact": True,
    }.items():
        _require(reward.get(key) == expected, f"protocol reward binding changed: {key}")

    scenario = value.get("scenario")
    _require(isinstance(scenario, Mapping), "protocol scenario section missing")
    for key, expected in {
        "name": SCENARIO,
        "reset_mode": "warm_start_rings",
        "total_requested_uavs": EXPECTED_REQUESTS,
        "warm_start_uavs": 12,
        "warm_start_ring_allocation": [6, 4, 2],
        "continuing_arrivals": 12,
        "arrival_interval_seconds": 15,
        "horizon_seconds": FULL_HORIZON_SECONDS,
        "lz_assignment_mode": "nearest_available",
        "release_arbitration_mode": "downstream_priority",
        "final_queue_capacity": 3,
    }.items():
        _require(scenario.get(key) == expected, f"protocol scenario binding changed: {key}")

    cohorts = value.get("cohorts")
    _require(isinstance(cohorts, Mapping), "protocol cohorts section missing")
    for cohort_name in COHORTS:
        cohort = cohorts.get(cohort_name)
        _require(isinstance(cohort, Mapping), f"cohort missing: {cohort_name}")
        _require(cohort.get("initial_soc_mode") == "uniform", f"SOC mode changed: {cohort_name}")
        _require(cohort.get("controllers") == list(CONTROLLERS), f"controller order changed: {cohort_name}")
        _require(cohort.get("paired_same_seed") is True, f"pairing changed: {cohort_name}")
        _require(cohort.get("deterministic") is True, f"determinism changed: {cohort_name}")
        _require(cohort.get("expected_records") == 6, f"record count changed: {cohort_name}")
        seeds = cohort.get("seeds")
        _require(isinstance(seeds, list) and len(seeds) == 3 and len(set(seeds)) == 3,
                 f"seed binding changed: {cohort_name}")
        hard = cohort.get("hard_gates")
        _require(isinstance(hard, Mapping), f"hard gates missing: {cohort_name}")
        if cohort_name == "high_soc_qualification":
            expected_hard = {
                "each_controller_requested_complete": True,
                "requested_uavs_per_record": EXPECTED_REQUESTS,
                "completion_fraction": 1.0,
                "no_diverted_requests": True,
                "no_unresolved_requests": True,
                "terminal_within_horizon": True,
                "safety_counters_zero": True,
                "exact_action_tape_replay": True,
                "no_teacher": True,
                "no_labels": True,
                "no_updates": True,
                "runtime_shortlist_contract": True,
            }
            _require(cohort.get("initial_soc_min") == 0.90 and cohort.get("initial_soc_max") == 1.00,
                     "high-SOC range changed")
        else:
            expected_hard = {
                "exact_request_accounting": True,
                "requested_uavs_per_record": EXPECTED_REQUESTS,
                "safety_counters_zero": True,
                "exact_action_tape_replay": True,
                "no_teacher": True,
                "no_labels": True,
                "no_updates": True,
                "runtime_shortlist_contract": True,
                "completion_fraction_is_reported_outcome": True,
                "low_soc_diversions_are_reported_outcome": True,
            }
            _require(cohort.get("initial_soc_min") == 0.55 and cohort.get("initial_soc_max") == 0.90,
                     "low-SOC range changed")
        for key, expected in expected_hard.items():
            _require(hard.get(key) == expected, f"hard-gate binding changed: {cohort_name}/{key}")

    authorization = value.get("authorization")
    _require(isinstance(authorization, Mapping), "protocol authorization section missing")
    for key in ("PPO_ALLOWED", "P3_QUALIFIED", "P4_ALLOWED"):
        _require(authorization.get(key) is False, f"authorization binding changed: {key}")
    return value


def source_hashes() -> dict[str, str]:
    """Validate the imported read-only execution sources and return their hashes."""
    result: dict[str, str] = {}
    for path in SOURCE_FILES:
        _require(path.is_file(), f"source input missing: {path}")
        digest = _sha256(path)
        _require(len(digest) == 64, f"invalid source hash: {path}")
        result[path.as_posix()] = digest
    _require(student_rollout_v4_2.OBS_SCHEMA == OBSERVATION_SCHEMA, "student observation schema changed")
    _require(student_rollout_v4_2.ACTION_SCHEMA_V4_2 == ACTION_SCHEMA, "student action schema changed")
    return result


def scenario_config(
    cohort: str,
    *,
    raw_protocol: Mapping[str, Any] | None = None,
):
    """Construct the fixed dense24 config from the frozen protocol bindings."""
    value = dict(raw_protocol) if raw_protocol is not None else load_protocol()
    _require(cohort in COHORTS, f"unknown cohort: {cohort}")
    scenario = value["scenario"]
    attributes = value["attributes"]
    reward = value["reward"]
    cohort_value = value["cohorts"][cohort]
    warm = int(scenario["warm_start_uavs"])
    continuing = int(scenario["continuing_arrivals"])
    interval = float(scenario["arrival_interval_seconds"])
    schedule = (0.0,) * warm + tuple(interval * index for index in range(1, continuing + 1))
    _require(len(schedule) == EXPECTED_REQUESTS, "dense24 schedule/count mismatch")
    return replace(
        p2_config(EXPECTED_REQUESTS),
        high_level_enabled=False,
        experiment_mode="fixed_low_student",
        reset_mode="warm_start_rings",
        warm_start_uavs=warm,
        warm_start_steady_motion=True,
        fixed_arrival_schedule=schedule,
        total_requested_uavs=EXPECTED_REQUESTS,
        max_simulation_seconds=float(scenario["horizon_seconds"]),
        safe_distance=SAFE_DISTANCE_M,
        nmac_distance=PROTECTED_FLOOR_M,
        collision_distance=COLLISION_DISTANCE_M,
        energy_capacity=ENERGY_CAPACITY,
        soc_reserve=TOUCHDOWN_SOC_MIN,
        initial_soc_mode="uniform",
        initial_soc=None,
        initial_soc_min=float(cohort_value["initial_soc_min"]),
        initial_soc_max=float(cohort_value["initial_soc_max"]),
        priority_mode=str(attributes["priority_mode"]),
        initial_priority=None,
        mission_probabilities=tuple(float(item) for item in attributes["mission_probabilities"]),
        low_level_priority_time_weight=float(reward["low_level_priority_time_weight"]),
        low_level_soc_urgency_time_weight=float(reward["low_level_soc_urgency_time_weight"]),
        low_level_soc_urgency_threshold=float(reward["low_level_soc_urgency_threshold"]),
        lz_assignment_mode=str(scenario["lz_assignment_mode"]),
        release_arbitration_mode=str(scenario["release_arbitration_mode"]),
        final_queue_capacity=int(scenario["final_queue_capacity"]),
        runtime_proposal_shortlist_k=RUNTIME_SHORTLIST_K,
        teacher_enabled=False,
        teacher_release_label_enabled=False,
        scheduler_mode="execution",
    )


def _cohort_seeds(
    protocol_value: Mapping[str, Any], cohort: str, *, smoke: bool,
) -> tuple[int, ...]:
    _require(cohort in COHORTS, f"unknown cohort: {cohort}")
    formal = tuple(int(seed) for seed in protocol_value["cohorts"][cohort]["seeds"])
    if not smoke:
        return formal
    smoke_seed = int(SMOKE_SEEDS[cohort][0])
    _require(smoke_seed not in formal, "smoke seed overlaps formal cohort")
    return (smoke_seed,)


def build_case_matrix(
    cohort: str,
    *,
    protocol_value: Mapping[str, Any] | None = None,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """Return paired rule/original-K3 cases for one cohort."""
    value = dict(protocol_value) if protocol_value is not None else load_protocol()
    seeds = _cohort_seeds(value, cohort, smoke=smoke)
    cases = []
    for seed in seeds:
        for controller in CONTROLLERS:
            cases.append({
                "case_id": f"dense24_{cohort}_seed{seed}_{controller}",
                "cohort": cohort,
                "scenario": SCENARIO,
                "seed": int(seed),
                "controller": controller,
            })
    return cases


def _normalise_outcomes(outcomes: Any) -> dict[str, int]:
    if not isinstance(outcomes, Mapping):
        raise BaselineEpisodeFailure("summary outcomes field is missing or not a mapping")
    values = list(outcomes.values())
    if values and all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        result = {str(key): int(value) for key, value in outcomes.items()}
    else:
        result = dict(Counter(str(getattr(value, "value", value)) for value in values))
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BaselineEpisodeFailure(f"invalid outcome count: {key}={value!r}")
    return result


def _outcome_accounting(outcomes: Mapping[str, int], expected_requests: int) -> dict[str, Any]:
    """Keep every outcome class while deriving diversion/unresolved counts."""
    total = sum(int(value) for value in outcomes.values())
    if total > expected_requests:
        raise BaselineEpisodeFailure(
            f"raw outcome accounting exceeds requests: {total} > {expected_requests}"
        )
    diversions = int(outcomes.get("diverted_low_soc", 0))
    other = {
        str(key): int(value)
        for key, value in outcomes.items()
        if str(key) not in {"completed", "diverted_low_soc"}
    }
    missing = max(0, int(expected_requests) - total)
    return {
        "accounting_total": total,
        "accounting_complete": total == int(expected_requests),
        "diversions": diversions,
        "other_outcomes": other,
        "missing_outcome_count": missing,
        "unresolved": sum(other.values()) + missing,
    }


def _validate_raw_summary(
    summary_path: Path,
    *,
    expected_requests: int = EXPECTED_REQUESTS,
    horizon_seconds: float = FULL_HORIZON_SECONDS,
    require_full_accounting: bool = True,
) -> dict[str, Any]:
    """Validate raw runner evidence before any derived gate is trusted."""
    summary = _read_json_object(summary_path)
    safety = summary.get("safety")
    if not isinstance(safety, Mapping):
        raise BaselineEpisodeFailure("raw summary safety counters are missing")
    counters: dict[str, int] = {}
    for key in SAFETY_COUNTER_KEYS:
        if key not in safety:
            raise BaselineEpisodeFailure(f"required raw safety counter is missing: {key}")
        value = safety[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BaselineEpisodeFailure(f"invalid raw safety counter: {key}={value!r}")
        counters[key] = int(value)
    for raw_key, value in safety.items():
        key = str(raw_key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BaselineEpisodeFailure(f"invalid raw safety counter: {key}={value!r}")
        counters[key] = int(value)
    if any(counters.values()):
        raise BaselineEpisodeFailure(f"raw safety counters are nonzero: {counters}")
    if summary.get("safety_error") not in (None, ""):
        raise BaselineEpisodeFailure("raw summary reports a safety error")
    outcomes = _normalise_outcomes(summary.get("outcomes"))
    accounting = _outcome_accounting(outcomes, expected_requests)
    accounting_total = int(accounting["accounting_total"])
    if require_full_accounting and accounting_total != expected_requests:
        raise BaselineEpisodeFailure(
            f"raw outcome accounting mismatch: {accounting_total} != {expected_requests}"
        )
    completed = summary.get("completed")
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
        raise BaselineEpisodeFailure(f"invalid raw completed count: {completed!r}")
    if completed != outcomes.get("completed", 0):
        raise BaselineEpisodeFailure("raw completed count does not match outcomes")
    seconds = summary.get("seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise BaselineEpisodeFailure("raw simulation seconds are missing or invalid")
    if not math.isfinite(float(seconds)) or float(seconds) < 0.0 or float(seconds) > horizon_seconds:
        raise BaselineEpisodeFailure("raw simulation seconds exceed the frozen horizon")
    return {
        "safety": counters,
        "outcomes": outcomes,
        "accounting_total": accounting_total,
        "accounting_complete": bool(accounting["accounting_complete"]),
        "completed": int(completed),
        "seconds": float(seconds),
        "terminal": bool(summary.get("terminal")),
        "action_tape": summary.get("action_tape"),
        "PPO_ALLOWED": summary.get("PPO_ALLOWED", False),
    }


def _empty_shortlist_check() -> dict[str, Any]:
    return {
        "shortlists_supplied": 0,
        "fallback_selected": 0,
        "exhausted": 0,
        "max_candidates_checked": 0,
        "requested_is_first": True,
        "unique": True,
        "bounded": True,
        "teacher_free": True,
        "all": True,
        "not_applicable": True,
    }


def _execute_episode(
    case: Mapping[str, Any],
    *,
    config: Any,
    policy: Any,
    output: Path,
    smoke: bool,
) -> dict[str, Any]:
    case_dir = output / str(case["case_id"])
    if case_dir.exists():
        raise BaselineEpisodeFailure(f"case output already exists: {case_dir}")
    controller = str(case["controller"])
    if controller == "rule":
        result = dict(effectiveness._rule_result(config, int(case["seed"]), case_dir, smoke=smoke))
        shortlist = _empty_shortlist_check()
    elif controller == "original_k3":
        result = dict(effectiveness._student_result(policy, config, int(case["seed"]), case_dir, smoke=smoke))
        tape_path = result.get("summary", {}).get("action_tape")
        if not isinstance(tape_path, str) or not tape_path:
            raise BaselineEpisodeFailure("student action tape path is missing")
        try:
            shortlist = runtime_shortlist._tape_shortlist_checks(
                tape_path, expected_k=RUNTIME_SHORTLIST_K
            )
        except Exception as exc:
            raise BaselineEpisodeFailure("runtime shortlist replay failed") from exc
    else:
        raise BaselineInputError(f"unsupported controller: {controller}")

    raw_summary = _validate_raw_summary(
        case_dir / "summary.json",
        require_full_accounting=not smoke,
    )
    tape_path = raw_summary.get("action_tape") or result.get("summary", {}).get("action_tape")
    if not isinstance(tape_path, str) or not tape_path:
        raise BaselineEpisodeFailure("action tape path is missing for replay telemetry")
    if result.get("replay_exact") is not True:
        raise BaselineEpisodeFailure("episode replay is not exact")
    try:
        replay_telemetry, replay, replay_error, _ = safety_energy._replay_telemetry(
            config, tape_path
        )
    except Exception as exc:
        raise BaselineEpisodeFailure("replay telemetry collection failed") from exc
    if replay_error is not None:
        raise BaselineEpisodeFailure(f"replay telemetry reported an error: {replay_error}")
    if not isinstance(replay, Mapping) or replay.get("exact") is not True:
        raise BaselineEpisodeFailure("replay telemetry replay is not exact")
    replay_safety: dict[str, int] = {}
    for key in ("hard_separation_violations", "nmac", "collisions"):
        if not isinstance(replay_telemetry, Mapping) or key not in replay_telemetry:
            raise BaselineEpisodeFailure(f"replay telemetry counter missing: {key}")
        value = replay_telemetry[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BaselineEpisodeFailure(f"invalid replay telemetry counter: {key}={value!r}")
        replay_safety[key] = int(value)
    if any(replay_safety.values()):
        raise BaselineEpisodeFailure(f"replay telemetry safety counters are nonzero: {replay_safety}")
    replay_exact = True
    if raw_summary.get("PPO_ALLOWED") is True or result.get("training_called") is True:
        raise BaselineEpisodeFailure("episode reports an update authorization")

    no_teacher = result.get("teacher_enabled") is False
    no_labels = int(result.get("label_rows", 0)) == 0
    no_updates = result.get("training_called") is False
    if not (no_teacher and no_labels and no_updates):
        raise BaselineEpisodeFailure("episode is not Teacher-free/no-label/no-update")

    completed = int(raw_summary["completed"])
    requests = EXPECTED_REQUESTS
    accounting = _outcome_accounting(raw_summary["outcomes"], requests)
    diversions = int(accounting["diversions"])
    other_outcomes = dict(accounting["other_outcomes"])
    missing_outcomes = int(accounting["missing_outcome_count"])
    unresolved = int(accounting["unresolved"])
    terminal = bool(raw_summary["terminal"])
    runtime_gate = bool(shortlist.get("all"))
    record = {
        **dict(case),
        "status": "completed",
        "requests": requests,
        "completed": completed,
        "completion_fraction": completed / requests,
        "diversions": diversions,
        "unresolved": unresolved,
        "terminal": terminal,
        "seconds": float(raw_summary["seconds"]),
        "safety": dict(raw_summary["safety"]),
        "replay_safety": replay_safety,
        "replay_error": replay_error,
        "outcomes": dict(raw_summary["outcomes"]),
        "other_outcomes": other_outcomes,
        "missing_outcome_count": missing_outcomes,
        "exact_request_accounting": bool(raw_summary["accounting_complete"]),
        "replay_exact": replay_exact,
        "teacher_free": no_teacher,
        "labels_zero": no_labels,
        "updates_zero": no_updates,
        "runtime_shortlist_checks": shortlist,
        "runtime_shortlist_contract": runtime_gate,
        "mean_wait_seconds_completed_only": result.get("mean_wait_seconds"),
        "mean_request_to_touchdown_seconds_completed_only": result.get(
            "mean_request_to_touchdown_seconds"
        ),
        "completed_only_wait_condition": "completed_uavs_only; null when no completed UAVs",
        "action_tape": str(raw_summary.get("action_tape") or result.get("summary", {}).get("action_tape")),
        "raw_summary": str(case_dir / "summary.json"),
        "evaluation": str(case_dir / "evaluation.json"),
    }
    high = str(case["cohort"]) == "high_soc_qualification"
    record["hard_gates"] = {
        "exact_request_accounting": record["exact_request_accounting"],
        "requested_complete": completed == requests,
        "no_diverted_requests": diversions == 0,
        "no_unresolved_requests": unresolved == 0,
        "terminal_within_horizon": terminal and record["seconds"] <= FULL_HORIZON_SECONDS,
        "safety_counters_zero": not any(record["safety"].values()),
        "replay_safety_counters_zero": not any(record["replay_safety"].values()),
        "exact_action_tape_replay": replay_exact,
        "no_teacher": no_teacher,
        "no_labels": no_labels,
        "no_updates": no_updates,
        "runtime_shortlist_contract": runtime_gate,
    }
    record["hard_gates"]["all"] = all(record["hard_gates"].values())
    record["valid_for_pair_metrics"] = bool(
        record["hard_gates"]["all"]
        and completed == requests
        and diversions == 0
        and unresolved == 0
        and terminal
    )
    if not high:
        # Low-SOC completion and diversion are report-only outcomes.  The
        # hard gate intentionally excludes their values.
        low_keys = (
            "exact_request_accounting", "safety_counters_zero",
            "replay_safety_counters_zero", "exact_action_tape_replay",
            "no_teacher", "no_labels", "no_updates",
            "runtime_shortlist_contract",
        )
        record["hard_gates"]["all"] = all(record["hard_gates"][key] for key in low_keys)
        record["valid_for_pair_metrics"] = bool(
            record["hard_gates"]["all"]
            and completed == requests
            and diversions == 0
            and unresolved == 0
            and terminal
        )
    return record


def _failed_record(case: Mapping[str, Any], error: BaseException) -> dict[str, Any]:
    return {
        **dict(case),
        "status": "failed",
        "requests": EXPECTED_REQUESTS,
        "completed": None,
        "completion_fraction": None,
        "diversions": None,
        "unresolved": None,
        "terminal": False,
        "seconds": None,
        "safety": None,
        "replay_safety": None,
        "replay_error": None,
        "outcomes": None,
        "replay_exact": False,
        "teacher_free": False,
        "labels_zero": False,
        "updates_zero": False,
        "runtime_shortlist_checks": None,
        "runtime_shortlist_contract": False,
        "valid_for_pair_metrics": False,
        "hard_gates": {"all": False},
        "error": f"{type(error).__name__}: {error}",
    }


def _persist_state(
    output: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    attempted_cases: Sequence[Mapping[str, Any]],
    consumed_seeds: Sequence[int],
    planned_cases: int,
    phase: str,
    current_case: Mapping[str, Any] | None = None,
) -> None:
    state = {
        "stage": STAGE,
        "phase": phase,
        "attempted_cases": list(attempted_cases),
        "consumed_seeds": sorted(set(int(seed) for seed in consumed_seeds)),
        "completed_records": sum(row.get("status") == "completed" for row in records),
        "failed_records": sum(row.get("status") == "failed" for row in records),
        "record_count": len(records),
        "expected_records": planned_cases,
        "current_case": None if current_case is None else dict(current_case),
    }
    _write_json_atomic(output / "progress.json", state)
    _write_json_atomic(output / "records.json", {"records": list(records)})


def _weighted_completed_mean(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values: list[tuple[float, int]] = []
    for row in records:
        value = row.get(key)
        count = row.get("completed")
        if value is None or count is None:
            continue
        try:
            number = float(value)
            weight = int(count)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and weight > 0:
            values.append((number, weight))
    if not values:
        return None
    return float(sum(number * weight for number, weight in values) / sum(weight for _, weight in values))


def paired_metrics(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return paired deltas only when both controller records are complete and valid."""
    grouped: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in records:
        grouped.setdefault(int(row["seed"]), {})[str(row["controller"])] = row
    result = []
    for seed, values in sorted(grouped.items()):
        rule = values.get("rule")
        k3 = values.get("original_k3")
        eligible = bool(
            rule is not None and k3 is not None
            and rule.get("valid_for_pair_metrics") is True
            and k3.get("valid_for_pair_metrics") is True
        )
        row: dict[str, Any] = {"seed": seed, "eligible": eligible}
        if not eligible:
            row["reason"] = "both_controller_records_not_complete_and_valid"
            row["rule_minus_original_k3"] = None
        else:
            row["reason"] = None
            row["rule_minus_original_k3"] = {
                "seconds": float(rule["seconds"]) - float(k3["seconds"]),
                "mean_wait_seconds_completed_only": (
                    None
                    if rule.get("mean_wait_seconds_completed_only") is None
                    or k3.get("mean_wait_seconds_completed_only") is None
                    else float(rule["mean_wait_seconds_completed_only"])
                    - float(k3["mean_wait_seconds_completed_only"])
                ),
                "mean_request_to_touchdown_seconds_completed_only": (
                    None
                    if rule.get("mean_request_to_touchdown_seconds_completed_only") is None
                    or k3.get("mean_request_to_touchdown_seconds_completed_only") is None
                    else float(rule["mean_request_to_touchdown_seconds_completed_only"])
                    - float(k3["mean_request_to_touchdown_seconds_completed_only"])
                ),
            }
        result.append(row)
    return result


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    cohort: str,
    expected_records: int,
) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    high = cohort == "high_soc_qualification"
    for controller in CONTROLLERS:
        rows = [row for row in records if row.get("controller") == controller]
        safety: dict[str, int] = {}
        for row in rows:
            for key, value in (row.get("safety") or {}).items():
                safety[key] = safety.get(key, 0) + int(value)
        hard = [row.get("hard_gates", {}).get("all") is True for row in rows]
        aggregates[controller] = {
            "records": len(rows),
            "expected_records": expected_records // len(CONTROLLERS),
            "requests": sum(int(row.get("requests") or 0) for row in rows),
            "completed": sum(int(row.get("completed") or 0) for row in rows),
            "completion_fraction": (
                sum(int(row.get("completed") or 0) for row in rows)
                / max(1, sum(int(row.get("requests") or 0) for row in rows))
            ),
            "diversions": sum(int(row.get("diversions") or 0) for row in rows),
            "unresolved": sum(int(row.get("unresolved") or 0) for row in rows),
            "terminal_records": sum(bool(row.get("terminal")) for row in rows),
            "safety": safety,
            "safety_all_zero": not any(safety.values()),
            "all_episode_hard_gates": bool(rows) and all(hard),
            "mean_wait_seconds_completed_only": _weighted_completed_mean(
                rows, "mean_wait_seconds_completed_only"
            ),
            "mean_request_to_touchdown_seconds_completed_only": _weighted_completed_mean(
                rows, "mean_request_to_touchdown_seconds_completed_only"
            ),
            "condition": "completed_uavs_only; null when no completed UAVs",
        }
        if not high:
            aggregates[controller]["completion_is_report_only"] = True
            aggregates[controller]["diversion_is_report_only"] = True
    return aggregates


def _final_gates(
    records: Sequence[Mapping[str, Any]],
    *,
    cohort: str,
    expected_records: int,
) -> dict[str, bool]:
    high = cohort == "high_soc_qualification"
    common = {
        "expected_record_count": len(records) == expected_records,
        "all_records_present": bool(records),
        "all_safety_zero": bool(records) and all(
            row.get("safety") is not None and not any(row["safety"].values())
            for row in records
        ),
        "all_replay_safety_zero": bool(records) and all(
            row.get("replay_safety") is not None and not any(row["replay_safety"].values())
            for row in records
        ),
        "all_replay_exact": bool(records) and all(row.get("replay_exact") is True for row in records),
        "all_teacher_free": bool(records) and all(row.get("teacher_free") is True for row in records),
        "all_labels_zero": bool(records) and all(row.get("labels_zero") is True for row in records),
        "all_updates_zero": bool(records) and all(row.get("updates_zero") is True for row in records),
        "all_runtime_shortlist_contract": bool(records) and all(
            row.get("runtime_shortlist_contract") is True for row in records
        ),
        "no_failed_records": bool(records) and all(row.get("status") == "completed" for row in records),
    }
    if high:
        common.update({
            "all_requested_complete": bool(records) and all(row.get("completed") == EXPECTED_REQUESTS for row in records),
            "no_diverted_requests": bool(records) and all(row.get("diversions") == 0 for row in records),
            "no_unresolved_requests": bool(records) and all(row.get("unresolved") == 0 for row in records),
            "all_terminal_within_horizon": bool(records) and all(
                row.get("terminal") is True and float(row.get("seconds", float("inf"))) <= FULL_HORIZON_SECONDS
                for row in records
            ),
        })
    else:
        common["exact_request_accounting"] = bool(records) and all(
            sum((row.get("outcomes") or {}).values()) == EXPECTED_REQUESTS for row in records
        )
    common["all"] = all(common.values())
    return common


def _fail_manifest(
    *,
    output: Path,
    protocol_value: Mapping[str, Any],
    cohort: str,
    seeds: Sequence[int],
    source_input_hashes: Mapping[str, str],
    checkpoint: Path,
    records: Sequence[Mapping[str, Any]],
    attempted_cases: Sequence[Mapping[str, Any]],
    consumed_seeds: Sequence[int],
    error: BaseException,
) -> dict[str, Any]:
    manifest = {
        "stage": STAGE,
        "status": "FAIL",
        "PASS": False,
        "failure": f"{type(error).__name__}: {error}",
        "failure_stop": "safety_or_replay_or_episode_error",
        "cohort": cohort,
        "scenario": SCENARIO,
        "protocol": str(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": ORIGINAL_K3_SHA256,
        "source_hashes": dict(source_input_hashes),
        "seeds": list(seeds),
        "attempted_cases": list(attempted_cases),
        "consumed_seeds": sorted(set(int(seed) for seed in consumed_seeds)),
        "records": list(records),
        "PPO_ALLOWED": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "training_called": False,
        "labels_collected": False,
        "protocol_value": _jsonable(protocol_value),
    }
    _write_json_atomic(output / "manifest.json", manifest)
    return manifest


def run(
    *,
    cohort: str,
    output: str | Path,
    checkpoint: str | Path,
    history_root: str | Path,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run one frozen cohort; smoke uses a non-formal seed and no reservation."""
    protocol_value = load_protocol()
    source_input_hashes = source_hashes()
    output_path = Path(output)
    checkpoint_path = Path(checkpoint)
    history_path = Path(history_root)
    _require(cohort in COHORTS, f"unknown cohort: {cohort}")
    _require(not output_path.exists(), f"output already exists: {output_path}")
    _require(checkpoint_path.is_file(), f"checkpoint not found: {checkpoint_path}")
    _require(_sha256(checkpoint_path) == ORIGINAL_K3_SHA256, "original K3 checkpoint SHA mismatch")
    policy, checkpoint_payload = effectiveness.load_candidate_checkpoint(
        checkpoint_path, expected_sha256=ORIGINAL_K3_SHA256
    )
    _require(policy.observation_dim == OBSERVATION_DIM, "checkpoint observation dimension changed")
    _require(checkpoint_payload.get("PPO_ALLOWED") is False, "checkpoint PPO authorization changed")

    seeds = _cohort_seeds(protocol_value, cohort, smoke=smoke)
    cases = build_case_matrix(cohort, protocol_value=protocol_value, smoke=smoke)
    expected_records = len(cases)
    config = scenario_config(cohort, raw_protocol=protocol_value)
    _require(config.total_requested_uavs == EXPECTED_REQUESTS, "config request count changed")
    _require(config.safe_distance == SAFE_DISTANCE_M, "config safe distance changed")
    _require(config.nmac_distance == PROTECTED_FLOOR_M, "config protected floor changed")
    _require(config.collision_distance == COLLISION_DISTANCE_M, "config collision distance changed")
    _require(config.max_simulation_seconds == FULL_HORIZON_SECONDS, "config horizon changed")

    output_path.mkdir(parents=True, exist_ok=False)
    history = historical_registry(history_path, output_path)
    overlap = sorted(set(seeds) & set(history.get("seeds", set())))
    _require(not overlap, f"cohort seeds already consumed: {overlap}")
    _write_json_atomic(output_path / "pre_reset_registry.json", {
        "stage": STAGE,
        "protocol_sha256": PROTOCOL_SHA,
        "checkpoint_sha256": ORIGINAL_K3_SHA256,
        "cohort": cohort,
        "seeds": list(seeds),
        "controllers": list(CONTROLLERS),
        "smoke": smoke,
        "historical_seed_count": len(history.get("seeds", set())),
        "reserved_before_first_reset": True,
        "formal_seed_consumption": not smoke,
        "source_hashes": dict(source_input_hashes),
    })

    records: list[dict[str, Any]] = []
    attempted_cases: list[dict[str, Any]] = []
    consumed_seeds: list[int] = []
    for case in cases:
        attempted = {**dict(case), "status": "started"}
        attempted_cases.append(attempted)
        if int(case["seed"]) not in consumed_seeds:
            consumed_seeds.append(int(case["seed"]))
        _persist_state(
            output_path,
            records=records,
            attempted_cases=attempted_cases,
            consumed_seeds=consumed_seeds,
            planned_cases=expected_records,
            phase="before_episode_reset",
            current_case=attempted,
        )
        try:
            record = _execute_episode(
                case,
                config=config,
                policy=policy,
                output=output_path,
                smoke=smoke,
            )
        except Exception as exc:
            record = _failed_record(case, exc)
            records.append(record)
            _persist_state(
                output_path,
                records=records,
                attempted_cases=attempted_cases,
                consumed_seeds=consumed_seeds,
                planned_cases=expected_records,
                phase="failed_stop",
                current_case=record,
            )
            return _fail_manifest(
                output=output_path,
                protocol_value=protocol_value,
                cohort=cohort,
                seeds=seeds,
                source_input_hashes=source_input_hashes,
                checkpoint=checkpoint_path,
                records=records,
                attempted_cases=attempted_cases,
                consumed_seeds=consumed_seeds,
                error=exc,
            )
        records.append(record)
        attempted_cases[-1] = {**dict(case), "status": "completed"}
        _persist_state(
            output_path,
            records=records,
            attempted_cases=attempted_cases,
            consumed_seeds=consumed_seeds,
            planned_cases=expected_records,
            phase="after_episode_complete",
            current_case=record,
        )

    gates = _final_gates(records, cohort=cohort, expected_records=expected_records)
    aggregates = aggregate_records(records, cohort=cohort, expected_records=expected_records)
    paired = paired_metrics(records)
    manifest = {
        "stage": STAGE,
        "status": "SMOKE" if smoke else "COMPLETE",
        "PASS": bool(gates["all"] and not smoke),
        "cohort": cohort,
        "scenario": SCENARIO,
        "protocol": str(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": ORIGINAL_K3_SHA256,
        "source_hashes": dict(source_input_hashes),
        "seeds": list(seeds),
        "controllers": list(CONTROLLERS),
        "expected_records": expected_records,
        "records": records,
        "aggregates": aggregates,
        "paired_metrics": paired,
        "gates": gates,
        "consumed_seeds": sorted(set(consumed_seeds)),
        "attempted_cases": attempted_cases,
        "checkpoint_payload_stage": checkpoint_payload.get("stage"),
        "training_called": False,
        "labels_collected": False,
        "PPO_ALLOWED": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "protocol_value": _jsonable(protocol_value),
    }
    _write_json_atomic(output_path / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    result = run(
        cohort=args.cohort,
        output=args.output,
        checkpoint=args.checkpoint,
        history_root=args.history_root,
        smoke=args.smoke,
    )
    print(json.dumps({
        "stage": result["stage"],
        "status": result["status"],
        "PASS": result["PASS"],
        "cohort": result["cohort"],
        "records": len(result.get("records", [])),
        "PPO_ALLOWED": result["PPO_ALLOWED"],
    }, sort_keys=True), flush=True)
    if result.get("status") == "FAIL" or (
        result.get("status") == "COMPLETE" and result.get("PASS") is not True
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
