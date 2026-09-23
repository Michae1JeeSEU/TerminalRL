"""Paired diagnostic for SOC priority under phase-motion energy semantics.

The runner compares the frozen original V4.2 K3 policy with the nominal rule
controller on the same low-SOC requests and seeds.  It is descriptive only:
there is no optimizer, Teacher, label collection, or checkpoint promotion
path.  The protocol is checked before the first environment reset.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from terminal_marl.core.arrival_process import ArrivalProcess
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness
from terminal_marl.training.evaluate_v4_2_p2_gate_runtime_shortlist_v1 import (
    _tape_shortlist_checks,
)
from terminal_marl.training.run_v4_2_curriculum_pilot import episode as rule_episode
from terminal_marl.training.run_v4_2_i2_dagger_preflight import _policy_tape_logprob_exact
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config
from terminal_marl.training.student_rollout_v4_2 import rollout


PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "configs" / "low_v4_2" / "emergency_soc_priority_paired_v1.yaml"
PROTOCOL_SHA256 = "6504740d11dc7ceb005155e57ed26ff5edfdf392bf7056bd9d8a5f1ef6893842"
CHECKPOINT_SHA256 = "4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c"
FORMAL_SEEDS = (840301, 840302)
SMOKE_SEED = 840300
CONTROLLERS = ("rule", "original_k3")
ACTION_SCHEMA = "low_action_schema_v4_2_hierarchical"
OBSERVATION_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
OBSERVATION_DIMENSION = 88
RUNTIME_SHORTLIST_K = 3
SOURCE_PATHS = (
    Path(__file__),
    Path(__file__).resolve().parents[1] / "envs" / "parallel_env.py",
    Path(__file__).resolve().parents[1] / "core" / "admission.py",
    Path(__file__).resolve().parents[1] / "core" / "arrival_process.py",
    Path(__file__).resolve().parents[1] / "core" / "config.py",
    Path(__file__).resolve().parents[1] / "core" / "energy.py",
    Path(__file__).resolve().parents[1] / "core" / "lifecycle.py",
    Path(__file__).resolve().parents[1] / "core" / "trajectory_generator.py",
    Path(__file__).resolve().parents[1] / "core" / "uav.py",
    Path(__file__).resolve().parents[1] / "core" / "safety_trace.py",
    Path(__file__).resolve().parents[1] / "training" / "student_rollout_v4_2.py",
    Path(__file__).resolve().parents[1] / "training" / "run_v4_2_curriculum_pilot.py",
)
SAFETY_COUNTER_KEYS = (
    "collisions", "nmac", "hard_separation_violations",
    "uncertified_execution", "SafetyInvariantError",
)


class ProtocolError(ValueError):
    """Raised when a frozen binding is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Load the preregistered YAML and validate the complete scenario binding."""
    path = Path(path)
    _require(path.resolve() == PROTOCOL_PATH.resolve(), "only the frozen paired protocol is accepted")
    _require(bool(PROTOCOL_SHA256), "protocol SHA256 is not populated")
    _require(_sha256(path) == PROTOCOL_SHA256, "paired protocol SHA256 mismatch")
    value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "paired protocol root must be a mapping")
    _require(value.get("protocol") == "EMERGENCY_SOC_PRIORITY_PAIRED_V1", "protocol id changed")
    _require(value.get("status") == "PREREGISTERED_SOC_PRIORITY_DIAGNOSTIC", "protocol status changed")

    freeze = value.get("freeze")
    _require(isinstance(freeze, Mapping), "freeze section missing")
    for key in ("diagnostic_only", "no_training", "no_optimizer", "no_teacher",
                "no_labels", "no_updates", "deterministic", "no_output_overwrite",
                "no_rerun_no_reset_no_tuning"):
        _require(freeze.get(key) is True, f"freeze.{key} must be true")
    _require(freeze.get("ppo_allowed") is False, "freeze.ppo_allowed must be false")

    checkpoint = value.get("checkpoint")
    _require(isinstance(checkpoint, Mapping), "checkpoint section missing")
    _require(checkpoint.get("controller") == "original_k3", "checkpoint controller changed")
    _require(checkpoint.get("role") == "frozen_original_k3", "checkpoint role changed")
    _require(checkpoint.get("sha256") == CHECKPOINT_SHA256, "checkpoint SHA binding changed")

    scenario = value.get("scenario")
    _require(isinstance(scenario, Mapping), "scenario section missing")
    expected = {
        "reset_mode": "warm_start_rings",
        "warm_start_uavs": 6,
        "warm_start_steady_motion": True,
        "total_requested_uavs": 12,
        "initial_soc_mode": "uniform",
        "initial_soc_min": 0.35,
        "initial_soc_max": 0.55,
        "horizon_seconds": 1200,
        "lz_assignment_mode": "nearest_available",
        "release_arbitration_mode": "downstream_priority",
        "final_queue_capacity": 3,
        "energy_model_mode": "phase_motion_v2",
        "phase_motion_nominal_solo_energy": 240.9,
        "emergency_landing_enabled": True,
        "emergency_trigger_soc": 0.40,
        "emergency_energy_margin_seconds": 15,
    }
    for key, expected_value in expected.items():
        _require(scenario.get(key) == expected_value, f"scenario.{key} must equal {expected_value!r}")
    schedule = tuple(float(item) for item in scenario.get("fixed_arrival_schedule", ()))
    _require(schedule == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0),
             "scenario.fixed_arrival_schedule changed")

    low_level = value.get("low_level")
    _require(isinstance(low_level, Mapping), "low_level section missing")
    for key, expected_value in {
        "action_schema": ACTION_SCHEMA,
        "observation_schema": OBSERVATION_SCHEMA,
        "observation_dimension": OBSERVATION_DIMENSION,
        "runtime_proposal_shortlist_k": RUNTIME_SHORTLIST_K,
    }.items():
        _require(low_level.get(key) == expected_value, f"low_level.{key} changed")

    safety = value.get("safety")
    _require(isinstance(safety, Mapping), "safety section missing")
    for key, expected_value in (("safe_distance_m", 80), ("protected_floor_m", 40), ("collision_distance_m", 10)):
        _require(safety.get(key) == expected_value, f"safety.{key} changed")

    matrix = value.get("matrix")
    _require(isinstance(matrix, Mapping), "matrix section missing")
    _require(tuple(int(seed) for seed in matrix.get("seeds", ())) == FORMAL_SEEDS,
             "formal seed binding changed")
    _require(matrix.get("expected_records") == len(FORMAL_SEEDS) * len(CONTROLLERS),
             "formal record count changed")
    _require(matrix.get("controllers") == list(CONTROLLERS), "controller order changed")
    _require(matrix.get("paired_same_seed") is True, "paired_same_seed must be true")
    _require(matrix.get("deterministic") is True, "matrix.deterministic must be true")

    smoke = value.get("smoke")
    _require(isinstance(smoke, Mapping), "smoke section missing")
    _require(smoke.get("seed") == SMOKE_SEED, "smoke seed changed")
    _require(isinstance(smoke.get("steps"), int) and smoke["steps"] > 0, "smoke steps invalid")
    _require(smoke.get("formal_marker") is False, "smoke formal marker must be false")

    hard = value.get("hard_gates")
    _require(isinstance(hard, Mapping), "hard_gates section missing")
    for key in ("exact_request_accounting", "all_records_terminal", "zero_safety_events",
                "zero_failed_energy", "zero_in_terminal_low_soc_diversion",
                "exact_action_replay", "no_teacher", "no_labels", "no_updates",
                "runtime_shortlist_k3"):
        _require(hard.get(key) is True, f"hard_gates.{key} must be true")

    authorization = value.get("authorization")
    _require(isinstance(authorization, Mapping), "authorization section missing")
    for key in ("PPO_ALLOWED", "training_allowed", "checkpoint_promotion_allowed"):
        _require(authorization.get(key) is False, f"authorization.{key} must be false")
    return dict(value)


def build_config(protocol: Mapping[str, Any], *, horizon_seconds: float | None = None):
    """Build the exact static-geometry configuration used by both controllers."""
    scenario = protocol["scenario"]
    low_level = protocol["low_level"]
    safety = protocol["safety"]
    return replace(
        p2_config(int(scenario["total_requested_uavs"])),
        action_schema_version=str(low_level["action_schema"]),
        observation_schema_version=str(low_level["observation_schema"]),
        experiment_mode="fixed_low_student",
        high_level_enabled=False,
        reset_mode=str(scenario["reset_mode"]),
        warm_start_uavs=int(scenario["warm_start_uavs"]),
        warm_start_steady_motion=bool(scenario["warm_start_steady_motion"]),
        total_requested_uavs=int(scenario["total_requested_uavs"]),
        fixed_arrival_schedule=tuple(float(item) for item in scenario["fixed_arrival_schedule"]),
        max_simulation_seconds=float(
            scenario["horizon_seconds"] if horizon_seconds is None else horizon_seconds
        ),
        initial_soc_mode=str(scenario["initial_soc_mode"]),
        initial_soc=None,
        initial_soc_min=float(scenario["initial_soc_min"]),
        initial_soc_max=float(scenario["initial_soc_max"]),
        lz_assignment_mode=str(scenario["lz_assignment_mode"]),
        release_arbitration_mode=str(scenario["release_arbitration_mode"]),
        final_queue_capacity=int(scenario["final_queue_capacity"]),
        energy_model_mode=str(scenario["energy_model_mode"]),
        phase_motion_nominal_solo_energy=float(scenario["phase_motion_nominal_solo_energy"]),
        emergency_landing_enabled=bool(scenario["emergency_landing_enabled"]),
        emergency_trigger_soc=float(scenario["emergency_trigger_soc"]),
        emergency_energy_margin_seconds=float(scenario["emergency_energy_margin_seconds"]),
        runtime_proposal_shortlist_k=int(low_level["runtime_proposal_shortlist_k"]),
        safe_distance=float(safety["safe_distance_m"]),
        nmac_distance=float(safety["protected_floor_m"]),
        collision_distance=float(safety["collision_distance_m"]),
        teacher_enabled=False,
        teacher_release_label_enabled=False,
        scheduler_mode="execution",
        shield_enabled=True,
    )


def _terminal(env: TerminalParallelEnv) -> bool:
    return bool(
        env.arrivals.exhausted()
        and not env.active
        and not env.admission.queue.items
        and not env.service_uavs
    )


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _replay_case(config: Any, tape_path: str | Path) -> tuple[TerminalParallelEnv, dict[str, Any], dict[str, Any], str | None]:
    """Replay a tape into a fresh env while collecting geometry and peaks."""
    env = TerminalParallelEnv(config)
    geometry: dict[str, float] = {
        "samples": 0, "reverse_samples": 0, "max_inner_excursion_m": 0.0,
    }
    peaks = {"active": 0, "queue": 0}

    def observe(world: TerminalParallelEnv) -> None:
        peaks["active"] = max(peaks["active"], len(world.active))
        peaks["queue"] = max(peaks["queue"], len(world.admission.queue.items))
        for uav in world.active.values():
            plan = uav.committed_plan
            if plan is None or plan.mode != "ring_transfer":
                continue
            xy, velocity = uav.position[:2], uav.velocity[:2]
            radius = float(np.linalg.norm(xy))
            geometry["samples"] += 1
            cross = (xy[0] * velocity[1] - xy[1] * velocity[0]) / max(radius, 1e-9)
            geometry["reverse_samples"] += int(cross < -0.001)
            endpoint_radius = float(np.linalg.norm(plan.points[-1].position[:2]))
            geometry["max_inner_excursion_m"] = max(
                geometry["max_inner_excursion_m"], endpoint_radius - radius,
            )

    env.set_step_observer(observe)
    try:
        replay = SafetyTraceRecorder.replay_action_tape(env, tape_path)
        error = None
    except Exception as exc:  # Keep a fail-closed diagnostic record.
        replay = {"exact": False, "error": str(exc), "steps_replayed": 0, "steps_expected": None}
        error = str(exc)
    geometry.update({"active_peak": peaks["active"], "queue_peak": peaks["queue"]})
    return env, replay, geometry, error


def reconstruct_requests(config: Any, seed: int) -> dict[int, Any]:
    # The cutoff deliberately exceeds the final scheduled arrival so smoke
    # runs can still report the formal request set and its SOC bins.
    cutoff = max(float(config.max_simulation_seconds), max(config.fixed_arrival_schedule) + 1.0)
    process = ArrivalProcess(config, seed)
    requests = process.requests_until(cutoff)
    return {int(request.arrival_id): request for request in requests}


def _horizon_penalized_delays(
    request_times: Mapping[int, float],
    touchdown_times: Mapping[int, float],
    *,
    horizon_seconds: float,
) -> dict[str, Any]:
    """Return one delay for every request, charging horizon for no touchdown."""
    per_request = {
        str(arrival_id): float(
            touchdown_times[arrival_id] - request_time
            if arrival_id in touchdown_times else horizon_seconds - request_time
        )
        for arrival_id, request_time in sorted(request_times.items())
    }
    values = list(per_request.values())
    return {
        "horizon_seconds": float(horizon_seconds),
        "per_request_seconds": per_request,
        "mean_seconds": None if not values else float(np.mean(values)),
        "sum_seconds": None if not values else float(np.sum(values)),
        "requests": len(per_request),
    }


def _safety_counters(env: TerminalParallelEnv, raw_summary: Mapping[str, Any], replay_error: str | None) -> tuple[dict[str, int], dict[str, int]]:
    run_safety = {key: int((raw_summary.get("safety") or {}).get(key, 0)) for key in SAFETY_COUNTER_KEYS}
    if raw_summary.get("safety_error") is not None:
        run_safety["SafetyInvariantError"] = 1
    replay_safety = {key: int(env.telemetry.get(key, 0)) for key in SAFETY_COUNTER_KEYS[:-1]}
    replay_safety["SafetyInvariantError"] = int(replay_error is not None)
    return run_safety, replay_safety


def _outcome_map(env: TerminalParallelEnv) -> dict[int, str]:
    return {
        int(arrival_id): str(getattr(outcome, "value", outcome))
        for arrival_id, outcome in env.ledger.outcomes.items()
    }


def _soc_bin(initial_soc: float) -> str:
    return "below_0p45" if float(initial_soc) < 0.45 else "at_or_above_0p45"


def _lifecycle_metrics(
    env: TerminalParallelEnv,
    config: Any,
    seed: int,
    *,
    raw_summary: Mapping[str, Any],
    replay: Mapping[str, Any],
    geometry: Mapping[str, Any],
    replay_error: str | None,
    requested_likelihood_exact: bool | None,
    controller: str,
    action_tape: str | Path,
) -> dict[str, Any]:
    requests = reconstruct_requests(config, seed)
    expected_ids = set(requests)
    actual_requested = set(int(item) for item in env.ledger.requested)
    outcome_map = _outcome_map(env)
    unresolved = set(int(item) for item in env.ledger.unresolved)
    accounting_identity = (
        actual_requested == set(outcome_map) | unresolved
        and set(outcome_map).isdisjoint(unresolved)
        and len(actual_requested) == len(outcome_map) + len(unresolved)
    )
    exact_request_accounting = bool(
        accounting_identity and actual_requested == expected_ids
        and len(expected_ids) == int(config.total_requested_uavs)
        and not unresolved
    )

    latest_energy: dict[int, Mapping[str, Any]] = {}
    for row in env.energy_outcomes:
        if isinstance(row, Mapping) and row.get("arrival_id") is not None:
            latest_energy[int(row["arrival_id"])] = row
    upstream_diversions = sum(
        row.get("outcome") == "diverted_low_soc" and row.get("phase") == "external_queue"
        for row in latest_energy.values()
    )
    in_terminal_diversions = sum(
        row.get("outcome") == "diverted_low_soc" and row.get("phase") != "external_queue"
        for row in latest_energy.values()
    )

    completed_by_id = {
        int(uav.arrival_id): uav
        for uav in env.completed_uavs
        if uav.touchdown_time is not None
    }
    touchdown_times = {
        arrival_id: float(uav.touchdown_time)
        for arrival_id, uav in completed_by_id.items()
        if uav.touchdown_time is not None
    }
    request_times = {
        int(arrival_id): float(request.request_time)
        for arrival_id, request in requests.items()
    }
    all_request_delay = _horizon_penalized_delays(
        request_times, touchdown_times, horizon_seconds=float(config.max_simulation_seconds),
    )

    normal_delays = [
        float(uav.touchdown_time - uav.request_time)
        for uav in completed_by_id.values()
        if str(getattr(uav.terminal_outcome, "value", uav.terminal_outcome)) == "completed"
        and uav.touchdown_time is not None
    ]
    emergency_delays = [
        float(uav.touchdown_time - uav.request_time)
        for uav in completed_by_id.values()
        if str(getattr(uav.terminal_outcome, "value", uav.terminal_outcome)) == "emergency_landed"
        and uav.touchdown_time is not None
    ]

    outcome_counts = Counter(outcome_map.values())
    soc_bins: dict[str, dict[str, Any]] = {}
    for bin_name in ("below_0p45", "at_or_above_0p45"):
        ids = [arrival_id for arrival_id, request in requests.items() if _soc_bin(request.initial_soc) == bin_name]
        bin_outcomes = Counter(outcome_map.get(arrival_id, "unresolved") for arrival_id in ids)
        bin_touchdowns = {
            arrival_id: touchdown_times[arrival_id]
            for arrival_id in ids if arrival_id in touchdown_times
        }
        bin_requests = {arrival_id: request_times[arrival_id] for arrival_id in ids}
        bin_all_delay = _horizon_penalized_delays(
            bin_requests, bin_touchdowns, horizon_seconds=float(config.max_simulation_seconds),
        )
        bin_completed_delays = [
            touchdown_times[arrival_id] - request_times[arrival_id]
            for arrival_id in ids if arrival_id in touchdown_times
        ]
        soc_bins[bin_name] = {
            "initial_soc_threshold": 0.45,
            "request_ids": ids,
            "requests": len(ids),
            "outcome_counts": dict(sorted(bin_outcomes.items())),
            "completed": int(bin_outcomes.get("completed", 0)),
            "emergency_landed": int(bin_outcomes.get("emergency_landed", 0)),
            "diverted_low_soc": int(bin_outcomes.get("diverted_low_soc", 0)),
            "failed_energy": int(bin_outcomes.get("failed_energy", 0)),
            "unresolved": int(bin_outcomes.get("unresolved", 0)),
            "mean_touchdown_delay_seconds": (
                None if not bin_completed_delays else float(np.mean(bin_completed_delays))
            ),
            "all_request_horizon_penalized_delay": bin_all_delay,
        }

    run_safety, replay_safety = _safety_counters(env, raw_summary, replay_error)
    combined_safety = {
        key: max(run_safety.get(key, 0), replay_safety.get(key, 0))
        for key in SAFETY_COUNTER_KEYS
    }
    runtime_shortlist = (
        {"all": True, "shortlists_supplied": 0, "controller": "rule"}
        if controller == "rule"
        else _tape_shortlist_checks(action_tape, expected_k=RUNTIME_SHORTLIST_K)
    )
    terminal_run = bool(raw_summary.get("terminal"))
    terminal_replay = _terminal(env)
    no_teacher = bool(raw_summary.get("teacher_enabled", False) is False and config.teacher_enabled is False)
    no_labels = int(raw_summary.get("label_rows", 0)) == 0
    no_updates = raw_summary.get("training_called", False) is False
    gates = {
        "exact_request_accounting": exact_request_accounting,
        "all_records_terminal": terminal_run and terminal_replay,
        "zero_safety_events": not any(combined_safety.values()),
        "zero_failed_energy": int(outcome_counts.get("failed_energy", 0)) == 0,
        "zero_in_terminal_low_soc_diversion": int(in_terminal_diversions) == 0,
        "exact_action_replay": bool(replay.get("exact")),
        "no_teacher": no_teacher,
        "no_labels": no_labels,
        "no_updates": no_updates,
        "runtime_shortlist_k3": bool(runtime_shortlist.get("all")),
    }
    return {
        "controller": controller,
        "seed": int(seed),
        "requests": len(expected_ids),
        "actual_requested": len(actual_requested),
        "completed": int(outcome_counts.get("completed", 0)),
        "emergency_landed": int(outcome_counts.get("emergency_landed", 0)),
        "successes": int(outcome_counts.get("completed", 0) + outcome_counts.get("emergency_landed", 0)),
        "outcomes": {str(key): value for key, value in sorted(outcome_map.items())},
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "unresolved_ids": sorted(unresolved),
        "terminal": bool(terminal_run and terminal_replay),
        "terminal_run": terminal_run,
        "terminal_replay": terminal_replay,
        "seconds": float(raw_summary.get("seconds", getattr(env, "now", 0.0))),
        "request_times": {str(key): value for key, value in sorted(request_times.items())},
        "initial_soc": {str(key): float(request.initial_soc) for key, request in sorted(requests.items())},
        "all_request_horizon_penalized_delay": all_request_delay,
        "normal_touchdown_delay_seconds": {
            "count": len(normal_delays),
            "mean": None if not normal_delays else float(np.mean(normal_delays)),
            "values": normal_delays,
        },
        "emergency_touchdown_delay_seconds": {
            "count": len(emergency_delays),
            "mean": None if not emergency_delays else float(np.mean(emergency_delays)),
            "values": emergency_delays,
        },
        "soc_bins": soc_bins,
        "energy_outcomes": list(env.energy_outcomes),
        "upstream_low_soc_diversions": int(upstream_diversions),
        "in_terminal_low_soc_diversions": int(in_terminal_diversions),
        "failed_energy": int(outcome_counts.get("failed_energy", 0)),
        "safety": combined_safety,
        "run_safety": run_safety,
        "replay_safety": replay_safety,
        "geometry": dict(geometry),
        "replay": dict(replay),
        "replay_error": replay_error,
        "action_tape": str(action_tape),
        "requested_likelihood_exact": requested_likelihood_exact,
        "teacher_enabled": bool(config.teacher_enabled),
        "label_rows": int(raw_summary.get("label_rows", 0)),
        "training_called": bool(raw_summary.get("training_called", False)),
        "runtime_shortlist_checks": runtime_shortlist,
        "hard_gates": gates | {"all": bool(gates) and all(gates.values())},
        "diagnostic_only": True,
        "PPO_ALLOWED": False,
    }


def _run_case(
    policy: Any,
    config: Any,
    seed: int,
    controller: str,
    output: Path,
    *,
    max_steps: int,
) -> dict[str, Any]:
    _require(not output.exists(), f"case output already exists: {output}")
    if controller == "rule":
        raw = dict(rule_episode(config, seed, output, max_steps=max_steps))
        likelihood = None
    elif controller == "original_k3":
        _, raw = rollout(
            policy, config, seed=seed, max_steps=max_steps, output=output,
            deterministic=True, collect_labels=False,
        )
        likelihood = bool(_policy_tape_logprob_exact(policy, config, raw["action_tape"]))
    else:
        raise ProtocolError(f"unknown controller: {controller}")
    tape_path = Path(raw["action_tape"])
    replay_env, replay, geometry, replay_error = _replay_case(config, tape_path)
    record = _lifecycle_metrics(
        replay_env, config, seed, raw_summary=raw, replay=replay,
        geometry=geometry, replay_error=replay_error,
        requested_likelihood_exact=likelihood, controller=controller,
        action_tape=tape_path,
    )
    _write_json(output / "diagnostic.json", record)
    return record


def _delta(left: Any, right: Any) -> float | None:
    try:
        if left is None or right is None:
            return None
        return float(left) - float(right)
    except (TypeError, ValueError):
        return None


def paired_deltas(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["seed"]), {})[str(record["controller"])] = record
    rows = []
    for seed, controllers in sorted(grouped.items()):
        rule = controllers.get("rule")
        policy = controllers.get("original_k3")
        if rule is None or policy is None:
            continue
        policy_delay = policy.get("all_request_horizon_penalized_delay", {}).get("mean_seconds")
        rule_delay = rule.get("all_request_horizon_penalized_delay", {}).get("mean_seconds")
        policy_normal = policy.get("normal_touchdown_delay_seconds", {}).get("mean")
        rule_normal = rule.get("normal_touchdown_delay_seconds", {}).get("mean")
        policy_emergency = policy.get("emergency_touchdown_delay_seconds", {}).get("mean")
        rule_emergency = rule.get("emergency_touchdown_delay_seconds", {}).get("mean")
        rows.append({
            "seed": seed,
            "original_k3_minus_rule": {
                "all_request_horizon_penalized_delay_mean_seconds": _delta(policy_delay, rule_delay),
                "normal_touchdown_delay_mean_seconds": _delta(policy_normal, rule_normal),
                "emergency_touchdown_delay_mean_seconds": _delta(policy_emergency, rule_emergency),
                "successes": int(policy.get("successes", 0)) - int(rule.get("successes", 0)),
                "emergency_landed": int(policy.get("emergency_landed", 0)) - int(rule.get("emergency_landed", 0)),
                "upstream_low_soc_diversions": int(policy.get("upstream_low_soc_diversions", 0)) - int(rule.get("upstream_low_soc_diversions", 0)),
                "in_terminal_low_soc_diversions": int(policy.get("in_terminal_low_soc_diversions", 0)) - int(rule.get("in_terminal_low_soc_diversions", 0)),
                "failed_energy": int(policy.get("failed_energy", 0)) - int(rule.get("failed_energy", 0)),
            },
        })
    return rows


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    safety = {
        key: sum(int(record.get("safety", {}).get(key, 0)) for record in records)
        for key in SAFETY_COUNTER_KEYS
    }
    outcomes = Counter()
    for record in records:
        outcomes.update(record.get("outcome_counts", {}))
    hard_gates = {
        "expected_records": len(records) == len(FORMAL_SEEDS) * len(CONTROLLERS),
        "exact_request_accounting": all(record.get("hard_gates", {}).get("exact_request_accounting", False) for record in records),
        "all_records_terminal": all(record.get("hard_gates", {}).get("all_records_terminal", False) for record in records),
        "zero_safety_events": not any(safety.values()),
        "zero_failed_energy": int(outcomes.get("failed_energy", 0)) == 0,
        "zero_in_terminal_low_soc_diversion": all(record.get("hard_gates", {}).get("zero_in_terminal_low_soc_diversion", False) for record in records),
        "exact_action_replay": all(record.get("hard_gates", {}).get("exact_action_replay", False) for record in records),
        "no_teacher": all(record.get("hard_gates", {}).get("no_teacher", False) for record in records),
        "no_labels": all(record.get("hard_gates", {}).get("no_labels", False) for record in records),
        "no_updates": all(record.get("hard_gates", {}).get("no_updates", False) for record in records),
        "runtime_shortlist_k3": all(record.get("hard_gates", {}).get("runtime_shortlist_k3", False) for record in records),
    }
    return {
        "records": len(records),
        "outcome_counts": dict(sorted(outcomes.items())),
        "safety_totals": safety,
        "hard_gates": hard_gates,
        "PASS": bool(hard_gates) and all(hard_gates.values()),
    }


def run(*, checkpoint: str | Path, output: str | Path, smoke: bool = False) -> dict[str, Any]:
    """Run the smoke case or the two-seed paired diagnostic."""
    protocol = load_protocol()
    output = Path(output)
    _require(not output.exists(), f"refusing to overwrite existing output: {output}")
    checkpoint = Path(checkpoint)
    _require(checkpoint.is_file(), f"checkpoint not found: {checkpoint}")
    _require(_sha256(checkpoint) == CHECKPOINT_SHA256, "checkpoint SHA256 mismatch")
    # Loading validates lineage and ensures no optimizer state is restored;
    # this occurs before the first TerminalParallelEnv reset below.
    policy, payload = effectiveness.load_candidate_checkpoint(
        checkpoint, expected_sha256=CHECKPOINT_SHA256,
    )
    _require(payload.get("PPO_ALLOWED") is False, "checkpoint PPO_ALLOWED binding changed")
    _require(len(policy.optimizer.state) == 0, "checkpoint optimizer state was loaded")

    output.mkdir(parents=True, exist_ok=False)
    smoke_value = protocol["smoke"]
    seeds = (int(smoke_value["seed"]),) if smoke else FORMAL_SEEDS
    max_steps = int(smoke_value["steps"]) if smoke else int(
        float(protocol["scenario"]["horizon_seconds"]) / build_config(protocol).dt
    )
    # Smoke bounds execution steps while retaining the formal 1200 s horizon
    # for nonnegative horizon-penalized delays of arrivals after the bound.
    config = build_config(protocol)
    records: list[dict[str, Any]] = []
    for seed in seeds:
        for controller in CONTROLLERS:
            case_output = output / f"{controller}_seed_{seed}"
            records.append(_run_case(policy, config, seed, controller, case_output, max_steps=max_steps))
            _write_json(output / "progress.json", {
                "completed_records": len(records),
                "expected_records": len(seeds) * len(CONTROLLERS),
                "records": records,
            })

    aggregate = _aggregate(records) if not smoke else {
        "records": len(records),
        "outcome_counts": dict(sorted(
            sum((Counter(record.get("outcome_counts", {})) for record in records), Counter()).items()
        )),
        "safety_totals": {
            key: sum(int(record.get("safety", {}).get(key, 0)) for record in records)
            for key in SAFETY_COUNTER_KEYS
        },
        "hard_gates": {key: False for key in (
            "expected_records", "exact_request_accounting", "all_records_terminal",
            "zero_safety_events", "zero_failed_energy", "zero_in_terminal_low_soc_diversion",
            "exact_action_replay", "no_teacher", "no_labels", "no_updates", "runtime_shortlist_k3",
        )},
        "PASS": False,
    }
    manifest = {
        "protocol": protocol["protocol"],
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_path": str(checkpoint),
        "checkpoint_payload": {key: payload.get(key) for key in ("stage", "architecture_id", "qualification_pass", "PPO_ALLOWED")},
        "source_sha256": {str(path): _sha256(path) for path in SOURCE_PATHS},
        "smoke": bool(smoke),
        "seeds": list(seeds),
        "controllers": list(CONTROLLERS),
        "records": records,
        "paired_deltas": paired_deltas(records),
        "aggregate": aggregate,
        "diagnostic_only": True,
        "training_called": False,
        "teacher_used": False,
        "labels_collected": False,
        "optimizer_loaded": False,
        "PPO_ALLOWED": False,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    manifest = run(checkpoint=args.checkpoint, output=args.output, smoke=args.smoke)
    print(json.dumps({
        "PASS": manifest["aggregate"]["PASS"],
        "smoke": manifest["smoke"],
        "records": len(manifest["records"]),
    }, sort_keys=True), flush=True)
    return 0 if manifest["smoke"] or manifest["aggregate"]["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
