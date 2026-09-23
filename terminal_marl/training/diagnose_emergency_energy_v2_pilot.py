"""Bounded rule-only diagnostic for the emergency-energy V2 curriculum.

This module deliberately has no optimizer, teacher, policy, or checkpoint
imports.  It executes a frozen protocol and writes one JSON record per case.
The formal matrix is only started through the command line; tests use the
separate seed 840000 and a short step limit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from terminal_marl.core.config import SimConfig
from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.envs.parallel_env import TerminalParallelEnv


PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "configs" / "low_v4_2" / "emergency_energy_v2_pilot.yaml"
PROTOCOL_SHA256 = "2c76e342c9edfe397a298fcc01362e021f4fc56be2d94c2e3f9cfdc3ed71f66b"
REPAIR_PROTOCOL_PATH = PROTOCOL_PATH.with_name("emergency_energy_v2_repair_pilot.yaml")
REPAIR_PROTOCOL_SHA256 = "e99cb11f8cf9756f26643f6e3fae8dc294394b22275ff9b846acafdf00377bc8"
ROUTE_REPAIR_PROTOCOL_PATH = PROTOCOL_PATH.with_name("emergency_energy_v2_route_repair_pilot.yaml")
ROUTE_REPAIR_PROTOCOL_SHA256 = "8c13dcf6a4b5f3b7f965d01429f4c9e1ffed403e8c21cbc5e7fb0d574f2b0108"
SOURCE_PATHS = (
    Path(__file__),
    Path(__file__).resolve().parents[1] / "core" / "admission.py",
    Path(__file__).resolve().parents[1] / "core" / "energy.py",
    Path(__file__).resolve().parents[1] / "core" / "trajectory_generator.py",
    Path(__file__).resolve().parents[1] / "envs" / "parallel_env.py",
)
DEV_SEEDS = (840001, 840002)
ROUTE_REPAIR_SEEDS = (840201, 840202)
TRIGGER_SOCS = (0.35, 0.40)
ARRIVAL_SCHEDULE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)


class ProtocolError(ValueError):
    """Raised when the frozen diagnostic protocol does not bind exactly."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Load and validate every frozen protocol binding before an env reset."""

    path = Path(path)
    expected_hash = {
        PROTOCOL_PATH.name: PROTOCOL_SHA256,
        REPAIR_PROTOCOL_PATH.name: REPAIR_PROTOCOL_SHA256,
        ROUTE_REPAIR_PROTOCOL_PATH.name: ROUTE_REPAIR_PROTOCOL_SHA256,
    }.get(path.name)
    _require(expected_hash is not None and _sha256(path) == expected_hash, "protocol SHA256 mismatch")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProtocolError(f"protocol file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ProtocolError(f"invalid YAML protocol: {path}") from exc
    _require(isinstance(payload, dict), "protocol root must be a mapping")
    protocol_name = payload.get("protocol")
    repair_protocol = protocol_name == "EMERGENCY_ENERGY_V2_REPAIR_PILOT"
    route_repair_protocol = protocol_name == "EMERGENCY_ENERGY_V2_ROUTE_REPAIR_PILOT"
    expected_protocol = (
        "EMERGENCY_ENERGY_V2_ROUTE_REPAIR_PILOT" if route_repair_protocol
        else "EMERGENCY_ENERGY_V2_REPAIR_PILOT" if repair_protocol
        else "EMERGENCY_ENERGY_V2_PILOT"
    )
    expected_status = (
        "PREREGISTERED_ROUTE_REPAIR_DIAGNOSTIC" if route_repair_protocol
        else "PREREGISTERED_REPAIR_DIAGNOSTIC" if repair_protocol
        else "PREREGISTERED_DIAGNOSTIC"
    )
    _require(payload.get("protocol") == expected_protocol, "unexpected protocol name")
    _require(payload.get("status") == expected_status, "protocol must be preregistered diagnostic")

    freeze = payload.get("freeze")
    _require(isinstance(freeze, dict), "freeze section is required")
    for key in ("diagnostic_only", "no_training", "no_optimizer", "no_teacher", "no_checkpoint"):
        _require(freeze.get(key) is True, f"freeze.{key} must be true")
    if repair_protocol or route_repair_protocol:
        _require(freeze.get("no_rerun_no_reset_no_tuning") is True,
                 "freeze.no_rerun_no_reset_no_tuning must be true")
    _require(freeze.get("ppo_allowed") is False, "freeze.ppo_allowed must be false")

    scenario = payload.get("scenario")
    _require(isinstance(scenario, dict), "scenario section is required")
    expected_scenario = {
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
        "emergency_landing_enabled": True,
        "emergency_energy_margin_seconds": 15,
    }
    for key, expected in expected_scenario.items():
        _require(scenario.get(key) == expected, f"scenario.{key} must equal {expected!r}")
    schedule = tuple(float(value) for value in scenario.get("fixed_arrival_schedule", ()))
    _require(schedule == ARRIVAL_SCHEDULE, "scenario.fixed_arrival_schedule does not match the 6+6 schedule")
    triggers = tuple(float(value) for value in scenario.get("emergency_trigger_soc_values", ()))
    expected_triggers = (0.40,) if repair_protocol or route_repair_protocol else TRIGGER_SOCS
    _require(triggers == expected_triggers, f"scenario.emergency_trigger_soc_values must be {expected_triggers}")

    safety = payload.get("safety")
    _require(isinstance(safety, dict), "safety section is required")
    for key, expected in (("safe_distance_m", 80), ("protected_floor_m", 40), ("collision_distance_m", 10)):
        _require(safety.get(key) == expected, f"safety.{key} must equal {expected}")

    matrix = payload.get("matrix")
    _require(isinstance(matrix, dict), "matrix section is required")
    seeds = tuple(int(value) for value in matrix.get("seeds", ()))
    expected_seeds = (
        ROUTE_REPAIR_SEEDS if route_repair_protocol
        else (840101, 840102) if repair_protocol
        else DEV_SEEDS
    )
    expected_records = 2 if repair_protocol or route_repair_protocol else 4
    _require(seeds == expected_seeds, f"matrix.seeds must be {expected_seeds}")
    _require(matrix.get("expected_records") == expected_records,
             f"matrix.expected_records must be {expected_records}")

    gates = payload.get("hard_gates")
    _require(isinstance(gates, dict), "hard_gates section is required")
    for key in ("exact_request_accounting", "zero_hard_separation", "zero_nmac", "zero_collision",
                "zero_failed_energy", "at_least_one_emergency_landed"):
        _require(gates.get(key) is True, f"hard_gates.{key} must be true")
    if repair_protocol or route_repair_protocol:
        _require(gates.get("all_records_terminal") is True, "repair requires terminal records")
        _require(gates.get("zero_in_terminal_low_soc_diversion") is True,
                 "repair requires explicit in-terminal energy outcomes")
        repair = payload.get("repair")
        _require(isinstance(repair, dict), "repair section is required")
        if route_repair_protocol:
            _require(repair.get("predecessor_protocols") == [
                "EMERGENCY_ENERGY_V2_PILOT", "EMERGENCY_ENERGY_V2_REPAIR_PILOT"
            ], "route repair predecessor protocols changed")
            _require(repair.get("predecessor_seeds_consumed") == [840001, 840002, 840101, 840102],
                     "route repair predecessor seeds changed")
            _require(repair.get("continuous_tangential_emergency_departure") is True,
                     "route repair continuous departure binding changed")
        else:
            _require(repair.get("predecessor_seeds_consumed") == [840001, 840002],
                     "repair predecessor seeds changed")
        _require(repair.get("phase_motion_nominal_solo_energy") == 240.9,
                 "repair solo energy binding changed")
        _require(repair.get("active_low_soc_outcome_must_be_explicit") is True,
                 "repair explicit outcome binding changed")

    authorization = payload.get("authorization")
    _require(isinstance(authorization, dict), "authorization section is required")
    _require(authorization.get("PPO_ALLOWED") is False, "authorization.PPO_ALLOWED must be false")
    _require(authorization.get("training_allowed") is False, "authorization.training_allowed must be false")
    _require(authorization.get("checkpoint_promotion_allowed") is False,
             "authorization.checkpoint_promotion_allowed must be false")
    return payload


def build_config(protocol: Mapping[str, Any], trigger_soc: float, *, max_simulation_seconds: float | None = None) -> SimConfig:
    """Bind one trigger value to the frozen simulator configuration."""

    scenario = protocol["scenario"]
    safety = protocol["safety"]
    return SimConfig(
        reset_mode=str(scenario["reset_mode"]),
        warm_start_uavs=int(scenario["warm_start_uavs"]),
        warm_start_steady_motion=bool(scenario["warm_start_steady_motion"]),
        total_requested_uavs=int(scenario["total_requested_uavs"]),
        arrival_model="finite_cohort",
        arrival_rate_per_second=0.0,
        fixed_arrival_schedule=ARRIVAL_SCHEDULE,
        initial_soc_mode=str(scenario["initial_soc_mode"]),
        initial_soc_min=float(scenario["initial_soc_min"]),
        initial_soc_max=float(scenario["initial_soc_max"]),
        max_simulation_seconds=float(max_simulation_seconds if max_simulation_seconds is not None else scenario["horizon_seconds"]),
        lz_assignment_mode=str(scenario["lz_assignment_mode"]),
        release_arbitration_mode=str(scenario["release_arbitration_mode"]),
        final_queue_capacity=int(scenario["final_queue_capacity"]),
        energy_model_mode=str(scenario["energy_model_mode"]),
        emergency_landing_enabled=bool(scenario["emergency_landing_enabled"]),
        emergency_trigger_soc=float(trigger_soc),
        emergency_energy_margin_seconds=float(scenario["emergency_energy_margin_seconds"]),
        phase_motion_nominal_solo_energy=float(
            protocol.get("repair", {}).get("phase_motion_nominal_solo_energy", 240.9)
        ),
        safe_distance=float(safety["safe_distance_m"]),
        nmac_distance=float(safety["protected_floor_m"]),
        collision_distance=float(safety["collision_distance_m"]),
        experiment_mode="fixed_rule",
        high_level_enabled=False,
        teacher_enabled=False,
        teacher_release_label_enabled=False,
        scheduler_mode="execution",
        shield_enabled=True,
    )


def case_matrix(protocol: Mapping[str, Any], *, seeds: Sequence[int] | None = None,
                triggers: Sequence[float] | None = None) -> list[dict[str, Any]]:
    matrix = protocol["matrix"]
    selected_seeds = tuple(int(value) for value in (matrix["seeds"] if seeds is None else seeds))
    selected_triggers = tuple(float(value) for value in (
        protocol["scenario"]["emergency_trigger_soc_values"] if triggers is None else triggers
    ))
    _require(selected_seeds and selected_triggers, "diagnostic matrix must contain at least one seed and trigger")
    return [
        {
            "case_id": f"seed_{seed}_trigger_{trigger:.2f}".replace(".", "p"),
            "seed": seed,
            "trigger_soc": trigger,
        }
        for seed in selected_seeds
        for trigger in selected_triggers
    ]


def _terminal(env: TerminalParallelEnv) -> bool:
    return bool(
        env.arrivals.exhausted()
        and not env.active
        and not env.admission.queue.items
        and not env.service_uavs
    )


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _record_for_episode(env: TerminalParallelEnv, *, case: Mapping[str, Any], steps: int,
                        terminal: bool, horizon_reached: bool,
                        safety_error: str | None) -> dict[str, Any]:
    requested_ids = set(env.ledger.requested)
    outcome_rows = {str(arrival_id): outcome.value for arrival_id, outcome in env.ledger.outcomes.items()}
    outcome_counts = Counter(outcome_rows.values())
    in_terminal_low_soc_diversions = sum(
        row.get("outcome") == "diverted_low_soc" and row.get("phase") != "external_queue"
        for row in env.energy_outcomes
    )
    upstream_low_soc_diversions = sum(
        row.get("outcome") == "diverted_low_soc" and row.get("phase") == "external_queue"
        for row in env.energy_outcomes
    )
    unresolved_ids = sorted(int(value) for value in env.ledger.unresolved)
    accounting_identity = (
        requested_ids == set(env.ledger.outcomes) | set(unresolved_ids)
        and set(env.ledger.outcomes).isdisjoint(set(unresolved_ids))
        and len(requested_ids) == len(env.ledger.outcomes) + len(unresolved_ids)
    )
    protocol_total = int(env.config.total_requested_uavs or 0)
    exact_request_accounting = bool(accounting_identity and len(requested_ids) == protocol_total and not unresolved_ids)

    telemetry = env.telemetry
    safety_counters = {
        "collisions": int(telemetry.get("collisions", 0)),
        "nmac": int(telemetry.get("nmac", 0)),
        "hard_separation_violations": int(telemetry.get("hard_separation_violations", 0)),
        "uncertified_execution": int(telemetry.get("uncertified_execution", 0)),
        "SafetyInvariantError": int(safety_error is not None),
    }
    emergency_keys = (
        "emergency_attempts", "emergency_committed", "emergency_blocked_safety",
        "emergency_blocked_energy", "emergency_blocked_lz", "emergency_blocked_dynamics",
        "emergency_touchdowns",
    )
    emergency_telemetry = {key: int(telemetry.get(key, 0)) for key in emergency_keys}
    minimum_separations = {
        "actual_m": _finite_or_none(telemetry.get("minimum_actual_separation")),
        "swept_m": _finite_or_none(telemetry.get("minimum_swept_separation")),
    }
    return {
        "case_id": case["case_id"],
        "seed": int(case["seed"]),
        "emergency_trigger_soc": float(case["trigger_soc"]),
        "diagnostic_only": True,
        "PPO_ALLOWED": False,
        "training_called": False,
        "teacher_used": False,
        "checkpoint_used": False,
        "steps": int(steps),
        "seconds": float(env.now),
        "terminal": bool(terminal),
        "horizon_reached": bool(horizon_reached),
        "requested": len(requested_ids),
        "requested_ids": sorted(int(value) for value in requested_ids),
        "resolved": len(env.ledger.outcomes),
        "unresolved_ids": unresolved_ids,
        "exact_request_accounting": exact_request_accounting,
        "accounting_identity": accounting_identity,
        "outcomes": outcome_rows,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "energy_outcomes": list(env.energy_outcomes),
        "in_terminal_low_soc_diversions": int(in_terminal_low_soc_diversions),
        "upstream_low_soc_diversions": int(upstream_low_soc_diversions),
        "emergency_telemetry": emergency_telemetry,
        "minimum_separations": minimum_separations,
        "safety_counters": safety_counters,
        "safety_error": safety_error,
    }


def run_episode(config: SimConfig, *, case: Mapping[str, Any], output: Path,
                max_steps: int | None = None) -> dict[str, Any]:
    """Run one rule-only case until terminal or the configured step bound."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = TerminalParallelEnv(config)
    # Protocol validation is performed by run_matrix before this seeded reset.
    env.reset(seed=int(case["seed"]))
    limit = int(max_steps if max_steps is not None else math.ceil(config.max_simulation_seconds / config.dt))
    if limit <= 0:
        raise ValueError("max_steps must be positive")
    terminal = False
    safety_error: str | None = None
    steps = 0
    for _ in range(limit):
        try:
            env.step({})
        except SafetyInvariantError as exc:
            safety_error = str(exc)
            steps += 1
            break
        steps += 1
        terminal = _terminal(env)
        if terminal:
            break
    terminal = _terminal(env)
    horizon_reached = bool(not terminal and steps >= limit)
    record = _record_for_episode(
        env,
        case=case,
        steps=steps,
        terminal=terminal,
        horizon_reached=horizon_reached,
        safety_error=safety_error,
    )
    _write_json_atomic(output / "summary.json", record)
    return record


def aggregate_records(records: Sequence[Mapping[str, Any]], *, expected_records: int,
                      require_explicit_terminal_energy: bool = False) -> dict[str, Any]:
    outcome_counts: Counter[str] = Counter()
    for record in records:
        outcome_counts.update(record.get("outcome_counts", {}))
    safety_totals = {
        key: sum(int(record.get("safety_counters", {}).get(key, 0)) for record in records)
        for key in ("collisions", "nmac", "hard_separation_violations", "SafetyInvariantError")
    }
    gates = {
        "expected_records": len(records) == int(expected_records),
        "exact_request_accounting": all(bool(record.get("exact_request_accounting")) for record in records),
        "zero_hard_separation": safety_totals["hard_separation_violations"] == 0,
        "zero_nmac": safety_totals["nmac"] == 0,
        "zero_collision": safety_totals["collisions"] == 0,
        "zero_failed_energy": int(outcome_counts.get("failed_energy", 0)) == 0,
        "at_least_one_emergency_landed": int(outcome_counts.get("emergency_landed", 0)) >= 1,
    }
    if require_explicit_terminal_energy:
        gates["all_records_terminal"] = all(bool(record.get("terminal")) for record in records)
        gates["zero_in_terminal_low_soc_diversion"] = sum(
            int(record.get("in_terminal_low_soc_diversions", 0)) for record in records
        ) == 0
    return {
        "records": len(records),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "safety_totals": safety_totals,
        "hard_gates": gates,
        "PASS": bool(all(gates.values())),
    }


def run_matrix(output: Path, *, protocol_path: Path = PROTOCOL_PATH,
               seeds: Sequence[int] | None = None, triggers: Sequence[float] | None = None,
               max_steps: int | None = None) -> dict[str, Any]:
    """Validate protocol, refuse overwrite, and execute a bounded matrix."""

    protocol = load_protocol(Path(protocol_path))
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    cases = case_matrix(protocol, seeds=seeds, triggers=triggers)
    records: list[dict[str, Any]] = []
    for case in cases:
        config = build_config(protocol, float(case["trigger_soc"]))
        record = run_episode(config, case=case, output=output / str(case["case_id"]), max_steps=max_steps)
        records.append(record)
        _write_json_atomic(output / "progress.json", {"completed_cases": len(records), "total_cases": len(cases), "records": records})
    repair_protocol = protocol["protocol"] in {
        "EMERGENCY_ENERGY_V2_REPAIR_PILOT",
        "EMERGENCY_ENERGY_V2_ROUTE_REPAIR_PILOT",
    }
    aggregate = aggregate_records(records, expected_records=len(cases),
                                  require_explicit_terminal_energy=repair_protocol)
    manifest = {
        "protocol": protocol["protocol"],
        "protocol_path": str(Path(protocol_path)),
        "protocol_sha256": _sha256(Path(protocol_path)),
        "runner_sha256": _sha256(Path(__file__)),
        "source_sha256": {str(path): _sha256(path) for path in SOURCE_PATHS},
        "diagnostic_only": True,
        "PPO_ALLOWED": False,
        "training_called": False,
        "teacher_used": False,
        "checkpoint_used": False,
        "cases": cases,
        "records": records,
        "aggregate": aggregate,
    }
    _write_json_atomic(output / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory for JSON diagnostic records")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH,
                        help="frozen V1, repair, or route-repair diagnostic protocol")
    args = parser.parse_args(argv)
    manifest = run_matrix(args.output, protocol_path=args.protocol)
    print(json.dumps({"PASS": manifest["aggregate"]["PASS"], "records": len(manifest["records"])}, sort_keys=True), flush=True)
    return 0 if manifest["aggregate"]["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
