"""Enumerate the legal one-UAV P3R cases for the initial-SOC floor.

The diagnostic runs the nominal rule controller on a flow-from-empty scene
with one request at ``t=10 s``.  Gate and mission distributions are made
one-hot, so the sixteen cases cover every legal gate/mission combination
without relying on a sampled arrival attribute.  Each case is first run at
full SOC to measure the energy used by the uncongested landing, then replayed
with the proposed ``0.55`` initial SOC floor.

This is a bounded simulator diagnostic.  Its PASS result applies only to the
enumerated one-UAV cases and does not establish a guarantee for congested or
multi-UAV traffic.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import (
    diagnose_v4_2_p3_mixed_priority_soc_pilot_v2 as p3_v2,
)
from terminal_marl.training.run_v4_2_curriculum_pilot import episode as rule_episode


STAGE = "V4_2_P3_SOLO_SOC_FLOOR_DIAGNOSTIC_V1"
PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v4.yaml"
)
SCENARIO = "flow6"
REQUEST_TIME_SECONDS = 10.0
ONE_UAV_HORIZON_SECONDS = 910.0
SAFE_DISTANCE_M = 80.0
PROTECTED_FLOOR_M = 40.0
COLLISION_DISTANCE_M = 10.0
ENERGY_CAPACITY = 1500.0
TOUCHDOWN_SOC_FLOOR = 0.25
PROPOSED_RANDOM_SOC_FLOOR = 0.55
DEFAULT_SEED = 930001
DEFAULT_MAX_CASES = 16
GATE_COUNT = 4
MISSION_TYPES = ("standard", "cargo", "emergency", "surveillance")
P3R_LZ_ASSIGNMENT_MODE = "nearest_available"
P3R_RELEASE_ARBITRATION_MODE = "downstream_priority"
P3R_FINAL_QUEUE_CAPACITY = 3


class DiagnosticInputError(ValueError):
    """Raised for malformed diagnostic inputs."""


def _jsonable(value: Any) -> Any:
    """Convert simulator values to strict JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _jsonable(value.value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8"
    )


def load_p3_v4_protocol(path: str | Path = PROTOCOL) -> dict[str, Any]:
    """Load the frozen P3 V4 scenario source used by the existing builder."""
    source = Path(path)
    value = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise DiagnosticInputError("P3 V4 protocol must be a mapping")
    scenarios = value.get("scenarios")
    if not isinstance(scenarios, Mapping) or SCENARIO not in scenarios:
        raise DiagnosticInputError(f"P3 V4 protocol is missing {SCENARIO}")
    attributes = value.get("attributes")
    reward = value.get("reward")
    if not isinstance(attributes, Mapping) or not isinstance(reward, Mapping):
        raise DiagnosticInputError("P3 V4 attributes and reward sections are required")
    return value


def _one_hot(index: int, size: int = 4) -> tuple[float, ...]:
    if not isinstance(index, int) or not 0 <= index < size:
        raise DiagnosticInputError(f"one-hot index must be in [0, {size})")
    return tuple(1.0 if item == index else 0.0 for item in range(size))


def scenario_config(
    gate_index: int,
    mission_index: int,
    *,
    initial_soc: float,
    raw_protocol: Mapping[str, Any] | None = None,
):
    """Build one one-UAV P3 V4 config through ``p3_v2.scenario_config``."""
    try:
        soc = float(initial_soc)
    except (TypeError, ValueError) as exc:
        raise DiagnosticInputError("initial_soc must be finite") from exc
    if not math.isfinite(soc) or not 0.0 < soc <= 1.0:
        raise DiagnosticInputError("initial_soc must satisfy 0 < initial_soc <= 1")

    raw = dict(raw_protocol) if raw_protocol is not None else load_p3_v4_protocol()
    # The existing builder owns the P3 observation/action/schema defaults and
    # the P3 V4 scenario semantics.  The diagnostic changes only cohort size,
    # arrival timing, categorical coverage, SOC mode, and the fixed P3R
    # resource settings required for this solo check.
    config = p3_v2.scenario_config(SCENARIO, raw)
    reward = raw["reward"]
    return replace(
        config,
        total_requested_uavs=1,
        fixed_arrival_schedule=(REQUEST_TIME_SECONDS,),
        max_simulation_seconds=ONE_UAV_HORIZON_SECONDS,
        gate_probabilities=_one_hot(gate_index),
        mission_probabilities=_one_hot(mission_index),
        initial_soc_mode="fixed",
        initial_soc=soc,
        initial_soc_min=None,
        initial_soc_max=None,
        safe_distance=SAFE_DISTANCE_M,
        nmac_distance=PROTECTED_FLOOR_M,
        collision_distance=COLLISION_DISTANCE_M,
        energy_capacity=ENERGY_CAPACITY,
        lz_assignment_mode=P3R_LZ_ASSIGNMENT_MODE,
        release_arbitration_mode=P3R_RELEASE_ARBITRATION_MODE,
        final_queue_capacity=P3R_FINAL_QUEUE_CAPACITY,
        low_level_priority_time_weight=float(
            reward["low_level_priority_time_weight"]
        ),
        low_level_soc_urgency_time_weight=float(
            reward["low_level_soc_urgency_time_weight"]
        ),
        low_level_soc_urgency_threshold=float(
            reward["low_level_soc_urgency_threshold"]
        ),
    )


def build_case_matrix(
    *,
    max_cases: int = DEFAULT_MAX_CASES,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Return the deterministic gate × mission case matrix in row-major order."""
    if not isinstance(max_cases, int) or not 1 <= max_cases <= DEFAULT_MAX_CASES:
        raise DiagnosticInputError(
            f"max_cases must be an integer in [1, {DEFAULT_MAX_CASES}]"
        )
    if not isinstance(seed, int) or seed < 0:
        raise DiagnosticInputError("seed must be a non-negative integer")
    cases: list[dict[str, Any]] = []
    for gate_index in range(GATE_COUNT):
        for mission_index, mission_name in enumerate(MISSION_TYPES):
            cases.append(
                {
                    "case_id": f"solo_gate{gate_index}_{mission_name}",
                    "seed": seed,
                    "gate_index": gate_index,
                    "mission_index": mission_index,
                    "mission_type": mission_name,
                }
            )
    return cases[:max_cases]


def _safety_counters(summary: Mapping[str, Any]) -> dict[str, int]:
    values = summary.get("safety", {})
    if not isinstance(values, Mapping):
        values = {}
    counters: dict[str, int] = {}
    for key, value in values.items():
        try:
            counters[str(key)] = int(value)
        except (TypeError, ValueError):
            counters[str(key)] = 1
    return counters


def _touchdown_observation(
    config: Any, tape_path: str | Path, summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay a nominal rule tape and extract the completed UAV evidence."""
    env = TerminalParallelEnv(config)
    replay = SafetyTraceRecorder.replay_action_tape(env, tape_path)
    completed = list(env.completed_uavs)
    if not completed:
        return {
            "replay_exact": bool(replay["exact"]),
            "touchdown": False,
            "touchdown_time_seconds": None,
            "energy_consumed_at_touchdown": None,
            "sampled_gate_index": None,
            "sampled_mission_type": None,
            "sampled_initial_soc": None,
            "outcome": None,
            "completed_count": 0,
            "replay_steps": int(replay.get("steps_replayed", 0)),
            "replay_expected_steps": int(replay.get("steps_expected", 0)),
            "safety_counters": _safety_counters(summary),
        }
    if len(completed) != 1:
        raise RuntimeError(f"solo replay produced {len(completed)} completed UAVs")
    uav = completed[0]
    touchdown_time = None if uav.touchdown_time is None else float(uav.touchdown_time)
    sampled_initial_soc = float(
        (uav.energy_remaining + uav.energy_consumed) / uav.energy_capacity
    )
    return {
        "replay_exact": bool(replay["exact"]),
        "touchdown": touchdown_time is not None,
        "touchdown_time_seconds": touchdown_time,
        "energy_consumed_at_touchdown": float(uav.energy_consumed),
        "sampled_gate_index": int(uav.assigned_gate),
        "sampled_mission_type": str(uav.mission_type.value),
        "sampled_initial_soc": sampled_initial_soc,
        "outcome": None
        if uav.terminal_outcome is None
        else str(uav.terminal_outcome.value),
        "completed_count": 1,
        "replay_steps": int(replay.get("steps_replayed", 0)),
        "replay_expected_steps": int(replay.get("steps_expected", 0)),
        "safety_counters": _safety_counters(summary),
    }


def _run_rule_case(
    case: Mapping[str, Any],
    *,
    raw_protocol: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Run full-SOC and fixed-floor rule episodes for one matrix row."""
    gate_index = int(case["gate_index"])
    mission_index = int(case["mission_index"])
    seed = int(case["seed"])
    case_dir = output_dir / str(case["case_id"])
    full_dir = case_dir / "soc_1p0"
    floor_dir = case_dir / "soc_0p55"
    full_config = scenario_config(
        gate_index, mission_index, initial_soc=1.0, raw_protocol=raw_protocol
    )
    floor_config = scenario_config(
        gate_index,
        mission_index,
        initial_soc=PROPOSED_RANDOM_SOC_FLOOR,
        raw_protocol=raw_protocol,
    )
    full_summary = rule_episode(full_config, seed, full_dir, max_steps=None)
    full_tape = full_summary.get("action_tape")
    if not isinstance(full_tape, str) or not full_tape:
        raise RuntimeError("full-SOC rule episode did not produce an action tape")
    full = _touchdown_observation(full_config, full_tape, full_summary)

    floor_summary = rule_episode(floor_config, seed, floor_dir, max_steps=None)
    floor_tape = floor_summary.get("action_tape")
    if not isinstance(floor_tape, str) or not floor_tape:
        raise RuntimeError("fixed-floor rule episode did not produce an action tape")
    floor = _touchdown_observation(floor_config, floor_tape, floor_summary)

    expected_mission = MISSION_TYPES[mission_index]

    def request_matches(observation: Mapping[str, Any], expected_soc: float) -> bool:
        actual_soc = observation.get("sampled_initial_soc")
        return bool(
            observation.get("sampled_gate_index") == gate_index
            and observation.get("sampled_mission_type") == expected_mission
            and actual_soc is not None
            and math.isclose(float(actual_soc), expected_soc, rel_tol=0.0, abs_tol=1e-9)
        )

    full_request_match = request_matches(full, 1.0)
    floor_request_match = request_matches(floor, PROPOSED_RANDOM_SOC_FLOOR)
    required_soc = None
    if full["energy_consumed_at_touchdown"] is not None:
        required_soc = TOUCHDOWN_SOC_FLOOR + float(
            full["energy_consumed_at_touchdown"]
        ) / ENERGY_CAPACITY
    full_safety_violation = any(value != 0 for value in full["safety_counters"].values())
    floor_safety_violation = any(
        value != 0 for value in floor["safety_counters"].values()
    )
    full_pass = bool(
        full["touchdown"]
        and full["outcome"] == "completed"
        and full["replay_exact"]
        and full_request_match
        and not full_safety_violation
    )
    floor_pass = bool(
        floor["touchdown"]
        and floor["outcome"] == "completed"
        and floor["replay_exact"]
        and floor_request_match
        and not floor_safety_violation
    )
    return {
        **dict(case),
        "config": {
            "safe_distance_m": SAFE_DISTANCE_M,
            "protected_floor_m": PROTECTED_FLOOR_M,
            "collision_distance_m": COLLISION_DISTANCE_M,
            "energy_capacity": ENERGY_CAPACITY,
            "touchdown_soc_floor": TOUCHDOWN_SOC_FLOOR,
            "request_time_seconds": REQUEST_TIME_SECONDS,
            "reset_mode": full_config.reset_mode,
            "lz_assignment_mode": full_config.lz_assignment_mode,
            "release_arbitration_mode": full_config.release_arbitration_mode,
            "final_queue_capacity": full_config.final_queue_capacity,
        },
        "full_soc_1p0": {
            **full,
            "summary": {
                "terminal": bool(full_summary.get("terminal")),
                "completed": int(full_summary.get("completed", 0)),
                "seconds": float(full_summary.get("seconds", 0.0)),
                "action_tape": str(full_tape),
            },
            "safety_violation": full_safety_violation,
            "request_match": full_request_match,
            "pass": full_pass,
        },
        "fixed_soc_0p55": {
            **floor,
            "summary": {
                "terminal": bool(floor_summary.get("terminal")),
                "completed": int(floor_summary.get("completed", 0)),
                "seconds": float(floor_summary.get("seconds", 0.0)),
                "action_tape": str(floor_tape),
            },
            "safety_violation": floor_safety_violation,
            "request_match": floor_request_match,
            "pass": floor_pass,
        },
        "required_initial_soc": required_soc,
        "floor_margin": None
        if required_soc is None
        else float(PROPOSED_RANDOM_SOC_FLOOR - required_soc),
        "pass": bool(full_pass and floor_pass and required_soc is not None),
    }


def aggregate_records(
    records: Sequence[Mapping[str, Any]], *,
    expected_case_count: int | None = None,
    configured_soc_floor: float = PROPOSED_RANDOM_SOC_FLOOR,
) -> dict[str, Any]:
    """Aggregate case evidence and apply only the bounded solo hard gates."""
    required = [
        float(row["required_initial_soc"])
        for row in records
        if row.get("required_initial_soc") is not None
    ]
    max_required = max(required) if required else None
    safety_clean = all(
        not bool(row.get("full_soc_1p0", {}).get("safety_violation"))
        and not bool(row.get("fixed_soc_0p55", {}).get("safety_violation"))
        for row in records
    )
    gates = {
        "all_cases_present": len(records) > 0,
        "expected_case_count": expected_case_count is None
        or len(records) == int(expected_case_count),
        "all_case_pass_flags": bool(records)
        and all(bool(row.get("pass")) for row in records),
        "full_soc_touchdown_all": bool(records)
        and all(bool(row.get("full_soc_1p0", {}).get("touchdown")) for row in records),
        "full_soc_safety_all_zero": bool(records)
        and all(
            not bool(row.get("full_soc_1p0", {}).get("safety_violation"))
            for row in records
        ),
        "fixed_soc_touchdown_all": bool(records)
        and all(bool(row.get("fixed_soc_0p55", {}).get("touchdown")) for row in records),
        "fixed_soc_safety_all_zero": bool(records)
        and all(
            not bool(row.get("fixed_soc_0p55", {}).get("safety_violation"))
            for row in records
        ),
        "required_soc_defined_all": bool(records)
        and len(required) == len(records)
        and all(math.isfinite(value) for value in required),
        "configured_floor_sufficient": max_required is not None
        and float(configured_soc_floor) >= max_required,
    }
    return {
        "case_count": len(records),
        "expected_case_count": None
        if expected_case_count is None
        else int(expected_case_count),
        "max_required_initial_soc": max_required,
        "configured_soc_floor": float(configured_soc_floor),
        "margin_above_max_required_soc": None
        if max_required is None
        else float(configured_soc_floor - max_required),
        "safety_all_zero": bool(safety_clean),
        "gates": gates,
        "PASS": bool(gates) and all(gates.values()),
        "scope": "enumerated one-UAV flow-from-empty gate/mission cases only",
    }


def run(
    *,
    output_dir: Path,
    max_cases: int = DEFAULT_MAX_CASES,
    seed: int = DEFAULT_SEED,
    protocol_path: str | Path = PROTOCOL,
) -> dict[str, Any]:
    """Run the bounded diagnostic and write ``manifest.json``."""
    if output_dir.exists():
        raise DiagnosticInputError(f"output directory already exists: {output_dir}")
    raw_protocol = load_p3_v4_protocol(protocol_path)
    cases = build_case_matrix(max_cases=max_cases, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for case in cases:
        records.append(
            _run_rule_case(case, raw_protocol=raw_protocol, output_dir=output_dir)
        )
        _write_json(
            output_dir / "progress.json",
            {"stage": STAGE, "completed_cases": len(records), "expected_cases": len(cases)},
        )
    aggregate = aggregate_records(records, expected_case_count=len(cases))
    manifest = {
        "stage": STAGE,
        "seed": int(seed),
        "max_cases": int(max_cases),
        "protocol": str(protocol_path),
        "controller": "rule_only",
        "records": records,
        "aggregate": aggregate,
        "PASS": bool(aggregate["PASS"]),
        "training_called": False,
        "PPO_ALLOWED": False,
        "scope": aggregate["scope"],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=DEFAULT_MAX_CASES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = run(
        output_dir=args.output_dir,
        max_cases=args.max_cases,
        seed=args.seed,
        protocol_path=args.protocol,
    )
    print(
        json.dumps(
            {
                "stage": manifest["stage"],
                "PASS": manifest["PASS"],
                "case_count": manifest["aggregate"]["case_count"],
                "max_required_initial_soc": manifest["aggregate"][
                    "max_required_initial_soc"
                ],
                "margin_above_max_required_soc": manifest["aggregate"][
                    "margin_above_max_required_soc"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
