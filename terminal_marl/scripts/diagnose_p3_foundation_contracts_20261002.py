"""Read-only P3 foundation-contract probes.

This module intentionally does not import torch, a policy, a training runner,
or any tape/evaluation helper.  It constructs a zero-arrival environment and
the same deterministic inner-ring holder used by the final-approach unit
tests.  The output is an observational JSON artifact; it is not a rollout or
qualification result.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
import sys
from typing import Any

import numpy as np

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.energy_reachability import minimum_immediate_emergency_soc
from terminal_marl.core.energy import EnergyModel
from terminal_marl.core.merge_scheduler import (
    LowLevelIntent,
    MergeReservation,
    canonical_plan_slot_time,
)
from terminal_marl.core.uav import (
    ArrivalRequest,
    FlightPhase,
    MISSION_SERVICE_SECONDS,
    MissionType,
    QueuedArrivalState,
    UAVState,
)
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot


PROPOSAL_VALUES = (-1.0, -0.5, 0.0, 0.5, 1.0)


def _config(*, geometry_mode: str, observation_schema: str) -> SimConfig:
    """Build the task-neutral phase-motion V4.2 diagnostic configuration."""

    return SimConfig(
        action_schema_version="low_action_schema_v4_2_hierarchical",
        observation_schema_version=observation_schema,
        arrival_model="finite_cohort",
        total_requested_uavs=0,
        arrival_rate_per_second=0.0,
        fixed_arrival_schedule=(),
        max_simulation_seconds=120.0,
        final_approach_geometry_mode=geometry_mode,
        runtime_proposal_shortlist_k=3,
        v4_2_rejected_release_hold_mode="canonical_wait",
        scheduler_mode="execution",
        teacher_enabled=False,
        high_level_enabled=False,
        shield_enabled=True,
        reconcile_late_merge_slots_enabled=False,
        task_priority_enabled=False,
        low_level_priority_time_weight=0.0,
        low_level_soc_urgency_time_weight=2.0,
        energy_model_mode="phase_motion_v2",
        phase_motion_nominal_solo_energy=240.9,
        emergency_landing_enabled=True,
        emergency_trigger_soc=0.40,
        emergency_energy_margin_seconds=15.0,
        lz_assignment_mode="nearest_available",
        release_arbitration_mode="downstream_priority",
        final_queue_capacity=3,
    )


def _env(*, geometry_mode: str, observation_schema: str = "phaseA_low_obs_v2_1_retry_state") -> TerminalParallelEnv:
    return TerminalParallelEnv(
        _config(geometry_mode=geometry_mode, observation_schema=observation_schema)
    )


def _ring_holder(
    env: TerminalParallelEnv,
    *,
    mission: MissionType = MissionType.STANDARD,
    role: RingRole = RingRole.INNER,
    theta: float = 0.0,
    speed: float | None = None,
    soc: float = 0.80,
    uid: int = 0,
    base_priority: int = 1,
) -> UAVState:
    """Create the deterministic ring-holder fixture used by the unit tests."""

    config = env.config
    request = ArrivalRequest(
        uid + 1,
        0.0,
        0,
        mission,
        base_priority,
        config.energy_capacity,
        soc,
    )
    queued = QueuedArrivalState(request, request.energy_capacity * soc)
    uav = UAVState.from_request(uid, request, queued, np.zeros(3), env.now)
    uav.phase = {
        RingRole.OUTER: FlightPhase.RING_OUTER,
        RingRole.MIDDLE: FlightPhase.RING_MIDDLE,
        RingRole.INNER: FlightPhase.RING_INNER,
    }[role]
    uav.ring_role = role
    uav.ring_progress = float(theta)
    ring = env.airspace.actual_ring(role)
    nominal_speed = float(ring.nominal_speed if speed is None else speed)
    uav.position = np.asarray(
        (ring.radius * math.cos(theta), ring.radius * math.sin(theta), ring.height),
        dtype=float,
    )
    uav.speed = nominal_speed
    uav.velocity = np.asarray(
        (-math.sin(theta) * nominal_speed, math.cos(theta) * nominal_speed, 0.0),
        dtype=float,
    )
    uav.acceleration = np.asarray(
        (
            -math.cos(theta) * nominal_speed**2 / ring.radius,
            -math.sin(theta) * nominal_speed**2 / ring.radius,
            0.0,
        ),
        dtype=float,
    )
    env.active[uav.uid] = uav
    env._refresh_airspace_occupancy()
    return uav


def _error_payload(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _final_probe(geometry_mode: str) -> dict[str, Any]:
    env = _env(geometry_mode=geometry_mode)
    uav = _ring_holder(env)
    lz_id = env.lz_manager.choose_rule_zone(uav, env.now)
    if lz_id is None:
        return {
            "geometry_mode": geometry_mode,
            "proposal_count": 25,
            "error": {"type": "NoLandingZone", "message": "choose_rule_zone returned None"},
        }

    snapshot = WorldSnapshot(env.now, env.active, env.airspace)
    rows: list[dict[str, Any]] = []
    index = 0
    for speed_residual in PROPOSAL_VALUES:
        for merge_residual in PROPOSAL_VALUES:
            intent = LowLevelIntent(1.0, speed_residual, merge_residual)
            row: dict[str, Any] = {
                "proposal_index": index,
                "speed_residual": speed_residual,
                "merge_time_residual": merge_residual,
                "speed_residual_mps": speed_residual * env.config.max_speed_residual_mps,
                "candidate_delay_seconds": 0.0,
                "lz_id": int(lz_id),
            }
            try:
                proposal = env.scheduler.propose(
                    uav, intent, snapshot, "final", delay_seconds=0.0
                )
                reservation = MergeReservation(
                    f"p3-contract-{geometry_mode}-{index}",
                    uav.uid,
                    "ring_inner",
                    "final",
                    float(proposal.merge_time),
                    proposal.merge_angle,
                    None,
                    None,
                )
                plan = env.generator.final_approach(
                    uav,
                    env.lz_manager.zones[lz_id],
                    reservation,
                    env.now,
                    scheduled_delay=0.0,
                    speed_residual_mps=float(
                        speed_residual * env.config.max_speed_residual_mps
                    ),
                    airspace=env.airspace,
                )
                dynamics_errors = env.generator.validate_kinematics(plan)
                safety = env.safety.filter_plan(plan, {}, snapshot)
                canonical_departure = canonical_plan_slot_time(plan)
                row.update(
                    {
                        "scheduler_requested_slot": float(proposal.merge_time),
                        "scheduler_headway": float(proposal.headway),
                        "canonical_departure": float(canonical_departure),
                        "scheduled_departure_metadata": float(
                            plan.metadata["scheduled_departure"]
                        ),
                        "plan_start": float(plan.start_time),
                        "plan_end": float(plan.end_time),
                        "duration": float(plan.end_time - plan.start_time),
                        "trajectory_hash": plan.trajectory_hash,
                        "initial_position_error_m": float(
                            np.linalg.norm(plan.points[0].position - uav.position)
                        ),
                        "initial_velocity_error_mps": float(
                            np.linalg.norm(plan.points[0].velocity - uav.velocity)
                        ),
                        "dynamics_errors": list(dynamics_errors),
                        "safety_approved": bool(safety.approved),
                        "safety_reason": safety.reason,
                        "safety_checked_plan_id": safety.checked_plan_id,
                        "safety_checked_trajectory_hash": safety.checked_trajectory_hash,
                    }
                )
            except Exception as exc:  # Diagnostic output keeps each failed case explicit.
                row["error"] = _error_payload(exc)
            rows.append(row)
            index += 1

    def _unique(key: str) -> list[Any]:
        values = [row[key] for row in rows if key in row and "error" not in row]
        if not values:
            return []
        if isinstance(values[0], (int, float)):
            return sorted({round(float(value), 9) for value in values})
        return sorted(set(values))

    return {
        "geometry_mode": geometry_mode,
        "fixture": {
            "role": RingRole.INNER.value,
            "theta": 0.0,
            "speed_mps": float(uav.speed),
            "soc": float(uav.soc),
            "mission": uav.mission_type.value,
            "lz_id": int(lz_id),
            "same_pad_free_at_probe": env.lz_manager.zones[lz_id].status.value == "free",
        },
        "proposal_grid": list(PROPOSAL_VALUES),
        "proposal_count": len(rows),
        "rows": rows,
        "aggregate": {
            "rows_with_error": sum("error" in row for row in rows),
            "unique_trajectory_hashes": _unique("trajectory_hash"),
            "unique_canonical_departures": _unique("canonical_departure"),
            "unique_durations": _unique("duration"),
            "unique_scheduler_requested_slots": _unique("scheduler_requested_slot"),
        },
    }


def _observation_probe() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    observations88: dict[str, np.ndarray] = {}
    missions = {
        "standard": MissionType.STANDARD,
        "cargo": MissionType.CARGO,
        "emergency": MissionType.EMERGENCY,
        "surveillance": MissionType.SURVEILLANCE,
    }
    for label, mission in missions.items():
        row: dict[str, Any] = {
            "requested_label": label,
            "mission_enum": mission.value,
            "service_seconds": float(MISSION_SERVICE_SECONDS[mission]),
        }
        try:
            env88 = _env(
                geometry_mode="brake_bezier_v1",
                observation_schema="phaseA_low_obs_v2_1_retry_state",
            )
            env90 = _env(
                geometry_mode="brake_bezier_v1",
                observation_schema="phaseA_low_obs_v2_4_service_context_90d",
            )
            uav88 = _ring_holder(env88, mission=mission)
            uav90 = _ring_holder(env90, mission=mission)
            obs88 = np.asarray(env88._observations()[str(uav88.uid)], dtype=np.float32)
            obs90 = np.asarray(env90._observations()[str(uav90.uid)], dtype=np.float32)
            observations88[label] = obs88.copy()
            prefix_delta = np.abs(obs88.astype(np.float64) - obs90[:88].astype(np.float64))
            changed = np.flatnonzero(prefix_delta > 0.0).astype(int).tolist()
            row.update(
                {
                    "obs88_dimension": int(obs88.size),
                    "obs90_dimension": int(obs90.size),
                    "prefix_equal": bool(np.array_equal(obs88, obs90[:88])),
                    "changed_indices_in_88_prefix": changed,
                    "max_prefix_abs_delta": float(prefix_delta.max(initial=0.0)),
                    "obs90_service_context": obs90[88:].astype(float).tolist(),
                    "obs88": obs88.astype(float).tolist(),
                    "obs90": obs90.astype(float).tolist(),
                }
            )
        except Exception as exc:
            row["error"] = _error_payload(exc)
        rows.append(row)

    cross_mission_pairs: list[dict[str, Any]] = []
    baseline = observations88.get("standard")
    if baseline is not None:
        for label in missions:
            if label == "standard":
                continue
            candidate = observations88.get(label)
            if candidate is None:
                cross_mission_pairs.append(
                    {
                        "baseline_mission": "standard",
                        "candidate_mission": label,
                        "error": {
                            "type": "MissingObservation",
                            "message": "candidate 88D observation was not produced",
                        },
                    }
                )
                continue
            delta = np.abs(baseline.astype(np.float64) - candidate.astype(np.float64))
            cross_mission_pairs.append(
                {
                    "baseline_mission": "standard",
                    "candidate_mission": label,
                    "byte_equal": bool(np.array_equal(baseline, candidate)),
                    "differing_indices": np.flatnonzero(delta > 0.0).astype(int).tolist(),
                    "max_abs_delta": float(delta.max(initial=0.0)),
                }
            )
    return {
        "task_priority_enabled": False,
        "rows": rows,
        "cross_mission_88d_pairs": cross_mission_pairs,
    }


def _emergency_probe() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    config_trigger = 0.40
    for role in (RingRole.OUTER, RingRole.MIDDLE, RingRole.INNER):
        for theta_label, theta in (("theta0", 0.0), ("theta_pi", math.pi)):
            for speed_label, speed in (("nominal", None), ("zero", 0.0)):
                env = _env(geometry_mode="brake_bezier_v1")
                uav = _ring_holder(
                    env,
                    role=role,
                    theta=theta,
                    speed=speed,
                    soc=0.80,
                )
                row: dict[str, Any] = {
                    "ring_role": role.value,
                    "theta_label": theta_label,
                    "theta": theta,
                    "speed_label": speed_label,
                    "speed_mps": float(uav.speed),
                    "config_emergency_trigger_soc": config_trigger,
                }
                try:
                    required = minimum_immediate_emergency_soc(
                        uav,
                        role,
                        env.now,
                        env.airspace,
                        env.lz_manager,
                        env.generator,
                        EnergyModel(env.config),
                        env.config,
                    )
                    row.update(
                        {
                            "minimum_immediate_emergency_soc": float(required),
                            "trigger_minus_required_soc": float(config_trigger - required),
                            "trigger_ge_required_soc": bool(config_trigger >= required),
                            "scope": "solo_energy_only_no_safety_guarantee",
                        }
                    )
                except Exception as exc:
                    row["error"] = _error_payload(exc)
                rows.append(row)
    return {
        "energy_model_mode": "phase_motion_v2",
        "rows": rows,
        "row_count": len(rows),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _technical_error_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return rows with execution errors; semantic findings are not errors."""

    failures: list[dict[str, Any]] = []
    for mode, probe in result["final_approach"].items():
        if "error" in probe:
            failures.append({"probe": f"final_approach.{mode}", "error": probe["error"]})
        for row in probe.get("rows", []):
            if "error" in row:
                failures.append({"probe": f"final_approach.{mode}", "row": row})
    for row in result["observation_service_context"].get("rows", []):
        if "error" in row:
            failures.append({"probe": "observation_service_context", "row": row})
    for row in result["minimum_immediate_emergency_soc"].get("rows", []):
        if "error" in row:
            failures.append({"probe": "minimum_immediate_emergency_soc", "row": row})
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="JSON artifact path")
    args = parser.parse_args()

    result = {
        "diagnostic": "p3_foundation_contracts_20261002",
        "synthetic_fixture": True,
        "diagnostic_only": True,
        "no_torch_models_training_tape_or_heldout": True,
        "scope": {
            "scene": "zero-arrival environment plus deterministic synthetic ring holders",
            "no_rollout": True,
            "no_policy_model": True,
            "no_training_or_optimizer": True,
            "no_tape_or_heldout": True,
            "configuration_is_synthetic_effective_probe_config": True,
        },
        "runtime": {
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "argv": list(sys.argv),
        },
        "config": {
            "action_schema_version": "low_action_schema_v4_2_hierarchical",
            "observation_schema_v2_1_dimension": 88,
            "observation_schema_v2_4_dimension": 90,
            "task_priority_enabled": False,
            "energy_model_mode": "phase_motion_v2",
            "runtime_proposal_shortlist_k": 3,
            "emergency_trigger_soc": 0.40,
            "min_landing_soc": 0.10,
            "soc_reserve": 0.25,
            "final_queue_capacity": 3,
            "lz_assignment_mode": "nearest_available",
            "release_arbitration_mode": "downstream_priority",
            "max_speed_residual_mps": 2.0,
            "max_merge_time_residual_seconds": 6.0,
            "reward_weights_and_scales": {
                "low_level_priority_time_weight": 0.0,
                "low_level_soc_urgency_time_weight": 2.0,
                "low_level_soc_urgency_threshold": 0.65,
                "low_level_time_penalty_per_second": 0.01,
                "low_level_progress_reward_per_meter": 0.001,
                "low_level_touchdown_reward": 25.0,
                "low_level_release_approved_reward": 0.10,
                "low_level_release_rejected_penalty": 0.05,
                "low_level_collision_penalty": 25.0,
                "low_level_energy_failure_penalty": 20.0,
                "low_level_emergency_landing_penalty": 8.0,
            },
        },
        "final_approach": {
            "brake_bezier_v1": _final_probe("brake_bezier_v1"),
            "continuous_inward_v1": _final_probe("continuous_inward_v1"),
        },
        "observation_service_context": _observation_probe(),
        "minimum_immediate_emergency_soc": _emergency_probe(),
    }
    technical_errors = _technical_error_rows(result)
    result["technical_errors"] = technical_errors
    status = "TECHNICAL_ERROR" if technical_errors else "CONTRACTS_READY"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "output": str(args.output),
                "technical_error_count": len(technical_errors),
            }
        )
    )
    return 1 if technical_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
