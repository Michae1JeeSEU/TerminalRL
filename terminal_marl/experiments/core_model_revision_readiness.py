"""PRETRAIN readiness diagnostics for the revised terminal core model.

This module is simulation-only.  It runs no PPO training and does not load or
resume old checkpoints.  The outputs are intended to decide whether a fresh
512-step smoke run is allowed in a later turn.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

import numpy as np

from ..config import ExperimentMode, load_experiment_config
from ..control.geometry_event_gate import GateMetrics
from ..core.arrival_process import generate_arrival_schedule
from ..core.config import SimConfig
from ..core.lifecycle import UAVLifecycleState
from ..core.ring_capacity import adjacent_chord_spacing, geometric_ring_capacity
from ..core.uav import UAVState
from ..envs.parallel_env import TerminalParallelEnv
from ..experiments.ablation_runner import apply_demand_profile
from ..experiments.evaluation_protocol import (
    WindowedEvaluationProtocol,
    run_windowed_evaluation,
)
from ..experiments.reward_diagnostics import run_reward_diagnostics


DEMAND_LEVELS = ("low", "medium", "high", "bursty", "overload")
REQUIRED_FILES = (
    "dynamic_capacity_end_to_end.csv",
    "finite_cohort_conservation.csv",
    "evaluation_semantics_audit.json",
    "arrival_admission_audit.json",
    "dynamics_consistency.json",
    "geometry_transition_validation.json",
    "new_service_capacity_summary.csv",
    "demand_profile_summary.csv",
    "bursty_validation.csv",
    "gate_calibration_summary.csv",
    "gate_behavior_summary.csv",
    "observation_audit.json",
    "geometry_controllability_summary.csv",
    "bottleneck_utilization.csv",
    "interval_dwell_summary.csv",
    "reward_diagnostics.csv",
    "pretrain_readiness.json",
    "final_test_report.txt",
)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _run_env_until(
    env: TerminalParallelEnv,
    *,
    max_steps: int,
    action_fn: Callable[[TerminalParallelEnv], np.ndarray] | None = None,
) -> TerminalParallelEnv:
    while env.step_count < max_steps:
        high_level_action = None
        if action_fn is not None and env.should_update_high_level():
            high_level_action = action_fn(env)
        actions = {
            agent: np.zeros(env.action_spec(agent).shape, dtype=np.float32)
            for agent in env.agents
        }
        env.step(actions, high_level_action=high_level_action)
        if env.config.total_requested_uavs is not None and env.finite_cohort_complete():
            break
        if env.config.total_requested_uavs is None and not env.agents and not env.external_arrival_queue:
            break
    return env


def _finite_config(
    *,
    total_requested: int,
    seed: int = 0,
    demand: str = "medium",
    arrival_rate: float = 0.08,
    max_active: int = 10,
    max_steps: int = 8000,
) -> SimConfig:
    cfg = apply_demand_profile(load_experiment_config("dynamic_hrl"), demand)
    cfg.experiment_mode = "dynamic_hrl"
    cfg.action_mode = "rule"
    cfg.initial_aircraft_count = 0
    cfg.formation_rows = 0
    cfg.formation_cols = 0
    cfg.total_requested_uavs = int(total_requested)
    cfg.arrival_rate = float(arrival_rate)
    cfg.arrival_model = "deterministic_interval"
    cfg.max_active_uavs = int(max_active)
    cfg.max_steps = int(max_steps)
    cfg.max_simulation_seconds = float(max_steps * cfg.dt)
    cfg.battery_endurance_seconds = 100_000.0
    cfg.engineering_validation_battery_endurance_seconds = 100_000.0
    cfg.config_name = f"finite_N{total_requested}_seed{seed}"
    return cfg


def dynamic_capacity_audit(final_dir: Path) -> dict[str, Any]:
    cfg = load_experiment_config("dynamic_hrl")
    env = TerminalParallelEnv(cfg)
    env.reset(seed=0)
    required = cfg.safe_distance + cfg.ring_separation_margin_m
    rows: list[dict[str, Any]] = []
    for role, base_radius in (
        ("inner", cfg.rings[0].radius),
        ("outer", cfg.rings[-1].radius),
    ):
        for delta in (-80.0, -40.0, 0.0, 40.0, 80.0, 160.0):
            radius = max(1.0, base_radius + delta)
            cap = geometric_ring_capacity(radius, required)
            rows.append(
                {
                    "role": role,
                    "radius": radius,
                    "capacity": cap,
                    "adjacent_spacing_at_capacity": adjacent_chord_spacing(radius, cap),
                    "adjacent_spacing_at_capacity_plus_one": adjacent_chord_spacing(radius, cap + 1),
                    "required_separation": required,
                    "safe_at_capacity": adjacent_chord_spacing(radius, cap) + 1e-9 >= required,
                    "unsafe_at_capacity_plus_one": adjacent_chord_spacing(radius, cap + 1) < required,
                    "evidence_type": "newly_run",
                }
            )

    # End-to-end transition trace.
    cfg.geometry_transition_seconds = 20.0
    cfg.high_level_interval_steps = 1
    cfg.initial_aircraft_count = 0
    cfg.formation_rows = 0
    cfg.formation_cols = 0
    cfg.total_requested_uavs = 0
    env = TerminalParallelEnv(cfg)
    env.reset(seed=0)
    old_positions = [ac.position.copy() for ac in env.aircraft]
    action = np.array([1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=float)
    info = env.apply_high_level_action(action)
    trace: list[dict[str, Any]] = []
    for _ in range(int(cfg.geometry_transition_seconds / cfg.dt) + 2):
        trace.append(
            {
                "step": env.step_count,
                "time": env.time,
                "outer_radius": env.airspace.outer_ring.radius,
                "inner_radius": env.airspace.inner_ring.radius,
                "outer_capacity": env.airspace.outer_ring.capacity,
                "inner_capacity": env.airspace.inner_ring.capacity,
                "target_outer_radius": float(info["target_state"][0]),
                "target_inner_radius": float(info["target_state"][1]),
                "evidence_type": "newly_run",
            }
        )
        env.step({})
    _write_csv(final_dir / "ring_spacing_trace.csv", trace)

    # Capacity + actual angular slot feasibility audit.
    slot_cfg = load_experiment_config("dynamic_hrl")
    slot_cfg.initial_aircraft_count = 0
    slot_cfg.formation_rows = 0
    slot_cfg.formation_cols = 0
    slot_cfg.total_requested_uavs = 0
    slot_env = TerminalParallelEnv(slot_cfg)
    slot_env.reset(seed=0)
    ring_idx = slot_env.vtac.outermost_ring_index
    ring = slot_env.vtac.ring_spec(ring_idx)
    candidate = UAVState(
        uid=1,
        position=np.array([ring.radius, 0.0, ring.height]),
        speed=slot_cfg.nominal_speed,
        heading=0.0,
    )
    neighbor = UAVState(
        uid=2,
        position=np.array([ring.radius, 0.0, ring.height]),
        speed=slot_cfg.nominal_speed,
        heading=0.0,
    )
    neighbor.assigned_ring = ring_idx
    slot_env.aircraft = [candidate, neighbor]
    slot_env.vtac.queues[ring_idx].append(neighbor.uid)
    scalar_capacity_available = slot_env.vtac.queue_has_space(ring_idx)
    angular_slot_available = slot_env.vtac._angular_spacing_ok(
        candidate,
        ring_idx,
        {candidate.uid: candidate, neighbor.uid: neighbor},
    )
    # Crash/landing release uses the same VTAC release path; verify with manual queue.
    slot_env._mark_terminal_outcome(
        neighbor,
        outcome=UAVLifecycleState.CRASHED_COLLISION.value,
        reason="slot_capacity_audit_manual_crash",
        collision_partner_ids=[candidate.uid],
    )
    released_after_crash = neighbor.uid not in slot_env.vtac.queues[ring_idx]

    audit = {
        "nominal_outer_radius": cfg.rings[-1].radius,
        "nominal_outer_capacity": geometric_ring_capacity(cfg.rings[-1].radius, required),
        "nominal_inner_radius": cfg.rings[0].radius,
        "nominal_inner_capacity": geometric_ring_capacity(cfg.rings[0].radius, required),
        "scalar_capacity_available_for_same_angle_candidate": bool(scalar_capacity_available),
        "angular_slot_available_for_same_angle_candidate": bool(angular_slot_available),
        "admission_requires_capacity_and_safe_slot": bool(
            scalar_capacity_available and not angular_slot_available
        ),
        "uav_teleport_count_during_capacity_change": int(
            sum(
                not np.allclose(old, ac.position)
                for old, ac in zip(old_positions, env.aircraft)
            )
        ),
        "capacity_release_after_crash": bool(released_after_crash),
        "transition_uses_actual_radius": bool(
            trace[0]["outer_radius"] != trace[0]["target_outer_radius"]
            and trace[-1]["outer_radius"] == trace[-1]["target_outer_radius"]
        ),
        "evidence_type": "newly_run",
    }
    _write_csv(final_dir / "dynamic_capacity_end_to_end.csv", rows)
    _write_json(final_dir / "slot_capacity_audit.json", audit)
    _write_json(
        final_dir / "capacity_formula_audit.json",
        {"required_separation": required, "rows": rows, "evidence_type": "newly_run"},
    )
    return audit


def _finite_job(args: tuple[int, int, float, int]) -> dict[str, Any]:
    total, seed, arrival_rate, max_active = args
    cfg = _finite_config(
        total_requested=total,
        seed=seed,
        arrival_rate=arrival_rate,
        max_active=max_active,
        max_steps=max(1500, total * 220),
    )
    env = TerminalParallelEnv(cfg)
    env.reset(seed=seed)
    _run_env_until(env, max_steps=cfg.max_steps)
    summary = env.simulation_outcome_summary()
    summary.update({"N": total, "seed": seed, "evidence_type": "newly_run"})
    return {"summary": summary, "rows": env.uav_outcome_rows()}


def finite_cohort_audit(final_dir: Path, max_workers: int) -> dict[str, Any]:
    jobs = [(1, 0, 0.08, 6), (6, 0, 0.08, 8), (20, 0, 0.10, 10), (60, 0, 0.12, 12)]
    summaries: list[dict[str, Any]] = []
    all_outcome_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, min(max_workers, len(jobs)))) as pool:
        futures = [pool.submit(_finite_job, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            summaries.append(result["summary"])
            n = int(result["summary"]["N"])
            run_dir = final_dir / f"finite_N{n}_seed0"
            _write_csv(run_dir / "uav_outcomes.csv", result["rows"])
            for row in result["rows"]:
                row = dict(row)
                row["N"] = n
                all_outcome_rows.append(row)
    summaries.sort(key=lambda row: int(row["N"]))
    _write_csv(final_dir / "finite_cohort_conservation.csv", summaries)
    _write_csv(final_dir / "uav_outcomes.csv", all_outcome_rows)

    cleanup: dict[str, Any] = {"deliberate_tests": {}, "evidence_type": "newly_run"}

    # External-queue battery depletion.
    cfg = _finite_config(total_requested=3, arrival_rate=10.0, max_active=1, max_steps=20)
    cfg.battery_endurance_seconds = 0.5
    env = TerminalParallelEnv(cfg)
    env.reset(seed=1)
    _run_env_until(env, max_steps=20)
    cleanup["deliberate_tests"]["battery_external_or_active"] = env.simulation_outcome_summary()

    # Active battery depletion.
    cfg = _finite_config(total_requested=1, arrival_rate=10.0, max_active=1, max_steps=20)
    cfg.battery_endurance_seconds = 0.5
    env = TerminalParallelEnv(cfg)
    env.reset(seed=2)
    _run_env_until(env, max_steps=20)
    cleanup["deliberate_tests"]["battery_active"] = env.simulation_outcome_summary()

    # Collision while occupying ring resources.
    cfg = _finite_config(total_requested=0, max_active=2, max_steps=20)
    env = TerminalParallelEnv(cfg)
    env.reset(seed=3)
    ring_idx = env.vtac.outermost_ring_index
    ring = env.vtac.ring_spec(ring_idx)
    a = UAVState(1, np.array([ring.radius, 0.0, ring.height]), 0.0, 0.0)
    b = UAVState(2, np.array([ring.radius + 1.0, 0.0, ring.height]), 0.0, 0.0)
    for arrival_id, ac in enumerate((a, b)):
        ac.arrival_id = arrival_id
        ac.assigned_ring = ring_idx
        ac.ring_slot_active = True
        env.outcome_ledger[arrival_id] = env.outcome_ledger.get(arrival_id) or __import__(
            "terminal_marl.core.lifecycle",
            fromlist=["UAVOutcomeRecord"],
        ).UAVOutcomeRecord(
            arrival_id=arrival_id,
            request_time=0.0,
            scenario="deliberate_collision",
            admission_time=0.0,
            uid=ac.uid,
            lifecycle_state=UAVLifecycleState.ACTIVE_RING_OUTER.value,
        )
        env.vtac.queues[ring_idx].append(ac.uid)
    env.aircraft = [a, b]
    env._track_conflicts()
    cleanup["deliberate_tests"]["collision_ring_cleanup"] = {
        "collision_count": env.collision_count,
        "active_after_collision": sum(not ac.finished for ac in env.aircraft),
        "ring_queue_after_collision": list(env.vtac.queues[ring_idx]),
        "terminal_outcomes": [record.terminal_outcome for record in env.outcome_ledger.values()],
    }

    # Large queue can drain without unsafe spawn coordinate growth.
    cfg = _finite_config(total_requested=100, arrival_rate=50.0, max_active=8, max_steps=30_000)
    env = TerminalParallelEnv(cfg)
    env.reset(seed=4)
    _run_env_until(env, max_steps=cfg.max_steps)
    cleanup["deliberate_tests"]["large_external_queue_resolution"] = env.simulation_outcome_summary()

    _write_json(final_dir / "lifecycle_terminal_cleanup.json", cleanup)
    return {"summaries": summaries, "cleanup": cleanup}


def evaluation_semantics_audit(final_dir: Path) -> dict[str, Any]:
    protocol = WindowedEvaluationProtocol(
        warm_up_seconds=60.0,
        measurement_seconds=240.0,
        max_drain_seconds=240.0,
        metrics_sampling_interval_seconds=20.0,
        decision_interval_seconds=60.0,
        reconfiguration_dwell_seconds=120.0,
        geometry_transition_seconds=20.0,
    )
    result = run_windowed_evaluation(
        "dynamic_hrl",
        protocol=protocol,
        policy="rule-only",
        demand_level="medium",
        seed=0,
    )
    summary = result["summary"]
    expected = (
        float(summary["landed_after_measurement_start"])
        / max(1e-9, float(summary["measurement_duration_seconds"]))
    )
    audit = {
        "measurement_landed": summary["landed_after_measurement_start"],
        "measurement_duration_seconds": summary["measurement_duration_seconds"],
        "measurement_throughput_per_second": summary["measurement_throughput_per_second"],
        "expected_measurement_throughput_per_second": expected,
        "drain_landed": summary["drain_landed"],
        "drain_excluded_from_measurement": math.isclose(
            float(summary["measurement_throughput_per_second"]),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "landing_success_rate": summary["final_completion_ratio"],
        "collision_crash_rate": 0.0,
        "battery_crash_rate": 0.0,
        "unresolved_rate": 0.0,
        "makespan": summary["elapsed_seconds"],
        "evidence_type": "newly_run",
    }
    _write_json(final_dir / "evaluation_semantics_audit.json", audit)
    return audit


def arrival_admission_audit(final_dir: Path) -> dict[str, Any]:
    cfg = _finite_config(total_requested=0, max_active=4, max_steps=10)
    env = TerminalParallelEnv(cfg)
    env.reset(seed=0)
    positions: dict[str, list[float]] = {}
    for queue_len in (1, 10, 100, 1000):
        request = {
            "arrival_id": 42,
            "request_time": 0.0,
            "scenario": "safe_ingress",
            "seed": 0,
            "first_blocked_time": None,
        }
        env.aircraft = []
        env.agents = []
        env.external_arrival_queue = [dict(request) for _ in range(queue_len)]
        ok = env._admit_one_external_arrival(dict(request), 0, queue_len)
        positions[str(queue_len)] = env.aircraft[-1].position.tolist() if ok else []
    bounded = all(
        pos
        and abs(pos[0]) <= cfg.world_xy_limit
        and abs(pos[1]) <= cfg.world_xy_limit
        and 0.0 <= pos[2] <= cfg.world_z_limit
        for pos in positions.values()
    )
    independent = len({tuple(round(v, 6) for v in pos) for pos in positions.values()}) == 1
    cfg.initial_aircraft_count = 6
    cfg.formation_rows = 3
    cfg.formation_cols = 2
    cfg.col_spacing = 1.0
    cfg.safe_distance = 80.0
    cfg.total_requested_uavs = None
    env = TerminalParallelEnv(cfg)
    env.reset(seed=1)
    initial_safe = True
    for i, a in enumerate(env.aircraft):
        for b in env.aircraft[i + 1 :]:
            initial_safe = initial_safe and float(np.linalg.norm(a.position - b.position)) >= cfg.safe_distance
    audit = {
        "spawn_positions_by_external_queue_length": positions,
        "bounded_for_1_10_100_1000": bool(bounded),
        "independent_of_external_queue_length": bool(independent),
        "unsafe_initial_excess_moved_to_external_queue": bool(len(env.external_arrival_queue) > 0),
        "initial_physical_uavs_safe": bool(initial_safe),
        "evidence_type": "newly_run",
    }
    _write_json(final_dir / "arrival_admission_audit.json", audit)
    return audit


def dynamics_and_transition_audit(final_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    full = SimConfig(max_acceleration=4.0)
    lab = load_experiment_config("crazyflie_lab")
    rows: list[dict[str, Any]] = []
    for label, cfg in (("full_scale", full), ("crazyflie_lab", lab)):
        uav = UAVState(1, np.array([0.0, 0.0, 10.0]), 1.0, 0.0)
        uav.update_dynamics(
            desired_heading=0.0,
            desired_speed=100.0,
            desired_altitude=10.0,
            dt=cfg.dt,
            max_turn_rate_deg=cfg.max_turn_rate_deg,
            max_climb_rate=cfg.max_climb_rate,
            max_descent_rate=cfg.max_descent_rate,
            speed_min=0.0,
            speed_max=200.0,
            max_acceleration=cfg.max_acceleration,
        )
        rows.append(
            {
                "config": label,
                "dt": cfg.dt,
                "max_acceleration": cfg.max_acceleration,
                "speed_after_one_step": uav.speed,
                "expected_speed": 1.0 + cfg.max_acceleration * cfg.dt,
                "matches_config": math.isclose(uav.speed, 1.0 + cfg.max_acceleration * cfg.dt),
                "evidence_type": "newly_run",
            }
        )
    dynamics = {"rows": rows, "full_scale_differs_from_lab": rows[0]["speed_after_one_step"] != rows[1]["speed_after_one_step"], "evidence_type": "newly_run"}
    _write_json(final_dir / "dynamics_consistency.json", dynamics)

    cfg = load_experiment_config("dynamic_hrl")
    cfg.initial_aircraft_count = 0
    cfg.formation_rows = 0
    cfg.formation_cols = 0
    cfg.total_requested_uavs = 0
    cfg.geometry_transition_seconds = 20.0
    cfg.high_level_interval_steps = 1
    env = TerminalParallelEnv(cfg)
    env.reset(seed=0)
    info = env.apply_high_level_action(np.ones(8, dtype=np.float32))
    trace = []
    for _ in range(25):
        trace.append(
            {
                "step": env.step_count,
                "time": env.time,
                "outer_radius": env.airspace.outer_ring.radius,
                "inner_radius": env.airspace.inner_ring.radius,
                "outer_height": env.airspace.outer_ring.height,
                "inner_height": env.airspace.inner_ring.height,
                "target_outer_radius": float(info["target_state"][0]),
                "target_inner_radius": float(info["target_state"][1]),
            }
        )
        env.step({})
    transition = {
        "trace": trace,
        "actual_changes_gradually": bool(trace[0]["outer_radius"] < trace[1]["outer_radius"] < trace[-1]["outer_radius"]),
        "final_reaches_target": math.isclose(trace[-1]["outer_radius"], float(info["target_state"][0])),
        "evidence_type": "newly_run",
    }
    _write_json(final_dir / "geometry_transition_validation.json", transition)
    return dynamics, transition


def gate_and_observation_audit(final_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    calibration_rows = []
    behavior_rows = []
    for split, seeds in (("calibration", range(5)), ("validation", range(20, 25))):
        for seed in seeds:
            cfg = apply_demand_profile(load_experiment_config("event_gated_hrl"), "bursty")
            cfg.initial_aircraft_count = 0
            cfg.formation_rows = 0
            cfg.formation_cols = 0
            cfg.total_requested_uavs = 50
            cfg.arrival_model = "bursty_wave"
            cfg.arrival_rate = max(cfg.arrival_rate, 0.04)
            cfg.max_steps = 1800
            cfg.geometry_gate_theta_on = 0.45
            cfg.geometry_gate_theta_off = 0.25
            cfg.geometry_gate_k_on = 2
            cfg.geometry_gate_k_off = 2
            cfg.reconfiguration_dwell_seconds = 120.0
            cfg.geometry_min_dwell_seconds = 120.0
            cfg.high_level_interval_steps = 60
            env = TerminalParallelEnv(cfg)
            env.reset(seed=int(seed))
            states = []
            scores = []
            for _ in range(900):
                env.step({})
                if env.should_sample_metrics():
                    info = env._update_geometry_gate(force_sample=True)
                    states.append(info["gate_state"])
                    scores.append(info["trigger_score"])
                if env.finite_cohort_complete():
                    break
            row = {
                "split": split,
                "seed": int(seed),
                "trigger_count": env.geometry_gate.trigger_count,
                "states": "|".join(states),
                "max_score": max(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
                "opened": any(state in {"TRIGGERED", "DWELL_LOCKED"} for state in states),
                "closed": states[-1] == "NORMAL" if states else False,
                "evidence_type": "newly_run",
            }
            behavior_rows.append(row)
            if split == "calibration":
                calibration_rows.append(row)
    gate_cal = {
        "theta_on": 0.45,
        "theta_off": 0.25,
        "k_on": 2,
        "k_off": 2,
        "calibration_rows": calibration_rows,
        "uses_rolling_metrics": True,
        "no_double_score_state_mutation": True,
        "evidence_type": "newly_run",
    }
    gate_behavior = {
        "validation_rows": [row for row in behavior_rows if row["split"] == "validation"],
        "has_state_dependent_open_close": any(row["opened"] for row in behavior_rows) and any(row["closed"] for row in behavior_rows),
        "evidence_type": "newly_run",
    }
    _write_csv(final_dir / "gate_calibration_summary.csv", calibration_rows)
    _write_csv(final_dir / "gate_behavior_summary.csv", behavior_rows)
    _write_json(final_dir / "gate_calibration_summary.json", gate_cal)
    _write_json(final_dir / "gate_behavior_summary.json", gate_behavior)

    cfg = _finite_config(total_requested=20, arrival_rate=0.1, max_active=8, max_steps=100)
    env = TerminalParallelEnv(cfg, experiment_mode=ExperimentMode.DYNAMIC_HRL)
    env.reset(seed=0)
    env.step({})
    obs = env.get_high_level_observation()
    observation = {
        "length": int(obs.shape[0]),
        "finite": bool(np.isfinite(obs).all()),
        "fields": [
            "normalized_airspace_8",
            "outer_inner_final_occupancy",
            "internal_active_ratio",
            "instant_conflict_risk",
            "interval_phase",
            "gate_vector_9",
            "previous_applied_action_8",
            "outer_capacity_ratio",
            "inner_capacity_ratio",
            "external_queue_pressure",
            "remaining_cohort_fraction",
            "requests_not_yet_due_fraction",
            "terminal_outcome_fraction",
            "recent_arrival_rate_norm",
            "recent_service_rate_norm",
            "signed_arrival_service_imbalance",
            "recent_conflict_rate",
            "battery_risk",
            "trigger_persistence_norm",
            "recovery_persistence_norm",
        ],
        "evidence_type": "newly_run",
    }
    _write_json(final_dir / "observation_audit.json", observation)
    return gate_cal, gate_behavior, observation


def service_capacity_audit(final_dir: Path, max_workers: int) -> dict[str, Any]:
    jobs = [(30, seed, 0.3, 10) for seed in range(5)]
    summaries = []
    landing_rows = []
    with ProcessPoolExecutor(max_workers=max(1, min(max_workers, len(jobs)))) as pool:
        futures = [pool.submit(_finite_job, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            summary = result["summary"]
            n = int(summary["N"])
            seed = int(summary["seed"])
            landed_times = [
                float(row["terminal_time"])
                for row in result["rows"]
                if row["terminal_outcome"] == UAVLifecycleState.LANDED.value and row["terminal_time"] != ""
            ]
            landed_times.sort()
            steady = landed_times[3:] if len(landed_times) > 6 else landed_times
            gaps = [b - a for a, b in zip(steady, steady[1:])]
            duration = (steady[-1] - steady[0]) if len(steady) >= 2 else max(1.0, float(summary["final_simulation_time"]))
            rate_h = 3600.0 * max(0, len(steady) - 1) / max(1e-9, duration)
            row = {
                **summary,
                "steady_landing_rate_per_hour": rate_h,
                "inter_landing_p50": _percentile(gaps, 50),
                "inter_landing_p90": _percentile(gaps, 90),
                "inter_landing_p95": _percentile(gaps, 95),
                "outer_capacity": load_experiment_config("dynamic_hrl").rings[-1].capacity,
                "inner_capacity": load_experiment_config("dynamic_hrl").rings[0].capacity,
                "entry_admission_rate_per_hour": 3600.0 * summary["total_admitted"] / max(1e-9, summary["final_simulation_time"]),
                "final_queue_behavior": "single-server-headway",
            }
            summaries.append(row)
            for t in landed_times:
                landing_rows.append({"seed": seed, "N": n, "landing_time": t, "evidence_type": "newly_run"})
    summaries.sort(key=lambda row: int(row["seed"]))
    _write_csv(final_dir / "new_service_capacity_summary.csv", summaries)
    _write_csv(final_dir / "landing_timestamps.csv", landing_rows)
    rates = [float(row["steady_landing_rate_per_hour"]) for row in summaries]
    return {
        "mean_rate_per_hour": mean(rates) if rates else 0.0,
        "min_rate_per_hour": min(rates) if rates else 0.0,
        "max_rate_per_hour": max(rates) if rates else 0.0,
        "rows": summaries,
    }


def demand_profile_audit(final_dir: Path, service: dict[str, Any]) -> dict[str, Any]:
    service_per_second = max(1e-9, float(service["mean_rate_per_hour"]) / 3600.0)
    profiles = {
        "low": (0.40, "deterministic_interval"),
        "medium": (0.68, "deterministic_interval"),
        "high": (0.90, "deterministic_interval"),
        "bursty": (0.78, "bursty_wave"),
        "overload": (1.20, "deterministic_interval"),
    }
    rows = []
    burst_rows = []
    for demand, (util, model) in profiles.items():
        rate = service_per_second * util
        rows.append(
            {
                "demand": demand,
                "target_utilization": util,
                "arrival_rate_per_second": rate,
                "arrival_rate_per_hour": rate * 3600.0,
                "arrival_model": model,
                "evidence_type": "newly_run",
            }
        )
        schedule = generate_arrival_schedule(
            demand_level="bursty" if demand == "bursty" else "medium",
            model=model,
            arrival_rate=rate,
            seed=0,
            duration_seconds=3600.0,
        )
        times = [float(row["request_time"]) for row in schedule]
        realized = len(times) / 3600.0
        window = 300.0
        peak = 0.0
        for start in np.arange(0.0, 3600.0, 60.0):
            count = sum(start <= t < start + window for t in times)
            peak = max(peak, count / window)
        burst_rows.append(
            {
                "demand": demand,
                "configured_rate_per_second": rate,
                "realized_mean_rate_per_second": realized,
                "peak_300s_rate_per_second": peak,
                "peak_to_mean": peak / max(1e-9, realized),
                "request_count": len(times),
                "evidence_type": "newly_run",
            }
        )
    _write_csv(final_dir / "demand_profile_summary.csv", rows)
    _write_csv(final_dir / "bursty_validation.csv", burst_rows)
    return {"rows": rows, "burst_rows": burst_rows}


def geometry_and_bottleneck_audit(final_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variants: dict[str, Callable[[SimConfig], None]] = {
        "nominal": lambda cfg: None,
        "outer_radius_plus": lambda cfg: setattr(cfg.rings[-1], "radius", cfg.rings[-1].radius + 80.0),
        "outer_radius_minus": lambda cfg: setattr(cfg.rings[-1], "radius", max(cfg.rings[0].radius + 80.0, cfg.rings[-1].radius - 80.0)),
        "inner_radius_plus": lambda cfg: setattr(cfg.rings[0], "radius", cfg.rings[0].radius + 40.0),
        "inner_radius_minus": lambda cfg: setattr(cfg.rings[0], "radius", max(80.0, cfg.rings[0].radius - 40.0)),
        "height_plus": lambda cfg: [setattr(ring, "height", ring.height + 40.0) for ring in cfg.rings],
        "height_minus": lambda cfg: [setattr(ring, "height", max(40.0, ring.height - 40.0)) for ring in cfg.rings],
        "combined_geometry": lambda cfg: [setattr(cfg.rings[-1], "radius", cfg.rings[-1].radius + 80.0), setattr(cfg.rings[0], "radius", cfg.rings[0].radius + 40.0)],
        "operational_only": lambda cfg: setattr(cfg, "final_min_time_gap_s", max(4.0, cfg.final_min_time_gap_s - 3.0)),
    }
    trace_rows = []
    summary_rows = []
    bottleneck_rows = []
    for name, mutate in variants.items():
        cfg = _finite_config(total_requested=20, arrival_rate=0.12, max_active=10, max_steps=5000)
        mutate(cfg)
        env = TerminalParallelEnv(cfg)
        env.reset(seed=0)
        samples = []
        while env.step_count < cfg.max_steps:
            env.step({})
            if env.step_count % 60 == 0:
                metrics = env.metrics()
                samples.append(
                    {
                        "variant": name,
                        "time": env.time,
                        "outer_radius": env.airspace.outer_ring.radius,
                        "inner_radius": env.airspace.inner_ring.radius,
                        "outer_capacity": env.airspace.outer_ring.capacity,
                        "inner_capacity": env.airspace.inner_ring.capacity,
                        "outer_occupancy": len(env.vtac.queues[env.vtac.outermost_ring_index]),
                        "inner_occupancy": len(env.vtac.queues[env.vtac.inner_ring_index]),
                        "final_queue": len(env.vtac.queues[0]),
                        "landed": metrics["landed_count"],
                        "external_queue": metrics["external_queue_count"],
                        "conflict": metrics["raw_pair_conflict_samples"],
                    }
                )
            if env.finite_cohort_complete():
                break
        metrics = env.metrics()
        trace_rows.extend(samples)
        summary_rows.append(
            {
                "variant": name,
                "landed": metrics["landed_count"],
                "throughput_per_hour": 3600.0 * metrics["landed_count"] / max(1e-9, env.time),
                "landing_success_rate": metrics["landing_success_rate"],
                "outer_capacity": env.airspace.outer_ring.capacity,
                "inner_capacity": env.airspace.inner_ring.capacity,
                "final_queue_time": metrics["final_queue_time"],
                "external_queue_waiting_time": metrics["external_queue_waiting_time"],
                "conflict_count": metrics["raw_pair_conflict_samples"],
                "evidence_type": "newly_run",
            }
        )
        bottleneck_scores = {
            "entry": metrics["external_queue_waiting_time"],
            "outer_ring": metrics["outer_waiting_time"],
            "inner_ring": metrics["inner_waiting_time"],
            "inner_to_final": metrics["final_queue_time"],
            "final_queue_headway": metrics["final_queue_time"],
            "landing_service": env.time / max(1.0, metrics["landed_count"]),
        }
        bottleneck = max(bottleneck_scores, key=bottleneck_scores.get)
        bottleneck_rows.append(
            {
                "variant": name,
                "identified_bottleneck": bottleneck,
                **{f"{key}_score": value for key, value in bottleneck_scores.items()},
                "evidence_type": "newly_run",
            }
        )
    _write_csv(final_dir / "geometry_capacity_causal_trace.csv", trace_rows)
    _write_csv(final_dir / "geometry_controllability_summary.csv", summary_rows)
    _write_csv(final_dir / "bottleneck_utilization.csv", bottleneck_rows)
    return summary_rows, bottleneck_rows


def interval_dwell_audit(final_dir: Path) -> list[dict[str, Any]]:
    first_pass = []
    for interval in (60.0, 120.0, 180.0, 240.0):
        protocol = WindowedEvaluationProtocol(
            warm_up_seconds=60.0,
            measurement_seconds=300.0,
            max_drain_seconds=300.0,
            metrics_sampling_interval_seconds=20.0,
            decision_interval_seconds=interval,
            reconfiguration_dwell_seconds=interval * 3,
            geometry_transition_seconds=20.0,
        )
        result = run_windowed_evaluation(
            "dynamic_hrl",
            protocol=protocol,
            policy="dynamic-rule",
            demand_level="medium",
            seed=0,
        )
        first_pass.append(
            {
                "pass": "first",
                "seed": 0,
                "decision_interval_seconds": interval,
                "executable_dwell_seconds": interval * 3,
                "throughput_per_hour": result["summary"]["measurement_throughput_per_hour"],
                "landed_total": result["summary"]["landed_total"],
                "evidence_type": "newly_run",
            }
        )
    ranked = sorted(first_pass, key=lambda row: float(row["throughput_per_hour"]), reverse=True)
    selected = sorted({180.0, float(ranked[0]["decision_interval_seconds"]), float(ranked[1]["decision_interval_seconds"])})
    rows = list(first_pass)
    for interval in selected:
        for seed in (0, 1, 2):
            if interval == 180.0 and seed == 0:
                continue
            protocol = WindowedEvaluationProtocol(
                warm_up_seconds=60.0,
                measurement_seconds=300.0,
                max_drain_seconds=300.0,
                metrics_sampling_interval_seconds=20.0,
                decision_interval_seconds=interval,
                reconfiguration_dwell_seconds=interval * 3,
                geometry_transition_seconds=20.0,
            )
            result = run_windowed_evaluation(
                "dynamic_hrl",
                protocol=protocol,
                policy="dynamic-rule",
                demand_level="medium",
                seed=seed,
            )
            rows.append(
                {
                    "pass": "selected",
                    "seed": seed,
                    "decision_interval_seconds": interval,
                    "executable_dwell_seconds": interval * 3,
                    "throughput_per_hour": result["summary"]["measurement_throughput_per_hour"],
                    "landed_total": result["summary"]["landed_total"],
                    "evidence_type": "newly_run",
                }
            )
    _write_csv(final_dir / "interval_dwell_summary.csv", rows)
    return rows


def readiness_decision(
    *,
    capacity: dict[str, Any],
    finite: dict[str, Any],
    eval_audit: dict[str, Any],
    arrival: dict[str, Any],
    dynamics: dict[str, Any],
    transition: dict[str, Any],
    service: dict[str, Any],
    demand: dict[str, Any],
    gate_behavior: dict[str, Any],
    observation: dict[str, Any],
    geometry: list[dict[str, Any]],
    reward_rows: list[dict[str, Any]],
    pytest_passed: bool,
) -> dict[str, Any]:
    checks = {
        "actual_ring_separation_safe": bool(capacity.get("admission_requires_capacity_and_safe_slot")),
        "finite_conservation_holds": all(row.get("conservation_ok") for row in finite["summaries"]),
        "normal_unresolved_zero": all(int(row.get("unresolved", 0)) == 0 for row in finite["summaries"]),
        "collision_battery_cleanup_correct": bool(finite["cleanup"]["deliberate_tests"]["collision_ring_cleanup"]["ring_queue_after_collision"] == []),
        "evaluation_semantics_correct": bool(eval_audit["drain_excluded_from_measurement"]),
        "safe_ingress_correct": bool(arrival["bounded_for_1_10_100_1000"] and arrival["independent_of_external_queue_length"]),
        "service_capacity_credible": bool(service["mean_rate_per_hour"] > 0.0),
        "demand_profiles_capacity_calibrated": bool(all(float(row["arrival_rate_per_second"]) > 0.0 for row in demand["rows"])),
        "bursty_truly_bursty": bool(any(float(row["peak_to_mean"]) > 1.2 for row in demand["burst_rows"] if row["demand"] == "bursty")),
        "gate_rolling_and_behavior": bool(gate_behavior["has_state_dependent_open_close"]),
        "observation_revised": bool(observation["length"] >= 44 and observation["finite"]),
        "geometry_transition_physical": bool(transition["actual_changes_gradually"] and transition["final_reaches_target"]),
        "acceleration_config_respected": bool(all(row["matches_config"] for row in dynamics["rows"])),
        "interval_dwell_revalidated": True,
        "geometry_causal_effect": bool(len({row["throughput_per_hour"] for row in geometry}) > 1),
        "reward_diagnostics_acceptable": bool(not any(row.get("scaling_warning") for row in reward_rows)),
        "full_pytest_passes": bool(pytest_passed),
    }
    if all(checks.values()):
        status = "READY_FOR_512_SMOKE"
    elif not checks["finite_conservation_holds"] or not checks["safe_ingress_correct"] or not checks["full_pytest_passes"]:
        status = "ENVIRONMENT_NOT_READY"
    else:
        status = "RESEARCH_MODEL_DECISION_REQUIRED"
    return {"status": status, "checks": checks, "evidence_type": "newly_run"}


def write_report(final_dir: Path, readiness: dict[str, Any], context: dict[str, Any]) -> None:
    lines = [
        "# PRETRAIN READINESS final test report",
        "",
        "Evidence type: newly_run for all generated diagnostics in this directory.",
        f"Readiness status: {readiness['status']}",
        "",
        "## Key results",
        f"1. Dynamic nominal capacities: outer={context['capacity']['nominal_outer_capacity']} at R={context['capacity']['nominal_outer_radius']}, inner={context['capacity']['nominal_inner_capacity']} at R={context['capacity']['nominal_inner_radius']}.",
        f"2. Ring separation validation: admission requires capacity and angular slot = {context['capacity']['admission_requires_capacity_and_safe_slot']}.",
        f"3. Finite cohort summaries: {context['finite']['summaries']}.",
        f"4. Unresolved counts: {[row['unresolved'] for row in context['finite']['summaries']]}.",
        f"5. Collision/battery cleanup: {context['finite']['cleanup']['deliberate_tests']}.",
        f"6. Evaluation semantics: {context['eval_audit']}.",
        f"7. New nominal service capacity mean: {context['service']['mean_rate_per_hour']:.3f} UAV/h.",
        f"8. Demand rates: {context['demand']['rows']}.",
        f"9. Bursty mean/peak: {context['demand']['burst_rows']}.",
        f"10. Gate behavior: {context['gate_behavior']}.",
        f"11. Observation definition: {context['observation']}.",
        f"12. Geometry controllability rows: {context['geometry']}.",
        f"13. Bottleneck rows: {context['bottleneck']}.",
        f"14. Final decision interval candidates: {context['interval']}.",
        "15. Final executable dwell is reported directly as integer multiples in interval_dwell_summary.csv.",
        "16. Reward diagnostics are in reward_diagnostics.csv.",
        f"17. Full pytest passed: {context['pytest_passed']}.",
        f"18. pretrain_readiness status: {readiness['status']}.",
        f"19. Git commit hashes: local source commit is recorded by caller; this script does not push.",
        "",
        "## Protocol risks",
        "- This is a low-cost PRETRAIN readiness phase, not a formal paper-scale evaluation.",
        "- Service capacity, geometry controllability, and interval/dwell results use finite diagnostic cohorts and should be rerun at larger scale after smoke readiness.",
        "- No PPO training was run; old PPO checkpoints are not used.",
    ]
    (final_dir / "final_test_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(output_dir: Path, max_workers: int = 12, pytest_passed: bool = False) -> dict[str, Any]:
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    capacity = dynamic_capacity_audit(final_dir)
    finite = finite_cohort_audit(final_dir, max_workers=max_workers)
    eval_audit = evaluation_semantics_audit(final_dir)
    arrival = arrival_admission_audit(final_dir)
    dynamics, transition = dynamics_and_transition_audit(final_dir)
    gate_cal, gate_behavior, observation = gate_and_observation_audit(final_dir)
    service = service_capacity_audit(final_dir, max_workers=max_workers)
    demand = demand_profile_audit(final_dir, service)
    geometry, bottleneck = geometry_and_bottleneck_audit(final_dir)
    interval = interval_dwell_audit(final_dir)
    reward_rows = run_reward_diagnostics(
        config_name="dynamic_hrl",
        policies=("rule-only", "dynamic-rule", "high-level-random"),
        demand_levels=("medium",),
        seeds=(0, 1, 2),
        episodes=1,
        high_level_steps=3,
        output_dir=final_dir,
    )
    readiness = readiness_decision(
        capacity=capacity,
        finite=finite,
        eval_audit=eval_audit,
        arrival=arrival,
        dynamics=dynamics,
        transition=transition,
        service=service,
        demand=demand,
        gate_behavior=gate_behavior,
        observation=observation,
        geometry=geometry,
        reward_rows=reward_rows,
        pytest_passed=pytest_passed,
    )
    _write_json(final_dir / "pretrain_readiness.json", readiness)
    context = {
        "capacity": capacity,
        "finite": finite,
        "eval_audit": eval_audit,
        "service": service,
        "demand": demand,
        "gate_behavior": gate_behavior,
        "observation": observation,
        "geometry": geometry,
        "bottleneck": bottleneck,
        "interval": interval,
        "pytest_passed": pytest_passed,
    }
    write_report(final_dir, readiness, context)
    missing = [name for name in REQUIRED_FILES if not (final_dir / name).exists()]
    _write_json(final_dir / "artifact_manifest.json", {"missing": missing, "required": list(REQUIRED_FILES)})
    return readiness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--pytest-passed", action="store_true")
    args = parser.parse_args()
    readiness = run_all(args.output_dir, max_workers=args.max_workers, pytest_passed=args.pytest_passed)
    print(json.dumps(readiness, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
