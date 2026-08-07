"""Warm-up / measurement / drain evaluation utilities.

The protocol keeps physical time comparable across high-level decision
intervals.  It is intentionally simulation-only and never starts PPO learning;
an existing PPO checkpoint may only be loaded for deterministic evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config import load_experiment_config, load_sim_config
from ..core.config import SimConfig
from ..core.dynamic_airspace import DynamicAirspace
from ..core.uav import FlightPhase
from ..envs.parallel_env import TerminalParallelEnv
from ..policies import DynamicRuleHighLevelPolicy
from .ablation_runner import DEMAND_PROFILES, apply_demand_profile


@dataclass(frozen=True)
class WindowedEvaluationProtocol:
    """Physical-time evaluation windows for terminal-flow experiments."""

    warm_up_seconds: float = 600.0
    measurement_seconds: float = 1800.0
    max_drain_seconds: float = 1800.0
    metrics_sampling_interval_seconds: float = 20.0
    decision_interval_seconds: float = 20.0
    reconfiguration_dwell_seconds: float = 60.0
    geometry_transition_seconds: float = 20.0

    def __post_init__(self) -> None:
        for name in (
            "warm_up_seconds",
            "measurement_seconds",
            "max_drain_seconds",
            "metrics_sampling_interval_seconds",
            "decision_interval_seconds",
            "reconfiguration_dwell_seconds",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.metrics_sampling_interval_seconds <= 0.0:
            raise ValueError("metrics_sampling_interval_seconds must be positive")
        if self.decision_interval_seconds <= 0.0:
            raise ValueError("decision_interval_seconds must be positive")
        if self.geometry_transition_seconds < 0.0:
            raise ValueError("geometry_transition_seconds must be non-negative")


def resolve_config(config: SimConfig | str | Path) -> SimConfig:
    """Load or copy a simulation configuration."""

    if isinstance(config, SimConfig):
        return copy.deepcopy(config)
    path = Path(config)
    if path.suffix == ".yaml" or path.is_file():
        return load_sim_config(path)
    return load_experiment_config(str(config))


def configure_high_level_timing(
    config: SimConfig,
    protocol: WindowedEvaluationProtocol,
) -> SimConfig:
    """Return a copy whose high-level decision interval is physical-time based."""

    updated = copy.deepcopy(config)
    updated.metrics_sampling_interval_seconds = float(
        protocol.metrics_sampling_interval_seconds
    )
    updated.decision_interval_seconds = float(protocol.decision_interval_seconds)
    updated.reconfiguration_dwell_seconds = float(
        protocol.reconfiguration_dwell_seconds
    )
    updated.geometry_transition_seconds = float(protocol.geometry_transition_seconds)
    updated.high_level_interval_steps = max(
        1,
        int(round(updated.decision_interval_seconds / updated.dt)),
    )
    return updated


def set_runtime_arrivals_enabled(env: TerminalParallelEnv, enabled: bool) -> None:
    """Enable/disable new arrivals without changing already-active aircraft."""

    if enabled:
        env.arrival_process.arrival_rate = float(env.config.arrival_rate)
        if not math.isfinite(env.arrival_process.next_arrival_time):
            env.arrival_process.next_arrival_time = env.arrival_process._next_time(
                env.time
            )
    else:
        env.arrival_process.arrival_rate = 0.0
        env.arrival_process.next_arrival_time = math.inf


def phase_snapshot(env: TerminalParallelEnv) -> dict[int, tuple[FlightPhase, int | None, bool]]:
    return {
        aircraft.uid: (aircraft.phase, aircraft.assigned_ring, aircraft.finished)
        for aircraft in env.aircraft
    }


def count_phase_progression(
    previous: dict[int, tuple[FlightPhase, int | None, bool]],
    env: TerminalParallelEnv,
) -> dict[str, int]:
    """Count coarse terminal-pipeline transitions observed in the last step."""

    counts = {
        "approach_to_outer": 0,
        "outer_to_inner": 0,
        "inner_to_final": 0,
        "final_to_land": 0,
    }
    ring_indices = {ring.index for ring in env.config.rings}
    outer_index = max(ring_indices) if ring_indices else 2
    inner_index = min(ring_indices) if ring_indices else 1
    for aircraft in env.aircraft:
        old_phase, old_ring, old_finished = previous.get(
            aircraft.uid,
            (FlightPhase.APPROACH, None, False),
        )
        if old_finished:
            continue
        if old_phase == FlightPhase.APPROACH and aircraft.assigned_ring == outer_index:
            counts["approach_to_outer"] += 1
        if old_ring == outer_index and aircraft.assigned_ring == inner_index:
            counts["outer_to_inner"] += 1
        if old_ring == inner_index and aircraft.phase == FlightPhase.FINAL:
            counts["inner_to_final"] += 1
        if old_phase == FlightPhase.FINAL and aircraft.finished:
            counts["final_to_land"] += 1
    return counts


def occupancy_snapshot(env: TerminalParallelEnv) -> dict[str, float]:
    active = [aircraft for aircraft in env.aircraft if not aircraft.finished]
    ring_indices = {ring.index for ring in env.config.rings}
    outer_index = max(ring_indices) if ring_indices else 2
    inner_index = min(ring_indices) if ring_indices else 1
    return {
        "active_count": float(len(active)),
        "spawned_count": float(len(env.aircraft)),
        "landed_count": float(sum(aircraft.finished for aircraft in env.aircraft)),
        "approach_backlog": float(
            sum(
                not aircraft.finished and aircraft.sequence_number is None
                for aircraft in env.aircraft
            )
        ),
        "outer_occupancy": float(len(env.vtac.queues[outer_index])),
        "inner_occupancy": float(len(env.vtac.queues[inner_index])),
        "final_occupancy": float(len(env.vtac.queues[0])),
    }


def _extract_release_approvals(env: TerminalParallelEnv) -> int:
    return sum(
        1
        for approval in env.last_release_approval.values()
        if bool(approval.get("approved", False))
    ) + sum(
        1
        for approval in env.last_landing_approval.values()
        if bool(approval.get("approved", False))
    )


def collect_sample(
    env: TerminalParallelEnv,
    *,
    policy: str,
    demand_level: str,
    seed: int,
    window: str,
    decision_interval_seconds: float,
    action: np.ndarray | None = None,
    action_applied: bool = False,
    transition_counts: dict[str, int] | None = None,
    measurement_start: dict[str, float] | None = None,
) -> dict[str, float | str]:
    metrics = env.metrics()
    occupancy = occupancy_snapshot(env)
    transition_counts = transition_counts or {}
    airspace_state = env.airspace.state_vector()
    nominal = env._make_airspace().state_vector()
    action_l1 = float(np.abs(action).sum()) if action is not None else 0.0
    measurement_start = measurement_start or {}
    gate_info = env.geometry_gate.info(
        env.time,
        geometry_deviation_l1=float(np.abs(airspace_state[:4] - nominal[:4]).sum()),
    )
    return {
        "time": float(env.time),
        "step": float(env.step_count),
        "window": window,
        "policy": policy,
        "traffic_demand_level": demand_level,
        "seed": float(seed),
        "decision_interval_seconds": float(decision_interval_seconds),
        "action_applied": float(action_applied),
        "action_l1": action_l1,
        "release_approvals": float(_extract_release_approvals(env)),
        "throughput": (
            float(
                (
                    metrics["landed_count"]
                    - measurement_start.get("landed_count", 0.0)
                )
                / max(1e-9, env.time - measurement_start.get("time", 0.0))
            )
            if window in {"measurement", "drain"} and env.time > measurement_start.get("time", 0.0)
            else 0.0
        ),
        "normalized_conflict_rate": float(metrics["normalized_conflict_rate"]),
        "normalized_waiting": float(metrics["normalized_waiting"]),
        "normalized_congestion": float(metrics["normalized_congestion"]),
        "nmac_count": float(metrics["nmac_count"]),
        "collision_count": float(metrics["collision_count"]),
        "safety_override_count": float(metrics["safety_override_count"]),
        "raw_safety_override_count": float(metrics["raw_safety_override_count"]),
        "min_pairwise_distance": float(metrics["min_pairwise_distance"]),
        "airspace_outer_radius": float(airspace_state[0]),
        "airspace_inner_radius": float(airspace_state[1]),
        "airspace_outer_height": float(airspace_state[2]),
        "airspace_inner_height": float(airspace_state[3]),
        "airspace_outer_speed": float(airspace_state[4]),
        "airspace_inner_speed": float(airspace_state[5]),
        "airspace_ring_headway": float(airspace_state[6]),
        "airspace_final_headway": float(airspace_state[7]),
        "airspace_deviation_l1": float(np.abs(airspace_state - nominal).sum()),
        "trigger_score": float(gate_info["trigger_score"]),
        "geometry_gate_active": float(gate_info["geometry_gate_active"]),
        "geometry_action_allowed": float(gate_info["geometry_action_allowed"]),
        "geometry_dwell_satisfied": float(gate_info["geometry_dwell_satisfied"]),
        "geometry_gate_state": gate_info["gate_state"],
        "geometry_gate_reason": gate_info["geometry_gate_reason"],
        "geometry_opportunity_count": float(gate_info["geometry_opportunity_count"]),
        "applied_geometry_change_count": float(gate_info["applied_geometry_change_count"]),
        "nominal_recovery_count": float(gate_info["nominal_recovery_count"]),
        "approach_to_outer": float(transition_counts.get("approach_to_outer", 0)),
        "outer_to_inner": float(transition_counts.get("outer_to_inner", 0)),
        "inner_to_final": float(transition_counts.get("inner_to_final", 0)),
        "final_to_land": float(transition_counts.get("final_to_land", 0)),
        **occupancy,
    }


def _zero_low_level_actions(env: TerminalParallelEnv) -> dict[str, np.ndarray]:
    if not env.agents and not env.possible_agents:
        return {}
    size = env.action_spec(env.possible_agents[0] if env.possible_agents else env.agents[0]).shape[0]
    return {agent: np.zeros(size, dtype=np.float32) for agent in env.agents}


def select_high_level_action(
    env: TerminalParallelEnv,
    *,
    policy: str,
    rng: np.random.Generator,
    dynamic_rule_policy: DynamicRuleHighLevelPolicy,
    ppo_model: Any | None = None,
) -> np.ndarray:
    if policy == "high-level-random":
        return rng.uniform(-1.0, 1.0, size=DynamicAirspace.ACTION_SIZE).astype(np.float32)
    if policy in {"dynamic-rule", "event-rule"}:
        return dynamic_rule_policy.act(env)
    if policy == "high-level-ppo":
        if ppo_model is None:
            raise ValueError("high-level-ppo requires ppo_model")
        action, _ = ppo_model.predict(env.get_high_level_observation(), deterministic=True)
        return np.asarray(action, dtype=np.float32)
    return np.zeros(DynamicAirspace.ACTION_SIZE, dtype=np.float32)


def load_existing_ppo_model(model_path: str | Path | None):
    if model_path is None:
        return None
    from ..training.train_high_level_ppo import require_sb3

    PPO = require_sb3()[0]
    return PPO.load(model_path)


def run_windowed_evaluation(
    config: SimConfig | str | Path,
    *,
    protocol: WindowedEvaluationProtocol,
    policy: str = "dynamic-rule",
    demand_level: str = "medium",
    seed: int = 0,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one warm-up/measurement/drain rollout at a fixed physical duration."""

    sim_config = configure_high_level_timing(
        apply_demand_profile(resolve_config(config), demand_level),
        protocol,
    )
    total_seconds = (
        protocol.warm_up_seconds
        + protocol.measurement_seconds
        + protocol.max_drain_seconds
    )
    sim_config.max_steps = max(sim_config.max_steps, int(math.ceil(total_seconds / sim_config.dt)) + 2)
    env = TerminalParallelEnv(sim_config)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    dynamic_rule_policy = DynamicRuleHighLevelPolicy()
    ppo_model = load_existing_ppo_model(model_path) if policy == "high-level-ppo" else None
    samples: list[dict[str, float | str]] = []
    transitions = {
        "approach_to_outer": 0,
        "outer_to_inner": 0,
        "inner_to_final": 0,
        "final_to_land": 0,
    }
    measurement_start: dict[str, float] = {}
    decision_steps = max(1, int(round(protocol.decision_interval_seconds / sim_config.dt)))
    sampling_steps = max(1, int(round(protocol.metrics_sampling_interval_seconds / sim_config.dt)))
    warmup_end = float(protocol.warm_up_seconds)
    measurement_end = warmup_end + float(protocol.measurement_seconds)
    drain_end = measurement_end + float(protocol.max_drain_seconds)
    next_decision_step = 0
    next_sample_step = 0
    last_action = np.zeros(DynamicAirspace.ACTION_SIZE, dtype=np.float32)
    action_apply_count = 0
    action_sign_changes = 0
    previous_action: np.ndarray | None = None
    action_l1_sum = 0.0
    action_abs_values: list[float] = []
    boundary_samples = 0
    total_samples = 0

    while env.step_count < sim_config.max_steps and env.time < drain_end:
        if env.time >= measurement_end:
            set_runtime_arrivals_enabled(env, False)
            window = "drain"
        elif env.time >= warmup_end:
            if not measurement_start:
                metrics = env.metrics()
                measurement_start = {
                    "time": float(env.time),
                    "landed_count": float(metrics["landed_count"]),
                    "spawned_count": float(metrics["total_spawned_count"]),
                    "waiting": float(metrics["total_waiting_time"]),
                    "conflict": float(metrics["raw_pair_conflict_samples"]),
                }
            window = "measurement"
        else:
            window = "warm_up"

        high_level_action = None
        action_applied = False
        if env.step_count >= next_decision_step:
            high_level_action = select_high_level_action(
                env,
                policy=policy,
                rng=rng,
                dynamic_rule_policy=dynamic_rule_policy,
                ppo_model=ppo_model,
            )
            last_action = high_level_action
            if previous_action is not None:
                changed_signs = np.sign(previous_action) != np.sign(high_level_action)
                action_sign_changes += int(np.count_nonzero(changed_signs & (np.abs(high_level_action) > 1e-6)))
            previous_action = high_level_action.copy()
            action_l1 = float(np.abs(high_level_action).sum())
            action_l1_sum += action_l1
            action_abs_values.extend(float(abs(x)) for x in high_level_action)
            action_apply_count += 1
            action_applied = True
            next_decision_step += decision_steps

        previous = phase_snapshot(env)
        env.step(
            _zero_low_level_actions(env),
            high_level_action=high_level_action,
        )
        step_counts = count_phase_progression(previous, env)
        for key, value in step_counts.items():
            transitions[key] += value

        state = env.airspace.normalized_state_vector()
        boundary_samples += int(np.any(np.abs(state) >= 0.98))
        total_samples += 1

        if env.step_count >= next_sample_step:
            samples.append(
                collect_sample(
                    env,
                    policy=policy,
                    demand_level=demand_level,
                    seed=seed,
                    window=window,
                    decision_interval_seconds=protocol.decision_interval_seconds,
                    action=last_action,
                    action_applied=action_applied,
                    transition_counts=transitions,
                    measurement_start=measurement_start,
                )
            )
            next_sample_step += sampling_steps
        if env.time >= measurement_end and not any(not aircraft.finished for aircraft in env.aircraft):
            break

    metrics = env.metrics()
    per_aircraft = env.per_aircraft_metrics()
    landed = [row for row in per_aircraft.values() if row["completion_time"] is not None]
    measurement_spawned = max(
        0.0,
        float(metrics["total_spawned_count"]) - measurement_start.get("spawned_count", 0.0),
    )
    landed_after_measurement_start = max(
        0.0,
        float(metrics["landed_count"]) - measurement_start.get("landed_count", 0.0),
    )
    delays = [
        float(row["landing_delay"])
        for row in landed
        if row.get("landing_delay") is not None
    ]
    active_counts = [float(row["active_count"]) for row in samples if row["window"] == "measurement"]
    backlogs = [float(row["approach_backlog"]) for row in samples if row["window"] == "measurement"]
    landed_measurement_end = max(
        [
            float(row["landed_count"])
            for row in samples
            if row["window"] == "measurement"
        ],
        default=float(metrics["landed_count"]),
    )
    summary: dict[str, float | str] = {
        "config_name": sim_config.config_name,
        "experiment_mode": env.experiment_mode.value,
        "policy": policy,
        "traffic_demand_level": demand_level,
        "seed": float(seed),
        "warm_up_seconds": float(protocol.warm_up_seconds),
        "measurement_seconds": float(protocol.measurement_seconds),
        "max_drain_seconds": float(protocol.max_drain_seconds),
        "metrics_sampling_interval_seconds": float(protocol.metrics_sampling_interval_seconds),
        "decision_interval_seconds": float(protocol.decision_interval_seconds),
        "reconfiguration_dwell_seconds": float(protocol.reconfiguration_dwell_seconds),
        "geometry_transition_seconds": float(protocol.geometry_transition_seconds),
        "elapsed_seconds": float(env.time),
        "measurement_start_time": float(measurement_start.get("time", 0.0)),
        "spawned_total": float(metrics["total_spawned_count"]),
        "landed_total": float(metrics["landed_count"]),
        "landed_by_measurement_end": float(landed_measurement_end),
        "total_landed_after_drain": float(metrics["landed_count"]),
        "measurement_spawned": float(measurement_spawned),
        "landed_after_measurement_start": float(landed_after_measurement_start),
        "final_completion_ratio": (
            float(metrics["landed_count"]) / max(1.0, float(metrics["total_spawned_count"]))
        ),
        "landed_by_horizon_ratio": (
            float(landed_measurement_end) / max(1.0, float(metrics["total_spawned_count"]))
        ),
        "throughput": landed_after_measurement_start / max(1e-9, protocol.measurement_seconds),
        "mean_delay": float(np.mean(delays)) if delays else 0.0,
        "p95_delay": float(np.percentile(delays, 95)) if delays else 0.0,
        "mean_waiting": float(metrics["normalized_waiting"]),
        "p95_waiting_proxy": float(np.percentile(backlogs, 95)) if backlogs else 0.0,
        "mean_backlog": float(np.mean(backlogs)) if backlogs else 0.0,
        "max_backlog": float(max(backlogs)) if backlogs else 0.0,
        "active_slope": (
            (active_counts[-1] - active_counts[0]) / max(1.0, protocol.measurement_seconds)
            if len(active_counts) >= 2
            else 0.0
        ),
        "normalized_conflict_rate": float(metrics["normalized_conflict_rate"]),
        "nmac_count": float(metrics["nmac_count"]),
        "collision_count": float(metrics["collision_count"]),
        "safety_override_count": float(metrics["safety_override_count"]),
        "raw_safety_override_count": float(metrics["raw_safety_override_count"]),
        "reconfig_count": float(action_apply_count),
        "mean_action_l1": action_l1_sum / max(1, action_apply_count),
        "mean_action_abs": float(np.mean(action_abs_values)) if action_abs_values else 0.0,
        "action_nonzero_ratio": (
            float(np.count_nonzero(np.asarray(action_abs_values) > 1e-6))
            / max(1, len(action_abs_values))
        ),
        "geometry_oscillation_count": float(action_sign_changes),
        "mean_reversal_interval_seconds": (
            float(env.time) / action_sign_changes if action_sign_changes > 0 else 0.0
        ),
        "boundary_time_fraction": boundary_samples / max(1, total_samples),
        "drain_time": max(0.0, float(env.time) - measurement_end),
        **{key: float(value) for key, value in transitions.items()},
    }
    return {
        "summary": summary,
        "samples": samples,
        "per_aircraft": per_aircraft,
        "env": env,
    }


def summarize_rows(rows: Iterable[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in group_keys), []).append(row)
    summaries: list[dict[str, Any]] = []
    for key_values, group in sorted(groups.items(), key=lambda item: str(item[0])):
        out = {key: value for key, value in zip(group_keys, key_values)}
        numeric_keys = sorted(
            {
                key
                for row in group
                for key, value in row.items()
                if isinstance(value, (int, float)) and key not in group_keys
            }
        )
        for metric in numeric_keys:
            values = np.asarray([float(row[metric]) for row in group if metric in row], dtype=float)
            if values.size:
                out[f"{metric}_mean"] = float(np.mean(values))
                out[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                out[f"{metric}_min"] = float(np.min(values))
                out[f"{metric}_max"] = float(np.max(values))
        summaries.append(out)
    return summaries
