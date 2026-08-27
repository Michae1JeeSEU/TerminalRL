"""C0 nominal-capacity calibration for the A5-v2 low-level curriculum.

This module deliberately runs the rule policy under the FULL teacher safety
stack.  It never constructs a PPO optimizer: its only responsibility is to
freeze an empirically documented ``mu_hat`` before traffic labels such as
"medium" or "high" are used by a learning stage.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ..core.config import RingRole, SimConfig
from ..envs.parallel_env import TerminalParallelEnv
from ..policies import RuleLowLevelPolicy


DEFAULT_CAPACITY_RATES = (0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.075, 0.090)
REQUIRED_SAFETY = ("collisions", "nmac", "hard_separation_violations", "final_overcapacity_steps")


@dataclass(frozen=True)
class CapacityCalibrationProtocol:
    rates: tuple[float, ...] = DEFAULT_CAPACITY_RATES
    seeds: tuple[int, ...] = (101, 102, 103)
    horizon_seconds: float = 3600.0
    warmup_seconds: float = 600.0
    completion_ratio_threshold: float = 0.95
    max_abs_queue_slope_per_second: float = 0.001

    def __post_init__(self) -> None:
        if not self.rates or any(rate <= 0.0 for rate in self.rates):
            raise ValueError("C0 requires at least one positive arrival rate")
        if not self.seeds:
            raise ValueError("C0 requires fixed non-empty seeds")
        if not 0.0 <= self.warmup_seconds < self.horizon_seconds:
            raise ValueError("C0 warm-up must be non-negative and shorter than the horizon")
        if not 0.0 < self.completion_ratio_threshold <= 1.0:
            raise ValueError("completion ratio threshold must be in (0, 1]")
        if self.max_abs_queue_slope_per_second < 0.0:
            raise ValueError("queue slope tolerance must be non-negative")


@dataclass(frozen=True)
class CapacityTrial:
    rate: float
    seed: int
    requested_measurement: int
    admitted_measurement: int
    completed_measurement: int
    completion_to_request_ratio: float
    requested_rate_actual: float
    completed_rate: float
    external_queue_slope_per_second: float
    external_queue_mean: float
    external_queue_max: int
    active_mean: float
    active_peak: int
    ring_occupancy_mean: tuple[float, float, float]
    lz_utilization_mean: float
    final_utilization_mean: float
    delay_mean: float | None
    delay_p95: float | None
    release_intents: int
    teacher_interventions: int
    shield_interventions: int
    teacher_interventions_per_100_intents: float
    shield_interventions_per_100_intents: float
    collisions: int
    nmac: int
    hard_separation_violations: int
    final_overcapacity_steps: int


def c0_config(rate: float, horizon_seconds: float) -> SimConfig:
    """Return the only allowed C0 controller/environment configuration."""
    return SimConfig(
        experiment_mode="fixed_teacher_heavy", high_level_enabled=False,
        reset_mode="flow_from_empty", arrival_model="poisson",
        arrival_rate_per_second=rate, total_requested_uavs=None,
        max_simulation_seconds=horizon_seconds, scheduler_mode="teacher_full",
        teacher_enabled=True, teacher_mode="teacher_full", shield_enabled=True,
        initial_soc_mode="fixed", initial_soc=0.85,
        gate_probabilities=(0.25, 0.25, 0.25, 0.25),
        mission_probabilities=(1.0, 0.0, 0.0, 0.0),
    )


def _optional_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _optional_p95(values: list[float]) -> float | None:
    return float(np.percentile(values, 95)) if values else None


def run_capacity_trial(rate: float, seed: int, protocol: CapacityCalibrationProtocol) -> CapacityTrial:
    """Run one C0 common-random-number rollout and retain measurement-only metrics."""
    config = c0_config(rate, protocol.horizon_seconds)
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(seed)
    policy = RuleLowLevelPolicy()
    queue_times: list[float] = []
    queues: list[float] = []
    active_counts: list[float] = []
    occupancies: list[tuple[float, float, float]] = []
    lz_utilizations: list[float] = []
    final_utilizations: list[float] = []
    admitted_measurement = 0
    release_intents = teacher_interventions = shield_interventions = 0
    safety_at_warmup: dict[str, int] | None = None
    completed_before = 0
    while env.now < config.max_simulation_seconds:
        before_admitted = int(env.telemetry["arrivals_admitted"])
        observations, _, _, _, info = env.step(policy.act(observations))
        if env.now <= protocol.warmup_seconds:
            completed_before = len(env.completed_uavs)
            continue
        if safety_at_warmup is None:
            safety_at_warmup = {key: int(env.telemetry.get(key, 0)) for key in REQUIRED_SAFETY}
        admitted_measurement += int(env.telemetry["arrivals_admitted"]) - before_admitted
        queue_times.append(env.now)
        queues.append(float(len(env.admission.queue.items)))
        active_counts.append(float(len(env.active)))
        occupancies.append(tuple(float(env.airspace.occupancy(role)) for role in RingRole))
        lz_utilizations.append(float(sum(zone.status.value != "free" for zone in env.lz_manager.zones) / len(env.lz_manager.zones)))
        final_utilizations.append(float(sum(uav.phase.value == "final" for uav in env.active.values()) / config.final_queue_capacity))
        for trace in info["action_traces"].values():
            if trace["requested_low_action"][0] > config.release_threshold:
                release_intents += 1
            teacher_interventions += int(bool(trace["teacher_used"]))
            shield_interventions += int(bool(trace["shield_used"]))
    measurement_seconds = protocol.horizon_seconds - protocol.warmup_seconds
    requested_measurement = sum(protocol.warmup_seconds < time <= protocol.horizon_seconds for time in env.request_times.values())
    completed_measurement = sum(protocol.warmup_seconds < uav.completion_time <= protocol.horizon_seconds for uav in env.completed_uavs if uav.completion_time is not None)
    completed = env.completed_uavs[completed_before:]
    delays = [float(uav.completion_time - uav.request_time) for uav in completed if uav.completion_time is not None]
    slope = float(np.polyfit(queue_times, queues, 1)[0]) if len(queue_times) >= 2 else 0.0
    occupancy_array = np.asarray(occupancies or [(0.0, 0.0, 0.0)], dtype=float)
    safety_at_warmup = safety_at_warmup or {key: 0 for key in REQUIRED_SAFETY}
    safety = {key: int(env.telemetry.get(key, 0)) - safety_at_warmup[key] for key in REQUIRED_SAFETY}
    return CapacityTrial(
        rate=rate, seed=seed, requested_measurement=requested_measurement,
        admitted_measurement=admitted_measurement, completed_measurement=completed_measurement,
        completion_to_request_ratio=completed_measurement / requested_measurement if requested_measurement else 0.0,
        requested_rate_actual=requested_measurement / measurement_seconds,
        completed_rate=completed_measurement / measurement_seconds,
        external_queue_slope_per_second=slope, external_queue_mean=float(np.mean(queues)) if queues else 0.0,
        external_queue_max=int(max(queues, default=0.0)), active_mean=float(np.mean(active_counts)) if active_counts else 0.0,
        active_peak=int(max(active_counts, default=0.0)), ring_occupancy_mean=tuple(float(value) for value in occupancy_array.mean(axis=0)),
        lz_utilization_mean=float(np.mean(lz_utilizations)) if lz_utilizations else 0.0,
        final_utilization_mean=float(np.mean(final_utilizations)) if final_utilizations else 0.0,
        delay_mean=_optional_mean(delays), delay_p95=_optional_p95(delays), release_intents=release_intents,
        teacher_interventions=teacher_interventions, shield_interventions=shield_interventions,
        teacher_interventions_per_100_intents=100.0 * teacher_interventions / max(1, release_intents),
        shield_interventions_per_100_intents=100.0 * shield_interventions / max(1, release_intents),
        collisions=safety["collisions"], nmac=safety["nmac"], hard_separation_violations=safety["hard_separation_violations"],
        final_overcapacity_steps=safety["final_overcapacity_steps"],
    )


def _trial_task(args: tuple[float, int, CapacityCalibrationProtocol]) -> CapacityTrial:
    return run_capacity_trial(*args)


def freeze_capacity(trials: Iterable[CapacityTrial], protocol: CapacityCalibrationProtocol) -> dict:
    """Freeze μ̂ only when the predeclared safety, ratio and queue gates pass."""
    grouped: dict[float, list[CapacityTrial]] = {}
    for trial in trials:
        grouped.setdefault(trial.rate, []).append(trial)
    rates: list[dict] = []
    for rate in sorted(grouped):
        rows = grouped[rate]
        ratio = float(np.mean([row.completion_to_request_ratio for row in rows]))
        slope = float(np.mean([row.external_queue_slope_per_second for row in rows]))
        safety_passed = all(all(getattr(row, key) == 0 for key in REQUIRED_SAFETY) for row in rows)
        qualified = safety_passed and ratio >= protocol.completion_ratio_threshold and abs(slope) <= protocol.max_abs_queue_slope_per_second
        rates.append({"rate": rate, "seed_count": len(rows), "completion_to_request_ratio_mean": ratio,
                      "queue_slope_per_second_mean": slope, "safety_passed": safety_passed, "qualified": qualified,
                      "completed_rate_mean": float(np.mean([row.completed_rate for row in rows]))})
    qualified = [row["rate"] for row in rates if row["qualified"]]
    mu_hat = max(qualified) if qualified else None
    level = lambda factor: round(factor * mu_hat, 12)
    return {"mu_hat": mu_hat, "frozen": mu_hat is not None, "criteria": {
        "completion_ratio_threshold": protocol.completion_ratio_threshold,
        "max_abs_queue_slope_per_second": protocol.max_abs_queue_slope_per_second,
        "safety_metrics": list(REQUIRED_SAFETY),
    }, "rates": rates,
    "traffic_levels": None if mu_hat is None else {
        "low": [level(0.30), level(0.40)], "medium": [level(0.45), level(0.60)],
        "high": [level(0.70), level(0.85)], "near_capacity": [level(0.85), level(0.95)],
        "overload": [level(1.05), level(1.20)],
    }}


def write_capacity_report(trials: tuple[CapacityTrial, ...], protocol: CapacityCalibrationProtocol, output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frozen = freeze_capacity(trials, protocol)
    report = {"stage": "C0_capacity_calibration", "evidence_type": "newly_run", "protocol": asdict(protocol),
              "controller": {"teacher_mode": "teacher_full", "scheduler_mode": "teacher_full", "shield_enabled": True,
                             "high_level_enabled": False, "geometry": "nominal_fixed"},
              "measurement_definition": "counts are collected after warm-up; completion/request is a same-window rate diagnostic and queue slope is the stability gate",
              "trials": [asdict(trial) for trial in trials], "frozen_capacity": frozen}
    (output / "c0_capacity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "c0_capacity_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(asdict(trials[0])) if trials else ("rate", "seed"))
        writer.writeheader()
        for trial in trials:
            writer.writerow(asdict(trial))
    lines = ["# A5-v2 C0 Capacity Calibration", "", f"Frozen μ̂: **{frozen['mu_hat']}**", "",
             "| λ (UAV/s) | Seeds | Completion/request | Queue slope (UAV/s) | Safety | Qualified |",
             "|---:|---:|---:|---:|---|---|"]
    for row in frozen["rates"]:
        lines.append(f"| {row['rate']:.3f} | {row['seed_count']} | {row['completion_to_request_ratio_mean']:.3f} | {row['queue_slope_per_second_mean']:.6f} | {'PASS' if row['safety_passed'] else 'FAIL'} | {'YES' if row['qualified'] else 'NO'} |")
    (output / "c0_capacity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_capacity_calibration(protocol: CapacityCalibrationProtocol | None = None, *, output_dir: str | Path = "runs/a5v2_c0_capacity", workers: int = 1) -> dict:
    protocol = protocol or CapacityCalibrationProtocol()
    tasks = [(rate, seed, protocol) for rate in protocol.rates for seed in protocol.seeds]
    if workers <= 1:
        trials = tuple(_trial_task(task) for task in tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            trials = tuple(executor.map(_trial_task, tasks))
    return write_capacity_report(trials, protocol, output_dir)


if __name__ == "__main__":
    print(json.dumps(run_capacity_calibration(), indent=2))
