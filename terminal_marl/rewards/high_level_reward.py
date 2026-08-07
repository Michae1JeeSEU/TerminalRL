"""Interval-level reward for dynamic terminal airspace management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np

if TYPE_CHECKING:
    from ..envs.parallel_env import TerminalParallelEnv


@dataclass(frozen=True)
class HighLevelRewardWeights:
    landed: float = 8.0
    throughput: float = 20.0
    phase_progress: float = 1.5
    waiting: float = 2.0
    conflict: float = 5.0
    safety_override: float = 1.0
    airspace_change: float = 0.25
    geometry_change: float = 0.05
    ring_imbalance: float = 0.5
    congestion: float = 1.0
    smoothness: float = 0.25


def high_level_snapshot(env: "TerminalParallelEnv") -> dict[str, float]:
    """Capture cumulative/system state at one high-level boundary."""

    occupancy = [
        len(env.vtac.queues[ring.index]) / max(1, ring.capacity)
        for ring in sorted(env.config.rings, key=lambda item: item.index)
    ]
    metrics = env.metrics()
    phase_progress_score = 0.0
    for aircraft in env.aircraft:
        if aircraft.finished:
            phase_progress_score += 4.0
        else:
            phase_progress_score += float(int(aircraft.phase))
    return {
        "time": float(env.time),
        "landed_count": float(sum(aircraft.finished for aircraft in env.aircraft)),
        "final_queue_time": float(env.final_queue_time),
        "conflict_count": float(env.nmac_count + env.collision_count),
        "raw_pair_conflict_samples": float(env.raw_pair_conflict_samples),
        "pair_opportunities": float(env.pair_opportunities),
        "normalized_conflict_rate": float(metrics["normalized_conflict_rate"]),
        "safety_override_count": float(env.safety_override_count),
        "total_waiting_time": float(env.total_waiting_time),
        "normalized_waiting": float(metrics["normalized_waiting"]),
        "congestion_pressure_time": float(env.congestion_pressure_time),
        "normalized_congestion": float(metrics["normalized_congestion"]),
        "phase_progress_score": float(phase_progress_score),
        "max_active_uavs": float(env.config.max_active_uavs),
        "ring_occupancy_0": occupancy[0] if occupancy else 0.0,
        "ring_occupancy_1": occupancy[1] if len(occupancy) > 1 else 0.0,
    }


class HighLevelReward:
    """Compute one reward after a complete high-level control interval."""

    def __init__(self, weights: HighLevelRewardWeights | None = None) -> None:
        self.weights = weights or HighLevelRewardWeights()
        self.previous_airspace_change = np.zeros(8, dtype=np.float64)

    def reset(self) -> None:
        self.previous_airspace_change = np.zeros(8, dtype=np.float64)

    def compute(
        self,
        before: Mapping[str, float],
        after: Mapping[str, float],
        old_airspace: np.ndarray,
        new_airspace: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        duration = max(1e-9, float(after["time"] - before["time"]))
        landed_delta = float(after["landed_count"] - before["landed_count"])
        throughput = landed_delta / duration
        phase_progress_delta = max(
            0.0,
            float(after.get("phase_progress_score", 0.0))
            - float(before.get("phase_progress_score", 0.0)),
        )
        waiting_raw_delta = max(
            0.0,
            float(after.get("total_waiting_time", after.get("final_queue_time", 0.0)))
            - float(before.get("total_waiting_time", before.get("final_queue_time", 0.0))),
        )
        max_active = max(
            1.0,
            float(after.get("max_active_uavs", before.get("max_active_uavs", 1.0))),
        )
        waiting_denominator = max(1.0, duration * max_active)
        waiting_delta = waiting_raw_delta / waiting_denominator
        raw_conflict_delta = max(
            0.0,
            float(after.get("raw_pair_conflict_samples", after.get("conflict_count", 0.0)))
            - float(before.get("raw_pair_conflict_samples", before.get("conflict_count", 0.0))),
        )
        pair_opportunity_delta = max(
            0.0,
            float(after.get("pair_opportunities", 0.0))
            - float(before.get("pair_opportunities", 0.0)),
        )
        conflict_delta = raw_conflict_delta / max(1.0, pair_opportunity_delta)
        override_delta = max(
            0.0,
            float(after["safety_override_count"] - before["safety_override_count"]),
        )
        occupancy = np.asarray(
            [
                after.get("ring_occupancy_0", 0.0),
                after.get("ring_occupancy_1", 0.0),
            ],
            dtype=np.float64,
        )
        ring_imbalance = float(np.std(occupancy))
        congestion_raw_delta = max(
            0.0,
            float(after.get("congestion_pressure_time", 0.0))
            - float(before.get("congestion_pressure_time", 0.0)),
        )
        congestion = max(
            float(np.maximum(occupancy - 1.0, 0.0).sum()),
            congestion_raw_delta / duration,
        )
        change = np.asarray(new_airspace, dtype=np.float64) - np.asarray(
            old_airspace,
            dtype=np.float64,
        )
        scale = np.maximum(np.abs(np.asarray(old_airspace, dtype=np.float64)), 1.0)
        normalized_change = change / scale
        airspace_change = float(np.linalg.norm(normalized_change))
        geometry_change = float(np.linalg.norm(normalized_change[:4]))
        operational_change = float(np.linalg.norm(normalized_change[4:]))
        smoothness = float(
            np.linalg.norm(normalized_change - self.previous_airspace_change)
        )
        self.previous_airspace_change = normalized_change

        components = {
            "landed": self.weights.landed * landed_delta,
            "throughput": self.weights.throughput * throughput,
            "phase_progress": self.weights.phase_progress * phase_progress_delta,
            "waiting": -self.weights.waiting * waiting_delta,
            "conflict": -self.weights.conflict * conflict_delta,
            "safety_override": -self.weights.safety_override * override_delta,
            "airspace_change": -self.weights.airspace_change * airspace_change,
            "geometry_change": -self.weights.geometry_change * geometry_change,
            "ring_imbalance": -self.weights.ring_imbalance * ring_imbalance,
            "congestion": -self.weights.congestion * congestion,
            "smoothness": -self.weights.smoothness * smoothness,
        }
        reward = float(sum(components.values()))
        diagnostics = {
            **{f"component_{key}": float(value) for key, value in components.items()},
            "landed_delta": landed_delta,
            "landed_count_cumulative": float(after["landed_count"]),
            "throughput": throughput,
            "phase_progress_delta": phase_progress_delta,
            "waiting_delta": waiting_delta,
            "waiting_raw_delta": waiting_raw_delta,
            "normalized_waiting": waiting_delta,
            "final_queue_time_cumulative": float(after["final_queue_time"]),
            "conflict_delta": conflict_delta,
            "raw_conflict_delta": raw_conflict_delta,
            "pair_opportunity_delta": pair_opportunity_delta,
            "normalized_conflict_rate": conflict_delta,
            "conflict_count_cumulative": float(after["conflict_count"]),
            "safety_override_delta": override_delta,
            "safety_override_count_cumulative": float(after["safety_override_count"]),
            "airspace_change_norm": airspace_change,
            "geometry_change_norm": geometry_change,
            "operational_change_norm": operational_change,
            "ring_imbalance": ring_imbalance,
            "congestion": congestion,
            "smoothness_norm": smoothness,
            "interval_duration": duration,
            "total_reward": reward,
        }
        return reward, diagnostics
