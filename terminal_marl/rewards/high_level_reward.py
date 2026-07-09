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
    waiting: float = 0.02
    conflict: float = 5.0
    safety_override: float = 1.0
    airspace_change: float = 0.25
    ring_imbalance: float = 0.5
    congestion: float = 1.0
    smoothness: float = 0.25


def high_level_snapshot(env: "TerminalParallelEnv") -> dict[str, float]:
    """Capture cumulative/system state at one high-level boundary."""

    occupancy = [
        len(env.vtac.queues[ring.index]) / max(1, ring.capacity)
        for ring in sorted(env.config.rings, key=lambda item: item.index)
    ]
    return {
        "time": float(env.time),
        "landed_count": float(sum(aircraft.finished for aircraft in env.aircraft)),
        "final_queue_time": float(env.final_queue_time),
        "conflict_count": float(env.nmac_count + env.collision_count),
        "safety_override_count": float(env.safety_override_count),
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
        waiting_delta = max(
            0.0,
            float(after["final_queue_time"] - before["final_queue_time"]),
        )
        conflict_delta = max(
            0.0,
            float(after["conflict_count"] - before["conflict_count"]),
        )
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
        congestion = float(np.maximum(occupancy - 1.0, 0.0).sum())
        change = np.asarray(new_airspace, dtype=np.float64) - np.asarray(
            old_airspace,
            dtype=np.float64,
        )
        scale = np.maximum(np.abs(np.asarray(old_airspace, dtype=np.float64)), 1.0)
        normalized_change = change / scale
        airspace_change = float(np.linalg.norm(normalized_change))
        smoothness = float(
            np.linalg.norm(normalized_change - self.previous_airspace_change)
        )
        self.previous_airspace_change = normalized_change

        components = {
            "landed": self.weights.landed * landed_delta,
            "throughput": self.weights.throughput * throughput,
            "waiting": -self.weights.waiting * waiting_delta,
            "conflict": -self.weights.conflict * conflict_delta,
            "safety_override": -self.weights.safety_override * override_delta,
            "airspace_change": -self.weights.airspace_change * airspace_change,
            "ring_imbalance": -self.weights.ring_imbalance * ring_imbalance,
            "congestion": -self.weights.congestion * congestion,
            "smoothness": -self.weights.smoothness * smoothness,
        }
        reward = float(sum(components.values()))
        diagnostics = {
            **{f"component_{key}": float(value) for key, value in components.items()},
            "landed_delta": landed_delta,
            "throughput": throughput,
            "waiting_delta": waiting_delta,
            "conflict_delta": conflict_delta,
            "safety_override_delta": override_delta,
            "airspace_change_norm": airspace_change,
            "ring_imbalance": ring_imbalance,
            "congestion": congestion,
            "smoothness_norm": smoothness,
            "interval_duration": duration,
            "total_reward": reward,
        }
        return reward, diagnostics
