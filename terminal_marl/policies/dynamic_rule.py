"""Explainable dynamic-airspace heuristic baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DynamicRuleHighLevelPolicy:
    """Map traffic pressure and conflict risk to bounded airspace increments."""

    recovery_gain: float = 0.12
    max_abs_action: float = 0.35
    last_info: dict[str, Any] = field(default_factory=dict, init=False)

    def act(self, env) -> np.ndarray:
        action = np.zeros(8, dtype=np.float32)
        rings = sorted(env.config.rings, key=lambda ring: ring.radius, reverse=True)
        outer, inner = rings[0], rings[-1]
        outer_load = len(env.vtac.queues[outer.index]) / max(1, outer.capacity)
        inner_load = len(env.vtac.queues[inner.index]) / max(1, inner.capacity)
        final_load = len(env.vtac.queues[0]) / max(1, env.config.final_queue_capacity)
        active_count = sum(not item.finished for item in env.aircraft)
        active_ratio = active_count / max(1, env.config.max_active_uavs)
        approach_backlog = sum(
            not item.finished and item.sequence_number is None for item in env.aircraft
        )
        metrics = env.metrics()
        normalized_conflict = float(metrics.get("normalized_conflict_rate", 0.0))
        has_pair = bool(metrics.get("has_pair", False))
        queue_pressure = max(outer_load, inner_load, final_load, active_ratio)
        current = env.airspace.state_vector()
        nominal = env._make_airspace().state_vector()
        recovery = np.sign(nominal - current).astype(np.float32) * self.recovery_gain

        conflict_high = has_pair and normalized_conflict > 0.08
        conflict_low = normalized_conflict < 0.03
        backlog_high = approach_backlog > 0 or active_ratio > 0.75
        throughput_pressure = outer_load > 0.7 or inner_load > 0.5 or backlog_high

        action += recovery

        if throughput_pressure and conflict_low:
            # Prefer throughput-friendly speed/headway changes over geometry
            # expansion.  Large moving rings were observed to chase VTAC
            # release references and starve inner->final transitions.
            action[4] += 0.20
            action[5] += 0.15
            action[6] -= 0.20
            if final_load <= 0.0:
                action[7] -= 0.15

        if conflict_high:
            action[4:6] -= 0.12
            action[6] += 0.18
            if final_load > 0.0:
                action[7] += 0.12

        if final_load >= 1.0 and inner_load >= 1.0:
            action[7] += 0.10
        elif approach_backlog > 0 and final_load <= 0.0 and conflict_low:
            action[7] -= 0.15

        # Keep geometry close to nominal unless there is clear spare safety
        # margin.  This preserves VTAC timers and release reference geometry.
        if throughput_pressure and conflict_low and outer_load < 0.7:
            action[0] += 0.08

        action = np.clip(action, -self.max_abs_action, self.max_abs_action)
        self.last_info = {
            "outer_load": float(outer_load),
            "inner_load": float(inner_load),
            "final_load": float(final_load),
            "active_ratio": float(active_ratio),
            "approach_backlog": float(approach_backlog),
            "normalized_conflict_rate": normalized_conflict,
            "has_pair": has_pair,
            "queue_pressure": float(queue_pressure),
            "conflict_high": bool(conflict_high),
            "conflict_low": bool(conflict_low),
            "backlog_high": bool(backlog_high),
            "throughput_pressure": bool(throughput_pressure),
            "nominal_recovery": bool(np.any(recovery != 0.0)),
        }
        return action.astype(np.float32)
