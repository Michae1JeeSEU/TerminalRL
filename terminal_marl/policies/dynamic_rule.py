"""Explainable dynamic-airspace heuristic baseline."""

from __future__ import annotations

import numpy as np


class DynamicRuleHighLevelPolicy:
    """Map traffic pressure and conflict risk to bounded airspace increments."""

    def act(self, env) -> np.ndarray:
        action = np.zeros(8, dtype=np.float32)
        rings = sorted(env.config.rings, key=lambda ring: ring.radius, reverse=True)
        outer, inner = rings[0], rings[-1]
        outer_load = len(env.vtac.queues[outer.index]) / max(1, outer.capacity)
        inner_load = len(env.vtac.queues[inner.index]) / max(1, inner.capacity)
        final_load = len(env.vtac.queues[0]) / max(1, env.config.final_queue_capacity)
        active_ratio = sum(not item.finished for item in env.aircraft) / max(
            1, env.config.max_active_uavs
        )
        conflict_risk = bool(
            env.safety_filter.check_pairwise_distance(
                [item for item in env.aircraft if not item.finished]
            )
        )
        if outer_load > 0.7 or active_ratio > 0.7:
            action[0] += 0.4
            action[4] += 0.35
        if inner_load > 0.7 or final_load > 0.5:
            action[7] += 0.5
        if conflict_risk:
            action[4:6] -= 0.5
            action[6:8] += 0.5
        if outer_load < 0.2 and inner_load < 0.2 and final_load == 0.0:
            current = env.airspace.state_vector()
            initial = env._make_airspace().state_vector()
            action += 0.2 * np.sign(initial - current).astype(np.float32)
        return np.clip(action, -1.0, 1.0)
