"""Deterministic pressure-gradient 10-D high-level baseline."""
from __future__ import annotations

import numpy as np


class DynamicRulePolicy:
    """Meter each interface from its upstream/downstream resource gradient."""

    upstream_pressure = 0.65
    downstream_slack = 0.65
    downstream_saturated = 0.85

    def act(self, observation: np.ndarray, context: dict) -> np.ndarray:
        del observation
        external = float(context["external_queue"])
        outer, middle, inner = np.asarray(context["occupancy_ratio"], dtype=float)
        final_pressure = float(context["final_pressure"])
        lz_utilization = float(context["lz_utilization"])
        downstream_full = final_pressure >= self.downstream_saturated or lz_utilization >= self.downstream_saturated
        action = np.zeros(10, dtype=np.float32)

        # O2M: feed middle only when its ring has room; otherwise meter outer.
        if (external > 0 or outer >= self.upstream_pressure) and middle < self.downstream_slack and not downstream_full:
            action[7] = -0.50
        elif middle >= self.downstream_saturated or downstream_full:
            action[7] = 0.35

        # M2I: use the same local gradient, independent of O2M.
        if middle >= self.upstream_pressure and inner < self.downstream_slack and not downstream_full:
            action[8] = -0.50
        elif inner >= self.downstream_saturated or downstream_full:
            action[8] = 0.35

        # I2F is constrained by both the final corridor and LZ service.
        if inner >= self.upstream_pressure and final_pressure < self.downstream_slack and lz_utilization < self.downstream_slack:
            action[9] = -0.50
        elif downstream_full:
            action[9] = 0.50

        # Structural expansion is meaningful only for a real upstream queue.
        # DynamicAirspace projects it to the nearest legal capacity threshold
        # or records a structural no-op reason; capacity is never an action.
        if external >= 4 and outer >= .45 and middle < self.downstream_saturated:
            action[0] = .60
        if middle >= .70 and inner < self.downstream_slack and not downstream_full:
            action[1] = .50
        if inner >= .70 and not downstream_full:
            action[2] = .35
        return action
