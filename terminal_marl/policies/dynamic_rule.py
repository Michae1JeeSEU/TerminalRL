"""Deterministic canonical 10-D dynamic-airspace baseline."""
from __future__ import annotations

import numpy as np


class DynamicRulePolicy:
    """Pressure-responsive geometry/headway intent, not a hidden low policy."""

    def act(self, observation: np.ndarray, context: dict) -> np.ndarray:
        external = float(context["external_queue"])
        ratios = np.asarray(context["occupancy_ratio"], dtype=float)
        final_pressure = float(context["final_pressure"])
        lz_utilization = float(context["lz_utilization"])
        pressure = max(external / 4.0, float(ratios.max(initial=0.0)), final_pressure, lz_utilization)
        action = np.zeros(10, dtype=np.float32)
        # More radial room helps only while an event gate says restructuring is
        # meaningful; the wrapper retains the authoritative mask.
        if pressure >= 0.55:
            action[:3] = (0.8, 0.55, 0.30)
            action[3:6] = (0.20, 0.15, 0.10)
            action[6] = 0.35
            action[7:10] = (-0.55, -0.45, -0.25)
        elif pressure <= 0.20 and external == 0:
            action[:3] = (-0.25, -0.15, -0.10)
            action[7:10] = (0.15, 0.10, 0.10)
        else:
            action[6] = 0.10
            action[7:10] = (-0.15, -0.10, -0.05)
        return action
