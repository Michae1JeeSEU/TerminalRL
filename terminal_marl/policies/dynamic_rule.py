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
            # Expand the upstream buffer first; inner/final geometry is only
            # touched after the coordinator has created an explicit window.
            action[:3] = (0.10, 0.0, 0.0)
            action[3:6] = (0.0, 0.0, 0.0)
            action[6] = 0.0
            # During an active protected reconfiguration window, retain a
            # conservative release buffer; throughput is not traded for a
            # hard-separation risk.
            action[7:10] = (0.25, 0.20, 0.15)
        elif pressure <= 0.20 and external == 0:
            action[:3] = (-0.25, -0.15, -0.10)
            action[7:10] = (0.15, 0.10, 0.10)
        else:
            action[6] = 0.10
            action[7:10] = (-0.15, -0.10, -0.05)
        return action
