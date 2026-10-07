"""Target-ring occupancy at a proposed merge time, without candidate queries."""
from __future__ import annotations

import math
import numpy as np

from .safety_filter import SafetyFilter


def target_ring_angles(config, ego, target, moment, snapshot):
    """Forecast current holders and commitments arriving before this slot.

    A transfer has no current ring_role. Its immutable target and end time,
    followed by nominal ring continuation, determine future membership.
    A committed departure must not be extrapolated as an indefinite holder.
    Full trajectory/swept checks remain the safety filter's responsibility.
    """
    predictor = SafetyFilter(config)
    times = np.asarray([float(moment)])
    angles = []
    for other in snapshot.active.values():
        if other.uid == ego.uid:
            continue
        plan = other.committed_plan
        if plan is None:
            if other.ring_role is not target:
                continue
            position = predictor._holder_positions(other, snapshot, times)[0]
        else:
            if (plan.metadata.get('target_phase') != 'ring_' + target.value
                    or plan.end_time > moment + 1e-9):
                continue
            position = predictor._committed_positions(plan, snapshot, times)[0]
        angles.append(math.atan2(float(position[1]), float(position[0])) % (2 * math.pi))
    return sorted(angles)


def largest_gap_center(angles):
    """Cyclic midpoint; the only holder has a full-circle gap behind it."""
    if not angles:
        raise ValueError('an empty ring has no target-holder gap')
    if len(angles) == 1:
        return (angles[0] + math.pi) % (2 * math.pi)
    gap, start = max(((angles[(i + 1) % len(angles)] - angle) % (2 * math.pi), angle)
                     for i, angle in enumerate(angles))
    return (start + gap / 2) % (2 * math.pi)
