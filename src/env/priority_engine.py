from src.env.uav import UAV, UAVStatus


class PriorityEngine:
    """Computes hybrid priorities and provides sorted UAV ordering."""

    def compute(self, uav: UAV) -> float:
        """Delegates to UAV.compute_priority()."""
        return uav.compute_priority()

    def get_sorting_key(self, uavs: list[UAV]) -> list[UAV]:
        """Return UAVs sorted by descending priority (highest first)."""
        active = [u for u in uavs if u.status != UAVStatus.LANDED]
        return sorted(active, key=lambda u: u.compute_priority(), reverse=True)

    def get_priority_violations(self, uavs: list[UAV]) -> float:
        """Count priority inversions in the landing order.

        A violation occurs when a lower-priority UAV has landed while a
        higher-priority UAV is still waiting.
        """
        sorted_uavs = self.get_sorting_key(uavs)
        landed = [u for u in uavs if u.status == UAVStatus.LANDED]
        waiting = [u for u in uavs if u.status != UAVStatus.LANDED]
        violations = 0.0
        for w in waiting:
            for l in landed:
                if w.compute_priority() > l.compute_priority():
                    violations += 1.0
        return violations
