import numpy as np

from src.config import EnvConfig


class HeuristicPolicy:
    """Greedy heuristic: descend toward inner ring as fast as possible.

    Strategy per UAV:
    - Outer ring (ring 1): move radially inward to middle ring + descend to H2
    - Middle ring (ring 2): move radially inward to inner ring + descend to H3
    - Inner ring (ring 3): pick nearest landing zone and land immediately
    """

    def __init__(self, config: EnvConfig):
        self.config = config
        self.n_uavs = config.num_uavs
        self.k = config.sim.num_waypoints
        self.m = config.sim.num_landing_zones
        self._landing_zone_assignments: dict[int, int] = {}

    def predict(self, obs: np.ndarray) -> dict:
        n, k, m = self.n_uavs, self.k, self.m
        offset = 9 + 3  # skip airspace(9) + ring_counts(3)

        ll_control = np.zeros(n, dtype=np.int32)
        ll_waypoints = np.zeros((n, k, 3), dtype=np.float64)
        ll_zones = np.zeros(n, dtype=np.int32)

        radii = obs[:3]    # [R1, R2, R3]
        heights = obs[3:6]  # [H1, H2, H3]

        for i in range(n):
            uav_start = offset + i * 5
            if uav_start + 5 > len(obs):
                break
            px, py, pz = obs[uav_start], obs[uav_start + 1], obs[uav_start + 2]
            battery = obs[uav_start + 3]
            if battery <= 0:
                continue

            r = np.sqrt(px * px + py * py)
            ring_id = 1
            for ri in range(2, -1, -1):
                if r <= radii[ri] and pz <= heights[ri]:
                    ring_id = ri + 1
                    break

            if ring_id == 3:
                # Inner ring: land at nearest zone
                ll_control[i] = 2
                ll_zones[i] = self._pick_nearest_zone(px, py, obs[-m * 3:])
                ll_waypoints[i] = np.zeros((k, 3), dtype=np.float64)
            else:
                # Move inward to next ring
                ll_control[i] = 2
                next_r = radii[ring_id]  # radius of next inner ring
                next_h = heights[ring_id]  # height of next inner ring
                angle = np.arctan2(py, px)
                target = np.array([next_r * np.cos(angle),
                                   next_r * np.sin(angle), next_h])
                current = np.array([px, py, pz])
                ll_waypoints[i] = self._waypoints_toward(current, target, k)

        hl = np.array(
            list(self.config.ring.default_radii)
            + list(self.config.ring.default_heights)
            + list(self.config.ring.default_speeds),
            dtype=np.float64,
        )

        return {
            "high_level": hl,
            "low_level": {
                "control": ll_control,
                "waypoints": ll_waypoints,
                "landing_zone": ll_zones,
            },
        }

    def _waypoints_toward(self, current: np.ndarray, target: np.ndarray, num: int
                          ) -> np.ndarray:
        """Generate waypoint offsets moving from current to target.

        Returns offsets in [-1, 1] normalized by R1. The env computes:
            waypoint_abs = uav.position + offset * R1
        The last waypoint is exactly at the target position.
        """
        R1 = self.config.ring.default_radii[0]
        wp = np.zeros((num, 3), dtype=np.float64)
        for i in range(num):
            frac = (i + 1) / num  # evenly spaced, last at target
            wp[i] = (current + (target - current) * frac - current) / R1
        return wp

    def _pick_nearest_zone(self, px: float, py: float, zones_flat: np.ndarray) -> int:
        best = 0
        best_dist = float("inf")
        for j in range(self.m):
            zx = zones_flat[j * 3]
            zy = zones_flat[j * 3 + 1]
            dist = (px - zx) ** 2 + (py - zy) ** 2
            if dist < best_dist:
                best_dist = dist
                best = j
        return best
