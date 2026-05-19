import numpy as np

from src.config import EnvConfig


class HeuristicPolicy:
    """Greedy heuristic: descend toward center with fixed flight path angle.

    Strategy per UAV:
    - Outer/middle ring (ring 1, 2): DESCEND with psi=toward center, gamma=-12deg
    - Inner ring (ring 3): DESCEND + nearest landing zone → land
    - Wait to descend (ring-wait not yet satisfied): CIRCLE at 1.0x ring speed

    Outputs raw low_level dict (bypasses PerUAVDiscreteWrapper).
    control: 0=CIRCLE, 1=DESCEND. speed sets circling speed_factor [0.3, 1.3].
    """

    def __init__(self, config: EnvConfig):
        self.config = config
        self.max_n = config.max_uavs
        self.m = config.sim.num_landing_zones
        self.safe_dist = config.uav.safe_separation
        # Landing zone positions (approximate, inner ring 60% radius)
        inner_r = config.ring.default_radii[2] * 0.6
        angles = np.linspace(0, 2 * np.pi, self.m, endpoint=False)
        self._lz_positions = np.zeros((self.m, 2), dtype=np.float64)
        self._lz_positions[:, 0] = inner_r * np.cos(angles)
        self._lz_positions[:, 1] = inner_r * np.sin(angles)

    def predict(self, obs: np.ndarray) -> dict:
        # obs shape: (max_n, local_obs_dim)
        max_n, m = self.max_n, self.m
        R1 = self.config.ring.default_radii[0]
        H1 = self.config.ring.default_heights[0]

        controls = np.zeros(max_n, dtype=np.int32)   # 0=CIRCLE, 1=DESCEND
        psi_arr = np.zeros(max_n, dtype=np.float64)
        gamma_arr = np.zeros(max_n, dtype=np.float64)
        speed_arr = np.ones(max_n, dtype=np.float64)  # CIRCLE speed_factor (1.0 = ring speed)
        zones = np.full(max_n, m, dtype=np.int32)

        for i in range(max_n):
            row = obs[i]
            sin_t, cos_t = row[9], row[10]
            ring_norm = row[11]
            r_norm = row[12]
            z_norm = row[13]
            can_descend = row[17]

            if abs(sin_t) < 1e-9 and abs(cos_t) < 1e-9 and abs(ring_norm) < 1e-9:
                continue

            ring_id = int(round(ring_norm * 3))
            if ring_id < 1 or ring_id > 3:
                continue

            if can_descend < 0.5:
                continue  # stay CIRCLE (control=0, speed=1.0)

            # DESCEND
            controls[i] = 1

            r = r_norm * R1
            z = z_norm * H1
            theta = np.arctan2(sin_t, cos_t)
            px = r * cos_t
            py = r * sin_t

            if ring_id == 3:
                best_lz = 0
                best_dist = float('inf')
                for j, lz in enumerate(self._lz_positions):
                    d = (px - lz[0])**2 + (py - lz[1])**2
                    if d < best_dist:
                        best_dist = d
                        best_lz = j
                psi_arr[i] = np.arctan2(self._lz_positions[best_lz][1] - py,
                                        self._lz_positions[best_lz][0] - px)
                zones[i] = best_lz
                gamma_arr[i] = -np.deg2rad(25.0)
                speed_arr[i] = self.config.uav.descent_speed
            else:
                psi_arr[i] = theta + np.pi
                gamma_arr[i] = -np.deg2rad(12.0)
                speed_arr[i] = self.config.uav.max_speed

        hl = np.array(
            list(self.config.ring.default_radii)
            + list(self.config.ring.default_heights)
            + list(self.config.ring.default_speeds),
            dtype=np.float64,
        )

        return {
            "high_level": hl,
            "low_level": {
                "control": controls,
                "psi": psi_arr,
                "gamma": gamma_arr,
                "speed": speed_arr,
                "landing_zone": zones,
            },
        }
