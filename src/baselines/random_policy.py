import numpy as np

from src.config import EnvConfig


class RandomPolicy:
    """Random action baseline for terminal area environment."""

    def __init__(self, config: EnvConfig, rng: np.random.Generator | None = None):
        self.config = config
        self.rng = rng or np.random.default_rng()
        self.n_uavs = config.num_uavs
        self.k = config.sim.num_waypoints
        self.m = config.sim.num_landing_zones

    def predict(self, obs: np.ndarray) -> dict:
        """Generate a random action."""
        hl = self.rng.uniform(
            low=[self.config.ring.r_min] * 3 + [self.config.ring.h_min] * 3 + [self.config.ring.v_min] * 3,
            high=[self.config.ring.r_max] * 3 + [self.config.ring.h_max] * 3 + [self.config.ring.v_max] * 3,
        ).astype(np.float64)

        ll_control = self.rng.integers(0, 3, size=self.n_uavs)
        ll_waypoints = self.rng.uniform(-1, 1, size=(self.n_uavs, self.k, 3)).astype(np.float64)
        ll_zones = self.rng.integers(0, self.m, size=self.n_uavs)

        return {
            "high_level": hl,
            "low_level": {
                "control": ll_control,
                "waypoints": ll_waypoints,
                "landing_zone": ll_zones,
            },
        }
