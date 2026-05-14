import numpy as np

from src.config import EnvConfig


class RandomPolicy:
    """Random action baseline for terminal area environment."""

    def __init__(self, config: EnvConfig, rng: np.random.Generator | None = None):
        self.config = config
        self.rng = rng or np.random.default_rng()
        self.max_n = config.max_uavs
        self.m = config.sim.num_landing_zones

    def predict(self, obs: np.ndarray) -> dict:
        """Generate a random action."""
        hl = self.rng.uniform(
            low=[self.config.ring.r_min] * 3 + [self.config.ring.h_min] * 3 + [self.config.ring.v_min] * 3,
            high=[self.config.ring.r_max] * 3 + [self.config.ring.h_max] * 3 + [self.config.ring.v_max] * 3,
        ).astype(np.float64)

        ll_control = self.rng.integers(0, 2, size=self.max_n).astype(np.int32)
        ll_psi = self.rng.uniform(0, 2 * np.pi, size=self.max_n).astype(np.float64)
        ll_gamma = self.rng.uniform(-np.pi / 2, np.pi / 2, size=self.max_n).astype(np.float64)
        ll_speed = self.rng.uniform(5, 40, size=self.max_n).astype(np.float64)
        ll_zones = self.rng.integers(0, self.m + 1, size=self.max_n).astype(np.int32)

        return {
            "high_level": hl,
            "low_level": {
                "control": ll_control,
                "psi": ll_psi,
                "gamma": ll_gamma,
                "speed": ll_speed,
                "landing_zone": ll_zones,
            },
        }
