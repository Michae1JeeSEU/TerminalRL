"""Gym wrappers for the terminal area environment."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.env.terminal_area_env import TerminalAreaEnv


class FixedAirspaceWrapper(gym.Wrapper):
    """Fixes the high-level airspace action and exposes only low-level control.

    For single-UAV or fixed-configuration training scenarios.
    """

    def __init__(self, env: TerminalAreaEnv):
        super().__init__(env)
        self.n_uavs = env.config.num_uavs
        self.k = env.config.sim.num_waypoints
        self.m = env.config.sim.num_landing_zones

        self._fixed_hl = np.array(
            list(env.config.ring.default_radii)
            + list(env.config.ring.default_heights)
            + list(env.config.ring.default_speeds),
            dtype=np.float64,
        )

        # Flattened low-level action: control(3) + waypoints(K*3) + landing_zone(M)
        flat_dim = 3 * self.n_uavs + self.k * 3 * self.n_uavs + self.m * self.n_uavs
        self.action_space = spaces.Box(-1.0, 1.0, shape=(flat_dim,), dtype=np.float64)
        self.observation_space = env.observation_space

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        ll = unflatten_action(action, self.n_uavs, self.k, self.m)
        full_action = {"high_level": self._fixed_hl, "low_level": ll}
        return self.env.step(full_action)


class DiscreteActionWrapper(gym.Wrapper):
    """Shared discrete control: one action applied to all UAVs.

    Action space: MultiDiscrete([3]) — CIRCLE/HOVER/DESCEND for all UAVs.
    Landing zone auto-set to nearest zone (0). Waypoints are zero-offset.
    """

    def __init__(self, env: TerminalAreaEnv):
        super().__init__(env)
        self.n_uavs = env.config.num_uavs
        self.k = env.config.sim.num_waypoints
        self._fixed_hl = np.array(
            list(env.config.ring.default_radii)
            + list(env.config.ring.default_heights)
            + list(env.config.ring.default_speeds),
            dtype=np.float64,
        )
        self.action_space = spaces.MultiDiscrete([3])
        self.observation_space = env.observation_space

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        ctrl = int(action[0])
        controls = np.full(self.n_uavs, ctrl, dtype=np.int32)
        zones = np.zeros(self.n_uavs, dtype=np.int32)
        ll = {
            "control": controls,
            "waypoints": self._auto_waypoints(),
            "landing_zone": zones,
        }
        full_action = {"high_level": self._fixed_hl, "low_level": ll}
        return self.env.step(full_action)

    def _auto_waypoints(self) -> np.ndarray:
        return np.zeros((self.n_uavs, self.k, 3), dtype=np.float64)


def unflatten_action(flat: np.ndarray, n_uavs: int, k: int, m: int) -> dict:
    """Convert a flat continuous action vector to the low-level dict format.

    Layout: control(3*N) + waypoints(N*K*3) + landing_zone(N*M)
    Discrete parts use argmax to select from the flat representation.
    """
    idx = 0

    control_raw = flat[idx:idx + 3 * n_uavs].reshape(n_uavs, 3)
    idx += 3 * n_uavs
    control = np.argmax(control_raw, axis=1).astype(np.int32)

    wp = flat[idx:idx + n_uavs * k * 3].reshape(n_uavs, k, 3)
    idx += n_uavs * k * 3

    zone_raw = flat[idx:idx + n_uavs * m].reshape(n_uavs, m)
    landing_zone = np.argmax(zone_raw, axis=1).astype(np.int32)

    return {"control": control, "waypoints": wp, "landing_zone": landing_zone}
