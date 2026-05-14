"""Gym wrappers for the terminal area environment."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FixedAirspaceWrapper(gym.Wrapper):
    """Fixes the high-level airspace action and exposes only low-level control."""

    def __init__(self, env):
        super().__init__(env)
        self.max_n = env.max_n
        self.m = env.config.sim.num_landing_zones
        self._fixed_hl = np.array(
            list(env.config.ring.default_radii)
            + list(env.config.ring.default_heights)
            + list(env.config.ring.default_speeds),
            dtype=np.float64,
        )
        self.action_space = env.low_level_action_space
        self.observation_space = env.observation_space

    def step(self, action):
        return self.env.step({"high_level": self._fixed_hl, "low_level": action})


class DiscreteActionWrapper(gym.Wrapper):
    """Shared discrete control: one action (CIRCLE=0/DESCEND=1) applied to all UAVs.

    DESCEND uses auto-computed continuous params:
    - psi: toward center (radially inward)
    - gamma: -15 degrees
    - speed: config.descent_speed
    - landing_zone: nearest zone (auto-selected by env or fixed to 0)
    """

    def __init__(self, env):
        super().__init__(env)
        self.max_n = env.max_n
        self._fixed_hl = np.array(
            list(env.config.ring.default_radii)
            + list(env.config.ring.default_heights)
            + list(env.config.ring.default_speeds),
            dtype=np.float64,
        )
        self.action_space = spaces.MultiDiscrete([2])  # 0=CIRCLE, 1=DESCEND
        self.observation_space = env.observation_space

    def step(self, action):
        ctrl = int(action[0])
        controls = np.full(self.max_n, ctrl, dtype=np.int32)

        # Auto descent parameters — compute psi toward center from UAV positions
        psi = np.zeros(self.max_n, dtype=np.float64)
        gamma = np.full(self.max_n, -np.deg2rad(15.0), dtype=np.float64)
        speed = np.full(self.max_n, self.unwrapped.config.uav.max_speed, dtype=np.float64)
        m = self.unwrapped.config.sim.num_landing_zones
        zones = np.full(self.max_n, m, dtype=np.int32)  # m = no-op

        for i, u in enumerate(self.unwrapped.uav_manager.uavs):
            if i >= self.max_n:
                break
            if u.is_controlled:
                theta = np.arctan2(u.position[1], u.position[0])
                psi[i] = theta + np.pi  # toward center
                if u.ring_id == 3:
                    zones[i] = 0  # nearest landing zone

        ll = {
            "control": controls,
            "psi": psi,
            "gamma": gamma,
            "speed": speed,
            "landing_zone": zones,
        }
        return self.env.step({"high_level": self._fixed_hl, "low_level": ll})


def unflatten_action(flat: np.ndarray, max_n: int, m: int) -> dict:
    """Convert a flat continuous action vector to the low-level dict format.

    Layout: control(2*N) + psi(N) + gamma(N) + speed(N) + landing_zone(N*(m+1))
    Discrete parts use argmax.
    """
    idx = 0
    control_raw = flat[idx:idx + 2 * max_n].reshape(max_n, 2)
    idx += 2 * max_n
    control = np.argmax(control_raw, axis=1).astype(np.int32)

    psi = flat[idx:idx + max_n] * np.pi  # scale from [-1,1] to [0, 2π]
    idx += max_n
    gamma = flat[idx:idx + max_n] * (np.pi / 2)  # scale from [-1,1] to [-π/2, π/2]
    idx += max_n
    speed = 5.0 + (flat[idx:idx + max_n] + 1.0) / 2.0 * 35.0  # scale to [5, 40]
    idx += max_n

    zone_raw = flat[idx:idx + max_n * (m + 1)].reshape(max_n, m + 1)
    landing_zone = np.argmax(zone_raw, axis=1).astype(np.int32)

    return {
        "control": control,
        "psi": psi,
        "gamma": gamma,
        "speed": speed,
        "landing_zone": landing_zone,
    }
