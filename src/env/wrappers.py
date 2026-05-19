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

        # Auto descent parameters
        psi = np.zeros(self.max_n, dtype=np.float64)
        m = self.unwrapped.config.sim.num_landing_zones
        zones = np.full(self.max_n, m, dtype=np.int32)  # m = no-op

        gamma = np.zeros(self.max_n, dtype=np.float64)
        if ctrl == 0:  # CIRCLE
            speed = np.ones(self.max_n, dtype=np.float64)  # neutral multiplier
        else:  # DESCEND
            speed = np.full(self.max_n, self.unwrapped.config.uav.max_speed, dtype=np.float64)
            gamma = np.full(self.max_n, -np.deg2rad(12.0), dtype=np.float64)  # ring geometry slope

        env = self.unwrapped
        lz_positions = env.landing_zones

        for i, u in enumerate(env.uav_manager.uavs):
            if i >= self.max_n:
                break
            if u.is_controlled:
                theta = np.arctan2(u.position[1], u.position[0])
                if u.ring_id == 3 and u._landing_target is not None:
                    # Landing approach: steer toward landing zone
                    lz = u._landing_target
                    psi[i] = np.arctan2(lz[1] - u.position[1], lz[0] - u.position[0])
                    zones[i] = u.landing_zone if u.landing_zone >= 0 else 0
                    speed[i] = self.unwrapped.config.uav.descent_speed
                    gamma[i] = -np.deg2rad(25.0)  # steep toward landing zone
                elif u.ring_id == 3 and ctrl == 1:
                    # Initiate landing: point toward nearest landing zone
                    nearest_lz = 0
                    nearest_dist = float('inf')
                    for j, lz in enumerate(lz_positions):
                        d = np.linalg.norm(u.position[:2] - lz[:2])
                        if d < nearest_dist:
                            nearest_dist = d
                            nearest_lz = j
                    psi[i] = np.arctan2(lz_positions[nearest_lz][1] - u.position[1],
                                        lz_positions[nearest_lz][0] - u.position[0])
                    zones[i] = nearest_lz
                    speed[i] = self.unwrapped.config.uav.descent_speed
                    gamma[i] = -np.deg2rad(25.0)  # steep toward landing zone
                else:
                    psi[i] = theta + np.pi  # toward center (radially inward)

        ll = {
            "control": controls,
            "psi": psi,
            "gamma": gamma,
            "speed": speed,
            "landing_zone": zones,
        }
        return self.env.step({"high_level": self._fixed_hl, "low_level": ll})


class PerUAVDiscreteWrapper(gym.Wrapper):
    """Per-UAV control: each UAV independently chooses CIRCLE or DESCEND.

    Action space: Box(max_n, 2) — per UAV:
      dim 0 (control_raw): > 0 → DESCEND, ≤ 0 → CIRCLE
      dim 1 (speed_raw):   scaled [-1,1] → [0.3, 1.3] ring speed multiplier (CIRCLE only)

    Continuous params (psi, gamma, speed, landing_zone) are auto-computed:
    - CIRCLE: speed set from dim 1 → ring_speed * speed_factor in UAV.step()
    - DESCEND ring 1/2: psi toward center, gamma=-12°, speed=max_speed
    - DESCEND ring 3: psi toward nearest LZ, gamma=-25°, speed=descent_speed
    - DESCEND ring 3 with ongoing approach: psi toward existing landing_target

    Ring-wait constraint is enforced by env._parse_low_level_actions (DESCEND
    from policy gets overridden to CIRCLE when can_descend() is False).
    """

    def __init__(self, env):
        super().__init__(env)
        self.max_n = env.max_n
        self.m = self.unwrapped.config.sim.num_landing_zones
        self._fixed_hl = np.array(
            list(env.config.ring.default_radii)
            + list(env.config.ring.default_heights)
            + list(env.config.ring.default_speeds),
            dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.max_n, 2), dtype=np.float64,
        )
        self.observation_space = env.observation_space

    def step(self, action):
        raw = np.asarray(action, dtype=np.float64)
        if raw.ndim == 3:  # vec env wrapper may add batch dim
            raw = raw.squeeze(0)
        max_n = self.max_n
        m = self.m

        # Column 0: control_raw > 0 → DESCEND, ≤ 0 → CIRCLE
        controls = np.where(raw[:, 0] > 0, 1, 0).astype(np.int32)
        # Column 1: scale [-1, 1] → [0.3, 1.3] for CIRCLE speed
        speed_raw = raw[:, 1]
        circle_speed = 0.3 + (speed_raw + 1.0) / 2.0 * 1.0
        circle_speed = np.clip(circle_speed, 0.3, 1.3)

        psi = np.zeros(max_n, dtype=np.float64)
        gamma = np.zeros(max_n, dtype=np.float64)
        speed = circle_speed.copy()  # default: use CIRCLE speed from action
        zones = np.full(max_n, m, dtype=np.int32)

        env = self.unwrapped
        lz_positions = env.landing_zones

        for i, u in enumerate(env.uav_manager.uavs):
            if i >= max_n:
                break
            if controls[i] == 1 and u.is_controlled:
                # DESCEND — auto-compute continuous parameters
                theta = np.arctan2(u.position[1], u.position[0])
                if u.ring_id == 3:
                    if u._landing_target is not None:
                        lz = u._landing_target
                        psi[i] = np.arctan2(lz[1] - u.position[1], lz[0] - u.position[0])
                        zones[i] = u.landing_zone if u.landing_zone >= 0 else 0
                    else:
                        nearest = int(np.argmin(
                            [np.linalg.norm(u.position[:2] - lz[:2]) for lz in lz_positions]
                        ))
                        psi[i] = np.arctan2(
                            lz_positions[nearest][1] - u.position[1],
                            lz_positions[nearest][0] - u.position[0],
                        )
                        zones[i] = nearest
                    gamma[i] = -np.deg2rad(25.0)
                    speed[i] = self.unwrapped.config.uav.descent_speed
                else:
                    psi[i] = theta + np.pi
                    gamma[i] = -np.deg2rad(12.0)
                    speed[i] = self.unwrapped.config.uav.max_speed

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
