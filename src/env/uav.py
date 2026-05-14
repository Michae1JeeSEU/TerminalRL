from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from src.config import UAVConfig


class UAVStatus(enum.IntEnum):
    CRUISING = 0
    DESCENDING = 1
    LANDED = 2


@dataclass
class LowLevelAction:
    """Per-UAV low-level action with continuous descent parameters."""
    control: int            # 0=CIRCLE, 1=DESCEND
    psi: float = 0.0        # heading angle [0, 2π)
    gamma: float = 0.0      # flight path angle [-π/2, π/2], negative = descending
    speed: float = 0.0      # flight speed (m/s)
    landing_zone: int = -1  # ≥0 triggers landing when on ring 3 + DESCEND


class UAV:
    """Single UAV with kinematic model, energy consumption, and priority."""

    def __init__(self, uav_id: int, config: UAVConfig, base_priority: int = 1):
        self.id = uav_id
        self.config = config
        self.base_priority = base_priority

        self.position = np.zeros(3, dtype=np.float64)      # (x, y, z)
        self.velocity = np.zeros(3, dtype=np.float64)       # (vx, vy, vz)
        self.ring_id = 1                                    # start in outer ring
        self.battery = config.max_battery
        self.status = UAVStatus.CRUISING
        self.wait_time = 0.0
        self.energy_consumed = 0.0
        self.landing_zone: int = -1
        self.is_controlled: bool = True   # False = background (auto-CIRCLE only)
        self._angle: float = 0.0

    def reset(self, position: np.ndarray, ring_id: int = 1,
              battery: float | None = None, wait_time: float = 0.0,
              status: UAVStatus = UAVStatus.CRUISING) -> None:
        self.position = position.copy()
        self.velocity = np.zeros(3, dtype=np.float64)
        self.ring_id = ring_id
        self.battery = battery if battery is not None else self.config.max_battery
        self.status = status
        self.wait_time = wait_time
        self.energy_consumed = 0.0
        self.landing_zone = -1
        self._angle = np.arctan2(position[1], position[0])

    @property
    def angle(self) -> float:
        return self._angle

    def step(self, action: LowLevelAction, ring_speed: float, ring_radius: float,
             ring_height: float, dt: float) -> None:
        """Advance UAV state by dt seconds given the current ring's orbit parameters."""
        if self.status == UAVStatus.LANDED:
            return

        if action.control == 0:  # CIRCLE
            self._step_circle(ring_speed, ring_radius, ring_height, dt)
        elif action.control == 1:  # DESCEND
            self._step_descend(action, dt)
            # Ring 3 + DESCEND + landing_zone specified → land
            if self.ring_id == 3 and action.landing_zone >= 0:
                self.landing_zone = action.landing_zone
                self.status = UAVStatus.LANDED
                self.velocity = np.zeros(3)
                self.position[2] = 0.0

        self._update_battery(dt)
        if self.status != UAVStatus.LANDED:
            self.wait_time += dt

    def _step_circle(self, ring_speed: float, ring_radius: float, ring_height: float,
                     dt: float) -> None:
        """Circle on the ring orbit at fixed radius and height."""
        self.status = UAVStatus.CRUISING
        omega = ring_speed / max(ring_radius, 1e-6)
        self._angle += omega * dt
        self.position[0] = ring_radius * np.cos(self._angle)
        self.position[1] = ring_radius * np.sin(self._angle)
        self.position[2] = ring_height
        self.velocity = np.array([
            -ring_speed * np.sin(self._angle),
            ring_speed * np.cos(self._angle),
            0.0,
        ])

    def _step_descend(self, action: LowLevelAction, dt: float) -> None:
        """Descend using 3D velocity from (psi, gamma, V)."""
        self.status = UAVStatus.DESCENDING
        psi = action.psi
        gamma = action.gamma
        V = max(action.speed, 1e-6)

        # 3D velocity: [V*cosγ*cosψ, V*cosγ*sinψ, V*sinγ]
        # Negative gamma = descending (sin negative)
        cos_gamma = np.cos(gamma)
        vx = V * cos_gamma * np.cos(psi)
        vy = V * cos_gamma * np.sin(psi)
        vz = V * np.sin(gamma)

        self.velocity = np.array([vx, vy, vz], dtype=np.float64)
        self.position += self.velocity * dt
        self._angle = np.arctan2(self.position[1], self.position[0])

    def _update_battery(self, dt: float) -> None:
        if self.status == UAVStatus.LANDED:
            return
        if not self.is_controlled:
            return  # Background UAVs don't consume battery
        power_map = {
            UAVStatus.CRUISING: self.config.power_cruise,
            UAVStatus.DESCENDING: self.config.power_descend,
        }
        power = power_map.get(self.status, self.config.power_cruise)
        self.energy_consumed += power * dt
        self.battery -= power * dt
        if self.battery <= 0:
            self.battery = 0.0

    @property
    def is_critical_battery(self) -> bool:
        return self.battery <= self.config.battery_critical * self.config.max_battery

    @property
    def is_dead(self) -> bool:
        return self.battery <= 0 and self.status != UAVStatus.LANDED

    def compute_priority(self) -> float:
        """Hybrid priority: base x f(wait_time) x g(battery)."""
        cfg = self.config
        f_wait = 1.0
        if cfg.max_battery > 0:
            wait_ratio = self.wait_time / 3600.0  # normalize by 1 hour
            f_wait = 1.0 + cfg.priority_alpha * (wait_ratio ** cfg.priority_beta)
        battery_ratio = self.battery / cfg.max_battery if cfg.max_battery > 0 else 0.0
        g_battery = 1.0
        if battery_ratio < cfg.battery_critical and battery_ratio > 0:
            g_battery = 1.0 + cfg.priority_gamma * (
                (1.0 - battery_ratio / cfg.battery_critical) ** cfg.priority_delta
            )
        return self.base_priority * f_wait * g_battery
