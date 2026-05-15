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

    def __init__(self, uav_id: int, config: UAVConfig, base_priority: int = 1,
                 mission_type: int = 1):
        self.id = uav_id
        self.config = config
        self.base_priority = base_priority
        self.mission_type = mission_type

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
        self.ring_entry_time: float = 0.0  # wait_time when entered current ring
        self._prev_ring_id: int = 1
        self._landing_target: np.ndarray | None = None  # set when landing approach begins

    def reset(self, position: np.ndarray, ring_id: int = 1,
              battery: float | None = None, wait_time: float = 0.0,
              status: UAVStatus = UAVStatus.CRUISING,
              mission_type: int | None = None) -> None:
        self.position = position.copy()
        self.velocity = np.zeros(3, dtype=np.float64)
        self.ring_id = ring_id
        self.battery = battery if battery is not None else self.config.max_battery
        self.status = status
        self.wait_time = wait_time
        self.energy_consumed = 0.0
        self.landing_zone = -1
        self._angle = np.arctan2(position[1], position[0])
        self.ring_entry_time = wait_time
        self._prev_ring_id = ring_id
        self._landing_target = None
        if mission_type is not None:
            self.mission_type = mission_type

    @property
    def angle(self) -> float:
        return self._angle

    def can_descend(self) -> bool:
        """Whether this UAV is allowed to DESCEND based on ring residency and priority."""
        if self.is_critical_battery:
            return True  # Full exemption: must land ASAP
        if self.base_priority >= self.config.priority_emergency:
            return True  # Full exemption: emergency/medical mission

        ring_residency = self.wait_time - self.ring_entry_time
        # Linear priority factor: p1=1.0(full wait), p5=0(no wait)
        priority_factor = max(0.0, 1.0 - (self.base_priority - 1) / 4.0)
        required_wait = self.config.min_ring_wait * priority_factor if hasattr(self.config, 'min_ring_wait') else 30.0
        return ring_residency >= required_wait

    def step(self, action: LowLevelAction, ring_speed: float, ring_radius: float,
             ring_height: float, dt: float) -> None:
        """Advance UAV state by dt seconds given the current ring's orbit parameters."""
        if self.status == UAVStatus.LANDED:
            return

        if action.control == 0:  # CIRCLE
            speed_factor = 1.0
            if action.speed > 0.01:
                speed_factor = float(np.clip(action.speed, 0.7, 1.3))
            self._step_circle(ring_speed, ring_radius, ring_height, dt, speed_factor)
        elif action.control == 1:  # DESCEND
            self._step_descend(action, dt)
            # Ground contact: land immediately regardless of ring
            if self.position[2] <= 1.0:
                self.status = UAVStatus.LANDED
                self.velocity = np.zeros(3)
                self.position[2] = 0.0
                if action.landing_zone >= 0:
                    self.landing_zone = action.landing_zone
            # Ring 3 + DESCEND + landing_zone specified → continuous landing approach
            elif self.ring_id == 3 and action.landing_zone >= 0:
                self.landing_zone = action.landing_zone

        self._update_battery(dt)
        if self.status != UAVStatus.LANDED:
            self.wait_time += dt

    def _step_circle(self, ring_speed: float, ring_radius: float, ring_height: float,
                     dt: float, speed_factor: float = 1.0) -> None:
        """Circle on the ring's prescribed orbit (R_i, H_i).

        Locks UAV to the ring orbit, correcting any drift from descent.
        speed_factor: multiplier on ring_speed for separation control [0.7, 1.3].
        """
        self.status = UAVStatus.CRUISING
        effective_speed = ring_speed * speed_factor
        omega = effective_speed / max(ring_radius, 1e-6)
        self._angle += omega * dt
        self.position[0] = ring_radius * np.cos(self._angle)
        self.position[1] = ring_radius * np.sin(self._angle)
        self.position[2] = ring_height
        tangential_speed = omega * ring_radius
        self.velocity = np.array([
            -tangential_speed * np.sin(self._angle),
            tangential_speed * np.cos(self._angle),
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
