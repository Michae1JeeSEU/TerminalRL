from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from src.config import UAVConfig


class UAVStatus(enum.IntEnum):
    CRUISING = 0
    HOLDING = 1       # hovering in place
    DESCENDING = 2
    LANDED = 3


@dataclass
class LowLevelAction:
    """Per-UAV low-level action."""
    control: int          # 0=CIRCLE, 1=HOVER, 2=DESCEND
    waypoints: np.ndarray | None = None  # shape (K, 3), only valid for DESCEND
    landing_zone: int = -1               # only valid when in inner ring + landing


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
        self._waypoint_idx = 0
        self._waypoints: np.ndarray | None = None
        self._target_ring_id: int | None = None

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
        self._waypoint_idx = 0
        self._waypoints = None
        self._target_ring_id = None

    def step(self, action: LowLevelAction, ring_speed: float, ring_radius: float, dt: float) -> None:
        """Advance UAV state by dt seconds given the current ring's speed."""
        if self.status == UAVStatus.LANDED:
            return

        if action.control == 0:  # CIRCLE
            self._step_circle(ring_speed, ring_radius, dt)
        elif action.control == 1:  # HOVER
            self._step_hover(dt)
        elif action.control == 2:  # DESCEND
            self._step_descend(dt)

        if action.landing_zone >= 0 and self.status != UAVStatus.LANDED:
            self.landing_zone = action.landing_zone
            self.status = UAVStatus.LANDED
            self.velocity = np.zeros(3)

        self._update_battery(dt)
        if self.status != UAVStatus.LANDED:
            self.wait_time += dt

    def _step_circle(self, ring_speed: float, ring_radius: float, dt: float) -> None:
        """Move along the ring circumference at ring_speed."""
        self.status = UAVStatus.CRUISING
        r = np.linalg.norm(self.position[:2])
        if r < 1e-6:
            return
        angular_velocity = ring_speed / max(r, 1e-6)
        angle = np.arctan2(self.position[1], self.position[0])
        angle += angular_velocity * dt
        self.position[0] = r * np.cos(angle)
        self.position[1] = r * np.sin(angle)
        self.velocity = np.array([-ring_speed * np.sin(angle),
                                  ring_speed * np.cos(angle), 0.0])

    def _step_hover(self, dt: float) -> None:
        self.status = UAVStatus.HOLDING
        self.velocity = np.zeros(3)

    def _step_descend(self, dt: float) -> None:
        """Descend along waypoints toward the inner ring."""
        self.status = UAVStatus.DESCENDING
        if self._waypoints is not None and self._waypoint_idx < len(self._waypoints):
            target = self._waypoints[self._waypoint_idx]
            direction = target - self.position
            dist = np.linalg.norm(direction)
            step_dist = self.config.descent_speed * dt
            if dist <= step_dist:
                self.position = target.copy()
                self._waypoint_idx += 1
            else:
                self.position += direction / dist * step_dist
            self.velocity = np.zeros(3)
        else:
            # Default descent: move both downward and radially inward
            speed = self.config.descent_speed
            step_dist = speed * dt
            r = np.linalg.norm(self.position[:2])
            if r > 1.0:
                inward_dir = -self.position[:2] / r  # unit vector toward center
            else:
                inward_dir = np.zeros(2)
            self.position[:2] += inward_dir * step_dist
            self.position[2] -= step_dist
            self.velocity = np.array([inward_dir[0] * speed,
                                      inward_dir[1] * speed, -speed])
        if self._waypoints is not None and self._waypoint_idx >= len(self._waypoints):
            self._waypoints = None
            self._waypoint_idx = 0

    def set_descent_path(self, waypoints: np.ndarray, target_ring_id: int) -> None:
        """Set the waypoint sequence for descent."""
        self._waypoints = waypoints.copy()
        self._waypoint_idx = 0
        self._target_ring_id = target_ring_id

    def _update_battery(self, dt: float) -> None:
        if self.status == UAVStatus.LANDED:
            return
        power_map = {
            UAVStatus.CRUISING: self.config.power_cruise,
            UAVStatus.HOLDING: self.config.power_hover,
            UAVStatus.DESCENDING: self.config.power_descend,
        }
        self.energy_consumed += power_map[self.status] * dt
        self.battery -= power_map[self.status] * dt
        if self.battery <= 0:
            self.battery = 0.0

    @property
    def is_critical_battery(self) -> bool:
        return self.battery <= self.config.battery_critical * self.config.max_battery

    @property
    def is_dead(self) -> bool:
        return self.battery <= 0 and self.status != UAVStatus.LANDED

    def compute_priority(self) -> float:
        """Hybrid priority: base × f(wait_time) × g(battery)."""
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
