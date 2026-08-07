from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import math
import numpy as np

from .geometry import wrap_angle, clamp


class FlightPhase(IntEnum):
    APPROACH = 0
    RING = 1
    FINAL = 2
    FINISHED = 3


@dataclass
class UAVState:
    uid: int
    position: np.ndarray
    speed: float
    heading: float
    pitch: float = 0.0
    phase: FlightPhase = FlightPhase.APPROACH
    assigned_gate: Optional[int] = None
    assigned_ring: Optional[int] = None
    sequence_number: Optional[int] = None
    ring_slot_active: bool = False
    final_cleared: bool = False
    final_entry_fix_reached: bool = False
    final_entry_target_time: Optional[float] = None
    finished: bool = False
    assigned_ring_step: Optional[int] = None
    ring_established_step: Optional[int] = None
    last_transfer_step: Optional[int] = None

    transition_active: bool = False
    transition_kind: Optional[str] = None
    transition_p0: Optional[np.ndarray] = None
    transition_p1: Optional[np.ndarray] = None
    transition_p2: Optional[np.ndarray] = None
    transition_p3: Optional[np.ndarray] = None
    transition_progress: float = 0.0

    # Episode-level metrics
    initial_position: Optional[np.ndarray] = None
    completion_step: Optional[int] = None
    completion_time: Optional[float] = None
    path_length_total: float = 0.0
    waiting_ring_distance: float = 0.0
    cumulative_turn_angle_rad: float = 0.0
    cumulative_altitude_change: float = 0.0
    residual_action_l1_total: float = 0.0
    touchdown_hold_steps: int = 0
    spawn_time: float = 0.0
    entry_time: Optional[float] = None
    planned_nominal_landing_time: Optional[float] = None
    actual_landing_time: Optional[float] = None
    landing_delay: Optional[float] = None
    arrival_id: Optional[int] = None
    request_time: float = 0.0
    admission_time: Optional[float] = None
    terminal_time: Optional[float] = None
    terminal_outcome: Optional[str] = None
    terminal_reason: Optional[str] = None
    battery_endurance_seconds: Optional[float] = None
    remaining_endurance_seconds: Optional[float] = None
    collision_partner_ids: list[int] = field(default_factory=list)

    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.initial_position is None:
            self.initial_position = self.position.copy()

    def snapshot(self) -> None:
        self.history.append(self.position.copy())

    def clear_transition(self) -> None:
        self.transition_active = False
        self.transition_kind = None
        self.transition_p0 = None
        self.transition_p1 = None
        self.transition_p2 = None
        self.transition_p3 = None
        self.transition_progress = 0.0

    def record_step_metrics(
        self,
        *,
        heading_step_rad: float,
        dz: float,
        path_distance: float,
        residual_action: np.ndarray,
        phase_before_step: FlightPhase,
    ) -> None:
        self.path_length_total += float(path_distance)
        if phase_before_step == FlightPhase.RING:
            self.waiting_ring_distance += float(path_distance)
        self.cumulative_turn_angle_rad += abs(float(heading_step_rad))
        self.cumulative_altitude_change += abs(float(dz))
        residual = np.asarray(residual_action, dtype=np.float32).reshape(-1)
        self.residual_action_l1_total += float(np.abs(residual).sum())

    def mark_finished(self, step: int, time: float) -> None:
        if self.completion_step is None:
            self.completion_step = int(step)
        if self.completion_time is None:
            self.completion_time = float(time)
        if self.actual_landing_time is None:
            self.actual_landing_time = float(time)
        if (
            self.landing_delay is None
            and self.planned_nominal_landing_time is not None
        ):
            self.landing_delay = (
                self.actual_landing_time - self.planned_nominal_landing_time
            )

    def mark_terminal(
        self,
        *,
        step: int,
        time: float,
        outcome: str,
        reason: str,
        collision_partner_ids: Optional[list[int]] = None,
    ) -> None:
        """Set one primary terminal outcome for the UAV."""

        if self.terminal_outcome is not None:
            return
        self.finished = True
        self.phase = FlightPhase.FINISHED
        self.terminal_time = float(time)
        self.terminal_outcome = str(outcome)
        self.terminal_reason = str(reason)
        if collision_partner_ids is not None:
            self.collision_partner_ids = [int(uid) for uid in collision_partner_ids]
        self.mark_finished(step=step, time=time)

    def direct_distance_to_goal(self, center: tuple[float, float, float]) -> float:
        c = np.asarray(center, dtype=float)
        return float(np.linalg.norm(np.asarray(self.initial_position, dtype=float) - c))

    def path_detour_rate(self, center: tuple[float, float, float]) -> float:
        direct = self.direct_distance_to_goal(center)
        if direct <= 1e-6:
            return 0.0
        return max(0.0, self.path_length_total / direct - 1.0)

    def metrics_dict(self, center: tuple[float, float, float]) -> dict:
        return {
            "finished": bool(self.finished),
            "completion_step": None if self.completion_step is None else int(self.completion_step),
            "completion_time": None if self.completion_time is None else float(self.completion_time),
            "spawn_time": float(self.spawn_time),
            "entry_time": None if self.entry_time is None else float(self.entry_time),
            "planned_nominal_landing_time": (
                None
                if self.planned_nominal_landing_time is None
                else float(self.planned_nominal_landing_time)
            ),
            "actual_landing_time": (
                None
                if self.actual_landing_time is None
                else float(self.actual_landing_time)
            ),
            "landing_delay": (
                None if self.landing_delay is None else float(self.landing_delay)
            ),
            "path_length": float(self.path_length_total),
            "path_detour_rate": float(self.path_detour_rate(center)),
            "waiting_ring_distance": float(self.waiting_ring_distance),
            "cumulative_turn_angle_deg": float(math.degrees(self.cumulative_turn_angle_rad)),
            "cumulative_height_change": float(self.cumulative_altitude_change),
            "residual_action_l1": float(self.residual_action_l1_total),
        }

    def update_dynamics(
        self,
        desired_heading: float,
        desired_speed: float,
        desired_altitude: float,
        dt: float,
        max_turn_rate_deg: float,
        max_climb_rate: float,
        max_descent_rate: float,
        speed_min: float,
        speed_max: float,
        max_acceleration: float = 2.0,
    ) -> dict:
        old_position = self.position.copy()
        max_turn = math.radians(max_turn_rate_deg) * dt
        heading_error = wrap_angle(desired_heading - self.heading)
        heading_step = clamp(heading_error, -max_turn, max_turn)
        self.heading = wrap_angle(self.heading + heading_step)

        speed_error = desired_speed - self.speed
        max_accel_step = max(0.0, float(max_acceleration)) * dt
        self.speed += clamp(speed_error, -max_accel_step, max_accel_step)
        self.speed = clamp(self.speed, speed_min, speed_max)

        altitude_error = desired_altitude - self.position[2]
        if altitude_error >= 0.0:
            dz = min(altitude_error, max_climb_rate * dt)
        else:
            dz = max(altitude_error, -max_descent_rate * dt)

        horizontal_speed = self.speed * math.cos(self.pitch)
        self.position[0] += horizontal_speed * math.cos(self.heading) * dt
        self.position[1] += horizontal_speed * math.sin(self.heading) * dt
        self.position[2] += dz

        if abs(horizontal_speed) > 1e-6:
            self.pitch = math.atan2(dz, horizontal_speed * dt)
        else:
            self.pitch = 0.0

        path_distance = float(np.linalg.norm(self.position - old_position))
        return {
            "heading_step_rad": float(heading_step),
            "dz": float(dz),
            "path_distance": path_distance,
        }
