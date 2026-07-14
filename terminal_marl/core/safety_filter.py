"""Basic, explainable safety filtering for airspace, intentions, and plans."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .dynamic_airspace import DynamicAirspace
from .trajectory_generator import TrajectoryPlan, TrajectoryPoint


class SafetyFilter:
    """Apply lightweight constraints before later MPC/CBF safety work.

    This phase deliberately implements deterministic clipping, rejection, and
    hover override behavior only.  It has no hardware or ROS dependencies.
    """

    def __init__(
        self,
        *,
        safe_distance: float,
        boundary: tuple[float, float, float, float, float, float],
        speed_max: float,
        acceleration_max: float = math.inf,
        min_segment_duration: float = 1e-3,
    ) -> None:
        if safe_distance <= 0.0:
            raise ValueError("safe_distance must be positive")
        if speed_max <= 0.0 or acceleration_max <= 0.0:
            raise ValueError("kinematic limits must be positive")
        if min_segment_duration <= 0.0:
            raise ValueError("min_segment_duration must be positive")
        if len(boundary) != 6:
            raise ValueError("boundary must be (xmin, xmax, ymin, ymax, zmin, zmax)")
        xmin, xmax, ymin, ymax, zmin, zmax = (float(value) for value in boundary)
        if not (xmin < xmax and ymin < ymax and zmin < zmax):
            raise ValueError("boundary limits must be strictly ordered")

        self.safe_distance = float(safe_distance)
        self.boundary = (xmin, xmax, ymin, ymax, zmin, zmax)
        self.speed_max = float(speed_max)
        self.acceleration_max = float(acceleration_max)
        self.min_segment_duration = float(min_segment_duration)

    def filter_airspace_action(
        self,
        action: np.ndarray,
        airspace: DynamicAirspace,
    ) -> tuple[np.ndarray, dict]:
        """Clip a high-level action and override it in fixed-airspace mode."""

        raw = np.asarray(action, dtype=np.float64)
        if raw.shape != (DynamicAirspace.ACTION_SIZE,):
            raise ValueError(f"airspace action must have shape ({DynamicAirspace.ACTION_SIZE},)")
        if not np.all(np.isfinite(raw)):
            raise ValueError("airspace action must contain finite values")

        filtered = np.clip(raw, -1.0, 1.0)
        violations: list[str] = []
        if not np.array_equal(filtered, raw):
            violations.append("action_bounds")
        if not airspace.dynamic_enabled and np.any(np.abs(filtered) > 0.0):
            filtered = np.zeros_like(filtered)
            violations.append("fixed_airspace_override")
        return filtered, {
            "approved": not violations,
            "modified": bool(violations),
            "violations": violations,
        }

    def filter_intention_action(
        self,
        action: np.ndarray,
        *,
        release_allowed: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """Clip ``[release, speed, slot, timing]`` and block unsafe release."""

        raw = np.asarray(action, dtype=np.float32)
        if raw.shape != (4,):
            raise ValueError("intention action must have shape (4,)")
        if not np.all(np.isfinite(raw)):
            raise ValueError("intention action must contain finite values")

        filtered = np.clip(raw, -1.0, 1.0).astype(np.float32)
        violations: list[str] = []
        if not np.array_equal(filtered, raw):
            violations.append("action_bounds")
        if not release_allowed and filtered[0] > -1.0:
            filtered[0] = -1.0
            violations.append("release_override")
        return filtered, {
            "approved": not violations,
            "modified": bool(violations),
            "violations": violations,
            "semantics": (
                "release_request",
                "speed_residual",
                "slot_residual",
                "timing_residual",
            ),
        }

    def filter_trajectory_plan(
        self,
        plan: TrajectoryPlan,
    ) -> tuple[TrajectoryPlan, dict]:
        """Approve a valid plan or replace it with a bounded stationary hover."""

        kinematic_detail = self.trajectory_kinematics_diagnostics(plan)
        checks = {
            "boundary": self.check_boundary(plan),
            "height": self.check_height_range(plan),
            "kinematics": self.check_speed_acceleration(plan),
            "kinematic_detail": kinematic_detail,
        }
        violations = [
            violation
            for name, result in checks.items()
            if name != "kinematic_detail"
            for violation in result
        ]
        violations = list(dict.fromkeys(violations))
        if not violations:
            return plan, {
                "approved": True,
                "is_safe": True,
                "modified": False,
                "violations": [],
                "reason": None,
                "diagnostics": checks,
                "override": None,
            }

        if not plan.points:
            return plan, {
                "approved": False,
                "is_safe": False,
                "modified": False,
                "violations": violations,
                "reason": violations[0],
                "diagnostics": checks,
                "override": None,
            }

        first = plan.points[0]
        safe_position = self._clip_position(np.array([first.x, first.y, first.z], dtype=float))
        start_time = float(first.t) if math.isfinite(first.t) else 0.0
        point_count = max(2, len(plan.points))
        hover_dt = max(self.min_segment_duration, 0.2)
        safe_points = [
            TrajectoryPoint(
                x=float(safe_position[0]),
                y=float(safe_position[1]),
                z=float(safe_position[2]),
                yaw=float(first.yaw) if math.isfinite(first.yaw) else 0.0,
                t=start_time + index * hover_dt,
            )
            for index in range(point_count)
        ]
        metadata = dict(plan.metadata)
        metadata.update(
            {
                "safety_override": "emergency_hover",
                "source_mode": plan.mode,
                "source_violations": violations,
            }
        )
        override = TrajectoryPlan(
            uav_id=plan.uav_id,
            mode="safety_override_hover",
            points=safe_points,
            metadata=metadata,
        )
        return override, {
            "approved": False,
            "is_safe": False,
            "modified": True,
            "violations": violations,
            "reason": violations[0],
            "diagnostics": checks,
            "override": "emergency_hover",
        }

    def approve_release(
        self,
        state: object,
        target_stage: str,
        states: Sequence[object],
        *,
        candidate_plan: TrajectoryPlan | None = None,
        trajectory_states: Sequence[object] | None = None,
        final_occupied: bool = False,
        landing_pad_occupied: bool = False,
    ) -> dict:
        """Approve a stage transition or override it to HOLD.

        ``target_stage`` is one of ``outer_to_inner``, ``inner_to_final``, or
        ``final_to_land``.  Approval never mutates an aircraft state.
        """

        supported = {"outer_to_inner", "inner_to_final", "final_to_land"}
        if target_stage not in supported:
            raise ValueError(f"unsupported target_stage: {target_stage}")

        state_id = self._state_id(state)
        pairwise = [
            violation
            for violation in self.check_pairwise_distance(states)
            if state_id in {violation["first"], violation["second"]}
        ]
        current_min_distance = self._minimum_distance_to_others(state, states)
        violations: list[str] = []
        if pairwise:
            violations.append("minimum_separation")
        if target_stage == "inner_to_final" and final_occupied:
            violations.append("final_approach_occupied")
        if target_stage == "final_to_land" and landing_pad_occupied:
            violations.append("landing_pad_occupied")

        filtered_plan = candidate_plan
        trajectory_info: dict | None = None
        trajectory_separation: list[dict] = []
        if candidate_plan is not None:
            separation_states = states if trajectory_states is None else trajectory_states
            filtered_plan, trajectory_info = self.filter_trajectory_plan(candidate_plan)
            if not trajectory_info["is_safe"]:
                violations.extend(
                    f"trajectory_{reason}" for reason in trajectory_info["violations"]
                )
            trajectory_separation = self.check_trajectory_separation(
                candidate_plan,
                separation_states,
                ignore_uav_id=state_id,
            )
            if trajectory_separation:
                violations.append("trajectory_minimum_separation")

        violations = list(dict.fromkeys(violations))
        approved = not violations
        return {
            "approved": approved,
            "is_safe": approved,
            "decision": "RELEASE" if approved else "HOLD",
            "target_stage": target_stage,
            "violations": violations,
            "reason": None if approved else violations[0],
            "filtered_plan": filtered_plan,
            "diagnostics": {
                "pairwise": pairwise,
                "current_min_distance": current_min_distance,
                "safe_distance": self.safe_distance,
                "trajectory": trajectory_info,
                "trajectory_separation": trajectory_separation,
                "trajectory_min_distance": (
                    None
                    if candidate_plan is None
                    else self._trajectory_minimum_distance(
                        candidate_plan,
                        states if trajectory_states is None else trajectory_states,
                        ignore_uav_id=state_id,
                    )
                ),
                "trajectory_state_count": (
                    None
                    if candidate_plan is None
                    else len(states if trajectory_states is None else trajectory_states)
                ),
                "final_occupied": bool(final_occupied),
                "landing_pad_occupied": bool(landing_pad_occupied),
            },
        }

    def check_pairwise_distance(self, states: Sequence[object]) -> list[dict]:
        """Return every current pair below the configured safe distance."""

        violations: list[dict] = []
        for i, first in enumerate(states):
            first_position = self._state_position(first)
            for second in states[i + 1 :]:
                second_position = self._state_position(second)
                distance = float(np.linalg.norm(first_position - second_position))
                if distance < self.safe_distance:
                    violations.append(
                        {
                            "first": self._state_id(first),
                            "second": self._state_id(second),
                            "distance": distance,
                            "minimum": self.safe_distance,
                        }
                    )
        return violations

    def check_boundary(self, plan: TrajectoryPlan) -> list[str]:
        """Check finite waypoints and the configured axis-aligned boundary."""

        if not plan.points:
            return ["empty_trajectory"]
        xmin, xmax, ymin, ymax, zmin, zmax = self.boundary
        for point in plan.points:
            values = (point.x, point.y, point.z, point.yaw, point.t)
            if not all(math.isfinite(value) for value in values):
                return ["non_finite_waypoint"]
            if not (
                xmin <= point.x <= xmax
                and ymin <= point.y <= ymax
                and zmin <= point.z <= zmax
            ):
                return ["boundary_violation"]
        return []

    def check_height_range(self, plan: TrajectoryPlan) -> list[str]:
        """Check waypoint altitude against the configured vertical range."""

        if not plan.points:
            return ["empty_trajectory"]
        zmin, zmax = self.boundary[4], self.boundary[5]
        if any(
            not math.isfinite(point.z) or point.z < zmin or point.z > zmax
            for point in plan.points
        ):
            return ["height_range"]
        return []

    def check_speed_acceleration(self, plan: TrajectoryPlan) -> list[str]:
        """Check strict time ordering, segment speed, and acceleration."""

        if not plan.points:
            return ["empty_trajectory"]
        if len(plan.points) == 1:
            return []
        positions = np.asarray(
            [[point.x, point.y, point.z] for point in plan.points],
            dtype=np.float64,
        )
        times = np.asarray([point.t for point in plan.points], dtype=np.float64)
        intervals = np.diff(times)
        if np.any(~np.isfinite(intervals)) or np.any(intervals <= 0.0):
            return ["non_monotonic_time"]

        velocities = np.diff(positions, axis=0) / intervals[:, None]
        violations: list[str] = []
        if np.any(intervals < self.min_segment_duration - 1e-12):
            violations.append("minimum_segment_duration")
        if np.any(np.linalg.norm(velocities, axis=1) > self.speed_max + 1e-9):
            violations.append("speed_limit")
        if len(velocities) > 1:
            acceleration_dt = 0.5 * (intervals[:-1] + intervals[1:])
            accelerations = np.diff(velocities, axis=0) / acceleration_dt[:, None]
            if np.any(
                np.linalg.norm(accelerations, axis=1) > self.acceleration_max + 1e-9
            ):
                violations.append("acceleration_limit")
        return violations

    def trajectory_kinematics_diagnostics(self, plan: TrajectoryPlan) -> dict:
        """Return numeric speed/acceleration diagnostics for explainable rejects."""

        diagnostics = {
            "point_count": len(plan.points),
            "speed_max_allowed": self.speed_max,
            "acceleration_max_allowed": self.acceleration_max,
            "min_segment_duration_allowed": self.min_segment_duration,
            "min_segment_duration_observed": None,
            "max_speed_observed": 0.0,
            "max_acceleration_observed": 0.0,
            "non_monotonic_time_indices": [],
            "minimum_segment_duration_segments": [],
            "speed_violation_segments": [],
            "acceleration_violation_segments": [],
        }
        if len(plan.points) <= 1:
            return diagnostics

        positions = np.asarray(
            [[point.x, point.y, point.z] for point in plan.points],
            dtype=np.float64,
        )
        times = np.asarray([point.t for point in plan.points], dtype=np.float64)
        intervals = np.diff(times)
        invalid = np.where(~np.isfinite(intervals) | (intervals <= 0.0))[0]
        diagnostics["non_monotonic_time_indices"] = [int(index) for index in invalid]
        if invalid.size:
            return diagnostics

        diagnostics["min_segment_duration_observed"] = float(np.min(intervals))
        short_segments = np.where(intervals < self.min_segment_duration - 1e-12)[0]
        diagnostics["minimum_segment_duration_segments"] = [
            int(index) for index in short_segments
        ]

        velocities = np.diff(positions, axis=0) / intervals[:, None]
        speeds = np.linalg.norm(velocities, axis=1)
        diagnostics["max_speed_observed"] = float(np.max(speeds)) if speeds.size else 0.0
        speed_segments = np.where(speeds > self.speed_max + 1e-9)[0]
        diagnostics["speed_violation_segments"] = [int(index) for index in speed_segments]

        if len(velocities) > 1:
            acceleration_dt = 0.5 * (intervals[:-1] + intervals[1:])
            accelerations = np.diff(velocities, axis=0) / acceleration_dt[:, None]
            acceleration_norms = np.linalg.norm(accelerations, axis=1)
            diagnostics["max_acceleration_observed"] = (
                float(np.max(acceleration_norms)) if acceleration_norms.size else 0.0
            )
            acceleration_segments = np.where(
                acceleration_norms > self.acceleration_max + 1e-9
            )[0]
            diagnostics["acceleration_violation_segments"] = [
                int(index) for index in acceleration_segments
            ]
        return diagnostics

    def check_trajectory_separation(
        self,
        plan: TrajectoryPlan,
        states: Sequence[object],
        *,
        ignore_uav_id: str | None = None,
    ) -> list[dict]:
        """Check a candidate path against other aircraft held at current state."""

        violations: list[dict] = []
        for state in states:
            state_id = self._state_id(state)
            if ignore_uav_id is not None and state_id == str(ignore_uav_id):
                continue
            position = self._state_position(state)
            distances = [
                float(
                    np.linalg.norm(
                        np.array([point.x, point.y, point.z], dtype=float) - position
                    )
                )
                for point in plan.points
            ]
            minimum = min(distances, default=math.inf)
            if minimum < self.safe_distance:
                violations.append(
                    {
                        "uav_id": state_id,
                        "distance": minimum,
                        "minimum": self.safe_distance,
                    }
                )
        return violations

    def _clip_position(self, position: np.ndarray) -> np.ndarray:
        xmin, xmax, ymin, ymax, zmin, zmax = self.boundary
        return np.clip(
            position,
            np.array([xmin, ymin, zmin], dtype=float),
            np.array([xmax, ymax, zmax], dtype=float),
        )

    def _minimum_distance_to_others(
        self,
        state: object,
        states: Sequence[object],
    ) -> float | None:
        state_id = self._state_id(state)
        position = self._state_position(state)
        distances = [
            float(np.linalg.norm(position - self._state_position(other)))
            for other in states
            if self._state_id(other) != state_id
        ]
        return float(min(distances)) if distances else None

    def _trajectory_minimum_distance(
        self,
        plan: TrajectoryPlan,
        states: Sequence[object],
        *,
        ignore_uav_id: str | None = None,
    ) -> float | None:
        distances: list[float] = []
        for state in states:
            state_id = self._state_id(state)
            if ignore_uav_id is not None and state_id == str(ignore_uav_id):
                continue
            position = self._state_position(state)
            for point in plan.points:
                distances.append(
                    float(
                        np.linalg.norm(
                            np.array([point.x, point.y, point.z], dtype=float)
                            - position
                        )
                    )
                )
        return float(min(distances)) if distances else None

    @staticmethod
    def _state_position(state: object) -> np.ndarray:
        if isinstance(state, Mapping):
            raw = state.get("position")
        else:
            raw = getattr(state, "position", None)
        position = np.asarray(raw, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("each state must expose a finite three-dimensional position")
        return position

    @staticmethod
    def _state_id(state: object) -> str:
        for name in ("uav_id", "uid", "id"):
            if isinstance(state, Mapping) and name in state:
                return str(state[name])
            if hasattr(state, name):
                return str(getattr(state, name))
        return "unknown"
