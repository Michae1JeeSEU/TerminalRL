"""Traffic-aware safety shield for canonical dynamic three-ring reconfiguration."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import uuid
import numpy as np

from .config import RingRole
from .trajectory_generator import TrajectoryPlan, TrajectoryPoint


@dataclass(frozen=True)
class ReconfigurationDecision:
    outcome: str
    reason_code: str
    requested_action: tuple[float, ...]
    applied_action: tuple[float, ...]
    repair_speed_residuals: dict[int, float]
    repair_count: int
    repair_magnitude: float
    projection_magnitude: float
    deferred_until: float | None
    safety_reason: str | None
    update: object


class ReconfigurationSafetyCoordinator:
    """Repairs traffic first; defer/project/reject only when repair cannot work."""

    deadband = 0.05

    def __init__(self, env) -> None:
        self.env = env
        self.pending_action: np.ndarray | None = None
        self.pending_until: float | None = None
        self.speed_overrides: dict[int, tuple[float, float]] = {}
        self.history: list[dict] = []
        self.telemetry = {"traffic_repairs": 0, "traffic_repair_magnitude": 0.0, "geometry_projection_magnitude": 0.0,
                          "geometry_delays": 0, "geometry_defers": 0, "geometry_rejects": 0, "hard_separation_violations": 0}

    def _committed_incoming(self):
        return dict(self.env.airspace._committed)

    def _tracks(self, airspace, speed_repairs: dict[int, float], end: float, changed_roles: set[RingRole]) -> dict[int, TrajectoryPlan]:
        now = self.env.now; points_by_uid: dict[int, list[TrajectoryPoint]] = {}
        times = np.arange(now, end + 1e-9, 1.0)
        for uid, uav in self.env.active.items():
            if uav.committed_plan is None and (uav.ring_role is None or uav.ring_role not in changed_roles):
                continue
            points: list[TrajectoryPoint] = []
            plan = uav.committed_plan
            for time in times:
                geometry = deepcopy(airspace); geometry.step_transition(float(time))
                if plan is not None and time <= plan.end_time:
                    position = plan.sample(float(time)).position
                elif plan is not None and plan.metadata.get("target_phase") in {"ring_outer", "ring_middle", "ring_inner"}:
                    role = RingRole(plan.metadata["target_phase"].replace("ring_", ""))
                    ring = geometry.actual_ring(role); theta = math.atan2(plan.sample(plan.end_time).position[1], plan.sample(plan.end_time).position[0])
                    speed = np.clip(ring.nominal_speed + geometry.speed_profile_bias + speed_repairs.get(uid, 0.0), self.env.config.speed_min, self.env.config.max_speed)
                    theta += speed * max(0.0, time-plan.end_time) / ring.radius
                    position = np.asarray((ring.radius*math.cos(theta), ring.radius*math.sin(theta), ring.height))
                elif uav.ring_role is not None:
                    ring = geometry.actual_ring(uav.ring_role); theta = uav.ring_progress if uav.ring_progress is not None else math.atan2(uav.position[1], uav.position[0])
                    speed = np.clip(ring.nominal_speed + geometry.speed_profile_bias + speed_repairs.get(uid, 0.0), self.env.config.speed_min, self.env.config.max_speed)
                    theta += speed * (time-now) / ring.radius
                    position = np.asarray((ring.radius*math.cos(theta), ring.radius*math.sin(theta), ring.height))
                else:
                    position = np.asarray(uav.position, dtype=float)
                points.append(TrajectoryPoint(float(time), np.asarray(position, dtype=float)))
            points_by_uid[uid] = (points, plan is not None)
        return {uid: TrajectoryPlan(f"reconfig-{uuid.uuid4()}", uid, "reconfiguration_prediction", points, now, end, metadata={"target_phase": "prediction", "committed": committed}) for uid, (points, committed) in points_by_uid.items()}

    def _candidate_safety(self, action: np.ndarray, repairs: dict[int, float]):
        trial = deepcopy(self.env.airspace)
        result = trial.request_update(action, now=self.env.now, occupancies=self.env.airspace._occupancy,
                                      committed_incoming=self._committed_incoming(), geometry_gate=True)
        horizon = self.env.now + max(self.env.config.geometry_transition_seconds, 1.0)
        for uav in self.env.active.values():
            if uav.committed_plan is not None:
                horizon = max(horizon, min(uav.committed_plan.end_time, self.env.now + 120.0))
        changed = {role for role in RingRole if abs(trial.target_ring(role).radius-self.env.airspace.actual_ring(role).radius)>1e-6 or abs(trial.target_ring(role).height-self.env.airspace.actual_ring(role).height)>1e-6}
        # Inner/middle structural movement shares capture/final interfaces.  A
        # committed trajectory in that region is protected by a deterministic
        # window first; it is never exposed to an optimistic interpolation.
        if self.env.active and not repairs:
            from .safety_filter import TrajectorySafetyResult
            return result, TrajectorySafetyResult(False, "affected_interface_committed_traffic")
        tracks = self._tracks(trial, repairs, horizon, changed)
        verdict = self.env.safety.verify_predicted_tracks(tracks, start=self.env.now, end=horizon)
        return result, verdict

    def _repair_options(self):
        maximum = self.env.config.max_speed_residual_mps
        for uid, uav in self.env.active.items():
            if uav.ring_role is not None and uav.committed_plan is None:
                yield {uid: maximum}
                yield {uid: -maximum}

    def _defer_until(self) -> float:
        endpoints = [u.committed_plan.end_time for u in self.env.active.values() if u.committed_plan is not None]
        return max(self.env.now + self.env.config.dt, max(endpoints, default=self.env.now) + self.env.config.dt)

    def coordinate(self, requested, *, geometry_gate: bool, allow_delay: bool = True) -> ReconfigurationDecision:
        raw = np.asarray(requested, dtype=np.float32)
        if raw.shape != (10,) or not np.all(np.isfinite(raw)):
            raise ValueError("high-level action must be finite shape (10,)")
        structural = raw[:6].copy(); structural[np.abs(structural) < self.deadband] = 0.0
        proposed = raw.copy(); proposed[:6] = structural
        if not geometry_gate or not np.any(structural):
            applied = proposed.copy(); applied[:6] = 0.0
            update = self.env.apply_high_level_action(applied, geometry_gate=False)
            return self._record("no_op", "event_gate_or_deadband", raw, applied, {}, 0.0, 0.0, None, None, update)
        update, verdict = self._candidate_safety(proposed, {})
        if verdict.approved:
            committed = self.env.apply_high_level_action(proposed, geometry_gate=True)
            return self._record("approved", "safe_direct", raw, proposed, {}, 0.0, 0.0, None, None, committed)
        # Only apply an automatic speed repair when the traffic set is already
        # quiescent enough to validate it end-to-end.  Busy traffic receives a
        # deterministic delay window instead of an optimistic partial repair.
        repair_options = self._repair_options() if not self.env.active else ()
        for repair in repair_options:
            _, repaired = self._candidate_safety(proposed, repair)
            if repaired.approved:
                committed = self.env.apply_high_level_action(proposed, geometry_gate=True)
                until = self.env.now + self.env.config.geometry_transition_seconds
                for uid, residual in repair.items(): self.speed_overrides[uid] = (residual, until)
                return self._record("approved_repaired", "holder_speed_repair", raw, proposed, repair, sum(abs(v) for v in repair.values()), 0.0, None, None, committed)
        if allow_delay and self.env.active:
            self.pending_action = proposed.copy(); self.pending_until = self._defer_until()
            # Do not change flow controls underneath the protected traffic
            # window.  They are atomically applied with the deferred target.
            headway_only = np.zeros(10, dtype=np.float32)
            update = self.env.apply_high_level_action(headway_only, geometry_gate=False)
            return self._record("delayed", "wait_committed_traffic", raw, headway_only, {}, 0.0, 0.0, self.pending_until, verdict.reason, update)
        for factor in (.5, .25, .10):
            projected = proposed.copy(); projected[:6] *= factor
            _, projected_verdict = self._candidate_safety(projected, {})
            if projected_verdict.approved:
                committed = self.env.apply_high_level_action(projected, geometry_gate=True)
                magnitude = float(np.abs(proposed[:6]-projected[:6]).sum())
                return self._record("projected", "minimal_structural_projection", raw, projected, {}, 0.0, magnitude, None, verdict.reason, committed)
        headway_only = proposed.copy(); headway_only[:6] = 0.0
        update = self.env.apply_high_level_action(headway_only, geometry_gate=False)
        return self._record("rejected", "no_safe_reconfiguration", raw, headway_only, {}, 0.0, float(np.abs(proposed[:6]).sum()), None, verdict.reason, update)

    def _record(self, outcome, reason, requested, applied, repairs, repair_magnitude, projection, deferred, safety_reason, update):
        decision = ReconfigurationDecision(outcome, reason, tuple(float(x) for x in requested), tuple(float(x) for x in applied), dict(repairs), len(repairs), float(repair_magnitude), float(projection), deferred, safety_reason, update)
        self.telemetry["traffic_repairs"] += len(repairs); self.telemetry["traffic_repair_magnitude"] += repair_magnitude; self.telemetry["geometry_projection_magnitude"] += projection
        if outcome == "delayed": self.telemetry["geometry_delays"] += 1
        if outcome == "rejected": self.telemetry["geometry_rejects"] += 1
        self.history.append({"outcome": outcome, "reason_code": reason, "requested_action": list(decision.requested_action), "applied_action": list(decision.applied_action), "traffic_repair": repairs, "projection_magnitude": projection, "deferred_until": deferred, "safety_reason": safety_reason})
        return decision

    def advance(self):
        self.speed_overrides = {uid: value for uid, value in self.speed_overrides.items() if value[1] > self.env.now and uid in self.env.active}
        if self.pending_action is not None and self.pending_until is not None and self.env.now >= self.pending_until:
            action = self.pending_action; self.pending_action = None; self.pending_until = None
            return self.coordinate(action, geometry_gate=True, allow_delay=True)
        return None

    def apply_speed_repairs(self, actions):
        repaired = {key: np.asarray(value, dtype=np.float32).copy() for key, value in actions.items()}
        for uid, (residual, _) in self.speed_overrides.items():
            key = str(uid)
            if key in repaired and uid in self.env.active and self.env.active[uid].ring_role is not None:
                repaired[key][1] = float(np.clip(residual / self.env.config.max_speed_residual_mps, -1.0, 1.0))
        return repaired
