"""Traffic-aware safety shield for canonical dynamic three-ring reconfiguration."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
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
    # Telemetry semantics are deliberately separate from legacy coordinator
    # control outcomes.  They describe what actually became effective and
    # do not alter the safety or simulation path.
    execution_outcome: str = "NO_OP"
    request_id: str | None = None
    original_request_time: float | None = None
    actual_application_time: float | None = None
    application_delay_seconds: float | None = None
    operational_effective: bool = False
    structural_effective: bool = False
    capacity_effective: bool = False


class PendingState(str, Enum):
    IDLE = "idle"
    PENDING = "pending"
    WAITING_FOR_EVENT = "waiting_for_event"
    REPAIRING = "repairing"
    READY = "ready"
    EXECUTING = "executing"
    DONE = "done"


class ReconfigurationSafetyCoordinator:
    """Repairs traffic first; defer/project/reject only when repair cannot work."""

    deadband = 0.05

    def __init__(self, env) -> None:
        self.env = env
        self.pending_action: np.ndarray | None = None
        self.pending_until: float | None = None
        self.pending_request_id: str | None = None
        self.pending_request_time: float | None = None
        self._request_sequence = 0
        self.pending_state = PendingState.IDLE
        self._last_retry_signature: tuple | None = None
        self.speed_overrides: dict[int, tuple[float, float]] = {}
        self.history: list[dict] = []
        self._candidate_airspace = None
        self._candidate_audit: dict = {}
        self.telemetry = {"traffic_repairs": 0, "traffic_repair_magnitude": 0.0, "geometry_projection_magnitude": 0.0,
                          "geometry_delays": 0, "geometry_defers": 0, "geometry_rejects": 0,
                          "coordinator_attempts": 0, "pending_retries": 0, "effective_updates": 0}

    def _committed_incoming(self):
        return dict(self.env.airspace._committed)

    def _protected_horizon(self, airspace) -> float:
        """The complete protected window, never a local 120 s truncation."""
        now = self.env.now
        transition_end = airspace.transition.end_time if airspace.transition is not None else now
        committed_end = max((uav.committed_plan.end_time for uav in self.env.active.values()
                             if uav.committed_plan is not None), default=now)
        # This buffer protects the canonical transfer -> target-ring holder
        # handoff; it is intentionally independent of the policy action.
        post_handoff_end = committed_end + max(30.0, self.env.config.geometry_transition_seconds)
        return max(now + self.env.config.dt, transition_end, post_handoff_end)

    def _geometry_timeline(self, airspace, times: np.ndarray) -> list[object]:
        """Actual R(t), H(t) samples with the exact transition implementation."""
        result = []
        for moment in times:
            geometry = deepcopy(airspace)
            geometry.step_transition(float(moment))
            result.append(geometry)
        return result

    def _holder_track(self, uid, uav, plan, times, geometries, speed_repairs):
        """Runtime-equivalent holder motion, including transfer handoff.

        Runtime updates geometry at the beginning of an interval and then
        advances ring progress on that actual reference.  This discrete
        construction deliberately matches that execution order rather than
        using a static-radius closed form.
        """
        now = self.env.now
        target_names = {"ring_outer": RingRole.OUTER, "ring_middle": RingRole.MIDDLE, "ring_inner": RingRole.INNER}
        role = uav.ring_role
        theta = uav.ring_progress if uav.ring_progress is not None else math.atan2(uav.position[1], uav.position[0])
        handoff_index = None
        active_end = float(times[-1])
        if plan is not None:
            target = target_names.get(plan.metadata.get("target_phase"))
            if target is not None:
                # ``TerminalParallelEnv._advance_plan`` hands a plan off on
                # the first simulation tick at or after its exact endpoint.
                # Predict that same discrete executor event, rather than
                # treating a fractional plan endpoint as an instantaneous
                # holder transition.
                handoff_time = now + math.ceil((plan.end_time - now) / self.env.config.dt - 1e-9) * self.env.config.dt
                handoff_index = next((index for index, value in enumerate(times) if value >= handoff_time - 1e-9), None)
            else:
                # Final/LZ plans remove the aircraft from the low-level
                # airborne set on their executor handoff; it has no phantom
                # static continuation after touchdown.
                active_end = min(float(times[-1]), now + math.ceil((plan.end_time - now) / self.env.config.dt - 1e-9) * self.env.config.dt)
        points = [TrajectoryPoint(float(times[0]), np.asarray(uav.position, dtype=float))]
        for index in range(1, len(times)):
            previous, moment = float(times[index - 1]), float(times[index])
            if plan is not None and handoff_index is not None and index <= handoff_index:
                position = plan.sample(min(moment, plan.end_time)).position
                if index == handoff_index:
                    role = target_names[plan.metadata["target_phase"]]
                    theta = math.atan2(position[1], position[0])
            elif plan is not None and handoff_index is None:
                position = plan.sample(min(moment, plan.end_time)).position
            elif role is not None:
                # Use the geometry that runtime has installed for this exact
                # motion interval.  For the first step it is the transition
                # start state; subsequent samples follow actual R/H updates.
                geometry = geometries[index - 1]
                ring = geometry.actual_ring(role)
                residual = 0.0 if plan is not None else speed_repairs.get(uid, 0.0)
                speed = float(np.clip(ring.nominal_speed + geometry.speed_profile_bias + residual,
                                      self.env.config.speed_min, self.env.config.max_speed))
                theta = (theta + speed * (moment - previous) / ring.radius) % (2.0 * math.pi)
                position = np.asarray((ring.radius * math.cos(theta), ring.radius * math.sin(theta), ring.height))
            else:
                # Every airborne state must have a canonical plan or ring
                # semantics.  A missing protected trajectory is fail-closed.
                raise RuntimeError(f"missing protected trajectory for active UAV {uid}")
            points.append(TrajectoryPoint(moment, np.asarray(position, dtype=float)))
        return TrajectoryPlan(f"reconfig-{uuid.uuid4()}", uid, "reconfiguration_prediction", points,
                              now, float(times[-1]), metadata={"target_phase": "prediction", "committed": plan is not None,
                                                                 "active_end": active_end, "source_plan_id": None if plan is None else plan.plan_id})

    def _tracks(self, airspace, speed_repairs: dict[int, float], end: float) -> dict[int, TrajectoryPlan]:
        now = self.env.now
        times = np.arange(now, end + 1e-9, self.env.config.dt)
        if len(times) == 0 or times[-1] < end:
            times = np.append(times, end)
        geometries = self._geometry_timeline(airspace, times)
        return {uid: self._holder_track(uid, uav, uav.committed_plan, times, geometries, speed_repairs)
                for uid, uav in self.env.active.items()}

    def _candidate_safety(self, action: np.ndarray, repairs: dict[int, float]):
        self._candidate_airspace = None
        self._candidate_audit = {}
        trial = deepcopy(self.env.airspace)
        result = trial.request_update(action, now=self.env.now, occupancies=self.env.airspace._occupancy,
                                      committed_incoming=self._committed_incoming(), geometry_gate=True)
        horizon = self._protected_horizon(trial)
        try:
            tracks = self._tracks(trial, repairs, horizon)
        except RuntimeError as error:
            self._candidate_airspace = None
            return result, type("Verdict", (), {"approved": False, "reason": f"missing_protected_traffic:{error}"})()
        verdict = self.env.safety.verify_predicted_tracks(tracks, start=self.env.now, end=horizon)
        self._candidate_airspace = trial
        prediction = dict(self.env.safety.last_predicted_track_audit)
        self._candidate_audit = {"protected_window": (self.env.now, horizon), "track_count": len(tracks),
                                 "transition_start": None if trial.transition is None else trial.transition.start_time,
                                 "transition_end": None if trial.transition is None else trial.transition.end_time,
                                 "prediction": prediction}
        return result, verdict

    def _apply_operational_if_safe(self, raw: np.ndarray, *, reason: str, deferred: float | None = None):
        """Apply speed/headway controls only after their own prospective check.

        Headways are prospective and therefore preserve committed reservations;
        speed-profile changes alter moving holders and must pass the full
        protected-window verifier.  A failed operational check is fail-closed
        instead of leaking a directly-applied speed change around the shield.
        """
        operational = raw.copy(); operational[:6] = 0.0
        if not np.any(np.abs(operational[6:]) > 1e-9):
            update = self.env.apply_high_level_action(operational, geometry_gate=False)
            return operational, update, None, {}, 0.0
        update, verdict = self._candidate_safety(operational, {})
        if verdict.approved:
            return operational, self._atomic_commit(operational, update), None, {}, 0.0
        for repair in self._repair_options():
            update, repaired = self._candidate_safety(operational, repair)
            if repaired.approved:
                committed = self._atomic_commit(operational, update)
                until = self.env.now + self.env.config.geometry_transition_seconds
                for uid, residual in repair.items(): self.speed_overrides[uid] = (residual, until)
                return operational, committed, None, repair, sum(abs(v) for v in repair.values())
        # Do not apply an unsafe speed/headway compound.  The all-zero update
        # is still recorded, making the rejection visible to high-level PPO.
        zero = np.zeros(10, dtype=np.float32)
        return zero, self.env.apply_high_level_action(zero, geometry_gate=False), verdict.reason, {}, 0.0

    def _atomic_commit(self, action, fallback_result):
        """Commit exactly the staged timeline which SafetyFilter approved."""
        if self._candidate_airspace is None:
            # Test doubles may replace candidate safety; production never
            # takes this path.  Preserve existing isolated coordinator tests.
            return self.env.apply_high_level_action(action, geometry_gate=True)
        return self.env.commit_high_level_candidate(self._candidate_airspace, fallback_result)

    def _repair_options(self):
        maximum = self.env.config.max_speed_residual_mps
        for uid, uav in self.env.active.items():
            if uav.ring_role is not None and uav.committed_plan is None:
                yield {uid: maximum}
                yield {uid: -maximum}

    def _traffic_signature(self) -> tuple:
        return tuple(sorted((uid, uav.phase.value, uav.ring_role.value if uav.ring_role else None,
                             uav.trajectory_plan_id, uav.merge_reservation_id) for uid, uav in self.env.active.items()))

    def _defer_until(self) -> float:
        endpoints = [u.committed_plan.end_time for u in self.env.active.values() if u.committed_plan is not None]
        return max(self.env.now + self.env.config.dt, max(endpoints, default=self.env.now) + self.env.config.dt)

    def _clear_pending(self, state: PendingState = PendingState.DONE) -> None:
        self.pending_action = None
        self.pending_until = None
        self.pending_request_id = None
        self.pending_request_time = None
        self._last_retry_signature = None
        self.pending_state = state

    def coordinate(self, requested, *, geometry_gate: bool, allow_delay: bool = True,
                   request_id: str | None = None, original_request_time: float | None = None) -> ReconfigurationDecision:
        self.telemetry["coordinator_attempts"] += 1
        if request_id is None:
            self._request_sequence += 1
            request_id = f"high-{self._request_sequence}"
        if original_request_time is None:
            original_request_time = self.env.now
        control_before = self.env.airspace.state_vector().copy()
        raw = np.asarray(requested, dtype=np.float32)
        if raw.shape != (10,) or not np.all(np.isfinite(raw)):
            raise ValueError("high-level action must be finite shape (10,)")
        structural = raw[:6].copy(); structural[np.abs(structural) < self.deadband] = 0.0
        proposed = raw.copy(); proposed[:6] = structural
        if not geometry_gate or not np.any(structural):
            applied, update, unsafe, repairs, magnitude = self._apply_operational_if_safe(
                proposed, reason="event_gate_or_deadband")
            outcome = "no_op" if unsafe is None else "operational_rejected"
            reason = "event_gate_or_deadband" if unsafe is None else "unsafe_operational_action"
            return self._record(outcome, reason, raw, applied, repairs, magnitude, 0.0, None, unsafe, update,
                                control_before, request_id, original_request_time)
        update, verdict = self._candidate_safety(proposed, {})
        if verdict.approved:
            committed = self._atomic_commit(proposed, update)
            self._clear_pending()
            return self._record("approved", "safe_direct", raw, proposed, {}, 0.0, 0.0, None, None, committed,
                                control_before, request_id, original_request_time)
        self.pending_state = PendingState.REPAIRING
        repair_options = self._repair_options()
        for repair in repair_options:
            _, repaired = self._candidate_safety(proposed, repair)
            if repaired.approved:
                committed = self._atomic_commit(proposed, update)
                until = self.env.now + self.env.config.geometry_transition_seconds
                for uid, residual in repair.items(): self.speed_overrides[uid] = (residual, until)
                self._clear_pending()
                return self._record("approved_repaired", "holder_speed_repair", raw, proposed, repair, sum(abs(v) for v in repair.values()), 0.0, None, None, committed,
                                    control_before, request_id, original_request_time)
        if allow_delay and self.env.active:
            self.pending_action = proposed.copy(); self.pending_until = self._defer_until()
            self.pending_request_id = request_id; self.pending_request_time = original_request_time
            self.pending_state = PendingState.WAITING_FOR_EVENT
            self._last_retry_signature = self._traffic_signature()
            # Structural and operational controls are decoupled, but the
            # latter never bypasses predictive verification.
            applied, update, unsafe, repairs, magnitude = self._apply_operational_if_safe(proposed, reason="structural_delayed", deferred=self.pending_until)
            return self._record("delayed", "wait_committed_traffic", raw, applied, repairs, magnitude, 0.0, self.pending_until,
                                verdict.reason if unsafe is None else unsafe, update, control_before, request_id, original_request_time)
        for factor in (.5, .25, .10):
            projected = proposed.copy(); projected[:6] *= factor
            projected_update, projected_verdict = self._candidate_safety(projected, {})
            if projected_verdict.approved:
                committed = self._atomic_commit(projected, projected_update)
                magnitude = float(np.abs(proposed[:6]-projected[:6]).sum())
                self._clear_pending()
                return self._record("projected", "minimal_structural_projection", raw, projected, {}, 0.0, magnitude, None, verdict.reason, committed,
                                    control_before, request_id, original_request_time)
        headway_only, update, unsafe, repairs, magnitude = self._apply_operational_if_safe(proposed, reason="structural_rejected")
        self._clear_pending()
        return self._record("rejected", "no_safe_reconfiguration", raw, headway_only, repairs, magnitude, float(np.abs(proposed[:6]).sum()), None,
                            verdict.reason if unsafe is None else unsafe, update, control_before, request_id, original_request_time)

    def _record(self, outcome, reason, requested, applied, repairs, repair_magnitude, projection, deferred, safety_reason, update,
                control_before, request_id, original_request_time):
        control_after = self.env.airspace.state_vector()
        operational = bool(np.any(np.abs(control_after[6:] - control_before[6:]) > 1e-8))
        structural = bool(update.structural_applied)
        capacity = bool(tuple(update.capacity_before) != tuple(update.capacity_after) and structural)
        if structural:
            label = "PROJECTED_AND_APPLIED" if outcome == "projected" else ("STRUCTURAL_DELAYED_THEN_APPLIED" if self.env.now > original_request_time + 1e-9 else "STRUCTURAL_APPLIED")
        elif operational:
            label = "OPERATIONAL_APPLIED"
        elif outcome in {"rejected", "operational_rejected"}:
            label = "SAFETY_REJECTED"
        elif outcome == "delayed":
            label = "STRUCTURAL_DELAYED"
        else:
            label = "DEADBAND_NOOP" if reason == "event_gate_or_deadband" and np.any(np.abs(requested[:6]) > 1e-8) else "NO_OP"
        actual_time = self.env.now if structural or operational else None
        delay = None if actual_time is None else max(0., actual_time - original_request_time)
        decision = ReconfigurationDecision(outcome, reason, tuple(float(x) for x in requested), tuple(float(x) for x in applied), dict(repairs), len(repairs), float(repair_magnitude), float(projection), deferred, safety_reason, update,
                                           label, request_id, original_request_time, actual_time, delay, operational, structural, capacity)
        self.telemetry["traffic_repairs"] += len(repairs); self.telemetry["traffic_repair_magnitude"] += repair_magnitude; self.telemetry["geometry_projection_magnitude"] += projection
        if outcome == "delayed": self.telemetry["geometry_delays"] += 1
        if outcome == "rejected": self.telemetry["geometry_rejects"] += 1
        if update.structural_applied: self.telemetry["effective_updates"] += 1
        self.history.append({"outcome": outcome, "execution_outcome": label, "reason_code": reason, "requested_action": list(decision.requested_action), "applied_action": list(decision.applied_action), "traffic_repair": repairs, "projection_magnitude": projection, "deferred_until": deferred, "safety_reason": safety_reason, "request_id": request_id, "original_request_time": original_request_time, "actual_application_time": actual_time, "application_delay_seconds": delay, "operational_effective": operational, "structural_effective": structural, "capacity_effective": capacity, "candidate_audit": deepcopy(self._candidate_audit)})
        return decision

    def advance(self):
        self.speed_overrides = {uid: value for uid, value in self.speed_overrides.items() if value[1] > self.env.now and uid in self.env.active}
        if self.pending_action is not None and self.pending_until is not None and self.env.now >= self.pending_until:
            signature = self._traffic_signature()
            if signature == self._last_retry_signature:
                self.pending_state = PendingState.WAITING_FOR_EVENT
                return None
            action = self.pending_action; request_id = self.pending_request_id; request_time = self.pending_request_time
            self.pending_action = None; self.pending_until = None
            self.pending_request_id = None; self.pending_request_time = None
            self.pending_state = PendingState.READY; self.telemetry["pending_retries"] += 1
            return self.coordinate(action, geometry_gate=True, allow_delay=True, request_id=request_id, original_request_time=request_time)
        return None

    def apply_speed_repairs(self, actions):
        repaired = {key: np.asarray(value, dtype=np.float32).copy() for key, value in actions.items()}
        for uid, (residual, _) in self.speed_overrides.items():
            key = str(uid)
            if key in repaired and uid in self.env.active and self.env.active[uid].ring_role is not None:
                repaired[key][1] = float(np.clip(residual / self.env.config.max_speed_residual_mps, -1.0, 1.0))
        return repaired
