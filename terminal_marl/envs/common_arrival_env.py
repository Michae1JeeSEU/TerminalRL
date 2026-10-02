"""Opt-in free-flight bridge with shared V3 trajectory/safety/service execution.

There are NO mandatory rings. A controller may choose a circular reference
waypoint, but any certified trajectory inside the common disk is permitted.
The three-ring training environment and its qualification are not changed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import SimpleNamespace

import numpy as np

from ..core.common_arrival import CommonArrivalConfig, validate_requests, wrap_angle
from ..core.config import SimConfig
from ..core.energy import EnergyModel
from ..core.lifecycle import OutcomeLedger
from ..core.runtime_separation import RuntimeSeparationMonitor
from ..core.safety_filter import SafetyFilter
from ..core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from ..core.uav import FlightPhase, TerminalOutcome


@dataclass
class BridgeAircraft:
    request: object
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    heading: float = 0.0
    yaw_rate: float = 0.0
    phase: str = "arrival"
    admission_time: float | None = None
    capture_time: float | None = None
    touchdown_time: float | None = None
    completion_time: float | None = None
    outcome_time: float | None = None
    energy_consumed: float = 0.0
    path_length: float = 0.0
    plan: object = None
    outcome: str | None = None
    reason: str | None = None

    @property
    def uid(self):
        return self.request.uid

    @property
    def speed(self):
        return float(np.linalg.norm(self.velocity[:2]))


class CommonArrivalEnv:
    def __init__(self, config: CommonArrivalConfig, requests):
        validate_requests(requests, config)
        self.config, self.requests = config, tuple(requests)
        # Reuse the main project's V3 certifier, dynamics and separation
        # monitor. No exception bypass or relaxed safety constants are used.
        self.execution_config = SimConfig(
            trajectory_schema_version="executable_trajectory_v3",
            high_level_enabled=False, task_priority_enabled=False,
            dt=config.dt, physics_dt=config.physics_dt,
            safety_sampling_dt=config.physics_dt,
            safe_distance=config.safety_distance, nmac_distance=config.nmac_distance,
            collision_distance=config.collision_distance, energy_model_mode="phase_motion_v2",
        )
        self.safety = SafetyFilter(self.execution_config)
        self.monitor = RuntimeSeparationMonitor(self.execution_config)
        self.energy = EnergyModel(self.execution_config)
        self.ledger = OutcomeLedger()
        self.now = 0.0
        self.aircraft = {}
        self.queue = []
        self.next_request = 0
        self.pad_uid = None
        self.grant_uid = None
        self.grant_not_before = 0.0
        self.planned_touchdown = None
        self.status = "running"
        self.stop_reason = None
        self.events = []
        self.action_rows = []
        self.tick_rows = []
        self.trajectory_rows = []
        self.intervention_steps = 0
        self.requested_steps = 0
        self.resource_violations = 0
        self.pad_reserved_seconds = 0.0
        self.pad_service_seconds = 0.0
        self.integrated_airborne = 0.0
        self.integrated_queue = 0.0
        self.max_airborne = 0
        self.max_queue = 0
        self._plan_sequence = 0
        for r in self.requests:
            self.ledger.request(r.uid)

    @property
    def flying(self):
        return {uid: a for uid, a in self.aircraft.items() if a.phase in {"arrival", "landing"}}

    @property
    def arrivals(self):
        return {uid: a for uid, a in self.aircraft.items() if a.phase == "arrival"}

    def event(self, kind, uid=None, **details):
        self.events.append({"time": self.now, "event": kind, "uid": uid, **details})

    def available_to_controller(self):
        """Only already-admitted current aircraft; never the future tape."""
        return self.arrivals

    def set_grant(self, uid, *, not_before=None, planned_touchdown=None):
        if self.grant_uid is not None or self.pad_uid is not None:
            return
        if uid not in self.arrivals:
            raise ValueError("grant must refer to a currently visible arrival")
        when = self.now if not_before is None else float(not_before)
        if not math.isfinite(when):
            raise ValueError("nonfinite grant time")
        self.grant_uid = uid
        self.grant_not_before = max(self.now, when)
        self.planned_touchdown = planned_touchdown
        self.event("grant", uid, not_before=self.grant_not_before,
                   planned_touchdown=planned_touchdown)

    def prepare(self):
        """Expose due requests and safely admit at their assigned gate."""
        while self.next_request < len(self.requests) and self.requests[self.next_request].request_time <= self.now + 1e-9:
            r = self.requests[self.next_request]
            angle = (math.pi / 2, 0.0, -math.pi / 2, math.pi)[r.gate]
            position = np.array([self.config.radius * math.cos(angle),
                                 self.config.radius * math.sin(angle), self.config.altitude])
            self.aircraft[r.uid] = BridgeAircraft(r, position, np.zeros(3), phase="queued")
            self.queue.append(r.uid)
            self.next_request += 1
            self.event("request", r.uid, gate=r.gate)
        for uid in list(self.queue):
            a = self.aircraft[uid]
            if self.soc(a) <= self.execution_config.min_landing_soc:
                self._resolve(a, TerminalOutcome.DIVERTED_LOW_SOC, "external_energy_floor")
                self.queue.remove(uid)
                continue
            # A fixed, policy-independent admission buffer includes room to
            # brake. Blocked requests stay at their gate in the external queue.
            margin = self.config.safety_distance + self.config.cruise_speed ** 2 / 1.5
            if any(np.linalg.norm(a.position - b.position) < margin for b in self.flying.values()):
                continue
            a.heading = math.atan2(-a.position[1], -a.position[0])
            a.velocity = a.request.speed * np.array([math.cos(a.heading), math.sin(a.heading), 0.0])
            a.phase, a.admission_time = "arrival", self.now
            self.queue.remove(uid)
            self.event("admission", uid, external_wait=self.now - a.request.request_time)

    def soc(self, a):
        return max(0.0, a.request.initial_soc - a.energy_consumed / self.config.energy_capacity)

    def reference_goal(self, a):
        """Soft navigation target, NOT an environmental ring constraint."""
        if a.uid == self.grant_uid and self.pad_uid is None:
            eta = max(0.0, np.linalg.norm(a.position[:2]) - self.config.capture_radius) / self.config.cruise_speed
            if self.now + eta >= self.grant_not_before - 1e-8:
                return np.zeros(2)
        angle = math.atan2(a.position[1], a.position[0]) + math.pi / 3
        return self.config.holding_radius * np.array([math.cos(angle), math.sin(angle)])

    def guidance_turn(self, a, *, repel=False):
        target = self.reference_goal(a)
        vector = target - a.position[:2]
        if repel:
            norm = np.linalg.norm(vector)
            vector = vector / max(norm, 1e-8)
            for b in self.flying.values():
                if b.uid == a.uid:
                    continue
                difference = a.position[:2] - b.position[:2]
                distance = np.linalg.norm(difference)
                if distance < 260 and abs(a.position[2] - b.position[2]) < self.config.safety_distance:
                    vector += difference / max(distance, 1e-8) * 4 * (1 - distance / 260)
        desired = math.atan2(vector[1], vector[0])
        return float(np.clip(wrap_angle(desired - a.heading),
                             -math.radians(self.config.max_turn_deg_s),
                             math.radians(self.config.max_turn_deg_s)))

    def _plan(self, a, end, duration, mode):
        self._plan_sequence += 1
        start = TrajectoryPoint(self.now, a.position, a.velocity, a.acceleration, a.heading, a.yaw_rate)
        return TrajectoryPlan(
            f"bridge-{a.uid}-{self._plan_sequence}", a.uid, mode, (start, end), self.now, self.now + duration,
            metadata={"trajectory_schema_version": "executable_trajectory_v3",
                      "benchmark": self.config.schema},
        )

    def motion_plan(self, a, turn, *, braking=False):
        duration = self.config.prediction_horizon
        if braking:
            duration = max(duration, a.speed * 2.2 / 1.5)
            # Quintic interpolation reaches rest; the exact curve is checked.
            end_position = a.position + a.velocity * duration / 2
            end = TrajectoryPoint(self.now + duration, end_position, np.zeros(3), np.zeros(3), a.heading, 0.0)
            return self._plan(a, end, duration, "bridge_brake")
        omega = float(np.clip(turn, -math.radians(self.config.max_turn_deg_s), math.radians(self.config.max_turn_deg_s)))
        v0 = a.speed
        tangential = float(np.clip((a.request.speed - v0) / duration, -0.6, 0.6))
        times = np.linspace(0.0, duration, 41)
        headings = a.heading + omega * times
        speeds = v0 + tangential * times
        velocities = np.column_stack((speeds * np.cos(headings), speeds * np.sin(headings), np.zeros(len(times))))
        integrate = getattr(np, "trapezoid", None) or np.trapz
        displacement = integrate(velocities, times, axis=0)
        h, v = headings[-1], speeds[-1]
        acceleration = np.array([tangential * math.cos(h) - v * omega * math.sin(h),
                                 tangential * math.sin(h) + v * omega * math.cos(h), 0.0])
        end = TrajectoryPoint(self.now + duration, a.position + displacement,
                              velocities[-1], acceleration, h, omega)
        return self._plan(a, end, duration, "bridge_flight")

    def landing_plan(self, a):
        distance = float(np.linalg.norm(a.position[:2]))
        base = max(40.0, 2 * distance / max(a.speed, 8.0), 2 * a.position[2] / 5.0)
        for multiplier in (1.0, 1.25, 1.6, 2.0, 2.5):
            duration = math.ceil(base * multiplier)
            end = TrajectoryPoint(self.now + duration, np.zeros(3), np.zeros(3), np.zeros(3), a.heading, 0.0)
            plan = self._plan(a, end, duration, "bridge_landing")
            if not self.safety.dynamics.validate_trajectory(plan):
                return plan
        return None

    def _boundary_ok(self, a, plan, end):
        times = np.arange(self.now, end + 1e-8, self.config.physics_dt)
        if times[-1] < end:
            times = np.append(times, end)
        positions = plan.sample_positions(times)
        radii = np.linalg.norm(positions[:, :2], axis=1)
        if np.max(radii) > self.config.radius + 1e-5:
            return False, "outer_boundary"
        if a.phase == "arrival" and a.uid != self.grant_uid and np.min(radii) < self.config.capture_radius:
            return False, "ungranted_capture_entry"
        if a.phase == "arrival" and a.uid == self.grant_uid and self.now < self.grant_not_before:
            early = times < self.grant_not_before - 1e-8
            if np.any(radii[early] < self.config.capture_radius):
                return False, "early_capture_entry"
        return True, "approved"

    def _verify(self, tracks):
        self.safety.begin_prediction_step()
        snapshot = SimpleNamespace(active={}, now=self.now, airspace=None)
        for uid, plan in tracks.items():
            a = self.aircraft[uid]
            end = min(plan.end_time, self.now + self.config.prediction_horizon)
            okay, reason = self._boundary_ok(a, plan, end)
            if not okay:
                return False, reason
            # A previous immutable certificate covers this plan's dynamics.
            # Rechecking its *past* against newly admitted aircraft would
            # compare states that never coexisted. Check future pairs below.
            if (plan.start_time < self.now - 1e-8
                    and self.safety.checked_trajectory_hashes.get(plan.plan_id) == plan.trajectory_hash
                    and plan.verify_trajectory_integrity()):
                continue
            result = self.safety.filter_plan(plan, tracks, snapshot)
            if not result.approved:
                return False, result.reason
            if result.checked_plan_id != plan.plan_id or result.checked_trajectory_hash != plan.trajectory_hash:
                raise RuntimeError("safety certificate does not bind the executed plan")
        if tracks:
            result = self.safety.verify_predicted_tracks(
                tracks, start=self.now, end=self.now + self.config.prediction_horizon)
            if not result.approved:
                return False, result.reason
        return True, "approved"

    def _resolve(self, a, outcome, reason):
        self.ledger.resolve(a.uid, outcome)
        a.outcome, a.reason = outcome.value, reason
        a.outcome_time = self.now
        a.phase = "completed" if outcome is TerminalOutcome.COMPLETED else "failed"
        if outcome is TerminalOutcome.COMPLETED:
            a.completion_time = self.now
        self.event("outcome", a.uid, outcome=a.outcome, reason=reason)
        if self.grant_uid == a.uid:
            self.grant_uid = None
        if self.pad_uid == a.uid:
            self.pad_uid = None

    def _try_capture(self):
        if self.pad_uid is not None or self.grant_uid not in self.arrivals or self.now < self.grant_not_before:
            return
        a = self.aircraft[self.grant_uid]
        if np.linalg.norm(a.position[:2]) > self.config.capture_radius:
            return
        plan = self.landing_plan(a)
        if plan is None:
            self.event("landing_rejected", a.uid, reason="dynamics")
            return
        # The existing plans forecast every other airborne vehicle.
        tracks = {uid: b.plan for uid, b in self.flying.items() if uid != a.uid and b.plan is not None}
        old_phase = a.phase
        a.phase = "landing"
        okay, reason = self._verify({**tracks, a.uid: plan})
        a.phase = old_phase
        if not okay:
            self.event("landing_rejected", a.uid, reason=reason)
            return
        required = self.energy.estimate_plan_energy(plan) + self.execution_config.min_landing_soc * self.config.energy_capacity
        remaining = a.request.initial_soc * self.config.energy_capacity - a.energy_consumed
        if remaining < required:
            self._resolve(a, TerminalOutcome.FAILED_ENERGY, "landing_energy_insufficient")
            return
        a.plan, a.phase, a.capture_time = plan, "landing", self.now
        self.pad_uid = a.uid
        self.event("capture", a.uid, expected_touchdown=plan.end_time,
                   planned_touchdown=self.planned_touchdown,
                   planned_touchdown_error=None if self.planned_touchdown is None else plan.end_time - self.planned_touchdown)

    def step(self, turns):
        if self.status != "running":
            raise RuntimeError("episode already stopped")
        self._try_capture()
        flying = self.flying
        requested = {uid: float(turns.get(uid, 0.0)) for uid in self.arrivals}
        if not all(math.isfinite(v) for v in requested.values()):
            raise ValueError("nonfinite controller action")
        self.requested_steps += bool(self.arrivals)
        tracks = {uid: a.plan for uid, a in flying.items() if a.phase == "landing"}
        tracks.update({uid: self.motion_plan(a, requested[uid]) for uid, a in self.arrivals.items()})
        okay, reason = self._verify(tracks)
        selection = "requested"
        initial_reason = reason
        if not okay:
            self.intervention_steps += 1
            # Common, observable fallback: bounded vector-field avoidance,
            # then braking, then the already certified remaining prefix.
            for scale in (1.0, 0.6, 0.25):
                alternative = {uid: a.plan for uid, a in flying.items() if a.phase == "landing"}
                alternative.update({uid: self.motion_plan(a, scale * self.guidance_turn(a, repel=True))
                                    for uid, a in self.arrivals.items()})
                okay, reason = self._verify(alternative)
                if okay:
                    tracks, selection = alternative, "shared_avoidance"
                    break
            if not okay:
                alternative = {uid: (a.plan if a.phase == "landing" else self.motion_plan(a, 0.0, braking=True))
                               for uid, a in flying.items()}
                okay, reason = self._verify(alternative)
                if okay:
                    tracks, selection = alternative, "shared_brake"
            if not okay and all(a.plan is not None and a.plan.end_time >= self.now + self.config.dt for a in flying.values()):
                # This prefix was certified jointly in the previous tick.
                alternative = {uid: a.plan for uid, a in flying.items()}
                okay, reason = self._verify(alternative)
                if okay:
                    tracks, selection = alternative, "certified_continuation"
            self.event("shield", reason=initial_reason, selection=selection if okay else "fail_closed")
        if not okay:
            self.status, self.stop_reason = "fail_closed", reason
            self.event("stop", reason=reason)
            return
        for uid, a in flying.items():
            a.plan = tracks[uid]
            self.action_rows.append({"time": self.now, "uid": uid,
                                     "requested_turn_rad_s": requested.get(uid),
                                     "selection": selection, "plan_id": a.plan.plan_id,
                                     "trajectory_hash": a.plan.trajectory_hash,
                                     "phase": a.phase, "grant_uid": self.grant_uid})
        self.integrated_airborne += len(flying) * self.config.dt
        self.integrated_queue += len(self.queue) * self.config.dt
        self.max_airborne = max(self.max_airborne, len(flying))
        self.max_queue = max(self.max_queue, len(self.queue))
        if self.pad_uid is not None:
            self.pad_reserved_seconds += self.config.dt
            if self.aircraft[self.pad_uid].phase == "service":
                self.pad_service_seconds += self.config.dt
        substeps = int(round(self.config.dt / self.config.physics_dt))
        for _ in range(substeps):
            previous = {uid: a.position.copy() for uid, a in self.flying.items()}
            subtime = self.now + self.config.physics_dt
            for uid, a in list(self.flying.items()):
                point = a.plan.sample(subtime)
                a.path_length += float(np.linalg.norm(point.position - a.position))
                a.position, a.velocity, a.acceleration = point.position.copy(), point.velocity.copy(), point.acceleration.copy()
                a.heading = math.atan2(a.velocity[1], a.velocity[0]) if a.speed > 1e-6 else a.heading
                a.yaw_rate = 0.0 if point.yaw_rate is None else point.yaw_rate
                a.energy_consumed += self.energy.motion_power(a.velocity) * self.config.physics_dt
            self.now = subtime
            encounters = self.monitor.observe(self.flying, previous)
            if encounters:
                self.event("separation", pairs=[{"pair": e.pair, "distance": e.swept_min_distance} for e in encounters])
            if self.monitor.telemetry["hard_separation_violations"]:
                self.status, self.stop_reason = "invalid_safety", "protected_floor_violation"
                return
            for a in list(self.flying.values()):
                if self.soc(a) < self.execution_config.min_landing_soc:
                    self._resolve(a, TerminalOutcome.FAILED_ENERGY, "airborne_energy_floor")
                elif a.phase == "landing" and self.now >= a.plan.end_time - 1e-8:
                    a.phase, a.touchdown_time = "service", self.now
                    self.event("touchdown", a.uid)
            for a in self.aircraft.values():
                if a.phase == "service" and self.now >= a.touchdown_time + a.request.service_seconds - 1e-8:
                    self._resolve(a, TerminalOutcome.COMPLETED, "service_complete")
            occupants = [a.uid for a in self.aircraft.values() if a.phase in {"landing", "service"}]
            if len(occupants) > 1 or (occupants and occupants[0] != self.pad_uid):
                self.resource_violations += 1
                self.status, self.stop_reason = "invalid_safety", "pad_resource_invariant"
                return
            for uid in self.queue:
                self.aircraft[uid].energy_consumed += self.energy.phase_power(FlightPhase.EXTERNAL_QUEUE) * self.config.physics_dt
        self.now = round(self.now, 9)
        self.tick_rows.append({"time": self.now, "airborne": len(self.flying), "queue": len(self.queue),
                               "service": int(self.pad_uid is not None and self.aircraft[self.pad_uid].phase == "service"),
                               "completed": sum(v is TerminalOutcome.COMPLETED for v in self.ledger.outcomes.values()),
                               "shield_steps": self.intervention_steps})
        if math.isclose(self.now / self.config.trace_period, round(self.now / self.config.trace_period)):
            for a in self.aircraft.values():
                if a.phase in {"arrival", "landing", "service"}:
                    self.trajectory_rows.append({"time": self.now, "uid": a.uid, "phase": a.phase,
                                                 "x": a.position[0], "y": a.position[1], "z": a.position[2],
                                                 "speed": a.speed, "soc": self.soc(a)})
        if self.ledger.closes():
            self.status, self.stop_reason = "complete", "all_requests_resolved"
        elif self.now >= self.config.duration - 1e-8:
            self.status, self.stop_reason = "truncated", "horizon"
