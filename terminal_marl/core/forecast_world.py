"""Lightweight, non-observing world used only by Full Teacher future repair."""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from .config import RingRole
from .lifecycle import transition
from .merge_scheduler import MergeReservation
from .trajectory_generator import TrajectoryGenerator
from .safety_filter import SafetyFilter
from .uav import FlightPhase


class ForecastWorld:
    """Minimal deterministic projection of active airborne traffic.

    Unlike ``TerminalParallelEnv`` this class owns neither admission,
    observations, rewards, runtime telemetry, history nor any rendering state.
    It advances only the already-active traffic required to determine whether
    a delayed Teacher repair has a canonical, SafetyFilter-approved witness.
    """

    def __init__(self, env) -> None:
        self.config = env.config
        self.now = float(env.now)
        self.airspace = deepcopy(env.airspace)
        self.active = deepcopy(env.active)
        self.scheduler = deepcopy(env.scheduler)
        self.lz_manager = deepcopy(env.lz_manager)
        self.generator = TrajectoryGenerator(self.config)
        self.safety = SafetyFilter(self.config)
        for uav in self.active.values():
            if uav.committed_plan is not None:
                self.safety.checked_trajectory_hashes[uav.committed_plan.plan_id] = uav.committed_plan.trajectory_hash

    def fork(self) -> "ForecastWorld":
        other = object.__new__(ForecastWorld)
        other.config = self.config
        other.now = self.now
        other.airspace = deepcopy(self.airspace)
        other.active = deepcopy(self.active)
        other.scheduler = deepcopy(self.scheduler)
        other.lz_manager = deepcopy(self.lz_manager)
        other.generator = self.generator
        other.safety = SafetyFilter(self.config)
        for uav in other.active.values():
            if uav.committed_plan is not None:
                other.safety.checked_trajectory_hashes[uav.committed_plan.plan_id] = uav.committed_plan.trajectory_hash
        return other

    def _finish_plan(self, uav, plan) -> None:
        if uav.merge_reservation_id is not None:
            self.scheduler.consume(uav.merge_reservation_id)
            uav.merge_reservation_id = None
        uav.committed_plan = None
        uav.trajectory_plan_id = None
        target = plan.metadata["target_phase"]
        phases = {"ring_outer": (FlightPhase.RING_OUTER, RingRole.OUTER),
                  "ring_middle": (FlightPhase.RING_MIDDLE, RingRole.MIDDLE),
                  "ring_inner": (FlightPhase.RING_INNER, RingRole.INNER)}
        if target not in phases:
            # A landing is not a candidate source in this narrow forecast.
            self.active.pop(uav.uid, None)
            return
        phase, role = phases[target]
        transition(uav, phase, self.now)
        uav.ring_role = role
        uav.ring_progress = math.atan2(uav.position[1], uav.position[0]) % (2.0 * math.pi)
        ring = self.airspace.actual_ring(role)
        tangent = np.asarray((-math.sin(uav.ring_progress), math.cos(uav.ring_progress), 0.0), dtype=float)
        radial = np.asarray((-math.cos(uav.ring_progress), -math.sin(uav.ring_progress), 0.0), dtype=float)
        uav.velocity = tangent * uav.speed
        uav.acceleration = radial * (uav.speed * uav.speed / ring.radius)

    def _advance_one(self, uav, subject_uid: int, subject_speed_residual: float) -> None:
        plan = uav.committed_plan
        if plan is not None:
            moment = min(self.now + self.config.dt, plan.end_time)
            point = plan.sample(moment)
            uav.position, uav.velocity, uav.acceleration = point.position, point.velocity, point.acceleration
            uav.speed = float(np.linalg.norm(point.velocity[:2]))
            if self.now + self.config.dt + 1e-9 >= plan.end_time:
                self._finish_plan(uav, plan)
            return
        if uav.ring_role is None:
            return
        ring = self.airspace.actual_ring(uav.ring_role)
        residual = subject_speed_residual if uav.uid == subject_uid else 0.0
        target_speed = float(np.clip(ring.nominal_speed + self.airspace.speed_profile_bias + residual * self.config.max_speed_residual_mps,
                                     self.config.speed_min, self.config.max_speed))
        theta = uav.ring_progress if uav.ring_progress is not None else math.atan2(uav.position[1], uav.position[0])
        speed, elapsed = max(0.0, uav.speed), 0.0
        while elapsed < self.config.dt - 1e-9:
            dt = min(self.config.physics_dt, self.config.dt - elapsed)
            motion = self.generator.dynamics.ring_motion_step(ring, theta, speed, target_speed, dt)
            theta, speed, elapsed = motion.theta, motion.speed, elapsed + dt
            uav.position, uav.velocity, uav.acceleration = motion.position, motion.velocity, motion.acceleration
        uav.speed, uav.ring_progress = speed, theta % (2.0 * math.pi)

    def advance(self, subject_uid: int, speed_residual: float, duration: float) -> None:
        for _ in range(int(round(duration / self.config.dt))):
            self.airspace.step_transition(self.now)
            for uav in list(self.active.values()):
                if uav.uid in self.active:
                    self._advance_one(uav, subject_uid, speed_residual)
            self.now += self.config.dt

    def candidate_plan(self, uav_id: int, candidate, target, *, lz_id: int | None):
        """Return the exact future-state candidate and its canonical verdict."""
        uav = self.active.get(uav_id)
        if uav is None or uav.committed_plan is not None or uav.ring_role is None:
            return None, None, "forecast_source_unavailable"
        from terminal_marl.envs.parallel_env import WorldSnapshot
        snapshot = WorldSnapshot(self.now, self.active, self.airspace)
        proposal = self.scheduler.propose(uav, candidate.intent, snapshot, target)
        if not self.scheduler.can_reserve(proposal):
            return None, None, "headway_conflict"
        temporary = MergeReservation("forecast-candidate", uav.uid, uav.phase.value, str(getattr(target, "value", target)),
                                     proposal.merge_time, proposal.merge_angle, None, None, status="future",
                                     expires_at=proposal.merge_time + 60.0)
        residual = candidate.intent.speed_residual * self.config.max_speed_residual_mps
        if target == "final":
            if lz_id is None:
                lz_id = self.lz_manager.choose_rule_zone(uav, self.now)
            if lz_id is None:
                return None, None, "lz_unavailable"
            plan = self.generator.final_approach(uav, self.lz_manager.zones[lz_id], temporary, self.now,
                                                 speed_residual_mps=residual, airspace=self.airspace)
        else:
            plan = self.generator.ring_transfer(uav, uav.ring_role, target, temporary, self.airspace, self.now, residual)
        committed = {uid: other.committed_plan for uid, other in self.active.items() if other.committed_plan is not None}
        verdict = self.safety.filter_plan(plan, committed, snapshot)
        return proposal, plan, None if verdict.approved else verdict.reason
