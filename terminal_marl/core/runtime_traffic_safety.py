"""Deterministic runtime protection for holders around committed trajectories."""
from __future__ import annotations

import math
import uuid
import numpy as np

from .runtime_separation import swept_minimum_distance
from .trajectory_generator import TrajectoryPlan, TrajectoryPoint


class RuntimeTrafficSafetyController:
    """Use only bounded holder-speed authority; never rewrite committed plans."""

    def __init__(self, config) -> None:
        self.config = config
        self.overrides: dict[int, tuple[float, float]] = {}
        self.telemetry = {"runtime_traffic_interventions": 0, "runtime_traffic_repair_magnitude": 0.0}

    def _holder_position(self, uav, airspace, now: float, residual: float) -> np.ndarray:
        ring = airspace.actual_ring(uav.ring_role)
        theta = uav.ring_progress if uav.ring_progress is not None else math.atan2(uav.position[1], uav.position[0])
        speed = float(np.clip(ring.nominal_speed + airspace.speed_profile_bias + residual, self.config.speed_min, self.config.max_speed))
        theta += speed * max(0., now) / ring.radius
        return np.asarray((ring.radius * math.cos(theta), ring.radius * math.sin(theta), ring.height))

    def _minimum_to_plan(self, holder, plan, airspace, now: float, residual: float) -> float:
        start = max(now, plan.start_time); horizon = min(plan.end_time, now + 30.0)
        times = np.arange(start, horizon + 1e-9, self.config.dt)
        if len(times) < 2: return float("inf")
        minimum = float("inf")
        for left, right in zip(times, times[1:]):
            holder_left = self._holder_position(holder, airspace, left - now, residual)
            holder_right = self._holder_position(holder, airspace, right - now, residual)
            distance, _ = swept_minimum_distance(holder_left, holder_right, plan.sample(float(left)).position, plan.sample(float(right)).position)
            minimum = min(minimum, distance)
        return minimum

    def _repair_plan(self, holder, airspace, now: float, end: float, residual: float) -> TrajectoryPlan:
        """Create the canonical bounded ring-speed prediction for a repair."""
        times = np.arange(now, end + 1e-9, self.config.dt)
        if times[-1] < end:
            times = np.append(times, end)
        points = [TrajectoryPoint(float(time), self._holder_position(holder, airspace, time - now, residual)) for time in times]
        return TrajectoryPlan(
            f"runtime-holder-repair-{uuid.uuid4()}", holder.uid, "runtime_holder_repair", points,
            now, end, metadata={"target_phase": "ring_holder", "committed": False},
        )

    def _passes_safety_filter(self, env, holder, plan, residual: float) -> bool:
        """SafetyFilter remains the final arbiter for a speed repair."""
        start = max(env.now, plan.start_time)
        end = min(plan.end_time, env.now + 30.0)
        if end <= start:
            return False
        holder_plan = self._repair_plan(holder, env.airspace, start, end, residual)
        committed = TrajectoryPlan(
            plan.plan_id, plan.uav_id, plan.mode, plan.points, plan.start_time, plan.end_time,
            reservation_ids=list(plan.reservation_ids), metadata={**plan.metadata, "committed": True},
        )
        # A speed correction that clears a committed transfer but closes into
        # a neighbouring holder is not a repair.  Include all relevant
        # uncommitted ring holders in the same predictive, swept check.  This
        # keeps the controller inside its canonical bounded-speed authority
        # and makes SafetyFilter the single final arbiter.
        tracks = {holder.uid: holder_plan, plan.uav_id: committed}
        for other in env.active.values():
            if other.uid in tracks or other.ring_role is None or other.committed_plan is not None:
                continue
            tracks[other.uid] = self._repair_plan(other, env.airspace, start, end, 0.0)
        return env.safety.verify_predicted_tracks(tracks, start=start, end=end).approved

    def protect(self, env, intents: dict[int, object]) -> dict[int, object]:
        self.overrides = {uid: value for uid, value in self.overrides.items() if value[1] > env.now and uid in env.active}
        committed = [uav.committed_plan for uav in env.active.values() if uav.committed_plan is not None]
        for holder in env.active.values():
            if holder.ring_role is None or holder.committed_plan is not None:
                continue
            for plan in committed:
                if plan.uav_id == holder.uid or plan.end_time <= env.now:
                    continue
                # Shift only an uncommitted holder.  Candidate plans are
                # sampled using the same swept-distance semantics as runtime.
                options = [self.config.max_speed_residual_mps, -self.config.max_speed_residual_mps]
                scored = [(self._minimum_to_plan(holder, plan, env.airspace, env.now, option), option) for option in options]
                best_distance, best = max(scored)
                nominal = self._minimum_to_plan(holder, plan, env.airspace, env.now, 0.0)
                if (nominal < self.config.safe_distance and best_distance >= self.config.safe_distance
                        and self._passes_safety_filter(env, holder, plan, best)):
                    self.overrides[holder.uid] = (best, min(plan.end_time, env.now + 30.0))
                    self.telemetry["runtime_traffic_interventions"] += 1
                    self.telemetry["runtime_traffic_repair_magnitude"] += abs(best)
        repaired = dict(intents)
        for uid, (residual, _) in self.overrides.items():
            if uid in intents:
                intent = intents[uid]
                repaired[uid] = type(intent)(intent.release_request, float(np.clip(residual / self.config.max_speed_residual_mps, -1., 1.)), intent.merge_time_residual)
        return repaired
