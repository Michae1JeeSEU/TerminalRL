"""Deterministic runtime protection for holders around committed trajectories."""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
import numpy as np

from .runtime_separation import swept_minimum_distance
from .trajectory_generator import TrajectoryPlan, TrajectoryPoint
from .action_trace import LowLevelActionTrace
from .vehicle_dynamics import VehicleDynamics


class PredictionCache:
    """Per-shield-step immutable prediction reuse boundary."""

    def __init__(self) -> None:
        self.holder_positions: dict[tuple, tuple[np.ndarray, ...]] = {}
        self.plan_positions: dict[tuple, tuple[np.ndarray, ...]] = {}
        self.repair_plans: dict[tuple, TrajectoryPlan] = {}

    @staticmethod
    def _time_key(times: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(times, dtype=float), 8))


@dataclass(frozen=True)
class RiskAssessment:
    predicted_min_swept_separation: float
    d_cpa: float
    t_cpa: float
    conflict_pair: tuple[int, int] | None
    trigger_reason: str | None


@dataclass(frozen=True)
class ShieldResult:
    intent: object
    used: bool
    reason: str | None = None
    correction_magnitude: float = 0.0


class PredictiveSafetyShield:
    """Pre-breach minimum shield; committed plans are immutable.

    The shield predicts a swept separation over a short horizon and changes
    only an *uncommitted* ring holder.  It is intentionally not a scheduler:
    it has no merge/release candidate grid and fails closed when a bounded,
    verifier-approved correction does not exist.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.dynamics = VehicleDynamics(config.vehicle_profile)
        self.overrides: dict[int, tuple[float, float]] = {}
        self.telemetry = {"shield_interventions": 0, "shield_correction_magnitude": 0.0,
                          "runtime_traffic_interventions": 0, "runtime_traffic_repair_magnitude": 0.0,
                          "shield_fail_closed": 0, "shield_broad_phase_rejected": 0,
                          "shield_narrow_phase_checked": 0}

    def assess(self, env) -> RiskAssessment:
        best: tuple[float, float, float, tuple[int, int]] | None = None
        committed = [uav.committed_plan for uav in env.active.values() if uav.committed_plan is not None]
        for holder in env.active.values():
            if holder.ring_role is None or holder.committed_plan is not None:
                continue
            for plan in committed:
                if plan.uav_id == holder.uid or plan.end_time <= env.now:
                    continue
                start = max(env.now, plan.start_time)
                minimum = self._minimum_to_plan(holder, plan, env.airspace, env.now, 0.0)
                holder_start = self._holder_position(holder, env.airspace, start - env.now, 0.0)
                relative = plan.sample(start).position - holder_start
                next_time = min(plan.end_time, start + env.config.dt)
                holder_next = self._holder_position(holder, env.airspace, next_time - env.now, 0.0)
                velocity = (plan.sample(next_time).position - plan.sample(start).position) / max(next_time - start, 1e-9)
                velocity -= (holder_next - holder_start) / max(next_time - start, 1e-9)
                relative_speed_sq = float(np.dot(velocity, velocity))
                t_cpa = 0.0 if relative_speed_sq <= 1e-9 else float(np.clip(-np.dot(relative, velocity) / relative_speed_sq, 0.0, self.config.shield_horizon_seconds))
                d_cpa = float(np.linalg.norm(relative + velocity * t_cpa))
                row = (minimum, d_cpa, t_cpa, (holder.uid, plan.uav_id))
                if best is None or row < best:
                    best = row
        if best is None:
            return RiskAssessment(float("inf"), float("inf"), 0.0, None, None)
        minimum, d_cpa, t_cpa, pair = best
        return RiskAssessment(minimum, d_cpa, t_cpa, pair,
                              "predicted_swept_separation" if minimum < self.config.safe_distance else None)

    def _holder_position(self, uav, airspace, now: float, residual: float) -> np.ndarray:
        ring = airspace.actual_ring(uav.ring_role)
        theta = uav.ring_progress if uav.ring_progress is not None else math.atan2(uav.position[1], uav.position[0])
        target_speed = float(np.clip(ring.nominal_speed + airspace.speed_profile_bias + residual, self.config.speed_min, self.config.max_speed))
        remaining, speed = max(0., now), max(0.0, uav.speed)
        while remaining > 1e-9:
            step = min(self.config.physics_dt, remaining)
            motion = self.dynamics.ring_motion_step(ring, theta, speed, target_speed, step)
            theta, speed, remaining = motion.theta, motion.speed, remaining - step
        return np.asarray((ring.radius * math.cos(theta), ring.radius * math.sin(theta), ring.height), dtype=float)

    def _holder_positions(self, uav, airspace, origin: float, times: np.ndarray, residual: float) -> list[np.ndarray]:
        """Return ordered executor-equivalent holder positions without replays.

        ``_minimum_to_plan`` evaluates adjacent swept intervals.  Replaying a
        holder from the current state for both ends of every interval made its
        30-second prediction quadratic in physics integrations, once for each
        residual option and protected holder/plan pair.  This advances the
        identical discrete integrator exactly once per ordered interval.
        """
        if len(times) == 0:
            return []
        if np.any(np.diff(times) < -1e-9):
            raise ValueError("holder prediction times must be ordered")
        ring = airspace.actual_ring(uav.ring_role)
        theta = uav.ring_progress if uav.ring_progress is not None else math.atan2(uav.position[1], uav.position[0])
        speed = max(0.0, uav.speed)
        target_speed = float(np.clip(ring.nominal_speed + airspace.speed_profile_bias + residual, self.config.speed_min, self.config.max_speed))
        previous = float(origin); positions = []
        for value in times:
            remaining = max(0.0, float(value) - previous)
            while remaining > 1e-9:
                step = min(self.config.physics_dt, remaining)
                motion = self.dynamics.ring_motion_step(ring, theta, speed, target_speed, step)
                theta, speed, remaining = motion.theta, motion.speed, remaining - step
            positions.append(np.asarray((ring.radius * math.cos(theta), ring.radius * math.sin(theta), ring.height), dtype=float))
            previous = max(previous, float(value))
        return positions

    def _minimum_to_plan(self, holder, plan, airspace, now: float, residual: float, cache: PredictionCache | None = None) -> float:
        start = max(now, plan.start_time); horizon = min(plan.end_time, now + 30.0)
        times = np.arange(start, horizon + 1e-9, self.config.safety_sampling_dt)
        if len(times) < 2: return float("inf")
        minimum = float("inf")
        time_key = PredictionCache._time_key(times)
        holder_key = (holder.uid, round(float(residual), 8), round(float(now), 8), time_key)
        plan_key = (plan.plan_id, plan.trajectory_hash, time_key)
        if cache is not None and holder_key in cache.holder_positions:
            holder_positions = cache.holder_positions[holder_key]
        else:
            holder_positions = tuple(self._holder_positions(holder, airspace, now, times, residual))
            if cache is not None: cache.holder_positions[holder_key] = holder_positions
        if cache is not None and plan_key in cache.plan_positions:
            plan_positions = cache.plan_positions[plan_key]
        else:
            plan_positions = tuple(plan.sample(float(time)).position for time in times)
            if cache is not None: cache.plan_positions[plan_key] = plan_positions
        for index in range(len(times) - 1):
            distance, _ = swept_minimum_distance(holder_positions[index], holder_positions[index + 1],
                                                  plan_positions[index], plan_positions[index + 1])
            minimum = min(minimum, distance)
        return minimum

    def _broad_phase_reject(self, holder, plan, airspace, now: float) -> bool:
        """Reject only pairs with a geometric proof of no shield interaction.

        CPA is intentionally calculated as a cheap diagnostic/ranking value;
        it never by itself approves a pair because curved trajectories could
        invalidate a constant-velocity assumption.  The actual rejection is
        limited to a conservative vertical envelope that cannot intersect the
        holder's constant ring altitude even under linear plan interpolation.
        """
        start, end = max(now, plan.start_time), min(plan.end_time, now + self.config.shield_horizon_seconds)
        if end <= start:
            return True
        ring_height = float(airspace.actual_ring(holder.ring_role).height)
        samples = plan.points
        vertical_clearance = min(
            0.0 if min(left.position[2], right.position[2]) <= ring_height <= max(left.position[2], right.position[2])
            else min(abs(left.position[2] - ring_height), abs(right.position[2] - ring_height))
            for left, right in zip(samples, samples[1:])
            if right.t >= start and left.t <= end
        )
        return vertical_clearance > self.config.safe_distance + self.config.shield_broad_phase_margin

    def _repair_plan(self, holder, airspace, now: float, end: float, residual: float) -> TrajectoryPlan:
        """Create the canonical bounded ring-speed prediction for a repair."""
        times = np.arange(now, end + 1e-9, self.config.physics_dt)
        if times[-1] < end:
            times = np.append(times, end)
        ring = airspace.actual_ring(holder.ring_role)
        theta = holder.ring_progress if holder.ring_progress is not None else math.atan2(holder.position[1], holder.position[0])
        speed = max(0.0, holder.speed)
        tangent = np.asarray((-math.sin(theta), math.cos(theta), 0.0), dtype=float)
        radial = np.asarray((-math.cos(theta), -math.sin(theta), 0.0), dtype=float)
        points = [TrajectoryPoint(float(now), np.asarray(holder.position, dtype=float), tangent * speed,
                                  radial * (speed * speed / ring.radius))]
        target_speed = float(np.clip(ring.nominal_speed + airspace.speed_profile_bias + residual, self.config.speed_min, self.config.max_speed))
        for left, right in zip(times, times[1:]):
            motion = self.dynamics.ring_motion_step(ring, theta, speed, target_speed, float(right - left))
            theta, speed = motion.theta, motion.speed
            points.append(TrajectoryPoint(float(right), motion.position, motion.velocity, motion.acceleration))
        return TrajectoryPlan(
            f"runtime-holder-repair-{uuid.uuid4()}", holder.uid, "runtime_holder_repair", points,
            now, end, metadata={"target_phase": "ring_holder", "committed": False},
        )

    def _passes_safety_filter(self, env, holder, plan, residual: float, cache: PredictionCache) -> bool:
        """SafetyFilter remains the final arbiter for a speed repair."""
        start = max(env.now, plan.start_time)
        end = min(plan.end_time, env.now + 30.0)
        if end <= start:
            return False
        def holder_plan_for(other, speed_residual: float) -> TrajectoryPlan:
            key = (other.uid, round(float(speed_residual), 8), round(start, 8), round(end, 8))
            if key not in cache.repair_plans:
                cache.repair_plans[key] = self._repair_plan(other, env.airspace, start, end, speed_residual)
            return cache.repair_plans[key]

        holder_plan = holder_plan_for(holder, residual)
        committed = TrajectoryPlan(
            plan.plan_id, plan.uav_id, plan.mode, plan.points, plan.start_time, plan.end_time,
            reservation_ids=list(plan.reservation_ids), metadata={**plan.metadata, "committed": True},
        )
        # Only pairs involving the changed holder require a new proof.  The
        # committed plan and unrelated holders retain their already-approved
        # canonical trajectories.  Same-ring relative motion is bounded by
        # the 2 m/s residual authority, so an arc-distance envelope safely
        # culls neighbours that cannot approach within the shield horizon.
        tracks = {holder.uid: holder_plan, plan.uav_id: committed}
        ring = env.airspace.actual_ring(holder.ring_role)
        holder_theta = holder.ring_progress if holder.ring_progress is not None else math.atan2(holder.position[1], holder.position[0])
        reachable_gap = env.config.safe_distance + 2.0 * env.config.max_speed_residual_mps * (end - start) + env.config.shield_broad_phase_margin
        for other in env.active.values():
            if other.uid in tracks or other.ring_role is None or other.committed_plan is not None:
                continue
            if other.ring_role is not holder.ring_role:
                continue
            other_theta = other.ring_progress if other.ring_progress is not None else math.atan2(other.position[1], other.position[0])
            gap = abs((other_theta - holder_theta + math.pi) % (2.0 * math.pi) - math.pi) * ring.radius
            if gap <= reachable_gap:
                tracks[other.uid] = holder_plan_for(other, 0.0)
        return env.safety.verify_predicted_tracks(tracks, start=start, end=end).approved

    def protect(self, env, intents: dict[int, object], traces: dict[int, LowLevelActionTrace] | None = None) -> dict[int, object]:
        self.overrides = {uid: value for uid, value in self.overrides.items() if value[1] > env.now and uid in env.active}
        committed = [uav.committed_plan for uav in env.active.values() if uav.committed_plan is not None]
        cache = PredictionCache()
        for holder in env.active.values():
            if holder.ring_role is None or holder.committed_plan is not None:
                continue
            for plan in committed:
                if plan.uav_id == holder.uid or plan.end_time <= env.now:
                    continue
                if self._broad_phase_reject(holder, plan, env.airspace, env.now):
                    self.telemetry["shield_broad_phase_rejected"] += 1
                    continue
                # Shift only an uncommitted holder.  Candidate plans are
                # sampled using the same swept-distance semantics as runtime.
                options = [self.config.max_speed_residual_mps, -self.config.max_speed_residual_mps]
                scored = [(self._minimum_to_plan(holder, plan, env.airspace, env.now, option, cache), option) for option in options]
                best_distance, best = max(scored)
                nominal = self._minimum_to_plan(holder, plan, env.airspace, env.now, 0.0, cache)
                self.telemetry["shield_narrow_phase_checked"] += 1
                repaired = best_distance >= self.config.safe_distance and self._passes_safety_filter(env, holder, plan, best, cache)
                if nominal < self.config.safe_distance and repaired:
                    self.overrides[holder.uid] = (best, min(plan.end_time, env.now + 30.0))
                    self.telemetry["shield_interventions"] += 1
                    self.telemetry["shield_correction_magnitude"] += abs(best)
                    self.telemetry["runtime_traffic_interventions"] += 1
                    self.telemetry["runtime_traffic_repair_magnitude"] += abs(best)
                elif nominal < self.config.safe_distance:
                    # The shield does not rewrite a committed plan or invent
                    # hover behaviour.  Record the fail-closed condition so a
                    # protected run cannot mistake it for a safe no-op.
                    self.telemetry["shield_fail_closed"] += 1
        repaired = dict(intents)
        for uid, (residual, _) in self.overrides.items():
            if uid in intents:
                intent = intents[uid]
                shield_intent = type(intent)(intent.release_request, float(np.clip(residual / self.config.max_speed_residual_mps, -1., 1.)), intent.merge_time_residual)
                repaired[uid] = shield_intent
                if traces is not None and uid in traces:
                    trace = traces[uid]
                    trace.shield_applied = shield_intent
                    trace.shield_used = True
                    trace.shield_reason = "predicted_swept_separation"
                    trace.correction_magnitude += abs(shield_intent.speed_residual - trace.execution_applied.speed_residual)
        return repaired


# Compatibility alias for archived A6 diagnostics.  New code should import
# ``PredictiveSafetyShield`` so its deployment role is explicit.
RuntimeTrafficSafetyController = PredictiveSafetyShield
