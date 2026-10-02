"""Deterministic runtime protection for holders around committed trajectories."""
from __future__ import annotations

from bisect import bisect_right
import itertools
import math
import uuid
from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np

from .trajectory_generator import TrajectoryPlan, TrajectoryPoint
from .action_trace import LowLevelActionTrace
from .vehicle_dynamics import VehicleDynamics


class PredictionCache:
    """Per-shield-step immutable prediction reuse boundary."""

    def __init__(self) -> None:
        self.holder_positions: dict[tuple, tuple[np.ndarray, ...]] = {}
        self.plan_positions: dict[tuple, tuple[np.ndarray, ...]] = {}
        self.repair_plans: dict[tuple, TrajectoryPlan] = {}
        self.plan_envelopes: dict[tuple, tuple[float, float, float, float]] = {}

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


@dataclass(frozen=True)
class JointSafetyAssessment:
    """Result of one deterministic, execution-equivalent joint check."""

    approved: bool
    minimum_separation: float
    unsafe_pairs: tuple[tuple[int, int], ...]


class SafetyInvariantError(RuntimeError):
    """Raised before physics when no certified bounded safety action exists.

    This is deliberately an execution barrier, rather than an instruction to
    use nominal (zero) residual.  Callers must treat it as a safety-regression
    failure and never advance the affected state through physics.
    """

    def __init__(self, message: str, *, diagnostic: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic or {}


class PredictiveSafetyShield:
    """Pre-breach minimum shield; committed plans are immutable.

    The shield predicts a swept separation over a short horizon and changes
    only an *uncommitted* ring holder.  It is intentionally not a scheduler:
    it has no merge/release candidate grid and fails closed when a bounded,
    verifier-approved correction does not exist.
    """

    def __init__(self, config) -> None:
        self.config = config
        self._v3_enabled = config.trajectory_schema_version == "executable_trajectory_v3"
        self.dynamics = VehicleDynamics(config.vehicle_profile, continuous_curve_limits=self._v3_enabled)
        self.overrides: dict[int, tuple[float, float]] = {}
        self.telemetry = {"shield_interventions": 0, "shield_correction_magnitude": 0.0,
                          "runtime_traffic_interventions": 0, "runtime_traffic_repair_magnitude": 0.0,
                          "shield_fail_closed": 0, "shield_broad_phase_rejected": 0,
                          "shield_narrow_phase_checked": 0,
                          "joint_safety_checks": 0, "joint_safety_corrections": 0,
                          "joint_safety_fail_closed": 0, "admissibility_rejections": 0,
                          "recursive_feasibility_checks": 0,
                          "recursive_feasibility_rejections": 0,
                          "joint_backup_uses": 0,
                          "joint_unrecoverable_components": 0,
                          "operational_buffer_takeovers": 0,
                          "joint_search_components": 0,
                          "joint_search_candidates": 0,
                          "joint_search_pruned": 0}
        self.unrecoverable_components: tuple[tuple[int, ...], ...] = ()
        self.last_certification_diagnostic: dict[str, object] = {}
        self.last_recursive_gate = "not_run"
        self._joint_track_cache: dict[tuple, np.ndarray] = {}
        self._joint_track_info: dict[int, dict[str, object]] = {}
        self._joint_assessment_times: np.ndarray | None = None
        # Speed integration for holders depends on the ring geometry, target
        # speed and per-step durations, but not on the holder's phase angle.
        # Keep this cache scoped to one protect tick; position tracks still
        # remain keyed by the holder snapshot and residual.
        self._joint_speed_sequence_cache: dict[
            tuple, tuple[tuple[float, float, float], ...]
        ] = {}
        self._joint_sampled_track_cache: dict[tuple, np.ndarray] = {}
        # Pair verdicts are scoped to one ``protect`` call.  The cache is
        # intentionally inactive for direct ``_joint_assessment`` calls so a
        # caller cannot accidentally reuse a result across policy ticks.
        self._joint_pair_verdict_cache: dict[tuple, tuple[bool, float]] | None = None
        self._joint_pair_snapshot_refs: dict[int, object] | None = None

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
                relative = plan.sample_position(start) - holder_start
                next_time = min(plan.end_time, start + env.config.dt)
                holder_next = self._holder_position(holder, env.airspace, next_time - env.now, 0.0)
                velocity = (plan.sample_position(next_time) - plan.sample_position(start)) / max(next_time - start, 1e-9)
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

    @property
    def _horizontal_acceleration_bound(self) -> float:
        profile = self.config.vehicle_profile
        return min(
            float(profile.max_horizontal_accel),
            float(profile.max_horizontal_decel),
            float(profile.max_lateral_accel),
        ) + 1.0e-3

    @property
    def _flight_acceleration_bound(self) -> float:
        profile = self.config.vehicle_profile
        vertical_bound = max(
            float(profile.max_vertical_accel_up),
            float(profile.max_vertical_accel_down),
        ) + 1.0e-3
        return math.sqrt(self._horizontal_acceleration_bound ** 2 + vertical_bound ** 2)

    @property
    def _holder_acceleration_bound(self) -> float:
        return self._horizontal_acceleration_bound

    def _track_descriptor(self, plan):
        if not self._v3_enabled:
            return "legacy"
        metadata = getattr(plan, "metadata", {})
        if not isinstance(metadata, dict):
            return None
        if metadata.get("runtime_executor_equivalent", False):
            return {"kind": "holder_proxy", "resampled": False}
        if metadata.get("trajectory_schema_version") == "executable_trajectory_v3":
            try:
                if bool(plan.verify_trajectory_integrity()):
                    return {"kind": "flight_v3", "resampled": False}
            except (AttributeError, TypeError, ValueError):
                return None
            return None
        return None

    def _track_error_budget(self, descriptor, dt: float) -> float | None:
        if not self._v3_enabled:
            return 0.0
        kind = descriptor.get("kind") if isinstance(descriptor, dict) else descriptor
        if kind == "flight_v3":
            return self._flight_acceleration_bound * float(dt) ** 2 / 8.0
        if kind == "holder_proxy":
            resampled = isinstance(descriptor, dict) and bool(descriptor.get("resampled", False))
            source_error = float(self.config.physics_dt) ** 2 if resampled else 0.0
            return self._holder_acceleration_bound * (float(dt) ** 2 + source_error) / 8.0
        if kind == "mixed":
            return (self._flight_acceleration_bound + self._holder_acceleration_bound) * float(dt) ** 2 / 8.0
        if kind == "legacy":
            return 0.0
        return None

    def _pair_curve_error(self, left_descriptor, right_descriptor, dt: float) -> float | None:
        left = self._track_error_budget(left_descriptor, dt)
        right = self._track_error_budget(right_descriptor, dt)
        if left is None or right is None:
            return None
        return left + right

    def _interval_descriptor(self, info, start: float, end: float):
        if not self._v3_enabled:
            return "legacy"
        if not isinstance(info, dict):
            return None
        kind = info.get("kind")
        if kind != "mixed_committed":
            return info
        plan_end = float(info.get("plan_end", end))
        if end <= plan_end + 1.0e-9:
            return {"kind": "flight_v3", "resampled": False}
        if start >= plan_end - 1.0e-9:
            return {"kind": "holder_proxy", "resampled": False}
        return {"kind": "mixed", "resampled": False}

    def _point_error(self, info) -> float | None:
        if not self._v3_enabled:
            return 0.0
        if not isinstance(info, dict):
            return None
        kind = info.get("kind")
        if kind == "holder_proxy" and info.get("resampled", False):
            return self._holder_acceleration_bound * float(self.config.physics_dt) ** 2 / 8.0
        if kind in {"holder_proxy", "flight_v3", "mixed_committed"}:
            return 0.0
        return None

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
                next_speed = self.dynamics.limited_tangential_speed(speed, target_speed, ring.radius, step)
                theta += .5 * (speed + next_speed) * step / ring.radius
                speed, remaining = next_speed, remaining - step
            positions.append(np.asarray((ring.radius * math.cos(theta), ring.radius * math.sin(theta), ring.height), dtype=float))
            previous = max(previous, float(value))
        return positions

    def _minimum_to_plan(self, holder, plan, airspace, now: float, residual: float, cache: PredictionCache | None = None) -> float:
        start = max(now, plan.start_time); horizon = min(plan.end_time, now + 30.0)
        times = np.arange(start, horizon + 1e-9, self.config.safety_sampling_dt)
        if self._v3_enabled and len(times) and times[-1] < horizon:
            times = np.append(times, horizon)
        if len(times) < 2: return float("inf")
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
            plan_positions = tuple(plan.sample_position(float(time)) for time in times)
            if cache is not None: cache.plan_positions[plan_key] = plan_positions
        holder_positions = np.asarray(holder_positions, dtype=float)
        plan_positions = np.asarray(plan_positions, dtype=float)
        relative_start = holder_positions[:-1] - plan_positions[:-1]
        relative_delta = ((holder_positions[1:] - holder_positions[:-1])
                          - (plan_positions[1:] - plan_positions[:-1]))
        denominator = np.einsum("ij,ij->i", relative_delta, relative_delta)
        projection = np.zeros_like(denominator)
        moving = denominator > 1e-12
        projection[moving] = -np.einsum(
            "ij,ij->i", relative_start[moving], relative_delta[moving]
        ) / denominator[moving]
        closest = relative_start + np.clip(projection, 0.0, 1.0)[:, None] * relative_delta
        swept = np.linalg.norm(closest, axis=1)
        plan_descriptor = self._track_descriptor(plan)
        if self._v3_enabled and plan_descriptor is None:
            return float("-inf")
        holder_descriptor = {"kind": "holder_proxy", "resampled": False}
        corrections = []
        for dt in np.diff(times):
            correction = self._pair_curve_error(holder_descriptor, plan_descriptor, float(dt))
            corrections.append(np.inf if correction is None else correction)
        return float(np.min(np.maximum(0.0, swept - np.asarray(corrections, dtype=float))))

    def _broad_phase_reject(self, holder, plan, airspace, now: float, cache: PredictionCache | None = None) -> bool:
        """Reject only pairs with a geometric proof of no shield interaction.

        CPA is intentionally calculated as a cheap diagnostic/ranking value;
        it never by itself approves a pair because curved trajectories could
        invalidate a constant-velocity assumption.  The actual rejection is
        limited to a conservative vertical envelope that cannot intersect the
        holder's constant ring altitude even under linear plan interpolation.
        """
        if self._v3_enabled:
            return False
        start, end = max(now, plan.start_time), min(plan.end_time, now + self.config.shield_horizon_seconds)
        if end <= start:
            return True
        ring = airspace.actual_ring(holder.ring_role)
        envelope_key = (plan.plan_id, plan.trajectory_hash, round(start, 8), round(end, 8))
        envelope = None if cache is None else cache.plan_envelopes.get(envelope_key)
        if envelope is None:
            # The plan is piecewise linear in the geometric envelope.  Only
            # the clipped boundary points need interpolation; all interior
            # segment endpoints are already available in ``plan.points``.
            # This avoids scanning and sampling every point outside the
            # current overlap interval.
            point_times = np.fromiter((point.t for point in plan.points), dtype=float)
            first_segment = max(0, min(
                len(plan.points) - 2, bisect_right(point_times, start) - 1,
            ))
            last_segment = max(first_segment, min(
                len(plan.points) - 2, bisect_right(point_times, end) - 1,
            ))
            segment_indices = np.arange(first_segment, last_segment + 1, dtype=int)
            segment_starts = np.maximum(start, point_times[segment_indices])
            segment_ends = np.minimum(end, point_times[segment_indices + 1])
            valid = segment_ends >= segment_starts
            if not np.any(valid):
                return True
            segment_indices = segment_indices[valid]
            segment_starts = segment_starts[valid]
            segment_ends = segment_ends[valid]
            start_position = plan.sample_position(float(start))
            end_position = plan.sample_position(float(end))
            left_positions = np.asarray(
                [start_position if moment == start else plan.points[index].position
                 for index, moment in zip(segment_indices, segment_starts)],
                dtype=float,
            )
            right_positions = np.asarray(
                [end_position if moment == end else plan.points[index + 1].position
                 for index, moment in zip(segment_indices, segment_ends)],
                dtype=float,
            )
            left_xy = left_positions[:, :2]
            delta = right_positions[:, :2] - left_xy
            denominator = np.einsum("ij,ij->i", delta, delta)
            projection = np.zeros_like(denominator)
            moving = denominator > 1e-12
            projection[moving] = -np.einsum(
                "ij,ij->i", left_xy[moving], delta[moving]
            ) / denominator[moving]
            projection = np.clip(projection, 0.0, 1.0)
            closest = left_xy + projection[:, None] * delta
            radial_min = float(np.min(np.linalg.norm(closest, axis=1)))
            radial_max = float(np.max(np.linalg.norm(
                np.concatenate((left_xy, right_positions[:, :2]), axis=0), axis=1,
            )))
            z_values = np.concatenate((left_positions[:, 2], right_positions[:, 2]))
            z_min = float(np.min(z_values))
            z_max = float(np.max(z_values))
            envelope = (radial_min, radial_max, z_min, z_max)
            if cache is not None:
                cache.plan_envelopes[envelope_key] = envelope
        radial_min, radial_max, z_min, z_max = envelope
        radial_gap = max(radial_min - ring.radius, ring.radius - radial_max, 0.0)
        vertical_gap = max(z_min - ring.height, ring.height - z_max, 0.0)
        lower_bound = math.hypot(radial_gap, vertical_gap)
        return lower_bound > self.config.safe_distance + self.config.shield_broad_phase_margin

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
            now, end, metadata={
                "target_phase": "ring_holder", "committed": False,
                # The points are generated by the identical
                # ``VehicleDynamics.ring_motion_step`` loop used by
                # TerminalParallelEnv._advance_ring_hold.  Marking this
                # provenance lets SafetyFilter avoid validating the sampled
                # chord as a free-form transfer trajectory (whose segment
                # velocity can be more conservative than the executor's
                # bounded tangent integration).
                "runtime_executor_equivalent": True,
            },
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

    def _protect_holder_headways(self, env, cache: PredictionCache) -> None:
        """Protect the next same-ring holder gap under finite speed response."""
        horizon = env.now + self.config.shield_horizon_seconds
        for role in env.config.rings:
            holders = [item for item in env.active.values()
                       if item.ring_role is role and item.committed_plan is None]
            if len(holders) < 2:
                continue
            ring = env.airspace.actual_ring(role)
            holders.sort(key=lambda item: (
                item.ring_progress if item.ring_progress is not None else math.atan2(item.position[1], item.position[0]),
                item.uid,
            ))

            def plan_for(item, residual: float) -> TrajectoryPlan:
                key = ("holder-headway", item.uid, round(float(residual), 8), round(env.now, 8), round(horizon, 8))
                if key not in cache.repair_plans:
                    cache.repair_plans[key] = self._repair_plan(item, env.airspace, env.now, horizon, residual)
                return cache.repair_plans[key]

            for index, trailing in enumerate(holders):
                leader = holders[(index + 1) % len(holders)]
                trailing_theta = trailing.ring_progress if trailing.ring_progress is not None else math.atan2(trailing.position[1], trailing.position[0])
                leader_theta = leader.ring_progress if leader.ring_progress is not None else math.atan2(leader.position[1], leader.position[0])
                arc_gap = ((leader_theta - trailing_theta) % (2.0 * math.pi)) * ring.radius
                current_distance = float(np.linalg.norm(
                    np.asarray(leader.position, dtype=float) - np.asarray(trailing.position, dtype=float)
                ))
                trailing_base = self.overrides.get(trailing.uid, (0.0, env.now))[0]
                leader_base = self.overrides.get(leader.uid, (0.0, env.now))[0]
                if (abs(trailing.speed - leader.speed) <= 1e-10 and abs(trailing_base - leader_base) <= 1e-10
                        and current_distance >= self.config.safe_distance):
                    # Equal-speed holders on the same fixed ring preserve
                    # their angular separation exactly; no sampled repair
                    # tracks are needed for this overwhelmingly common case.
                    continue
                reachable = (self.config.safe_distance + 2.0 * self.config.max_speed_residual_mps *
                             self.config.shield_horizon_seconds + self.config.shield_broad_phase_margin)
                if arc_gap > reachable:
                    continue
                nominal = {trailing.uid: plan_for(trailing, trailing_base),
                           leader.uid: plan_for(leader, leader_base)}
                if env.safety.verify_predicted_tracks(nominal, start=env.now, end=horizon).approved:
                    continue
                options = (
                    {trailing.uid: -self.config.max_speed_residual_mps, leader.uid: leader_base},
                    {trailing.uid: trailing_base, leader.uid: self.config.max_speed_residual_mps},
                    {trailing.uid: -self.config.max_speed_residual_mps,
                     leader.uid: self.config.max_speed_residual_mps},
                )
                repaired = False
                pair = {trailing.uid: trailing, leader.uid: leader}
                for residuals in options:
                    tracks = {uid: plan_for(pair[uid], residual) for uid, residual in residuals.items()}
                    if env.safety.verify_predicted_tracks(tracks, start=env.now, end=horizon).approved:
                        changed = [(uid, residual) for uid, residual in residuals.items() if residual != 0.0]
                        for uid, residual in changed:
                            self.overrides[uid] = (residual, horizon)
                        magnitude = sum(abs(residual) for _, residual in changed)
                        self.telemetry["shield_interventions"] += len(changed)
                        self.telemetry["shield_correction_magnitude"] += magnitude
                        self.telemetry["runtime_traffic_interventions"] += len(changed)
                        self.telemetry["runtime_traffic_repair_magnitude"] += magnitude
                        repaired = True
                        break
                if not repaired:
                    self.telemetry["shield_fail_closed"] += 1

    def _joint_pairs(self, env) -> tuple[tuple[int, int], ...]:
        """Return the complete set of potentially interacting runtime pairs.

        Same-ring holders only need their cyclic immediate neighbours: a
        non-neighbour cannot meet without first crossing one of those two
        neighbours, which the continuous swept check forbids.  Every holder
        versus a committed transfer is retained unless the conservative
        vertical/radial broad-phase proves the paths disjoint.  Immutable
        committed--committed pairs were proven atomically at commit time.
        """
        pairs: set[tuple[int, int]] = set()
        holders_by_ring: dict[object, list[object]] = {}
        committed = []
        for uav in env.active.values():
            if uav.committed_plan is not None:
                committed.append(uav)
            elif uav.ring_role is not None:
                holders_by_ring.setdefault(uav.ring_role, []).append(uav)
        for holders in holders_by_ring.values():
            if len(holders) < 2:
                continue
            holders.sort(key=lambda uav: (uav.ring_progress if uav.ring_progress is not None else
                                           math.atan2(uav.position[1], uav.position[0]), uav.uid))
            for index, holder in enumerate(holders):
                neighbour = holders[(index + 1) % len(holders)]
                pairs.add(tuple(sorted((holder.uid, neighbour.uid))))
        cache = PredictionCache()
        for holders in holders_by_ring.values():
            for holder in holders:
                for other in committed:
                    plan = other.committed_plan
                    if plan is None or plan.end_time <= env.now:
                        continue
                    # A transfer that hands off onto this holder's ring inside
                    # the protected horizon is never a broad-phase-cullable
                    # pair.  At the hand-off instant it becomes a same-ring
                    # holder with immutable pre-handoff motion; excluding it
                    # left a one-tick coverage hole and caused the observed
                    # holder-holder hard violation after plan completion.
                    target_role = {"ring_outer": "outer", "ring_middle": "middle", "ring_inner": "inner"}.get(
                        plan.metadata.get("target_phase")
                    )
                    imminent_handoff = (
                        target_role == getattr(holder.ring_role, "value", None)
                        and plan.end_time <= env.now + self.config.shield_horizon_seconds
                    )
                    if imminent_handoff:
                        pairs.add(tuple(sorted((holder.uid, other.uid))))
                        continue
                    if self._broad_phase_reject(holder, plan, env.airspace, env.now, cache):
                        self.telemetry["shield_broad_phase_rejected"] += 1
                        continue
                    pairs.add(tuple(sorted((holder.uid, other.uid))))
        return tuple(sorted(pairs))

    @staticmethod
    def _resample_position_track(
        source_times: np.ndarray, source_positions: np.ndarray, sample_times: np.ndarray,
    ) -> np.ndarray:
        """Match ``TrajectoryPlan.sample`` without constructing point objects.

        Joint safety checks need positions only.  The former path wrapped the
        same physics-grid positions in thousands of immutable
        ``TrajectoryPoint`` objects, hashed each temporary plan, and then
        sampled it again for every pair.  This performs the identical linear
        interpolation directly on the shared arrays.
        """
        source_times = np.asarray(source_times, dtype=float)
        source_positions = np.asarray(source_positions, dtype=float)
        sample_times = np.asarray(sample_times, dtype=float)
        right = np.searchsorted(source_times, sample_times, side="right")
        right = np.clip(right, 1, len(source_times) - 1)
        left = right - 1
        spans = source_times[right] - source_times[left]
        fractions = np.divide(
            sample_times - source_times[left], spans,
            out=np.zeros_like(sample_times), where=spans > 0.0,
        )
        result = source_positions[left] + fractions[:, None] * (
            source_positions[right] - source_positions[left]
        )
        result[sample_times <= source_times[0]] = source_positions[0]
        result[sample_times >= source_times[-1]] = source_positions[-1]
        return result

    def _joint_position_tracks(
        self, env, residuals: dict[int, float], uids: set[int], *, horizon: float | None = None,
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Build one shared position array per UAV for a joint proof.

        This is deliberately a *joint* proof.  Pairwise repairs accepted in
        isolation can be mutually inconsistent when several holders receive
        residual commands on the same policy tick.  Committed trajectories
        retain absolute priority; their post-plan holder continuation is
        sampled through the same SafetyFilter dynamics used at commit time.
        """
        start = float(env.now)
        end = start + float(self.config.shield_horizon_seconds if horizon is None else horizon)
        times = np.arange(start, end + 1e-9, self.config.physics_dt)
        if times[-1] < end:
            times = np.append(times, end)
        samples = np.arange(start, end + 1e-9, self.config.safety_sampling_dt)
        if samples[-1] < end:
            samples = np.append(samples, end)
        snapshot = SimpleNamespace(now=start, active=env.active, airspace=env.airspace)
        tracks: dict[int, np.ndarray] = {}
        self._joint_track_info = {}
        self._joint_assessment_times = samples
        for uid, uav in sorted(env.active.items()):
            if uid not in uids:
                continue
            direct_v3 = False
            if uav.committed_plan is not None:
                key = (
                    "committed", uav.committed_plan.plan_id, start, end,
                    float(env.airspace.speed_profile_bias),
                )
                plan = uav.committed_plan
                plan_schema = plan.metadata.get("trajectory_schema_version", "executable_trajectory_v2")
                certified_v3 = False
                if self._v3_enabled and plan_schema == "executable_trajectory_v3":
                    try:
                        certified_v3 = bool(plan.verify_trajectory_integrity())
                    except (AttributeError, TypeError, ValueError):
                        certified_v3 = False
                direct_v3 = self._v3_enabled and certified_v3
                if direct_v3:
                    direct_key = ("committed-v3-direct", plan.plan_id, plan.trajectory_hash, start, end,
                                  float(env.airspace.speed_profile_bias), tuple(np.round(samples, 8)))
                    positions = self._joint_sampled_track_cache.get(direct_key)
                    if positions is None:
                        positions = np.asarray(
                            env.safety._committed_positions(plan, snapshot, samples),
                            dtype=float,
                        )
                        self._joint_sampled_track_cache[direct_key] = positions
                    if plan.end_time < end - 1e-9:
                        track_info = {"kind": "mixed_committed", "plan_end": float(plan.end_time)}
                    else:
                        track_info = {"kind": "flight_v3", "resampled": False}
                else:
                    positions = self._joint_track_cache.get(key)
                    if positions is None:
                        positions = np.asarray(
                            env.safety._committed_positions(plan, snapshot, times),
                            dtype=float,
                        )
                        self._joint_track_cache[key] = positions
                    track_info = {"kind": "legacy" if not self._v3_enabled else "unknown", "resampled": True}
            elif uav.ring_role is not None:
                ring = env.airspace.actual_ring(uav.ring_role)
                theta = (
                    float(uav.ring_progress) if uav.ring_progress is not None
                    else math.atan2(float(uav.position[1]), float(uav.position[0]))
                )
                residual = float(residuals.get(uid, 0.0))
                key = (
                    "holder", uid, start, end, uav.ring_role.value,
                    tuple(float(value) for value in np.asarray(uav.position, dtype=float)),
                    theta, float(uav.speed), residual,
                    float(ring.radius), float(ring.height), float(ring.nominal_speed),
                    float(env.airspace.speed_profile_bias),
                )
                positions = self._joint_track_cache.get(key)
                if positions is None:
                    speed = max(0.0, float(uav.speed))
                    target_speed = float(np.clip(
                        ring.nominal_speed + env.airspace.speed_profile_bias + residual,
                        self.config.speed_min, self.config.max_speed,
                    ))
                    values = [np.asarray(uav.position, dtype=float).copy()]
                    dt_sequence = tuple(
                        float(right - left) for left, right in zip(times, times[1:])
                    )
                    speed_key = (
                        float(ring.radius), speed, target_speed, dt_sequence,
                    )
                    speed_sequence = self._joint_speed_sequence_cache.get(speed_key)
                    if speed_sequence is None:
                        speed0 = speed
                        sequence: list[tuple[float, float, float]] = []
                        for step in dt_sequence:
                            speed1 = self.dynamics.limited_tangential_speed(
                                speed0, target_speed, float(ring.radius), step,
                            )
                            angular_increment = .5 * (speed0 + speed1) * step / float(ring.radius)
                            sequence.append((speed0, speed1, angular_increment))
                            speed0 = speed1
                        speed_sequence = tuple(sequence)
                        self._joint_speed_sequence_cache[speed_key] = speed_sequence
                    for step, (speed0, speed1, _angular_increment) in zip(
                        dt_sequence, speed_sequence,
                    ):
                        # Keep the holder-specific theta update in the same
                        # order as ring_motion_step.  The shared sequence
                        # removes repeated speed integration while allowing
                        # holders with different starting angles to reuse it.
                        theta += .5 * (speed0 + speed1) * step / float(ring.radius)
                        speed = speed1
                        values.append(np.asarray((
                            float(ring.radius) * math.cos(theta),
                            float(ring.radius) * math.sin(theta),
                            float(ring.height),
                        ), dtype=float))
                    positions = np.asarray(values, dtype=float)
                    self._joint_track_cache[key] = positions
                track_info = {"kind": "holder_proxy", "resampled": True}
            else:
                continue
            sampled_key = (
                "sampled", key,
                float(self.config.physics_dt),
                float(self.config.safety_sampling_dt),
            )
            sampled = self._joint_sampled_track_cache.get(sampled_key)
            if direct_v3:
                sampled = positions
            else:
                if sampled is None:
                    sampled = self._resample_position_track(times, positions, samples)
                    self._joint_sampled_track_cache[sampled_key] = sampled
            tracks[uid] = sampled
            self._joint_track_info[uid] = track_info
        return samples, tracks

    @staticmethod
    def _position_track_distances(
        left: np.ndarray, right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized point and linearly swept separation for one pair."""
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        point = np.linalg.norm(left - right, axis=1)
        if len(left) < 2:
            return point, np.empty(0, dtype=float)
        relative_start = left[:-1] - right[:-1]
        relative_delta = (left[1:] - left[:-1]) - (right[1:] - right[:-1])
        denominator = np.einsum("ij,ij->i", relative_delta, relative_delta)
        projection = np.divide(
            -np.einsum("ij,ij->i", relative_start, relative_delta), denominator,
            out=np.zeros_like(denominator), where=denominator > 1e-12,
        )
        projection = np.clip(projection, 0.0, 1.0)
        closest = relative_start + projection[:, None] * relative_delta
        return point, np.linalg.norm(closest, axis=1)

    def _assess_position_pair(
        self, left: np.ndarray, right: np.ndarray, threshold: float | None = None,
        *, times: np.ndarray | None = None, left_info: dict | None = None,
        right_info: dict | None = None,
    ) -> tuple[bool, float]:
        """Reproduce the pairwise order and threshold semantics of the arbiter."""
        threshold = self.config.safe_distance if threshold is None else float(threshold)
        point, swept = self._position_track_distances(left, right)
        if self._v3_enabled and times is None:
            return False, float("-inf")
        if self._v3_enabled and times is not None:
            values = np.asarray(times, dtype=float)
            if len(values) != len(point) or len(values) != len(left) or len(values) != len(right):
                point = np.full_like(point, -np.inf)
                swept = np.full_like(swept, -np.inf)
            else:
                left_point_error = self._point_error(left_info)
                right_point_error = self._point_error(right_info)
                if left_point_error is None or right_point_error is None:
                    point = np.full_like(point, -np.inf)
                else:
                    point = point - left_point_error - right_point_error
                adjusted = []
                for index, dt in enumerate(np.diff(values)):
                    left_descriptor = self._interval_descriptor(left_info, values[index], values[index + 1])
                    right_descriptor = self._interval_descriptor(right_info, values[index], values[index + 1])
                    correction = self._pair_curve_error(left_descriptor, right_descriptor, float(dt))
                    adjusted.append(-np.inf if correction is None else max(0.0, swept[index] - correction))
                swept = np.asarray(adjusted, dtype=float)
        bad_point = np.flatnonzero(point < threshold)
        if len(bad_point):
            index = int(bad_point[0])
            return False, float(np.min(point[:index + 1]))
        minimum = float(np.min(point)) if len(point) else float("inf")
        bad_swept = np.flatnonzero(swept < threshold)
        if len(bad_swept):
            index = int(bad_swept[0])
            return False, min(minimum, float(np.min(swept[:index + 1])))
        if len(swept):
            minimum = min(minimum, float(np.min(swept)))
        return True, minimum

    def _joint_pair_cache_key(
        self, env, pair: tuple[int, int], residuals: dict[int, float],
        effective_horizon: float, threshold: float,
    ) -> tuple | None:
        """Return a within-protect identity key for one endpoint pair.

        ``id(env)`` identifies the immutable snapshot presented to a joint
        assessment.  While the cache is active, the strong reference table
        keeps every snapshot alive so CPython cannot recycle an id for a
        different successor projection during the same protect call.
        """
        cache = self._joint_pair_verdict_cache
        refs = self._joint_pair_snapshot_refs
        if cache is None or refs is None:
            return None
        snapshot_id = id(env)
        refs.setdefault(snapshot_id, env)
        left, right = pair
        endpoint_residuals = (
            float(residuals.get(left, 0.0)),
            float(residuals.get(right, 0.0)),
        )
        return (
            snapshot_id,
            (left, right),
            endpoint_residuals,
            float(effective_horizon),
            float(threshold),
            bool(self._v3_enabled),
            float(self.config.physics_dt),
            float(self.config.safety_sampling_dt),
            round(float(self._flight_acceleration_bound), 8),
            round(float(self._holder_acceleration_bound), 8),
        )

    def _joint_assessment(self, env, residuals: dict[int, float], *, pairs: tuple[tuple[int, int], ...] | None = None,
                          horizon: float | None = None, threshold: float | None = None) -> JointSafetyAssessment:
        """Check all relevant pairs and retain the current predicted conflict graph."""
        threshold = self.config.safe_distance if threshold is None else float(threshold)
        pairs = self._joint_pairs(env) if pairs is None else pairs
        self.telemetry["joint_safety_checks"] += 1
        if not pairs:
            return JointSafetyAssessment(True, float("inf"), ())
        active_pairs = tuple(pair for pair in pairs if all(uid in env.active for uid in pair))
        if not active_pairs:
            return JointSafetyAssessment(True, float("inf"), ())
        effective_horizon = float(self.config.shield_horizon_seconds if horizon is None else horizon)
        pair_records: list[tuple[tuple[int, int], tuple | None, tuple[bool, float] | None]] = []
        missing_pairs: list[tuple[int, int]] = []
        cache = self._joint_pair_verdict_cache
        for pair in active_pairs:
            key = self._joint_pair_cache_key(
                env, pair, residuals, effective_horizon, threshold,
            )
            verdict = None if cache is None or key is None else cache.get(key)
            pair_records.append((pair, key, verdict))
            if verdict is None:
                missing_pairs.append(pair)
        if missing_pairs:
            _, tracks = self._joint_position_tracks(
                env, residuals, {uid for pair in missing_pairs for uid in pair},
                horizon=effective_horizon,
            )
        else:
            tracks = {}
        minimum = float("inf")
        unsafe: list[tuple[int, int]] = []
        for pair, key, verdict in pair_records:
            left, right = pair
            if verdict is None:
                if self._v3_enabled:
                    verdict = self._assess_position_pair(
                        tracks[left], tracks[right], threshold,
                        times=self._joint_assessment_times,
                        left_info=self._joint_track_info.get(left),
                        right_info=self._joint_track_info.get(right),
                    )
                else:
                    verdict = self._assess_position_pair(tracks[left], tracks[right], threshold)
                if cache is not None and key is not None:
                    cache[key] = verdict
            approved, pair_minimum = verdict
            minimum = min(minimum, pair_minimum)
            if not approved:
                unsafe.append((left, right))
        return JointSafetyAssessment(not unsafe, minimum, tuple(sorted(unsafe)))

    def _joint_approved(self, env, residuals: dict[int, float]) -> bool:
        """Compatibility predicate retained for existing diagnostics/tests."""
        return self._joint_assessment(env, residuals).approved

    @staticmethod
    def _components(edges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, ...], ...]:
        """Deterministic connected components of the current conflict graph."""
        neighbours: dict[int, set[int]] = {}
        for left, right in edges:
            neighbours.setdefault(left, set()).add(right)
            neighbours.setdefault(right, set()).add(left)
        components: list[tuple[int, ...]] = []
        remaining = set(neighbours)
        while remaining:
            root = min(remaining)
            stack, component = [root], set()
            while stack:
                uid = stack.pop()
                if uid in component:
                    continue
                component.add(uid)
                stack.extend(sorted(neighbours.get(uid, ()), reverse=True))
            remaining.difference_update(component)
            components.append(tuple(sorted(component)))
        return tuple(components)

    def _next_executor_state(self, env, residuals: dict[int, float]):
        """Advance one control tick with the same plan-handoff semantics as physics.

        Recursive feasibility must evaluate the state that execution will
        actually present on the next policy tick.  In particular, a committed
        transfer can complete during this tick and become a ring holder; it
        must not remain an expired immutable plan in the successor state.
        """
        # Recursive feasibility mutates only top-level UAVState fields below.
        # TrajectoryPlan is immutable and the remaining nested dictionaries are
        # read-only in this projection, so copying the entire plan point arrays
        # for every shield tick is unnecessary and extremely expensive.
        active = {uid: copy(uav) for uid, uav in env.active.items()}
        elapsed = float(env.config.dt)
        for uid, uav in list(active.items()):
            plan = uav.committed_plan
            if plan is not None:
                target_time = min(float(env.now + elapsed), float(plan.end_time))
                point = plan.sample(target_time)
                uav.position = np.asarray(point.position, dtype=float)
                uav.velocity = np.asarray(point.velocity, dtype=float)
                uav.acceleration = np.asarray(point.acceleration, dtype=float)
                uav.speed = float(np.linalg.norm(uav.velocity[:2]))
                if env.now + elapsed + 1e-9 < plan.end_time:
                    continue
                target = plan.metadata.get("target_phase")
                if target == "landing":
                    active.pop(uid, None)
                    continue
                target_name = {"ring_outer": "outer", "ring_middle": "middle", "ring_inner": "inner"}.get(target)
                role = next((candidate for candidate in env.config.rings
                             if candidate.value == target_name), None)
                if role is None:
                    # A non-ring terminal/unknown plan cannot be steered by a
                    # holder residual.  Retain it as immutable so the normal
                    # pair verifier remains conservative.
                    continue
                uav.committed_plan = None
                uav.trajectory_plan_id = None
                uav.ring_role = role
                uav.ring_progress = math.atan2(uav.position[1], uav.position[0])
                ring = env.airspace.actual_ring(role)
                tangent = np.asarray((-math.sin(uav.ring_progress), math.cos(uav.ring_progress), 0.0), dtype=float)
                radial = np.asarray((-math.cos(uav.ring_progress), -math.sin(uav.ring_progress), 0.0), dtype=float)
                uav.velocity = tangent * uav.speed
                uav.acceleration = radial * (uav.speed * uav.speed / ring.radius)
                # The authoritative executor continues nominal ring motion
                # for any fraction left after a plan ends inside this control
                # tick.  Mirror that exact handoff instead of leaving a
                # frozen endpoint in the recursive successor state.
                remaining = max(0.0, float(env.now + elapsed - plan.end_time))
                target_speed = float(np.clip(ring.nominal_speed + env.airspace.speed_profile_bias,
                                             env.config.speed_min, env.config.max_speed))
                while remaining > 1e-9:
                    step = min(env.config.physics_dt, remaining)
                    motion = self.dynamics.ring_motion_step(ring, uav.ring_progress, uav.speed, target_speed, step)
                    uav.ring_progress, uav.speed, remaining = motion.theta, motion.speed, remaining - step
                    uav.position, uav.velocity, uav.acceleration = motion.position, motion.velocity, motion.acceleration
                continue
            if uav.ring_role is None:
                continue
            ring = env.airspace.actual_ring(uav.ring_role)
            theta = uav.ring_progress if uav.ring_progress is not None else math.atan2(uav.position[1], uav.position[0])
            target = float(np.clip(ring.nominal_speed + env.airspace.speed_profile_bias + residuals.get(uid, 0.0),
                                   env.config.speed_min, env.config.max_speed))
            remaining = elapsed
            while remaining > 1e-9:
                step = min(env.config.physics_dt, remaining)
                motion = self.dynamics.ring_motion_step(ring, theta, uav.speed, target, step)
                theta, uav.speed, remaining = motion.theta, motion.speed, remaining - step
                uav.position, uav.velocity, uav.acceleration = motion.position, motion.velocity, motion.acceleration
            uav.ring_progress = theta
        return SimpleNamespace(now=float(env.now + elapsed), active=active, airspace=env.airspace,
                               config=env.config, safety=env.safety,
                               future_release_reservations=getattr(env, "future_release_reservations", {}))

    def _headway_backup_request(self, env, residuals: dict[int, float], pairs: tuple[tuple[int, int], ...]) -> dict[int, float]:
        """Deterministic leader/follower proposal within existing residual bounds."""
        proposal = dict(residuals)
        for left, right in pairs:
            first, second = env.active.get(left), env.active.get(right)
            if first is None or second is None or first.committed_plan is not None or second.committed_plan is not None:
                continue
            if first.ring_role is None or first.ring_role is not second.ring_role:
                continue
            ring = env.airspace.actual_ring(first.ring_role)
            theta_a = first.ring_progress if first.ring_progress is not None else math.atan2(first.position[1], first.position[0])
            theta_b = second.ring_progress if second.ring_progress is not None else math.atan2(second.position[1], second.position[0])
            # Positive angular direction is the executor's ring travel direction.
            if (theta_b - theta_a) % (2.0 * math.pi) < math.pi:
                follower, leader = first, second
            else:
                follower, leader = second, first
            gap = ((leader.ring_progress - follower.ring_progress) % (2.0 * math.pi)) * ring.radius
            closing = follower.speed - leader.speed
            if gap < 1.5 * env.config.safe_distance or closing > 0.0:
                proposal[follower.uid] = -env.config.max_speed_residual_mps
                proposal[leader.uid] = env.config.max_speed_residual_mps
        return proposal

    def _ring_profile_backup_candidates(self, env, residuals: dict[int, float],
                                        pairs: tuple[tuple[int, int], ...]) -> list[dict[int, float]]:
        """Return certified-candidate proposals that preserve ring headways.

        A cyclic holder component cannot safely be repaired by applying the
        follower/leader rule independently to every edge: each holder is both
        a leader and a follower, so the last edge overwrites the preceding
        edge's command.  The resulting alternating command pattern was the
        source of the large, apparently ``infeasible`` components seen in the
        S3b t=1 replays.

        For every ring touched by a predicted conflict, add a *common target
        profile* for all mutable holders on that ring.  Equal target speeds
        preserve the cyclic ordering; it is not an unchecked ``all brake``
        fallback.  We include the demand-centred common target and a small,
        deterministic set of legal profiles, then require the ordinary swept
        proof before any one is used.  This also gives recursive feasibility a
        bounded deterministic policy for components larger than the N<=3
        diagnostic solver limit.
        """
        touched_roles = {
            uav.ring_role
            for pair in pairs
            for uid in pair
            for uav in (env.active.get(uid),)
            if uav is not None and uav.ring_role is not None and uav.committed_plan is None
        }
        profiles: list[tuple[tuple[int, ...], tuple[float, ...]]] = []
        for role in sorted(touched_roles, key=lambda item: str(item)):
            holders = tuple(sorted(
                uid for uid, uav in env.active.items()
                if uav.ring_role is role and uav.committed_plan is None and uid in residuals
            ))
            if len(holders) < 2:
                continue
            scale = float(env.config.max_speed_residual_mps)
            centre = float(np.clip(np.median([residuals[uid] / scale for uid in holders]), -1.0, 1.0))
            # Order from the least disruptive shared profile.  Include both
            # directions; certification, rather than a hard-coded brake,
            # decides whether a profile is admissible.
            # This is an online deterministic *backup-policy* catalogue, not
            # the offline exhaustive solver.  Keeping the demand-centred,
            # maximum-braking, nominal, and maximum-advance profiles bounds a
            # three-ring joint evaluation to 4^3 candidates.  Interior cases
            # remain available to the component solver/local probes below;
            # this avoids letting a 6^3 profile product dominate every PPO
            # safety tick.
            levels = tuple(sorted({centre, -1.0, 0.0, 1.0},
                                  key=lambda value: (abs(value - centre), abs(value), value)))
            profiles.append((holders, levels))
        if not profiles:
            return []
        # Conflicts on different rings can occur in the same control tick.
        # A proposal which restores only one ring leaves the other ring
        # unsafe, so it cannot certify an existential backup.  Enumerate the
        # compact product of ring profiles as one joint control (at most
        # 6^3=216 for the three terminal rings); the demand-centred tuple is
        # always first.  This is dramatically smaller than a per-UAV beam and
        # still entirely within the pre-existing action bounds.
        candidates: list[dict[int, float]] = []
        for selected in itertools.product(*(levels for _, levels in profiles)):
            trial = dict(residuals)
            for (holders, _), normalized in zip(profiles, selected):
                for uid in holders:
                    trial[uid] = float(normalized * env.config.max_speed_residual_mps)
            candidates.append(trial)
        return candidates

    def _backup_safe(self, env, residuals: dict[int, float], pairs: tuple[tuple[int, int], ...],
                     threshold: float | None = None) -> tuple[dict[int, float] | None, JointSafetyAssessment]:
        """Find an existential next-state backup; it need not equal current action.

        This is deliberately a policy, not a nested online optimizer.  The
        headway proposal is checked first, followed by deterministic uniform
        component actions and one-holder safe-interval probes.  Every action
        still receives the complete swept-separation proof.
        """
        # Fixed-rule flow derives a nominal holding contract with no residual
        # speed authority.  Keep the complete joint/swept assessment, but do
        # not enter the normalized profile/search catalogue (which divides by
        # the residual scale).  A nominal conflict therefore remains an
        # explicit rejected backup rather than being repaired implicitly.
        scale = float(getattr(
            getattr(env, "config", None),
            "max_speed_residual_mps",
            self.config.max_speed_residual_mps,
        ))
        if abs(scale) <= 1e-12:
            zero_residuals = {uid: 0.0 for uid in residuals}
            assessment_kwargs = {"pairs": pairs}
            if threshold is not None:
                assessment_kwargs["threshold"] = float(threshold)
            assessment = self._joint_assessment(env, zero_residuals, **assessment_kwargs)
            if assessment.approved:
                return zero_residuals, assessment
            return None, assessment
        baseline = self._headway_backup_request(env, residuals, pairs)
        assessment_kwargs = {"pairs": pairs}
        if threshold is not None:
            assessment_kwargs["threshold"] = float(threshold)
        baseline_assessment = self._joint_assessment(env, baseline, **assessment_kwargs)
        # Try ring-wide headway-preserving profiles *before* the local
        # leader/follower proposal.  A local proposal can be safe over one
        # horizon yet inject an alternating cyclic speed pattern which has no
        # viable backup at the next decision.  A shared ring profile preserves
        # the ordering and is therefore the deterministic invariant backup
        # policy used by the recursive-feasibility certificate.
        candidates: list[dict[int, float]] = self._ring_profile_backup_candidates(env, residuals, pairs)
        candidates.append(baseline)
        for component in self._components(baseline_assessment.unsafe_pairs or pairs):
            holders = tuple(uid for uid in component if uid in baseline)
            for normalized in (-1.0, -0.5, 0.0, 0.5, 1.0):
                trial = dict(baseline)
                for uid in holders:
                    trial[uid] = normalized * env.config.max_speed_residual_mps
                candidates.append(trial)
            # Holder-committed safe residual interval probe: only the holder
            # varies while immutable plans remain in the verification set.
            for uid in holders:
                for normalized in (-1.0, -0.5, 0.0, 0.5, 1.0):
                    trial = dict(baseline)
                    trial[uid] = normalized * env.config.max_speed_residual_mps
                    candidates.append(trial)
        best = baseline_assessment
        certified: list[tuple[tuple[float, float, tuple[float, ...]], dict[int, float], JointSafetyAssessment]] = []
        baseline_correction = float(sum(abs(baseline.get(uid, residuals.get(uid, 0.0)) - residuals.get(uid, 0.0))
                                       for uid in sorted(residuals)))
        baseline_action_key = tuple(round(baseline.get(uid, 0.0) / self.config.max_speed_residual_mps, 8)
                                    for uid in sorted(residuals))
        baseline_key = tuple((uid, float(baseline[uid])) for uid in sorted(baseline))
        evaluated_candidates: set[tuple[tuple[int, float], ...]] = {baseline_key}
        if baseline_assessment.approved:
            certified.append(((-baseline_assessment.minimum_separation,
                               baseline_correction, baseline_action_key),
                              baseline, baseline_assessment))
        for trial in candidates:
            candidate_key = tuple(
                (uid, float(trial[uid])) for uid in sorted(trial)
            )
            if candidate_key in evaluated_candidates:
                continue
            evaluated_candidates.add(candidate_key)
            verdict = self._joint_assessment(env, trial, **assessment_kwargs)
            if verdict.approved:
                # Backup control is selected for maximum certified separation
                # margin, not merely the first safe lattice point.  This is
                # important before an imminent handoff: a marginally-safe
                # zero-like profile can leave too little braking distance for
                # the successor state, while a legal leader/follower or ring
                # profile already has a robust margin.  Correction and action
                # vector supply deterministic ties only.
                correction = float(sum(abs(trial.get(uid, residuals.get(uid, 0.0)) - residuals.get(uid, 0.0))
                                       for uid in sorted(residuals)))
                action_key = tuple(round(trial.get(uid, 0.0) / self.config.max_speed_residual_mps, 8)
                                   for uid in sorted(residuals))
                certified.append(((-verdict.minimum_separation, correction, action_key), trial, verdict))
                continue
            if verdict.minimum_separation > best.minimum_separation:
                best = verdict
        if certified:
            certified.sort(key=lambda item: item[0])
            _, choice, verdict = certified[0]
            return choice, verdict
        # A small component can require a coordinated interior residual (for
        # example [-1.0, -0.6, -1.0]), which neither a uniform action nor a
        # single-holder interval probe represents.  Use the deterministic
        # bounded component solver only here, at the next state, and only for
        # N<=3; this supplies an existential backup without reintroducing an
        # unbounded nested search for large components.
        for component in self._components(baseline_assessment.unsafe_pairs or pairs):
            mutable = tuple(uid for uid in component if uid in baseline)
            if not mutable or len(mutable) > 3:
                continue
            choice, verdict = self._component_search(
                env, baseline, baseline, component, pairs, backup=True, recursive=False,
                threshold=threshold,
            )
            if choice is not None and verdict is not None and verdict.approved:
                return choice, verdict
        return None, best

    @staticmethod
    def _has_imminent_ring_handoff(env, *, within_seconds: float) -> bool:
        """Whether a committed plan will become a controllable ring holder soon."""
        ring_targets = {"ring_outer", "ring_middle", "ring_inner"}
        deadline = float(env.now + within_seconds + 1e-9)
        return any(
            uav.committed_plan is not None
            and uav.committed_plan.metadata.get("target_phase") in ring_targets
            and float(uav.committed_plan.end_time) <= deadline
            for uav in env.active.values()
        )

    @staticmethod
    def _successor_action_domain(env, residuals: dict[int, float]) -> dict[int, float]:
        """Give every controllable successor holder an explicit legal action.

        A transfer may complete between recursive-certification ticks.  The
        resulting ring holder did not exist in the current policy action map,
        but it *does* have legal shield authority at the successor state.  If
        it is omitted, ``_backup_safe`` incorrectly treats a
        committed--new-holder conflict as immutable and reports a spurious
        empty action set.  Zero here is only the requested value for a newly
        introduced variable; it still goes through the ordinary joint search
        and swept proof and is never a safety fallback.
        """
        domain = dict(residuals)
        for uid, uav in env.active.items():
            if uav.ring_role is not None and uav.committed_plan is None:
                domain.setdefault(uid, 0.0)
        return domain

    def _recursive_safe(self, env, residuals: dict[int, float], pairs: tuple[tuple[int, int], ...],
                        threshold: float | None = None) -> JointSafetyAssessment:
        """Current action is safe iff a (possibly different) next backup exists."""
        self.telemetry["recursive_feasibility_checks"] += 1
        assessment_kwargs = {"pairs": pairs}
        if threshold is not None:
            assessment_kwargs["threshold"] = float(threshold)
        current = self._joint_assessment(env, residuals, **assessment_kwargs)
        if not current.approved:
            self.last_recursive_gate = "current_horizon"
            return current
        next_env = self._next_executor_state(env, residuals)
        next_pairs = self._joint_pairs(next_env)
        successor_domain = self._successor_action_domain(next_env, residuals)
        if threshold is None:
            backup, future = self._backup_safe(next_env, successor_domain, next_pairs)
        else:
            backup, future = self._backup_safe(next_env, successor_domain, next_pairs, threshold=threshold)
        if backup is None:
            self.telemetry["recursive_feasibility_rejections"] += 1
            self.last_recursive_gate = "backup_existence"
            return JointSafetyAssessment(False, min(current.minimum_separation, future.minimum_separation), future.unsafe_pairs)
        # A transfer which completes in the next two executor ticks needs one
        # additional finite-step proof.  Without it, a successor at t+dt can
        # have an action that is safe *now* but whose first execution places
        # the newly handed-off holder inside the safety invariant at t+2dt.
        # This is a bounded handoff-specific viability check, not an unbounded
        # recursive optimizer: it asks only for the existence of the
        # deterministic backup policy after that first backup action.
        if self._has_imminent_ring_handoff(env, within_seconds=2.0 * float(env.config.dt)):
            after_backup = self._next_executor_state(next_env, backup)
            final_pairs = self._joint_pairs(after_backup)
            final_domain = self._successor_action_domain(after_backup, backup)
            if threshold is None:
                final_backup, final_assessment = self._backup_safe(after_backup, final_domain, final_pairs)
            else:
                final_backup, final_assessment = self._backup_safe(
                    after_backup, final_domain, final_pairs, threshold=threshold,
                )
            if final_backup is None:
                self.telemetry["recursive_feasibility_rejections"] += 1
                self.last_recursive_gate = "handoff_backup_existence"
                return JointSafetyAssessment(
                    False,
                    min(current.minimum_separation, future.minimum_separation, final_assessment.minimum_separation),
                    final_assessment.unsafe_pairs,
                )
        self.last_recursive_gate = "approved"
        return JointSafetyAssessment(True, min(current.minimum_separation, future.minimum_separation), ())

    def _component_search(self, env, requested: dict[int, float], residuals: dict[int, float],
                          component: tuple[int, ...], all_pairs: tuple[tuple[int, int], ...], *, backup: bool,
                          recursive: bool = True, threshold: float | None = None) -> tuple[dict[int, float] | None, JointSafetyAssessment | None]:
        """Search a bounded, deterministic joint action beam.

        Only mutable ring holders in the predicted conflict component are
        decision variables.  Committed plans remain fixed.  The grid is
        expressed in existing normalized PPO residual bounds.  Full Cartesian
        enumeration is intentionally forbidden here: a 16-holder component
        makes a 9^16 scan non-terminating in practice.  This beam never
        *approves* an unchecked action; each retained/full candidate has an
        H+dt execution-equivalent proof.  Beam exhaustion is reported as an
        uncertified condition, not misreported as an empty safe action set.
        """
        variables = tuple(uid for uid in component if uid in requested)
        if not variables:
            return None, None
        self.telemetry["joint_search_components"] += 1
        # Five symmetric values are a bounded action lattice over exactly the
        # pre-existing shield authority.  The policy request is inserted so a
        # no-op projection remains reachable even off lattice.
        grids = {
            # Dense near the deceleration boundary: feasibility diagnostics
            # found legitimate -0.6 residual solutions that the former
            # {-1,-.5,0,.5,1} lattice could not represent.
            uid: tuple(sorted({-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, 0.0, 0.5, 1.0,
                               float(np.clip(requested[uid] / self.config.max_speed_residual_mps, -1.0, 1.0))},
                              key=lambda item: (abs(item - requested[uid] / self.config.max_speed_residual_mps),
                                                abs(item), item)))
            for uid in variables
        }
        relevant_pairs = tuple(pair for pair in all_pairs if set(pair).intersection(component))
        beam_width = 16
        # Seed the beam with coordinated component actions.  A uniformly
        # populated ring preserves its ordering under the shared executor,
        # whereas starting only from independent requested residuals can trap
        # a coordinate beam behind every neighbour simultaneously changing
        # speed.  These are joint candidates, not a zero-residual fallback.
        beam: list[dict[int, float]] = [dict(residuals)]
        collective_values = sorted({value for values in grids.values() for value in values})
        for normalized in collective_values:
            collective = dict(residuals)
            for uid in variables:
                collective[uid] = float(normalized * self.config.max_speed_residual_mps)
            beam.append(collective)
        for uid in variables:
            ranked: list[tuple[tuple, dict[int, float], JointSafetyAssessment]] = []
            for base in beam:
                for normalized in grids[uid]:
                    trial = dict(base)
                    trial[uid] = float(normalized * self.config.max_speed_residual_mps)
                    if recursive:
                        future = (self._recursive_safe(env, trial, relevant_pairs) if threshold is None else
                                  self._recursive_safe(env, trial, relevant_pairs, threshold=threshold))
                    else:
                        future = (self._joint_assessment(env, trial, pairs=relevant_pairs) if threshold is None else
                                  self._joint_assessment(env, trial, pairs=relevant_pairs, threshold=threshold))
                    self.telemetry["joint_search_candidates"] += 1
                    correction = float(sum(abs(trial[item] - requested[item]) for item in variables))
                    action_key = tuple(round(trial[item] / self.config.max_speed_residual_mps, 8) for item in variables)
                    # Unsafe candidates stay in the beam only when they reduce
                    # the number of conflicts / improve the minimum margin.
                    score = (0 if future.approved else 1, len(future.unsafe_pairs),
                             -future.minimum_separation, correction, action_key)
                    ranked.append((score, trial, future))
            ranked.sort(key=lambda row: row[0])
            if len(ranked) > beam_width:
                self.telemetry["joint_search_pruned"] += len(ranked) - beam_width
            beam = [trial for _, trial, _ in ranked[:beam_width]]
            if not beam:
                break
        certified: list[tuple[tuple, dict[int, float], JointSafetyAssessment]] = []
        for trial in beam:
            if recursive:
                future = (self._recursive_safe(env, trial, relevant_pairs) if threshold is None else
                          self._recursive_safe(env, trial, relevant_pairs, threshold=threshold))
            else:
                future = (self._joint_assessment(env, trial, pairs=relevant_pairs) if threshold is None else
                          self._joint_assessment(env, trial, pairs=relevant_pairs, threshold=threshold))
            self.telemetry["joint_search_candidates"] += 1
            if not future.approved:
                self.telemetry["recursive_feasibility_rejections"] += 1
                continue
            correction = float(sum(abs(trial[uid] - requested[uid]) for uid in variables))
            action_key = tuple(round(trial[uid] / self.config.max_speed_residual_mps, 8) for uid in variables)
            score = ((-future.minimum_separation, correction, action_key) if backup
                     else (correction, -future.minimum_separation, action_key))
            certified.append((score, trial, future))
        if not certified:
            return None, None
        certified.sort(key=lambda row: row[0])
        _, best, assessment = certified[0]
        return best, assessment

    def _holder_priority(self, env, uid: int) -> tuple[int, int]:
        """Higher-priority intent is changed last (then stable UID order)."""
        uav = env.active[uid]
        if uav.committed_plan is not None:
            return (0, uid)
        if uid in env.future_release_reservations:
            return (1, uid)
        return (2, uid)

    @staticmethod
    def _candidate_residuals(requested: float) -> tuple[float, ...]:
        grid = np.linspace(-1.0, 1.0, 17)
        values = sorted({float(np.clip(requested, -1.0, 1.0)), *(float(item) for item in grid)},
                        key=lambda item: (abs(item - requested), abs(item), item))
        return tuple(values)

    def protect(self, env, intents: dict[int, object], traces: dict[int, LowLevelActionTrace] | None = None) -> dict[int, object]:
        """Run one protected policy tick with a scoped pair-verdict cache."""
        self._joint_pair_verdict_cache = {}
        self._joint_pair_snapshot_refs = {}
        self._joint_speed_sequence_cache.clear()
        try:
            return self._protect_impl(env, intents, traces)
        finally:
            # A verdict is valid only while the exact snapshot and residual
            # inputs remain within this protect call.  Clear both tables on
            # normal return and on every exception before the next policy
            # tick can observe them.
            if self._joint_pair_verdict_cache is not None:
                self._joint_pair_verdict_cache.clear()
            if self._joint_pair_snapshot_refs is not None:
                self._joint_pair_snapshot_refs.clear()
            self._joint_pair_verdict_cache = None
            self._joint_pair_snapshot_refs = None
            self._joint_speed_sequence_cache.clear()

    def _protect_impl(self, env, intents: dict[int, object], traces: dict[int, LowLevelActionTrace] | None = None) -> dict[int, object]:
        """Project every holder speed residual through a joint admissibility test.

        This is the final pre-physics gate: no holder speed residual reaches
        ``_advance_ring_hold`` unless a joint, recursively-feasible swept
        separation proof exists.  Nominal (zero) residual is merely one legal
        candidate; it is never an unchecked fail-closed fallback.
        """
        # One policy tick may evaluate dozens of bounded joint candidates.
        # Their immutable committed tracks and repeated holder residuals are
        # shared only within this tick and are invalidated before any new
        # authoritative state can be executed.
        self._joint_track_cache.clear()
        self._joint_sampled_track_cache.clear()
        repaired = dict(intents)
        holders = [uid for uid, uav in env.active.items()
                   if uid in repaired and uav.ring_role is not None and uav.committed_plan is None]
        if not holders:
            return repaired
        residuals = {
            uid: float(np.clip(repaired[uid].speed_residual, -1.0, 1.0)) * self.config.max_speed_residual_mps
            for uid in holders
        }
        requested = dict(residuals)
        pairs = self._joint_pairs(env)
        component_records: list[dict[str, object]] = []
        soft_mode = False
        soft_takeover_uids: set[int] = set()
        # A single H+dt proof implies current-horizon safety and proves that
        # the candidate remains a legal backup after one executor tick.
        assessment = self._recursive_safe(env, residuals, pairs)
        if abs(float(env.config.max_speed_residual_mps)) <= 1e-12:
            # The fixed-rule flow contract has no shield speed authority.  A
            # nominal action is still accepted only after the complete
            # recursive joint/swept proof above; a nominal conflict fails
            # closed instead of entering the residual search or 40 m buffer.
            if not assessment.approved:
                self.unrecoverable_components = tuple(
                    self._components(assessment.unsafe_pairs or pairs)
                )
                self.telemetry["shield_fail_closed"] += 1
                self.telemetry["joint_safety_fail_closed"] += 1
                self.telemetry["joint_unrecoverable_components"] += len(self.unrecoverable_components)
                self.last_certification_diagnostic = {
                    "time": float(env.now),
                    "pairs": pairs,
                    "unsafe_pairs": assessment.unsafe_pairs,
                    "initial_minimum_separation": assessment.minimum_separation,
                    "unrecoverable_components": self.unrecoverable_components,
                    "recursive_gate": self.last_recursive_gate,
                    "speed_residual_contract": "nominal_zero",
                    "operational_threshold": float(self.config.safe_distance),
                }
                raise SafetyInvariantError(
                    "Nominal fixed-rule holding action is not recursively safe; physics execution blocked",
                    diagnostic=self.last_certification_diagnostic,
                )
            self.unrecoverable_components = ()
            return {
                uid: type(repaired[uid])(
                    repaired[uid].release_request,
                    0.0,
                    repaired[uid].merge_time_residual,
                ) if uid in holders else repaired[uid]
                for uid in repaired
            }
        if not assessment.approved:
            # Before a general component search, use the deterministic backup
            # policy at the *current* state.  It is already a swept-certified
            # joint control, and must not be hidden behind a sparse beam.
            backup_now, _ = self._backup_safe(env, residuals, pairs)
            if backup_now is not None:
                backup_assessment = self._recursive_safe(env, backup_now, pairs)
                if backup_assessment.approved:
                    residuals, assessment = backup_now, backup_assessment
                    self.telemetry["joint_backup_uses"] += 1
        if not assessment.approved:
            # The *failed* predicted pairs are the conflict graph.  Solve one
            # connected component at a time, including every pair incident to
            # that component in each candidate proof so a repair cannot create
            # a fresh edge outside it.
            failures = assessment.unsafe_pairs or pairs
            unresolved: list[tuple[int, ...]] = []
            for component in self._components(failures):
                choice, result = self._component_search(env, requested, residuals, component, pairs, backup=False)
                if choice is None:
                    # This is a verified deterministic backup controller, not
                    # an all-UAV slowdown: it maximizes component separation
                    # margin subject to the existing legal residual bounds.
                    choice, result = self._component_search(env, requested, residuals, component, pairs, backup=True)
                    if choice is not None:
                        self.telemetry["joint_backup_uses"] += 1
                if choice is None:
                    unresolved.append(component)
                else:
                    residuals = choice
                component_records.append({
                    "uids": component, "size": len(component),
                    "requested_residual": {str(uid): requested[uid] for uid in component if uid in requested},
                    "best_candidate": None if choice is None else {str(uid): choice[uid] for uid in component if uid in choice},
                    "best_minimum_separation": None if result is None else result.minimum_separation,
                    "rejection_gate": self.last_recursive_gate,
                    "unsafe_pairs": () if result is None else result.unsafe_pairs,
                })
            assessment = self._recursive_safe(env, residuals, pairs)
            if unresolved or not assessment.approved:
                # The 80 m operating separation is a desired buffer.  When
                # it cannot be certified, retain the same bounded residual
                # search and recursive executor-equivalent proof at the 40 m
                # protected floor.  Only failure at that floor blocks physics.
                soft_threshold = float(self.config.nmac_distance)
                soft_failures = assessment.unsafe_pairs or pairs
                soft_components = self._components(soft_failures)
                soft_takeover_uids = {uid for component in soft_components for uid in component}
                soft_choice, _ = self._backup_safe(
                    env, residuals, pairs, threshold=soft_threshold,
                )
                soft_assessment = None
                if soft_choice is not None:
                    soft_assessment = self._recursive_safe(
                        env, soft_choice, pairs, threshold=soft_threshold,
                    )
                    if soft_assessment.approved:
                        residuals, assessment = soft_choice, soft_assessment
                        soft_mode = True
                        self.telemetry["joint_backup_uses"] += 1
                if not soft_mode:
                    soft_unresolved: list[tuple[int, ...]] = []
                    for component in soft_components:
                        choice, result = self._component_search(
                            env, requested, residuals, component, pairs,
                            backup=True, threshold=soft_threshold,
                        )
                        if choice is None:
                            soft_unresolved.append(component)
                        else:
                            residuals = choice
                        component_records.append({
                            "threshold": soft_threshold,
                            "uids": component, "size": len(component),
                            "requested_residual": {str(uid): requested[uid] for uid in component if uid in requested},
                            "best_candidate": None if choice is None else {str(uid): choice[uid] for uid in component if uid in choice},
                            "best_minimum_separation": None if result is None else result.minimum_separation,
                            "rejection_gate": self.last_recursive_gate,
                            "unsafe_pairs": () if result is None else result.unsafe_pairs,
                        })
                    soft_assessment = self._recursive_safe(
                        env, residuals, pairs, threshold=soft_threshold,
                    )
                    if not soft_unresolved and soft_assessment.approved:
                        assessment = soft_assessment
                        soft_mode = True
                if not soft_mode:
                    self.unrecoverable_components = tuple(
                        soft_unresolved if soft_unresolved
                        else self._components(soft_assessment.unsafe_pairs if soft_assessment is not None and soft_assessment.unsafe_pairs else pairs)
                    )
                    self.telemetry["shield_fail_closed"] += 1
                    self.telemetry["joint_safety_fail_closed"] += 1
                    self.telemetry["joint_unrecoverable_components"] += len(self.unrecoverable_components)
                    self.last_certification_diagnostic = {
                        "time": float(env.now), "pairs": pairs,
                        "initial_minimum_separation": assessment.minimum_separation,
                        "rejected_components": component_records,
                        "unrecoverable_components": self.unrecoverable_components,
                        "recursive_gate": self.last_recursive_gate,
                        "operational_threshold": soft_threshold,
                        "controlled_holder_reachable_tube": "not_used_in_runtime_shield",
                    }
                    raise SafetyInvariantError(
                        "No certified bounded joint backup action above protected floor; physics execution blocked for components "
                        f"{self.unrecoverable_components}", diagnostic=self.last_certification_diagnostic,
                    )
        else:
            self.unrecoverable_components = ()
        if soft_mode:
            self.unrecoverable_components = ()
        approved = assessment.approved
        shielded_uids: set[int] = set()
        for uid in holders:
            actual = residuals[uid]
            before = requested[uid]
            if abs(actual - before) <= 1e-10:
                continue
            intent = repaired[uid]
            shield_intent = type(intent)(intent.release_request,
                                         float(np.clip(actual / self.config.max_speed_residual_mps, -1.0, 1.0)),
                                         intent.merge_time_residual)
            repaired[uid] = shield_intent
            magnitude = abs(shield_intent.speed_residual - intent.speed_residual)
            self.telemetry["shield_interventions"] += 1
            self.telemetry["shield_correction_magnitude"] += magnitude
            self.telemetry["runtime_traffic_interventions"] += 1
            self.telemetry["runtime_traffic_repair_magnitude"] += magnitude
            self.telemetry["joint_safety_corrections"] += 1
            self.telemetry["admissibility_rejections"] += 1
            if traces is not None and uid in traces:
                trace = traces[uid]
                trace.shield_applied = shield_intent
                trace.shield_used = True
                trace.shield_reason = "joint_recursive_admissibility_projection"
                trace.correction_magnitude += magnitude
            shielded_uids.add(uid)
        if soft_mode:
            # Even a zero-delta action is a shield takeover in the 40--80 m
            # operating buffer.  Preserve the applied intent explicitly so
            # the existing environment reward charges the intervention cost.
            for uid in sorted(soft_takeover_uids.intersection(holders)):
                actual = residuals[uid]
                intent = repaired[uid]
                shield_intent = type(intent)(
                    intent.release_request,
                    float(np.clip(actual / self.config.max_speed_residual_mps, -1.0, 1.0)),
                    intent.merge_time_residual,
                )
                if uid not in shielded_uids:
                    self.telemetry["shield_interventions"] += 1
                    self.telemetry["runtime_traffic_interventions"] += 1
                self.telemetry["operational_buffer_takeovers"] += 1
                if traces is not None and uid in traces:
                    trace = traces[uid]
                    trace.shield_applied = shield_intent
                    trace.shield_used = True
                    trace.shield_reason = "operational_buffer_takeover"
        return repaired


# Compatibility alias for archived A6 diagnostics.  New code should import
# ``PredictiveSafetyShield`` so its deployment role is explicit.
RuntimeTrafficSafetyController = PredictiveSafetyShield
