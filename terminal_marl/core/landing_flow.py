"""Opt-in finite-candidate landing-flow coordination.

The historical V4.2 release path remains owned by ``TerminalParallelEnv``.
This coordinator is deliberately a small, fixed-rule execution layer used by
the dense flow repair profile.  It accepts only the current release intent,
builds a bounded set of canonical flow trajectories, and then runs every
candidate through the existing energy and safety authorities before binding
any scheduler or landing-zone state.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from .config import RingRole
from .flow_trajectories import FlowTrajectoryGenerator
from .lifecycle import transition
from .merge_scheduler import LowLevelIntent, MergeProposal
from .uav import FlightPhase, UAVState

if TYPE_CHECKING:  # pragma: no cover - imports are only for static readers
    from ..envs.parallel_env import TerminalParallelEnv


class LandingFlowCoordinator:
    """Coordinate one fixed-rule landing-flow release at a time.

    The coordinator owns no simulation state of its own beyond the parent
    environment reference.  That keeps reset, rollback, and the legacy false
    path easy to audit.  All durable reservations and terminal state remain
    in the existing scheduler, landing-zone manager, and UAV objects.
    """

    RING_DURATIONS = {
        # The source-ring geometry determines the minimum useful duration;
        # each interface retains four duration candidates, while M -> I also
        # enumerates three tangential speed biases.
        RingRole.MIDDLE: (48.0, 60.0, 72.0, 84.0),  # O -> M
        RingRole.INNER: (60.0, 72.0, 84.0, 96.0),   # M -> I
    }
    # Tangential repair candidates are scoped to M -> I.  The outer transfer
    # and final approach retain their historical single geometry path.
    INNER_TANGENTIAL_SPEED_BIASES_MPS = (0.0, -4.0, -8.0)
    FINAL_DURATIONS = (20.0, 28.0, 36.0, 48.0, 60.0, 90.0)
    NORMAL_RETRY_SECONDS = 2.0
    EMERGENCY_RETRY_SECONDS = 5.0
    _RING_TRAVEL_SECONDS = {
        RingRole.OUTER: 125.0,
        RingRole.MIDDLE: 85.0,
        RingRole.INNER: 45.0,
    }

    def __init__(self, env: "TerminalParallelEnv") -> None:
        self.env = env
        self.trajectory = FlowTrajectoryGenerator(env.config, env.generator)

    @staticmethod
    def _source_target(uav: UAVState) -> tuple[str, RingRole | str, FlightPhase] | None:
        if uav.phase is FlightPhase.RING_OUTER and uav.ring_role is RingRole.OUTER:
            return "o2m", RingRole.MIDDLE, FlightPhase.TRANSFER_O2M
        if uav.phase is FlightPhase.RING_MIDDLE and uav.ring_role is RingRole.MIDDLE:
            return "m2i", RingRole.INNER, FlightPhase.TRANSFER_M2I
        if uav.phase is FlightPhase.RING_INNER and uav.ring_role is RingRole.INNER:
            return "i2f", "final", FlightPhase.FINAL
        return None

    @staticmethod
    def _target_role(target: RingRole | str) -> RingRole | None:
        return target if isinstance(target, RingRole) else None

    @staticmethod
    def _target_phase_name(target: RingRole | str) -> str:
        if isinstance(target, RingRole):
            return {
                RingRole.OUTER: "ring_outer",
                RingRole.MIDDLE: "ring_middle",
                RingRole.INNER: "ring_inner",
            }[target]
        return "landing"

    def _snapshot(self):
        # SafetyFilter consumes this structural view only; keeping it local
        # avoids a core -> env import cycle.
        return SimpleNamespace(
            now=self.env.now,
            active=self.env.active,
            airspace=self.env.airspace,
        )

    def _committed_plans(self) -> dict[int, object]:
        return {
            uid: other.committed_plan
            for uid, other in self.env.active.items()
            if other.committed_plan is not None
        }

    def _refresh_occupancy(self) -> None:
        self.env._refresh_airspace_occupancy()

    def _ring_capacity_available(self, target: RingRole) -> bool:
        """Count holders and committed incoming plans against actual capacity."""
        self._refresh_occupancy()
        incoming = int(getattr(self.env.airspace, "_committed", {}).get(target, 0))
        return (
            int(self.env.airspace.occupancy(target)) + incoming
            < int(self.env.airspace.capacity(target))
        )

    def _inner_departure_gap_angle(self) -> float:
        """Return the i2f egress headway as an angular separation.

        The spacing is derived from the existing i2f time headway and the
        nominal inner-ring speed.  It is a queue-spacing policy for M->I
        admission only; the normal energy and SafetyFilter checks remain the
        authorities for an executable candidate and for final approach.
        """
        ring = self.env.airspace.actual_ring(RingRole.INNER)
        radius = float(ring.radius)
        speed = float(ring.nominal_speed)
        if radius <= 0.0 or speed <= 0.0:
            raise ValueError("inner-ring radius and nominal speed must be positive")
        return float(self.env.airspace.headways["i2f"]) * speed / radius

    def _inner_departure_capacity_limit(self) -> int:
        """Cap inner holders plus committed M->I arrivals by egress spacing."""
        angle = self._inner_departure_gap_angle()
        spacing_capacity = int(math.floor((2.0 * math.pi) / angle))
        return min(int(self.env.airspace.capacity(RingRole.INNER)), spacing_capacity)

    def _inner_departure_occupancy(self) -> tuple[int, int]:
        """Return current inner occupancy and the derived egress cap."""
        self._refresh_occupancy()
        current = int(self.env.airspace.occupancy(RingRole.INNER))
        incoming = int(
            getattr(self.env.airspace, "_committed", {}).get(RingRole.INNER, 0)
        )
        return current + incoming, self._inner_departure_capacity_limit()

    @staticmethod
    def _angular_gap(first: float, second: float) -> float:
        return abs((float(first) - float(second) + math.pi) % (2.0 * math.pi) - math.pi)

    def _inner_departure_headway_reason(self, uav: UAVState, plan) -> str | None:
        """Check projected M->I queue spacing before full SafetyFilter work.

        Current inner holders and committed M->I plans are projected at the
        later of their two handoff times.  Committed final plans are omitted
        because they are departing the inner ring.  This check only governs
        inner queue spacing; full energy and safety certification still runs
        after it.
        """
        if plan.metadata.get("target_phase") != "ring_inner":
            return None
        required = self._inner_departure_gap_angle()
        snapshot = self._snapshot()
        candidate_end = float(plan.end_time)

        def candidate_position(moment: float) -> np.ndarray:
            times = np.asarray([float(moment)], dtype=float)
            if moment <= candidate_end + 1e-9:
                return np.asarray(plan.sample_position(moment), dtype=float)
            return np.asarray(
                self.env.safety._post_plan_positions(plan, snapshot, times)[0],
                dtype=float,
            )

        for other in self.env.active.values():
            if other.uid == uav.uid:
                continue
            other_plan = other.committed_plan
            if other_plan is not None:
                # Only future holders of the inner ring participate.  Final
                # approaches (target_phase=landing) are intentionally ignored.
                if other_plan.metadata.get("target_phase") != "ring_inner":
                    continue
                other_end = float(other_plan.end_time)
            elif other.ring_role is RingRole.INNER:
                other_end = float(self.env.now)
            else:
                continue

            moment = max(candidate_end, other_end)
            times = np.asarray([float(moment)], dtype=float)
            if other_plan is not None:
                other_position = np.asarray(
                    self.env.safety._committed_positions(other_plan, snapshot, times)[0],
                    dtype=float,
                )
            else:
                other_position = np.asarray(
                    self.env.safety._holder_positions(other, snapshot, times, 0.0)[0],
                    dtype=float,
                )
            candidate = candidate_position(moment)
            candidate_angle = math.atan2(float(candidate[1]), float(candidate[0]))
            other_angle = math.atan2(float(other_position[1]), float(other_position[0]))
            if self._angular_gap(candidate_angle, other_angle) + 1e-9 < required:
                return "inner_departure_headway"
        return None

    def _urgent_spacing_energy_limit(self, uav: UAVState) -> float:
        """Return one current-ring circuit of energy at current phase power."""
        role = uav.ring_role
        if not isinstance(role, RingRole):
            return math.nan
        ring = self.env.airspace.actual_ring(role)
        radius = float(ring.radius)
        speed = float(ring.nominal_speed)
        phase_power = float(self.env.energy.phase_power(uav.phase))
        if radius <= 0.0 or speed <= 0.0 or phase_power < 0.0:
            return math.nan
        return (2.0 * math.pi * radius / speed) * phase_power

    def _urgent_spacing_energy_ready(
        self, uav: UAVState, previous_diagnostic: object | None
    ) -> bool:
        """Require a fresh, route-energy one-circuit budget."""
        diagnostic = self.env.energy_reachability_diagnostics.get(uav.uid)
        if diagnostic is None or diagnostic is previous_diagnostic:
            return False
        if diagnostic.get("selected_lz_id") is None:
            return False
        try:
            slack = float(diagnostic["slack_energy"])
        except (KeyError, TypeError, ValueError):
            return False
        limit = self._urgent_spacing_energy_limit(uav)
        return (
            math.isfinite(slack)
            and slack >= -1e-12
            and math.isfinite(limit)
            and slack <= limit + 1e-9
        )

    def _final_capacity_available(self) -> bool:
        """Apply the final queue cap to active normal final approaches."""
        count = 0
        for other in self.env.active.values():
            if other.phase is FlightPhase.FINAL:
                count += 1
            elif (
                other.committed_plan is not None
                and other.committed_plan.metadata.get("target_phase") == "landing"
            ):
                count += 1
        return count < int(self.env.config.final_queue_capacity)

    def _available_lz_ids(self, uav: UAVState) -> tuple[int, ...]:
        # LandingZoneManager filters both physical status and mission
        # compatibility.  No emergency-priority pseudo reservation is used.
        return tuple(self.env.lz_manager.rank_available_zones(uav, self.env.now))

    def _proposal(self, target: RingRole | str, plan) -> MergeProposal:
        if isinstance(target, RingRole):
            interface = {
                RingRole.MIDDLE: "o2m",
                RingRole.INNER: "m2i",
            }[target]
            endpoint = np.asarray(plan.points[-1].position, dtype=float)
            merge_angle = float(math.atan2(endpoint[1], endpoint[0]) % (2.0 * math.pi))
            merge_time = float(plan.end_time)
        else:
            interface = "i2f"
            # The final slot is the inner-ring departure event.  The helper
            # stores this metadata explicitly, so touchdown time is never
            # confused with the i2f headway event.
            merge_time = float(plan.metadata["scheduled_departure"])
            merge_angle = None
        return MergeProposal(
            target=target,
            merge_time=merge_time,
            merge_angle=merge_angle,
            headway=float(self.env.airspace.headways[interface]),
        )

    def _row_start(self, uav: UAVState, intent: LowLevelIntent, interface: str) -> dict:
        row = self.env._start_release_intent(uav, intent, interface)
        row.update(
            {
                "flow_coordination": True,
                "flow_candidate_durations": [],
                "flow_candidates_checked": 0,
                "flow_candidate_rejection_reasons": [],
                "flow_selected_duration": None,
                "flow_selected_lz_id": None,
                "flow_selected_tangential_speed_bias_mps": None,
                "flow_deadline_fallback": False,
            }
        )
        return row

    @staticmethod
    def _record_candidate(
        row: dict,
        duration: float,
        reason: str,
        lz_id: int | None = None,
        tangential_speed_bias_mps: float = 0.0,
    ) -> None:
        row["flow_candidates_checked"] += 1
        row["flow_candidate_durations"].append(
            {
                "duration": float(duration),
                "lz_id": lz_id,
                "tangential_speed_bias_mps": float(tangential_speed_bias_mps),
                "reason": reason,
            }
        )
        row["flow_candidate_rejection_reasons"].append(reason)

    def _finish(
        self,
        uav: UAVState,
        row: dict,
        decision: str,
        *,
        reason: str | None = None,
        plan_id: str | None = None,
    ) -> str:
        if decision not in {
            "approved",
            "not_requested",
            "plan_active",
            "phase_masked",
            # A retry response means the previous rejection's deadline is
            # still active.  Extending that deadline on every one-second
            # policy tick would make a two-second retry permanently expire.
            "safety_wait",
        }:
            uav.safety_retry_after = self.env.now + self.NORMAL_RETRY_SECONDS
        self.env._finish_release_intent(
            uav,
            row,
            decision,
            safety_reason=reason,
            plan_id=plan_id,
            selection_source="flow_coordination",
        )
        return decision

    def _candidate_plan(
        self,
        uav: UAVState,
        target: RingRole | str,
        lz_id: int | None,
        duration: float,
        tangential_speed_bias_mps: float = 0.0,
        allow_urgent_spacing: bool = False,
    ):
        if isinstance(target, RingRole):
            interface = {
                RingRole.MIDDLE: "o2m",
                RingRole.INNER: "m2i",
            }[target]
            known_slot = self.env.now + float(duration)
        else:
            interface = "i2f"
            known_slot = self.env.now
        if not self.env.scheduler.can_reserve_at(
            target,
            known_slot,
            headway=float(self.env.airspace.headways[interface]),
        ):
            return None, "headway_conflict"
        if target is RingRole.INNER:
            spacing_rejected = False
            # Prepare and prescreen the array representation first.  A
            # rejected queue-spacing candidate must not allocate the complete
            # point tuple or a trajectory hash.
            try:
                (
                    prepared_duration,
                    prepared_bias,
                    source,
                    target_role,
                    target_ring,
                    start,
                ) = self.trajectory._ring_transfer_context(
                    uav,
                    target,
                    self.env.airspace,
                    self.env.now,
                    duration,
                    tangential_speed_bias_mps,
                )
                prepared = self.trajectory._prepare_ring_transfer(
                    uav,
                    target_ring,
                    prepared_duration,
                    prepared_bias,
                )
                preview = self.trajectory._ring_transfer_preview(
                    uav,
                    source,
                    target_role,
                    start,
                    prepared_duration,
                    prepared_bias,
                    prepared,
                )
            except (RuntimeError, ValueError, KeyError, AssertionError):
                return None, "trajectory_infeasible"

            proposal = self._proposal(target, preview)
            if not self.env.scheduler.can_reserve(proposal):
                return None, "headway_conflict"

            # The M->I queue-spacing guard is cheaper than materialisation and
            # the route-energy budget.  The preview is read-only and cannot be
            # submitted to safety or commit.
            headway_reason = self._inner_departure_headway_reason(uav, preview)
            if headway_reason is not None:
                if not (
                    allow_urgent_spacing
                    and headway_reason == "inner_departure_headway"
                ):
                    return None, headway_reason
                spacing_rejected = True

            try:
                plan = self.trajectory._materialize_ring_transfer(
                    uav,
                    source,
                    target_role,
                    start,
                    prepared_duration,
                    prepared_bias,
                    prepared,
                )
                errors = self.trajectory.dynamics.validate_plan(plan.points)
            except (RuntimeError, ValueError, KeyError, AssertionError):
                return None, "trajectory_infeasible"
            if errors:
                return None, "trajectory_infeasible"
        else:
            try:
                if target == "final":
                    assert lz_id is not None
                    plan = self.trajectory.final_approach(
                        uav,
                        self.env.lz_manager.zones[lz_id],
                        self.env.airspace,
                        self.env.now,
                        duration,
                        emergency=False,
                    )
                else:
                    plan = self.trajectory.ring_transfer(
                        uav,
                        target,
                        self.env.airspace,
                        self.env.now,
                        duration,
                        tangential_speed_bias_mps=tangential_speed_bias_mps,
                    )
            except (RuntimeError, ValueError, KeyError, AssertionError):
                return None, "trajectory_infeasible"

            proposal = self._proposal(target, plan)
            if not self.env.scheduler.can_reserve(proposal):
                return plan, "headway_conflict"

        previous_energy_diagnostic = self.env.energy_reachability_diagnostics.get(
            uav.uid
        )
        energy_reason = self.env._candidate_energy_guard_reason(uav, plan, target)
        if energy_reason is not None:
            return plan, energy_reason

        if target is RingRole.INNER and spacing_rejected:
            if not self._urgent_spacing_energy_ready(uav, previous_energy_diagnostic):
                return plan, "inner_departure_headway"

        result = self.env.safety.filter_plan(
            plan,
            self._committed_plans(),
            self._snapshot(),
        )
        if not result.approved:
            return plan, result.reason or "predicted_separation"
        return plan, None

    def _commit_normal(
        self,
        uav: UAVState,
        target: RingRole | str,
        phase: FlightPhase,
        plan,
        lz_id: int | None,
    ) -> tuple[bool, str]:
        lz_reserved = False
        if target == "final":
            assert lz_id is not None
            reservation = self.env.lz_manager.reserve(lz_id, uav.uid, self.env.now)
            if not reservation.approved:
                return False, "lz_reservation_reject"
            lz_reserved = True
        try:
            proposal = self._proposal(target, plan)
            reservation = self.env.scheduler.reserve(uav, proposal, now=self.env.now)
        except (ValueError, KeyError):
            if lz_reserved:
                self.env.lz_manager.release(lz_id, uav.uid, "headway_conflict")
            return False, "headway_conflict"

        bound_plan = plan.with_reservation_ids((reservation.reservation_id,))
        try:
            committed = self.env._commit(uav, bound_plan)
        except Exception:
            # Safety/hash invariant failures are programming or protocol
            # errors.  Roll back the tentative resources, then preserve the
            # original exception so the caller cannot mistake it for an
            # ordinary candidate rejection.
            self.env.scheduler.cancel(reservation.reservation_id, "safety_exception")
            if lz_reserved:
                self.env.lz_manager.release(lz_id, uav.uid, "safety_exception")
            raise
        if committed:
            uav.merge_reservation_id = reservation.reservation_id
            if lz_id is not None:
                uav.assigned_lz = lz_id
                self.env._record_lz_assignment(uav, lz_id)
            transition(uav, phase, self.env.now)
            uav.ring_role = None
            self._refresh_occupancy()
            return True, "approved"

        self.env.scheduler.cancel(reservation.reservation_id, "safety_reject")
        if lz_reserved:
            self.env.lz_manager.release(lz_id, uav.uid, "safety_reject")
        return False, self.env.last_safety_decisions.get(uav.uid, "safety_reject")

    def release(self, uav: UAVState, intent: LowLevelIntent) -> str:
        """Attempt one normal O->M, M->I, or I->final release."""
        if uav.committed_plan is not None:
            return "plan_active"
        if intent.release_request <= self.env.config.release_threshold:
            return "not_requested"
        source = self._source_target(uav)
        if source is None:
            return "phase_masked"
        interface, target, phase = source
        row = self._row_start(uav, intent, interface)
        if interface in self.env.airspace.affected_interfaces():
            return self._finish(uav, row, "geometry_frozen")
        if uav.soc <= 0.0:
            return self._finish(uav, row, "soc")
        if self.env.now < uav.safety_retry_after:
            return self._finish(
                uav,
                row,
                "safety_wait",
                reason=self.env.last_safety_decisions.get(uav.uid),
            )

        if isinstance(target, RingRole):
            durations = self.RING_DURATIONS[target]
            if target is RingRole.INNER:
                occupied, limit = self._inner_departure_occupancy()
                if occupied >= limit:
                    self._record_candidate(row, durations[0], "inner_departure_capacity")
                    return self._finish(
                        uav,
                        row,
                        "capacity",
                        reason="inner_departure_capacity",
                    )
            if not self._ring_capacity_available(target):
                self._record_candidate(row, durations[0], "capacity")
                return self._finish(uav, row, "capacity", reason="capacity")
            lz_ids: tuple[int | None, ...] = (None,)
        else:
            if not self._final_capacity_available():
                self._record_candidate(row, self.FINAL_DURATIONS[0], "final_capacity")
                return self._finish(uav, row, "final_capacity", reason="final_capacity")
            lz_ids = self._available_lz_ids(uav)
            if not lz_ids:
                self._record_candidate(row, self.FINAL_DURATIONS[0], "lz_unavailable")
                return self._finish(uav, row, "lz_unavailable", reason="lz_unavailable")
            durations = self.FINAL_DURATIONS

        last_reason = "no_certified_candidate"
        spacing_fallback_candidates: list[tuple[float, float]] = []
        spacing_fallback_seen: set[tuple[float, float]] = set()
        for duration in durations:
            # The target ring can fill between candidates in a same-tick
            # arbitration, so recheck its real occupancy before every plan.
            if isinstance(target, RingRole):
                if target is RingRole.INNER:
                    occupied, limit = self._inner_departure_occupancy()
                    if occupied >= limit:
                        self._record_candidate(
                            row, duration, "inner_departure_capacity"
                        )
                        last_reason = "inner_departure_capacity"
                        continue
                if not self._ring_capacity_available(target):
                    self._record_candidate(row, duration, "capacity")
                    last_reason = "capacity"
                    continue
            candidate_lz_ids = lz_ids
            if target == "final":
                candidate_lz_ids = self._available_lz_ids(uav)
                if not candidate_lz_ids:
                    self._record_candidate(row, duration, "lz_unavailable")
                    last_reason = "lz_unavailable"
                    continue
            biases = (
                self.INNER_TANGENTIAL_SPEED_BIASES_MPS
                if target is RingRole.INNER
                else (0.0,)
            )
            for candidate_lz_id in candidate_lz_ids:
                for tangential_speed_bias_mps in biases:
                    row["flow_candidates_checked"] += 1
                    # Keep the historical four-argument call for the
                    # zero-bias and non-M2I paths.  This preserves existing
                    # adapters while allowing only M2I to request repairs.
                    if tangential_speed_bias_mps == 0.0:
                        plan, reason = self._candidate_plan(
                            uav, target, candidate_lz_id, duration
                        )
                    else:
                        plan, reason = self._candidate_plan(
                            uav,
                            target,
                            candidate_lz_id,
                            duration,
                            tangential_speed_bias_mps=tangential_speed_bias_mps,
                        )
                    if reason is not None or plan is None:
                        reason = reason or "candidate_invalid"
                        row["flow_candidate_durations"].append(
                            {
                                "duration": float(duration),
                                "lz_id": candidate_lz_id,
                                "tangential_speed_bias_mps": float(
                                    tangential_speed_bias_mps
                                ),
                                "reason": reason,
                            }
                        )
                        row["flow_candidate_rejection_reasons"].append(reason)
                        if (
                            target is RingRole.INNER
                            and reason == "inner_departure_headway"
                        ):
                            candidate_key = (
                                float(duration),
                                float(tangential_speed_bias_mps),
                            )
                            if candidate_key not in spacing_fallback_seen:
                                spacing_fallback_seen.add(candidate_key)
                                spacing_fallback_candidates.append(candidate_key)
                        last_reason = reason
                        continue
                    committed, commit_reason = self._commit_normal(
                        uav, target, phase, plan, candidate_lz_id
                    )
                    if committed:
                        row["flow_selected_duration"] = float(duration)
                        row["flow_selected_lz_id"] = candidate_lz_id
                        row["flow_selected_tangential_speed_bias_mps"] = float(
                            tangential_speed_bias_mps
                        )
                        row["flow_candidate_durations"].append(
                            {
                                "duration": float(duration),
                                "lz_id": candidate_lz_id,
                                "tangential_speed_bias_mps": float(
                                    tangential_speed_bias_mps
                                ),
                                "reason": "approved",
                            }
                        )
                        return self._finish(
                            uav,
                            row,
                            "approved",
                            plan_id=plan.plan_id,
                        )
                    reason = commit_reason
                    row["flow_candidate_durations"].append(
                        {
                            "duration": float(duration),
                            "lz_id": candidate_lz_id,
                            "tangential_speed_bias_mps": float(
                                tangential_speed_bias_mps
                            ),
                            "reason": reason,
                        }
                    )
                    row["flow_candidate_rejection_reasons"].append(reason)
                    last_reason = reason

        if target is RingRole.INNER and spacing_fallback_candidates:
            for duration, tangential_speed_bias_mps in spacing_fallback_candidates:
                occupied, limit = self._inner_departure_occupancy()
                if occupied >= limit:
                    self._record_candidate(
                        row, duration, "inner_departure_capacity",
                        tangential_speed_bias_mps=tangential_speed_bias_mps,
                    )
                    last_reason = "inner_departure_capacity"
                    continue
                if not self._ring_capacity_available(target):
                    self._record_candidate(
                        row, duration, "capacity",
                        tangential_speed_bias_mps=tangential_speed_bias_mps,
                    )
                    last_reason = "capacity"
                    continue
                if tangential_speed_bias_mps == 0.0:
                    plan, reason = self._candidate_plan(
                        uav,
                        target,
                        None,
                        duration,
                        allow_urgent_spacing=True,
                    )
                else:
                    plan, reason = self._candidate_plan(
                        uav,
                        target,
                        None,
                        duration,
                        tangential_speed_bias_mps=tangential_speed_bias_mps,
                        allow_urgent_spacing=True,
                    )
                row["flow_candidates_checked"] += 1
                if reason is not None or plan is None:
                    reason = reason or "candidate_invalid"
                    row["flow_candidate_durations"].append(
                        {
                            "duration": float(duration),
                            "lz_id": None,
                            "tangential_speed_bias_mps": float(
                                tangential_speed_bias_mps
                            ),
                            "reason": reason,
                        }
                    )
                    row["flow_candidate_rejection_reasons"].append(reason)
                    last_reason = reason
                    continue
                committed, commit_reason = self._commit_normal(
                    uav, target, phase, plan, None
                )
                if committed:
                    row["flow_selected_duration"] = float(duration)
                    row["flow_selected_lz_id"] = None
                    row["flow_selected_tangential_speed_bias_mps"] = float(
                        tangential_speed_bias_mps
                    )
                    row["flow_deadline_fallback"] = True
                    row["flow_candidate_durations"].append(
                        {
                            "duration": float(duration),
                            "lz_id": None,
                            "tangential_speed_bias_mps": float(
                                tangential_speed_bias_mps
                            ),
                            "reason": "approved",
                        }
                    )
                    return self._finish(
                        uav,
                        row,
                        "approved",
                        plan_id=plan.plan_id,
                    )
                reason = commit_reason
                row["flow_candidate_durations"].append(
                    {
                        "duration": float(duration),
                        "lz_id": None,
                        "tangential_speed_bias_mps": float(
                            tangential_speed_bias_mps
                        ),
                        "reason": reason,
                    }
                )
                row["flow_candidate_rejection_reasons"].append(reason)
                last_reason = reason
        if self._energy_guard_only(row["flow_candidate_rejection_reasons"]):
            decision = "emergency_energy_guard"
        else:
            decision = "safety_reject"
        return self._finish(uav, row, decision, reason=last_reason)

    def _normal_energy_guard_only(self, uav: UAVState) -> bool:
        row = self.env.last_release_telemetry.get(uav.uid)
        if not row or not row.get("flow_coordination"):
            return False
        return self._energy_guard_only(row.get("flow_candidate_rejection_reasons", ()))

    @staticmethod
    def _energy_guard_only(reasons) -> bool:
        """Classify energy fallback after removing known geometry failures."""
        relevant = [
            str(reason)
            for reason in reasons
            if str(reason) != "trajectory_infeasible"
        ]
        return bool(relevant) and all(
            reason == "emergency_energy_guard" for reason in relevant
        )

    def _emergency_slack_budget(
        self, uav: UAVState, plan
    ) -> tuple[float, float, float, float]:
        """Return route requirement, slack, wait budget, and slack limit.

        The wait budget covers the larger of the existing retry/headway
        interval and one nominal circuit of the UAV's current ring.  It is a
        local urgency window for the next geometry opportunity; it is not a
        guarantee about future pad or route availability.
        """
        required = float(self.env.energy.emergency_required_energy(uav, plan))
        slack = float(uav.energy_remaining) - required
        role = uav.ring_role
        if not isinstance(role, RingRole):
            role = RingRole(role)
        ring = self.env.airspace.actual_ring(role)
        circuit_seconds = 2.0 * math.pi * float(ring.radius) / float(ring.nominal_speed)
        retry_headway_seconds = (
            2.0 * float(self.env.airspace.headways["i2f"])
            + self.EMERGENCY_RETRY_SECONDS
        )
        wait_seconds = max(retry_headway_seconds, circuit_seconds)
        phase_power = float(self.env.energy.phase_power(uav.phase))
        limit = wait_seconds * phase_power
        return required, slack, wait_seconds, limit

    def _emergency_slack_allowed(self, uav: UAVState, plan) -> bool:
        required, slack, _wait_seconds, limit = self._emergency_slack_budget(uav, plan)
        if uav.energy_remaining + 1e-9 < required:
            self.env.telemetry["emergency_blocked_energy"] += 1
            return False
        return slack <= limit + 1e-9

    def try_emergency(self, uav: UAVState) -> bool:
        """Attempt emergency takeover only when the route slack is tight."""
        if (
            not self.env.config.emergency_landing_enabled
            or uav.emergency_landing
            or uav.assigned_lz is not None
            or uav.ring_role is None
            or uav.committed_plan is not None
            or self.env.now < uav.emergency_retry_after
        ):
            return False
        self.env.telemetry["emergency_attempts"] += 1
        allow_energy_guard_override = self._normal_energy_guard_only(uav)
        zones = self._available_lz_ids(uav)
        if not zones:
            self.env.telemetry["emergency_blocked_lz"] += 1
            uav.emergency_retry_after = self.env.now + self.EMERGENCY_RETRY_SECONDS
            return False
        for duration in (28.0, 36.0, 48.0, 60.0, 90.0):
            for lz_id in zones:
                try:
                    plan = self.trajectory.final_approach(
                        uav,
                        self.env.lz_manager.zones[lz_id],
                        self.env.airspace,
                        self.env.now,
                        duration,
                        emergency=True,
                    )
                except (RuntimeError, ValueError, KeyError):
                    self.env.telemetry["emergency_blocked_dynamics"] += 1
                    continue
                required, slack, wait_seconds, slack_limit = self._emergency_slack_budget(
                    uav, plan
                )
                if uav.energy_remaining + 1e-9 < required:
                    self.env.telemetry["emergency_blocked_energy"] += 1
                    continue
                if slack > slack_limit + 1e-9 and not allow_energy_guard_override:
                    self.env.telemetry["emergency_blocked_slack"] = (
                        self.env.telemetry.get("emergency_blocked_slack", 0) + 1
                    )
                    continue
                safety = self.env.safety.filter_plan(
                    plan,
                    self._committed_plans(),
                    self._snapshot(),
                )
                if not safety.approved:
                    self.env.telemetry["emergency_blocked_safety"] += 1
                    continue
                reservation = self.env.lz_manager.reserve(lz_id, uav.uid, self.env.now)
                if not reservation.approved:
                    continue
                try:
                    committed = self.env._commit(uav, plan)
                except Exception:
                    self.env.lz_manager.release(
                        lz_id, uav.uid, "emergency_safety_exception"
                    )
                    raise
                if not committed:
                    self.env.lz_manager.release(lz_id, uav.uid, "emergency_safety_reject")
                    self.env.telemetry["emergency_blocked_safety"] += 1
                    continue
                pending = self.env.future_release_reservations.pop(uav.uid, None)
                if pending is not None:
                    self.env.scheduler.cancel(pending["reservation_id"], "emergency_landing")
                pre_transition_phase_power = float(
                    self.env.energy.phase_power(uav.phase)
                )
                uav.assigned_lz = lz_id
                uav.emergency_landing = True
                transition(uav, FlightPhase.EMERGENCY_LANDING, self.env.now)
                uav.ring_role = None
                self.env.telemetry["emergency_committed"] += 1
                self.env.telemetry["flow_emergency_redirects"] = (
                    self.env.telemetry.get("flow_emergency_redirects", 0) + 1
                )
                intent_id = self.env._active_release_intents.get(uav.uid)
                if intent_id is not None:
                    row = self.env.release_intents[intent_id]
                    row["flow_emergency_wait_budget_seconds"] = float(wait_seconds)
                    row["flow_emergency_slack_energy"] = float(slack)
                    row["flow_emergency_slack_seconds"] = (
                        float(slack / pre_transition_phase_power)
                        if pre_transition_phase_power > 0.0
                        else None
                    )
                    self.env._finish_release_intent(
                        uav,
                        row,
                        "emergency_redirected",
                        plan_id=plan.plan_id,
                        selection_source="flow_emergency",
                    )
                self.env.last_release_decisions[uav.uid] = "emergency_redirected"
                self.env.events.append(("emergency_departure", uav.uid, self.env.now))
                return True
        uav.emergency_retry_after = self.env.now + self.EMERGENCY_RETRY_SECONDS
        return False

    def _energy_slack_key(self, uav: UAVState) -> tuple[float, int, float, int]:
        role = uav.ring_role
        if role is None:
            return (math.inf, 3, -float(uav.dynamic_priority), int(uav.uid))
        estimated = self._RING_TRAVEL_SECONDS[role] * float(self.env.config.power_reference)
        slack = (
            float(uav.energy_remaining)
            - float(uav.energy_capacity) * float(self.env.config.min_landing_soc)
            - estimated
        )
        downstream_rank = {
            RingRole.INNER: 0,
            RingRole.MIDDLE: 1,
            RingRole.OUTER: 2,
        }[role]
        return (slack, downstream_rank, -float(uav.dynamic_priority), int(uav.uid))

    def processing_order(self) -> list[tuple[int, UAVState]]:
        for uav in self.env.active.values():
            self.env.priority.update(uav, self.env.now)
        downstream_rank = {
            RingRole.INNER: 0,
            RingRole.MIDDLE: 1,
            RingRole.OUTER: 2,
        }

        def key(row: tuple[int, UAVState]) -> tuple[int, float, float, int]:
            _uid, uav = row
            slack, _legacy_rank, priority_key, uid = self._energy_slack_key(uav)
            return (
                downstream_rank.get(uav.ring_role, 3),
                slack,
                priority_key,
                uid,
            )

        return sorted(self.env.active.items(), key=key)

    def emergency_processing_order(self) -> list[UAVState]:
        for uav in self.env.active.values():
            self.env.priority.update(uav, self.env.now)
        return [
            uav
            for _, uav in sorted(
                self.env.active.items(), key=lambda row: self._energy_slack_key(row[1])
            )
        ]
