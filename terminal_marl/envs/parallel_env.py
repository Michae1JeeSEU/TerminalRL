"""Fixed-airspace Phase-A A5 parallel environment.

Only airborne UAVs are active agents.  A low-level terminal reward is emitted
at touchdown; the independent LZ service lifecycle never rewards a terminated
low-level agent.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from copy import deepcopy
import math
from typing import Callable
import numpy as np

from ..core.admission import AdmissionManager
from ..core.arrival_process import ArrivalProcess
from ..core.config import RingRole, SimConfig
from ..core.dynamic_airspace import DynamicAirspace
from ..core.energy import EnergyModel
from ..core.lifecycle import OutcomeLedger, transition
from ..core.landing_zone import LandingZoneManager
from ..core.merge_scheduler import LowLevelIntent, MergeScheduler, MergeProposal, RepairCandidate
from ..core.execution_scheduler import ExecutionScheduler
from ..core.teacher_scheduler import TeacherScheduler
from ..core.action_trace import LowLevelActionTrace
from ..core.observations import build_low_observation
from ..core.topology_neighbors import TopologyCache
from ..core.priority import PriorityEngine
from ..core.rewards import LowLevelReward
from ..core.safety_filter import SafetyFilter
from ..core.runtime_separation import RuntimeSeparationMonitor
from ..core.runtime_traffic_safety import PredictiveSafetyShield
from ..core.trajectory_generator import TrajectoryGenerator
from ..core.forecast_world import ForecastWorld
from ..core.uav import FlightPhase, TerminalOutcome, UAVState
from ..core.uav import QueuedArrivalState


@dataclass
class WorldSnapshot:
    now: float
    active: dict[int, UAVState]
    airspace: DynamicAirspace


class TerminalParallelEnv:
    def __init__(self, config: SimConfig | None = None) -> None:
        self.config = config or SimConfig()
        # Deliberately generic: the core environment must not know about any
        # visualization backend or construct render data on the normal path.
        self._step_observer: Callable[["TerminalParallelEnv"], None] | None = None
        self.energy = EnergyModel(self.config)
        self.priority = PriorityEngine(self.config)
        self.generator = TrajectoryGenerator(self.config)
        self.reset()

    @property
    def agents(self) -> list[str]:
        return [str(uid) for uid in self.active]

    def action_spec(self, _agent: str | None = None) -> dict[str, tuple[int, ...]]:
        """Factorized V3 scheduling categories (not a 50-way joint action)."""
        return {"release": (2,), "speed_residual": (5,), "merge_time_residual": (5,)}

    def set_step_observer(self, observer: Callable[["TerminalParallelEnv"], None] | None) -> None:
        """Register an optional post-tick observer without changing step semantics."""
        self._step_observer = observer

    def clear_step_observer(self) -> None:
        """Remove the optional post-tick observer."""
        self._step_observer = None

    def reset(self, seed: int | None = None):
        self.now = 0.0
        self.step_count = 0
        self.active: dict[int, UAVState] = {}
        self.service_uavs: dict[int, UAVState] = {}
        self.uid = 0
        self.ledger = OutcomeLedger()
        self.request_times: dict[int, float] = {}
        self.resolution_times: dict[int, float] = {}
        self.arrivals = ArrivalProcess(self.config, seed)
        self.admission = AdmissionManager(self.config)
        self.airspace = DynamicAirspace(self.config)
        self.lz_manager = LandingZoneManager()
        self.scheduler = MergeScheduler(self.config)
        self.execution_scheduler = ExecutionScheduler(self.config, self.scheduler)
        self.teacher_scheduler = TeacherScheduler(self.config, self.scheduler) if self.config.teacher_enabled else None
        self.safety = SafetyFilter(self.config)
        self.events: list[tuple[str, int, float]] = []
        self.conflict_events: list[dict] = []
        self.separation_monitor = RuntimeSeparationMonitor(self.config)
        self.runtime_traffic = PredictiveSafetyShield(self.config)
        self.runtime_trace: deque[dict] = deque(maxlen=60)
        self.last_safety_decisions: dict[int, str] = {}
        self.last_release_decisions: dict[int, str] = {}
        self.last_release_telemetry: dict[int, dict] = {}
        self.last_action_traces: dict[int, LowLevelActionTrace] = {}
        self.release_intents: dict[int, dict] = {}
        self._active_release_intents: dict[int, int] = {}
        self._next_release_intent_id = 1
        self.future_release_reservations: dict[int, dict] = {}
        self.telemetry = {
            "arrivals_admitted": 0, "release_approved": 0, "release_rejected": 0, "safety_interventions": 0,
            "merge_time_modified": 0, "geometry_updates": 0,
            "max_concurrent_final": 0, "final_overcapacity_steps": 0, "final_busy_steps": 0,
            "repair_candidates_checked": 0, "repair_used": 0, "future_release_reservations": 0,
            "checked_executed_plan_matches": 0, "checked_executed_plan_samples": 0,
            "checked_executed_plan_hash_mismatches": 0,
            "trajectory_infeasible_candidates": 0,
        }
        self.telemetry.update(self.separation_monitor.telemetry)
        self.telemetry.update(self.runtime_traffic.telemetry)
        self.completed_uavs: list[UAVState] = []
        if self.config.reset_mode == "warm_start_rings" and self.config.warm_start_uavs:
            self._warm_start()
        return self._observations(), {}

    def _warm_start(self) -> None:
        """Create a deterministic, separation-safe curriculum state.

        A warm start is a legal traffic snapshot, not a random scattering of
        aircraft.  Per-ring slot limits enforce the v3 protocol's initial
        ``1.2 * safe_distance`` margin while the deterministic allocation
        preserves ring ordering and replayability.
        """
        requests = self.arrivals.requests_until(0.0)
        if len(requests) < self.config.warm_start_uavs:
            raise ValueError("safe warm-start schedule did not yield every requested initial UAV")
        layout = self._safe_warm_start_layout(self.config.warm_start_uavs)
        for request, (role, theta) in zip(requests[:self.config.warm_start_uavs], layout):
            self.ledger.request(request.arrival_id)
            gate = self.admission.gates[request.gate_id]
            queued = QueuedArrivalState(request, request.energy_capacity * request.initial_soc)
            uav = UAVState.from_request(self.uid, request, queued, gate.spawn_point, self.now)
            self.uid += 1
            ring = self.airspace.actual_ring(role)
            uav.phase = {RingRole.OUTER: FlightPhase.RING_OUTER, RingRole.MIDDLE: FlightPhase.RING_MIDDLE, RingRole.INNER: FlightPhase.RING_INNER}[role]
            uav.ring_role = role; uav.ring_progress = theta
            uav.position = np.array((ring.radius * math.cos(theta), ring.radius * math.sin(theta), ring.height))
            if self.config.warm_start_steady_motion:
                # A populated live scene represents aircraft already
                # established in steady ring flight.  Initialize the complete
                # dynamics state so prediction and execution start identically.
                uav.speed = min(ring.nominal_speed, self.generator.dynamics.ring_speed_limit(ring.radius))
                tangent = np.asarray((-math.sin(theta), math.cos(theta), 0.0), dtype=float)
                radial = np.asarray((-math.cos(theta), -math.sin(theta), 0.0), dtype=float)
                uav.velocity = tangent * uav.speed
                uav.acceleration = radial * (uav.speed * uav.speed / ring.radius)
            self.active[uav.uid] = uav
        self._refresh_airspace_occupancy()

    def _safe_warm_start_layout(self, count: int) -> list[tuple[RingRole, float]]:
        margin = 1.2 * self.config.safe_distance
        capacities: dict[RingRole, int] = {}
        for role in RingRole:
            radius = self.airspace.actual_ring(role).radius
            # Chord distance 2r sin(pi/n) is the minimum separation of n
            # evenly spaced points on one ring.  Search is clearer and less
            # error-prone than inverting the trigonometric constraint.
            maximum = 1
            while 2.0 * radius * math.sin(math.pi / (maximum + 1)) >= margin:
                maximum += 1
            capacities[role] = maximum
        if count > sum(capacities.values()):
            raise ValueError(f"safe warm-start count {count} exceeds {sum(capacities.values())} separation-safe slots")
        assigned = {role: 0 for role in RingRole}
        for _ in range(count):
            role = min(RingRole, key=lambda candidate: (assigned[candidate] / capacities[candidate], list(RingRole).index(candidate)))
            if assigned[role] >= capacities[role]:
                raise ValueError("no separation-safe warm-start ring slot remains")
            assigned[role] += 1
        result: list[tuple[RingRole, float]] = []
        for role in RingRole:
            for index in range(assigned[role]):
                result.append((role, 2.0 * math.pi * index / assigned[role]))
        return result

    def _observations(self) -> dict[str, np.ndarray]:
        cache = None
        if self.config.observation_schema_version == "phaseA_low_obs_v2_topology_87d":
            cache = TopologyCache(self.airspace, self.active.values(), now=self.now,
                                 cpa_horizon=self.config.shield_horizon_seconds)
        return {str(uid): build_low_observation(uav, self, topology_cache=cache) for uid, uav in self.active.items()}

    def _refresh_airspace_occupancy(self) -> None:
        values = {role: 0 for role in RingRole}
        committed = {role: 0 for role in RingRole}
        for uav in self.active.values():
            if uav.ring_role is not None:
                values[uav.ring_role] += 1
            if uav.committed_plan is not None:
                target = uav.committed_plan.metadata.get("target_phase")
                role = {"ring_outer": RingRole.OUTER, "ring_middle": RingRole.MIDDLE, "ring_inner": RingRole.INNER}.get(target)
                if role is not None:
                    committed[role] += 1
        self.airspace.set_occupancy(values, committed)

    def apply_high_level_action(self, action: np.ndarray, *, geometry_gate: bool):
        """Apply a canonical 10-D action without changing any committed plan."""
        self._refresh_airspace_occupancy()
        result = self.airspace.request_update(
            action, now=self.now, occupancies=self.airspace._occupancy,
            committed_incoming=self.airspace._committed, geometry_gate=geometry_gate,
        )
        if result.structural_applied:
            self.telemetry["geometry_updates"] += 1
        self.events.append(("high_action", -1, self.now))
        return result

    def commit_high_level_candidate(self, candidate_airspace, checked_result):
        """Atomically install the exact airspace timeline already checked.

        ReconfigurationSafetyCoordinator stages ``candidate_airspace`` and
        verifies its future actual geometry against protected traffic before
        calling this method.  Reconstructing an update here would allow a
        checked and executed timeline to diverge.
        """
        self.airspace = deepcopy(candidate_airspace)
        if checked_result.structural_applied:
            self.telemetry["geometry_updates"] += 1
        self.events.append(("high_action", -1, self.now))
        return checked_result

    def _resolve(self, uav: UAVState, outcome: TerminalOutcome, reason: str) -> None:
        if uav.assigned_lz is not None:
            self.lz_manager.release(uav.assigned_lz, uav.uid, reason)
        uav.terminal_outcome = outcome
        uav.terminal_reason = reason
        self.ledger.resolve(uav.arrival_id, outcome)
        self.resolution_times[uav.arrival_id] = self.now
        self.active.pop(uav.uid, None)

    def _record_final_occupancy(self) -> None:
        concurrent = sum(uav.phase is FlightPhase.FINAL for uav in self.active.values())
        self.telemetry["max_concurrent_final"] = max(self.telemetry["max_concurrent_final"], concurrent)
        self.telemetry["final_busy_steps"] += int(concurrent > 0)
        self.telemetry["final_overcapacity_steps"] += int(concurrent > self.config.final_queue_capacity)

    def _commit(self, uav: UAVState, plan) -> bool:
        committed = {uid: other.committed_plan for uid, other in self.active.items() if other.committed_plan is not None}
        result = self.safety.filter_plan(plan, committed, WorldSnapshot(self.now, self.active, self.airspace))
        self.last_safety_decisions[uav.uid] = result.reason or "approved"
        if not result.approved:
            self.telemetry["safety_interventions"] += 1
            return False
        if result.checked_plan_id != plan.plan_id or result.checked_trajectory_hash != plan.trajectory_hash:
            raise RuntimeError("A6.12 safety approval did not identify the executed canonical plan")
        uav.committed_plan = plan
        uav.trajectory_plan_id = plan.plan_id
        self.telemetry["checked_executed_plan_matches"] += 1
        return True

    @staticmethod
    def _primary_blocker(decision: str, safety_reason: str | None = None) -> str:
        """One and only one release blocker, independent of diagnostic details."""
        if decision in {"final_capacity", "final_unavailable"}: return "FINAL_RESOURCE"
        if decision in {"lz_unavailable", "lz_reservation_reject"}: return "LZ_RESOURCE"
        if decision == "geometry_frozen": return "INTERFACE_FROZEN"
        if decision in {"headway_conflict"}: return "HEADWAY"
        if decision in {"capacity"}: return "CAPACITY"
        if decision in {"soc"}: return "SOC"
        if safety_reason and "holder" in safety_reason: return "NO_MERGE_GAP"
        if decision in {"safety_wait", "safety_reject"}: return "PREDICTED_SEPARATION"
        return "OTHER"

    def _start_release_intent(self, uav: UAVState, intent: LowLevelIntent, interface: str) -> dict:
        existing = self._active_release_intents.get(uav.uid)
        if existing is None:
            existing = self._next_release_intent_id; self._next_release_intent_id += 1
            self._active_release_intents[uav.uid] = existing
            self.release_intents[existing] = {
                "release_intent_id": existing, "uav_id": uav.uid, "arrival_id": uav.arrival_id,
                "interface": interface, "first_attempt_time": self.now, "approval_time": None,
                "attempt_count": 0, "retry_count": 0, "scheduled_delay": None,
                "repair_used": False, "requested_speed_residual": float(intent.speed_residual),
                "requested_merge_time_residual": float(intent.merge_time_residual),
                "applied_speed_residual": None, "applied_merge_time_residual": None,
                "primary_blocker": None, "secondary_constraints": [], "trajectory_plan_id": None,
                "decision_history": [], "forecast_attempted": False,
            }
        row = self.release_intents[existing]
        row["attempt_count"] += 1
        row["retry_count"] = max(0, row["attempt_count"] - 1)
        return row

    def _finish_release_intent(self, uav: UAVState, row: dict, decision: str, *, candidate: RepairCandidate | None = None,
                               safety_reason: str | None = None, plan_id: str | None = None,
                               secondary: list[str] | None = None) -> None:
        row["decision_history"].append(decision)
        if decision == "approved":
            row.update({"approval_time": self.now, "scheduled_delay": 0.0 if candidate is None else candidate.delay_seconds,
                        "repair_used": bool(candidate is not None and candidate.kind != "requested"),
                        "candidate_kind": None if candidate is None else candidate.kind,
                        "applied_speed_residual": None if candidate is None else float(candidate.intent.speed_residual),
                        "applied_merge_time_residual": None if candidate is None else float(candidate.intent.merge_time_residual),
                        "primary_blocker": None, "trajectory_plan_id": plan_id})
            self._active_release_intents.pop(uav.uid, None)
            if row["repair_used"]: self.telemetry["repair_used"] += 1
            if candidate is not None and candidate.delay_seconds > 0.0: self.telemetry["future_release_reservations"] += 1
        else:
            row["primary_blocker"] = self._primary_blocker(decision, safety_reason)
            row["secondary_constraints"] = list(secondary or ([] if safety_reason is None else [safety_reason]))
        self.last_release_telemetry[uav.uid] = dict(row)

    def _candidate_plan(self, uav: UAVState, candidate: RepairCandidate, target: RingRole | str, snapshot: WorldSnapshot, lz_id: int | None):
        proposal = self.scheduler.propose(uav, candidate.intent, snapshot, target, delay_seconds=candidate.delay_seconds)
        if not self.scheduler.can_reserve(proposal):
            return None, None, "headway_conflict"
        # A lightweight temporary reservation gives the generator the exact
        # slot it would receive on commit.  No scheduler or LZ state changes
        # occur until the candidate has passed the authoritative filter.
        from ..core.merge_scheduler import MergeReservation
        temporary = MergeReservation("candidate", uav.uid, uav.phase.value, str(getattr(target, "value", target)),
                                     proposal.merge_time, proposal.merge_angle, None, None,
                                     status="future" if proposal.merge_time > self.now else "committed", expires_at=proposal.merge_time + 60.0)
        residual = candidate.intent.speed_residual * self.config.max_speed_residual_mps
        try:
            if target == "final":
                plan = self.generator.final_approach(uav, self.lz_manager.zones[lz_id], temporary, self.now,
                                                     scheduled_delay=candidate.delay_seconds, speed_residual_mps=residual,
                                                     airspace=self.airspace)
            else:
                plan = self.generator.ring_transfer(uav, uav.ring_role, target, temporary, self.airspace, self.now, residual)
        except RuntimeError:
            # Canonical parameterization is part of executable feasibility.
            # A failed candidate is rejected fail-closed; it is never turned
            # into an approximate path nor permitted to abort a teacher scan.
            self.telemetry["repair_candidates_checked"] += 1
            self.telemetry["trajectory_infeasible_candidates"] += 1
            return proposal, None, "trajectory_infeasible"
        committed = {uid: other.committed_plan for uid, other in self.active.items() if other.committed_plan is not None}
        self.telemetry["repair_candidates_checked"] += 1
        verdict = self.safety.filter_plan(plan, committed, snapshot)
        return proposal, plan, None if verdict.approved else verdict.reason

    def _forecast_future_repair(self, uav: UAVState, target: RingRole | str, phase: FlightPhase,
                                candidates: tuple[RepairCandidate, ...]) -> dict | None:
        """Find a future slot with a minimal active-traffic forecast world."""
        delayed = [(index, candidate) for index, candidate in enumerate(candidates) if candidate.delay_seconds > 0.0]
        # Simulating the intervening hold is by far the expensive portion.  A
        # common speed residual shares that canonical source-ring evolution,
        # so project it once, then evaluate all merge-time variants from that
        # frozen future state.  We retain the global candidate index and pick
        # the least-intervention successful candidate afterwards.
        base_world = ForecastWorld(self)
        projected: dict[tuple[float, float], ForecastWorld] = {}
        # Candidate order is already minimum-intervention order.  Evaluate it
        # directly and stop at the first verified witness.  A candidate probe
        # is read-only, so merge-time variants sharing one projected world do
        # not require another state copy.
        for _, candidate in delayed:
            key = (candidate.delay_seconds, candidate.intent.speed_residual)
            world = projected.get(key)
            if world is None:
                prior = [item for item in projected if item[1] == candidate.intent.speed_residual
                         and item[0] < candidate.delay_seconds]
                if prior:
                    prior_key = max(prior, key=lambda item: item[0])
                    world = projected[prior_key].fork()
                    remaining = candidate.delay_seconds - prior_key[0]
                else:
                    world = base_world.fork()
                    remaining = candidate.delay_seconds
                world.advance(uav.uid, candidate.intent.speed_residual, remaining)
                projected[key] = world
            proposal, plan, reason = world.candidate_plan(uav.uid, candidate, target, lz_id=None)
            if reason is not None or plan is None or proposal is None or not self.scheduler.can_reserve(proposal):
                continue
            return {"candidate": candidate, "not_before": self.now + candidate.delay_seconds,
                    "proposal": proposal, "forecast_plan_id": plan.plan_id, "phase": phase,
                    "target": target, "forecast_safety": "approved"}
        return None

    def _commit_future_release(self, uav: UAVState, pending: dict) -> str:
        """Atomically turn a future slot into the exact executed plan."""
        row = self.release_intents[self._active_release_intents[uav.uid]]
        reservation = self.scheduler.reservations.get(pending["reservation_id"])
        candidate: RepairCandidate = pending["candidate"]
        target = pending["target"]
        if reservation is None:
            self._finish_release_intent(uav, row, "safety_wait", safety_reason="future_reservation_missing")
            return "safety_wait"
        if target == "final":
            if sum(item.phase is FlightPhase.FINAL for item in self.active.values()) >= self.config.final_queue_capacity:
                self.scheduler.cancel(reservation.reservation_id, "final_capacity")
                self._finish_release_intent(uav, row, "final_capacity")
                return "final_capacity"
            lz_id = self.lz_manager.choose_rule_zone(uav, self.now)
            if lz_id is None or not self.lz_manager.reserve(lz_id, uav.uid, self.now).approved:
                self.scheduler.cancel(reservation.reservation_id, "lz_unavailable")
                self._finish_release_intent(uav, row, "lz_unavailable")
                return "lz_unavailable"
        else:
            lz_id = None
        try:
            if target == "final":
                plan = self.generator.final_approach(uav, self.lz_manager.zones[lz_id], reservation, self.now,
                                                     speed_residual_mps=candidate.intent.speed_residual * self.config.max_speed_residual_mps,
                                                     airspace=self.airspace)
            else:
                plan = self.generator.ring_transfer(uav, uav.ring_role, target, reservation, self.airspace, self.now,
                                                    candidate.intent.speed_residual * self.config.max_speed_residual_mps)
        except RuntimeError:
            if lz_id is not None:
                self.lz_manager.release(lz_id, uav.uid, "trajectory_infeasible")
            self.scheduler.cancel(reservation.reservation_id, "trajectory_infeasible")
            uav.safety_retry_after = self.now + max(5.0, self.airspace.headways[row["interface"]])
            self._finish_release_intent(uav, row, "safety_wait", safety_reason="trajectory_infeasible")
            return "safety_wait"
        if self._commit(uav, plan):
            uav.merge_reservation_id = reservation.reservation_id
            if lz_id is not None: uav.assigned_lz = lz_id
            transition(uav, pending["phase"], self.now); uav.ring_role = None
            self._finish_release_intent(uav, row, "approved", candidate=candidate, plan_id=plan.plan_id)
            return "approved"
        if lz_id is not None: self.lz_manager.release(lz_id, uav.uid, "safety_reject")
        self.scheduler.cancel(reservation.reservation_id, "forecast_recheck_failed")
        uav.safety_retry_after = self.now + max(5.0, self.airspace.headways[row["interface"]])
        self._finish_release_intent(uav, row, "safety_wait", safety_reason=self.last_safety_decisions.get(uav.uid))
        return "safety_wait"

    @staticmethod
    def _phase_pair(left: UAVState, right: UAVState) -> str:
        def label(uav: UAVState) -> str:
            if uav.phase is FlightPhase.INBOUND: return "INBOUND"
            if uav.phase is FlightPhase.FINAL: return "FINAL"
            if uav.committed_plan is not None: return "TRANSFER"
            if uav.ring_role is not None: return "holder"
            return uav.phase.value.upper()
        a, b = label(left), label(right)
        if a == b == "holder" and left.ring_role == right.ring_role: return "same-ring holder-holder"
        return "-".join(sorted((a, b)))

    def _uav_trace(self, uav: UAVState) -> dict:
        return {"uid": uav.uid, "arrival_id": uav.arrival_id, "phase": uav.phase.value,
                "ring_role": None if uav.ring_role is None else uav.ring_role.value,
                "position": uav.position.tolist(), "velocity": uav.velocity.tolist(), "acceleration": uav.acceleration.tolist(), "speed": uav.speed,
                "ring_progress": uav.ring_progress, "entry_gate": uav.assigned_gate,
                "trajectory_plan_id": uav.trajectory_plan_id, "merge_reservation_id": uav.merge_reservation_id,
                "trajectory_hash": None if uav.committed_plan is None else uav.committed_plan.trajectory_hash,
                "reservation_ids": [] if uav.committed_plan is None else list(uav.committed_plan.reservation_ids),
                "committed": uav.committed_plan is not None, "phase_entry_time": uav.phase_entry_time,
                "last_safety_decision": self.last_safety_decisions.get(uav.uid),
                "last_release_decision": self.last_release_decisions.get(uav.uid)}

    def _next_state_prediction(self, intents: dict[int, LowLevelIntent]) -> dict[int, np.ndarray]:
        """Diagnostic-only one-tick predictor sharing executor dynamics.

        It never feeds the transition.  The report makes any future divergence
        between canonical-plan execution and holder integration observable.
        """
        expected: dict[int, np.ndarray] = {}
        for uid, uav in self.active.items():
            plan = uav.committed_plan
            if plan is not None:
                target_time = min(self.now + self.config.dt, plan.end_time)
                endpoint = plan.sample(target_time)
                # A canonical transfer may end within this control interval.
                # Continue its post-handoff nominal ring motion for the
                # remaining fraction, exactly as SafetyFilter's
                # ``_post_plan_positions`` does.  Stopping at the endpoint
                # until the next tick created an unmodelled 0--dt hold.
                target_name = {"ring_outer": "outer", "ring_middle": "middle", "ring_inner": "inner"}.get(
                    plan.metadata.get("target_phase")
                )
                remaining = max(0.0, self.now + self.config.dt - plan.end_time)
                if target_name is None or remaining <= 1e-9:
                    expected[uid] = endpoint.position.copy()
                    continue
                role = next((item for item in RingRole if item.value == target_name), None)
                if role is None:
                    expected[uid] = endpoint.position.copy()
                    continue
                ring = self.airspace.actual_ring(role)
                theta = math.atan2(endpoint.position[1], endpoint.position[0])
                speed = float(np.linalg.norm(endpoint.velocity[:2]))
                target = float(np.clip(ring.nominal_speed + self.airspace.speed_profile_bias,
                                       self.config.speed_min, self.config.max_speed))
                position = endpoint.position.copy()
                elapsed = 0.0
                while elapsed < remaining - 1e-9:
                    dt = min(self.config.physics_dt, remaining - elapsed)
                    motion = self.generator.dynamics.ring_motion_step(ring, theta, speed, target, dt)
                    theta, speed, position = motion.theta, motion.speed, motion.position
                    elapsed += dt
                expected[uid] = np.asarray(position, dtype=float)
                continue
            intent = intents.get(uid)
            if uav.ring_role is None or intent is None:
                continue
            ring = self.airspace.actual_ring(uav.ring_role)
            theta = uav.ring_progress if uav.ring_progress is not None else math.atan2(uav.position[1], uav.position[0])
            speed = max(0.0, uav.speed)
            target = float(np.clip(ring.nominal_speed + self.airspace.speed_profile_bias +
                                   intent.speed_residual * self.config.max_speed_residual_mps,
                                   self.config.speed_min, self.config.max_speed))
            elapsed = 0.0
            position = uav.position.copy()
            while elapsed < self.config.dt - 1e-9:
                dt = min(self.config.physics_dt, self.config.dt - elapsed)
                motion = self.generator.dynamics.ring_motion_step(ring, theta, speed, target, dt)
                theta, speed, position = motion.theta, motion.speed, motion.position
                elapsed += dt
            expected[uid] = np.asarray(position, dtype=float)
        return expected

    def _first_conflict_trace(self, encounter) -> dict:
        left, right = (self.active[uid] for uid in encounter.pair)
        return {"kind": "first_hard_separation_trace", "time": self.now + self.config.dt,
                "pair": encounter.pair, "phase_pair": self._phase_pair(left, right),
                "actual_distance": encounter.actual_distance, "swept_min_distance": encounter.swept_min_distance,
                "swept_fraction": encounter.swept_fraction, "uavs": [self._uav_trace(left), self._uav_trace(right)],
                "recent_trace": list(self.runtime_trace), "lz_state": [{"id": zone.lz_id, "status": zone.status.value,
                    "reserved_uid": zone.reserved_by, "service_remaining": zone.service_remaining} for zone in self.lz_manager.zones],
                "reservations": {key: vars(value).copy() for key, value in self.scheduler.reservations.items()}}

    def _advance_plan(self, uav: UAVState) -> tuple[float, bool]:
        plan = uav.committed_plan
        if plan is None:
            return 0.0, False
        checked_hash = self.safety.checked_trajectory_hashes.get(plan.plan_id)
        if checked_hash != plan.trajectory_hash:
            raise RuntimeError("attempted to execute a trajectory that was not safety-checked by content hash")
        distance = 0.0
        target_time = min(self.now + self.config.dt, plan.end_time)
        sample_times = np.arange(self.now + self.config.physics_dt, target_time + 1e-9, self.config.physics_dt)
        if len(sample_times) == 0 or sample_times[-1] < target_time - 1e-9:
            sample_times = np.append(sample_times, target_time)
        for moment in sample_times:
            if self.safety.checked_trajectory_hashes.get(plan.plan_id) != plan.trajectory_hash:
                self.telemetry["checked_executed_plan_hash_mismatches"] += 1
                raise RuntimeError("trajectory content changed after its safety approval")
            before = uav.position.copy()
            point = plan.sample(float(moment))
            uav.position = point.position
            uav.velocity = point.velocity
            uav.acceleration = point.acceleration
            uav.yaw_rate = 0.0 if point.yaw_rate is None else point.yaw_rate
            uav.speed = float(np.linalg.norm(uav.velocity[:2]))
            distance += float(np.linalg.norm(uav.position - before))
            self.telemetry["checked_executed_plan_samples"] += 1
        uav.path_length += distance
        if self.now + self.config.dt + 1e-9 < plan.end_time:
            return distance, False
        if uav.merge_reservation_id is not None:
            self.scheduler.consume(uav.merge_reservation_id)
            uav.merge_reservation_id = None
        uav.committed_plan = None
        uav.trajectory_plan_id = None
        target = plan.metadata["target_phase"]
        completion_time = float(plan.end_time)
        if target == "ring_outer":
            transition(uav, FlightPhase.RING_OUTER, completion_time)
            uav.ring_role = RingRole.OUTER
        elif target == "ring_middle":
            transition(uav, FlightPhase.RING_MIDDLE, completion_time)
            uav.ring_role = RingRole.MIDDLE
        elif target == "ring_inner":
            transition(uav, FlightPhase.RING_INNER, completion_time)
            uav.ring_role = RingRole.INNER
        elif target == "landing":
            transition(uav, FlightPhase.LANDING, self.now + self.config.dt)
            transition(uav, FlightPhase.SERVICE, self.now + self.config.dt)
            self.lz_manager.touchdown(uav.assigned_lz, uav.uid, uav.mission_type)
            self.service_uavs[uav.uid] = uav
            self.active.pop(uav.uid, None)
            self.events.append(("touchdown", uav.uid, self.now + self.config.dt))
            return distance, True
        uav.ring_progress = math.atan2(uav.position[1], uav.position[0])
        ring = self.airspace.actual_ring(uav.ring_role)
        tangent = np.asarray((-math.sin(uav.ring_progress), math.cos(uav.ring_progress), 0.0), dtype=float)
        radial = np.asarray((-math.cos(uav.ring_progress), -math.sin(uav.ring_progress), 0.0), dtype=float)
        uav.velocity = tangent * uav.speed
        uav.acceleration = radial * (uav.speed * uav.speed / ring.radius)
        # The plan may have completed between executor ticks.  Advance the
        # remainder under the canonical nominal holder policy rather than
        # freezing at the endpoint.  The action for this tick belonged to the
        # already committed transfer; a fresh residual is considered only on
        # the next control tick by the Shield.
        remaining = max(0.0, self.now + self.config.dt - completion_time)
        if remaining > 1e-9:
            distance += self._advance_ring_hold(uav, LowLevelIntent(0.0, 0.0, 0.0), duration=remaining)
        return distance, False

    def _advance_ring_hold(self, uav: UAVState, intent: LowLevelIntent, *, duration: float | None = None) -> float:
        if uav.ring_role is None or uav.committed_plan is not None:
            return 0.0
        ring = self.airspace.actual_ring(uav.ring_role)
        # A ring holder follows the moving *actual* reference during geometry
        # interpolation.  Its committed transfer (if any) is never rewritten.
        theta0 = uav.ring_progress if uav.ring_progress is not None else math.atan2(uav.position[1], uav.position[0])
        target_speed = float(np.clip(
            ring.nominal_speed + self.airspace.speed_profile_bias + intent.speed_residual * self.config.max_speed_residual_mps,
            self.config.speed_min, self.config.max_speed,
        ))
        duration = self.config.dt if duration is None else float(duration)
        theta, speed = theta0, max(0.0, uav.speed)
        elapsed, distance = 0.0, 0.0
        while elapsed < duration - 1e-9:
            step = min(self.config.physics_dt, duration - elapsed)
            before = uav.position.copy()
            motion = self.generator.dynamics.ring_motion_step(ring, theta, speed, target_speed, step)
            theta, speed = motion.theta, motion.speed
            uav.position, uav.velocity, uav.acceleration = motion.position, motion.velocity, motion.acceleration
            uav.speed = speed
            distance += float(np.linalg.norm(uav.position - before))
            elapsed += step
        uav.ring_progress = theta % (2 * math.pi)
        uav.path_length += distance
        uav.internal_wait_time += duration
        uav.holding_time_by_ring[uav.ring_role] = uav.holding_time_by_ring.get(uav.ring_role, 0.0) + duration
        return distance

    def _release_legacy(self, uav: UAVState, intent: LowLevelIntent) -> str:
        """Unchanged single-candidate scheduler for the A6.12 CRN control."""
        if intent.release_request <= self.config.release_threshold: return "not_requested"
        if uav.committed_plan is not None: return "plan_active"
        if uav.phase is FlightPhase.RING_OUTER: interface = "o2m"
        elif uav.phase is FlightPhase.RING_MIDDLE: interface = "m2i"
        elif uav.phase is FlightPhase.RING_INNER: interface = "i2f"
        else: return "phase_masked"
        row = self._start_release_intent(uav, intent, interface)
        def finish(decision: str, reason: str | None = None) -> str:
            self._finish_release_intent(uav, row, decision, candidate=RepairCandidate("requested", intent) if decision == "approved" else None,
                                        safety_reason=reason, plan_id=uav.trajectory_plan_id if decision == "approved" else None)
            return decision
        if self.now < uav.safety_retry_after: return finish("safety_wait", self.last_safety_decisions.get(uav.uid))
        snapshot = WorldSnapshot(self.now, self.active, self.airspace); residual = intent.speed_residual * self.config.max_speed_residual_mps
        if uav.phase is FlightPhase.RING_OUTER:
            if "o2m" in self.airspace.affected_interfaces(): return finish("geometry_frozen")
            target, phase = RingRole.MIDDLE, FlightPhase.TRANSFER_O2M
        elif uav.phase is FlightPhase.RING_MIDDLE:
            if "m2i" in self.airspace.affected_interfaces(): return finish("geometry_frozen")
            target, phase = RingRole.INNER, FlightPhase.TRANSFER_M2I
        else:
            if "i2f" in self.airspace.affected_interfaces(): return finish("geometry_frozen")
            if sum(item.phase is FlightPhase.FINAL for item in self.active.values()) >= self.config.final_queue_capacity: return finish("final_capacity")
            lz_id = self.lz_manager.choose_rule_zone(uav, self.now)
            if lz_id is None: return finish("lz_unavailable")
            if not self.lz_manager.reserve(lz_id, uav.uid, self.now).approved: return finish("lz_reservation_reject")
            proposal = self.execution_scheduler.propose_requested(uav, intent, snapshot, "final")
            if not self.execution_scheduler.check_requested(proposal).approved:
                self.lz_manager.release(lz_id, uav.uid, "headway_conflict")
                return finish("headway_conflict")
            reservation = self.execution_scheduler.reserve_requested(uav, proposal, now=self.now)
            plan = self.generator.final_approach(uav, self.lz_manager.zones[lz_id], reservation, self.now,
                                                 airspace=self.airspace)
            if self._commit(uav, plan):
                uav.assigned_lz = lz_id; uav.merge_reservation_id = reservation.reservation_id; transition(uav, FlightPhase.FINAL, self.now); uav.ring_role = None
                return finish("approved")
            self.lz_manager.release(lz_id, uav.uid, "safety_reject"); self.execution_scheduler.cancel(reservation.reservation_id, "safety_reject")
            uav.safety_retry_after = self.now + 5.0
            return finish("safety_reject", self.last_safety_decisions.get(uav.uid))
        proposal = self.execution_scheduler.propose_requested(uav, intent, snapshot, target)
        if not self.execution_scheduler.check_requested(proposal).approved:
            return finish("headway_conflict")
        reservation = self.execution_scheduler.reserve_requested(uav, proposal, now=self.now)
        plan = self.generator.ring_transfer(uav, uav.ring_role, target, reservation, self.airspace, self.now, residual)
        if self._commit(uav, plan):
            uav.merge_reservation_id = reservation.reservation_id; transition(uav, phase, self.now); uav.ring_role = None
            return finish("approved")
        self.execution_scheduler.cancel(reservation.reservation_id, "safety_reject")
        uav.safety_retry_after = self.now + max(5.0, self.airspace.headways[interface])
        return finish("safety_reject", self.last_safety_decisions.get(uav.uid))

    def _release(self, uav: UAVState, intent: LowLevelIntent) -> str:
        # Label-only is intentionally execution-equivalent to student mode;
        # training code may query TeacherScheduler separately for an auxiliary
        # target, but it must not turn the label into an environment action.
        if not self.config.teacher_enabled or self.config.teacher_mode == "teacher_label_only":
            return self._release_legacy(uav, intent)
        if intent.release_request <= self.config.release_threshold:
            return "not_requested"
        if uav.committed_plan is not None:
            return "plan_active"
        pending = self.future_release_reservations.get(uav.uid)
        if pending is not None:
            if self.now + 1e-9 < pending["not_before"]:
                row = self.release_intents[self._active_release_intents[uav.uid]]
                row["decision_history"].append("future_reservation_wait")
                self.last_release_telemetry[uav.uid] = dict(row)
                return "future_reservation_wait"
            self.future_release_reservations.pop(uav.uid, None)
            return self._commit_future_release(uav, pending)
        if uav.phase is FlightPhase.RING_OUTER:
            interface, target, phase = "o2m", RingRole.MIDDLE, FlightPhase.TRANSFER_O2M
        elif uav.phase is FlightPhase.RING_MIDDLE:
            interface, target, phase = "m2i", RingRole.INNER, FlightPhase.TRANSFER_M2I
        elif uav.phase is FlightPhase.RING_INNER:
            interface, target, phase = "i2f", "final", FlightPhase.FINAL
        else:
            return "phase_masked"
        row = self._start_release_intent(uav, intent, interface)
        if interface in self.airspace.affected_interfaces():
            self._finish_release_intent(uav, row, "geometry_frozen")
            return "geometry_frozen"
        if uav.soc <= 0.0:
            self._finish_release_intent(uav, row, "soc")
            return "soc"
        if self.now < uav.safety_retry_after:
            self._finish_release_intent(uav, row, "safety_wait", safety_reason=self.last_safety_decisions.get(uav.uid))
            return "safety_wait"
        lz_id = None
        if target == "final":
            if sum(item.phase is FlightPhase.FINAL for item in self.active.values()) >= self.config.final_queue_capacity:
                self._finish_release_intent(uav, row, "final_capacity")
                return "final_capacity"
            lz_id = self.lz_manager.choose_rule_zone(uav, self.now)
            if lz_id is None:
                self._finish_release_intent(uav, row, "lz_unavailable")
                return "lz_unavailable"
        snapshot = WorldSnapshot(self.now, self.active, self.airspace)
        candidates = self.teacher_scheduler.candidates(intent) if self.teacher_scheduler is not None else (RepairCandidate("requested", intent),)
        # The immediate phase is deliberately closed before future-delay
        # forecasting.  Delayed candidates already contain a source-holder
        # prefix, so generating every one here used to repeat full canonical
        # trajectory construction before any current repair had been ruled
        # out.
        immediate_candidates = tuple(candidate for candidate in candidates if candidate.delay_seconds == 0.0)
        last_reason = None
        for candidate in immediate_candidates:
            proposal, plan, reason = self._candidate_plan(uav, candidate, target, snapshot, lz_id)
            if reason is not None:
                last_reason = reason
                continue
            # This is the atomic commit point: resources are reserved only
            # after the exact canonical plan passed candidate verification.
            if target == "final":
                if not self.lz_manager.reserve(lz_id, uav.uid, self.now).approved:
                    last_reason = "lz_reservation_reject"; continue
            try:
                reservation = self.scheduler.reserve(uav, proposal, now=self.now)
            except ValueError:
                if target == "final": self.lz_manager.release(lz_id, uav.uid, "headway_conflict")
                last_reason = "headway_conflict"; continue
            # Re-run the same plan through SafetyFilter at the atomic state;
            # _commit installs this very object, never a reconstructed path.
            plan = plan.with_reservation_ids((reservation.reservation_id,))
            if self._commit(uav, plan):
                uav.merge_reservation_id = reservation.reservation_id
                if target == "final": uav.assigned_lz = lz_id
                transition(uav, phase, self.now); uav.ring_role = None
                self._finish_release_intent(uav, row, "approved", candidate=candidate, plan_id=plan.plan_id)
                return "approved"
            if target == "final": self.lz_manager.release(lz_id, uav.uid, "safety_reject")
            self.scheduler.cancel(reservation.reservation_id, "safety_reject")
            last_reason = self.last_safety_decisions.get(uav.uid, "predicted_separation")
        # A single bounded forecast belongs to one release intent.  Repeating
        # the same expensive projection after every retry would recreate the
        # old retry storm under a different name.
        forecast = None if row["forecast_attempted"] else self._forecast_future_repair(uav, target, phase, candidates)
        row["forecast_attempted"] = True
        if forecast is not None:
            reservation = self.scheduler.reserve(uav, forecast["proposal"], now=self.now)
            forecast["reservation_id"] = reservation.reservation_id
            self.future_release_reservations[uav.uid] = forecast
            row.update({"scheduled_delay": forecast["candidate"].delay_seconds, "repair_used": True,
                        "applied_speed_residual": forecast["candidate"].intent.speed_residual,
                        "applied_merge_time_residual": forecast["candidate"].intent.merge_time_residual,
                        "trajectory_plan_id": forecast["forecast_plan_id"]})
            row["decision_history"].append("future_reserved")
            self.last_release_telemetry[uav.uid] = dict(row)
            self.telemetry["future_release_reservations"] += 1
            return "future_reserved"
        # A full bounded search failed.  This is a deterministic backoff for
        # a truly currently infeasible intent, not the old one-second retry.
        uav.safety_retry_after = self.now + max(5.0, self.airspace.headways[interface])
        decision = "safety_wait" if self.config.teacher_enabled else "safety_reject"
        self._finish_release_intent(uav, row, decision, safety_reason=last_reason)
        return decision

    def step(self, actions: dict[str, np.ndarray] | None = None):
        self.safety.begin_prediction_step()
        previous = dict(self.active)
        self.last_release_telemetry = {}
        rewards = {uid: LowLevelReward(time=-self.config.low_level_time_penalty_per_second * self.config.dt) for uid in previous}
        actions = actions or {}
        intents: dict[int, LowLevelIntent] = {}
        traces: dict[int, LowLevelActionTrace] = {}
        decisions: dict[int, str] = {}
        action_rows = []
        for request in self.arrivals.requests_until(self.now):
            self.ledger.request(request.arrival_id); self.request_times[request.arrival_id] = request.request_time; self.admission.enqueue(request)
        self.admission.queue.step(self.config.dt, self.energy, self.priority, self.now)
        self.execution_scheduler.expire(self.now, set(self.active))
        self._refresh_airspace_occupancy()
        self.airspace.step_transition(self.now)
        admitted, records = self.admission.step(self.now, self.active, self.airspace, self.energy, self.priority)
        for record in records:
            if record.decision.value == "divert":
                outcome = TerminalOutcome.DIVERTED_LOW_SOC if record.reason_code == "low_soc" else TerminalOutcome.DIVERTED_CAPACITY
                self.ledger.resolve(record.request.arrival_id, outcome)
                self.resolution_times[record.request.arrival_id] = self.now
        for queued, gate in admitted:
            uav = UAVState.from_request(self.uid, queued.request, queued, gate.spawn_point, self.now)
            self.uid += 1; self.active[uav.uid] = uav
            if not self._commit(uav, self.generator.inbound(uav, gate, self.airspace, self.now)):
                # Admission must never create a plan-less airborne agent.  Put
                # this request back in the external queue for a deterministic
                # retry rather than letting two inbound spawns overlap.
                self.active.pop(uav.uid, None)
                self.admission.queue.items.append(queued)
                gate.last_admission_time = min(gate.last_admission_time, self.now - self.config.gate_min_time_gap_seconds)
            else:
                self.telemetry["arrivals_admitted"] += 1
        # Service is global only; it does not resurrect terminated low-level agents.
        for uid in self.lz_manager.step(self.config.dt):
            uav = self.service_uavs.pop(uid)
            transition(uav, FlightPhase.COMPLETED, self.now + self.config.dt)
            uav.completion_time = self.now + self.config.dt; uav.terminal_outcome = TerminalOutcome.COMPLETED
            self.ledger.resolve(uav.arrival_id, TerminalOutcome.COMPLETED)
            self.resolution_times[uav.arrival_id] = self.now + self.config.dt
            self.completed_uavs.append(uav)
            self.events.append(("completed", uid, self.now + self.config.dt))
        pre_motion_positions = {uid: uav.position.copy() for uid, uav in self.active.items()}
        for uid, uav in list(self.active.items()):
            intent = LowLevelIntent.from_array(actions.get(str(uid), np.array([1.0, 0.0, 0.0])))
            intents[uid] = intent; traces[uid] = LowLevelActionTrace(requested=intent); self.priority.update(uav, self.now)
            decision = self._release(uav, intent); decisions[uid] = decision
            self.last_release_decisions[uid] = decision
            release_row = self.last_release_telemetry.get(uid)
            if release_row and release_row.get("repair_used"):
                corrected = LowLevelIntent(intent.release_request, float(release_row["applied_speed_residual"]), float(release_row["applied_merge_time_residual"]))
                traces[uid].teacher_corrected = corrected
                traces[uid].execution_applied = corrected
                traces[uid].teacher_used = True
                traces[uid].teacher_kind = str(release_row.get("candidate_kind", "teacher_repair"))
                traces[uid].teacher_delay_seconds = float(release_row.get("scheduled_delay") or 0.0)
                traces[uid].teacher_witness_plan_id = release_row.get("trajectory_plan_id")
                traces[uid].teacher_verified = traces[uid].teacher_witness_plan_id is not None
                traces[uid].correction_magnitude = abs(corrected.speed_residual-intent.speed_residual) + abs(corrected.merge_time_residual-intent.merge_time_residual)
                rewards.setdefault(uid, LowLevelReward()).teacher -= self.config.teacher_intervention_penalty + self.config.teacher_correction_magnitude_penalty * traces[uid].correction_magnitude
            pending = self.future_release_reservations.get(uid)
            if pending is not None and decision in {"future_reserved", "future_reservation_wait"}:
                # This is the source-holder portion of the future canonical
                # release, not an unmodelled temporary avoidance manoeuvre.
                candidate = pending["candidate"]
                intents[uid] = LowLevelIntent(0.0, candidate.intent.speed_residual, 0.0)
            if intent.release_request > self.config.release_threshold:
                rewards.setdefault(uid, LowLevelReward()).release += self.config.low_level_release_approved_reward if decision == "approved" else -self.config.low_level_release_rejected_penalty
                if decision == "approved": self.telemetry["release_approved"] += 1
                else: self.telemetry["release_rejected"] += 1
            if abs(intent.merge_time_residual) > 1e-8: self.telemetry["merge_time_modified"] += 1
            action_rows.append((intent.release_request, intent.speed_residual, intent.merge_time_residual, decision))
        if self.config.shield_enabled:
            intents = self.runtime_traffic.protect(self, intents, traces)
            self.telemetry.update(self.runtime_traffic.telemetry)
        for uid, trace in traces.items():
            if trace.shield_used:
                magnitude = abs(trace.shield_applied.speed_residual-trace.execution_applied.speed_residual)
                rewards.setdefault(uid, LowLevelReward()).shield -= self.config.shield_intervention_penalty + self.config.shield_correction_magnitude_penalty*magnitude
        self.last_action_traces = traces
        predicted_positions = self._next_state_prediction(intents)
        for uid, uav in list(self.active.items()):
            phase = uav.phase
            had_committed_plan = uav.committed_plan is not None
            distance, touchdown = self._advance_plan(uav)
            # A canonical plan already advances to its endpoint for this
            # simulation interval.  Do not append an additional full holder
            # step on the transition tick; that would execute a state which
            # was never safety-checked.
            if uid in intents and not had_committed_plan:
                distance += self._advance_ring_hold(uav, intents[uid])
            self.energy.consume(uav, phase, self.config.dt)
            if uid in rewards: rewards[uid].progress += distance * self.config.low_level_progress_reward_per_meter
            if touchdown and uid in rewards: rewards[uid].touchdown += self.config.low_level_touchdown_reward
            if uav.energy_remaining <= 0 and uid in self.active:
                self._resolve(uav, TerminalOutcome.FAILED_ENERGY, "energy_depleted")
        encounters = self.separation_monitor.observe(self.active, pre_motion_positions)
        self.telemetry.update(self.separation_monitor.telemetry)
        prediction_execution_error = {
            str(uid): float(np.linalg.norm(prediction - self.active[uid].position))
            for uid, prediction in predicted_positions.items() if uid in self.active
        }
        for encounter in encounters:
            if any(uid not in self.active for uid in encounter.pair):
                continue
            if encounter.hard_entry and not any(item.get("kind") == "first_hard_separation_trace" for item in self.conflict_events):
                self.conflict_events.append(self._first_conflict_trace(encounter))
            if encounter.nmac_entry or encounter.collision_entry:
                left, right = (self.active[uid] for uid in encounter.pair)
                kind = "collision" if encounter.collision_entry else "nmac_entry"
                self.conflict_events.append({"kind": kind, "time": self.now + self.config.dt, "uids": encounter.pair,
                    "arrival_ids": (left.arrival_id, right.arrival_id), "distance": encounter.actual_distance,
                    "swept_min_distance": encounter.swept_min_distance, "phases": (left.phase.value, right.phase.value),
                    "plan_ids": (left.trajectory_plan_id, right.trajectory_plan_id)})
            if encounter.collision_entry:
                for uid in encounter.pair:
                    if uid in self.active: self._resolve(self.active[uid], TerminalOutcome.FAILED_COLLISION, "collision")
        self._record_final_occupancy()
        self.runtime_trace.append({"time": self.now + self.config.dt, "active": [self._uav_trace(uav) for uav in self.active.values()],
            "release_decisions": dict(decisions), "release_intents": {str(uid): dict(row) for uid, row in self.last_release_telemetry.items()},
            "action_traces": {str(uid): trace.as_dict() for uid, trace in traces.items()},
            "prediction_execution_position_error": prediction_execution_error,
            "telemetry": dict(self.telemetry)})
        for uid, uav in previous.items():
            if uav.terminal_outcome is TerminalOutcome.FAILED_COLLISION: rewards[uid].failure -= self.config.low_level_collision_penalty
            if uav.terminal_outcome is TerminalOutcome.FAILED_ENERGY: rewards[uid].failure -= self.config.low_level_energy_failure_penalty
        self.now += self.config.dt; self.step_count += 1; self._refresh_airspace_occupancy()
        observation = self._observations()
        # This is intentionally after every canonical state update and before
        # returning the existing API value.  Disabled runs only pay this null
        # check; snapshot creation belongs exclusively to an attached observer.
        if self._step_observer is not None:
            self._step_observer(self)
        return observation, {str(uid): r.total for uid, r in rewards.items()}, {str(uid): uid not in self.active for uid in previous}, {str(uid): self.now >= self.config.max_simulation_seconds for uid in previous}, {
            "time": self.now, "queue_size": len(self.admission.queue.items), "outcomes": dict(self.ledger.outcomes),
            "reward_components": {str(uid): r.as_dict() for uid, r in rewards.items()}, "release_decisions": {str(uid): x for uid, x in decisions.items()},
            "release_intent_telemetry": {str(uid): dict(row) for uid, row in self.last_release_telemetry.items()},
            "actions": action_rows, "action_traces": {str(uid): trace.as_dict() for uid, trace in traces.items()},
            "prediction_execution_position_error": prediction_execution_error,
            "telemetry": dict(self.telemetry), "conflict_events": list(self.conflict_events),
        }

    def run_rule_only(self, max_steps: int | None = None) -> OutcomeLedger:
        for _ in range(max_steps or int(self.config.max_simulation_seconds / self.config.dt)):
            self.step()
            if self.arrivals.exhausted() and not self.active and not self.admission.queue.items and not self.service_uavs: break
        return self.ledger
