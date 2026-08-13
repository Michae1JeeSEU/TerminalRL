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
import numpy as np

from ..core.admission import AdmissionManager
from ..core.arrival_process import ArrivalProcess
from ..core.config import RingRole, SimConfig
from ..core.dynamic_airspace import DynamicAirspace
from ..core.energy import EnergyModel
from ..core.lifecycle import OutcomeLedger, transition
from ..core.landing_zone import LandingZoneManager
from ..core.merge_scheduler import LowLevelIntent, MergeScheduler
from ..core.observations import build_a5_observation
from ..core.priority import PriorityEngine
from ..core.rewards import LowLevelReward
from ..core.safety_filter import SafetyFilter
from ..core.runtime_separation import RuntimeSeparationMonitor
from ..core.runtime_traffic_safety import RuntimeTrafficSafetyController
from ..core.trajectory_generator import TrajectoryGenerator
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
        self.energy = EnergyModel(self.config)
        self.priority = PriorityEngine(self.config)
        self.generator = TrajectoryGenerator(self.config)
        self.reset()

    @property
    def agents(self) -> list[str]:
        return [str(uid) for uid in self.active]

    def action_spec(self, _agent: str | None = None) -> tuple[int]:
        return (3,)

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
        self.safety = SafetyFilter(self.config)
        self.events: list[tuple[str, int, float]] = []
        self.conflict_events: list[dict] = []
        self.separation_monitor = RuntimeSeparationMonitor(self.config)
        self.runtime_traffic = RuntimeTrafficSafetyController(self.config)
        self.runtime_trace: deque[dict] = deque(maxlen=60)
        self.last_safety_decisions: dict[int, str] = {}
        self.last_release_decisions: dict[int, str] = {}
        self.telemetry = {
            "release_approved": 0, "release_rejected": 0, "safety_interventions": 0,
            "merge_time_modified": 0, "geometry_updates": 0,
            "max_concurrent_final": 0, "final_overcapacity_steps": 0, "final_busy_steps": 0,
        }
        self.telemetry.update(self.separation_monitor.telemetry)
        self.telemetry.update(self.runtime_traffic.telemetry)
        self.completed_uavs: list[UAVState] = []
        if self.config.reset_mode == "warm_start_rings" and self.config.warm_start_uavs:
            self._warm_start()
        return self._observations(), {}

    def _warm_start(self) -> None:
        """Curriculum-only seeded ring state; formal evaluation stays flow-from-empty."""
        requests = self.arrivals.requests_until(0.0)
        for index, request in enumerate(requests[:self.config.warm_start_uavs]):
            self.ledger.request(request.arrival_id)
            gate = self.admission.gates[request.gate_id]
            queued = QueuedArrivalState(request, request.energy_capacity * request.initial_soc)
            uav = UAVState.from_request(self.uid, request, queued, gate.spawn_point, self.now)
            self.uid += 1
            role = (RingRole.OUTER, RingRole.MIDDLE, RingRole.INNER)[index % 3]
            ring = self.airspace.actual_ring(role)
            theta = 2 * math.pi * index / max(1, self.config.warm_start_uavs)
            uav.phase = {RingRole.OUTER: FlightPhase.RING_OUTER, RingRole.MIDDLE: FlightPhase.RING_MIDDLE, RingRole.INNER: FlightPhase.RING_INNER}[role]
            uav.ring_role = role; uav.ring_progress = theta
            uav.position = np.array((ring.radius * math.cos(theta), ring.radius * math.sin(theta), ring.height))
            self.active[uav.uid] = uav
        self._refresh_airspace_occupancy()

    def _observations(self) -> dict[str, np.ndarray]:
        return {str(uid): build_a5_observation(uav, self) for uid, uav in self.active.items()}

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
        uav.committed_plan = plan
        uav.trajectory_plan_id = plan.plan_id
        return True

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
                "position": uav.position.tolist(), "velocity": uav.velocity.tolist(), "speed": uav.speed,
                "ring_progress": uav.ring_progress, "entry_gate": uav.assigned_gate,
                "trajectory_plan_id": uav.trajectory_plan_id, "merge_reservation_id": uav.merge_reservation_id,
                "committed": uav.committed_plan is not None, "phase_entry_time": uav.phase_entry_time,
                "last_safety_decision": self.last_safety_decisions.get(uav.uid),
                "last_release_decision": self.last_release_decisions.get(uav.uid)}

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
        before = uav.position.copy()
        point = plan.sample(min(self.now + self.config.dt, plan.end_time))
        uav.position = point.position
        uav.velocity = point.velocity if point.velocity is not None else np.zeros(3)
        uav.speed = float(np.linalg.norm(uav.velocity))
        distance = float(np.linalg.norm(uav.position - before))
        uav.path_length += distance
        if self.now + self.config.dt + 1e-9 < plan.end_time:
            return distance, False
        if uav.merge_reservation_id is not None:
            self.scheduler.consume(uav.merge_reservation_id)
            uav.merge_reservation_id = None
        uav.committed_plan = None
        uav.trajectory_plan_id = None
        target = plan.metadata["target_phase"]
        if target == "ring_outer":
            transition(uav, FlightPhase.RING_OUTER, self.now + self.config.dt)
            uav.ring_role = RingRole.OUTER
        elif target == "ring_middle":
            transition(uav, FlightPhase.RING_MIDDLE, self.now + self.config.dt)
            uav.ring_role = RingRole.MIDDLE
        elif target == "ring_inner":
            transition(uav, FlightPhase.RING_INNER, self.now + self.config.dt)
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
        return distance, False

    def _advance_ring_hold(self, uav: UAVState, intent: LowLevelIntent) -> float:
        if uav.ring_role is None or uav.committed_plan is not None:
            return 0.0
        ring = self.airspace.actual_ring(uav.ring_role)
        # A ring holder follows the moving *actual* reference during geometry
        # interpolation.  Its committed transfer (if any) is never rewritten.
        before = uav.position.copy()
        theta0 = uav.ring_progress if uav.ring_progress is not None else math.atan2(uav.position[1], uav.position[0])
        uav.position = np.array((ring.radius * math.cos(theta0), ring.radius * math.sin(theta0), ring.height))
        speed = float(np.clip(
            ring.nominal_speed + self.airspace.speed_profile_bias + intent.speed_residual * self.config.max_speed_residual_mps,
            self.config.speed_min, self.config.max_speed,
        ))
        theta = theta0 + speed * self.config.dt / ring.radius
        uav.ring_progress = theta % (2 * math.pi)
        uav.position = np.array((ring.radius * math.cos(uav.ring_progress), ring.radius * math.sin(uav.ring_progress), ring.height))
        uav.velocity = (uav.position - before) / self.config.dt
        uav.speed = speed
        distance = float(np.linalg.norm(uav.position - before))
        uav.path_length += distance
        uav.internal_wait_time += self.config.dt
        uav.holding_time_by_ring[uav.ring_role] = uav.holding_time_by_ring.get(uav.ring_role, 0.0) + self.config.dt
        return distance

    def _release(self, uav: UAVState, intent: LowLevelIntent) -> str:
        if intent.release_request <= self.config.release_threshold:
            return "not_requested"
        if uav.committed_plan is not None:
            return "plan_active"
        if self.now < uav.safety_retry_after:
            return "safety_wait"
        snapshot = WorldSnapshot(self.now, self.active, self.airspace)
        residual = intent.speed_residual * self.config.max_speed_residual_mps
        if uav.phase is FlightPhase.RING_OUTER:
            if "o2m" in self.airspace.affected_interfaces(): return "geometry_frozen"
            target, phase = RingRole.MIDDLE, FlightPhase.TRANSFER_O2M
        elif uav.phase is FlightPhase.RING_MIDDLE:
            if "m2i" in self.airspace.affected_interfaces(): return "geometry_frozen"
            target, phase = RingRole.INNER, FlightPhase.TRANSFER_M2I
        elif uav.phase is FlightPhase.RING_INNER:
            if "i2f" in self.airspace.affected_interfaces(): return "geometry_frozen"
            if sum(item.phase is FlightPhase.FINAL for item in self.active.values()) >= self.config.final_queue_capacity:
                return "final_capacity"
            lz_id = self.lz_manager.choose_rule_zone(uav, self.now)
            if lz_id is None: return "lz_unavailable"
            if not self.lz_manager.reserve(lz_id, uav.uid, self.now).approved: return "lz_reservation_reject"
            reservation = self.scheduler.reserve(uav, self.scheduler.propose(uav, intent, snapshot, "final"))
            plan = self.generator.final_approach(uav, self.lz_manager.zones[lz_id], reservation, self.now)
            if self._commit(uav, plan):
                uav.assigned_lz = lz_id; uav.merge_reservation_id = reservation.reservation_id; transition(uav, FlightPhase.FINAL, self.now); uav.ring_role = None
                return "approved"
            self.lz_manager.release(lz_id, uav.uid, "safety_reject"); self.scheduler.cancel(reservation.reservation_id, "safety_reject")
            uav.safety_retry_after = self.now + 5.0
            return "safety_reject"
        else:
            return "phase_masked"
        reservation = self.scheduler.reserve(uav, self.scheduler.propose(uav, intent, snapshot, target))
        plan = self.generator.ring_transfer(uav, uav.ring_role, target, reservation, self.airspace, self.now, residual)
        if self._commit(uav, plan):
            uav.merge_reservation_id = reservation.reservation_id; transition(uav, phase, self.now); uav.ring_role = None
            return "approved"
        self.scheduler.cancel(reservation.reservation_id, "safety_reject")
        # A rejected canonical plan is a deterministic hold-future-release
        # decision.  Back off by the affected interface headway rather than
        # retrying the exact same unsafe merge on every low-level step.
        key = {RingRole.MIDDLE: "o2m", RingRole.INNER: "m2i"}[target]
        uav.safety_retry_after = self.now + max(5.0, self.airspace.headways[key])
        return "safety_reject"

    def step(self, actions: dict[str, np.ndarray] | None = None):
        previous = dict(self.active)
        rewards = {uid: LowLevelReward(time=-self.config.low_level_time_penalty_per_second * self.config.dt) for uid in previous}
        actions = actions or {}
        intents: dict[int, LowLevelIntent] = {}
        decisions: dict[int, str] = {}
        action_rows = []
        for request in self.arrivals.requests_until(self.now):
            self.ledger.request(request.arrival_id); self.request_times[request.arrival_id] = request.request_time; self.admission.enqueue(request)
        self.admission.queue.step(self.config.dt, self.energy, self.priority, self.now)
        self.scheduler.expire(self.now, set(self.active))
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
            intents[uid] = intent; self.priority.update(uav, self.now)
            decision = self._release(uav, intent); decisions[uid] = decision
            self.last_release_decisions[uid] = decision
            if intent.release_request > self.config.release_threshold:
                rewards.setdefault(uid, LowLevelReward()).release += self.config.low_level_release_approved_reward if decision == "approved" else -self.config.low_level_release_rejected_penalty
                if decision == "approved": self.telemetry["release_approved"] += 1
                else: self.telemetry["release_rejected"] += 1
            if abs(intent.merge_time_residual) > 1e-8: self.telemetry["merge_time_modified"] += 1
            action_rows.append((intent.release_request, intent.speed_residual, intent.merge_time_residual, decision))
        intents = self.runtime_traffic.protect(self, intents)
        self.telemetry.update(self.runtime_traffic.telemetry)
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
            "release_decisions": dict(decisions), "telemetry": dict(self.telemetry)})
        for uid, uav in previous.items():
            if uav.terminal_outcome is TerminalOutcome.FAILED_COLLISION: rewards[uid].failure -= self.config.low_level_collision_penalty
            if uav.terminal_outcome is TerminalOutcome.FAILED_ENERGY: rewards[uid].failure -= self.config.low_level_energy_failure_penalty
        self.now += self.config.dt; self.step_count += 1; self._refresh_airspace_occupancy()
        observation = self._observations()
        return observation, {str(uid): r.total for uid, r in rewards.items()}, {str(uid): uid not in self.active for uid in previous}, {str(uid): self.now >= self.config.max_simulation_seconds for uid in previous}, {
            "time": self.now, "queue_size": len(self.admission.queue.items), "outcomes": dict(self.ledger.outcomes),
            "reward_components": {str(uid): r.as_dict() for uid, r in rewards.items()}, "release_decisions": {str(uid): x for uid, x in decisions.items()},
            "actions": action_rows, "telemetry": dict(self.telemetry), "conflict_events": list(self.conflict_events),
        }

    def run_rule_only(self, max_steps: int | None = None) -> OutcomeLedger:
        for _ in range(max_steps or int(self.config.max_simulation_seconds / self.config.dt)):
            self.step()
            if self.arrivals.exhausted() and not self.active and not self.admission.queue.items and not self.service_uavs: break
        return self.ledger
