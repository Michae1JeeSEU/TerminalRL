"""External demand queue and physical four-corridor admission."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import math, numpy as np
from .config import SimConfig, RingRole
from .uav import ArrivalRequest, FlightPhase, QueuedArrivalState, TerminalOutcome

@dataclass
class EntryGate:
    gate_id:int; angle:float; spawn_point:np.ndarray; gate_point:np.ndarray
    active_inbound:list[int]=field(default_factory=list); last_admission_time:float=-math.inf
class AdmissionDecision(str,Enum): ADMIT="admit"; WAIT="wait"; DIVERT="divert"
@dataclass(frozen=True)
class AdmissionRecord:
    request: ArrivalRequest
    decision: AdmissionDecision
    reason_code: str
    energy_remaining: float | None = None
    energy_consumed: float | None = None
    soc: float | None = None

class ExternalQueue:
    def __init__(self): self.items:list[QueuedArrivalState]=[]
    def enqueue(self,request:ArrivalRequest)->None: self.items.append(QueuedArrivalState(request,request.energy_capacity*request.initial_soc))
    def step(self,dt,energy,priority,now):
        for q in self.items:
            energy.consume(q, "external_queue", dt); q.external_wait_time+=dt; priority.update(q,now)
    def ordered(self,priority,now):
        for q in self.items: priority.update(q,now)
        return sorted(self.items,key=lambda q: priority.rank_key(q,now))

class AdmissionManager:
    def __init__(self,config:SimConfig):
        self.config=config; self.queue=ExternalQueue(); self.gates=self._make_gates()
    def _make_gates(self):
        gates=[]
        for i,degrees in enumerate(self.config.gate_angles_deg):
            a=math.radians(degrees); unit=np.array((math.cos(a),math.sin(a),0.))
            gates.append(EntryGate(i,a,unit*self.config.spawn_radius+np.array((0.,0.,self.config.gate_height)), unit*self.config.approach_threshold_radius+np.array((0.,0.,self.config.gate_height))))
        return gates
    def enqueue(self, request): self.queue.enqueue(request)
    def step(self,now,active_uavs,airspace,energy_model,priority_engine,*,lz_manager=None):
        records=[]; admitted=[]
        if self.config.energy_reachability_guard_enabled and lz_manager is None:
            raise ValueError("energy_reachability_guard_enabled requires lz_manager")
        for q in list(self.queue.ordered(priority_engine,now)):
            r=q.request
            def record(decision, reason):
                return AdmissionRecord(r, decision, reason, float(q.energy_remaining),
                                       float(q.energy_consumed), float(q.soc))
            if q.soc < self.config.soc_reserve or q.energy_remaining < energy_model.admission_required_energy(q):
                q.diverted_outcome=TerminalOutcome.DIVERTED_LOW_SOC; q.diversion_reason="insufficient_soc"; self.queue.items.remove(q); records.append(record(AdmissionDecision.DIVERT,"low_soc")); continue
            if self.config.max_external_wait_seconds is not None and q.external_wait_time>=self.config.max_external_wait_seconds:
                q.diverted_outcome=TerminalOutcome.DIVERTED_CAPACITY; self.queue.items.remove(q); records.append(record(AdmissionDecision.DIVERT,"external_wait_limit")); continue
            if self.config.energy_reachability_guard_enabled:
                gate = self.gates[r.gate_id]
                finite_extra_waits = []
                for zone in lz_manager.zones:
                    if r.mission_type not in zone.compatible_missions:
                        continue
                    release_time = float(lz_manager.earliest_release_time(zone.lz_id, now=float(now)))
                    if not math.isfinite(release_time):
                        continue
                    horizontal_distance = float(np.linalg.norm(
                        np.asarray(zone.position, dtype=float)[:2]
                        - np.asarray(gate.spawn_point, dtype=float)[:2]
                    ))
                    minimum_travel_time = horizontal_distance / float(self.config.max_speed)
                    finite_extra_waits.append(max(0.0, release_time - float(now) - minimum_travel_time))
                if not finite_extra_waits:
                    records.append(record(AdmissionDecision.WAIT,"lz_slot_unknown")); continue
                extra_pad_wait = min(finite_extra_waits)
                required_energy = (
                    float(energy_model.admission_required_energy(q))
                    + extra_pad_wait * float(energy_model.phase_power(FlightPhase.RING_INNER))
                    + float(self.config.emergency_energy_margin_seconds)
                    * float(self.config.power_reference)
                )
                if q.energy_remaining < required_energy:
                    q.diverted_outcome=TerminalOutcome.DIVERTED_LOW_SOC; q.diversion_reason="low_soc_slot"; self.queue.items.remove(q); records.append(record(AdmissionDecision.DIVERT,"low_soc_slot")); continue
            gate=self.gates[r.gate_id]
            outer=airspace.capacity(RingRole.OUTER)
            if len(active_uavs)>=self.config.max_active_uavs: records.append(record(AdmissionDecision.WAIT,"active_cap")); continue
            if now-gate.last_admission_time<self.config.gate_min_time_gap_seconds: records.append(record(AdmissionDecision.WAIT,"gate_headway")); continue
            if airspace.occupancy(RingRole.OUTER)>=outer: records.append(record(AdmissionDecision.WAIT,"outer_capacity")); continue
            self.queue.items.remove(q); gate.last_admission_time=now; admitted.append((q,gate)); records.append(record(AdmissionDecision.ADMIT,"ok"))
        return admitted,records
