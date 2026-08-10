"""Phase-A landing-only rule/intent environment.

Active agents are airborne UAVs only. SERVICE is owned by the LZ manager after
touchdown, so low-level termination and completion accounting differ.
"""
from __future__ import annotations
from dataclasses import dataclass
import math,numpy as np
from ..core.config import SimConfig,RingRole
from ..core.uav import FlightPhase,UAVState,TerminalOutcome
from ..core.arrival_process import ArrivalProcess
from ..core.admission import AdmissionManager
from ..core.dynamic_airspace import DynamicAirspace
from ..core.energy import EnergyModel
from ..core.priority import PriorityEngine
from ..core.landing_zone import LandingZoneManager
from ..core.lifecycle import OutcomeLedger,transition
from ..core.merge_scheduler import LowLevelIntent,MergeScheduler
from ..core.trajectory_generator import TrajectoryGenerator
from ..core.safety_filter import SafetyFilter

@dataclass
class WorldSnapshot: now:float; active:dict[int,UAVState]; airspace:DynamicAirspace
class TerminalParallelEnv:
    def __init__(self,config:SimConfig|None=None,policy=None):
        self.config=config or SimConfig();self.policy=policy;self.energy=EnergyModel(self.config);self.priority=PriorityEngine(self.config);self.generator=TrajectoryGenerator(self.config);self.reset()
    @property
    def agents(self):return [str(uid) for uid in self.active]
    def action_spec(self,agent=None):return (3,)
    def reset(self,seed=None):
        self.now=0.;self.step_count=0;self.active={};self.service_uavs={};self.uid=0;self.ledger=OutcomeLedger();self.arrivals=ArrivalProcess(self.config,seed);self.admission=AdmissionManager(self.config);self.airspace=DynamicAirspace(self.config);self.lz_manager=LandingZoneManager();self.scheduler=MergeScheduler(self.config);self.safety=SafetyFilter(self.config);self.events=[]
        return self._observations(),{}
    def _observations(self):
        result={}
        for uid,u in self.active.items():
            role=[float(u.ring_role is r) for r in RingRole] if u.ring_role else [0.,0.,0.]
            result[str(uid)]=np.asarray([u.soc,u.dynamic_priority/5.,*role,*[self.airspace.occupancy(r)/max(1,self.airspace.capacity(r)) for r in RingRole],len(self.admission.queue.items)/max(1,self.config.max_active_uavs),sum(z.status.value=='free' for z in self.lz_manager.zones)/6.],dtype=np.float32)
        return result
    def _refresh_airspace_occupancy(self):
        values={r:0 for r in RingRole}
        for u in self.active.values():
            if u.ring_role is not None:values[u.ring_role]+=1
        self.airspace.set_occupancy(values)
    def _resolve(self,uav,outcome,reason):
        if uav.assigned_lz is not None:self.lz_manager.release(uav.assigned_lz,uav.uid,reason)
        uav.terminal_outcome=outcome;uav.terminal_reason=reason;self.ledger.resolve(uav.arrival_id,outcome);self.active.pop(uav.uid,None)
    def _advance_plan(self,u):
        plan=u.committed_plan
        if plan is None:return
        before=u.position.copy();point=plan.sample(min(self.now+self.config.dt,plan.end_time));u.position=point.position;u.velocity=point.velocity if point.velocity is not None else np.zeros(3);u.speed=float(np.linalg.norm(u.velocity));u.path_length+=float(np.linalg.norm(u.position-before))
        if self.now+self.config.dt+1e-9<plan.end_time:return
        u.committed_plan=None;u.trajectory_plan_id=None;target=plan.metadata.get("target_phase")
        if target=="ring_outer":transition(u,FlightPhase.RING_OUTER,self.now+self.config.dt);u.ring_role=RingRole.OUTER;u.ring_progress=math.atan2(u.position[1],u.position[0])
        elif target=="ring_middle":transition(u,FlightPhase.RING_MIDDLE,self.now+self.config.dt);u.ring_role=RingRole.MIDDLE;u.ring_progress=math.atan2(u.position[1],u.position[0])
        elif target=="ring_inner":transition(u,FlightPhase.RING_INNER,self.now+self.config.dt);u.ring_role=RingRole.INNER;u.ring_progress=math.atan2(u.position[1],u.position[0])
        elif target=="landing":
            transition(u,FlightPhase.LANDING,self.now+self.config.dt);transition(u,FlightPhase.SERVICE,self.now+self.config.dt);self.lz_manager.touchdown(u.assigned_lz,u.uid,u.mission_type);self.service_uavs[u.uid]=u;self.active.pop(u.uid,None);self.events.append(("touchdown",u.uid,self.now+self.config.dt))
    def _commit(self,u,plan):
        plans={uid: x.committed_plan for uid,x in self.active.items() if x.committed_plan is not None};result=self.safety.filter_plan(plan,plans,WorldSnapshot(self.now,self.active,self.airspace))
        if result.approved:u.committed_plan=plan;u.trajectory_plan_id=plan.plan_id;return True
        return False
    def _request_release(self,u,intent):
        if intent.release_request<=self.config.release_threshold or u.committed_plan is not None:return
        snapshot=WorldSnapshot(self.now,self.active,self.airspace)
        if u.phase is FlightPhase.RING_OUTER:
            if "o2m" in self.airspace.affected_interfaces():return
            r=self.scheduler.reserve(u,self.scheduler.propose(u,intent,snapshot,RingRole.MIDDLE));plan=self.generator.ring_transfer(u,RingRole.OUTER,RingRole.MIDDLE,r,self.airspace,self.now)
            if self._commit(u,plan):transition(u,FlightPhase.TRANSFER_O2M,self.now);u.ring_role=None;u.merge_reservation_id=r.reservation_id
            else:self.scheduler.cancel(r.reservation_id,"safety_reject")
        elif u.phase is FlightPhase.RING_MIDDLE:
            if "m2i" in self.airspace.affected_interfaces():return
            r=self.scheduler.reserve(u,self.scheduler.propose(u,intent,snapshot,RingRole.INNER));plan=self.generator.ring_transfer(u,RingRole.MIDDLE,RingRole.INNER,r,self.airspace,self.now)
            if self._commit(u,plan):transition(u,FlightPhase.TRANSFER_M2I,self.now);u.ring_role=None;u.merge_reservation_id=r.reservation_id
            else:self.scheduler.cancel(r.reservation_id,"safety_reject")
        elif u.phase is FlightPhase.RING_INNER:
            if "i2f" in self.airspace.affected_interfaces():return
            lz_id=self.lz_manager.choose_rule_zone(u,self.now)
            if lz_id is None:return
            if not self.lz_manager.reserve(lz_id,u.uid,self.now).approved:return
            r=self.scheduler.reserve(u,self.scheduler.propose(u,intent,snapshot,"final"));plan=self.generator.final_approach(u,self.lz_manager.zones[lz_id],r,self.now)
            if self._commit(u,plan):u.assigned_lz=lz_id;transition(u,FlightPhase.FINAL,self.now);u.ring_role=None;u.merge_reservation_id=r.reservation_id
            else:self.lz_manager.release(lz_id,u.uid,"safety_reject");self.scheduler.cancel(r.reservation_id,"safety_reject")
    def step(self,actions=None):
        for request in self.arrivals.requests_until(self.now):self.ledger.request(request.arrival_id);self.admission.enqueue(request)
        self.admission.queue.step(self.config.dt,self.energy,self.priority,self.now)
        self._refresh_airspace_occupancy();admitted,records=self.admission.step(self.now,self.active,self.airspace,self.energy,self.priority)
        for record in records:
            if record.decision.value=="divert":self.ledger.resolve(record.request.arrival_id,TerminalOutcome.DIVERTED_LOW_SOC if record.reason_code=="low_soc" else TerminalOutcome.DIVERTED_CAPACITY)
        for queued,gate in admitted:
            u=UAVState.from_request(self.uid,queued.request,queued,gate.spawn_point,self.now);self.uid+=1;self.active[u.uid]=u;self._commit(u,self.generator.inbound(u,gate,self.airspace,self.now))
        self.airspace.step_transition(self.now)
        for uid in self.lz_manager.step(self.config.dt):
            u=self.service_uavs.pop(uid);transition(u,FlightPhase.COMPLETED,self.now+self.config.dt);u.completion_time=self.now+self.config.dt;u.terminal_outcome=TerminalOutcome.COMPLETED;self.ledger.resolve(u.arrival_id,TerminalOutcome.COMPLETED);self.events.append(("completed",uid,self.now+self.config.dt))
        actions=actions or {}
        for uid,u in list(self.active.items()):
            intent=LowLevelIntent.from_array(actions.get(str(uid),np.array([1.,0.,0.])));self.priority.update(u,self.now);self._request_release(u,intent)
        for u in list(self.active.values()):
            phase=u.phase;self._advance_plan(u);self.energy.consume(u,phase,self.config.dt)
            if u.energy_remaining<=0 and u.uid in self.active:self._resolve(u,TerminalOutcome.FAILED_ENERGY,"energy_depleted")
        # The trajectory-level filter prevents expected conflicts; this detects any
        # realised collision as a terminal safety failure instead of hiding it.
        for uid,u in list(self.active.items()):
            for other_uid,other in list(self.active.items()):
                if uid >= other_uid: continue
                if np.linalg.norm(u.position-other.position) < self.config.collision_distance:
                    self._resolve(u,TerminalOutcome.FAILED_COLLISION,"collision")
                    if other_uid in self.active:self._resolve(other,TerminalOutcome.FAILED_COLLISION,"collision")
        self.now+=self.config.dt;self.step_count+=1;self._refresh_airspace_occupancy();obs=self._observations();rewards={a:-self.config.dt*.01 for a in obs};terminated={a:False for a in obs};truncated={a:self.now>=self.config.max_simulation_seconds for a in obs};infos={"time":self.now,"queue_size":len(self.admission.queue.items),"outcomes":dict(self.ledger.outcomes),"touchdowns":sum(x[0]=='touchdown' for x in self.events)}
        return obs,rewards,terminated,truncated,infos
    def run_rule_only(self,max_steps=None):
        for _ in range(max_steps or int(self.config.max_simulation_seconds/self.config.dt)):
            self.step()
            if self.arrivals.exhausted() and not self.active and not self.admission.queue.items and not self.service_uavs:break
        return self.ledger
