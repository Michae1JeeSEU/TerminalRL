"""Time/gap based merge reservations; no fixed release angles."""
from __future__ import annotations
from dataclasses import dataclass
import math, uuid
from .config import RingRole
from .uav import UAVState

@dataclass(frozen=True)
class LowLevelIntent:
    release_request:float; speed_residual:float; merge_time_residual:float
    @classmethod
    def from_array(cls,raw):
        values=[float(v) for v in raw]
        if len(values)!=3 or any(not math.isfinite(v) for v in values):raise ValueError("low-level intent must be finite shape (3,)")
        return cls(*(max(-1.,min(1.,v)) for v in values))
@dataclass
class MergeReservation:
    reservation_id:str; uav_id:int; source_stage:str; target_stage:str; merge_time:float; merge_angle:float|None; predecessor_id:int|None; successor_id:int|None; status:str="committed"; expires_at:float=0.
@dataclass(frozen=True)
class MergeProposal: target:RingRole|str; merge_time:float; merge_angle:float|None; headway:float

@dataclass(frozen=True)
class RepairCandidate:
    """A bounded, canonical-intention candidate for one release attempt."""
    kind: str
    intent: LowLevelIntent
    delay_seconds: float = 0.0

class MergeScheduler:
    def __init__(self,config):
        self.config=config; self.reservations={}
        # ``last_release`` is the latest *active tentative* reservation.  It
        # must not be confused with the last executed interface release.
        self.last_release={"o2m":-math.inf,"m2i":-math.inf,"i2f":-math.inf}
        self.last_executed_release={"o2m":-math.inf,"m2i":-math.inf,"i2f":-math.inf}
    def _key(self,target):return {RingRole.MIDDLE:"o2m",RingRole.INNER:"m2i","final":"i2f"}[target]
    def transfer_duration_estimate(self,uav,target,snapshot):
        """Conservative canonical ring-transfer duration before a merge slot.

        A reservation is an arrival-at-merge-time slot, not permission to
        depart immediately through a source-ring holder.  The estimate uses
        the same vertical smoothstep bound as ``TrajectoryGenerator`` and a
        conservative radial travel lower bound; it creates a bounded holding
        window before a future slot without adding a free-flight control.
        """
        if not isinstance(target,RingRole) or uav.ring_role is None:
            return 0.0
        source=snapshot.airspace.actual_ring(uav.ring_role); destination=snapshot.airspace.actual_ring(target)
        vertical=abs(destination.height-source.height)/max(1e-6,self.config.max_climb_rate)*1.90
        radial=abs(destination.radius-source.radius)
        # A curved merge must accelerate/decelerate and travel farther than a
        # radial chord.  This lower-bound surrogate is deliberately shared
        # with the EH216-S acceleration envelope, so reservations cannot ask
        # the time parameterizer to catch an impossible slot.
        target_speed=min(destination.nominal_speed,self.config.max_speed)
        horizontal=3.5*radial/max(1.,target_speed)+source.nominal_speed/self.config.max_acceleration+target_speed/self.config.max_acceleration
        return max(6.0,vertical,horizontal)
    def baseline_time(self,uav,target,snapshot):
        # Nominal merge preparation time.  A bounded residual modifies the
        # scheduler's reservation subject to the hard earliest headway.
        key=self._key(target); previous=max(self.last_release[key],self.last_executed_release[key])
        earliest=max(snapshot.now,snapshot.airspace.headways[key]+previous)
        return earliest + 6.0 + self.transfer_duration_estimate(uav,target,snapshot)
    def propose(self,uav,intent,snapshot,target, *, delay_seconds: float = 0.0):
        if delay_seconds < 0.0:
            raise ValueError("prospective release delay must be non-negative")
        base=self.baseline_time(uav,target,snapshot); key=self._key(target); previous=max(self.last_release[key],self.last_executed_release[key]); hard_earliest=max(snapshot.now,snapshot.airspace.headways[key]+previous)
        t=max(hard_earliest, base+intent.merge_time_residual*self.config.max_merge_time_residual_seconds+delay_seconds); angle=None
        if isinstance(target,RingRole):
            ring=snapshot.airspace.actual_ring(target); angle=self._gap_center(uav,target,t,snapshot,ring.nominal_speed)
        return MergeProposal(target,t,angle,snapshot.airspace.headways[self._key(target)])

    def repair_candidates(self, requested: LowLevelIntent) -> tuple[RepairCandidate, ...]:
        """Minimum-intervention ordering over the existing ±6 s / ±2 m/s authority.

        The finite endpoints match the frozen-state audit witnesses.  A
        candidate remains an ordinary three-dimensional low-level intention;
        no geometric or free-flight command is introduced here.
        """
        values = (-1.0, 1.0)
        rows: list[RepairCandidate] = [RepairCandidate("requested", requested)]
        def add(kind: str, speed: float, merge: float, delay: float = 0.0) -> None:
            candidate = LowLevelIntent(requested.release_request, speed, merge)
            row = RepairCandidate(kind, candidate, delay)
            if row not in rows:
                rows.append(row)
        # Requested -> one-axis timing -> one-axis speed -> both axes.
        for merge in values:
            add("merge_time_repair", requested.speed_residual, merge)
        for speed in values:
            add("speed_repair", speed, requested.merge_time_residual)
        for speed in values:
            for merge in values:
                add("combined_repair", speed, merge)
        # Only after all immediate legal repairs fail do we reserve a future
        # slot.  Use the same finite residual options, now with a bounded
        # prospective delay, so no one-second retry loop is needed.
        immediate = tuple(rows)
        for delay in self.config.prospective_repair_delays_seconds:
            for row in immediate:
                add("future_delay" if row.kind == "requested" else f"future_delay_{row.kind}",
                    row.intent.speed_residual, row.intent.merge_time_residual, float(delay))
        return tuple(rows)

    def can_reserve(self, proposal: MergeProposal) -> bool:
        key = self._key(proposal.target)
        return proposal.merge_time >= self.last_release[key] + proposal.headway - 1e-9
    def _gap_center(self,uav,target,time,snapshot,speed):
        states=[u for u in snapshot.active.values() if u.ring_role is target]
        # An empty target ring has no target-holder gap to optimize.  A fixed
        # zero-angle capture made every O2M request bend through the same
        # arbitrary corridor.  With no target traffic, use the requester's
        # own phase propagated to the merge time.  This keeps the transfer
        # progressing in the circulation direction even when conservative
        # slot lead time exceeds the eventual canonical curve duration.  It
        # remains a scheduler-derived
        # canonical capture phase, not a free-flight heading command.
        if not states:
            source_role = uav.ring_role or {
                "ring_outer": RingRole.OUTER, "ring_middle": RingRole.MIDDLE, "ring_inner": RingRole.INNER,
            }.get(uav.phase.value)
            if source_role is None:
                return float(uav.ring_progress if uav.ring_progress is not None else math.atan2(uav.position[1], uav.position[0])) % (2 * math.pi)
            source_ring=snapshot.airspace.actual_ring(source_role)
            horizon=max(0.0,time-snapshot.now)
            phase = uav.ring_progress if uav.ring_progress is not None else math.atan2(uav.position[1], uav.position[0])
            return float(phase + source_ring.nominal_speed * horizon / source_ring.radius) % (2*math.pi)
        horizon = max(0.0, time - snapshot.now)
        angles=sorted(((u.ring_progress or 0.)+speed*horizon/snapshot.airspace.actual_ring(target).radius)%(2*math.pi) for u in states)
        gaps=[((angles[(i+1)%len(angles)]-a)%(2*math.pi),a) for i,a in enumerate(angles)]
        gap,start=max(gaps); return (start+gap/2)%(2*math.pi)
    def reserve(self,uav,proposal, *, now: float | None = None):
        key=self._key(proposal.target)
        if not self.can_reserve(proposal): raise ValueError("headway reservation conflict")
        rid=str(uuid.uuid4()); status="future" if now is not None and proposal.merge_time > now + 1e-9 else "committed"; r=MergeReservation(rid,uav.uid,uav.phase.value,str(getattr(proposal.target,'value',proposal.target)),proposal.merge_time,proposal.merge_angle,None,None,status=status,expires_at=proposal.merge_time+60.)
        self.reservations[rid]=r;self.last_release[key]=proposal.merge_time;return r
    def cancel(self,reservation_id,reason=""):
        """Remove a rejected tentative reservation from future-release state."""
        reservation = self.reservations.pop(reservation_id, None)
        if reservation is None:
            return
        key = self._key(RingRole(reservation.target_stage) if reservation.target_stage in {role.value for role in RingRole} else "final")
        committed = [item.merge_time for item in self.reservations.values() if self._key(RingRole(item.target_stage) if item.target_stage in {role.value for role in RingRole} else "final") == key]
        self.last_release[key] = max(committed, default=-math.inf)

    def consume(self,reservation_id):
        """Close an executed reservation while retaining its real headway."""
        reservation=self.reservations.pop(reservation_id,None)
        if reservation is None:return
        key=self._key(RingRole(reservation.target_stage) if reservation.target_stage in {role.value for role in RingRole} else "final")
        self.last_executed_release[key]=max(self.last_executed_release[key],reservation.merge_time)
        committed=[item.merge_time for item in self.reservations.values() if self._key(RingRole(item.target_stage) if item.target_stage in {role.value for role in RingRole} else "final")==key]
        self.last_release[key]=max(committed,default=-math.inf)

    def expire(self,now,active_uav_ids):
        """Cancel stale unexecuted reservations; return IDs for audit trace."""
        stale=[rid for rid,item in self.reservations.items() if item.expires_at<=now or item.uav_id not in active_uav_ids]
        for rid in stale:self.cancel(rid,"expired_or_orphaned")
        return stale
