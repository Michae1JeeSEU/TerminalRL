"""Preselection frozen baseline with all revealed downstream demand."""
from dataclasses import dataclass, replace
from types import MappingProxyType
import math
from .scope import digest

EVALUATION_INPUT_POLICY = "global_preselection_frozen_v1"


class FrameIncomplete(RuntimeError):
    """A bounded baseline calculation did not produce a complete frame."""


@dataclass(frozen=True)
class GlobalEvaluationFrame:
    snapshot_id: str
    effective_time: float
    baselines: tuple
    calendars: tuple
    order: tuple
    assignment: tuple
    pad_count: int
    upstream_requests: tuple = ()
    policy: str = EVALUATION_INPUT_POLICY

    @property
    def identity(self): return digest(self)
    @property
    def baseline_by_id(self): return MappingProxyType({d.aircraft_id:d for d in self.baselines})
    @property
    def upstream_request_ids(self): return tuple(row[0] for row in self.upstream_requests)
    @property
    def objective_eligibility(self):
        return {**{d.aircraft_id:"ADMITTED_AIRCRAFT_V8" for d in self.baselines},
                **{row[0]:"NOT_ADMITTED_NO_FLIGHT_STATE" for row in self.upstream_requests}}

    def downstream(self, demands, replan_ids):
        from ..stage2_downstream import JointDownstream, ticks
        fixed=tuple(d for d in self.baselines if d.aircraft_id not in replan_ids)
        return JointDownstream(demands,self.calendars,self.order,now=ticks(self.effective_time),pad_count=self.pad_count,
            fixed_demands=fixed,assignment=dict(self.assignment),frame_id=self.identity)


def build_global_frame(snapshot, provider, context):
    """No full references or new candidate trajectories. Only FOLLOW facts/cache."""
    from ..stage2_planning import _node, NormalReference, energy_slack
    from ..stage2_downstream import Demand, JointDownstream, describe_column, ticks, joules
    from ..config import service_duration
    from ..world import freeze
    baselines=[]; priorities={}
    for aid,a in sorted(snapshot.aircraft.items()):
        if context.budget is not None and not context.budget.evaluation_available():
            raise FrameIncomplete("GLOBAL_FRAME_BUDGET_UNKNOWN")
        node=_node(snapshot,a)
        if node is not None:
            # _column reads current continuations and cache only. This is a
            # baseline descriptor, never a replan generator or new maneuver.
            c=provider._column(snapshot,a,(),"BACKGROUND",context)
            if c is None: raise FrameIncomplete("FIXED_CONTINUATION_PROTECTION_UNKNOWN")
            d=describe_column(snapshot,c,provider.references,context)
            priorities[aid]=energy_slack(snapshot,a,NormalReference("UNKNOWN"))
        else:
            plans=context.all_plans(aid,())
            final=plans[-1] if plans else None
            if final is None and a.status!="LANDED_SERVICE": continue
            touchdown=final.end_time if final is not None else (a.touchdown_time if a.touchdown_time is not None else snapshot.time)
            intervals=tuple((r.resource[1],ticks(r.start),ticks(r.end),aid) for r in (context.resources(a,()) or ())
                            if isinstance(r.resource,tuple) and r.resource[0]=="pad" and math.isfinite(r.end))
            portions=tuple((p,max(snapshot.time,p.start_time),p.end_time) for p in plans if p.end_time>snapshot.time)
            q=context.energy_portions(portions)[0] if portions else 0.
            capacity=snapshot.config.energy.equivalent_voltage_V*snapshot.config.energy.reference_capacity_Ah*3600.
            d=Demand(aid,True,ticks(touchdown-13.5),ticks(13.5+service_duration(snapshot.config,a.service_class)),
                joules(q),0,joules((a.soc-.25)*capacity),0,0,-int(touchdown<=snapshot.time+240.),ticks(touchdown),None,
                MappingProxyType(dict(reference_status="KNOWN",reference_unknown_flag=False,
                    reference_estimate_kind="committed_final_or_ground_service",window_j=q,tail_motion_j=0.)),intervals)
            priorities[aid]=(-1,0.,aid)
        # Freeze diagnostics too: candidate enumeration cannot mutate frame.
        baselines.append(replace(d,diagnostics=freeze(dict(d.diagnostics))))
    order=tuple(sorted((d.aircraft_id for d in baselines),key=lambda aid:(priorities[aid],aid)))
    calendars=tuple((r.resource if isinstance(r.resource,int) else r.resource[1],ticks(r.start),ticks(r.end),r.aircraft_id)
                    for r in snapshot.pads if math.isfinite(r.end))
    baseline_map={}; fixed=tuple(baselines)
    provisional=JointDownstream(baseline_map,calendars,order,now=ticks(snapshot.time),pad_count=snapshot.config.pads.count,fixed_demands=fixed)
    # Every aircraft may replace FOLLOW by a full candidate. Freeze a virtual
    # pad for every nonfull baseline; full baselines are never open normal IDs.
    revealed_time=getattr(getattr(snapshot,"planning_timing",None),"snapshot_time",snapshot.time)
    upstream=tuple((rid,r.release_time,r.initial_soc,r.service_class,getattr(r,"gate_id",None),None,"UNKNOWN", "NOT_ADMITTED_NO_FLIGHT_STATE")
        for rid,r in sorted(snapshot.requests.items()) if rid not in snapshot.aircraft and r.release_time<=revealed_time)
    return GlobalEvaluationFrame(snapshot.version,snapshot.time,fixed,calendars,order,
        tuple(sorted(provisional.assignment.items())),snapshot.config.pads.count,upstream)
