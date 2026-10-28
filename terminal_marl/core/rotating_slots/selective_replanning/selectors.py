"""Deterministic mandatory + four-queue Stage A selection."""
import math
from .scope import PlanningScope
from .relations import relation_digest


class RuleSelector:
    def __init__(self, nominal_k=8, max_k=12, *, variant="baseline"):
        if nominal_k != 8 or max_k != 12: raise ValueError("Stage A freezes K8/max12")
        if variant not in ("baseline","outer_priority_v1"): raise ValueError("unknown selective rule variant")
        self.nominal_k, self.max_k = nominal_k, max_k
        self.variant=variant

    def select(self, snapshot, frame, history, relations=(), *, method="RULE_LOCAL_K8", now=None):
        from ..stage2_planning import _decision_aircraft, _node
        from ..energy import stable_power
        from ..rescue_rules import normal_energy_lower_bound
        from ..stage2_downstream import earliest_gap
        eligible = {a.id:a for a in _decision_aircraft(snapshot)}
        ids = tuple(sorted(eligible))
        now = snapshot.time if now is None else now
        ages = {aid:history.ages(aid, now) for aid in ids}
        reasons = {aid:[] for aid in ids}; mandatory=[]
        energy = {}
        opportunity = {}
        for aid, a in eligible.items():
            d = frame.baseline_by_id.get(aid); node=_node(snapshot,a)
            e = getattr(a, "emergency", None)
            estimate_source=getattr(e,"normal_estimate_source",None)
            estimate_epoch=getattr(e,"normal_estimate_epoch",None)
            known=bool(estimate_source is not None and estimate_epoch is not None and
                0<=now-estimate_epoch<=5.+1.e-8 and getattr(e,"normal_path_soc",None) is not None and
                tuple(estimate_source[:3])==(node.ring,node.slot,node.version))
            source="current_shared_path_estimate" if known else "normal_energy_lower_bound_proxy"
            required=e.normal_path_soc if known else normal_energy_lower_bound(snapshot.config,node.ring)
            margin=node.soc-.25-required
            power=stable_power(node.ring,snapshot.config)
            qhold=power.current_a*5./(snapshot.config.energy.reference_capacity_Ah*3600.)
            peak_margin=120.-node.peak
            peak_next=5.*int(power.shaft_w>snapshot.config.vehicle.rotor_count*snapshot.config.energy.reference_motor_continuous_power_kW_each*1000)
            energy[aid]=(known,margin,source,estimate_epoch,estimate_source,qhold)
            # A lower-bound deficit proves urgency, but a positive proxy never
            # certifies that normal flight can safely be postponed.
            urgent = margin<=qhold or peak_margin<=peak_next
            if urgent: reasons[aid].append("ENERGY_OR_PEAK_NEXT_OPPORTUNITY"); mandatory.append(aid)
            if ages[aid]["review_age"] is not None and ages[aid]["review_age"] >= 60.:
                reasons[aid].append("REVIEW_AGE_60S"); mandatory.append(aid)
            if not known: reasons[aid].append("ENERGY_ESTIMATE_UNKNOWN")
            pad=dict(frame.assignment).get(aid)
            calendar=[(lo,hi) for p,lo,hi,owner in frame.calendars if p==pad and owner!=aid]
            opportunity[aid]=earliest_gap(calendar,d.ready,d.duration) if d is not None and pad is not None else math.inf
        for r in relations:
            if r.required and r.definite and r.target in eligible:
                mandatory.append(r.target); reasons[r.target].append("VALIDATED_REQUIRED_BLOCKER")
        def tie(aid): return (getattr(eligible[aid], "release_time", math.inf), aid)
        mandatory=tuple(sorted(set(mandatory), key=lambda aid:(0 if "ENERGY_OR_PEAK_NEXT_OPPORTUNITY" in reasons[aid] else 1, tie(aid))))
        overflow=len(mandatory)>self.max_k
        if method == "FULL_H" or overflow:
            selected=list(ids); status="FULL_H" if method=="FULL_H" else "MANDATORY_OVERFLOW_FULL"
            for aid in selected: reasons[aid].append("FULL_H" if method=="FULL_H" else "EXPLICIT_OVERFLOW_FULL_FALLBACK")
        else:
            selected=list(mandatory); status="LOCAL"
            queues=[sorted(ids,key=lambda aid:(energy[aid][1], not energy[aid][0], tie(aid))),
                    sorted(ids,key=lambda aid:(opportunity[aid], tie(aid))),
                    sorted(ids,key=lambda aid:(ages[aid]["progress_unknown"], -(ages[aid]["progress_age"] or 0), tie(aid))),
                    sorted(ids,key=lambda aid:(ages[aid]["review_unknown"], -(ages[aid]["review_age"] or 0), tie(aid)))]
            labels=("ENERGY_QUEUE", "DOWNSTREAM_OPPORTUNITY_QUEUE", "PROGRESS_AGE_QUEUE", "REVIEW_AGE_QUEUE")
            if self.variant=="outer_priority_v1":
                outer=[aid for aid in ids if _node(snapshot,eligible[aid]).ring=="outer"]
                queues.insert(0,sorted(outer,key=lambda aid:(ages[aid]["progress_unknown"],
                    -(ages[aid]["progress_age"] or 0),tie(aid))))
                labels=("OUTER_RING_QUEUE",)+labels
            limit=min(len(ids), max(self.nominal_k, len(mandatory)))
            while len(selected)<limit:
                advanced=False
                for queue,label in zip(queues,labels):
                    while queue and queue[0] in selected: queue.pop(0)
                    if not queue or len(selected)>=limit: continue
                    seed=queue.pop(0); selected.append(seed); reasons[seed].append(label); advanced=True
                    blockers=sorted({r.target for r in relations if r.source==seed and r.definite and
                                     r.target in eligible and r.target not in selected},key=tie)
                    if blockers and len(selected)<limit:
                        blocker=blockers[0]; selected.append(blocker); reasons[blocker].append("EXPLICIT_RESOURCE_BLOCKER")
                if not advanced: break
        scope=PlanningScope(snapshot.version,snapshot.time,ids,tuple(sorted(selected)),
            tuple(sorted(set(snapshot.aircraft)-set(selected))),mandatory,self.nominal_k,len(selected),self.max_k,
            tuple((aid,tuple(reasons[aid])) for aid in selected),relation_digest(relations),frame.identity,status)
        return scope.validate(snapshot), dict(ages=ages, energy_masks={aid:dict(known=v[0],margin_soc=v[1],source=v[2],estimate_epoch=v[3],
            estimate_source=v[4],next_opportunity_wait_soc=v[5],can_safely_postpone=False if not v[0] else None) for aid,v in energy.items()},
            status="MANDATORY_OVERFLOW" if overflow else status, unserved_mandatory_ids=[],rule_variant=self.variant,
            eligible_ring_by_id={aid:_node(snapshot,eligible[aid]).ring for aid in ids})
