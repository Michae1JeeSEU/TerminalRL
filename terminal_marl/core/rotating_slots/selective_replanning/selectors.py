"""Deterministic mandatory + four-queue Stage A selection."""
import math
from .scope import PlanningScope
from .relations import relation_digest


class RuleSelector:
    def __init__(self, nominal_k=8, max_k=12, *, variant="baseline"):
        if nominal_k != 8 or max_k != 12: raise ValueError("Stage A freezes K8/max12")
        if variant not in ("baseline","outer_priority_v1","opportunity_cooperative_v2"): raise ValueError("unknown selective rule variant")
        self.nominal_k, self.max_k = nominal_k, max_k
        self.variant=variant

    def select(self, snapshot, frame, history, relations=(), *, method="RULE_LOCAL_K8", now=None,
               references=None, opportunity_seed_offset=0):
        if opportunity_seed_offset not in (0,1,2): raise ValueError("diagnostic opportunity offset must be 0..2")
        if self.variant=="opportunity_cooperative_v2":
            return self._select_v2(snapshot,frame,history,relations,method=method,now=now,
                references=references,opportunity_seed_offset=opportunity_seed_offset)
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

    def _select_v2(self,snapshot,frame,history,relations,*,method,now,references,opportunity_seed_offset):
        from ..stage2_planning import _decision_aircraft, _node
        from ..energy import stable_power
        from ..rescue_rules import normal_energy_lower_bound
        from .near_term import summarize_opportunities,cooperative_seed_groups
        eligible={a.id:a for a in _decision_aircraft(snapshot)}; ids=tuple(sorted(eligible))
        timing=getattr(snapshot,"planning_timing",None)
        observed=getattr(timing,"snapshot_time",snapshot.time)
        now=observed if now is None else now
        ages={aid:history.ages(aid,now) for aid in ids}
        reasons={aid:[] for aid in ids}; masks={}; hard=set()
        for aid,a in eligible.items():
            node=_node(snapshot,a); e=a.emergency
            epoch=getattr(e,"normal_estimate_epoch",None); source=getattr(e,"normal_estimate_source",None)
            known=bool(source is not None and epoch is not None and 0<=now-epoch<=5.+1.e-8
                and e.normal_path_soc is not None and tuple(source[:3])==(node.ring,node.slot,node.version))
            required=e.normal_path_soc if known else normal_energy_lower_bound(snapshot.config,node.ring)
            effective=getattr(timing,"effective_time",snapshot.time)
            lead=max(0.,effective-observed)
            access=max(effective,node.time)
            deferred=max(effective+lead,node.time)
            # Projected aircraft/node SOC already includes the lead and all
            # immutable commitments up to node.time. Charge only any remainder.
            uncharged=max(0.,deferred-node.time)
            total_wait=max(0.,deferred-observed)
            power=stable_power(node.ring,snapshot.config)
            rate=power.current_a/(snapshot.config.energy.reference_capacity_Ah*3600.)
            wait_soc=rate*uncharged; margin=node.soc-.25-required
            peak_rate=int(power.shaft_w>snapshot.config.vehicle.rotor_count*snapshot.config.energy.reference_motor_continuous_power_kW_each*1000)
            urgent=margin<=wait_soc or 120.-node.peak<=uncharged*peak_rate
            if urgent: hard.add(aid); reasons[aid].append("ENERGY_OR_PEAK_BEFORE_ADJUSTMENT")
            if not known: reasons[aid].append("ENERGY_ESTIMATE_UNKNOWN")
            masks[aid]=dict(known=known,margin_soc=margin,source="current_shared_path_estimate" if known else "normal_energy_lower_bound_proxy",
                estimate_epoch=epoch,estimate_source=source,can_safely_postpone=None,
                snapshot_time=observed,energy_basis_time=node.time,adjustment_time=access,
                current_adjustment_time=access,deferred_adjustment_time=deferred,
                deferred_adjustment_source="effective_boundary_plus_current_lead_estimate" if timing is not None else "timing_unknown_no_lead_assumed",
                next_opportunity_wait_seconds=total_wait,next_opportunity_wait_soc=rate*total_wait,
                additional_uncharged_wait_seconds=uncharged,additional_uncharged_wait_soc=wait_soc,
                projected_wait_already_charged_seconds=max(0.,node.time-observed),urgent_evidence=urgent)
        def closure(seed, *, required_only=False):
            result=set(seed); frontier=list(seed)
            while frontier:
                current=frontier.pop()
                for r in relations:
                    if (r.source==current and r.definite and (r.required or not required_only)
                            and r.target in eligible and r.target not in result):
                        result.add(r.target); frontier.append(r.target)
            return result
        urgent_hard=set(hard)
        hard=closure(hard,required_only=True)
        for aid in hard-urgent_hard: reasons[aid].append("VALIDATED_REQUIRED_BLOCKER")
        def tie(aid): return (getattr(eligible[aid],"release_time",math.inf),aid)
        mandatory=tuple(sorted(hard,key=lambda aid:(not masks[aid]["urgent_evidence"],tie(aid))))
        opportunities=summarize_opportunities(snapshot,eligible,references)
        groups=cooperative_seed_groups(opportunities) if ids else ()
        overflow=len(hard)>self.max_k
        overdue=[aid for aid in ids if ages[aid]["review_age"] is not None and ages[aid]["review_age"]>=60.]
        fair_bonus={aid:max(0.,(ages[aid]["review_age"] or 0.)-40.) for aid in ids}
        fair=[]
        if method=="FULL_H" or overflow:
            selected=list(ids); status="FULL_H" if method=="FULL_H" else "MANDATORY_OVERFLOW_FULL"
            for aid in selected: reasons[aid].append("FULL_H" if method=="FULL_H" else "EXPLICIT_HARD_OVERFLOW_FULL_FALLBACK")
        else:
            selected=list(mandatory); status="LOCAL"
            seats=min(2,max(0,self.nominal_k-len(selected)))
            old=sorted((aid for aid in ids if aid not in hard and fair_bonus[aid]>0),key=lambda aid:(-fair_bonus[aid],tie(aid)))
            unknown=sorted((aid for aid in ids if aid not in hard and ages[aid]["review_unknown"]),key=tie)
            # Unknown age receives bounded review access, never invented debt.
            if seats and old: fair.append(old.pop(0))
            if len(fair)<seats and unknown: fair.append(unknown.pop(0))
            fair.extend(old[:seats-len(fair)])
            if len(fair)<seats: fair.extend(unknown[:seats-len(fair)])
            accepted_fair=[]; skipped=[]
            for aid in fair:
                additions=closure((aid,))-set(selected)
                if len(selected)+len(additions)>self.max_k:
                    skipped.append(dict(members=[aid],reason="FAIRNESS_DEPENDENCY_CLOSURE_EXCEEDS_MAX12")); continue
                accepted_fair.append(aid)
                for member in sorted(additions,key=tie):
                    selected.append(member)
                    reasons[member].append(("UNKNOWN_AGE_BOUNDED_REVIEW" if ages[aid]["review_unknown"] else "CONTINUOUS_FAIRNESS_REVIEW")
                        if member==aid else "DEFINITE_DEPENDENCY_CLOSURE")
            fair=accepted_fair
            # Reserve hard/fair first, then put a whole cooperative opportunity
            # into the remaining nominal seats. Only definite dependency closure
            # may expand a complete seed from K8 to max12.
            seed_order=list(groups)
            if seed_order:
                offset=min(opportunity_seed_offset,len(seed_order)-1)
                seed_order=seed_order[offset:]+seed_order[:offset]
            ordered=sorted(ids,key=lambda aid:(opportunities[aid]["unknown"],
                (opportunities[aid]["earliest_arrival"] or math.inf)-fair_bonus[aid],
                masks[aid]["known"],masks[aid]["margin_soc"],tie(aid)))
            for seed in seed_order+[(aid,) for aid in ordered]:
                add=set(seed)-set(selected)
                if not add: continue
                if len(set(selected)|set(seed))>self.nominal_k:
                    skipped.append(dict(members=list(seed),reason="GROUP_EXCEEDS_REMAINING_NOMINAL_SEATS")); continue
                expanded=closure(seed)-set(selected)
                if len(selected)+len(expanded)>self.max_k:
                    skipped.append(dict(members=list(seed),reason="DEPENDENCY_CLOSURE_EXCEEDS_MAX12")); continue
                for aid in sorted(expanded,key=tie):
                    selected.append(aid); reasons[aid].append("COOPERATIVE_OPPORTUNITY_GROUP" if aid in seed else "DEFINITE_DEPENDENCY_CLOSURE")
                if len(selected)>=self.nominal_k: break
        scope=PlanningScope(snapshot.version,snapshot.time,ids,tuple(sorted(selected)),
            tuple(sorted(set(snapshot.aircraft)-set(selected))),mandatory,self.nominal_k,len(selected),self.max_k,
            tuple((aid,tuple(reasons[aid])) for aid in selected),relation_digest(relations),frame.identity,status).validate(snapshot)
        group_rows=[dict(group_id=i,members=list(group),reason="SPECIFIC_RESOURCE_RELEASE_CHAIN",
            selected_members=sorted(set(group)&set(selected))) for i,group in enumerate(groups)]
        return scope,dict(ages=ages,energy_masks=masks,rule_variant=self.variant,status="MANDATORY_OVERFLOW" if overflow else status,
            hard_mandatory_count=len(hard),hard_urgent_count=sum(masks[aid]["urgent_evidence"] for aid in hard),
            overflow_cause="HARD_MANDATORY_GT_MAX12" if overflow else None,actual_k=len(selected),unserved_mandatory_ids=[],
            fairness_selected_ids=fair,fairness_bonus=fair_bonus,fairness_overdue_ids=overdue,
            fairness_overdue_count=len(overdue),fairness_unselected_overdue_ids=sorted(set(overdue)-set(selected)),
            unknown_review_age_ids=[aid for aid in ids if ages[aid]["review_unknown"]],
            near_term_opportunities=opportunities,cooperative_groups=group_rows,opportunity_seed_offset=opportunity_seed_offset,
            skipped_optional_groups=skipped if status=="LOCAL" else [],
            unselected_reasons={aid:"FIXED_BACKGROUND_NOMINAL_SCOPE_CAPACITY" for aid in ids if aid not in selected},
            eligible_ring_by_id={aid:_node(snapshot,eligible[aid]).ring for aid in ids})
