"""S candidates + complete fixed world, at most two shared-budget solves."""
from dataclasses import replace, asdict
import time
from .scope import PlanningScope
from .policy import SearchPolicy


class LocalPlanner:
    def __init__(self, provider, scheduler, *, progressive_v2=False, executable_v3=False, search_policy=None):
        self.provider, self.scheduler = provider, scheduler
        self.progressive_v2=progressive_v2 or executable_v3
        self.executable_v3=executable_v3
        self.search_policy=search_policy if search_policy is not None else SearchPolicy(executable_v3,executable_v3,executable_v3)
        if not isinstance(self.search_policy,SearchPolicy): raise ValueError("resolved SearchPolicy required")

    def plan(self, snapshot, scope, global_frame, shared_budget):
        if self.progressive_v2: return self._plan_v2(snapshot,scope,global_frame,shared_budget)
        from ..stage2_planning import EvaluationContext, PlanningResult, _selection_key
        budget=shared_budget; scope.validate(snapshot)
        calls=[]; completed=set(); all_blockers=[]; aggregate_pairs={}; first=None; incumbent=None
        for index in range(2):
            before_materialized={aid:budget.screening.get("materialized:"+aid,0) for aid in snapshot.aircraft}
            budget.allowed_replan_ids=set(scope.replan_ids)
            context=EvaluationContext(snapshot,budget=budget,validation_cache=self.provider.validation_cache,
                reuse_prefix_results=True,scope=scope,global_frame=global_frame)
            budget._certificate_context=context.snapshot_key
            with budget.measure("candidate_generation_and_single_vehicle"),budget.candidate_work():
                catalog=None
                for catalog in self.provider.iter_progressive_columns(snapshot,budget,context,scope=scope): pass
            if catalog is None: raise RuntimeError("scoped generator did not publish FOLLOW catalog")
            if set(catalog.columns)!=set(scope.replan_ids): raise ValueError("catalog/scope open-variable mismatch")
            seed=None
            if incumbent is not None and incumbent.certified:
                old={c.aircraft_id:c for c in incumbent.selected}
                rows=tuple(next((c for c in catalog.columns[aid] if aid in old and c.signature==old[aid].signature),catalog.columns[aid][0])
                           for aid in scope.replan_ids)
                seed=replace(incumbent,selected=rows,prefix=context.prefix(rows),objective=_selection_key(catalog,rows))
            with budget.measure("selection"):
                result=self.scheduler.plan(snapshot,catalog,context,budget,incumbent=seed)
            completed.update(context.completed_evaluations)
            all_blockers.extend(context.background_blockers)
            for key,value in context.cache_stats.items():
                if key.startswith("pair_"): aggregate_pairs[key]=aggregate_pairs.get(key,0)+value
            calls.append(dict(scope=scope.to_dict(),scope_identity=scope.identity,certified=result.certified,status=result.status,
                objective=list(result.objective),blockers=list(context.background_blockers),
                pair_conflict_evidence=list(context.pair_conflict_evidence),
                pair_conflict_evidence_dropped=context.pair_conflict_evidence_dropped,
                solver_records=list(result.solver_records),candidate_count=sum(map(len,catalog.columns.values())),
                candidate_generation_by_aircraft={aid:budget.screening.get("materialized:"+aid,0)-before_materialized[aid] for aid in snapshot.aircraft}))
            if context.diagnostics is not None:
                calls[-1]["candidate_diagnostics"]=context.diagnostics.export(catalog,result.selected)
            if first is None: first=(result,catalog,context,scope)
            if result.certified and (incumbent is None or result.objective<=incumbent.objective):
                incumbent=result; best=(result,catalog,context,scope)
            if index or scope.status!="LOCAL" or not budget.search_available() or scope.actual_k>=scope.max_k: break
            eligible=set(scope.eligible_ids); opened=set(scope.replan_ids)
            blockers=sorted({row["target"] for row in context.background_blockers
                if row["target"] in eligible-opened and row["reason"]=="RESOURCE_CONFLICT" and row["resource"]})
            if not blockers: break
            add=tuple(blockers[:scope.max_k-len(opened)])
            if not add: break
            opened.update(add)
            scope=replace(scope,replan_ids=tuple(sorted(opened)),fixed_background_ids=tuple(sorted(set(snapshot.aircraft)-opened)),
                actual_k=len(opened),selection_reasons=scope.selection_reasons+tuple((aid,("BOUNDED_VALIDATED_BLOCKER_EXPANSION",)) for aid in add))
            scope.validate(snapshot)
        result,catalog,context,accepted_scope=best if incumbent is not None else first
        self.last_catalog,self.last_context,self.last_scope=catalog,context,accepted_scope
        self.completed_evaluations=completed
        self.telemetry=dict(scope=accepted_scope.to_dict(),scope_identity=accepted_scope.identity,
            objective_frame_id=global_frame.identity,evaluation_input_policy=global_frame.policy,
            known_aircraft_demand_ids=list(global_frame.order),upstream_request_ids=list(global_frame.upstream_request_ids),
            objective_eligibility=global_frame.objective_eligibility,
            upstream_requests=[dict(request_id=r[0],release_time=r[1],initial_soc=r[2],service_class=r[3],gate_id=r[4],
                admission_ready_time=r[5],admission_readiness=r[6],mask=r[7]) for r in global_frame.upstream_requests],
            solve_calls=len(calls),scope_expansions=max(0,len(calls)-1),calls=calls,
            attempted_replan_ids=sorted({aid for call in calls for aid in call["scope"]["replan_ids"]}),
            candidate_generation_by_aircraft={aid:budget.screening.get("materialized:"+aid,0) for aid in snapshot.aircraft},
            pair_checks=aggregate_pairs,background_blockers=all_blockers,
            model_scale=[r["model_scale"] for call in calls for r in call["solver_records"] if "model_scale" in r],
            status="LOCAL_SEARCH_INCOMPLETE" if not result.prefix else "CERTIFIED_PREFIX",
            completed_evaluation_ids=sorted(completed))
        return result

    @staticmethod
    def _coverage(snapshot,catalog):
        from ..stage3_learning.catalog_control import coverage_report
        effective=getattr(getattr(snapshot,"planning_timing",None),"effective_time",None)
        rows=coverage_report(snapshot,catalog.columns,effective_time=effective)
        for aid,row in rows.items():
            moves=[c for c in catalog.columns[aid] if c.kind=="MOVE" and c.plans]
            row.update(move_candidates=len(moves),only_keep_wait=not moves,target_count=len(row["first_directions"]))
            if not row["delayed"]: row["missing_reasons"].append("NO_DELAYED_MOVE_OBSERVED")
            if len(row["first_directions"])<2: row["missing_reasons"].append("NO_ALTERNATE_TARGET_OBSERVED")
        return rows

    @staticmethod
    def _mapped_incumbent(old,catalog,context,budget):
        """Re-evaluate a previous choice in this catalog and complete frame."""
        from ..stage2_planning import _selection_key, RollingHorizonCPSATScheduler
        from ..stage2_downstream import SERVICE_EVALUATION_PROFILE
        if old is None or not old.certified: return None
        service_profile=getattr(catalog.downstream,'evaluation_profile',None)==SERVICE_EVALUATION_PROFILE
        known=None
        if service_profile:
            state,_=RollingHorizonCPSATScheduler._continuation(context.snapshot,catalog,context)
            known=state.get('incumbent')
        # Exact same context and full column contents preserve the original
        # certificate, prefix, and score even when generation used its slice.
        if known is not None:
            reused=RollingHorizonCPSATScheduler._certified_seed(old,catalog,context,budget,known=known)
            if reused is not None: return reused
        if not budget.evaluation_available(): return None
        by_id={c.aircraft_id:c for c in old.selected}; selected=[]
        for aid,rows in catalog.columns.items():
            previous=by_id.get(aid)
            matched=(next((c for c in rows if c.signature==previous.signature),None)
                if previous is not None else next((c for c in rows if c.kind=="KEEP"),None))
            if matched is None: return None
            selected.append(matched)
        selected=tuple(selected)
        result,_=context.check_batch(selected)
        if not result.accepted or result.status=="UNKNOWN": return None
        evaluation=context.check_commit_prefix(selected) if budget.evaluation_available() else None
        if evaluation is None or not evaluation.accepted or getattr(evaluation,"refinement_required",False): return None
        return replace(old,selected=selected,prefix=context.prefix(selected),evaluation=evaluation,
            objective=_selection_key(catalog,selected),certified=True)

    def _plan_v2(self,snapshot,scope,frame,budget):
        from ..stage2_planning import EvaluationContext
        from ..stage2_downstream import SERVICE_EVALUATION_PROFILE
        service_profile=getattr(self.provider,'evaluation_profile',None)==SERVICE_EVALUATION_PROFILE
        scope.validate(snapshot)
        calls=[]; catalog_rows=[]; completed=set(); blockers=[]; joints=[]; pair_counts={}
        first=None; best=None; incumbent=None; expansions=0; catalogs=None; context=None
        scope_round=0; old_call_limit=getattr(budget,"solver_call_seconds",None); first_certified=None; skipped_expansions=[]
        budget._selective_progressive_v2=True
        budget._selective_target_coverage_order=self.search_policy.target_coverage_order
        try:
            for index in range(4 if service_profile else 2):
                if service_profile and incumbent is not None and budget.remaining_seconds()<=budget.solver_validation_headroom():
                    break
                before_materialized={aid:budget.screening.get("materialized:"+aid,0) for aid in snapshot.aircraft}
                if catalogs is None:
                    budget.allowed_replan_ids=set(scope.replan_ids)
                    context=EvaluationContext(snapshot,budget=budget,validation_cache=self.provider.validation_cache,
                        reuse_prefix_results=True,scope=scope,global_frame=frame)
                    budget._certificate_context=context.snapshot_key
                    budget._strict_resource_context=context
                    catalogs=self.provider.iter_progressive_columns(snapshot,budget,context,scope=scope,early_publish=True)
                context.local_round=index
                started=time.perf_counter()
                with budget.measure("candidate_generation_and_single_vehicle"),budget.candidate_work():
                    try: catalog=next(catalogs)
                    except StopIteration:
                        if not calls: raise RuntimeError("v2 generator did not publish initial FOLLOW catalog")
                        break
                generation_wall=time.perf_counter()-started
                if set(catalog.columns)!=set(scope.replan_ids): raise ValueError("v2 catalog/scope mismatch")
                coverage=self._coverage(snapshot,catalog)
                budget.bind_catalog(catalog)
                mapped=self._mapped_incumbent(incumbent,catalog,context,budget)
                # Only current-frame re-evaluation supplies the comparison key.
                # If mapping is UNKNOWN/missing, the prior certified tuple is
                # retained rather than comparing incompatible old objectives.
                started=time.perf_counter(); solver_before=budget.solver_wall_used
                budget.solver_call_seconds=(min(max(0.,budget.remaining_seconds()-budget.solver_validation_headroom()),
                    max(0.,budget.solver_seconds-budget.solver_wall_used),
                    budget.solver_seconds if index==3 else max(.5,budget.solver_seconds/4.)) if service_profile else
                    min(2.,max(0.,budget.solver_seconds-budget.solver_wall_used)) if index==0 else max(0.,budget.solver_seconds-budget.solver_wall_used))
                with budget.measure("selection"):
                    result=self.scheduler.plan(snapshot,catalog,context,budget,incumbent=mapped)
                reserve=getattr(self.provider,"reserve_suffix_completion",None)
                if reserve is not None: reserve(snapshot,catalog,budget)
                solve_wall=time.perf_counter()-started
                if result.certified and first_certified is None:
                    first_certified=dict(call=index,wall_elapsed_s=budget.elapsed,objective=list(result.objective),
                        prefix_hashes=[p.trajectory_hash for p in result.prefix])
                if first is None: first=(result,catalog,context,scope)
                retained_previous=False
                if result.certified:
                    if incumbent is None or (mapped is not None and result.objective<=mapped.objective):
                        incumbent=result; best=(result,catalog,context,scope)
                    elif mapped is not None:
                        incumbent=mapped; best=(mapped,catalog,context,scope); retained_previous=True
                    else: retained_previous=True
                elif incumbent is not None:
                    if mapped is not None: incumbent=mapped; best=(mapped,catalog,context,scope)
                    retained_previous=True
                completed.update(context.completed_evaluations)
                blockers.extend(row for row in context.background_blockers if row not in blockers)
                for row in context.joint_conflict_observations:
                    if row not in joints: joints.append(row)
                for key,value in context.cache_stats.items():
                    if key.startswith("pair_"): pair_counts[key]=value+sum(c.get("pair_checks",{}).get(key,0) for c in calls if c["scope_identity"]!=scope.identity)
                catalog_row=dict(call=index,scope_identity=scope.identity,generation_wall_s=generation_wall,
                    coverage=coverage,generation_coverage=catalog.generation_coverage,
                    candidate_count=sum(map(len,catalog.columns.values())),wall_elapsed_s=budget.elapsed,
                    search_remaining_s=budget.remaining_seconds(),candidate_active_s=budget.candidate_elapsed)
                if service_profile:
                    from ..stage2_planning import _catalog_identity
                    catalog_row['catalog_identity']=_catalog_identity(catalog)
                catalog_rows.append(catalog_row)
                calls.append(dict(scope=scope.to_dict(),scope_identity=scope.identity,certified=result.certified,status=result.status,
                    objective=list(result.objective),best_objective=None if incumbent is None else list(incumbent.objective),
                    previous_best_retained=retained_previous,incumbent_mapped=mapped is not None,
                    first_certified=first_certified,blockers=list(context.background_blockers),
                    joint_conflict_observations=list(context.joint_conflict_observations),pair_checks=dict(context.cache_stats),
                    pair_conflict_evidence=list(context.pair_conflict_evidence),
                    pair_conflict_evidence_dropped=context.pair_conflict_evidence_dropped,
                    solver_records=list(result.solver_records),candidate_count=sum(map(len,catalog.columns.values())),
                    catalog_identity=catalog_row.get('catalog_identity'),
                    generation_wall_s=generation_wall,solve_wall_s=solve_wall,wall_elapsed_s=budget.elapsed,
                    search_remaining_s=budget.remaining_seconds(),solver_remaining_s=max(0.,budget.solver_seconds-budget.solver_wall_used),
                    solver_call_allocation_s=budget.solver_call_seconds,solver_wall_used_this_call_s=budget.solver_wall_used-solver_before,
                    candidate_generation_by_aircraft={aid:budget.screening.get("materialized:"+aid,0)-before_materialized[aid] for aid in snapshot.aircraft}))
                if context.diagnostics is not None:
                    calls[-1]["candidate_diagnostics"]=context.diagnostics.export(catalog,result.selected)
                if (index and not service_profile) or not budget.search_available(): break
                eligible=set(scope.eligible_ids); opened=set(scope.replan_ids)
                targets=sorted({row["target"] for row in context.background_blockers if row["target"] in eligible-opened
                    and row["snapshot_id"]==snapshot.version and row["reason"]=="RESOURCE_CONFLICT" and row["resource"]})
                if self.search_policy.ordinary_immediate_filter:
                    from .near_term import releasable_at_effective
                    executable_targets=[]
                    for aid in targets:
                        evidence=[releasable_at_effective(snapshot,aid,row["resource"]) for row in context.background_blockers
                            if row["target"]==aid and row["reason"]=="RESOURCE_CONFLICT" and row["resource"]]
                        if any(row["releasable"] for row in evidence): executable_targets.append(aid)
                        else: skipped_expansions.append(dict(aircraft_id=aid,reason="BLOCKER_CANNOT_RELEASE_AT_EFFECTIVE",access=evidence))
                    targets=executable_targets
                add=(targets[:max(0,scope.max_k-len(opened))] if scope.status=="LOCAL" and
                     (not service_profile or expansions<1) else [])
                if add:
                    catalogs.close(); catalogs=None; opened.update(add); expansions+=1; scope_round+=1
                    scope=replace(scope,replan_ids=tuple(sorted(opened)),fixed_background_ids=tuple(sorted(set(snapshot.aircraft)-opened)),
                        actual_k=len(opened),selection_reasons=scope.selection_reasons+tuple((aid,("BOUNDED_VALIDATED_BLOCKER_EXPANSION",)) for aid in add))
                    budget._selective_retained_columns={} if incumbent is None else {c.aircraft_id:c for c in incumbent.selected}
                    budget._selective_retained_snapshot=snapshot.version
                else:
                    gaps={aid:tuple(row["missing_reasons"]) for aid,row in coverage.items() if row["missing_reasons"]}
                    if not gaps: break
                    budget._selective_coverage_gaps=gaps
        finally:
            if catalogs is not None: catalogs.close()
            if old_call_limit is None: budget.__dict__.pop("solver_call_seconds",None)
            else: budget.solver_call_seconds=old_call_limit
            budget.__dict__.pop("_strict_resource_context",None)
            budget.__dict__.pop("_selective_target_coverage_order",None)
        result,catalog,context,accepted_scope=best if best is not None else first
        self.last_catalog,self.last_context,self.last_scope=catalog,context,accepted_scope
        self.completed_evaluations=completed
        self.telemetry=dict(scope=accepted_scope.to_dict(),scope_identity=accepted_scope.identity,objective_frame_id=frame.identity,
            evaluation_input_policy=frame.policy,known_aircraft_demand_ids=list(frame.order),upstream_request_ids=list(frame.upstream_request_ids),
            objective_eligibility=frame.objective_eligibility,upstream_requests=[dict(request_id=r[0],release_time=r[1],initial_soc=r[2],
                service_class=r[3],gate_id=r[4],admission_ready_time=r[5],admission_readiness=r[6],mask=r[7]) for r in frame.upstream_requests],
            solve_calls=len(calls),scope_expansions=expansions,calls=calls,catalog_coverage=catalog_rows,
            skipped_unexecutable_expansions=skipped_expansions,
            attempted_replan_ids=sorted({aid for call in calls for aid in call["scope"]["replan_ids"]}),
            candidate_generation_by_aircraft={aid:budget.screening.get("materialized:"+aid,0) for aid in snapshot.aircraft},
            pair_checks=pair_counts,background_blockers=blockers,joint_conflict_observations=joints,
            model_scale=[r["model_scale"] for call in calls for r in call["solver_records"] if "model_scale" in r],
            status="CERTIFIED_PREFIX" if result.prefix else "CERTIFIED_WAIT" if result.certified else "LOCAL_SEARCH_INCOMPLETE",
            completed_evaluation_ids=sorted(completed),first_certified=first_certified,
            first_nonempty_certified=budget.first_nonempty_certified,
            best_objective=list(result.objective),actual_fallback=bool(result.details.get("fallback",False)) or not result.certified,wall_elapsed_s=budget.elapsed,
            search_remaining_s=budget.remaining_seconds(),solver_remaining_s=max(0.,budget.solver_seconds-budget.solver_wall_used))
        report=getattr(self.provider,"suffix_completion_report",lambda budget:None)(budget)
        if report is not None: self.telemetry["suffix_completion"]=report
        return result
