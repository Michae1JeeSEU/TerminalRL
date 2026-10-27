"""S candidates + complete fixed world, at most two shared-budget solves."""
from dataclasses import replace, asdict
from .scope import PlanningScope


class LocalPlanner:
    def __init__(self, provider, scheduler): self.provider, self.scheduler = provider, scheduler

    def plan(self, snapshot, scope, global_frame, shared_budget):
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
                solver_records=list(result.solver_records),candidate_count=sum(map(len,catalog.columns.values())),
                candidate_generation_by_aircraft={aid:budget.screening.get("materialized:"+aid,0)-before_materialized[aid] for aid in snapshot.aircraft}))
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
