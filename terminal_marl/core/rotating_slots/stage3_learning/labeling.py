"""Finite same-catalog reference labels; missing evidence remains masked."""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
import time

from ..stage2_campaign import atomic_json,digest,load_json
from .data import PortableObserver,StratifiedProbe,restore_root,auxiliary_labels,summarize_coverage


def comparable(a,b):
    keys=("snapshot_id","source_node_id","aircraft_id","parent_prefix_id","construction_stage")
    return all(a.get(k)==b.get(k) for k in keys)


def derive_pairs(descriptors, facts, reference_solutions, *, conditional_attempts=(), context_id, root_id, split, fingerprint):
    """Only certified observed strict preferences; no unselected pseudo-negatives."""
    from .campaign import contract_config
    limits=contract_config()["labeling"]
    if len(reference_solutions)>limits["certified_reference_solutions_max"]:
        raise ValueError("saved reference support exceeds frozen solution cap")
    if len(conditional_attempts)>limits["conditional_resolves_max_per_snapshot"]:
        raise ValueError("conditional attempts exceed frozen resolve cap")
    by_id={d["descriptor_id"]:d for d in descriptors}
    supported=set()
    for solution in reference_solutions:
        if solution.get("certified"):supported.update(solution.get("descriptor_ids",()))
    invalid={f["descriptor_id"] for f in facts if f.get("outcome")=="SINGLE_INVALID_CERTIFIED" and f.get("validity_label_mask",False) and not f.get("censored",False)}
    pairs={}
    def add(a,b,source,evidence):
        if a==b or a not in by_id or b not in by_id or not comparable(by_id[a],by_id[b]):return
        pairs[(a,b)]=dict(record_type="PreferencePair",context_id=by_id[a]["source_node_id"],preferred_descriptor=a,other_descriptor=b,
            label_source=source,objective_id=fingerprint["objective_id"],label_policy_id="bounded_reference_supported_pair_v1_0",
            ancestry_root_id=root_id,split=split,evidence_refs=list(evidence))
    for a in sorted(supported):
        for b in sorted(invalid):
            evidence=[s["solution_id"] for s in reference_solutions if s.get("certified") and a in s.get("descriptor_ids",())]
            add(a,b,"supported_vs_certified_single_invalid",evidence+["fact:"+b])
    conditional=[s for s in conditional_attempts if s.get("certified") and s.get("required_descriptor")]
    for a,b in combinations(conditional,2):
        if a["catalog_id"]!=b["catalog_id"] or a["budget_settings"]!=b["budget_settings"]:continue
        if tuple(a["objective"])==tuple(b["objective"]):continue
        better,worse=(a,b) if tuple(a["objective"])<tuple(b["objective"]) else (b,a)
        add(better["required_descriptor"],worse["required_descriptor"],
            "bounded_conditional_certified_objective_preference",[better["solution_id"],worse["solution_id"]])
    return list(pairs.values())


def conditional_catalog(catalog, descriptor_id, ancestry):
    """Require >=1 generated descendant while retaining original downstream object.

    A descriptor belongs to exactly one aircraft. Restricting that aircraft's
    choices is the exact branch constraint; no objective or baseline is rebuilt.
    """
    targets={aid:[c for c in columns if descriptor_id in ancestry.get(c.signature,())]
             for aid,columns in catalog.columns.items()}
    targets={aid:values for aid,values in targets.items() if values}
    if len(targets)!=1:return None
    aid,columns=next(iter(targets.items()))
    return replace(catalog,columns=MappingProxyType(dict(catalog.columns,**{aid:tuple(columns)})))


def _column_ancestry(catalog, records):
    rows=[r for r in records if r.get("record_type")=="ColumnAncestry"]
    result={}
    for aid,columns in catalog.columns.items():
        for c in columns:
            hashes=tuple(p.trajectory_hash for p in c.plans)
            matches=[r for r in rows if r.get("aircraft_id")==aid and tuple(r.get("trajectory_hashes",()))==hashes]
            result[c.signature]=tuple(dict.fromkeys(d for r in matches for d in r.get("descriptor_ids",())))
    return result


def _solution(result, ancestry, catalog_id, method, *, required=None,budget=None):
    # `certified` alone is insufficient: original final prefix authorization is mandatory.
    certified=bool(result.certified and result.evaluation is not None and result.evaluation.accepted)
    descriptors=sorted({d for c in result.selected for d in ancestry.get(c.signature,())})
    if required is not None and required not in descriptors:certified=False
    row=dict(method=method,catalog_id=catalog_id,certified=certified,status=result.status,
        objective=list(result.objective),descriptor_ids=descriptors,required_descriptor=required,
        selected_signatures=[repr(c.signature) for c in result.selected],budget_settings=budget.settings() if budget else {})
    row["solution_id"]="solution:"+digest(row)
    return row


def label_snapshot(output,root_id,binding):
    from ..stage2_planning import (PlanningBudget,CandidateProvider,EvaluationContext,
        NormalSchedulerCallback,EnergySlackScheduler,RollingHorizonCPSATScheduler)
    from .runtime import AdvisorRuntime
    from .campaign import contract_config
    output=Path(output);root=load_json(output/"roots"/root_id/"root.json")
    directory=output/"labels"/root_id;directory.mkdir(parents=True,exist_ok=True)
    if (directory/"labels.json").exists():return load_json(directory/"labels.json")
    raw=output/"raw"/root_id
    # Failed partial jobs may be retried, but cannot append a second trace to stale facts.
    if raw.exists() and any(raw.iterdir()):
        raise ValueError("partial raw trace exists; retain evidence and return to primary before retry")
    observer=PortableObserver(raw)
    cfg=contract_config()["labeling"];started=time.perf_counter();deadline=started+cfg["per_snapshot_total_wall_seconds"]
    world=restore_root(root,NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8"))
    snapshot=world.build_planning_snapshot()
    runtime=AdvisorRuntime(mode="SHADOW_OFFLINE",fingerprint=binding["fingerprint"],observer=observer,
        ancestry_root_id=root_id,evaluation=False,offline_probe=StratifiedProbe(cfg["offline_probe_seed"]).select)
    budget=PlanningBudget.for_profile("quality_v5")
    budget.candidate_seconds=min(cfg["reference_candidate_wall_cap_seconds"],budget.candidate_seconds*cfg["reference_candidate_budget_multiplier"])
    budget.max_attempts*=2;budget.max_per_aircraft*=2
    budget.soft_seconds=max(budget.soft_seconds,budget.candidate_seconds)
    context=EvaluationContext(snapshot,budget=budget,validation_cache={})
    provider=CandidateProvider(planning_profile="downstream_service_v8",advisor_runtime=runtime)
    catalog=provider.get_columns(snapshot,budget,context)
    ancestry=_column_ancestry(catalog,observer.records)
    catalog_id=digest([repr(c.signature) for cs in catalog.columns.values() for c in cs])
    solutions=[]
    def reference(method,limit,required=None):
        remaining=deadline-time.perf_counter()
        if remaining<=0:return None
        local=PlanningBudget.for_profile("quality_v5")
        local.soft_seconds=local.watchdog_seconds=min(limit,remaining)
        if required is not None:
            local.solver_seconds=min(3.,local.soft_seconds)
            local.hint_seconds=min(.5,local.soft_seconds)
            local.solver_reserve_seconds=0.
        candidates=catalog if required is None else conditional_catalog(catalog,required,ancestry)
        if candidates is None:return None
        # Independent cache and scheduler per reference; same materialized catalog and target.
        ctx=EvaluationContext(snapshot,budget=local,validation_cache={})
        scheduler=EnergySlackScheduler() if method=="energy_slack" else RollingHorizonCPSATScheduler()
        result=scheduler.plan(snapshot,candidates,ctx,local)
        return _solution(result,ancestry,catalog_id,method,required=required,budget=local)
    for method in cfg["reference_solvers"]:
        solution=reference(method,30.)
        if solution is not None and solution["certified"]:solutions.append(solution)
    descriptors=list({r["descriptor_id"]:r for r in observer.records if r.get("record_type")=="WorkDescriptor"}.values())
    # Bound conditional attempts before observing conditional outcomes.
    eligible=sorted({d for values in ancestry.values() for d in values})
    conditional_ids=[]
    candidates=[d for d in descriptors if d["descriptor_id"] in eligible]
    for a,b in combinations(candidates,2):
        if comparable(a,b):
            for d in (a["descriptor_id"],b["descriptor_id"]):
                if d not in conditional_ids:conditional_ids.append(d)
        if len(conditional_ids)>=cfg["conditional_resolves_max_per_snapshot"]:break
    conditional_attempts=[]
    for descriptor_id in conditional_ids[:cfg["conditional_resolves_max_per_snapshot"]]:
        solution=reference("rh_cp_sat",cfg["conditional_resolve_wall_cap_seconds"],descriptor_id)
        if solution is not None:conditional_attempts.append(solution)
    facts=[r for r in observer.records if r.get("record_type") in ("FactualResult","SingleVehicleEvidence")]
    # Final single-vehicle evidence augments the recorded pre-query timed fact.
    evidence={r["descriptor_id"]:r for r in facts if r.get("record_type")=="SingleVehicleEvidence"}
    timed=[dict(r,**{k:v for k,v in evidence.get(r["descriptor_id"],{}).items() if k in ("outcome","validity_label_mask","energy_j")})
        for r in facts if r.get("record_type")=="FactualResult"]
    for fact in timed:
        energy=evidence.get(fact["descriptor_id"],{})
        fact["generation_wall_s"]=fact["wall_s"]
        fact["energy_wall_s"]=energy.get("wall_s")
        fact["wall_s"]=(fact["wall_s"] or 0.)+(energy.get("wall_s") or 0.)
        fact["cpu_s"]=(fact.get("cpu_s") or 0.)+(energy.get("cpu_s") or 0.)
        fact["cost_boundary"]="GENERATOR_PLUS_OBSERVED_SINGLE_VEHICLE_ENERGY_EXCLUDES_JOINT_VALIDATION"
    context_id=getattr(runtime,"snapshot_id",None) or (descriptors[0]["snapshot_id"] if descriptors else root["snapshot_id"])
    reference_solutions=[];seen=set()
    for solution in solutions+conditional_attempts:
        key=tuple(solution["selected_signatures"])
        if solution["certified"] and key not in seen and len(reference_solutions)<cfg["certified_reference_solutions_max"]:
            seen.add(key);reference_solutions.append(solution)
    pairs=derive_pairs(descriptors,timed,reference_solutions,conditional_attempts=conditional_attempts,context_id=context_id,root_id=root_id,split=root["split"],fingerprint=binding["fingerprint"])
    support={d for solution in reference_solutions for d in solution["descriptor_ids"]}
    result=dict(status="LABELED" if pairs else "DATA_INSUFFICIENT",context_id=context_id,root_id=root_id,split=root["split"],
        sealed=root["split"]=="test",fingerprint=binding["fingerprint"],catalog_id=catalog_id,
        reference_solutions=reference_solutions,conditional_attempts=conditional_attempts,preference_pairs=pairs,
        auxiliary_labels=auxiliary_labels(timed),facts=timed,wall_s=time.perf_counter()-started,
        coverage_summary=summarize_coverage(descriptors,timed,support,support_kind="SAVED_BOUNDED_REFERENCE_ANCESTRY"),
        unsupported_valid_is_negative=False,conditional_is_global_oracle=False)
    atomic_json(directory/"labels.json",result)
    return result
