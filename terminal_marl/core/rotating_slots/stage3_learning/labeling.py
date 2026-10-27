"""Finite same-catalog reference labels; missing evidence remains masked."""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
import time

from ..stage2_campaign import atomic_json,digest,load_json
from .data import PortableObserver,StratifiedProbe,restore_root,auxiliary_labels,summarize_coverage

DEFAULT_LABEL_POLICY = 'bounded_reference_supported_pair_v1_0'
JOINT_CONTRIBUTION_POLICY = 'bounded_joint_contribution_v2'


def _label_policy(policy):
    policy = DEFAULT_LABEL_POLICY if policy is None else policy
    if policy not in (DEFAULT_LABEL_POLICY, JOINT_CONTRIBUTION_POLICY):
        raise ValueError('unsupported label policy: '+str(policy))
    return policy


def comparable(a,b):
    keys=("snapshot_id","source_node_id","aircraft_id","parent_prefix_id","construction_stage")
    return all(a.get(k)==b.get(k) for k in keys)


def derive_pairs(descriptors, facts, reference_solutions, *, conditional_attempts=(), context_id, root_id, split, fingerprint,
                 label_policy=None, increments=()):
    """Only certified observed strict preferences; no unselected pseudo-negatives."""
    from .campaign import contract_config
    policy=_label_policy(label_policy)
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
            label_source=source,objective_id=fingerprint["objective_id"],label_policy_id=policy,
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
    result=list(pairs.values())
    if policy==JOINT_CONTRIBUTION_POLICY:
        contribution=joint_contribution_pairs(descriptors,facts,conditional_attempts,increments,
            root_id=root_id,split=split,fingerprint=fingerprint)
        replaced={frozenset((p['preferred_descriptor'],p['other_descriptor'])) for p in contribution}
        result=[p for p in result if p['label_source']!='bounded_conditional_certified_objective_preference' or
            frozenset((p['preferred_descriptor'],p['other_descriptor'])) not in replaced]
        result.extend(contribution)
        unordered=[frozenset((p['preferred_descriptor'],p['other_descriptor'])) for p in result]
        assert len(unordered)==len(set(unordered)), 'duplicate or contradictory unordered preference pair'
    return result


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


def _planning_snapshot_provenance(snapshot,provenance):
    """Validate caller-supplied revealed projection metadata without projecting it."""
    from collections.abc import Mapping
    from copy import deepcopy
    import math
    from ..world import PlanningSnapshot
    if snapshot is None:
        if provenance is not None:raise ValueError('planning snapshot provenance requires a snapshot')
        return None
    if not isinstance(snapshot,PlanningSnapshot):
        raise ValueError('planning_snapshot must be an immutable PlanningSnapshot')
    required={'source_time','effective_time','version','projection'}
    if not isinstance(provenance,Mapping) or not required.issubset(provenance):
        raise ValueError('planning snapshot provenance requires source_time/effective_time/version/projection')
    source,effective=provenance['source_time'],provenance['effective_time']
    if (not all(isinstance(t,(int,float)) and not isinstance(t,bool) and math.isfinite(t) for t in (source,effective))
            or effective<source):
        raise ValueError('planning snapshot effective_time must be finite and >= source_time')
    if snapshot.time!=effective or snapshot.version!=provenance['version']:
        raise ValueError('planning snapshot provenance time/version mismatch')
    if not isinstance(provenance['projection'],Mapping) or not provenance['projection']:
        raise ValueError('planning snapshot projection provenance must be a nonempty mapping')
    return deepcopy(dict(provenance))


def label_snapshot(output,root_id,binding, *, schema_id=None,label_policy=None,
                   planning_snapshot=None,planning_snapshot_provenance=None):
    from ..stage2_planning import (PlanningBudget,CandidateProvider,EvaluationContext,
        NormalSchedulerCallback,EnergySlackScheduler,RollingHorizonCPSATScheduler)
    from .runtime import AdvisorRuntime
    from .campaign import contract_config
    from .features import SCHEMA_V2
    policy=_label_policy(label_policy)
    v2 = schema_id == SCHEMA_V2
    if policy==JOINT_CONTRIBUTION_POLICY:
        if not v2:raise ValueError('joint contribution policy requires SCHEMA_V2')
        if binding['fingerprint'].get('label_policy_id')!=policy:
            raise ValueError('binding fingerprint label policy does not match requested policy')
    snapshot_provenance=_planning_snapshot_provenance(planning_snapshot,planning_snapshot_provenance)
    output=Path(output);root=load_json(output/"roots"/root_id/"root.json")
    ancestry_root_id=root.get('ancestry_root_id',root_id)
    directory=output/"labels"/root_id;directory.mkdir(parents=True,exist_ok=True)
    if (directory/"labels.json").exists():
        saved=load_json(directory/'labels.json')
        if v2 and saved.get('feature_schema_id') != SCHEMA_V2:
            raise ValueError('existing labels use a different feature schema')
        if saved.get('label_policy_id',DEFAULT_LABEL_POLICY)!=policy:
            raise ValueError('existing labels use a different label policy')
        if saved.get('planning_snapshot_provenance')!=snapshot_provenance:
            raise ValueError('existing labels use a different planning snapshot')
        return saved
    raw=output/"raw"/root_id
    # Failed partial jobs may be retried, but cannot append a second trace to stale facts.
    if raw.exists() and any(raw.iterdir()):
        raise ValueError("partial raw trace exists; retain evidence and return to primary before retry")
    observer=PortableObserver(raw)
    cfg=contract_config()["labeling"];started=time.perf_counter();deadline=started+cfg["per_snapshot_total_wall_seconds"]
    if planning_snapshot is None:
        world=restore_root(root,NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8"))
        snapshot=world.build_planning_snapshot()
    else:
        snapshot=planning_snapshot
    runtime=AdvisorRuntime(mode="SHADOW_OFFLINE",fingerprint=binding["fingerprint"],observer=observer,
        ancestry_root_id=ancestry_root_id,evaluation=False,offline_probe=StratifiedProbe(cfg["offline_probe_seed"]).select,
        feature_schema=schema_id)
    budget=PlanningBudget.for_profile("quality_v5")
    budget.candidate_seconds=min(cfg["reference_candidate_wall_cap_seconds"],budget.candidate_seconds*cfg["reference_candidate_budget_multiplier"])
    budget.max_attempts*=2;budget.max_per_aircraft*=2
    budget.soft_seconds=max(budget.soft_seconds,budget.candidate_seconds)
    context=EvaluationContext(snapshot,budget=budget,validation_cache={})
    provider=CandidateProvider(planning_profile="downstream_service_v8",advisor_runtime=runtime)
    if v2:
        budget.candidate_active_seconds=0.
        batches=provider.iter_progressive_columns(snapshot,budget,context)
        catalog=None
        while True:
            try:
                with budget.candidate_work():
                    catalog=next(batches)
            except StopIteration:
                break
        if catalog is None:
            raise ValueError('progressive collection produced no catalog')
    else:
        catalog=provider.get_columns(snapshot,budget,context)
    ancestry=_column_ancestry(catalog,observer.records)
    catalog_id=digest([repr(c.signature) for cs in catalog.columns.values() for c in cs])
    solutions=[]
    def reference(method,limit,required=None,excluded=None):
        remaining=deadline-time.perf_counter()
        if remaining<=0:return None
        local=PlanningBudget.for_profile("quality_v5")
        local.soft_seconds=local.watchdog_seconds=min(limit,remaining)
        if required is not None or excluded is not None:
            local.solver_seconds=min(3.,local.soft_seconds)
            local.hint_seconds=min(.5,local.soft_seconds)
            local.solver_reserve_seconds=0.
        candidates=catalog if required is None else conditional_catalog(catalog,required,ancestry)
        if candidates is None:return None
        if excluded is not None:
            filtered={aid:tuple(c for c in cs if excluded not in ancestry.get(c.signature,())) for aid,cs in catalog.columns.items()}
            if any(not cs for cs in filtered.values()):return None
            candidates=replace(catalog,columns=MappingProxyType(filtered))
        # Independent cache and scheduler per reference; same materialized catalog and target.
        ctx=EvaluationContext(snapshot,budget=local,validation_cache={})
        scheduler=EnergySlackScheduler() if method=="energy_slack" else RollingHorizonCPSATScheduler()
        result=scheduler.plan(snapshot,candidates,ctx,local)
        row = _solution(result,ancestry,catalog_id,method,required=required,budget=local)
        if excluded is not None:
            row.update(excluded_descriptor=excluded,removed_signatures=[repr(c.signature) for cs in catalog.columns.values() for c in cs if excluded in ancestry.get(c.signature,())],
                counterfactual_catalog_id=digest([repr(c.signature) for cs in candidates.columns.values() for c in cs]))
            row['solution_id']=canonical_solution_id(row)
        return row
    for method in cfg["reference_solvers"]:
        solution=reference(method,30.)
        if solution is not None and solution["certified"]:solutions.append(solution)
    descriptors=list({r["descriptor_id"]:r for r in observer.records if r.get("record_type")=="WorkDescriptor"}.values())
    # Bound conditional attempts before observing conditional outcomes.
    eligible=sorted({d for values in ancestry.values() for d in values})
    if v2:
        terminal_hashes={c.plans[-1].trajectory_hash for cs in catalog.columns.values() for c in cs if c.plans}
        eligible=sorted({ids[-1] for h,ids in runtime._plans.items() if h in terminal_hashes and ids})
    conditional_ids=[]
    candidates=[d for d in descriptors if d["descriptor_id"] in eligible]
    for a,b in combinations(candidates,2):
        if comparable(a,b):
            for d in (a["descriptor_id"],b["descriptor_id"]):
                if d not in conditional_ids:conditional_ids.append(d)
        if len(conditional_ids)>=cfg["conditional_resolves_max_per_snapshot"]:break
    conditional_attempts=[]
    incremental=[]
    limit=cfg["conditional_resolves_max_per_snapshot"]
    for descriptor_id in conditional_ids[:limit//2 if v2 else limit]:
        solution=reference("rh_cp_sat",cfg["conditional_resolve_wall_cap_seconds"],descriptor_id)
        if solution is not None:conditional_attempts.append(solution)
        if v2:
            removal=reference('rh_cp_sat',cfg['conditional_resolve_wall_cap_seconds'],excluded=descriptor_id)
            if removal is not None:conditional_attempts.append(removal)
            if solution is not None and removal is not None:
                incremental.append(incremental_evidence(descriptor_id,solution,removal))
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
    if v2:
        timed=complete_cost_facts(observer.records)
    context_id=getattr(runtime,"snapshot_id",None) or (descriptors[0]["snapshot_id"] if descriptors else root["snapshot_id"])
    reference_solutions=[];seen=set()
    for solution in solutions+conditional_attempts:
        key=tuple(solution["selected_signatures"])
        if solution["certified"] and key not in seen and len(reference_solutions)<cfg["certified_reference_solutions_max"]:
            seen.add(key);reference_solutions.append(solution)
    pairs=derive_pairs(descriptors,timed,reference_solutions,conditional_attempts=conditional_attempts,context_id=context_id,root_id=ancestry_root_id,split=root["split"],fingerprint=binding["fingerprint"],label_policy=policy,increments=incremental)
    if v2 and policy==DEFAULT_LABEL_POLICY:
        pairs += supplementary_pairs(descriptors,timed,conditional_attempts,incremental,root_id=ancestry_root_id,split=root['split'],fingerprint=binding['fingerprint'])
    support={d for solution in reference_solutions for d in solution["descriptor_ids"]}
    result=dict(status="LABELED" if pairs else "DATA_INSUFFICIENT",context_id=context_id,root_id=root_id,split=root["split"],
        sealed=root["split"]=="test",fingerprint=binding["fingerprint"],catalog_id=catalog_id,
        reference_solutions=reference_solutions,conditional_attempts=conditional_attempts,preference_pairs=pairs,
        auxiliary_labels=auxiliary_labels(timed),facts=timed,wall_s=time.perf_counter()-started,
        coverage_summary=summarize_coverage(descriptors,timed,support,support_kind="SAVED_BOUNDED_REFERENCE_ANCESTRY"),
        unsupported_valid_is_negative=False,conditional_is_global_oracle=False)
    if policy==JOINT_CONTRIBUTION_POLICY:
        result.update(label_policy_id=policy,contribution_is_global_oracle=False,
            contribution_is_causal_guarantee=False)
    if snapshot_provenance is not None:
        result['planning_snapshot_provenance']=snapshot_provenance
    if v2:
        result.update(feature_schema_id=SCHEMA_V2,incremental_contribution_evidence=incremental,ancestry_root_id=ancestry_root_id,
            candidate_timer_regime='CUMULATIVE_ACTIVE_GENERATION_SHARED_ORIGINAL_CLOCK',
            complete_cost_facts=sum(bool(f.get('cost_complete')) for f in timed),incremental_labeled_descriptors=sum(bool(i.get('contribution_mask')) for i in incremental))
    atomic_json(directory/"labels.json",result)
    return result


def complete_cost_facts(records):
    """Join measured components by exact query reference, never descriptor overwrite."""
    import math
    energies,validations={},{}
    for row in records:
        ref=row.get('features_before_query_ref')
        if ref and row.get('record_type')=='SingleVehicleEvidence':energies.setdefault(ref,[]).append(row)
        if ref and row.get('record_type')=='ColumnValidationCost':validations.setdefault(ref,[]).append(row)
    result=[]
    def measured(row):
        value=row.get('wall_s')
        return isinstance(value,(int,float)) and math.isfinite(value) and value>=0
    for source in records:
        if source.get('record_type')!='FactualResult':continue
        fact=dict(source);ref=fact.get('features_before_query_ref');es=energies.get(ref,[]);vs=validations.get(ref,[])
        if es:
            fact.update({k:v for k,v in es[-1].items() if k in ('outcome','validity_label_mask','energy_j')})
        # A generated plan requires observed energy and terminal validation. A
        # bounded no-plan return has no energy/validation operation to charge.
        no_plan=fact.get('trajectory_hash') is None
        components=[source]+es+vs
        complete=not fact.get('censored',False) and measured(source) and (no_plan or bool(es and vs)) and all(measured(r) for r in components)
        fact.update(cost_scope='GENERATOR_ENERGY_TERMINAL_VALIDATION_V2',cost_complete=complete,
            generation_wall_s=source.get('wall_s'),energy_wall_s=sum(r['wall_s'] for r in es if measured(r)) if es else None,
            column_validation_wall_s=sum(r['wall_s'] for r in vs if measured(r)) if vs else None,
            wall_s=sum(r['wall_s'] for r in components if measured(r)) if complete else None,
            cost_component_refs=[ref] if ref else [])
        result.append(fact)
    return result


def incremental_evidence(descriptor_id,inclusion,removal):
    """Bounded same-catalog paired solves; local contribution, never global optimum."""
    certified=bool(inclusion.get('certified') and removal.get('certified') and
        inclusion.get('catalog_id')==removal.get('catalog_id') and inclusion.get('budget_settings')==removal.get('budget_settings') and
        inclusion.get('required_descriptor')==descriptor_id and removal.get('excluded_descriptor')==descriptor_id and
        descriptor_id in inclusion.get('descriptor_ids',()) and descriptor_id not in removal.get('descriptor_ids',()) and len(removal.get('removed_signatures',()))==1)
    return dict(descriptor_id=descriptor_id,contribution_mask=certified,
        included_objective=inclusion.get('objective'),removed_objective=removal.get('objective'),
        increment=[b-a for a,b in zip(inclusion['objective'],removal['objective'])] if certified else None,
        evidence_refs=[inclusion['solution_id'],removal['solution_id']],
        catalog_id=inclusion.get('catalog_id'),counterfactual_catalog_id=removal.get('counterfactual_catalog_id'),
        removed_signatures=removal.get('removed_signatures',[]),scope='BOUNDED_CERTIFIED_INCLUSION_REMOVAL',
        missing_reason=None if certified else 'UNCERTIFIED_NONCOMPARABLE_OR_MULTIPLE_DESCENDANT_REMOVAL')


def joint_contribution_pairs(descriptors,facts,solutions,increments,*,root_id,split,fingerprint):
    """Observed finite-budget relative contribution, with four exact solve bindings.

    These comparisons neither establish a global optimum nor a causal effect.
    An increment is admitted only when its two references uniquely identify the
    certified, single-signature inclusion/removal interventions that produced it.
    """
    import math
    by_id={d['descriptor_id']:d for d in descriptors}
    solutions_by_ref={}
    for row in solutions:
        solutions_by_ref.setdefault(row.get('solution_id'),[]).append(row)
    increments_by_descriptor={}
    for row in increments:
        increments_by_descriptor.setdefault(row['descriptor_id'],[]).append(row)
    bound={}
    for did,rows in increments_by_descriptor.items():
        if len(rows)!=1 or not rows[0].get('contribution_mask'):continue
        row=rows[0];refs=row.get('evidence_refs',())
        if len(refs)!=2 or len(set(refs))!=2 or any(len(solutions_by_ref.get(ref,()))!=1 for ref in refs):continue
        included,removed=(solutions_by_ref[ref][0] for ref in refs)
        if included.get('excluded_descriptor') is not None or removed.get('required_descriptor') is not None:continue
        objectives=(included.get('objective'),removed.get('objective'))
        if (not objectives[0] or not objectives[1] or len(objectives[0])!=len(objectives[1]) or
                not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for obj in objectives for v in obj)):continue
        verified=incremental_evidence(did,included,removed)
        if not verified['contribution_mask']:continue
        fields=('included_objective','removed_objective','increment','catalog_id','counterfactual_catalog_id','removed_signatures')
        if any(row.get(key)!=verified.get(key) for key in fields):continue
        if not included.get('catalog_id') or not included.get('budget_settings'):continue
        bound[did]=(row,included,removed)
    by_fact={}
    for fact in facts:
        if fact.get('cost_complete'):by_fact.setdefault(fact['descriptor_id'],[]).append(fact)
    pairs=[]
    for da,db in combinations(sorted(bound),2):
        if da not in by_id or db not in by_id or not comparable(by_id[da],by_id[db]):continue
        ia,a,ar=bound[da];ib,b,br=bound[db]
        if a['catalog_id']!=b['catalog_id'] or a['budget_settings']!=b['budget_settings']:continue
        refs=ia['evidence_refs']+ib['evidence_refs']
        if len(set(refs))!=4:continue
        delta_a,delta_b=tuple(ia['increment']),tuple(ib['increment'])
        if len(delta_a)!=len(delta_b):continue
        source=None;cost_refs=[]
        if delta_a!=delta_b:
            preferred,other=(da,db) if delta_a>delta_b else (db,da)
            source='observed_bounded_joint_contribution_preference'
        elif len(by_fact.get(da,[]))==len(by_fact.get(db,[]))==1:
            fa,fb=by_fact[da][0],by_fact[db][0]
            costs=(fa.get('wall_s'),fb.get('wall_s'))
            if (fa.get('cost_scope') and fa.get('cost_scope')==fb.get('cost_scope') and
                    all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0 for v in costs) and
                    costs[0]!=costs[1] and fa.get('features_before_query_ref') and fb.get('features_before_query_ref') and
                    fa['features_before_query_ref']!=fb['features_before_query_ref']):
                preferred,other=(da,db) if costs[0]<costs[1] else (db,da)
                source='observed_bounded_joint_contribution_measured_cost_preference'
                cost_refs=[fa['features_before_query_ref'],fb['features_before_query_ref']]
        if source:
            pairs.append(dict(record_type='PreferencePair',context_id=by_id[da]['source_node_id'],preferred_descriptor=preferred,other_descriptor=other,
                label_source=source,objective_id=fingerprint['objective_id'],label_policy_id=JOINT_CONTRIBUTION_POLICY,
                ancestry_root_id=root_id,split=split,evidence_refs=list(refs),cost_evidence_refs=cost_refs,
                catalog_id=a['catalog_id'],budget_settings=dict(a['budget_settings']),
                contribution_evidence=[dict(descriptor_id=did,included_solution_ref=inc['solution_id'],removed_solution_ref=rem['solution_id'],
                    included_objective=list(inc['objective']),removed_objective=list(rem['objective']),delta=list(increment['increment']),
                    counterfactual_catalog_id=rem.get('counterfactual_catalog_id'),removed_signatures=list(rem['removed_signatures']))
                    for did,increment,inc,rem in ((da,ia,a,ar),(db,ib,b,br))],
                contribution_scope='OBSERVED_BOUNDED_SAME_CATALOG_SINGLE_SIGNATURE_INCLUSION_REMOVAL',
                global_oracle=False,causal_guarantee=False))
    return pairs


def supplementary_pairs(descriptors,facts,solutions,increments,*,root_id,split,fingerprint):
    """Original quality dominates; use contribution/cost only at certified ties."""
    by_id={d['descriptor_id']:d for d in descriptors}
    by_increment={r['descriptor_id']:r for r in increments if r.get('contribution_mask')}
    by_fact={}
    for fact in facts:
        if fact.get('cost_complete'):by_fact.setdefault(fact['descriptor_id'],[]).append(fact)
    eligible=[s for s in solutions if s.get('certified') and s.get('required_descriptor')]
    pairs=[]
    for a,b in combinations(eligible,2):
        da,db=a['required_descriptor'],b['required_descriptor']
        if da not in by_id or db not in by_id or not comparable(by_id[da],by_id[db]):continue
        if a['catalog_id']!=b['catalog_id'] or a['budget_settings']!=b['budget_settings'] or a['objective']!=b['objective']:continue
        ia,ib=by_increment.get(da),by_increment.get(db)
        source=None;evidence=[]
        if ia and ib and ia['increment']!=ib['increment']:
            preferred,other=(da,db) if tuple(ia['increment'])>tuple(ib['increment']) else (db,da)
            source='equal_quality_incremental_preference';evidence=ia['evidence_refs']+ib['evidence_refs']
        elif ia and ib and ia['increment']==ib['increment'] and len(by_fact.get(da,[]))==len(by_fact.get(db,[]))==1:
            fa,fb=by_fact[da][0],by_fact[db][0]
            if fa['cost_scope']==fb['cost_scope'] and fa['wall_s']!=fb['wall_s']:
                preferred,other=(da,db) if fa['wall_s']<fb['wall_s'] else (db,da)
                source='equal_quality_measured_cost_preference';evidence=[fa['features_before_query_ref'],fb['features_before_query_ref']]
        if source:
            pairs.append(dict(record_type='PreferencePair',context_id=by_id[da]['source_node_id'],preferred_descriptor=preferred,other_descriptor=other,
                label_source=source,objective_id=fingerprint['objective_id'],label_policy_id='bounded_reference_supported_pair_v1_0',
                ancestry_root_id=root_id,split=split,evidence_refs=evidence))
    return pairs


def canonical_solution_id(row):
    """Intervention-qualified content identity, computed after all metadata exists."""
    return 'solution:'+digest({k:v for k,v in row.items() if k!='solution_id'})


def reindex_saved_labels(labels,descriptors):
    """Repair saved evidence references only; no generation, scheduler or solves.

    Matching interventions uses required/excluded descriptors, never legacy IDs
    which may collide. The original saved reference subset and order are kept.
    """
    from copy import deepcopy
    result=deepcopy(labels)
    for key in ('conditional_attempts','reference_solutions'):
        for row in result[key]:row['solution_id']=canonical_solution_id(row)
    conditional=result['conditional_attempts']
    increments=[]
    for original in labels.get('incremental_contribution_evidence',[]):
        did=original['descriptor_id']
        included=[r for r in conditional if r.get('required_descriptor')==did]
        removed=[r for r in conditional if r.get('excluded_descriptor')==did]
        if len(included)!=1 or len(removed)!=1:
            raise ValueError('ambiguous saved intervention match: '+did)
        increments.append(incremental_evidence(did,included[0],removed[0]))
    kwargs=dict(root_id=labels.get('ancestry_root_id',labels['root_id']),split=labels['split'],fingerprint=labels['fingerprint'])
    policy=labels.get('label_policy_id',DEFAULT_LABEL_POLICY)
    pairs=derive_pairs(descriptors,labels['facts'],result['reference_solutions'],conditional_attempts=conditional,context_id=labels['context_id'],
        label_policy=policy,increments=increments,**kwargs)
    if policy==DEFAULT_LABEL_POLICY:
        pairs+=supplementary_pairs(descriptors,labels['facts'],conditional,increments,**kwargs)
    def pair_semantics(rows):
        return sorted((r['context_id'],r['preferred_descriptor'],r['other_descriptor'],r['label_source'],r['objective_id'],r['split'],r['ancestry_root_id']) for r in rows)
    if pair_semantics(pairs)!=pair_semantics(labels['preference_pairs']):
        raise ValueError('reference repair changed preference semantics/count')
    strip_refs=lambda rows:[{k:v for k,v in r.items() if k!='evidence_refs'} for r in rows]
    if strip_refs(increments)!=strip_refs(labels.get('incremental_contribution_evidence',[])):
        raise ValueError('reference repair changed contribution semantics')
    result.update(preference_pairs=pairs,incremental_contribution_evidence=increments,
        reindex_provenance=dict(policy='INTERVENTION_QUALIFIED_SOLUTION_IDS_ONLY',source_labels_content_sha256=digest(labels),
            source_fingerprint=deepcopy(labels['fingerprint']),no_new_solves=True,reference_subset_preserved=True))
    return result
