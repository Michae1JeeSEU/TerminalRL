"""One real t=0 v8 callback per selected baseline on the frozen medium inputs.

Validation output only: this script does not prepare, restore, branch, advance,
or run a formal campaign. The baseline directory is read only. Omit --method
for two sequential callbacks, or use separate --method invocations on two cores.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

# Set before importing numpy/scipy or the scheduler. Each CP solver already
# uses the frozen single-worker setting; no budget or solver parameter changes.
THREAD_VARIABLES=("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")
for _thread_variable in THREAD_VARIABLES: os.environ[_thread_variable]="1"
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from terminal_marl.core.rotating_slots.config import load_service_profile
from terminal_marl.core.rotating_slots.stage2_campaign import (
    METHODS, atomic_json, busy_fixture, campaign_config, canonical, digest,
    load_json, read_manifest,
)
from terminal_marl.core.rotating_slots.stage2_planning import (
    CandidateProvider, NormalSchedulerCallback, PLANNING_PROFILE_DOWNSTREAM,
    PlanningBudget, QUALITY_BUDGET_PROFILE, candidate_policy_id, objective_schema_id,
)


def file_hash(path):
    value=sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): value.update(block)
    return value.hexdigest()


def source_files():
    names=("stage2_planning.py","stage2_downstream.py","stage2_campaign.py","world.py",
           "rescue_rules.py","config.py","energy.py","safety.py","trajectory.py")
    return {"terminal_marl/core/rotating_slots/"+name:
        file_hash(ROOT/"terminal_marl/core/rotating_slots"/name) for name in names}


class ObservedProvider(CandidateProvider):
    """Keep returned catalog references; serialize only after callback timing."""
    def __init__(self):
        super().__init__(planning_profile=PLANNING_PROFILE_DOWNSTREAM)
        self.observed=[]

    def get_columns(self,snapshot,budget=None,context=None):
        catalog=super().get_columns(snapshot,budget,context)
        self.observed.append((snapshot,catalog))
        return catalog


def candidate_facts(snapshot,catalog):
    rows=[]
    counts=Counter()
    per_aircraft={}
    for aid,columns in catalog.columns.items():
        ring=snapshot.aircraft[aid].ring
        own=Counter()
        for column in columns:
            full=bool(column.plans and column.plans[-1].pad_id is not None)
            timing=("NO_MOVE" if not column.plans else "IMMEDIATE"
                if column.plans[0].start_time<=snapshot.time+1e-8 else "DELAYED")
            category="COMPLETE" if full else "PARTIAL" if column.plans else "NO_MOVE"
            counts[category]+=1; own[category]+=1
            # NO_MOVE is both a category and a timing label. Count its shared
            # key once; the combined category/timing key remains separate.
            if timing!=category: counts[timing]+=1; own[timing]+=1
            counts[category+"_"+timing]+=1; own[category+"_"+timing]+=1
            rows.append(dict(aircraft_id=aid,source_ring=ring,kind=column.kind,
                signature=column.signature,complete=full,timing=timing,
                plans=[plan.to_dict() for plan in column.plans],
                objective=column.objective,energy_j=column.energy_j,
                peak_seconds=column.peak_seconds,downstream=dict(column.downstream)))
        assert sum(own[key] for key in ("COMPLETE","PARTIAL","NO_MOVE"))==len(columns)
        per_aircraft[aid]=dict(source_ring=ring,counts=dict(own),columns_count=len(columns),
            has_complete=bool(own["COMPLETE"]),has_immediate=bool(own["IMMEDIATE"]),
            has_delayed=bool(own["DELAYED"]),catalog_incomplete=aid in catalog.incomplete)
    assert sum(counts[key] for key in ("COMPLETE","PARTIAL","NO_MOVE"))==len(rows)
    return dict(time=snapshot.time,counts=dict(counts),aircraft=per_aircraft,
        catalog_incomplete=list(catalog.incomplete),columns=rows,
        columns_count=sum(map(len,catalog.columns.values())))


def solver_facts(records):
    rows=[]
    for index,record in enumerate(records):
        model=record.get("model_integer_key")
        selected=record.get("canonical_selected_integer_key")
        phase=record.get("phase")
        layer=("F","L","risk","Ehat","tau")[phase] if isinstance(phase,int) and 0<=phase<5 else None
        rows.append(dict(record_index=index,phase=phase,objective_layer=layer,round=record.get("round"),status=record.get("status"),
            uniform_refinement_restart=record.get("uniform_refinement_restart"),
            model_integer_key=model,canonical_selected_integer_key=selected,
            keys_both_observed=model is not None and selected is not None,
            keys_equal=None if model is None or selected is None else tuple(model)==tuple(selected),
            raw_record=record))
    return rows


def validate_inputs(baseline):
    required=("campaign.json","provenance.json","fixture.json","manifest.json","config.yaml","environment.lock.json")
    files={name:baseline/name for name in required}
    missing=[str(path) for path in files.values() if not path.is_file()]
    if missing: raise ValueError("missing baseline input: "+", ".join(missing))
    campaign=load_json(files["campaign.json"])
    provenance=load_json(files["provenance.json"])
    for name,expected in (("service_profile","service_half_v1"),("scenario_profile","medium24_rate150_v1"),
                          ("rescue_rule_profile","rescue_need_v7")):
        if campaign.get(name)!=expected or provenance.get(name)!=expected:
            raise ValueError("baseline "+name+" is not "+expected)
    config=campaign_config(baseline,campaign)
    approved=load_service_profile("service_half_v1")
    if config.config_hash!=approved.config_hash or config.raw_hash!=approved.raw_hash:
        raise ValueError("baseline differs from the approved service_half_v1 bytes")
    manifest=read_manifest(files["manifest.json"])
    if manifest.seed!=71 or manifest.rate_per_hour!=150. or manifest.arrival_window!=3600.:
        raise ValueError("baseline manifest seed/rate/window differs from medium protocol")
    if digest(manifest.to_dict())!=campaign["identity"]["manifest_sha256"]:
        raise ValueError("baseline manifest identity mismatch")
    if file_hash(files["fixture.json"])!=campaign["fixture_sha256"]:
        raise ValueError("baseline fixture identity mismatch")
    if digest(load_json(files["environment.lock.json"]))!=campaign["environment_lock_sha256"]:
        raise ValueError("baseline environment lock identity mismatch")
    actual_source=source_files()
    numeric={name:value for name,value in actual_source.items() if name.endswith(("energy.py","safety.py","trajectory.py"))}
    if digest(numeric)!=provenance["numeric_backend_hash"]:
        raise ValueError("current numerical backend differs from baseline")
    return files,campaign,config,manifest,load_json(files["fixture.json"]),actual_source


def run_method(output,method,config,manifest,fixture,input_facts):
    directory=output/method
    directory.mkdir(parents=True,exist_ok=False)
    facts=dict(schema="stage2_v8_t0_callback_validation_v1",status="running",method=method,
        interpretation="ONE_FRESH_T0_CALLBACK_NOT_A_FORMAL_CAMPAIGN",
        inputs=input_facts,planning_profile=PLANNING_PROFILE_DOWNSTREAM,
        candidate_policy_id=candidate_policy_id(PLANNING_PROFILE_DOWNSTREAM),
        objective_id=objective_schema_id(PLANNING_PROFILE_DOWNSTREAM),
        expected_budget_profile=QUALITY_BUDGET_PROFILE,
        expected_budget=PlanningBudget.for_profile(QUALITY_BUDGET_PROFILE).settings(),
        numerical_threads={name:os.environ[name] for name in THREAD_VARIABLES},
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None)
    atomic_json(directory/"facts.json",facts)
    world=None
    try:
        world=busy_fixture(manifest,config=config,scenario_profile="medium24_rate150_v1",rescue_rule_profile="rescue_need_v7")
        actual=dict(aircraft=[a.to_dict() for a in world.aircraft.values()],slots=world.slots.to_dict(),pads=world.pads.to_dict())
        fixture_equal={key:canonical(actual[key])==canonical(fixture[key]) for key in actual}
        facts.update(initial_time_s=world.time,fixture_equal=fixture_equal,
            initial_fixture_digest=digest(actual),initial_metrics=world.metrics(),
            effective_config_hash=world.config.config_hash,effective_profile_hash=world.config.profile_hash)
        if not all(fixture_equal.values()): raise ValueError("reconstructed t0 fixture differs from frozen baseline")
        provider=ObservedProvider()
        callback=NormalSchedulerCallback(method,candidate_provider=provider,planning_profile=PLANNING_PROFILE_DOWNSTREAM)
        if callback.records or world.time!=0.: raise ValueError("callback must start fresh at t0")
        events_before=len(world.events)
        started=time.perf_counter()
        result=callback(world._scheduler_port)
        callback_wall=time.perf_counter()-started
        if world.time!=0. or len(callback.records)!=1: raise ValueError("expected exactly one real t0 callback and no advance")
        row=callback.records[0]
        atomic_json(directory/"callback.json",row)
        catalogs=[candidate_facts(snapshot,catalog) for snapshot,catalog in provider.observed]
        atomic_json(directory/"candidates.json",catalogs)
        atomic_json(directory/"scheduler_state.json",callback.state_dict())
        atomic_json(directory/"selected_prefix.json",[plan.to_dict() for plan in result.prefix])
        new_events=world.events[events_before:]
        atomic_json(directory/"callback_events.json",new_events)
        candidate_roundtrip=load_json(directory/"candidates.json")
        facts.update(callback_count=1,callback_wall_s=callback_wall,callback_time_s=world.time,
            actual_planning_profile=row.get("planning_profile"),actual_budget_profile=row.get("budget_profile"),
            actual_budget=row.get("planning_budget"),callback_record=row,
            candidate_catalog_passes=len(catalogs),candidate_coverage=[{k:v for k,v in entry.items() if k!="columns"} for entry in catalogs],
            candidate_json_roundtrip_equal=canonical(candidate_roundtrip)==canonical(catalogs),
            solver_stage_comparisons=solver_facts(row.get("solver",())),
            result_prefix_count=len(result.prefix),submitted_prefix_count=row.get("submitted"),
            commit_status=row.get("commit_status"),
            actual_committed_events=[event for event in new_events if event["type"] in ("plan_committed","soft_plan_reserved")],
            actual_active_count=len(world.active),actual_soft_count=len(world.soft),
            post_callback_metrics=world.metrics(),config_hash_unchanged=world.config.config_hash==config.config_hash,
            profile_hash_unchanged=world.config.profile_hash==config.profile_hash)
        if row.get("planning_profile")!=PLANNING_PROFILE_DOWNSTREAM: raise ValueError("callback planning profile differs")
        if row.get("budget_profile")!=QUALITY_BUDGET_PROFILE or row.get("planning_budget")!=facts["expected_budget"]:
            raise ValueError("callback budget differs from unchanged quality_v5")
        facts["status"]="completed"
    except Exception as error:
        facts.update(status="failed",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        atomic_json(directory/"facts.json",facts)
        if world is not None and world.state_sink is not None: world.state_sink.close()
    return dict(method=method,status=facts["status"],facts=str(directory/"facts.json"),callback_wall_s=facts["callback_wall_s"])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root",type=Path,required=True,help="read-only existing v7 medium 600s root")
    parser.add_argument("--output",type=Path,required=True,help="new validation output outside the baseline and project source")
    parser.add_argument("--method",choices=METHODS,help="one method only; default executes the pair sequentially")
    args=parser.parse_args(argv)
    baseline=args.baseline_root.expanduser().resolve(); output=args.output.expanduser().resolve()
    if output==baseline or output.is_relative_to(baseline) or baseline.is_relative_to(output):
        parser.error("baseline and output must be disjoint")
    if output.is_relative_to(ROOT): parser.error("server validation output must be outside the project source tree")
    files,campaign,config,manifest,fixture,current_source=validate_inputs(baseline)
    before={name:file_hash(path) for name,path in files.items()}
    versions={}
    for package in ("numpy","scipy","PyYAML","ortools"):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    if (args.method is None or args.method=="rh_cp_sat") and versions["ortools"]!="9.15.6755":
        parser.error("real CP callback requires the frozen ortools==9.15.6755 environment")
    input_facts=dict(baseline_root=str(baseline),baseline_files_sha256=before,
        baseline_identity=campaign["identity"],manifest_digest=digest(manifest.to_dict()),
        current_source_files_sha256=current_source,script_sha256=file_hash(__file__),
        installed_versions=versions,baseline_environment_lock_sha256=before["environment.lock.json"])
    output.mkdir(parents=True,exist_ok=True)
    summary=dict(schema="stage2_v8_t0_validation_summary_v1",status="running",inputs=input_facts,methods=[])
    summary_path=output/("summary_"+(args.method or "pair")+".json")
    try:
        for method in ((args.method,) if args.method else METHODS):
            summary["methods"].append(run_method(output,method,config,manifest,fixture,input_facts))
        summary["status"]="completed"
    except Exception as error:
        summary.update(status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        after={name:file_hash(path) for name,path in files.items()}
        summary.update(baseline_files_after_sha256=after,baseline_files_unchanged=before==after)
        atomic_json(summary_path,summary)
    if before!=after: raise RuntimeError("read-only baseline input bytes changed during validation")
    print(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))


if __name__=="__main__": main()
