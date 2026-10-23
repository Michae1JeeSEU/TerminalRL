"""Fresh medium validation through 12s, including the real fractional arrival.

This is a bounded error-recovery check on the existing medium seed/fixture,
not a formal campaign. No checkpoint or campaign budget is created or resumed.
The baseline root is read only; every method output directory is exclusive.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

THREAD_VARIABLES=("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")
for _thread_variable in THREAD_VARIABLES: os.environ[_thread_variable]="1"
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))

from v8_runtime_validation import validate_inputs, file_hash, source_files
from terminal_marl.core.rotating_slots.replay import BufferedStateJSONLSink
from terminal_marl.core.rotating_slots.stage2_campaign import (
    METHODS, atomic_json, busy_fixture, canonical, digest,
    shared_fifo_admission, pad_opportunity_samples, annotate_pad_opportunities,
)
from terminal_marl.core.rotating_slots.stage2_planning import (
    NormalSchedulerCallback, PLANNING_PROFILE_DOWNSTREAM, PlanningBudget,
    QUALITY_BUDGET_PROFILE, candidate_policy_id, objective_schema_id,
)

TARGET=12.


def write_jsonl(path,rows):
    with Path(path).open("wb") as stream:
        for row in rows: stream.write(canonical(row)+b"\n")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root",type=Path,required=True,help="read-only existing v7 medium 600s root")
    parser.add_argument("--output",type=Path,required=True,help="validation output outside baseline and project source")
    parser.add_argument("--method",choices=METHODS,required=True,help="one method per invocation, with no automatic retry")
    args=parser.parse_args(argv)
    baseline=args.baseline_root.expanduser().resolve(); output=args.output.expanduser().resolve()
    if output==baseline or output.is_relative_to(baseline) or baseline.is_relative_to(output):
        parser.error("baseline and output must be disjoint")
    if output.is_relative_to(ROOT): parser.error("server validation output must be outside the project source tree")
    files,campaign,config,manifest,fixture,current_source=validate_inputs(baseline)
    if args.method=="rh_cp_sat":
        try: solver_version=importlib.metadata.version("ortools")
        except importlib.metadata.PackageNotFoundError: solver_version=None
        if solver_version!="9.15.6755": parser.error("real CP validation requires ortools==9.15.6755")
    before={name:file_hash(path) for name,path in files.items()}
    directory=output/args.method
    directory.mkdir(parents=True,exist_ok=False)
    expected_budget=PlanningBudget.for_profile(QUALITY_BUDGET_PROFILE).settings()
    facts=dict(schema="stage2_v8_fractional_12s_validation_v1",status="running",method=args.method,
        interpretation="BOUNDED_12S_ERROR_RECOVERY_VALIDATION_NOT_FORMAL_CAMPAIGN",
        target_s=TARGET,baseline_root=str(baseline),baseline_identity=campaign["identity"],
        baseline_files_before_sha256=before,current_source_files_before_sha256=current_source,
        script_sha256=file_hash(__file__),manifest_digest=digest(manifest.to_dict()),
        planning_profile=PLANNING_PROFILE_DOWNSTREAM,candidate_policy_id=candidate_policy_id(PLANNING_PROFILE_DOWNSTREAM),
        objective_id=objective_schema_id(PLANNING_PROFILE_DOWNSTREAM),budget_profile=QUALITY_BUDGET_PROFILE,
        expected_budget=expected_budget,numerical_threads={name:os.environ[name] for name in THREAD_VARIABLES},
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
        callback_calls=[],manual_initial_callback=False,automatic_retries=0)
    atomic_json(directory/"facts.json",facts)
    world=None; sink=None; scheduler=None
    started=time.perf_counter()
    try:
        sink=BufferedStateJSONLSink(directory/"states.jsonl")
        world=busy_fixture(manifest,state_sink=sink,config=config,
            scenario_profile="medium24_rate150_v1",rescue_rule_profile="rescue_need_v7")
        actual=dict(aircraft=[a.to_dict() for a in world.aircraft.values()],slots=world.slots.to_dict(),pads=world.pads.to_dict())
        facts["fixture_equal"]={key:canonical(actual[key])==canonical(fixture[key]) for key in actual}
        facts["initial_fixture_digest"]=digest(actual)
        facts["initial_time_s"]=world.time
        if world.time!=0. or not all(facts["fixture_equal"].values()):
            raise ValueError("fresh t0 medium world differs from frozen baseline fixture")
        world.admission_callback=shared_fifo_admission
        scheduler=NormalSchedulerCallback(args.method,planning_profile=PLANNING_PROFILE_DOWNSTREAM)

        def callback(snapshot,port):
            # Same pad observation/annotation wrapper as run_episode_block.
            pads=pad_opportunity_samples(world)
            protected={plan.pad_id for plan in (*world.active.values(),*world.soft.values()) if plan.kind=="FINAL_EMERGENCY"}
            call=dict(time=world.time,status="running")
            facts["callback_calls"].append(call)
            count_before=len(scheduler.records)
            call_started=time.perf_counter()
            try:
                result=scheduler(port)
                if scheduler.records:
                    row=scheduler.records[-1]
                    annotate_pad_opportunities(pads,row,protected)
                call.update(status="completed",records_added=len(scheduler.records)-count_before,
                    submitted=scheduler.records[-1].get("submitted") if len(scheduler.records)>count_before else None,
                    commit_status=scheduler.records[-1].get("commit_status") if len(scheduler.records)>count_before else None)
                return result
            except Exception as error:
                call.update(status="failed",error_type=type(error).__name__,error=str(error))
                raise
            finally:
                call["wall_s"]=time.perf_counter()-call_started
                atomic_json(directory/"facts.json",facts)

        # The executor performs its own t0 dispatch and actual fractional
        # arrival dispatch. There is deliberately no manual initial callback.
        world.advance(TARGET,callback,settle_only_at_end=True)
        facts.update(final_time_s=world.time,failed=world.failed,
            terminated_invalid=getattr(world,"terminated_invalid",False),
            metrics=world.metrics(),callback_count=len(scheduler.records),
            fractional_callback_times=[row["time"] for row in scheduler.records if abs(row["time"]-round(row["time"]))>1e-8],
            actual_budgets=[row.get("planning_budget") for row in scheduler.records],
            actual_planning_profiles=[row.get("planning_profile") for row in scheduler.records],
            config_hash_unchanged=world.config.config_hash==config.config_hash,
            profile_hash_unchanged=world.config.profile_hash==config.profile_hash)
        if abs(world.time-TARGET)>1e-8 or world.failed or getattr(world,"terminated_invalid",False):
            raise ValueError("12s validation did not reach a valid complete target")
        if not all(row.get("planning_budget")==expected_budget and row.get("budget_profile")==QUALITY_BUDGET_PROFILE
                   and row.get("planning_profile")==PLANNING_PROFILE_DOWNSTREAM for row in scheduler.records):
            raise ValueError("callback changed frozen planning profile or quality_v5 budget")
        releases=[request.release_time for request in manifest.requests if 0<request.release_time<TARGET]
        facts["manifest_releases_before_12s"]=releases
        facts["fractional_release_callbacks_observed"]={str(t):any(abs(row["time"]-t)<1e-8 for row in scheduler.records) for t in releases}
        if not releases or not all(facts["fractional_release_callbacks_observed"].values()):
            raise ValueError("validation lacks the actual medium arrival callback before 12s")
        facts["status"]="completed"
    except Exception as error:
        facts.update(status="failed",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        if world is not None:
            facts.update(final_time_s=world.time,failed=world.failed,terminated_invalid=getattr(world,"terminated_invalid",False))
            write_jsonl(directory/"events.jsonl",world.events)
            write_jsonl(directory/"dispatch_timings.jsonl",world.dispatch_timings)
            atomic_json(directory/"metrics.json",world.metrics())
        if scheduler is not None: write_jsonl(directory/"decisions.jsonl",scheduler.records)
        if sink is not None: sink.close()
        after={name:file_hash(path) for name,path in files.items()}
        sources_after=source_files()
        facts.update(wall_s=time.perf_counter()-started,baseline_files_after_sha256=after,
            baseline_files_unchanged=before==after,current_source_files_after_sha256=sources_after,
            current_source_files_unchanged=current_source==sources_after)
        atomic_json(directory/"facts.json",facts)
    if before!=after: raise RuntimeError("read-only baseline inputs changed during validation")
    if current_source!=sources_after: raise RuntimeError("project source files changed during validation")
    print(json.dumps(dict(status=facts["status"],method=args.method,final_time_s=facts["final_time_s"],
        callback_count=facts["callback_count"],fractional_callback_times=facts["fractional_callback_times"],
        facts=str(directory/"facts.json")),indent=2,ensure_ascii=False,allow_nan=False))


if __name__=="__main__": main()
