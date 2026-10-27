"""One fixed t0 interface check per H/L arm; no fitting or scenario sweep."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import time

if __package__:
    from . import run_latency_interface as runner
else:
    import run_latency_interface as runner

DRIVER=Path(__file__).resolve()
END=20.001
JOB_CAP,TOTAL_CAP=90.,180.


class SingleTask:
    """Forward all executor services; suppress only starts after the first task."""
    def __init__(self,inner):
        self.inner=inner
        self.started=False
        self.suppressed_starts=0

    def __getattr__(self,name):
        return getattr(self.inner,name)

    def __call__(self,*args,**kwargs):
        if self.started and self.inner.pending is None:
            self.suppressed_starts+=1
            return
        self.started=True
        return self.inner(*args,**kwargs)


def protocol(d):
    value=runner.protocol(d.dependency_protocol,20.,4.,d.handoff_version)
    value.update(schema="search_handoff_fixed_state_v1",target_sim_seconds=END,
        job_wall_cap_s=JOB_CAP,total_evaluation_wall_cap_s=TOTAL_CAP,
        normal_callbacks_per_arm=1,physical_states=1,
        observation_boundaries_s=[5.,10.,15.,END],
        endpoint_policy="original world advances through the 20 s effective event to 20.001; final endpoint is settle-only; no second normal task",
        interpretation="Specified t0 interface check; not an additional short-scenario quality experiment")
    value.pop("checkpoint_seconds");value.pop("boundary_seconds")
    return value


def prepare(args,d):
    output,parent,initial=(Path(p).resolve() for p in (args.output,args.model_parent,args.initial_parent))
    runner.require(not output.exists(),"fresh fixed-state output required; no retry")
    runner.require(not any(output.is_relative_to(p) or p.is_relative_to(output) for p in (parent,initial)),"output overlaps input archive")
    frozen=runner.load(args.source_freeze)
    runner.verify_source(frozen,d)
    runner.require(runner.file_hash(DRIVER)==frozen["source_files"]["scripts/stage3/probe_search_handoff.py"],"probe differs from source freeze")
    binding=runner.verify_model(parent)
    original=initial/"roots"/runner.ROOT_ID
    row,manifest,state,checkpoint=(runner.load(original/name) for name in
        ("root.json","manifest.json","initial_state.json","initial.checkpoint.json"))
    runner.require((row["root_id"],row["seed"],row["scenario_profile"])==(runner.ROOT_ID,runner.SEED,"medium24_rate150_v1"),"wrong fixed root")
    runner.require(manifest["seed"]==runner.SEED and manifest["rate_per_hour"]==150 and manifest["arrival_window"]==3600.
        and len(manifest["requests"])==141,"manifest changed")
    runner.require(d.digest(manifest)==row["manifest_hash"] and d.digest(state)==row["initial_state_hash"],"initial digest mismatch")
    runner.require(checkpoint["sha256"]==d.digest(checkpoint["payload"]) and checkpoint["payload"]["identity"]==row["identity"]
        and checkpoint["payload"]["time"]==0. and checkpoint["payload"]["sink"] is None,"invalid original t0 checkpoint")
    runner.require(sum(a["status"].startswith("HOLD_") for a in state["aircraft"])==24
        and sum(a["status"]=="LANDED_SERVICE" for a in state["aircraft"])==3,"initial aircraft counts changed")
    output.mkdir(parents=True)
    runner.exclusive(output/"prepare.started.json",dict(started_at=d.campaign.utc_now()))
    try:
        (output/"model").mkdir()
        root=output/"roots"/runner.ROOT_ID
        root.mkdir(parents=True)
        for name in ("model.pt","feature_normalizer.json","model_card.json"):
            shutil.copyfile(parent/"model"/name,output/"model"/name)
        shutil.copyfile(parent/"run_binding.generated.json",output/"run_binding.generated.json")
        shutil.copyfile(args.source_freeze,output/"source_freeze.json")
        for name in ("manifest.json","initial_state.json","initial.checkpoint.json"):
            shutil.copyfile(original/name,root/name)
        shutil.copyfile(original/"root.json",root/"original_root.json")
        relocated=deepcopy(row)
        relocated.update(checkpoint_path=str(root/"initial.checkpoint.json"),raw_snapshot_path=str(root/"initial_state.json"))
        d.atomic_json(root/"root.json",relocated)
        scope=dict(protocol=protocol(d),source_commit=frozen["source_commit"],source_hash=frozen["source_hash"],checkout=str(d.checkout),
            model_fingerprint=deepcopy(binding["fingerprint"]),shared_manifest_hash=row["manifest_hash"],initial_state_hash=row["initial_state_hash"],
            initial_identity=deepcopy(row["identity"]),
            model_parent_readonly=str(parent),initial_parent_readonly=str(initial),driver_sha256=runner.file_hash(DRIVER))
        scope["scope_id"]=d.digest(scope)
        scope["prepared_file_sha256"]={p.relative_to(output).as_posix():runner.file_hash(p)
            for p in output.rglob("*") if p.is_file() and p.name!="prepare.started.json"}
        d.atomic_json(output/"execution_scope.json",scope)
        ledger=d.campaign.CampaignLedger(output/"budget_ledger.json")
        ledger.data.update(total_s=TOTAL_CAP,compute_stop_s=TOTAL_CAP,phase_caps={"evaluation":TOTAL_CAP})
        ledger.persist();ledger.pause()
        d.atomic_json(output/"prepare.finished.json",dict(status="COMPLETED",scope_id=scope["scope_id"]))
        return dict(status="PREPARED",scope_id=scope["scope_id"])
    except Exception as exc:
        d.atomic_json(output/"prepare.finished.json",dict(status="FAILED",error_type=type(exc).__name__,error=str(exc)))
        raise


def verify(output,d):
    scope=runner.load(output/"execution_scope.json")
    prepared=runner.load(output/"prepare.finished.json")
    runner.require(prepared["status"]=="COMPLETED" and prepared["scope_id"]==scope["scope_id"],"probe preparation changed")
    runner.require(scope["protocol"]==protocol(d) and Path(scope["checkout"]).resolve()==d.checkout,"probe scope changed")
    runner.require(scope["scope_id"]==d.digest({k:v for k,v in scope.items() if k not in ("scope_id","prepared_file_sha256")}),"scope digest mismatch")
    runner.verify_source(runner.load(output/"source_freeze.json"),d)
    runner.require(runner.file_hash(DRIVER)==scope["driver_sha256"],"probe driver changed")
    for name,expected in scope["prepared_file_sha256"].items():
        runner.require(runner.file_hash(output/name)==expected,"prepared input changed: "+name)
    binding=runner.verify_model(output)
    runner.require(binding["fingerprint"]==scope["model_fingerprint"],"model fingerprint changed")
    ledger=d.campaign.CampaignLedger(output/"budget_ledger.json")
    runner.require(ledger.data["total_s"]==ledger.data["compute_stop_s"]==TOTAL_CAP
        and ledger.data["phase_caps"]=={"evaluation":TOTAL_CAP},"probe cap changed")
    return scope,binding,ledger


def task_facts(tasks,callbacks):
    rows=[]
    by_time={r.get("planning_timing",{}).get("snapshot_time"):r for r in callbacks}
    for task in tasks:
        handoff=task.get("search_handoff",by_time.get(task["snapshot_time"],{}).get("search_handoff"))
        row={key:task.get(key) for key in ("snapshot_time","result_time","preparation_started_time","ready_time",
            "effective_time","finished_time","status","reason","submitted")}
        row["search_handoff"]=handoff
        row["first_nonempty_certified_observed"]=None if handoff is None else handoff.get("first_nonempty_certified") is not None
        row["preparation_started_observed"]=task.get("preparation_started_time") is not None
        row["actual_join_observed"]=task.get("status")=="COMMITTED" and (task.get("submitted") or 0)>0
        row["interface_check_status"]=("UNAVAILABLE_TRACE" if handoff is None else
            "NO_NONEMPTY_CERTIFICATE" if not row["first_nonempty_certified_observed"] else
            "PREPARATION_STARTED" if row["preparation_started_observed"] else "PREPARATION_NOT_OBSERVED")
        rows.append(row)
    return rows


def job(output,arm,d):
    scope,binding,ledger=verify(output,d)
    runner.require(arm in runner.ARMS and (output/"run.started.json").is_file(),"fixed parent run required")
    receipt=ledger.data["jobs"].get(arm,{})
    runner.require(receipt.get("status")=="running" and receipt.get("timeout_s")==JOB_CAP
        and (ledger.data.get("active_phase") or {}).get("phase")=="evaluation","parent 90 s watchdog receipt required")
    if hasattr(os,"sched_setaffinity"):os.sched_setaffinity(0,{runner.CPU})
    directory=output/"evaluation"/arm
    directory.mkdir(parents=True,exist_ok=False)
    d.atomic_json(directory/"job.started.json",dict(pid=os.getpid(),arm=arm,started_at=d.campaign.utc_now()))
    started=time.perf_counter();world=scheduler=gate=None;error=None;finalization_errors=[];setup={}
    row=runner.load(output/"roots"/runner.ROOT_ID/"root.json")
    def logs():
        if gate is not None:
            d.atomic_json(directory/"normal_tasks.json",gate.records)
            d.atomic_json(directory/"online_services.json",gate.online_services)
            d.atomic_json(directory/"scheduler_records.json",scheduler.records)
    try:
        world=d.restore_root(row,d.NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8"))
        world.configure_practical_runtime(True)
        scheduler=d.evaluation._scheduler(output,arm,binding,d.OnlineObserver(),runner.ROOT_ID,setup)
        scheduler.provider.advisor_runtime.ranking_scope=runner.RANKING
        inner=d.LatencyAwareScheduler(scheduler,lead_seconds=20.,preparation_margin=1.,search_handoff_reserve_seconds=4.)
        gate=SingleTask(inner)
        world.decision_callback=gate;world.admission_callback=d.shared_fifo_admission
        runner.require(world.time==0. and world.scheduler_period==5.,"original t0 or heartbeat changed")
        for boundary in (5.,10.,15.,END):
            world.advance(boundary,gate,settle_only_at_end=True)
            logs()
            runner.require(abs(world.time-boundary)<1e-7,"world did not reach the observation boundary")
            runner.require(len(scheduler.records)<=1 and len(gate.records)<=1,"probe started more than one normal callback")
            runner.require(not world.failed and not getattr(world,"terminated_invalid",False),"world failed during interface observation")
        runner.require(len(scheduler.records)==1 and len(gate.records)==1,"t0 callback was not observed exactly once")
    except Exception as exc:
        error=dict(error_type=type(exc).__name__,error=str(exc))
    finally:
        terminal=dict(status="UNAVAILABLE_SETUP_FAILURE");world_record=None
        try:
            logs()
            if world is not None:
                terminal=d.evaluation.terminal_observation(world)
                d.atomic_json(directory/"terminal_observation.json",terminal)
                world_record=world.to_dict()
                d.atomic_json(directory/"world_record.json",world_record)
        except Exception as exc:
            finalization_errors.append(dict(error_type=type(exc).__name__,error=str(exc)))
        facts=task_facts(gate.records,scheduler.records) if gate is not None else []
        result=dict(status="COMPLETE_OBSERVATION" if error is None and not finalization_errors else "PARTIAL_OBSERVATION",
            arm=arm,scope=scope,model_fingerprint=binding["fingerprint"],terminal_observation=terminal,task_facts=facts,
            callback_count=len(scheduler.records) if scheduler is not None else 0,suppressed_later_starts=gate.suppressed_starts if gate else 0,
            world_record_path=str(directory/"world_record.json") if world_record is not None else None,
            observed_maneuver_start_events=[event for event in (world_record or {}).get("events",[]) if event.get("type")=="maneuver_started"],
            wall_s=time.perf_counter()-started,error=error,finalization_errors=finalization_errors,
            interpretation="Specified fixed-state interface evidence; certificate, preparation and actual joining are separate observations")
        d.atomic_json(directory/"result.json",result)
        d.atomic_json(directory/"job.finished.json",dict(status=result["status"],error=error,wall_s=result["wall_s"]))
    if error or finalization_errors:raise RuntimeError("fixed-state observation failed; partial evidence preserved: "+str(error or finalization_errors))
    return result


def run(output,d):
    with d.campaign.campaign_lock(output):
        scope,_,ledger=verify(output,d)
        runner.require(not (output/"run.started.json").exists() and not ledger.data["jobs"],"probe already attempted; no retry")
        resource=d.campaign.resource_probe()
        runner.require(runner.CPU in resource.get("free_physical_cores",[]),"fixed CPU12 unavailable")
        runner.exclusive(output/"run.started.json",dict(scope_id=scope["scope_id"],resource_probe=resource,started_at=d.campaign.utc_now()))
        receipts=[];error=None
        try:
            with ledger.phase("evaluation"):
                for arm in runner.ARMS:
                    runner.require(ledger.remaining("evaluation")>=JOB_CAP+1.,"full equal 90 s child cap unavailable")
                    command=[sys.executable,str(DRIVER),"job","--output",str(output),"--checkout",str(d.checkout),"--arm",arm]
                    receipt=d.campaign.run_subprocess(ledger,"evaluation",arm,command,output/"logs",cap=JOB_CAP)
                    receipts.append(receipt);d.atomic_json(output/"receipts"/(arm+".json"),receipt)
                    if receipt["status"]=="BUDGET_LIMITED":
                        d.atomic_json(output/"evaluation"/arm/"watchdog_partial.json",dict(status="BUDGET_LIMITED",receipt=receipt,
                            interpretation="Hard timeout; existing task/service logs are partial, no retry"))
                    runner.require(receipt["status"]=="completed" and receipt.get("exit_code")==0,"probe child did not complete; return to primary")
                    runner.require(runner.load(output/"evaluation"/arm/"result.json")["status"]=="COMPLETE_OBSERVATION","probe child output is partial")
        except Exception as exc:error=dict(error_type=type(exc).__name__,error=str(exc))
        finally:
            ledger.pause()
            result=dict(status="COMPLETED" if error is None and len(receipts)==2 else "INCOMPLETE_RETURN_TO_PRIMARY",receipts=receipts,error=error)
            d.atomic_json(output/"run.finished.json",result)
        if error:raise RuntimeError(error["error"])
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","run","job"))
    parser.add_argument("--output",type=Path,default=Path("/home/user/training_outputs/terminal/stage3_search_handoff_20261027/fixed_state"))
    parser.add_argument("--checkout",type=Path,required=True)
    parser.add_argument("--source-freeze",type=Path)
    parser.add_argument("--model-parent",type=Path,default=Path("/home/user/training_outputs/terminal/stage3_online_v2_20261025"))
    parser.add_argument("--initial-parent",type=Path,default=Path("/home/user/training_outputs/terminal/stage3_learning_v1_0_20261023"))
    parser.add_argument("--arm",choices=runner.ARMS)
    args=parser.parse_args(argv)
    if args.stage=="prepare" and args.source_freeze is None:parser.error("prepare requires --source-freeze")
    if args.stage=="job" and args.arm is None:parser.error("job requires --arm")
    d=runner.dependencies(args.checkout);output=args.output.resolve()
    result=prepare(args,d) if args.stage=="prepare" else run(output,d) if args.stage=="run" else job(output,args.arm,d)
    print(json.dumps(dict(status=result["status"],output=str(output)),sort_keys=True))


if __name__=="__main__":main()
