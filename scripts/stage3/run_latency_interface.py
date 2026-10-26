"""One preregistered medium H/L pair, 120 simulated seconds, without fitting."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[2]
DRIVER = Path(__file__).resolve()
ARMS = ("heuristic_reduced", "learned_reduced")
ROOT_ID, SEED, CPU = "s3r23", 7324, 12
HORIZON, STEP, CHECKPOINT_STEP, JOB_CAP, TOTAL_CAP = 120, 5, 60, 900., 1800.
INTERFACE, RANKING = "planning_effective_state_v1", "comparable_group_v1"
MODEL_HASH = "dba25b3393a5580bdc3971d9adf7d6e3cff6360404fdb08bc8fbb6f0b36e3b48"
NORMALIZER_HASH = "1e06b066dad2d657acbe16de6706b180dd4f4ed8ba1bc106cefcc2a3d4f4e7be"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
EXTRA_SOURCES = ("scripts/stage3/run_latency_interface.py", "tests/test_rotating_slots_latency_campaign.py",
                 "tests/test_rotating_slots_planning_latency.py", "tests/test_rotating_slots_latency_ranking.py")


def require(condition, message):
    if not condition: raise ValueError(message)


def load(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_hash(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def exclusive(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)


def dependencies(checkout):
    """No project import occurs until the explicit execution checkout is selected."""
    checkout = Path(checkout).resolve()
    require(checkout.is_dir(), "execution checkout is missing")
    os.chdir(checkout); sys.path.insert(0, str(checkout))
    os.environ.update({k:"1" for k in THREAD_KEYS})
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPATH"] = str(checkout)+(os.pathsep+os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    from terminal_marl.core.rotating_slots.stage3_learning import campaign, evaluation
    from terminal_marl.core.rotating_slots.stage3_learning.data import restore_root
    from terminal_marl.core.rotating_slots.stage3_learning.online_observer import OnlineObserver
    from terminal_marl.core.rotating_slots.stage2_campaign import atomic_json, digest, shared_fifo_admission, episode_metrics, save_checkpoint
    from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
    from terminal_marl.core.rotating_slots.planning_latency import LatencyAwareScheduler
    from terminal_marl.core.rotating_slots.replay import ReplayBundle
    require(campaign.ROOT.resolve()==checkout and Path(evaluation.__file__).resolve().is_relative_to(checkout),
            "project imports are outside the selected checkout")
    # This dependency uses stdlib at import and aggregates saved scalar fields.
    path = checkout/"scripts/stage3/summarize_online_v2.py"
    spec = importlib.util.spec_from_file_location("_latency_saved_aggregate", path)
    aggregate = importlib.util.module_from_spec(spec); spec.loader.exec_module(aggregate)
    return SimpleNamespace(checkout=checkout,campaign=campaign,evaluation=evaluation,restore_root=restore_root,
        OnlineObserver=OnlineObserver,atomic_json=atomic_json,digest=digest,shared_fifo_admission=shared_fifo_admission,
        episode_metrics=episode_metrics,save_checkpoint=save_checkpoint,NormalSchedulerCallback=NormalSchedulerCallback,
        LatencyAwareScheduler=LatencyAwareScheduler,ReplayBundle=ReplayBundle,aggregate=aggregate.aggregate_decisions)


def protocol():
    return dict(schema="normal_latency_120_v1",root_id=ROOT_ID,seed=SEED,scenario_profile="medium24_rate150_v1",
        target_sim_seconds=HORIZON,boundary_seconds=STEP,checkpoint_seconds=CHECKPOINT_STEP,
        cpu=CPU,arms=list(ARMS),formal_jobs=2,ordering="H_THEN_L_SERIAL",job_wall_cap_s=JOB_CAP,
        total_evaluation_wall_cap_s=TOTAL_CAP,lead_seconds=30.,preparation_margin=1.,scheduler_period=5.,
        interface=INTERFACE,ranking_scope=RANKING,emergency_timing_mode="existing_priority_response_cpu_charged",
        candidate_active_seconds=6.,cp_seconds=8.,hint_seconds=3.,soft_seconds=30.,callback_watchdog_seconds=60.,
        attempts_per_aircraft=32,total_attempts=1024,model_sha256=MODEL_HASH,normalizer_sha256=NORMALIZER_HASH,
        endpoint_policy="settle_only_at_end; dispatch/delivery exactly at 120 is censored and remains pending",
        training="NONE",automatic_retries=False)


def verify_model(parent):
    require(file_hash(parent/"model/model.pt")==MODEL_HASH, "model hash mismatch")
    require(file_hash(parent/"model/feature_normalizer.json")==NORMALIZER_HASH, "normalizer hash mismatch")
    binding = load(parent/"run_binding.generated.json"); card = load(parent/"model/model_card.json")
    require(binding.get("online_v2") and card["model_frozen"] and card["model_hash"]==MODEL_HASH,
            "expected frozen online-v2 model")
    require(card["fingerprint"]==binding["fingerprint"] and card["parameter_count"]==119555
            and card["seed"]==20261023, "model metadata/fingerprint mismatch")
    require(load(parent/"model/feature_normalizer.json")["schema_id"]==binding["fingerprint"]["feature_schema_id"],
            "normalizer feature schema mismatch")
    return binding


def verify_source(frozen, d):
    require(bool(frozen.get("source_commit")), "primary-provided execution commit is required")
    required = {p.relative_to(d.checkout).as_posix() for p in d.campaign.source_files()} | set(EXTRA_SOURCES)
    require(required <= set(frozen["source_files"]), "execution freeze omits required source/test dependencies")
    d.campaign.verify_sources(frozen)
    require(file_hash(DRIVER)==frozen["source_files"][EXTRA_SOURCES[0]], "invoked driver differs from execution freeze")


def prepare(args, d):
    output, parent, initial = (Path(p).resolve() for p in (args.output,args.model_parent,args.initial_parent))
    require(not output.exists(), "fresh output required; no retry/reset")
    require(not any(output.is_relative_to(p) or p.is_relative_to(output) for p in (parent,initial)), "output overlaps an input archive")
    frozen = load(args.source_freeze); verify_source(frozen,d); binding = verify_model(parent)
    original = initial/"roots"/ROOT_ID
    row, manifest, state, checkpoint = (load(original/name) for name in
        ("root.json","manifest.json","initial_state.json","initial.checkpoint.json"))
    require((row["root_id"],row["seed"],row["scenario_profile"])==(ROOT_ID,SEED,"medium24_rate150_v1"), "wrong fixed physical root")
    require(manifest["seed"]==SEED and manifest["rate_per_hour"]==150 and manifest["arrival_window"]==3600.
            and len(manifest["requests"])==141, "manifest changed")
    require(d.digest(manifest)==row["manifest_hash"] and d.digest(state)==row["initial_state_hash"], "initial input digest mismatch")
    require(sum(a["status"].startswith("HOLD_") for a in state["aircraft"])==24
            and sum(a["status"]=="LANDED_SERVICE" for a in state["aircraft"])==3,"original initial aircraft counts changed")
    require(checkpoint["sha256"]==d.digest(checkpoint["payload"]) and checkpoint["payload"]["identity"]==row["identity"]
            and checkpoint["payload"]["time"]==0. and checkpoint["payload"]["sink"] is None, "invalid original t0 checkpoint")
    output.mkdir(parents=True)
    exclusive(output/"prepare.started.json",dict(started_at=d.campaign.utc_now()))
    try:
        (output/"model").mkdir(); directory=output/"roots"/ROOT_ID; directory.mkdir(parents=True)
        for name in ("model.pt","feature_normalizer.json","model_card.json"):
            shutil.copyfile(parent/"model"/name,output/"model"/name)
        shutil.copyfile(parent/"run_binding.generated.json",output/"run_binding.generated.json")
        require(file_hash(output/"run_binding.generated.json")==file_hash(parent/"run_binding.generated.json"),"binding copy changed")
        shutil.copyfile(parent/"source_freeze.json",output/"training_source_freeze.json")
        shutil.copyfile(args.source_freeze,output/"source_freeze.json")
        for name in ("manifest.json","initial_state.json","initial.checkpoint.json"):
            shutil.copyfile(original/name,directory/name)
        shutil.copyfile(original/"root.json",directory/"original_root.json")
        relocated=deepcopy(row); relocated.update(checkpoint_path=str(directory/"initial.checkpoint.json"),raw_snapshot_path=str(directory/"initial_state.json"))
        d.atomic_json(directory/"root.json",relocated)
        scope=dict(protocol=protocol(),source_commit=frozen["source_commit"],source_hash=frozen["source_hash"],
            checkout=str(d.checkout),driver_sha256=file_hash(DRIVER),model_fingerprint=deepcopy(binding["fingerprint"]),
            training_source_hash=binding["source_hash"],training_source_commit=binding["fingerprint"]["source_commit"],
            shared_manifest_hash=row["manifest_hash"],initial_state_hash=row["initial_state_hash"],
            model_parent_readonly=str(parent),initial_parent_readonly=str(initial),
            interpretation="New normal-planning execution scope; original model binding/fingerprint is unchanged")
        scope["scope_id"]=d.digest(scope)
        scope["prepared_file_sha256"]={p.relative_to(output).as_posix():file_hash(p) for base in (output/"model",directory) for p in base.iterdir()}
        for name in ("run_binding.generated.json","source_freeze.json","training_source_freeze.json"):
            scope["prepared_file_sha256"][name]=file_hash(output/name)
        d.atomic_json(output/"execution_scope.json",scope)
        ledger=d.campaign.CampaignLedger(output/"budget_ledger.json")
        ledger.data.update(total_s=TOTAL_CAP,compute_stop_s=TOTAL_CAP,phase_caps={"evaluation":TOTAL_CAP})
        ledger.persist(); ledger.pause()
        d.atomic_json(output/"prepare.finished.json",dict(status="COMPLETED",scope_id=scope["scope_id"]))
        return dict(status="PREPARED",scope_id=scope["scope_id"])
    except Exception as exc:
        d.atomic_json(output/"prepare.finished.json",dict(status="FAILED",error_type=type(exc).__name__,error=str(exc))); raise


def verify(output, d):
    require(load(output/"prepare.finished.json")["status"]=="COMPLETED", "prepare did not complete")
    scope=load(output/"execution_scope.json"); verify_source(load(output/"source_freeze.json"),d)
    require(scope["protocol"]==protocol() and Path(scope["checkout"]).resolve()==d.checkout, "execution scope changed")
    require(scope["scope_id"]==d.digest({k:v for k,v in scope.items() if k not in ("scope_id","prepared_file_sha256")}), "scope digest mismatch")
    for name,expected in scope["prepared_file_sha256"].items(): require(file_hash(output/name)==expected,"prepared input changed: "+name)
    binding=verify_model(output)
    require(binding["fingerprint"]==scope["model_fingerprint"], "original fingerprint changed")
    require((output/"budget_ledger.json").is_file(), "existing fresh-run ledger required")
    ledger=d.campaign.CampaignLedger(output/"budget_ledger.json")
    require(ledger.data["total_s"]==TOTAL_CAP and ledger.data["compute_stop_s"]==TOTAL_CAP
            and ledger.data["phase_caps"]=={"evaluation":TOTAL_CAP}, "evaluation limits changed")
    return scope,binding,ledger


def measured(values):
    values=sorted(float(x) for x in values if isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x))
    return dict(count=len(values),total_s=math.fsum(values),p95_s=values[max(0,math.ceil(.95*len(values))-1)] if values else None,
                max_s=max(values) if values else None,quantile_rule="NEAREST_RANK")


def lifecycle_summary(tasks, callbacks, services):
    by_time={r["planning_timing"]["snapshot_time"]:r for r in callbacks if r.get("planning_timing")}
    proposed=[r for r in tasks if by_time.get(r["snapshot_time"],{}).get("deferred") is True
              and by_time[r["snapshot_time"]].get("noop") is False]
    finished=[r for r in proposed if "finished_time" in r]
    committed=[r for r in finished if r.get("status")=="COMMITTED" and r.get("submitted",0)>0]
    reasons=Counter(r.get("reason","MISSING") for r in tasks if r.get("status")=="REJECTED")
    base_reasons=Counter()
    for reason,count in reasons.items(): base_reasons[reason.split(":",1)[0]]+=count
    stale_reasons={"TRAFFIC_COMMITMENT_CHANGED","DEPENDENCY_CHANGED","CONTINUATION_STATE_CHANGED","NEW_TRAFFIC",
        "RESOURCE_DEPENDENCY_CHANGED","RELEASE_DEPENDENCY_CHANGED","RELEASE_TIME_CHANGED","EMERGENCY_POLICY_CHANGED",
        "EMERGENCY_PRIORITY_CHANGED","LATENCY_PACKET_EXPIRED"}
    energy_reasons={"ENERGY_BASIS_CHANGED","ENERGY_BOUND_CHANGED","SOC_INSUFFICIENT","LANDING_RESERVE",
        "PEAK_BUDGET","PREPARED_ENERGY_BOUND","PROJECTED_ENERGY_INSUFFICIENT"}
    components=("normal_work_s","prepare_work_s","ready_registration_bookkeeping_s",
                "publication_bookkeeping_wall_s","normal_evidence_wall_s")
    task_work=[dict(snapshot_time=r["snapshot_time"],status=r["status"],finished="finished_time" in r,
        full_normal_planning_work_s=math.fsum(r.get(k) or 0. for k in components),
        observed_components={k:r.get(k) for k in components}) for r in tasks]
    timing={key:measured(r.get(key) for r in tasks) for key in
        ("normal_work_s","normal_work_cpu_s","prepare_work_s","prepare_work_cpu_s","ready_guard_work_s",
         "ready_registration_bookkeeping_s","publication_bookkeeping_wall_s","publication_bookkeeping_cpu_s",
         "normal_evidence_wall_s","normal_evidence_cpu_s")}
    kinds=sorted({r["kind"] for r in services})
    service={kind:dict(wall=measured(r.get("wall_s") for r in services if r["kind"]==kind),
                       cpu=measured(r.get("cpu_s") for r in services if r["kind"]==kind)) for kind in kinds}
    return dict(task_count=len(tasks),finished_task_count=sum("finished_time" in r for r in tasks),
        nonempty_proposed_groups=len(proposed),finished_nonempty_groups=len(finished),committed_nonempty_groups=len(committed),
        pending_nonempty_groups=len(proposed)-len(finished),pending_tasks=sum("finished_time" not in r for r in tasks),
        no_executable_prefix_groups=sum(r.get("reason")=="NO_EXECUTABLE_PREFIX" for r in tasks),
        join_rate=len(committed)/len(finished) if finished else None,
        join_rate_denominator="finished nonempty proposed groups; pending and no-prefix excluded",
        ready_deadline_rejections=reasons.get("READY_DEADLINE_MISSED",0),rejection_reasons=dict(reasons),
        stale_or_dependency_rejections=sum(base_reasons[r] for r in stale_reasons),stale_reason_set=sorted(stale_reasons),
        energy_rejections=sum(base_reasons[r] for r in energy_reasons),energy_reason_set=sorted(energy_reasons),
        unknown_energy_rejections=base_reasons["ENERGY_UNKNOWN"],
        status_counts=dict(Counter(r["status"] for r in tasks)),timings=timing,online_service_by_kind=service,
        per_task_work=task_work,full_normal_planning_work=measured(r["full_normal_planning_work_s"] for r in task_work),
        all_task_work_includes_pending_partial=True,
        finished_task_full_planning_work=measured(r["full_normal_planning_work_s"] for r in task_work if r["finished"]),
        old_5s_full_planning_overrun_groups=sum(r["full_normal_planning_work_s"]>5. for r in task_work),
        old_callback_budget_miss_count=sum(bool(r.get("deadline_miss",False)) for r in callbacks),
        note="prepare_work_s includes ready_guard_work_s; publication is separately measured atomic bookkeeping, not backdated latency")


def run_job(output,arm,d,*,started=None):
    require(arm in ARMS,"unexpected arm")
    started=time.perf_counter() if started is None else started; cpu_started=time.process_time()
    scope,binding,ledger=verify(output,d)
    require((output/"run.started.json").is_file(),"job is internal to the fixed parent run")
    receipt=ledger.data["jobs"].get(arm,{})
    require((ledger.data.get("active_phase") or {}).get("phase")=="evaluation"
            and receipt.get("status")=="running" and receipt.get("timeout_s")==JOB_CAP,
            "parent-owned 900s watchdog receipt required")
    if hasattr(os,"sched_setaffinity"): os.sched_setaffinity(0,{CPU})
    directory=output/"evaluation"/arm; directory.mkdir(parents=True,exist_ok=False)
    exclusive(directory/"job.started.json",dict(pid=os.getpid(),started_at=d.campaign.utc_now(),arm=arm))
    world=scheduler=wrapper=None; restore_observer=lambda:None; holding={}; setup={}; last=None; completed=0; error=None
    advance_wall=advance_cpu=0.
    row=load(output/"roots"/ROOT_ID/"root.json")
    # Original identity restores t0; new checkpoints identify this execution source.
    checkpoint_identity=dict(row["identity"],source_hash=scope["source_hash"],normal_planning_interface=INTERFACE,ranking_scope=RANKING)
    def progress(phase,target=None):
        pending=wrapper.pending if wrapper is not None else None
        d.atomic_json(output/"progress"/(arm+".json"),dict(pid=os.getpid(),arm=arm,phase=phase,
            sim_time_s=None if world is None else world.time,completed_boundary_s=completed,target_boundary_s=target,
            wall_s=time.perf_counter()-started,cpu_s=time.process_time()-cpu_started,
            pending_stage=None if pending is None else pending["stage"],pending_due=None if pending is None else pending["due"],
            dirty=None if wrapper is None else wrapper.dirty))
    def costs():
        value=d.evaluation._costs(scheduler,observer,time.perf_counter()-started,time.process_time()-cpu_started,
            sum(r.get("dispatch_wall_s",0.) for r in world.dispatch_timings),setup_timings=setup,dispatch_records=world.dispatch_timings)
        value.update(world_advance_wall_s=advance_wall,world_advance_cpu_s=advance_cpu,
            full_online_wall_s_scope="JOB_ELAPSED_INCLUDING_SETUP_AND_PRIOR_CHECKPOINT_BOUNDARY_LOG_IO; NOT_PURE_PLANNING",
            measured_full_planning=lifecycle_summary(wrapper.records,scheduler.records,wrapper.online_services)["finished_task_full_planning_work"],
            all_tasks_executed_planning_work=lifecycle_summary(wrapper.records,scheduler.records,wrapper.online_services)["full_normal_planning_work"],
            simulation_integration_and_other_executor_wall_s=advance_wall-value["full_dispatch_wall_s"],
            simulation_timing_scope="world.advance minus measured dispatch; includes integration/execution checks/recording, not pure physics alone")
        return value
    def logs():
        d.atomic_json(directory/"normal_tasks.json",wrapper.records)
        d.atomic_json(directory/"online_services.json",wrapper.online_services)
        d.atomic_json(directory/"scheduler_records.json",scheduler.records)
        d.atomic_json(directory/"dispatch_timings.json",world.dispatch_timings)
    try:
        world=d.restore_root(row,d.NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8"))
        world.configure_practical_runtime(True); observer=d.OnlineObserver()
        scheduler=d.evaluation._scheduler(output,arm,binding,observer,ROOT_ID,setup)
        scheduler.provider.advisor_runtime.ranking_scope=RANKING
        wrapper=d.LatencyAwareScheduler(scheduler,lead_seconds=30.,preparation_margin=1.)
        world.decision_callback=wrapper; world.admission_callback=d.shared_fifo_admission
        require(world.scheduler_period==5.,"original scheduling period changed")
        holding,restore_observer=d.evaluation.observe_executed_holding(world)
        d.save_checkpoint(directory/"checkpoint_0000.json",world,wrapper,checkpoint_identity)
        for boundary in range(STEP,HORIZON+1,STEP):
            progress("advancing",boundary)
            before, before_cpu=time.perf_counter(),time.process_time()
            try: world.advance(float(boundary),wrapper,settle_only_at_end=True)
            finally:
                advance_wall+=time.perf_counter()-before; advance_cpu+=time.process_time()-before_cpu
            if abs(world.time-boundary)>1e-7 or world.failed or getattr(world,"terminated_invalid",False): break
            completed=boundary
            last=dict(status="COMPLETE_BOUNDARY",root_id=ROOT_ID,arm=arm,time=world.time,
                quality=d.episode_metrics(world),costs=costs(),holding_energy_by_ring=dict(holding),
                shared_manifest_hash=row["manifest_hash"],initial_state_hash=row["initial_state_hash"],
                execution_scope_id=scope["scope_id"],normal_lifecycle=lifecycle_summary(wrapper.records,scheduler.records,wrapper.online_services))
            if boundary%CHECKPOINT_STEP==0:
                d.save_checkpoint(directory/f"checkpoint_{boundary:04d}.json",world,wrapper,checkpoint_identity,boundary_summary=last)
            d.atomic_json(directory/f"boundary_{boundary:04d}.json",last); logs(); progress("boundary_saved")
    except Exception as exc:
        error=dict(error_type=type(exc).__name__,error=str(exc))
    finally:
        finalization_errors=[]; terminal=dict(status="UNAVAILABLE_SETUP_FAILURE"); replay_status="UNAVAILABLE"
        result=deepcopy(last) if last else dict(root_id=ROOT_ID,arm=arm,time=None,quality=None)
        try:
            if world is not None:
                terminal=d.evaluation.terminal_observation(world)
                d.atomic_json(directory/"terminal_observation.json",terminal)
                if wrapper is not None:
                    logs()
                    if completed!=HORIZON or world.failed or getattr(world,"terminated_invalid",False):
                        d.save_checkpoint(directory/"checkpoint_terminal.json",world,wrapper,checkpoint_identity)
                    result.update(terminal_costs=costs(),normal_lifecycle=lifecycle_summary(wrapper.records,scheduler.records,wrapper.online_services),
                        all_callbacks=d.aggregate(scheduler.records,costs(),world.dispatch_timings))
                    record=world.to_dict()
                    record.update(decisions=scheduler.records,stage3_trace=[dict(record_type="NormalTask",**r) for r in wrapper.records],
                        stage2_provenance=dict(training_binding=binding,execution_scope=scope),
                        tentative_predictions=[dict(aircraft_id=aid,as_of=world.time,plans=[p.to_dict() for p in plans])
                                               for aid,plans in scheduler.provider.previous.items()])
                    d.ReplayBundle.write(directory/"replay",record=record,code_commit=scope["source_commit"],
                        config_path=world.config.path,command=str(DRIVER),seed=SEED,eager=False)
                    replay_status="WRITTEN_HUMAN_PENDING"
        except Exception as exc:
            finalization_errors.append(dict(error_type=type(exc).__name__,error=str(exc)))
        finally: restore_observer()
        result.update(status="COMPLETE" if completed==HORIZON and error is None and not finalization_errors else "EVALUATION_PARTIAL",
            completed_boundary_s=completed,terminal_observation=terminal,replay_status=replay_status,
            replay_path=str(directory/"replay") if replay_status.startswith("WRITTEN") else None,
            execution_scope=scope,model_fingerprint=binding["fingerprint"],error=error,finalization_errors=finalization_errors,
            full_job_wall_s=time.perf_counter()-started,process_cpu_s=time.process_time()-cpu_started,
            actual_cpu=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None)
        d.atomic_json(directory/"result.json",result)
        d.atomic_json(directory/"job.finished.json",dict(status=result["status"],completed_boundary_s=completed,
            observed_terminal_time_s=terminal.get("observed_time_s"),full_job_wall_s=result["full_job_wall_s"],
            error=error,finalization_errors=finalization_errors,finished_at=d.campaign.utc_now()))
        progress("complete" if result["status"]=="COMPLETE" else "partial")
    if error or finalization_errors: raise RuntimeError("evaluation failed; preserved partial evidence: "+str(error or finalization_errors))
    return result


def job_command(output,checkout,arm):
    return [sys.executable,str(DRIVER),"job","--output",str(output),"--checkout",str(checkout),"--arm",arm]


def hard_timeout_evidence(output,arm,receipt,d):
    directory=output/"evaluation"/arm; directory.mkdir(parents=True,exist_ok=True)
    boundaries=sorted(directory.glob("boundary_*.json"))
    last=load(boundaries[-1]) if boundaries else None
    terminal=(load(directory/"terminal_observation.json") if (directory/"terminal_observation.json").exists()
              else dict(status="UNAVAILABLE_HARD_TIMEOUT",observed_time_s=None))
    value=dict(status="BUDGET_LIMITED",arm=arm,receipt=receipt,time=None,
        completed_boundary_s=last["time"] if last else 0,last_completed_boundary=last,
        terminal_observation=terminal,replay_status="PRESENT_NOT_VALIDATED" if (directory/"replay/replay.json").exists() else "UNAVAILABLE_HARD_TIMEOUT",
        interpretation="Last completed boundary is not the observed state at process kill; no recovery/retry performed")
    d.atomic_json(directory/"watchdog_partial.json",value)
    if not (directory/"terminal_observation.json").exists(): d.atomic_json(directory/"terminal_observation.json",terminal)
    if not (directory/"result.json").exists(): d.atomic_json(directory/"result.json",value)
    return value


def run(output,d):
    with d.campaign.campaign_lock(output):
        scope,_,ledger=verify(output,d)
        require(ledger.data.get("paused_at") is not None and not ledger.data.get("active_phase") and not ledger.data["jobs"],"fresh paused run required")
        require(not (output/"run.started.json").exists(),"run already attempted; no retry")
        resource=d.campaign.resource_probe(); require(CPU in resource.get("free_physical_cores",[]),"fixed CPU12 unavailable; return to primary")
        exclusive(output/"run.started.json",dict(started_at=d.campaign.utc_now(),resource_probe=resource,scope_id=scope["scope_id"]))
        receipts=[]; error=None
        try:
            with ledger.phase("evaluation"):
                for arm in ARMS:
                    require(ledger.remaining("evaluation")>=JOB_CAP+1.,"full equal 900s cap unavailable; return to primary")
                    receipt=d.campaign.run_subprocess(ledger,"evaluation",arm,job_command(output,d.checkout,arm),output/"logs",cap=JOB_CAP)
                    receipts.append(receipt); d.atomic_json(output/"receipts"/(arm+".json"),receipt)
                    if receipt["status"]=="BUDGET_LIMITED": hard_timeout_evidence(output,arm,receipt,d)
                    require(receipt["timeout_s"]==JOB_CAP,"child cap was shortened")
                    require(receipt["status"]=="completed" and receipt.get("exit_code")==0,"child did not complete; no automatic retry")
                    require(load(output/"evaluation"/arm/"result.json")["status"]=="COMPLETE","partial result; return to primary")
        except Exception as exc: error=dict(error_type=type(exc).__name__,error=str(exc))
        finally:
            ledger.pause()
            result=dict(status="COMPLETED" if len(receipts)==2 and error is None else "INCOMPLETE_RETURN_TO_PRIMARY",
                receipts=receipts,error=error,finished_at=d.campaign.utc_now(),no_automatic_retry=True)
            d.atomic_json(output/"run.finished.json",result)
        if error: raise RuntimeError(error["error"])
        return result


def summarize(output,d):
    scope,_,ledger=verify(output,d)
    require(ledger.data.get("paused_at") is not None and not ledger.data.get("active_phase"),"run must be paused before summarizing")
    require(not (output/"summary.json").exists() and not (output/"README.md").exists(),"summary already exists")
    paths={arm:output/"evaluation"/arm for arm in ARMS}
    boundaries={arm:{int(p.stem.split("_")[-1]):p for p in base.glob("boundary_*.json")} for arm,base in paths.items()}
    common=max(set(boundaries[ARMS[0]]) & set(boundaries[ARMS[1]]),default=0)
    rows={arm:load(values[common]) for arm,values in boundaries.items()} if common else {}
    results={arm:load(base/"result.json") if (base/"result.json").exists() else dict(status="NOT_RUN") for arm,base in paths.items()}
    values=dict(schema="normal_latency_summary_v1",execution_scope=scope,common_completed_boundary_s=common,
        common_rows=rows,independent_results=results,
        quality_differences=d.evaluation.quality_differences(rows[ARMS[0]]["quality"],rows[ARMS[1]]["quality"]) if rows else None,
        interpretation="One fixed paired rollout; pending is neither failed nor successful joining; replay HUMAN_PENDING")
    d.atomic_json(output/"summary.json",values)
    text="# Normal-planning latency interface: fixed 120s H/L pair\n\n"
    text+=f"Common completed boundary: {common}s. Source: `{scope['source_commit']}`.\n\n"
    text+="H is heuristic candidate order with the same CP-SAT backend as L. Original model fingerprint is unchanged; execution_scope.json records the new interface and ranking scope.\n\n"
    for arm in ARMS:
        result=results[arm]; life=result.get("normal_lifecycle",{})
        text+=f"- {arm}: {result['status']}; committed/finished nonempty groups {life.get('committed_nonempty_groups')}/{life.get('finished_nonempty_groups')}; pending groups {life.get('pending_nonempty_groups')}; join rate {life.get('join_rate')}.\n"
    text+="\nBoundary costs end before that boundary checkpoint; terminal_costs include terminal bookkeeping before replay export; full_job_wall_s includes replay export; subprocess receipt also includes process startup. World.advance-minus-dispatch includes integration, execution checks and recording, and is not pure physics.\n\n"
    text+="Publication is an atomic event with separately measured bookkeeping CPU, not backdated validation. Shared emergency response timestamps are unchanged and its CPU service delays normal tasks. No training, seed scan, retries or statistical/generalization claim. Exact-120 dispatch is censored. See independent terminals, actual scheduler records and authoritative replay; HUMAN_PENDING.\n"
    (output/"README.md").write_text(text,encoding="utf8")
    return dict(status="SUMMARIZED",common_completed_boundary_s=common)


def main(argv=None):
    started=time.perf_counter(); parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","run","job","summarize"))
    parser.add_argument("--output",type=Path,default=Path("/home/user/training_outputs/terminal/stage3_latency_interface_20261026"))
    parser.add_argument("--checkout",type=Path,required=True)
    parser.add_argument("--source-freeze",type=Path)
    parser.add_argument("--model-parent",type=Path,default=Path("/home/user/training_outputs/terminal/stage3_online_v2_20261025"))
    parser.add_argument("--initial-parent",type=Path,default=Path("/home/user/training_outputs/terminal/stage3_learning_v1_0_20261023"))
    parser.add_argument("--arm",choices=ARMS)
    args=parser.parse_args(argv)
    if args.stage=="prepare" and args.source_freeze is None: parser.error("prepare requires primary-provided --source-freeze")
    if args.stage=="job" and args.arm is None: parser.error("job requires --arm")
    output=args.output.resolve(); d=dependencies(args.checkout)
    result=(prepare(args,d) if args.stage=="prepare" else run(output,d) if args.stage=="run" else
            run_job(output,args.arm,d,started=started) if args.stage=="job" else summarize(output,d))
    print(json.dumps(dict(stage=args.stage,output=str(output),status=result["status"],
        completed_boundary_s=result.get("completed_boundary_s"),scope_id=result.get("scope_id")),sort_keys=True))


if __name__=="__main__": main()
