"""One 300 s high-traffic H/L pair using the existing frozen future-state model."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

DRIVER=Path(__file__).resolve()
ROOT_ID,SEED,PROFILE,CPU,HORIZON="s3r24",7325,"stress36_rate180_v1",12,300
STEP,CHECKPOINT_STEP,JOB_CAP=5,60,1800.
PHASE_CAPS=dict(preflight_tests=300.,evaluation=3605.,archive=300.)
TOTAL_CAP,COMPUTE_STOP=4205.,3905.
CORE_COMMIT="79fc55c81c28a2b5e59e28c939d99cf40b438d1b"
CORE_HASH="52852f54bd7372a44f4ff7053ef94dcf67522b05779e03f9643c7167f9ccd524"
MODEL_HASH="29d942674fad65805bd33598aecd880add6d9cad223b1fd8f11d2b90c6ac292d"
NORMALIZER_HASH="6df1cc091824ac4bf066ad8ff8039bd542310a14e6f3e46630a39481384abeb0"
ARMS=("heuristic_reduced","learned_reduced")
MODEL_FILES=("model/model.pt","model/feature_normalizer.json","model/model_card.json",
             "run_binding.generated.json","formal_training.guard.json","source_freeze.json")


def require(condition,message):
    if not condition:raise ValueError(message)
def load(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def file_hash(path):return sha256(Path(path).read_bytes()).hexdigest()
def exclusive(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as stream:json.dump(value,stream,sort_keys=True,allow_nan=False)


def module_from(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def configure_latency(latency):
    """Only case/horizon/watchdog constants change in the imported frozen runner."""
    latency.ROOT_ID,latency.SEED,latency.CPU=ROOT_ID,SEED,CPU
    latency.HORIZON,latency.STEP,latency.CHECKPOINT_STEP,latency.JOB_CAP=HORIZON,STEP,CHECKPOINT_STEP,JOB_CAP
    return latency


def dependencies(checkout):
    checkout=Path(checkout).resolve()
    require(checkout.is_dir(),"explicit frozen checkout required")
    future=module_from(checkout/"scripts/stage3/run_future_learning.py","_high300_frozen_future")
    d=future.dependencies(checkout);d.future=future;configure_latency(d.latency)
    return d


def protocol(d):
    p=d.latency.protocol(d.dependency_protocol,20.,4.,d.handoff_version)
    p.update(schema="future_learning_high300_20261027_v1",root_id=ROOT_ID,seed=SEED,scenario_profile=PROFILE,
        target_sim_seconds=HORIZON,cpu=CPU,boundary_seconds=STEP,checkpoint_seconds=CHECKPOINT_STEP,
        ranking_scope="comparable_group_lazy_v2",feature_state_semantics="planning_effective_state_v1",
        runtime_projection_version=d.projection_version,model_sha256=MODEL_HASH,normalizer_sha256=NORMALIZER_HASH,
        training="FROZEN_NO_TRAINING",evaluation_role="HIGH_TRAFFIC_SINGLE_FROZEN_MODEL_PAIR",
        job_wall_cap_s=JOB_CAP,total_evaluation_wall_cap_s=PHASE_CAPS["evaluation"],
        total_campaign_wall_cap_s=TOTAL_CAP,compute_stop_s=COMPUTE_STOP,phase_caps=deepcopy(PHASE_CAPS),
        endpoint_policy="settle_only_at_end; dispatch/delivery exactly at 300 is censored and remains pending",
        initial_airborne=36,initial_ground=3,arrival_rate_per_hour=180,automatic_stage_transition=False,
        initial_estimated_wall_s=900.,estimate_basis="previous medium120 pair ~198s * 2.5 horizon * 1.5 traffic = ~743s, plus margin",
        reporting="300s timeline/full saved replay PVA; no legacy 0/60/120 contact sheet")
    return p


def verify_core(frozen,d):
    require((frozen.get("source_commit"),frozen.get("source_hash"),len(frozen.get("source_files",{})))==
        (CORE_COMMIT,CORE_HASH,106),"wrong approved frozen core identity")
    d.campaign.verify_sources(frozen)


def verify_parent(parent,d):
    parent=Path(parent);binding=load(parent/"run_binding.generated.json")
    hashes=d.future.verify_model(parent,binding)
    require(hashes==(MODEL_HASH,NORMALIZER_HASH),"wrong approved frozen model/normalizer")
    verify_core(load(parent/"source_freeze.json"),d)
    return binding


def validate_case(row,manifest,state,d):
    require((row["root_id"],row["seed"],row["scenario_profile"])==(ROOT_ID,SEED,PROFILE),"wrong high-traffic physical root")
    require(manifest["seed"]==SEED and manifest["rate_per_hour"]==180 and manifest["arrival_window"]==3600.
        and len(manifest["requests"])==167 and sum(r["release_time"]<=HORIZON for r in manifest["requests"])==13,"high-traffic manifest changed")
    require(d.digest(manifest)==row["manifest_hash"] and d.digest(state)==row["initial_state_hash"],"raw high-traffic input digest changed")
    require(len(state["aircraft"])==39 and sum(a["status"].startswith("HOLD_") for a in state["aircraft"])==36
        and sum(a["status"]=="LANDED_SERVICE" for a in state["aircraft"])==3,"36 airborne plus 3 ground required")
    require([sum(a["status"]==status for a in state["aircraft"]) for status in ("HOLD_OUTER","HOLD_MIDDLE","HOLD_INNER")]==[14,12,10],"initial ring distribution changed")


def prepare(args,d):
    started=time.perf_counter();output=args.output.resolve();parent=args.model_parent.resolve();initial=args.initial_parent.resolve()
    require(not output.exists(),"fresh high300 output required; no retry/reset")
    require(not any(output.is_relative_to(p) or p.is_relative_to(output) for p in (parent,initial)),"new output overlaps readonly inputs")
    binding=verify_parent(parent,d)
    resource=d.campaign.resource_probe()
    for package in ("numpy","scipy","PyYAML","ortools","torch"):
        observed=resource.get("versions",{}).get(package)
        require(observed is not None and observed==binding.get("environment",{}).get("versions",{}).get(package),"numerical dependency changed: "+package)
    require(all(resource.get("thread_environment",{}).get(key)=="1" for key in d.latency.THREAD_KEYS),"one numerical thread required before Python starts")
    original=initial/"roots"/ROOT_ID
    validate_case(load(original/"root.json"),load(original/"manifest.json"),load(original/"initial_state.json"),d)
    output.mkdir(parents=True);exclusive(output/"prepare.started.json",dict(started_at=d.campaign.utc_now()))
    ledger=d.campaign.CampaignLedger(output/"budget_ledger.json")
    ledger.data.update(total_s=TOTAL_CAP,compute_stop_s=COMPUTE_STOP,phase_caps=deepcopy(PHASE_CAPS));ledger.persist();ledger.pause()
    initial_elapsed=ledger.elapsed;ledger.data["phase_used"]["preflight_tests"]=initial_elapsed;ledger.persist()
    try:
        for name in MODEL_FILES:
            target=output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/name,target)
            require(file_hash(target)==file_hash(parent/name),"frozen training artifact copy changed: "+name)
        row=d.future.copy_root(initial,output,ROOT_ID,d)
        validate_case(row,load(output/"roots"/ROOT_ID/"manifest.json"),load(output/"roots"/ROOT_ID/"initial_state.json"),d)
        scope=dict(protocol=protocol(d),checkout=str(d.checkout),source_commit=CORE_COMMIT,source_hash=CORE_HASH,
            driver_path=str(DRIVER),driver_sha256=file_hash(DRIVER),model_fingerprint=deepcopy(binding["fingerprint"]),
            model_parent_readonly=str(parent),initial_parent_readonly=str(initial),training_source_hash=binding["source_hash"],
            shared_manifest_hash=row["manifest_hash"],initial_state_hash=row["initial_state_hash"],resource_probe=resource,
            interpretation="New high-traffic execution scope; copied original training fingerprint/guard remain unchanged; no training")
        scope["prepared_file_sha256"]={p.relative_to(output).as_posix():file_hash(p) for base in (output/"model",output/"roots"/ROOT_ID) for p in base.iterdir()}
        scope["prepared_file_sha256"].update({name:file_hash(output/name) for name in MODEL_FILES if not name.startswith("model/")})
        scope["scope_id"]=d.digest(scope)
        d.atomic_json(output/"execution_scope.json",scope)
        d.atomic_json(output/"prepare.finished.json",dict(status="COMPLETED",scope_id=scope["scope_id"]))
    finally:
        ledger.credit("high300_prepare",max(0.,time.perf_counter()-started-initial_elapsed),phase="preflight_tests");ledger.pause()
    return dict(status="PREPARED",scope_id=scope["scope_id"])


def verify(output,d):
    scope=load(output/"execution_scope.json")
    require(load(output/"prepare.finished.json")==dict(status="COMPLETED",scope_id=scope["scope_id"]),"prepare did not complete")
    require(scope["scope_id"]==d.digest({k:v for k,v in scope.items() if k!="scope_id"}),"execution scope identity changed")
    require(scope["protocol"]==protocol(d) and Path(scope["checkout"]).resolve()==d.checkout,"saved high300 protocol changed")
    require(scope["driver_sha256"]==file_hash(DRIVER) and Path(scope["driver_path"]).resolve()==DRIVER,"reviewed high300 wrapper changed")
    verify_core(load(output/"source_freeze.json"),d)
    binding=verify_parent(Path(scope["model_parent_readonly"]),d)
    require(binding["fingerprint"]==scope["model_fingerprint"] and load(output/"run_binding.generated.json")==binding,"original training fingerprint changed")
    for name,expected in scope["prepared_file_sha256"].items():require(file_hash(output/name)==expected,"prepared file changed: "+name)
    require(file_hash(output/"model/model.pt")==MODEL_HASH and file_hash(output/"model/feature_normalizer.json")==NORMALIZER_HASH,"copied frozen model hash changed")
    validate_case(load(output/"roots"/ROOT_ID/"root.json"),load(output/"roots"/ROOT_ID/"manifest.json"),load(output/"roots"/ROOT_ID/"initial_state.json"),d)
    require((output/"budget_ledger.json").is_file(),"existing ledger required")
    ledger=d.campaign.CampaignLedger(output/"budget_ledger.json")
    require(ledger.data["total_s"]==TOTAL_CAP and ledger.data["compute_stop_s"]==COMPUTE_STOP and ledger.data["phase_caps"]==PHASE_CAPS,"high300 ledger caps changed")
    return scope,binding,ledger


def command(output,d,stage,arm=None):
    result=[sys.executable,str(DRIVER),stage,"--checkout",str(d.checkout),"--output",str(output)]
    return result+(["--arm",arm] if arm else [])


def run(output,d):
    with d.campaign.campaign_lock(output):
        scope,_,ledger=verify(output,d)
        require(ledger.data.get("paused_at") is not None and not ledger.data.get("active_phase") and not ledger.data["jobs"],"fresh paused evaluation required")
        require(not (output/"run.started.json").exists(),"high300 run already attempted; no retry")
        resource=d.campaign.resource_probe();require(CPU in resource.get("free_physical_cores",[]),"CPU12 unavailable; return to primary")
        exclusive(output/"run.started.json",dict(scope_id=scope["scope_id"],resource_probe=resource,started_at=d.campaign.utc_now()))
        receipts=[];error=None
        try:
            with ledger.phase("evaluation"):
                for arm in ARMS:
                    require(ledger.remaining("evaluation")>=JOB_CAP+1.,"full equal 1800s arm cap unavailable")
                    receipt=d.campaign.run_subprocess(ledger,"evaluation",arm,command(output,d,"job",arm),output/"logs",cap=JOB_CAP)
                    receipts.append(receipt);d.atomic_json(output/"receipts"/(arm+".json"),receipt)
                    if receipt["status"]=="BUDGET_LIMITED":d.latency.hard_timeout_evidence(output,arm,receipt,d)
                    require(receipt["timeout_s"]==JOB_CAP,"arm watchdog cap was shortened")
                    require(receipt["status"]=="completed" and receipt.get("exit_code")==0,"arm failed; no retry")
                    require(load(output/"evaluation"/arm/"result.json")["status"]=="COMPLETE","arm partial; return to primary")
        except Exception as exc:error=dict(error_type=type(exc).__name__,error=str(exc))
        finally:
            ledger.pause();result=dict(status="COMPLETED" if len(receipts)==2 and error is None else "INCOMPLETE_RETURN_TO_PRIMARY",
                receipts=receipts,error=error,finished_at=d.campaign.utc_now(),no_automatic_retry=True)
            d.atomic_json(output/"run.finished.json",result)
        if error:raise RuntimeError(error["error"])
        return result


def saved_report(output,d,scope):
    paths={arm:output/"evaluation"/arm for arm in ARMS}
    boundaries={a:{int(p.stem.split("_")[-1]):p for p in base.glob("boundary_*.json")} for a,base in paths.items()}
    common=max(set(boundaries[ARMS[0]])&set(boundaries[ARMS[1]]),default=0)
    rows={a:load(b[common]) for a,b in boundaries.items()} if common else {}
    results={a:load(base/"result.json") if (base/"result.json").is_file() else dict(status="NOT_RUN") for a,base in paths.items()}
    d.atomic_json(output/"summary.json",dict(schema="normal_latency_summary_v1",execution_scope=scope,
        common_completed_boundary_s=common,common_rows=rows,independent_results=results,
        quality_differences=d.evaluation.quality_differences(rows[ARMS[0]]["quality"],rows[ARMS[1]]["quality"]) if rows else None,
        interpretation="One fixed 300s high-traffic frozen-model pair; pending is censored, replay HUMAN_PENDING"))
    (output/"README.md").write_text(f"# High-traffic frozen-model H/L pair: 300 s\n\nCommon completed boundary: {common} s. No training, tuning or retries. Exact-300 dispatch is censored. Original model fingerprint is preserved; execution_scope.json records this case. Replay HUMAN_PENDING.\n",encoding="utf8")
    if not all((base/"result.json").is_file() for base in paths.values()):return dict(status="SUMMARY_ONLY_INCOMPLETE_PAIR")
    report=module_from(d.checkout/"scripts/stage3/report_latency_interface.py","_high300_saved_report")
    value,timeline,quality,errors,contacts=report.collect(output,report.saved_core(d.checkout))
    # Frozen collect's legacy s3r23 lookup affects these two root checks only.
    # Recheck against the actual case, without inventing an alias root.
    root=load(output/"roots"/ROOT_ID/"root.json")
    for field,key in (("same_manifest","shared_manifest_hash"),("same_initial_state","initial_state_hash")):
        root_key="manifest_hash" if field=="same_manifest" else "initial_state_hash"
        value["identity_checks"][field]=report.same_known([results[a].get(key) for a in ARMS]+[scope.get(key),root[root_key]])
    report.write_csv(output/"timeline.csv",timeline,tuple(timeline[0]) if timeline else ("arm","snapshot_time","status"))
    report.write_csv(output/"boundary_quality.csv",quality,("arm","time_s","L_normal","L_emergency","F_energy_total","N_airborne_pending","N_admitted_total","E_terminal_J","holding_time"))
    report.write_csv(output/"replay_sample_errors.csv",errors,("arm","time","aircraft_id","trajectory_hash","status","p_error","v_error","a_error"))
    report.figures(output,value,timeline,quality,contacts,False)
    value.update(derived_script_sha256=file_hash(DRIVER),high300_report_adapter=dict(actual_root_id=ROOT_ID,
        endpoint_seconds=HORIZON,legacy_root_checks_recomputed=True,legacy_contact_sheet_omitted=True),
        figure_qa=dict(visual_review="PENDING_PRIMARY_VIEW_IMAGE",human_replay_acceptance="HUMAN_PENDING"))
    d.atomic_json(output/"analysis_summary.json",value)
    text=report.report_text(value,timeline).replace("censored at 120 s","censored at 300 s")
    text=text.replace("# Normal planning latency interface — saved-artifact report","# High-traffic frozen-model 300 s H/L pair — saved-artifact report")
    (output/"FINAL_REPORT.md").write_text(text,encoding="utf8")
    d.atomic_json(output/"figure_QA.json",value["figure_qa"])
    return dict(status="SAVED_REPORT_WRITTEN",common_completed_boundary_s=common,visual_review="PENDING_PRIMARY_VIEW_IMAGE",human_replay_acceptance="HUMAN_PENDING")


def archive(output,d):
    with d.campaign.campaign_lock(output):
        scope,_,ledger=verify(output,d)
        require(ledger.data.get("paused_at") is not None and not ledger.data.get("active_phase"),"paused inactive ledger required")
        require((output/"run.finished.json").is_file() and not (output/"archive.started.json").exists(),"run terminal/first archive required")
        exclusive(output/"archive.started.json",dict(started_at=d.campaign.utc_now()))
        try:
            with ledger.phase("archive"):result=saved_report(output,d,scope)
        finally:ledger.pause()
        d.atomic_json(output/"archive.finished.json",dict(result,finished_at=d.campaign.utc_now()))
        return result


def main(argv=None):
    started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","run","job","archive"))
    parser.add_argument("--checkout",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("/home/user/training_outputs/terminal/stage3_future_learning_high300_20261027"))
    parser.add_argument("--model-parent",type=Path,default=Path("/home/user/training_outputs/terminal/stage3_future_learning_20261027"))
    parser.add_argument("--initial-parent",type=Path,default=Path("/home/user/training_outputs/terminal/stage3_learning_v1_0_20261023"))
    parser.add_argument("--arm",choices=ARMS)
    args=parser.parse_args(argv)
    if args.stage=="job" and args.arm is None:parser.error("internal job requires --arm")
    d=dependencies(args.checkout);output=args.output.resolve()
    result=prepare(args,d) if args.stage=="prepare" else run(output,d) if args.stage=="run" else archive(output,d) if args.stage=="archive" else d.latency.run_job(output,args.arm,d,started=started,execution_verifier=verify)
    print(json.dumps(dict(stage=args.stage,status=result["status"],output=str(output)),sort_keys=True))


if __name__=="__main__":main()
