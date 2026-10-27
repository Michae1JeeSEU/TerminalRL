"""Small serialized-input and frozen-runner fixtures; no new scientific run."""
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.stage3 import run_future_learning_high300 as runner
from terminal_marl.core.rotating_slots.stage3_learning.campaign import CampaignLedger

PROJECT=Path(__file__).resolve().parents[1]
FROZEN=PROJECT/".worktrees/future-learning-repair-20261027"


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False),encoding="utf8")
def digest(value):return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
@contextmanager
def lock(output):yield


def prepared(tmp_path,monkeypatch):
    latency=runner.configure_latency(runner.module_from(FROZEN/"scripts/stage3/run_latency_interface.py","_high300_fixture_latency"))
    future=runner.module_from(FROZEN/"scripts/stage3/run_future_learning.py","_high300_fixture_future")
    parent=tmp_path/"model_parent";(parent/"model").mkdir(parents=True)
    (parent/"model/model.pt").write_bytes(b"original future weights")
    atomic(parent/"model/feature_normalizer.json",dict(schema_id="v2"))
    monkeypatch.setattr(runner,"MODEL_HASH",runner.file_hash(parent/"model/model.pt"))
    monkeypatch.setattr(runner,"NORMALIZER_HASH",runner.file_hash(parent/"model/feature_normalizer.json"))
    fingerprint=dict(feature_schema_id="v2",training_source="original",model_id="NO_MODEL")
    environment=dict(versions={name:"fixture_v1" for name in ("numpy","scipy","PyYAML","ortools","torch")},
        thread_environment={name:"1" for name in latency.THREAD_KEYS},free_physical_cores=[12])
    atomic(parent/"model/model_card.json",dict(fingerprint=fingerprint,model_hash=runner.MODEL_HASH,normalizer_sha256=runner.NORMALIZER_HASH))
    atomic(parent/"run_binding.generated.json",dict(fingerprint=fingerprint,source_hash=runner.CORE_HASH,environment=environment))
    atomic(parent/"formal_training.guard.json",dict(output="/original/server/training/model",seed=20261023))
    atomic(parent/"source_freeze.json",dict(source_commit=runner.CORE_COMMIT,source_hash=runner.CORE_HASH,
        source_files={str(i):"fixture" for i in range(106)}))
    initial=tmp_path/"initial";root=initial/"roots/s3r24";root.mkdir(parents=True)
    manifest=dict(seed=7325,rate_per_hour=180,arrival_window=3600.,requests=[dict(release_time=10.*i if i<13 else 400.+i) for i in range(167)])
    state=dict(aircraft=[dict(status=status) for status,count in (("HOLD_OUTER",14),("HOLD_MIDDLE",12),("HOLD_INNER",10),("LANDED_SERVICE",3)) for _ in range(count)])
    identity={"physical":"original high root"}
    row=dict(root_id="s3r24",seed=7325,scenario_profile="stress36_rate180_v1",split="test",identity=identity,
        manifest_hash=digest(manifest),initial_state_hash=digest(state))
    payload=dict(time=0.,identity=identity,sink=None)
    for name,value in (("root.json",row),("manifest.json",manifest),("initial_state.json",state),
        ("initial.checkpoint.json",dict(payload=payload,sha256=digest(payload)))):atomic(root/name,value)
    verified=[]
    def verify_model(path,binding):
        assert Path(path)==parent;verified.append(str(path))
        return runner.MODEL_HASH,runner.NORMALIZER_HASH
    d=SimpleNamespace(checkout=FROZEN.resolve(),latency=latency,projection_version="planning_effective_state_v1",
        dependency_protocol="dependency",handoff_version="handoff",future=SimpleNamespace(copy_root=future.copy_root,verify_model=verify_model),
        digest=digest,atomic_json=atomic,campaign=SimpleNamespace(CampaignLedger=CampaignLedger,verify_sources=lambda frozen:None,
            utc_now=lambda:"UTC",resource_probe=lambda:deepcopy(environment),campaign_lock=lock))
    args=SimpleNamespace(output=tmp_path/"output",model_parent=parent,initial_parent=initial)
    assert runner.prepare(args,d)["status"]=="PREPARED"
    return args.output,d,parent,initial,verified


def test_prepare_serializes_verifies_and_copies_original_guard_inputs_unchanged(tmp_path,monkeypatch):
    output,d,parent,initial,verified=prepared(tmp_path,monkeypatch)
    scope,binding,ledger=runner.verify(output,d)
    assert json.loads(json.dumps(runner.protocol(d)))==runner.protocol(d)==scope["protocol"]
    assert (scope["protocol"]["root_id"],scope["protocol"]["seed"],scope["protocol"]["target_sim_seconds"])==("s3r24",7325,300)
    assert scope["protocol"]["training"]=="FROZEN_NO_TRAINING" and scope["protocol"]["ranking_scope"]=="comparable_group_lazy_v2"
    assert ledger.data["total_s"]==4205. and ledger.data["compute_stop_s"]==3905.
    assert ledger.data["phase_caps"]==dict(preflight_tests=300.,evaluation=3605.,archive=300.)
    for name in runner.MODEL_FILES:assert (output/name).read_bytes()==(parent/name).read_bytes()
    for name in ("manifest.json","initial_state.json","initial.checkpoint.json"):
        assert (output/"roots/s3r24"/name).read_bytes()==(initial/"roots/s3r24"/name).read_bytes()
    assert binding==runner.load(parent/"run_binding.generated.json") and len(verified)==2
    assert runner.load(output/"formal_training.guard.json")["output"]=="/original/server/training/model"
    changed=deepcopy(scope);changed["protocol"]["target_sim_seconds"]=600;changed["scope_id"]=digest({k:v for k,v in changed.items() if k!="scope_id"})
    atomic(output/"execution_scope.json",changed);atomic(output/"prepare.finished.json",dict(status="COMPLETED",scope_id=changed["scope_id"]))
    with pytest.raises(ValueError,match="protocol changed"):runner.verify(output,d)


def test_wrong_case_and_frozen_model_hash_are_rejected(tmp_path,monkeypatch):
    output,d,parent,initial,_=prepared(tmp_path,monkeypatch)
    root=initial/"roots/s3r24";row=runner.load(root/"root.json");row["seed"]=7324
    with pytest.raises(ValueError,match="physical root"):runner.validate_case(row,runner.load(root/"manifest.json"),runner.load(root/"initial_state.json"),d)
    (output/"model/model.pt").write_bytes(b"changed model")
    with pytest.raises(ValueError,match="prepared file changed|model hash"):runner.verify(output,d)


def test_frozen_job_advances_only_to_300_and_censors_exact_endpoint(tmp_path,monkeypatch):
    output,d,_,_,_=prepared(tmp_path,monkeypatch)
    atomic(output/"run.started.json",{})
    ledger=CampaignLedger(output/"budget_ledger.json");ledger.data["active_phase"]=dict(phase="evaluation",started_epoch=ledger.clock())
    ledger.data["jobs"]={a:dict(status="running",timeout_s=1800.) for a in runner.ARMS};ledger.persist()
    scheduler=SimpleNamespace(records=[],provider=SimpleNamespace(previous={},advisor_runtime=SimpleNamespace(ranking_scope="legacy")))
    calls=[];saved=[]
    class Wrapper:
        def __init__(self,callback,**options):
            assert options==dict(lead_seconds=20.,preparation_margin=1.,search_handoff_reserve_seconds=4.)
            self.records=[];self.online_services=[];self.pending=None;self.dirty=False
        def __call__(self,*args):raise AssertionError("endpoint must not dispatch")
    class World:
        time=0.;failed=False;terminated_invalid=False;scheduler_period=5.;config=SimpleNamespace(path="fixture.yaml")
        dispatch_timings=[]
        def configure_practical_runtime(self,value):assert value
        def advance(self,target,wrapper,*,settle_only_at_end):
            assert settle_only_at_end and target<=300 and self.decision_callback is wrapper
            assert scheduler.provider.advisor_runtime.ranking_scope=="comparable_group_lazy_v2"
            calls.append(target);self.time=target
        def to_dict(self):return dict(states=[],events=[],trajectories={})
    world=World()
    d.restore_root=lambda *args:world;d.NormalSchedulerCallback=lambda *a,**k:object();d.OnlineObserver=lambda:object()
    d.LatencyAwareScheduler=Wrapper;d.shared_fifo_admission=object()
    d.save_checkpoint=lambda path,w,*args,**kwargs:(saved.append(w.time),atomic(path,dict(time=w.time)))
    d.ReplayBundle=SimpleNamespace(write=lambda path,**kwargs:atomic(Path(path)/"replay.json",{}))
    d.aggregate=lambda *args:{}
    d.evaluation=SimpleNamespace(_scheduler=lambda *args:scheduler,
        _costs=lambda *args,**kwargs:dict(full_dispatch_wall_s=0.),observe_executed_holding=lambda w:({},lambda:None),
        terminal_observation=lambda w:dict(observed_time_s=w.time))
    d.episode_metrics=lambda w:dict(L_normal=0)
    monkeypatch.setattr(d.latency.os,"sched_setaffinity",lambda *args:None,raising=False)
    result=d.latency.run_job(output,runner.ARMS[0],d,execution_verifier=runner.verify)
    assert result["status"]=="COMPLETE" and result["completed_boundary_s"]==300
    assert calls==list(range(5,301,5)) and saved==[0.,60.,120.,180.,240.,300.]
    assert len(list((output/"evaluation"/runner.ARMS[0]).glob("boundary_*.json")))==60
    assert result["root_id"]=="s3r24" and result["terminal_observation"]["observed_time_s"]==300.


def test_second_arm_never_receives_shortened_watchdog(tmp_path,monkeypatch):
    output,d,_,_,_=prepared(tmp_path,monkeypatch);calls=[]
    def subprocess(ledger,phase,arm,command,directory,*,cap):
        calls.append((arm,cap));atomic(output/"evaluation"/arm/"result.json",dict(status="COMPLETE"))
        ledger.data["phase_used"]["evaluation"]=1805.;ledger.persist()
        return dict(status="completed",exit_code=0,timeout_s=1800.)
    d.campaign.run_subprocess=subprocess
    with pytest.raises(RuntimeError,match="full equal 1800s"):runner.run(output,d)
    assert calls==[(runner.ARMS[0],1800.)]
    assert runner.load(output/"run.finished.json")["status"]=="INCOMPLETE_RETURN_TO_PRIMARY"


def test_actual_frozen_saved_report_exports_300_timeline_without_alias_root(tmp_path,monkeypatch):
    output,d,_,_,_=prepared(tmp_path,monkeypatch);scope,_,_=runner.verify(output,d)
    root=runner.load(output/"roots/s3r24/root.json")
    quality=dict(L_normal=1,L_emergency=0,F_energy_total=0,N_airborne_pending=35,N_admitted_total=39,E_terminal_J=10.,holding_time=2.)
    for arm in runner.ARMS:
        base=output/"evaluation"/arm
        atomic(base/"boundary_0300.json",dict(time=300.,quality=quality))
        atomic(base/"normal_tasks.json",[dict(snapshot_time=280.,effective_time=300.,ready_deadline=299.,ready_time=298.,
            result_time=298.,status="PREPARED",submitted=0,normal_work_s=1.,prepare_work_s=1.,revalidation_work_s=0.,
            revalidation_count=0,revalidations=[])])
        atomic(base/"scheduler_records.json",[])
        atomic(base/"result.json",dict(status="COMPLETE",completed_boundary_s=300,time=300.,quality=quality,
            terminal_observation=dict(observed_time_s=300.,quality=quality),execution_scope=scope,model_fingerprint=scope["model_fingerprint"],
            shared_manifest_hash=root["manifest_hash"],initial_state_hash=root["initial_state_hash"],all_callbacks={}))
    d.evaluation=SimpleNamespace(quality_differences=lambda a,b:{})
    original_module_from=runner.module_from
    report=original_module_from(FROZEN/"scripts/stage3/report_latency_interface.py","_high300_actual_report_fixture")
    # No replay/world is synthesized; the actual collector reports unavailable replay.
    monkeypatch.setattr(report,"saved_core",lambda path:None)
    monkeypatch.setattr(runner,"module_from",lambda path,name:report if Path(path).name=="report_latency_interface.py" else original_module_from(path,name))
    result=runner.saved_report(output,d,scope)
    assert result["common_completed_boundary_s"]==300
    text=(output/"FINAL_REPORT.md").read_text(encoding="utf8")
    assert "300 s H/L pair" in text and "censored at 300 s" in text and "censored at 120 s" not in text
    summary=runner.load(output/"analysis_summary.json")
    assert summary["identity_checks"]["same_manifest"] is True and summary["identity_checks"]["same_initial_state"] is True
    assert (output/"planning_timeline.png").stat().st_size>0 and (output/"planning_timeline.svg").is_file()
    assert not (output/"roots/s3r23").exists() and not (output/"saved_xy_contact.png").exists()
