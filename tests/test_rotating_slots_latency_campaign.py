"""Driver input/serialization/receipt fixtures; no physics, solve or inference."""
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.stage3 import run_latency_interface as runner


def atomic(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False),encoding="utf8")


def digest(value): return sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


class Ledger:
    def __init__(self,path):
        self.path=Path(path)
        self.data=runner.load(path) if self.path.exists() else dict(paused_at="UTC",active_phase=None,jobs={},phase_used={})
    def persist(self): atomic(self.path,self.data)
    def pause(self): self.data["paused_at"]="UTC"; self.persist()
    def remaining(self,phase): return 1800.
    @contextmanager
    def phase(self,name): yield


@contextmanager
def lock(output): yield


@pytest.mark.parametrize("lead_seconds,reserve_seconds",[(20.,0.),(30.,0.),(20.,4.)])
def test_prepare_preserves_fingerprint_and_predeclares_exact_pair(tmp_path,monkeypatch,lead_seconds,reserve_seconds):
    checkout=tmp_path/"checkout"; checkout.mkdir()
    for name in runner.EXTRA_SOURCES:
        p=checkout/name; p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(runner.DRIVER.read_bytes() if name==runner.EXTRA_SOURCES[0] else b"fixture source")
    hashes={name:runner.file_hash(checkout/name) for name in runner.EXTRA_SOURCES}
    frozen=dict(source_commit="primary_new_commit",source_files=hashes,source_hash=digest(hashes))
    freeze=tmp_path/"frozen.json"; atomic(freeze,frozen)
    parent=tmp_path/"model_parent"; (parent/"model").mkdir(parents=True)
    (parent/"model/model.pt").write_bytes(b"unchanged weights")
    atomic(parent/"model/feature_normalizer.json",dict(schema_id="v2"))
    monkeypatch.setattr(runner,"MODEL_HASH",runner.file_hash(parent/"model/model.pt"))
    monkeypatch.setattr(runner,"NORMALIZER_HASH",runner.file_hash(parent/"model/feature_normalizer.json"))
    fp=dict(source_commit="old_training_commit",feature_schema_id="v2",evaluation_protocol_id="original")
    binding=dict(online_v2=True,fingerprint=fp,source_hash="old_training_source")
    atomic(parent/"run_binding.generated.json",binding)
    atomic(parent/"source_freeze.json",dict(source_commit="old_training_commit"))
    atomic(parent/"model/model_card.json",dict(model_frozen=True,model_hash=runner.MODEL_HASH,
        parameter_count=119555,seed=20261023,fingerprint=fp))
    initial=tmp_path/"initial"; original=initial/"roots/s3r23"; original.mkdir(parents=True)
    manifest=dict(seed=7324,rate_per_hour=150,arrival_window=3600.,requests=[dict(release_time=t) for t in range(141)])
    state=dict(aircraft=[dict(status="HOLD_INNER") for _ in range(24)]+[dict(status="LANDED_SERVICE") for _ in range(3)])
    identity={"original":"identity"}
    row=dict(root_id="s3r23",seed=7324,scenario_profile="medium24_rate150_v1",identity=identity,
        manifest_hash=digest(manifest),initial_state_hash=digest(state))
    payload=dict(identity=identity,time=0.,sink=None)
    for name,value in (("root.json",row),("manifest.json",manifest),("initial_state.json",state),
                       ("initial.checkpoint.json",dict(payload=payload,sha256=digest(payload)))): atomic(original/name,value)
    def verify_sources(value):
        assert value["source_hash"]==digest(value["source_files"])
        assert all(runner.file_hash(checkout/k)==v for k,v in value["source_files"].items())
    d=SimpleNamespace(checkout=checkout,atomic_json=atomic,digest=digest,dependency_protocol="affected_traffic_revalidation_v1",handoff_version="search_handoff_reserve_v1",campaign=SimpleNamespace(
        source_files=lambda:[checkout/p for p in runner.EXTRA_SOURCES],verify_sources=verify_sources,
        utc_now=lambda:"UTC",CampaignLedger=Ledger))
    output=tmp_path/"output"
    args=SimpleNamespace(output=output,model_parent=parent,initial_parent=initial,source_freeze=freeze,lead_seconds=lead_seconds,
        search_handoff_reserve_seconds=reserve_seconds)
    assert runner.prepare(args,d)["status"]=="PREPARED"
    scope,bound,_=runner.verify(output,d)
    assert (output/"run_binding.generated.json").read_bytes()==(parent/"run_binding.generated.json").read_bytes()
    assert bound["fingerprint"]==fp and scope["training_source_hash"]=="old_training_source"
    assert scope["source_commit"]=="primary_new_commit"
    p=scope["protocol"]
    assert p["arms"]==list(runner.ARMS) and p["formal_jobs"]==2 and p["target_sim_seconds"]==120
    assert p["dependency_protocol"]=="affected_traffic_revalidation_v1"
    assert (p["interface"],p["ranking_scope"])==(runner.INTERFACE,runner.RANKING)
    assert p["search_handoff_reserve_seconds"]==reserve_seconds
    assert p.get("search_handoff_protocol")==("search_handoff_reserve_v1" if reserve_seconds else None)
    assert (p["cpu"],p["lead_seconds"],p["preparation_margin"],p["job_wall_cap_s"],p["total_evaluation_wall_cap_s"])==(12,lead_seconds,1.,900.,1800.)
    for field,value,message in (("lead_seconds",25.,"lead seconds"),("cp_seconds",9.,"execution scope changed"),
                                ("lead_seconds",50.-lead_seconds,"prepared scope identity changed")):
        changed=deepcopy(scope); changed["protocol"][field]=value
        changed["scope_id"]=digest({k:v for k,v in changed.items() if k not in ("scope_id","prepared_file_sha256")})
        atomic(output/"execution_scope.json",changed)
        with pytest.raises(ValueError,match=message): runner.verify(output,d)
    atomic(output/"execution_scope.json",scope)
    assert runner.verify(output,d)[0]["protocol"]["lead_seconds"]==lead_seconds
    forged=deepcopy(scope)
    forged["protocol"]=runner.protocol(d.dependency_protocol,lead_seconds,4.-reserve_seconds,d.handoff_version)
    forged["scope_id"]=digest({k:v for k,v in forged.items() if k not in ("scope_id","prepared_file_sha256")})
    atomic(output/"execution_scope.json",forged)
    with pytest.raises(ValueError,match="prepared scope identity changed"):runner.verify(output,d)
    atomic(output/"execution_scope.json",scope)
    if reserve_seconds==0.:
        legacy=deepcopy(scope);legacy["protocol"].pop("search_handoff_reserve_seconds")
        legacy["scope_id"]=digest({k:v for k,v in legacy.items() if k not in ("scope_id","prepared_file_sha256")})
        atomic(output/"execution_scope.json",legacy)
        atomic(output/"prepare.finished.json",dict(status="COMPLETED",scope_id=legacy["scope_id"]))
        assert "search_handoff_reserve_seconds" not in runner.verify(output,d)[0]["protocol"]
        atomic(output/"execution_scope.json",scope)
        atomic(output/"prepare.finished.json",dict(status="COMPLETED",scope_id=scope["scope_id"]))
    args.output=tmp_path/"invalid_output"; args.lead_seconds=0.
    with pytest.raises(ValueError,match="lead seconds"): runner.prepare(args,d)
    assert not args.output.exists()
    args.output=output; args.lead_seconds=lead_seconds
    with pytest.raises(ValueError,match="fresh output"): runner.prepare(args,d)


class Scheduler:
    def __init__(self):
        self.records=[]; self.provider=SimpleNamespace(previous={},advisor_runtime=SimpleNamespace(ranking_scope="legacy"))


class Wrapper:
    def __init__(self,scheduler,**options):
        assert options["lead_seconds"] in (20.,30.) and options["preparation_margin"]==1.
        self.options=options
        self.callback=scheduler; self.records=[]; self.online_services=[]; self.pending=None; self.dirty=False
    def planning_event(self,port): pass
    def __call__(self,*args): pass


class World:
    def __init__(self,fail_at=None,lead_seconds=30.):
        self.time=0.; self.failed=False; self.terminated_invalid=False; self.dispatch_timings=[]
        self.scheduler_period=5.; self.config=SimpleNamespace(path="fixture.yaml"); self.fail_at=fail_at; self.steps=[]
        self.lead_seconds=lead_seconds
    def configure_practical_runtime(self,value): assert value is True
    def advance(self,boundary,wrapper,*,settle_only_at_end):
        assert callable(wrapper.planning_event) and self.decision_callback is wrapper and settle_only_at_end is True
        assert wrapper.options==dict(lead_seconds=self.lead_seconds,preparation_margin=1.,search_handoff_reserve_seconds=self.reserve_seconds)
        assert wrapper.callback.provider.advisor_runtime.ranking_scope==runner.RANKING
        self.steps.append(boundary); self.time=boundary
        if boundary==self.fail_at: raise RuntimeError("fixture failure")
        self.dispatch_timings.append(dict(time=boundary,dispatch_wall_s=.001,normal_wall_s=.0005))
    def to_dict(self): return dict(states=[dict(time=0.),dict(time=self.time)],events=[],trajectories={})


def job_fixture(tmp_path,monkeypatch,fail_at=None,lead_seconds=30.,reserve_seconds=0.):
    output=tmp_path/"output"; output.mkdir()
    atomic(output/"roots/s3r23/root.json",dict(identity={"original":"id"},manifest_hash="manifest",initial_state_hash="state"))
    world=World(fail_at,lead_seconds);world.reserve_seconds=reserve_seconds; scheduler=Scheduler(); saved=[]; restored=[]
    scope=dict(source_hash="new_source",source_commit="new_commit",scope_id="scope",
        protocol=runner.protocol("affected_traffic_revalidation_v1",lead_seconds,reserve_seconds,"search_handoff_reserve_v1"))
    binding=dict(fingerprint={"old":"fingerprint"})
    atomic(output/"run.started.json",dict(scope_id="scope"))
    ledger=SimpleNamespace(data=dict(active_phase=dict(phase="evaluation"),
        jobs={arm:dict(status="running",timeout_s=900.) for arm in runner.ARMS}))
    monkeypatch.setattr(runner,"verify",lambda *args:(scope,binding,ledger))
    monkeypatch.setattr(runner.os,"sched_setaffinity",lambda *args:None,raising=False)
    def checkpoint(path,w,wrapper,identity,**kwargs):
        assert isinstance(wrapper,Wrapper)
        saved.append(w.time); atomic(path,dict(time=w.time,identity=identity))
    class Replay:
        @staticmethod
        def write(path,**kwargs):
            atomic(Path(path)/"replay.json",dict(code_commit=kwargs["code_commit"]))
    quality=lambda w:dict(L_normal=1,N_airborne_pending=2,time=w.time)
    d=SimpleNamespace(checkout=tmp_path,restore_root=lambda *args:world,NormalSchedulerCallback=lambda *a,**k:object(),
        OnlineObserver=lambda:object(),LatencyAwareScheduler=Wrapper,shared_fifo_admission=object(),
        save_checkpoint=checkpoint,atomic_json=atomic,ReplayBundle=Replay,episode_metrics=quality,
        campaign=SimpleNamespace(utc_now=lambda:"UTC"),
        evaluation=SimpleNamespace(_scheduler=lambda *args:scheduler,
            _costs=lambda s,o,wall,cpu,dispatch,**kwargs:dict(full_dispatch_wall_s=dispatch,full_online_wall_s=wall),
            observe_executed_holding=lambda w:({},lambda:restored.append(True)),
            terminal_observation=lambda w:dict(status="OBSERVED",observed_time_s=w.time,quality=quality(w))),
        aggregate=lambda *args:dict(actual_callbacks=len(scheduler.records)))
    return output,d,world,saved,restored


@pytest.mark.parametrize("lead_seconds,reserve_seconds",[(20.,0.),(30.,0.),(20.,4.)])
def test_job_serialization_uses_wrapper_and_0_60_120_checkpoints(tmp_path,monkeypatch,lead_seconds,reserve_seconds):
    output,d,w,saved,restored=job_fixture(tmp_path,monkeypatch,lead_seconds=lead_seconds,reserve_seconds=reserve_seconds)
    result=runner.run_job(output,runner.ARMS[0],d)
    base=output/"evaluation"/runner.ARMS[0]
    assert w.steps==list(range(5,121,5)) and saved==[0.,60.,120.]
    assert len(list(base.glob("boundary_*.json")))==24
    assert result["status"]=="COMPLETE" and result["terminal_observation"]["observed_time_s"]==120.
    assert runner.load(base/"checkpoint_0120.json")["identity"]["source_hash"]=="new_source"
    assert result["model_fingerprint"]=={"old":"fingerprint"} and restored==[True]
    assert runner.load(base/"normal_tasks.json")==[]
    assert runner.load(base/"replay/replay.json")["code_commit"]=="new_commit"


def test_caught_failure_keeps_independent_terminal_and_real_last_boundary(tmp_path,monkeypatch):
    output,d,w,saved,restored=job_fixture(tmp_path,monkeypatch,fail_at=35.)
    with pytest.raises(RuntimeError,match="preserved partial evidence"): runner.run_job(output,runner.ARMS[1],d)
    base=output/"evaluation"/runner.ARMS[1]; result=runner.load(base/"result.json")
    assert result["time"]==30. and result["completed_boundary_s"]==30
    assert result["terminal_observation"]["observed_time_s"]==35.
    assert result["status"]=="EVALUATION_PARTIAL" and result["replay_status"]=="WRITTEN_HUMAN_PENDING"
    assert saved==[0.,35.] and restored==[True]
    assert not (base/"boundary_0035.json").exists()


@pytest.mark.parametrize("stage",["run","job","summarize"])
def test_lead_cli_cannot_override_existing_scope(stage,tmp_path,monkeypatch):
    def forbidden(*args): raise AssertionError("CLI override imported project dependencies")
    monkeypatch.setattr(runner,"dependencies",forbidden)
    with pytest.raises(SystemExit) as error:
        runner.main([stage,"--checkout",str(tmp_path),"--lead-seconds","20"])
    assert error.value.code==2


def test_lead_default_and_invalid_prepare_cli(tmp_path,monkeypatch):
    assert runner.protocol("affected_traffic_revalidation_v1")==runner.protocol("affected_traffic_revalidation_v1",30.)
    def forbidden(*args): raise AssertionError("invalid lead imported project dependencies")
    monkeypatch.setattr(runner,"dependencies",forbidden)
    with pytest.raises(SystemExit) as error:
        runner.main(["prepare","--checkout",str(tmp_path),"--source-freeze","unused.json","--lead-seconds","25"])
    assert error.value.code==2


def test_lifecycle_rates_exclude_pending_and_no_prefix(tmp_path,monkeypatch):
    tasks=[dict(snapshot_time=0.,status="COMMITTED",submitted=2,finished_time=30.,normal_work_s=10.,prepare_work_s=2.,ready_guard_work_s=.2,
                ready_registration_bookkeeping_s=.1,publication_bookkeeping_wall_s=.1,normal_evidence_wall_s=.1),
           dict(snapshot_time=5.,status="REJECTED",submitted=0,finished_time=34.,reason="READY_DEADLINE_MISSED"),
           dict(snapshot_time=10.,status="PREPARED",submitted=0,normal_work_s=20.),
           dict(snapshot_time=15.,status="NO_EXECUTABLE_PREFIX",submitted=0,finished_time=20.,reason="NO_EXECUTABLE_PREFIX")]
    callbacks=[dict(planning_timing=dict(snapshot_time=t),deferred=True,noop=t==15.) for t in (0.,5.,10.,15.)]
    value=runner.lifecycle_summary(tasks,callbacks,[dict(kind="monitor",wall_s=1.,cpu_s=.5)])
    assert value["nonempty_proposed_groups"]==3 and value["finished_nonempty_groups"]==2
    assert value["pending_nonempty_groups"]==1 and value["committed_nonempty_groups"]==1 and value["join_rate"]==.5
    assert value["no_executable_prefix_groups"]==1 and value["ready_deadline_rejections"]==1
    assert value["timings"]["prepare_work_s"]["total_s"]==2.
    assert value["full_normal_planning_work"]["total_s"]==pytest.approx(32.3)
    assert value["full_normal_planning_work"]["p95_s"]==20.
    assert value["finished_task_full_planning_work"]["total_s"]==pytest.approx(12.3)
    assert value["finished_task_full_planning_work"]["p95_s"]==pytest.approx(12.3)
    assert value["per_task_work"][0]["full_normal_planning_work_s"]==pytest.approx(12.3)
    assert value["old_5s_full_planning_overrun_groups"]==2
    atomic(tmp_path/"summary.json",value)
    assert runner.load(tmp_path/"summary.json")["online_service_by_kind"]["monitor"]["wall"]["total_s"]==1.
    output=tmp_path/"saved_pair"; output.mkdir()
    for arm in runner.ARMS:
        atomic(output/"evaluation"/arm/"boundary_0005.json",dict(time=5.,quality={"normal":1}))
        atomic(output/"evaluation"/arm/"result.json",dict(status="EVALUATION_PARTIAL",normal_lifecycle=value))
    atomic(output/"evaluation"/runner.ARMS[0]/"boundary_0010.json",dict(time=10.,quality={"normal":2}))
    ledger=SimpleNamespace(data=dict(paused_at="UTC",active_phase=None))
    monkeypatch.setattr(runner,"verify",lambda *args:({"source_commit":"new_commit"},{},ledger))
    d=SimpleNamespace(atomic_json=atomic,evaluation=SimpleNamespace(quality_differences=lambda a,b:dict(equal=a==b)))
    assert runner.summarize(output,d)["common_completed_boundary_s"]==5
    summary=runner.load(output/"summary.json")
    assert summary["independent_results"][runner.ARMS[1]]["normal_lifecycle"]["pending_nonempty_groups"]==1
    assert summary["quality_differences"]==dict(equal=True) and (output/"README.md").is_file()


@pytest.mark.parametrize("timeout",[False,True])
def test_parent_exact_order_caps_and_hard_timeout_output(tmp_path,monkeypatch,timeout):
    output=tmp_path/"output"; output.mkdir(); ledger=Ledger(output/"ledger.json"); calls=[]
    monkeypatch.setattr(runner,"verify",lambda *args:({"scope_id":"fixed"},{},ledger))
    def child(ledger,phase,arm,command,directory,*,cap):
        calls.append((arm,cap,command))
        base=output/"evaluation"/arm
        if timeout:
            atomic(base/"boundary_0030.json",dict(time=30.,quality={"actual":"boundary only"}))
            return dict(status="BUDGET_LIMITED",timeout_s=900.,exit_code=-9)
        atomic(base/"result.json",dict(status="COMPLETE"))
        return dict(status="completed",timeout_s=900.,exit_code=0)
    d=SimpleNamespace(checkout=tmp_path,atomic_json=atomic,campaign=SimpleNamespace(campaign_lock=lock,
        resource_probe=lambda:dict(free_physical_cores=[12]),utc_now=lambda:"UTC",run_subprocess=child))
    if timeout:
        with pytest.raises(RuntimeError,match="no automatic retry"): runner.run(output,d)
        assert len(calls)==1
        result=runner.load(output/"evaluation"/runner.ARMS[0]/"result.json")
        assert result["time"] is None and result["completed_boundary_s"]==30.
        assert result["terminal_observation"]==dict(status="UNAVAILABLE_HARD_TIMEOUT",observed_time_s=None)
        assert result["replay_status"]=="UNAVAILABLE_HARD_TIMEOUT"
    else:
        assert runner.run(output,d)["status"]=="COMPLETED"
        assert [r[0] for r in calls]==list(runner.ARMS)
        assert all(r[1]==900. and r[2][-2:]==["--arm",r[0]] for r in calls)
    assert ledger.data["paused_at"]=="UTC"
    assert runner.load(output/"run.finished.json")["no_automatic_retry"] is True


def test_revalidation_work_once_multiple_joins_and_censored_round():
    event=dict(started_time=20.,due=21.,completed_time=21.,affected_aircraft_ids=["new-a"],
        pairs_checked=3,reused_pairs=7,full_fallback=False,reason="NEW_TRAFFIC",status="CERTIFIED",wall_s=1.,cpu_s=.8)
    tasks=[dict(snapshot_time=0.,status="COMMITTED",submitted=13,finished_time=30.,normal_work_s=4.,
                prepare_work_s=2.,revalidation_work_s=1.,revalidation_work_cpu_s=.8,dependency_guard_work_s=.3,
                dependency_guard_work_cpu_s=.25,ready_registration_bookkeeping_cpu_s=.02,revalidation_count=1,revalidations=[event]),
           dict(snapshot_time=35.,status="COMMITTED",submitted=2,finished_time=65.,revalidation_count=0,revalidations=[]),
           dict(snapshot_time=90.,status="REVALIDATING",submitted=0,revalidation_work_s=.4,
                revalidation_count=1,revalidations=[dict(event,started_time=119.,due=120.,status="GUARDING",geometry_completed_time=119.5,guard_delivery_due=120.)])]
    del tasks[-1]["revalidations"][0]["completed_time"]
    callbacks=[dict(planning_timing=dict(snapshot_time=r["snapshot_time"]),deferred=True,noop=False) for r in tasks]
    value=runner.lifecycle_summary(tasks,callbacks,[dict(kind="dependency_guard",wall_s=.3,cpu_s=.25)])
    assert value["full_normal_planning_work"]["total_s"]==pytest.approx(7.7)
    assert value["timings"]["dependency_guard_work_s"]["total_s"]==pytest.approx(.3)
    assert value["timings"]["dependency_guard_work_cpu_s"]["total_s"]==pytest.approx(.25)
    assert value["timings"]["ready_registration_bookkeeping_cpu_s"]["total_s"]==pytest.approx(.02)
    assert value["actual_joined_groups"]==2 and value["actual_submitted_plans"]==15
    assert value["submitted_plans_by_joined_group"]==[13,2]
    assert value["join_rate"]==1. and value["pending_nonempty_groups"]==1
    reval=value["revalidation"]
    assert reval["recorded_rounds"]==2 and reval["certified_rounds"]==1 and reval["pending_rounds"]==1
    assert reval["affected_aircraft_ids"]==["new-a"] and reval["reused_pairs"]==14
    assert runner.revalidation_summary([dict(snapshot_time=0.)])["declared_count"] is None
    assert runner.revalidation_summary([dict(snapshot_time=0.)])["availability"]=="UNAVAILABLE_LEGACY_FIELDS"
    rejection=runner.lifecycle_summary([dict(snapshot_time=i,status="REJECTED",finished_time=30.,reason=reason)
        for i,reason in enumerate(("READY_DEADLINE_MISSED","TRAFFIC_CHANGED_AFTER_DEADLINE","EXTERNAL_TRAFFIC_MISSING","REVALIDATION_WORK_DEADLINE"))],[],[])
    assert rejection["ready_deadline_rejections"]==1 and rejection["traffic_changed_after_deadline_rejections"]==1
    assert rejection["external_traffic_missing_rejections"]==1
    assert rejection["revalidation_work_deadline_rejections"]==1
    assert runner.search_reuse_summary([{}])["cp"]["pair_cuts_reused"] is None


def test_search_handoff_reserve_cannot_override_frozen_scope_from_cli():
    for stage in ("run","job","summarize"):
        with pytest.raises(SystemExit) as error:
            runner.main([stage,"--checkout","unused","--search-handoff-reserve-seconds","4"])
        assert error.value.code==2
