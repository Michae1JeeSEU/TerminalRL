"""Fixed-state driver fixtures only: no simulation, solve or model inference."""
from hashlib import sha256
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.stage3 import probe_search_handoff as probe


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False),encoding="utf8")


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


class Ledger:
    def __init__(self,path):
        self.path=Path(path)
        self.data=probe.runner.load(path) if self.path.exists() else dict(jobs={})
    def persist(self):atomic(self.path,self.data)
    def pause(self):self.data["paused_at"]="UTC";self.persist()


def test_single_task_forwards_pending_events_services_and_suppresses_only_later_starts():
    class Inner:
        def __init__(self):self.pending=None;self.calls=0;self.services=[];self.events=0;self.normal_starts=0
        def __call__(self,*args):
            self.calls+=1
            if self.pending is None:self.normal_starts+=1;self.pending={"stage":"RESULT"}
        def planning_event(self,*args):self.events+=1;self.pending=None
        def online_service(self,*args):self.services.append(args)
        def inspect_dependencies(self,port):self.services.append(("inspect",port))
    inner=Inner();gate=probe.SingleTask(inner)
    gate("t0");gate("coalesced")
    gate.online_service(5.,.1,.1,"monitor");gate.inspect_dependencies("port")
    gate.planning_event("port");gate("would restart");gate.online_service(20.,.2,.1,"emergency")
    assert inner.normal_starts==1 and inner.calls==2 and inner.events==1
    assert len(inner.services)==3 and gate.pending is None and gate.suppressed_starts==1


def test_prepare_freezes_original_identity_model_and_exact_probe_protocol(tmp_path,monkeypatch):
    parent=tmp_path/"model_parent";(parent/"model").mkdir(parents=True)
    for name in ("model.pt","feature_normalizer.json","model_card.json"):(parent/"model"/name).write_bytes(b"unchanged")
    atomic(parent/"run_binding.generated.json",dict(fingerprint={"original":"fingerprint"}))
    initial=tmp_path/"initial";root=initial/"roots/s3r23";root.mkdir(parents=True)
    manifest=dict(seed=7324,rate_per_hour=150,arrival_window=3600.,requests=[{"release_time":100.+i} for i in range(141)])
    state=dict(aircraft=[dict(status="HOLD_OUTER") for _ in range(24)]+[dict(status="LANDED_SERVICE") for _ in range(3)])
    identity=dict(original="checkpoint-identity")
    row=dict(root_id="s3r23",seed=7324,scenario_profile="medium24_rate150_v1",identity=identity,
        manifest_hash=digest(manifest),initial_state_hash=digest(state))
    payload=dict(identity=identity,time=0.,sink=None)
    for name,value in (("root.json",row),("manifest.json",manifest),("initial_state.json",state),
        ("initial.checkpoint.json",dict(payload=payload,sha256=digest(payload)))):atomic(root/name,value)
    freeze=tmp_path/"freeze.json"
    atomic(freeze,dict(source_commit="execution",source_hash="source",source_files={
        "scripts/stage3/probe_search_handoff.py":probe.runner.file_hash(probe.DRIVER)}))
    monkeypatch.setattr(probe.runner,"verify_source",lambda *args:None)
    binding=dict(fingerprint={"original":"fingerprint"})
    monkeypatch.setattr(probe.runner,"verify_model",lambda *args:binding)
    d=SimpleNamespace(checkout=tmp_path,digest=digest,atomic_json=atomic,dependency_protocol="affected_traffic_revalidation_v1",
        handoff_version="search_handoff_reserve_v1",campaign=SimpleNamespace(utc_now=lambda:"UTC",CampaignLedger=Ledger))
    output=tmp_path/"probe"
    args=SimpleNamespace(output=output,model_parent=parent,initial_parent=initial,source_freeze=freeze)
    before={p:p.read_bytes() for base in (parent,initial) for p in base.rglob("*") if p.is_file()}
    assert probe.prepare(args,d)["status"]=="PREPARED"
    scope,bound,_=probe.verify(output,d)
    assert scope["initial_identity"]==identity and bound["fingerprint"]==binding["fingerprint"]
    assert scope["protocol"]["target_sim_seconds"]==20.001 and scope["protocol"]["normal_callbacks_per_arm"]==1
    assert scope["protocol"]["lead_seconds"]==20. and scope["protocol"]["search_handoff_reserve_seconds"]==4.
    assert (scope["protocol"]["job_wall_cap_s"],scope["protocol"]["total_evaluation_wall_cap_s"])==(90.,180.)
    assert scope["protocol"]["observation_boundaries_s"]==[5.,10.,15.,20.001]
    assert "checkpoint_seconds" not in scope["protocol"] and "120" not in scope["protocol"]["endpoint_policy"]
    assert all(path.read_bytes()==data for path,data in before.items())
    assert (output/"roots/s3r23/manifest.json").read_bytes()==(root/"manifest.json").read_bytes()
    with pytest.raises(ValueError,match="fresh fixed-state"):probe.prepare(args,d)


@pytest.mark.parametrize("fail",[False,True])
def test_job_runs_one_callback_preserves_services_and_reads_back_partial_evidence(tmp_path,monkeypatch,fail):
    output=tmp_path/"probe";output.mkdir()
    atomic(output/"run.started.json",dict(scope_id="scope"))
    atomic(output/"roots/s3r23/root.json",dict(identity={"original":"identity"}))
    scheduler=SimpleNamespace(records=[],provider=SimpleNamespace(advisor_runtime=SimpleNamespace()))
    options=[]
    class Inner:
        def __init__(self,callback,**kwargs):
            options.append(kwargs);self.callback=callback;self.pending=None;self.records=[];self.online_services=[]
        def __call__(self,observation,port):
            if self.pending is not None:return
            assert port.time==0.
            handoff=dict(first_nonempty_certified=dict(wall_elapsed_s=2.),callback_return=dict(wall_elapsed_s=14.,certified=True))
            self.callback.records.append(dict(planning_timing=dict(snapshot_time=0.),search_handoff=handoff))
            self.records.append(dict(snapshot_time=0.,status="COMPUTING",search_handoff=handoff,submitted=0))
            self.pending={"stage":"RESULT"}
        def planning_event(self,port):
            self.records[0].update(preparation_started_time=15.,finished_time=20.,status="COMMITTED",submitted=1)
            self.pending=None
        def online_service(self,*args):self.online_services.append(args)
    class World:
        time=0.;scheduler_period=5.;failed=False;terminated_invalid=False
        def configure_practical_runtime(self,value):assert value is True
        def advance(self,boundary,gate,*,settle_only_at_end):
            assert self.decision_callback is gate and self.admission_callback is admission and settle_only_at_end
            gate(None,SimpleNamespace(time=self.time))
            gate.online_service(self.time,.1,.05,"monitor")
            if fail: self.time=2.;raise RuntimeError("fixture interruption")
            self.time=boundary
            if boundary>=20.:gate.planning_event(SimpleNamespace(time=boundary))
            if boundary==probe.END:gate(None,SimpleNamespace(time=boundary))
        def to_dict(self):
            return dict(states=[dict(time=0.,aircraft=[dict(id="a",p=[1.,0.,0.],v=[0.,1.,0.],a=[-1.,0.,0.],soc=.8)]),
                dict(time=self.time,aircraft=[dict(id="a",p=[1.,0.,0.],v=[0.,1.,0.],a=[-1.,0.,0.],soc=.79)])],
                events=[] if fail else [dict(type="maneuver_started",time=20.,aircraft_id="a",details=dict(trajectory_hash="joined"))],
                trajectories={"joined":dict(kind="TRANSFER")})
    world=World();admission=object()
    scope=dict(scope_id="scope",model_fingerprint={"original":"fp"})
    binding=dict(fingerprint=scope["model_fingerprint"])
    ledger=SimpleNamespace(data=dict(active_phase=dict(phase="evaluation"),jobs={probe.runner.ARMS[0]:dict(status="running",timeout_s=90.)}))
    monkeypatch.setattr(probe,"verify",lambda *args:(scope,binding,ledger))
    monkeypatch.setattr(probe.os,"sched_setaffinity",lambda *args:None,raising=False)
    d=SimpleNamespace(restore_root=lambda *args:world,NormalSchedulerCallback=lambda *a,**k:object(),OnlineObserver=lambda:object(),
        evaluation=SimpleNamespace(_scheduler=lambda *args:scheduler,terminal_observation=lambda w:dict(status="OBSERVED",observed_time_s=w.time)),
        LatencyAwareScheduler=Inner,shared_fifo_admission=admission,atomic_json=atomic,campaign=SimpleNamespace(utc_now=lambda:"UTC"))
    if fail:
        with pytest.raises(RuntimeError,match="partial evidence preserved"):probe.job(output,probe.runner.ARMS[0],d)
    else:probe.job(output,probe.runner.ARMS[0],d)
    base=output/"evaluation"/probe.runner.ARMS[0]
    result=probe.runner.load(base/"result.json")
    assert options==[dict(lead_seconds=20.,preparation_margin=1.,search_handoff_reserve_seconds=4.)]
    assert result["callback_count"]==1 and len(probe.runner.load(base/"normal_tasks.json"))==1
    assert probe.runner.load(base/"online_services.json")
    assert result["status"]==("PARTIAL_OBSERVATION" if fail else "COMPLETE_OBSERVATION")
    assert result["terminal_observation"]["observed_time_s"]==(2. if fail else probe.END)
    assert result["task_facts"][0]["first_nonempty_certified_observed"] is True
    assert result["task_facts"][0]["actual_join_observed"] is (not fail)
    assert result["task_facts"][0]["preparation_started_observed"] is (not fail)
    record=probe.runner.load(base/"world_record.json")
    assert record["states"][-1]["aircraft"][0]["soc"]==.79
    assert len(result["observed_maneuver_start_events"])==(0 if fail else 1)


@pytest.mark.parametrize("timeout",[False,True])
def test_parent_probe_serial_order_fixed_caps_and_hard_timeout_receipt(tmp_path,monkeypatch,timeout):
    output=tmp_path/"probe";output.mkdir();calls=[]
    @contextmanager
    def phase(name):yield
    @contextmanager
    def lock(path):yield
    ledger=SimpleNamespace(data=dict(jobs={}),phase=phase,remaining=lambda name:180.,pause=lambda:None)
    monkeypatch.setattr(probe,"verify",lambda *args:({"scope_id":"scope"},{},ledger))
    def child(ledger,phase,arm,command,directory,*,cap):
        calls.append((arm,cap,command))
        if timeout:return dict(status="BUDGET_LIMITED",exit_code=-9,timeout_s=90.)
        atomic(output/"evaluation"/arm/"result.json",dict(status="COMPLETE_OBSERVATION"))
        return dict(status="completed",exit_code=0,timeout_s=90.)
    d=SimpleNamespace(checkout=tmp_path,atomic_json=atomic,campaign=SimpleNamespace(campaign_lock=lock,
        resource_probe=lambda:dict(free_physical_cores=[12]),utc_now=lambda:"UTC",run_subprocess=child))
    if timeout:
        with pytest.raises(RuntimeError,match="return to primary"):probe.run(output,d)
        assert len(calls)==1
        assert probe.runner.load(output/"evaluation"/probe.runner.ARMS[0]/"watchdog_partial.json")["status"]=="BUDGET_LIMITED"
    else:
        assert probe.run(output,d)["status"]=="COMPLETED"
        assert [row[0] for row in calls]==list(probe.runner.ARMS)
        assert all(row[1]==90. and row[2][-2:]==["--arm",row[0]] for row in calls)
    assert probe.runner.load(output/"run.finished.json")["status"]==("INCOMPLETE_RETURN_TO_PRIMARY" if timeout else "COMPLETED")
