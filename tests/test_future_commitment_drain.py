"""Small checkpoint/control fixtures; no normal solver or campaign is run."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts.stage3 import run_future_commitment_drain as drain
from terminal_marl.core.rotating_slots import stage2_campaign as archive
from terminal_marl.core.rotating_slots.demand import ArrivalManifest, ArrivalRequest
from terminal_marl.core.rotating_slots.planning_latency import LatencyAwareScheduler
from terminal_marl.core.rotating_slots.replay import BufferedStateJSONLSink, ReplayBundle
from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID


def dependencies():
    return SimpleNamespace(NormalSchedulerCallback=NormalSchedulerCallback,
        LatencyAwareScheduler=LatencyAwareScheduler, restore_checkpoint=archive.restore_checkpoint)


def request(key, t):
    return ArrivalRequest(key, t, 0, .8, "standard")


def serialized_fixture(tmp_path, *, pending=True, streamed=True):
    directory = tmp_path / "original"
    directory.mkdir()
    sink = BufferedStateJSONLSink(directory / "states.jsonl") if streamed else None
    world = RotatingSlotWorld(manifest=ArrivalManifest((request("queued", 319.),request("future",601.)),7324,150.,3600.),
        state_sink=sink,experiment_policy_id=STAGE2_POLICY_ID)
    world.auto_emergency=False
    world.seed_hold("a","middle",0,.8)
    world.time = 320.
    world._manifest_index=1
    world.requests={"queued":world.manifest.requests[0]}
    world._q_cache[("persisted",320.)]=.25
    callback=drain.common.common_callback(dependencies(),"FULL_H",evaluation_profile=drain.campaign.PROFILE)
    callback.initialize_imported_history(world.build_planning_snapshot(),events=world.events)
    plan=world._base(world.aircraft["a"])
    callback.provider.previous={"a":(plan,)}
    callback.provider.references.templates={("fixture",):plan}
    callback.provider.references.hits=7
    wrapper=LatencyAwareScheduler(callback,lead_seconds=20.,preparation_margin=1.,
        search_handoff_reserve_seconds=4.,restart_policy=drain.common.RESTART)
    if pending:
        wrapper.records=[dict(status="PREPARED",snapshot_time=300.,submitted=0)]
        wrapper.pending=dict(stage="PREPARED",token="pending_fixture",record_index=0,due=334.5)
        world._planning_latency_packets={"pending_fixture":dict(absolute_time=334.5,certificate=("exact",320.))}
        world._future_reservations={"future_fixture":dict(aircraft_id="a",state="RESERVED",start_time=345.75,
            end_time=400.,trajectory_hash=plan.trajectory_hash,plan=plan)}
    world._record(force=True)
    identity=archive.checkpoint_identity(dict(source_hash="original",base_config_hash=world.config.config_hash,
        overlay_sha256="fixture",planning_profile="downstream_service_v8"),archive.digest(world.manifest.to_dict()))
    path=directory/"checkpoint_terminal.json"
    archive.save_checkpoint(path,world,wrapper,identity)
    if sink: sink.close()
    payload=drain.base.load(path)["payload"]
    metadata=dict(common_summary=dict(baseline=dict(event_count=0,dispatch_count=0)),initializer_tasks=[],
                  initializer_callback_records=[],initializer_services=[],t0=20.)
    return SimpleNamespace(world=world,wrapper=wrapper,callback=callback,path=path,payload=payload,metadata=metadata)


@pytest.mark.parametrize("streamed",[True,False])
def test_real_restore_retains_pending_absolute_reservations_persisted_caches_and_source(tmp_path,streamed):
    f=serialized_fixture(tmp_path,streamed=streamed)
    before={p:p.read_bytes() for p in f.path.parent.iterdir() if p.is_file()}
    world,callback,wrapper,checks=drain.restore(f.path,f.payload,"FULL_H",tmp_path/"copy.jsonl",d=dependencies())
    try:
        assert checks["status"]=="PASS" and checks["physical_advance_calls"]==checks["solver_calls"]==0
        assert wrapper.pending==f.wrapper.pending and wrapper.pending["due"]==334.5
        assert world._future_reservations["future_fixture"]["start_time"]==345.75
        assert world._q_cache==f.world._q_cache and callback.provider.references.hits==7
        assert callback.state_dict()==f.callback.state_dict()
        assert callback.provider.validation_cache=={}
        if streamed:
            assert world.state_sink.path==tmp_path/"copy.jsonl"
            world.state_sink.flush()
        else:
            assert world.state_sink is None and world.records==f.world.records
    finally:
        if world.state_sink: world.state_sink.close()
    assert {p:p.read_bytes() for p in before}==before


def test_cutoff_settles_due_arrivals_keeps_queue_and_stops_future_release_and_admission(tmp_path):
    world=RotatingSlotWorld(manifest=ArrivalManifest((request("queued",590.),request("exact",600.),request("later",600.1)),
        7324,150.,3600.),experiment_policy_id=STAGE2_POLICY_ID)
    world.time=599.8; world.auto_emergency=False; world._manifest_index=1
    world.requests={"queued":world.manifest.requests[0]}
    admissions=[]
    world.admission_callback=lambda snapshot,port: admissions.append(port.time)
    world.advance(600.,False,settle_only_at_end=True)
    assert set(world.requests)=={"queued","exact"} and 600. not in admissions
    identity,receipt=drain.apply_cutoff(world,{"manifest_sha256":"original"})
    assert receipt["excluded_upstream_ids"]==["exact","queued"]
    assert receipt["suppressed_request_ids"]==["later"]
    assert receipt["cohort_airborne_ids"]==[] and world.manifest.arrival_window==600.
    assert identity["manifest_sha256"]==archive.digest(world.manifest.to_dict())
    before=len(world.events); count=len(admissions)
    world.advance(605.,False,settle_only_at_end=True)
    assert len(admissions)==count and set(world.requests)=={"queued","exact"}
    assert not any(e["type"] in ("request_released","admitted") for e in world.events[before:])
    assert drain.completion(world,receipt)=="COMPLETE"


def test_cutoff_checkpoint_restore_does_not_reenable_admission(tmp_path):
    f=serialized_fixture(tmp_path,pending=False)
    world,callback,wrapper,_=drain.restore(f.path,f.payload,"FULL_H",tmp_path/"first.jsonl",d=dependencies())
    world.time=600.
    world._record(force=True)
    identity,receipt=drain.apply_cutoff(world,f.payload["identity"])
    path=tmp_path/"cutoff.json"
    archive.save_checkpoint(path,world,wrapper,identity,boundary_summary=dict(cutoff=receipt))
    world.state_sink.close()
    before=(tmp_path/"first.jsonl").read_bytes()
    restored,_,_,checks=drain.restore(path,drain.base.load(path)["payload"],"FULL_H",tmp_path/"second.jsonl",
                                     cutoff=receipt,d=dependencies())
    assert checks["status"]=="PASS" and restored.admission_callback is None
    assert restored.requests.keys()==world.requests.keys() and restored.manifest.requests==world.manifest.requests
    restored.state_sink.close()
    assert (tmp_path/"first.jsonl").read_bytes()==before


@pytest.mark.parametrize("kind,expected",[("normal","NORMAL_LANDING"),("emergency","EMERGENCY_LANDING"),
    ("battery","BATTERY_DEPLETED"),("peak","PEAK_BUDGET_EXHAUSTED")])
def test_terminal_outcomes_are_explicit_and_ground_service_is_resolved(kind,expected):
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    aircraft=world.seed_hold("a","inner",0,.8)
    if kind in ("normal","emergency"):
        aircraft=replace(aircraft,status="LANDED_SERVICE",touchdown_time=601.25,service_end=721.25,
                         emergency=replace(aircraft.emergency,charged=kind=="emergency"))
    else:
        aircraft=replace(aircraft,status="ENERGY_MISSION_FAILED",soc=0. if kind=="battery" else .3)
        world.failure_archive["a"]=dict(time=601.25,reason=expected)
    world.aircraft["a"]=aircraft
    receipt=dict(cohort_airborne_ids=["a"],admitted_ids=["a"])
    assert drain.aircraft_outcome(world,aircraft)["outcome"]==expected
    assert drain.completion(world,receipt)=="COMPLETE"
    world.terminated_invalid=True
    assert drain.completion(world,receipt)=="INVALID"
    world.terminated_invalid=False
    world.aircraft["a"]=replace(aircraft,status="CENSORED_DUE_TO_FAILURE",touchdown_time=None)
    assert drain.completion(world,receipt)=="UNKNOWN_TERMINAL"


def test_initial_ground_excluded_and_mixed_statistics_are_finite_json(tmp_path):
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.seed_service("ground",0,20.,"standard",.8)
    world.seed_hold("a","inner",0,.8)
    world.time=600.
    identity,cutoff=drain.apply_cutoff(world,{})
    assert cutoff["admitted_ids"]==cutoff["cohort_airborne_ids"]==["a"]
    assert drain.statistics(world)["normal"]==0
    assert drain.completion(world,cutoff)=="DRAINING"
    drain.write(tmp_path/"statistics.json",dict(statistics=drain.statistics(world),cutoff=cutoff,
        pending=dict(stage="RESULT",due=610.,packet=("mixed",None))))
    assert drain.base.load(tmp_path/"statistics.json")["pending"]["packet"]==["mixed",None]


def prepared_segment(tmp_path,monkeypatch,*,pending=False):
    f=serialized_fixture(tmp_path,pending=pending)
    monkeypatch.setattr(drain,"verify_algorithm_source",lambda source:{"world":"frozen"})
    monkeypatch.setattr(drain,"code_binding",lambda:dict(files={"runner":"reviewed"},source_hash="reviewed"))
    monkeypatch.setattr(drain,"source_inputs",lambda *args:(f.path,deepcopy(f.payload),dict(binding_id="original"),f.metadata))
    source=tmp_path/"source.json"; drain.write(source,{"fixture":True})
    args=SimpleNamespace(stage="run",output=tmp_path/"new",source_output=tmp_path/"old",source_freeze=source,
        root="s3r23",method="FULL_H",cpu=16,code_commit="reviewed",resume=False,segment_wall_s=1800.)
    return f,args


def test_segment_cutoff_completion_replay_and_explicit_resume_preserve_source(tmp_path,monkeypatch):
    f,args=prepared_segment(tmp_path,monkeypatch)
    original_bytes={p:p.read_bytes() for p in f.path.parent.iterdir() if p.is_file()}
    def advance(world,target,wrapper,**kwargs):
        assert kwargs==dict(settle_only_at_end=True)
        world.time=target
        world._settle()
        if target>600.:
            a=world.aircraft["a"]
            world.aircraft["a"]=replace(a,status="LANDED_SERVICE",touchdown_time=601.,service_end=721.)
        world._record(force=True)
    monkeypatch.setattr(RotatingSlotWorld,"advance",advance)
    # Pause at600, after the cutoff transaction, then resume from full state.
    calls=[0]
    def clock():
        calls[0]+=1
        return 1801. if calls[0]>=114 else 0.
    first=drain.run_segment(args,d=dependencies(),clock=clock)
    assert first["status"]=="PAUSED_CHECKPOINT" and first["time"]==600.
    directory=drain.comparison.evaluation_directory(args.output,args.root,args.method)
    first_stream=(directory/"segment_0001/states.jsonl").read_bytes()
    args.resume=True
    result=drain.run_segment(args,d=dependencies(),clock=lambda:0.)
    assert result["status"]=="COMPLETE" and result["time"]==605.
    assert result["terminal"]["airborne"]==result["arrivals_after600"]==result["admissions_after600"]==0
    assert result["admitted_outcomes"][0]["terminal_time"]==601.
    assert result["last_cohort_terminal_time"]==601. and result["time"]==605.
    assert result["normal_landed_before320"]==result["emergency_landed_before320"]==0
    assert result["initial_finals"]==result["inherited_final_outcomes"]==[]
    assert "inherited_normal_count" not in result
    assert result["cutoff"]["excluded_upstream_ids"]==["queued"]
    assert (directory/"segment_0001/states.jsonl").read_bytes()==first_stream
    ReplayBundle.validate(directory/result["replay_directory"])
    assert {p:p.read_bytes() for p in original_bytes}==original_bytes
    with pytest.raises(ValueError,match="normally paused"):
        drain.run_segment(args,d=dependencies(),clock=lambda:0.)


def test_smoke_does_not_create_formal_arm_or_replay(tmp_path,monkeypatch):
    f,args=prepared_segment(tmp_path,monkeypatch,pending=True)
    args.stage="smoke"
    def forbidden(*args,**kwargs): pytest.fail("restore smoke advanced physics")
    monkeypatch.setattr(RotatingSlotWorld,"advance",forbidden)
    result=drain.run_segment(args,d=dependencies(),clock=lambda:0.)
    assert result["status"]=="RESTORE_ONLY_PASS"
    assert not (args.output/"arms").exists()
    assert (args.output/"restore_checks/s3r23/FULL_H/receipt.json").is_file()
    assert not list(args.output.rglob("replay*"))


def test_initial_finals_capture_active_and_future_commitments_separately_from_landed_counts():
    def plan(key,kind,start,end):
        return SimpleNamespace(aircraft_id=key,kind=kind,start_time=start,end_time=end,trajectory_hash=key+"_hash")
    world=SimpleNamespace(active={"a":plan("a","FINAL_NORMAL",310.,350.),"other":plan("other","TRANSFER",315.,360.)},
                          soft={"b":plan("b","FINAL_EMERGENCY",335.,375.)})
    rows=drain.initial_finals(world)
    assert [(row["aircraft_id"],row["source"]) for row in rows]==[("a","active"),("b","soft")]
    assert rows[1]["start_time"]==335. and rows[0]["trajectory_hash"]=="a_hash"


def test_invalid_segment_cannot_publish_complete(tmp_path,monkeypatch):
    f,args=prepared_segment(tmp_path,monkeypatch)
    def advance(world,target,wrapper,**kwargs):
        world.failed=world.terminated_invalid=True
    monkeypatch.setattr(RotatingSlotWorld,"advance",advance)
    result=drain.run_segment(args,d=dependencies(),clock=lambda:0.)
    assert result["status"]=="ERROR" and result["error"] is not None
    directory=drain.comparison.evaluation_directory(args.output,args.root,args.method)
    assert drain.base.load(directory/"latest.json")["status"]=="ERROR"
