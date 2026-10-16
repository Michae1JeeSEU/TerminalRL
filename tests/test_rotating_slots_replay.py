"""Replay semantics; these checks do not stand in for researcher observation."""
from hashlib import sha256

import pytest
import json
from pathlib import Path

from terminal_marl.core.rotating_slots.config import load_config, DEFAULT_CONFIG
from terminal_marl.core.rotating_slots.replay import ReplayBundle, ReplayArchive, BufferedStateJSONLSink
from terminal_marl.core.rotating_slots.trajectory import make_stable_plan
from terminal_marl.visualization.rotating_slots_replay import ReplayController


def record():
    c=load_config()
    plan=make_stable_plan("u1","outer",0,end_time=2.)
    rows=[]
    for t in (0.,.2,.33,.4,.6):
        k=plan.state(t)
        rows.append({"time":t,"aircraft":[{"id":"u1","status":"HOLD_OUTER","soc":.8-t*.001,
            "p":k.p,"v":k.v,"a":k.a,"jerk":k.jerk,"ring":"outer","slot":0,
            "service_class":"standard","trajectory_hash":plan.trajectory_hash,
            "battery_power_w":50000.,"emergency":{"declared":False}}],"slots":[],"pads":[],"metrics":{}})
    return {"config_hash":c.config_hash,"sample_dt_s":.2,"manifest":{"seed":11,"requests":[]},"states":rows,
        "trajectories":{plan.trajectory_hash:plan.to_dict()},"events":[{"time":.33,"type":"TEST_EVENT","aircraft_id":"u1","details":{}}],
        "metrics":{},"rejections":[{"time":.2,"aircraft_id":"u1","reason":"NOT_FOUND","details":{}}]}


def bundle(tmp_path, source=None):
    return ReplayBundle.write(tmp_path/"bundle",record=source or record(),code_commit="test-fixture",seed=11,command="pytest")


def test_bundle_exact_bytes_hashes_and_pending_acceptance(tmp_path):
    b=bundle(tmp_path)
    assert (b.path/"config.yaml").read_bytes()==DEFAULT_CONFIG.read_bytes()
    assert b.index["config_bytes_sha256"]==sha256(DEFAULT_CONFIG.read_bytes()).hexdigest()
    assert b.data["visual_acceptance"]["researcher_visual_acceptance"]=="PENDING"
    assert b.index["code_commit"]=="test-fixture"
    with pytest.raises(AttributeError): b.data.time=2
    with pytest.raises(TypeError): b.data["metrics"]["x"]=1
    (b.path/"states.jsonl").write_text("{}\n")
    with pytest.raises(ValueError,match="content hash mismatch"): ReplayBundle.load(b.path)


def test_arbitrary_seek_uses_analytic_plan_not_linear_samples(tmp_path):
    b=bundle(tmp_path); controller=ReplayController(b)
    before=(b.path/"trajectories.json").read_bytes()
    snapshot=controller.seek(.137)
    plan=next(iter(controller._plans.values()))
    assert snapshot.uavs[0].position==plan.state(.137).p
    left=plan.state(0.).p; right=plan.state(.2).p
    linear=tuple(a+(z-a)*.137/.2 for a,z in zip(left,right))
    assert snapshot.uavs[0].position != linear
    controller.seek(.59); controller.seek(.137)
    assert controller.snapshot().uavs[0].position==snapshot.uavs[0].position
    assert (b.path/"trajectories.json").read_bytes()==before
    with pytest.raises(TypeError): snapshot.telemetry["x"]=1


def test_step_event_jump_pause_speeds_and_trajectory_selection(tmp_path):
    controller=ReplayController(bundle(tmp_path))
    assert controller.step().time==.2
    assert controller.jump_event().time==.33
    assert controller.step().time==.4
    assert controller.jump_event(-1).time==.33
    assert controller.advance(1.).time==.33
    for speed in controller.speeds:
        controller.set_speed(speed); assert controller.speed==speed
    with pytest.raises(ValueError): controller.set_speed(2.)
    controller.select("u1",follow=True)
    assert controller.follow
    controller.set_trajectories("hidden")
    assert controller.snapshot().uavs[0].trajectory_points==()
    controller.set_trajectories("selected")
    assert controller.snapshot().uavs[0].trajectory_points
    controller.paused=False; controller.seek(0.); controller.set_speed(.25)
    assert controller.advance(.4).time==.1


def test_exact_lifecycle_service_clearance_and_frozen_failure(tmp_path):
    source=record()
    source["states"][1]["aircraft"][0].update(status="LANDED_SERVICE",pad_id=0,ring=None,slot=None,trajectory_hash=None,p=[100.,0.,0.],v=[0.,0.,0.])
    source["states"][1]["pads"]=[{"resource":0,"aircraft_id":"u1","start":.2,"end":.33,"kind":"service"}]
    source["states"][2]["aircraft"][0].update(status="CLEARED",trajectory_hash=None)
    source["states"][3]["aircraft"][0].update(status="ENERGY_FAILURE",p=[12.,34.,56.])
    source["states"][4]["aircraft"][0].update(status="ENERGY_FAILURE",p=[12.,34.,56.])
    controller=ReplayController(bundle(tmp_path,source))
    frame=controller.seek(.25)
    assert frame.service_count==1 and frame.landing_zones[0].status=="occupied"
    assert frame.landing_zones[0].service_remaining==pytest.approx(.08)
    frame=controller.seek(.33)
    assert frame.completed_count==1 and not frame.uavs and frame.landing_zones[0].status=="free"
    assert controller.seek(.5).uavs[0].position==(12.,34.,56.)


def test_departure_protection_is_calendar_interval_and_soft_dependency_visible(tmp_path):
    source=record()
    source["states"][0]["slots"]=[
        {"resource":["outer",0],"aircraft_id":"u1","start":0.,"end":.1,"kind":"departure_protection"},
        {"resource":["middle",0],"aircraft_id":"u1","start":.3,"end":2.,"kind":"target","dependencies":["release-1"]}]
    controller=ReplayController(bundle(tmp_path,source))
    assert controller.slots_at()[0]["state"]=="DEPARTING"
    assert controller.overlays()["future_soft_dependencies"][0]["dependencies"]==("release-1",)
    controller.seek(.15)
    assert controller.slots_at()[0]["state"]=="OCCUPIED"
    assert len(controller.slots_at())==83


def test_logged_curves_do_not_generate_missing_power(tmp_path):
    source=record(); source["states"][1]["aircraft"][0].pop("battery_power_w")
    b=bundle(tmp_path,source)
    rows=b.curves("u1")
    assert rows[0]["battery_power"]==50000.
    assert rows[1]["battery_power"] is None
    assert set(rows[0])=={"time","speed","acceleration","jerk","battery_power","SOC"}


def test_missing_plan_rejects_arbitrary_airborne_pose(tmp_path):
    source=record(); source["trajectories"]={}
    for s in source["states"]: s["aircraft"][0]["trajectory_hash"]=None
    controller=ReplayController(bundle(tmp_path,source))
    with pytest.raises(ValueError,match="requires an authoritative plan"): controller.seek(.1)


def test_sample_dt_and_profile_mismatch(tmp_path):
    source=record(); source["sample_dt_s"]=.1
    assert bundle(tmp_path,source).index["sample_dt_s"]==.1
    source=record(); h=next(iter(source["trajectories"]))
    source["trajectories"][h]["profile_hash"]="wrong"
    with pytest.raises(ValueError,match="hash mismatch"):
        ReplayBundle.write(tmp_path/"invalid",record=source,code_commit="fixture")


def test_actual_world_export_matches_authoritative_pose_and_logged_power(tmp_path):
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
    world=RotatingSlotWorld(physics_dt=.1)
    world.auto_emergency=False  # This fixture tests replay, not emergency search.
    world.seed_hold("actual","middle",3,soc=.8)
    world.advance(.4)
    b=ReplayBundle.write(tmp_path/"actual",world=world,code_commit="test-world-export",seed=11)
    controller=ReplayController(b)
    actual=controller.seek(.137).uavs[0]
    h=world.aircraft["actual"].trajectory_hash
    assert actual.position==controller._plans[h].state(.137).p
    assert controller.slots_at()[37+3]["state"]=="OCCUPIED"
    assert b.index["sample_dt_s"]==.1
    assert b.curves("actual")[-1]["battery_power"]>0
    assert b.data["manifest"]["seed"] is None


def test_jsonl_sink_replaces_same_timestamp_even_after_flush(tmp_path):
    path=tmp_path/"source.jsonl"
    with BufferedStateJSONLSink(path) as sink:
        sink({"time":0.,"commits":[]})
        sink.flush()
        sink({"time":0.,"commits":["first"]})
        sink.flush()
        sink({"time":0.,"commits":["latest", "larger", "payload"]})
        sink.flush()
        sink({"time":0.,"commits":["final"]})
        sink({"time":.137,"commits":["exact_event"]})
        sink({"time":.2,"commits":["sample"]})
        assert sink.frame_count==3
        with pytest.raises(ValueError,match="must not decrease"):
            sink({"time":.1})
        sink.flush(); sink.flush()
    rows=[json.loads(line) for line in path.read_bytes().splitlines()]
    assert rows==[{"time":0.,"commits":["final"]},{"time":.137,"commits":["exact_event"]},
                  {"time":.2,"commits":["sample"]}]


def test_streamed_world_bundle_parity_and_one_frame_retention(tmp_path):
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
    streamed_sink=BufferedStateJSONLSink(tmp_path/"streamed.jsonl")
    memory=RotatingSlotWorld()
    streamed=RotatingSlotWorld(state_sink=streamed_sink)
    try:
        for world in (memory,streamed):
            world.auto_emergency=False
            world.seed_hold("hold","outer",0,soc=.8)
            world.to_dict()  # Mid-run flush must not freeze the current timestamp.
            world.seed_service("service",pad_id=0,remaining=.3)
            world.to_dict()  # Same timestamp is replaced, including this fixture.
            world.advance(.6)
        eager=ReplayBundle.write(tmp_path/"memory_bundle",world=memory,code_commit="stream-parity",seed=11)
        archive=ReplayBundle.write(tmp_path/"stream_bundle",world=streamed,code_commit="stream-parity",seed=11)
        assert isinstance(archive,ReplayArchive)
        assert len(streamed.records)==1
        loaded=archive.open()
        assert eager.data["states"]==loaded.data["states"]
        assert eager.data["events"]==loaded.data["events"]
        assert eager.data["metrics"]==loaded.data["metrics"]
        assert eager.data["trajectories"]==loaded.data["trajectories"]
        times=[float(row["time"]) for row in loaded.data["states"]]
        assert len(times)==len(set(times))==archive.state_count
        assert .3 in times  # Exact service event between .2 s physics samples.
        assert len(loaded.data["states"][0]["aircraft"])==2
        assert loaded.data["states"][-1]["aircraft"][1]["status"]=="CLEARED"
        assert archive.time_bounds==(0.,.6)
    finally:
        streamed_sink.close()


def test_streamed_write_and_validate_do_not_eagerly_read_states(tmp_path,monkeypatch):
    source=record()
    spool=tmp_path/"spool.jsonl"
    with BufferedStateJSONLSink(spool) as sink:
        for frame in source["states"]: sink(frame)
        sink.flush()
        source.update(states=[],states_jsonl=str(spool),states_count=sink.frame_count,states_streamed=True)
        original_read=Path.read_bytes
        def guarded_read(path):
            if path.name in ("states.jsonl","spool.jsonl"):
                raise AssertionError("state file was eagerly read")
            return original_read(path)
        monkeypatch.setattr(Path,"read_bytes",guarded_read)
        archive=ReplayBundle.write(tmp_path/"lazy",record=source,code_commit="streaming")
        assert isinstance(archive,ReplayArchive) and archive.state_count==5
        assert ReplayBundle.validate(archive.path).state_count==5


def test_partial_callback_stream_is_not_mislabeled_complete(tmp_path):
    source=record(); source["states_streamed"]=True; source["states_complete"]=False
    with pytest.raises(ValueError,match="complete streamed states"):
        ReplayBundle.write(tmp_path/"incomplete",record=source,code_commit="streaming")


def test_seek_uses_current_emergency_fee_not_future_episode_outcome(tmp_path):
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
    from terminal_marl.core.rotating_slots.trajectory import generate_final
    world=RotatingSlotWorld(); world.auto_emergency=False
    world.seed_hold("rescue","inner",0,soc=.8,emergency=True)
    world.advance(.2)
    plan=generate_final("rescue","inner",0,0,.2,emergency=True).plan
    assert plan is not None and world.submit((plan,)).accepted
    archive=ReplayBundle.write(tmp_path/"fee",world=world,code_commit="test-fee")
    controller=ReplayController(archive)
    controller.seek(.1)
    assert controller.overlays()["logged_metrics"]["cost_emergency"]==0
    assert controller.overlays()["episode_metrics"]["cost_emergency"]==1200
    assert not controller.overlays()["current_commits"]
    assert not controller.snapshot().uavs[0].committed
    controller.seek(.2)
    assert controller.overlays()["logged_metrics"]["cost_emergency"]==1200
    assert [a["id"] for a in controller.overlays()["current_commits"]]==["rescue"]
    assert controller.snapshot().uavs[0].committed


def test_soft_dependencies_exclude_committed_future_reservations(tmp_path):
    source=record()
    source["states"][0]["slots"]=[
        {"resource":["outer",1],"aircraft_id":"u1","start":.3,"end":2.,"committed":True,"dependencies":["active-inbound"]},
        {"resource":["middle",0],"aircraft_id":"u1","start":.4,"end":2.,"committed":False,"dependencies":["soft-release"]}]
    controller=ReplayController(bundle(tmp_path,source))
    dependencies=controller.overlays()["future_soft_dependencies"]
    assert len(dependencies)==1 and dependencies[0]["dependencies"]==("soft-release",)
    assert not controller.overlays()["current_commits"]


def test_analytic_bundle_has_explicit_no_episode_outcome_banner(tmp_path):
    source=record(); source["metrics"]={"fixture_only":True,"normal_landed":99}
    source["states"][0]["metrics"]={"fixture_only":True}
    overlay=ReplayController(bundle(tmp_path,source)).overlays()
    assert overlay["fixture_banner"]=="ANALYTIC FIXTURE - no physical episode outcome"
    assert "normal_landed" not in overlay["logged_metrics"]
    assert overlay["episode_metrics"]["normal_landed"]==99


def test_separation_evidence_labels_fast_interval_and_refined_certificate(tmp_path):
    source=record()
    source["metrics"]={"minimum_separation":{"lower_bound_m":80.,"upper_bound_m":200.,"time":.1,"pair":["a","b"]}}
    fast=ReplayController(bundle(tmp_path/"fast",source)).overlays()["separation_evidence"]
    assert fast["label"]=="Logged separation interval evidence"
    assert (fast["lower_bound_m"],fast["upper_bound_m"],fast["witness_time"])==(80.,200.,.1)
    source["metrics"]["continuous_minimum"]={"status":"CERTIFIED","lower_bound_m":79.9999,
        "upper_bound_m":80.0001,"witness_time":.137,"witness_distance_m":80.}
    refined=ReplayController(bundle(tmp_path/"refined",source)).overlays()["separation_evidence"]
    assert refined["label"]=="Certified continuous minimum (whole episode)"
    assert (refined["lower_bound_m"],refined["upper_bound_m"],refined["witness_time"],refined["witness_distance_m"])==(
        79.9999,80.0001,.137,80.)
    source["metrics"]["certified_continuous_minimum"]={"status":"CERTIFIED","lower_bound_m":90.,
        "upper_bound_m":90.,"witness_time":.2,"witness_distance_m":90.}
    explicit=ReplayController(bundle(tmp_path/"explicit",source)).overlays()["separation_evidence"]
    assert explicit["lower_bound_m"]==90. and explicit["witness_time"]==.2


def test_frozen_censored_aircraft_remains_in_current_physical_pair(tmp_path):
    source=record()
    for frame in source["states"]:
        failed=frame["aircraft"][0]
        failed.update(status="ENERGY_FAILURE",p=[0.,0.,0.])
        censored=dict(failed,id="censored",status="CENSORED_DUE_TO_FAILURE")
        cleared=dict(failed,id="cleared",status="CLEARED")
        frame["aircraft"]=[failed,censored,cleared]
    controller=ReplayController(bundle(tmp_path,source))
    controller.seek(.137)
    distance,a,b=controller.overlays()["minimum_current_3d_pair"]
    assert distance==0. and {a,b}=={"u1","censored"}
    assert {u.position for u in controller.snapshot().uavs}=={(0.,0.,0.)}
