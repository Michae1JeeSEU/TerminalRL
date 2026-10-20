"""Narrow campaign/replay evidence; test names map only to exercised clauses."""
import json
from pathlib import Path

import pytest

from terminal_marl.core.rotating_slots.demand import ArrivalManifest, ArrivalRequest
from terminal_marl.core.rotating_slots.replay import BufferedStateJSONLSink, ReplayBundle
from terminal_marl.core.rotating_slots.stage2_campaign import (
    busy_fixture, frozen_manifest, CampaignBudget, common_boundary, atomic_json,
    episode_rank, shared_fifo_admission, save_checkpoint, restore_checkpoint,
    checkpoint_identity, digest, quantiles)
from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID


def identity(world):
    return checkpoint_identity(dict(source_hash="test_source",base_config_hash=world.config.config_hash,
                                    overlay_sha256="test_overlay"),digest(world.manifest.to_dict()))


@pytest.fixture(scope="module")
def fixture_world(): return busy_fixture(frozen_manifest())


def test_T01_frozen_busy_fixture_geometry_and_cohorts(fixture_world):
    world=fixture_world
    assert len(world.aircraft)==39
    for ring,count,slots in (("outer",14,37),("middle",12,28),("inner",10,18)):
        rows=[a for a in world.aircraft.values() if a.ring==ring]
        assert len(rows)==count
        assert sorted(a.slot for a in rows)==[k*slots//count for k in range(count)]
        assert all(.60<=a.soc<=.90 and a.fixture and a.release_time==0 and a.peak_seconds==0
                   and a.energy_j==0 and a.airborne_time==0 for a in rows)
    ground=[a for a in world.aircraft.values() if a.status=="LANDED_SERVICE"]
    assert sorted((a.pad_id,a.service_end,a.soc) for a in ground)==[(0,30.,.6),(2,60.,.6),(4,90.,.6)]
    metrics=world.metrics()
    assert metrics["initial_airborne_count"]==36 and metrics["initial_ground_count"]==3
    assert metrics["L_success"]==0 and metrics["N_admitted_total"]==36 and metrics["conservation"]
    # Constructor's seed methods already ran the authoritative continuous kernel.
    import math
    assert min(math.dist(a.p,b.p) for i,a in enumerate(world.aircraft.values())
               for b in list(world.aircraft.values())[i+1:])>=80.


def test_frozen_manifest_right_open_shared_and_deterministic():
    a,b=frozen_manifest(),frozen_manifest()
    assert a.to_dict()==b.to_dict()
    assert a.seed==71 and a.rate_per_hour==180.
    assert a.requests and all(0<=r.release_time<3600 for r in a.requests)


def test_T22_T23_episode_lexicographic_order():
    low=dict(F_energy_total=0,L_success=2,energy_j=1e12,cost_emergency=10000)
    high=dict(F_energy_total=0,L_success=3,energy_j=1e13,cost_emergency=20000)
    lost=dict(F_energy_total=1,L_success=300,energy_j=0)
    assert episode_rank(high)<episode_rank(low)<episode_rank(lost)


def test_T24_pending_is_not_success_or_failure():
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.seed_hold("pending","inner",0,.8)
    result=world.metrics()
    assert result["L_success"]==result["F_energy_total"]==0
    assert result["N_airborne_pending"]==1 and result["conservation"]


@pytest.mark.parametrize("touchdown_time",(3600.,3600.1))
def test_T24_literal_3600_actual_final_boundary_and_resume(touchdown_time):
    import time
    from terminal_marl.core.rotating_slots.trajectory import generate_transfer,generate_final
    started=time.perf_counter()
    request=ArrivalRequest("late",3200.,0,.9,"short")
    world=RotatingSlotWorld(manifest=ArrivalManifest((request,)),experiment_policy_id=STAGE2_POLICY_ID)
    world.auto_emergency=False
    world.advance(3200.,settle_only_at_end=True)
    assert world.admit(request.id).accepted
    world.advance(world.active[request.id].end_time,settle_only_at_end=True)
    for source in ("outer","middle"):
        aircraft=world.aircraft[request.id]
        assert aircraft.ring==source
        plan=None
        for target in range(28 if source=="outer" else 18):
            result=generate_transfer(request.id,source,aircraft.slot,target,world.time,
                source_version=aircraft.version)
            if result.accepted and world.submit((result.plan,)).accepted:
                plan=result.plan;break
        assert plan is not None
        world.advance(plan.end_time,settle_only_at_end=True)
    aircraft=world.aircraft[request.id]
    final=None
    for duration in range(20,301,5):
        start=touchdown_time-duration-13.5
        if start<world.time: continue
        result=generate_final(request.id,"inner",aircraft.slot,0,start,
            source_version=aircraft.version,durations=(float(duration),))
        if result.accepted and world.submit((result.plan,)).accepted:
            final=result.plan;break
    assert final is not None and final.end_time==touchdown_time
    world.advance(3600.,settle_only_at_end=True)
    metric=world.metrics()
    assert metric["F_energy_total"]==0 and metric["conservation"]
    assert metric["L_success"]==int(touchdown_time==3600.)
    assert metric["N_airborne_pending"]==int(touchdown_time>3600.)
    assert all(t<3600.-1e-9 for t in world.decision_times)
    if touchdown_time>3600.:
        # Continuation preserves the absolute rotation clock and records only
        # the later actual touchdown, without backdating the cutoff count.
        world.advance(touchdown_time,settle_only_at_end=True)
        assert world.aircraft[request.id].touchdown_time==touchdown_time
        assert world.metrics()["L_success"]==1
    assert world.time==touchdown_time
    print(json.dumps(dict(test="T24_literal_3600",touchdown_time=touchdown_time,wall_s=time.perf_counter()-started)))


def test_T27_budget_survives_restart_and_common_complete_markers(tmp_path):
    clock=[100000.]
    budget=CampaignBudget(tmp_path/"budget.json",120.,clock=lambda:clock[0])
    clock[0]+=34081.
    budget.persist()
    restarted=CampaignBudget(tmp_path/"budget.json",clock=lambda:clock[0])
    assert restarted.compute_remaining==0 and restarted.elapsed>=34201.
    assert restarted.data["started_epoch"]==99880.
    for method,times in (("energy_slack",(60,120,180)),("rh_cp_sat",(60,120))):
        for t in times: atomic_json(tmp_path/method/f"summary_{t:04d}.json",{})
    (tmp_path/"rh_cp_sat"/"summary_0180.json.partial").write_text("unfinished")
    assert common_boundary(tmp_path)==120


def test_common_admission_gate_fifo_only_revealed():
    from types import SimpleNamespace
    requests=[ArrivalRequest("a",1,0,.7,"standard"),ArrivalRequest("b",2,0,.7,"standard"),
              ArrivalRequest("c",3,1,.7,"standard")]
    called=[]
    def admit(aid): called.append(aid);return SimpleNamespace(accepted=False)
    port=SimpleNamespace(requests={r.id:r for r in requests},aircraft={},admit=admit)
    shared_fifo_admission(None,port)
    assert called==["a","c"]


def test_T28_exact_checkpoint_pending_dispatch_and_absolute_phase(tmp_path):
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID,
                           state_sink=BufferedStateJSONLSink(tmp_path/"states.jsonl"))
    world.auto_emergency=False
    world.seed_hold("a","inner",0,.8)
    scheduler=NormalSchedulerCallback()
    calls=[]
    world.advance(.4,lambda snapshot,port:calls.append(port.time),settle_only_at_end=True)
    position=world.aircraft["a"].p
    checkpoint=tmp_path/"checkpoint.json"
    save_checkpoint(checkpoint,world,scheduler,identity(world))
    world.state_sink.close()
    loaded=restore_checkpoint(checkpoint,NormalSchedulerCallback(),identity(world))
    assert loaded.time==.4 and loaded.aircraft["a"].p==position
    assert loaded.slots.to_dict()==world.slots.to_dict()
    assert loaded._targeted_energy==world._targeted_energy
    loaded.advance(.8,lambda snapshot,port:calls.append(port.time),settle_only_at_end=True)
    assert loaded.time==.8 and loaded.aircraft["a"].p!=position
    assert not loaded.failed
    loaded.state_sink.close()


def test_T28_checkpoint_rejects_identity_corruption_and_unknown_fields(tmp_path):
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    scheduler=NormalSchedulerCallback()
    path=tmp_path/"checkpoint.json"
    save_checkpoint(path,world,scheduler,identity(world))
    bad=dict(identity(world),source_hash="different")
    with pytest.raises(ValueError,match="identity/content"):
        restore_checkpoint(path,scheduler,bad)
    world.unapproved_field=1
    with pytest.raises(ValueError,match="inventory"):
        save_checkpoint(path,world,scheduler,identity(world))


def test_T26_timing_quantiles_keep_raw_counts():
    summary=quantiles([dict(wall_s=x) for x in (0.,1.,2.,10.)],"wall_s")
    assert summary["count"]==4 and summary["over_5s"]==1
    assert summary["p50"]==1.5 and summary["raw_s"]==[0.,1.,2.,10.]


def test_T28_stage2_replay_separate_predictions_and_failure_ledger(tmp_path):
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.auto_emergency=False;world.seed_hold("a","inner",0,.8)
    record=world.to_dict()
    record.update(experiment_policy_id=STAGE2_POLICY_ID,
        decisions=[dict(time=0.,wall_s=.4)],dispatch_timings=[dict(time=0.,normal_wall_s=.4,dispatch_wall_s=.7)],
        failure_archive={},tentative_predictions=[dict(aircraft_id="a",plans=[])],stage2_provenance={"revision":"test"})
    bundle=ReplayBundle.write(tmp_path/"bundle",record=record,code_commit="test")
    assert bundle.index["human_acceptance"]=="HUMAN_PENDING"
    assert bundle.data["tentative_predictions"][0]["aircraft_id"]=="a"
    assert bundle.data["dispatch_timings"][0]["dispatch_wall_s"]==.7
    assert "failure_archive.json" in bundle.index["files"]
    assert bundle.index["prediction_authority"]=="TENTATIVE_ONLY"
    index=load_summary(tmp_path/"bundle/replay.json")
    del index["files"]["decisions.jsonl"]
    atomic_json(tmp_path/"bundle/replay.json",index)
    with pytest.raises(ValueError,match="required members"): ReplayBundle.validate(tmp_path/"bundle")
    with pytest.raises(ValueError,match="required members"): ReplayBundle.load(tmp_path/"bundle")


def test_T27_T28_resident_episode_continues_60_to_120_without_reset(tmp_path,monkeypatch):
    from terminal_marl.core.rotating_slots import stage2_campaign as campaign
    empty=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    expected_identity=identity(empty)
    atomic_json(tmp_path/"campaign.json",{"identity":expected_identity})
    atomic_json(tmp_path/"manifest.json",ArrivalManifest().to_dict())
    atomic_json(tmp_path/"provenance.json",{"revision":"test"})
    CampaignBudget(tmp_path/"budget.json")
    constructions=[]
    def construct(manifest,state_sink):
        constructions.append(1)
        world=RotatingSlotWorld(manifest=manifest,state_sink=state_sink,experiment_policy_id=STAGE2_POLICY_ID)
        world.auto_emergency=False
        return world
    monkeypatch.setattr(campaign,"busy_fixture",construct)
    monkeypatch.setattr(campaign,"verify_source_provenance",lambda x:None)
    class Connection:
        def __init__(self): self.targets=iter((60.,120.,None));self.results=[]
        def recv(self): return next(self.targets)
        def send(self,value): self.results.append(value)
    connection=Connection()
    campaign.run_episode_block(tmp_path,"energy_slack",None,connection=connection)
    assert constructions==[1]
    assert connection.results==[("completed",60.),("completed",120.)]
    restored=restore_checkpoint(tmp_path/"energy_slack/checkpoint.json",NormalSchedulerCallback(),expected_identity)
    assert restored.time==120. and restored.decision_times==pytest.approx([float(t) for t in range(0,120,5)])
    assert restored._next_heartbeat==120.
    assert load_summary(tmp_path/"energy_slack/summary_0120.json")["normal_timing"]["count"]==24
    restored.state_sink.close()
    # Publishing a replay of the earlier shared window cannot truncate a
    # faster method's live history or invalidate its later checkpoint.
    stream=tmp_path/"energy_slack/states.jsonl"
    before=stream.read_bytes()
    campaign.export_episode(tmp_path,"energy_slack",60)
    assert stream.read_bytes()==before
    # A killed publisher can recover summary metadata without resimulating.
    (tmp_path/"energy_slack/summary_0120.json").unlink()
    campaign.run_episode_block(tmp_path,"energy_slack",120.)
    recovered=load_summary(tmp_path/"energy_slack/summary_0120.json")
    assert recovered["checkpoint_timing_status"]=="UNKNOWN_AFTER_INTERRUPTED_PUBLICATION"
    assert constructions==[1]


def load_summary(path): return json.loads(Path(path).read_text())


def test_T28_viewer_isolated_marker_and_actual_dispatch_hud(tmp_path):
    from terminal_marl.visualization.rotating_slots_replay import ReplayController
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.auto_emergency=False
    world.seed_hold("loss","inner",0,.00000001)
    world.advance(.2,settle_only_at_end=True)
    record=world.to_dict()
    record.update(experiment_policy_id=STAGE2_POLICY_ID,failure_archive=world.failure_archive,
        dispatch_timings=[dict(time=0.,normal_wall_s=.4,dispatch_wall_s=.7)],
        tentative_predictions=[dict(as_of=.1,aircraft_id="loss",plans=[])])
    bundle=ReplayBundle.write(tmp_path/"bundle",record=record,code_commit="test")
    controller=ReplayController(bundle);controller.seek(.2)
    overlay=controller.overlays()
    assert len(overlay["failed_isolated"])==1
    assert overlay["minimum_current_3d_pair"] is None
    assert overlay["failure_archive"]["loss"]["reason"]=="BATTERY_DEPLETED"
    assert any("full dispatch wall=0.7000s" in line for line in controller.hud_lines())
    assert not any("OLD MODEL" in line for line in controller.hud_lines())
    assert "capacity-only reroute" in overlay["admission_model"]
    controller.seek(0.)
    assert controller.overlays()["tentative_predictions"]==[]
