"""Frozen equal-linear rotation: phase, continuous motion and output closure."""
import json
import math

import pytest

from terminal_marl.core.rotating_slots.config import (
    load_config, load_service_profile, CANONICAL_CONFIG_HASH, SERVICE_HALF_CONFIG_HASH,
    config_for_hash, digest)
from terminal_marl.core.rotating_slots.geometry import ring_angular_speed, slot_phase, slot_state
from terminal_marl.core.rotating_slots.trajectory import (
    TrajectoryPlan, generate_transfer, generate_final, make_stable_plan, screen_normal_templates)
from terminal_marl.core.rotating_slots.safety import validate_plan, validate_separation, certified_minimum_distance
from terminal_marl.core.rotating_slots.energy import integrate_energy, stable_power, evaluate_power


@pytest.fixture(scope="module")
def config(): return load_service_profile("service_half_v1","equal_linear_21_v1")


def test_profiles_are_orthogonal_and_old_payloads_unchanged():
    assert load_config().config_hash==CANONICAL_CONFIG_HASH
    assert load_service_profile("service_half_v1").config_hash==SERVICE_HALF_CONFIG_HASH
    for service in ("baseline","service_half_v1"):
        old=load_service_profile(service)
        new=load_service_profile(service,"equal_linear_21_v1")
        data=new.to_dict(); data["airspace"].pop("rotation_profile");data["airspace"].pop("common_tangential_speed_mps")
        assert data==old.to_dict()
        assert new.service_profile==old.service_profile==service
        assert config_for_hash(new.config_hash).to_dict()==new.to_dict()
    with pytest.raises(ValueError,match="rotation profile"): load_service_profile(rotation_profile="other")


@pytest.mark.parametrize("time",(0.,37.,600.,1234.))
def test_equal_speed_and_drifting_phase(config,time):
    for ring,radius in (("outer",600.),("middle",450.),("inner",300.)):
        assert ring_angular_speed(ring,config)==21./radius
        state=slot_state(ring,0,time,config)
        assert math.hypot(*state.v[:2])==pytest.approx(21.)
        assert slot_phase(ring,0,time,config)==-21./radius*time
    assert slot_phase("middle",0,time,config)-slot_phase("outer",0,time,config)==pytest.approx((.035-21/450)*time)


@pytest.mark.parametrize("ring,target,time",(("outer",1,0.),("outer",4,600.),("middle",0,0.),("middle",0,600.)))
def test_transfer_c2_endpoints_and_serialized_continuation(config,ring,target,time):
    result=generate_transfer("a",ring,0,target,time,config=config)
    assert result.accepted
    plan=TrajectoryPlan.from_dict(json.loads(json.dumps(result.plan.to_dict())),config)
    assert plan.to_dict()==result.plan.to_dict()
    assert validate_plan(plan,config).accepted
    for t,r,k in ((plan.start_time,ring,0),(plan.end_time,plan.target_ring,target),
                  (plan.start_time-3,ring,0),(plan.end_time+3,plan.target_ring,target)):
        actual=plan.state(t); expected=slot_state(r,k,t,config)
        for field in ("p","v","a"):
            assert getattr(actual,field)==pytest.approx(getattr(expected,field),abs=1e-7)
    for start,end,r,k in ((time-4,time-1,ring,0),(plan.end_time+1,plan.end_time+4,plan.target_ring,target)):
        segment=next(plan.intervals(start,end))
        assert segment.state((start+end)/2).v==pytest.approx(slot_state(r,k,(start+end)/2,config).v)
        energy=integrate_energy(plan,start,end)
        assert energy.valid
        assert energy.joules==pytest.approx(stable_power(r,config).battery_w*(end-start),rel=1e-12)
    with pytest.raises(ValueError,match="profile hash mismatch"):
        TrajectoryPlan.from_dict(plan.to_dict(),load_service_profile("service_half_v1"))


@pytest.mark.parametrize("ring",("outer","middle","inner"))
def test_emergency_final_starts_at_each_ring_clock(config,ring):
    result=generate_final("a",ring,0,0,37.,emergency=True,config=config)
    assert result.accepted
    expected=slot_state(ring,0,37.,config)
    for name in ("p","v","a"):
        assert getattr(result.plan.state(37.),name)==pytest.approx(getattr(expected,name),abs=1e-7)
    restored=TrajectoryPlan.from_dict(result.plan.to_dict())
    assert restored.state(36.).v==pytest.approx(slot_state(ring,0,36.,config).v)


def test_feasibility_cache_uses_actual_angular_coefficients(config):
    from terminal_marl.core.rotating_slots.trajectory import _TRANSFER_FEASIBILITY
    _TRANSFER_FEASIBILITY.clear()
    early=generate_transfer("a","middle",0,2,0.,config=config,durations=(40,))
    late=generate_transfer("a","middle",0,2,100.,config=config,durations=(40,))
    assert early.accepted and not late.accepted
    assert generate_transfer("a","middle",0,2,0.,config=config,durations=(40,)).accepted
    assert not generate_transfer("a","middle",0,2,100.,config=config,durations=(40,)).accepted


@pytest.mark.parametrize("ring,time",(("outer",600.),("middle",1234.),("inner",600.)))
def test_template_rejections_agree_with_direct_scalar_candidates(config,ring,time):
    table,hit=screen_normal_templates(ring,0,time,config)
    again,hit=screen_normal_templates(ring,0,time,config)
    assert hit and again==table
    count={"outer":28,"middle":18,"inner":6}[ring]
    retained=rejected=0
    for target in range(count):
        for duration in (40,80,120,180):
            result=(generate_final("a",ring,0,target,time,config=config,durations=(duration,)) if ring=="inner" else
                    generate_transfer("a",ring,0,target,time,config=config,durations=(duration,)))
            if table[target,duration] is not None:
                rejected+=1
                assert not result.accepted,(ring,target,time,duration)
            elif result.accepted: retained+=1
    assert retained>0 and rejected>0
    if ring!="inner":
        early,_=screen_normal_templates(ring,0,0.,config)
        assert early!=table


def test_stable_separation_uses_phase_independent_lower_bound(config):
    a=make_stable_plan("a","outer",0,0,10,config=config)
    b=make_stable_plan("b","middle",0,0,10,config=config)
    assert math.dist(a.state(0).p,b.state(0).p)!=pytest.approx(math.dist(a.state(100).p,b.state(100).p))
    result=validate_separation(a,b,0.,1000.,config)
    assert result.accepted and result.details["method"]=="concentric_ring_radial_vertical_bound"
    assert result.details["lower_bound_m"]==pytest.approx(math.hypot(150,40),abs=1e-8)
    minimum=certified_minimum_distance(a,b,0.,60.,config,tolerance_m=.05)
    assert minimum["status"]=="CERTIFIED"
    assert minimum["lower_bound_m"]>=80
    same=make_stable_plan("c","outer",1,0,10,config=config)
    assert validate_separation(a,same,0.,1000.,config).details["method"]=="common_rotation_structure_with_roundoff_enclosure"


@pytest.mark.parametrize("ring",("outer","middle","inner"))
def test_stable_energy_matches_profile_state(config,ring):
    plan=make_stable_plan("a",ring,3,20,21,config=config)
    state=slot_state(ring,3,20,config)
    power=evaluate_power(state.v,state.a,config)
    result=integrate_energy(TrajectoryPlan.from_dict(plan.to_dict()),15.,30.)
    assert power.valid and result.valid
    assert result.joules==pytest.approx(power.battery_w*15,rel=1e-12)


def test_campaign_prepare_identity_and_checkpoint_replay(config,tmp_path,monkeypatch):
    from terminal_marl.core.rotating_slots import stage2_campaign as campaign
    from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
    from terminal_marl.core.rotating_slots.replay import ReplayBundle
    # Profile/serialization wiring only: dependency validation and expensive
    # initial reference search are explicitly stubbed. Server smoke uses both.
    monkeypatch.setattr(campaign,"environment_lock",lambda:{"test_stub":True})
    monkeypatch.setattr(campaign,"initial_references",lambda *args:[])
    provenance=tmp_path/"frozen.json"
    campaign.freeze_provenance(provenance,"service_half_v1",rotation_profile="equal_linear_21_v1",
        scenario_profile="diagnostic18_rate120_v1",rescue_rule_profile="rescue_need_v7")
    prepared=campaign.prepare_campaign(tmp_path/"campaign",provenance_path=provenance,
        service_profile="service_half_v1",rotation_profile="equal_linear_21_v1",
        scenario_profile="diagnostic18_rate120_v1",rescue_rule_profile="rescue_need_v7")
    assert prepared["rotation_profile"]==prepared["identity"]["rotation_profile"]=="equal_linear_21_v1"
    assert campaign.campaign_config(tmp_path/"campaign",prepared).config_hash==config.config_hash
    bad=dict(prepared,rotation_profile="equal_angular")
    with pytest.raises(ValueError,match="configuration identity"): campaign.campaign_config(tmp_path/"campaign",bad)
    world=campaign.busy_fixture(campaign.frozen_manifest("diagnostic18_rate120_v1"),config=config,
        scenario_profile="diagnostic18_rate120_v1",rescue_rule_profile="rescue_need_v7")
    world.auto_emergency=False
    world.advance(.4,settle_only_at_end=True)
    path=tmp_path/"checkpoint.json"; scheduler=NormalSchedulerCallback()
    campaign.save_checkpoint(path,world,scheduler,prepared["identity"])
    restored=campaign.restore_checkpoint(path,NormalSchedulerCallback(),prepared["identity"])
    assert restored.config.rotation_profile=="equal_linear_21_v1"
    assert restored.aircraft==world.aircraft
    bundle=ReplayBundle.write(tmp_path/"replay",world=world,config_path=config.path,code_commit="test")
    assert bundle.index["rotation_profile"]=="equal_linear_21_v1"
    assert bundle.config.config_hash==config.config_hash
    ReplayBundle.validate(tmp_path/"replay")


@pytest.mark.parametrize("method",("energy_slack","rh_cp_sat"))
def test_real_two_method_tiny_output_closure(config,tmp_path,method):
    pytest.importorskip("ortools",reason="real CP closure requires existing pinned server environment")
    from terminal_marl.core.rotating_slots import stage2_campaign as campaign
    from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback, QUALITY_BUDGET_PROFILE
    from terminal_marl.core.rotating_slots.replay import ReplayBundle
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID
    world=RotatingSlotWorld(config=config,experiment_policy_id=STAGE2_POLICY_ID)
    world.rescue_rule_profile="rescue_need_v7"
    world.seed_hold("a","inner",0,.85)
    scheduler=NormalSchedulerCallback(method)
    assert scheduler.budget_profile==QUALITY_BUDGET_PROFILE
    world.advance(.4,scheduler,settle_only_at_end=True)
    assert world.time==.4 and not world.failed and not world.terminated_invalid
    assert scheduler.records and world.dispatch_timings
    identity=campaign.checkpoint_identity(dict(source_hash="tiny_test",base_config_hash=config.config_hash,
        overlay_sha256="tiny_test",service_profile=config.service_profile,rotation_profile=config.rotation_profile,
        rescue_rule_profile="rescue_need_v7"),digest(world.manifest.to_dict()))
    path=tmp_path/"checkpoint.json"
    campaign.save_checkpoint(path,world,scheduler,identity)
    restored_scheduler=NormalSchedulerCallback(method)
    restored=campaign.restore_checkpoint(path,restored_scheduler,identity)
    assert restored.time==world.time and restored.aircraft==world.aircraft
    assert restored_scheduler.state_dict()==scheduler.state_dict()
    record=world.to_dict()
    record.update(decisions=scheduler.records,dispatch_timings=world.dispatch_timings,
        failure_archive=world.failure_archive,tentative_predictions=[],stage2_provenance={"test":True})
    bundle=ReplayBundle.write(tmp_path/"replay",record=record,config_path=config.path,code_commit="tiny_test")
    assert bundle.index["rotation_profile"]=="equal_linear_21_v1"
    ReplayBundle.validate(tmp_path/"replay")
    summary=dict(method=method,time=world.time,metrics=campaign.episode_metrics(world),
        decision_count=len(scheduler.records),dispatch_count=len(world.dispatch_timings),
        trajectory_count=len(record["trajectories"]),config_hash=config.config_hash,
        submitted=sum(row["submitted"] for row in scheduler.records),
        solver_records=[entry for row in scheduler.records for entry in row["solver"]],
        dispatch_statuses=[dict(status=row["status"],solver_entered=row["solver_entered"],
            skip_reason=row["skip_reason"],commit_status=row["commit_status"]) for row in scheduler.records])
    campaign.atomic_json(tmp_path/"summary.json",summary)
    assert summary["metrics"]["conservation"]
    assert summary["submitted"]>0,summary
    if method=="rh_cp_sat":
        assert summary["solver_records"],summary
        assert all(row.get("status") for row in summary["solver_records"]),summary
