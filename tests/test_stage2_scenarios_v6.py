"""Frozen v6 scenario inputs, identity, observation denominators, and CLI plumbing."""
import json
from dataclasses import replace

import pytest

from terminal_marl.core.rotating_slots import stage2_campaign as campaign
from terminal_marl.core.rotating_slots.config import load_service_profile
from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
from terminal_marl.core.rotating_slots.world import AIRBORNE


@pytest.mark.parametrize("scenario,counts,rate",(("stress36_rate180_v1",(14,12,10),180.),
                                               ("diagnostic18_rate120_v1",(6,6,6),120.)))
def test_scenario_inputs_are_frozen_and_method_independent(scenario,counts,rate):
    manifest=campaign.frozen_manifest(scenario)
    assert manifest.seed==71 and manifest.rate_per_hour==rate and manifest.arrival_window==3600.
    assert all(r.release_time<3600. for r in manifest.requests)
    worlds=[campaign.busy_fixture(manifest,config=load_service_profile("service_half_v1"),
        scenario_profile=scenario,rescue_rule_profile="rescue_opportunity_v6") for _ in campaign.METHODS]
    assert [sum(a.status in AIRBORNE and a.ring==ring for a in worlds[0].aircraft.values())
        for ring in ("outer","middle","inner")]==list(counts)
    payload=lambda w: campaign.canonical(dict(manifest=w.manifest.to_dict(),
        aircraft=[a.to_dict() for a in w.aircraft.values()],slots=w.slots.to_dict(),pads=w.pads.to_dict()))
    assert payload(worlds[0])==payload(worlds[1])
    assert sorted(a.service_end for a in worlds[0].aircraft.values() if a.status=="LANDED_SERVICE")==[15.,30.,45.]
    assert all(.60<=a.soc<=.90 for a in worlds[0].aircraft.values() if a.status in AIRBORNE)
    if scenario=="stress36_rate180_v1":
        assert campaign.canonical(manifest.to_dict())==campaign.canonical(campaign.frozen_manifest().to_dict())
        legacy=campaign.busy_fixture(manifest,config=load_service_profile("service_half_v1"))
        assert payload(worlds[0])==payload(legacy)


def test_v6_prepare_materializes_profiles_and_rejects_runtime_identity_changes(tmp_path,monkeypatch):
    profiles=dict(scenario_profile="diagnostic18_rate120_v1",rescue_rule_profile="rescue_opportunity_v6")
    provenance=campaign.source_provenance("service_half_v1",**profiles)
    assert any(p.endswith("/rescue_rules.py") for p in provenance["source_files"])
    path=tmp_path/"frozen.json";campaign.atomic_json(path,provenance)
    monkeypatch.setattr(campaign,"environment_lock",lambda:{"test":"serialization only"})
    monkeypatch.setattr(campaign,"initial_references",lambda world,*args:[])
    root=tmp_path/"pair"
    result=campaign.prepare_campaign(root,provenance_path=path,service_profile="service_half_v1",**profiles)
    assert all(result[name]==result["identity"][name]==value for name,value in profiles.items())
    assert campaign.read_manifest(root/"manifest.json").rate_per_hour==120.
    assert len(campaign.load_json(root/"fixture.json")["aircraft"])==21
    world=campaign.busy_fixture(campaign.read_manifest(root/"manifest.json"),
        config=campaign.campaign_config(root,result),**profiles)
    checkpoint=root/"state.json";scheduler=NormalSchedulerCallback()
    campaign.save_checkpoint(checkpoint,world,scheduler,result["identity"])
    restored=campaign.restore_checkpoint(checkpoint,NormalSchedulerCallback(),result["identity"])
    assert restored.rescue_rule_profile=="rescue_opportunity_v6"
    world.rescue_rule_profile="legacy_v5"
    with pytest.raises(ValueError,match="rescue rule profile"):
        campaign.save_checkpoint(root/"wrong.json",world,scheduler,result["identity"])
    with pytest.raises(ValueError,match="scenario_profile identity"):
        campaign.campaign_config(root,dict(result,scenario_profile="stress36_rate180_v1"))


def test_observations_distinguish_wait_censoring_and_use_explicit_denominators():
    world=campaign.busy_fixture(scenario_profile="diagnostic18_rate120_v1",rescue_rule_profile="rescue_opportunity_v6")
    world.time=20.
    first,second=list(world.aircraft)[:2]
    world.aircraft[first]=replace(world.aircraft[first],emergency=replace(world.aircraft[first].emergency,start_time=15.))
    world.events=[dict(time=5.,type="emergency_triggered",aircraft_id=first,details={}),
        dict(time=10.,type="emergency_triggered",aircraft_id=second,details={}),
        dict(time=3.,type="rescue_warning",aircraft_id=first,details={})]
    pads=campaign.pad_opportunity_samples(world)
    normal=[dict(timings={"candidate_generation_and_single_vehicle":(2.,1.),"safe_progress_generation":(.2,.1),
        "reference_seed_generation":(.3,.1),"geometry_template_screen":(.4,.2),
        "trajectory_validation":(.6,.3),"prefix_validation":(.7,.3),"selected_validation":(.5,.4)},
        solver_wall_s=.7,pad_opportunities=pads,aircraft_choices=[dict(aircraft_id=first,move_candidates=1,
        immediate_move_candidates=0,selected_kind="KEEP",source_reference_status="KNOWN")])]
    observed=campaign.rescue_observations(world,normal,[{},{}])
    assert observed["denominators"]==dict(dispatch_calls=2,normal_calls=1,aircraft_planning_observations=1,
        pad_dispatch_samples=6,ground_idle_offered_pad_samples=3,
        idle_without_active_approach_offered_pad_samples=3,approach_reserved_pad_samples=0)
    assert observed["rescue_wait_completed"]["raw_s"]==[10.]
    assert observed["rescue_wait_censored"]["raw_s"]==[10.]
    assert observed["candidate_exclusive_timing"]["mean"]==pytest.approx(2.9)
    assert observed["safety_validation_timing"]["mean"]==pytest.approx(1.3)
    assert observed["selected_validation_overhead_timing"]["mean"]==.5
    assert observed["warning_aircraft"]==1
    assert observed["pad_reason_counts"]["GROUNDED_SERVICE"]==3
    # JSON output accepts the actual summary payload without nonfinite placeholders.
    assert json.loads(campaign.canonical(observed))["denominators"]["dispatch_calls"]==2


@pytest.mark.parametrize("action",("run","resume"))
@pytest.mark.parametrize("option,value",(("--scenario-profile","stress36_rate180_v1"),
                                         ("--rescue-rule-profile","rescue_opportunity_v6")))
def test_cli_rejects_profile_changes_after_prepare(monkeypatch,action,option,value):
    from scripts.server import run_rotating_slots_stage2 as cli
    monkeypatch.setattr("sys.argv",["stage2",action,"--output","unused",option,value])
    with pytest.raises(SystemExit) as error: cli.main()
    assert error.value.code==2


def test_read_only_legacy_tables_mark_missing_v6_fields_unknown(tmp_path):
    campaign.atomic_json(tmp_path/"campaign.json",{})
    for method in campaign.METHODS:
        campaign.atomic_json(tmp_path/method/"summary_0480.json",dict(metrics=dict(L_normal=1,L_emergency=2,
            L_success=3,F_energy_total=4,N_airborne_pending=5,N_upstream_pending=6)))
    before={p:p.read_bytes() for p in tmp_path.rglob("*.json")}
    tables=campaign.rescue_comparison_tables(tmp_path)
    assert tables["common_time_s"]==480 and len(tables["rows"])==2
    for row in tables["rows"]:
        assert row["total"]==3 and row["warning_aircraft"]=="UNKNOWN" and row["candidate_coverage"]=="UNKNOWN"
    assert before=={p:p.read_bytes() for p in tmp_path.rglob("*.json")}


def test_pad_categories_merge_observed_evidence_without_post_planning_state():
    world=campaign.busy_fixture(scenario_profile="diagnostic18_rate120_v1",rescue_rule_profile="rescue_opportunity_v6")
    pads=campaign.pad_opportunity_samples(world)
    world.time=100.  # Recorded samples must retain their original timestamp.
    row=dict(aircraft_choices=[dict(candidate_pad_ids=[3],selected_pad_ids=[])],submitted=0,
        pad_opportunities={"3":dict(candidate_count=2,submitted_count=0),"5":dict(explicit_conflict_count=1)})
    campaign.annotate_pad_opportunities(pads,row,protected={1})
    assert [r["reason"] for r in pads]==["GROUNDED_SERVICE","EMERGENCY_PROTECTED","GROUNDED_SERVICE",
        "PLANNING_NOT_SUBMITTED","GROUNDED_SERVICE","EXPLICIT_TRAJECTORY_CONFLICT_OBSERVED"]
    assert all(r["time"]==0. for r in pads)
    assert pads[3]["planning_evidence"]["candidate_count"]==2


def test_cli_cpu_pair_is_explicit_and_output_remains_json(monkeypatch,capsys):
    from scripts.server import run_rotating_slots_stage2 as cli
    seen=[]
    monkeypatch.setattr(campaign,"run_campaign",lambda path,**kwargs:seen.append(kwargs) or {"ok":True})
    monkeypatch.setattr(campaign,"load_json",lambda path:{"status":"PARTIAL_REQUESTED_WINDOW"})
    monkeypatch.setattr("sys.argv",["stage2","run","--output","unused","--target","480","--cpus","2","4"])
    cli.main()
    assert seen==[dict(target=480.,cpus=[2,4])]
    assert json.loads(capsys.readouterr().out)=={"ok":True}


def test_actual_busy_fixture_initial_ground_is_excluded_from_low_reserve_touchdowns():
    world=campaign.busy_fixture(campaign.frozen_manifest(),rescue_rule_profile="rescue_opportunity_v6")
    ground={a.id for a in world.aircraft.values() if a.status=="LANDED_SERVICE"}
    assert len(ground)==world.metrics()["initial_ground_count"]==3
    for aid in ground:
        world.aircraft[aid]=replace(world.aircraft[aid],soc=.20)
    observed=campaign.rescue_observations(world,[],[])
    assert observed["initial_ground_excluded"]==3
    assert observed["low_reserve_touchdowns"]==0
    assert len(observed["aircraft"])==36
    assert not ground.intersection(row["aircraft_id"] for row in observed["aircraft"])


def test_ground_idle_and_active_approach_are_separate_and_future_protection_is_counted():
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID
    from terminal_marl.core.rotating_slots.trajectory import generate_final
    world=RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.rescue_rule_profile="rescue_opportunity_v6";world.auto_emergency=False
    world.seed_hold("a","inner",0,.9,"standard")
    generated=generate_final("a","inner",0,0,0.,config=world.config)
    assert generated.accepted and world.submit((generated.plan,)).accepted
    active=campaign.pad_opportunity_samples(world)[0]
    assert active["ground_idle"] and active["approach_reserved"]
    assert not active["idle_without_active_approach"]
    assert active["reason"]=="ACTIVE_APPROACH_RESERVED"
    # A direct future-calendar observation can simultaneously be protected.
    pads=campaign.pad_opportunity_samples(campaign.busy_fixture())
    pads[1]["future_calendar"]=True;pads[1]["reason"]="FUTURE_CALENDAR"
    row=dict(aircraft_choices=[],pad_opportunities={})
    campaign.annotate_pad_opportunities(pads,row,protected={1})
    assert pads[1]["reason"]=="FUTURE_CALENDAR" and pads[1]["emergency_protected"]
    observed=campaign.rescue_observations(world,[row],[])
    assert observed["emergency_protected_pad_samples"]==1
    assert observed["future_calendar_emergency_protected_pad_samples"]==1


@pytest.mark.parametrize("scenario_profile,method",
    [(scenario,method) for scenario in ("stress36_rate180_v1","diagnostic18_rate120_v1") for method in campaign.METHODS],
    ids=[f"{scenario}-{method}" for scenario in ("stress36_rate180_v1","diagnostic18_rate120_v1") for method in campaign.METHODS])
def test_actual_callback_v6_checkpoint_roundtrip(scenario_profile,method,tmp_path):
    """Server integration: one real t=0 dispatch, never a 480-second episode."""
    if method=="rh_cp_sat": pytest.importorskip("ortools.sat.python.cp_model")
    from dataclasses import asdict
    world=campaign.busy_fixture(campaign.frozen_manifest(scenario_profile),
        config=load_service_profile("service_half_v1"),scenario_profile=scenario_profile,
        rescue_rule_profile="rescue_opportunity_v6")
    scheduler=NormalSchedulerCallback(method,budget_profile="quality_v5")
    world.admission_callback=campaign.shared_fifo_admission

    def normal(snapshot,port):
        pads=campaign.pad_opportunity_samples(world)
        protected={p.pad_id for p in (*world.active.values(),*world.soft.values()) if p.kind=="FINAL_EMERGENCY"}
        result=scheduler(port)
        campaign.annotate_pad_opportunities(pads,scheduler.records[-1],protected)
        return result

    world.advance(0.,normal,settle_only_at_end=False)
    assert world.time==0. and not world.failed and not world.terminated_invalid
    assert len(scheduler.records)==len(world.dispatch_timings)==1
    record=scheduler.records[0]
    assert record["time"]==0. and record["method"]==method and record["budget_profile"]=="quality_v5"
    assert record["planning_budget"]["soft_seconds"]==30.
    assert record["planning_budget"]["solver_seconds"]==8.
    assert record["planning_budget"]["max_refinement_rounds"]==64
    assert len(record["aircraft_choices"])>0 and len(record["pad_opportunities"])==6
    assert "candidate_generation_and_single_vehicle" in record["timings"]
    assert all(len(timing)==2 and all(value>=0. for value in timing) for timing in record["timings"].values())
    assert all("planning_evidence" in pad and "ground_idle" in pad and "emergency_protected" in pad
               for pad in record["pad_opportunities"])
    observed=campaign.rescue_observations(world,scheduler.records,world.dispatch_timings)
    assert observed["denominators"]["normal_calls"]==1 and observed["denominators"]["pad_dispatch_samples"]==6
    assert observed["candidate_exclusive_timing"]["count"]==1
    # allow_nan=False is deliberate: NaN or infinity in any actual record fails.
    json.dumps(dict(decision=record,dispatch=world.dispatch_timings,observations=observed),allow_nan=False)
    assert json.loads(campaign.canonical(observed))["initial_ground_excluded"]==3

    identity=campaign.checkpoint_identity(dict(source_hash="actual_callback_test",overlay_sha256="test_overlay",
        base_config_hash=world.config.config_hash,service_profile="service_half_v1",
        scenario_profile=scenario_profile,rescue_rule_profile="rescue_opportunity_v6"),campaign.digest(world.manifest.to_dict()))
    emergency_before={aid:asdict(a.emergency) for aid,a in world.aircraft.items()}
    expected_fields={"warning","normal_status","normal_margin","normal_path_soc","normal_path_wait_seconds",
        "normal_estimate_epoch","normal_estimate_source","rescue_wait_seconds","rescue_reason","rescue_slack_seconds"}
    assert all(expected_fields<=set(state) for state in emergency_before.values())
    prefix_before=dict(active={aid:p.to_dict() for aid,p in world.active.items()},
        soft={aid:p.to_dict() for aid,p in world.soft.items()},
        previous={aid:[p.to_dict() for p in plans] for aid,plans in scheduler.provider.previous.items()})
    checkpoint=tmp_path/"actual_callback_checkpoint.json"
    campaign.save_checkpoint(checkpoint,world,scheduler,identity)
    restored_scheduler=NormalSchedulerCallback(method,budget_profile="quality_v5")
    restored=campaign.restore_checkpoint(checkpoint,restored_scheduler,identity)
    assert restored.time==world.time==0. and restored.rescue_rule_profile=="rescue_opportunity_v6"
    assert restored.config.config_hash==world.config.config_hash
    assert {aid:asdict(a.emergency) for aid,a in restored.aircraft.items()}==emergency_before
    prefix_after=dict(active={aid:p.to_dict() for aid,p in restored.active.items()},
        soft={aid:p.to_dict() for aid,p in restored.soft.items()},
        previous={aid:[p.to_dict() for p in plans] for aid,plans in restored_scheduler.provider.previous.items()})
    assert campaign.canonical(prefix_after)==campaign.canonical(prefix_before)
    assert campaign.canonical(restored_scheduler.records)==campaign.canonical(scheduler.records)
    assert campaign.canonical(restored.dispatch_timings)==campaign.canonical(world.dispatch_timings)
