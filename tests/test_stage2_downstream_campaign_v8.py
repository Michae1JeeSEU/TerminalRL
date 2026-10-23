"""Frozen algorithm selection and realized per-ring analysis, without long runs."""
import importlib.util
from pathlib import Path

import pytest

from terminal_marl.core.rotating_slots import stage2_campaign as campaign
from terminal_marl.core.rotating_slots import stage2_planning as planning
from terminal_marl.core.rotating_slots.demand import ArrivalManifest
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID


def analyzer():
    path=Path(__file__).resolve().parents[1]/"artifacts/rotating_slots_stage2_20261023/v8_analyze_medium.py"
    spec=importlib.util.spec_from_file_location("v8_analyze_medium",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_freeze_profiles_separate_algorithm_and_unchanged_quality_budget(tmp_path):
    legacy=campaign.source_provenance()
    current=campaign.freeze_provenance(tmp_path/"v8.json", "service_half_v1",
        scenario_profile="medium24_rate150_v1",rescue_rule_profile="rescue_need_v7",
        planning_profile=planning.PLANNING_PROFILE_DOWNSTREAM)
    assert current==campaign.load_json(tmp_path/"v8.json")
    assert current["planning_profile"]==planning.PLANNING_PROFILE_DOWNSTREAM
    assert current["candidate_policy_id"]==planning.candidate_policy_id(planning.PLANNING_PROFILE_DOWNSTREAM)
    assert current["objective_id"]==planning.objective_schema_id(planning.PLANNING_PROFILE_DOWNSTREAM)
    assert current["planning_budget"]==legacy["planning_budget"]
    assert current["planning_budget_profile"]==legacy["planning_budget_profile"]=="quality_v5"
    assert (current["planning_budget"]["soft_seconds"],current["planning_budget"]["solver_seconds"],
        current["planning_budget"]["max_refinement_rounds"])==(30.,8.,64)
    assert campaign.planning_metadata({})["planning_profile"]==planning.PLANNING_PROFILE_LEGACY
    assert "planning_profile" not in legacy
    with pytest.raises(ValueError,match="metadata mismatch"):
        campaign.planning_metadata(dict(current,candidate_policy_id="wrong"))


def test_historical_provenance_identity_structure_and_real_medium_bytes():
    # candidate/objective already existed in legacy provenance, but did not
    # belong to its historical checkpoint identity.
    provenance=dict(source_hash="s",base_config_hash="c",overlay_sha256="o",
        service_profile="service_half_v1",scenario_profile="medium24_rate150_v1",rescue_rule_profile="rescue_need_v7",
        candidate_policy_id=planning.candidate_policy_id(planning.PLANNING_PROFILE_LEGACY),
        objective_id=planning.objective_schema_id(planning.PLANNING_PROFILE_LEGACY))
    expected=dict(policy_id=STAGE2_POLICY_ID,source_hash="s",base_config_hash="c",overlay_sha256="o",manifest_sha256="m",
        service_profile="service_half_v1",scenario_profile="medium24_rate150_v1",rescue_rule_profile="rescue_need_v7")
    assert campaign.checkpoint_identity(provenance,"m")==expected
    real=Path("D:/7_codespace/training_outputs/terminal/stage2_rescue_need_v7_medium_600s_20261023")
    if not real.exists(): pytest.skip("local historical medium archive not mounted")
    old=campaign.load_json(real/"provenance.json")
    manifest=campaign.read_manifest(real/"manifest.json")
    assert campaign.checkpoint_identity(old,campaign.digest(manifest.to_dict()))==campaign.load_json(real/"campaign.json")["identity"]


def test_branch_rejects_downstream_algorithm_before_creating_output(tmp_path,monkeypatch):
    new=campaign.planning_metadata(dict(planning_profile="downstream_service_v8"))
    monkeypatch.setattr(campaign,"campaign_config",lambda *args:None)
    def read(path): return new if Path(path).name=="new.json" else {}
    monkeypatch.setattr(campaign,"load_json",read)
    output=tmp_path/"new_root"
    with pytest.raises(ValueError,match="must start from zero"):
        campaign.branch_campaign(tmp_path/"parent",output,60.,provenance_path=tmp_path/"new.json")
    assert not output.exists()


def test_prepare_run_restore_export_read_frozen_profile(tmp_path,monkeypatch):
    profile=planning.PLANNING_PROFILE_DOWNSTREAM
    frozen=tmp_path/"frozen.json"
    provenance=campaign.source_provenance(planning_profile=profile)
    campaign.atomic_json(frozen,provenance)
    monkeypatch.setattr(campaign,"environment_lock",lambda:{"scope":"serialization only"})
    monkeypatch.setattr(campaign,"initial_references",lambda *args,**kwargs:[])
    monkeypatch.setattr(campaign,"frozen_manifest",ArrivalManifest)
    # Files may be edited by another authorized collaborator during this test.
    monkeypatch.setattr(campaign,"verify_source_provenance",lambda value:campaign.planning_metadata(value))
    def empty_fixture(manifest=None,*,state_sink=None,config=None):
        world=RotatingSlotWorld(manifest=manifest,state_sink=state_sink,config=config,experiment_policy_id=STAGE2_POLICY_ID)
        world.auto_emergency=False
        return world
    monkeypatch.setattr(campaign,"busy_fixture",empty_fixture)
    root=tmp_path/"campaign"
    prepared=campaign.prepare_campaign(root,provenance_path=frozen,planning_profile=profile)
    with pytest.raises(ValueError,match="planning_profile.*frozen provenance"):
        campaign.prepare_campaign(tmp_path/"mismatch",provenance_path=frozen)
    with pytest.raises(ValueError,match="planning_profile.*frozen provenance"):
        campaign.prepare_campaign(tmp_path/"mismatch",provenance_path=frozen)
    assert prepared["planning_profile"]==prepared["identity"]["planning_profile"]==profile
    campaign.campaign_config(root,prepared)
    campaign.run_episode_block(root,"energy_slack",.2)
    campaign.run_episode_block(root,"energy_slack",.4)
    checkpoint=root/"energy_slack/checkpoint_0000.json"
    scheduler=planning.NormalSchedulerCallback(planning_profile=profile)
    world=campaign.restore_checkpoint(checkpoint,scheduler,prepared["identity"])
    assert world.time==.4 and scheduler.planning_profile==profile
    world.state_sink.close()
    with pytest.raises(ValueError,match="planning profile"):
        campaign.restore_checkpoint(checkpoint,planning.NormalSchedulerCallback(),prepared["identity"])
    bundle=Path(campaign.export_episode(root,"energy_slack",0))
    assert campaign.load_json(bundle/"stage2_provenance.json")["planning_profile"]==profile
    summary=campaign.load_json(root/"energy_slack/summary_0000.json")
    assert summary["candidate_policy_id"]==provenance["candidate_policy_id"]
    assert summary["objective_id"]==provenance["objective_id"]
    with pytest.raises(ValueError,match="identity mismatch"):
        campaign.campaign_config(root,dict(prepared,planning_profile=planning.PLANNING_PROFILE_LEGACY,
            candidate_policy_id=planning.candidate_policy_id(planning.PLANNING_PROFILE_LEGACY),
            objective_id=planning.objective_schema_id(planning.PLANNING_PROFILE_LEGACY)))


@pytest.mark.parametrize("action",("freeze-provenance","prepare"))
def test_cli_planning_profile_reaches_freeze_prepare(monkeypatch,tmp_path,action):
    from scripts.server import run_rotating_slots_stage2 as cli
    seen={}
    def capture(*args,**kwargs): seen.update(kwargs);return {}
    monkeypatch.setattr(campaign,"freeze_provenance",capture)
    monkeypatch.setattr(campaign,"prepare_campaign",capture)
    monkeypatch.setattr("sys.argv",["stage2",action,"--output",str(tmp_path/"unused"),
        "--planning-profile","downstream_service_v8"])
    cli.main()
    assert seen["planning_profile"]=="downstream_service_v8"


@pytest.mark.parametrize("action",("run","resume","branch","report","status"))
def test_cli_cannot_override_frozen_algorithm(monkeypatch,action):
    from scripts.server import run_rotating_slots_stage2 as cli
    argv=["stage2",action,"--output","unused","--planning-profile","downstream_service_v8"]
    if action=="branch": argv += ["--parent-output","parent","--from-time","60","--provenance","p.json"]
    monkeypatch.setattr("sys.argv",argv)
    with pytest.raises(SystemExit) as error: cli.main()
    assert error.value.code==2


def test_event_integral_nonuniform_gaps_transfers_reservations_and_censoring():
    module=analyzer()
    fixture=[dict(id="a",ring="outer",status="HOLD_OUTER"),dict(id="b",ring="inner",status="HOLD_INNER")]
    plans={"om":dict(source_ring="outer",target_ring="middle",kind="TRANSFER"),
        "mi":dict(source_ring="middle",target_ring="inner",kind="TRANSFER"),
        "final":dict(source_ring="inner",target_ring=None,kind="FINAL_NORMAL")}
    def event(t,kind,**details): return dict(time=t,type=kind,aircraft_id="a",details=details)
    events=[event(0,"plan_committed",trajectory_hash="om",start_time=3.,end_time=7.),
        event(2,"rescue_warning"),event(3,"maneuver_started",trajectory_hash="om"),event(7,"transfer_completed"),
        event(10,"maneuver_started",trajectory_hash="mi"),event(13,"transfer_completed"),
        event(18,"maneuver_started",trajectory_hash="final"),event(20,"touchdown",emergency=False)]
    segments,transfers,latencies=module.holding_segments(fixture,events,plans,20.)
    seconds={r:sum(stop-start for _,ring,start,stop in segments if ring==r) for r in module.RINGS}
    assert seconds==dict(outer=3.,middle=3.,inner=25.)
    assert transfers=={"outer->middle":1,"middle->inner":1}
    assert latencies[0]["inner_to_final_s"]==5. and latencies[0]["inner_to_touchdown_s"]==7.
    assert latencies[1]["inner_to_final_censored"] and latencies[1]["observed_since_entry_s"]==20.
    # W * s = J; transfer periods and deferred-reservation start are not conflated.
    assert sum(seconds[r]*dict(outer=10.,middle=20.,inner=30.)[r] for r in seconds)==840.
    states=[]
    for t,ring,status in ((0,"outer","HOLD_OUTER"),(3,"outer","TRANSFER_O_M"),(7,"middle","HOLD_MIDDLE"),
        (10,"middle","TRANSFER_M_I"),(13,"inner","HOLD_INNER"),(18,"inner","FINAL_NORMAL"),(20,None,"LANDED_SERVICE")):
        states.append(dict(time=t,aircraft=[dict(id="a",ring=ring,status=status,soc=.3,emergency={}),
            dict(id="b",ring="inner",status="HOLD_INNER",soc=.3,emergency={})]))
    assert module.state_holding_integral(states,20.)==seconds
    queues=module.queues(states,events,segments,20.,dict.fromkeys(module.RINGS,.1))
    assert queues["aircraft_seconds"]["outer"]["warning"]==1.
    assert queues["aircraft_seconds"]["outer"]["lower_bound_shortage"]==3.
    assert queues["terminal"]["inner"]["stable_backlog"]==1
    assert queues["integral_precision"]["max_state_gap_s"]==5.
    assert queues["warning_integral_status"]=="APPROXIMATE_NO_CLEAR_EVENTS"
    # A later observed false flag clears warning; it must not latch forever.
    assert queues["aircraft_seconds"]["middle"]["warning"]==0.


def test_failure_stops_stable_holding_without_recounting_isolation():
    module=analyzer()
    fixture=[dict(id="a",ring="middle",status="HOLD_MIDDLE")]
    events=[dict(time=4.,type="aircraft_energy_failure",aircraft_id="a",details={}),
            dict(time=4.,type="failure_isolated",aircraft_id="a",details={})]
    segments,_,_=module.holding_segments(fixture,events,{},10.)
    assert segments==[("a","middle",0.,4.)]
