"""Explicit service scenario, preserving the frozen physical/safety payload."""
from pathlib import Path

import pytest
import yaml

from terminal_marl.core.rotating_slots.config import (
    CANONICAL_CONFIG_HASH, DEFAULT_CONFIG, SERVICE_HALF_CONFIG_HASH,
    load_config, load_service_profile, service_duration,
)
from terminal_marl.core.rotating_slots import stage2_campaign as campaign
from terminal_marl.core.rotating_slots.demand import ArrivalManifest
from terminal_marl.core.rotating_slots.replay import ReplayBundle
from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
from terminal_marl.core.rotating_slots.trajectory import generate_final
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID


def test_two_exact_frozen_profiles_and_unchanged_physical_energy_profile(tmp_path):
    baseline=load_config(); half=load_service_profile("service_half_v1")
    assert baseline.config_hash==CANONICAL_CONFIG_HASH
    assert half.config_hash==SERVICE_HALF_CONFIG_HASH!=baseline.config_hash
    assert baseline.profile_hash==half.profile_hash
    assert baseline.service_profile=="baseline" and half.service_profile=="service_half_v1"
    expected=baseline.to_dict()
    for name,duration in (("short",30.),("standard",60.),("long",120.)):
        expected["pads"]["service_classes"][name]["duration_s"]=duration
        assert service_duration(half,name)==duration
    assert half.to_dict()==expected
    # Both complete profiles still reject any unrelated physical/safety drift.
    for config in (baseline,half):
        changed=config.to_dict(); changed["safety"]["hard_separation_m"]=79.
        path=tmp_path/(config.service_profile+".yaml")
        path.write_text(yaml.safe_dump(changed),encoding="utf-8")
        with pytest.raises(ValueError,match="noncanonical"): load_config(path)
    changed=half.to_dict();changed["pads"]["service_classes"]["standard"]["duration_s"]=61.
    path=tmp_path/"unapproved.yaml";path.write_text(yaml.safe_dump(changed),encoding="utf-8")
    with pytest.raises(ValueError,match="frozen"): load_config(path)
    assert load_config().config_hash==CANONICAL_CONFIG_HASH


@pytest.mark.parametrize("emergency",(False,True))
@pytest.mark.parametrize("profile,duration",(("baseline",120.),("service_half_v1",60.)))
def test_config_drives_normal_and_emergency_calendar_traffic_and_touchdown(profile,duration,emergency):
    config=load_service_profile(profile)
    world=RotatingSlotWorld(config=config,experiment_policy_id=STAGE2_POLICY_ID)
    world.auto_emergency=False
    world.seed_hold("a","inner",0,.9,"standard",emergency=emergency)
    result=generate_final("a","inner",0,0,0.,emergency=emergency,config=config)
    assert result.accepted
    plan=result.plan
    assert world.submit((plan,)).accepted
    assert world._traffic(world.aircraft["a"]).end_time==plan.end_time+duration
    reservation=next(r for r in world.pads.reservations if r.aircraft_id=="a")
    assert reservation.end==plan.end_time+duration
    assert reservation.kind==("emergency" if emergency else "final")
    world.advance(plan.end_time,settle_only_at_end=True)
    assert world.aircraft["a"].service_end==plan.end_time+duration
    assert world.aircraft["a"].status=="LANDED_SERVICE"
    assert world._traffic(world.aircraft["a"]).end_time==plan.end_time+duration
    world.advance(plan.end_time+duration,settle_only_at_end=True)
    assert world.aircraft["a"].status=="CLEARED"


def test_half_fixture_preserves_manifest_soc_classes_and_airborne_state():
    manifest=campaign.frozen_manifest()
    baseline=campaign.busy_fixture(manifest)
    half=campaign.busy_fixture(manifest,config=load_service_profile("service_half_v1"))
    assert baseline.manifest.to_dict()==half.manifest.to_dict()==manifest.to_dict()
    for aid,aircraft in baseline.aircraft.items():
        other=half.aircraft[aid]
        if aircraft.status=="LANDED_SERVICE":
            assert other.service_end==aircraft.service_end/2
        else:
            # Trajectory hashes include the experiment configuration hash.
            old,new=aircraft.to_dict(),other.to_dict()
            old.pop("trajectory_hash");new.pop("trajectory_hash")
            assert old==new
    assert sorted(a.service_end for a in half.aircraft.values() if a.status=="LANDED_SERVICE")==[15.,30.,45.]


def test_prepare_checkpoint_and_export_roundtrip_exact_config(tmp_path,monkeypatch):
    config=load_service_profile("service_half_v1")
    provenance=campaign.source_provenance("service_half_v1")
    assert provenance["candidate_policy_id"]=="shared_multistep_columns_v6_reuse_diversity"
    assert provenance["planning_budget_profile"]=="quality_v5"
    assert provenance["planning_budget"]==dict(soft_seconds=30.,watchdog_seconds=60.,candidate_seconds=12.,
        hint_seconds=3.,solver_seconds=8.,solver_reserve_seconds=8.,max_per_aircraft=64,max_attempts=2048,
        max_refinement_rounds=64,hint_validation_reserve_seconds=.5)
    assert provenance["emergency_policy_id"]=="energy_urgency_yield_v5"
    assert provenance["objective_id"]=="reliability_throughput_progress_tail_v2"
    frozen=tmp_path/"frozen.json";campaign.atomic_json(frozen,provenance)
    campaign.verify_source_provenance(provenance)
    bad=dict(provenance,original_config_hash="wrong")
    with pytest.raises(RuntimeError,match="configuration/contract"): campaign.verify_source_provenance(bad)
    # Environment/initial reference computation are outside this serialization test.
    monkeypatch.setattr(campaign,"environment_lock",lambda:{"test":"serialization only"})
    monkeypatch.setattr(campaign,"initial_references",lambda *args,**kwargs:[])
    monkeypatch.setattr(campaign,"frozen_manifest",ArrivalManifest)
    def empty_fixture(manifest=None,*,state_sink=None,config=None):
        world=RotatingSlotWorld(config=config,manifest=manifest,state_sink=state_sink,experiment_policy_id=STAGE2_POLICY_ID)
        world.auto_emergency=False
        return world
    monkeypatch.setattr(campaign,"busy_fixture",empty_fixture)
    root=tmp_path/"campaign"
    prepared=campaign.prepare_campaign(root,provenance_path=frozen,service_profile="service_half_v1",
        campaign_wall_budget_seconds=3600.,archive_reserve_seconds=300.)
    budget_before=(root/"budget.json").read_bytes()
    budget=campaign.CampaignBudget(root/"budget.json")
    assert budget.data["total_s"]==3600. and budget.data["compute_stop_s"]==3300.
    with pytest.raises(ValueError,match="already prepared"):
        campaign.prepare_campaign(root,campaign_wall_budget_seconds=7200.,archive_reserve_seconds=300.)
    assert (root/"budget.json").read_bytes()==budget_before
    assert prepared["service_profile"]==prepared["identity"]["service_profile"]=="service_half_v1"
    assert (root/"config.yaml").read_bytes()==Path(config.path).read_bytes()
    assert campaign.campaign_config(root,prepared).config_hash==config.config_hash
    campaign.run_episode_block(root,"energy_slack",.4)
    directory=root/"energy_slack"
    checkpoint=directory/"checkpoint_0000.json"
    restored=campaign.restore_checkpoint(checkpoint,NormalSchedulerCallback(),prepared["identity"])
    assert restored.config.config_hash==config.config_hash and restored.time==.4
    with pytest.raises(ValueError,match="configuration identity"):
        campaign.save_checkpoint(directory/"wrong.json",restored,NormalSchedulerCallback(),
            dict(prepared["identity"],base_config_hash=CANONICAL_CONFIG_HASH))
    restored.state_sink.close()
    bundle_path=campaign.export_episode(root,"energy_slack",0)
    ReplayBundle.validate(bundle_path)
    bundle=ReplayBundle.load(bundle_path)
    assert bundle.config.config_hash==config.config_hash
    assert (Path(bundle_path)/"config.yaml").read_bytes()==Path(config.path).read_bytes()
    assert load_config().config_hash==CANONICAL_CONFIG_HASH
    (root/"config.yaml").write_bytes(DEFAULT_CONFIG.read_bytes())
    with pytest.raises(ValueError,match="identity/content"): campaign.campaign_config(root,prepared)


def test_prepare_rejects_profile_provenance_mismatch(tmp_path,monkeypatch):
    path=tmp_path/"provenance.json"
    campaign.freeze_provenance(path,"baseline")
    monkeypatch.setattr(campaign,"environment_lock",lambda:{})
    with pytest.raises(ValueError,match="service profile"):
        campaign.prepare_campaign(tmp_path/"wrong",provenance_path=path,service_profile="service_half_v1")


def test_explicit_budget_persists_and_rejects_reinitialization(tmp_path):
    path=tmp_path/"budget.json"
    first=campaign.CampaignBudget(path,100.,clock=lambda:10000.,total_seconds=3600.,archive_reserve_seconds=300.)
    assert first.data["started_epoch"]==9900. and first.compute_remaining==pytest.approx(3200.,abs=.1)
    before=path.read_bytes()
    with pytest.raises(ValueError,match="first creation"):
        campaign.CampaignBudget(path,total_seconds=3600.,archive_reserve_seconds=300.)
    assert path.read_bytes()==before
    restored=campaign.CampaignBudget(path,clock=lambda:10100.)
    assert restored.data["total_s"]==3600. and restored.compute_remaining==pytest.approx(3100.,abs=.1)
    default=campaign.CampaignBudget(tmp_path/"default.json")
    assert default.data["total_s"]==36000. and default.data["compute_stop_s"]==34200.
    for i,(total,reserve) in enumerate(((0,1),(1,0),(10,10),(10,11),(float("inf"),1),(10,float("nan")))):
        with pytest.raises(ValueError,match="finite"):
            campaign.CampaignBudget(tmp_path/f"invalid{i}.json",total_seconds=total,archive_reserve_seconds=reserve)


@pytest.mark.parametrize("action",("run","resume"))
@pytest.mark.parametrize("option,value",(("--campaign-wall-budget-seconds","3600"),("--archive-reserve-seconds","300"),
                                         ("--service-profile","service_half_v1")))
def test_cli_run_resume_cannot_override_frozen_budget_or_profile(monkeypatch,action,option,value):
    from scripts.server import run_rotating_slots_stage2 as cli
    monkeypatch.setattr("sys.argv",["stage2",action,"--output","unused",option,value])
    with pytest.raises(SystemExit) as error: cli.main()
    assert error.value.code==2
