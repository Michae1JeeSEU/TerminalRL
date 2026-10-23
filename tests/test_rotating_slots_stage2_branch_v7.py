"""Small local branch codec checks; no formal episode or solver run."""
from dataclasses import replace
from hashlib import sha256
import json

import pytest

from terminal_marl.core.rotating_slots import stage2_campaign as campaign
from terminal_marl.core.rotating_slots.config import load_service_profile
from terminal_marl.core.rotating_slots.replay import BufferedStateJSONLSink, ReplayBundle
from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
from terminal_marl.core.rotating_slots.world import AIRBORNE, RotatingSlotWorld, STAGE2_POLICY_ID


@pytest.mark.parametrize("scenario,counts,rate", (
    ("diagnostic18_rate120_v1", (6,6,6), 120.),
    ("medium24_rate150_v1", (8,8,8), 150.),
    ("stress36_rate180_v1", (14,12,10), 180.)))
def test_frozen_scenarios_same_inputs_for_both_methods(scenario,counts,rate):
    manifest=campaign.frozen_manifest(scenario)
    assert (manifest.seed,manifest.rate_per_hour,manifest.arrival_window)==(71,rate,3600.)
    worlds=[campaign.busy_fixture(manifest,config=load_service_profile("service_half_v1"),
        scenario_profile=scenario,rescue_rule_profile="rescue_need_v7") for _ in campaign.METHODS]
    assert tuple(sum(a.status in AIRBORNE and a.ring==ring for a in worlds[0].aircraft.values())
        for ring in ("outer","middle","inner"))==counts
    payload=lambda world:campaign.canonical(dict(manifest=world.manifest.to_dict(),
        aircraft=[a.to_dict() for a in world.aircraft.values()],slots=world.slots.to_dict(),pads=world.pads.to_dict()))
    assert payload(worlds[0])==payload(worlds[1])
    assert sorted(a.service_end for a in worlds[0].aircraft.values() if a.status=="LANDED_SERVICE")==[15.,30.,45.]
    assert all(.60<=a.soc<=.90 and a.energy_j==0 for a in worlds[0].aircraft.values() if a.status in AIRBORNE)
    assert all(r.release_time<3600. for r in manifest.requests)


@pytest.mark.parametrize("scenario,expected",(
    ("stress36_rate180_v1","10b97225bde7992af59c79857b0c387bde69ec3b1153e36867a9d42406ad4bc6"),
    ("diagnostic18_rate120_v1","b827cae3e8d961a3e94102378c6a9c552bf20f1b12fd30d0445b5fce8cb52743")))
def test_old_manifest_bytes_unchanged(scenario,expected):
    assert campaign.digest(campaign.frozen_manifest(scenario).to_dict())==expected


def test_medium_prepare_materializes_frozen_v7_inputs(tmp_path,monkeypatch):
    profiles=dict(scenario_profile="medium24_rate150_v1",rescue_rule_profile="rescue_need_v7")
    provenance=campaign.source_provenance("service_half_v1",**profiles)
    assert provenance["emergency_policy_id"]=="rescue_need_v7"
    frozen=tmp_path/"frozen.json";campaign.atomic_json(frozen,provenance)
    monkeypatch.setattr(campaign,"environment_lock",lambda:{"test":"medium prepare"})
    monkeypatch.setattr(campaign,"initial_references",lambda *args:[])
    root=tmp_path/"medium"
    result=campaign.prepare_campaign(root,provenance_path=frozen,service_profile="service_half_v1",
        campaign_wall_budget_seconds=21600.,archive_reserve_seconds=1800.,**profiles)
    assert campaign.read_manifest(root/"manifest.json").rate_per_hour==150.
    assert len(campaign.load_json(root/"fixture.json")["aircraft"])==27
    assert result["identity"]["rescue_rule_profile"]=="rescue_need_v7"
    assert result["identity"]["scenario_profile"]=="medium24_rate150_v1"
    assert campaign.campaign_config(root,result).service_profile=="service_half_v1"


@pytest.fixture
def parent_pair(tmp_path,monkeypatch):
    # Freeze real file hashes inside each test, after all local worker edits.
    profiles=dict(scenario_profile="diagnostic18_rate120_v1",rescue_rule_profile="rescue_opportunity_v6")
    old=campaign.source_provenance("service_half_v1",**profiles)
    old_path=tmp_path/"old.json";campaign.atomic_json(old_path,old)
    monkeypatch.setattr(campaign,"environment_lock",lambda:{"test":"local codec fixture"})
    monkeypatch.setattr(campaign,"initial_references",lambda *args:[])
    parent=tmp_path/"parent"
    prepared=campaign.prepare_campaign(parent,provenance_path=old_path,service_profile="service_half_v1",
        campaign_wall_budget_seconds=21600.,archive_reserve_seconds=1800.,**profiles)
    for method in campaign.METHODS:
        directory=parent/method;directory.mkdir()
        world=RotatingSlotWorld(config=campaign.campaign_config(parent,prepared),
            manifest=campaign.read_manifest(parent/"manifest.json"),experiment_policy_id=STAGE2_POLICY_ID,
            state_sink=BufferedStateJSONLSink(directory/"states.jsonl"))
        world.rescue_rule_profile="rescue_opportunity_v6";world.auto_emergency=False
        world.seed_hold("preserved","inner",0,.85,"standard")
        scheduler=NormalSchedulerCallback(method)
        # A genuine physical 0 -> 60 prefix, with cheap recorded dispatches.
        def normal(snapshot,port):
            scheduler.records.append(dict(time=port.time,method=method,wall_s=0.,submitted=0,aircraft=0))
        world.advance(60.,normal,settle_only_at_end=True)
        from terminal_marl.core.rotating_slots.trajectory import generate_final
        plan=generate_final("preserved","inner",0,0,60.,config=world.config)
        assert plan.accepted and world.submit((plan.plan,)).accepted
        # Explicit nonzero inherited cost and pending rescue state are codec data.
        assert world.metrics()["energy_j"]>0
        world.aircraft["preserved"]=replace(world.aircraft["preserved"],
            emergency=replace(world.aircraft["preserved"].emergency,warning=True,rescue_reason="INHERITED_TEST"))
        summary=dict(time=60.,method=method,metrics=campaign.episode_metrics(world),block_wall_s=2.)
        campaign.save_checkpoint(directory/"checkpoint_0060.json",world,scheduler,prepared["identity"],boundary_summary=summary)
        campaign.atomic_json(directory/"summary_0060.json",summary)
        world.state_sink.close()
    new=campaign.source_provenance("service_half_v1",scenario_profile=profiles["scenario_profile"],rescue_rule_profile="rescue_need_v7")
    new_path=tmp_path/"new.json";campaign.atomic_json(new_path,new)
    return parent,tmp_path/"branch",new_path,prepared


def test_branch_roundtrip_preserves_parent_energy_commitments_and_prefix(parent_pair,monkeypatch):
    parent,root,provenance,old=parent_pair
    before={p.relative_to(parent):p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    prepared=campaign.branch_campaign(parent,root,60.,provenance_path=provenance)
    assert prepared["identity"]["rescue_rule_profile"]=="rescue_need_v7"
    assert prepared["policy_interpretation"]=="MIXED_POLICY_CONTINUATION_NOT_V7_FROM_ZERO"
    metadata=campaign.load_json(root/"branch.json")
    assert metadata["parent_budget"]==campaign.load_json(parent/"budget.json")
    assert campaign.load_json(root/"budget.json")["total_s"]==21600.
    for name in ("config.yaml","fixture.json","manifest.json","environment.lock.json"):
        assert (root/name).read_bytes()==(parent/name).read_bytes()
    for method in campaign.METHODS:
        entry=metadata["methods"][method]
        assert entry["inherited_energy_j"]>0
        assert entry["inherited_energy_kwh"]==entry["inherited_energy_j"]/3600000.
        assert entry["parent_checkpoint_bytes_sha256"]==sha256((parent/method/"checkpoint_0060.json").read_bytes()).hexdigest()
        old_scheduler,new_scheduler=NormalSchedulerCallback(method),NormalSchedulerCallback(method)
        old_world=campaign.restore_checkpoint(parent/method/"checkpoint_0060.json",old_scheduler,old["identity"],
            stream_copy=root/method/"inspect_parent.jsonl")
        new_world=campaign.restore_checkpoint(root/method/"checkpoint_0060.json",new_scheduler,prepared["identity"])
        try:
            assert campaign.canonical(old_scheduler.state_dict())==campaign.canonical(new_scheduler.state_dict())
            assert campaign.canonical(old_world.aircraft)==campaign.canonical(new_world.aircraft)
            assert campaign.canonical(old_world.active)==campaign.canonical(new_world.active)
            assert campaign.canonical(old_world.soft)==campaign.canonical(new_world.soft)
            assert campaign.canonical(old_world.dependencies)==campaign.canonical(new_world.dependencies)
            assert entry["retained_commitments"]["active"]
            assert new_world.aircraft["preserved"].emergency.rescue_reason=="INHERITED_TEST"
            assert new_world.time==60. and new_world._next_heartbeat==60.
            assert new_world.metrics()["energy_j"]==entry["inherited_energy_j"]
            assert not any(name in vars(new_world) for name in campaign.DERIVED_RESCUE_FIELDS)
        finally:
            old_world.state_sink.close();new_world.state_sink.close()
        with pytest.raises(ValueError,match="identity/content"):
            campaign.restore_checkpoint(root/method/"checkpoint_0060.json",NormalSchedulerCallback(method),old["identity"])
        def cheap_call(self,port,*args):
            self.records.append(dict(time=port.time,method=self.scheduler.method,wall_s=0.,submitted=0,aircraft=0))
        monkeypatch.setattr(NormalSchedulerCallback,"__call__",cheap_call)
        campaign.run_episode_block(root,method,120.)
        summary=campaign.load_json(root/method/"summary_0120.json")
        assert summary["suffix_energy_j"]==summary["metrics"]["energy_j"]-entry["inherited_energy_j"]
        assert summary["suffix_energy_kwh"]==summary["suffix_energy_j"]/3600000.
        decisions=[json.loads(line) for line in (root/method/"decisions.jsonl").read_text().splitlines()]
        times=[row["time"] for row in decisions]
        assert times[:len(old_scheduler.records)]==[row["time"] for row in old_scheduler.records]
        assert sum(abs(t-60.)<1e-7 for t in times)==1
        assert len({round(t,7) for t in times})==len(times)
        assert times==sorted(times)
        states=[json.loads(line)["time"] for line in (root/method/"states.jsonl").read_text().splitlines()]
        assert len(states)==len(set(states)) and min(states)==0. and max(states)==120.
        assert campaign.load_json(root/method/"summary_0060.json")["policy_segment"]=="INHERITED_V6_PREFIX"
        archive=campaign.export_episode(root,method,120)
        bundle=ReplayBundle.load(archive)
        assert campaign.canonical(bundle.data["stage2_provenance"]["policy_branch"])==campaign.canonical(metadata)
    assert before=={p.relative_to(parent):p.read_bytes() for p in parent.rglob("*") if p.is_file()}


@pytest.mark.parametrize("change",("profile","budget","backend","source","hash"))
def test_branch_rejects_unapproved_change(parent_pair,change):
    parent,root,path,_=parent_pair
    value=campaign.load_json(path)
    if change=="profile": value["rescue_rule_profile"]="legacy_v5"
    elif change=="budget": value["planning_budget"]["solver_seconds"]+=1
    elif change=="backend": value["numeric_backend_hash"]="changed"
    elif change=="source":
        old=campaign.load_json(parent/"provenance.json")
        old["source_files"]["terminal_marl/core/rotating_slots/energy.py"]="older_backend"
        old["source_hash"]=campaign.digest(old["source_files"])
        campaign.atomic_json(parent/"provenance.json",old)
        prepared=campaign.load_json(parent/"campaign.json")
        prepared["identity"]["source_hash"]=old["source_hash"]
        campaign.atomic_json(parent/"campaign.json",prepared)
    else: value["source_hash"]="corrupt"
    campaign.atomic_json(path,value)
    with pytest.raises((ValueError,RuntimeError)):
        campaign.branch_campaign(parent,root,60.,provenance_path=path)
    assert not root.exists()


def test_cli_branch_requires_explicit_parent_boundary_provenance(monkeypatch,capsys):
    from scripts.server import run_rotating_slots_stage2 as cli
    monkeypatch.setattr("sys.argv",["stage2","branch","--output","unused"])
    with pytest.raises(SystemExit) as error: cli.main()
    assert error.value.code==2
    calls=[]
    monkeypatch.setattr(campaign,"branch_campaign",lambda *args,**kwargs:calls.append((args,kwargs)) or {"ok":True})
    monkeypatch.setattr("sys.argv",["stage2","branch","--output","new","--parent-output","old","--from-time","240",
        "--provenance","frozen.json","--campaign-wall-budget-seconds","21600","--archive-reserve-seconds","1800"])
    cli.main()
    assert calls[0][0][2]==240. and calls[0][1]["campaign_wall_budget_seconds"]==21600.
    assert json.loads(capsys.readouterr().out)=={"ok":True}
