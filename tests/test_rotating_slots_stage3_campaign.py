"""Small protocol checks; no formal campaign or training starts here."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from terminal_marl.core.rotating_slots.stage3_learning.campaign import CampaignLedger,contract_config,source_files
from terminal_marl.core.rotating_slots.stage3_learning.data import (connected_splits,freeze_split,auxiliary_labels,
    validate_independent_initial_states,StratifiedProbe)
from terminal_marl.core.rotating_slots.stage3_learning.labeling import derive_pairs,conditional_catalog
from terminal_marl.core.rotating_slots.stage3_learning.evaluation import rotated_arms,open_test_gate,observe_executed_holding
from terminal_marl.core.rotating_slots.stage2_campaign import atomic_json,digest,busy_fixture


def descriptor(name,parent=None,stage="alternative"):
    return dict(descriptor_id=name,snapshot_id="snapshot",source_node_id="node",aircraft_id="a",parent_prefix_id=parent,
        construction_stage=stage,source_ring="outer",target_kind="next_ring_slot",start_offset_s=0.)


def test_masks_unknown_timeout_and_unselected_are_not_negative():
    facts=[dict(descriptor_id=k,outcome=k,wall_s=.01,censored=k=="TIMEOUT") for k in ("UNKNOWN","TIMEOUT","NOT_ATTEMPTED","SINGLE_VALID","SINGLE_INVALID_CERTIFIED","FIXED_BACKGROUND_CONFLICT")]
    labels={r["descriptor_id"]:r for r in auxiliary_labels(facts)}
    assert labels["SINGLE_VALID"]["validity"]==1.
    assert labels["SINGLE_INVALID_CERTIFIED"]["validity"]==0.
    assert all(not labels[k]["validity_mask"] for k in ("UNKNOWN","TIMEOUT","NOT_ATTEMPTED","FIXED_BACKGROUND_CONFLICT"))
    assert labels["UNKNOWN"]["cost_mask"]
    assert not labels["TIMEOUT"]["cost_mask"]


def test_unsupported_valid_not_a_negative_and_parent_stage_exact():
    descriptors=[descriptor("supported"),descriptor("valid"),descriptor("invalid"),descriptor("different_parent",parent="p"),descriptor("other_stage",stage="progress")]
    solution=dict(solution_id="s",certified=True,descriptor_ids=["supported"])
    facts=[dict(descriptor_id="valid",outcome="SINGLE_VALID",validity_label_mask=True),
        *[dict(descriptor_id=k,outcome="SINGLE_INVALID_CERTIFIED",validity_label_mask=True) for k in ("invalid","different_parent","other_stage")]]
    pairs=derive_pairs(descriptors,facts,[solution],context_id="snapshot",root_id="r",split="train",fingerprint={"objective_id":"objective"})
    assert [(p["preferred_descriptor"],p["other_descriptor"]) for p in pairs]==[("supported","invalid")]
    assert pairs[0]["context_id"]=="node"


def test_conditional_comparison_requires_certification_strict_key_and_same_catalog():
    descriptors=[descriptor("a"),descriptor("b")]
    base=dict(solution_id="s",certified=True,descriptor_ids=[],required_descriptor="a",catalog_id="catalog",budget_settings={"seconds":5},objective=[0,1,0,0,0])
    other=dict(base,solution_id="t",required_descriptor="b",objective=[0,2,0,0,0])
    kwargs=dict(context_id="snapshot",root_id="r",split="train",fingerprint={"objective_id":"objective"})
    assert len(derive_pairs(descriptors,[],[],conditional_attempts=[base,other],**kwargs))==1
    for changed in (dict(other,certified=False),dict(other,objective=base["objective"]),dict(other,catalog_id="other"),dict(other,budget_settings={"seconds":4})):
        assert derive_pairs(descriptors,[],[],conditional_attempts=[base,changed],**kwargs)==[]


def test_conditional_catalog_retains_bound_downstream_and_forces_descendant():
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateCatalog
    keep=SimpleNamespace(signature=("a","KEEP"));move=SimpleNamespace(signature=("a","MOVE"))
    downstream=object();catalog=CandidateCatalog({"a":(keep,move)},{"a":0.},{},(),downstream=downstream)
    filtered=conditional_catalog(catalog,"d",{move.signature:("d",)})
    assert filtered.columns["a"]==(move,)
    assert filtered.downstream is downstream
    assert catalog.columns["a"]==(keep,move)


def test_split_transitive_component_and_development_seed_guard():
    rows=[dict(ancestry_root_id="a",manifest_lineage_id="m",checkpoint_parent_id=None,split="train"),
        dict(ancestry_root_id="b",manifest_lineage_id="m",checkpoint_parent_id="c",split="train"),
        dict(ancestry_root_id="d",manifest_lineage_id="n",checkpoint_parent_id="c",split="test")]
    with pytest.raises(ValueError,match="crosses"):connected_splits(rows)
    with pytest.raises(ValueError,match="seed71"):connected_splits([dict(rows[0],seed=71)])
    with pytest.raises(ValueError,match="identical t0"):
        validate_independent_initial_states([dict(rows[0],root_id="r1",initial_state_hash="same"),dict(rows[0],root_id="r2",initial_state_hash="same")])


def test_seed_collision_allocation_frozen_exact_24():
    import yaml
    from terminal_marl.core.rotating_slots.stage3_learning.campaign import PACKAGE
    plan=yaml.safe_load((PACKAGE/"stage3_dataset_plan_v1_0.yaml").read_text(encoding="utf-8"))
    split=freeze_split(plan,{7301,7302})
    assert len({r["seed"] for r in split["roots"]})==24
    assert all(r["seed"] not in (71,7301,7302) for r in split["roots"])
    assert [r["split"] for r in split["roots"]].count("train")==16
    assert split==freeze_split(plan,{7301,7302})


def test_seed_factory_defaults_preserved_and_new_roots_differ():
    from terminal_marl.core.rotating_slots.demand import ArrivalManifest
    old=busy_fixture(scenario_profile="diagnostic18_rate120_v1")
    default=busy_fixture(scenario_profile="diagnostic18_rate120_v1",initial_seed=None)
    new=busy_fixture(scenario_profile="diagnostic18_rate120_v1",initial_seed=7301)
    assert [a.to_dict() for a in old.aircraft.values()]==[a.to_dict() for a in default.aircraft.values()]
    assert [a.soc for a in old.aircraft.values()]!=[a.soc for a in new.aircraft.values()]
    assert [a.slot for a in old.aircraft.values()]==[a.slot for a in new.aircraft.values()]
    assert all(.6<=a.soc<=.9 for a in new.aircraft.values())
    with pytest.raises(ValueError):ArrivalManifest.generate(seed=7301,rate_per_hour=120)
    assert ArrivalManifest.generate(seed=7301,rate_per_hour=120,allow_new_seed=True)==ArrivalManifest.generate(seed=7301,rate_per_hour=120,allow_new_seed=True)
    assert ArrivalManifest.generate(seed=71,rate_per_hour=120)==ArrivalManifest.generate(seed=71,rate_per_hour=120,allow_new_seed=True)


def test_budget_resume_credits_and_explicit_edit_pause(tmp_path):
    clock=[100.];path=tmp_path/"ledger.json"
    ledger=CampaignLedger(path,clock=lambda:clock[0]);ledger.credit("external",5.)
    clock[0]+=10;ledger.pause();clock[0]+=100
    resumed=CampaignLedger(path,clock=lambda:clock[0]);assert resumed.elapsed==15.
    resumed.resume();clock[0]+=2;assert resumed.elapsed==17.
    resumed.credit("external",5.);assert resumed.elapsed==17.
    with pytest.raises(ValueError):resumed.credit("external",6.)
    with resumed.phase("training"):clock[0]+=20
    assert resumed.phase_used("training")==20.
    assert resumed.remaining("training")==1780.


def test_budget_interrupted_active_phase_charged_on_resume(tmp_path):
    clock=[0.];path=tmp_path/"ledger.json";ledger=CampaignLedger(path,clock=lambda:clock[0])
    ledger.data["active_phase"]={"phase":"training","started_epoch":0.};ledger.persist()
    clock[0]=100.;resumed=CampaignLedger(path,clock=lambda:clock[0])
    with resumed.phase("training"):clock[0]+=10
    assert resumed.phase_used("training")==110.


def test_rotation_and_test_gate_never_opens_before_freeze(tmp_path):
    assert len({rotated_arms(i)[0] for i in range(3)})==3
    with pytest.raises(ValueError):open_test_gate(tmp_path,{"model_frozen":False})
    atomic_json(tmp_path/"split_manifest.json",{"split_manifest_id":"split"})
    atomic_json(tmp_path/"model/feature_normalizer.json",{})
    card={"model_frozen":True,"model_hash":"model"}
    assert open_test_gate(tmp_path,card)==open_test_gate(tmp_path,card)
    with pytest.raises(ValueError):open_test_gate(tmp_path,dict(card,model_hash="other"))


def test_offline_stratified_probe_probability_and_repeatability():
    pool=[descriptor(str(i)) for i in range(5)]
    a=StratifiedProbe();b=StratifiedProbe()
    assert a.select(pool,pool[0],4)==b.select(pool,pool[0],4)
    selected,metadata=a.select(pool,pool[0],9)
    assert selected in pool and metadata["selection_probability"]==.2
    assert not a.select(pool,pool[0],0)[1]["offline_probe"]


def test_executed_holding_observation_changes_no_energy_or_world_fields():
    class World:
        def __init__(self):self.aircraft={"a":SimpleNamespace(ring="outer",holding_time=0.,energy_j=0.)}
        def _advance_energy(self,end):
            self.aircraft["a"].holding_time+=end;self.aircraft["a"].energy_j+=end*100.
    world=World();fields=set(vars(world));totals,restore=observe_executed_holding(world)
    try:world._advance_energy(2.)
    finally:restore()
    assert totals=={"outer":200.,"middle":0.,"inner":0.}
    assert world.aircraft["a"].energy_j==200. and set(vars(world))==fields


def test_serialization_no_pickle_and_physical_sources_covered():
    from terminal_marl.core.rotating_slots.config import load_service_profile
    files=source_files()
    assert Path(load_service_profile("service_half_v1").path) in files
    assert Path(load_service_profile("baseline").path) in files
    assert contract_config()["compute"]["campaign_wall_seconds"]==14400


def test_replay_proposal_attempt_commit_distinction(tmp_path):
    from terminal_marl.core.rotating_slots.config import load_config
    from terminal_marl.core.rotating_slots.replay import ReplayBundle
    from terminal_marl.visualization.rotating_slots_controller import ReplayController
    source=dict(config_hash=load_config().config_hash,states=[dict(time=0.,aircraft=[],slots=[],pads=[],metrics={})],
        trajectories={},manifest={"requests":[]},events=[],metrics={},rejections=[],stage3_trace=[
            dict(record_type="WorkDescriptor",descriptor_id="proposal"),dict(record_type="FactualResult",descriptor_id="attempt")])
    bundle=ReplayBundle.write(tmp_path/"replay",record=source,code_commit="test")
    overlay=ReplayController(bundle).overlays()
    assert len(overlay["stage3_proposals"])==1 and len(overlay["stage3_attempts"])==1
    assert overlay["current_commits"]==[]
    assert bundle.data["visual_acceptance"]["researcher_visual_acceptance"]=="PENDING"


def test_serialized_training_export_groups_and_sealed_test_poison(tmp_path):
    from terminal_marl.core.rotating_slots.stage3_learning.campaign import build_training_dataset,PACKAGE
    from terminal_marl.core.rotating_slots.stage3_learning.learning import _groups
    from terminal_marl.core.rotating_slots.stage3_learning.features import QUERY_DIM,SET_DIMS,SCHEMA_ID
    schema=json.loads((PACKAGE/"stage3_data_schema_v1_0.json").read_text(encoding="utf-8"))
    fingerprint={key:key for key in schema["$defs"]["Fingerprint"]["required"]};fingerprint["model_id"]="NO_MODEL"
    roots=[dict(root_id="train",split="train"),dict(root_id="val",split="validation"),dict(root_id="test",split="test")]
    atomic_json(tmp_path/"split_manifest.json",dict(roots=roots));atomic_json(tmp_path/"run_binding.generated.json",dict(fingerprint=fingerprint))
    for root in roots[:2]:
        rid=root["root_id"];context="node:"+rid
        desc=[dict(descriptor(rid+suffix),source_node_id=context) for suffix in ("a","b")]
        facts=[dict(record_type="FactualResult",descriptor_id=d["descriptor_id"],context_id=context,ancestry_root_id=rid,
            fingerprint=fingerprint,outcome="SINGLE_VALID",wall_s=.1,cpu_s=.1,censored=False,cache_state={},features_before_query_ref="f:"+d["descriptor_id"]) for d in desc]
        pair=dict(record_type="PreferencePair",context_id=context,preferred_descriptor=desc[0]["descriptor_id"],other_descriptor=desc[1]["descriptor_id"],
            label_source="bounded_conditional_certified_objective_preference",objective_id=fingerprint["objective_id"],label_policy_id="bounded_reference_supported_pair_v1_0",ancestry_root_id=rid,split=root["split"],evidence_refs=["s1","s2"])
        atomic_json(tmp_path/"labels"/rid/"labels.json",dict(status="LABELED",context_id=context,fingerprint=fingerprint,facts=facts,preference_pairs=[pair]))
        raw=tmp_path/"raw"/rid;raw.mkdir(parents=True)
        (raw/"descriptors.jsonl").write_text("\n".join(json.dumps(d) for d in desc),encoding="utf-8")
        encoded=[dict(record_type="FeaturesBeforeQuery",descriptor_id=d["descriptor_id"],features=dict(descriptor_id=d["descriptor_id"],schema_id=SCHEMA_ID,
            query=[0.]*QUERY_DIM,sets={name:[] for name in SET_DIMS})) for d in desc]
        (raw/"features.jsonl").write_text("\n".join(json.dumps(e) for e in encoded),encoding="utf-8")
    sealed=tmp_path/"labels/test/labels.json";sealed.parent.mkdir(parents=True);sealed.write_text("SEALED_POISON_MUST_NOT_BE_PARSED",encoding="utf-8")
    dataset=build_training_dataset(tmp_path)
    groups=_groups(json.loads((tmp_path/"training_dataset.json").read_text(encoding="utf-8")),fingerprint)
    assert len(groups)==2 and all(len(g["pairs"])==1 for g in groups)
    assert {g["context"]["split"] for g in groups}=={"train","validation"}
    assert all(c["ancestry_root_id"]!="test" for c in dataset["contexts"])


def test_quality_flags_fewer_normal_even_with_better_lexicographic_total():
    from terminal_marl.core.rotating_slots.stage3_learning.evaluation import quality_differences
    control=dict(F_energy_total=0,L_normal=3,L_success=3,L_emergency=0,K_episode=[0,-3,0,100,10])
    learned=dict(F_energy_total=0,L_normal=2,L_success=4,L_emergency=2,K_episode=[0,-4,0,90,10])
    delta=quality_differences(control,learned)
    assert delta["quality_regression_observed"] and not delta["lexicographic_episode_worse"]
    assert set(delta["forbidden_deterioration_fields"])=={"L_normal","L_emergency"}


def test_parallel_root_coverage_two_distinct_cores_and_no_duplicate_jobs(tmp_path,monkeypatch):
    import threading
    import time
    from terminal_marl.core.rotating_slots.stage3_learning import campaign
    monkeypatch.setattr(campaign,"resource_probe",lambda:dict(free_physical_cores=[2,4]))
    active=set();seen=[];lock=threading.Lock();maximum=[0]
    def fake_run(ledger,phase,job_id,command,directory,*,cap=None,parallel=False):
        payload=json.loads(Path(command[-1]).read_text(encoding="utf-8"));cpu=payload["cpu"]
        with lock:
            assert cpu not in active;active.add(cpu);maximum[0]=max(maximum[0],len(active));seen.append(job_id)
        time.sleep(.005)
        with lock:active.remove(cpu)
        return dict(job_id=job_id,status="completed",exit_code=0)
    monkeypatch.setattr(campaign,"run_subprocess",fake_run)
    ledger=CampaignLedger(tmp_path/"ledger.json")
    roots=[dict(root_id=f"r{i}") for i in range(5)]
    result=campaign.run_root_jobs(tmp_path,ledger,"collect",roots)
    assert len(result)==5 and len(set(seen))==5 and maximum[0]==2
    assert {r["root_id"] for r in result}=={r["root_id"] for r in roots}
    assert ledger.data["parallel_batches"][-1]["cores"]==[2,4]


def test_budget_completed_job_resume_does_not_spawn_again(tmp_path,monkeypatch):
    from terminal_marl.core.rotating_slots.stage3_learning import campaign
    ledger=CampaignLedger(tmp_path/"ledger.json")
    ledger.data["jobs"]["done"]={"status":"completed","exit_code":0};ledger.persist()
    def forbidden(*args,**kwargs):raise AssertionError("completed job spawned twice")
    monkeypatch.setattr(campaign.subprocess,"Popen",forbidden)
    assert campaign.run_subprocess(ledger,"training","done",["never"],tmp_path)["exit_code"]==0


def test_reference_support_cap_is_separate_from_conditional_observations():
    descriptors=[descriptor(str(i)) for i in range(11)]
    references=[dict(solution_id="reference:"+str(i),certified=True,descriptor_ids=[str(i)]) for i in range(8)]
    conditional=[dict(solution_id="conditional:"+str(i),certified=True,descriptor_ids=[str(i)],required_descriptor=str(i),
        catalog_id="catalog",budget_settings={"seconds":5},objective=[0,i,0,0,0]) for i in (8,9)]
    facts=[dict(descriptor_id="10",outcome="SINGLE_INVALID_CERTIFIED",validity_label_mask=True)]
    kwargs=dict(context_id="snapshot",root_id="r",split="train",fingerprint={"objective_id":"objective"})
    pairs=derive_pairs(descriptors,facts,references,conditional_attempts=conditional,**kwargs)
    supported=[p for p in pairs if p["label_source"]=="supported_vs_certified_single_invalid"]
    assert {p["preferred_descriptor"] for p in supported}==set(map(str,range(8)))
    assert all("conditional" not in ref for p in supported for ref in p["evidence_refs"])
    assert any(p["preferred_descriptor"]=="8" and p["other_descriptor"]=="9" for p in pairs)
    with pytest.raises(ValueError,match="solution cap"):
        derive_pairs(descriptors,facts,references+[conditional[0]],**kwargs)
    with pytest.raises(ValueError,match="resolve cap"):
        derive_pairs(descriptors,facts,references,conditional_attempts=conditional*5,**kwargs)


def test_coverage_reports_distinct_proposals_queries_validity_support_by_ring_stage():
    from terminal_marl.core.rotating_slots.stage3_learning.data import summarize_coverage
    a=descriptor("a");b=dict(descriptor("b",stage="progress"),source_ring="inner")
    facts=[dict(record_type="FactualResult",descriptor_id="a",outcome="UNKNOWN",physical_attempts=2,censored=False),
        dict(record_type="SingleVehicleEvidence",descriptor_id="a",outcome="SINGLE_VALID",validity_label_mask=True),
        dict(record_type="FactualResult",descriptor_id="b",outcome="NOT_ATTEMPTED",physical_attempts=0,censored=True)]
    report=summarize_coverage([a,a,b],facts,["a"],support_kind="SAVED_BOUNDED_REFERENCE_ANCESTRY")
    assert report["counts"]==dict(proposed_descriptors=2,attempted_descriptors=1,single_valid_descriptors=1,
        certified_single_invalid_descriptors=0,supported_descriptors=1,materialization_queries=2,physical_attempts=2)
    assert report["by_ring"]["outer"]["supported_descriptors"]==1
    assert report["by_stage"]["progress"]["attempted_descriptors"]==0


def test_latency_uses_observed_fields_and_explicit_timeouts_only():
    from terminal_marl.core.rotating_slots.stage3_learning.evaluation import latency_summary,same_state_latency
    records=[dict(wall_s=v,deadline_miss=v>5.) for v in (1.,2.,7.,9.)]
    records[-1].update(watchdog=True,truncations=["WATCHDOG"])
    report=latency_summary(records)
    assert (report["p50_s"],report["p95_s"],report["max_s"])==(2.,9.,9.)
    assert report["timeout_count"]==1 and report["deadline_miss_count"]==2
    rows=[dict(arm="learned_reduced",scheduler_records=records),dict(arm="learned_reduced",status="BUDGET_LIMITED")]
    aggregate=same_state_latency(rows)["learned_reduced"]
    assert aggregate["planning"]["count"]==4 and aggregate["hard_deadline_job_count"]==1


def test_evaluation_full_wall_includes_scheduler_model_setup(tmp_path,monkeypatch):
    from terminal_marl.core.rotating_slots.stage3_learning import evaluation
    from terminal_marl.core.rotating_slots import stage2_planning
    clock=[0.]
    monkeypatch.setattr(evaluation.time,"perf_counter",lambda:clock[0])
    monkeypatch.setattr(evaluation.time,"process_time",lambda:0.)
    class World:
        def configure_practical_runtime(self,enabled):clock[0]+=1.
    def restore(*args,**kwargs):clock[0]+=2.;return World()
    monkeypatch.setattr(evaluation,"restore_root",restore)
    monkeypatch.setattr(stage2_planning,"NormalSchedulerCallback",lambda *args,**kwargs:object())
    class Scheduler:
        records=[dict(wall_s=7.)]
        last_result=SimpleNamespace(certified=True,objective=(0,0,0,0,0))
        def __call__(self,port):clock[0]+=7.
    def setup(output,arm,binding,observer,root_id,timings):
        clock[0]+=5.;timings.update(model_setup_wall_s=5.,scheduler_setup_wall_s=5.,setup_included_in_full_online_wall=True)
        return Scheduler()
    monkeypatch.setattr(evaluation,"_scheduler",setup)
    atomic_json(tmp_path/"heldout_access.json",{})
    atomic_json(tmp_path/"roots/r/root.json",dict(snapshot_id="s",initial_state_hash="i",manifest_hash="m"))
    row=evaluation.evaluate_job(dict(output=str(tmp_path),root_id="r",arm="learned_reduced",kind="same_state"),{})
    assert row["costs"]["full_online_wall_s"]==15.
    assert row["costs"]["setup_timings"]["model_setup_wall_s"]==5.
    assert row["costs"]["setup_timings"]["pre_scheduler_setup_wall_s"]==3.
    assert row["costs"]["planning_latency"]["p50_s"]==7.


def test_actual_history_curve_and_data_insufficient_absence(tmp_path):
    from terminal_marl.core.rotating_slots.stage3_learning.campaign import plot_training_history
    no_data=tmp_path/"no_data"
    atomic_json(no_data/"model/model_card.json",dict(status="DATA_INSUFFICIENT",formal_training_started=False))
    absent=plot_training_history(no_data)
    assert absent["status"]=="SKIPPED_DATA_INSUFFICIENT" and not absent["weights_present"] and not absent["curve_present"]
    trained=tmp_path/"trained"
    atomic_json(trained/"model/model_card.json",dict(status="TRAINED_PILOT",formal_training_started=True))
    history=[dict(epoch=1,train_loss=1.,validation_rank_loss=.8),dict(epoch=2,train_loss=.7,validation_rank_loss=.6)]
    (trained/"model/training_history.jsonl").write_text("".join(json.dumps(row)+"\n" for row in history),encoding="utf-8")
    rendered=plot_training_history(trained)
    if rendered["status"]=="RENDERED_ACTUAL_HISTORY":
        assert rendered["epochs"]==2 and Path(rendered["curve_path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    else:
        assert rendered["status"]=="CURVE_NOT_RENDERED" and rendered["error_type"] in ("ImportError","ModuleNotFoundError") and not rendered["curve_present"]


def test_training_campaign_rejects_heldout_even_without_formal_guard(tmp_path,monkeypatch):
    from terminal_marl.core.rotating_slots.stage3_learning import campaign
    atomic_json(tmp_path/"heldout_access.json",dict(model_hash="frozen"))
    monkeypatch.setattr(campaign,"require_checks",lambda output:None)
    def forbidden(*args,**kwargs):raise AssertionError("training started after heldout access")
    monkeypatch.setattr(campaign,"build_training_dataset",forbidden)
    monkeypatch.setattr(campaign,"run_subprocess",forbidden)
    assert campaign.main(["train","--output",str(tmp_path)])==2
    assert not (tmp_path/"formal_training.guard.json").exists()
    assert "heldout" in json.loads((tmp_path/"campaign_error.json").read_text(encoding="utf-8"))["error"]


def test_unmapped_failed_regression_blocks_targeted_gate_until_same_node_passes(tmp_path):
    from terminal_marl.core.rotating_slots.stage3_learning.evaluation import CHECK_TESTS,record_targeted_checks
    import xml.etree.ElementTree as ET
    suite=ET.Element("testsuite")
    for check,names in CHECK_TESTS.items():ET.SubElement(suite,"testcase",classname=check,name=names[0]+"_observed")
    failed=ET.SubElement(suite,"testcase",classname="extra",name="test_unmapped_regression")
    ET.SubElement(failed,"failure",message="observed failure")
    initial=tmp_path/"initial.xml";ET.ElementTree(suite).write(initial,encoding="utf-8")
    receipt=dict(exit_code=0,status="completed")
    result=record_targeted_checks(tmp_path,[initial],receipt)
    assert not result["all_automated_pass"] and result["merged_failed_tests"][0]["name"]=="test_unmapped_regression"
    rerun=ET.Element("testsuite");ET.SubElement(rerun,"testcase",classname="extra",name="test_unmapped_regression")
    later=tmp_path/"later.xml";ET.ElementTree(rerun).write(later,encoding="utf-8")
    assert record_targeted_checks(tmp_path,[initial,later],receipt)["all_automated_pass"]


def test_archive_missing_plot_dependency_reports_absence_without_failing(tmp_path,monkeypatch):
    import builtins
    from terminal_marl.core.rotating_slots.stage3_learning.campaign import plot_training_history
    atomic_json(tmp_path/"model/model_card.json",dict(status="TRAINED_PILOT",formal_training_started=True))
    (tmp_path/"model/training_history.jsonl").write_text(json.dumps(dict(epoch=1,train_loss=.8,validation_rank_loss=.7))+"\n",encoding="utf-8")
    original=builtins.__import__
    def missing(name,*args,**kwargs):
        if name.startswith("matplotlib"):raise ImportError("matplotlib unavailable in observed environment")
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",missing)
    result=plot_training_history(tmp_path)
    assert result["status"]=="CURVE_NOT_RENDERED" and "matplotlib unavailable" in result["reason"]
    assert not result["curve_present"] and not (tmp_path/"model/training_curve.png").exists()


def test_terminal_failure_is_preserved_beyond_latest_completed_boundary(tmp_path,monkeypatch):
    from terminal_marl.core.rotating_slots.stage3_learning import evaluation
    from terminal_marl.core.rotating_slots import stage2_planning
    class World:
        time=0.;failed=False;terminated_invalid=False;aircraft={};dispatch_timings=[]
        def configure_practical_runtime(self,enabled):pass
        def _advance_energy(self,end):pass
        def advance(self,*args,**kwargs):self.time=2.;self.failed=True
        def to_dict(self):return dict(states=[])
    world=World()
    monkeypatch.setattr(evaluation,"restore_root",lambda *args,**kwargs:world)
    monkeypatch.setattr(stage2_planning,"NormalSchedulerCallback",lambda *args,**kwargs:object())
    scheduler=SimpleNamespace(records=[],provider=SimpleNamespace(previous={}))
    monkeypatch.setattr(evaluation,"_scheduler",lambda *args,**kwargs:scheduler)
    monkeypatch.setattr(evaluation,"save_checkpoint",lambda *args,**kwargs:None)
    monkeypatch.setattr(evaluation,"episode_metrics",lambda world:dict(F_energy_total=1,K_episode=[1,0,0,0,0]))
    atomic_json(tmp_path/"heldout_access.json",{})
    atomic_json(tmp_path/"roots/r/root.json",dict(identity={},snapshot_id="s",initial_state_hash="i",manifest_hash="m"))
    row=evaluation.evaluate_job(dict(output=str(tmp_path),root_id="r",arm="heuristic_reduced",kind="closed_loop"),{})
    assert row["status"]=="EVALUATION_PARTIAL" and row["time"]==0.
    assert row["terminal_observation"]["observed_time_s"]==2. and row["terminal_observation"]["failed"]
    assert row["terminal_observation"]["quality"]["F_energy_total"]==1
