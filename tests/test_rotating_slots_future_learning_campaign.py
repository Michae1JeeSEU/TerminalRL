"""Pure campaign fixtures: no physics rollout, CP solve, inference or fit."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.stage3 import run_future_learning as runner
from terminal_marl.core.rotating_slots.stage3_learning.features import SCHEMA_V2
from terminal_marl.core.rotating_slots.stage3_learning.learning import _groups, LABEL_POLICY


def fingerprint(policy=runner.LABEL_POLICY):
    return dict(objective_id="objective",feature_schema_id=SCHEMA_V2,label_policy_id=policy,model_id="NO_MODEL")


def pair(source="bounded_conditional_certified_objective_preference",policy=runner.LABEL_POLICY,**changes):
    return dict(context_id="context",ancestry_root_id="s3r17",split="validation",preferred_descriptor="a",other_descriptor="b",
        label_source=source,label_policy_id=policy,objective_id="objective",evidence_refs=["solution:a","solution:b"],**changes)


def feature(did,value): return dict(descriptor_id=did,schema_id=SCHEMA_V2,query=[value,1.]*33)


def context(fp,split="validation",root="s3r17",semantics=runner.FEATURE_SEMANTICS):
    return dict(context_id="context",ancestry_root_id=root,split=split,fingerprint=fp,feature_state_semantics=semantics,
        features=[feature("a",1.),feature("b",2.)],facts=[])


def test_protocol_freezes_one_training_and_one_seen_pair():
    from scripts.stage3 import run_latency_interface
    d=SimpleNamespace(latency=run_latency_interface,dependency_protocol="dependency",handoff_version="handoff",projection_version="projection")
    p=runner.protocol(d)
    assert (p["lead_seconds"],p["scheduler_period"],p["preparation_margin"],p["search_handoff_reserve_seconds"])==(20.,5.,1.,4.)
    assert (p["candidate_active_seconds"],p["cp_seconds"],p["hint_seconds"],p["soft_seconds"],p["callback_watchdog_seconds"])==(6.,8.,3.,30.,60.)
    assert (p["attempts_per_aircraft"],p["total_attempts"],p["labels_total"],p["root_job_wall_cap_s"])==(32,1024,6,900.)
    assert p["collection_cpus"]==list(runner.COLLECTION_CPUS)==[18,22]
    assert p["cpu"]==12
    assert p["model_sha256"] is None and p["normalizer_sha256"] is None
    assert sum(p["phase_caps"].values())==p["campaign_wall_cap_s"]==7200.
    assert p["validation"]=="NEW_FUTURE_S3R17_ONLY" and not p["automatic_stage_transition"]


def test_first_query_exact_ref_join_keeps_later_fact_out():
    fp=fingerprint();provenance={"source_time":30.,"effective_time":50.,"version":"v"}
    labels=dict(fingerprint=fp,planning_snapshot_provenance=provenance,preference_pairs=[pair()],facts=[
        dict(context_id="context",descriptor_id="a",features_before_query_ref="later"),
        dict(context_id="context",descriptor_id="a",features_before_query_ref="first")])
    rows=[dict(descriptor_id="a",features_before_query_ref="first",features=feature("a",1.)),
        dict(descriptor_id="a",features_before_query_ref="later",features=feature("a",9.)),
        dict(descriptor_id="b",features_before_query_ref="b",features=feature("b",2.))]
    descriptors=[dict(descriptor_id=d,source_node_id="context") for d in ("a","b")]
    contexts,pairs=runner.fresh_contexts(labels,descriptors,rows,runner.ROOTS[2],fp)
    assert contexts[0]["selected_feature_refs"]["a"]=="first"
    assert contexts[0]["features"][0]["query"][0]==1.
    assert [f["features_before_query_ref"] for f in contexts[0]["facts"]]==["first"]
    assert pairs==labels["preference_pairs"]


def test_contribution_replaces_opposite_old_quality_same_pair():
    old=pair();new=pair(source="observed_bounded_joint_contribution_preference")
    new.update(preferred_descriptor="b",other_descriptor="a")
    assert runner.prefer_contribution_pairs([old,new])==[new]
    conflicting=deepcopy(new);conflicting.update(preferred_descriptor="a",other_descriptor="b")
    with pytest.raises(ValueError,match="contradictory"):runner.prefer_contribution_pairs([new,conflicting])


def test_projection_provenance_never_reads_future_manifest():
    class Visible:
        time=30.;version="source";aircraft={"seen":object()}
        @property
        def manifest(self): raise AssertionError("future manifest accessed")
    projected=SimpleNamespace(time=50.,version="future",aircraft={"seen":object()})
    provenance=runner.projection_provenance(Visible(),projected,SimpleNamespace(projection_version="protocol"))
    assert provenance["visible_aircraft_ids"]==["seen"]
    assert not provenance["projection"]["future_manifest_access"]
    projected.aircraft["future_arrival"]=object()
    with pytest.raises(ValueError,match="unrevealed"):runner.projection_provenance(Visible(),projected,SimpleNamespace(projection_version="protocol"))


def test_old_train_migration_keeps_numeric_features_qualified_cost_and_masks_partial():
    source_fp=fingerprint(LABEL_POLICY);fp=fingerprint()
    train=context(source_fp,"train","s3r01","CURRENT_STATE_V2")
    train["facts"]=[dict(context_id="context",descriptor_id=did,ancestry_root_id="s3r01",fingerprint=source_fp,
        cost_complete=complete,cost_scope="GENERATOR_ENERGY_TERMINAL_VALIDATION_V2" if complete else "PARTIAL",wall_s=7.,
        features_before_query_ref="exact:"+did) for did,complete in (("a",True),("b",False))]
    validation=deepcopy(train);validation.update(context_id="old_val",split="validation",ancestry_root_id="s3r17")
    old=dict(fingerprint=source_fp,contexts=[train,validation],preference_pairs=[dict(pair(policy=LABEL_POLICY),split="train",ancestry_root_id="s3r01")])
    before=deepcopy(old);migrated=runner.migrate_old_train(old,fp,source_fp)
    assert old==before and len(migrated["contexts"])==1
    result=migrated["contexts"][0]
    assert result["features"]==train["features"] and [f["wall_s"] for f in result["facts"]]==[7.,7.]
    assert [f["cost_complete"] for f in result["facts"]]==[True,False]
    assert result["migration_provenance"]["source_fingerprint"]==source_fp
    assert _groups(migrated,fp,schema_id=SCHEMA_V2)[0]["pairs"]==[("a","b")]


def test_new_validation_forbids_legacy_policy_and_new_unbound_contribution():
    fp=fingerprint();c=context(fp)
    dataset=dict(fingerprint=fp,contexts=[c],preference_pairs=[pair(policy=LABEL_POLICY)])
    with pytest.raises(ValueError,match="old-current train migration"):_groups(dataset,fp,schema_id=SCHEMA_V2)
    dataset["preference_pairs"]=[pair(source="observed_bounded_joint_contribution_preference")]
    with pytest.raises(ValueError,match="four-solve"):_groups(dataset,fp,schema_id=SCHEMA_V2)


def test_default_learning_policy_keeps_old_behavior():
    fp=fingerprint(LABEL_POLICY);c=context(fp)
    data=dict(fingerprint=fp,contexts=[c],preference_pairs=[pair(policy=LABEL_POLICY)])
    assert _groups(data,fp,schema_id=SCHEMA_V2)[0]["pairs"]==[("a","b")]
    data["preference_pairs"][0]["label_policy_id"]=runner.LABEL_POLICY
    with pytest.raises(ValueError,match="unapproved preference"):_groups(data,fp,schema_id=SCHEMA_V2)


def test_unique_model_card_resolves_real_new_hash_and_rejects_changed_weights(tmp_path):
    fp=fingerprint();model=tmp_path/"model";model.mkdir()
    (model/"model.pt").write_bytes(b"new weights")
    runner.exclusive(model/"feature_normalizer.json",dict(schema_id=SCHEMA_V2))
    runner.exclusive(tmp_path/"formal_training.guard.json",dict(seed=20261023,output=str(model.resolve())))
    runner.exclusive(model/"model_card.json",dict(status="TRAINED_PILOT",model_frozen=True,formal_training_started=True,
        fingerprint=fp,seed=20261023,parameter_count=119555,model_hash=runner.file_hash(model/"model.pt"),
        normalizer_sha256=runner.file_hash(model/"feature_normalizer.json")))
    hashes=runner.verify_model(tmp_path,dict(fingerprint=fp))
    assert hashes==(runner.file_hash(model/"model.pt"),runner.file_hash(model/"feature_normalizer.json"))
    (model/"model.pt").write_bytes(b"changed weights")
    with pytest.raises(ValueError,match="model hash"):runner.verify_model(tmp_path,dict(fingerprint=fp))


def test_physical_baseline_is_latest_execution_not_old_training_source():
    latest=dict(source_files={"terminal_marl/core/rotating_slots/world.py":"accepted_latency_world",
        "terminal_marl/core/rotating_slots/stage3_learning/runtime.py":"old_ranking"})
    current=dict(source_files={"terminal_marl/core/rotating_slots/world.py":"accepted_latency_world",
        "terminal_marl/core/rotating_slots/stage3_learning/runtime.py":"new_lazy_ranking"})
    old_training=dict(source_files={"terminal_marl/core/rotating_slots/world.py":"older_training_world"})
    assert current["source_files"]["terminal_marl/core/rotating_slots/world.py"]!=old_training["source_files"]["terminal_marl/core/rotating_slots/world.py"]
    runner.verify_physical_baseline(current,latest)
    with pytest.raises(ValueError,match="physical source changed"):runner.verify_physical_baseline(current,old_training)


def test_new_contribution_learning_rechecks_actual_four_solve_evidence():
    from terminal_marl.core.rotating_slots.stage3_learning import labeling
    fp=fingerprint();descriptors=[dict(descriptor_id=did,snapshot_id="snapshot",source_node_id="context",aircraft_id="aircraft",
        parent_prefix_id="prefix",construction_stage="alternative") for did in ("a","b")]
    solutions=[];increments=[]
    for did,included,removed in (("a",[0,-3,10],[0,-2,10]),("b",[0,-1,8],[0,2,8])):
        common=dict(certified=True,catalog_id="catalog",budget_settings={"seconds":3},method="rh_cp_sat",status="FEASIBLE")
        inc=dict(common,required_descriptor=did,descriptor_ids=[did],objective=included,selected_signatures=[did])
        rem=dict(common,required_descriptor=None,excluded_descriptor=did,descriptor_ids=[],objective=removed,
            selected_signatures=[],removed_signatures=[did],counterfactual_catalog_id="without:"+did)
        for row in (inc,rem):row["solution_id"]=labeling.canonical_solution_id(row)
        solutions.extend((inc,rem));increments.append(labeling.incremental_evidence(did,inc,rem))
    approved=labeling.joint_contribution_pairs(descriptors,[],solutions,increments,root_id="s3r17",split="validation",fingerprint=fp)
    c=context(fp);c.update(descriptor_records=descriptors,contribution_solutions=solutions,incremental_contribution_evidence=increments)
    data=dict(fingerprint=fp,contexts=[c],preference_pairs=approved)
    assert _groups(data,fp,schema_id=SCHEMA_V2)[0]["pairs"]==[("b","a")]
    data["preference_pairs"][0]["evidence_refs"][0]="unbound"
    with pytest.raises(ValueError,match="four-solve"):_groups(data,fp,schema_id=SCHEMA_V2)


def test_dataset_gate_stops_without_new_contribution_and_future_validation(tmp_path):
    old_fp=fingerprint(LABEL_POLICY);fp=fingerprint()
    old=dict(fingerprint=old_fp,contexts=[context(old_fp,"train","s3r01","CURRENT_STATE_V2")],
        preference_pairs=[dict(pair(policy=LABEL_POLICY),split="train",ancestry_root_id="s3r01")])
    runner.exclusive(tmp_path/"old_dataset.json",old)
    for spec in runner.ROOTS:
        for target in (30,90):
            rid=spec["root_id"]+"_t"+str(target)
            runner.exclusive(tmp_path/"labels"/rid/"labels.json",dict(fingerprint=fp,facts=[],preference_pairs=[],planning_snapshot_provenance={}))
            raw=tmp_path/"raw"/rid;raw.mkdir(parents=True)
            for name in ("descriptors.jsonl","features.jsonl"):(raw/name).write_text("",encoding="utf8")
    binding=dict(fingerprint=fp,execution_provenance=dict(old_training_fingerprint=old_fp))
    index=runner.build_dataset(tmp_path,binding,dict(old_dataset=str(tmp_path/"old_dataset.json"),old_dataset_sha256="source"),
        SimpleNamespace(atomic_json=runner.exclusive))
    assert index["status"]=="DATA_INSUFFICIENT"
    assert "NO_NEW_FUTURE_TRAIN_JOINT_CONTRIBUTION_PAIRS" in index["reasons"]
    assert "NO_NEW_FUTURE_VALIDATION_RANK_PAIRS" in index["reasons"]
    assert index["source_counts"]["old_current_train"]==dict(contexts=1,pairs=1)
    assert index["old_validation_used"] is False
