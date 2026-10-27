"""One guarded future-state learning campaign; every public stage stops for review."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

DRIVER = Path(__file__).resolve()
LABEL_POLICY = "bounded_joint_contribution_v2"
RANKING = "comparable_group_lazy_v2"
FEATURE_SEMANTICS = "planning_effective_state_v1"
TRAIN_SEMANTICS = "MIXED_TRAIN_FUTURE_TARGET_V1"
PHASE_CAPS = dict(data_and_labels=3600.,training=900.,evaluation=1800.,archive=300.,preflight_tests=600.)
TOTAL_CAP = 7200.
COLLECTION_CPUS = (18,22)
ROOTS = (
    dict(root_id="s3r01",seed=7302,split="train",scenario_profile="diagnostic18_rate120_v1"),
    dict(root_id="s3r02",seed=7303,split="train",scenario_profile="medium24_rate150_v1"),
    dict(root_id="s3r17",seed=7318,split="validation",scenario_profile="medium24_rate150_v1"),
)
CONTRIBUTION_SOURCES = {"observed_bounded_joint_contribution_preference",
                        "observed_bounded_joint_contribution_measured_cost_preference"}


def require(condition,message):
    if not condition: raise ValueError(message)


def load(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def file_hash(path): return sha256(Path(path).read_bytes()).hexdigest()


def exclusive(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as stream: json.dump(value,stream,sort_keys=True,allow_nan=False)


def dependencies(checkout):
    checkout=Path(checkout).resolve()
    require(checkout.is_dir(),"explicit checkout is missing")
    # Import the existing evaluator only after selecting the execution checkout.
    path=checkout/"scripts/stage3/run_latency_interface.py"
    spec=importlib.util.spec_from_file_location("_future_latency",path)
    latency=importlib.util.module_from_spec(spec); spec.loader.exec_module(latency)
    d=latency.dependencies(checkout); d.latency=latency
    from terminal_marl.core.rotating_slots.planning_latency import EFFECTIVE_STATE_VERSION
    d.projection_version=EFFECTIVE_STATE_VERSION
    return d


def protocol(d):
    p=d.latency.protocol(d.dependency_protocol,20.,4.,d.handoff_version)
    p.update(schema="future_learning_20261027_v1",ranking_scope=RANKING,
        feature_state_semantics=FEATURE_SEMANTICS,runtime_projection_version=d.projection_version,
        training_feature_state_semantics=TRAIN_SEMANTICS,training="ONE_NEW_FORMAL_TRAINING_RUN",
        model_sha256=None,normalizer_sha256=None,evaluation_role="DEVELOPMENT_SEEN",
        label_policy_id=LABEL_POLICY,collection_roots=deepcopy(list(ROOTS)),collection_times_s=[30.,90.],
        labels_total=6,root_job_wall_cap_s=900.,collection_cpus=list(COLLECTION_CPUS),
        acquisition="H_LATENCY_AWARE_SHARED_RULES_NO_LEARNED_MODEL_ONE_CONTIGUOUS_ROLLOUT",
        label_projection="SOURCE_T_PLUS_20_VISIBLE_COMMITMENTS_ONLY_NO_MANIFEST_LOOKAHEAD",
        per_snapshot_wall_cap_s=180.,conditional_solve_cap=8,
        campaign_wall_cap_s=TOTAL_CAP,phase_caps=PHASE_CAPS,automatic_stage_transition=False,
        validation="NEW_FUTURE_S3R17_ONLY",old_training="TRAIN_SPLIT_ONLY_CURRENT_STATE_V2")
    return p


def verify_source(frozen,d):
    require(frozen.get("source_commit"),"primary-reviewed source commit required")
    required={p.relative_to(d.checkout).as_posix() for p in d.campaign.source_files()}
    required.update(d.latency.EXTRA_SOURCES)
    required.update(("scripts/stage3/run_future_learning.py","tests/test_rotating_slots_future_learning_campaign.py",
        "scripts/stage3/summarize_online_v2.py","tests/test_rotating_slots_lazy_ranking.py","tests/test_rotating_slots_stage3_joint_contribution.py"))
    require(required<=set(frozen["source_files"]),"source freeze omits campaign source/tests")
    d.campaign.verify_sources(frozen)
    require(file_hash(DRIVER)==frozen["source_files"]["scripts/stage3/run_future_learning.py"],"invoked driver differs from selected checkout")


def verify_physical_baseline(frozen,baseline):
    names={"energy.py","safety.py","trajectory.py","world.py","rescue_rules.py","config.py","geometry.py","demand.py"}
    physical={name:value for name,value in baseline["source_files"].items() if Path(name).name in names}
    require(physical,"latest execution baseline contains no physical sources")
    for name,expected in physical.items():require(frozen["source_files"].get(name)==expected,"physical source changed: "+name)


def copy_root(source,output,rid,d):
    source=Path(source)/"roots"/rid; directory=output/"roots"/rid
    directory.mkdir(parents=True,exist_ok=False)
    row=load(source/"root.json")
    envelope=load(source/"initial.checkpoint.json")
    require(envelope["sha256"]==d.digest(envelope["payload"]) and envelope["payload"]["time"]==0.,"original t0 checkpoint required")
    require(envelope["payload"]["identity"]==row["identity"] and envelope["payload"]["sink"] is None,"invalid physical root checkpoint")
    for name in ("manifest.json","initial_state.json","initial.checkpoint.json"):
        shutil.copyfile(source/name,directory/name)
    require(d.digest(load(directory/"manifest.json"))==row["manifest_hash"],"manifest digest changed")
    require(d.digest(load(directory/"initial_state.json"))==row["initial_state_hash"],"physical root digest changed")
    row.update(checkpoint_path=str(directory/"initial.checkpoint.json"),raw_snapshot_path=str(directory/"initial_state.json"))
    d.atomic_json(directory/"root.json",row)
    return row


def prepare(args,d):
    started=time.perf_counter(); output=args.output.resolve(); parent=args.parent.resolve(); initial=args.initial_parent.resolve()
    require(not output.exists(),"fresh output required; no reset/retry")
    require(not any(output.is_relative_to(p) or p.is_relative_to(output) for p in (parent,initial)),"output overlaps readonly archive")
    frozen=load(args.source_freeze); verify_source(frozen,d)
    baseline_path=args.baseline_execution_freeze.resolve(); baseline=load(baseline_path)
    old=load(parent/"run_binding.generated.json"); split=load(parent/"split_manifest.json")
    environment=d.campaign.resource_probe()
    for package in ("numpy","scipy","PyYAML","ortools","torch"):
        require(environment["versions"].get(package) is not None and environment["versions"].get(package)==old["environment"]["versions"].get(package),"numerical dependency changed: "+package)
    require(all(environment["thread_environment"].get(key)=="1" for key in d.latency.THREAD_KEYS),"one numerical thread required before Python starts")
    payload=deepcopy(split); require(payload.pop("split_manifest_id")==d.digest(payload),"inherited split digest changed")
    split_rows={r["root_id"]:r for r in split["roots"]}
    for spec in ROOTS:
        require(all(split_rows[spec["root_id"]][k]==v for k,v in spec.items()),"fixed collection root changed")
    # Compare this round's physical files with the latest accepted execution,
    # rather than an older training checkout predating authorized interface work.
    verify_physical_baseline(frozen,baseline)
    output.mkdir(parents=True)
    exclusive(output/"prepare.started.json",dict(started_at=d.campaign.utc_now()))
    ledger=d.campaign.CampaignLedger(output/"budget_ledger.json")
    ledger.data.update(total_s=TOTAL_CAP,compute_stop_s=TOTAL_CAP,phase_caps=deepcopy(PHASE_CAPS));ledger.persist();ledger.pause()
    initial_ledger_elapsed=ledger.elapsed
    ledger.data["phase_used"]["preflight_tests"]=initial_ledger_elapsed;ledger.persist()
    try:
        p=protocol(d)
        fp=dict(old["fingerprint"],source_commit=frozen["source_commit"],workspace_diff_hash=frozen["workspace_diff_hash"],
            label_policy_id=LABEL_POLICY,feature_state_semantics=TRAIN_SEMANTICS,
            runtime_projection_version=d.projection_version,evaluation_protocol_id=d.digest(p),model_id="NO_MODEL")
        binding=deepcopy(old)
        binding.update(fingerprint=fp,source_hash=frozen["source_hash"],source_files=frozen["source_files"],online_v2=True,
            future_learning=True,collection_cpus=list(p["collection_cpus"]),evaluation_cpu=12,environment=environment,
            execution_provenance=dict(source_commit=frozen["source_commit"],source_hash=frozen["source_hash"],
                old_training_fingerprint=old["fingerprint"],feature_state_semantics=TRAIN_SEMANTICS,
                old_feature_state_semantics="CURRENT_STATE_V2",future_feature_state_semantics=FEATURE_SEMANTICS,
                runtime_projection_version=d.projection_version,development_role="DEVELOPMENT_SEEN"))
        for spec in ROOTS:
            row=copy_root(initial,output,spec["root_id"],d)
            require(all(row[k]==v for k,v in spec.items()),"collection physical root mismatch")
        row=copy_root(initial,output,"s3r23",d)
        require(row["seed"]==7324 and row["scenario_profile"]=="medium24_rate150_v1","seen development root changed")
        shutil.copyfile(parent/"split_manifest.json",output/"split_manifest.json")
        d.atomic_json(output/"source_freeze.json",frozen)
        d.atomic_json(output/"run_binding.generated.json",binding)
        d.atomic_json(output/"future_learning_protocol.json",dict(protocol=p,source_hash=frozen["source_hash"],
            checkout=str(d.checkout),old_dataset=str(parent/"training_dataset.json"),old_dataset_sha256=file_hash(parent/"training_dataset.json"),
            old_binding_sha256=file_hash(parent/"run_binding.generated.json"),parent_readonly=str(parent),initial_parent_readonly=str(initial),
            baseline_execution_freeze=str(baseline_path),baseline_execution_freeze_sha256=file_hash(baseline_path),
            split_manifest_sha256=file_hash(output/"split_manifest.json")))
        d.atomic_json(output/"prepare.finished.json",dict(status="COMPLETED"))
    finally:
        ledger.credit("prepare_preflight",max(0.,time.perf_counter()-started-initial_ledger_elapsed),phase="preflight_tests");ledger.pause()
    return dict(status="PREPARED")


def verify(output,d):
    require(load(output/"prepare.finished.json")["status"]=="COMPLETED","prepare incomplete")
    frozen=load(output/"source_freeze.json");verify_source(frozen,d)
    meta=load(output/"future_learning_protocol.json"); binding=load(output/"run_binding.generated.json")
    require(meta["protocol"]==protocol(d) and Path(meta["checkout"]).resolve()==d.checkout,"frozen campaign protocol changed")
    require(meta["source_hash"]==binding["source_hash"]==frozen["source_hash"],"binding source mismatch")
    require(file_hash(output/"split_manifest.json")==meta["split_manifest_sha256"],"inherited split changed")
    require(file_hash(meta["old_dataset"])==meta["old_dataset_sha256"],"readonly source dataset changed")
    require(file_hash(Path(meta["parent_readonly"])/"run_binding.generated.json")==meta["old_binding_sha256"],"readonly source binding changed")
    require(file_hash(meta["baseline_execution_freeze"])==meta["baseline_execution_freeze_sha256"],"baseline execution freeze changed")
    require((output/"budget_ledger.json").is_file(),"existing ledger required")
    ledger=d.campaign.CampaignLedger(output/"budget_ledger.json")
    require(ledger.data["phase_caps"]==PHASE_CAPS and ledger.data["total_s"]==ledger.data["compute_stop_s"]==TOTAL_CAP,"ledger caps changed")
    return meta,binding,ledger


def projection_provenance(source,projected,d):
    visible=sorted(source.aircraft)
    require(sorted(projected.aircraft)==visible,"projection revealed unrevealed aircraft")
    require(projected.time==source.time+20.,"projection effective time changed")
    return dict(source_time=source.time,effective_time=projected.time,version=projected.version,
        source_version=source.version,visible_aircraft_ids=visible,feature_state_semantics=FEATURE_SEMANTICS,
        projection=dict(protocol=d.projection_version,visible_commitments_only=True,
            future_manifest_access=False,source_time=source.time,effective_time=projected.time))


def root_job(output,rid,binding,d):
    from terminal_marl.core.rotating_slots.stage3_learning.runtime import AdvisorRuntime
    from terminal_marl.core.rotating_slots.stage3_learning.advisor import HeuristicAdvisor
    from terminal_marl.core.rotating_slots.stage3_learning.features import SCHEMA_V2
    from terminal_marl.core.rotating_slots.stage3_learning.labeling import label_snapshot
    from terminal_marl.core.rotating_slots.planning_latency import PlanningTiming,project_visible_snapshot
    original=load(output/"roots"/rid/"root.json")
    cold=d.NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8")
    world=d.restore_root(original,cold);world.configure_practical_runtime(True)
    observer=d.OnlineObserver()
    runtime=AdvisorRuntime(mode="GUIDED_LIMITED",advisor=HeuristicAdvisor(),fingerprint=binding["fingerprint"],observer=observer,
        ancestry_root_id=rid,deterministic_only=True,trace_mode="online",feature_schema=SCHEMA_V2,ranking_scope=RANKING)
    scheduler=d.NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8",advisor_runtime=runtime,
        budget_scale=.5,online_policy="progressive_v1",prefix_reuse=True)
    wrapper=d.LatencyAwareScheduler(scheduler,lead_seconds=20.,preparation_margin=1.,search_handoff_reserve_seconds=4.)
    world.decision_callback=wrapper;world.admission_callback=d.shared_fifo_admission
    cases=[]
    # Complete one physical rollout first; offline labeling never consumes its timeline.
    for target in (30.,90.):
        world.advance(target,wrapper,settle_only_at_end=True)
        require(world.time==target and not world.failed and not getattr(world,"terminated_invalid",False),"H acquisition failed to reach boundary")
        case=rid+"_t"+str(int(target)); directory=output/"roots"/case;directory.mkdir(parents=True,exist_ok=False)
        # Cold normal search state excludes pending latency task state, preserving
        # every world commitment and arrival/emergency state in the checkpoint.
        checkpoint=d.save_checkpoint(directory/"initial.checkpoint.json",world,cold,original["identity"])
        state=dict(aircraft=[a.to_dict() for a in world.aircraft.values()],slots=world.slots.to_dict(),pads=world.pads.to_dict())
        row=dict(original,root_id=case,ancestry_root_id=rid,snapshot_time_s=target,snapshot_id=case,
            checkpoint_path=str(directory/"initial.checkpoint.json"),raw_snapshot_path=str(directory/"initial_state.json"),
            initial_state_hash=d.digest(state),acquisition_policy="H_LATENCY_AWARE_SHARED_RULES",
            cold_label_scheduler=True,pending_normal_task_discarded=wrapper.pending is not None)
        d.atomic_json(directory/"root.json",row);d.atomic_json(directory/"initial_state.json",state)
        shutil.copyfile(output/"roots"/rid/"manifest.json",directory/"manifest.json")
        d.atomic_json(directory/"acquisition_summary.json",dict(time_s=target,checkpoint_sha256=checkpoint["sha256"],
            scheduler_records=deepcopy(scheduler.records),normal_tasks=deepcopy(wrapper.records),
            pending_task_present=wrapper.pending is not None,online_telemetry=observer.summary()))
        d.atomic_json(output/"progress"/(rid+".json"),dict(stage="acquisition",time_s=target,pid=os.getpid()))
        cases.append(case)
    for case in cases:
        row=load(output/"roots"/case/"root.json")
        label_world=d.restore_root(row,d.NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8"))
        source=label_world.build_planning_snapshot();timing=PlanningTiming.from_lead(source.time,20.,1.)
        projected=project_visible_snapshot(source,timing.effective_time,timing=timing)
        provenance=projection_provenance(source,projected,d)
        d.atomic_json(output/"roots"/case/"projection_provenance.json",provenance)
        labels=label_snapshot(output,case,binding,schema_id=SCHEMA_V2,label_policy=LABEL_POLICY,
            planning_snapshot=projected,planning_snapshot_provenance=provenance)
        d.atomic_json(output/"progress"/(rid+".json"),dict(stage="label_complete",case_id=case,
            status=labels["status"],facts=len(labels.get("facts",[])),pairs=len(labels.get("preference_pairs",[])),pid=os.getpid()))
    return dict(status="COLLECTED",root_id=rid,cases=cases)


def fresh_contexts(labels,descriptors,features,spec,fp):
    """Select FIRST_QUERY_EXACT_REF, then join facts to that exact observation."""
    selected={}
    for row in features: selected.setdefault(row["descriptor_id"],row)
    desc={r["descriptor_id"]:r for r in descriptors};by_context={}
    for did,row in selected.items(): by_context.setdefault(desc[did]["source_node_id"],{})[did]=row
    contexts=[];pairs=[]
    require(labels["fingerprint"]==fp,"new label binding mismatch")
    for cid,rows in by_context.items():
        refs={did:r["features_before_query_ref"] for did,r in rows.items()}
        facts=[f for f in labels["facts"] if f.get("context_id")==cid and refs.get(f["descriptor_id"])==f.get("features_before_query_ref")]
        contexts.append(dict(context_id=cid,ancestry_root_id=spec["root_id"],split=spec["split"],fingerprint=deepcopy(fp),
            features=[r["features"] for r in rows.values()],facts=deepcopy(facts),selected_feature_refs=refs,
            feature_observation_policy="FIRST_QUERY_EXACT_REF",feature_state_semantics=FEATURE_SEMANTICS,
            projection_provenance=deepcopy(labels["planning_snapshot_provenance"]),
            descriptor_records=[deepcopy(desc[did]) for did in rows],
            contribution_solutions=deepcopy(labels.get("conditional_attempts",[])),
            incremental_contribution_evidence=deepcopy(labels.get("incremental_contribution_evidence",[]))))
    for pair in labels["preference_pairs"]:
        rows=by_context.get(pair["context_id"],{})
        require(pair["ancestry_root_id"]==spec["root_id"] and pair["split"]==spec["split"],"new pair ancestry/split mismatch")
        if pair["preferred_descriptor"] in rows and pair["other_descriptor"] in rows:
            cost_refs=pair.get("cost_evidence_refs",[])
            if cost_refs and set(cost_refs)!={rows[did]["features_before_query_ref"] for did in (pair["preferred_descriptor"],pair["other_descriptor"])}:continue
            pairs.append(deepcopy(pair))
    return contexts,prefer_contribution_pairs(pairs)


def prefer_contribution_pairs(pairs):
    grouped={}
    for pair in pairs:
        key=(pair["context_id"],tuple(sorted((pair["preferred_descriptor"],pair["other_descriptor"]))))
        grouped.setdefault(key,[]).append(pair)
    result=[]
    for group in grouped.values():
        contributions=[p for p in group if p["label_source"] in CONTRIBUTION_SOURCES]
        retained=contributions or group
        directions={(p["preferred_descriptor"],p["other_descriptor"]) for p in retained}
        require(len(directions)==1,"contradictory pair supervision in the same context")
        # Exactly one target for a descriptor pair; contribution takes priority.
        result.append(retained[0])
    return result


def migrate_old_train(old,fp,expected_fp):
    from terminal_marl.core.rotating_slots.stage3_learning.data import migrate_v1_dataset
    from terminal_marl.core.rotating_slots.stage3_learning.features import SCHEMA_V2
    original=old
    require(original["fingerprint"]==expected_fp,"source dataset fingerprint mismatch")
    require(all(c["split"] in ("train","validation") for c in old["contexts"]),"sealed source context encountered")
    # Filter only by the frozen split before deep-copy migration. The large old
    # validation tensor is excluded from both model selection and copy overhead.
    train_ids={c["context_id"] for c in old["contexts"] if c["split"]=="train"}
    source_train=dict(old,contexts=[c for c in old["contexts"] if c["context_id"] in train_ids],
        preference_pairs=[p for p in old["preference_pairs"] if p["split"]=="train" and p["context_id"] in train_ids])
    migrated=migrate_v1_dataset(source_train,fp,expected_source_fingerprint=expected_fp)
    source_by_id={c["context_id"]:c for c in original["contexts"]}
    migrated["contexts"]=[c for c in migrated["contexts"] if c["split"]=="train"]
    ids={c["context_id"] for c in migrated["contexts"]}
    migrated["preference_pairs"]=[p for p in migrated["preference_pairs"] if p["split"]=="train" and p["context_id"] in ids]
    for c in migrated["contexts"]:
        source=source_by_id[c["context_id"]]
        require(c["ancestry_root_id"]!="s3r23","development root in inherited training")
        if all(f["schema_id"]==SCHEMA_V2 for f in source["features"]):
            require(c["features"]==source["features"],"legacy V2 numeric feature changed during binding migration")
        c["feature_state_semantics"]="CURRENT_STATE_V2"
        c["prior_source_provenance"]={key:deepcopy(source[key]) for key in ("source_fingerprint","migration_provenance","prior_source_provenance") if key in source}
        c["migration_provenance"].update(source_feature_state_semantics="CURRENT_STATE_V2",target_feature_state_semantics=TRAIN_SEMANTICS)
        for fact,source_fact in zip(c.get("facts",[]),source.get("facts",[])):
            fact["prior_source_provenance"]={key:deepcopy(source_fact[key]) for key in ("source_fingerprint","migration_provenance","prior_source_provenance") if key in source_fact}
            # Existing complete V2 costs remain qualified evidence. Partial costs
            # retain explicit masking; neither old kind becomes future-state data.
            if source_fact.get("cost_scope")=="GENERATOR_ENERGY_TERMINAL_VALIDATION_V2" and source_fact.get("cost_complete"):
                fact["cost_complete"]=True
            for key,value in source_fact.items():
                if key not in {"fingerprint","source_fingerprint","cost_complete","prior_source_provenance"}:require(fact.get(key)==value,"legacy fact numeric/evidence field changed: "+key)
    return migrated


def build_dataset(output,binding,meta,d):
    from terminal_marl.core.rotating_slots.stage3_learning.data import merge_v2_datasets,read_jsonl
    from terminal_marl.core.rotating_slots.stage3_learning.learning import _groups
    from terminal_marl.core.rotating_slots.stage3_learning.features import SCHEMA_V2
    old=migrate_old_train(load(meta["old_dataset"]),binding["fingerprint"],binding["execution_provenance"]["old_training_fingerprint"])
    contexts=[];pairs=[];sources=[]
    for spec in ROOTS:
        for target in (30,90):
            rid=spec["root_id"]+"_t"+str(target); raw=output/"raw"/rid
            labels=load(output/"labels"/rid/"labels.json")
            new_c,new_p=fresh_contexts(labels,read_jsonl(raw/"descriptors.jsonl"),read_jsonl(raw/"features.jsonl"),spec,binding["fingerprint"])
            contexts.extend(new_c);pairs.extend(new_p)
            sources.append(dict(case_id=rid,ancestry_root_id=spec["root_id"],split=spec["split"],contexts=len(new_c),pairs=len(new_p),
                label_source_counts=dict(Counter(p["label_source"] for p in new_p)),
                masked_contributions=sum(not bool(e.get("contribution_mask")) for e in labels.get("incremental_contribution_evidence",[]))))
    fresh=dict(fingerprint=binding["fingerprint"],contexts=contexts,preference_pairs=pairs)
    dataset=merge_v2_datasets([old,fresh],binding["fingerprint"])
    dataset["preference_pairs"]=prefer_contribution_pairs(dataset["preference_pairs"])
    require(not any(c["ancestry_root_id"]=="s3r23" for c in dataset["contexts"]),"development root leakage")
    groups=_groups(dataset,binding["fingerprint"],schema_id=SCHEMA_V2)
    counts={name:dict(contexts=sum(c["split"]==split for c in data["contexts"]),
        pairs=sum(p["split"]==split for p in data["preference_pairs"])) for name,data,split in
        (("old_current_train",old,"train"),("new_future_train",fresh,"train"),("new_future_validation",fresh,"validation"))}
    train_contribution_pairs=sum(p["split"]=="train" and p["label_source"] in CONTRIBUTION_SOURCES for p in fresh["preference_pairs"])
    validation_pairs=sum(p["split"]=="validation" for p in fresh["preference_pairs"])
    reasons=[]
    if not train_contribution_pairs:reasons.append("NO_NEW_FUTURE_TRAIN_JOINT_CONTRIBUTION_PAIRS")
    if not validation_pairs:reasons.append("NO_NEW_FUTURE_VALIDATION_RANK_PAIRS")
    if not all(any(g["context"]["split"]==s and g["pairs"] for g in groups) for s in ("train","validation")):reasons.append("MISSING_TRAIN_OR_VALIDATION_RANK_GROUP")
    index=dict(status="DATA_INSUFFICIENT" if reasons else "READY",reasons=reasons,
        new_future_train_contribution_pairs=train_contribution_pairs,new_future_validation_rank_pairs=validation_pairs,
        source_counts=counts,label_source_counts=dict(Counter(p["label_source"] for p in dataset["preference_pairs"])),
        labels=sources,contexts=len(dataset["contexts"]),pairs=len(dataset["preference_pairs"]),
        training_feature_state_semantics=TRAIN_SEMANTICS,heldout_labels_read=False,old_validation_used=False,
        old_dataset_sha256=meta["old_dataset_sha256"],source_fingerprint=binding["execution_provenance"]["old_training_fingerprint"])
    d.atomic_json(output/"training_dataset.json",dataset);d.atomic_json(output/"dataset_index.json",index)
    return index


def verify_model(output,binding):
    card=load(output/"model/model_card.json")
    require(card["status"]=="TRAINED_PILOT" and card.get("model_frozen") and card.get("formal_training_started"),"unique training did not complete")
    require((output/"formal_training.guard.json").is_file(),"formal training guard required")
    guard=load(output/"formal_training.guard.json")
    require(guard.get("seed")==20261023 and Path(guard.get("output","")).resolve()==(output/"model").resolve(),"unique formal training guard binding changed")
    require(card["fingerprint"]==binding["fingerprint"] and card["seed"]==20261023 and card["parameter_count"]==119555,"trained model metadata changed")
    require(file_hash(output/"model/model.pt")==card["model_hash"],"trained model hash mismatch")
    normalizer=load(output/"model/feature_normalizer.json")
    require(normalizer["schema_id"]==binding["fingerprint"]["feature_schema_id"],"trained normalizer schema mismatch")
    require(file_hash(output/"model/feature_normalizer.json")==card.get("normalizer_sha256"),"trained normalizer hash mismatch")
    return card["model_hash"],file_hash(output/"model/feature_normalizer.json")


def evaluation_scope(output,binding,meta,d):
    model_hash,normalizer_hash=verify_model(output,binding)
    p=d.latency.protocol(d.dependency_protocol,20.,4.,d.handoff_version)
    p.update(ranking_scope=RANKING,model_sha256=model_hash,normalizer_sha256=normalizer_hash,
        training="ONE_NEW_FORMAL_TRAINING_RUN",feature_state_semantics=FEATURE_SEMANTICS,
        runtime_projection_version=d.projection_version,evaluation_role="DEVELOPMENT_SEEN")
    row=load(output/"roots/s3r23/root.json")
    scope=dict(protocol=p,checkout=str(d.checkout),source_commit=binding["fingerprint"]["source_commit"],source_hash=binding["source_hash"],
        driver_sha256=file_hash(DRIVER),model_fingerprint=deepcopy(binding["fingerprint"]),
        training_source_hash=binding["source_hash"],shared_manifest_hash=row["manifest_hash"],initial_state_hash=row["initial_state_hash"],
        interpretation="One future-state model, one 120s DEVELOPMENT_SEEN H/L pair; no timing sweep or heldout claim")
    scope["scope_id"]=d.digest(scope)
    return scope


def verify_evaluation(output,d):
    meta,binding,ledger=verify(output,d)
    scope=load(output/"execution_scope.json")
    require(scope==evaluation_scope(output,binding,meta,d),"evaluation source/model/protocol scope changed")
    return scope,binding,ledger


def job_command(output,d,kind,**payload):
    return [sys.executable,str(DRIVER),"job","--checkout",str(d.checkout),"--output",str(output),"--payload",json.dumps(dict(kind=kind,**payload))]


def execute_job(output,payload,d):
    meta,binding,_=verify(output,d)
    cpu=payload.get("cpu",12)
    if hasattr(os,"sched_setaffinity"):os.sched_setaffinity(0,{cpu})
    if payload["kind"]=="root":return root_job(output,payload["root_id"],binding,d)
    if payload["kind"]=="train":
        from terminal_marl.core.rotating_slots.stage3_learning.learning import train_model
        result=train_model(output/"training_dataset.json",output/"model",binding,config=d.campaign.contract_config(),
            deadline=payload["deadline"],schema_id=binding["fingerprint"]["feature_schema_id"])
        if result.get("status")=="TRAINED_PILOT":
            card=load(output/"model/model_card.json")
            card.update(normalizer_sha256=file_hash(output/"model/feature_normalizer.json"),
                training_dataset_sha256=file_hash(output/"training_dataset.json"),feature_state_semantics=TRAIN_SEMANTICS,
                runtime_projection_version=d.projection_version)
            d.atomic_json(output/"model/model_card.json",card)
        return result
    if payload["kind"]=="evaluation":return d.latency.run_job(output,payload["arm"],d,execution_verifier=verify_evaluation)
    if payload["kind"]=="report":
        path=d.checkout/"scripts/stage3/report_latency_interface.py"
        spec=importlib.util.spec_from_file_location("_future_saved_report",path)
        report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
        report.main(["--output",str(output),"--checkout",str(d.checkout),"--contact-sheet"])
        return dict(status="REPORT_WRITTEN")
    raise ValueError("unknown internal job")


def run_stage(output,stage,d):
    with d.campaign.campaign_lock(output):
        meta,binding,ledger=verify(output,d)
        require(ledger.data.get("paused_at") is not None and not ledger.data.get("active_phase"),"paused inactive campaign required")
        require(not (output/(stage+".started.json")).exists(),"stage already attempted; no retry/recollection/retraining")
        if stage=="train":
            require(load(output/"collect.finished.json")["status"]=="COMPLETED","collection incomplete")
            require(load(output/"dataset_index.json")["status"]=="READY","DATA_INSUFFICIENT; no further acquisition authorized")
            require(not (output/"formal_training.guard.json").exists() and not (output/"model").exists(),"fresh formal training required")
        if stage=="evaluate":
            require(load(output/"train.finished.json")["status"]=="COMPLETED","training stage incomplete")
            scope=evaluation_scope(output,binding,meta,d)
        if stage=="archive": require((output/"evaluate.finished.json").is_file(),"evaluation must be attempted before archive")
        resource=d.campaign.resource_probe() if stage in ("collect","train","evaluate") else {}
        cpus=meta["protocol"]["collection_cpus"] if stage=="collect" else [12]
        if resource:require(set(cpus)<=set(resource.get("free_physical_cores",[])),"fixed CPUs unavailable; return to primary")
        exclusive(output/(stage+".started.json"),dict(started_at=d.campaign.utc_now(),resource_probe=resource))
        phase={"collect":"data_and_labels","train":"training","evaluate":"evaluation","archive":"archive"}[stage]
        result=None;error=None
        try:
            with ledger.phase(phase):
                if stage=="collect":
                    receipts=[]
                    # Two explicit waves on frozen collection CPUs; each root has a 900s cap.
                    for wave in (ROOTS[:2],ROOTS[2:]):
                        require(ledger.remaining(phase)>=900.,"full root cap unavailable")
                        def launch(item):
                            index,spec=item
                            return d.campaign.run_subprocess(ledger,phase,"collect:"+spec["root_id"],
                                job_command(output,d,"root",root_id=spec["root_id"],cpu=cpus[index]),output/"logs",cap=900.,parallel=True)
                        with ThreadPoolExecutor(max_workers=len(wave)) as pool: wave_receipts=list(pool.map(launch,enumerate(wave)))
                        receipts.extend(wave_receipts)
                        require(all(r["status"]=="completed" and r.get("exit_code")==0 for r in wave_receipts),"collection job failed; no retry")
                    index=build_dataset(output,binding,meta,d)
                    result=dict(status="COMPLETED" if index["status"]=="READY" else "DATA_INSUFFICIENT",receipts=receipts,dataset_index=index)
                elif stage=="train":
                    receipt=d.campaign.run_subprocess(ledger,phase,"train_future_once",job_command(output,d,"train",deadline=time.time()+ledger.remaining(phase),cpu=12),output/"logs",cap=900.)
                    require(receipt["status"]=="completed" and receipt.get("exit_code")==0,"training process failed; no retry")
                    verify_model(output,binding)
                    result=dict(status="COMPLETED",receipt=receipt)
                elif stage=="evaluate":
                    d.atomic_json(output/"execution_scope.json",scope)
                    exclusive(output/"run.started.json",dict(scope_id=scope["scope_id"],started_at=d.campaign.utc_now()))
                    receipts=[]
                    for arm in d.latency.ARMS:
                        require(ledger.remaining(phase)>=d.latency.JOB_CAP,"full equal evaluation cap unavailable")
                        receipt=d.campaign.run_subprocess(ledger,phase,arm,job_command(output,d,"evaluation",arm=arm,cpu=12),output/"logs",cap=d.latency.JOB_CAP)
                        receipts.append(receipt)
                        d.atomic_json(output/"receipts"/(arm+".json"),receipt)
                        if receipt["status"]=="BUDGET_LIMITED":d.latency.hard_timeout_evidence(output,arm,receipt,d)
                        require(receipt["status"]=="completed" and receipt.get("exit_code")==0,"evaluation failed; no retry")
                        require(load(output/"evaluation"/arm/"result.json")["status"]=="COMPLETE","evaluation partial")
                    result=dict(status="COMPLETED",receipts=receipts)
                else:
                    d.latency.summarize(output,d,execution_verifier=verify_evaluation)
                    complete_saved_pair=all((output/"evaluation"/arm/"result.json").is_file() for arm in d.latency.ARMS)
                    report_receipt=None
                    if complete_saved_pair:
                        report_receipt=d.campaign.run_subprocess(ledger,phase,"archive_saved_report",job_command(output,d,"report",cpu=12),output/"logs")
                        require(report_receipt["status"]=="completed" and report_receipt.get("exit_code")==0,"saved-artifact report did not complete")
                    result=dict(status="ARCHIVED",evaluation=load(output/"evaluate.finished.json"),ledger_elapsed_s=ledger.elapsed,
                        report_receipt=report_receipt,report_status="WRITTEN_VISUAL_QA_PENDING" if complete_saved_pair else "UNAVAILABLE_INCOMPLETE_PAIR",
                        human_replay_acceptance="HUMAN_PENDING")
                    d.atomic_json(output/"archive_inventory.json",result)
        except Exception as exc:
            error=dict(error_type=type(exc).__name__,error=str(exc));result=dict(status="FAILED_RETURN_TO_PRIMARY",error=error)
        finally:
            ledger.pause()
            d.atomic_json(output/(stage+".finished.json"),dict(result or {},finished_at=d.campaign.utc_now()))
            if stage=="evaluate":d.atomic_json(output/"run.finished.json",dict(result or {},finished_at=d.campaign.utc_now(),no_automatic_retry=True))
        if error:raise RuntimeError(error["error"])
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","collect","train","evaluate","archive","job"))
    parser.add_argument("--checkout",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("/home/user/training_outputs/terminal/stage3_future_learning_20261027"))
    parser.add_argument("--source-freeze",type=Path)
    parser.add_argument("--parent",type=Path,default=Path("/home/user/training_outputs/terminal/stage3_online_v2_20261025"))
    parser.add_argument("--initial-parent",type=Path,default=Path("/home/user/training_outputs/terminal/stage3_learning_v1_0_20261023"))
    parser.add_argument("--baseline-execution-freeze",type=Path,default=Path("/home/user/training_outputs/terminal/stage3_search_handoff_20261027/closed_loop/source_freeze.json"))
    parser.add_argument("--payload")
    args=parser.parse_args(argv)
    if args.stage=="prepare" and args.source_freeze is None:parser.error("prepare requires primary-reviewed --source-freeze")
    if args.stage=="job" and args.payload is None:parser.error("job requires --payload")
    d=dependencies(args.checkout);output=args.output.resolve()
    result=prepare(args,d) if args.stage=="prepare" else execute_job(output,json.loads(args.payload),d) if args.stage=="job" else run_stage(output,args.stage,d)
    print(json.dumps(dict(stage=args.stage,status=result["status"],output=str(output)),sort_keys=True))


if __name__=="__main__":main()
