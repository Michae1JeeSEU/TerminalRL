"""One separately prepared online-v2 round; each expensive stage is explicit.

No command automatically advances from training/dev review to formal evaluation.
The parent archives, their split bytes, and their initial physical identities are
inputs only. This module never scans or regenerates seeds.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from terminal_marl.core.rotating_slots.stage3_learning import campaign
from terminal_marl.core.rotating_slots.stage2_campaign import atomic_json, digest, load_json

CONFIG = campaign.ROOT / "configs/stage3/online_v2.yaml"
ARMS = ("reference_full", "heuristic_reduced", "learned_reduced")


def config():
    import yaml
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def probe_environment():
    observed=campaign.resource_probe()
    missing=[k for k in ("numpy","scipy","PyYAML","ortools","torch","pytest") if observed["versions"].get(k) is None]
    if missing:raise ValueError("required dependencies missing: "+", ".join(missing))
    if any(observed["thread_environment"].get(k)!="1" for k in campaign.THREAD_ENV):raise ValueError("thread environment must equal1 before Python starts")
    if 12 not in observed.get("free_physical_cores",[]):raise ValueError("same physical CPU12 is not verified free")
    return observed


def freeze(output):
    # The existing source-only freezer performs local Git checks, never seed scan.
    from scripts.stage3.reevaluate import freeze as source_only_freeze
    return source_only_freeze(output)


def validate_mid_roots(split):
    rows = {r["root_id"]:r for r in split["roots"]}
    for spec in config()["mid_snapshots"]:
        row = rows[spec["root_id"]]
        for key in ("seed", "split", "scenario_profile"):
            if row[key] != spec[key]:
                raise ValueError("frozen middle-root manifest mismatch: " + spec["root_id"] + ":" + key)
    return [rows[s["root_id"]] for s in config()["mid_snapshots"]]


def copy_checkpoint(source, destination, expected_time, *, scheduler=None, allow_cold_projection=False):
    """Keep input bytes, projecting only incompatible paired-dev search state."""
    envelope = load_json(source)
    payload = envelope["payload"]
    if envelope["sha256"] != digest(payload) or payload["time"] != expected_time:
        raise ValueError("checkpoint content/time mismatch: " + str(source))
    if payload.get("sink") is not None:
        raise ValueError("recording sink requires explicit stream_copy restore; return to primary")
    projected=False;restore_error=None
    result=deepcopy(payload)
    if scheduler is not None:
        from terminal_marl.core.rotating_slots.stage2_campaign import Stage2Codec
        codec=Stage2Codec();cold_state=scheduler.state_dict()
        try:scheduler.load_state_dict(codec.decode(payload["scheduler"]))
        except ValueError as exc:
            if (not allow_cold_projection or expected_time!=30. or str(exc) not in
                ("Stage3 callback binding mismatch","online catalog policy mismatch","prefix reuse policy mismatch")):raise
            restore_error=str(exc);projected=True
            result["scheduler"]=codec.encode(cold_state)
    unchanged=lambda value:digest({k:v for k,v in value.items() if k!="scheduler"})
    if unchanged(result)!=unchanged(payload):raise ValueError("cold search projection changed non-scheduler payload")
    if projected:
        shutil.copyfile(source,Path(destination).with_name("source.checkpoint.json"))
        atomic_json(destination,dict(payload=result,sha256=digest(result)))
    else:shutil.copyfile(source,destination)
    return dict(original_checkpoint_sha256=file_hash(source), original_time_s=payload["time"],
                recording_sink_detached=False, identity=payload["identity"],cold_search_projection=projected,
                cold_restore_error=restore_error,non_scheduler_payload_hash=unchanged(payload),
                copied_checkpoint_sha256=file_hash(destination))


def _copy_root(root_dir, checkpoint, output, case_id, expected_time, *, role=None):
    original = load_json(Path(root_dir) / "root.json")
    directory = Path(output) / "roots" / case_id
    directory.mkdir(parents=True)
    from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
    from terminal_marl.core.rotating_slots.stage3_learning.data import restore_root
    cold=NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8")
    provenance = copy_checkpoint(checkpoint, directory / "initial.checkpoint.json", expected_time,scheduler=cold,
        allow_cold_projection=role=="DEVELOPMENT_SEEN" and expected_time==30.)
    if provenance["identity"] != original["identity"]:
        raise ValueError("root/checkpoint physical identity mismatch")
    shutil.copyfile(Path(root_dir)/"manifest.json",directory/"manifest.json")
    if digest(load_json(directory / "manifest.json")) != original["manifest_hash"]:
        raise ValueError("manifest hash mismatch")
    row = dict(original, root_id=case_id, ancestry_root_id=original.get("ancestry_root_id",original["root_id"]),
        checkpoint_path=str(directory / "initial.checkpoint.json"), snapshot_id=case_id+":t"+str(expected_time),
        snapshot_time_s=expected_time, checkpoint_provenance=provenance,
        raw_snapshot_path=str(directory / "initial_state.json"))
    # This is the physical checkpoint state, which may be at30s. The original
    # root's t0 identity/hash remains explicit instead of being relabeled.
    world=restore_root(row,NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8"))
    state=dict(aircraft=[a.to_dict() for a in world.aircraft.values()],slots=world.slots.to_dict(),pads=world.pads.to_dict())
    row["original_initial_state_hash"]=original["initial_state_hash"]
    row["initial_state_hash"]=digest(state)
    if expected_time==0. and row["initial_state_hash"]!=original["initial_state_hash"]:raise ValueError("restored t0 state differs")
    atomic_json(directory/"initial_state.json",state)
    if role:row["evaluation_role"] = role
    atomic_json(directory / "root.json", row)
    return row


def prepare(parent, output, source_freeze, dev_specs):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if parent == output or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError("new output must be separate from readonly parent")
    if output.exists() and any(output.iterdir()):
        raise ValueError("nonempty output; no restart/budget reset")
    frozen = load_json(Path(source_freeze) / "source_freeze.json")
    campaign.verify_sources(frozen)
    old = load_json(parent / "run_binding.generated.json")
    split = load_json(parent / "split_manifest.json")
    payload = dict(split)
    if payload.pop("split_manifest_id") != digest(payload):raise ValueError("split identity mismatch")
    mids = validate_mid_roots(split)
    if len(dev_specs) != 2 or len({s["case_id"] for s in dev_specs}) != 2:
        raise ValueError("exactly two explicit development checkpoints required")
    if float(dev_specs[0]["expected_time_s"]) != 0.:
        raise ValueError("first development state must be original closed-loop t0")
    from terminal_marl.core.rotating_slots.stage2_planning import PlanningBudget, objective_schema_id
    from terminal_marl.core.rotating_slots.config import load_service_profile
    if old["planning_budget"] != PlanningBudget.for_profile("quality_v5").settings():
        raise ValueError("planning budget changed")
    if old["fingerprint"]["objective_id"] != objective_schema_id("downstream_service_v8"):
        raise ValueError("Stage2 objective changed")
    if old["fingerprint"]["physical_config_hash"] != load_service_profile("service_half_v1").config_hash:
        raise ValueError("physical config changed")
    environment=probe_environment()
    for package in ("numpy","scipy","PyYAML","ortools","torch"):
        if environment["versions"].get(package)!=old["environment"]["versions"].get(package):
            raise ValueError("frozen numerical dependency mismatch: "+package)
    # Check only physical source families; planning/runtime are this authorized change.
    for name, expected in old["source_files"].items():
        if Path(name).name in {"energy.py","safety.py","trajectory.py","world.py","rescue_rules.py","config.py","geometry.py","demand.py"}:
            if frozen["source_files"].get(name) != expected:raise ValueError("physical source changed: "+name)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "prepare.started.json").open("x", encoding="utf8") as stream:json.dump(dict(status="PREPARING"),stream)
    shutil.copyfile(parent / "split_manifest.json", output / "split_manifest.json")
    if (output / "split_manifest.json").read_bytes() != (parent / "split_manifest.json").read_bytes():raise ValueError("split copy differs")
    rows=[]
    for root in mids:
        directory=parent / "roots" / root["root_id"]
        rows.append(_copy_root(directory,directory/"initial.checkpoint.json",output,root["root_id"],0.))
    for spec in dev_specs:
        rows.append(_copy_root(spec["root_dir"],spec["checkpoint_path"],output,spec["case_id"],float(spec["expected_time_s"]),role="DEVELOPMENT_SEEN"))
    initial=next(r for r in rows if r["root_id"]==dev_specs[0]["case_id"])
    if initial["ancestry_root_id"] != "s3r23" or initial["seed"] != 7324:raise ValueError("closed loop requires original s3r23 seed7324")
    cfg=config(); binding=deepcopy(old)
    fp=dict(old["fingerprint"],source_commit=frozen["source_commit"],workspace_diff_hash=frozen["workspace_diff_hash"],
        feature_schema_id=cfg["feature_schema_id"],feature_cost_policy_id=cfg["feature_cost_policy_id"],
        behavior_policy_id=cfg["online_policy_id"],cost_ranking_policy_id=cfg["cost_ranking_policy_id"],
        evaluation_protocol_id=digest(cfg),model_id="NO_MODEL")
    binding.update(fingerprint=fp,source_hash=frozen["source_hash"],source_files=frozen["source_files"],online_v2=True,
        evaluation_cpu=12,environment=environment,collection_cpus=environment["free_physical_cores"][:2],
        execution_provenance=dict(source_commit=frozen["source_commit"],source_hash=frozen["source_hash"],
        old_training_fingerprint=old["fingerprint"],development_role="DEVELOPMENT_SEEN"))
    atomic_json(output/"run_binding.generated.json",binding)
    atomic_json(output/"source_freeze.json",frozen)
    atomic_json(output/"raw_snapshot_index.json",dict(roots=rows,complete=True))
    atomic_json(output/"online_v2_protocol.json",dict(status="PREPARED",config=cfg,parent_readonly=str(parent),
        inherited_split_sha256=file_hash(parent/"split_manifest.json"),training_dataset_input=str(parent/"training_dataset.json"),
        development_case_ids=[s["case_id"] for s in dev_specs],closed_loop_case_id=dev_specs[0]["case_id"],
        development_callbacks=6,closed_loop_jobs=2,heldout_signal_measurement=False,
        generalization_claim="NOT_ASSESSED_DEVELOPMENT_SEEN",source_hash=frozen["source_hash"]))
    ledger=campaign.CampaignLedger(output/"budget_ledger.json");ledger.pause()
    return binding


def verify(output):
    output=Path(output); protocol=load_json(output/"online_v2_protocol.json")
    if protocol["config"] != config():raise ValueError("prepared protocol changed")
    if file_hash(output/"split_manifest.json") != protocol["inherited_split_sha256"]:raise ValueError("inherited split changed")
    campaign.verify_sources(load_json(output/"source_freeze.json"))
    binding=load_json(output/"run_binding.generated.json")
    if binding["source_hash"] != protocol["source_hash"]:raise ValueError("source binding mismatch")
    if not (output/"budget_ledger.json").is_file():raise ValueError("existing ledger required")
    ledger=campaign.CampaignLedger(output/"budget_ledger.json")
    compute=campaign.contract_config()["compute"]
    if ledger.data["phase_caps"]!=compute["phase_caps_seconds"] or ledger.data["total_s"]!=compute["campaign_wall_seconds"] or ledger.data["compute_stop_s"]!=compute["compute_stop_seconds"]:
        raise ValueError("ledger caps changed")
    return protocol,binding,ledger


def job_command(output,kind,**payload):
    command=[sys.executable,str(Path(__file__).resolve()),"job","--output",str(Path(output).resolve()),"--payload",json.dumps(dict(kind=kind,**payload))]
    return command


def acquisition_monitor(world,scheduler,path,*,clock=time.perf_counter):
    """Collection-only progress after completed callbacks; no simulator stepping."""
    started=clock();completed=0;last_wall=None
    def publish(stage="acquisition",**details):
        atomic_json(path,dict(stage=stage,time_s=world.time,elapsed_wall_s=clock()-started,
            completed_callbacks=completed,last_callback_wall_s=last_wall,requested_sim_seconds=30.,
            pid=os.getpid(),**details))
    def callback(snapshot,port):
        nonlocal completed,last_wall
        callback_started=clock()
        result=scheduler(port)
        last_wall=clock()-callback_started;completed+=1
        publish()
        return result
    publish(status="STARTED")
    return callback,publish


def _mid_job(output, root_id, binding):
    from terminal_marl.core.rotating_slots.stage3_learning.data import restore_root
    from terminal_marl.core.rotating_slots.stage3_learning.runtime import AdvisorRuntime
    from terminal_marl.core.rotating_slots.stage3_learning.advisor import HeuristicAdvisor
    from terminal_marl.core.rotating_slots.stage3_learning.online_observer import OnlineObserver
    from terminal_marl.core.rotating_slots.stage3_learning.features import SCHEMA_V2
    from terminal_marl.core.rotating_slots.stage3_learning.labeling import label_snapshot
    from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
    from terminal_marl.core.rotating_slots.stage2_campaign import save_checkpoint
    original=load_json(output/"roots"/root_id/"root.json")
    cold=NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8")
    world=restore_root(original,cold);world.configure_practical_runtime(True)
    observer=OnlineObserver()
    runtime=AdvisorRuntime(mode="GUIDED_LIMITED",advisor=HeuristicAdvisor(),fingerprint=binding["fingerprint"],observer=observer,
        ancestry_root_id=root_id,deterministic_only=True,trace_mode="online",feature_schema=SCHEMA_V2)
    scheduler=NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8",advisor_runtime=runtime,budget_scale=.5,online_policy="progressive_v1",prefix_reuse=True)
    callback,publish_progress=acquisition_monitor(world,scheduler,output/"acquisition_progress"/(root_id+".json"))
    world.advance(30.,callback,settle_only_at_end=True)
    if world.time != 30. or world.failed or getattr(world,"terminated_invalid",False):raise ValueError("middle acquisition did not reach physical30s")
    rid=root_id+"_t30";directory=output/"roots"/rid;directory.mkdir(parents=True)
    checkpoint=directory/"initial.checkpoint.json"
    save_checkpoint(checkpoint,world,cold,original["identity"])
    state=dict(aircraft=[a.to_dict() for a in world.aircraft.values()],slots=world.slots.to_dict(),pads=world.pads.to_dict())
    row=dict(original,root_id=rid,ancestry_root_id=root_id,snapshot_id=rid+":t30",snapshot_time_s=30.,checkpoint_path=str(checkpoint),
        initial_state_hash=digest(state),raw_snapshot_path=str(directory/"initial_state.json"),acquisition_policy="H_PROGRESSIVE_LIGHT_ONLINE")
    atomic_json(directory/"root.json",row);atomic_json(directory/"initial_state.json",state)
    shutil.copyfile(output/"roots"/root_id/"manifest.json",directory/"manifest.json")
    atomic_json(directory/"acquisition_summary.json",dict(time_s=world.time,online_telemetry=observer.summary(),scheduler_records=scheduler.records))
    publish_progress(stage="labeling")
    labels=label_snapshot(output,rid,binding,schema_id=SCHEMA_V2)
    publish_progress(stage="complete",label_status=labels["status"],facts=len(labels.get("facts",[])),
        preference_pairs=len(labels.get("preference_pairs",[])))
    return labels


def execute_job(output,payload):
    protocol,binding,_=verify(output)
    if hasattr(os,"sched_setaffinity"):os.sched_setaffinity(0,{payload.get("cpu",12)})
    if payload["kind"]=="mid":return _mid_job(output,payload["root_id"],binding)
    if payload["kind"] in ("same_state","closed_loop"):
        from terminal_marl.core.rotating_slots.stage3_learning.evaluation import evaluate_job
        return evaluate_job(dict(payload,output=str(output)),binding)
    if payload["kind"]=="train":
        from terminal_marl.core.rotating_slots.stage3_learning.learning import train_model
        return train_model(output/"training_dataset.json",output/"model",binding,config=campaign.contract_config(),
            deadline=payload["deadline"],schema_id=binding["fingerprint"]["feature_schema_id"])
    raise ValueError("unknown job")


def build_dataset(output,binding,protocol):
    from terminal_marl.core.rotating_slots.stage3_learning.data import migrate_v1_dataset,merge_v2_datasets,read_jsonl
    old=load_json(protocol["training_dataset_input"])
    migrated=migrate_v1_dataset(old,binding["fingerprint"],expected_source_fingerprint=binding["execution_provenance"]["old_training_fingerprint"])
    contexts=[];pairs=[];label_counts=[]
    for spec in config()["mid_snapshots"]:
        rid=spec["root_id"]+"_t30";raw=output/"raw"/rid;labels=load_json(output/"labels"/rid/"labels.json")
        descriptors={d["descriptor_id"]:d for d in read_jsonl(raw/"descriptors.jsonl")}
        # Explicit first-query projection. Facts join the exact reference selected
        # here; a later observation never silently replaces earlier visible input.
        selected={}
        for row in read_jsonl(raw/"features.jsonl"):
            selected.setdefault(row["descriptor_id"],row)
        by_context={}
        for did,row in selected.items():
            by_context.setdefault(descriptors[did]["source_node_id"],{})[did]=row
        for cid,features in by_context.items():
            refs={d:r["features_before_query_ref"] for d,r in features.items()}
            facts=[f for f in labels["facts"] if refs.get(f["descriptor_id"])==f.get("features_before_query_ref")]
            contexts.append(dict(context_id=cid,ancestry_root_id=spec["root_id"],split=spec["split"],fingerprint=binding["fingerprint"],
                features=[r["features"] for r in features.values()],facts=facts,selected_feature_refs=refs,feature_observation_policy="FIRST_QUERY_EXACT_REF"))
        for pair in labels["preference_pairs"]:
            features=by_context.get(pair["context_id"],{})
            if pair["preferred_descriptor"] in features and pair["other_descriptor"] in features:pairs.append(pair)
        label_counts.append(dict(root_id=rid,ancestry_root_id=spec["root_id"],split=spec["split"],
            complete_cost_facts=sum(bool(f.get("cost_complete")) for f in labels["facts"]),
            incremental_labeled_descriptors=labels.get("incremental_labeled_descriptors"),
            contribution_labels=sum(bool(r.get("contribution_mask")) for r in labels.get("incremental_contribution_evidence",[])),
            preference_pairs=len(labels["preference_pairs"]),candidate_timer_regime=labels.get("candidate_timer_regime")))
    fresh=dict(fingerprint=binding["fingerprint"],contexts=contexts,preference_pairs=pairs)
    dataset=merge_v2_datasets([migrated,fresh],binding["fingerprint"])
    atomic_json(output/"training_dataset.json",dataset)
    from terminal_marl.core.rotating_slots.stage3_learning.learning import auxiliary_targets
    from terminal_marl.core.rotating_slots.stage3_learning.features import SCHEMA_V2
    def visible_new_context(context):
        return any(any(f["query"][49::2]) for f in context["features"])
    summary=dict(contexts=len(dataset["contexts"]),pairs=len(dataset["preference_pairs"]),heldout_labels_read=False,
        added_middle_labels=label_counts,split_counts={split:dict(contexts=sum(c["split"]==split for c in dataset["contexts"]),
            pairs=sum(p["split"]==split for p in dataset["preference_pairs"]),
            nonmissing_new_context_groups=sum(c["split"]==split and visible_new_context(c) for c in dataset["contexts"]),
            complete_cost_targets=sum(auxiliary_targets(f,schema_id=SCHEMA_V2)[1] is not None for c in dataset["contexts"] if c["split"]==split for f in c.get("facts",[])),
            contribution_labels=sum(r["contribution_labels"] for r in label_counts if r["split"]==split)) for split in ("train","validation")})
    atomic_json(output/"dataset_index.json",summary)
    return summary


def matrix(protocol,kind):
    if kind=="dev":return [(rid,arm) for rid in protocol["development_case_ids"] for arm in ARMS]
    if kind=="evaluate":return [(protocol["closed_loop_case_id"],arm) for arm in ARMS[1:]]
    raise ValueError("unknown matrix")


def closed_comparison(output,protocol,rows):
    """Compare only a completed boundary shared by both serial arms."""
    rid=protocol["closed_loop_case_id"];arms=ARMS[1:]
    completed={arm:{int(p.stem.split("_")[-1]):p for p in (output/"evaluation/closed_loop"/rid/arm).glob("boundary_*.json")} for arm in arms}
    boundary=max(set(completed[arms[0]])&set(completed[arms[1]]),default=0)
    common=[load_json(completed[arm][boundary]) for arm in arms] if boundary else []
    terminals={}
    for arm in arms:
        path=output/"evaluation/closed_loop"/rid/arm/"terminal_observation.json"
        terminals[arm]=load_json(path) if path.exists() else dict(status="NOT_PUBLISHED_INTERRUPTED",observed_time_s=None)
    return dict(status="COMPLETE" if boundary==60 else "EVALUATION_PARTIAL",rows=common,job_results=rows,
        common_completed_boundary_s=boundary,requested_sim_seconds=60,terminal_observations=terminals)


def run_stage(output,stage,*,reviewed_dev=False):
    output=Path(output)
    with campaign.campaign_lock(output):
        protocol,binding,ledger=verify(output)
        guard=output/(stage+".started.json")
        if guard.exists():raise ValueError("stage already attempted; return to primary, no retry")
        if stage in ("collect","train","dev","evaluate"):
            if not load_json(output/"online_v2_checks.json").get("passed"):raise ValueError("online v2 checks required")
        if stage=="train" and (output/"heldout_access.json").exists():raise ValueError("development/test gate already open; no training")
        if stage=="evaluate" and (not reviewed_dev or not (output/"dev.finished.json").exists() or load_json(output/"dev.finished.json")["status"]!="COMPLETED"):
            raise ValueError("primary review of completed development results required; pass --reviewed-dev")
        atomic_json(guard,dict(started_at=campaign.utc_now()))
        phase={"tests":"preflight_tests","collect":"data_and_labels","train":"training","dev":"evaluation","evaluate":"evaluation","archive":"archive"}[stage]
        try:
            with ledger.phase(phase):
                if stage=="tests":
                    receipt=campaign.run_subprocess(ledger,phase,"online_v2_checks",[sys.executable,"-m","pytest",
                        "tests/test_rotating_slots_stage3_online_campaign.py","tests/test_rotating_slots_stage3_online_runtime.py",
                        "tests/test_rotating_slots_stage3_prefix_reuse.py","tests/test_rotating_slots_stage3_online_catalog.py",
                        "tests/test_rotating_slots_stage3_cost_ranking.py","tests/test_stage2_candidate_validation_v6.py","tests/test_stage2_quality_budget.py",
                        "-q","--junitxml",str(output/"online_v2_pytest.xml")],output/"logs")
                    result=dict(passed=receipt.get("exit_code")==0,receipt=receipt);atomic_json(output/"online_v2_checks.json",result)
                    if not result["passed"]:raise ValueError("online v2 checks failed")
                elif stage=="collect":
                    cpus=binding["collection_cpus"]
                    def collect(item):
                        index,spec=item
                        return campaign.run_subprocess(ledger,phase,"mid:"+spec["root_id"],job_command(output,"mid",root_id=spec["root_id"],cpu=cpus[index%len(cpus)]),output/"logs",parallel=True)
                    receipts=[];specs=list(enumerate(config()["mid_snapshots"]))
                    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
                        for start in range(0,len(specs),len(cpus)):
                            batch=list(pool.map(collect,specs[start:start+len(cpus)]))
                            receipts.extend(batch)
                            if any(r["status"]!="completed" for r in batch):raise RuntimeError("middle acquisition/label job incomplete")
                    result=build_dataset(output,binding,protocol)
                elif stage=="train":
                    result=campaign.run_subprocess(ledger,phase,"train_v2",job_command(output,"train",deadline=time.time()+ledger.remaining(phase)),output/"logs")
                    if result["status"]!="completed":raise RuntimeError("guarded training job incomplete")
                elif stage in ("dev","evaluate"):
                    from terminal_marl.core.rotating_slots.stage3_learning.evaluation import open_test_gate,quality_differences
                    open_test_gate(output,load_json(output/"model/model_card.json"))
                    rows=[];tasks=matrix(protocol,stage);cap=90. if stage=="dev" else ledger.remaining(phase)/2.
                    kind="same_state" if stage=="dev" else "closed_loop"
                    for rid,arm in tasks:
                        receipt=campaign.run_subprocess(ledger,phase,stage+":"+rid+":"+arm,job_command(output,kind,root_id=rid,arm=arm),output/"logs",cap=cap)
                        path=output/"evaluation"/kind/rid/arm/"result.json"
                        rows.append(load_json(path) if path.exists() else dict(root_id=rid,arm=arm,status=receipt["status"]))
                        if receipt["status"]=="failed":raise RuntimeError("evaluation child failed; retain logs")
                    result=dict(status="COMPLETE" if all(r.get("status")=="COMPLETE" for r in rows) else "EVALUATION_PARTIAL",rows=rows,
                        role="DEVELOPMENT_SEEN",generalization_claim="NOT_ASSESSED",reference_is_global_optimum=False,
                        quality_degradation_allowance="NOT_AUTHORIZED",online_benefit="ONLINE_BENEFIT_NOT_ESTABLISHED",serial_arms=True,evaluation_cpu=12)
                    if stage=="dev":
                        result["reference_comparisons"]=[]
                        for rid in protocol["development_case_ids"]:
                            local={r["arm"]:r for r in rows if r["root_id"]==rid}
                            reference=local.get("reference_full",{})
                            for arm in ARMS[1:]:
                                row=local.get(arm,{})
                                comparable=reference.get("certified") and row.get("certified") and reference.get("objective") is not None and row.get("objective") is not None
                                result["reference_comparisons"].append(dict(root_id=rid,arm=arm,comparable=bool(comparable),
                                    bounded_reference_better=tuple(reference["objective"])<tuple(row["objective"]) if comparable else None,
                                    reference_objective=reference.get("objective"),arm_objective=row.get("objective")))
                    else:
                        result.update(closed_comparison(output,protocol,rows))
                        common=result["rows"]
                        if len(common)==2:result["quality_differences"]=quality_differences(common[0]["quality"],common[1]["quality"])
                    atomic_json(output/("same_state_comparison.json" if stage=="dev" else "closed_loop_comparison.json"),result)
                else:
                    result=dict(status="ARCHIVED",ledger_elapsed_s=ledger.elapsed,training=load_json(output/"model/model_card.json") if (output/"model/model_card.json").exists() else {},
                        development=load_json(output/"same_state_comparison.json") if (output/"same_state_comparison.json").exists() else {},
                        evaluation=load_json(output/"closed_loop_comparison.json") if (output/"closed_loop_comparison.json").exists() else {},generalization_claim="NOT_ASSESSED_DEVELOPMENT_SEEN")
                    atomic_json(output/"final_status.json",result)
                    (output/"FINAL_REPORT.md").write_text("# Stage3 online v2\n\n"
                        +"Training, development and paired evaluation evidence are saved in model/model_card.json, same_state_comparison.json and closed_loop_comparison.json.\n\n"
                        +"Both development inputs were previously seen. Independent generalization and an online benefit are not established automatically. No quality degradation tolerance is authorized.\n\n"
                        +"Campaign elapsed wall: %.3fs. Full budget/job receipts: budget_ledger.json.\n"%ledger.elapsed,encoding="utf8")
            atomic_json(output/(stage+".finished.json"),dict(status="COMPLETED",result=result,finished_at=campaign.utc_now()))
            return result
        except Exception as exc:
            atomic_json(output/(stage+".finished.json"),dict(status="FAILED",error_type=type(exc).__name__,error=str(exc)));raise
        finally:ledger.pause()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("freeze","prepare","tests","collect","train","dev","evaluate","archive","job"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--parent",type=Path)
    parser.add_argument("--source-freeze",type=Path)
    parser.add_argument("--dev-specs",type=Path,help="JSON list of exactly2 case_id/root_dir/checkpoint_path/expected_time_s objects")
    parser.add_argument("--reviewed-dev",action="store_true")
    parser.add_argument("--payload")
    args=parser.parse_args(argv)
    if args.stage=="freeze":return freeze(args.output)
    if args.stage=="prepare":
        if args.parent is None or args.source_freeze is None or args.dev_specs is None:parser.error("prepare requires parent/source-freeze/dev-specs")
        return prepare(args.parent,args.output,args.source_freeze,load_json(args.dev_specs))
    if args.stage=="job":return execute_job(args.output,json.loads(args.payload))
    return run_stage(args.output,args.stage,reviewed_dev=args.reviewed_dev)


if __name__=="__main__":main()
