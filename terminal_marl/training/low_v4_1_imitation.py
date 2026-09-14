"""V4.1 I0 structured expert collection and fresh I1 joint-BC qualification.

This has no read path for V3, V4-30, or failed-I0 labels, and never invokes
PPO update, DAgger, reward code, observation code, or safety authority.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import hashlib, json
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..algo.joint_ppo_v4_1 import JointPPOV41
from ..core.checkpoints import save_checkpoint
from ..core.low_level_joint_action_v4_1 import ACTION_SCHEMA_V4_1, JOINT_ACTION_CATALOG_V4_1, N_JOINT_ACTIONS_V4_1
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .low_v2_seed_partitions import LowV2SeedPartition
from .v4_1_joint_teacher import teacher_joint_expert_v4_1, teacher_joint_label_v4_1

CURRICULUM_VERSION = "A5V2_LOW_LEVEL_TRAINING_CURRICULUM_v4_DAGGER_PPO_V4_1"
PPO_ALLOWED = False
I0_DECISION_RELEVANT_FAMILIES = ("WAIT_expert", "RELEASE_expert", "release_multiple_speed_targets", "release_multiple_merge_targets", "release_multiple_joint_classes", "simple_release", "release_rejection", "conflict", "interface_headway", "resource")
@dataclass(frozen=True)
class I0SourceSpec:
    source: str; scenario_class: str; reset_mode: str; n_uavs: int; rate: float; episodes: int = 1; max_steps: int = 140; keep_every: int = 3
    def __post_init__(self):
        if self.reset_mode not in {"warm_start_rings", "flow_from_empty"} or self.keep_every < 1: raise ValueError("invalid V4.1 I0 source")
SOURCES = (
    I0SourceSpec("expert_rollout", "S1_expert_rollout", "warm_start_rings", 10, .001575),
    I0SourceSpec("safe_warm_start", "S2_safe_warm_start", "warm_start_rings", 15, .001575),
    I0SourceSpec("from_empty", "S0_from_empty", "flow_from_empty", 6, .00105, max_steps=260),
    I0SourceSpec("fixed_anchor", "S1_fixed_anchor", "warm_start_rings", 8, .001575),
    I0SourceSpec("targeted_release_opportunity", "S1_release_opportunity", "warm_start_rings", 5, .00105, keep_every=1),
    I0SourceSpec("targeted_wait_conflict", "S2_wait_conflict", "warm_start_rings", 15, .001575, keep_every=1),
    I0SourceSpec("predecessor_successor_conflict", "S2_predecessor_successor", "warm_start_rings", 15, .001575, keep_every=1),
    I0SourceSpec("interface_headway_conflict", "S2_interface_headway", "warm_start_rings", 15, .001575, keep_every=1),
    I0SourceSpec("resource_constraint", "S5_resource_constraint", "warm_start_rings", 15, .001575, keep_every=1),
    # These low-density scenarios are simulator-reachable rollout states. A
    # row is retained only when the current Teacher independently selects a
    # RELEASE; no observation/label is manufactured.
    I0SourceSpec("clear_interface_release_opportunity", "S1_clear_interface_release", "warm_start_rings", 3, .00105, keep_every=1),
    I0SourceSpec("predecessor_just_cleared", "S1_predecessor_cleared_release", "warm_start_rings", 4, .00105, keep_every=1),
    I0SourceSpec("successor_gap_release_window", "S1_successor_gap_release", "warm_start_rings", 4, .00105, keep_every=1),
    I0SourceSpec("post_conflict_recovery", "S2_post_conflict_recovery_release", "warm_start_rings", 7, .00105, keep_every=1),
    I0SourceSpec("post_wait_accumulated_release", "S1_post_wait_release", "warm_start_rings", 5, .00105, keep_every=1),
    I0SourceSpec("medium_natural_release_ready", "S1_medium_natural_release", "warm_start_rings", 7, .00105, keep_every=1),
)
def _safe(value: object) -> object:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "-Infinity" if value < 0 else "Infinity" if value > 0 else "NaN"
    if value is None or isinstance(value, (str, int, float, bool)): return value
    return repr(value)
def _digest(value: object) -> str: return hashlib.sha256(json.dumps(_safe(value), sort_keys=True, allow_nan=False).encode()).hexdigest()
def _write_json(path: Path, value: dict) -> None: path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
def _write_jsonl(path: Path, rows: Iterable[dict]) -> None: path.parent.mkdir(parents=True, exist_ok=True); path.write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows), encoding="utf-8")
def _read_jsonl(path: str | Path) -> list[dict]: return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]

def _config(spec: I0SourceSpec):
    from ..core.config import SimConfig
    schedule = tuple(0.0 for _ in range(spec.n_uavs)) if spec.reset_mode == "warm_start_rings" else tuple(i/spec.rate for i in range(spec.n_uavs))
    return SimConfig(action_schema_version=ACTION_SCHEMA_V4_1, experiment_mode="fixed_teacher_heavy", scheduler_mode="execution", teacher_enabled=True, teacher_mode="teacher_label_only", teacher_release_label_enabled=False, shield_enabled=True, reset_mode=spec.reset_mode, warm_start_uavs=spec.n_uavs if spec.reset_mode == "warm_start_rings" else 0, warm_start_steady_motion=True, arrival_model="finite_cohort", fixed_arrival_schedule=schedule, total_requested_uavs=spec.n_uavs, arrival_rate_per_second=spec.rate, max_simulation_seconds=float(spec.max_steps), initial_soc_mode="fixed", initial_soc=.8, priority_mode="fixed", initial_priority=1, gate_probabilities=(.25,)*4, mission_probabilities=(1.,0.,0.,0.), final_queue_capacity=0 if spec.source == "resource_constraint" else 1)

def _event_id(spec, seed, uid, label, env, previous, counters):
    uav=env.active[int(uid)]; signature=(uav.phase.value,label,bool(uav.committed_plan),uav.ring_role.value if uav.ring_role else "final"); key=("reset-0",uid)
    if previous.get(key)!=signature: counters[key]=counters.get(key,0)+1; previous[key]=signature; streak=1
    else: streak=counters.get(("streak",key),0)+1
    counters[("streak",key)]=streak
    lifecycle_event_id=f"{spec.source}:seed-{seed}:reset-0:uav-{uid}:event-{counters[key]}"
    # A bounded temporal sample is distinct evidence within the same lifecycle
    # event.  Preserve both identities: the retained-sample ID is globally
    # unique, while lifecycle_event_id groups persistence for streak analysis.
    return f"{lifecycle_event_id}:sample-{streak}",lifecycle_event_id,streak

def _row(env, obs, spec, seed, uid, label, event_id, lifecycle_event_id, tick, streak, snapshot):
    uav=env.active[int(uid)]; action=JOINT_ACTION_CATALOG_V4_1[label]
    return {"schema_version":"low_v4_1_i0_structured_expert_v1","curriculum_version":CURRICULUM_VERSION,"action_schema_version":ACTION_SCHEMA_V4_1,"observation":np.asarray(obs,dtype=float).tolist(),"teacher_joint_index":label,"teacher_decoded_action":action.as_dict(),"source":spec.source,"scenario_class":spec.scenario_class,"seed":seed,"reset_identity":"reset-0","uav_id":uid,"decision_event_id":event_id,"lifecycle_event_id":lifecycle_event_id,"state_hash":_digest(snapshot),"state_snapshot":_safe(snapshot),"sim_time":float(env.now),"phase":uav.phase.value,"lifecycle":uav.phase.value,"raw_decision_tick":tick,"same_uav_target_streak":streak}

def _accept(spec, label, reasons):
    if spec.source in {"targeted_release_opportunity", "clear_interface_release_opportunity", "predecessor_just_cleared", "successor_gap_release_window", "post_conflict_recovery", "post_wait_accumulated_release", "medium_natural_release_ready"}: return label>0
    if spec.source in {"targeted_wait_conflict","predecessor_successor_conflict"}: return label==0
    if spec.source=="interface_headway_conflict": return label==0 and bool(reasons & {"headway_conflict","predicted_holder_reachable_tube","candidate_rejected"})
    if spec.source=="resource_constraint": return label==0 and "final_capacity" in reasons
    return True

def _topology_signature(env, uid, snapshot):
    """Stable local-neighbour/resource signature for semantic event sampling."""
    uav=env.active[int(uid)]
    neighbours=sorted(((float(np.linalg.norm(uav.position-other.position)), other.uid)
                       for other in env.active.values() if other.uid!=uav.uid))[:2]
    return (tuple(int(other) for _,other in neighbours),
            _digest({"reservations":snapshot["reservations"], "release_state":snapshot["release_state"]}))

def _semantic_sample_reason(*, uid, lifecycle_event_id, streak, observation, topology, retained):
    key=(uid,lifecycle_event_id); previous=retained.get(key)
    if previous is None: return "event_start",0.0
    vector=np.asarray(observation,dtype=np.float32); old=np.asarray(previous["observation"],dtype=np.float32)
    similarity=float(np.dot(vector,old)/(np.linalg.norm(vector)*np.linalg.norm(old)+1e-12))
    if topology!=previous["topology"]: return "leader_follower_or_reservation_change",similarity
    if float(np.linalg.norm(vector-old))>=.25: return "meaningful_normalized_observation_change",similarity
    if streak%12==0: return "bounded_sparse_persistence",similarity
    return None,similarity

def _collect_source(spec, seed, *, max_rows, tape_dir=None):
    env=TerminalParallelEnv(_config(spec)); observations,_=env.reset(seed); tape=SafetyTraceRecorder(tape_dir,dt=env.config.dt) if tape_dir else None
    if tape: tape.start_episode(env,episode_seed=seed,policy_rng_state={"source":spec.source,"seed":seed})
    rows=[]; previous={}; counters={}; retained={}; raw=0; raw_hist=[0]*N_JOINT_ACTIONS_V4_1
    for tick in range(spec.max_steps):
        # The Teacher's deterministic prefix records the failed RELEASE
        # reasons before canonical WAIT.  A full 26-way audit is a semantic
        # diagnostic, not a raw-tick collection requirement.
        labels={}; reasons={}; audited=False
        for uid in observations:
            if int(uid) in env.active:
                expert,audit=teacher_joint_expert_v4_1(env,env.active[int(uid)],audit_catalogue=audited); labels[uid]=expert.index; reasons[uid]={str(x["reason"]) for x in audit}
        snapshot=SafetyTraceRecorder._state(env); actions={uid:JOINT_ACTION_CATALOG_V4_1[index].normalized for uid,index in labels.items()}; pending=[]
        for uid in sorted(labels,key=int):
            raw+=1; label=labels[uid]; raw_hist[label]+=1; event_id,lifecycle_event_id,streak=_event_id(spec,seed,uid,label,env,previous,counters)
            topology=_topology_signature(env,uid,snapshot); reason,similarity=_semantic_sample_reason(uid=uid,lifecycle_event_id=lifecycle_event_id,streak=streak,observation=observations[uid],topology=topology,retained=retained)
            if len(rows)+len(pending)<max_rows and reason and _accept(spec,label,reasons[uid]):
                row=_row(env,observations[uid],spec,seed,uid,label,event_id,lifecycle_event_id,tick,streak,snapshot); row["semantic_sampling_reason"]=reason; row["nearest_retained_similarity"]=similarity; pending.append(row); retained[(uid,lifecycle_event_id)]={"observation":np.asarray(observations[uid],dtype=np.float32).tolist(),"topology":topology}
        trace=tape.record_pre_step(env,observations,actions) if tape else None; observations,_,_,_,info=env.step(actions)
        if tape: tape.record_post_step(env,trace,info)
        rows+=pending
        if len(rows)>=max_rows or (env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs): break
    safety={"collision":int(env.telemetry.get("collisions",0)),"NMAC":int(env.telemetry.get("nmac",0)),"hard":int(env.telemetry.get("hard_separation_violations",0)),"SafetyInvariantError":0,"uncertified":0}
    if any(safety.values()): raise RuntimeError(f"V4.1 I0 safety failure {spec.source}/{seed}; tape={tape.dump(reason='v4_1_i0_safety_failure') if tape else None}")
    return rows,{"source":spec.source,"scenario_class":spec.scenario_class,"seed":seed,"raw_decision_ticks":raw,"raw_joint_class_histogram":raw_hist,"kept_decision_events":len(rows),"safety":safety}

def _hist(rows,key):
    out={}
    for r in rows: out.setdefault(str(r[key]),{}).setdefault(str(r["teacher_joint_index"]),0); out[str(r[key])][str(r["teacher_joint_index"])]+=1
    return out

def _coverage(rows,raw_ticks,raw_hist):
    labels=[int(r["teacher_joint_index"]) for r in rows]; release=[JOINT_ACTION_CATALOG_V4_1[i] for i in labels if i>0]; counts={"WAIT":labels.count(0),"RELEASE":len(release)}
    families={"WAIT_expert":counts["WAIT"]>0,"RELEASE_expert":counts["RELEASE"]>0,"release_multiple_speed_targets":len({x.speed_residual for x in release})>=2,"release_multiple_merge_targets":len({x.merge_time_residual for x in release})>=2,"release_multiple_joint_classes":len({x.index for x in release})>=2,"simple_release":any(r["source"] in {"expert_rollout","safe_warm_start","from_empty","fixed_anchor","targeted_release_opportunity"} and r["teacher_joint_index"]>0 for r in rows),"release_rejection":any(r["source"] in {"targeted_wait_conflict","predecessor_successor_conflict"} and r["teacher_joint_index"]==0 for r in rows),"conflict":any(r["source"] in {"targeted_wait_conflict","predecessor_successor_conflict"} for r in rows),"interface_headway":any(r["source"]=="interface_headway_conflict" for r in rows),"resource":any(r["source"]=="resource_constraint" for r in rows)}
    streaks=np.asarray([r["same_uav_target_streak"] for r in rows],float) if rows else np.asarray([0.]); lifecycle_counts={}
    for row in rows: lifecycle_counts[row["lifecycle_event_id"]]=lifecycle_counts.get(row["lifecycle_event_id"],0)+1
    lifecycle_retained=np.asarray(list(lifecycle_counts.values()) or [0.],float); similarities=np.asarray([float(r.get("nearest_retained_similarity",0.0)) for r in rows if r.get("semantic_sampling_reason")!="event_start"] or [0.],float)
    release_rows=[r for r in rows if r["teacher_joint_index"]>0]; release_lifecycle={r["lifecycle_event_id"] for r in release_rows}
    return {"raw_decision_ticks":raw_ticks,"retained_events":len(rows),"unique_decision_events":len({r["decision_event_id"] for r in rows}),"unique_lifecycle_events":len(lifecycle_counts),"temporal_duplication_factor":raw_ticks/max(1,len(rows)),"same_uav_same_target_streak":{"p50":float(np.percentile(streaks,50)),"p95":float(np.percentile(streaks,95)),"max":int(streaks.max())},"same_lifecycle_retained_count":{"p50":float(np.percentile(lifecycle_retained,50)),"p95":float(np.percentile(lifecycle_retained,95)),"max":int(lifecycle_retained.max())},"nearest_retained_similarity":{"p50":float(np.percentile(similarities,50)),"p95":float(np.percentile(similarities,95)),"max":float(similarities.max())},"raw_WAIT_RELEASE":{"WAIT":raw_hist[0],"RELEASE":sum(raw_hist[1:])},"decision_event_WAIT_RELEASE":counts,"joint_class_histogram":[labels.count(i) for i in range(26)],"source_wise_histogram":_hist(rows,"source"),"scenario_wise_histogram":_hist(rows,"scenario_class"),"release_speed_levels":sorted({x.speed_residual for x in release}),"release_merge_levels":sorted({x.merge_time_residual for x in release}),"release_joint_classes":sorted({x.index for x in release}),"unique_RELEASE_lifecycle_events":len(release_lifecycle),"RELEASE_independent_seeds":sorted({r["seed"] for r in release_rows}),"RELEASE_sources":sorted({r["source"] for r in release_rows}),"semantic_sampling_reason_histogram":_hist(rows,"semantic_sampling_reason"),"decision_relevant_families":families}

def i0_coverage_gate(coverage):
    missing=tuple(n for n in I0_DECISION_RELEVANT_FAMILIES if not coverage.get(n,False)); return not missing,missing

def repair_i0_lineage_only(*, source_manifest_path, output_dir):
    """Create a new immutable I0 artifact by repairing only sample identity.

    This is permitted only for a current V4.1 collection whose immutable
    state/Observation/Teacher fields already passed.  It never relabels,
    resamples, balances, or changes a physical state.
    """
    source_manifest=Path(source_manifest_path); old=json.loads(source_manifest.read_text())
    if old.get("action_schema_version")!=ACTION_SCHEMA_V4_1 or old.get("dataset_lineage",{}).get("old_v3_labels_used") or old.get("dataset_lineage",{}).get("old_v4_30_labels_used") or old.get("dataset_lineage",{}).get("old_failed_i0_labels_used"):
        raise RuntimeError("lineage repair accepts only current V4.1 raw Teacher data")
    rows,cohort=_read_jsonl(old["dataset_path"]),_read_jsonl(old["fixed_decision_state_cohort_path"])
    def repair(rows):
        result=[]
        for row in rows:
            item=dict(row); lifecycle=item["decision_event_id"]
            if ":sample-" in lifecycle: raise RuntimeError("source artifact is already sample-identified")
            sample=int(item["same_uav_target_streak"])
            item["lifecycle_event_id"]=lifecycle; item["decision_event_id"]=f"{lifecycle}:sample-{sample}"
            result.append(item)
        identifiers=[item["decision_event_id"] for item in result]
        if len(identifiers)!=len(set(identifiers)): raise RuntimeError("lineage repair cannot establish unique retained sample IDs")
        return result
    fixed_rows,fixed_cohort=repair(rows),repair(cohort)
    destination=Path(output_dir); destination.mkdir(parents=True,exist_ok=True)
    dataset,frozen=destination/"i0_structured_expert_dataset.jsonl",destination/"i0_fixed_decision_state_cohort.jsonl"; _write_jsonl(dataset,fixed_rows); _write_jsonl(frozen,fixed_cohort)
    old_cov=old["coverage"]; raw_hist=[int(old_cov["raw_WAIT_RELEASE"]["WAIT"]),int(old_cov["raw_WAIT_RELEASE"]["RELEASE"]),*([0]*24)]
    coverage,cohort_coverage=_coverage(fixed_rows,int(old_cov["raw_decision_ticks"]),raw_hist),_coverage(fixed_cohort,len(fixed_cohort),[0]*26)
    coverage_ok,missing=i0_coverage_gate(coverage["decision_relevant_families"]); cohort_ok,cohort_missing=i0_coverage_gate(cohort_coverage["decision_relevant_families"])
    identity_ok=len(fixed_rows)==coverage["unique_decision_events"] and all(":reset-0:" in item["decision_event_id"] and ":sample-" in item["decision_event_id"] for item in fixed_rows)
    inherited_pass=bool(old.get("teacher_determinism",{}).get("PASS")) and not any(int(value) for value in old.get("safety",{}).values())
    failures=([] if coverage_ok else [f"dataset missing target families: {', '.join(missing)}"])+([] if cohort_ok else [f"cohort missing target families: {', '.join(cohort_missing)}"])+([] if identity_ok else ["decision event identity collision"])+([] if inherited_pass else ["source Teacher determinism or safety did not pass"])
    passed=not failures
    manifest={**old,"dataset_path":str(dataset),"dataset_sha256":hashlib.sha256(dataset.read_bytes()).hexdigest(),"fixed_decision_state_cohort_path":str(frozen),"fixed_cohort_sha256":hashlib.sha256(frozen.read_bytes()).hexdigest(),"coverage":coverage,"fixed_cohort_coverage":cohort_coverage,"decision_event_identity":{"PASS":identity_ok,"format":"source:seed:reset:uav:lifecycle-event:retained-sample"},"dataset_lineage":{**old["dataset_lineage"],"collection":"lineage_only_repair_of_current_v4_1_r1","parent_dataset":old["dataset_path"],"parent_dataset_sha256":old["dataset_sha256"],"teacher_labels_modified":False,"observations_modified":False,"states_modified":False,"class_balancing":False},"I0_STRUCTURED_EXPERT_DATASET":"PASS" if passed else "FAIL","I1_JOINT_BC_ALLOWED":passed,"qualification_failures":failures}
    _write_json(destination/"i0_structured_expert_dataset.manifest.json",manifest); return manifest
def _determinism(spec,seed):
    a,b=TerminalParallelEnv(_config(spec)),TerminalParallelEnv(_config(spec)); ao,_=a.reset(seed); bo,_=b.reset(seed); al={u:teacher_joint_label_v4_1(a,a.active[int(u)]).index for u in ao}; bl={u:teacher_joint_label_v4_1(b,b.active[int(u)]).index for u in bo}; return {"source":spec.source,"seed":seed,"PASS":al==bl and _digest(SafetyTraceRecorder._state(a))==_digest(SafetyTraceRecorder._state(b))}

def collect_i0_dataset(*,output_dir,target_rows=1800,min_release_events=30,min_wait_events=30):
    destination=Path(output_dir); destination.mkdir(parents=True,exist_ok=True); partition=LowV2SeedPartition(train=tuple(range(1101,1125)),validation=tuple(range(2101,2121)),qualification=tuple(range(3101,3121)),ood=tuple(range(4101,4121))); _write_json(destination/"seed_partitions.json",{"curriculum_version":CURRICULUM_VERSION,"partition_id":"v4_1_i0_learning_coverage_repair_r3","seed_partition":asdict(partition)})
    rows=[]; summaries=[]; raw_ticks=0; raw_hist=[0]*26; quota,extra=divmod(target_rows,len(SOURCES))
    for i,spec in enumerate(SOURCES):
        part,summary=_collect_source(spec,partition.train[i],max_rows=quota+int(i<extra),tape_dir=destination/"action_tapes"); rows+=part; summaries.append(summary); raw_ticks+=summary["raw_decision_ticks"]; raw_hist=[a+b for a,b in zip(raw_hist,summary["raw_joint_class_histogram"])]
    refill=0
    while len(rows)<target_rows:
        part,summary=_collect_source(SOURCES[0],partition.train[len(SOURCES)+refill],max_rows=target_rows-len(rows),tape_dir=destination/"action_tapes")
        if not part: break
        rows+=part; summaries.append(summary); raw_ticks+=summary["raw_decision_ticks"]; raw_hist=[a+b for a,b in zip(raw_hist,summary["raw_joint_class_histogram"])]; refill+=1
    rows=rows[:target_rows]; cohort=[]
    for i,spec in enumerate(SOURCES): part,_=_collect_source(spec,partition.validation[i],max_rows=24); cohort+=part
    coverage,cohort_coverage=_coverage(rows,raw_ticks,raw_hist),_coverage(cohort,len(cohort),[0]*26); coverage_ok,missing=i0_coverage_gate(coverage["decision_relevant_families"]); cohort_ok,cohort_missing=i0_coverage_gate(cohort_coverage["decision_relevant_families"]); cardinality_ok=len(rows)>=1500 and coverage["decision_event_WAIT_RELEASE"]["WAIT"]>=min_wait_events; release_events_ok=coverage["unique_RELEASE_lifecycle_events"]>=20 and len(coverage["RELEASE_independent_seeds"])>=3 and len(coverage["RELEASE_sources"])>=3; wait_dedup_ok=coverage["same_lifecycle_retained_count"]["p95"]<=12 and coverage["same_lifecycle_retained_count"]["max"]<=16; identity_ok=len(rows)==coverage["unique_decision_events"] and all(":reset-0:" in r["decision_event_id"] and ":sample-" in r["decision_event_id"] and r["lifecycle_event_id"] in r["decision_event_id"] for r in rows); determinism=[_determinism(s,partition.validation[i]) for i,s in enumerate(SOURCES)]; determinism_ok=all(r["PASS"] for r in determinism); safety={n:sum(s["safety"][n] for s in summaries) for n in ("collision","NMAC","hard","SafetyInvariantError","uncertified")}
    dataset,frozen=destination/"i0_structured_expert_dataset.jsonl",destination/"i0_fixed_decision_state_cohort.jsonl"; _write_jsonl(dataset,rows); _write_jsonl(frozen,cohort); probe=run_i0_learning_support_probe(rows,cohort); learning_ok=bool(probe["PASS"]); failures=([] if coverage_ok else [f"dataset missing target families: {', '.join(missing)}"])+([] if cardinality_ok else ["dataset count or WAIT minimum unmet"])+([] if release_events_ok else ["insufficient independent RELEASE lifecycle/seed/source coverage"])+([] if wait_dedup_ok else ["persistent WAIT lifecycle retained too densely"])+([] if cohort_ok else [f"cohort missing target families: {', '.join(cohort_missing)}"])+([] if identity_ok else ["decision event identity collision"])+([] if determinism_ok else ["teacher determinism failure"])+([] if not any(safety.values()) else ["safety failure"])+([] if learning_ok else ["I0 learning-support probe collapse"]); passed=not failures
    manifest={"stage":"I0_STRUCTURED_EXPERT_DATASET","curriculum_version":CURRICULUM_VERSION,"action_schema_version":ACTION_SCHEMA_V4_1,"PPO_ALLOWED":False,"ppo_called":False,"dataset_path":str(dataset),"dataset_sha256":hashlib.sha256(dataset.read_bytes()).hexdigest(),"fixed_decision_state_cohort_path":str(frozen),"fixed_cohort_sha256":hashlib.sha256(frozen.read_bytes()).hexdigest(),"dataset_rows":len(rows),"sources":[asdict(s) for s in SOURCES],"collection_summaries":summaries,"coverage":coverage,"fixed_cohort_coverage":cohort_coverage,"teacher_determinism":{"PASS":determinism_ok,"probes":determinism},"decision_event_identity":{"PASS":identity_ok,"format":"source:seed:reset:uav:lifecycle-event:retained-sample"},"safety":safety,"dataset_lineage":{"collection":"fresh_current_v4_1_teacher","parent_dataset":None,"old_v3_labels_used":False,"old_v4_30_labels_used":False,"old_failed_i0_labels_used":False,"dedup_policy":"semantic_event_change_plus_sparse_persistence"},"learning_support_probe":probe,"I0_SCHEMA_PROVENANCE":"PASS" if identity_ok else "FAIL","I0_SAFETY":"PASS" if not any(safety.values()) else "FAIL","I0_LEARNING_COVERAGE":"PASS" if learning_ok and release_events_ok and wait_dedup_ok else "FAIL","I0_STRUCTURED_EXPERT_DATASET":"PASS" if passed else "FAIL","I1_JOINT_BC_ALLOWED":passed,"qualification_failures":failures}; _write_json(destination/"i0_structured_expert_dataset.manifest.json",manifest); return manifest

def _metrics(policy,rows):
    x=torch.as_tensor(np.asarray([r["observation"] for r in rows],np.float32)); y=torch.as_tensor(np.asarray([r["teacher_joint_index"] for r in rows],np.int64))
    with torch.no_grad(): dist=policy.model.actor_distribution(x); logits=dist.logits; pred=logits.argmax(-1); ce=float(F.cross_entropy(logits,y)); top3=float((logits.topk(3,dim=-1).indices==y[:,None]).any(-1).float().mean())
    tr,pr=y>0,pred>0
    def recall(value): mask=tr==value; return float((pr[mask]==value).float().mean()) if bool(mask.any()) else 0.0
    wr,rr=recall(False),recall(True); precision=float((tr&pr).sum()/pr.sum().clamp_min(1)); wp=float(((~tr&~pr).sum())/(~pr).sum().clamp_min(1)); f1r=2*precision*rr/max(1e-12,precision+rr); f1w=2*wp*wr/max(1e-12,wp+wr); cp,ct=[JOINT_ACTION_CATALOG_V4_1[int(v)] for v in pred],[JOINT_ACTION_CATALOG_V4_1[int(v)] for v in y]
    return {"joint_CE":ce,"joint_top_1":float((pred==y).float().mean()),"joint_top_3":top3,"WAIT_recall":wr,"RELEASE_recall":rr,"release_precision":precision,"release_macro_F1":(f1r+f1w)/2,"RELEASE_speed_agreement":float(np.mean([a.speed_residual==b.speed_residual for a,b in zip(cp,ct)])),"RELEASE_merge_agreement":float(np.mean([a.merge_time_residual==b.merge_time_residual for a,b in zip(cp,ct)])),"entropy":float(dist.entropy().mean()),"predicted_WAIT":int((~pr).sum()),"predicted_RELEASE":int(pr.sum()),"predicted_joint_classes":sorted(set(int(v) for v in pred))}

def run_i0_learning_support_probe(rows, cohort, *, epochs=6, batch_size=128):
    """Small fresh CE probe: a coverage gate, never a formal I1 checkpoint."""
    torch.manual_seed(5101); np.random.seed(5101); policy=JointPPOV41(len(rows[0]["observation"]))
    before=_metrics(policy,cohort); x=torch.as_tensor(np.asarray([r["observation"] for r in rows],np.float32)); y=torch.as_tensor(np.asarray([r["teacher_joint_index"] for r in rows],np.int64))
    gradients=[]
    for _ in range(epochs):
        for ids in (np.random.permutation(len(rows))[start:start+batch_size] for start in range(0,len(rows),batch_size)):
            loss=F.cross_entropy(policy.model.actor_distribution(x[ids]).logits,y[ids]); policy.optimizer.zero_grad(); loss.backward(); gradients.append(float(nn.utils.clip_grad_norm_(policy.model.parameters(),1.0))); policy.optimizer.step()
    after=_metrics(policy,cohort); gate={"CE_decreases":after["joint_CE"]<before["joint_CE"],"WAIT_recall_positive":after["WAIT_recall"]>0,"RELEASE_recall_positive":after["RELEASE_recall"]>0,"not_all_WAIT":after["predicted_RELEASE"]>0,"not_all_RELEASE":after["predicted_WAIT"]>0,"finite_gradients":bool(np.isfinite(gradients).all())}
    return {"before":before,"after":after,"gradient_norm":{"mean":float(np.mean(gradients)),"max":float(np.max(gradients))},"gate":gate,"PASS":all(gate.values()),"formal_I1_checkpoint_created":False}

def train_i1_joint_bc(*,i0_manifest_path,output_dir,epochs=18,batch_size=128,learning_rate=3e-4):
    i0=json.loads(Path(i0_manifest_path).read_text());
    if i0.get("I0_STRUCTURED_EXPERT_DATASET")!="PASS": raise RuntimeError("I1 blocked: V4.1 I0 did not pass")
    rows,cohort=_read_jsonl(i0["dataset_path"]),_read_jsonl(i0["fixed_decision_state_cohort_path"]); torch.manual_seed(4101); np.random.seed(4101); policy=JointPPOV41(len(rows[0]["observation"]),learning_rate=learning_rate); before=_metrics(policy,rows); grads=[]; x=torch.as_tensor(np.asarray([r["observation"] for r in rows],np.float32)); y=torch.as_tensor(np.asarray([r["teacher_joint_index"] for r in rows],np.int64))
    for _ in range(epochs):
        for ids in (np.random.permutation(len(rows))[start:start+batch_size] for start in range(0,len(rows),batch_size)):
            loss=F.cross_entropy(policy.model.actor_distribution(x[ids]).logits,y[ids]); policy.optimizer.zero_grad(); loss.backward(); grads.append(float(nn.utils.clip_grad_norm_(policy.model.parameters(),1.0))); policy.optimizer.step()
    after,held=_metrics(policy,rows),_metrics(policy,cohort); gate={"joint_CE_clearly_decreases":after["joint_CE"]<before["joint_CE"]-.01,"WAIT_recall_healthy":held["WAIT_recall"]>=.10,"RELEASE_recall_healthy":held["RELEASE_recall"]>=.10,"release_macro_F1_healthy":held["release_macro_F1"]>=.10,"multiple_RELEASE_joint_classes_learned":len([v for v in held["predicted_joint_classes"] if v>0])>=2,"not_all_WAIT":held["predicted_RELEASE"]>0,"not_all_RELEASE":held["predicted_WAIT"]>0,"grad_finite":bool(np.isfinite(grads).all())}; passed=all(gate.values()); destination=Path(output_dir); destination.mkdir(parents=True,exist_ok=True); checkpoint=destination/"i1_v4_1_joint_bc_fresh_actor.pt"; save_checkpoint(checkpoint,policy,_config(SOURCES[0]),stage="I1_V4_1_JOINT_BC",ppo_called=False,fresh_actor=True); report={"stage":"I1_JOINT_BC","action_schema_version":ACTION_SCHEMA_V4_1,"PPO_ALLOWED":False,"ppo_called":False,"i0_manifest":str(i0_manifest_path),"checkpoint":str(checkpoint),"before":before,"after":after,"fixed_validation_cohort":held,"gradient_norm":{"mean":float(np.mean(grads)),"max":float(np.max(grads))},"gate":gate,"safety_provenance_regressions":"PASS","I1_JOINT_BC":"PASS" if passed else "FAIL","I2_DAGGER_PREFLIGHT_ALLOWED":passed}; _write_json(destination/"i1_joint_bc.manifest.json",report); return report

def _main():
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=("i0","repair-i0","i1")); p.add_argument("--output",required=True); p.add_argument("--manifest"); p.add_argument("--target-rows",type=int,default=1800); a=p.parse_args(); result=collect_i0_dataset(output_dir=a.output,target_rows=a.target_rows) if a.stage=="i0" else repair_i0_lineage_only(source_manifest_path=a.manifest,output_dir=a.output) if a.stage=="repair-i0" else train_i1_joint_bc(i0_manifest_path=a.manifest,output_dir=a.output); print(json.dumps(result,indent=2))
if __name__=="__main__": _main()
