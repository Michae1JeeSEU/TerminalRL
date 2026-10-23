"""Frozen, same-state and short paired online evaluation with honest partials."""
from __future__ import annotations

from pathlib import Path
import os
import sys
import time
import math
import xml.etree.ElementTree as ET

from ..stage2_campaign import atomic_json,load_json,digest,episode_metrics,save_checkpoint
from .data import PortableObserver,restore_root,read_jsonl,summarize_coverage

ARMS=("reference_full","heuristic_reduced","learned_reduced")


def open_test_gate(output,model_card):
    output=Path(output)
    if not model_card.get("model_frozen") or not model_card.get("model_hash"):
        raise ValueError("test outcomes require a frozen model and normalizer")
    if not (output/"model/feature_normalizer.json").is_file():raise ValueError("normalizer not frozen")
    value=dict(model_hash=model_card["model_hash"],split_manifest_id=load_json(output/"split_manifest.json")["split_manifest_id"],
        access="ONE_FROZEN_EVALUATION_NO_RETRAIN",opened_at=time.time())
    path=output/"heldout_access.json"
    if path.exists():
        prior=load_json(path)
        if any(prior[k]!=value[k] for k in ("model_hash","split_manifest_id")):raise ValueError("heldout already opened for another frozen model")
        return prior
    atomic_json(path,value);return value


def rotated_arms(case_index):
    n=case_index%len(ARMS)
    return ARMS[n:]+ARMS[:n]


def quality_differences(control,learned):
    keys=("F_energy_total","L_normal","L_success","L_emergency","cost_emergency","cost_reroute","energy_j","airborne_time","upstream_wait_time")
    delta={key:learned.get(key,0.)-control.get(key,0.) for key in keys}
    forbidden=[]
    for key in ("F_energy_total","L_emergency","cost_emergency","cost_reroute"):
        if delta[key]>0:forbidden.append(key)
    if delta["L_normal"]<0:forbidden.append("L_normal")
    return dict(learned_minus_control=delta,forbidden_deterioration_fields=forbidden,
        quality_regression_observed=bool(forbidden),lexicographic_episode_worse=tuple(learned["K_episode"])>tuple(control["K_episode"]),
        authorized_degradation_tolerance="NONE")


def measure_heldout_signal(output,binding):
    """Open sealed rank labels only inside the frozen evaluation gate."""
    from .model import LearnedAdvisor
    output=Path(output)
    if not (output/"heldout_access.json").exists():raise ValueError("heldout gate not open")
    path=output/"heldout_signal.json"
    if path.exists():return load_json(path)
    advisor=LearnedAdvisor(output/"model",fingerprint=binding["fingerprint"],evaluation=True)
    advisor._load();rows=[]
    for root in load_json(output/"split_manifest.json")["roots"]:
        if root["split"]!="test":continue
        label_path=output/"labels"/root["root_id"]/"labels.json"
        if not label_path.exists():continue
        labels=load_json(label_path)
        features={r["descriptor_id"]:r["features"] for r in read_jsonl(output/"raw"/root["root_id"]/"features.jsonl")}
        scores={};values=list(features.values())
        for offset in range(0,len(values),64):scores.update(advisor.rank(values[offset:offset+64]).scores)
        for pair in labels["preference_pairs"]:
            a,b=pair["preferred_descriptor"],pair["other_descriptor"]
            if a not in scores or b not in scores:continue
            margin=float(scores[a])-float(scores[b]);loss=max(0.,-margin)+math.log1p(math.exp(-abs(margin)))
            rows.append(dict(root_id=root["root_id"],context_id=pair["context_id"],margin=margin,loss=loss,
                concordance=1. if margin>0 else 0. if margin<0 else .5))
    groups={}
    for row in rows:groups.setdefault(row["context_id"],[]).append(row)
    mean=lambda key:sum(sum(r[key] for r in values)/len(values) for values in groups.values())/len(groups) if groups else None
    value=dict(status="MEASURED" if rows else "DATA_INSUFFICIENT",effective_pairs=len(rows),contexts=len(groups),
        context_mean_concordance=mean("concordance"),context_mean_pairwise_loss=mean("loss"),
        signal_status="LEARNABLE_SIGNAL_NOT_ESTABLISHED",interpretation="Pilot measurements only; no automatic significance or performance threshold",rows=rows)
    atomic_json(path,value);return value


def _scheduler(output,arm,binding,observer,root_id,setup_timings=None):
    setup_started=time.perf_counter();model_setup=0.
    from ..stage2_planning import NormalSchedulerCallback
    from .runtime import AdvisorRuntime
    from .advisor import HeuristicAdvisor
    advisor=HeuristicAdvisor() if arm=="heuristic_reduced" else None
    if arm=="learned_reduced":
        model_started=time.perf_counter()
        from .model import LearnedAdvisor
        advisor=LearnedAdvisor(Path(output)/"model",fingerprint=binding["fingerprint"],evaluation=True)
        advisor._load()
        model_setup=time.perf_counter()-model_started
    fingerprint=dict(binding["fingerprint"],model_id=load_json(Path(output)/"model/model_card.json")["model_hash"])
    runtime=AdvisorRuntime(mode="GUIDED_LIMITED" if arm!="reference_full" else "RECORD",advisor=advisor,
        fingerprint=fingerprint,observer=observer,ancestry_root_id=root_id,evaluation=True,deterministic_only=arm=="heuristic_reduced")
    scheduler=NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8",advisor_runtime=runtime,
        budget_scale=1. if arm=="reference_full" else .5)
    if setup_timings is not None:
        setup_timings.update(model_setup_wall_s=model_setup,scheduler_setup_wall_s=time.perf_counter()-setup_started,
            setup_included_in_full_online_wall=True)
    return scheduler


def latency_summary(records,wall_field="wall_s"):
    """Nearest-rank quantiles of observed wall fields, without timing inference."""
    values=sorted(float(r[wall_field]) for r in records if isinstance(r.get(wall_field),(int,float)) and math.isfinite(r[wall_field]) and r[wall_field]>=0)
    def percentile(q):return values[max(0,math.ceil(q*len(values))-1)] if values else None
    timeout_tokens={"TIMEOUT","WATCHDOG"}
    def timed_out(row):
        tags={str(row.get(k,"")) for k in ("status","stop_reason","solver_stop_reason")}
        tags.update(row.get("truncations",()))
        return bool(row.get("watchdog",False) or tags&timeout_tokens)
    return dict(wall_field=wall_field,quantile_rule="NEAREST_RANK",count=len(values),
        p50_s=percentile(.5),p95_s=percentile(.95),max_s=max(values) if values else None,
        timeout_count=sum(timed_out(r) for r in records),deadline_miss_count=sum(bool(r.get("deadline_miss",False)) for r in records),
        truncation_counts={tag:sum(tag in r.get("truncations",()) for r in records) for tag in sorted({tag for r in records for tag in r.get("truncations",())})})


def same_state_latency(rows):
    return {arm:dict(planning=latency_summary([record for row in rows if row.get("arm")==arm for record in row.get("scheduler_records",())]),
        hard_deadline_job_count=sum(row.get("arm")==arm and row.get("status")=="BUDGET_LIMITED" for row in rows)) for arm in ARMS}


def _costs(scheduler,observer,wall,cpu,dispatch,*,setup_timings=None,dispatch_records=()):
    facts=[r for r in observer.records if r.get("record_type")=="FactualResult"]
    energies=[r for r in observer.records if r.get("record_type")=="SingleVehicleEvidence"]
    descriptors=[r for r in observer.records if r.get("record_type")=="WorkDescriptor"]
    supported={did for row in observer.records if row.get("record_type")=="PlanningSelection" and row.get("certified") for did in row.get("selected_descriptor_ids",())}
    return dict(full_online_wall_s=wall,process_cpu_s=cpu,full_dispatch_wall_s=dispatch,
        setup_timings=dict(setup_timings or {}),online_timing_boundary="JOB_ENTRY_THROUGH_COMPLETED_PLANNING_OR_SIMULATION_BOUNDARY",
        planning_latency=latency_summary(scheduler.records),dispatch_latency=latency_summary(dispatch_records,"dispatch_wall_s"),
        physical_attempts=sum(r.get("physical_attempts",0) for r in facts),descriptor_queries=len(facts),
        generation_wall_s=sum(r.get("wall_s",0.) or 0. for r in facts),
        energy_wall_s=sum(r.get("wall_s",0.) or 0. for r in energies),
        exclusive_planning_timings=[r.get("timings",r.get("phase_timings",{})) for r in scheduler.records],
        fallback_traces=[r for r in observer.records if r.get("record_type")=="Fallback"],
        coverage_summary=summarize_coverage(descriptors,facts+energies,supported,support_kind="CERTIFIED_SELECTED_COLUMN_ANCESTRY"),
        coverage=[r for r in observer.records if r.get("record_type")=="Coverage"])


def observe_executed_holding(world):
    """Read actual energy deltas at the executor's unchanged integration boundary.

    The isolated evaluation process owns this class wrapper. No world field or
    numerical result is added/changed, so strict checkpoints remain portable.
    """
    cls=type(world);original=cls._advance_energy;totals={r:0. for r in ("outer","middle","inner")}
    def observed(instance,end):
        before={aid:(a.ring,a.holding_time,a.energy_j) for aid,a in instance.aircraft.items()}
        result=original(instance,end)
        for aid,(ring,held,energy) in before.items():
            after=instance.aircraft.get(aid)
            if after is not None and ring in totals and after.holding_time>held:
                totals[ring]+=after.energy_j-energy
        return result
    cls._advance_energy=observed
    return totals,lambda:setattr(cls,"_advance_energy",original)


def terminal_observation(world):
    return dict(status="OBSERVED",observed_time_s=world.time,failed=bool(world.failed),
        terminated_invalid=bool(getattr(world,"terminated_invalid",False)),quality=episode_metrics(world))


def evaluate_job(payload,binding):
    start=time.perf_counter();cpu_start=time.process_time()
    from ..stage2_planning import NormalSchedulerCallback
    from ..world import SchedulerPort
    from ..stage2_campaign import shared_fifo_admission
    from ..replay import ReplayBundle
    output=Path(payload["output"]);root_id=payload["root_id"];arm=payload["arm"];kind=payload["kind"]
    if not (output/"heldout_access.json").exists():raise ValueError("heldout gate not open")
    root=load_json(output/"roots"/root_id/"root.json")
    # Restore physical t0 with matching full/OFF identity; experimental arm starts fresh.
    world=restore_root(root,NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8"))
    world.configure_practical_runtime(True)
    directory=output/"evaluation"/kind/root_id/arm;directory.mkdir(parents=True,exist_ok=True)
    observer=PortableObserver(directory/"trace")
    setup_timings={"pre_scheduler_setup_wall_s":time.perf_counter()-start}
    scheduler=_scheduler(output,arm,binding,observer,root_id,setup_timings)
    world.admission_callback=shared_fifo_admission
    if kind=="same_state":
        result=scheduler(SchedulerPort(world))
        costs=_costs(scheduler,observer,time.perf_counter()-start,time.process_time()-cpu_start,None,setup_timings=setup_timings)
        last=scheduler.last_result
        row=dict(status="COMPLETE",root_id=root_id,arm=arm,snapshot_id=root["snapshot_id"],
            original_initial_state_hash=root["initial_state_hash"],manifest_hash=root["manifest_hash"],
            certified=bool(last and last.certified),objective=None if last is None else list(last.objective),
            scheduler_records=scheduler.records,costs=costs,cache_regime="FRESH_PROCESS_COLD_PER_ARM",cpu=os.sched_getaffinity(0) if hasattr(os,"sched_getaffinity") else None)
        if isinstance(row["cpu"],set):row["cpu"]=sorted(row["cpu"])
        atomic_json(directory/"result.json",row)
        return row
    # Both serial arms publish identical 5-second completed boundaries. Hard
    # parent termination can only discard the currently uncommitted boundary.
    def normal(snapshot,port):return scheduler(port)
    holding_energy,restore_observer=observe_executed_holding(world)
    save_checkpoint(directory/"checkpoint_0000.json",world,scheduler,root["identity"])
    for boundary in range(5,61,5):
        world.advance(float(boundary),normal,settle_only_at_end=True)
        if abs(world.time-boundary)>1.e-7 or world.failed or getattr(world,"terminated_invalid",False):break
        row=dict(status="COMPLETE_BOUNDARY",root_id=root_id,arm=arm,time=world.time,
            quality=episode_metrics(world),holding_energy_by_ring=dict(values_j=dict(holding_energy),status="OBSERVED_EXECUTOR_ENERGY_DELTAS"),
            costs=_costs(scheduler,observer,time.perf_counter()-start,time.process_time()-cpu_start,
                sum(r.get("dispatch_wall_s",0.) for r in world.dispatch_timings),setup_timings=setup_timings,dispatch_records=world.dispatch_timings),
            shared_manifest_hash=root["manifest_hash"],initial_state_hash=root["initial_state_hash"],
            dispatch_timings=world.dispatch_timings,cpu=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None)
        save_checkpoint(directory/f"checkpoint_{boundary:04d}.json",world,scheduler,root["identity"],boundary_summary=row)
        atomic_json(directory/f"boundary_{boundary:04d}.json",row)
    completed=sorted(directory.glob("boundary_*.json"))
    restore_observer()
    terminal=terminal_observation(world)
    atomic_json(directory/"terminal_observation.json",terminal)
    row=load_json(completed[-1]) if completed else dict(status="EVALUATION_PARTIAL",time=0.,root_id=root_id,arm=arm)
    row["status"]="COMPLETE" if row["time"]==60. else "EVALUATION_PARTIAL"
    row["terminal_observation"]=terminal
    # Bundle trajectory commits separately from proposal/attempt trace records.
    record=world.to_dict();record.update(decisions=scheduler.records,stage3_trace=observer.records,
        stage2_provenance=binding,tentative_predictions=[dict(aircraft_id=aid,as_of=world.time,plans=[p.to_dict() for p in plans]) for aid,plans in scheduler.provider.previous.items()])
    if record.get("states") and not (directory/"replay").exists():
        ReplayBundle.write(directory/"replay",record=record,code_commit=binding["fingerprint"]["source_commit"],config_path=world.config.path,
            command="scripts/stage3/evaluate.py",seed=root["seed"],eager=False)
        row["replay_path"]=str(directory/"replay")
    atomic_json(directory/"result.json",row)
    return row


def run_evaluation(output,ledger):
    from .campaign import run_subprocess,job_command,contract_config
    output=Path(output);card_path=output/"model/model_card.json"
    if not card_path.exists() or not load_json(card_path).get("model_frozen"):
        reason="MODEL_MISSING" if not card_path.exists() else load_json(card_path).get("status","DATA_INSUFFICIENT")
        skipped=dict(status="SKIPPED",reason=reason,online_benefit="ONLINE_BENEFIT_NOT_ESTABLISHED",random_deployment=False)
        atomic_json(output/"same_state_comparison.json",skipped);atomic_json(output/"closed_loop_comparison.json",skipped);return skipped
    checks=load_json(output/"targeted_checks.json") if (output/"targeted_checks.json").exists() else {}
    if not checks.get("all_automated_pass"):raise ValueError("targeted automated checks required before evaluation")
    open_test_gate(output,load_json(card_path))
    measure_heldout_signal(output,load_json(output/"run_binding.generated.json"))
    roots=load_json(output/"raw_snapshot_index.json")["roots"]
    test=[r for r in roots if r["split"]=="test"]
    cases=(test+[r for r in roots if r["split"]=="validation"])[:contract_config()["evaluation"]["same_snapshot_cases_max"]]
    rows=[]
    for index,root in enumerate(cases):
        for arm in rotated_arms(index):
            case_id=root["root_id"]+"_"+arm
            receipt=run_subprocess(ledger,"evaluation","same_state:"+case_id,
                job_command(output,"same_state",case_id=case_id,root_id=root["root_id"],arm=arm),output/"logs",cap=90.)
            path=output/"evaluation/same_state"/root["root_id"]/arm/"result.json"
            if path.exists():rows.append(load_json(path))
            elif receipt["status"]!="completed":rows.append(dict(root_id=root["root_id"],arm=arm,status=receipt["status"]))
            if receipt["status"]=="failed":
                atomic_json(output/"same_state_comparison.json",dict(status="FAILED",rows=rows,failed_job=receipt))
                atomic_json(output/"closed_loop_comparison.json",dict(status="SKIPPED",reason="SAME_STATE_STRUCTURAL_FAILURE"))
                raise RuntimeError("same-state evaluation child failed; inspect receipt/log")
    atomic_json(output/"same_state_comparison.json",dict(status="COMPLETE" if len(rows)==len(cases)*3 and all(r.get("status")=="COMPLETE" for r in rows) else "EVALUATION_PARTIAL",
        rows=rows,latency_by_arm=same_state_latency(rows),heldout_cases=sum(r["split"]=="test" for r in cases),reference_is_global_optimum=False,
        quality_degradation_allowance="NOT_AUTHORIZED",online_benefit="ONLINE_BENEFIT_NOT_ESTABLISHED"))
    candidates=[r for r in test if r["scenario_profile"]=="medium24_rate150_v1"]
    if not candidates:
        result=dict(status="SKIPPED",reason="EVALUATION_INPUT_MISSING_NO_TRAIN_ROOT_SUBSTITUTION")
        atomic_json(output/"closed_loop_comparison.json",result);return result
    root=candidates[0];arms=("heuristic_reduced","learned_reduced")
    # Reserve equal wall for serial arms from the same physical core/cache regime.
    allowance=max(0.,ledger.remaining("evaluation")/2.)
    for arm in arms:
        case_id=root["root_id"]+"_"+arm
        receipt=run_subprocess(ledger,"evaluation","closed_loop:"+case_id,
            job_command(output,"closed_loop",case_id=case_id,root_id=root["root_id"],arm=arm),output/"logs",cap=allowance)
        if receipt["status"]=="failed":
            atomic_json(output/"closed_loop_comparison.json",dict(status="FAILED",failed_job=receipt,reason="STRUCTURAL_FAILURE"))
            raise RuntimeError("closed-loop evaluation child failed; inspect receipt/log")
    completed={arm:{int(p.stem.split("_")[-1]):p for p in (output/"evaluation/closed_loop"/root["root_id"]/arm).glob("boundary_*.json")} for arm in arms}
    common=set(completed[arms[0]])&set(completed[arms[1]])
    boundary=max(common,default=0)
    rows=[load_json(completed[a][boundary]) for a in arms] if boundary else []
    differences=quality_differences(rows[0]["quality"],rows[1]["quality"]) if rows else None
    terminals={}
    for arm in arms:
        path=output/"evaluation/closed_loop"/root["root_id"]/arm/"terminal_observation.json"
        terminals[arm]=load_json(path) if path.exists() else dict(status="NOT_PUBLISHED_INTERRUPTED",observed_time_s=None,failed=None,terminated_invalid=None)
    result=dict(status="COMPLETE" if boundary==60 else "EVALUATION_PARTIAL",root_id=root["root_id"],common_completed_boundary_s=boundary,
        requested_sim_seconds=60,rows=rows,quality_differences=differences,quality_regression_observed=bool(differences and differences["quality_regression_observed"]),
        terminal_observations=terminals,terminal_failure_observed=any(row.get("failed") or row.get("terminated_invalid") for row in terminals.values()),
        terminal_comparison_scope="SEPARATE_OBSERVATIONS_MAY_EXTEND_BEYOND_COMMON_COMPLETED_BOUNDARY",
        latency_by_arm={row["arm"]:dict(planning=row["costs"]["planning_latency"],dispatch=row["costs"]["dispatch_latency"]) for row in rows},
        hard_deadline_jobs=[job_id for job_id,receipt in ledger.data["jobs"].items() if job_id.startswith("closed_loop:") and receipt["status"]=="BUDGET_LIMITED"],
        online_benefit="ONLINE_BENEFIT_NOT_ESTABLISHED",human_visual_review="HUMAN_PENDING",
        evaluation_cpu=load_json(output/"run_binding.generated.json")["evaluation_cpu"],serial_arms=True)
    atomic_json(output/"closed_loop_comparison.json",result);return result


CHECK_TESTS={
    "S3-T01":["test_off_import","test_off_record"],"S3-T02":["test_features_are_readonly_cheap"],"S3-T03":["test_permutation_and_identifier"],
    "S3-T04":["test_features_are_readonly_cheap_and_forbidden","test_test_context"],"S3-T05":["test_real_parent"],"S3-T06":["test_masks","test_unsupported","test_auxiliary_masks"],
    "S3-T07":["test_runtime_fallback","test_fingerprint"],"S3-T08":["test_off_record","test_reduced_budget","test_guided_initial"],
    "S3-T09":["test_high_advisor_score"],"S3-T10":["test_split","test_seed"],
    "S3-T11":["test_budget","test_rotation"],"S3-T12":["test_replay"]}


def run_targeted_checks(output,ledger,*,external_xmls=()):
    """Map actually executed pytest node names; never turn missing checks into PASS."""
    from .campaign import ROOT,run_subprocess
    output=Path(output);report=output/"targeted_pytest.xml"
    files=["tests/test_rotating_slots_stage3_campaign.py"] if external_xmls else ["tests/test_rotating_slots_stage3_runtime.py","tests/test_rotating_slots_stage3_model.py","tests/test_rotating_slots_stage3_campaign.py"]
    command=[sys.executable,"-m","pytest",*files,"-q","--junitxml",str(report)]
    receipt=run_subprocess(ledger,"preflight_tests","targeted_checks",command,output/"logs")
    return record_targeted_checks(output,[*external_xmls,report],receipt)


def record_targeted_checks(output,xml_paths,receipt):
    """Merge actual test results; a later run supersedes the same pytest node."""
    output=Path(output);executed={}
    for xml_path in xml_paths:
        if Path(xml_path).exists():
            for case in ET.parse(xml_path).iter("testcase"):
                key=(case.attrib.get("classname",""),case.attrib.get("name",""))
                executed[key]=dict(name=key[1],source_xml=str(xml_path),passed=not any(case.find(tag) is not None for tag in ("failure","error","skipped")))
    rows=[]
    for check,names in CHECK_TESTS.items():
        matches=[r for r in executed.values() if any(name in r["name"] for name in names)]
        rows.append(dict(id=check,status="PASS" if matches and all(r["passed"] for r in matches) else "NOT_COVERED" if not matches else "FAIL",
            tests=[r["name"] for r in matches],human_status="HUMAN_PENDING" if check=="S3-T12" else "NOT_APPLICABLE"))
    value=dict(checks=rows,pytest_receipt=receipt,source_xmls=list(map(str,xml_paths)),
        merged_executed_test_count=len(executed),merged_failed_tests=[dict(classname=key[0],**row) for key,row in executed.items() if not row["passed"]],
        all_automated_pass=receipt.get("exit_code")==0 and all(r["status"]=="PASS" for r in rows) and all(r["passed"] for r in executed.values()),human_visual_review="HUMAN_PENDING")
    atomic_json(output/"targeted_checks.json",value);return value
