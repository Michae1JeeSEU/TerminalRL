"""Explicit v3 action or opt-in cold checkpoint diagnostic; never a matrix.

Historical checkpoints contain physical boundary state and pending proposals,
not the source/future planning snapshots. A boundary diagnostic must identify
that difference explicitly; it is not a replay of the historical solver input.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import fields, is_dataclass
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.stage3 import run_selective_reoptimization_a as base
from scripts.stage3 import run_selective_reoptimization_repair as old

DRIVER = Path(__file__).resolve()
OUTPUT_NAME = "rs_selective_repair_v3_20261029"
RULE = "opportunity_executable_v3"
VARIANTS = (old.RULE, RULE)
POLICY = dict(cap_child_wall_s=5400., prior_audits_child_wall_s=0.,
    historical_campaign_used_s=7323.47022066447, campaign="rs_selective_repair_v3_20261029",
    previous_repair_settled_child_wall_s=1477.5308457711415,
    previous_repair_ledger_readonly="rs_selective_repair_20261029/repair_budget_ledger.json")
PROBE_IDS = ("initial_middle_01", "initial_middle_02")
CONTINUATION_NAME = "s3r23_serialization_fix"
CONTINUATION_CHECKPOINT_SHA256 = "091469fdf2ca568885241953c84df3c37f7813488e3e10c0904ab909b35073fc"
CONTINUATION_PARENT_SOURCE_HASH = "2a5629bf756be0b48273b1f47523818370d0fa23f8a4f787f35b06e301a0ac76"
CONTINUATION_BOUNDARIES = tuple(range(65,301,5))


def ledger_transaction(output, *, write=True):
    return old.ledger_transaction(output, write=write, policy=POLICY)


def run_bounded_job(output, job_id, command, *, cap_s, poll_s=.05):
    return old.run_bounded_job(output, job_id, command, cap_s=cap_s, poll_s=poll_s, policy=POLICY)


def credit_observed_receipt(output, job_id, path):
    """Charge one measured receipt once, including failed attempts; run no job."""
    path=Path(path).resolve(); receipt=base.load(path)
    wall_ms=receipt.get("wall_ms")
    base.require(type(wall_ms) in (float,int) and math.isfinite(wall_ms) and wall_ms>=0.,
                 "finite nonnegative measured wall_ms required")
    base.require(bool(receipt.get("command")) and type(receipt.get("exit_code")) is int
                 and receipt.get("start_utc") and receipt.get("end_utc"), "complete measured receipt evidence required")
    wall_s=wall_ms/1000.
    with ledger_transaction(output) as ledger:
        base.require(job_id not in ledger["jobs"], "cost/job already recorded")
        base.require(not any(row.get("receipt_path")==str(path) or row.get("receipt")==receipt
                             for row in ledger["jobs"].values()), "receipt already recorded")
        ledger["jobs"][job_id]=dict(status="credited_preexisting_measurement",child_wall_s=wall_s,
            receipt_path=str(path),receipt=deepcopy(receipt),exit_code=receipt["exit_code"],
            original_status=receipt.get("status"),evidence_type=receipt.get("evidence_type"))
        ledger["settled_child_wall_s"]+=wall_s
        exhausted=old.active_elapsed(ledger)>=ledger["cap_child_wall_s"]
    return dict(status="CREDITED",child_wall_s=wall_s,receipt_path=str(path),
                original_exit_code=receipt["exit_code"],budget_exhausted=exhausted)


def validated_output(path):
    output = Path(path).resolve()
    base.require(output.name == OUTPUT_NAME and output.parent == (Path.home()/"training_outputs/terminal").resolve(),
                 "v3 outputs must use the authorized fresh server directory")
    return output


def treatment(variant, *, diagnostic=False):
    base.require(variant in VARIANTS, "explicit supported selector variant required")
    base.require(diagnostic or variant == RULE, "v2 is diagnostic-only in the v3 runner")
    row = old.treatment()
    row.update(protocol="rs_selective_executable_v3_20261029", selective_rule_variant=variant,
               diagnostic_only=diagnostic)
    return row


def _tagged_mapping(value):
    """Read just a tagged mapping without importing the physical codec/classes."""
    base.require(value.get("$") in ("dict", "OrderedDict", "section"), "checkpoint tagged mapping required")
    return dict(value["items"])


def _tagged_time(value):
    return float.fromhex(value["hex"]) if isinstance(value, dict) and value.get("$") == "float" else value


def json_view(value):
    """Avoid dataclasses.asdict deepcopy of immutable mapping proxy fields."""
    if is_dataclass(value): return {field.name:json_view(getattr(value,field.name)) for field in fields(value)}
    if hasattr(value,"items"): return {str(key):json_view(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)): return [json_view(item) for item in value]
    if isinstance(value,(set,frozenset)): return sorted(json_view(item) for item in value)
    if isinstance(value,Path): return str(value)
    if isinstance(value,float) and math.isinf(value): return {"$nonfinite_float":"+inf" if value>0 else "-inf"}
    return value


def checkpoint_observation(path, expected_source_time):
    path = Path(path); envelope = base.load(path); payload = envelope["payload"]
    base.require(envelope["sha256"] == base.digest(payload), "checkpoint content digest changed")
    saved = _tagged_mapping(payload["scheduler"])
    world = _tagged_mapping(payload["world"])
    packets = _tagged_mapping(world["_planning_latency_packets"]) if "_planning_latency_packets" in world else {}
    packet_rows=[]
    for token,encoded in packets.items():
        packet=_tagged_mapping(encoded)
        packet_rows.append(dict(token=token, timing={key:_tagged_time(value) for key,value in packet["timing"]["fields"]},
            preparation_source_time=_tagged_time(dict(packet["source"]["fields"])["time"]),
            preparation_projected_time=_tagged_time(dict(packet["projected"]["fields"])["time"]),
            meaning="PREPARATION_SNAPSHOT_NOT_ORIGINAL_NORMAL_CALLBACK_INPUT"))
    pending = _tagged_mapping(saved["pending"]) if saved.get("pending") is not None else None
    pending_timing = ({key:_tagged_time(value) for key,value in pending["timing"]["fields"]}
                      if pending else None)
    difference = payload["time"]-expected_source_time
    return dict(checkpoint=str(path.resolve()), checkpoint_sha256=base.file_hash(path),
        checkpoint_payload_sha256=envelope["sha256"], checkpoint_identity=payload["identity"],
        checkpoint_time=payload["time"], expected_historical_source_time=expected_source_time,
        boundary_minus_historical_source_s=difference, pending_keys=sorted(pending) if pending else [],
        pending_timing=pending_timing, original_source_snapshot_serialized=False,
        original_future_snapshot_serialized=False, historical_input_recovered=False,
        saved_prepared_packets=packet_rows,
        pending_handling="OLD_CALLBACK_AND_UNEXECUTED_PROPOSAL_NOT_RESUMED_PHYSICAL_COMMITMENTS_PRESERVED",
        prepared_packet_resources="SEPARATE_REGISTRY_NOT_IN_PHYSICAL_SNAPSHOT_RESERVATIONS",
        input_basis="COLD_PHYSICAL_CHECKPOINT_BOUNDARY",
        time_relation="SAME_TIME_BOUNDARY_NOT_ORIGINAL_INPUT" if abs(difference)<1.e-9 else "DIFFERENT_TIME_BOUNDARY")


def write_new_json(path, value):
    """Exclusive creation: reruns cannot overwrite evidence, even on failure."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(json_view(value), stream, sort_keys=True, indent=2, allow_nan=False)


def probe_combinations(diagnostic, ids=PROBE_IDS):
    """At most one existing immediate MOVE per UID; no new column generation."""
    candidates = diagnostic.get("candidates", [])
    effective = diagnostic.get("snapshot_time")
    by_id = {}
    for aid in ids:
        rows = [row for row in candidates if row["aircraft_id"] == aid and row["kind"] == "MOVE"
                and row.get("in_catalog", True) and row.get("first_start") is not None
                and effective is not None and abs(row["first_start"]-effective) <= 1.e-7]
        by_id[aid] = sorted(rows, key=lambda row: row["candidate_hash"])
    chosen = {aid:rows[0]["candidate_hash"] for aid,rows in by_id.items() if rows}
    cases = [dict(label=aid, forced_choices={aid:chosen[aid]}) for aid in ids if aid in chosen]
    if len(chosen) == len(ids): cases.append(dict(label="combined", forced_choices=chosen))
    return dict(candidate_policy="FIRST_HASH_SORTED_EXISTING_IMMEDIATE_MOVE_PER_UID",
        candidate_hashes_by_aircraft={aid:[row["candidate_hash"] for row in rows] for aid,rows in by_id.items()},
        missing_immediate_move_ids=[aid for aid in ids if not by_id[aid]], cases=cases[:3])


def binding(args):
    source = base.load(args.source_freeze); base.verify_sources(source)
    required = (DRIVER.relative_to(PROJECT).as_posix(),
                "tests/test_rotating_slots_selective_repair_v3_campaign.py")
    base.require(all(name in source["files"] for name in required), "reviewed v3 runner/tests freeze required")
    if args.stage != "snapshot":
        with ledger_transaction(args.output) as ledger:
            bound = ledger.setdefault("formal_source_hash", source["source_hash"])
            base.require(bound == source["source_hash"], "formal source differs from first attempted smoke/arm")
    origin_path = args.origin_output/"resolved_run_binding.json"; origin = base.load(origin_path)
    base.require(origin["binding_id"] == base.digest({k:v for k,v in origin.items() if k != "binding_id"}), "origin binding changed")
    item = deepcopy(origin["inputs"][args.root])
    base.require(base.file_hash(item["run_checkpoint"]) == item["initial_checkpoint_sha256"], "original t0 changed")
    base.require(item["row"]["seed"] == base.ROOTS[args.root][0], "original root seed changed")
    return dict(schema="rs_selective_repair_binding_v3", source=source, source_commit=source["commit"],
        source_hash=source["source_hash"], inputs={args.root:item}, root=args.root, seed=base.ROOTS[args.root][0],
        origin_input_binding=origin["binding_id"], origin_input_binding_sha256=base.file_hash(origin_path),
        treatment=treatment(args.variant, diagnostic=args.stage=="snapshot"),
        episode_seconds=300, new_child_wall_cap_s=POLICY["cap_child_wall_s"],
        previous_repair_ledger=deepcopy(POLICY), no_automatic_retry=True, human_replay_status="HUMAN_PENDING")


def restore_cold_with_history(checkpoint,d,directory):
    """Restore all physical state and capture saved scheduler bytes only."""
    cold=base.ColdRestore()
    identity=base.load(checkpoint)["payload"]["identity"]
    world=d.restore_checkpoint(checkpoint,cold,identity,stream_copy=directory/"restored_states.jsonl")
    world.configure_practical_runtime(True)
    return world,cold.saved["callback"]


def restore_selector_history(callback,saved):
    selective=saved.get("selective_replanning")
    base.require(selective is not None and selective["method"]=="RULE_LOCAL_K8", "saved selector history required")
    callback.selective_history.load_state_dict(deepcopy(selective["history"]))
    callback.selective_blockers=deepcopy(selective.get("blockers",[]))
    return dict(history=callback.selective_history.state_dict(),blockers=deepcopy(callback.selective_blockers),
        basis="SAVED_CHECKPOINT_SELECTOR_HISTORY",candidate_cache="COLD_NOT_HISTORICAL_WARM_CACHE",
        historical_callback_records_resumed=False)


def arm(args, d):
    treatment(args.variant)
    old.set_cpu(args,d)
    directory = args.output/("smokes" if args.stage=="smoke" else "arms")/args.root
    directory.mkdir(parents=True,exist_ok=False)
    row = binding(args); row["episode_seconds"] = 10 if args.stage=="smoke" else 300
    row.update(cpu=args.cpu, run_role="SMOKE_NOT_FORMAL" if args.stage=="smoke" else "FORMAL_SINGLE_ARM")
    row["binding_id"] = base.digest(row)
    write_new_json(directory/"resolved_run_binding.json",row)
    result = base.closed_job(directory,args.root,"RULE_LOCAL_K8",row,d,lead_seconds=20.,
        selector_variant=args.variant,episode_seconds=row["episode_seconds"],opportunity_seed_offset=0,
        restart_policy=old.RESTART)
    write_new_json(directory/"arm.finished.json",result)
    return result


def continuation_inputs(args):
    """Bind the one failed medium arm's saved 60s state, never a t0 retry."""
    base.require(args.root=="s3r23" and args.cpu==12 and args.variant==RULE,"fixed s3r23 CPU12 v3 continuation required")
    parent=args.output/"arms/s3r23"
    expected=parent/"evaluation/s3r23/RULE_LOCAL_K8/checkpoint_0060.json"
    base.require(Path(args.checkpoint).resolve()==expected.resolve(),"only the original failed arm's checkpoint0060 is authorized")
    base.require(base.file_hash(expected)==CONTINUATION_CHECKPOINT_SHA256,"authorized continuation checkpoint bytes differ")
    envelope=base.load(expected);payload=envelope["payload"]
    base.require(envelope["sha256"]==base.digest(payload) and payload["time"]==60.,"continuation checkpoint digest/time differs")
    base.require(payload["identity"]["source_hash"]==CONTINUATION_PARENT_SOURCE_HASH
                 and payload["identity"]["manifest_sha256"]==base.RAW_HASHES["s3r23"][0],"continuation checkpoint source/root differs")
    old_binding_path=parent/"resolved_run_binding.json";old_binding=base.load(old_binding_path)
    base.require(old_binding["binding_id"]==base.digest({k:v for k,v in old_binding.items() if k!="binding_id"}),"failed arm binding digest differs")
    base.require(old_binding["source_hash"]==CONTINUATION_PARENT_SOURCE_HASH and old_binding["seed"]==7324
                 and old_binding["treatment"]==treatment(RULE),"failed arm treatment/seed/source differs")
    source=base.load(args.source_freeze);base.verify_sources(source)
    base.require(DRIVER.relative_to(PROJECT).as_posix() in source["files"] and
        "tests/test_rotating_slots_selective_repair_v3_campaign.py" in source["files"],"reviewed continuation runner/tests freeze required")
    with ledger_transaction(args.output,write=False) as ledger:
        failed=deepcopy(ledger["jobs"].get("arm_s3r23",{}))
        base.require(failed.get("status")=="failed" and failed.get("exit_code")==1,"original FAILED arm receipt must remain present")
        base.require(ledger.get("formal_source_hash")==CONTINUATION_PARENT_SOURCE_HASH,"original formal source binding changed")
    summary=payload["boundary_summary"]
    base.require(summary["time"]==60. and summary["root_id"]=="s3r23","prefix boundary summary differs")
    provenance=dict(schema="rs_selective_checkpoint_continuation_v1",reason="AUTHORIZED_SERIALIZATION_METADATA_FIX",
        ledger_job_id="continuation_"+CONTINUATION_NAME,
        source_checkpoint=str(expected),source_checkpoint_sha256=CONTINUATION_CHECKPOINT_SHA256,
        source_checkpoint_payload_sha256=envelope["sha256"],resume_sim_time=60.,target_sim_time=300.,
        parent_arm_binding=str(old_binding_path),parent_arm_binding_sha256=base.file_hash(old_binding_path),
        parent_arm_binding_id=old_binding["binding_id"],old_source_hash=old_binding["source_hash"],old_source_commit=old_binding["source_commit"],
        old_freeze_basis="EMBEDDED_SOURCE_MANIFEST_IN_PARENT_ARM_BINDING",new_source_freeze=str(args.source_freeze),
        new_source_hash=source["source_hash"],new_source_commit=source["commit"],original_failed_job=failed,
        original_failed_child_wall_s=failed.get("child_wall_s"),prefix_boundary_costs=deepcopy(summary["costs"]),
        prefix_frames_policy="RESTORE_ALL_SAVED_0_TO_60_FRAMES_EVENTS_TRAJECTORIES",
        pending_policy="RESTORE_FULL_WRAPPER_AND_PREPARED_PACKET_NO_PROPOSAL_RESOLVE",
        unsaved_cache_policy=dict(provider_validation_cache="NOT_SERIALIZED_STARTS_EMPTY",process_global_accelerators="NOT_SERIALIZED",
            reference_cache_templates_previous="RESTORED_FROM_CALLBACK_STATE",world_energy_cache="RESTORED_FROM_WORLD_STATE",
            exact_future_wall_service_reproduction=False),
        interruption_policy="NO_RECOMPUTATION_AT_OR_BEFORE_60_REMAINING_65_TO_300_AFTER_PROCESS_RESTART",
        accounting_scope="PREFIX_BOUNDARY_COST_PLUS_CONTINUATION_ONLY_FAILED_SERIALIZATION_OVERHEAD_EXCLUDED",
        original_failed_output_readonly=str(parent),historical_callback_and_service_records_counted_once=True)
    row=deepcopy(old_binding)
    row.pop("binding_id",None)
    row.update(schema="rs_selective_repair_continuation_binding_v3",source=source,source_hash=source["source_hash"],
        source_commit=source["commit"],run_role="CHECKPOINT_CONTINUATION_SAME_EPISODE",continuation=provenance)
    row["binding_id"]=base.digest(row)
    return payload,row


def restore_continuation(checkpoint,payload,d,directory):
    """Restore scheduler and world together; intentionally never ColdRestore."""
    callback=base.scheduler(d,"RULE_LOCAL_K8",selector_variant=RULE,opportunity_seed_offset=0)
    wrapper=d.LatencyAwareScheduler(callback,lead_seconds=20.,preparation_margin=1.,
        search_handoff_reserve_seconds=4.,restart_policy=old.RESTART)
    world=d.restore_checkpoint(checkpoint,wrapper,payload["identity"],stream_copy=directory/"restored_states.jsonl")
    world.decision_callback,world.admission_callback=wrapper,d.shared_fifo_admission
    base.require(world.time==60. and world.scheduler_period==5.,"restored continuation time/heartbeat differs")
    return world,callback,wrapper


def continuation_invariants(world,callback,wrapper,payload,codec):
    """Observe exact persisted state before any callback, publication or advance."""
    saved_world=codec.decode(payload["world"]);saved_scheduler=codec.decode(payload["scheduler"])
    saved_callback=saved_scheduler["callback"];actual_callback=callback.state_dict()
    checks=dict(world_serialized_fields_match=all(codec.encode(getattr(world,key))==codec.encode(value)
        for key,value in saved_world.items()),
        pending_matches=codec.encode(wrapper.pending)==codec.encode(saved_scheduler["pending"]),
        wrapper_records_match=codec.encode(wrapper.records)==codec.encode(saved_scheduler["records"]),
        online_services_match=codec.encode(wrapper.online_services)==codec.encode(saved_scheduler["online_services"]),
        selector_history_matches=actual_callback["selective_replanning"]["history"]==saved_callback["selective_replanning"]["history"],
        previous_matches=actual_callback["previous"]==saved_callback["previous"],
        reference_cache_matches=actual_callback["references"]==saved_callback["references"],
        reference_templates_match=actual_callback["templates"]==saved_callback["templates"],
        reference_hits_match=actual_callback["reference_hits"]==saved_callback["reference_hits"])
    wrapper_state=wrapper.state_dict()
    checks["wrapper_simulation_clock_and_restart_matches"]=all(codec.encode(wrapper_state[key])==codec.encode(saved_scheduler[key])
        for key in ("dirty","idle_service","handoff_service","service_until","restart_policy","restart_requested","last_normal_start_time","last_restart_time"))
    pending=wrapper.pending;packets=getattr(world,"_planning_latency_packets",{})
    checks["prepared_pending_packet_present"]=bool(pending and pending["stage"]=="PREPARED" and pending["token"] in packets)
    times=[record["time"] for record in world.records]
    checks["complete_saved_0_to_60_frame_prefix"]=bool(times and times[0]==0. and times[-1]==60. and all(a<b for a,b in zip(times,times[1:])))
    return dict(status="PASS" if all(checks.values()) else "FAIL",checks=checks,physical_execution=False,solver_called=False,
        time=world.time,frame_count=len(times),first_frame_time=times[0] if times else None,last_frame_time=times[-1] if times else None,
        callback_record_count=len(callback.records),normal_task_count=len(wrapper.records),online_service_count=len(wrapper.online_services),
        active_count=len(world.active),soft_count=len(world.soft),pending_stage=pending["stage"] if pending else None,
        pending_due=pending["due"] if pending else None,provider_validation_cache_size=len(callback.provider.validation_cache),
        comparison_scope="ALL_SERIALIZED_WORLD_FIELDS_AND_PERSISTED_SCHEDULER_STATE_NO_ADVANCE")


def continuation_costs(prefix,*,job_wall_s,process_cpu_s,advance_wall_s,advance_cpu_s):
    return dict(job_elapsed_wall_s=prefix["job_elapsed_wall_s"]+job_wall_s,
        process_cpu_s=prefix["process_cpu_s"]+process_cpu_s,
        world_advance_wall_s=prefix["world_advance_wall_s"]+advance_wall_s,
        world_advance_cpu_s=prefix["world_advance_cpu_s"]+advance_cpu_s,
        continuation_job_wall_s=job_wall_s,continuation_process_cpu_s=process_cpu_s,
        continuation_world_advance_wall_s=advance_wall_s,continuation_world_advance_cpu_s=advance_cpu_s,
        cumulative_cost_scope="PREFIX_BOUNDARY_COST_PLUS_CONTINUATION_ONLY_FAILED_SERIALIZATION_OVERHEAD_EXCLUDED",
        ledger_cost_scope="ORIGINAL_FAILED_CHILD_AND_CONTINUATION_CHILD_CHARGED_SEPARATELY_NO_PREFIX_RECREDIT")


def continuation(args,d):
    from terminal_marl.core.rotating_slots.stage2_campaign import Stage2Codec
    from terminal_marl.core.rotating_slots.world import AIRBORNE
    started,cpu_started=time.perf_counter(),time.process_time()
    payload,row=continuation_inputs(args)
    if args.stage=="continue":
        check_root=args.output/"restore_checks"/CONTINUATION_NAME
        checked=base.load(check_root/"result.json");checked_binding=base.load(check_root/"resolved_run_binding.json")
        base.require(checked.get("status")=="RESTORE_ONLY_PASS" and checked.get("physical_execution") is False,
                     "successful restore-only validation required before continuation")
        base.require(checked_binding["source_hash"]==row["source_hash"] and
            checked_binding["continuation"]["source_checkpoint_sha256"]==CONTINUATION_CHECKPOINT_SHA256,
            "restore-only validation used a different source or checkpoint")
    old.set_cpu(args,d)
    arm_root=args.output/("restore_checks" if args.stage=="restore-check" else "continuations")/CONTINUATION_NAME
    arm_root.mkdir(parents=True,exist_ok=False)
    directory=arm_root/"evaluation/s3r23/RULE_LOCAL_K8";directory.mkdir(parents=True,exist_ok=False)
    write_new_json(arm_root/"resolved_run_binding.json",row)
    write_new_json(arm_root/"continuation_provenance.json",row["continuation"])
    world,callback,wrapper=restore_continuation(args.checkpoint,payload,d,directory)
    invariants=continuation_invariants(world,callback,wrapper,payload,Stage2Codec())
    write_new_json(arm_root/"restore_invariants.json",invariants)
    base.require(invariants["status"]=="PASS","restored continuation state invariants failed")
    if args.stage=="restore-check":
        result=dict(status="RESTORE_ONLY_PASS",restore_invariants=str(arm_root/"restore_invariants.json"),
            physical_execution=False,solver_called=False,restored_time=world.time,pending_not_recomputed=True)
        write_new_json(arm_root/"result.json",result)
        return result
    with ledger_transaction(args.output) as ledger:
        source_hash=ledger.setdefault("continuation_serialization_fix_source_hash",row["source_hash"])
        base.require(source_hash==row["source_hash"],"continuation source differs from its first attempt")
    # Prefix boundary artifacts retain their original binding/identity verbatim.
    parent_directory=Path(args.checkpoint).parent
    for boundary in range(5,61,5):
        source_path=parent_directory/f"boundary_{boundary:04d}.json"
        base.require(source_path.is_file(),"saved prefix boundary missing")
        shutil.copyfile(source_path,directory/source_path.name)
    shutil.copyfile(args.checkpoint,directory/"checkpoint_0060.json")
    summary=payload["boundary_summary"];prefix=summary["costs"]
    initial_counts=deepcopy(summary["initial_counts"])
    holding,restore_observer=d.evaluation.observe_executed_holding(world)
    holding.update(summary["holding_energy_by_ring"])
    advance_wall=advance_cpu=0.;completed=60;error=None
    identity=dict(payload["identity"],source_hash=row["source_hash"])
    try:
        for boundary in CONTINUATION_BOUNDARIES:
            d.atomic_json(arm_root/"progress/s3r23_RULE_LOCAL_K8.json",dict(root="s3r23",method="RULE_LOCAL_K8",pid=os.getpid(),
                phase="CONTINUING_FROM_SAVED_60",completed_boundary_s=completed,target=boundary,continuation_job_wall_s=time.perf_counter()-started))
            before,before_cpu=time.perf_counter(),time.process_time()
            world.advance(float(boundary),wrapper,settle_only_at_end=True)
            advance_wall+=time.perf_counter()-before;advance_cpu+=time.process_time()-before_cpu
            if abs(world.time-boundary)>1.e-7 or world.failed or getattr(world,"terminated_invalid",False):break
            completed=boundary
            quality=d.episode_metrics(world)
            quality.update(unresolved_low_soc_count=sum(a.status in AIRBORNE and a.soc<.25 for a in world.aircraft.values()),unresolved_low_soc_threshold=.25)
            costs=continuation_costs(prefix,job_wall_s=time.perf_counter()-started,process_cpu_s=time.process_time()-cpu_started,
                advance_wall_s=advance_wall,advance_cpu_s=advance_cpu)
            costs.update(dispatch_wall_s=sum(item.get("dispatch_wall_s",0.) for item in world.dispatch_timings),
                simulation_timing_scope=prefix["simulation_timing_scope"])
            boundary_row=dict(root_id="s3r23",method="RULE_LOCAL_K8",time=world.time,quality=quality,initial_counts=initial_counts,
                normal_lifecycle=base.lifecycle_summary(wrapper.records,callback.records,wrapper.online_services,d.latency),
                holding_energy_by_ring=dict(holding),costs=costs,pending=wrapper.pending is not None,source_binding=row["binding_id"],
                continuation=dict(resume_sim_time=60.,prefix_callbacks_inherited=True),
                selection_summary=base.module(PROJECT/"scripts/stage3/report_selective_reoptimization_a.py","_selective_cont_boundary").selection_summary(callback.records))
            d.save_checkpoint(directory/f"checkpoint_{boundary:04d}.json",world,wrapper,identity,boundary_summary=boundary_row)
            d.atomic_json(directory/f"boundary_{boundary:04d}.json",boundary_row)
            base.save_logs(directory,world,callback,wrapper,row,"s3r23",d)
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc))
    finally:
        restore_observer()
    # Finalization is explicit and only writes the new continuation tree.
    base.save_logs(directory,world,callback,wrapper,row,"s3r23",d)
    terminal=d.evaluation.terminal_observation(world)
    d.atomic_json(directory/"terminal_observation.json",terminal)
    d.save_checkpoint(directory/"checkpoint_terminal.json",world,wrapper,identity)
    record=world.to_dict()
    record.update(decisions=callback.records,stage3_trace=[dict(record_type="NormalTask",**item) for item in wrapper.records],
        stage2_provenance=dict(execution_scope=row),tentative_predictions=[dict(aircraft_id=aid,as_of=world.time,plans=[plan.to_dict() for plan in plans])
            for aid,plans in callback.provider.previous.items()])
    d.ReplayBundle.write(directory/"replay",record=record,code_commit=row["source_commit"],config_path=world.config.path,
        command=str(DRIVER)+" continue",seed=7324,eager=False)
    costs=continuation_costs(prefix,job_wall_s=time.perf_counter()-started,process_cpu_s=time.process_time()-cpu_started,
        advance_wall_s=advance_wall,advance_cpu_s=advance_cpu)
    result=dict(status="COMPLETE" if completed==300 and error is None else "EVALUATION_PARTIAL",root_id="s3r23",method="RULE_LOCAL_K8",
        completed_boundary_s=completed,terminal_observation=terminal,error=error,full_job_wall_s=costs["continuation_job_wall_s"],
        process_cpu_s=costs["continuation_process_cpu_s"],continuation_costs=costs,original_failed_child_wall_s=row["continuation"]["original_failed_child_wall_s"],
        actual_cpu=sorted(os.sched_getaffinity(0)),replay_status="WRITTEN_HUMAN_PENDING",source_binding=row["binding_id"],
        continued_from_sim_time=60.,inherited_callback_records=invariants["callback_record_count"],
        full_job_wall_scope="CONTINUATION_CHILD_ONLY_ORIGINAL_FAILED_CHILD_REMAINS_SEPARATE_IN_LEDGER")
    d.atomic_json(directory/"result.json",result)
    write_new_json(arm_root/"arm.finished.json",result)
    write_new_json(arm_root/"continuation_finished_provenance.json",dict(row["continuation"],continuation_costs=costs,
        final_status=result["status"],final_sim_time=world.time))
    if error:raise RuntimeError(str(error))
    return result


def snapshot(args, d):
    """One cold local solve and optional fixed-catalog probes; no publication."""
    from terminal_marl.core.rotating_slots.selective_replanning.diagnostics import DiagnosticConfig, CandidateDiagnostics, plain
    from terminal_marl.core.rotating_slots.performance.parallel import _clear_worker_accelerators
    old.set_cpu(args,d)
    observation = checkpoint_observation(args.checkpoint,args.expected_source_time)
    base.require(observation["checkpoint_identity"]["manifest_sha256"] == base.RAW_HASHES[args.root][0], "checkpoint root differs")
    base.require(abs(observation["boundary_minus_historical_source_s"])<1.e-9 or args.allow_boundary_difference,
                 "checkpoint is a different-time boundary; explicit --allow-boundary-difference required")
    directory = args.output/"snapshots"/args.root/args.case
    directory.mkdir(parents=True,exist_ok=False)
    row = binding(args); row["checkpoint_observation"] = observation; row["binding_id"] = base.digest(row)
    write_new_json(directory/"resolved_run_binding.json",row)
    reset_started=time.perf_counter(); reset=_clear_worker_accelerators(); reset_s=time.perf_counter()-reset_started
    world,saved_callback=restore_cold_with_history(args.checkpoint,d,directory)
    source=world.build_planning_snapshot(); before=source.version
    timing=d.PlanningTiming.from_lead(source.time,20.,1.)
    setup_started=time.perf_counter()
    callback=d.NormalSchedulerCallback("rh_cp_sat",planning_profile="downstream_service_v8",online_policy="progressive_v1",
        selective_method="RULE_LOCAL_K8",selective_rule_variant=args.variant,opportunity_seed_offset=0,diagnostics=DiagnosticConfig())
    history=restore_selector_history(callback,saved_callback)
    write_new_json(directory/"selector_history_before.json",json_view(history))
    setup_s=time.perf_counter()-setup_started
    started=time.perf_counter(); projected=d.project_visible_snapshot(source,timing.effective_time,timing=timing)
    projection_s=time.perf_counter()-started
    port=d.FuturePort(projected,source,started+19.,search_wall_deadline=started+15.,planning_wall_origin=started,
                      search_handoff_reserve_seconds=4.)
    callback_started=time.perf_counter(); callback(port); callback_s=time.perf_counter()-callback_started
    unchanged=world.build_planning_snapshot().version==before
    base.require(unchanged,"snapshot diagnostic changed physical state")
    problem=callback.last_diagnostic_problem
    diagnostic=None; probes=dict(status="NOT_REQUESTED",cases=[])
    if problem is not None:
        # This independent export retains the complete actual catalog regardless
        # of bounded telemetry limits; all solver observations remain in records.
        count=sum(len(columns) for columns in problem["catalog"].columns.values())
        collector=CandidateDiagnostics(DiagnosticConfig(max_candidates=count,max_trajectory_candidates=count),problem["snapshot"])
        for columns in problem["catalog"].columns.values():
            for column in columns: collector.candidate(column,trajectory=True)
        diagnostic=collector.export(problem["catalog"],problem["selected"])
        diagnostic["scope"]=json_view(problem["catalog"].scope)
        diagnostic["global_frame"]=json_view(problem["global_frame"])
        diagnostic["actual_selected_columns"]=json_view(problem["selected"])
        diagnostic["fixed_background_ids"]=list(problem["catalog"].scope.fixed_background_ids)
        diagnostic["catalog_metadata"]={name:json_view(getattr(problem["catalog"],name)) for name in
            ("slacks","attempts","incomplete","cache_hits","slack_reference_statuses","fixed_background","generation_coverage")}
    # Persist actual choice/catalog evidence BEFORE any extra validation work.
    actual=dict(schema="rs_selective_checkpoint_diagnostic_v3",checkpoint_observation=observation,
        actual_source_time=source.time,actual_source_version=source.version,projected_effective_time=projected.time,
        projected_version=projected.version,physical_source_unchanged=unchanged,physical_execution=False,
        actual_publication=False,cold_normal_cache=True,cache_reset=plain(reset),treatment=row["treatment"],
        selector_history_before=history,
        scheduler_records=plain(callback.records),actual_catalog=diagnostic,
        costs=dict(cache_reset_s=reset_s,callback_setup_and_history_io_s=setup_s,projection_s=projection_s,callback_s=callback_s),
        physical_unchanged_basis="REVEALED_PHYSICAL_SNAPSHOT_VERSION_INCLUDES_ACTIVE_SOFT_RESERVATIONS_AND_ENERGY",
        nonfinite_float_encoding="OBJECT_WITH_$nonfinite_float_SIGNED_INF",
        status="ACTUAL_CATALOG_SAVED" if problem is not None else "NO_RETAINED_CATALOG")
    write_new_json(directory/"actual_diagnostic.json",actual)
    if args.probe_immediate:
        plan=probe_combinations(diagnostic or {})
        probes=dict(status="COMPLETE" if problem is not None else "NO_RETAINED_CATALOG",**plan)
        results=[]
        for case in plan["cases"]:
            probe_started=time.perf_counter()
            result=callback.probe_last_choices(case["forced_choices"],seconds=args.probe_seconds)
            item=dict(**case,result=plain(result),extra_validation_wall_s=time.perf_counter()-probe_started)
            write_new_json(directory/("probe_"+case["label"]+".json"),item)
            results.append(item)
        probes["results"]=results
    unchanged_after=world.build_planning_snapshot().version==before
    base.require(unchanged_after,"counterfactual validation changed physical state")
    write_new_json(directory/"probe_results.json",probes)
    result=dict(status=actual["status"],actual_diagnostic=str(directory/"actual_diagnostic.json"),
        probes=str(directory/"probe_results.json"),physical_source_unchanged=unchanged_after,
        actual_publication=False,physical_execution=False,additional_formal_cp_calls=0,
        extra_validation_wall_s=sum(case["extra_validation_wall_s"] for case in probes.get("results",[])))
    write_new_json(directory/"result.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("snapshot","smoke","arm","restore-check","continue"))
    parser.add_argument("--output",type=Path,default=Path.home()/"training_outputs/terminal"/OUTPUT_NAME)
    parser.add_argument("--origin-output",type=Path,default=Path.home()/"training_outputs/terminal/rs_selective_reoptimization_a_20261027")
    parser.add_argument("--source-freeze",type=Path,required=True)
    parser.add_argument("--root",choices=tuple(old.ROOT_CPUS),required=True)
    parser.add_argument("--cpu",type=int,required=True)
    parser.add_argument("--variant",choices=VARIANTS,required=True)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--expected-source-time",type=float)
    parser.add_argument("--allow-boundary-difference",action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--probe-immediate",action="store_true")
    parser.add_argument("--probe-seconds",type=float,default=10.)
    parser.add_argument("--_child-job-id",help=argparse.SUPPRESS)
    args=parser.parse_args(argv); args.output=validated_output(args.output)
    treatment(args.variant,diagnostic=args.stage=="snapshot")
    if args.stage=="snapshot":
        base.require(args.checkpoint and args.expected_source_time is not None and args.case,
                     "snapshot requires explicit checkpoint/expected-source-time/case")
        base.require(args.case.isascii() and all(c.isalnum() or c in "_-" for c in args.case), "simple unique case name required")
        base.require(math.isfinite(args.expected_source_time) and args.expected_source_time>=0.,"finite historical source time required")
        base.require(0<args.probe_seconds<=10.,"probe validation allowance must be at most ten seconds")
    elif args.stage in ("restore-check","continue"):
        base.require(args.root=="s3r23" and args.cpu==12 and args.checkpoint and args.variant==RULE,
                     "explicit s3r23 CPU12 v3 checkpoint continuation required")
        base.require(not args.probe_immediate and not args.case,"snapshot case/probes forbidden for continuation")
    else:
        base.require(not args.checkpoint and not args.probe_immediate and not args.case,"snapshot options forbidden for rollout")
    job_id=("continuation_"+CONTINUATION_NAME if args.stage=="continue" else
            "restore_check_"+CONTINUATION_NAME if args.stage=="restore-check" else
            args.stage+"_"+args.root+("_"+args.case if args.stage=="snapshot" else ""))
    if not args._child_job_id:
        command=[sys.executable,str(DRIVER),*(sys.argv[1:] if argv is None else argv),"--_child-job-id",job_id]
        result=run_bounded_job(args.output,job_id,command,cap_s=1200. if args.stage in ("arm","continue") else 180.)
        base.print_result(dict(job_id=job_id,**result));return
    base.require(args._child_job_id==job_id,"child job identity differs")
    with ledger_transaction(args.output,write=False) as ledger:
        row=ledger["jobs"].get(job_id,{})
        base.require(row.get("status")=="running" and row.get("parent_pid")==os.getppid(),"active parent watchdog required")
    d=base.dependencies(PROJECT);d.launch_threads={key:os.environ.get(key) for key in base.THREADS}
    base.print_result(snapshot(args,d) if args.stage=="snapshot" else
                      continuation(args,d) if args.stage in ("restore-check","continue") else arm(args,d))


if __name__=="__main__":main()
