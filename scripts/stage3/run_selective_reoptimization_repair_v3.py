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
    parser.add_argument("stage",choices=("snapshot","smoke","arm"))
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
    else:
        base.require(not args.checkpoint and not args.probe_immediate and not args.case,"snapshot options forbidden for rollout")
    job_id=args.stage+"_"+args.root+("_"+args.case if args.stage=="snapshot" else "")
    if not args._child_job_id:
        command=[sys.executable,str(DRIVER),*(sys.argv[1:] if argv is None else argv),"--_child-job-id",job_id]
        result=run_bounded_job(args.output,job_id,command,cap_s=1200. if args.stage=="arm" else 180.)
        base.print_result(dict(job_id=job_id,**result));return
    base.require(args._child_job_id==job_id,"child job identity differs")
    with ledger_transaction(args.output,write=False) as ledger:
        row=ledger["jobs"].get(job_id,{})
        base.require(row.get("status")=="running" and row.get("parent_pid")==os.getppid(),"active parent watchdog required")
    d=base.dependencies(PROJECT);d.launch_threads={key:os.environ.get(key) for key in base.THREADS}
    base.print_result(snapshot(args,d) if args.stage=="snapshot" else arm(args,d))


if __name__=="__main__":main()
