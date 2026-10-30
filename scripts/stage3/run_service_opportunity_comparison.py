"""One bounded service-opportunity follow-up action; never launch a matrix.

Common physical checkpoints are copied byte-for-byte from the prior campaign.
All children, including smoke and audits, share the new 7200 child-second cap.
First-iteration smoke covers 20 simulation seconds (absolute t=20 through t=40);
the explicitly authorized supply-r2 and local-recovery smoke cover 25 seconds
(t=20 through t=45).
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.stage3 import run_selective_reoptimization_a as base
from scripts.stage3 import run_selective_reoptimization_repair as old
from scripts.stage3 import run_selective_common_comparison_a4 as comparison

DRIVER = Path(__file__).resolve()
OUTPUT_NAME = "rs_service_opportunity_20261030"
SUPPLY_R2_OUTPUT_NAME = "rs_service_opportunity_supply_r2_20261030"
LOCAL_RECOVERY_OUTPUT_NAME = "rs_local_recovery_20261031"
LOCAL_RECOVERY_RESTOREFIX_OUTPUT_NAME = "rs_local_recovery_restorefix_20261031"
OUTPUT_NAMES = (OUTPUT_NAME, SUPPLY_R2_OUTPUT_NAME, LOCAL_RECOVERY_OUTPUT_NAME,
                LOCAL_RECOVERY_RESTOREFIX_OUTPUT_NAME)
PROFILE = "service_opportunity_v1"
ORIGIN_NAME = "rs_selective_quality_a4_20261030"
COMMON_HASHES = {
    "s3r23": "0f67e40052c6d80ba35efebaf1e47fb113a1aca7046332d20b0c44f8cc29efc7",
    "s3r24": "255f3ce5a3c45b01d3fb59ce5069ce39dec73b420735759ecdd6ce510f5eda77",
}
COMMON_FILES = ("common.checkpoint.json", "boundary_summary.json", "initializer_tasks.json",
                "initializer_callback_records.json", "initializer_services.json")
CPUS = {("s3r23", "FULL_H"): 16, ("s3r23", "RULE_LOCAL_K8"): 17,
        ("s3r24", "FULL_H"): 18, ("s3r24", "RULE_LOCAL_K8"): 19}
POLICY = dict(cap_child_wall_s=7200., prior_audits_child_wall_s=0.,
    historical_campaign_used_s="SEE_READ_ONLY_PREVIOUS_LEDGER", campaign=OUTPUT_NAME,
    previous_ledger_readonly=ORIGIN_NAME+"/repair_budget_ledger.json",
    run_role="AUTHORIZED_SERVICE_OPPORTUNITY_FOLLOWUP", old_ledgers_reset=False)
CAPS = dict(prepare=120., arm=1200., audit=300., smoke=120.)


def validated_output(path):
    output = Path(path).resolve()
    base.require(output.name in OUTPUT_NAMES and
        output.parent == (Path.home()/"training_outputs/terminal").resolve(),
        "follow-up outputs must use the authorized fresh server directory")
    return output


def policy_for_output(output):
    """Retain history; a restore-only continuation consumes the old R3 budget."""
    output=validated_output(output)
    policy = dict(POLICY, campaign=output.name)
    if policy["campaign"] == SUPPLY_R2_OUTPUT_NAME:
        policy.update(previous_ledger_readonly=OUTPUT_NAME+"/repair_budget_ledger.json",
                      run_role="AUTHORIZED_SUPPLY_RESERVATION_REPAIR_FOLLOWUP")
    elif policy["campaign"] == LOCAL_RECOVERY_OUTPUT_NAME:
        policy.update(previous_ledger_readonly=SUPPLY_R2_OUTPUT_NAME+"/repair_budget_ledger.json",
                      run_role="AUTHORIZED_LOCAL_RECOVERY_FOLLOWUP")
    elif policy["campaign"] == LOCAL_RECOVERY_RESTOREFIX_OUTPUT_NAME:
        previous=output.parent/LOCAL_RECOVERY_OUTPUT_NAME/"repair_budget_ledger.json"
        base.require(previous.is_file(),"original local-recovery ledger required for remaining budget")
        previous_hash=base.file_hash(previous)
        ledger=base.load(previous)
        base.require(base.file_hash(previous)==previous_hash,"previous local-recovery ledger changed while read")
        base.require(ledger.get("campaign")==LOCAL_RECOVERY_OUTPUT_NAME and
                     ledger.get("cap_child_wall_s")==POLICY["cap_child_wall_s"],
                     "previous local-recovery campaign/cap binding differs")
        terminal={"completed","failed","BUDGET_LIMITED","credited_preexisting_measurement"}
        base.require(isinstance(ledger.get("jobs"),dict) and
                     all(row.get("status") in terminal for row in ledger["jobs"].values()),
                     "previous local-recovery jobs must all be terminal; no active/unknown jobs")
        used=ledger.get("settled_child_wall_s")
        base.require(type(used) in (int,float) and math.isfinite(used) and
                     0.<=used<POLICY["cap_child_wall_s"],"previous local-recovery settled budget invalid or exhausted")
        policy.update(cap_child_wall_s=POLICY["cap_child_wall_s"]-used,
            previous_ledger_readonly=LOCAL_RECOVERY_OUTPUT_NAME+"/repair_budget_ledger.json",
            previous_ledger_sha256=previous_hash,previous_campaign_settled_child_wall_s=used,
            shared_original_cap_child_wall_s=POLICY["cap_child_wall_s"],
            continuation_reason="COLD_COMMON_TRAFFIC_POLICY_RESTORE_FIX; PRIOR_PREFLIGHTS_REMAIN_CHARGED",
            run_role="AUTHORIZED_LOCAL_RECOVERY_RESTORE_FIX_CONTINUATION")
    return policy


def launch_threads():
    """Capture literal launch settings before loading numerical dependencies."""
    observed = {key: os.environ.get(key) for key in base.THREADS}
    base.require(all(value == "1" for value in observed.values()),
                 "five literal numerical thread settings of 1 required at launch")
    return observed


def copy_common(output, root, *, origin=None, expected_hash=None):
    """Read the historical archive only; never regenerate or rewrite its state."""
    output = Path(output)
    origin = output.parent/ORIGIN_NAME if origin is None else Path(origin)
    source = origin/"common_initial"/root
    target = output/"common_initial"/root
    checkpoint = source/"common.checkpoint.json"
    expected = COMMON_HASHES[root] if expected_hash is None else expected_hash
    base.require(base.file_hash(checkpoint) == expected, "authorized common checkpoint bytes differ")
    summary = base.load(checkpoint)["payload"]["boundary_summary"]
    base.require(summary["status"] == "COMMON_INITIAL_READY" and summary["t0"] == 20. and
                 summary["source_binding"]["root"] == root, "common root/time/readiness differs")
    files = [source/name for name in COMMON_FILES]
    # An attached recorder may reference a sibling file. Preserve its bytes in
    # the copy as well, while retaining the unmodified checkpoint's old path.
    sink = base.load(checkpoint)["payload"].get("sink")
    if sink is not None:
        stream = Path(sink["path"])
        base.require(stream.parent.resolve() == source.resolve() and stream.is_file(),
                     "common stream must be a readable sibling of the checkpoint")
        if stream not in files: files.append(stream)
    base.require(all(path.is_file() for path in files), "complete common ancillary files required")
    base.require(not target.exists(), "common copy already exists; no overwrite or retry")
    target.mkdir(parents=True, exist_ok=False)
    copied = {}
    for path in files:
        with path.open("rb") as src, (target/path.name).open("xb") as dst:
            shutil.copyfileobj(src, dst)
        copied[path.name] = base.file_hash(target/path.name)
        base.require(copied[path.name] == base.file_hash(path), "common copy bytes differ")
    return dict(root=root, t0=20., horizon=[20.,320.], source=str(source), files=copied,
                checkpoint_sha256=expected, initializer_regenerated=False,
                source_sink_reference="UNCHANGED_READ_ONLY_SOURCE; RESTORE_CLONES_BEFORE_WRITE")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", choices=tuple(CAPS), required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--root", choices=tuple(base.ROOTS), required=True)
    result.add_argument("--method", choices=base.METHODS, required=True)
    result.add_argument("--cpu", type=int, required=True)
    result.add_argument("--source-freeze", type=Path, required=True)
    result.add_argument("--evaluation-profile", choices=(PROFILE,), default=PROFILE)
    result.add_argument("--_child-job-id", help=argparse.SUPPRESS)
    return result


def main(argv=None):
    values = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(values)
    threads = launch_threads()
    args.output = validated_output(args.output)
    policy = policy_for_output(args.output)
    base.require(args.cpu == CPUS[args.root,args.method], "fixed physical CPU16-19 arm mapping required")
    job_id = "service_"+args.stage+"_"+args.root+("" if args.stage=="prepare" else "_"+args.method)
    if not args._child_job_id:
        command = [sys.executable, str(DRIVER), *values, "--_child-job-id", job_id]
        receipt = old.run_bounded_job(args.output, job_id, command, cap_s=CAPS[args.stage], policy=policy)
        base.print_result(dict(job_id=job_id, **receipt))
        return receipt
    base.require(args._child_job_id == job_id, "child identity differs")
    with old.ledger_transaction(args.output, write=False, policy=policy) as ledger:
        row = ledger["jobs"].get(job_id, {})
        base.require(row.get("status")=="running" and row.get("parent_pid")==os.getppid(),
                     "active parent watchdog required")
    source = base.load(args.source_freeze)
    base.verify_sources(source)
    required = (DRIVER.relative_to(PROJECT).as_posix(),
                "scripts/stage3/report_service_opportunity_comparison.py",
                "tests/test_service_opportunity_campaign.py")
    if args.output.name == LOCAL_RECOVERY_RESTOREFIX_OUTPUT_NAME:
        required += ("scripts/stage3/run_selective_common_initial_a4.py",
                     "tests/test_selective_common_initial_a4.py")
    base.require(all(name in source["files"] for name in required), "reviewed follow-up sources/tests required")
    with old.ledger_transaction(args.output, policy=policy) as ledger:
        bound = ledger.setdefault("service_source_hash", source["source_hash"])
        profile = ledger.setdefault("service_evaluation_profile", PROFILE)
        base.require(bound==source["source_hash"] and profile==PROFILE, "source/profile changed after first attempt")
    d = base.dependencies(PROJECT)
    d.launch_threads = threads
    old.set_cpu(argparse.Namespace(stage="comparison", cpu=args.cpu, root=args.root), d)
    if args.stage == "prepare":
        result = copy_common(args.output,args.root)
        comparison.write_new(args.output/"common_initial"/args.root/"copy_receipt.json", result)
    elif args.stage == "audit":
        saved = base.load(args.output/"arms"/args.root/args.method/"resolved_run_binding.json")
        base.require(saved["source_hash"]==source["source_hash"] and saved["evaluation_profile"]==PROFILE,
                     "saved arm source/profile differs")
        result = comparison.audit_arm(args.output,args.root,args.method,checkout=PROJECT)
    else:
        checkpoint = args.output/"common_initial"/args.root/"common.checkpoint.json"
        base.require(base.file_hash(checkpoint)==COMMON_HASHES[args.root], "common checkpoint changed")
        run_output = args.output/"smoke_only" if args.stage=="smoke" else args.output
        smoke_seconds = 25 if args.output.name in (SUPPLY_R2_OUTPUT_NAME, LOCAL_RECOVERY_OUTPUT_NAME,
                                                 LOCAL_RECOVERY_RESTOREFIX_OUTPUT_NAME) else 20
        result = comparison.run_arm(run_output,args.root,args.method,checkpoint,
            dict(source_commit=source["commit"],source_hash=source["source_hash"],cpu=args.cpu,
                 campaign=args.output.name,run_role="SMOKE_NOT_FORMAL" if args.stage=="smoke" else "FORMAL_SINGLE_ARM"),
            evaluation_profile=PROFILE,d=d,episode_seconds=smoke_seconds if args.stage=="smoke" else 300)
    base.print_result(result)
    return result


if __name__ == "__main__": main()
