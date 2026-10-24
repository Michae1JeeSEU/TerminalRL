"""One bounded online reevaluation of the already frozen Stage3 pilot model.

No seed scan, collection, labeling, training, restart, or budget reset is exposed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
import shutil
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from terminal_marl.core.rotating_slots.stage3_learning import campaign
from terminal_marl.core.rotating_slots.stage2_campaign import atomic_json, digest, load_json

MODEL_HASH = "b565c2a98af7f30c5b8179e823a6447bb1ccecebcbf4da2dfdcb44679ce85969"
NORMALIZER_HASH = "007de6c700959a8cadc0397e8896f279aeca0070fe12bbac5441c3c5b2e63491"
CASE_IDS = ("s3r21", "s3r22", "s3r23", "s3r24", "s3r17", "s3r18")
CORE = "terminal_marl/core/rotating_slots/"
# Explicit execution-only changes reviewed for this reevaluation. Everything else,
# including all physical sources, package contracts and configuration, must match.
ALLOWED_CHANGED = {
    CORE + "stage3_learning/features.py", CORE + "stage3_learning/model.py",
    CORE + "stage3_learning/evaluation.py", CORE + "replay.py",
    "terminal_marl/visualization/rotating_slots_controller.py",
    "scripts/stage3/benchmark_encoding.py", "scripts/stage3/reevaluate.py",
    "tests/test_rotating_slots_stage3_encoding_optimization.py",
    "tests/test_rotating_slots_stage3_replay_projection.py",
    "tests/test_rotating_slots_stage3_reevaluation.py",
}


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def freeze(output):
    """Local Git/source freeze only: deliberately never calls freeze_local."""
    output = Path(output).resolve()
    if not output.is_relative_to(campaign.ROOT):
        raise ValueError("source freeze must be inside local repository for Syncthing")
    if output.exists() and any(output.iterdir()):
        raise ValueError("source freeze already exists")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=campaign.ROOT).decode("utf-8", errors="replace")
    files = campaign.source_files()
    relative = [p.relative_to(campaign.ROOT).as_posix() for p in files]
    diff = git("diff", "--binary", "HEAD", "--", *relative)
    tracked = set(git("ls-files", "--", *relative).splitlines())
    if diff or set(relative) - tracked:
        raise ValueError("commit all execution sources before freezing")
    hashes = {name: file_hash(campaign.ROOT / name) for name in relative}
    value = dict(source_commit=git("rev-parse", "HEAD").strip(),
                 workspace_status=git("status", "--short"),
                 workspace_diff_hash=sha256(diff.encode()).hexdigest(),
                 source_files=hashes, source_hash=digest(hashes),
                 branch=git("branch", "--show-current").strip(), created_at=campaign.utc_now(),
                 freeze_kind="REEVALUATION_SOURCE_ONLY_NO_SEED_SCAN")
    output.mkdir(parents=True, exist_ok=True)
    (output / "workspace.diff").write_text(diff, encoding="utf-8")
    atomic_json(output / "source_freeze.json", value)
    return value


def verify_source_contract(frozen, old):
    campaign.verify_sources(frozen)
    current = {p.relative_to(campaign.ROOT).as_posix() for p in campaign.source_files()}
    if set(frozen["source_files"]) != current:
        raise ValueError("source freeze file inventory mismatch")
    changed = sorted(k for k in set(old["source_files"]) | current
                     if old["source_files"].get(k) != frozen["source_files"].get(k))
    if set(changed) - ALLOWED_CHANGED:
        raise ValueError("physical/protocol/config source drift: " + ", ".join(set(changed) - ALLOWED_CHANGED))
    if not frozen.get("source_commit"):
        raise ValueError("actual execution commit missing")
    return changed


def probe_environment(old):
    environment = campaign.resource_probe()
    if old["evaluation_cpu"] != 12 or 12 not in environment.get("free_physical_cores", ()):
        raise ValueError("inherited evaluation physical CPU12 is not verified free")
    if hasattr(os, "sched_getaffinity") and 12 not in os.sched_getaffinity(0):
        raise ValueError("inherited evaluation CPU12 is outside allowed affinity")
    if any(environment["thread_environment"].get(k) != "1" for k in campaign.THREAD_ENV):
        raise ValueError("numerical thread environment must equal1 before Python starts")
    for package in ("numpy", "scipy", "PyYAML", "ortools", "torch", "pytest"):
        version = environment["versions"].get(package)
        if version is None or version != old["environment"]["versions"].get(package):
            raise ValueError("numerical dependency mismatch: " + package)
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    environment.update(torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads())
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    model.AddBoolOr([model.NewBoolVar("reevaluation_probe")])
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    environment["cp_probe_status"] = solver.StatusName(solver.Solve(model))
    if environment["cp_probe_status"] != "OPTIMAL":
        raise ValueError("CP probe failed")
    return environment


def selected_roots(index):
    roots = index["roots"]
    tests = [r for r in roots if r["split"] == "test"]
    cases = (tests + [r for r in roots if r["split"] == "validation"])[:campaign.contract_config()["evaluation"]["same_snapshot_cases_max"]]
    if tuple(r["root_id"] for r in cases) != CASE_IDS:
        raise ValueError("frozen six-case order mismatch")
    medium = [r for r in tests if r["scenario_profile"] == "medium24_rate150_v1"]
    if not medium or medium[0]["root_id"] != "s3r23":
        raise ValueError("frozen closed-loop case mismatch")
    return cases


def validate_parent(parent, frozen):
    old = load_json(parent / "run_binding.generated.json")
    changed = verify_source_contract(frozen, old)
    if load_json(parent / "final_status.json").get("evaluation") != "COMPLETE":
        raise ValueError("parent evaluation not complete")
    card = load_json(parent / "model/model_card.json")
    if not card.get("model_frozen") or card.get("model_hash") != MODEL_HASH:
        raise ValueError("invalid frozen model card")
    if file_hash(parent / "model/model.pt") != MODEL_HASH or file_hash(parent / "model/feature_normalizer.json") != NORMALIZER_HASH:
        raise ValueError("invalid frozen model/normalizer bytes")
    if card["fingerprint"] != old["fingerprint"]:
        raise ValueError("frozen model fingerprint mismatch")
    split = load_json(parent / "split_manifest.json")
    payload = dict(split)
    if payload.pop("split_manifest_id") != digest(payload) or split["split_manifest_id"] != old["fingerprint"]["split_manifest_id"]:
        raise ValueError("frozen split identity mismatch")
    cfg = campaign.contract_config()
    from terminal_marl.core.rotating_slots.stage2_planning import PlanningBudget, HORIZON
    from terminal_marl.core.rotating_slots.config import load_service_profile
    physical = load_service_profile("service_half_v1")
    if (old["resolved_budgets"] != cfg["compute"] or old["planning_budget"] != PlanningBudget.for_profile("quality_v5").settings()
            or old["horizon_s"] != HORIZON or old["fingerprint"]["evaluation_protocol_id"] != digest(cfg["evaluation"])
            or old["fingerprint"]["physical_config_hash"] != physical.config_hash
            or old["inputs"]["physical_config_sha256"] != physical.raw_hash):
        raise ValueError("physical/protocol/planning budget mismatch")
    if old["contract_hash"] != file_hash(campaign.PACKAGE / cfg["contract_file"]):
        raise ValueError("contract mismatch")
    index = load_json(parent / "raw_snapshot_index.json")
    if not index.get("complete") or len(index["roots"]) != 24:
        raise ValueError("parent root index incomplete")
    cases = selected_roots(index)
    split_rows = {r["root_id"]: r for r in split["roots"]}
    from terminal_marl.core.rotating_slots.stage3_learning.data import validate_independent_initial_states, restore_root
    from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
    validate_independent_initial_states(index["roots"])
    for row in cases:
        rid = row["root_id"]
        directory = parent / "roots" / rid
        if load_json(directory / "root.json") != row or any(row.get(k) != v for k, v in split_rows[rid].items()):
            raise ValueError("root/split/index mismatch: " + rid)
        if digest(load_json(directory / "initial_state.json")) != row["initial_state_hash"] or digest(load_json(directory / "manifest.json")) != row["manifest_hash"]:
            raise ValueError("initial state/manifest mismatch: " + rid)
        checkpoint = load_json(directory / "initial.checkpoint.json")
        payload = checkpoint["payload"]
        if (checkpoint["sha256"] != digest(payload) or payload["identity"] != row["identity"]
                or payload["time"] != 0. or payload.get("sink") is not None):
            raise ValueError("checkpoint identity/content/sink mismatch: " + rid)
        local_row = dict(row, checkpoint_path=str(directory / "initial.checkpoint.json"))
        world = restore_root(local_row, NormalSchedulerCallback("rh_cp_sat", planning_profile="downstream_service_v8"))
        actual = dict(aircraft=[a.to_dict() for a in world.aircraft.values()], slots=world.slots.to_dict(), pads=world.pads.to_dict())
        if digest(actual) != row["initial_state_hash"]:
            raise ValueError("restored initial state mismatch: " + rid)
    # Strict model load exercises the unchanged compatibility gate with its
    # original fingerprint, never an invented new-source fingerprint.
    from terminal_marl.core.rotating_slots.stage3_learning.model import LearnedAdvisor
    LearnedAdvisor(parent / "model", fingerprint=old["fingerprint"], evaluation=True)._load()
    return old, index, cases, changed


def prepare(parent, output, source_freeze):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if parent == output or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError("reevaluation output must be separate from parent")
    if output.exists() and any(output.iterdir()):
        raise ValueError("nonempty output: prepare/restart/budget reset prohibited")
    frozen = load_json(Path(source_freeze) / "source_freeze.json")
    old, index, cases, changed = validate_parent(parent, frozen)
    environment = probe_environment(old)
    # All validation above is read-only. Failures after this point retain the
    # nonempty output, so another prepare can never grant a fresh budget.
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive claim also prevents two concurrent prepares from both passing
    # the initial empty-directory check and constructing separate ledgers.
    with (output / "prepare.started.json").open("x", encoding="utf-8") as stream:
        stream.write('{"status":"PREPARING"}\n')
    inputs = {}
    def copy(relative, destination=None):
        source = parent / relative
        target = output / (relative if destination is None else destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inputs[str(relative)] = file_hash(source)
        if file_hash(target) != inputs[str(relative)]:
            raise ValueError("copy hash mismatch: " + str(relative))
    for path in sorted((parent / "model").iterdir()):
        if path.is_file():
            copy(Path("model") / path.name)
    for name in ("split_manifest.json", "heldout_signal.json"):
        copy(name)
    for name in ("same_state_comparison.json", "closed_loop_comparison.json", "final_status.json", "run_binding.generated.json", "source_freeze.json", "heldout_access.json"):
        copy(name, Path("prior_evidence") / name)
    rows = deepcopy(index["roots"])
    for case in cases:
        rid = case["root_id"]
        for name in ("initial.checkpoint.json", "initial_state.json", "manifest.json"):
            copy(Path("roots") / rid / name)
    # Keep all24 roots in original order. Nonselected rows are metadata only;
    # evaluation accesses exactly the six materialized cases and no raw streams.
    for row in rows:
        directory = output / "roots" / row["root_id"]
        row["checkpoint_path"] = str(directory / "initial.checkpoint.json")
        row["raw_snapshot_path"] = str(directory / "initial_state.json")
        if row["root_id"] in CASE_IDS:
            atomic_json(directory / "root.json", row)
    atomic_json(output / "raw_snapshot_index.json", dict(index, roots=rows, materialized_root_ids=list(CASE_IDS)))
    atomic_json(output / "source_freeze.json", frozen)
    binding = deepcopy(old)
    provenance = dict(source_commit=frozen["source_commit"], source_hash=frozen["source_hash"],
                      workspace_diff_hash=frozen["workspace_diff_hash"], changed_source_files=changed,
                      environment=environment, frozen_model_contract_reused=True)
    binding.update(source_hash=frozen["source_hash"], source_files=frozen["source_files"],
                   execution_provenance=provenance, frozen_model_contract=deepcopy(old["fingerprint"]),
                   fingerprint_role="FROZEN_MODEL_COMPATIBILITY_CONTRACT_NOT_EXECUTION_PROVENANCE",
                   environment=environment, created_at=campaign.utc_now())
    atomic_json(output / "run_binding.generated.json", binding)
    prepared_files = ["run_binding.generated.json", "source_freeze.json", "raw_snapshot_index.json"]
    prepared_files += [f"roots/{rid}/root.json" for rid in CASE_IDS]
    atomic_json(output / "reevaluation_protocol.json", dict(
        status="PREPARED", parent_readonly=str(parent), input_sha256=inputs,
        source_hash=frozen["source_hash"], source_commit=frozen["source_commit"],
        model_hash=MODEL_HASH, normalizer_hash=NORMALIZER_HASH, same_state_case_ids=list(CASE_IDS),
        closed_loop_root_id="s3r23", same_state_job_cap_s=90., evaluation_cap_s=3600.,
        prior_evidence_reuse=["frozen_model", "frozen_normalizer", "frozen_split", "initial_checkpoints", "heldout_signal"],
        heldout_signal_policy="BYTE_IDENTICAL_PRIOR_EVIDENCE_NO_NEW_MEASUREMENT", training_authorized=False,
        execution_provenance=provenance))
    protocol = load_json(output / "reevaluation_protocol.json")
    protocol["prepared_sha256"] = {name: file_hash(output / name) for name in prepared_files}
    atomic_json(output / "reevaluation_protocol.json", protocol)
    ledger = campaign.CampaignLedger(output / "budget_ledger.json")
    ledger.pause()
    return binding


def verify_prepared(output):
    output = Path(output)
    protocol = load_json(output / "reevaluation_protocol.json")
    for relative, expected in protocol["prepared_sha256"].items():
        if file_hash(output / relative) != expected:
            raise ValueError("prepared binding/root metadata changed: " + relative)
    frozen = load_json(output / "source_freeze.json")
    campaign.verify_sources(frozen)
    binding = load_json(output / "run_binding.generated.json")
    if (binding["execution_provenance"]["source_hash"] != frozen["source_hash"]
            or binding["execution_provenance"]["source_commit"] != frozen["source_commit"]
            or binding["source_hash"] != frozen["source_hash"]
            or binding["source_files"] != frozen["source_files"]
            or binding["fingerprint"] != binding["frozen_model_contract"]):
        raise ValueError("actual source/frozen model contract mismatch")
    for relative, expected in protocol["input_sha256"].items():
        if relative in ("same_state_comparison.json", "closed_loop_comparison.json", "final_status.json", "run_binding.generated.json", "source_freeze.json", "heldout_access.json"):
            continue
        if file_hash(output / relative) != expected:
            raise ValueError("immutable reevaluation input changed: " + relative)
    selected_roots(load_json(output / "raw_snapshot_index.json"))
    if not (output / "budget_ledger.json").is_file():
        raise ValueError("existing ledger required; no budget reset")
    ledger = campaign.CampaignLedger(output / "budget_ledger.json")
    compute = campaign.contract_config()["compute"]
    if (ledger.data["phase_caps"] != compute["phase_caps_seconds"]
            or ledger.data["total_s"] != compute["campaign_wall_seconds"]
            or ledger.data["compute_stop_s"] != compute["compute_stop_seconds"]):
        raise ValueError("ledger caps changed")
    return protocol, binding, ledger


def run_stage(output, stage):
    from terminal_marl.core.rotating_slots.stage3_learning import evaluation
    output = Path(output)
    # campaign_lock is used only after prepare; no lock may populate an empty
    # output before prepare's strict nonempty refusal.
    with campaign.campaign_lock(output):
        protocol, binding, ledger = verify_prepared(output)
        guard = output / (stage + ".started.json")
        if guard.exists():
            raise ValueError("stage already attempted; no automatic retries")
        if stage == "evaluate":
            campaign.require_checks(output)
            if not load_json(output / "reevaluation_checks.json").get("passed"):
                raise ValueError("reevaluation/encoding checks required")
            probe_environment(binding)
        atomic_json(guard, dict(started_at=campaign.utc_now(), source_hash=binding["source_hash"]))
        try:
            with ledger.phase("preflight_tests" if stage == "tests" else "evaluation"):
                if stage == "tests":
                    checks = evaluation.run_targeted_checks(output, ledger)
                    receipt = campaign.run_subprocess(ledger, "preflight_tests", "reevaluation_encoding_checks",
                        [sys.executable, "-m", "pytest", "tests/test_rotating_slots_stage3_reevaluation.py",
                         "tests/test_rotating_slots_stage3_encoding_optimization.py", "-q", "--junitxml", str(output / "reevaluation_pytest.xml")], output / "logs")
                    passed = checks["all_automated_pass"] and receipt.get("exit_code") == 0
                    atomic_json(output / "reevaluation_checks.json", dict(passed=passed, receipt=receipt))
                    if not passed:
                        raise ValueError("reevaluation targeted checks failed")
                else:
                    evaluation.run_evaluation(output, ledger)
            atomic_json(output / (stage + ".finished.json"), dict(status="COMPLETED", finished_at=campaign.utc_now()))
        except Exception as exc:
            atomic_json(output / (stage + ".finished.json"), dict(status="FAILED", error_type=type(exc).__name__, error=str(exc)))
            raise
        finally:
            ledger.pause()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "prepare", "tests", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--source-freeze", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "freeze":
        freeze(args.output)
    elif args.stage == "prepare":
        if args.parent is None or args.source_freeze is None:
            parser.error("prepare requires --parent and --source-freeze")
        prepare(args.parent, args.output, args.source_freeze)
    else:
        run_stage(args.output, args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
