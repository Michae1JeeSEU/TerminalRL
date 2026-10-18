"""P0/P1 runtime integration. No core imports until byte-verified bootstrap.

Run through scripts/server/run_rotating_slots_performance.py in a fresh process.
The public CLI implements preflight, capture and bounded micro only.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

from . import provenance


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    return json.loads(Path(path).read_bytes())


def _write(path, data):
    from .checkpoint import atomic_write_json
    atomic_write_json(path, data)


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def parser():
    result = argparse.ArgumentParser(description="FT-RS-S1-PERF-1.0 P0/P1 runner (not performance qualification)")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("preflight", "capture", "micro"):
        command = commands.add_parser(name)
        command.add_argument("--reference", type=Path, required=True, help="byte-verified frozen R0 export")
        command.add_argument("--kernel", choices=("R0", "OPT"), default="R0")
        command.add_argument("--implementation-commit", required=True, help="exact locally supplied deployment commit label")
        command.add_argument("--deployment-manifest", type=Path, required=True)
        if name == "preflight":
            command.add_argument("--write-deployment-manifest", action="store_true",
                                 help="local-only explicit creation; never invokes server Git")
        else:
            command.add_argument("--output", type=Path, required=True, help="unique new attempt directory")
            command.add_argument("--inner-workers", type=int, choices=range(7), required=True)
            command.add_argument("--core-budget", type=int, required=True)
            command.add_argument("--process-budget", type=int, required=True)
            command.add_argument("--wall-budget", type=float, required=True, help="soft budget checked at natural cuts")
        if name == "capture":
            command.add_argument("--rate", type=int, choices=(60, 120, 180), required=True)
            command.add_argument("--seed", type=int, choices=(11, 23, 47, 71, 101), required=True)
            command.add_argument("--milestones", type=float, nargs="*", default=[])
            command.add_argument("--capture-until", type=float, required=True,
                                 help="request natural suspension; >= latest milestone + 60 seconds")
            command.add_argument("--checkpoint", type=Path, help="resume a complete checkpoint+runner sidecar in a NEW attempt")
            command.add_argument("--crash-evidence", type=Path,
                                 help="explicit JSON evidence for a crashed source attempt; timing remains INCOMPLETE")
            command.add_argument("--cache-mode", choices=("cold", "warm"), default="cold")
        if name == "micro":
            command.add_argument("--fixture", choices=("empty", "inner-hold", "service-events"), default="inner-hold")
            command.add_argument("--until", type=float, default=.4, help="fixed micro invocation target in (0,.4]")
            command.add_argument("--mode", choices=("benchmark", "profile"), default="benchmark")
    return result


def _existing_core_modules():
    return [name for name in sys.modules if name == "terminal_marl" or name.startswith("terminal_marl.")]


def bootstrap(args, helper_root):
    """Verify deployment and frozen bytes before importing any core instance."""
    helper_root = Path(helper_root).resolve()
    if _existing_core_modules():
        raise provenance.ProvenanceError("CORE_ALREADY_IMPORTED: use the stdlib bootstrap in a fresh process")
    if len(args.implementation_commit) != 40 or any(c not in "0123456789abcdef" for c in args.implementation_commit):
        raise provenance.ProvenanceError("IMPLEMENTATION_COMMIT_REQUIRES_EXACT_40_HEX")
    if getattr(args, "write_deployment_manifest", False):
        if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
            raise provenance.ProvenanceError("DEPLOYMENT_MANIFEST_CREATION_IS_LOCAL_ONLY")
        if args.deployment_manifest.exists():
            raise FileExistsError(args.deployment_manifest)
        manifest = provenance.implementation_fingerprint(helper_root)
        manifest["implementation_commit"] = args.implementation_commit
        _write(args.deployment_manifest, manifest)
    manifest = _read(args.deployment_manifest)
    deployment = provenance.verify_deployment(helper_root, manifest, supplied_commit=args.implementation_commit)
    reference_root = args.reference.resolve()
    # This API reads and verifies the fixed byte manifest; no Git is used here.
    reference = provenance._verify_export(reference_root)
    from .contract import preflight
    core_root = reference_root if args.kernel == "R0" else helper_root
    contract = preflight(helper_root / "rotating_slots_performance_contract_v1_0", core_root)
    source_semantics = contract["semantic_fingerprint"]
    reference_semantics = provenance.semantic_fingerprint(reference_root)
    provenance.require_same_semantics(reference_semantics, source_semantics)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(core_root))
    config_module = importlib.import_module("terminal_marl.core.rotating_slots.config")
    world_module = importlib.import_module("terminal_marl.core.rotating_slots.world")
    driver_module = importlib.import_module("terminal_marl.core.rotating_slots.driver")
    replay_module = importlib.import_module("terminal_marl.core.rotating_slots.replay")
    imported = _verify_imports(core_root, helper_root, deployment, reference, args.kernel)
    # Paired R0/OPT executions bind to identical frozen bytes, even when the
    # current Windows checkout has CRLF physical YAML. Source projection/raw
    # bytes remain separately recorded in provenance.
    config_path = reference_root / provenance.PHYSICAL_PATH
    config = config_module.load_config(config_path)
    semantics = dict(source_semantics, physical_config_bytes_sha256=config.raw_hash)
    if (world_module.POLICY_ID != provenance.R0_POLICY_ID or config.raw_hash != reference_semantics["physical_config_bytes_sha256"]
            or config.config_hash != semantics["config_hash"] or config.profile_hash != semantics["profile_hash"]):
        raise provenance.ProvenanceError("IMPORTED_CORE_CONFIG_IDENTITY_MISMATCH")
    from .observation import make_observed_advance
    observed = make_observed_advance(world_module.RotatingSlotWorld)
    helper_hashes = deployment["actual_source_files_sha256"]
    helper_execution = {name: digest for name, digest in helper_hashes.items()
                        if name.startswith("terminal_marl/core/rotating_slots/performance/")
                        or name == "scripts/server/run_rotating_slots_performance.py"}
    core_hashes = reference["actual_source_files_sha256"] if args.kernel == "R0" else {
        name: digest for name, digest in helper_hashes.items()
        if name.startswith("terminal_marl/core/rotating_slots/") and "/performance/" not in name
        or name == provenance.PHYSICAL_PATH}
    execution_sha = provenance.digest({"kernel": args.kernel, "core": core_hashes,
                                       "helpers": helper_execution, "observation_shim": observed.generated_source_sha256})
    source = {"kernel": args.kernel, "reference_commit": provenance.R0_COMMIT,
              "implementation_commit": args.implementation_commit,
              "kernel_commit": provenance.R0_COMMIT if args.kernel == "R0" else args.implementation_commit,
              "actual_core_files_sha256": core_hashes, "helper_execution_files_sha256": helper_execution,
              "helper_deployment_sha256": deployment["implementation_sha256"],
              "deployment_manifest_sha256": provenance.digest(manifest),
              "runner_execution_sha256": execution_sha, "semantic_fingerprint": semantics,
              "source_semantic_projection": source_semantics,
              "source_physical_config_bytes_sha256": source_semantics["physical_config_bytes_sha256"],
              "execution_physical_config_bytes_sha256": config.raw_hash,
              "physical_config_path": str(config_path), "imported_module_paths": imported,
              "reference_source_sha256": reference["source_sha256"],
              "observation_original_source_sha256": observed.original_source_sha256,
              "observation_generated_source_sha256": observed.generated_source_sha256}
    return dict(helper_root=helper_root, core_root=core_root, reference_root=reference_root, deployment=deployment,
                reference=reference, source=source, config=config, world_module=world_module,
                driver_module=driver_module, replay_module=replay_module, observed=observed,
                contract=contract)


def _verify_imports(core_root, helper_root, deployment, reference, kernel):
    result = {"core": {}, "helpers": {}}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if name == "terminal_marl" or name.startswith("terminal_marl."):
            if not path.is_relative_to(core_root):
                raise provenance.ProvenanceError("CORE_IMPORT_OUTSIDE_SELECTED_ROOT: " + str(path))
            relative = path.relative_to(core_root).as_posix()
            hashes = reference["actual_source_files_sha256"] if kernel == "R0" else deployment["actual_source_files_sha256"]
            if relative not in hashes or _sha(path) != hashes[relative]:
                raise provenance.ProvenanceError("CORE_IMPORT_UNVERIFIED: " + str(path))
            result["core"][name] = str(path)
        elif name == "performance" or name.startswith("performance.") or name in ("parallel", "process_metrics"):
            if not path.is_relative_to(helper_root):
                raise provenance.ProvenanceError("HELPER_IMPORT_OUTSIDE_ROOT: " + str(path))
            relative = path.relative_to(helper_root).as_posix()
            if deployment["actual_source_files_sha256"].get(relative) != _sha(path):
                raise provenance.ProvenanceError("HELPER_IMPORT_UNVERIFIED: " + str(path))
            result["helpers"][name] = str(path)
    return result


def _verify_finish(runtime, args):
    # verify_deployment deliberately runs before core import; its public default
    # rejects R0 modules outside helper_root. Finish checks the exact same bytes
    # directly and verifies each imported core/helper against its own root.
    for name, digest in runtime["deployment"]["actual_source_files_sha256"].items():
        if _sha(runtime["helper_root"] / name) != digest:
            raise provenance.ProvenanceError("DEPLOYMENT_BYTES_CHANGED_DURING_RUN: " + name)
    provenance._verify_export(runtime["reference_root"], runtime["reference"])
    return _verify_imports(runtime["core_root"], runtime["helper_root"], runtime["deployment"],
                           runtime["reference"], args.kernel)


def _checkpoint_identity(runtime):
    from .checkpoint import CheckpointIdentity
    semantics = runtime["source"]["semantic_fingerprint"]
    return CheckpointIdentity(policy_id=semantics["policy_id"], config_hash=semantics["config_hash"],
        profile_hash=semantics["profile_hash"], physical_config_bytes_sha256=semantics["physical_config_bytes_sha256"],
        effective_rules_hash=semantics["effective_rules_hash"], candidate_policy_hash=semantics["candidate_policy_hash"],
        numerical_policy_hash=semantics["numerical_policy_hash"],
        implementation_sha256=runtime["source"]["runner_execution_sha256"])


def _validate_run_args(args):
    if (not math.isfinite(args.wall_budget) or args.wall_budget <= 0 or args.core_budget < 1
            or args.process_budget < 1 or args.process_budget < args.inner_workers + 1
            or args.core_budget < max(1, args.inner_workers)):
        raise ValueError("invalid wall/process/core budget")
    if args.inner_workers and args.kernel != "R0":
        raise ValueError("P1 inner workers implement R0_PARALLEL only")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.command == "capture":
        if (not math.isfinite(args.capture_until) or not 0 < args.capture_until <= 5400
                or any(not math.isfinite(x) or x < 0 or x > 5340 for x in args.milestones)):
            raise ValueError("invalid capture request/milestone")
        if args.milestones and args.capture_until < max(args.milestones) + 60:
            raise ValueError("capture-until must be >= latest milestone + 60 simulated seconds")
    elif not math.isfinite(args.until) or not 0 < args.until <= .4:
        raise ValueError("micro invocation must be in (0,.4] simulated seconds")


def _workload(args):
    if args.command == "capture":
        return {"kind": "full_traffic_prefix", "rate_per_hour": args.rate, "seed": args.seed,
                "arrival_window_s": 3600., "external_phase_targets": [3600., 5400.]}
    return {"kind": "micro_fixture", "fixture": args.fixture, "original_target": args.until}


class AttemptLedger:
    def __init__(self, path, started_wall, started_utc, execution_sha, prior_wall=0., accounting="COMPLETE"):
        self.path = Path(path)
        self.started_wall, self.started_utc = started_wall, started_utc
        self.execution_sha, self.prior_wall, self.accounting = execution_sha, prior_wall, accounting
        self.status = "RUNNING"

    def snapshot(self):
        elapsed = time.perf_counter() - self.started_wall
        return {"schema": "ft_rs_attempt_ledger_v1_0", "pid": os.getpid(), "status": self.status,
                "started_utc": self.started_utc, "observed_utc": _utc(),
                "runner_execution_sha256": self.execution_sha, "prior_consumed_wall_s": self.prior_wall,
                "attempt_consumed_wall_s": elapsed, "cumulative_effective_wall_s": self.prior_wall + elapsed,
                "wall_accounting": self.accounting,
                "performance_qualification_eligible": self.accounting == "COMPLETE"}

    def flush(self):
        data = self.snapshot(); _write(self.path, data); return data


def _resume_metadata(args, runtime):
    from .checkpoint import load_checkpoint
    checkpoint = args.checkpoint.resolve()
    sidecar = _read(checkpoint.with_suffix(".runner.json"))
    if (sidecar.get("schema") != "ft_rs_runner_checkpoint_v1_0" or sidecar.get("checkpoint_sha256") != _sha(checkpoint)
            or sidecar.get("runner_execution_sha256") != runtime["source"]["runner_execution_sha256"]
            or sidecar.get("workload") != _workload(args)):
        raise ValueError("runner checkpoint sidecar/source/workload mismatch")
    source_attempt = Path(sidecar["source_attempt"]).resolve()
    index = _read(source_attempt / "checkpoints/checkpoint_index.json")
    if (checkpoint.parent != source_attempt / "checkpoints" or not any(
            row.get("file") == checkpoint.name and row.get("sha256") == sidecar["checkpoint_sha256"]
            for row in index.get("entries", []))):
        raise ValueError("checkpoint is not a retained complete indexed runner pair")
    ledger_path = source_attempt / "attempt_ledger.json"
    previous = _read(ledger_path)
    if (previous.get("runner_execution_sha256") != runtime["source"]["runner_execution_sha256"]
            or previous.get("cumulative_effective_wall_s", -1) < sidecar["cumulative_effective_wall_at_checkpoint_s"]):
        raise ValueError("source attempt ledger missing/mismatched consumed work")
    terminal = previous.get("status") in ("SUSPENDED_CAPTURE", "PERFORMANCE_TIMEOUT", "COMPLETED_EPISODE", "PHYSICAL_FAILURE", "IMPLEMENTATION_FAILURE")
    accounting = previous.get("wall_accounting", "INCOMPLETE")
    if not terminal:
        if args.crash_evidence is None:
            raise ValueError("source attempt not terminal; explicit crash evidence is required")
        evidence = _read(args.crash_evidence)
        if (evidence.get("source_attempt") != str(source_attempt) or evidence.get("source_pid") != previous["pid"]
                or evidence.get("confirmed_not_running") is not True or not evidence.get("observed_utc")):
            raise ValueError("crash evidence must identify source attempt/PID and confirm it is no longer running")
        accounting = "INCOMPLETE"
    payload = load_checkpoint(checkpoint, expected_identity=_checkpoint_identity(runtime))
    if args.capture_until <= sidecar["cut"]["actual_sim_time"]:
        raise ValueError("resumed capture-until must be later than the saved natural cut")
    return payload, sidecar, previous["cumulative_effective_wall_s"], accounting


@contextmanager
def _active_query_context(world, heartbeat):
    cls = type(world)
    original = cls._reference
    def reference(self, ring, slot, t, version=0):
        if self is not world:
            return original(self, ring, slot, t, version)
        heartbeat.publish(context={"phase": "emergency-reference", "ring": ring, "slot": slot,
                                   "query_time": t, "version": version, "pads": [0, 1, 2, 3, 4, 5]})
        try:
            return original(self, ring, slot, t, version)
        finally:
            heartbeat.publish(context={"phase": "advance", "original_reference_finished": True})
    cls._reference = reference
    try:
        yield
    finally:
        cls._reference = original


def execute(args, runtime, started_wall, started_utc):
    from .checkpoint import (AdvanceCursor, CheckpointStore, capture_checkpoint, restore_checkpoint)
    from .measurements import Measurements, PassiveHeartbeat
    from .observation import CutRequests, SuspendAtCut
    _validate_run_args(args)
    runtime["source"]["prefix_execution"] = "R0_PARALLEL" if args.inner_workers else args.kernel + "_SERIAL"
    runtime["source"]["prefix_equivalence_evidence"] = {
        "scope": "bounded_real_R0_inner_hold_and_small_prefix_only_not_full_P4",
        "tests": ["test_real_R0_micro_serial_pool_full_records_equal",
                  "test_isolated_R0_real_hold_nohook_hook_and_cold_warm_restore"],
        "qualification_status": "NOT_QUALIFIED"}
    root = args.output.resolve()
    identity = _checkpoint_identity(runtime)
    workload = _workload(args)
    measurements = Measurements("benchmark" if args.command == "capture" else args.mode)
    prior_wall, accounting, resume, resume_evidence = 0., "COMPLETE", None, None
    if args.command == "capture" and args.checkpoint:
        with measurements.span("checkpoint_load_validate"):
            payload, resume_evidence, prior_wall, accounting = _resume_metadata(args, runtime)
        with measurements.span("checkpoint_restore_copy"):
            resume = restore_checkpoint(payload, expected_identity=identity, cache_mode=args.cache_mode,
                                        new_attempt_dir=root, world_class=runtime["world_module"].RotatingSlotWorld)
        world, driver = resume.world, resume.driver
    else:
        root.mkdir(parents=True, exist_ok=False)
        sink = runtime["replay_module"].BufferedStateJSONLSink(root / "states.jsonl")
        world = runtime["world_module"].RotatingSlotWorld(config=runtime["config"], state_sink=sink)
        if args.command == "capture":
            demand = importlib.import_module("terminal_marl.core.rotating_slots.demand")
            # Whole immutable demand is generated once; no shorter window changes RNG.
            world.manifest = demand.ArrivalManifest.generate(args.seed, args.rate, 3600.)
            driver = runtime["driver_module"].DeterministicDriver()
        else:
            driver = False
            if args.fixture == "inner-hold":
                world.seed_hold("micro-inner-hold", "inner", 0, soc=.9)
            elif args.fixture == "service-events":
                world.seed_service("micro-service", pad_id=0, remaining=.2)
    ledger = AttemptLedger(root / "attempt_ledger.json", started_wall, started_utc,
                           runtime["source"]["runner_execution_sha256"], prior_wall, accounting)
    ledger.flush()
    _write(root / "source_identity.json", runtime["source"])
    (root / "observation.patch").write_bytes(runtime["observed"].patch.encode("utf8"))
    (root / "observation_generated.py").write_bytes(runtime["observed"].generated_source.encode("utf8"))
    (root / "observation_original.py").write_bytes(runtime["observed"].original_source.encode("utf8"))
    store = CheckpointStore(root / "checkpoints")
    checkpoints, cut_evidence = [], []
    current_cursor = resume.cursor if resume else None
    stop_reason, failure, pool, resource_report = None, None, None, None
    last_saved = None

    def save(w, cur, effective, evidence, *, pinned=False):
        nonlocal last_saved
        key = (w.time, cur.phase, cur.outer_phase)
        if key == last_saved:
            return
        with measurements.span("checkpoint_capture_save_flush"):
            payload = capture_checkpoint(w, identity=identity, cursor=cur, driver=effective)
            path = store.save(payload, pinned=pinned, corpus=True,
                              milestone=evidence.get("requested_sim_times") if pinned else None)
            snapshot = ledger.flush()
            sidecar = {"schema": "ft_rs_runner_checkpoint_v1_0", "checkpoint_sha256": _sha(path),
                       "runner_execution_sha256": runtime["source"]["runner_execution_sha256"],
                       "source_attempt": str(root), "workload": workload,
                       "cumulative_effective_wall_at_checkpoint_s": snapshot["cumulative_effective_wall_s"],
                       "cut": evidence, "cursor": {"original_target": cur.original_target, "phase": cur.phase,
                                                     "outer_phase": cur.outer_phase}}
            _write(path.with_suffix(".runner.json"), sidecar)
            # Flush the ledger after save and sidecar costs too. Resumption reads
            # this final consumed total rather than resetting at checkpoint time.
            ledger.flush()
        checkpoints.append(str(path)); cut_evidence.append(evidence); last_saved = key

    def capture_due(w, cur, effective, evidence):
        save(w, cur, effective, evidence, pinned="requested_milestone" in evidence["reasons"])

    requested = [] if args.command == "micro" else [x for x in args.milestones if x > world.time]
    cuts = CutRequests(capture_due, start_sim=world.time, requested_sim=requested)
    class LedgerHeartbeat(PassiveHeartbeat):
        def emit(self):
            state = super().emit()
            ledger.flush()
            return state
    heartbeat = LedgerHeartbeat(root / "heartbeat.json")

    def hook(w, cur, effective):
        nonlocal current_cursor, stop_reason
        current_cursor = cur
        heartbeat.publish(complete_sim_time=w.time,
                          context={"phase": cur.outer_phase, "advance_target": cur.original_target,
                                   "last_checkpoint": checkpoints[-1] if checkpoints else None,
                                   "cache_entries": {"q": len(w._q_cache), "energy": len(w._energy_cache)}})
        heartbeat.check_health()
        cuts(w, cur, effective)
        if w.failed:
            save(w, cur, effective, {"requested_sim_times": [], "actual_sim_time": w.time,
                 "simulation_deferral_s": [], "reasons": ["physical_failure"]}, pinned=True)
            return
        budget_reached = prior_wall + time.perf_counter() - started_wall >= args.wall_budget
        capture_reached = args.command == "capture" and w.time >= args.capture_until
        if budget_reached or capture_reached or cur.phase == "returned":
            evidence = {"requested_sim_times": [], "actual_sim_time": w.time,
                        "simulation_deferral_s": [], "reasons": ["soft_wall_budget" if budget_reached else
                        "capture_limit" if capture_reached else "invocation_returned"]}
            save(w, cur, effective, evidence, pinned=True)
        if budget_reached:
            stop_reason = "PERFORMANCE_TIMEOUT"
            raise SuspendAtCut()
        if capture_reached and not w.failed:
            stop_reason = "SUSPENDED_CAPTURE"
            raise SuspendAtCut()

    try:
        with ExitStack() as stack:
            if args.inner_workers:
                from .parallel import ReferencePool, install_parallel_reference
                core_directory = runtime["core_root"] / "terminal_marl/core/rotating_slots"
                core_manifest = {p.name: _sha(p) for p in core_directory.glob("*.py")}
                with measurements.span("worker_initialization"):
                    pool = ReferencePool(args.inner_workers, config=runtime["config"],
                        expected_core_root=core_directory, expected_core_manifest=core_manifest,
                        semantic_identity=runtime["source"]["semantic_fingerprint"],
                        process_budget=args.process_budget, core_budget=args.core_budget,
                        exclusive_process_accounting=True, mode="benchmark", rss_series_path=root / "rss_samples.jsonl")
                stack.enter_context(pool)
                stack.enter_context(install_parallel_reference(world, pool))
            # Linux fork pool and worker registration happen before either thread.
            stack.enter_context(heartbeat)
            stack.enter_context(_active_query_context(world, heartbeat))
            heartbeat.publish(complete_sim_time=world.time, context={"phase": "ready"})
            with measurements.span("rollout_and_record_flush"):
                try:
                    if args.command == "micro":
                        if args.mode == "profile":
                            with measurements.profile_batch(root / "micro.prof"):
                                runtime["observed"].run(world, args.until, driver, hook, outer_phase="micro")
                        else:
                            runtime["observed"].run(world, args.until, driver, hook, outer_phase="micro")
                        stop_reason = stop_reason or "COMPLETED_MICRO"
                    else:
                        phases = (("arrival_window", 3600.), ("drain_window", 5400.))
                        start = 0
                        if resume:
                            labels = [name for name, _ in phases]
                            if resume.cursor.outer_phase not in labels:
                                raise ValueError("unsupported resumed outer phase")
                            start = labels.index(resume.cursor.outer_phase)
                            if resume.cursor.original_target != phases[start][1]:
                                raise ValueError("original advance phase target mismatch")
                            runtime["observed"].resume(resume, hook)
                            start += 1
                        for phase_name, target in phases[start:]:
                            if world.failed:
                                break
                            runtime["observed"].run(world, target, driver, hook, outer_phase=phase_name)
                        stop_reason = stop_reason or ("PHYSICAL_FAILURE" if world.failed else "COMPLETED_EPISODE")
                except SuspendAtCut:
                    if stop_reason is None:
                        raise
                world.flush_records()
            if pool is not None:
                # Join before any post-run subprocess or replay audit is started.
                with measurements.span("worker_join"):
                    pool.close()
                resource_report = pool.summary()
    except Exception as exc:
        stop_reason = "IMPLEMENTATION_FAILURE"
        failure = {"type": type(exc).__name__, "message": str(exc)}
        if pool is not None:
            resource_report = pool.summary()
    replay = None
    try:
        with measurements.span("replay_bundle_and_verify"):
            replay = runtime["replay_module"].ReplayBundle.write(root / "replay", world=world,
                config_path=runtime["config"].path, code_commit=runtime["source"]["kernel_commit"],
                command=" ".join(sys.argv), seed=getattr(args, "seed", None), eager=False)
            runtime["replay_module"].ReplayBundle.validate(root / "replay")
        imported = _verify_finish(runtime, args)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        stop_reason = "IMPLEMENTATION_FAILURE"
        imported = runtime["source"]["imported_module_paths"]
    finally:
        world.state_sink.close()
    ledger.status = stop_reason
    consumed = ledger.flush()
    measured = measurements.snapshot()
    measured["execution_phase_wall_s"] = measured["end_to_end_wall_s"]
    measured["execution_phase_parent_cpu_s"] = measured["parent_process_cpu_s"]
    measured["end_to_end_wall_s"] = consumed["attempt_consumed_wall_s"]
    measured["parent_process_cpu_s"] = time.process_time() - runtime["started_cpu"]
    output = {"schema": "ft_rs_performance_attempt_v1_0", "pid": os.getpid(),
              "started_utc": started_utc, "finished_utc": _utc(), "command": list(sys.argv),
              "subcommand": args.command, "workload": workload, "source_identity": runtime["source"],
              "final_imported_module_paths": imported, "status": stop_reason, "stop_reason": stop_reason,
              "physical_outcome": {"failed": world.failed, "failure_time": world.failure_time,
                                   "episode_complete": stop_reason == "COMPLETED_EPISODE",
                                   "sim_time": world.time, "metrics": world.metrics()},
              "target_sim_time": args.capture_until if args.command == "capture" else args.until,
              "current_phase": None if current_cursor is None else current_cursor.outer_phase,
              "original_advance_target": None if current_cursor is None else current_cursor.original_target,
              "checkpoints": [str(root / "checkpoints" / x["file"]) for x in store.entries],
              "cuts": cut_evidence, "resume_source": resume_evidence,
              "timing": dict(measured, **{k: v for k, v in consumed.items() if k.endswith("_wall_s")}),
              "wall_accounting": consumed["wall_accounting"], "soft_wall_budget_s": args.wall_budget,
              "hard_timeout_enforced": False, "timeout_acceptance": "INCOMPLETE_SOFT_SAFE_BOUNDARY_ONLY",
              "inner_workers": args.inner_workers,
              "execution_backend": "R0_PARALLEL" if args.inner_workers else args.kernel + "_SERIAL",
              "resources": resource_report,
              "replay_path": str(root / "replay") if replay is not None else None,
              "replay_scope": "full_episode" if stop_reason == "COMPLETED_EPISODE" else "complete_observed_prefix",
              "recorded_state_count": world._record_count, "event_count": len(world.events),
              "driver_calls": None if driver is False or driver is None else driver.calls,
              "last_reference_fallback": world._last_reference_fallback,
              "failure": failure, "performance_status": "NOT_QUALIFIED", "human_status": "HUMAN_PENDING"}
    _write(root / "result.json", output)
    return output


def main(argv=None, *, helper_root=None):
    args = parser().parse_args(argv)
    started_wall, started_utc, started_cpu = time.perf_counter(), _utc(), time.process_time()
    root = Path(helper_root or Path(__file__).resolve().parents[4])
    try:
        runtime = bootstrap(args, root)
        runtime["started_cpu"] = started_cpu
        if args.command == "preflight":
            print(json.dumps({"status": "PREFLIGHT_OK", "source_identity": runtime["source"],
                              "performance_status": "NOT_QUALIFIED", "human_status": "HUMAN_PENDING"}, allow_nan=False))
            return 0
        result = execute(args, runtime, started_wall, started_utc)
        print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                          "sim_time": result["physical_outcome"]["sim_time"], "failure": result["failure"]}, allow_nan=False))
        return 1 if result["status"] == "IMPLEMENTATION_FAILURE" else 0
    except Exception as exc:
        print(json.dumps({"status": "PREFLIGHT_REJECTED", "error_type": type(exc).__name__, "error": str(exc)}, allow_nan=False))
        return 2
