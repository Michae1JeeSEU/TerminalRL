"""Real serial common-state windows; no paired experiment or qualification.

Use the stdlib CLI bootstrap in a fresh ``python -I -B`` process. Frozen source
streams remain absolute, read-only dependencies; a replay is never a checkpoint.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

from . import provenance, runner
from .checkpoint import (CheckpointIdentity, CheckpointStore, StateCodec, atomic_write_json,
                         capture_checkpoint, load_checkpoint, register_equivalence_migration,
                         restore_checkpoint, _prefix_hash)
from .corpus import CASES, WINDOWS, read_source, select_windows
from .measurements import Measurements, PassiveHeartbeat
from .observation import CutRequests, SuspendAtCut

PHASES = {"arrival_window": 3600., "drain_window": 5400.}
IDENTITY_KEYS = ("policy_id", "config_hash", "profile_hash", "physical_config_bytes_sha256",
                 "effective_rules_hash", "candidate_policy_hash", "numerical_policy_hash")


class WindowError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise WindowError(message)


def _read(path):
    return json.loads(Path(path).read_bytes())


def _sha(path):
    # Checkpoints and streams can exceed memory; hash them incrementally.
    from hashlib import sha256
    result = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def parser():
    result = argparse.ArgumentParser(description="Real serial common window; NOT_QUALIFIED")
    commands = result.add_subparsers(dest="command", required=True)
    command = commands.add_parser("window")
    command.add_argument("--corpus-manifest", type=Path, required=True)
    command.add_argument("--corpus-sha256", required=True, help="expected canonical corpus digest")
    command.add_argument("--window", choices=tuple(row[0] for row in WINDOWS), required=True)
    command.add_argument("--duration", type=int, choices=(10, 60), required=True)
    command.add_argument("--kernel", choices=("R0", "OPT"), required=True)
    command.add_argument("--mode", choices=("benchmark", "profile"), default="benchmark")
    command.add_argument("--reference", type=Path, required=True)
    command.add_argument("--deployment-manifest", type=Path, required=True)
    command.add_argument("--implementation-commit", required=True)
    command.add_argument("--migration-approval", type=Path,
                         help="required only when the exact checkpoint implementation changes")
    command.add_argument("--output", type=Path, required=True, help="new attempt directory only")
    return result


def validate_manifest(document, expected_digest, window, duration):
    """Pure structural/digest checks, deliberately separate from all file reads."""
    _require(_hex(expected_digest), "expected corpus SHA256 must be exact 64 lowercase hex")
    body = {k: v for k, v in document.items() if k != "manifest_sha256"}
    _require(provenance.digest(body) == document.get("manifest_sha256") == expected_digest,
             "corpus canonical digest mismatch")
    _require(document.get("schema") == "ft_rs_performance_corpus_v1_0" and
             document.get("status") == "FROZEN", "corpus must be FROZEN")
    _require(document.get("reference_commit") == provenance.R0_COMMIT and
             document.get("reference_source_sha256") == provenance.R0_SOURCE_SHA256,
             "corpus is not the fixed R0")
    _require(duration in (10, 60) and type(duration) is int, "window duration must be 10 or 60")
    rows = document.get("windows", [])
    _require(len(rows) == 6 and {r.get("window") for r in rows} == {r[0] for r in WINDOWS} and
             all(r.get("status") == "AVAILABLE" for r in rows), "six AVAILABLE windows required")
    sources = document.get("sources", {})
    _require(set(sources) == set(CASES), "fixed four corpus sources required")
    selected = select_windows(sources)
    by_name = {r["window"]: r for r in rows}
    for expected in selected:
        actual = by_name[expected["window"]]
        _require(all(actual.get(k) == v for k, v in expected.items()), "frozen selection changed")
        cut, validation = actual.get("cut"), actual.get("validation", {})
        _require(isinstance(cut, dict), "missing selected natural cut")
        source = sources[actual["case"]]
        source_identity = source["source_identity"]
        semantics = source_identity.get("semantic_fingerprint")
        _require(semantics == document.get("semantic_fingerprint") and
                 source_identity.get("kernel") == "R0" and
                 source_identity.get("reference_commit") == source_identity.get("kernel_commit") == provenance.R0_COMMIT and
                 source_identity.get("reference_source_sha256") == provenance.R0_SOURCE_SHA256,
                 "source semantic/fixed R0 mismatch")
        identity = CheckpointIdentity(**{k: semantics[k] for k in IDENTITY_KEYS},
                                     implementation_sha256=source_identity["runner_execution_sha256"])
        cursor = validation.get("cursor", {})
        _require(validation.get("identity") == asdict(identity) and
                 validation.get("cache_mode") == "cold" and validation.get("restored_without_advance") is True and
                 validation.get("world_time") == cut["actual_sim_time"] and
                 cursor.get("shim_source_sha256") == source_identity.get("observation_generated_source_sha256") and
                 {k: cursor.get(k) for k in ("original_target", "phase", "outer_phase")} == cut.get("cursor") and
                 cursor.get("phase") == "loop_start" and
                 cursor.get("original_target") == PHASES.get(cursor.get("outer_phase")),
                 "cold restore evidence/cursor/original phase target mismatch")
        _require(_number(actual["stop_sim_time"]) and _number(cut["actual_sim_time"]) and
                 actual["stop_sim_time"] - cut["actual_sim_time"] >= 60.,
                 "selected cut lacks observed 60 second suffix")
        _require(cut["actual_sim_time"] + 60. <= cursor["original_target"],
                 "selected window exceeds original invocation")
        _require(_hex(cut.get("checkpoint_sha256")) and _hex(cut.get("sidecar_sha256")),
                 "selected checkpoint/sidecar hashes missing")
    _require(window in by_name, "unknown window")
    return by_name[window]


def validate_approval(document, *, source, target, corpus_digest, source_metadata, target_metadata,
                      checkpoint_helper_sha256):
    """Consume an existing approval and exact evidence files; never issue approval."""
    source, target = asdict(source), asdict(target)
    _require(source != target and all(source[k] == target[k] for k in source if k != "implementation_sha256"),
             "migration cannot change semantics or identical implementation")
    _require(document.get("schema") == "window_migration_approval_v1" and document.get("qualified") is True,
             "qualified window migration approval required")
    _require(all(source_metadata[k] == target_metadata[k] for k in (
        "observation_original_source_sha256", "observation_generated_source_sha256")),
        "observation shim changes are outside this implementation migration")
    expected = {"corpus_sha256": corpus_digest,
                "target_execution_sha256": target["implementation_sha256"],
                "target_observation_original_sha256": target_metadata["observation_original_source_sha256"],
                "target_observation_generated_sha256": target_metadata["observation_generated_source_sha256"],
                "checkpoint_helper_sha256": checkpoint_helper_sha256}
    _require(all(document.get(k) == v for k, v in expected.items()), "stale approval corpus/target/shim/helper")
    pairs = document.get("migrations", [])
    matching = [p for p in pairs if p.get("source_identity") == source and p.get("target_identity") == target]
    _require(len(matching) == 1, "exact source/target migration pair missing or ambiguous")
    pair = matching[0]
    _require(isinstance(pair.get("migration_id"), str) and bool(pair["migration_id"]) and
             pair.get("source_observation_original_sha256") == source_metadata["observation_original_source_sha256"] and
             pair.get("source_observation_generated_sha256") == source_metadata["observation_generated_source_sha256"],
             "stale source shim or missing migration ID")
    evidence = document.get("supporting_evidence")
    _require(isinstance(evidence, list) and bool(evidence), "nonempty supporting evidence required")
    for row in evidence:
        _require(isinstance(row, dict) and set(row) == {"path", "sha256"} and
                 isinstance(row["path"], str) and Path(row["path"]).is_absolute() and
                 _hex(row["sha256"]), "invalid evidence path/SHA256")
        _require(Path(row["path"]).is_file() and _sha(row["path"]) == row["sha256"],
                 "supporting evidence missing or changed")
    return pair["migration_id"]


def validate_inputs(args, runtime, measurements):
    """Validate immutable source terminal evidence, both copies and real payload."""
    manifest_path = args.corpus_manifest.resolve()
    document = _read(manifest_path)
    row = validate_manifest(document, args.corpus_sha256, args.window, args.duration)
    _require(document["semantic_fingerprint"] == runtime["source"]["semantic_fingerprint"],
             "runtime/corpus semantics differ")
    root = args.output.resolve()
    _require(not root.exists(), "output already exists; overwrite is forbidden")
    source = document["sources"][row["case"]]
    source_root = Path(source["source_attempt"]).resolve()
    for protected in (manifest_path.parent, source_root, runtime["reference_root"]):
        _require(not root.is_relative_to(protected) and not protected.is_relative_to(root),
                 "output must be separate from corpus/source/reference")
    # This repeats the freeze's terminal/index/workload checks using actual
    # original source bytes. Absolute paths are intentionally never remapped.
    actual = read_source(source_root.parent, row["case"], runtime["reference"], document["semantic_fingerprint"])
    _require(actual == source, "original source terminal metadata/index changed since freeze")
    cut = row["cut"]
    checkpoint, sidecar_path = Path(row["frozen_checkpoint"]), Path(row["frozen_sidecar"])
    _require(checkpoint.resolve().parent == manifest_path.parent / "inputs" / args.window and
             sidecar_path.resolve() == checkpoint.resolve().with_suffix(".runner.json"),
             "frozen pair must be retained inside this corpus window")
    _require(_sha(checkpoint) == _sha(cut["checkpoint"]) == cut["checkpoint_sha256"] and
             _sha(sidecar_path) == _sha(cut["sidecar"]) == cut["sidecar_sha256"],
             "frozen/original checkpoint/sidecar bytes differ")
    sidecar = _read(sidecar_path)
    _require(sidecar.get("source_attempt") == source["source_attempt"] and sidecar.get("workload") == source["workload"] and
             sidecar.get("cursor") == cut["cursor"] and
             sidecar.get("runner_execution_sha256") == source["source_identity"]["runner_execution_sha256"],
             "frozen sidecar source/workload/cursor mismatch")
    identity = CheckpointIdentity(**row["validation"]["identity"])
    with measurements.span("checkpoint_load_validate"):
        payload = load_checkpoint(checkpoint, expected_identity=identity)
    codec = StateCodec()
    cursor, state = codec.decode(payload["cursor"]), codec.decode(payload["world"])
    _require(cursor == row["validation"]["cursor"] and state["time"] == cut["actual_sim_time"] and
             not state["failed"], "checkpoint natural state/cursor differs from cold validation")
    workload = source["workload"]
    demand = importlib.import_module("terminal_marl.core.rotating_slots.demand")
    expected_demand = demand.ArrivalManifest.generate(workload["seed"], workload["rate_per_hour"], 3600.).to_dict()
    _require(state["manifest"].to_dict() == expected_demand and state["physics_dt"] == .2 and
             state["scheduler_period"] == 5., "checkpoint workload/physics changed")
    _require(payload["state_stream"] == row["validation"]["state_stream"] and
             payload["state_stream"]["path"] == str(source_root / "states.jsonl"),
             "original immutable record stream dependency differs")
    archive = runtime["replay_module"].ReplayBundle.validate(source_root / "replay")
    replay_evidence = document["replay_validation"][row["case"]]
    _require(archive.state_count == source["recorded_state_count"] == replay_evidence["state_count"] and
             list(archive.time_bounds) == replay_evidence["time_bounds"] == [0., source["stop_time"]] and
             archive.index["code_commit"] == provenance.R0_COMMIT and
             _read(source_root / "replay/manifest.json") == expected_demand,
             "original replay workload/prefix changed")
    return document, row, source, identity, payload


def _counts(world, driver):
    """Existing reliable counters only; unavailable work counters stay MISSING."""
    sink = world.state_sink
    statuses = [a.status for a in world.aircraft.values()]
    airborne = importlib.import_module(type(world).__module__).AIRBORNE
    return {"sim_time": world.time, "state_count": world._record_count,
            "airborne_count": sum(s in airborne for s in statuses),
            "holding_count": sum(s.startswith("HOLD_") for s in statuses),
            "in_transit_count": sum(s in airborne and not s.startswith("HOLD_") for s in statuses),
            "service_count": statuses.count("LANDED_SERVICE"),
            "upstream_count": sum(k not in world.aircraft for k in world.requests),
            "decision_count": len(world.decision_times),
            "event_count": len(world.events), "rejection_count": len(world.rejections),
            "trajectory_count": len(world.trajectories),
            "driver_calls": None if driver is None or driver is False else list(driver.calls),
            "record_file_bytes": None if sink is None else sink.path.stat().st_size,
            "record_logical_bytes": None if sink is None else sink._offset + len(sink._pending or b""),
            "record_committed_count": None if sink is None else sink._committed_count,
            "reference_counts": dict(world.reference_counts), "cache_hits": world.cache_hits,
            "qe_logical_queries": "MISSING", "candidate_logical_count": "MISSING",
            "validation_logical_count": "MISSING", "validation_actual_count": "MISSING",
            "cache_evictions": "MISSING", "kernel_work_counts": "MISSING"}


def _cold_cache_snapshot(world):
    """Observe exact named caches without changing configuration or policy."""
    core = type(world).__module__.rsplit(".", 1)[0]
    numerical, empty = {}, True
    for module_name, names in (("trajectory", ("poly_bounds",)),
                               ("energy", ("_integrable_plan", "_segment_table")),
                               ("safety", ("_NUMERIC_CERTIFICATES", "_PLAN_CERTIFICATES"))):
        module = importlib.import_module(core + "." + module_name)
        for name in names:
            cache = getattr(module, name)
            key = module_name + "." + name
            if hasattr(cache, "cache_info"):
                numerical[key] = cache.cache_info()._asdict()
                empty = empty and numerical[key]["currsize"] == 0
            else:
                _require(type(cache) is dict, "unexpected module acceleration cache")
                numerical[key] = {"entries": len(cache)}
                empty = empty and not cache
    # R0 predates this optional OPT cache. Absence is valid and empty; named
    # caches above remain required and are never cleared to manufacture coldness.
    energy_module = importlib.import_module(core + ".energy")
    root_stats = getattr(energy_module, "root_cache_stats", None)
    if root_stats is None:
        numerical["energy._ROOT_CACHE"] = {"present": False, "entries": 0}
    else:
        _require(callable(root_stats), "unexpected optional root cache observer")
        observed = root_stats()
        _require(isinstance(observed, dict) and type(observed.get("entries")) is int
                 and observed["entries"] >= 0, "invalid optional root cache observation")
        numerical["energy._ROOT_CACHE"] = dict(observed, present=True)
        empty = empty and observed["entries"] == 0
    world_caches = {name: len(getattr(world, name)) for name in ("_q_cache", "_energy_cache", "_reference_templates")}
    config_module = importlib.import_module(core + ".config")
    parse_cache = config_module._default_config.cache_info()._asdict()
    return {"world_entries": world_caches, "module_numerical_caches": numerical,
            "configuration_parse_cache": parse_cache,
            "all_named_acceleration_caches_empty": bool(empty and all(value == 0 for value in world_caches.values())),
            "retained_tables": ["immutable config", "energy._GAUSS", "canonical query policy"],
            "observation_timing_scope": "checkpoint_cold_restore_copy", "module_caches_cleared": False}


def execute_window(args, runtime, *, started_wall=None, started_cpu=None):
    """Restore then resume the original invocation to its first eligible hook."""
    started_wall = time.perf_counter() if started_wall is None else started_wall
    started_cpu = time.process_time() if started_cpu is None else started_cpu
    started_utc = runtime.get("started_utc", runner._utc())
    measurements = Measurements(args.mode)
    root = args.output.resolve()
    _require(not root.exists(), "output already exists; overwrite is forbidden")
    with measurements.span("input_archive_and_checkpoint_load_validate"):
        document, row, source, original_identity, payload = validate_inputs(args, runtime, measurements)
    identity = runner._checkpoint_identity(runtime)
    migration_id, approval = None, None
    if original_identity != identity:
        _require(args.migration_approval is not None, "implementation changed: explicit qualified migration approval required")
        approval = _read(args.migration_approval)
        migration_id = validate_approval(approval, source=original_identity, target=identity,
            corpus_digest=args.corpus_sha256, source_metadata=source["source_identity"],
            target_metadata=runtime["source"], checkpoint_helper_sha256=_sha(Path(__file__).with_name("checkpoint.py")))
        register_equivalence_migration(migration_id, original_identity, identity)
    else:
        _require(args.migration_approval is None, "migration not required for identical implementation")
    _require(row["validation"]["cursor"]["shim_source_sha256"] == runtime["observed"].generated_source_sha256,
             "observation shim changed; cursor rebinding is forbidden")
    with measurements.span("checkpoint_cold_restore_copy"):
        restored = restore_checkpoint(payload, expected_identity=identity, cache_mode="cold", new_attempt_dir=root,
            world_class=runtime["world_module"].RotatingSlotWorld, migration_id=migration_id)
        cold_cache_evidence = _cold_cache_snapshot(restored.world)
        atomic_write_json(root / "cold_cache_report.json", cold_cache_evidence)
        if not cold_cache_evidence["all_named_acceleration_caches_empty"]:
            restored.world.state_sink.close()
            raise WindowError("cold restore module/world numerical caches are not empty; see cold_cache_report.json")
    rollout_started_wall, rollout_started_cpu = time.perf_counter(), time.process_time()
    world, driver = restored.world, restored.driver
    _require(root.is_dir() and world.time == row["cut"]["actual_sim_time"], "restore advanced time or lacks copied record stream")
    original_cursor = restored.cursor
    _require(original_cursor.shim_source_sha256 == runtime["observed"].generated_source_sha256,
             "observation shim changed; cursor rebinding is forbidden")
    _require(restored.cursor.phase == "loop_start", "window must resume an unfinished original invocation")
    start_sim, nominal_stop = world.time, world.time + args.duration
    runtime["source"]["prefix_execution"] = args.kernel + "_SERIAL"
    atomic_write_json(root / "source_identity.json", runtime["source"])
    for filename, text in (("observation.patch", runtime["observed"].patch),
                           ("observation_original.py", runtime["observed"].original_source),
                           ("observation_generated.py", runtime["observed"].generated_source)):
        (root / filename).write_bytes(text.encode("utf8"))
    atomic_write_json(root / "input_identity.json", {"corpus_path": str(args.corpus_manifest.resolve()),
        "corpus_sha256": args.corpus_sha256, "window": args.window, "selected_cut": row["cut"],
        "source_identity": asdict(original_identity), "target_identity": asdict(identity),
        "source_cursor": asdict(original_cursor), "runtime_cursor": asdict(restored.cursor),
        "migration_id": migration_id, "migration_approval_path": None if args.migration_approval is None else str(args.migration_approval.resolve()),
        "migration_approval_sha256": None if approval is None else _sha(args.migration_approval)})
    store = CheckpointStore(root / "checkpoints")
    cuts, current_cursor = [], restored.cursor
    last_complete_sim_time = start_sim
    status, failure, terminal_checkpoint = "RUNNING", None, None
    heartbeat = PassiveHeartbeat(root / "heartbeat.json")
    pending_cut = None

    def save(w, cursor, effective, evidence, *, pinned=False):
        nonlocal terminal_checkpoint
        with measurements.span("normal_checkpoint_capture_save_flush"):
            saved = capture_checkpoint(w, identity=identity, cursor=cursor, driver=effective)
            path = store.save(saved, pinned=pinned, corpus="simulation_interval" in evidence["reasons"])
            atomic_write_json(path.with_suffix(".runner.json"), {
                "schema": "ft_rs_window_checkpoint_v1_0", "checkpoint_sha256": _sha(path),
                "runner_execution_sha256": identity.implementation_sha256, "source_attempt": str(root),
                "corpus_sha256": args.corpus_sha256, "window": args.window, "workload": source["workload"],
                "cursor": asdict(cursor), "cut": evidence})
        cuts.append(evidence)
        terminal_checkpoint = str(path)

    def capture_due(w, cursor, effective, evidence):
        nonlocal pending_cut
        pending_cut = evidence

    requests = CutRequests(capture_due, start_sim=start_sim)

    def hook(w, cursor, effective):
        nonlocal current_cursor, pending_cut, status, last_complete_sim_time
        current_cursor, pending_cut = cursor, None
        last_complete_sim_time = w.time
        heartbeat.publish(complete_sim_time=w.time, context={"phase": cursor.outer_phase,
            "original_advance_target": cursor.original_target, "nominal_window_stop": nominal_stop,
            "last_checkpoint": terminal_checkpoint, "cache_entries": {
                "q": len(w._q_cache), "energy": len(w._energy_cache)}})
        heartbeat.check_health()
        requests(w, cursor, effective)
        terminal = "physical_failure" if w.failed else "window_limit" if w.time >= nominal_stop else (
            "original_invocation_returned" if cursor.phase == "returned" else None)
        if pending_cut is not None or terminal:
            evidence = pending_cut or {"requested_sim_times": [], "actual_sim_time": w.time,
                "simulation_deferral_s": [], "reasons": []}
            if terminal:
                evidence["reasons"].append(terminal)
                if terminal == "window_limit":
                    evidence["window_nominal_stop"] = nominal_stop
                    evidence["window_deferral_s"] = w.time - nominal_stop
            save(w, cursor, effective, evidence, pinned=terminal is not None)
        if w.failed:
            status = "PHYSICAL_FAILURE"
            raise SuspendAtCut()
        if w.time >= nominal_stop:
            status = "COMPLETED_WINDOW"
            raise SuspendAtCut()
        if cursor.phase == "returned":
            status = "INSUFFICIENT_CONTINUATION"
            raise SuspendAtCut()

    start_counts = _counts(world, driver)
    resource_report = None
    rollout_wall = rollout_cpu = None
    # Rollout starts immediately after cold restore, including all subsequent
    # observer/setup work, periodic saves and final record flushing.
    try:
        with ExitStack() as stack:
            from .process_metrics import OwnedProcessMetrics
            metrics = OwnedProcessMetrics(exclusive=True, rss_series_path=root / "rss_samples.jsonl")
            stack.callback(lambda: metrics.finish(joined=True))
            stack.enter_context(heartbeat)
            stack.enter_context(runner._active_query_context(world, heartbeat))
            heartbeat.publish(complete_sim_time=world.time, context={"phase": "ready", "window": args.window})
            try:
                with measurements.span("rollout_and_record_flush"):
                    try:
                        if args.mode == "profile":
                            with measurements.profile_batch(root / "window.prof"):
                                runtime["observed"].resume(restored, hook)
                        else:
                            runtime["observed"].resume(restored, hook)
                    except SuspendAtCut:
                        _require(status != "RUNNING", "unexpected unclassified suspension")
                    finally:
                        world.flush_records()
            finally:
                rollout_wall = time.perf_counter() - rollout_started_wall
                rollout_cpu = time.process_time() - rollout_started_cpu
            resource_report = metrics.finish(joined=True)
    except Exception as exc:
        status, failure = "IMPLEMENTATION_FAILURE", {"type": type(exc).__name__, "message": str(exc)}
        # A failure between natural cuts cannot be serialized as a new complete
        # state. Keep the last complete normal checkpoint and cursor evidence.
        if "metrics" in locals():
            resource_report = metrics.finish(joined=True)
    if status == "RUNNING":
        status = "INSUFFICIENT_CONTINUATION"
    end_counts = _counts(world, driver)
    replay_path, imported = None, runtime["source"]["imported_module_paths"]
    try:
        with measurements.span("output_replay_bundle_and_validate"):
            runtime["replay_module"].ReplayBundle.write(root / "replay", world=world,
                config_path=runtime["config"].path, code_commit=runtime["source"]["kernel_commit"],
                command=" ".join(sys.argv), seed=source["workload"]["seed"], eager=False)
            runtime["replay_module"].ReplayBundle.validate(root / "replay")
            replay_path = str(root / "replay")
        with measurements.span("source_and_deployment_finish_verify"):
            imported = runner._verify_finish(runtime, args)
            _require(_read(args.corpus_manifest) == document, "corpus changed during run")
            for name, digest in source["source_files_sha256"].items():
                _require(_sha(Path(source["source_attempt"]) / name) == digest, "source metadata changed during run")
            _require(_sha(row["frozen_checkpoint"]) == _sha(row["cut"]["checkpoint"]) == row["cut"]["checkpoint_sha256"] and
                     _sha(row["frozen_sidecar"]) == _sha(row["cut"]["sidecar"]) == row["cut"]["sidecar_sha256"],
                     "source/frozen pair changed during run")
            stream = payload["state_stream"]
            _require(_prefix_hash(stream["path"], stream["offset"]) == stream["prefix_sha256"],
                     "original source record prefix changed during run")
    except Exception as exc:
        status, failure = "IMPLEMENTATION_FAILURE", {"type": type(exc).__name__, "message": str(exc)}
    finally:
        world.state_sink.close()
    final_heartbeat = _read(heartbeat.path) if heartbeat.path.exists() else {}
    final_heartbeat.update(status=status, last_complete_sim_time=last_complete_sim_time,
                           finished_utc=runner._utc(), last_checkpoint=terminal_checkpoint)
    atomic_write_json(heartbeat.path, final_heartbeat)
    timing = measurements.snapshot()
    timing.update(rollout_wall_s=rollout_wall, rollout_parent_process_cpu_s=rollout_cpu,
                  total_wall_s=time.perf_counter() - started_wall,
                  total_parent_process_cpu_s=time.process_time() - started_cpu,
                  worker_cpu_s=0., serial_no_workers=True,
                  timing_scope="rollout begins after cold restore; includes natural checkpoints, observation and record flush")
    output = {"schema": "ft_rs_common_window_v1_0", "pid": os.getpid(), "status": status,
        "command": list(sys.argv), "started_utc": started_utc, "finished_utc": runner._utc(),
        "source_identity": runtime["source"], "final_imported_module_paths": imported,
        "corpus_sha256": args.corpus_sha256, "window": args.window, "duration_s": args.duration,
        "workload": source["workload"], "start_sim_time": start_sim, "nominal_stop_sim_time": nominal_stop,
        "actual_stop_sim_time": world.time, "actual_sim_delta_s": world.time - start_sim,
        "simulation_deferral_s": world.time - nominal_stop if world.time >= nominal_stop else None,
        "original_advance_target": original_cursor.original_target, "outer_phase": original_cursor.outer_phase,
        "last_observed_cursor": asdict(current_cursor), "terminal_checkpoint": terminal_checkpoint,
        "last_complete_sim_time": last_complete_sim_time,
        "last_complete_state_checkpoint": terminal_checkpoint or row["frozen_checkpoint"],
        "last_complete_state_scope": "retained_complete_checkpoint; live state after an implementation failure is diagnostic only",
        "checkpoints": [str(store.directory / x["file"]) for x in store.entries], "cuts": cuts,
        "physical_outcome": {"failed": world.failed, "failure_time": world.failure_time,
                             "episode_complete": False, "metrics": world.metrics()},
        "start_counts": start_counts, "end_counts": end_counts, "timing": timing,
        "resources": resource_report, "resource_qualification": "INCOMPLETE",
        "resource_scope": "parent-only serial; sample peak is a lower bound, lifetime HWM is an upper bound",
        "inner_workers": 0, "execution_backend": args.kernel + "_SERIAL", "mode": args.mode,
        "profile_timing_performance_eligible": False if args.mode == "profile" else None,
        "cache_mode": "cold", "cold_cache_evidence": cold_cache_evidence,
        "cache_warmup_excluded": False, "migration_id": migration_id,
        "replay_path": replay_path, "replay_scope": "diagnostic_prefix_after_implementation_failure" if failure else "complete_observed_prefix_not_full_episode",
        "failure": failure, "hard_timeout_enforced": False, "external_diagnostic_budget_enforced": False,
        "performance_status": "NOT_QUALIFIED", "human_status": "HUMAN_PENDING"}
    atomic_write_json(root / "result.json", output)
    return output


def main(argv=None, *, helper_root=None):
    args = parser().parse_args(argv)
    started_wall, started_cpu, started_utc = time.perf_counter(), time.process_time(), runner._utc()
    try:
        _require(not args.output.exists(), "output already exists; overwrite is forbidden")
        runtime = runner.bootstrap(args, Path(helper_root or Path(__file__).resolve().parents[4]))
        runtime["started_utc"] = started_utc
        bind_window_entrypoint(runtime, args.kernel)
        result = execute_window(args, runtime, started_wall=started_wall, started_cpu=started_cpu)
        print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                          "actual_sim_delta_s": result["actual_sim_delta_s"], "failure": result["failure"]}, allow_nan=False))
        return 0 if result["status"] in ("COMPLETED_WINDOW", "PHYSICAL_FAILURE") else 1
    except Exception as exc:
        print(json.dumps({"status": "PREFLIGHT_REJECTED", "error_type": type(exc).__name__,
                          "error": str(exc)}, allow_nan=False))
        return 2


def bind_window_entrypoint(runtime, kernel):
    """Include the deployment-verified window launcher in checkpoint identity."""
    # runner.bootstrap predates this entrypoint. Retain its exact verified
    # core/helper/shim formula while extending the actual executed helper set.
    entrypoint = "scripts/server/run_rotating_slots_performance_windows.py"
    helpers = runtime["source"]["helper_execution_files_sha256"]
    helpers[entrypoint] = runtime["deployment"]["actual_source_files_sha256"][entrypoint]
    runtime["source"]["runner_execution_sha256"] = provenance.digest({
        "kernel": kernel, "core": runtime["source"]["actual_core_files_sha256"],
        "helpers": helpers, "observation_shim": runtime["observed"].generated_source_sha256})
    return runtime["source"]["runner_execution_sha256"]
