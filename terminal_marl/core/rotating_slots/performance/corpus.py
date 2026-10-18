"""Freeze the six §15.2 inputs, without running or qualifying a benchmark."""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

from . import provenance
from .checkpoint import atomic_write_json

WINDOWS = (("W71-A", "rate180_seed71", 300.), ("W71-B", "rate180_seed71", 900.),
           ("W71-C", "rate180_seed71", 1200.), ("W60", "rate60_seed23", 600.),
           ("W120", "rate120_seed11", 600.), ("W180", "rate180_seed11", 540.))
CASES = {"rate180_seed71": (180, 71), "rate60_seed23": (60, 23),
         "rate120_seed11": (120, 11), "rate180_seed11": (180, 11)}
TERMINAL = {"SUSPENDED_CAPTURE", "PHYSICAL_FAILURE", "COMPLETED_EPISODE"}


class CorpusError(ValueError):
    pass


def _read(path):
    return json.loads(Path(path).read_bytes())


def _sha(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _require(condition, message):
    if not condition:
        raise CorpusError(message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def select_windows(sources):
    """Pure deterministic selection over validated source/cut metadata.

    A source contains status, stop_time, workload.seed and cuts. Cuts contain actual_sim_time,
    requested_sim_times, pinned and corpus. No wall measurement is inspected.
    """
    used, output = {}, []
    for window, case, requested in WINDOWS:
        source = sources[case]
        _require(source["status"] in TERMINAL, case + ": source is not an eligible terminal attempt")
        stop = source["stop_time"]
        _require(_number(stop), case + ": invalid stop time")
        cuts = source["cuts"]
        for cut in cuts:
            _require(_number(cut["actual_sim_time"]) and cut["actual_sim_time"] <= stop,
                     case + ": invalid cut time")
        previous = used.setdefault(source["workload"]["seed"], [])
        def eligible(cut):
            return (stop - cut["actual_sim_time"] >= 60. and
                    all(abs(cut["actual_sim_time"] - t) >= 30. for t in previous))
        preferred = [c for c in cuts if c["pinned"] and requested in c["requested_sim_times"]]
        _require(len(preferred) <= 1, case + ": ambiguous preferred milestone")
        selected = preferred[0] if preferred and eligible(preferred[0]) else None
        reason = ("preferred_requested_milestone" if selected else
                  "preferred_same_seed_spacing_conflict" if preferred and stop - preferred[0]["actual_sim_time"] >= 60.
                  else
                  "missing_preferred_pin_after_observed_60s_suffix" if not preferred and stop - requested >= 60.
                  else "preferred_unavailable_or_short")
        physical_short = (stop - (preferred[0]["actual_sim_time"] if preferred else requested) < 60.)
        if selected is None and source["status"] == "PHYSICAL_FAILURE" and physical_short:
            # Every validated saved natural cut is eligible, including wall-only cuts.
            available = [c for c in cuts if eligible(c)]
            available.sort(key=lambda c: c["actual_sim_time"], reverse=True)
            if available:
                _require(len(available) == 1 or available[0]["actual_sim_time"] != available[1]["actual_sim_time"],
                         case + ": ambiguous latest natural cut")
                selected = available[0]
                reason = "availability_only_physical_stop_before_preferred_60s_suffix"
        row = {"window": window, "case": case, "original_requested_sim_time": requested,
               "status": "AVAILABLE" if selected else "UNAVAILABLE", "selection_reason": reason,
               "stop_reason": source["status"], "stop_sim_time": stop, "cut": selected}
        if selected:
            previous.append(selected["actual_sim_time"])
        output.append(row)
    return output


def read_source(campaign, case, reference_manifest, semantics):
    """Read a terminal source, checking its retained index and exact identities."""
    root = (Path(campaign) / case).resolve()
    exit_path = root.parent / (case + ".exitcode.txt")
    _require(exit_path.is_file() and exit_path.read_text().strip() == "0", case + ": missing/nonzero numeric exit code")
    result = _read(root / "result.json")
    ledger = _read(root / "attempt_ledger.json")
    source = _read(root / "source_identity.json")
    _require(result.get("schema") == "ft_rs_performance_attempt_v1_0" and
             ledger.get("schema") == "ft_rs_attempt_ledger_v1_0", case + ": invalid result/ledger schema")
    status = result.get("status")
    _require(status in TERMINAL and status == result.get("stop_reason") == ledger.get("status"),
             case + ": running, timeout, failure or inconsistent terminal status")
    _require(result.get("finished_utc") and result.get("failure") is None and
             result.get("pid") == ledger.get("pid") and result.get("started_utc") == ledger.get("started_utc"),
             case + ": incomplete/inconsistent attempt identity")
    _require(source == result.get("source_identity"), case + ": result/source identity mismatch")
    _require(result.get("wall_accounting") == ledger.get("wall_accounting") == "COMPLETE",
             case + ": incomplete source accounting")
    _require(source.get("kernel") == "R0" and source.get("kernel_commit") == source.get("reference_commit") == provenance.R0_COMMIT
             and source.get("reference_source_sha256") == provenance.R0_SOURCE_SHA256,
             case + ": source is not the fixed R0")
    _require(source.get("actual_core_files_sha256") == reference_manifest["actual_source_files_sha256"],
             case + ": reference core bytes mismatch")
    _require(source.get("semantic_fingerprint") == semantics and source.get("source_semantic_projection") == semantics and
             source.get("execution_physical_config_bytes_sha256") == semantics["physical_config_bytes_sha256"] and
             source.get("source_physical_config_bytes_sha256") == semantics["physical_config_bytes_sha256"],
             case + ": semantic/physical configuration mismatch")
    backend = result.get("execution_backend")
    _require(backend in ("R0_SERIAL", "R0_PARALLEL") and source.get("prefix_execution") == backend and
             ((backend == "R0_SERIAL" and result.get("inner_workers") == 0) or
              (backend == "R0_PARALLEL" and type(result.get("inner_workers")) is int and result["inner_workers"] > 0)),
             case + ": execution label mismatch")
    helpers = source.get("helper_execution_files_sha256")
    _require(isinstance(helpers, dict) and helpers, case + ": missing executed helper hashes")
    execution = provenance.digest({"kernel": "R0", "core": source["actual_core_files_sha256"],
                                  "helpers": helpers, "observation_shim": source.get("observation_generated_source_sha256")})
    _require(execution == source.get("runner_execution_sha256") == ledger.get("runner_execution_sha256"),
             case + ": source execution hash mismatch")
    for filename, field in (("observation_generated.py", "observation_generated_source_sha256"),
                            ("observation_original.py", "observation_original_source_sha256")):
        _require(_sha(root / filename) == source.get(field), case + ": observation source hash mismatch")
    rate, seed = CASES[case]
    workload = {"kind": "full_traffic_prefix", "rate_per_hour": rate, "seed": seed,
                "arrival_window_s": 3600., "external_phase_targets": [3600., 5400.]}
    _require(result.get("subcommand") == "capture" and result.get("workload") == workload,
             case + ": full-traffic workload mismatch")
    _require(type(result.get("recorded_state_count")) is int and result["recorded_state_count"] > 0,
             case + ": missing recorded state count")
    phases = {"arrival_window": 3600., "drain_window": 5400.}
    _require(result.get("current_phase") in phases and result.get("original_advance_target") == phases[result["current_phase"]],
             case + ": original invocation target mismatch")
    physical = result["physical_outcome"]
    stop = physical.get("sim_time")
    _require(_number(stop) and stop <= 5400. and physical.get("failed") is (status == "PHYSICAL_FAILURE") and
             physical.get("episode_complete") is (status == "COMPLETED_EPISODE"), case + ": inconsistent physical stop")
    if status == "PHYSICAL_FAILURE":
        _require(physical.get("failure_time") == stop, case + ": physical failure time mismatch")
    if status == "SUSPENDED_CAPTURE":
        latest = max(t for _, c, t in WINDOWS if c == case)
        _require(_number(result.get("target_sim_time")) and result["target_sim_time"] >= latest + 60. and
                 stop >= result["target_sim_time"], case + ": suspended capture truncated before required target+60")
    if status == "COMPLETED_EPISODE":
        _require(stop == 5400., case + ": incomplete episode")
    index = _read(root / "checkpoints/checkpoint_index.json")
    _require(index.get("schema") == "ft_rs_checkpoint_v1_0", case + ": invalid index schema")
    cuts, names, sequences = [], set(), set()
    for entry in index["entries"]:
        name = entry["file"]
        _require(type(entry.get("sequence")) is int and entry["sequence"] >= 0 and
                 type(entry.get("pinned")) is bool and type(entry.get("corpus")) is bool,
                 case + ": invalid retained index flags/sequence")
        _require(Path(name).name == name and name not in names and entry["sequence"] not in sequences,
                 case + ": duplicate/unsafe indexed checkpoint")
        names.add(name); sequences.add(entry["sequence"])
        path = root / "checkpoints" / name
        sidecar_path = path.with_suffix(".runner.json")
        sidecar = _read(sidecar_path)
        _require(entry["sha256"] == _sha(path) == sidecar.get("checkpoint_sha256"), case + ": checkpoint file hash mismatch")
        _require(sidecar.get("schema") == "ft_rs_runner_checkpoint_v1_0" and
                 sidecar.get("runner_execution_sha256") == execution and sidecar.get("workload") == workload and
                 sidecar.get("source_attempt") == str(root),
                 case + ": checkpoint source/workload mismatch (absolute paths are not remapped)")
        _require(_number(sidecar.get("cumulative_effective_wall_at_checkpoint_s")) and
                 sidecar["cumulative_effective_wall_at_checkpoint_s"] <= ledger["cumulative_effective_wall_s"],
                 case + ": checkpoint ledger mismatch")
        cut = sidecar["cut"]
        _require(cut in result.get("cuts", []), case + ": checkpoint cut absent from terminal result")
        requests = cut["requested_sim_times"]
        # Older captures pinned terminal cuts with milestone=[] rather than None.
        # Accept those retained bytes only for an empty, explicitly terminal request.
        terminal_reasons = {"physical_failure", "capture_limit", "soft_wall_budget", "invocation_returned"}
        legacy_terminal = (entry["pinned"] is True and requests == [] and
                           bool(cut["reasons"]) and set(cut["reasons"]) <= terminal_reasons and
                           entry["milestone"] == [])
        _require((entry["milestone"] == (requests if "requested_milestone" in cut["reasons"] else None) or legacy_terminal) and
                 ("requested_milestone" not in cut["reasons"] or entry["pinned"] is True), case + ": milestone/index mismatch")
        _require(all(_number(t) and t <= cut["actual_sim_time"] for t in requests) and
                 cut["simulation_deferral_s"] == [cut["actual_sim_time"] - t for t in requests],
                 case + ": requested/actual milestone mismatch")
        cuts.append(dict(cut, pinned=entry["pinned"], corpus=entry["corpus"], checkpoint=str(path),
                         sidecar=str(sidecar_path), checkpoint_sha256=entry["sha256"], sidecar_sha256=_sha(sidecar_path),
                         cursor=sidecar["cursor"]))
    _require(set(result["checkpoints"]) == {c["checkpoint"] for c in cuts}, case + ": result/index checkpoint mismatch")
    _require(result.get("replay_path") == str(root / "replay"), case + ": replay source path mismatch (no remapping)")
    return {"status": status, "stop_time": stop, "cuts": cuts, "source_identity": source,
            "source_attempt": str(root), "workload": workload,
            "recorded_state_count": result.get("recorded_state_count"),
            "source_files_sha256": {n: _sha(root / n) for n in ("result.json", "attempt_ledger.json", "source_identity.json", "checkpoints/checkpoint_index.json")}}


def validate_reference_inputs(reference, sources, selections, validation_root):
    """Actual ReplayBundle and checkpoint restore validation in isolated R0."""
    helper = Path(__file__).resolve().parent
    for source in sources.values():
        expected = source["source_identity"]["helper_execution_files_sha256"]
        for name in ("__init__.py", "checkpoint.py"):
            key = "terminal_marl/core/rotating_slots/performance/" + name
            _require(expected.get(key) == _sha(helper / name), "checkpoint validator helper bytes differ from capture")
    request = {"helper": str(helper.parent), "sources": sources, "selections": selections,
               "validation_root": str(Path(validation_root).resolve())}
    request_path = Path(validation_root).resolve() / "validation_request.json"
    _require(not request_path.exists(), "validation request already exists")
    atomic_write_json(request_path, request)
    code = "import json, pathlib\nrequest = json.loads(pathlib.Path(" + repr(str(request_path)) + ").read_bytes())\n" + _VALIDATE_CHILD
    evidence = provenance.run_reference(reference, modules=("terminal_marl.core.rotating_slots.replay",),
                                        code=code, timeout=300)
    return dict(evidence["query_result"], reference_process={k: v for k, v in evidence.items() if k != "query_result"},
                request_path=str(request_path), request_sha256=_sha(request_path))


_VALIDATE_CHILD = r'''
import sys, json, pathlib
sys.path.insert(0, request["helper"])
from performance.checkpoint import CheckpointIdentity, StateCodec, load_checkpoint, restore_checkpoint
from terminal_marl.core.rotating_slots.replay import ReplayBundle, _plain
from terminal_marl.core.rotating_slots.demand import ArrivalManifest
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
codec = StateCodec()
replays, checkpoints = {}, {}
for case, source in request["sources"].items():
    root = pathlib.Path(source["source_attempt"])
    archive = ReplayBundle.validate(root / "replay")
    workload = source["workload"]
    manifest = ArrivalManifest.generate(workload["seed"], workload["rate_per_hour"], 3600.).to_dict()
    if (json.loads((root / "replay/manifest.json").read_bytes()) != manifest or
        archive.index["code_commit"] != "1c5afd37c891a0ed4776f64367fc7264db3c1c54" or
        archive.time_bounds != (0., source["stop_time"]) or
        archive.state_count != source["recorded_state_count"] or
        archive.index["config_hash"] != source["source_identity"]["semantic_fingerprint"]["config_hash"] or
        archive.index["config_bytes_sha256"] != source["source_identity"]["semantic_fingerprint"]["physical_config_bytes_sha256"]):
        raise ValueError("replay workload/commit/observed stop mismatch: " + case)
    replays[case] = {"state_count": archive.state_count, "time_bounds": list(archive.time_bounds),
                     "index": _plain(archive.index)}
for row in request["selections"]:
    if row["status"] != "AVAILABLE":
        continue
    source, cut = request["sources"][row["case"]], row["cut"]
    semantics = source["source_identity"]["semantic_fingerprint"]
    identity = CheckpointIdentity(**{k: semantics[k] for k in
        ("policy_id", "config_hash", "profile_hash", "physical_config_bytes_sha256",
         "effective_rules_hash", "candidate_policy_hash", "numerical_policy_hash")},
        implementation_sha256=source["source_identity"]["runner_execution_sha256"])
    payload = load_checkpoint(cut["checkpoint"], expected_identity=identity)
    cursor, world = codec.decode(payload["cursor"]), codec.decode(payload["world"])
    if ({k: cursor[k] for k in ("original_target", "phase", "outer_phase")} != cut["cursor"] or
        cursor["shim_source_sha256"] != source["source_identity"]["observation_generated_source_sha256"] or
        world["time"] != cut["actual_sim_time"] or cursor["original_target"] !=
        {"arrival_window": 3600., "drain_window": 5400.}.get(cursor["outer_phase"]) or
        world["manifest"].to_dict() != ArrivalManifest.generate(source["workload"]["seed"],
                 source["workload"]["rate_per_hour"], 3600.).to_dict() or
        world["physics_dt"] != .2 or world["scheduler_period"] != 5.):
        raise ValueError("checkpoint cursor/world/workload mismatch: " + row["window"])
    stream = payload["state_stream"]
    if stream is None or stream["path"] != str(pathlib.Path(source["source_attempt"]) / "states.jsonl"):
        raise ValueError("checkpoint lacks original recorded stream: " + row["window"])
    restored = restore_checkpoint(payload, expected_identity=identity, cache_mode="cold",
        new_attempt_dir=pathlib.Path(request["validation_root"]) / row["window"], world_class=RotatingSlotWorld)
    if restored.world.time != cut["actual_sim_time"]:
        raise ValueError("restored time mismatch")
    if restored.world.state_sink is not None:
        restored.world.state_sink.close()
    checkpoints[row["window"]] = {"identity": payload["identity"], "cursor": cursor,
        "world_time": world["time"], "state_stream": stream, "cache_mode": "cold",
        "restored_without_advance": True}
query_result = {"replays": replays, "checkpoints": checkpoints}
'''


def freeze_corpus(campaign, reference, output):
    """Create a new output; complete manifests appear only after validation/copy."""
    started = datetime.now(timezone.utc).isoformat()
    helper = Path(__file__).resolve().parent
    repo = helper.parents[3]
    collector_files = [helper / n for n in ("corpus.py", "__init__.py", "checkpoint.py", "provenance.py")]
    collector_files.append(repo / "scripts/server/freeze_rotating_slots_performance_inputs.py")
    collector = {"started_utc": started, "command": list(sys.argv), "python_executable": sys.executable,
                 "python_version": sys.version, "files_sha256": {str(p): _sha(p) for p in collector_files}}
    campaign, reference, output = (Path(p).resolve() for p in (campaign, reference, output))
    _require(not output.exists(), "output already exists; overwrite is forbidden")
    _require(not output.is_relative_to(campaign) and not campaign.is_relative_to(output) and
             not output.is_relative_to(reference) and not reference.is_relative_to(output),
             "output must be separate from campaign and reference")
    output.mkdir(parents=True, exist_ok=False)
    try:
        manifest = provenance._verify_export(reference)
        semantics = provenance.semantic_fingerprint(reference)
        sources = {case: read_source(campaign, case, manifest, semantics) for case in CASES}
        selections = select_windows(sources)
        validation = validate_reference_inputs(reference, sources, selections, output / "validation")
        for row in selections:
            if row["status"] != "AVAILABLE":
                continue
            cut = row["cut"]
            target = output / "inputs" / row["window"]
            target.mkdir(parents=True, exist_ok=False)
            for field, hash_field in (("checkpoint", "checkpoint_sha256"), ("sidecar", "sidecar_sha256")):
                path = Path(cut[field]); copied = target / path.name
                shutil.copyfile(path, copied)
                _require(_sha(path) == _sha(copied) == cut[hash_field], "source changed during freeze: " + str(path))
                row["frozen_" + field] = str(copied)
            row["validation"] = validation["checkpoints"][row["window"]]
        for source in sources.values():
            for name, digest in source["source_files_sha256"].items():
                _require(_sha(Path(source["source_attempt"]) / name) == digest, "source metadata changed during freeze")
        result = {"schema": "ft_rs_performance_corpus_v1_0", "status": "FROZEN" if all(
            r["status"] == "AVAILABLE" for r in selections) else "INCOMPLETE",
            "performance_status": "NOT_QUALIFIED", "human_status": "HUMAN_PENDING",
            "campaign": str(campaign), "reference": str(reference), "reference_commit": provenance.R0_COMMIT,
            "reference_source_sha256": provenance.R0_SOURCE_SHA256, "semantic_fingerprint": semantics,
            "sources": sources, "windows": selections, "replay_validation": validation["replays"],
            "reference_validation_process": {k: v for k, v in validation.items() if k not in ("replays", "checkpoints")},
            "collector": dict(collector, finished_utc=datetime.now(timezone.utc).isoformat()),
            "stream_dependency": "Frozen checkpoint bytes retain the original immutable absolute source states.jsonl path; retain source streams and their recorded prefix offset/hash. No path remapping or implementation migration is performed."}
        result["manifest_sha256"] = provenance.digest(result)
        atomic_write_json(output / "corpus_manifest.json", result)
        return result
    except Exception as exc:
        atomic_write_json(output / "freeze_report.json", {"status": "INCOMPLETE", "error": str(exc),
            "error_type": type(exc).__name__, "performance_status": "NOT_QUALIFIED", "human_status": "HUMAN_PENDING",
            "collector": dict(collector, finished_utc=datetime.now(timezone.utc).isoformat())})
        raise


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "reference", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = freeze_corpus(args.campaign, args.reference, args.output)
        print(json.dumps({k: result[k] for k in ("status", "manifest_sha256", "performance_status", "human_status")}))
        return 0 if result["status"] == "FROZEN" else 2
    except Exception as exc:
        print(json.dumps({"status": "INCOMPLETE", "error": str(exc), "error_type": type(exc).__name__}))
        return 2
