"""Bounded FT-RS-SELECTIVE-1.0 Stage A; each stage is explicitly requested."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[2]
DRIVER = Path(__file__).resolve()
PACKAGE = PROJECT / "new10.16/rs_selective_reoptimization_v1_0"
METHODS = ("FULL_H", "RULE_LOCAL_K8")
ROOTS = {"s3r23": (7324, "medium24_rate150_v1", (15, 30, 45, 60)),
         "s3r24": (7325, "stress36_rate180_v1", (60, 120, 180, 240))}
RAW_HASHES = {
    "s3r23": ("f5560d54108a4e9434900a02698cf3b26d43055ffe3a62b6bb62353e2d20b485", "d4a0513ec327a7bc77d33e300d1401194a5a057bf9e1229b3836e845fcfcca79"),
    "s3r24": ("8c234fddab5bf8378636cb03ba015e38cf516d96f248d61f4c3be21a8388ea14", "7cd897ff9d3562920c80504520d41891af7339f3b3ec8c71a020da9bd95457fe")}
SNAPSHOT_HASHES = {
    "s3r23": ("b99c9ea38806e052811acbc330c4a5f062d45eeed14fced19f0124cc9244d152", "5157ab963a44b8dd78f27a7aae2d1c2e56e7881769751b692901733f3f510cd0", "0b52f23a804d4931441d5895f019ac843831d103b6fca442ad467e28c872405b", "b9f4ed4be4e935ab3bb3b2a27f58121a4969195a7b3845dbf511e3311016a4ac"),
    "s3r24": ("22f19dc912083f3bbfe40a7d393def68a629b7155da5e4898ba062add3ffdfba", "5e54b4a849a11a101322954637fc4beb6afaa0a23885e5818540070b49ac181b", "7ae720c01fa28d9a6b3e9971633e49d43755219c9b87893d9925f21769fcdeaa", "dfc7d29e80b1ef1a33b89ad656dafc6fb977e6f4139baf84b605dd2fa0b452a2")}
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
CAP = 7200.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    # Normalize object keys to their stored JSON form before sorting: resource
    # probes can use integer CPU IDs while reloaded artifacts use string keys.
    normalized = json.loads(json.dumps(value, allow_nan=False))
    return sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def print_result(value):
    """Use the same plain, finite JSON representation as saved atomic outputs."""
    from terminal_marl.core.rotating_slots.stage2_campaign import canonical
    print(canonical(value).decode("utf8"))


def parse_contract(phase_path=PACKAGE / "phase_A_v1_0.yaml"):
    """Fail closed for any changed v1.0 control, including previously unknown keys.

    Full document equality is deliberate: this runner implements the immutable
    supplied specification, not arbitrary YAML overrides. Inactive B/C controls
    are retained in the binding and never drive a Stage A action.
    """
    import yaml
    phase_path = Path(phase_path)
    phase = yaml.safe_load(phase_path.read_text(encoding="utf8"))
    require(phase.get("phase") == "A", "only Stage A is authorized")
    require(phase.get("extends") == "selective_reoptimization_config_v1_0.yaml", "unexpected parent")
    parent_path = phase_path.parent / phase["extends"]
    parent = yaml.safe_load(parent_path.read_text(encoding="utf8"))
    canonical_parent = yaml.safe_load((PACKAGE / "selective_reoptimization_config_v1_0.yaml").read_text(encoding="utf8"))
    canonical_phase = yaml.safe_load((PACKAGE / "phase_A_v1_0.yaml").read_text(encoding="utf8"))
    # Package content identity is independent of the caller's path.
    require(file_hash(PACKAGE / "selective_reoptimization_config_v1_0.yaml") == "d21fb07d12661355280abbe37a77f3bfe8df12a0e69b1062433b7024fed5aa11", "immutable parent specification changed")
    require(file_hash(PACKAGE / "phase_A_v1_0.yaml") == "c1da3f0092ea8c788f4b1f17e5ba2467e89d93eba1fa389a0e615c66de4d4bc2", "immutable A specification changed")
    require(parent == canonical_parent and phase == canonical_phase, "unsupported or changed specification control")
    require(phase["settings"] == parent["phase_A"], "A override differs from parent")
    return dict(parent=parent, phase=phase, phase_sha256=file_hash(phase_path), parent_sha256=file_hash(parent_path),
                active_sections=["source", "preserve", "objective", "scope", "timing", "search", "execution", "neighborhood_pool", "campaign", "phase_A", "verification"],
                inactive_sections={k: "NOT_AUTHORIZED_IN_A" for k in ("model", "training", "adaptive_size", "phase_B", "phase_C")})


def dependencies(checkout):
    checkout = Path(checkout).resolve()
    require(checkout == PROJECT.resolve(), "runner must use its actual local/main synchronized source")
    latency = module(checkout / "scripts/stage3/run_latency_interface.py", "_selective_latency")
    d = latency.dependencies(checkout)
    d.latency = latency
    from terminal_marl.core.rotating_slots.stage2_campaign import restore_checkpoint
    from terminal_marl.core.rotating_slots.planning_latency import PlanningTiming, project_visible_snapshot, _FuturePort
    d.restore_checkpoint, d.PlanningTiming, d.project_visible_snapshot, d.FuturePort = restore_checkpoint, PlanningTiming, project_visible_snapshot, _FuturePort
    return d


def git(*args):
    return subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()


def source_binding(expected_commit):
    require(os.name == "nt", "Git source freeze is local-only; server must consume --source-freeze")
    require(git("rev-parse", "HEAD") == expected_commit, "actual source commit differs from approved commit")
    paths=("terminal_marl", "scripts", "tests", "configs/stage3", "new10.16/rs_selective_reoptimization_v1_0",
           "new10.16/rotating_slots_stage3_learning_v1_0", "new10.16/fixed_terminal_rotating_slots_stage1_config_v1_0.yaml",
           "new10.16/fixed_terminal_rotating_slots_stage2_service_half_v1.yaml")
    require(not git("diff", "HEAD", "--", *paths), "uncommitted source changes require a new reviewed freeze")
    names = git("ls-files", *paths).splitlines()
    require(DRIVER.relative_to(PROJECT).as_posix() in names, "new runner must be committed before prepare")
    files = {name: file_hash(PROJECT / name) for name in names if (PROJECT / name).is_file()}
    return dict(commit=expected_commit, branch=git("branch", "--show-current"), checkout=str(PROJECT),
                files=files, source_hash=digest(files), tracked_diff=git("diff", "HEAD"),
                status=git("status", "--short", "--untracked-files=no"), synchronization="SYNCTHING_ONLY")


def verify_sources(source):
    """Server source verification reads bytes only; it never invokes Git."""
    require(source["source_hash"] == digest(source["files"]), "source freeze manifest changed")
    for name, expected in source["files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe source manifest path")
        require(file_hash(PROJECT / relative) == expected, "source not synchronized to local freeze: " + name)
    require(DRIVER.relative_to(PROJECT).as_posix() in source["files"], "freeze does not include Stage A runner")


def validate_inputs(mapping, d):
    require(set(mapping) == {"initial_parent", "snapshots"}, "input map must contain initial_parent and snapshots only")
    require(set(mapping["snapshots"]) == set(ROOTS), "two exact preregistered development roots required")
    parent = Path(mapping["initial_parent"])
    result = {}
    for root, (seed, profile, times) in ROOTS.items():
        base = parent / "roots" / root
        row, manifest, initial = (load(base / name) for name in ("root.json", "manifest.json", "initial_state.json"))
        require((row["root_id"], row["seed"], row["scenario_profile"]) == (root, seed, profile), "wrong original root")
        require((d.digest(manifest), d.digest(initial)) == RAW_HASHES[root], "original raw root hashes changed")
        require((row["manifest_hash"], row["initial_state_hash"]) == RAW_HASHES[root], "root registry digests changed")
        checkpoint = base / "initial.checkpoint.json"
        require(load(checkpoint)["payload"]["identity"] == row["identity"], "original t0 checkpoint identity differs")
        snapshots = mapping["snapshots"][root]
        require(len(snapshots) == 4, "four preregistered checkpoints required per root")
        normalized = []
        for item, expected_time, expected_hash in zip(snapshots, times, SNAPSHOT_HASHES[root]):
            require(set(item) == {"time", "path", "sha256"}, "unexpected snapshot mapping control")
            require(item["time"] == expected_time and item["sha256"] == expected_hash, "snapshot preregistration differs")
            path = Path(item["path"])
            require(file_hash(path) == expected_hash, "saved development checkpoint hash differs")
            payload = load(path)["payload"]
            require(payload["time"] == expected_time and payload["identity"]["manifest_sha256"] == row["manifest_hash"], "saved checkpoint root/time differs")
            normalized.append(dict(item, identity=payload["identity"]))
        result[root] = dict(row=row, original_directory=str(base), initial_checkpoint=str(checkpoint),
                            initial_checkpoint_sha256=file_hash(checkpoint), snapshots=normalized)
    return result


def checks_evidence(paths):
    require(bool(paths), "actual passing targeted-test XML required")
    result = []
    for path in paths:
        tree = ET.parse(path).getroot()
        cases = list(tree.iter("testcase"))
        require(cases and not list(tree.iter("failure")) and not list(tree.iter("error")), "targeted tests missing or failed")
        require(not list(tree.iter("skipped")), "skipped targeted tests cannot establish entry gate")
        result.append(dict(path=str(Path(path).resolve()), sha256=file_hash(path), tests=len(cases)))
    return result


def prepare(args, d):
    started = time.perf_counter()
    contract = parse_contract(args.config)
    source = load(args.source_freeze)
    verify_sources(source)
    inputs = validate_inputs(load(args.input_map), d)
    checks = checks_evidence(args.checks_xml)
    require(args.cpu == 12, "preregistered physical CPU12 required")
    require(all(d.launch_threads.get(k) == "1" for k in THREADS), "set numerical thread variables before Python starts")
    resource = d.campaign.resource_probe()
    expected_versions={"numpy":"2.5.1","scipy":"1.18.1","ortools":"9.15.6755","PyYAML":"6.0.3"}
    require(all(resource.get("versions",{}).get(k)==v for k,v in expected_versions.items()), "numerical dependencies differ from preregistered preflight")
    require(args.cpu in resource.get("free_physical_cores", []), "preregistered physical CPU unavailable")
    output = args.output.resolve()
    require(not output.exists(), "fresh Stage A output required; no budget reset")
    require(not output.is_relative_to(PROJECT), "server outputs must be outside synchronized source")
    require(output.parent == Path.home() / "training_outputs/terminal", "output must be in ~/training_outputs/terminal")
    output.mkdir(parents=True)
    ledger = d.campaign.CampaignLedger(output / "budget_ledger.json")
    ledger.data.update(total_s=CAP, compute_stop_s=CAP, phase_caps={"A": CAP}, phase_used={}, jobs={})
    ledger.persist(); ledger.pause()
    ledger_initial = ledger.elapsed
    # The original t0 serialization is copied without changing physical state.
    for root, item in inputs.items():
        target = output / "roots" / root
        target.mkdir(parents=True)
        base = Path(item["original_directory"])
        for name in ("root.json", "manifest.json", "initial_state.json", "initial.checkpoint.json"):
            shutil.copyfile(base / name, target / name)
        item["run_checkpoint"] = str(target / "initial.checkpoint.json")
    binding = dict(schema="rs_selective_run_binding_v1", contract=contract, source=source,
                   source_commit=source["commit"], source_hash=source["source_hash"], inputs=inputs,
                   input_map_sha256=file_hash(args.input_map), environment=resource, cpu=args.cpu,
                   checks=checks, phase="A", no_auto_advance=True, human_replay_status="HUMAN_PENDING",
                   methods=list(METHODS), episode_seconds=300, boundaries_seconds=5, checkpoint_seconds=5,
                   snapshot_cache="COLD_NORMAL_ONLY_PRESERVE_PHYSICAL_COMMITMENTS", fixed_lead_s=20.,
                   phase_wall_cap_s=CAP, snapshot_pair_watchdog_s=150., closed_loop_watchdog_s=1200.)
    binding["binding_id"] = digest(binding)
    d.atomic_json(output / "resolved_run_binding.json", binding)
    ledger.credit("prepare", max(0.,time.perf_counter() - started-ledger_initial), phase="A"); ledger.pause()
    return dict(status="PREPARED", binding_id=binding["binding_id"])


def verify(output, d):
    binding = load(output / "resolved_run_binding.json")
    require(binding["binding_id"] == digest({k: v for k, v in binding.items() if k != "binding_id"}), "run binding changed")
    require(binding["phase"] == "A" and binding["methods"] == list(METHODS), "unauthorized phase/method")
    verify_sources(binding["source"])
    require(parse_contract() == binding["contract"], "bound contract changed")
    require(all(d.launch_threads.get(k)=="1" for k in THREADS), "one numeric thread required at process launch")
    for name in ("numpy","scipy","PyYAML","ortools"):
        require(importlib.metadata.version(name)==binding["environment"]["versions"][name], "bound dependency changed: "+name)
    ledger = d.campaign.CampaignLedger(output / "budget_ledger.json")
    require(ledger.data["total_s"] == CAP and ledger.data["compute_stop_s"] == CAP and ledger.data["phase_caps"] == {"A": CAP}, "Stage A ledger cap changed")
    return binding, ledger


class ColdRestore:
    """Accept historical scheduler bytes without reviving its normal computation."""
    planning_profile = "downstream_service_v8"
    def load_state_dict(self, state):
        self.saved = state


def restore(path, d, directory):
    path = Path(path)
    identity = load(path)["payload"]["identity"]
    directory.mkdir(parents=True, exist_ok=True)
    world = d.restore_checkpoint(path, ColdRestore(), identity, stream_copy=directory / "restored_states.jsonl")
    world.configure_practical_runtime(True)
    return world


def scheduler(d, method, frame=None):
    return d.NormalSchedulerCallback("rh_cp_sat", planning_profile="downstream_service_v8",
        online_policy="progressive_v1", selective_method=method, global_frame=frame)


def lifecycle_summary(tasks, callbacks, services, latency):
    """Extend existing accounting for the new charged history service once."""
    value=latency.lifecycle_summary(tasks,callbacks,services)
    per_task=value["per_task_work"]
    for row,task in zip(per_task,tasks):
        history=task.get("selective_history_work_s",0.)
        row["full_normal_planning_work_s"]+=history
        row["observed_components"]["selective_history_work_s"]=history
    attributed=sum(task.get("selective_history_work_s",0.) for task in tasks)
    service_total=sum(row.get("wall_s",0.) for row in services if row.get("kind")=="selective_history")
    unmatched=max(0.,service_total-attributed)
    value["selective_history_work"]=dict(task_attributed_s=attributed,service_total_s=service_total,unattributed_service_s=unmatched)
    value["full_normal_planning_work"]=latency.measured(row["full_normal_planning_work_s"] for row in per_task)
    value["full_normal_planning_work"]["total_s"]+=unmatched
    value["finished_task_full_planning_work"]=latency.measured(row["full_normal_planning_work_s"] for row in per_task if row["finished"])
    finished=[task for task in tasks if "finished_time" in task]
    deadline_reasons={"READY_DEADLINE_MISSED","LATENCY_PACKET_EXPIRED","TRAFFIC_CHANGED_AFTER_DEADLINE","REVALIDATION_WORK_DEADLINE"}
    late=sum(str(task.get("reason","")).split(":",1)[0] in deadline_reasons for task in finished)
    value.update(late_finished_tasks=late,late_rate=late/len(finished) if finished else None,
                 late_rate_denominator="ALL_FINISHED_TASKS_PENDING_EXCLUDED",late_reason_set=sorted(deadline_reasons),
                 availability_delay_s=latency.measured(task["finished_time"]-task["snapshot_time"] for task in finished))
    return value


def selection_envelopes(records, run_id, binding_id, root):
    """Translate internal telemetry to the contract's public record envelope."""
    result = []
    for index, row in enumerate(records):
        telemetry = row.get("selective_replanning")
        if not telemetry:
            continue
        # A failed/unknown frame has no selected set or common objective frame.
        # Keep that raw diagnostic in scheduler_records; do not invent Selection.
        if telemetry.get("scope") is None or not telemetry.get("scope",{}).get("objective_frame_id"):
            continue
        scope = deepcopy(telemetry["scope"])
        scope["selected_ids"] = list(scope.pop("replan_ids", scope.get("selected_ids", [])))
        scope["background_ids"] = list(scope.pop("fixed_background_ids", scope.get("background_ids", [])))
        scope.setdefault("objective_frame_id", telemetry.get("objective_frame_id", ""))
        scope.setdefault("actual_k", len(scope["selected_ids"]))
        scope.setdefault("nominal_k", 8)
        scope.setdefault("mandatory_ids", [])
        scope.setdefault("eligible_ids", [])
        require(set(scope["selected_ids"]) <= set(scope["eligible_ids"]), "selection outside eligibility")
        require(not set(scope["selected_ids"]) & set(scope["background_ids"]), "selection/background overlap")
        timing = row.get("planning_timing", {})
        snapshot_time = timing.get("snapshot_time", row.get("time", 0.))
        result.append(dict(schema_version="rs_select_record_v1", record_type="Selection", run_id=run_id,
            source_binding=binding_id, snapshot_id=str(scope.get("snapshot_id", f"{run_id}:{index}")),
            ancestry_root_id=root, split="development", snapshot_time=snapshot_time,
            effective_time=timing.get("effective_time", snapshot_time), scope=scope,
            measurements=telemetry, status=str(telemetry.get("status", row.get("status", "RECORDED")))))
    return result


def save_logs(directory, world, callback, wrapper, binding, root, d):
    for name, value in (("scheduler_records.json", callback.records), ("normal_tasks.json", wrapper.records),
                        ("online_services.json", wrapper.online_services), ("dispatch_timings.json", world.dispatch_timings),
                        ("selection_records.json", selection_envelopes(callback.records, directory.name, binding["binding_id"], root))):
        d.atomic_json(directory / name, value)


def closed_job(output, root, method, binding, d):
    started, cpu_started = time.perf_counter(), time.process_time()
    item = binding["inputs"][root]
    require(file_hash(item["run_checkpoint"]) == item["initial_checkpoint_sha256"], "copied t0 checkpoint changed")
    directory = output / "evaluation" / root / method
    require(not directory.exists(), "closed-loop task already attempted; no retry")
    world = restore(item["run_checkpoint"], d, directory)
    callback = scheduler(d, method)
    wrapper = d.LatencyAwareScheduler(callback, lead_seconds=20., preparation_margin=1., search_handoff_reserve_seconds=4.)
    world.decision_callback, world.admission_callback = wrapper, d.shared_fifo_admission
    require(world.time == 0 and world.scheduler_period == 5., "t0/heartbeat changed")
    identity = dict(item["row"]["identity"], source_hash=binding["source_hash"])
    holding, restore_observer = d.evaluation.observe_executed_holding(world)
    completed, error = 0, None
    advance_wall = advance_cpu = 0.
    initial_counts = d.episode_metrics(world)
    try:
        d.save_checkpoint(directory / "checkpoint_0000.json", world, wrapper, identity)
        for boundary in range(5, 301, 5):
            d.atomic_json(output / "progress" / f"{root}_{method}.json", dict(root=root, method=method, pid=os.getpid(), phase="ADVANCING", completed_boundary_s=completed, target=boundary, wall_s=time.perf_counter()-started))
            before, before_cpu = time.perf_counter(), time.process_time()
            world.advance(float(boundary), wrapper, settle_only_at_end=True)
            advance_wall += time.perf_counter()-before; advance_cpu += time.process_time()-before_cpu
            if abs(world.time-boundary)>1e-7 or world.failed or getattr(world,"terminated_invalid",False):
                break
            completed = boundary
            life = lifecycle_summary(wrapper.records, callback.records, wrapper.online_services,d.latency)
            quality=d.episode_metrics(world)
            from terminal_marl.core.rotating_slots.world import AIRBORNE
            quality["unresolved_low_soc_count"]=sum(a.status in AIRBORNE and a.soc<.25 for a in world.aircraft.values())
            quality["unresolved_low_soc_threshold"]=.25
            row = dict(root_id=root, method=method, time=world.time, quality=quality,
                initial_counts=initial_counts, normal_lifecycle=life, holding_energy_by_ring=dict(holding),
                costs=dict(job_elapsed_wall_s=time.perf_counter()-started, process_cpu_s=time.process_time()-cpu_started,
                    world_advance_wall_s=advance_wall, world_advance_cpu_s=advance_cpu,
                    dispatch_wall_s=sum(x.get("dispatch_wall_s",0.) for x in world.dispatch_timings),
                    simulation_timing_scope="ADVANCE_MINUS_DISPATCH_INCLUDES_INTEGRATION_EXECUTION_CHECKS_RECORDING_NOT_PURE_PHYSICS"),
                pending=wrapper.pending is not None, source_binding=binding["binding_id"],
                selection_summary=module(PROJECT/"scripts/stage3/report_selective_reoptimization_a.py","_selective_boundary").selection_summary(callback.records))
            d.save_checkpoint(directory / f"checkpoint_{boundary:04d}.json", world, wrapper, identity, boundary_summary=row)
            d.atomic_json(directory / f"boundary_{boundary:04d}.json", row)
            save_logs(directory, world, callback, wrapper, binding, root, d)
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
    finally:
        restore_observer()
        save_logs(directory, world, callback, wrapper, binding, root, d)
        terminal = d.evaluation.terminal_observation(world)
        d.atomic_json(directory / "terminal_observation.json", terminal)
        d.save_checkpoint(directory / "checkpoint_terminal.json", world, wrapper, identity)
        record = world.to_dict()
        record.update(decisions=callback.records, stage3_trace=[dict(record_type="NormalTask", **r) for r in wrapper.records],
                      stage2_provenance=dict(execution_scope=binding),
                      tentative_predictions=[dict(aircraft_id=aid, as_of=world.time, plans=[p.to_dict() for p in plans]) for aid, plans in callback.provider.previous.items()])
        d.ReplayBundle.write(directory / "replay", record=record, code_commit=binding["source_commit"],
                            config_path=world.config.path, command=str(DRIVER), seed=ROOTS[root][0], eager=False)
        result = dict(status="COMPLETE" if completed==300 and error is None else "EVALUATION_PARTIAL",
            root_id=root, method=method, completed_boundary_s=completed, terminal_observation=terminal,
            error=error, full_job_wall_s=time.perf_counter()-started, process_cpu_s=time.process_time()-cpu_started,
            actual_cpu=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
            replay_status="WRITTEN_HUMAN_PENDING", source_binding=binding["binding_id"])
        d.atomic_json(directory / "result.json", result)
    if error:
        raise RuntimeError(str(error))
    return result


def snapshot_job(output, root, stamp, binding, d):
    from terminal_marl.core.rotating_slots.selective_replanning import build_global_frame
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider, EvaluationContext, PlanningBudget, QUALITY_BUDGET_PROFILE
    from terminal_marl.core.rotating_slots.performance.parallel import _clear_worker_accelerators
    item = next(r for r in binding["inputs"][root]["snapshots"] if r["time"] == stamp)
    require(file_hash(item["path"]) == item["sha256"], "registered snapshot changed")
    directory = output / "snapshots" / root / f"{stamp:04d}"
    require(not directory.exists(), "snapshot comparison already attempted; no retry")
    world = restore(item["path"], d, directory)
    source = world.build_planning_snapshot()
    timing = d.PlanningTiming.from_lead(source.time,20.,1.)
    projection_started = time.perf_counter()
    projected = d.project_visible_snapshot(source,timing.effective_time,timing=timing)
    projection_wall = time.perf_counter()-projection_started
    frame_started = time.perf_counter()
    frame = build_global_frame(projected, CandidateProvider(planning_profile="downstream_service_v8"),
                               EvaluationContext(projected,budget=PlanningBudget.for_profile(QUALITY_BUDGET_PROFILE)))
    frame_wall = time.perf_counter()-frame_started
    rows = []
    for method in METHODS:
        method_started=time.perf_counter()
        cache_reset=_clear_worker_accelerators()
        reset_finished=time.perf_counter()
        cache_reset_wall=reset_finished-method_started
        callback = scheduler(d, method, frame)
        callback.initialize_imported_history(source, events=world.events)
        started = time.perf_counter()
        initialization_wall=started-reset_finished
        common_work = projection_wall + frame_wall
        port = d.FuturePort(projected, source, method_started+19.-common_work, search_wall_deadline=method_started+15.-common_work,
                            planning_wall_origin=method_started-common_work, search_handoff_reserve_seconds=4.)
        callback(port)
        callback_wall = time.perf_counter()-started
        preparation=counterfactual_preparation(world,port.proposal,timing,
            ready_wall_deadline=method_started+19.-common_work)
        require(world.build_planning_snapshot().version==source.version,"snapshot preparation changed source physical commitments/state")
        rows.append(dict(method=method, callback_wall_s=callback_wall,initialization_wall_s=initialization_wall,
            cache_reset_wall_s=cache_reset_wall,cache_reset=cache_reset,
            common_projection_wall_s=projection_wall, common_frame_wall_s=frame_wall,
            snapshot_observed_compute_s=projection_wall+frame_wall+cache_reset_wall+initialization_wall+callback_wall+preparation["wall_s"],
            snapshot_cost_scope="COMMON_PROJECTION_FRAME_PLUS_COLD_SETUP_PLANNING_AND_ONE_NONPUBLISHING_PREPARATION",
            preparation=preparation,full_online_pipeline_cost_s=None,
            physical_source_version=source.version,physical_source_unchanged=True,
            unobserved_online_stages=["AFFECTED_TRAFFIC_REVALIDATION","ONLINE_SERVICE_DELAY","ACTUAL_DELIVERY"],
            scheduler_records=callback.records, proposal_hashes=[p.trajectory_hash for p in (port.proposal or ())],
            selection_records=selection_envelopes(callback.records,f"{root}:{stamp}:{method}",binding["binding_id"],root)))
        d.atomic_json(directory / f"{method}.json", rows[-1])
    frame_ids = [row["selection_records"][0]["scope"]["objective_frame_id"] for row in rows if row["selection_records"]]
    require(len(frame_ids)==2 and len(set(frame_ids))==1, "paired snapshot did not use same frame")
    result = dict(status="COMPLETE", root_id=root, time=stamp, effective_time=timing.effective_time,
                  original_checkpoint_sha256=item["sha256"], same_frame=True, cold_normal_cache=True,
                  physical_execution=False, rows=rows, source_binding=binding["binding_id"])
    d.atomic_json(directory / "result.json", result)
    return result


def counterfactual_preparation(world,proposal,timing,*,ready_wall_deadline):
    """Certify once through the real source port, discard its token, never publish.

    Preparation has no interruptible subdeadline API. Its measured completion is
    checked against the same outer readiness limit; a late result is unusable.
    Later live-traffic revalidation and delivery are unobserved counterfactuals.
    """
    if not proposal:
        return dict(status="NO_PROPOSAL",accepted=None,ready_eligible=False,wall_s=0.,calls=0,
                    reason="NO_NONEMPTY_CERTIFIED_GROUP",published=False)
    started=time.perf_counter();token=None;evaluation=None;error=None
    try:
        evaluation,token=world._scheduler_port.prepare_latency_submission(tuple(proposal),timing)
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc))
    finally:
        if token is not None:
            world._scheduler_port.discard_latency_submission(token)
    finished=time.perf_counter()
    accepted=None if evaluation is None else bool(evaluation.accepted)
    raw_status=None if evaluation is None else evaluation.status
    late=finished>ready_wall_deadline
    missing_packet=bool(accepted and token is None)
    status=("UNKNOWN" if error or missing_packet or raw_status=="UNKNOWN" else
            "LATE" if late else "PREPARED_COUNTERFACTUAL" if accepted else "REJECTED")
    return dict(status=status,accepted=accepted,raw_status=raw_status,
        reason="MISSING_PREPARATION_TOKEN" if missing_packet else getattr(evaluation,"reason",None),
        ready_eligible=bool(accepted and not late and not missing_packet and error is None),
        late=late,wall_s=finished-started,calls=1,token_issued=token is not None,
        token_discarded=token is not None,published=False,error=error,
        readiness_margin_s=ready_wall_deadline-finished,
        full_online_delivery_status="UNOBSERVED_NO_PUBLISH_OR_FUTURE_ROLLOUT")


def run_stage(output, stage, d):
    with d.campaign.campaign_lock(output):
        binding, ledger = verify(output,d)
        require(not (output / f"{stage}.started.json").exists(), "stage already attempted; no automatic retry")
        if stage == "run":
            require(load(output / "snapshots.finished.json")["status"] == "COMPLETE", "complete snapshot stage required before closed loop")
        require(binding["cpu"] in d.campaign.resource_probe().get("free_physical_cores", []), "paired physical core unavailable")
        d.atomic_json(output / f"{stage}.started.json", dict(started_at=d.campaign.utc_now(), binding=binding["binding_id"]))
        receipts = []; status = "COMPLETE"
        jobs = [(root, stamp, None) for root in ROOTS for stamp in ROOTS[root][2]] if stage=="snapshots" else [(root,None,m) for root in ROOTS for m in METHODS]
        try:
            with ledger.phase("A"):
                for root, stamp, method in jobs:
                    job_id = f"snapshot_{root}_{stamp}" if stamp is not None else f"closed_{root}_{method}"
                    cap = binding["snapshot_pair_watchdog_s" if stamp is not None else "closed_loop_watchdog_s"]
                    if ledger.remaining("A") < cap:
                        status = "BUDGET_LIMITED"; break
                    command = [sys.executable,str(DRIVER),"job","--output",str(output),"--root",root]
                    command += ["--snapshot-time",str(stamp)] if stamp is not None else ["--method",method]
                    receipt = d.campaign.run_subprocess(ledger,"A",job_id,command,output/"logs",cap=cap)
                    receipts.append(receipt); d.atomic_json(output/"receipts"/(job_id+".json"),receipt)
                    if receipt["status"] == "BUDGET_LIMITED":
                        status = "BUDGET_LIMITED"
                        if stamp is not None: break
                        base=output/"evaluation"/root/method
                        d.atomic_json(base/"watchdog_partial.json",dict(status="BUDGET_LIMITED",receipt=receipt,
                            interpretation="Timeout is computational censoring; last saved completed boundary remains authoritative; process-kill state is unavailable"))
                        continue
                    if receipt["status"] != "completed":
                        status = "FAILED_RETURN_TO_PRIMARY"; break
                    result_path=(output/"snapshots"/root/f"{stamp:04d}" if stamp is not None else output/"evaluation"/root/method)/"result.json"
                    if load(result_path)["status"]!="COMPLETE":
                        status="PARTIAL_RETURN_TO_PRIMARY";break
        except Exception:
            status="FAILED_RETURN_TO_PRIMARY"
            raise
        finally:
            ledger.pause()
            result = dict(status=status,receipts=receipts,expected_jobs=len(jobs),attempted_jobs=len(receipts),
                          no_automatic_retry=True,finished_at=d.campaign.utc_now())
            d.atomic_json(output / f"{stage}.finished.json",result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze","prepare","snapshots","run","job","report","credit"))
    parser.add_argument("--output",type=Path,default=Path.home()/"training_outputs/terminal/rs_selective_reoptimization_a_20261027")
    parser.add_argument("--config",type=Path,default=PACKAGE/"phase_A_v1_0.yaml")
    parser.add_argument("--input-map",type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-freeze",type=Path)
    parser.add_argument("--checks-xml",type=Path,action="append",default=[])
    parser.add_argument("--cpu",type=int,default=12)
    parser.add_argument("--root",choices=tuple(ROOTS)); parser.add_argument("--method",choices=METHODS)
    parser.add_argument("--snapshot-time",type=int)
    parser.add_argument("--audit",action="store_true",help="internal bounded saved-replay audit job")
    parser.add_argument("--credit-id"); parser.add_argument("--credit-wall",type=float)
    args = parser.parse_args(argv)
    if args.stage=="freeze":
        require(args.source_commit and args.source_freeze,"local freeze requires --source-commit and --source-freeze")
        source=source_binding(args.source_commit)
        args.source_freeze.parent.mkdir(parents=True,exist_ok=True)
        with args.source_freeze.open("x",encoding="utf8") as stream: json.dump(source,stream,sort_keys=True,allow_nan=False)
        print(json.dumps(dict(status="LOCAL_SOURCE_FROZEN",source_hash=source["source_hash"]))); return
    launch_threads={key:os.environ.get(key) for key in THREADS}
    parse_contract(args.config)
    d = dependencies(PROJECT); d.launch_threads=launch_threads; output=args.output.resolve()
    if args.stage=="prepare":
        require(args.input_map is not None and args.source_freeze, "prepare requires explicit input map and local reviewed --source-freeze")
        result=prepare(args,d)
    elif args.stage=="credit":
        _,ledger=verify(output,d)
        require(args.credit_id and args.credit_wall is not None,"credit requires ID and measured seconds")
        ledger.credit(args.credit_id,args.credit_wall,phase="A"); result=dict(status="CREDITED",elapsed_s=ledger.elapsed)
    elif args.stage=="job":
        binding,ledger=verify(output,d)
        require(args.root and (args.method is not None) != (args.snapshot_time is not None), "job requires root and exactly method or snapshot time")
        job_id=f"audit_{args.root}_{args.method}" if args.audit else f"snapshot_{args.root}_{args.snapshot_time}" if args.snapshot_time is not None else f"closed_{args.root}_{args.method}"
        require((ledger.data.get("active_phase") or {}).get("phase")=="A" and ledger.data["jobs"].get(job_id,{}).get("status")=="running", "job requires active parent watchdog")
        require(hasattr(os,"sched_setaffinity"), "formal paired runs require physical CPU affinity")
        os.sched_setaffinity(0,{binding["cpu"]})
        result=(module(PROJECT/"scripts/stage3/report_selective_reoptimization_a.py","_selective_audit").audit_job(output/"evaluation"/args.root/args.method,PROJECT) if args.audit else
                snapshot_job(output,args.root,args.snapshot_time,binding,d) if args.snapshot_time is not None else closed_job(output,args.root,args.method,binding,d))
    elif args.stage=="report":
        binding,ledger=verify(output,d)
        with d.campaign.campaign_lock(output):
            try:
                with ledger.phase("A"):
                    for root in ROOTS:
                        for method in METHODS:
                            base=output/"evaluation"/root/method
                            if not (base/"replay/replay.json").is_file(): continue
                            job_id=f"audit_{root}_{method}"
                            if job_id in ledger.data["jobs"]: continue
                            allowance=min(600.,max(0.,ledger.remaining("A")-10.))
                            if allowance<=0: break
                            command=[sys.executable,str(DRIVER),"job","--output",str(output),"--root",root,"--method",method,"--audit"]
                            receipt=d.campaign.run_subprocess(ledger,"A",job_id,command,output/"logs",cap=allowance)
                            d.atomic_json(output/"receipts"/(job_id+".json"),receipt)
                    result=module(PROJECT/"scripts/stage3/report_selective_reoptimization_a.py","_selective_report").report(output,PROJECT)
            finally: ledger.pause()
    else: result=run_stage(output,args.stage,d)
    print_result(dict(stage=args.stage,output=str(output),**result))


if __name__=="__main__":
    main()
