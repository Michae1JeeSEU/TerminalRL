"""Read-only Stage2 catalog trace and separately budgeted physical witnesses.

No advance, submit, commit, refinement, or checkpoint stream restoration occurs.
Witness enumeration is diagnostic only; it does not feed CandidateProvider.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from hashlib import sha256
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from terminal_marl.core.rotating_slots.geometry import ring_config
from terminal_marl.core.rotating_slots import stage2_campaign as campaign
from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.stage2_campaign import busy_fixture, frozen_manifest
from terminal_marl.core.rotating_slots.stage2_planning import (
    CandidateProvider, EvaluationContext, PlanningBudget, ReferenceProvider, _node,
)
from terminal_marl.core.rotating_slots.trajectory import generate_final, generate_transfer, screen_normal_templates
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID, evaluate_planning_snapshot


class ProbeDeadline(TimeoutError):
    pass


def json_value(value):
    """Represent nonfinite diagnostics explicitly, never emit JSON NaN tokens."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [json_value(v) for v in value]
    return value


def result_row(result):
    return dict(accepted=result.accepted, status=result.status, reason=result.reason,
                details=json_value(result.details))


def plan_row(plan):
    return dict(aircraft_id=plan.aircraft_id, source_ring=plan.source_ring,
                source_slot=plan.source_slot, target_ring=plan.target_ring,
                target_slot=plan.target_slot, pad_id=plan.pad_id,
                start_time=plan.start_time, end_time=plan.end_time,
                duration=plan.end_time-plan.start_time, trajectory_hash=plan.trajectory_hash)


@contextmanager
def measured(report, name):
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        yield
    finally:
        report["timings"][name] = dict(wall_s=time.perf_counter()-wall,
                                     cpu_s=time.process_time()-cpu)


class Deadline:
    def __init__(self, seconds):
        self.end = time.perf_counter()+seconds

    def check(self):
        if time.perf_counter() >= self.end:
            raise ProbeDeadline("probe wall limit reached")


class TracedReferences(ReferenceProvider):
    def __init__(self, rows, deadline):
        super().__init__()
        self.rows, self.deadline = rows, deadline

    def generate(self, snapshot, a, node, target, start, duration, budget):
        self.deadline.check()
        row = dict(aircraft_id=a.id, source_ring=node.ring, source_slot=node.slot,
                   source_time=node.time, source_version=node.version,
                   target_ring=None if node.ring == "inner" else
                       {"outer": "middle", "middle": "inner"}[node.ring],
                   target_slot=None if node.ring == "inner" else target,
                   pad_id=target if node.ring == "inner" else None,
                   start=start, duration=duration, attempts_before=dict(budget.attempts),
                   aircraft_attempts_before=budget.attempts[a.id],
                   total_attempts_before=sum(budget.attempts.values()),
                   truncations_before=sorted(budget.truncations),
                   template_cache_hit=(snapshot.config.config_hash, a.id, node.ring,
                       node.slot, node.version, target, start, duration) in self.templates,
                   returned=False, returned_plan=False)
        self.rows.append(row)
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            ok, plan = super().generate(snapshot, a, node, target, start, duration, budget)
            row.update(returned=True, generation_allowed=ok, returned_plan=plan is not None,
                       plan=plan_row(plan) if plan is not None else None,
                       budget_truncated=not ok)
            return ok, plan
        finally:
            row.update(attempts_after=dict(budget.attempts),
                       aircraft_attempts_after=budget.attempts[a.id],
                       total_attempts_after=sum(budget.attempts.values()),
                       truncations_after=sorted(budget.truncations),
                       wall_s=time.perf_counter()-wall, cpu_s=time.process_time()-cpu)


class TracedContext(EvaluationContext):
    def __init__(self, snapshot, rows, deadline, budget):
        super().__init__(snapshot, budget=budget)
        self.rows, self.deadline = rows, deadline

    def check_single(self, column):
        self.deadline.check()
        row = dict(aircraft_id=column.aircraft_id, kind=column.kind,
                   chain=[plan_row(p) for p in column.plans], returned=False)
        self.rows.append(row)
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            result = super().check_single(column)
            row.update(returned=True, **result_row(result))
            return result
        finally:
            row.update(wall_s=time.perf_counter()-wall, cpu_s=time.process_time()-cpu)


def reference_row(reference):
    if reference is None:
        return None
    return dict(status=reference.status, source=reference.source,
                soc_required=reference.soc_required, peak_required=reference.peak_required,
                duration=reference.duration, energy_j=reference.energy_j,
                chain=[plan_row(p) for p in reference.plans])


def load_checkpoint_read_only(path):
    """Decode the world only; never restore, open, copy, or truncate its sink."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    envelope = json.loads(raw)
    payload = envelope["payload"]
    if payload["schema"] != campaign.SCHEMA or envelope["sha256"] != campaign.digest(payload):
        raise campaign.CheckpointError("Stage2 checkpoint schema/content mismatch")
    root = path.parent.parent
    metadata = campaign.load_json(root/"campaign.json")
    provenance = campaign.load_json(root/"provenance.json")
    if (campaign.digest(provenance["source_files"]) != provenance["source_hash"]
            or provenance["contract_sha256"] != sha256((campaign.CONTRACT_DIR/"rotating_slots_stage2_baseline_contract_v1_0.md").read_bytes()).hexdigest()
            or provenance["overlay_sha256"] != sha256((campaign.CONTRACT_DIR/"stage2_baseline_config_v1_0.yaml").read_bytes()).hexdigest()):
        raise campaign.CheckpointError("Stage2 frozen provenance/contract/overlay mismatch")
    manifest = campaign.load_json(root/"manifest.json")
    expected = campaign.checkpoint_identity(provenance, campaign.digest(manifest))
    if payload["identity"] != expected or metadata["identity"] != expected:
        raise campaign.CheckpointError("Stage2 checkpoint identity mismatch")
    state = campaign.Stage2Codec().decode(payload["world"])
    fields = campaign.WORLD_FIELDS | campaign.CACHE_FIELDS | campaign.OPTIONAL_RUNTIME_FIELDS | campaign.STAGE2_FIELDS
    if set(state) != fields or state["experiment_policy_id"] != STAGE2_POLICY_ID:
        raise campaign.CheckpointError("Stage2 checkpoint world inventory/policy mismatch")
    config = load_config()
    if (state["config"].config_hash != expected["base_config_hash"]
            or expected["base_config_hash"] != config.config_hash
            or state["config"].raw_hash != config.raw_hash
            or provenance["base_config_bytes_sha256"] != config.raw_hash
            or state["config"].profile_hash != config.profile_hash
            or state["time"] != payload["time"]):
        raise campaign.CheckpointError("Stage2 checkpoint configuration/time mismatch")
    if campaign.digest(state["manifest"].to_dict()) != expected["manifest_sha256"]:
        raise campaign.CheckpointError("Stage2 checkpoint manifest mismatch")
    if any(r.release_time > state["time"] for r in state["requests"].values()):
        raise campaign.CheckpointError("checkpoint revealed request is in the future")
    world = object.__new__(RotatingSlotWorld)
    world.__dict__.update(state)
    world.state_sink = world._scheduler_port = None
    world.admission_callback = world.decision_callback = None
    current_code = [Path(__file__), *(ROOT/"terminal_marl/core/rotating_slots"/name for name in
                    ("config.py", "world.py", "stage2_planning.py", "trajectory.py", "energy.py"))]
    return world, dict(path=str(path), bytes_sha256=sha256(raw).hexdigest(),
        schema=payload["schema"], identity=expected, time=payload["time"],
        validation="schema/digest/provenance/identity/config/manifest/inventory/revealed requests",
        old_run_revision=provenance["revision"],
        current_diagnostic_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        current_worktree_code_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in current_code},
        diagnostic_scope="read-only snapshot with current worktree; not exact old-code restore or continuation",
        baseline_semantic_raw_profile_match=True,
        sink_metadata_ignored=True, state_sink=None, scheduler_state_not_loaded=True)


def collect_catalog(snapshot, report, deadline):
    data = report["catalog"] = dict(status="running", generate_calls=[], single_checks=[], aircraft={})
    budget = PlanningBudget()  # Normative 3 s / 64 per aircraft / 2048 global.
    data["budget"] = {k: getattr(budget, k) for k in
                      ("candidate_seconds", "max_per_aircraft", "max_attempts", "soft_seconds", "watchdog_seconds")}
    references = TracedReferences(data["generate_calls"], deadline)
    context = TracedContext(snapshot, data["single_checks"], deadline, budget)
    provider = CandidateProvider(reference_provider=references)
    try:
        catalog = provider.get_columns(snapshot, budget, context)
        for aircraft_id, columns in catalog.columns.items():
            deadline.check()
            a = snapshot.aircraft[aircraft_id]
            moving = [c for c in columns if c.kind == "MOVE"]
            complete = [c for c in moving if c.plans and c.plans[-1].pad_id is not None]
            targets = Counter(json.dumps(c.first_target) for c in moving)
            data["aircraft"][aircraft_id] = dict(
                source_ring=a.ring, source_slot=a.slot, status=a.status,
                move_count=len(moving), complete_move_count=len(complete),
                partial_move_count=len(moving)-len(complete),
                column_kind_counts=dict(Counter(c.kind for c in columns)),
                first_target_distribution=dict(targets),
                attempts=catalog.attempts.get(aircraft_id, 0),
                incomplete=aircraft_id in catalog.incomplete,
                references=[reference_row(c.reference) for c in columns],
                columns=[dict(kind=c.kind, first_target=c.first_target,
                              complete_chain=bool(c.kind == "MOVE" and c.plans and c.plans[-1].pad_id is not None),
                              chain=[plan_row(p) for p in c.plans],
                              objective=c.objective, reference=reference_row(c.reference),
                              first_start_le_now_plus_5=bool(c.plans and c.plans[0].start_time <= snapshot.time+5.+1.e-8),
                              prefix_eligible=bool(context.prefix((c,)))) for c in columns])
            data["aircraft"][aircraft_id]["single_rejection_reasons"] = dict(Counter(
                r.get("reason", "UNRETURNED") for r in data["single_checks"]
                if r["aircraft_id"] == aircraft_id and not r.get("accepted", False)))
        data.update(status="completed", incomplete=list(catalog.incomplete), cache_hits=catalog.cache_hits)
    finally:
        data.update(attempts=dict(budget.attempts), truncations=sorted(budget.truncations),
                    budget_elapsed_wall_s=budget.elapsed, budget_timings=dict(budget.timings))


def resolve_aircraft(snapshot, name):
    return name if name in snapshot.aircraft else "initial_"+name


def auto_witnesses(snapshot, report, count):
    catalog = (report.get("catalog") or {}).get("aircraft", {})
    eligible = [a for a in snapshot.aircraft.values() if a.status == "HOLD_OUTER"
                and not (a.emergency.eligible or a.emergency.declared or a.emergency.pending)
                and a.id not in snapshot.active and a.id not in snapshot.soft]
    eligible.sort(key=lambda a: (catalog.get(a.id, {}).get("move_count", 0) != 0, a.id))
    selected = eligible[:count]
    report["witness_selection"] = dict(requested_count=count, eligible_count=len(eligible),
        rule="nonemergency HOLD_OUTER without fixed plan; catalog no-MOVE first, then aircraft ID",
        selected=[dict(aircraft_id=a.id, catalog_move_count=catalog.get(a.id, {}).get("move_count"),
                       basis="catalog_no_MOVE" if catalog.get(a.id, {}).get("move_count") == 0 else
                             "catalog_has_MOVE" if a.id in catalog else "catalog_unrecorded") for a in selected])
    return [a.id for a in selected]


def collect_witnesses(snapshot, aircraft_names, report, deadline, *, first_accepted=False, screen=False):
    data = report["witnesses"] = dict(status="running", separate_from_catalog_budget=True,
        purpose="diagnostic physical witnesses only; never algorithm candidates", aircraft=[])
    for name in aircraft_names:
        deadline.check()
        aircraft_id = resolve_aircraft(snapshot, name)
        a = snapshot.aircraft.get(aircraft_id)
        row = dict(requested_aircraft=name, aircraft_id=aircraft_id, targets=[], status="running")
        data["aircraft"].append(row)
        if a is None or _node(snapshot, a) is None:
            row.update(status="unavailable", reason="aircraft absent or no normal planning node")
            continue
        node = _node(snapshot, a)
        if first_accepted and (node.time != snapshot.time or a.id in snapshot.soft or a.id in snapshot.active):
            row.update(status="unavailable", reason="immediate witness requires current unfixed hold")
            continue
        targets = range(6 if node.ring == "inner" else
                        ring_config({"outer": "middle", "middle": "inner"}[node.ring], snapshot.config).slot_count)
        durations = range(20, 301 if node.ring == "inner" else 181, 5)
        row.update(source_ring=node.ring, source_slot=node.slot, start_time=node.time,
                   duration_domain=list(durations), first_accepted_stop=first_accepted,
                   screened_rejections=0, accepted=False)
        table = None
        if screen:
            deadline.check()
            wall, cpu = time.perf_counter(), time.process_time()
            table, hit = screen_normal_templates(node.ring, node.slot, node.time, snapshot.config)
            row["geometry_screen"] = dict(wall_s=time.perf_counter()-wall,
                cpu_s=time.process_time()-cpu, cache_hit=hit, rejection_only=True)
        for target in targets:
            deadline.check()
            target_row = dict(target=target, motion_feasible=False, attempts=[],
                              prefix_check=None, status="running")
            row["targets"].append(target_row)
            for duration in durations:
                deadline.check()
                if table is not None and table[(target, duration)] is not None:
                    row["screened_rejections"] += 1
                    continue
                wall, cpu = time.perf_counter(), time.process_time()
                generator = generate_final if node.ring == "inner" else generate_transfer
                result = generator(a.id, node.ring, node.slot, target, node.time,
                    source_version=node.version, config=snapshot.config, durations=(duration,))
                target_row["attempts"].append(dict(duration=duration, **result_row(result),
                    wall_s=time.perf_counter()-wall, cpu_s=time.process_time()-cpu))
                if result.accepted:
                    target_row.update(motion_feasible=True, plan=plan_row(result.plan))
                    deadline.check()
                    wall, cpu = time.perf_counter(), time.process_time()
                    checked = evaluate_planning_snapshot(snapshot, (result.plan,))
                    target_row["prefix_check"] = dict(**result_row(checked),
                        refinement_required=getattr(checked, "refinement_required", False),
                        wall_s=time.perf_counter()-wall, cpu_s=time.process_time()-cpu)
                    if checked.accepted:
                        row.update(accepted=True, accepted_plan=plan_row(result.plan))
                    if checked.accepted or not first_accepted:
                        break
            target_row["status"] = "completed"
            if first_accepted and row["accepted"]:
                break
        row["status"] = "completed"
    data["status"] = "completed"


def run_probe(*, catalog_only=False, witness_only=False, witness_aircraft=None,
              isolated_aircraft=None, max_wall_seconds=90., checkpoint=None,
              auto_witness_count=0, first_accepted=False, screen_witnesses=False):
    if not 0 < max_wall_seconds <= 90:
        raise ValueError("max wall seconds must be in (0,90]")
    deadline = Deadline(max_wall_seconds)
    report = dict(schema="stage2_normal_action_probe_v1", status="running", partial=False,
        read_only=True, checkpoint_supported=True, max_wall_seconds=max_wall_seconds,
        deadline_mode="signal_and_cooperative" if hasattr(signal, "setitimer") else "cooperative",
        timings={}, catalog=None, witnesses=None,
        nonfinite_json_representation="NaN/+Infinity/-Infinity strings")
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    old_handler = None
    if hasattr(signal, "setitimer"):
        def alarm(_signum, _frame):
            raise ProbeDeadline("probe wall limit reached")
        old_handler = signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, max_wall_seconds)
    world = snapshot = None
    try:
        with measured(report, "fixture_and_snapshot"):
            if checkpoint:
                world, report["checkpoint"] = load_checkpoint_read_only(checkpoint)
            else:
                world = busy_fixture(frozen_manifest())
            if isolated_aircraft:
                if checkpoint:
                    raise ValueError("checkpoint cannot be combined with isolated fixture")
                name = isolated_aircraft if isolated_aircraft in world.aircraft else "initial_"+isolated_aircraft
                a = world.aircraft[name]
                world = RotatingSlotWorld(config=world.config)
                world.seed_hold(a.id, a.ring, a.slot, a.soc, a.service_class)
            snapshot = world.build_planning_snapshot()
            report["snapshot"] = dict(time=snapshot.time, version=snapshot.version,
                config_hash=snapshot.config.config_hash, aircraft_count=len(snapshot.aircraft),
                statuses=dict(Counter(a.status for a in snapshot.aircraft.values())),
                isolated_aircraft=isolated_aircraft)
        if not witness_only:
            with measured(report, "catalog"):
                collect_catalog(snapshot, report, deadline)
        if not catalog_only:
            with measured(report, "witnesses"):
                names = auto_witnesses(snapshot, report, auto_witness_count) if auto_witness_count else (
                    witness_aircraft or ("outer_00", "middle_00", "inner_00"))
                collect_witnesses(snapshot, names, report, deadline,
                    first_accepted=first_accepted or bool(auto_witness_count),
                    screen=screen_witnesses or bool(auto_witness_count))
        report["status"] = "completed"
    except ProbeDeadline as exc:
        report.update(status="partial", partial=True, stopped_reason=str(exc))
    except Exception as exc:
        report.update(status="failed", partial=True, error_type=type(exc).__name__, error=str(exc))
    finally:
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
        if snapshot is not None:
            after = world.build_planning_snapshot()
            report["state_unchanged"] = after.version == snapshot.version and after.time == snapshot.time
        if report.get("checkpoint"):
            report["checkpoint_input_unchanged"] = sha256(Path(checkpoint).read_bytes()).hexdigest() == report["checkpoint"]["bytes_sha256"]
        report["timings"]["total"] = dict(wall_s=time.perf_counter()-started_wall,
                                               cpu_s=time.process_time()-started_cpu)
    return json_value(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--catalog-only", action="store_true")
    mode.add_argument("--witness-only", action="store_true")
    parser.add_argument("--witness-aircraft", nargs="+")
    parser.add_argument("--auto-witness-count", type=int, default=0, help="Select up to 3 nonemergency HOLD_OUTER; no-MOVE first")
    parser.add_argument("--first-accepted", action="store_true", help="Stop at first certified immediate witness per aircraft")
    parser.add_argument("--screen-witnesses", action="store_true", help="Use original rejection-only geometry screen before generation")
    parser.add_argument("--isolated-aircraft", help="Smoke only: retain one named fixture hold in a fresh world")
    parser.add_argument("--checkpoint", type=Path, help="Read-only Stage2 checkpoint; scheduler and sink metadata are never restored")
    parser.add_argument("--max-wall-seconds", type=float, default=90.)
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/rotating_slots_stage2_20261021/transfer_diagnosis/probe.json")
    args = parser.parse_args(argv)
    if args.checkpoint and (not args.checkpoint.is_file() or args.isolated_aircraft):
        parser.error("checkpoint must exist and cannot be combined with --isolated-aircraft")
    if not 0 <= args.auto_witness_count <= 3:
        parser.error("--auto-witness-count must be within [0,3]")
    if args.auto_witness_count and (args.catalog_only or args.witness_aircraft):
        parser.error("automatic witnesses cannot combine with --catalog-only or explicit --witness-aircraft")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if not 0 < args.max_wall_seconds <= 90:
        parser.error("--max-wall-seconds must be in (0,90]")
    report = run_probe(catalog_only=args.catalog_only, witness_only=args.witness_only,
        witness_aircraft=args.witness_aircraft, isolated_aircraft=args.isolated_aircraft,
        max_wall_seconds=args.max_wall_seconds, checkpoint=args.checkpoint,
        auto_witness_count=args.auto_witness_count, first_accepted=args.first_accepted,
        screen_witnesses=args.screen_witnesses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), status=report["status"], partial=report["partial"],
        timings=report["timings"], state_unchanged=report.get("state_unchanged")), allow_nan=False))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
