"""Read-only Stage2 catalog trace and separately budgeted physical witnesses.

No advance, submit, commit, refinement, or checkpoint stream restoration occurs.
Witness enumeration is diagnostic only; it does not feed CandidateProvider.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
import json
import math
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from terminal_marl.core.rotating_slots.geometry import ring_config
from terminal_marl.core.rotating_slots.stage2_campaign import busy_fixture, frozen_manifest
from terminal_marl.core.rotating_slots.stage2_planning import (
    CandidateProvider, EvaluationContext, PlanningBudget, ReferenceProvider, _node,
)
from terminal_marl.core.rotating_slots.trajectory import generate_final, generate_transfer
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, evaluate_planning_snapshot


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
            targets = Counter(json.dumps(c.first_target) for c in moving)
            data["aircraft"][aircraft_id] = dict(
                source_ring=a.ring, source_slot=a.slot, status=a.status,
                move_count=len(moving), column_kind_counts=dict(Counter(c.kind for c in columns)),
                first_target_distribution=dict(targets),
                attempts=catalog.attempts.get(aircraft_id, 0),
                incomplete=aircraft_id in catalog.incomplete,
                references=[reference_row(c.reference) for c in columns],
                columns=[dict(kind=c.kind, chain=[plan_row(p) for p in c.plans],
                              objective=c.objective, reference=reference_row(c.reference),
                              first_start_le_now_plus_5=bool(c.plans and c.plans[0].start_time <= snapshot.time+5.+1.e-8),
                              prefix_eligible=bool(context.prefix((c,)))) for c in columns])
        data.update(status="completed", incomplete=list(catalog.incomplete), cache_hits=catalog.cache_hits)
    finally:
        data.update(attempts=dict(budget.attempts), truncations=sorted(budget.truncations),
                    budget_elapsed_wall_s=budget.elapsed, budget_timings=dict(budget.timings))


def resolve_aircraft(snapshot, name):
    return name if name in snapshot.aircraft else "initial_"+name


def collect_witnesses(snapshot, aircraft_names, report, deadline):
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
        targets = range(6 if node.ring == "inner" else
                        ring_config({"outer": "middle", "middle": "inner"}[node.ring], snapshot.config).slot_count)
        durations = range(20, 301 if node.ring == "inner" else 181, 5)
        row.update(source_ring=node.ring, source_slot=node.slot, start_time=node.time,
                   duration_domain=list(durations))
        for target in targets:
            deadline.check()
            target_row = dict(target=target, motion_feasible=False, attempts=[],
                              prefix_check=None, status="running")
            row["targets"].append(target_row)
            for duration in durations:
                deadline.check()
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
                    break
            target_row["status"] = "completed"
        row["status"] = "completed"
    data["status"] = "completed"


def run_probe(*, catalog_only=False, witness_only=False, witness_aircraft=None,
              isolated_aircraft=None, max_wall_seconds=90.):
    if not 0 < max_wall_seconds <= 90:
        raise ValueError("max wall seconds must be in (0,90]")
    deadline = Deadline(max_wall_seconds)
    report = dict(schema="stage2_normal_action_probe_v1", status="running", partial=False,
        read_only=True, checkpoint_supported=False, max_wall_seconds=max_wall_seconds,
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
            world = busy_fixture(frozen_manifest())
            if isolated_aircraft:
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
                collect_witnesses(snapshot, witness_aircraft or ("outer_00", "middle_00", "inner_00"), report, deadline)
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
        report["timings"]["total"] = dict(wall_s=time.perf_counter()-started_wall,
                                               cpu_s=time.process_time()-started_cpu)
    return json_value(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--catalog-only", action="store_true")
    mode.add_argument("--witness-only", action="store_true")
    parser.add_argument("--witness-aircraft", nargs="+", default=["outer_00", "middle_00", "inner_00"])
    parser.add_argument("--isolated-aircraft", help="Smoke only: retain one named fixture hold in a fresh world")
    parser.add_argument("--checkpoint", type=Path, help="Not implemented: refused without reading or writing checkpoint streams")
    parser.add_argument("--max-wall-seconds", type=float, default=90.)
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/rotating_slots_stage2_20261021/transfer_diagnosis/probe.json")
    args = parser.parse_args(argv)
    if args.checkpoint:
        parser.error("--checkpoint is not implemented; this probe supports frozen t0 only")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if not 0 < args.max_wall_seconds <= 90:
        parser.error("--max-wall-seconds must be in (0,90]")
    report = run_probe(catalog_only=args.catalog_only, witness_only=args.witness_only,
        witness_aircraft=args.witness_aircraft, isolated_aircraft=args.isolated_aircraft,
        max_wall_seconds=args.max_wall_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), status=report["status"], partial=report["partial"],
        timings=report["timings"], state_unchanged=report.get("state_unchanged")), allow_nan=False))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
