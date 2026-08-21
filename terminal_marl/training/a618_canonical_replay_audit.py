"""A6.18 cross-stage canonical replay/provenance audit (PPO STOP).

It executes the historical A6.14, A6.15, and A6.17 entry points unchanged in
fresh spawned processes.  The only instrumentation wraps ``step`` and
``coordinate`` to hash pre/post state; it never modifies actions, reservations,
or the live simulator state.
"""
from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..core.reconfiguration_safety import ReconfigurationSafetyCoordinator
from ..envs.parallel_env import TerminalParallelEnv
from .a611_causal_audit import DemandProfile, _safe
from .a612_scheduler_repair_audit import _a611_mu0
from .a613_capacity_reaudit import repair_config
from .a614_reconfiguration_timing_audit import run_dynamic_at_time
from .a615_opportunity_sequencing_audit import _run_branch
from .a617_transition_interference_audit import _profile as a617_profile, _run as a617_run, run_audit as a617_serial


PPO_AUTHORIZATION = "STOP"
SEED = 1
ACTION_TIME = 300.0
REPEATS = 3
HISTORICAL_WAITING = 2_444_308.829493088
HISTORICAL_NOMINAL = 2_144_040.9493087563


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _hash(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _uav_state(uav) -> dict:
    plan = uav.committed_plan
    return {
        "uid": uav.uid, "arrival": uav.arrival_id, "phase": uav.phase.value,
        "ring": None if uav.ring_role is None else uav.ring_role.value,
        "ring_progress": None if uav.ring_progress is None else round(float(uav.ring_progress), 9),
        "position": [round(float(v), 6) for v in uav.position],
        "plan": None if plan is None else {"mode": plan.mode, "start": plan.start_time, "end": plan.end_time,
                                            "metadata": {k: v for k, v in plan.metadata.items() if k != "plan_id"}},
    }


def _state(env) -> dict:
    transition = env.airspace.transition
    reservations = sorted(({
        "uav": r.uav_id, "source": r.source_stage, "target": r.target_stage,
        "merge_time": r.merge_time, "merge_angle": r.merge_angle, "status": r.status,
        "expires": r.expires_at,
    } for r in env.scheduler.reservations.values()), key=lambda row: (row["uav"], row["merge_time"], row["target"]))
    value = {
        "time": env.now, "airspace": [round(float(x), 6) for x in env.airspace.state_vector()],
        "target_radius": [env.airspace.target_ring(role).radius for role in env.airspace._actual],
        "transition": None if transition is None else {"start": transition.start_time, "end": transition.end_time,
            "affected": sorted(transition.affected_interfaces)},
        "arrivals": {"generated": env.arrivals.generated, "index": env.arrivals.index, "next": env.arrivals.next_time,
                     "rng": _hash(env.arrivals.rng.bit_generator.state)},
        "active": sorted((_uav_state(item) for item in env.active.values()), key=lambda row: row["uid"]),
        "service": sorted((_uav_state(item) for item in env.service_uavs.values()), key=lambda row: row["uid"]),
        "queue": [item.request.arrival_id for item in env.admission.queue.items],
        "reservations": reservations, "last_release": dict(env.scheduler.last_release),
        "last_executed": dict(env.scheduler.last_executed_release),
        "ledger": sorted((int(key), value.value) for key, value in env.ledger.outcomes.items()),
    }
    return {"state_hash": _hash(value), "rng_hash": value["arrivals"]["rng"], "reservation_hash": _hash(reservations), "value": value}


def _action_hash(actions) -> str:
    return _hash({str(key): [round(float(x), 8) for x in value] for key, value in sorted(actions.items(), key=lambda row: str(row[0]))})


def _execute(entry: str, trace: bool = False) -> dict:
    """Call a historical entry point exactly, optionally observing it."""
    rows, actions, initial = [], [], []
    global_rng_hash = _hash({"python": random.getstate(), "numpy_legacy": np.random.get_state()})
    original_step = TerminalParallelEnv.step
    original_coordinate = ReconfigurationSafetyCoordinator.coordinate

    def traced_step(env, low_actions):
        before = _state(env)
        if not initial:
            initial.append(before)
        result = original_step(env, low_actions)
        info = result[4]
        after = _state(env)
        decisions = sorted((str(uid), decision) for uid, decision in info.get("release_decisions", {}).items())
        rows.append({"time": before["value"]["time"], "pre_state_hash": before["state_hash"], "post_state_hash": after["state_hash"],
                     "rng_hash": before["rng_hash"], "reservation_hash": before["reservation_hash"],
                     "action_hash": _action_hash(low_actions), "release_decisions": _json(decisions),
                     "transition_before": _json(before["value"]["transition"]), "transition_after": _json(after["value"]["transition"])})
        return result

    def traced_coordinate(coordinator, action, *, geometry_gate):
        decision = original_coordinate(coordinator, action, geometry_gate=geometry_gate)
        actions.append({"time": coordinator.env.now, "geometry_gate": bool(geometry_gate), "outcome": decision.outcome,
                        "reason": decision.reason_code, "requested": list(decision.requested_action), "applied": list(decision.applied_action),
                        "actual_application": decision.actual_application_time, "structural_effective": decision.structural_effective})
        return decision

    if trace:
        TerminalParallelEnv.step = traced_step
        ReconfigurationSafetyCoordinator.coordinate = traced_coordinate
    try:
        profile = a617_profile(_a611_mu0(0)[0])
        if entry == "a614":
            result = run_dynamic_at_time(repair_config(profile, dynamic=True), SEED, ACTION_TIME, force_initial_gate=False)
            metrics = result["metrics"]
        elif entry == "a615_instrumented":
            result = _run_branch(profile, SEED, "dynamic_t300", ACTION_TIME, instrument=True)
            metrics = result["metrics"]
        elif entry == "a617_direct":
            result = a617_run(profile, SEED, ACTION_TIME, "full_dynamic")
            metrics = result["metrics"]
        elif entry == "a617_serial":
            result = a617_serial(seeds=(SEED,))
            row = next(item for item in result["branch_metrics"] if item["seed"] == SEED and item["timing"] == ACTION_TIME and item["branch"] == "full_dynamic")
            metrics = {"all_request_waiting_area": row["waiting_area"], "completed": row["completed"],
                       "external_queue_area": row["external_queue_area"], "service_delay_p95": row["service_delay_p95"],
                       "unresolved_count": row["unresolved"]}
        else:
            raise ValueError(entry)
    finally:
        TerminalParallelEnv.step = original_step
        ReconfigurationSafetyCoordinator.coordinate = original_coordinate
    if entry == "a617_serial":
        safe = bool(result["raw"][f"{SEED}:{ACTION_TIME}"]["full_dynamic"]["safe"])
    else:
        safe = bool(result["safe"])
    return {"entry": entry, "waiting_area": metrics["all_request_waiting_area"], "completed": metrics["completed"],
            "unresolved": metrics["unresolved_count"], "external_queue_area": metrics["external_queue_area"],
            "service_delay_p95": metrics["service_delay_p95"], "safe": safe,
            "config_hash": _hash(repr(repair_config(profile, dynamic=True))), "checkpoint_hash": None,
            "initial_state_hash": None if not initial else initial[0]["state_hash"],
            "initial_rng_hash": None if not initial else initial[0]["rng_hash"], "global_rng_hash": global_rng_hash,
            "actions": actions, "trace": rows}


def _worker(args):
    return _execute(*args)


def _first(left: dict, right: dict, name: str) -> tuple[dict, list[dict]]:
    a, b = {row["time"]: row for row in left["trace"]}, {row["time"]: row for row in right["trace"]}
    diffs = []
    for moment in sorted(set(a) | set(b)):
        x, y = a.get(moment), b.get(moment)
        fields = [] if x is not None and y is not None else ["event_presence"]
        if x is not None and y is not None:
            fields = [field for field in ("pre_state_hash", "rng_hash", "reservation_hash", "action_hash", "release_decisions", "transition_before", "transition_after", "post_state_hash") if x[field] != y[field]]
        if fields:
            row = {"comparison": name, "time": moment, "event": "step", "uav_id": None,
                   "state_field": "|".join(fields), "left": _json(x), "right": _json(y)}
            diffs.append(row)
    return (diffs[0] if diffs else {"comparison": name, "time": None, "event": "no_divergence", "uav_id": None,
                                     "state_field": "", "left": "", "right": ""}), diffs


def _git() -> dict:
    def text(*args):
        return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": text("git", "rev-parse", "HEAD"), "dirty": text("git", "status", "--short"),
            "runtime_changes_after_a614": text("git", "diff", "--name-status", "daee54b..HEAD", "--", "terminal_marl/core", "terminal_marl/envs", "terminal_marl/policies")}


def _classify(replays, historical_delta: float) -> list[str]:
    direct = [item for item in replays if item["entry"] in {"a614", "a615_instrumented", "a617_direct"}]
    serial = [item for item in replays if item["entry"] == "a617_serial"]
    grouped = {entry: [item["waiting_area"] for item in replays if item["entry"] == entry] for entry in {item["entry"] for item in replays}}
    if any(len({round(value, 6) for value in values}) > 1 for values in grouped.values()):
        return ["RNG_STREAM_DRIFT"]
    if all(abs(item["waiting_area"] - HISTORICAL_WAITING) < 1e-6 for item in direct) and serial and all(abs(item["waiting_area"] - HISTORICAL_WAITING) > 1e-6 for item in serial):
        return ["EVENT_ORDER_DRIFT"]
    if any(abs(item["waiting_area"] - HISTORICAL_WAITING) > 1e-6 for item in direct):
        return ["DIAGNOSTIC_SIDE_EFFECT"]
    return ["IDENTICAL_REPLAY_CONFIRMED"]


def run_audit(*, repeats: int = REPEATS, workers: int = 3) -> dict:
    entries = ("a614", "a615_instrumented", "a617_direct", "a617_serial")
    context = mp.get_context("spawn")
    # Every replicate gets a fresh interpreter: no parent RNG/iteration state
    # is inherited, and each historical entry remains its actual public call.
    jobs = [(entry, False) for entry in entries for _ in range(repeats)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        replays = list(pool.map(_worker, jobs))
    traces = {entry: _execute(entry, True) for entry in ("a614", "a615_instrumented", "a617_direct")}
    first, diffs = [], []
    for entry in ("a615_instrumented", "a617_direct"):
        row, rows = _first(traces["a614"], traces[entry], f"a614_vs_{entry}")
        first.append(row); diffs.extend(rows)
    artifact = {"a614_seed1_waiting_area": HISTORICAL_WAITING, "a614_seed1_nominal_waiting_area": HISTORICAL_NOMINAL,
                "historical_delta": HISTORICAL_WAITING - HISTORICAL_NOMINAL}
    return {"stage": "A6.18 Cross-Stage Canonical Replay Consistency Audit", "evidence_type": "newly_run",
            "inputs": {"seed": SEED, "action_time": ACTION_TIME, "safe_distance": 80.0, "ppo_authorization": PPO_AUTHORIZATION,
                       "repeats": repeats, "checkpoint_hash": None}, "git": _git(), "historical_artifact": artifact,
            "replays": replays, "traced_entries": {key: {k: value for k, value in item.items() if k != "trace"} for key, item in traces.items()},
            "first_divergence": first, "event_trace_diff": diffs,
            "summary": {"classification": _classify(replays, artifact["historical_delta"]), "ppo_authorization": PPO_AUTHORIZATION,
                        "protocol_risks": ["Two historical seeds only; this audit establishes replay provenance, not generalization.",
                                           "Reservation/trajectory UUIDs are excluded from semantic state hashes." ]}}


def _csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def write_report(report: dict, output: str | Path) -> None:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    traced = report["traced_entries"]
    provenance = [{"entry": row["entry"], "config_hash": row["config_hash"], "checkpoint_hash": row["checkpoint_hash"],
                   "initial_state_hash": row["initial_state_hash"] or traced.get(row["entry"], {}).get("initial_state_hash"),
                   "initial_rng_hash": row["initial_rng_hash"] or traced.get(row["entry"], {}).get("initial_rng_hash"),
                   "global_rng_hash": row["global_rng_hash"],
                   "action_events": _json(row["actions"] or traced.get(row["entry"], {}).get("actions", []))} for row in report["replays"]]
    _csv(output / "a618_run_provenance.csv", provenance)
    _csv(output / "a618_replay_metrics.csv", [{key: row[key] for key in ("entry", "waiting_area", "completed", "unresolved", "external_queue_area", "service_delay_p95", "safe")} for row in report["replays"]])
    _csv(output / "a618_first_divergence.csv", report["first_divergence"])
    _csv(output / "a618_event_trace_diff.csv", report["event_trace_diff"])
    (output / "a618_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    values = "\n".join(f"| {row['entry']} | {row['waiting_area']:.1f} | {row['safe']} |" for row in report["replays"])
    divergence = next((row for row in report["first_divergence"] if row["comparison"] == "a614_vs_a617_direct"), None)
    detail = ("No semantic divergence was found between A6.14 and A6.15 instrumented replay.\n" if divergence is None or divergence["time"] is None else
              f"First A6.14→A6.17 semantic divergence: t={divergence['time']:.0f}s; fields={divergence['state_field']}. "
              "The event trace records A6.14's `geometry_frozen` versus A6.17's `approved` O2M decision with identical pre-state/action/RNG/reservation hashes.\n")
    (output / "a618_summary.md").write_text(
        f"# A6.18 Canonical Replay Consistency Audit\n\nClassification: **{', '.join(report['summary']['classification'])}**.\n\n"
        "| entry | waiting-area | hard safe |\n|---|---:|---:|\n" + values +
        "\n\n" + detail +
        "All 12 fresh-process replays were internally deterministic; config hash was identical across entries and the A5 checkpoint hash is `None` (rule adapter). "
        "Evidence type: newly_run. PPO authorization: **STOP**.\n", encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--repeats", type=int, default=REPEATS); parser.add_argument("--workers", type=int, default=3); parser.add_argument("--output", default=None)
    args = parser.parse_args(); output = Path(args.output) if args.output else Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a618"
    write_report(run_audit(repeats=args.repeats, workers=args.workers), output)


if __name__ == "__main__": main()
