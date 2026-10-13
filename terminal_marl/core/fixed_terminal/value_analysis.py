"""Artifact replay and same-state, fixed-continuation finite-window values.

No policy training labels or optimal-Q claims. Realized-future evaluations keep
future draws only in the simulator; every scheduler call receives its observed
clone. All physical dynamics, emergency certificates, and commitments are reused.
"""
from copy import deepcopy
from dataclasses import asdict
from math import floor, isclose
from pathlib import Path
from time import perf_counter
import json
from .domain import TerminalConfig, Ring, EnergyRates, Intent
from .scenarios import Request, Scenario, generate_scenario
from .baselines import observed_clone, apply_scheduler, common_emergency_fallback, Statistics
from .routes import build_plan
from terminal_marl.configs.fixed_terminal_stage2 import DEFAULT_SCHEDULER_CONFIG
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv


def canonical_json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def config_from_dict(data):
    data = dict(data)
    data["rings"] = tuple(Ring(**ring) for ring in data["rings"])
    data["energy"] = EnergyRates(**data["energy"])
    data["normal_pad"], data["emergency_pad"] = tuple(data["normal_pad"]), tuple(data["emergency_pad"])
    return TerminalConfig(**data)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_stage2_episode(directory):
    directory = Path(directory)
    frozen = json.loads((directory/"frozen_input.json").read_text(encoding="utf-8"))
    result = json.loads((directory/"result.json").read_text(encoding="utf-8"))
    config = config_from_dict(frozen["config"])
    s = frozen["scenario"]
    generated = generate_scenario(config, s["seed"], s["load"], s["soc_profile"])
    if canonical_json(asdict(generated)) != s:
        raise ValueError("frozen scenario does not match its independent seeded generator")
    if result["config"] != frozen["config"] or result["scenario"] != s or result["scheduler"] != frozen["scheduler"]:
        raise ValueError("stage2 result/frozen input mismatch")
    requests = read_jsonl(directory/"requests.jsonl")
    if requests != result["requests"]:
        raise ValueError("stage2 request ledger mismatch")
    decisions = read_jsonl(directory/"decisions.jsonl")
    events = read_jsonl(directory/"events.jsonl")
    times = [r["time"] for r in decisions]
    cadence = frozen["scheduler_config"]["decision_interval"]
    expected = []
    t = 0.
    next_decision = 0.
    while t < config.horizon-1e-9:
        if t >= next_decision-1e-9:
            expected.append(t)
            next_decision += cadence
        t = min(config.horizon, t+config.dt)
    if times != expected or any(r["method"] != frozen["scheduler"] for r in decisions):
        raise ValueError("decision cadence or scheduler label mismatch")
    if any(e["time"] < 0 or e["time"] > config.horizon for e in events):
        raise ValueError("event outside observation interval")
    return {"directory": str(directory), "frozen": frozen, "result": result, "config": config,
            "scenario": generated, "decisions": decisions, "events": events}


def plans_from_decision(env, row):
    plans = []
    for item in row["normal_plans"]:
        plan = env.propose(Intent(item["aircraft_id"], tuple(item["wait_cycles"])))
        if not isclose(plan.end, item["completion"], rel_tol=0., abs_tol=1e-8):
            raise ValueError("normal decision plan completion mismatch")
        plans.append(plan)
    return tuple(plans)


def _replay_decision(env, row, commits, search_limit):
    normal = plans_from_decision(env, row)
    normal_events = [e for e in commits if not e["emergency"]]
    if [p.aircraft_id for p in normal] != [e["aircraft_id"] for e in normal_events]:
        raise ValueError("normal decision and commit event identities/order mismatch")
    for plan, event in zip(normal, normal_events):
        _check_commit_record(plan, event, env)
    check = env.submit_batch(normal)
    if check.status != "feasible":
        raise ValueError("recorded normal batch no longer feasible: "+check.reason)
    for event in commits:
        if not event["emergency"]:
            continue
        a = env.aircraft[event["aircraft_id"]]
        plan = build_plan(a, env.time, env.config, tuple(event["wait_cycles"]), True)
        _check_commit_record(plan, event, env)
        check = env.feasibility.check(plan, a, list(env.aircraft.values()), env.time, search_limit=search_limit)
        if check.status != "feasible":
            raise ValueError("recorded emergency lacks current public certificate: "+check.reason)
        env._commit(a, plan, event["reason"])


def _check_commit_record(plan, event, env):
    from .routes import plan_energy
    if (tuple(event["wait_cycles"]) != plan.wait_cycles or
            not isclose(plan.end, event["planned_end"], rel_tol=0., abs_tol=1e-8) or
            not isclose(plan_energy(plan, env.config)+env.config.energy.reserve,
                        event["required_soc_including_reserve"], rel_tol=0., abs_tol=1e-10) or
            canonical_json([asdict(r) for r in plan.reservations]) != event["reservations"]):
        raise ValueError("commit window/resource/energy record mismatch")


def replay_episode(directory, stop_time=None):
    """Return the state immediately BEFORE the requested time's decision.

    Replay full normal batches and independently revalidate logged emergencies;
    never call the original slow scheduler. Full replay matches every request
    ledger field, including exact depletion time, SOC, and emergency costs.
    """
    source = load_stage2_episode(directory)
    config = source["config"]
    stop = config.horizon if stop_time is None else float(stop_time)
    if stop < 0 or stop > config.horizon or (stop != config.horizon and not isclose(stop/config.dt, round(stop/config.dt), abs_tol=1e-9)):
        raise ValueError("replay stop must be an episode tick or the final cutoff")
    env = FixedTerminalEnv(config, source["scenario"])
    rows = {row["time"]: row for row in source["decisions"]}
    commits = {}
    for event in source["events"]:
        if event["event"] == "plan_committed":
            if event["time"] not in rows:
                raise ValueError("commit has no matching decision")
            commits.setdefault(event["time"], []).append(event)
    while env.time < stop-1e-9:
        if env.time in rows:
            _replay_decision(env, rows[env.time], commits.get(env.time, []),
                             source["frozen"]["scheduler_config"]["normal_search_limit"])
        env.step()
    # All observable transition records are deterministic. Compare without
    # scheduling search telemetry, which is intentionally not recomputed.
    kinds = {"admitted", "admission_rejected", "plan_committed", "failed_energy", "normal_landed", "emergency_landed"}
    expected_events = [e for e in source["events"] if e["event"] in kinds and
                       (e["time"] < stop or (e["time"] == stop and e["event"] in {"failed_energy", "normal_landed", "emergency_landed"}))]
    actual_events = [e for e in env.events if e["event"] in kinds]
    # Global event log order may put exact landing timestamps before earlier
    # timestamp events in the same tick; preserve logged sequence verbatim.
    if canonical_json(actual_events) != expected_events:
        raise ValueError("replayed transition events differ from logged episode prefix")
    if stop == config.horizon and canonical_json(env.report()["requests"]) != source["result"]["requests"]:
        raise ValueError("full replay request outcomes differ from stage2 ledger")
    return env


def visible_snapshot(env):
    view = observed_clone(env)
    return {"time": view.time, "config": asdict(view.config),
            "aircraft": [asdict(a) for a in sorted(view.aircraft.values(), key=lambda a: a.aircraft_id)],
            "committed_resources": [{"aircraft_id": a.aircraft_id, **asdict(r)} for a in view.aircraft.values()
                                    if a.plan for r in a.plan.reservations if r.end >= view.time],
            "information_scope": "already_arrived_only; no pending IDs, draws, or arrival times"}


def system_value(env):
    """Actual finite-window outcomes, unweighted lexicographic system value."""
    report = env.report()
    counts = report["summary"]["outcomes"]
    normal = counts["normal_landed"]
    failed, emergency, censored = counts["failed_energy"], counts["emergency_landed"], counts["censored"]
    exposure = report["summary"]["observed_request_exposure_seconds"]
    delay = sum(row["delay_seconds"] for row in report["requests"] if row["outcome"] == "normal_landed")
    energy = sum(row["energy_consumed_soc"] for row in report["requests"])
    components = {"failed_energy": failed, "normal_landed": normal, "emergency_landed": emergency,
                  "arrived_unresolved": censored, "observed_request_seconds": exposure,
                  "normal_delay_sum_seconds": delay, "energy_consumed_soc_all_requests": energy}
    return {"J": (-failed, normal, -emergency, -censored, -exposure, -delay, -energy),
            "components": components, "request_denominator": len(report["requests"]),
            "arrived_request_count": report["summary"]["arrived_request_count"],
            "not_arrived_count": counts["not_arrived"],
            "normal_conditional_mean_delay_seconds": report["summary"]["normal_delay_mean_seconds"],
            "normal_conditional_denominator": normal, "time": env.time,
            "evaluation_stop": env.time, "episode_cumulative": True}


def delta_value(value, baseline):
    return {"delta_J": tuple(a-b for a, b in zip(value["J"], baseline["J"])),
            "delta_components": {key: value["components"][key]-baseline["components"][key] for key in value["components"]},
            "interpretation": "finite_window_same_state_fixed_continuation_difference; not optimal or expected Q"}


def evaluate_root(env, plans, stops, *, realized_future=False):
    """Root + common emergency, then same FCFS on global 5-second cadence.

    stops are absolute simulation times, including an optional current-time
    checkpoint. This function never mutates its input environment.
    """
    rollout_started = perf_counter()
    work = deepcopy(env) if realized_future else observed_clone(env)
    work.events = []
    check = work.feasibility.check_batch(tuple(plans), list(work.aircraft.values()), work.time)
    if check.status != "feasible":
        return {"status": check.status, "reason": check.reason, "values": {}, "events": []}
    work.submit_batch(tuple(plans))
    root_stats = Statistics()
    common_emergency_fallback(work, DEFAULT_SCHEDULER_CONFIG, root_stats)
    targets = sorted(set(float(t) for t in stops))
    if not targets or any(t < work.time or t > work.config.horizon for t in targets):
        raise ValueError("rollout stop outside current-state observation interval")
    if any(t != work.config.horizon and not isclose(t/work.config.dt, round(t/work.config.dt), abs_tol=1e-9) for t in targets):
        raise ValueError("rollout stops must be simulation ticks")
    cadence = DEFAULT_SCHEDULER_CONFIG.decision_interval
    next_decision = (floor(work.time/cadence)+1)*cadence
    values, reports, continuation = {}, {}, []
    for target in targets:
        while work.time < target-1e-9:
            if work.time >= next_decision-1e-9:
                decision = apply_scheduler(work, "fcfs", DEFAULT_SCHEDULER_CONFIG)
                continuation.append({"time": work.time, "normal_ids": [p.aircraft_id for p in decision.plans],
                                     "statistics": asdict(decision.statistics)})
                next_decision += cadence
            work.step()
        values[f"{target:g}"] = system_value(work)
        values[f"{target:g}"]["root_time"] = env.time
        reports[f"{target:g}"] = work.report()
        reports[f"{target:g}"]["scheduler"] = "stage3_root_then_fixed_fcfs"
        reports[f"{target:g}"]["assumptions"]["normal_domain"] = "root_selected_candidates_jointly_committed; unselected_holders_hold; existing_commitments_fixed"
        values[f"{target:g}"]["elapsed_through_stop_seconds"] = perf_counter()-rollout_started
    return {"status": "feasible", "reason": "root_independently_validated",
            "future_scope": "ex_post_realized_original_demand_path" if realized_future else "observed_arrived_cohort_only",
            "continuation": "FCFS_with_stage2_common_emergency_rules", "values": values, "reports": reports,
            "events": work.events, "continuation_decisions": continuation,
            "seconds": perf_counter()-rollout_started,
            "checkpoint_timing_scope": "one shared trajectory; cumulative measured time including earlier checkpoints/reports",
            "root_emergency_statistics": asdict(root_stats)}


def conditional_margin(env, full_batch, aircraft_id, full_evaluation=None):
    reduced = tuple(p for p in full_batch if p.aircraft_id != aircraft_id)
    if len(reduced) == len(full_batch):
        raise ValueError("conditional marginal aircraft is not in the full batch")
    for label, batch in (("full", full_batch), ("without", reduced)):
        result = env.feasibility.check_batch(batch, list(env.aircraft.values()), env.time)
        if result.status != "feasible":
            return {"status": result.status, "reason": label+":"+result.reason,
                    "aircraft_id": aircraft_id, "marginal": None, "additivity_claim": False}
    full = full_evaluation or evaluate_root(env, full_batch, (env.config.horizon,), realized_future=True)
    without = evaluate_root(env, reduced, (env.config.horizon,), realized_future=True)
    key = f"{env.config.horizon:g}"
    return {"status": "feasible", "aircraft_id": aircraft_id,
            "marginal": delta_value(full["values"][key], without["values"][key]),
            "without_evaluation": without, "additivity_claim": False,
            "interpretation": "conditional marginal of this full compatible batch under the same realized path and FCFS continuation"}
