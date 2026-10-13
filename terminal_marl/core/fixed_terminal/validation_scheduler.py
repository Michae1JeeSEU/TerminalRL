"""Stage5 total-budget screening with independent exact execution checks.

Search counters are accumulated operations, never counts of unique candidates.
All closed values reuse the original observed-cohort FCFS evaluator unchanged.
"""
from dataclasses import asdict
from hashlib import sha256
from math import ceil
from time import perf_counter

from .baselines import (Statistics, observed_clone, make_catalogue, joint_choices,
                        choose_normal_batch, common_emergency_fallback, objective)
from .learning_features import extract_features
from .learning_scheduler import prediction_order
from .reference import Root, action_key, value_comparison_key
from .routes import plan_energy
from .value_analysis import evaluate_root
from terminal_marl.configs.fixed_terminal_stage2 import DEFAULT_SCHEDULER_CONFIG
from terminal_marl.configs.fixed_terminal_stage3 import REFERENCE_SETTINGS


def make_pool(env, pool_limit=14):
    if not isinstance(pool_limit, int) or isinstance(pool_limit, bool) or pool_limit < 3:
        raise ValueError("pool_limit must be an integer >=3")
    started = perf_counter()
    view, stats = observed_clone(env), Statistics()
    catalogue = make_catalogue(view, REFERENCE_SETTINGS, stats)
    choices = joint_choices(view, catalogue, REFERENCE_SETTINGS, stats)
    pool, beam_order = {}, []
    validations = 0

    def add(plans, label):
        nonlocal validations
        key = action_key(plans)
        if key not in pool:
            check = view.feasibility.check_batch(plans, list(view.aircraft.values()), view.time)
            validations += 1
            stats.joint_assignments += 1
            stats.reject(check)
            if check.status != "feasible":
                raise RuntimeError("pool candidate rejected: "+check.status+":"+check.reason)
            root_id = "root-"+sha256(repr((view.time, key)).encode()).hexdigest()[:12]
            pool[key] = Root(root_id, tuple(plans), [], [], asdict(check))
        pool[key].labels.append(label)
        return pool[key]

    seen = set()
    for batch in choices:
        key = action_key(batch)
        if key in seen:
            continue
        seen.add(key)
        if len(beam_order) >= pool_limit-2:
            break
        beam_order.append(add(batch, "beam").root_id)
    hold = add((), "HOLD")
    control = choose_normal_batch("fcfs", view, DEFAULT_SCHEDULER_CONFIG)
    stats.merge(control.statistics)
    fcfs = add(control.plans, "FCFS")
    roots = sorted(pool.values(), key=lambda r: action_key(r.plans))
    return roots, {"settings": asdict(REFERENCE_SETTINGS), "pool_limit": pool_limit,
                   "beam_order": beam_order, "anchors": [hold.root_id, fcfs.root_id],
                   "statistics": asdict(stats), "pool_validation_calls": validations,
                   "generation_seconds": perf_counter()-started, "control_source": "same_state_live_FCFS",
                   "counter_scope": "accumulated_search_operations_not_unique_candidates"}


def complete_order(roots, metadata, method, predictions=None):
    by_action = [r.root_id for r in sorted(roots, key=lambda r: action_key(r.plans))]
    beam = list(metadata["beam_order"])
    beam += [root_id for root_id in by_action if root_id not in beam]
    if method in ("beam", "rolling", "full"):
        return beam
    kind = "value" if method == "value" else "rank"
    learned = prediction_order(roots, predictions, kind)
    if method != "hybrid":
        if method not in ("linear_rank", "mlp_rank", "gradient_boost", "value"):
            raise ValueError("unknown method")
        return learned
    order = []
    for pair in zip(beam, learned):
        for root_id in pair:
            if root_id not in order:
                order.append(root_id)
    return order


def shortlist(roots, metadata, method, budget, predictions=None):
    if method == "full":
        return list(roots)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("total budget must be a positive integer")
    ids = {r.root_id for r in roots}
    anchors = list(dict.fromkeys(metadata["anchors"]))
    if not set(anchors) <= ids or len(anchors) > budget:
        raise ValueError("budget cannot accommodate the unique anchors")
    selected = anchors[:]
    for root_id in complete_order(roots, metadata, method, predictions):
        if len(selected) >= min(budget, len(roots)):
            break
        if root_id not in selected:
            selected.append(root_id)
    if len(selected) != min(budget, len(roots)):
        raise ValueError("incomplete candidate order")
    # Exact evaluation and ties use one deterministic root/action order.
    return [r for r in roots if r.root_id in selected]


def _merge_rows(rows):
    total = Statistics()
    for row in rows:
        total.merge(Statistics(**row))
    return asdict(total)


def closed_evaluation(env, root):
    result = evaluate_root(env, root.plans, (env.config.horizon,), realized_future=False)
    if result["status"] != "feasible":
        raise RuntimeError("root rejected by original evaluator: "+result["reason"])
    continuation = result["continuation_decisions"]
    value = result["values"][f"{env.config.horizon:g}"]
    return {"status": "feasible", "value": value, "seconds": result["seconds"],
            "statistics": _merge_rows([result["root_emergency_statistics"]]+[r["statistics"] for r in continuation]),
            "root_emergency_statistics": result["root_emergency_statistics"],
            "continuation_decision_count": len(continuation),
            "work": {"rollouts": 1, "simulation_steps": ceil(max(0., env.config.horizon-env.time)/env.config.dt-1e-9),
                     "root_validation_calls": 2, "continuation_submission_calls": len(continuation)},
            "objective_scope": "actual_observed_cohort_J_under_FCFS_until_original_horizon"}


def rolling_evaluation(env, root, rolling_seconds):
    """Original stage2 joint continuation/surrogate, evaluated for this root once."""
    if rolling_seconds not in (20., 60.):
        raise ValueError("rolling_seconds must be20 or60")
    started = perf_counter()
    settings = DEFAULT_SCHEDULER_CONFIG
    future = observed_clone(env)
    check = future.submit_batch(root.plans)
    if check.status != "feasible":
        raise RuntimeError("rolling root rejected: "+check.reason)
    local = Statistics()
    common_emergency_fallback(future, settings, local)
    root_stats = asdict(local)
    stop = min(env.config.horizon, env.time+rolling_seconds)
    next_decision = env.time+settings.decision_interval
    steps, continuation = 0, []
    while future.time < stop-1e-9:
        future.step()
        steps += 1
        local.lookahead_states += 1
        if future.time >= next_decision-1e-9 and future.time < stop-1e-9:
            current = Statistics()
            candidates = make_catalogue(future, settings, current)
            batch = joint_choices(future, candidates, settings, current)[0]
            check = future.submit_batch(batch)
            if check.status != "feasible":
                raise RuntimeError("rolling continuation rejected: "+check.reason)
            common_emergency_fallback(future, settings, current)
            continuation.append({"time": future.time, "normal_ids": [p.aircraft_id for p in batch], "statistics": asdict(current)})
            local.merge(current)
            next_decision += settings.decision_interval
    normal = [a.plan for a in future.aircraft.values() if a.plan and not a.plan.emergency]
    failures = sum(a.state == "failed_energy" for a in future.aircraft.values())
    emergencies = sum(a.plan is not None and a.plan.emergency for a in future.aircraft.values())
    exposure = sum(max(0., min(future.time, a.completion_time if a.completion_time is not None else future.time)-a.arrival_time)
                   for a in future.aircraft.values())
    normal_value = objective(future, normal)
    energy = sum(plan_energy(a.plan, future.config) for a in future.aircraft.values() if a.plan)
    value = (-failures, normal_value[0], -emergencies, -exposure, normal_value[1], -energy)
    return {"status": "feasible", "surrogate": value, "evaluation_stop": future.time,
            "seconds": perf_counter()-started, "statistics": asdict(local), "root_emergency_statistics": root_stats,
            "continuation_decision_count": len(continuation),
            "work": {"rollouts": 1, "simulation_steps": steps, "root_validation_calls": 1,
                     "continuation_submission_calls": len(continuation)},
            "objective_scope": "stage2_commitment_surrogate; joint_continuation; not_closed_actual_J"}


def evaluate_selected(env, roots, rolling_seconds=None):
    if not roots:
        raise ValueError("no roots selected")
    evaluations = {r.root_id: closed_evaluation(env, r) if rolling_seconds is None else rolling_evaluation(env, r, rolling_seconds) for r in roots}
    def key(root):
        row = evaluations[root.root_id]
        return value_comparison_key(row["value"]) if rolling_seconds is None else tuple(row["surrogate"])
    best = max(map(key, roots))
    chosen = min((r for r in roots if key(r) == best), key=lambda r: action_key(r.plans))
    return chosen, evaluations


def apply_validation_scheduler(env, method, scorer=None, budget=5, pool_limit=14, rolling_seconds=None):
    start = perf_counter()
    if method not in ("beam", "full", "rolling", "linear_rank", "mlp_rank", "gradient_boost", "value", "hybrid"):
        raise ValueError("unknown validation scheduler")
    if (method == "rolling") != (rolling_seconds is not None):
        raise ValueError("rolling method/window mismatch")
    timing = {k: 0. for k in ("generation", "features", "inference", "exact_rollout", "final_validation_and_emergency")}
    roots, metadata = make_pool(env, pool_limit)
    timing["generation"] = perf_counter()-start
    degenerate = len(roots) == 1
    evaluations, predictions = {}, None
    if degenerate:
        selected, chosen = roots, roots[0]
    else:
        if method in ("linear_rank", "mlp_rank", "gradient_boost", "value", "hybrid"):
            kind = "value" if method == "value" else "rank"
            if scorer is None or scorer.kind != kind:
                raise ValueError("matching frozen scorer required")
            tick = perf_counter()
            feature_rows = [extract_features(env, r.plans) for r in roots]
            timing["features"] = perf_counter()-tick
            tick = perf_counter()
            predictions = scorer.predict(feature_rows)
            timing["inference"] = perf_counter()-tick
        selected = shortlist(roots, metadata, method, budget, predictions)
        tick = perf_counter()
        chosen, evaluations = evaluate_selected(env, selected, rolling_seconds)
        timing["exact_rollout"] = perf_counter()-tick
    tick = perf_counter()
    check = env.submit_batch(chosen.plans)
    if check.status != "feasible":
        raise RuntimeError("final independent batch validation: "+check.status+":"+check.reason)
    emergency_stats = Statistics()
    common_emergency_fallback(env, DEFAULT_SCHEDULER_CONFIG, emergency_stats)
    timing["final_validation_and_emergency"] = perf_counter()-tick
    combined = _merge_rows([metadata["statistics"], asdict(emergency_stats)]+[r["statistics"] for r in evaluations.values()])
    work = {name: sum(r["work"][name] for r in evaluations.values()) for name in
            ("rollouts", "simulation_steps", "root_validation_calls", "continuation_submission_calls")}
    work.update({"pool_validation_calls": metadata["pool_validation_calls"], "final_validation_calls": 1,
                 "counter_scope": "accumulated_search_operations; counters_may_revisit_candidates",
                 "closed_step_count": "ceil((horizon-root_time)/dt); original_evaluator_fixed_step_loop"})
    work["root_rollout_count"] = work["rollouts"]
    work["final_submit_count"] = 1
    row = {"time": env.time, "method": method, "budget": budget, "budget_scope": "total_including_unique_anchors",
            "pool_limit": pool_limit, "pool": [r.serializable() for r in roots], "pool_metadata": metadata,
            "pool_count": len(roots), "degenerate": degenerate,
            "fast_path": "single_safe_root_no_inference_or_rollout" if degenerate else None,
            "selected_root_ids": [r.root_id for r in selected], "chosen_root_id": chosen.root_id,
            "normal_plans": [asdict(p) for p in chosen.plans], "evaluations": evaluations,
            "final_check": asdict(check), "emergency_statistics": asdict(emergency_stats), "statistics": combined,
            "work": work, "timing": timing, "rolling_seconds": rolling_seconds,
            "timing_scope": "full_generation_features_inference_rollouts_final_validation_emergency; IO_model_load_env_steps_excluded"}
    timing["total"] = perf_counter()-start
    env.decision_calls += 1
    env.decision_seconds += timing["total"]
    env.search_candidates += combined["examined"]
    return row
