"""Nested exact search with a visible-cohort normal-completion certificate.

Frozen learned scores order only the initial TOTAL B3 shortlist. Expansions
append the existing beam order, cache each exact rollout once, and preserve the
original independent submission and emergency checks. A certificate concerns
the observed cohort under FCFS continuation, not future arrivals or optimal J.
"""
from dataclasses import asdict
from math import isfinite
from numbers import Real
from time import perf_counter

from .baselines import Statistics, observed_clone, common_emergency_fallback
from .learning_features import extract_features
from .reference import action_key, value_comparison_key
from .validation_scheduler import make_pool, closed_evaluation, complete_order, shortlist
from terminal_marl.configs.fixed_terminal_stage2 import DEFAULT_SCHEDULER_CONFIG
from terminal_marl.configs.fixed_terminal_stage6 import ONLINE_ARMS, BUDGETS, POOL_LIMIT


def visible_counts(env):
    """Read counts from a clone with all unarrived identities/draws removed."""
    view = observed_clone(env)
    return {"A": len(view.aircraft),
            "F0": sum(a.state == "failed_energy" for a in view.aircraft.values()),
            "E0": sum(a.state == "emergency_landed" for a in view.aircraft.values())}


def completion_certificate(value, counts):
    """Actual closed outcomes reaching the visible normal-completion upper bound.

    Historical failures/emergencies remain in the denominator. A still-active
    emergency commitment is deliberately not E0, making this test conservative.
    """
    names = {"F": "failed_energy", "N": "normal_landed", "E": "emergency_landed", "C": "arrived_unresolved"}
    outcomes = {key: value["components"][field] for key, field in names.items()}
    for number in list(counts.values())+list(outcomes.values())+[value["arrived_request_count"]]:
        if not isinstance(number, int) or isinstance(number, bool) or number < 0:
            raise ValueError("certificate counts must be nonnegative integers")
    a, f0, e0 = counts["A"], counts["F0"], counts["E0"]
    if f0+e0 > a or value["arrived_request_count"] != a or sum(outcomes.values()) != a:
        raise ValueError("closed certificate observed-cohort denominator mismatch")
    if outcomes["F"] < f0 or outcomes["E"] < e0:
        raise ValueError("closed certificate cannot erase historical terminal outcomes")
    passed = outcomes["F"] == f0 and outcomes["E"] == e0 and outcomes["N"] == a-f0-e0 and outcomes["C"] == 0
    return {"passed": passed, **outcomes, **counts, "normal_upper_bound": a-f0-e0,
            "new_failed": outcomes["F"]-f0, "new_emergency": outcomes["E"]-e0,
            "scope": "observed_cohort_actual_completion_under_FCFS; not_future_arrival_safety_or_full_J_optimality"}


def _checked_predictions(predictions, count):
    if isinstance(predictions, (str, bytes, dict)):
        raise ValueError("rank predictions must be a row sequence")
    try:
        rows = list(predictions)
    except TypeError as exc:
        raise ValueError("rank predictions must be a row sequence") from exc
    if len(rows) != count:
        raise ValueError("prediction count does not match pool")
    result = []
    for row in rows:
        if isinstance(row, (str, bytes, dict)):
            raise ValueError("each rank prediction must contain one finite number")
        try:
            values = list(row)
        except TypeError as exc:
            raise ValueError("each rank prediction must contain one finite number") from exc
        if (len(values) != 1 or not isinstance(values[0], Real) or isinstance(values[0], bool)
                or not isfinite(values[0])):
            raise ValueError("each rank prediction must contain one finite number")
        result.append([float(values[0])])
    return result


def _best(roots, evaluations):
    best = max(value_comparison_key(evaluations[r.root_id]["value"]) for r in roots)
    return min((r for r in roots if value_comparison_key(evaluations[r.root_id]["value"]) == best),
               key=lambda r: action_key(r.plans))


def apply_assisted_scheduler(env, arm_name, scorer=None, pool_limit=POOL_LIMIT):
    start = perf_counter()
    if not isinstance(arm_name, str):
        raise ValueError("arm_name must name a configured stage6 arm")
    arm = next((a for a in ONLINE_ARMS if a["name"] == arm_name), None)
    if arm is None:
        raise ValueError("unknown stage6 arm")
    if not isinstance(pool_limit, int) or isinstance(pool_limit, bool) or pool_limit < 3:
        raise ValueError("pool_limit must be an integer >=3")
    if arm["method"] != "full" and (type(arm["budget"]) is not int or arm["budget"] not in BUDGETS):
        raise ValueError("configured budget must be one of stage6 BUDGETS")
    timing = {k: 0. for k in ("generation", "features", "inference", "exact_rollout", "final_validation_and_emergency")}
    view = observed_clone(env)
    counts = visible_counts(view)
    roots, metadata = make_pool(view, pool_limit)
    timing["generation"] = perf_counter()-start
    degenerate = len(roots) == 1
    evaluations, stages = {}, []
    predictions = None
    certificate = None
    if degenerate:
        selected, chosen = roots, roots[0]
        stop_reason = "single_root"
    else:
        if arm["model_name"] is not None:
            if scorer is None or getattr(scorer, "kind", None) != "rank" or not callable(getattr(scorer, "predict", None)):
                raise ValueError("matching frozen rank scorer required")
            tick = perf_counter()
            features = [extract_features(view, r.plans) for r in roots]
            timing["features"] = perf_counter()-tick
            tick = perf_counter()
            predictions = _checked_predictions(scorer.predict(features), len(roots))
            timing["inference"] = perf_counter()-tick
        selected = shortlist(roots, metadata, arm["method"], arm["budget"], predictions)
        beam = complete_order(roots, metadata, "beam")
        targets = list(dict.fromkeys(min(b, len(roots)) for b in (*BUDGETS, len(roots)))) if arm["adaptive"] else [len(selected)]
        for stage_index, target in enumerate(targets):
            tick = perf_counter()
            selected_ids = {r.root_id for r in selected}
            if stage_index:
                for root_id in beam:
                    if len(selected_ids) >= target:
                        break
                    selected_ids.add(root_id)
                selected = [r for r in roots if r.root_id in selected_ids]
            if len(selected) != target:
                raise ValueError("nested shortlist does not fill target budget")
            new = [r for r in selected if r.root_id not in evaluations]
            for root in new:
                evaluation = closed_evaluation(view, root)
                if evaluation["status"] != "feasible":
                    raise RuntimeError("closed root evaluation rejected: "+evaluation["status"])
                evaluations[root.root_id] = evaluation
            timing["exact_rollout"] += perf_counter()-tick
            chosen = _best(selected, evaluations)
            certificate = completion_certificate(evaluations[chosen.root_id]["value"], counts)
            stages.append({"target_budget": target, "new_root_ids": [r.root_id for r in new],
                           "cumulative_count": len(evaluations), "chosen_root_id": chosen.root_id,
                           "certificate": certificate, "stage_seconds": perf_counter()-tick,
                           "elapsed_seconds": perf_counter()-start})
            if arm["adaptive"] and certificate["passed"]:
                stop_reason = "visible_normal_upper_bound_reached"
                break
        else:
            stop_reason = "full_pool_exhausted" if len(evaluations) == len(roots) else "fixed_budget"

    tick = perf_counter()
    # submit_batch itself independently calls check_batch before any commitment.
    check = env.submit_batch(chosen.plans)
    if check.status != "feasible":
        raise RuntimeError("final independent batch validation: "+check.status+":"+check.reason)
    emergency = Statistics()
    common_emergency_fallback(env, DEFAULT_SCHEDULER_CONFIG, emergency)
    timing["final_validation_and_emergency"] = perf_counter()-tick
    combined = Statistics()
    for item in [metadata["statistics"], asdict(emergency)]+[v["statistics"] for v in evaluations.values()]:
        combined.merge(Statistics(**item))
    statistics = asdict(combined)
    work = {name: sum(v["work"][name] for v in evaluations.values()) for name in
            ("rollouts", "simulation_steps", "root_validation_calls", "continuation_submission_calls")}
    work.update({"root_rollout_count": work["rollouts"], "pool_validation_calls": metadata["pool_validation_calls"],
                 "final_validation_calls": 1, "final_submit_count": 1,
                 "counter_scope": "accumulated_search_operations; counters_may_revisit_candidates",
                 "closed_step_count": "ceil((horizon-root_time)/dt); original_evaluator_fixed_step_loop"})
    row = {"time": env.time, "method": arm["method"], "arm": arm_name,
           "budget": len(evaluations), "configured_budget": arm["budget"],
           "budget_scope": "cumulative_unique_exact_rollouts; unique_anchors_included; single_root_zero",
           "pool_limit": pool_limit, "pool_count": len(roots), "pool": [r.serializable() for r in roots],
           "pool_metadata": metadata, "degenerate": degenerate,
           "fast_path": "single_safe_root_no_inference_or_rollout" if degenerate else None,
           "selected_root_ids": [r.root_id for r in selected], "chosen_root_id": chosen.root_id,
           "normal_plans": [asdict(p) for p in chosen.plans], "evaluations": evaluations,
           "final_check": asdict(check), "emergency_statistics": asdict(emergency), "statistics": statistics,
           "work": work, "timing": timing,
           "adaptive": {"enabled": arm["adaptive"], "stages": stages, "stop_reason": stop_reason,
                        "fallback_count": max(0, len(stages)-1),
                        "quality_certified": certificate["passed"] if certificate is not None else None,
                        "visible_counts": counts},
           "timing_scope": "full_generation_features_inference_rollouts_final_validation_emergency_bookkeeping; IO_model_load_env_steps_excluded"}
    timing["total"] = perf_counter()-start
    env.decision_calls += 1
    env.decision_seconds += timing["total"]
    env.search_candidates += statistics["examined"]
    return row
