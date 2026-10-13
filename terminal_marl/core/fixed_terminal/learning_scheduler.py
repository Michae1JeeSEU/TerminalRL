"""Shared deterministic pool and exact final choice; learning only shortlists."""
from dataclasses import asdict
from hashlib import sha256
from time import perf_counter
from .baselines import (Statistics, observed_clone, make_catalogue, joint_choices,
                        choose_normal_batch, common_emergency_fallback, apply_scheduler)
from .reference import Root, action_key, value_comparison_key
from .learning_features import extract_features
from .value_analysis import evaluate_root
from terminal_marl.configs.fixed_terminal_stage2 import DEFAULT_SCHEDULER_CONFIG
from terminal_marl.configs.fixed_terminal_stage4 import REFERENCE_SETTINGS, POOL_KEEP, TOP_K, METHODS


def make_pool(env):
    started = perf_counter()
    view, stats = observed_clone(env), Statistics()
    catalogue = make_catalogue(view, REFERENCE_SETTINGS, stats)
    choices = joint_choices(view, catalogue, REFERENCE_SETTINGS, stats)
    pool = {}
    beam_order = []

    def add(plans, label):
        key = action_key(plans)
        if key not in pool:
            check = view.feasibility.check_batch(plans, list(view.aircraft.values()), view.time)
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
        if len(beam_order) == POOL_KEEP:
            break
        beam_order.append(add(batch, "beam").root_id)
    hold = add((), "HOLD")
    fcfs = choose_normal_batch("fcfs", view, DEFAULT_SCHEDULER_CONFIG)
    stats.merge(fcfs.statistics)
    control = add(fcfs.plans, "FCFS")
    roots = sorted(pool.values(), key=lambda r: action_key(r.plans))
    assert len(roots) <= 14
    return roots, {"settings": asdict(REFERENCE_SETTINGS), "pool_limit": 14, "beam_order": beam_order,
                   "anchors": [hold.root_id, control.root_id], "statistics": asdict(stats),
                   "generation_seconds": perf_counter()-started, "control_source": "same_state_live_FCFS"}


def prediction_order(roots, predictions, kind):
    if len(predictions) != len(roots):
        raise ValueError("prediction count does not match pool")
    def key(index):
        value = predictions[index]
        return (tuple(round(float(x)) for x in value[:4])+tuple(float(x) for x in value[4:])
                if kind == "value" else (float(value[0]),))
    indices = sorted(range(len(roots)), key=lambda i: action_key(roots[i].plans))
    return [roots[i].root_id for i in sorted(indices, key=key, reverse=True)]


def shortlist(roots, metadata, method, predictions=None):
    if method == "full_reference":
        return list(roots)
    if method == "beam_topk":
        order = metadata["beam_order"]
    elif method in ("value_topk", "rank_topk"):
        order = prediction_order(roots, predictions, method.split("_")[0])
    else:
        raise ValueError("unknown pool method")
    ids = set(order[:TOP_K]) | set(metadata["anchors"])
    selected = [r for r in roots if r.root_id in ids]
    assert len(selected) <= 5 and set(metadata["anchors"]) <= {r.root_id for r in selected}
    return selected


def select_exact(env, roots):
    evaluations = {}
    for root in roots:
        result = evaluate_root(env, root.plans, (env.config.horizon,), realized_future=False)
        if result["status"] != "feasible":
            raise RuntimeError("independent root rollout rejected: "+result["reason"])
        value = result["values"][f"{env.config.horizon:g}"]
        evaluations[root.root_id] = {"value": value, "seconds": result["seconds"], "status": result["status"]}
    best = max(value_comparison_key(evaluations[r.root_id]["value"]) for r in roots)
    chosen = min((r for r in roots if value_comparison_key(evaluations[r.root_id]["value"]) == best), key=lambda r: action_key(r.plans))
    return chosen, evaluations


def apply_learning_scheduler(env, method, scorer=None):
    if method not in METHODS:
        raise ValueError("unknown stage4 method")
    start = perf_counter()
    timing = {name: 0. for name in ("generation", "features", "inference", "exact_rollout", "final_validation_and_emergency", "legacy_scheduler")}
    if method in ("fcfs", "rolling_horizon"):
        decision = apply_scheduler(env, method)
        timing["legacy_scheduler"] = decision.seconds
        timing["total"] = perf_counter()-start
        return {"time": env.time, "method": method, "normal_plans": [asdict(p) for p in decision.plans],
                "timing": timing, "statistics": asdict(decision.statistics), "degenerate": None,
                "timing_scope": "original_stage2_total_including_final_validation_and_emergency"}
    roots, metadata = make_pool(env)
    timing["generation"] = perf_counter()-start
    degenerate = len(roots) == 1
    evaluations, feature_rows = {}, []
    if degenerate:
        selected, chosen = roots, roots[0]
    else:
        if method in ("value_topk", "rank_topk"):
            if scorer is None or scorer.kind != method.split("_")[0]:
                raise ValueError("matching frozen model required")
            tick = perf_counter()
            feature_rows = [extract_features(env, r.plans) for r in roots]
            timing["features"] = perf_counter()-tick
            tick = perf_counter()
            predictions = scorer.predict(feature_rows)
            timing["inference"] = perf_counter()-tick
        else:
            predictions = None
        selected = shortlist(roots, metadata, method, predictions)
        tick = perf_counter()
        chosen, evaluations = select_exact(env, selected)
        timing["exact_rollout"] = perf_counter()-tick
    tick = perf_counter()
    check = env.submit_batch(chosen.plans)
    if check.status != "feasible":
        raise RuntimeError("final independent batch validation: "+check.status+":"+check.reason)
    emergency_stats = Statistics()
    common_emergency_fallback(env, DEFAULT_SCHEDULER_CONFIG, emergency_stats)
    timing["final_validation_and_emergency"] = perf_counter()-tick
    timing["total"] = perf_counter()-start
    env.decision_calls += 1
    env.decision_seconds += timing["total"]
    env.search_candidates += metadata["statistics"]["examined"]+emergency_stats.examined
    return {"time": env.time, "method": method, "pool": [r.serializable() for r in roots],
            "pool_metadata": metadata, "pool_count": len(roots), "degenerate": degenerate,
            "fast_path": "single_safe_root_no_inference_or_rollout" if degenerate else None,
            "selected_root_ids": [r.root_id for r in selected], "chosen_root_id": chosen.root_id,
            "normal_plans": [asdict(p) for p in chosen.plans], "evaluations": evaluations,
            "final_check": asdict(check), "emergency_statistics": asdict(emergency_stats), "timing": timing,
            "timing_scope": "generation_includes_all_pool_feasibility; final_exact_J; IO_and_environment_step_excluded"}
