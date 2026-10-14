"""Learning acts on partial-plan parents before deterministic child validation.

The finite reference uses width128 and at most twelve final beam batches plus
HOLD/live FCFS. It is neither exhaustive nor a global optimization certificate.
All caches, meters and traces belong to one method at one decision state.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from math import isfinite
from numbers import Real
from time import perf_counter

from .baselines import (Statistics, observed_clone, make_catalogue, objective,
                        choose_normal_batch, common_emergency_fallback)
from .safety import Feasibility
from .reference import Root, action_key, value_comparison_key
from .validation_scheduler import closed_evaluation
from .assisted_scheduler import visible_counts, completion_certificate
from .search_features import extract_search_features
from terminal_marl.configs.fixed_terminal_stage2 import DEFAULT_SCHEDULER_CONFIG
from terminal_marl.configs.fixed_terminal_stage7 import (ONLINE_ARMS, REFERENCE_SETTINGS,
    REFERENCE_WIDTH, ROOT_KEEP, POOL_LIMIT, EXPAND_THRESHOLD)

SEARCH_SCHEMA = "fixed-terminal-stage7-search-v1"


class _Meter:
    def __init__(self, root_time):
        self.root_time, self.phase = root_time, "catalogue"
        self.counts = Counter()

    def phase_at(self, now):
        if self.phase == "closed_rollout":
            return "closed_root" if now == self.root_time else "continuation"
        return self.phase


class _MeteredFeasibility(Feasibility):
    """Count actual calls once at each distinct API level, including nested checks.

    check_batch invokes check per candidate internally: these are separately
    reported batch_checks and candidate_checks, never added as the same unit.
    A deepcopy keeps this decision's shared meter, with identical physics.
    """
    def __init__(self, config, meter):
        super().__init__(config)
        self.meter = meter

    def __deepcopy__(self, memo):
        result = type(self)(deepcopy(self.config, memo), self.meter)
        memo[id(self)] = result
        return result

    def check(self, plan, aircraft, fleet, now, *, check_emergency=True, search_limit=256):
        phase = self.meter.phase_at(now)
        self.meter.counts[phase+"_candidate_checks"] += 1
        return super().check(plan, aircraft, fleet, now, check_emergency=check_emergency, search_limit=search_limit)

    def check_batch(self, plans, fleet, now):
        phase = self.meter.phase_at(now)
        self.meter.counts[phase+"_batch_checks"] += 1
        return super().check_batch(plans, fleet, now)


def _probabilities(policy, rows):
    if policy is None or getattr(policy, "kind", None) != "expand" or not callable(getattr(policy, "predict_expand", None)):
        raise ValueError("matching frozen expand policy required")
    values = policy.predict_expand(rows)
    if isinstance(values, (str, bytes, dict)):
        raise ValueError("expand probabilities must be a row sequence")
    values = list(values)
    if len(values) != len(rows) or any(isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("invalid expand probabilities")
    return [float(v) for v in values]


def dynamic_width(probabilities):
    mass = sum(probabilities)
    return 3 if mass <= 3 else 5 if mass <= 5 else 8


def _root_id(env, plans):
    return "root-"+sha256(repr((env.time, action_key(plans))).encode()).hexdigest()[:12]


def _best(roots, evaluations):
    best = max(value_comparison_key(evaluations[r.root_id]["value"]) for r in roots)
    return min((r for r in roots if value_comparison_key(evaluations[r.root_id]["value"]) == best), key=lambda r: action_key(r.plans))


class _Search:
    def __init__(self, env, policy=None):
        self.env, self.policy = env, policy
        self.meter = _Meter(env.time)
        self.env.feasibility = _MeteredFeasibility(env.config, self.meter)
        self.stats = Statistics()
        self.timing = {k: 0. for k in ("generation", "features", "inference", "exact_rollout", "final_validation_and_emergency")}
        self.nodes, self.stages, self.leaf_links = [], [], {}
        self.partial_cache, self.final_cache, self.evaluations, self.all_roots = {}, {}, {}, {}
        self.counters = Counter()
        tick = perf_counter()
        self.catalogue = make_catalogue(env, REFERENCE_SETTINGS, self.stats)
        self.meter.phase = "live_fcfs"
        control = choose_normal_batch("fcfs", env, DEFAULT_SCHEDULER_CONFIG)
        self.stats.merge(control.statistics)
        self.fcfs_plans = control.plans
        self.timing["generation"] += perf_counter()-tick

    def checked(self, plans, final=False):
        key = action_key(plans)
        cache = self.final_cache if final else self.partial_cache
        kind = "final_root" if final else "partial"
        self.counters[kind+"_check_attempts"] += 1
        if key in cache:
            self.counters[kind+"_check_cache_hits"] += 1
            self.last_check_cost = {"batch_checks": 0, "candidate_checks": 0, "check_attempts": 1, "cache_hits": 1}
            return cache[key], True
        self.meter.phase = kind
        before = self.meter.counts[kind+"_candidate_checks"]
        fleet = list(self.env.aircraft.values()) if final else (
            [a for a in self.env.aircraft.values() if a.state != "holding"]+
            [self.env.aircraft[p.aircraft_id] for p in plans])
        result = self.env.feasibility.check_batch(plans, fleet, self.env.time)
        self.stats.joint_assignments += 1
        self.stats.reject(result)
        cache[key] = result
        self.last_check_cost = {"batch_checks": 1, "candidate_checks": self.meter.counts[kind+"_candidate_checks"]-before, "check_attempts": 1, "cache_hits": 0}
        return result, False

    def node(self, stage, plans, depth, parent=None, option=None, key=None, result=None, cache_hit=False, width=128, frontier_count=1):
        node_id = f"s{stage}-n{len(self.nodes)}"
        lineage = list(parent["lineage"]) if parent else []
        if parent:
            lineage.append({"holder": key, "wait_cycles": list(option.wait_cycles) if option is not None else None})
        row = {"node_id": node_id, "stage": stage, "depth": depth,
               "parent_id": parent["node_id"] if parent else None, "lineage": lineage,
               "action_signature": [list((a, list(w))) for a, w in action_key(plans)],
               "candidate_count": len(self.catalogue[list(self.catalogue)[depth]]) if depth < len(self.catalogue) else 0,
               "options_count": len(self.catalogue[list(self.catalogue)[depth]])+1 if depth < len(self.catalogue) else 0,
               "validation": asdict(result) if result else None, "check_cache_hit": cache_hit,
               "operation_counts": dict(getattr(self, "last_check_cost", {})) if result else {"batch_checks": 0, "candidate_checks": 0, "check_attempts": 0, "cache_hits": 0},
               "status": "generated" if result is None or result.status == "feasible" else "rejected",
               "selected": False, "beam_discarded": False, "learned_pruned": False,
               "expanded": False, "p_expand": None, "width": width, "frontier_count": frontier_count}
        self.nodes.append(row)
        return row

    def run(self, width, mode="beam"):
        tick = perf_counter()
        index = len(self.stages)
        first = self.node(index, (), 0, width=width)
        first.update(selected=True, status="selected")
        frontier = [((), first)]
        layers = []
        failure = None
        for depth, (key, candidates) in enumerate(self.catalogue.items()):
            parent_width = width
            parent_count = len(frontier)
            for _, parent in frontier:
                parent.update(width=parent_width, frontier_count=len(frontier))
            probabilities = None
            if mode != "beam":
                feature_tick = perf_counter()
                rows = [extract_search_features(self.env, batch, self.catalogue, depth, parent_width, len(frontier)) for batch, _ in frontier]
                self.timing["features"] += perf_counter()-feature_tick
                inference_tick = perf_counter()
                try:
                    probabilities = _probabilities(self.policy, rows)
                except Exception as exc:
                    failure = "prediction_error:"+type(exc).__name__+":"+str(exc)
                    self.timing["inference"] += perf_counter()-inference_tick
                    break
                self.timing["inference"] += perf_counter()-inference_tick
                for (_, node), features, probability in zip(frontier, rows, probabilities):
                    node.update(features=features, p_expand=probability)
                if mode in ("learned_beam_width", "hybrid"):
                    width = dynamic_width(probabilities)
            expanded = []
            before = len(self.nodes)
            for position, (batch, parent) in enumerate(frontier):
                if mode in ("learned_pruning", "hybrid") and batch and probabilities[position] < EXPAND_THRESHOLD:
                    parent.update(learned_pruned=True, status="learned_pruned")
                    self.counters["learned_pruned_nodes"] += 1
                    continue
                parent["expanded"] = True
                self.counters["expanded_nodes"] += 1
                for option in (None,)+tuple(candidates):
                    plans = batch+((option,) if option is not None else ())
                    self.counters["child_attempts"] += 1
                    checked, hit = self.checked(plans)
                    node = self.node(index, plans, depth+1, parent, option, key, checked, hit, width, len(frontier))
                    if checked.status == "feasible":
                        expanded.append((plans, node))
            expanded.sort(key=lambda item: (objective(self.env, item[0]), action_key(item[0])), reverse=True)
            frontier = expanded[:width]
            hold = next((item for item in expanded if not item[0]), None)
            if hold is not None and all(batch for batch, _ in frontier):
                frontier.append(hold)
            selected_ids = {n["node_id"] for _, n in frontier}
            for _, node in expanded:
                node.update(selected=node["node_id"] in selected_ids,
                            beam_discarded=node["node_id"] not in selected_ids,
                            frontier_count=len(frontier),
                            status="selected" if node["node_id"] in selected_ids else "beam_discarded")
            self.stats.truncated |= len(expanded) > width
            layers.append({"depth": depth, "width": width, "parent_frontier_count": parent_count,
                           "probability_mass": sum(probabilities) if probabilities is not None else None,
                           "generated_nodes": len(self.nodes)-before, "feasible_nodes": len(expanded), "retained_nodes": len(frontier)})
        valid = []
        if failure is None:
            for batch, node in frontier:
                checked, hit = self.checked(batch, final=True)
                node.update(final_validation=asdict(checked), final_check_cache_hit=hit,
                            final_validation_operation_counts=dict(getattr(self, "last_check_cost", {})))
                if checked.status == "feasible":
                    valid.append((batch, node))
            valid.sort(key=lambda item: (objective(self.env, item[0]), action_key(item[0])), reverse=True)
        roots = {}

        def add(plans, label, node=None):
            checked, _ = self.checked(plans, final=True)
            if checked.status != "feasible":
                raise RuntimeError("independent anchor/root rejected: "+checked.status+":"+checked.reason)
            rid = _root_id(self.env, plans)
            if rid not in roots:
                roots[rid] = Root(rid, tuple(plans), [], [], asdict(checked))
            if label not in roots[rid].labels:
                roots[rid].labels.append(label)
            if node is not None:
                links = self.leaf_links.setdefault(rid, [])
                if node["node_id"] not in links:
                    links.append(node["node_id"])
                roots[rid].provenance.append({"source": "partial_search_leaf", "node_id": node["node_id"], "stage": index})
            else:
                roots[rid].provenance.append({"source": "independent_anchor", "label": label, "tree_lineage": None})
            if rid not in self.all_roots:
                self.all_roots[rid] = roots[rid]
            elif self.all_roots[rid] is not roots[rid]:
                previous = self.all_roots[rid]
                for name in roots[rid].labels:
                    if name not in previous.labels:
                        previous.labels.append(name)
                for provenance in roots[rid].provenance:
                    if provenance not in previous.provenance:
                        previous.provenance.append(provenance)
                roots[rid] = previous
            return rid

        beam_order = [add(batch, "beam", node) for batch, node in valid[:ROOT_KEEP]]
        hold_node = next((node for batch, node in valid if not batch), None)
        hold = add((), "HOLD", hold_node)
        fcfs = add(self.fcfs_plans, "FCFS")
        roots = sorted(roots.values(), key=lambda root: action_key(root.plans))
        generation = perf_counter()-tick
        # Feature and inference durations are already charged separately.
        self.timing["generation"] += generation-(self.timing["features"]-getattr(self, "_last_features", 0.))-(self.timing["inference"]-getattr(self, "_last_inference", 0.))
        self._last_features, self._last_inference = self.timing["features"], self.timing["inference"]
        stage = {"stage": index, "configured_width": first["width"], "effective_final_width": width,
                 "initial_width": first["width"], "mode": mode, "layers": layers, "prediction_failure": failure,
                 "root_ids": [r.root_id for r in roots], "beam_order": beam_order, "anchors": list(dict.fromkeys([hold, fcfs])),
                 "generation_seconds": generation}
        self.stages.append(stage)
        return roots, stage

    def evaluate(self, roots):
        tick = perf_counter()
        new = []
        for root in roots:
            if root.root_id in self.evaluations:
                self.counters["rollout_cache_hits"] += 1
                continue
            self.meter.phase = "closed_rollout"
            row = closed_evaluation(self.env, root)
            if row["status"] != "feasible":
                raise RuntimeError("closed evaluator rejected root")
            self.evaluations[root.root_id] = row
            new.append(root.root_id)
        self.timing["exact_rollout"] += perf_counter()-tick
        # Search stages have different leaves: retain every evaluated incumbent.
        accumulated = [r for rid, r in self.all_roots.items() if rid in self.evaluations]
        return _best(accumulated, self.evaluations), new

    def trace(self):
        counters = dict(self.counters)
        counters.update(self.meter.counts)
        counters["candidate_checks"] = sum(v for k, v in self.meter.counts.items() if k.endswith("_candidate_checks"))
        counters["batch_checks"] = sum(v for k, v in self.meter.counts.items() if k.endswith("_batch_checks"))
        for key in ("expanded_nodes", "child_attempts", "partial_check_attempts", "partial_check_cache_hits", "final_root_check_attempts", "final_root_check_cache_hits", "learned_pruned_nodes", "rollout_cache_hits"):
            counters.setdefault(key, 0)
        return {"schema": SEARCH_SCHEMA, "nodes": self.nodes, "stages": self.stages,
                "counters": counters, "leaf_links": self.leaf_links,
                "catalogue_holder_count": len(self.catalogue),
                "catalogue_candidate_counts": {k: len(v) for k, v in self.catalogue.items()},
                "counter_scope": "decision_local_actual_check_calls; plan_checks_and_batch_calls_separate; cache_hits_excluded",
                "reference_scope": "same_finite_catalogue_width128_top12_plus_HOLD_liveFCFS; not_exhaustive"}


def apply_search_scheduler(env, arm_name, policy=None):
    started = perf_counter()
    arm = next((a for a in ONLINE_ARMS if a["name"] == arm_name), None)
    if arm is None:
        raise ValueError("unknown stage7 arm")
    view = observed_clone(env)
    counts = visible_counts(view)
    state = _Search(view, policy)
    adaptive_stages, fallback_reasons = [], []
    learned = arm["model_name"] is not None
    widths = (3, 5, 8, REFERENCE_WIDTH) if arm_name == "beam_adaptive" else (arm["budget"],)
    for width in widths:
        roots, stage = state.run(width, arm["method"] if learned else "beam")
        chosen, new = state.evaluate(roots)
        certificate = completion_certificate(state.evaluations[chosen.root_id]["value"], counts)
        adaptive_stages.append({"width": width, "actual_widths": [r["width"] for r in stage["layers"]],
                                "new_root_ids": new, "cumulative_count": len(state.evaluations),
                                "chosen_root_id": chosen.root_id, "certificate": certificate,
                                "elapsed_seconds": perf_counter()-started})
        if learned and (stage["prediction_failure"] or not certificate["passed"]):
            fallback_reasons.append(stage["prediction_failure"] or "visible_normal_upper_bound_not_reached")
            roots, stage = state.run(REFERENCE_WIDTH)
            chosen, new = state.evaluate(roots)
            certificate = completion_certificate(state.evaluations[chosen.root_id]["value"], counts)
            adaptive_stages.append({"width": REFERENCE_WIDTH, "actual_widths": [r["width"] for r in stage["layers"]],
                                    "new_root_ids": new, "cumulative_count": len(state.evaluations),
                                    "chosen_root_id": chosen.root_id, "certificate": certificate,
                                    "elapsed_seconds": perf_counter()-started})
            break
        if arm_name == "beam_adaptive" and certificate["passed"]:
            break
    tick = perf_counter()
    old_checker = env.feasibility
    env.feasibility = _MeteredFeasibility(env.config, state.meter)
    try:
        state.meter.phase = "pre_submit"
        checked = env.feasibility.check_batch(chosen.plans, list(env.aircraft.values()), env.time)
        state.timing["final_validation_and_emergency"] += perf_counter()-tick
        if checked.status != "feasible" and learned:
            fallback_reasons.append("final_check:"+checked.status+":"+checked.reason)
            roots, stage = state.run(REFERENCE_WIDTH)
            chosen, new = state.evaluate(roots)
            certificate = completion_certificate(state.evaluations[chosen.root_id]["value"], counts)
            adaptive_stages.append({"width": REFERENCE_WIDTH, "actual_widths": [r["width"] for r in stage["layers"]], "new_root_ids": new,
                                    "cumulative_count": len(state.evaluations), "chosen_root_id": chosen.root_id,
                                    "certificate": certificate, "elapsed_seconds": perf_counter()-started})
        elif checked.status != "feasible":
            raise RuntimeError("final independent validation rejected: "+checked.reason)
        tick = perf_counter()
        state.meter.phase = "final_submit"
        check = env.submit_batch(chosen.plans)
        if check.status != "feasible":
            raise RuntimeError("final independent submit rejected: "+check.reason)
        emergency = Statistics()
        state.meter.phase = "emergency"
        common_emergency_fallback(env, DEFAULT_SCHEDULER_CONFIG, emergency)
    finally:
        env.feasibility = old_checker
    state.timing["final_validation_and_emergency"] += perf_counter()-tick
    combined = Statistics()
    for item in [asdict(state.stats), asdict(emergency)]+[v["statistics"] for v in state.evaluations.values()]:
        combined.merge(Statistics(**item))
    trace = state.trace()
    work = {name: sum(v["work"][name] for v in state.evaluations.values()) for name in
            ("rollouts", "simulation_steps", "root_validation_calls", "continuation_submission_calls")}
    work.update(trace["counters"])
    work.update(root_rollout_count=work["rollouts"], unique_rollouts=work["rollouts"],
                pool_validation_calls=trace["counters"].get("final_root_batch_checks", 0),
                final_validation_calls=trace["counters"].get("pre_submit_batch_checks", 0)+trace["counters"].get("final_submit_batch_checks", 0),
                final_submit_count=1, counter_scope=trace["counter_scope"])
    all_roots = sorted(state.all_roots.values(), key=lambda r: action_key(r.plans))
    row = {"time": env.time, "method": arm["method"], "arm": arm_name,
           "budget": len(state.evaluations), "configured_budget": arm["budget"],
           "budget_scope": "partial_plan_beam_width; root_rollouts_separately_measured; not_stage6_TOTAL",
           "pool_limit": POOL_LIMIT, "pool_count": len(all_roots), "pool": [r.serializable() for r in all_roots],
           "pool_metadata": {"settings": asdict(REFERENCE_SETTINGS), "beam_order": stage["beam_order"],
                             "anchors": stage["anchors"], "statistics": asdict(state.stats),
                             "pool_validation_calls": work["pool_validation_calls"], "control_source": "same_state_live_FCFS"},
           "selected_root_ids": list(state.evaluations), "chosen_root_id": chosen.root_id,
           "normal_plans": [asdict(p) for p in chosen.plans], "evaluations": state.evaluations,
           "final_check": asdict(check), "statistics": asdict(combined), "emergency_statistics": asdict(emergency),
           "work": work, "timing": state.timing, "search": trace,
           "degenerate": len(all_roots) == 1, "fast_path": None,
           "adaptive": {"enabled": arm["adaptive"], "stages": adaptive_stages,
                        "stop_reason": "prediction_or_certificate_fallback" if fallback_reasons else
                                       "visible_normal_upper_bound_reached" if certificate["passed"] and arm["adaptive"] else "configured_search_complete",
                        "fallback_count": len(fallback_reasons) if learned else max(0, len(adaptive_stages)-1),
                        "fallback_reasons": fallback_reasons, "quality_certified": certificate["passed"], "visible_counts": counts},
           "timing_scope": "full_catalogue_partial_search_features_inference_unique_rollouts_final_checks_emergency_trace_bookkeeping; IO_model_load_steps_excluded"}
    state.timing["total"] = perf_counter()-started
    env.decision_calls += 1
    env.decision_seconds += state.timing["total"]
    env.search_candidates += work["candidate_checks"]
    return row
