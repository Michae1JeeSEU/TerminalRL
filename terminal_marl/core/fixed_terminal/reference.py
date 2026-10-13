"""Larger-budget offline root reference evaluated by one fixed continuation.

Observed-cohort root selection and ex-post realized-path labels are separate.
Preexisting t0 control roots are warm starts, not free solver computation.
"""
from dataclasses import dataclass, asdict
from hashlib import sha256
from time import perf_counter
from .baselines import make_catalogue, joint_choices, choose_normal_batch, observed_clone, Statistics
from .value_analysis import load_stage2_episode, plans_from_decision, evaluate_root
from terminal_marl.configs.fixed_terminal_stage2 import METHODS, DEFAULT_SCHEDULER_CONFIG
from terminal_marl.configs.fixed_terminal_stage3 import REFERENCE_SETTINGS, ROOT_SEARCH_KEEP


def action_key(plans):
    return tuple(sorted((p.aircraft_id, p.wait_cycles) for p in plans))


def value_comparison_key(value):
    """Frozen numeric comparison precision; keep raw J in every artifact."""
    raw = value["J"]
    return tuple(int(x) for x in raw[:4])+(round(raw[4], 8), round(raw[5], 8), round(raw[6], 10))


@dataclass
class Root:
    root_id: str
    plans: tuple
    labels: list
    provenance: list
    feasibility: dict

    def serializable(self):
        return {"root_id": self.root_id, "plans": [asdict(p) for p in self.plans],
                "labels": self.labels, "provenance": self.provenance, "feasibility": self.feasibility}


def acquire_controls(env, context, stage2_root=None):
    controls, timing = {}, {}
    for method in METHODS:
        started = perf_counter()
        if env.time == 0 and stage2_root is not None:
            from pathlib import Path
            path = Path(stage2_root)/f"{method}-{context.load}-{context.soc}-{context.seed}"
            source = load_stage2_episode(path)
            if source["config"] != env.config or source["scenario"] != env.scenario:
                raise ValueError("control root source does not share the same initial scenario/config")
            row = source["decisions"][0]
            plans = plans_from_decision(env, row)
            provenance = {"source": "preexisting_stage2_root", "method": method, "path": str(path),
                          "original_decision_seconds": row["seconds"],
                          "solver_cost_interpretation": "precomputed warm start; reconstruction is not solve cost"}
        else:
            decision = choose_normal_batch(method, observed_clone(env), DEFAULT_SCHEDULER_CONFIG)
            plans = decision.plans
            provenance = {"source": "same_state_recomputed_control", "method": method,
                          "actual_seconds": decision.seconds, "statistics": asdict(decision.statistics)}
        check = env.feasibility.check_batch(plans, list(env.aircraft.values()), env.time)
        elapsed = perf_counter()-started
        provenance["acquisition_and_check_seconds"] = elapsed
        provenance["feasibility"] = asdict(check)
        controls[method] = (plans, provenance)
        timing[method] = elapsed
    return controls, timing


def make_root_pool(env, context, stage2_root=None):
    started = perf_counter()
    view = observed_clone(env)
    stats = Statistics()
    catalogue = make_catalogue(view, REFERENCE_SETTINGS, stats)
    choices = joint_choices(view, catalogue, REFERENCE_SETTINGS, stats)
    pool = {}

    def add(plans, label, provenance):
        key = action_key(plans)
        if key not in pool:
            root_id = "root-"+sha256(repr((env.time, key)).encode()).hexdigest()[:12]
            checked = env.feasibility.check_batch(plans, list(env.aircraft.values()), env.time)
            pool[key] = Root(root_id, tuple(plans), [], [], asdict(checked))
        pool[key].labels.append(label)
        pool[key].provenance.append(provenance)

    for rank, batch in enumerate(choices[:ROOT_SEARCH_KEEP]):
        add(batch, f"expanded_joint_rank_{rank}", {"source": "expanded_joint_search", "rank": rank})
    add((), "HOLD", {"source": "explicit_hold_control"})
    root_generation_seconds = perf_counter()-started
    control_started = perf_counter()
    controls, control_timings = acquire_controls(env, context, stage2_root)
    for method, (batch, provenance) in controls.items():
        add(batch, method, provenance)
    control_seconds = perf_counter()-control_started
    roots = sorted(pool.values(), key=lambda root: action_key(root.plans))
    if len(roots) > ROOT_SEARCH_KEEP+7:
        raise AssertionError("reference root pool exceeds preregistered maximum19")
    metadata = {"reference_type": "budget_bounded_offline_root_pool_not_global_optimum",
                "reference_settings": asdict(REFERENCE_SETTINGS), "control_settings": asdict(DEFAULT_SCHEDULER_CONFIG),
                "expanded_roots_limit": ROOT_SEARCH_KEEP, "pool_limit": ROOT_SEARCH_KEEP+7,
                "unique_roots": len(roots), "statistics": asdict(stats),
                "root_generation_seconds": root_generation_seconds,
                "control_acquisition_seconds": control_seconds, "control_acquisition_by_method": control_timings,
                "catalogue": {key: [asdict(p) for p in group] for key, group in catalogue.items()}}
    return roots, metadata


def observed_evaluation(env, root):
    stops = (min(env.config.horizon, env.time+20.), min(env.config.horizon, env.time+60.), env.config.horizon)
    return evaluate_root(env, root.plans, stops, realized_future=False)


def select_observed_roots(env, roots, evaluations):
    selections = {}
    for label, stop in (("20s", min(env.config.horizon, env.time+20.)),
                        ("60s", min(env.config.horizon, env.time+60.)), ("until240", env.config.horizon)):
        available = [(root, evaluations[root.root_id]["values"][f"{stop:g}"])
                     for root in roots if evaluations[root.root_id]["status"] == "feasible"]
        if not available:
            raise ValueError("no independently feasible evaluated root, including HOLD")
        # Fixed deterministic action key settles equal-J ties.
        best_value = max(value_comparison_key(value) for _, value in available)
        chosen, value = min(((root, value) for root, value in available if value_comparison_key(value) == best_value),
                            key=lambda pair: action_key(pair[0].plans))
        selections[label] = {"root_id": chosen.root_id, "labels": chosen.labels, "value": value,
                             "selection_information": "observed_arrived_cohort_only",
                             "comparison_key": value_comparison_key(value),
                             "comparison_precision": "counts_exact; exposure_delay_round8; SOC_energy_round10; raw_J_preserved",
                             "evaluated_root_count": len(available),
                             "no_worse_than_any_evaluated_root_in_this_pool": all(value_comparison_key(value) >= value_comparison_key(other) for _, other in available)}
    return selections
