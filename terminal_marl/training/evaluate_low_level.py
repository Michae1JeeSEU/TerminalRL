"""Common-random-number comparison for fixed_rule versus fixed_low_rl."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from .train_low_level_ppo import _episode
from ..core.config import SimConfig
from ..policies import RuleLowLevelPolicy

class _RuleAdapter:
    def act(self, observations, world, deterministic=False):
        n=len(observations);return np.tile(np.array((1.,0.,0.),np.float32),(n,1)),np.zeros(n),0.
def compare_fixed_rule_and_low_rl(policy, config: SimConfig, seeds: list[int]) -> dict:
    if config.total_requested_uavs is None:
        raise ValueError("evaluation requires a finite common-random-number arrival cohort")
    fixed=replace(config, experiment_mode="fixed_rule", high_level_enabled=False)
    rl=replace(config, experiment_mode="fixed_low_rl", high_level_enabled=False)
    rows={"fixed_rule": [_episode(_RuleAdapter(),fixed,s,None,True) for s in seeds], "fixed_low_rl": [_episode(policy,rl,s,None,True) for s in seeds]}
    return {name:{metric:float(np.mean([row[metric] for row in values])) for metric in ("delay","throughput","collisions","nmac","safety_interventions","energy","fairness")} for name,values in rows.items()} | {"per_seed":rows}

def assess_a5_gate(comparison: dict, training_history: list[dict]) -> dict:
    """Explicitly gate A6: no hidden promotion from an unverified A5 run."""
    rule, learned = comparison["fixed_rule"], comparison["fixed_low_rl"]
    action_active = all(any(abs(row.get(f"action_{name}_std", 0.0)) > 1e-4 for row in training_history) for name in ("release", "speed_residual", "merge_time_residual"))
    no_nan = all(np.isfinite(row["reward"]) for row in training_history)
    no_collision_regression = learned["collisions"] <= rule["collisions"] + 1e-9
    safety_not_markedly_worse = learned["safety_interventions"] <= max(1.0, rule["safety_interventions"]) * 1.25
    improves_primary = learned["delay"] < rule["delay"] or learned["throughput"] > rule["throughput"]
    checks = {"all_action_dimensions_active":action_active,"no_nan":no_nan,"no_collision_regression":no_collision_regression,"safety_not_markedly_worse":safety_not_markedly_worse,"improves_delay_or_throughput":improves_primary}
    return {"passed":all(checks.values()),"checks":checks,"fixed_rule":rule,"fixed_low_rl":learned}
