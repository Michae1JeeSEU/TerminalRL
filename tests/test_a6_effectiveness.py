import numpy as np

from terminal_marl.core.config import SimConfig
from terminal_marl.envs.high_level_wrapper import RuleLowLevelAdapter
from terminal_marl.training.a6_effectiveness import SCENARIOS, _run_dynamic, scenario_config

def test_a6_effectiveness_scenarios_are_long_flow_from_empty():
    assert [scenario.name for scenario in SCENARIOS] == ["low", "medium", "high", "bursty"]
    for scenario in SCENARIOS:
        config=scenario_config(scenario,"dynamic_rule")
        assert config.reset_mode == "flow_from_empty"
        assert config.max_simulation_seconds / config.high_level_decision_interval_seconds >= 25
        assert config.total_requested_uavs >= 80


def test_a6_effectiveness_retains_auditable_per_decision_action_trace():
    config = SimConfig(experiment_mode="dynamic_rule", high_level_enabled=True, reset_mode="flow_from_empty",
                       arrival_model="finite_cohort", total_requested_uavs=1, arrival_rate_per_second=0.5,
                       max_simulation_seconds=10.0)
    result = _run_dynamic(config, seed=3, high_action=lambda *_: np.zeros(10, np.float32), low_level=RuleLowLevelAdapter())
    assert result["action_events"]
    event = result["action_events"][0]
    assert len(event["requested_action"]) == len(event["applied_action"]) == 10
    assert {"time", "outcome", "reason_code", "projection_magnitude"} <= set(event)
    assert len(result["requested_action_summary"]["mean"]) == 10
