import numpy as np

from terminal_marl.core.config import SimConfig
from terminal_marl.core.dynamic_airspace import DynamicAirspace
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.policies import DynamicRulePolicy
from terminal_marl.training.a64_lever_calibration import HEADWAY_ACTIONS, capacity_threshold_probe, _safe


def test_a64_final_capacity_is_a_hard_runtime_resource_limit():
    config = SimConfig(arrival_model="finite_cohort", total_requested_uavs=20, arrival_rate_per_second=.1, max_simulation_seconds=500.)
    env = TerminalParallelEnv(config); env.reset(seed=0); env.run_rule_only()
    assert env.telemetry["max_concurrent_final"] <= config.final_queue_capacity
    assert env.telemetry["final_overcapacity_steps"] == 0


def test_a64_capacity_threshold_probe_changes_geometry_derived_capacity():
    result = capacity_threshold_probe()
    assert result["structural_applied"]
    assert any(after > before for before, after in zip(result["capacity_before"], result["capacity_after"]))
    assert all(delta > 0 for delta in result["applied_radius_delta"])


def test_a64_non_capacity_radius_change_is_explicit_noop_or_projection():
    airspace = DynamicAirspace(SimConfig())
    update = airspace.request_update(np.asarray((.1, 0, 0, 0, 0, 0, 0, 0, 0, 0), np.float32), now=0., geometry_gate=True)
    assert any(code.startswith("capacity_threshold_") for code in update.structural_reason_codes)
    assert update.capacity_after[0] > update.capacity_before[0] or update.applied_radius_delta[0] == 0.


def test_a64_pressure_gradient_meters_only_saturated_downstream_interface():
    policy = DynamicRulePolicy()
    action = policy.act(np.zeros(29, np.float32), {"external_queue": 8, "occupancy_ratio": [.8, .2, .1], "final_pressure": .1, "lz_utilization": .1})
    assert action[7] < 0 and action[8] == action[9] == 0
    saturated = policy.act(np.zeros(29, np.float32), {"external_queue": 8, "occupancy_ratio": [.8, .8, .8], "final_pressure": 1., "lz_utilization": 1.})
    assert all(saturated[index] > 0 for index in (7, 8, 9))
    assert set(HEADWAY_ACTIONS) == {"baseline_9_9_12", "o2m_only", "m2i_only", "i2f_only", "o2m_m2i", "all_decrease"}


def test_a64_screen_safety_requires_final_capacity_metrics():
    assert not _safe({"collisions": 0, "nmac": 0, "hard_separation_violations": 0})
    assert _safe({"collisions": 0, "nmac": 0, "hard_separation_violations": 0, "final_overcapacity_steps": 0, "max_concurrent_final": 1})
