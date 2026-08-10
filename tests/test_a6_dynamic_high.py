import numpy as np

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.dynamic_airspace import DynamicAirspace
from terminal_marl.envs.high_level_wrapper import DynamicHighLevelWrapper
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.policies import DynamicRulePolicy


def dynamic_config(**overrides):
    values = {"experiment_mode": "dynamic_rule", "high_level_enabled": True, "arrival_model": "finite_cohort", "total_requested_uavs": 10, "max_simulation_seconds": 700.0}
    values.update(overrides)
    return SimConfig(**values)


def test_a6_high_action_masks_structure_but_always_applies_speed_and_headways():
    airspace = DynamicAirspace(SimConfig())
    result = airspace.request_update(np.ones(10), now=0.0, geometry_gate=False)
    assert not result.structural_applied
    assert result.structural_mask == (False,) * 6
    assert result.applied_action[:6] == (0.0,) * 6
    assert result.applied_action[6:] == (1.0,) * 4
    assert airspace.speed_profile_bias > 0.0 and airspace.headways == {"o2m": 10.0, "m2i": 10.0, "i2f": 13.0}


def test_a6_wrapper_records_requested_applied_geometry_capacity_and_headways():
    wrapper = DynamicHighLevelWrapper(dynamic_config())
    observation, _ = wrapper.reset(seed=0)
    assert observation.shape == (wrapper.observation_dim,)
    action = np.asarray((1, 1, 1, .5, .5, .5, .2, -.2, -.2, -.2), np.float32)
    _, _, _, _, first = wrapper.step(action)
    first_action = first["high_action"]
    assert first_action["requested_action"] == list(action)
    assert first_action["structural_mask"] == [False] * 6
    assert first_action["applied_action"][6:] == list(action[6:])
    _, _, _, _, second = wrapper.step(action)
    second_action = second["high_action"]
    assert any(second_action["structural_mask"])
    assert second_action["actual_geometry"] == second_action["target_geometry"]
    assert second_action["capacity"] != [47, 31, 15]
    assert second_action["headways"]["o2m"] < 9.0


def test_a6_transition_freezes_new_interface_and_keeps_committed_transfer():
    config = dynamic_config(max_simulation_seconds=400.0)
    env = TerminalParallelEnv(config)
    env.reset(seed=0)
    committed = None
    for _ in range(160):
        env.step()
        committed = next((u for u in env.active.values() if u.committed_plan is not None and u.committed_plan.mode == "ring_transfer"), None)
        if committed is not None:
            break
    assert committed is not None
    plan_id = committed.trajectory_plan_id
    update = env.apply_high_level_action(np.asarray((1, 0, 0, 0, 0, 0, 0, 0, 0, 0), np.float32), geometry_gate=True)
    assert update.structural_applied and "o2m" in update.affected_interfaces
    assert env.airspace.actual_ring(RingRole.OUTER).radius != env.airspace.target_ring(RingRole.OUTER).radius
    env.step()
    assert committed.uid in env.active and env.active[committed.uid].trajectory_plan_id == plan_id


def test_a6_dynamic_rule_is_canonical_10d_and_pressure_responsive():
    policy = DynamicRulePolicy()
    quiet = policy.act(np.zeros(29, np.float32), {"external_queue": 0, "occupancy_ratio": [0., 0., 0.], "final_pressure": 0., "lz_utilization": 0.})
    busy = policy.act(np.zeros(29, np.float32), {"external_queue": 8, "occupancy_ratio": [.8, .2, .1], "final_pressure": 0., "lz_utilization": .2})
    assert quiet.shape == busy.shape == (10,)
    assert busy[0] > 0 and busy[7] < 0


def test_a60_safety_handoff_prevents_historical_fixed_rule_conflicts():
    config = SimConfig(arrival_model="finite_cohort", total_requested_uavs=10, max_simulation_seconds=1800.0)
    env = TerminalParallelEnv(config)
    env.reset(seed=2); env.run_rule_only()
    assert env.ledger.closes()
    assert env.telemetry["collisions"] == env.telemetry["nmac"] == 0
    assert env.telemetry["nmac_exposure_steps"] == 0
