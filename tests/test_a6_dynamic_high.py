import numpy as np

from terminal_marl.core.config import RingRole, SimConfig
from terminal_marl.core.dynamic_airspace import DynamicAirspace
from terminal_marl.core.reconfiguration_safety import ReconfigurationSafetyCoordinator
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
    assert second_action["shield_outcome"] in {"delayed", "approved_repaired", "approved", "projected"}
    # A committed inner/final corridor is protected rather than being moved
    # optimistically; flow controls remain applied in either case.
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
    # A pressured outer ring with slack in middle is released prospectively;
    # downstream saturation is what activates metering, not congestion alone.
    assert busy[0] > 0 and busy[7] < 0 and busy[8] == busy[9] == 0


def test_a60_safety_handoff_prevents_historical_fixed_rule_conflicts():
    config = SimConfig(arrival_model="finite_cohort", total_requested_uavs=10, max_simulation_seconds=1800.0)
    env = TerminalParallelEnv(config)
    env.reset(seed=2); env.run_rule_only()
    assert env.ledger.closes()
    assert env.telemetry["collisions"] == env.telemetry["nmac"] == 0
    assert env.telemetry["nmac_exposure_steps"] == 0


def test_a61_tiny_structural_noise_is_deadbanded_but_headway_remains_effective():
    wrapper = DynamicHighLevelWrapper(dynamic_config(max_simulation_seconds=200.0))
    wrapper.reset(seed=0)
    _, _, _, _, info = wrapper.step(np.asarray((.01, -.01, .02, 0, 0, 0, 0, -.2, 0, 0), np.float32))
    event = info["high_action"]
    assert event["shield_outcome"] == "no_op"
    assert event["structural_mask"] == [False] * 6
    assert event["headways"]["o2m"] < 9.0


def test_a61_unsafe_transition_is_delayed_not_immediately_cancelled():
    wrapper = DynamicHighLevelWrapper(dynamic_config(max_simulation_seconds=400.0))
    wrapper.reset(seed=0)
    # Establish an active committed trajectory before requesting an inner
    # update; the shield must preserve the request as a pending structure.
    for _ in range(90):
        wrapper.base.step()
        if any(u.committed_plan is not None for u in wrapper.base.active.values()):
            break
    coordinator = wrapper.coordinator
    decision = coordinator.coordinate(np.asarray((0, 0, 1, 0, 0, .5, 0, 0, 0, 0), np.float32), geometry_gate=True)
    assert decision.outcome in {"approved", "approved_repaired", "delayed"}
    if decision.outcome == "delayed":
        assert coordinator.pending_action is not None and decision.deferred_until is not None


def test_a61_projection_and_reject_have_explicit_reason_codes(monkeypatch):
    env = TerminalParallelEnv(dynamic_config())
    env.reset(seed=0)
    coordinator = ReconfigurationSafetyCoordinator(env)
    class Verdict:
        def __init__(self, approved): self.approved=approved; self.reason="forced_conflict"
    calls=[]
    def candidate(action, repairs):
        calls.append((action.copy(), repairs.copy()))
        return None, Verdict(len(calls) == 2)
    monkeypatch.setattr(coordinator, "_candidate_safety", candidate)
    monkeypatch.setattr(coordinator, "_repair_options", lambda: iter(()))
    decision = coordinator.coordinate(np.ones(10, np.float32), geometry_gate=True, allow_delay=False)
    assert decision.outcome == "projected" and decision.reason_code == "minimal_structural_projection"
    assert decision.projection_magnitude > 0
    monkeypatch.setattr(coordinator, "_candidate_safety", lambda action, repairs: (None, Verdict(False)))
    decision = coordinator.coordinate(np.ones(10, np.float32), geometry_gate=True, allow_delay=False)
    assert decision.outcome == "rejected" and decision.reason_code == "no_safe_reconfiguration"


def test_a61_dynamic_rule_random_actions_have_no_hard_conflict_smoke():
    rng = np.random.default_rng(7)
    for seed in (0, 1, 2):
        wrapper = DynamicHighLevelWrapper(dynamic_config(max_simulation_seconds=700.0))
        observation, _ = wrapper.reset(seed)
        while True:
            action = rng.uniform(-1, 1, 10).astype(np.float32)
            observation, _, terminated, truncated, info = wrapper.step(action)
            if terminated or truncated:
                break
        assert info["telemetry"]["collisions"] == info["telemetry"]["nmac"] == 0
        assert info["telemetry"]["hard_separation_violations"] == 0


def test_a64_dynamic_rule_only_meters_when_downstream_is_saturated():
    policy = DynamicRulePolicy()
    action = policy.act(np.zeros(29, np.float32), {"external_queue": 8, "occupancy_ratio": [.8, .8, .8], "final_pressure": 1., "lz_utilization": 1.})
    assert all(action[index] > 0 for index in (7, 8, 9))
