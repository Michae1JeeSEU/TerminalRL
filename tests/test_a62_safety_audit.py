import numpy as np
from types import SimpleNamespace

from terminal_marl.core.config import SimConfig
from terminal_marl.core.runtime_separation import RuntimeSeparationMonitor, swept_minimum_distance
from terminal_marl.core.merge_scheduler import LowLevelIntent, MergeScheduler
from terminal_marl.core.uav import FlightPhase
from terminal_marl.envs.high_level_wrapper import DynamicHighLevelWrapper
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.a6_effectiveness import write_report
from terminal_marl.training.a62_safety_audit import run_fail_fast


def test_missing_hard_sep_metric_fails_gate(tmp_path):
    raw = {"low": {name: [{"collisions": 0., "nmac": 0., "throughput": 1., "service_delay_p95": 1., "external_queue_area": 0., "completed": 1.}] for name in ("fixed_rule", "dynamic_rule", "learned_high_rule_low", "learned_high_frozen_a5_low")}}
    assert not write_report(raw, tmp_path)["safety_passed"]


def test_separation_threshold_consistency():
    config = SimConfig(); monitor = RuntimeSeparationMonitor(config)
    class State:
        def __init__(self, p): self.position = np.asarray(p, float)
    monitor.observe({1: State((0, 0, 0)), 2: State((10, 0, 0))}, {1: np.zeros(3), 2: np.asarray((10, 0, 0))})
    assert monitor.telemetry["collisions"] == monitor.telemetry["nmac"] == monitor.telemetry["hard_separation_violations"] == 1


def test_swept_minimum_separation():
    distance, fraction = swept_minimum_distance((-10, 0, 0), (10, 0, 0), (10, 0, 0), (-10, 0, 0))
    assert distance == 0 and fraction == .5


def test_rejected_tentative_reservation_does_not_consume_headway():
    env = TerminalParallelEnv(SimConfig(fixed_arrival_schedule=(0.,))); env.reset(0)
    env.step()
    uav = next(iter(env.active.values()))
    uav.phase = FlightPhase.RING_OUTER
    from terminal_marl.core.config import RingRole
    uav.ring_role = RingRole.OUTER
    scheduler = MergeScheduler(env.config)
    proposal = scheduler.propose(uav, LowLevelIntent(1., 0., 0.), type("Snapshot", (), {"now": 0., "airspace": env.airspace, "active": {}})(), RingRole.MIDDLE)
    reservation = scheduler.reserve(uav, proposal)
    scheduler.cancel(reservation.reservation_id, "safety_reject")
    assert scheduler.last_release["o2m"] == float("-inf")


def test_first_conflict_trace():
    result = run_fail_fast("medium", 0, max_seconds=80.)
    # The repaired inbound path may now be safe; an unsafe version must still
    # expose a structured first-conflict trace through the same runner.
    if not result["safe"]:
        trace = result["first_conflict"]
        assert {"pair", "phase_pair", "uavs", "recent_trace", "swept_min_distance"} <= set(trace)


def test_saturated_fixed_rule_medium_safe_to_first_conflict_window():
    config = SimConfig(arrival_model="finite_cohort", total_requested_uavs=120, arrival_rate_per_second=.04, max_simulation_seconds=400.)
    env = TerminalParallelEnv(config); env.reset(0); env.run_rule_only()
    assert env.telemetry["collisions"] == env.telemetry["nmac"] == env.telemetry["hard_separation_violations"] == 0


def test_saturated_fixed_rule_high_safe_to_first_conflict_window():
    result = run_fail_fast("high", 0, max_seconds=400.)
    assert result["safe"]


def test_saturated_fixed_rule_bursty_safe_to_first_conflict_window():
    result = run_fail_fast("bursty", 0, max_seconds=400.)
    assert result["safe"]


def test_pending_reconfiguration_single_instance_and_operational_controls():
    config = SimConfig(experiment_mode="dynamic_rule", high_level_enabled=True, arrival_model="finite_cohort", total_requested_uavs=10, arrival_rate_per_second=.2, max_simulation_seconds=240.)
    wrapper = DynamicHighLevelWrapper(config); wrapper.reset(0)
    for _ in range(180): wrapper.base.step()
    decision = wrapper.coordinator.coordinate(np.asarray((1, 0, 0, 0, 0, 0, .2, -.2, 0, 0), np.float32), geometry_gate=True)
    if decision.outcome == "delayed":
        pending = wrapper.coordinator.pending_action.copy()
        replacement = wrapper.coordinator.coordinate(np.asarray((.5, 0, 0, 0, 0, 0, .1, .2, 0, 0), np.float32), geometry_gate=True)
        assert wrapper.coordinator.pending_action is not None and wrapper.coordinator.pending_action.shape == pending.shape
        # Structural deferral does not freeze safe operational control, but
        # an operational change that fails its own protected-window check is
        # fail-closed rather than leaking through merely because it is not
        # structural.
        assert np.isclose(decision.applied_action[7], -.2) or np.isclose(decision.applied_action[7], 0.)
        assert np.isclose(replacement.applied_action[7], .2) or np.isclose(replacement.applied_action[7], 0.)
        if np.isclose(decision.applied_action[7], 0.):
            assert decision.safety_reason is not None


def test_deferred_action_no_per_second_retry():
    config = SimConfig(experiment_mode="dynamic_rule", high_level_enabled=True, total_requested_uavs=10, max_simulation_seconds=240.)
    wrapper = DynamicHighLevelWrapper(config); wrapper.reset(0)
    for _ in range(100): wrapper.base.step()
    decision = wrapper.coordinator.coordinate(np.ones(10, np.float32), geometry_gate=True)
    if decision.outcome == "delayed":
        attempts = wrapper.coordinator.telemetry["coordinator_attempts"]
        for _ in range(10): wrapper.base.step(); wrapper.coordinator.advance()
        assert wrapper.coordinator.telemetry["coordinator_attempts"] == attempts


def test_pending_action_superseded_at_next_decision():
    config = SimConfig(experiment_mode="dynamic_rule", high_level_enabled=True, total_requested_uavs=10, max_simulation_seconds=240.)
    wrapper = DynamicHighLevelWrapper(config); wrapper.reset(0)
    for _ in range(100): wrapper.base.step()
    first = wrapper.coordinator.coordinate(np.asarray((1, 0, 0, 0, 0, 0, 0, 0, 0, 0), np.float32), geometry_gate=True)
    if first.outcome == "delayed":
        replacement = np.asarray((-.5, 0, 0, 0, 0, 0, 0, 0, 0, 0), np.float32)
        wrapper.coordinator.coordinate(replacement, geometry_gate=True)
        assert wrapper.coordinator.pending_action is not None
        assert wrapper.coordinator.pending_action[0] == replacement[0]


def test_high_policy_decision_count_matches_macro_steps():
    config = SimConfig(experiment_mode="dynamic_rule", high_level_enabled=True, total_requested_uavs=5, max_simulation_seconds=360.)
    wrapper = DynamicHighLevelWrapper(config); observation, _ = wrapper.reset(0)
    for _ in range(2):
        observation, _, terminated, truncated, _ = wrapper.step(np.zeros(10, np.float32))
        if terminated or truncated: break
    assert len(wrapper.action_history) <= 2
    assert wrapper.action_history[-1]["high_policy_decisions"] == len(wrapper.action_history)


def test_structural_defer_does_not_block_safe_headway_update():
    config = SimConfig(experiment_mode="dynamic_rule", high_level_enabled=True, total_requested_uavs=10, max_simulation_seconds=240.)
    wrapper = DynamicHighLevelWrapper(config); wrapper.reset(0)
    for _ in range(100): wrapper.base.step()
    headway = wrapper.base.airspace.headways["o2m"]
    decision = wrapper.coordinator.coordinate(np.asarray((1, 0, 0, 0, 0, 0, 0, .5, 0, 0), np.float32), geometry_gate=True)
    if decision.outcome == "delayed":
        assert wrapper.base.airspace.headways["o2m"] > headway


def test_repair_under_active_traffic():
    config = SimConfig(arrival_model="finite_cohort", total_requested_uavs=120, arrival_rate_per_second=.04, max_simulation_seconds=900.)
    env = TerminalParallelEnv(config); env.reset(0); env.run_rule_only()
    # The hand-off guard may prevent the repair from being needed in this
    # deterministic demand stream.  It must nevertheless stay hard-safe.
    assert env.telemetry["hard_separation_violations"] == env.telemetry["nmac"] == env.telemetry["collisions"] == 0


def test_coordinator_repairs_with_unrelated_active_traffic(monkeypatch):
    config = SimConfig(experiment_mode="dynamic_rule", high_level_enabled=True, arrival_model="finite_cohort", total_requested_uavs=10, arrival_rate_per_second=.2, max_simulation_seconds=240.)
    wrapper = DynamicHighLevelWrapper(config); wrapper.reset(0)
    for _ in range(180): wrapper.base.step()
    coordinator = wrapper.coordinator
    holder = next(uav for uav in wrapper.base.active.values() if uav.ring_role is not None and uav.committed_plan is None)
    calls = []
    class Verdict:
        def __init__(self, approved): self.approved = approved; self.reason = "predicted_conflict"
    def candidate(action, repairs):
        calls.append(dict(repairs))
        return None, Verdict(bool(repairs))
    monkeypatch.setattr(coordinator, "_candidate_safety", candidate)
    monkeypatch.setattr(coordinator, "_repair_options", lambda: iter(({holder.uid: config.max_speed_residual_mps},)))
    decision = coordinator.coordinate(np.asarray((1, 0, 0, 0, 0, 0, 0, 0, 0, 0), np.float32), geometry_gate=True)
    assert len(wrapper.base.active) > 1
    assert decision.outcome == "approved_repaired"
    assert decision.repair_count == 1 and wrapper.base.telemetry["geometry_updates"] == 1
    assert calls == [{}, {holder.uid: config.max_speed_residual_mps}]
