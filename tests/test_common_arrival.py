"""Checks for shared requests, actual source calls, execution and honest output."""
from dataclasses import replace
import json
import math
import sys

import numpy as np
import pytest

from terminal_marl.core.common_arrival import CommonArrivalConfig, generate_requests, request_digest, validate_requests
from terminal_marl.envs.common_arrival_env import CommonArrivalEnv
from terminal_marl.policies.common_arrival_baselines import RulePolicy, YangMCTSPolicy, SaxenaMILPPolicy, make_policy, MissingCheckpointError
from terminal_marl.scripts.evaluate_common_arrival import run_one, metrics


def ready(**kwargs):
    c = CommonArrivalConfig(**kwargs)
    env = CommonArrivalEnv(c, generate_requests(c, 20261003))
    env.prepare()
    return c, env


def test_request_tape_independent_and_causal():
    c, env = ready(requests=3)
    tape = generate_requests(c, 20261003)
    np.random.seed(919)
    assert request_digest(tape) == request_digest(generate_requests(c, 20261003))
    assert list(env.available_to_controller()) == [0]
    with pytest.raises(ValueError):
        env.set_grant(1)
    with pytest.raises(ValueError):
        validate_requests([tape[0], tape[0]], c)


def test_free_flight_is_area_not_mandatory_ring():
    _, env = ready(requests=1)
    a = env.arrivals[0]
    a.position = np.array([570., 100., 100.])
    a.heading = math.pi
    a.velocity = np.array([-13., 0., 0.])
    plan = env.motion_plan(a, 0.0)
    assert env._boundary_ok(a, plan, plan.end_time)[0]
    assert env._verify({0: plan})[0]
    assert np.linalg.norm(plan.sample_position(1)[:2]) != pytest.approx(env.config.holding_radius)


def test_ungranted_capture_and_outer_boundary_rejected():
    _, env = ready(requests=1)
    a = env.arrivals[0]
    a.position = np.array([210., 0., 100.])
    a.velocity = np.array([-13., 0., 0.]); a.heading = math.pi
    assert env._boundary_ok(a, env.motion_plan(a, 0), 10)[1] == "ungranted_capture_entry"
    a.position = np.array([790., 0., 100.])
    a.velocity = np.array([13., 0., 0.]); a.heading = 0
    assert env._boundary_ok(a, env.motion_plan(a, 0), 10)[1] == "outer_boundary"


def test_source_mcts_runs_without_global_rng_or_module_leak():
    c, env = ready(requests=2, arrival_pattern="bursty", mcts_simulations=10)
    old = {k: sys.modules.get(k) for k in ("config_vertiport", "common", "nodes_multi", "search_multi")}
    policy = YangMCTSPolicy(c, 73)
    np.random.seed(51); expected = np.random.get_state()
    actions = policy.act(env)
    actual = np.random.get_state()
    assert np.array_equal(expected[1], actual[1]) and expected[2:] == actual[2:]
    assert all(sys.modules.get(k) is v for k, v in old.items())
    assert set(actions) == {0, 1}
    assert len(policy.metadata["source_hashes"]) == 4
    c2, e2 = ready(requests=2, arrival_pattern="bursty", mcts_simulations=10)
    assert actions == YangMCTSPolicy(c2, 73).act(e2)


def test_milp_preserves_tape_and_enforces_pad_intervals():
    pytest.importorskip("scipy")
    c, env = ready(requests=2, arrival_pattern="bursty")
    original = request_digest(env.requests)
    policy = SaxenaMILPPolicy(c)
    policy.act(env)
    row = policy.solver_records[0]
    assert row["accepted"] and row["max_violation"] <= 1e-6
    times = sorted(r["R2"] for r in row["node_times_relative"].values())
    assert times[0] >= (c.radius - c.capture_radius) / c.cruise_speed + 40 - 1e-6
    assert times[1] - times[0] >= c.service_seconds + 40 - 1e-6
    assert request_digest(env.requests) == original


@pytest.mark.parametrize("method", ["fcfs_rule", "yang_mcts_adapted", "saxena_milp_adapted"])
def test_complete_episode_outputs_and_service_clock(method, tmp_path):
    if method == "saxena_milp_adapted":
        pytest.importorskip("scipy")
    c = CommonArrivalConfig(requests=1, duration=300)
    result = run_one(c, generate_requests(c, 123), method, 123, tmp_path / method)
    assert result["status"] == "complete" and result["completed"] == 1
    assert result["simulated_seconds"] >= (c.radius - c.capture_radius) / c.cruise_speed + 40 + c.service_seconds
    assert result["conservation_ok"] and result["unresolved_total"] == 0
    assert result["resource_violations"] == 0
    assert result["safety"]["minimum_actual_separation"] is None
    assert result["safety"]["collisions"] == 0
    text = (tmp_path / method / "metrics.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["performance_valid"]


def test_missing_waltz_is_skip_not_fake_performance(tmp_path):
    c = CommonArrivalConfig(requests=1, duration=300)
    with pytest.raises(MissingCheckpointError):
        make_policy("waltz_actor_adapted", c)
    result = run_one(c, generate_requests(c, 0), "waltz_actor_adapted", 0, tmp_path / "waltz")
    assert result["status"] == "skipped_missing_checkpoint"
    assert not result["performance_valid"] and result["completed"] == 0
    assert result["simulated_seconds"] == 0


def test_unresolved_requests_are_not_hidden_by_success_mean():
    c, env = ready(requests=3, duration=200)
    policy = RulePolicy(c)
    env.step(policy.act(env))
    result, _ = metrics(env, policy)
    assert result["completed"] == 0 and result["unresolved_total"] == 3
    assert result["unresolved_due"] == 1 and result["unspawned"] == 2
    assert result["request_to_completion_seconds"]["mean"] is None
    assert result["unresolved_age_seconds"]["mean"] == 1
    assert env.status == "running"
