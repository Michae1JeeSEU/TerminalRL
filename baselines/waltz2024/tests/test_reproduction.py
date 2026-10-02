"""Semantic checks for the local no-training TUD_RL wrapper."""

from __future__ import annotations

import random
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from baselines.waltz2024.runner import (
    Arrival,
    LocalValidationEnv,
    ScenarioPlan,
    StrictActorPolicy,
    ZeroTurnPolicy,
    build_plan,
    load_upstream,
    main,
)


def _run_steps(env, count):
    policy = ZeroTurnPolicy()
    for _ in range(count):
        _, _, done, _ = env.step(policy)
        if done:
            break


def test_wave_counts_times_and_speed():
    np.random.seed(7)
    random.seed(7)
    env = LocalValidationEnv(build_plan("wave"), seed=7)
    _run_steps(env, 60)
    spawns = [e for e in env._spawn_events if e["event_type"] == "spawn"]
    assert len(spawns) == 12
    assert [e["time"] for e in spawns] == [0] * 4 + [30] * 4 + [60] * 4
    assert all(np.isclose(p.tas, 13.0) for p in env.planes)


def test_study_births_and_speed_noise_range():
    np.random.seed(11)
    random.seed(11)
    env = LocalValidationEnv(build_plan("study", n=5), seed=11, noise=20.0)
    _run_steps(env, 60)
    spawns = [e for e in env._spawn_events if e["event_type"] == "spawn"]
    assert len(spawns) == 5
    assert [e["time"] for e in spawns] == [0, 15, 30, 45, 60]
    assert all(10.0 <= p.tas <= 16.0 for p in env.planes)
    assert all(hasattr(p, "noisy_n") and hasattr(p, "noisy_e") for p in env.planes)


def test_gate_safe_distance_and_profile_normalization():
    env = LocalValidationEnv(build_plan("wave"), seed=1)
    mods = load_upstream()
    assert all(np.isclose(mods["ed"](env.dest.N, env.dest.E, g.N, g.E), 1000.0) for g in env.gates)
    assert not env._gate_is_free(env.gates[0])
    paper = LocalValidationEnv(build_plan("wave"), seed=1, profile="paper800")
    assert np.isclose(paper.dest.spawn_radius, 800.0)
    assert np.isclose(paper.distance_norm_m, 1000.0)
    assert np.isclose(paper.state[1], 0.8)


def test_turn_increment_and_destination_closure():
    env = LocalValidationEnv(build_plan("wave"), seed=2)
    p = env.planes[0]
    commanded, _ = p._RL_control(np.array([1.0], dtype=np.float32), False)
    assert np.isclose(commanded - p.hdg, 5.0)
    env.dest.close()
    for _ in range(60):
        env.dest.step([])
    assert env.dest.is_open


def test_noise_does_not_change_groundtruth_coordinates():
    np.random.seed(13)
    random.seed(13)
    clean = LocalValidationEnv(build_plan("study", n=1), seed=13, noise=0.0)
    np.random.seed(13)
    random.seed(13)
    noisy = LocalValidationEnv(build_plan("study", n=1), seed=13, noise=100.0)
    p0, p1 = clean.planes[0], noisy.planes[0]
    assert np.isclose(p1.n, p0.n) and np.isclose(p1.e, p0.e)
    assert np.isclose(p1.hdg, p0.hdg) and np.isclose(p1.tas, p0.tas)
    assert not np.isclose(p1.noisy_n, p1.n) or not np.isclose(p1.noisy_e, p1.e)
    assert not np.allclose(clean.state, noisy.state, equal_nan=True)


def test_poisson_schedule_is_four_cluster_south_gate_and_isolated_rng():
    plan = build_plan("poisson", poisson_seed=123)
    assert len(plan.cluster_counts) == 4
    assert all(a.gate == 2 and a.cluster in range(4) for a in plan.arrivals)
    for cluster, base in enumerate((0, 120, 240, 360)):
        times = [a.scheduled_t for a in plan.arrivals if a.cluster == cluster]
        assert times == [base + 10 * i for i in range(len(times))]
    np.random.seed(900)
    expected_first = np.random.random()
    expected_second = np.random.random()
    np.random.seed(900)
    observed_first = np.random.random()
    _ = build_plan("poisson", poisson_seed=123)
    observed_second = np.random.random()
    assert np.isclose(observed_first, expected_first)
    assert np.isclose(observed_second, expected_second)


def test_poisson_sorting_handles_cluster_overlap(monkeypatch):
    class FakeRng:
        def __init__(self):
            self.values = iter([13, 1, 1, 1])

        def poisson(self, _):
            return next(self.values)

    monkeypatch.setattr(np.random, "default_rng", lambda _seed: FakeRng())
    plan = build_plan("poisson", poisson_seed=1)
    assert plan.cluster_counts == (13, 1, 1, 1)
    assert plan.target_n == 16
    assert list(plan.arrivals) == sorted(plan.arrivals, key=lambda a: (a.scheduled_t, a.cluster))
    at_120 = [a for a in plan.arrivals if a.scheduled_t == 120]
    assert len(at_120) == 2
    assert {a.cluster for a in at_120} == {0, 1}
    assert plan.arrivals[-1].scheduled_t == 360


def test_poisson_zero_schedule_has_no_active_state(monkeypatch):
    class ZeroRng:
        def poisson(self, _):
            return 0

    monkeypatch.setattr(np.random, "default_rng", lambda _seed: ZeroRng())
    env = LocalValidationEnv(build_plan("poisson", poisson_seed=2), seed=2)
    assert env.plan.target_n == 0
    assert env._complete()
    _, reward, _, _ = env.step(ZeroTurnPolicy())
    assert reward == 0.0


def test_poisson_entry_check_retries_pending_south_request(monkeypatch):
    class TwoRng:
        def poisson(self, _):
            return 2 if _ == 5 else 0

    monkeypatch.setattr(np.random, "default_rng", lambda _seed: TwoRng())
    env = LocalValidationEnv(build_plan("poisson", poisson_seed=3), seed=3, entry_check=True)
    assert env._spawned_count == 1
    assert env._next_arrival == 1
    for _ in range(40):
        env.step(ZeroTurnPolicy())
    assert env._spawned_count == 2
    assert env.plan.target_n == env._spawned_count + env._unspawned_count


def test_empty_active_set_waits_for_future_birth():
    env = LocalValidationEnv(build_plan("study", n=2), seed=4)
    env.planes = []
    env.N_planes = 0
    _, _, done, _ = env.step(ZeroTurnPolicy())
    assert done is False
    for _ in range(14):
        env.step(ZeroTurnPolicy())
    assert env.sim_t == 15
    assert len(env.planes) == 1


def test_zero_separation_counts_as_valid_collision_measurement():
    env = LocalValidationEnv(build_plan("wave"), seed=5)
    env.planes[1].n = env.planes[0].n
    env.planes[1].e = env.planes[0].e
    env._pair_metrics()
    assert env._min_separation == 0.0
    assert env._accident_pair_steps >= 1
    assert env._incident_pair_steps >= 1


def test_wave_wrapper_matches_upstream_prefix_for_zero_turn_agent():
    mods = load_upstream()
    seed = 21
    np.random.seed(seed)
    random.seed(seed)
    source = mods["validation"](1, False, None, seed, False, 0.0)
    source.reset()
    policy = ZeroTurnPolicy()
    source_trace = []
    for _ in range(65):
        src_state, _, _, _ = source.step(policy)
        source_trace.append((src_state.copy(), [(p.n, p.e, p.hdg, p.tas, p.hist_len, p.s_hist.copy()) for p in source.planes]))
    np.random.seed(seed)
    random.seed(seed)
    wrapped = LocalValidationEnv(build_plan("wave"), seed=seed)
    for expected_state, expected_planes in source_trace:
        wrapped_state, _, _, _ = wrapped.step(policy)
        np.testing.assert_allclose(expected_state, wrapped_state, equal_nan=True)
        for p, expected in zip(wrapped.planes, expected_planes):
            np.testing.assert_allclose([p.n, p.e, p.hdg, p.tas], expected[:4])
            assert p.hist_len == expected[4]
            np.testing.assert_allclose(p.s_hist, expected[5], equal_nan=True)
        assert len(wrapped.planes) == len(expected_planes)
    keys = [(row["time"], row["uid"]) for row in wrapped._trajectory]
    assert len(keys) == len(set(keys))


def test_study_wrapper_matches_upstream_prefix_for_zero_turn_agent():
    mods = load_upstream()
    seed = 22
    np.random.seed(seed)
    random.seed(seed)
    source = mods["validation"](1, True, 6, seed, False, 0.0)
    source.reset()
    policy = ZeroTurnPolicy()
    source_trace = []
    for _ in range(65):
        src_state, _, _, _ = source.step(policy)
        source_trace.append((src_state.copy(), [(p.n, p.e, p.hdg, p.tas, p.hist_len, p.s_hist.copy()) for p in source.planes]))
    np.random.seed(seed)
    random.seed(seed)
    wrapped = LocalValidationEnv(build_plan("study", n=6), seed=seed)
    for expected_state, expected_planes in source_trace:
        wrapped_state, _, _, _ = wrapped.step(policy)
        np.testing.assert_allclose(expected_state, wrapped_state, equal_nan=True)
        for p, expected in zip(wrapped.planes, expected_planes):
            np.testing.assert_allclose([p.n, p.e, p.hdg, p.tas], expected[:4])
            assert p.hist_len == expected[4]
            np.testing.assert_allclose(p.s_hist, expected[5], equal_nan=True)
        assert len(wrapped.planes) == len(expected_planes)


def test_actor_architecture_forward_histories_and_nan_padding():
    mods = load_upstream()
    import torch

    actor = mods["nets"].LSTMRecActor(1, 3, 6, use_past_actions=False, device="cpu")
    obs = 3 + 6 * 11
    state = np.full(obs, np.nan, dtype=np.float32)
    state[:3] = [0.0, 1.0, -1.0]
    state[3:9] = [1.0, -1.0, -1.0, -1.0, 1.0, -1.0]
    s = torch.tensor(state).reshape(1, -1)
    h = torch.zeros((1, 2, obs), dtype=torch.float32)
    with torch.inference_mode():
        a0, _ = actor(s, h, None, torch.tensor([0]))
        a1, _ = actor(s, h, None, torch.tensor([1]))
        a2, _ = actor(s, h, None, torch.tensor([2]))
    assert a0.shape == a1.shape == a2.shape == (1, 1)
    assert torch.isfinite(a0).all() and torch.isfinite(a1).all() and torch.isfinite(a2).all()


def test_missing_actor_is_nonzero_and_writes_no_summary(tmp_path: Path):
    out = tmp_path / "missing"
    with pytest.raises(SystemExit) as exc:
        main(["--scenario", "wave", "--actor", str(tmp_path / "missing.pth"), "--output", str(out)])
    assert exc.value.code == 2
    assert not (out / "summary.json").exists()


def test_capped_diagnostic_writes_schema_and_marks_truncated(tmp_path: Path):
    out = tmp_path / "diagnostic"
    assert main([
        "--scenario", "wave", "--diagnostic-zero", "--max-steps", "1",
        "--output", str(out),
    ]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "runtime_diagnostic"
    assert summary["policy_verified"] is False
    assert summary["complete"] is False
    assert summary["truncated"] is True
    assert summary["conservation"]["requested_eq_spawned_plus_unspawned"] is True
    assert summary["conservation"]["spawned_eq_entered_plus_outside_plus_active"] is True
    with (out / "trajectories.csv").open(newline="", encoding="utf-8") as f:
        trajectory_rows = list(csv.DictReader(f))
    keys = [(row["time"], row["uid"]) for row in trajectory_rows]
    assert len(keys) == len(set(keys))
    assert {row["sigma"] for row in trajectory_rows} <= {"-1.0", "1.0"}
    assert (out / "events.csv").is_file()


def test_strict_incompatible_actor_state_rejected(tmp_path: Path, monkeypatch):
    import torch

    path = tmp_path / "bad.pth"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"wrong": torch.tensor(1.0)})
    with pytest.raises((RuntimeError, KeyError, TypeError)):
        StrictActorPolicy(path)
