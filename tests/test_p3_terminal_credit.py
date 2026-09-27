from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from terminal_marl.training import diagnose_p3_terminal_credit as runner


class _Arrivals:
    def __init__(self, env: "_ContinuationEnv") -> None:
        self.env = env

    def exhausted(self) -> bool:
        return self.env.step_count >= 2


class _ContinuationEnv:
    def __init__(self) -> None:
        self.step_count = 0
        self.now = 0.0
        self.active = {1: object()}
        self.service_uavs = {}
        self.admission = SimpleNamespace(queue=SimpleNamespace(items=[]))
        self.arrivals = _Arrivals(self)
        self.calls: list[dict[str, np.ndarray]] = []

    def step(self, actions, shortlists):
        self.calls.append({key: np.asarray(value, dtype=np.float32).copy() for key, value in actions.items()})
        self.step_count += 1
        self.now += 1.0
        if self.step_count >= 2:
            self.active = {}
        following = {"1": np.asarray([self.step_count], dtype=np.float32)} if self.active else {}
        rewards = {"1": 1.0}
        terminated = {"1": self.step_count >= 2}
        truncated = {"1": False}
        info = {
            "reward_components": {"1": {"progress": 1.0, "total": 1.0}},
            "time": self.now,
            "outcomes": {},
        }
        return following, rewards, terminated, truncated, info


def test_continuation_uses_new_observation_for_each_source_request(monkeypatch) -> None:
    env = _ContinuationEnv()
    seen: list[float] = []

    def request(_policy, _env, observations, _config):
        value = float(observations["1"][0])
        seen.append(value)
        return {"1": np.asarray([value, 0.0, 0.0], dtype=np.float32)}, {}, ()

    monkeypatch.setattr(runner, "_source_request", request)
    config = SimpleNamespace(runtime_proposal_shortlist_k=1)
    _following, transitions = runner._continue_source(
        policy=object(),
        env=env,
        observations={"1": np.asarray([0.0], dtype=np.float32)},
        config=config,
        horizon=5,
    )

    assert seen == [0.0, 1.0]
    assert [float(call["1"][0]) for call in env.calls] == [0.0, 1.0]
    assert [item["reward"] for item in transitions] == [1.0, 1.0]


def test_selection_is_four_train_states_per_registered_seed() -> None:
    rows = []
    for seed in runner.SOURCE_SEEDS:
        for step in range(5):
            rows.append(
                {
                    "seed": seed,
                    "split": "train",
                    "step": step,
                    "uid": step,
                    "arrival_id": step,
                    "observation": [0.0] * 88,
                    "source_proposal_logits": [0.0] * 25,
                    "certified_proposals": [] if step == 0 else [0],
                }
            )
    selected = runner._select_rows(rows)
    assert len(selected) == 12
    assert {seed: sum(int(row["seed"]) == seed for row in selected) for seed in runner.SOURCE_SEEDS} == {
        seed: 4 for seed in runner.SOURCE_SEEDS
    }
    assert all("selection_reason" in row and "q_k3" in row for row in selected)


def test_terminal_summary_keeps_normal_emergency_failed_and_requested_denominator(monkeypatch) -> None:
    rows = [
        {
            "arrival_id": 0,
            "base_priority": 1,
            "initial_soc": 0.8,
            "request_time": 0.0,
            "terminal_time": 10.0,
            "touchdown_time": 10.0,
            "completion_time": 10.0,
            "external_wait_time": 2.0,
            "internal_wait_time": 3.0,
            "wait_seconds": 5.0,
            "final_soc": 0.7,
            "outcome": "completed",
            "emergency_landing": False,
        },
        {
            "arrival_id": 1,
            "base_priority": 5,
            "initial_soc": 0.6,
            "request_time": 0.0,
            "terminal_time": 11.0,
            "touchdown_time": 11.0,
            "completion_time": 11.0,
            "external_wait_time": 1.0,
            "internal_wait_time": 2.0,
            "wait_seconds": 3.0,
            "final_soc": 0.5,
            "outcome": "emergency_landed",
            "emergency_landing": True,
        },
        {
            "arrival_id": 2,
            "base_priority": 3,
            "initial_soc": 0.7,
            "request_time": 0.0,
            "terminal_time": 12.0,
            "touchdown_time": None,
            "completion_time": None,
            "external_wait_time": 4.0,
            "internal_wait_time": 0.0,
            "wait_seconds": 4.0,
            "final_soc": 0.1,
            "outcome": "failed_energy",
            "emergency_landing": False,
        },
    ]
    env = SimpleNamespace(
        active={},
        service_uavs={},
        arrivals=SimpleNamespace(exhausted=lambda: True),
        ledger=SimpleNamespace(requested={0, 1, 2}, outcomes={0: "completed", 1: "emergency_landed", 2: "failed_energy"}),
    )
    monkeypatch.setattr(runner.terminal_lifecycle_metrics, "_completed_rows", lambda _env, _first_seen: rows)
    summary = runner._ledger_summary(
        env=env,
        captured=[],
        first_seen={},
        config=SimpleNamespace(total_requested_uavs=3),
    )
    assert summary["outcomes"] == {"completed": 1, "emergency_landed": 1, "failed_energy": 1}
    assert summary["requested_count"] == 3
    assert summary["all_requests_observed"] is True
    assert summary["ledger_alignment"] is True
    json.dumps(runner._jsonable(summary), allow_nan=False)


def test_reward_proxy_and_branch_copy_are_serializable() -> None:
    transitions = [
        {"reward": 2.0, "reward_components": {"progress": 2.0}},
        {"reward": -1.0, "reward_components": {"time": -1.0}},
    ]
    rewards = runner._aggregate_rewards(transitions)
    assert rewards["suffix_undiscounted_reward"] == 1.0
    assert rewards["suffix_discounted_reward"] == 2.0 - runner.GAMMA
    original = {"lifecycle": {"rows": [{"arrival_id": 0}], "outcomes": {"completed": 1}}}
    copied = runner._copy_branch(original, branch_name="runtime_certified_release", no_alternative=True)
    copied["lifecycle"]["rows"][0]["arrival_id"] = 7
    assert original["lifecycle"]["rows"][0]["arrival_id"] == 0
    json.dumps(runner._jsonable({"rewards": rewards, "branch": copied}), allow_nan=False)


class _BranchArrivals:
    def requests_until(self, _now: float) -> list[object]:
        return []


class _BranchEnv:
    def __init__(self) -> None:
        self.step_count = 0
        self.active = {1: object()}
        self.service_uavs = {}
        self.admission = SimpleNamespace(queue=SimpleNamespace(items=[]))
        self.arrivals = _BranchArrivals()
        self.telemetry = {
            "collisions": 0.0,
            "nmac": 0.0,
            "hard_separation_violations": 0.0,
            "checked_executed_plan_hash_mismatches": 0.0,
            "checked_executed_plan_matches": 1.0,
        }

    def _observations(self) -> dict[str, np.ndarray]:
        return {"1": np.zeros(88, dtype=np.float32)}

    def step(self, _actions, _shortlists):
        self.step_count += 1
        self.active = {}
        return (
            {},
            {"1": 0.0},
            {"1": True},
            {"1": False},
            {"time": float(self.step_count), "reward_components": {"1": {"total": 0.0}}},
        )


def test_branch_continuation_starts_from_same_rng_and_restores_parent(monkeypatch) -> None:
    continuation_rngs = []

    def continue_stub(*, policy, env, observations, config, horizon):
        del policy, env, observations, config, horizon
        continuation_rngs.append(runner._rng_snapshot())
        return {}, []

    monkeypatch.setattr(runner, "_continue_source", continue_stub)
    monkeypatch.setattr(runner, "_state_digest", lambda _env: "base")
    monkeypatch.setattr(runner, "_model_digest", lambda _policy: "weights")
    monkeypatch.setattr(runner, "_validate_target", lambda _env, _row: np.zeros(88, dtype=np.float32))
    monkeypatch.setattr(runner.action_credit_diagnostic.diagnostic, "_restore_shadow_methods", lambda _env: None)
    monkeypatch.setattr(
        runner.terminal_lifecycle_metrics,
        "_capture_terminal_rows",
        lambda _env: ([], {}),
    )
    monkeypatch.setattr(
        runner,
        "_ledger_summary",
        lambda **_kwargs: {
            "rows": [],
            "outcomes": {"completed": 1},
            "all_requests_observed": True,
            "ledger_alignment": True,
        },
    )
    monkeypatch.setattr(runner, "_real_terminal", lambda _env: True)
    monkeypatch.setattr(
        runner.p3_action_credit,
        "branch_inputs",
        lambda actions, shortlists, _uid, proposal, _logits, shortlist_k=3: (
            dict(actions),
            dict(shortlists),
        ),
    )

    branch_rng = runner._seeded_rng(20260928)
    parent_rng = runner._seeded_rng(1117)
    common = {
        "policy": object(),
        "base_env": _BranchEnv(),
        "row": {"uid": 1, "arrival_id": 1, "certified_proposals": []},
        "source_actions": {"1": np.zeros(3, dtype=np.float32)},
        "source_shortlists": {},
        "logits": np.zeros(25, dtype=np.float32),
        "branch_rng": branch_rng,
        "config": SimpleNamespace(runtime_proposal_shortlist_k=1),
        "horizon": 5,
    }

    wait = runner._run_branch(branch_name="wait", **common)
    assert runner._rng_equal(parent_rng, runner._rng_snapshot())
    runner._restore_rng(parent_rng)
    source_release = runner._run_branch(branch_name="source_release", **common)

    assert runner._rng_equal(parent_rng, runner._rng_snapshot())
    assert wait["rng_restored"] is True
    assert source_release["rng_restored"] is True
    assert source_release["proposal"] is not None
    assert len(continuation_rngs) == 2
    assert runner._rng_equal(continuation_rngs[0], continuation_rngs[1])
    assert source_release["safety"]["checked_executed_plan_matches"] == 1.0
    assert source_release["safety"]["uncertified_execution"]["available"] is False
    assert source_release["safety"]["uncertified_execution"]["after"] is None
    source_release["arrival_signature_match"] = True
    assert runner._branch_acceptance(source_release) is True


def _comparison_branch(*, priority: float, low_count: int, low_wait: float | None) -> dict:
    return {
        "lifecycle": {
            "outcomes": {"completed": 1},
            "priority_weighted_wait_seconds": priority,
            "low_soc": {"count": low_count, "mean_wait_seconds": low_wait},
        }
    }


def test_compare_pair_pareto_zero_low_soc_and_repeat_disagreement() -> None:
    wait = _comparison_branch(priority=10.0, low_count=2, low_wait=5.0)
    release = _comparison_branch(priority=10.0, low_count=2, low_wait=4.0)
    low_soc_only = runner._compare_pair(wait, release)
    assert low_soc_only["direction"] == "release"
    assert low_soc_only["reason"] == "wait_pareto"

    tradeoff = runner._compare_pair(
        _comparison_branch(priority=9.0, low_count=2, low_wait=4.0),
        _comparison_branch(priority=8.0, low_count=2, low_wait=5.0),
    )
    assert tradeoff["direction"] == "tie"

    priority_only = runner._compare_pair(
        _comparison_branch(priority=10.0, low_count=0, low_wait=None),
        _comparison_branch(priority=8.0, low_count=0, low_wait=None),
    )
    assert priority_only["direction"] == "release"
    assert priority_only["low_soc"]["comparable"] is False

    repeat_summary = runner._summarize_states(
        [
            {"state_index": 0, "seed": 873101, "step": 190, "uid": 2, "label": "wait"},
            {"state_index": 0, "seed": 873101, "step": 190, "uid": 2, "label": "release"},
        ]
    )
    assert repeat_summary["states"][0]["label"] == "uncertain"
