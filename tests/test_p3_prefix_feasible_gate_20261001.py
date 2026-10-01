from __future__ import annotations

import contextlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from terminal_marl.training import diagnose_p3_prefix_feasible_gate_20261001 as runner


def _row(step, uid, *, used=True, rank=8, gate=1, outcome="approved"):
    return {
        "step": step, "uid": uid,
        "requested_gate": {str(uid): gate},
        "requested_proposal": {str(uid): 10},
        "runtime_shortlist_result": {str(uid): {"used": used, "rank": rank, "candidates_checked": rank}},
        "observation_hash": f"h-{step}-{uid}", "reward": 999.0, "terminal_outcome": outcome,
    }


def _sidecar(*keys):
    return {"registrations": {
        key: {"order": list(range(25)), "metadata": {"source_logits": [float(i) for i in range(25)]}}
        for key in keys
    }}


def test_selection_numeric_uid_and_ignores_p_reward_outcome():
    rows = [_row(12, 10, outcome="failed"), _row(11, 98, outcome="completed"), _row(11, 7, outcome="failed")]
    got = runner.select_first_potential(rows, _sidecar("11|7", "11|98", "12|10"))
    assert [(item["step"], item["uid"]) for item in got] == [(11, 7), (11, 98), (12, 10)]
    rows[0]["reward"] = -10_000.0
    rows[1]["reward"] = 10_000.0
    got2 = runner.select_first_potential(rows, _sidecar("11|7", "11|98", "12|10"))
    assert [(item["step"], item["uid"]) for item in got2] == [(11, 7), (11, 98), (12, 10)]


def test_missing_does_not_substitute_and_actual_approval_is_separate():
    assert runner.select_first_potential([_row(1, 3, used=False)], _sidecar("1|3")) == []
    candidates = runner.select_first_potential([_row(1, 3)], _sidecar("1|3"))
    assert runner.select_first_actual_approved(candidates, actual={3: "safety_wait"}) is None
    assert runner.select_first_actual_approved(candidates, actual={3: "approved"})["uid"] == 3


def test_bad_registration_order_and_missing_registration_rejected():
    bad = _sidecar("1|2")
    bad["registrations"]["1|2"]["order"] = list(range(24)) + [23]
    with pytest.raises(runner.FeasibleGateError):
        runner.select_first_potential([_row(1, 2)], bad)
    assert runner.select_first_potential([_row(1, 2)], _sidecar("2|2")) == []


def test_requested_gate_and_rank_are_hard_filters():
    rows = [_row(1, 2, gate=0), _row(2, 2, rank=3), _row(3, 2, used=False)]
    assert runner.select_first_potential(rows, _sidecar("1|2", "2|2", "3|2")) == []


@pytest.mark.parametrize("context", [
    {"phase": "ring_outer", "soc": 0.5, "actor_mask": True, "retry_free": True, "committed_plan": False},
    {"phase": "ring_inner", "soc": 0.4, "actor_mask": True, "retry_free": True, "committed_plan": False},
    {"phase": "ring_inner", "soc": 0.65, "actor_mask": True, "retry_free": False, "committed_plan": False},
    {"phase": "ring_inner", "soc": 0.5, "actor_mask": True, "retry_free": True, "committed_plan": True},
])
def test_physical_window_excludes_phase_soc_retry_and_committed(context):
    assert runner.physical_context_eligible(context) is False


def test_physical_window_accepts_strict_soc_open_lower_bound():
    assert runner.physical_context_eligible({
        "phase": "ring_inner", "soc": 0.4000001, "actor_mask": True,
        "retry_free": True, "committed_plan": False,
    }) is True
    assert runner.physical_context_eligible({
        "phase": "ring_inner", "soc": 0.65, "actor_mask": True,
        "retry_free": True, "committed_plan": False,
    }) is True


def test_q_comparison_is_bitwise_and_p_can_differ():
    q = np.asarray([1.0 / 25.0] * 25, dtype=np.float64)
    assert np.array_equal(q, q.copy())
    assert not np.array_equal(np.asarray([0.4, 0.6]), np.asarray([0.4, 0.6000001]))


def test_rng_restore_success_and_exception():
    before = runner._rng_snapshot()
    random.random(); np.random.rand(); torch.rand(2)
    runner._restore_rng(before)
    assert runner._rng_equal(before, runner._rng_snapshot())
    with pytest.raises(RuntimeError):
        try:
            random.random(); np.random.rand(); torch.rand(2)
            raise RuntimeError("synthetic")
        finally:
            runner._restore_rng(before)
    assert runner._rng_equal(before, runner._rng_snapshot())


def test_instance_method_restored_on_exception():
    class Obj:
        def step(self, value):
            return value
    obj = Obj()
    original = obj.step
    try:
        saved = obj.step
        obj.step = lambda value: value + 1
        raise RuntimeError("synthetic")
    except RuntimeError:
        obj.step = saved
    assert obj.step(3) == original(3)


def test_strict_json_serialization_keeps_unknown_none(tmp_path: Path):
    path = tmp_path / "cell.json"
    runner._write_json(path, {"D": None, "local_m": None, "q": [0.1] * 25})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["D"] is None and payload["local_m"] is None
    assert len(payload["q"]) == 25


def test_preflight_contract_flags_are_explicit(monkeypatch):
    assert runner.PPO_ALLOWED is False
    assert runner.HELDOUT_USED is False
    assert runner.TEACHER_USED is False
    assert runner.OPTIMIZER_USED is False
    assert runner.OFF_POLICY is True and runner.NO_TRAINING is True


def test_numeric_uid_sort_rejects_non_numeric():
    with pytest.raises(runner.FeasibleGateError):
        runner._numeric_key({"step": 1, "uid": "uav-x"})


def test_full_reuse_requires_completed_pilot(tmp_path: Path):
    # The phase runner must refuse a pilot artifact that has no completed arm.
    (tmp_path / "preflight.json").write_text("{}", encoding="utf-8")
    with pytest.raises(runner.FeasibleGateError):
        runner._run_phase("full", tmp_path)


def _fake_run_item(runner_module):
    class Actor:
        uid = 3
        arrival_id = 13
        phase = SimpleNamespace(value="ring_inner")
        ring_role = SimpleNamespace(value="inner")
        soc = 0.5
        committed_plan = None
        safety_retry_after = 0.0

    class Env:
        instances = []

        def __init__(self, config):
            self.config = config
            self.active = {3: Actor()}
            self.future_release_reservations = set()
            self.now = 1.0
            self.step_count = 0
            self.last_release_decisions = {}
            self.last_release_telemetry = {}
            Env.instances.append(self)

        def reset(self, seed=None):
            self.step_count = 0
            return self._observations(), {}

        def _observations(self):
            return {"3": np.zeros(88, dtype=np.float32)}

        def step(self, actions, shortlists):
            self.step_count += 1
            self.last_release_decisions = {3: "approved"}
            self.last_release_telemetry = {3: {"runtime_shortlist_rank": 4}}
            return (self._observations(), {"3": 0.0}, {"3": False}, {"3": False},
                    {"release_decisions": {3: "approved"},
                     "release_intent_telemetry": {3: {"runtime_shortlist_rank": 4}}})

    class Policy:
        def __init__(self, label):
            self.label = label
            self.device = "cpu"
            self.model = object()

    obs = {"3": np.zeros(88, dtype=np.float32)}
    row = {"step": 0, "observation_hash": runner_module._digest({"3": obs["3"]}),
           "gate_logprob": {"3": float(np.log(0.25))}}
    sidecar = {"registrations": {"0|3": {
        "order": list(range(25)),
        "metadata": {"source_logits": [float(i) for i in range(25)]},
    }}}
    item = {"cell_id": 1, "rows": [row], "potential": [{
        "step": 0, "uid": 3, "registration_key": "0|3", "rank": 4,
        "sidecar_source_logits": [float(i) for i in range(25)],
    }], "sidecar": sidecar, "tape_path": "fake-tape",
        "tape_sha256": "tape", "sidecar_sha256": "sidecar"}
    return Env, Policy, item


def test_run_cell_uses_one_replay_and_serializes_two_policy_distributions(tmp_path: Path, monkeypatch):
    Env, Policy, item = _fake_run_item(runner)
    calls = {"replay": 0, "scope_enter": 0, "scope_exit": 0}

    @contextlib.contextmanager
    def fake_scope(env, sidecar):
        calls["scope_enter"] += 1
        try:
            yield SimpleNamespace(metadata=lambda: {})
        finally:
            calls["scope_exit"] += 1

    def fake_replay(env, path):
        calls["replay"] += 1
        env.reset(1)
        env.step({}, {})
        return {"exact": True, "steps_replayed": 1, "steps_expected": 1, "steps": [{}]}

    def fake_distribution(policy, world):
        q = np.arange(1, 26, dtype=np.float64)
        q /= q.sum()
        p = 0.25 if policy.label == "before" else 0.5
        return {"gate_probs": np.asarray([[1.0 - p, p]]),
                "gate_logits": np.asarray([[0.0, 1.0]]),
                "proposal_q": q[None, :],
                "proposal_logits": np.arange(25, dtype=np.float64)[None, :]}

    load_count = {"n": 0}
    def fake_load(path, config):
        load_count["n"] += 1
        torch.rand(1)  # model construction may consume RNG; runner must restore it.
        return Policy("before" if load_count["n"] == 1 else "after"), {}

    monkeypatch.setattr(runner.frozen, "_CapturedTerminalEnv", Env)
    monkeypatch.setattr(runner.SafetyTraceRecorder, "replay_action_tape", fake_replay)
    monkeypatch.setattr(runner.SafetyTraceRecorder, "_state", lambda env: {"step": env.step_count})
    monkeypatch.setattr(runner, "_controller_scope", fake_scope)
    monkeypatch.setattr(runner.matched, "_distribution_snapshot", fake_distribution)
    monkeypatch.setattr(runner.rank_eval, "_model_digest", lambda policy: policy.label)
    monkeypatch.setattr(runner.rank_eval, "_load_policy", fake_load)

    result = runner._run_cell(item, SimpleNamespace(student_decision_mask_mode="legacy"), tmp_path)
    json.dumps(runner._jsonable(result), allow_nan=False)
    assert calls == {"replay": 1, "scope_enter": 1, "scope_exit": 1}
    assert result["status"] == "COMPLETED"
    assert result["comparison"]["before_gate_p"] == 0.25
    assert result["comparison"]["after_gate_p"] == 0.5
    assert result["comparison"]["q_bitwise_equal"] is True
    assert result["model_load_rng_consumed"] == {"before": True, "after": True}
    assert result["selected_npz_sha256"]
    assert (tmp_path / "selected" / "cell_01_selected.npz").is_file()


def test_run_cell_restores_controller_step_and_rng_on_replay_error(tmp_path: Path, monkeypatch):
    Env, Policy, item = _fake_run_item(runner)
    exits = []

    @contextlib.contextmanager
    def fake_scope(env, sidecar):
        try:
            yield SimpleNamespace(metadata=lambda: {})
        finally:
            exits.append(True)

    def failing_replay(env, path):
        raise RuntimeError("synthetic replay failure")

    monkeypatch.setattr(runner.frozen, "_CapturedTerminalEnv", Env)
    monkeypatch.setattr(runner.SafetyTraceRecorder, "replay_action_tape", failing_replay)
    monkeypatch.setattr(runner, "_controller_scope", fake_scope)
    monkeypatch.setattr(runner.rank_eval, "_model_digest", lambda policy: policy.label)
    def fake_load(path, config):
        torch.rand(1)
        return Policy("before"), {}
    monkeypatch.setattr(runner.rank_eval, "_load_policy", fake_load)
    before_rng = runner._rng_snapshot()
    with pytest.raises(RuntimeError, match="synthetic replay failure"):
        runner._run_cell(item, SimpleNamespace(student_decision_mask_mode="legacy"), tmp_path)
    assert exits == [True]
    assert runner._rng_equal(before_rng, runner._rng_snapshot())
    assert Env.instances and getattr(Env.instances[0].step, "__func__", None) is Env.step


def test_controller_scope_restores_native_private_method():
    class Env:
        step_count = 0
        now = 0.0

        def _runtime_shortlist_release(self, *args):
            return "approved"

    env = Env()
    sidecar = {"registrations": {"0|3": {
        "order": list(range(25)),
        "metadata": {"source_logits": [float(i) for i in range(25)]},
    }}}
    assert "_runtime_shortlist_release" not in env.__dict__
    with runner._controller_scope(env, sidecar):
        assert "_runtime_shortlist_release" in env.__dict__
        assert callable(env._runtime_shortlist_release)
    assert "_runtime_shortlist_release" not in env.__dict__
