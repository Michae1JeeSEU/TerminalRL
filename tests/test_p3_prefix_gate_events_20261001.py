from __future__ import annotations

import contextlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from terminal_marl.training import diagnose_p3_prefix_gate_events_20261001 as runner


def _sidecar(keys):
    return {
        "registrations": {
            key: {"order": list(range(25)), "metadata": {"source_logits": [float(i) for i in range(25)]}}
            for key in keys
        }
    }


def _row(step, uid, *, rank=None, used=True, gate=1, obs_hash="obs"):
    runtime = {"used": used}
    if rank is not None:
        runtime["rank"] = rank
        runtime["candidates_checked"] = rank
    return {
        "step": step,
        "uid": uid,
        "requested_gate": {str(uid): gate},
        "requested_proposal": {str(uid): 10},
        "runtime_shortlist_result": {str(uid): runtime},
        "observation_hash": obs_hash,
        "gate_logprob": {str(uid): math.log(0.25)},
        "reward": -999.0,
        "terminal_outcome": "failed",
    }


def test_all_registered_requested_gate_events_include_first3_unknown_and_late():
    rows = [
        _row(3, 10, rank=2),
        _row(1, 7, rank=4),
        _row(2, 8, rank=None, used=False),
        _row(4, 9, rank=0, used=False),
        _row(5, 11, rank=5, gate=0),
    ]
    sidecar = _sidecar(["3|10", "1|7", "2|8", "4|9", "5|11"])
    got = runner.select_all_potential(rows, sidecar)
    assert [(x["step"], x["uid"], x["rank"], x["runtime_used"]) for x in got] == [
        (1, 7, 4, True), (2, 8, None, False), (3, 10, 2, True), (4, 9, None, False)
    ]


def test_unregistered_requested_gate_is_skipped_but_not_a_phase_error():
    rows = [_row(1, 7, rank=2), _row(2, 8, rank=4)]
    got = runner.select_all_potential(rows, _sidecar(["2|8"]))
    assert [(x["step"], x["uid"]) for x in got] == [(2, 8)]


@pytest.mark.parametrize(
    "context, expected",
    [
        ({"phase": "ring_outer", "soc": 0.5, "actor_mask": True, "retry_free": True, "committed_plan": False}, True),
        ({"phase": "ring_inner", "soc": 0.4, "actor_mask": True, "retry_free": True, "committed_plan": False}, False),
        ({"phase": "ring_middle", "soc": 0.7, "actor_mask": True, "retry_free": False, "committed_plan": False}, False),
        ({"phase": "ring_inner", "soc": 0.5, "actor_mask": True, "retry_free": True, "committed_plan": True}, False),
    ],
)
def test_normal_predicates_have_no_upper_soc_or_rank_gt3_filter(context, expected):
    state = runner._predicate_state(context, decision="approved", last_decision="approved", actual_rank=2)
    assert state["normal_eligible"] is expected
    assert state["rank_class"] == "first3"


def test_stale_approval_is_not_current_approval():
    context = {"phase": "ring_outer", "soc": 0.5, "actor_mask": True, "retry_free": True, "committed_plan": False}
    state = runner._predicate_state(context, decision="safety_wait", last_decision="approved", actual_rank=4)
    assert state["normal_eligible"] is False
    assert "current_info_approved" in state["false_predicates"]


def test_unknown_rank_is_descriptive_not_normal_exclusion():
    context = {"phase": "ring_inner", "soc": 0.55, "actor_mask": True, "retry_free": True, "committed_plan": False}
    state = runner._predicate_state(context, decision="approved", last_decision="approved", actual_rank=None)
    assert state["normal_eligible"] is True
    assert state["rank_class"] is None


def _fake_run_item():
    observations = {
        "3": np.zeros(88, dtype=np.float32),
        "4": np.ones(88, dtype=np.float32),
        "5": np.full(88, 2.0, dtype=np.float32),
        "6": np.full(88, 3.0, dtype=np.float32),
    }
    obs_hash = runner._digest({key: value for key, value in sorted(observations.items())})

    class Actor:
        def __init__(self, uid, arrival_id, phase, soc):
            self.uid = uid
            self.arrival_id = arrival_id
            self.phase = SimpleNamespace(value=phase)
            self.ring_role = SimpleNamespace(value=phase.replace("ring_", ""))
            self.soc = soc
            self.committed_plan = None
            self.safety_retry_after = 0.0

    class Env:
        instances = []

        def __init__(self, config):
            self.config = config
            self.active = {
                3: Actor(3, 103, "ring_outer", 0.55),
                4: Actor(4, 104, "ring_middle", 0.55),
                5: Actor(5, 105, "ring_inner", 0.30),
                6: Actor(6, 106, "ring_inner", 0.55),
            }
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
            return observations

        def step(self, actions, shortlists):
            step = self.step_count
            self.step_count += 1
            uid = [3, 4, 5, 6][step]
            decision = "safety_wait" if uid == 4 else "approved"
            # uid=4 deliberately has stale approved state in the environment.
            last = "approved" if uid == 4 else decision
            rank = {3: 2, 4: 4, 5: 5, 6: 6}[uid]
            self.last_release_decisions = {uid: last}
            self.last_release_telemetry = {uid: {"runtime_shortlist_rank": rank}}
            info = {
                "release_decisions": {uid: decision},
                "release_intent_telemetry": {uid: {"runtime_shortlist_rank": rank}},
            }
            return self._observations(), {}, {}, {}, info

    rows = [
        _row(0, 3, rank=2, obs_hash=obs_hash),
        _row(1, 4, rank=4, obs_hash=obs_hash),
        _row(2, 5, rank=5, obs_hash=obs_hash),
        _row(3, 6, rank=6, obs_hash=obs_hash),
    ]
    sidecar = _sidecar([f"{row['step']}|{row['uid']}" for row in rows])
    return Env, observations, {
        "cell_id": 1, "rows": rows, "potential": runner.select_all_potential(rows, sidecar),
        "sidecar": sidecar, "tape_path": "fake-tape", "tape_sha256": "tape", "sidecar_sha256": "sidecar",
        "full_sha256": "full", "arm_sha256": "arm",
    }


def test_fake_run_replays_once_and_records_all_events_across_phases(tmp_path: Path, monkeypatch):
    Env, observations, item = _fake_run_item()
    calls = {"replay": 0, "scope_enter": 0, "scope_exit": 0}

    class Policy:
        def __init__(self, label):
            self.label = label
            self.device = "cpu"
            self.model = object()

    @contextlib.contextmanager
    def fake_scope(env, sidecar):
        calls["scope_enter"] += 1
        try:
            yield SimpleNamespace()
        finally:
            calls["scope_exit"] += 1

    def fake_replay(env, path):
        calls["replay"] += 1
        env.reset(1)
        for _ in range(4):
            env.step({}, {})
        return {"exact": True, "steps_replayed": 4, "steps_expected": 4, "steps": []}

    def fake_distribution(policy, world):
        assert np.asarray(world).shape == (4, 88)
        q = np.arange(1, 26, dtype=np.float64)
        q /= q.sum()
        gate = 0.25 if policy.label == "before" else 0.5
        return {
            "gate_probs": np.tile([[1.0 - gate, gate]], (4, 1)),
            "proposal_q": np.tile(q, (4, 1)),
            "proposal_logits": np.tile(np.arange(25, dtype=np.float64), (4, 1)),
        }

    load_count = {"n": 0}

    def fake_load(path, config):
        load_count["n"] += 1
        torch.rand(1)
        return Policy("before" if load_count["n"] == 1 else "after"), {}

    monkeypatch.setattr(runner.base.frozen, "_CapturedTerminalEnv", Env)
    monkeypatch.setattr(runner.SafetyTraceRecorder, "replay_action_tape", fake_replay)
    monkeypatch.setattr(runner.base, "_controller_scope", fake_scope)
    monkeypatch.setattr(runner.base, "_distribution_snapshot", fake_distribution)
    monkeypatch.setattr(runner.rank_eval, "_model_digest", lambda policy: policy.label)
    monkeypatch.setattr(runner.rank_eval, "_load_policy", fake_load)

    before_rng = runner._rng_snapshot()
    result = runner._run_cell(item, SimpleNamespace(student_decision_mask_mode="legacy"), tmp_path)
    assert calls == {"replay": 1, "scope_enter": 1, "scope_exit": 1}
    assert runner._rng_equal(before_rng, runner._rng_snapshot())
    events = result["replay"]["events"]
    assert len(events) == 4
    assert result["replay"]["normal_event_count"] == 2
    assert events[0]["status"] == "NORMAL_ELIGIBLE" and events[0]["predicate_state"]["rank_class"] == "first3"
    assert events[1]["status"] == "EXCLUDED" and "current_info_approved" in events[1]["predicate_state"]["reasons"]
    assert events[2]["status"] == "EXCLUDED_AUTOEMERGENCY"
    assert events[3]["status"] == "NORMAL_ELIGIBLE" and events[3]["predicate_state"]["rank_class"] == "late_prefix_gt3"
    assert events[0]["comparison"]["q_bitwise_equal"] is True
    assert events[3]["comparison"]["delta_p"] == pytest.approx(0.25)
    assert events[0]["comparison"]["source_gate_logprob_check"]["checked_policy"] == "BEFORE"
    assert "after_gate_logprob" not in events[0]["comparison"]["source_gate_logprob_check"]
    assert result["caller_rng_unchanged_before_restore"] is True
    assert result["caller_rng_unchanged_after_restore"] is True
    assert events[0]["pre_world_matrix_88d"] and len(events[0]["pre_world_matrix_88d"]) == 4


def test_q_mismatch_stops_and_restores_scope_and_rng(tmp_path: Path, monkeypatch):
    Env, observations, item = _fake_run_item()
    exits = []

    class Policy:
        def __init__(self, label):
            self.label = label
            self.device = "cpu"
            self.model = object()

    @contextlib.contextmanager
    def fake_scope(env, sidecar):
        try:
            yield SimpleNamespace()
        finally:
            exits.append(True)

    def fake_replay(env, path):
        env.reset(1)
        for _ in range(4):
            env.step({}, {})
        return {"exact": True, "steps_replayed": 4, "steps_expected": 4, "steps": []}

    def mismatched(policy, world):
        q = np.arange(1, 26, dtype=np.float64)
        q /= q.sum()
        if policy.label == "after":
            q[0] += 1e-5
            q /= q.sum()
        return {"gate_probs": np.tile([[0.75, 0.25]], (4, 1)), "proposal_q": np.tile(q, (4, 1)),
                "proposal_logits": np.tile(np.arange(25, dtype=np.float64), (4, 1))}

    count = {"n": 0}

    def fake_load(path, config):
        count["n"] += 1
        torch.rand(1)
        return Policy("before" if count["n"] == 1 else "after"), {}

    monkeypatch.setattr(runner.base.frozen, "_CapturedTerminalEnv", Env)
    monkeypatch.setattr(runner.SafetyTraceRecorder, "replay_action_tape", fake_replay)
    monkeypatch.setattr(runner.base, "_controller_scope", fake_scope)
    monkeypatch.setattr(runner.base, "_distribution_snapshot", mismatched)
    monkeypatch.setattr(runner.rank_eval, "_model_digest", lambda policy: policy.label)
    monkeypatch.setattr(runner.rank_eval, "_load_policy", fake_load)
    before_rng = runner._rng_snapshot()
    with pytest.raises(runner.GateEventsError, match="proposal q mismatch"):
        runner._run_cell(item, SimpleNamespace(student_decision_mask_mode="legacy"), tmp_path)
    assert exits == [True]
    assert runner._rng_equal(before_rng, runner._rng_snapshot())


def test_q_failure_preserves_prior_events_in_cell_partial_and_restores_step(tmp_path: Path, monkeypatch):
    Env, observations, item = _fake_run_item()
    exits = []

    class Policy:
        def __init__(self, label):
            self.label = label
            self.device = "cpu"
            self.model = object()

    @contextlib.contextmanager
    def fake_scope(env, sidecar):
        try:
            yield SimpleNamespace()
        finally:
            exits.append(True)

    def fake_replay(env, path):
        env.reset(1)
        for _ in range(4):
            env.step({}, {})
        return {"exact": True, "steps_replayed": 4, "steps_expected": 4, "steps": []}

    dist_calls = {"n": 0}

    def q_fails_on_last_normal_event(policy, world):
        dist_calls["n"] += 1
        q = np.arange(1, 26, dtype=np.float64)
        q /= q.sum()
        if policy.label == "after" and dist_calls["n"] >= 4:
            q[0] += 1e-5
            q /= q.sum()
        return {
            "gate_probs": np.tile([[0.75, 0.25]], (4, 1)),
            "proposal_q": np.tile(q, (4, 1)),
            "proposal_logits": np.tile(np.arange(25, dtype=np.float64), (4, 1)),
        }

    count = {"n": 0}

    def fake_load(path, config):
        count["n"] += 1
        torch.rand(1)
        return Policy("before" if count["n"] == 1 else "after"), {}

    monkeypatch.setattr(runner.base.frozen, "_CapturedTerminalEnv", Env)
    monkeypatch.setattr(runner.SafetyTraceRecorder, "replay_action_tape", fake_replay)
    monkeypatch.setattr(runner.base, "_controller_scope", fake_scope)
    monkeypatch.setattr(runner.base, "_distribution_snapshot", q_fails_on_last_normal_event)
    monkeypatch.setattr(runner.rank_eval, "_model_digest", lambda policy: policy.label)
    monkeypatch.setattr(runner.rank_eval, "_load_policy", fake_load)
    before_rng = runner._rng_snapshot()
    with pytest.raises(runner.GateEventsError, match="proposal q mismatch"):
        runner._run_cell(item, SimpleNamespace(student_decision_mask_mode="legacy"), tmp_path)
    partial = json.loads((tmp_path / "cell_01_partial.json").read_text(encoding="utf-8"))
    assert partial["status"] == "PARTIAL"
    assert partial["events_recorded"] == 3
    assert partial["events"][0]["comparison"]["D"] is None
    assert partial["events"][0]["comparison"]["local_m"] is None
    assert partial["current_step"] == 3
    assert exits == [True]
    assert Env.instances[-1].step.__func__ is Env.step
    assert runner._rng_equal(before_rng, runner._rng_snapshot())


def test_invalid_after_gate_probability_fails_closed(tmp_path: Path, monkeypatch):
    _, _, item = _fake_run_item()

    class Policy:
        def __init__(self, label):
            self.label = label

    def invalid_after(policy, world):
        q = np.arange(1, 26, dtype=np.float64)
        q /= q.sum()
        return {
            "gate_probs": np.tile([[0.75, np.nan if policy.label == "after" else 0.25]], (4, 1)),
            "proposal_q": np.tile(q, (4, 1)),
            "proposal_logits": np.tile(np.arange(25, dtype=np.float64), (4, 1)),
        }

    monkeypatch.setattr(runner.base, "_distribution_snapshot", invalid_after)
    with pytest.raises(runner.GateEventsError, match="AFTER gate probability is invalid"):
        runner._event_comparison(
            item["potential"][0],
            {"index": 0, "full_world": np.zeros((4, 88), dtype=np.float32)},
            item["rows"][0], Policy("before"), Policy("after"),
        )


def test_model_constructor_rng_is_separate_and_json_unknowns_are_strict(tmp_path: Path, monkeypatch):
    Env, observations, item = _fake_run_item()
    class Policy:
        def __init__(self, label):
            self.label = label
            self.device = "cpu"
            self.model = object()
    monkeypatch.setattr(runner.base.frozen, "_CapturedTerminalEnv", Env)
    def fake_replay(env, path):
        env.reset(1)
        for _ in range(4): env.step({}, {})
        return {"exact": True, "steps_replayed": 4, "steps_expected": 4, "steps": []}
    monkeypatch.setattr(runner.SafetyTraceRecorder, "replay_action_tape", fake_replay)
    monkeypatch.setattr(runner.base, "_controller_scope", lambda env, sidecar: contextlib.nullcontext())
    def dist(policy, world):
        q = np.arange(1, 26, dtype=np.float64); q /= q.sum()
        p = 0.25 if policy.label == "before" else 0.5
        return {"gate_probs": np.tile([[1-p, p]], (4, 1)), "proposal_q": np.tile(q, (4, 1)),
                "proposal_logits": np.tile(np.arange(25, dtype=np.float64), (4, 1))}
    monkeypatch.setattr(runner.base, "_distribution_snapshot", dist)
    monkeypatch.setattr(runner.rank_eval, "_model_digest", lambda policy: policy.label)
    count = {"n": 0}
    def fake_load(path, config):
        count["n"] += 1; torch.rand(3); return Policy("before" if count["n"] == 1 else "after"), {}
    monkeypatch.setattr(runner.rank_eval, "_load_policy", fake_load)
    result = runner._run_cell(item, SimpleNamespace(student_decision_mask_mode="legacy"), tmp_path)
    encoded = json.dumps(runner._jsonable(result), allow_nan=False)
    assert json.loads(encoded)["comparison"]["D"] is None
    assert result["model_load_rng_consumed"] == {"before": True, "after": True}


def test_pilot_reuse_rejects_tampered_cell(tmp_path: Path):
    (tmp_path / "cells").mkdir()
    (tmp_path / "preflight.json").write_text('{"status":"PREFLIGHT_OK"}', encoding="utf-8")
    cell = tmp_path / "cells" / "cell_01.json"
    cell.write_text('{"status":"COMPLETED"}', encoding="utf-8")
    pilot = {"status": "PILOT_OK", "cell_ids": [1], "preflight_sha256": runner._sha256(tmp_path / "preflight.json"),
             "cell_sha256": runner._sha256(cell)}
    (tmp_path / "pilot.json").write_text(json.dumps(pilot), encoding="utf-8")
    assert runner._validate_pilot_reuse(tmp_path, {"status": "PREFLIGHT_OK"})["status"] == "PILOT_OK"
    cell.write_text('{"status":"COMPLETED","tampered":true}', encoding="utf-8")
    with pytest.raises(runner.GateEventsError, match="pilot cell binding changed"):
        runner._validate_pilot_reuse(tmp_path, {"status": "PREFLIGHT_OK"})


def test_serialized_pilot_requires_preflight_hash(tmp_path: Path):
    (tmp_path / "cells").mkdir()
    preflight = tmp_path / "preflight.json"
    preflight.write_text('{"status":"PREFLIGHT_OK"}', encoding="utf-8")
    cell = tmp_path / "cells" / "cell_01.json"
    cell.write_text('{"status":"COMPLETED"}', encoding="utf-8")
    (tmp_path / "pilot.json").write_text(json.dumps({
        "status": "PILOT_OK", "cell_ids": [1],
        "preflight_sha256": "tampered", "cell_sha256": runner._sha256(cell),
    }), encoding="utf-8")
    with pytest.raises(runner.GateEventsError, match="pilot preflight binding changed"):
        runner._validate_pilot_reuse(tmp_path, {"status": "PREFLIGHT_OK"})


def test_preflight_flags_are_no_forward_and_no_training():
    assert runner.PPO_ALLOWED is False
    assert runner.HELDOUT_USED is False
    assert runner.TEACHER_USED is False
    assert runner.OPTIMIZER_USED is False
    assert runner.NO_TRAINING is True and runner.OFF_POLICY is True
