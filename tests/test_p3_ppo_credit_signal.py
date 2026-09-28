from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.training import diagnose_p3_ppo_credit_signal as diagnostic


def _capture(step: int, *, reward: float = 0.0, terminal: bool = False, arrival_id: int = 12) -> dict:
    return {
        "step": step,
        "world": np.zeros((1, 88), dtype=np.float32),
        "actor_mask": np.asarray([True], dtype=bool),
        "requested_gate": np.asarray([0], dtype=np.int64),
        "requested_proposal": np.asarray([-1], dtype=np.int64),
        "gate_logprob": np.asarray([0.0], dtype=np.float32),
        "proposal_logprob": np.asarray([np.nan], dtype=np.float32),
        "value": 0.0,
        "post_value": 7.0,
        "next_value": 99.0 if terminal else 0.0,
        "reward": reward,
        "rewards_by_uav": {"42": reward},
        "reward_components": {"time": reward},
        "terminated": terminal,
        "episode_end": terminal,
        "identities": [{"uid": 42, "arrival_id": arrival_id, "phase": "ring_outer"}],
        "requested_gate_by_uid": {"42": 0},
        "release_probability_by_uid": {"42": 0.25},
        "phase_by_uid": {"42": "ring_outer"},
        "actor_count": 1,
    }


def _critical_captures(*, wrong_arrival_step: int | None = None) -> list[dict]:
    rows = [_capture(step) for step in range(105)]
    rows[-1]["reward"] = 1.0
    rows[-1]["rewards_by_uav"] = {"42": 1.0}
    rows[-1]["reward_components"] = {"time": 1.0}
    rows[-1]["terminated"] = True
    rows[-1]["episode_end"] = True
    rows[-1]["next_value"] = 99.0
    if wrong_arrival_step is not None:
        rows[wrong_arrival_step]["identities"][0]["arrival_id"] = 999
    return rows


def test_team_gae_is_called_and_terminal_transition_does_not_bootstrap(monkeypatch) -> None:
    calls: list[tuple[float, float]] = []
    original = diagnostic.ppo.team_gae

    def wrapped(steps, gamma, gae_lambda):
        calls.append((gamma, gae_lambda))
        return original(steps, gamma, gae_lambda)

    monkeypatch.setattr(diagnostic.ppo, "team_gae", wrapped)
    result = diagnostic.analyze_captures(_critical_captures())

    assert calls == [(0.99, 0.95), (1.0, 0.95)]
    assert result["gae"]["gamma_0_99"]["raw"]["advantage"][-1] == pytest.approx(1.0)
    assert result["gae"]["gamma_1_0"]["raw"]["advantage"][-1] == pytest.approx(1.0)


def test_cutoff_marks_episode_end_without_terminal() -> None:
    env = SimpleNamespace(
        now=2.0,
        config=SimpleNamespace(max_simulation_seconds=2.0),
        arrivals=None,
        active=None,
        admission=None,
        service_uavs=None,
    )
    assert diagnostic._global_terminal(env, {}) is False
    assert diagnostic._episode_end(env, False, {"truncated": {"42": True}}) is True


def test_mc_discount_changes_delayed_reward() -> None:
    discounted = diagnostic.mc_returns([0.0, 0.0, 1.0], 0.99)
    undiscounted = diagnostic.mc_returns([0.0, 0.0, 1.0], 1.0)

    assert discounted.tolist() == pytest.approx([0.9801, 0.99, 1.0])
    assert undiscounted.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_actor_only_normalization_and_zero_actor() -> None:
    normalized = diagnostic.actor_step_normalize([1.0, 3.0, 5.0], [True, False, True])
    assert normalized["actor_count"] == 2
    assert normalized["mean"] == pytest.approx(3.0)
    assert normalized["values"][0] == pytest.approx(-1.0)
    assert normalized["values"][2] == pytest.approx(1.0)

    zero = diagnostic.actor_step_normalize([1.0, 2.0], [False, False])
    assert zero["available"] is False
    assert zero["actor_count"] == 0
    assert zero["values"].tolist() == [1.0, 2.0]


def test_critical_identity_mismatch_fails() -> None:
    with pytest.raises(diagnostic.PPOCreditSignalError, match="critical arrival mismatch"):
        diagnostic.analyze_captures(_critical_captures(wrong_arrival_step=103))


def test_analyze_output_is_jsonable_and_contains_critical_rows() -> None:
    result = diagnostic.analyze_captures(_critical_captures())
    encoded = json.dumps(diagnostic._jsonable(result), allow_nan=False)
    assert isinstance(encoded, str)
    assert [row["step"] for row in result["critical_rows"]] == [102, 103, 104]
    assert result["discount_half_life"]["gamma_0_99_steps"] == pytest.approx(68.9675, rel=1e-4)


def _fake_lifecycle() -> dict:
    rows = [
        {
            "arrival_id": index,
            "uid": index,
            "mission_type": "standard",
            "outcome": "completed",
            "wait_seconds": 0.0,
            "base_priority": 1,
        }
        for index in range(24)
    ]
    return {
        "terminal": True,
        "all_requests_observed": True,
        "ledger_alignment": True,
        "request_rows": [{"arrival_id": index, "mission": "standard", "mission_type": "standard"} for index in range(24)],
        "rows": rows,
    }


def test_run_writes_numeric_json_with_injected_replay(tmp_path: Path, monkeypatch) -> None:
    tape = tmp_path / "source_tape.json"
    tape.write_text(json.dumps({"action_tape": [{"step": step, "a_requested": {}} for step in range(105)]}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    config = SimpleNamespace(total_requested_uavs=24, dt=1.0)
    records = [{"seed": 873103, "scenario": "dense24", "split": "train", "action_tape": str(tape)}]
    monkeypatch.setattr(diagnostic.collector, "_load_source_inputs", lambda root: ({}, {"dense24": config}, records, manifest))
    monkeypatch.setattr(diagnostic.collector, "_load_source_policy", lambda protocol, cfg: {"policy": object()})
    monkeypatch.setattr(diagnostic, "_model_digest", lambda policy: "digest")
    monkeypatch.setattr(
        diagnostic,
        "_capture_replay",
        lambda policy, config, tape_path, tape_rows, **kwargs: {
            "env": SimpleNamespace(telemetry={}),
            "replay": {"exact": True, "steps_replayed": 105},
            "captures": _critical_captures(),
            "lifecycle": _fake_lifecycle(),
            "request_rows": [],
        },
    )
    output = tmp_path / "output"
    result = diagnostic.run(output_dir=output, formal_root=tmp_path)

    assert result["status"] == "COMPLETED"
    for name in ("trace.json", "report.json", "progress.json"):
        assert (output / name).is_file()
        json.loads((output / name).read_text(encoding="utf-8"))
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    trace = json.loads((output / "trace.json").read_text(encoding="utf-8"))
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert report["source"]["steps_expected"] == report["analysis"]["steps"] == trace["steps_captured"] == progress["processed_steps"]
    assert report["analysis"]["reward_components"]["sum_matches_reward_total"] is True


def test_run_failure_keeps_failed_report(tmp_path: Path, monkeypatch) -> None:
    tape = tmp_path / "source_tape.json"
    tape.write_text(json.dumps({"action_tape": [{"step": 0, "a_requested": {}}]}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    config = SimpleNamespace(total_requested_uavs=24, dt=1.0)
    records = [{"seed": 873103, "scenario": "dense24", "split": "train", "action_tape": str(tape)}]
    monkeypatch.setattr(diagnostic.collector, "_load_source_inputs", lambda root: ({}, {"dense24": config}, records, manifest))
    monkeypatch.setattr(diagnostic.collector, "_load_source_policy", lambda protocol, cfg: {"policy": object()})
    monkeypatch.setattr(diagnostic, "_model_digest", lambda policy: "digest")
    monkeypatch.setattr(diagnostic, "_capture_replay", lambda *args: (_ for _ in ()).throw(RuntimeError("fixture failure")))
    output = tmp_path / "failed"
    result = diagnostic.run(output_dir=output, formal_root=tmp_path)

    assert result["status"] == "FAILED"
    assert json.loads((output / "report.json").read_text(encoding="utf-8"))["status"] == "FAILED"
    assert json.loads((output / "progress.json").read_text(encoding="utf-8"))["status"] == "FAILED"
    assert result["status"] != "COMPLETED"


def test_capture_replay_wraps_one_fake_step_and_serializes_capture(tmp_path: Path, monkeypatch) -> None:
    class FakeActor:
        committed_plan = None
        phase = "ring_outer"
        arrival_id = 12

    class FakeEnv:
        def __init__(self, config):
            self.config = config
            self.step_count = 0
            self.now = 0.0
            self.active = {42: FakeActor()}
            self.future_release_reservations = {}

        def _observations(self):
            return {"42": np.zeros(88, dtype=np.float32)}

        def step(self, actions, proposal_shortlists=None):
            self.step_count += 1
            self.now += 1.0
            return self._observations(), {"42": 2.0}, {}, {}, {
                "terminal": True,
                "reward_components": {"42": {"time": 2.0, "total": 2.0}},
            }

    config = SimpleNamespace(dt=1.0, total_requested_uavs=24, max_simulation_seconds=10.0)
    tape = tmp_path / "fake_tape.json"
    tape.write_text("{}", encoding="utf-8")
    tape_rows = [{
        "step": 0,
        "a_requested": {"42": [0.0, 0.0, 0.0]},
        "requested_gate": {"42": 0},
        "requested_proposal": {"42": -1},
    }]
    monkeypatch.setattr(diagnostic, "TerminalParallelEnv", FakeEnv)
    monkeypatch.setattr(diagnostic.student_rollout_v4_2, "world_value", lambda policy, world: 3.0)
    monkeypatch.setattr(diagnostic, "_policy_release_probabilities", lambda policy, world: np.asarray([0.4], dtype=np.float32))
    monkeypatch.setattr(diagnostic.terminal_lifecycle_metrics, "_capture_terminal_rows", lambda env: ([], {}))
    monkeypatch.setattr(diagnostic.terminal_credit, "_request_capture", lambda env: [])
    monkeypatch.setattr(diagnostic, "_request_lifecycle", lambda env, captured, first_seen, requests, config: _fake_lifecycle())

    calls = {"replay": 0}

    def fake_replay(env, path):
        calls["replay"] += 1
        env.step({"42": np.zeros(3, dtype=np.float32)}, {})
        return {"exact": True, "steps_replayed": 1}

    monkeypatch.setattr(diagnostic.SafetyTraceRecorder, "replay_action_tape", fake_replay)
    result = diagnostic._capture_replay(object(), config, tape, tape_rows)

    assert calls["replay"] == 1
    assert len(result["captures"]) == 1
    assert result["captures"][0]["reward"] == pytest.approx(2.0)
    assert diagnostic._jsonable(result["captures"][0])["release_probability_by_uid"]["42"] == pytest.approx(0.4)
