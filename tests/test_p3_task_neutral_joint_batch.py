from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.algo.student_ppo_v4_2 import StudentStepV42, team_gae
from terminal_marl.training import run_p3_task_neutral_joint_batch as runner


def _step(reward: float, *, episode_end: bool, terminated: bool, next_value: float = 0.0) -> StudentStepV42:
    return StudentStepV42(
        world=np.zeros((1, 88), dtype=np.float32),
        requested_gate=np.array([0], dtype=np.int64),
        requested_proposal=np.array([-1], dtype=np.int64),
        gate_logprob=np.array([0.0], dtype=np.float32),
        proposal_logprob=np.array([np.nan], dtype=np.float32),
        actor_mask=np.array([True], dtype=bool),
        value=0.0,
        next_value=float(next_value),
        reward=float(reward),
        terminated=bool(terminated),
        episode_end=bool(episode_end),
        stochastic_behavior=True,
    )


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        task_priority_enabled=False,
        low_level_priority_time_weight=0.0,
        max_simulation_seconds=1.0,
        dt=1.0,
    )


def test_joint_team_gae_respects_each_episode_boundary() -> None:
    first = [_step(1.0, episode_end=False, terminated=False, next_value=0.0),
             _step(2.0, episode_end=True, terminated=True)]
    second = [_step(3.0, episode_end=False, terminated=False, next_value=0.0),
              _step(4.0, episode_end=True, terminated=True)]
    combined_adv, combined_ret = team_gae(first + second, 0.99, 0.95)
    first_adv, first_ret = team_gae(first, 0.99, 0.95)
    second_adv, second_ret = team_gae(second, 0.99, 0.95)
    np.testing.assert_allclose(combined_adv, np.concatenate([first_adv, second_adv]))
    np.testing.assert_allclose(combined_ret, np.concatenate([first_ret, second_ret]))


def test_collect_six_passes_policy_seeds_and_retains_terminal_failure(monkeypatch, tmp_path) -> None:
    calls: list[tuple[int, int]] = []
    step = _step(1.0, episode_end=True, terminated=True)

    def fake_training_record(**kwargs):
        calls.append((int(kwargs["seed"]), int(kwargs["policy_seed"])))
        return [step], {"scenario": kwargs["scenario"], "seed": kwargs["seed"],
                        "policy_seed": kwargs["policy_seed"], "weights_unchanged": True}

    monkeypatch.setattr(runner.pilot, "_training_record", fake_training_record)
    monkeypatch.setattr(runner.pilot, "_record_annotations", lambda record, config: dict(record))
    monkeypatch.setattr(runner, "_training_stop_reason", lambda record, steps: None)
    monkeypatch.setattr(runner, "_training_cell_record", lambda record, steps, cell: {
        **dict(record), "cell": dict(cell), "terminal_failure_retained": True,
    })
    monkeypatch.setattr(runner.pilot, "_model_digest", lambda policy: "unchanged")
    policy = SimpleNamespace()
    all_steps, records, collection, reason = runner._collect_training(
        output=tmp_path,
        policy=policy,
        checkpoint=tmp_path / "candidate.pt",
        config=_config(),
    )
    assert reason is None
    assert len(records) == 6
    assert len(all_steps) == 6
    assert calls == [(int(cell["seed"]), int(cell["policy_seed"])) for cell in runner.TRAINING_CELLS]
    assert collection["model_digest_after_collection"] == "unchanged"
    assert all(record["terminal_failure_retained"] for record in records)


@pytest.mark.parametrize("freeze_ok", [True, False])
def test_joint_update_occurs_after_collection_and_only_once(monkeypatch, tmp_path, freeze_ok) -> None:
    events: list[str] = []
    config = _config()
    configs = {name: config for name in ("dense24", "flow6", "flow12")}
    step = _step(1.0, episode_end=True, terminated=True)
    policy = SimpleNamespace(model=SimpleNamespace(state_dict=lambda: {}), gate_reference=None)
    source_policy = SimpleNamespace(model=SimpleNamespace(state_dict=lambda: {}))

    monkeypatch.setattr(runner.neutral_eval, "_build_neutral_inputs", lambda: (configs, configs, {}))
    monkeypatch.setattr(runner, "_preflight", lambda neutral_configs: (
        tmp_path / "candidate.pt", policy, {"checkpoint_sha256": runner.EXPECTED_CANDIDATE_SHA256},
        source_policy, {"warmup_sha256": runner.WARMUP_CHECKPOINT_SHA256,
                        "warmup_path": str(tmp_path / "warmup.pt")},
        {"source_q_digest": "q"},
    ))
    monkeypatch.setattr(runner.pilot, "_configure_trainable_policy", lambda policy: {"scope": "test"})
    monkeypatch.setattr(runner, "_collect_training", lambda **kwargs: (
        events.append("collect") or [step] * 6,
        [{"scenario": "dense24", "seed": int(cell["seed"]), "weights_unchanged": True}
         for cell in runner.TRAINING_CELLS],
        {"model_digest_after_collection": "p"}, None,
    ))
    monkeypatch.setattr(runner.pilot, "_model_digest", lambda obj: "p" if obj is policy else "q")
    monkeypatch.setattr(runner.pilot, "actor_world_observation_weights", lambda steps: (
        np.zeros((len(steps), 88), dtype=np.float32),
        np.full(len(steps), 1.0 / len(steps), dtype=np.float32),
    ))
    monkeypatch.setattr(runner.pilot, "_build_gate_reference", lambda *args: object())
    monkeypatch.setattr(runner.pilot, "_gate_reference_digest", lambda reference: "gate")
    monkeypatch.setattr(runner.pilot, "_actor_gate_release_mean", lambda *args: 0.5)
    monkeypatch.setattr(runner.pilot, "_frozen_bitwise_check", lambda before, after: {
        "frozen_parameter_bitwise_unchanged": freeze_ok, "proposal_bitwise_unchanged": freeze_ok,
        "gate_temperature_bitwise_unchanged": freeze_ok,
    })
    def update(steps, **kwargs):
        events.append("update")
        assert len(steps) == 6
        assert kwargs["epochs"] == 2
        return {"updates": 1, "accepted_updates": 1, "rejected_updates": 0, "approx_kl": 0.0}
    policy.update = update
    stage_metadata: dict = {}
    def fake_roundtrip(*args):
        events.append("roundtrip")
        stage_metadata.update(args[3])
        return {"checkpoint_sha256": "after"}
    monkeypatch.setattr(runner.pilot, "_roundtrip_pilot", fake_roundtrip)
    monkeypatch.setattr(runner, "_evaluate_arm", lambda **kwargs: ([], None))
    monkeypatch.setattr(runner.neutral_eval, "_aggregate_arm", lambda records: {"records": len(records)})
    monkeypatch.setattr(runner, "_pair_evaluation_records", lambda before, after: [])

    report = runner.run(output_dir=tmp_path / "run")
    assert report["status"] == ("COMPLETED" if freeze_ok else "FAILED")
    assert events == (["collect", "update", "roundtrip"] if freeze_ok else ["collect", "update"])
    assert report["ppo_updates"] == 1
    assert report["training"]["update_calls"] == 1
    if freeze_ok:
        assert stage_metadata["repair_method"] == runner.REPAIR_METHOD
        assert stage_metadata["training_budget"]["updates"] == 1
        assert stage_metadata["training_budget"]["permutation_seed"] == runner.PERMUTATION_SEED
    else:
        assert "joint_checkpoint" not in report["artifacts"]
        assert report["evaluation_rollouts"] == 0


def test_terminal_failed_energy_is_not_a_training_stop_but_safety_is(monkeypatch) -> None:
    step = [_step(0.0, episode_end=True, terminated=True)]
    monkeypatch.setattr(runner.neutral_eval, "_record_metrics", lambda record: {
        "requests": 24, "technical_ok": True, "lifecycle_valid": True,
        "safety_nonzero": False, "failed_energy": 1,
    })
    record = {"scenario": "dense24", "weights_unchanged": True}
    assert runner._training_stop_reason(record, step) is None
    monkeypatch.setattr(runner.neutral_eval, "_record_metrics", lambda record: {
        "requests": 24, "technical_ok": True, "lifecycle_valid": True,
        "safety_nonzero": True, "failed_energy": 0,
    })
    assert runner._training_stop_reason(record, step) == "training rollout reported a safety violation"
    monkeypatch.setattr(runner.neutral_eval, "_record_metrics", lambda record: {
        "requests": 24, "technical_ok": False, "lifecycle_valid": False,
        "safety_nonzero": False, "failed_energy": 0,
    })
    assert runner._training_stop_reason(record, step) == "training lifecycle fields are incomplete"


def test_check_only_has_zero_runtime_and_update_calls(monkeypatch, tmp_path) -> None:
    config = _config()
    configs = {name: config for name in ("dense24", "flow6", "flow12")}
    policy = SimpleNamespace()
    source_policy = SimpleNamespace()
    monkeypatch.setattr(runner.neutral_eval, "_build_neutral_inputs", lambda: (configs, configs, {}))
    monkeypatch.setattr(runner, "_preflight", lambda neutral_configs: (
        tmp_path / "candidate.pt", policy, {"checkpoint_sha256": runner.EXPECTED_CANDIDATE_SHA256},
        source_policy, {"warmup_sha256": runner.WARMUP_CHECKPOINT_SHA256,
                        "warmup_path": str(tmp_path / "warmup.pt")},
        {"source_q_digest": "q"},
    ))
    calls = {"training": 0, "runtime": 0, "update": 0}
    monkeypatch.setattr(runner.pilot, "_training_record", lambda **kwargs: calls.__setitem__("training", calls["training"] + 1))
    report = runner.run(output_dir=tmp_path / "check", check_only=True)
    assert report["status"] == "CHECK_ONLY_OK"
    assert report["checks"]["runtime_rollout_calls"] == 0
    assert report["checks"]["update_calls"] == 0
    assert report["checks"]["env_steps"] == 0
    assert calls == {"training": 0, "runtime": 0, "update": 0}
