from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.algo.student_ppo_v4_2 import StudentStepV42, team_gae
from terminal_marl.training import run_p3_task_neutral_joint_batch as runner


def _step(
    reward: float, *, episode_end: bool, terminated: bool, next_value: float = 0.0,
    actor: bool = True, value: float = 0.0,
) -> StudentStepV42:
    return StudentStepV42(
        world=np.zeros((1, 88), dtype=np.float32),
        requested_gate=np.array([0], dtype=np.int64),
        requested_proposal=np.array([-1], dtype=np.int64),
        gate_logprob=np.array([0.0], dtype=np.float32),
        proposal_logprob=np.array([np.nan], dtype=np.float32),
        actor_mask=np.array([actor], dtype=bool),
        value=float(value),
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


def test_collect_training_accepts_explicit_batch_cells_without_changing_default(monkeypatch, tmp_path) -> None:
    calls: list[tuple[int, int]] = []
    step = _step(1.0, episode_end=True, terminated=True)
    selected = (
        {"scenario": "dense24", "seed": 880101, "policy_seed": 880201, "split": "curve_01"},
        {"scenario": "dense24", "seed": 880102, "policy_seed": 880202, "split": "curve_01"},
    )

    def fake_training_record(**kwargs):
        calls.append((int(kwargs["seed"]), int(kwargs["policy_seed"])))
        return [step], {"scenario": kwargs["scenario"], "seed": kwargs["seed"],
                        "policy_seed": kwargs["policy_seed"], "weights_unchanged": True}

    monkeypatch.setattr(runner.pilot, "_training_record", fake_training_record)
    monkeypatch.setattr(runner.pilot, "_record_annotations", lambda record, config: dict(record))
    monkeypatch.setattr(runner, "_training_stop_reason", lambda record, steps: None)
    monkeypatch.setattr(runner.pilot, "_model_digest", lambda policy: "unchanged")
    policy = SimpleNamespace()
    all_steps, records, collection, reason = runner._collect_training(
        output=tmp_path,
        policy=policy,
        checkpoint=tmp_path / "candidate.pt",
        config=_config(),
        training_cells=selected,
    )
    assert reason is None
    assert calls == [(880101, 880201), (880102, 880202)]
    assert len(records) == len(selected)
    assert len(all_steps) == len(selected)
    assert collection["bitwise_unchanged"] is True


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


def test_full_episode_mc_returns_reset_at_episode_boundaries_and_include_no_actor_tail() -> None:
    first_values = [5.0] + [6.0 + 0.1 * index for index in range(60)]
    first = []
    for index, value in enumerate(first_values):
        first.append(_step(
            1.0 if index == 0 else 25.0 if index == 60 else 0.0,
            episode_end=index == 60,
            terminated=index == 60,
            next_value=0.0 if index == 60 else first_values[index + 1],
            actor=index == 0,
            value=value,
        ))
    second = [
        _step(2.0, episode_end=False, terminated=False, next_value=4.0, value=3.0),
        _step(7.0, episode_end=True, terminated=True, value=4.0),
    ]
    combined = first + second
    expected = np.asarray([26.0] + [25.0] * 60 + [9.0, 7.0], dtype=np.float32)
    np.testing.assert_allclose(runner._full_episode_mc_returns(combined), expected)
    _, returns = team_gae(combined, gamma=1.0, gae_lambda=1.0)
    np.testing.assert_allclose(returns, expected)


def test_default_credit_kwargs_are_unchanged_and_full_mode_is_explicit() -> None:
    assert runner._credit_update_kwargs() == dict(runner.pilot.UPDATE_KWARGS)
    full = runner._credit_update_kwargs(full_episode_credit=True)
    assert full["gamma"] == 1.0
    assert full["gae_lambda"] == 1.0
    for key in ("epochs", "minibatch_size", "clip_ratio", "value_coef", "entropy_coef", "target_kl"):
        assert full[key] == runner.pilot.UPDATE_KWARGS[key]


def test_joint_team_actor_kwargs_keep_original_credit_and_set_reduction() -> None:
    joint = runner._credit_update_kwargs(joint_team_actor=True)
    assert joint["gamma"] == runner.pilot.UPDATE_KWARGS["gamma"] == 0.99
    assert joint["gae_lambda"] == runner.pilot.UPDATE_KWARGS["gae_lambda"] == 0.95
    assert joint["actor_reduction"] == "joint_team"


def test_joint_team_actor_cli_flag_and_credit_modes_are_mutually_exclusive(monkeypatch, tmp_path, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "CHECK_ONLY_OK"}

    monkeypatch.setattr(runner, "run", fake_run)
    assert runner.main([
        "--output-dir", str(tmp_path / "joint"), "--check-only", "--joint-team-actor",
    ]) == 0
    assert captured["joint_team_actor"] is True
    assert captured["full_episode_credit"] is False
    with pytest.raises(SystemExit):
        runner.main([
            "--output-dir", str(tmp_path / "both"), "--full-episode-credit", "--joint-team-actor",
        ])
    assert "not allowed with" in capsys.readouterr().err


@pytest.mark.parametrize("mc_match", [True, False])
def test_full_mode_passes_credit_kwargs_and_records_mc_match(monkeypatch, tmp_path, mc_match) -> None:
    config = _config()
    configs = {name: config for name in ("dense24", "flow6", "flow12")}
    step = _step(1.0, episode_end=True, terminated=True)
    policy = SimpleNamespace(model=SimpleNamespace(state_dict=lambda: {}), gate_reference=None)
    source_policy = SimpleNamespace(model=SimpleNamespace(state_dict=lambda: {}))
    captured: dict[str, object] = {}

    monkeypatch.setattr(runner.neutral_eval, "_build_neutral_inputs", lambda: (configs, configs, {}))
    monkeypatch.setattr(runner, "_preflight", lambda neutral_configs: (
        tmp_path / "candidate.pt", policy, {"checkpoint_sha256": runner.EXPECTED_CANDIDATE_SHA256},
        source_policy, {"warmup_sha256": runner.WARMUP_CHECKPOINT_SHA256,
                        "warmup_path": str(tmp_path / "warmup.pt")},
        {"source_q_digest": "q"},
    ))
    monkeypatch.setattr(runner.pilot, "_configure_trainable_policy", lambda policy: {"scope": "test"})

    def collect(**kwargs):
        captured["collect_kwargs"] = kwargs
        records = [{"scenario": cell["scenario"], "seed": cell["seed"], "weights_unchanged": True}
                   for cell in runner.TRAINING_CELLS]
        return [step] * 6, records, {"model_digest_after_collection": "p"}, None

    monkeypatch.setattr(runner, "_collect_training", collect)
    monkeypatch.setattr(runner.pilot, "_model_digest", lambda obj: "p" if obj is policy else "q")
    monkeypatch.setattr(runner.pilot, "actor_world_observation_weights", lambda steps: (
        np.zeros((len(steps), 88), dtype=np.float32),
        np.full(len(steps), 1.0 / len(steps), dtype=np.float32),
    ))
    monkeypatch.setattr(runner.pilot, "_build_gate_reference", lambda *args: object())
    monkeypatch.setattr(runner.pilot, "_gate_reference_digest", lambda reference: "gate")
    monkeypatch.setattr(runner.pilot, "_actor_gate_release_mean", lambda *args: 0.5)
    monkeypatch.setattr(runner.pilot, "_frozen_bitwise_check", lambda before, after: {
        "frozen_parameter_bitwise_unchanged": True,
        "proposal_bitwise_unchanged": True,
        "gate_temperature_bitwise_unchanged": True,
    })
    if not mc_match:
        monkeypatch.setattr(
            runner, "_full_episode_mc_returns",
            lambda steps: np.full(len(steps), 999.0, dtype=np.float32),
        )

    def update(steps, **kwargs):
        captured["update_kwargs"] = kwargs
        return {"updates": 1, "accepted_updates": 1, "rejected_updates": 0, "approx_kl": 0.0}

    policy.update = update
    stage_metadata: dict[str, object] = {}

    def fake_roundtrip(*args):
        stage_metadata.update(args[3])
        return {"checkpoint_sha256": "after"}

    monkeypatch.setattr(runner.pilot, "_roundtrip_pilot", fake_roundtrip)
    monkeypatch.setattr(runner, "_evaluate_arm", lambda **kwargs: ([], None))
    monkeypatch.setattr(runner.neutral_eval, "_aggregate_arm", lambda records: {"records": len(records)})
    monkeypatch.setattr(runner, "_pair_evaluation_records", lambda before, after: [])

    report = runner.run(output_dir=tmp_path / "full", full_episode_credit=True)
    assert report["status"] == ("COMPLETED" if mc_match else "FAILED")
    assert report["credit_method"] == "full_episode_mc"
    assert report["binding"]["credit_gamma"] == 1.0
    assert report["binding"]["credit_gae_lambda"] == 1.0
    assert captured["collect_kwargs"]["update_kwargs"]["gamma"] == 1.0
    assert captured["collect_kwargs"]["update_kwargs"]["gae_lambda"] == 1.0
    if mc_match:
        assert captured["update_kwargs"]["gamma"] == 1.0
        assert captured["update_kwargs"]["gae_lambda"] == 1.0
        assert report["training"]["pooled"]["full_episode_mc_return_match"] is True
        assert report["training"]["pooled"]["full_episode_mc_return_max_abs_error"] == 0.0
        assert stage_metadata["credit_method"] == "full_episode_mc"
        assert stage_metadata["training_budget"]["ppo_update_kwargs"]["gamma"] == 1.0
        assert report["evaluation"]["kind"] == "development_existing_scenes"
        assert report["evaluation"]["fresh_evaluation"] is False
    else:
        assert "update_kwargs" not in captured
        assert report["ppo_updates"] == 0
        assert report["training"]["pooled"]["full_episode_mc_return_match"] is False


@pytest.mark.parametrize("mode", ["full_episode_credit", "joint_team_actor"])
def test_reuse_check_only_reports_planned_and_actual_rollouts_separately(monkeypatch, tmp_path, mode) -> None:
    config = _config()
    configs = {name: config for name in ("dense24", "flow6", "flow12")}
    policy = SimpleNamespace()
    source_policy = SimpleNamespace()
    reuse_root = tmp_path / "before"
    reuse_root.mkdir()
    monkeypatch.setattr(runner.neutral_eval, "_build_neutral_inputs", lambda: (configs, configs, {}))
    monkeypatch.setattr(runner, "_preflight", lambda neutral_configs: (
        tmp_path / "candidate.pt", policy, {"checkpoint_sha256": runner.EXPECTED_CANDIDATE_SHA256},
        source_policy, {"warmup_sha256": runner.WARMUP_CHECKPOINT_SHA256,
                        "warmup_path": str(tmp_path / "warmup.pt")},
        {"source_q_digest": "q"},
    ))
    monkeypatch.setattr(runner, "_reuse_before_records", lambda *args, **kwargs: (
        [{"seed": int(cell["seed"])} for cell in runner.EVALUATION_CELLS],
        [str(reuse_root / f"{cell['scenario']}_{cell['seed']}.json") for cell in runner.EVALUATION_CELLS],
    ))
    report = runner.run(
        output_dir=tmp_path / f"check_{mode}",
        check_only=True,
        reuse_before_root=reuse_root,
        **{mode: True},
    )
    budgets = report["binding"]["simulation_budgets"]
    assert report["status"] == "CHECK_ONLY_OK"
    assert budgets["paired_evaluation_total_horizon_steps"] == 12
    assert budgets["reused_before_rollouts_expected"] == 6
    assert budgets["new_eval_rollouts_expected"] == 6
    assert budgets["new_evaluation_horizon_steps"] == 6
    assert report["checks"]["reused_before_rollouts_expected"] == 6
    assert report["checks"]["new_eval_rollouts_expected"] == 6
    assert report["checks"]["reused_before_rollouts"] == 0
    assert report["checks"]["new_eval_rollouts"] == 0
    assert report["checks"]["runtime_rollout_calls"] == 0


def _write_reuse_before_root(tmp_path, config, *, mutate: tuple[str, object] | None = None):
    root = tmp_path / "evaluation" / "records" / "before"
    snapshot = runner.neutral_eval._config_snapshot(config)
    for cell in runner.EVALUATION_CELLS:
        scenario = str(cell["scenario"])
        seed = int(cell["seed"])
        record = {
            "arm": "before",
            "cell": dict(cell),
            "scenario": scenario,
            "seed": seed,
            "policy_seed": int(cell["policy_seed"]),
            "requests": int(runner.neutral_eval.EXPECTED_REQUESTS[scenario]),
            "checkpoint_sha256": runner.EXPECTED_CANDIDATE_SHA256,
            "status": "completed",
            "lifecycle_valid": True,
            "technical_ok": True,
            "weights_unchanged": True,
            "training_or_update_called": False,
            "teacher_called": False,
            "task_priority_enabled": False,
            "low_level_priority_time_weight": 0.0,
            "config": snapshot,
            "summary": {"seed": seed, "policy_seed": int(cell["policy_seed"])},
        }
        if mutate is not None:
            field, value = mutate
            record[field] = value
        path = root / f"{scenario}_{seed}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
    return root


@pytest.mark.parametrize("field,value", [
    ("checkpoint_sha256", "wrong-sha"),
    ("seed", 999999),
    ("task_priority_enabled", True),
])
def test_reuse_before_rejects_sha_seed_or_neutral_profile(monkeypatch, tmp_path, field, value) -> None:
    config = _config()
    configs = {name: config for name in ("dense24", "flow6", "flow12")}
    monkeypatch.setattr(runner.neutral_eval, "_record_metrics", lambda record: {
        "technical_ok": True, "safety_nonzero": False,
    })
    monkeypatch.setattr(runner.neutral_eval, "_record_fatal_reason", lambda record, smoke: None)
    root = _write_reuse_before_root(tmp_path, config, mutate=(field, value))
    with pytest.raises(runner.JointBatchError):
        runner._reuse_before_records(
            root, output=None, neutral_configs=configs,
            evaluation_kind="development_existing_scenes",
        )


def test_reuse_before_validates_and_copies_exact_six_without_runtime(monkeypatch, tmp_path) -> None:
    config = _config()
    configs = {name: config for name in ("dense24", "flow6", "flow12")}
    monkeypatch.setattr(runner.neutral_eval, "_record_metrics", lambda record: {
        "technical_ok": True, "safety_nonzero": False,
    })
    monkeypatch.setattr(runner.neutral_eval, "_record_fatal_reason", lambda record, smoke: None)
    root = _write_reuse_before_root(tmp_path, config)
    output = tmp_path / "run"
    records, paths = runner._reuse_before_records(
        root, output=output, neutral_configs=configs,
        evaluation_kind="development_existing_scenes",
    )
    assert len(records) == 6
    assert len(paths) == 6
    assert len(list((output / "evaluation" / "records" / "before").glob("*.json"))) == 6
    assert all(record["source_provenance"]["kind"] == "reused_before_record" for record in records)


def test_reuse_before_requires_full_episode_credit(tmp_path) -> None:
    report = runner.run(
        output_dir=tmp_path / "default",
        reuse_before_root=tmp_path / "before",
    )
    assert report["status"] == "FAILED"
    assert "requires --full-episode-credit" in report["unresolved"][0]


def test_full_and_joint_team_actor_modes_are_mutually_exclusive(tmp_path) -> None:
    report = runner.run(
        output_dir=tmp_path / "both",
        full_episode_credit=True,
        joint_team_actor=True,
    )
    assert report["status"] == "FAILED"
    assert "mutually exclusive" in report["unresolved"][0]
