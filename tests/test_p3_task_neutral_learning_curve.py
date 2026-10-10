from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.algo.student_ppo_v4_2 import StudentStepV42
from terminal_marl.training import run_p3_task_neutral_learning_curve as runner


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        task_priority_enabled=False,
        low_level_priority_time_weight=0.0,
        max_simulation_seconds=1800.0,
        dt=1.0,
    )


def _step(*, reward: float = 1.0, actor: bool = True) -> StudentStepV42:
    return StudentStepV42(
        world=np.zeros((1, 88), dtype=np.float32),
        requested_gate=np.array([0], dtype=np.int64),
        requested_proposal=np.array([-1], dtype=np.int64),
        gate_logprob=np.array([0.0], dtype=np.float32),
        proposal_logprob=np.array([np.nan], dtype=np.float32),
        actor_mask=np.array([actor], dtype=bool),
        value=0.0,
        next_value=0.0,
        reward=float(reward),
        terminated=True,
        episode_end=True,
        stochastic_behavior=True,
    )


def _sequence_step(
    *, reward: float, actor: bool, episode_end: bool, terminated: bool,
    value: float = 0.0, next_value: float = 0.0,
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


def _fake_record(cell: dict[str, object], *, arm: str = "before") -> dict[str, object]:
    return {
        "arm": arm,
        "scenario": str(cell["scenario"]),
        "seed": int(cell["seed"]),
        "policy_seed": int(cell["policy_seed"]),
        "record_path": f"{arm}_{cell['seed']}.json",
    }


def _install_full_mocks(monkeypatch, tmp_path, *, freeze_ok: bool = True,
                        q_ok: bool = True, accepted: int = 1):
    config = _config()
    configs = {name: config for name in ("dense24", "flow6", "flow12")}

    class Model:
        def state_dict(self):
            return {}

    policy = SimpleNamespace(model=Model(), gate_reference=None)
    source_policy = SimpleNamespace(model=Model())
    warmup_path = tmp_path / "warmup.pt"
    policy.optimizer = None
    monkeypatch.setattr(
        runner.neutral_eval, "_build_neutral_inputs",
        lambda: (configs, configs, {}),
    )
    monkeypatch.setattr(
        runner.joint, "_preflight",
        lambda neutral_configs: (
            tmp_path / "candidate.pt", policy,
            {"checkpoint_sha256": runner.pilot.EXPECTED_CANDIDATE_SHA256},
            source_policy,
            {"warmup_path": str(warmup_path), "warmup_sha256": "warmup"},
            {"source_q_digest": "q"},
        ),
    )
    baseline = [_fake_record(cell) for cell in runner.EVALUATION_CELLS]
    monkeypatch.setattr(
        runner, "_baseline_records",
        lambda config, output=None: (list(baseline), [row["record_path"] for row in baseline]),
    )

    optimizer = object()
    configure_calls = []

    def configure(active):
        configure_calls.append(active)
        if active.optimizer is None:
            active.optimizer = optimizer
        return {"scope": "test", "learning_rate": 3e-5}

    monkeypatch.setattr(runner.pilot, "_configure_trainable_policy", configure)
    monkeypatch.setattr(runner.pilot, "_model_digest", lambda obj: "p" if obj is policy else "q")
    weight_calls: list[int] = []

    def observation_weights(steps):
        weight_calls.append(len(steps))
        return (
            np.zeros((len(steps), 88), dtype=np.float32),
            np.full(len(steps), 1.0 / len(steps), dtype=np.float32),
        )

    monkeypatch.setattr(
        runner.pilot, "actor_world_observation_weights", observation_weights,
    )
    monkeypatch.setattr(runner.pilot, "_build_gate_reference", lambda *args: object())
    monkeypatch.setattr(
        runner.pilot, "_frozen_bitwise_check",
        lambda before, after: {
            "frozen_parameter_bitwise_unchanged": freeze_ok,
            "proposal_bitwise_unchanged": freeze_ok,
            "gate_temperature_bitwise_unchanged": freeze_ok,
        },
    )
    monkeypatch.setattr(
        runner.neutral_eval, "_record_metrics", lambda record: {"failed_energy": 0},
    )
    monkeypatch.setattr(
        runner, "_metric_summary",
        lambda records: {
            "requests": len(records) * 24,
            "normal": len(records) * 24,
            "emergency": 0,
            "failed": 0,
            "mean_wait": 1.0,
            "p95_wait": 1.0,
            "max_wait": 1.0,
            "lowSOC_wait": 1.0,
            "reward_total": float(len(records)),
            "safety_nonzero_cells": 0,
            "technical_ok": True,
        },
    )

    collect_calls: list[dict[str, object]] = []

    def collect(**kwargs):
        selected = tuple(kwargs["training_cells"])
        collect_calls.append({"cells": selected, "checkpoint": kwargs["checkpoint"]})
        steps = [_step(reward=float(index + 1)) for index in range(len(selected))]
        records = [_fake_record(cell, arm="training") for cell in selected]
        return steps, records, {"bitwise_unchanged": True}, None

    monkeypatch.setattr(runner.joint, "_collect_training", collect)
    update_calls: list[tuple[int, dict[str, object]]] = []

    def update(steps, **kwargs):
        update_calls.append((len(steps), dict(kwargs)))
        return {
            "updates": 1,
            "accepted_updates": accepted,
            "rejected_updates": 0 if accepted else 1,
            "approx_kl": 0.0,
            "final_rollout_kl": 0.0,
        }

    policy.update = update
    roundtrips: list[dict[str, object]] = []

    def roundtrip(path, active, dense, metadata):
        roundtrips.append(dict(metadata))
        return {"checkpoint_sha256": f"after-{metadata['batch_index']}", "exact": True}

    monkeypatch.setattr(runner.pilot, "_roundtrip_pilot", roundtrip)
    eval_calls: list[tuple[int, Path]] = []

    def evaluate(**kwargs):
        batch_index = int(kwargs["batch_index"])
        eval_calls.append((batch_index, kwargs["checkpoint"]))
        return [_fake_record(cell, arm="after") for cell in runner.EVALUATION_CELLS], None

    monkeypatch.setattr(runner, "_run_evaluation", evaluate)
    monkeypatch.setattr(runner, "_paired_metrics", lambda before, after: {"after_minus_before": {}})
    if not q_ok:
        calls = {"source": 0}

        def changing_digest(obj):
            if obj is source_policy:
                calls["source"] += 1
                return "q" if calls["source"] == 1 else "changed"
            return "p"

        monkeypatch.setattr(runner.pilot, "_model_digest", changing_digest)
    return {
        "config": config,
        "policy": policy,
        "source_policy": source_policy,
        "configure_calls": configure_calls,
        "collect_calls": collect_calls,
        "update_calls": update_calls,
        "roundtrips": roundtrips,
        "eval_calls": eval_calls,
        "weight_calls": weight_calls,
    }


def _install_episodic_preflight_mocks(monkeypatch, tmp_path, calls):
    policy = calls["policy"]
    source_policy = calls["source_policy"]
    candidate = tmp_path / "candidate.pt"
    warmup = tmp_path / "episodic_warmup.pt"
    candidate.write_bytes(b"candidate")
    warmup.write_bytes(b"episodic-warmup")
    warmup_sha = runner.neutral_eval._sha256(warmup)
    monkeypatch.setattr(
        runner.pilot, "_load_binding",
        lambda config: (
            candidate,
            policy,
            {"checkpoint_sha256": runner.pilot.EXPECTED_CANDIDATE_SHA256},
        ),
    )
    monkeypatch.setattr(
        runner.rank_eval, "_load_policy",
        lambda checkpoint, config: (source_policy, {"stage": runner.STAGE}),
    )

    def load_warmup(path, active, config, *, expected_gamma=None, expected_reward_scale=None):
        calls["expected_gamma"] = expected_gamma
        calls["expected_reward_scale"] = expected_reward_scale
        return warmup, {
            "checkpoint_sha256": warmup_sha,
            "gamma": expected_gamma,
            "reward_scale": expected_reward_scale,
        }

    monkeypatch.setattr(runner.pilot, "_load_initial_warmup", load_warmup)
    return warmup, warmup_sha


def test_training_matrix_has_six_batches_and_fixed_seed_pairs() -> None:
    batches = runner.training_batches()
    assert len(batches) == 6
    assert all(len(batch) == 6 for batch in batches)
    assert [int(batch[0]["seed"]) for batch in batches] == [877101, 877107, 877113, 877119, 877125, 877131]
    assert [int(batch[0]["policy_seed"]) for batch in batches] == [878201, 878207, 878213, 878219, 878225, 878231]
    cells = runner._training_cells()
    assert len(cells) == 36
    assert len({(cell["seed"], cell["policy_seed"]) for cell in cells}) == 36


def test_larger_batch_profile_has_four_batches_of_24_and_fixed_seed_pairs() -> None:
    profile = runner.LARGER_BATCH_PROFILE
    batches = runner.training_batches(profile)
    assert len(batches) == 4
    assert all(len(batch) == 24 for batch in batches)
    assert [int(batch[0]["seed"]) for batch in batches] == [879101, 879125, 879149, 879173]
    assert [int(batch[-1]["seed"]) for batch in batches] == [879124, 879148, 879172, 879196]
    assert [int(batch[0]["policy_seed"]) for batch in batches] == [879201, 879225, 879249, 879273]
    assert [int(batch[-1]["policy_seed"]) for batch in batches] == [879224, 879248, 879272, 879296]
    cells = runner._training_cells(profile)
    assert len(cells) == 96
    assert len({(cell["seed"], cell["policy_seed"]) for cell in cells}) == 96
    assert profile.permutation_seeds == (879001, 879002, 879003, 879004)
    assert profile.evaluation_batches == (2, 4)
    assert profile.minibatch_size == 128
    args = runner.build_parser().parse_args([
        "--output-dir", "curve", "--larger-batch", "--episodic-return",
        "--initial-warmup-checkpoint", "warmup.pt", "--reward-scale", "0.01",
    ])
    assert args.larger_batch is True


def test_larger_batch_cli_requires_episodic_inputs(monkeypatch, tmp_path) -> None:
    calls = _install_full_mocks(monkeypatch, tmp_path)
    report = runner.run(output_dir=tmp_path / "larger_requires_episodic", larger_batch=True)
    assert report["status"] == "FAILED"
    assert "requires --episodic-return" in report["unresolved"][0]
    assert calls["collect_calls"] == []
    scaled = runner.run(
        output_dir=tmp_path / "larger_requires_scale", larger_batch=True,
        episodic_return=True, initial_warmup_checkpoint=tmp_path / "missing.pt",
        reward_scale=1.0,
    )
    assert scaled["status"] == "FAILED"
    assert "requires --reward-scale 0.01" in scaled["unresolved"][0]


def test_episodic_kwargs_and_cli_flags_leave_default_contract_unchanged() -> None:
    default = runner._update_kwargs(episodic_return=False)
    episodic = runner._update_kwargs(episodic_return=True)
    assert default == runner.UPDATE_KWARGS
    assert default["gamma"] == 0.99
    assert default["gae_lambda"] == 0.95
    assert episodic["gamma"] == 1.0
    assert episodic["gae_lambda"] == 1.0
    assert runner.UPDATE_KWARGS["gamma"] == 0.99
    assert runner.UPDATE_KWARGS["gae_lambda"] == 0.95
    args = runner.build_parser().parse_args([
        "--output-dir", "curve", "--episodic-return",
        "--initial-warmup-checkpoint", "warmup.pt",
    ])
    assert args.episodic_return is True
    assert str(args.initial_warmup_checkpoint) == "warmup.pt"


def test_reward_scale_requires_episodic_mode_and_scaled_steps_preserve_raw_inputs() -> None:
    raw = [
        _sequence_step(reward=3.0, value=1.0, next_value=0.5,
                       actor=True, episode_end=False, terminated=False),
        _sequence_step(reward=2.0, value=0.5, next_value=0.0,
                       actor=True, episode_end=True, terminated=True),
    ]
    scaled = runner._scaled_training_steps(raw, 0.01)
    assert [row.reward for row in raw] == [3.0, 2.0]
    assert [row.reward for row in scaled] == pytest.approx([0.03, 0.02])
    assert [row.value for row in scaled] == [1.0, 0.5]
    assert [row.next_value for row in scaled] == [0.5, 0.0]
    assert runner._validate_reward_scale(1.0, episodic_return=False) == 1.0
    with pytest.raises(runner.LearningCurveError, match="requires --episodic-return"):
        runner._validate_reward_scale(0.01, episodic_return=False)


def test_uniform_reward_and_value_scaling_preserves_normalized_advantage() -> None:
    raw = [
        _sequence_step(reward=3.0, value=1.0, next_value=0.5,
                       actor=True, episode_end=False, terminated=False),
        _sequence_step(reward=2.0, value=0.5, next_value=0.0,
                       actor=True, episode_end=True, terminated=True),
    ]
    scaled = [
        dataclasses.replace(
            row, reward=row.reward * 0.01, value=row.value * 0.01,
            next_value=row.next_value * 0.01,
        )
        for row in raw
    ]
    raw_diag = runner.pilot._gae_diagnostics(raw, gamma=0.99, gae_lambda=0.95)
    scaled_diag = runner.pilot._gae_diagnostics(scaled, gamma=0.99, gae_lambda=0.95)
    assert np.asarray(scaled_diag["normalized_advantages"]) == pytest.approx(
        np.asarray(raw_diag["normalized_advantages"]), abs=1e-5,
    )


def test_raw_reward_record_total_and_scaled_learning_total_are_checked() -> None:
    steps = [_step(reward=2.0), _step(reward=3.0)]
    records = [
        {"summary": {"reward_total": 2.0}},
        {"rollout_summary": {"reward_total": 3.0}},
    ]
    diagnostic = runner._reward_scale_diagnostics(steps, records, 0.01)
    assert diagnostic["raw_reward_total_source"] == "records"
    assert diagnostic["raw_environment_reward_total"] == pytest.approx(5.0)
    assert diagnostic["learning_reward_total"] == pytest.approx(0.05)
    assert diagnostic["reward_scale_consistent"] is True
    assert [row.reward for row in steps] == [2.0, 3.0]


def test_check_only_has_zero_runtime_and_reports_curve_budgets(monkeypatch, tmp_path) -> None:
    _install_full_mocks(monkeypatch, tmp_path)
    report = runner.run(output_dir=tmp_path / "check", check_only=True)
    assert report["status"] == "CHECK_ONLY_OK"
    assert report["checks"]["runtime_rollouts"] == 0
    assert report["checks"]["update_calls"] == 0
    assert report["checks"]["evaluation_rollouts"] == 0
    assert report["binding"]["training_rollouts_expected"] == 36
    assert report["binding"]["training_max_steps"] == 64800
    assert report["binding"]["evaluation_new_max_steps"] == 21600
    assert report["evaluation"]["baseline_reused_records"] == 4


def test_full_curve_reuses_one_optimizer_and_updates_each_batch(monkeypatch, tmp_path) -> None:
    calls = _install_full_mocks(monkeypatch, tmp_path)
    report = runner.run(output_dir=tmp_path / "full")
    assert report["status"] == "COMPLETED"
    assert len(calls["configure_calls"]) == 1
    assert len(calls["collect_calls"]) == 6
    assert len(calls["update_calls"]) == 6
    assert all(size == 6 for size, _ in calls["update_calls"])
    assert all(kwargs["actor_reduction"] == "joint_team" for _, kwargs in calls["update_calls"])
    assert all(kwargs["gamma"] == 0.99 and kwargs["gae_lambda"] == 0.95
               for _, kwargs in calls["update_calls"])
    assert [len(item["cells"]) for item in calls["collect_calls"]] == [6] * 6
    assert [item["anchor_steps"] for item in report["training"]["batches"]] == [6, 12, 18, 24, 30, 36]
    assert calls["weight_calls"] == [6, 12, 18, 24, 30, 36]
    assert [item["actor_count_normalizer"] for item in report["training"]["batches"]] == [1.0] * 6
    assert all(item["optimizer_identity_unchanged"] is True
               for item in report["training"]["batches"])
    assert [batch for batch, _ in calls["eval_calls"]] == [2, 4, 6]
    assert report["evaluation"]["new_rollouts"] == 12
    assert [item["batch_index"] for item in calls["roundtrips"]] == [1, 2, 3, 4, 5, 6]
    assert all(item["repair_method"] == runner.REPAIR_METHOD for item in calls["roundtrips"])


@pytest.mark.parametrize("failure", ["accepted", "freeze", "q"])
def test_curve_stops_after_update_failure_without_evaluation(monkeypatch, tmp_path, failure) -> None:
    calls = _install_full_mocks(
        monkeypatch,
        tmp_path,
        freeze_ok=failure != "freeze",
        q_ok=failure != "q",
        accepted=0 if failure == "accepted" else 1,
    )
    report = runner.run(output_dir=tmp_path / f"failed_{failure}")
    assert report["status"] == "FAILED"
    assert len(calls["collect_calls"]) == 1
    assert len(calls["update_calls"]) == 1
    assert calls["eval_calls"] == []
    assert calls["roundtrips"] == []
    assert report["training"]["updates"] == 1


def test_batch_summary_uses_batch_normalizer_and_cumulative_anchor_count() -> None:
    steps = [_step(), _step(actor=False), _step()]
    summary = runner._batch_summary(
        batch_index=2,
        cells=runner.training_batches()[1],
        steps=steps,
        records=[_fake_record(runner.training_batches()[1][0], arm="training")],
        stats={"updates": 1, "accepted_updates": 1},
        freeze={"frozen_parameter_bitwise_unchanged": True},
        collection={"bitwise_unchanged": True},
        anchor_steps=12,
        anchor_observations=12,
        anchor_weight_sum=1.0,
        optimizer_identity_unchanged=True,
        checkpoint=None,
    )
    assert summary["actor_decisions"] == 2
    assert summary["actor_count_normalizer"] == pytest.approx(2 / 3)
    assert summary["anchor_steps"] == 12


def test_evaluation_loads_a_fresh_policy_for_each_cell(monkeypatch, tmp_path) -> None:
    loaded: list[object] = []

    def load_policy(checkpoint, config):
        policy = object()
        loaded.append(policy)
        return policy, {"stage": runner.STAGE}

    monkeypatch.setattr(runner.rank_eval, "_load_policy", load_policy)
    monkeypatch.setattr(runner.pilot, "_validate_pilot_payload", lambda payload, config: {})
    monkeypatch.setattr(
        runner.rank_eval,
        "_runtime_record",
        lambda **kwargs: {
            "scenario": kwargs["scenario"], "seed": kwargs["seed"],
            "policy_seed": kwargs["policy_seed"], "weights_unchanged": True,
        },
    )
    monkeypatch.setattr(runner.pilot, "_record_annotations", lambda record, config: dict(record))
    monkeypatch.setattr(runner.neutral_eval, "_record_fatal_reason", lambda record, smoke: None)
    records, reason = runner._run_evaluation(
        output=tmp_path,
        batch_index=2,
        checkpoint=tmp_path / "batch_02.pt",
        config=_config(),
    )
    assert reason is None
    assert len(records) == len(runner.EVALUATION_CELLS) == 4
    assert len({id(policy) for policy in loaded}) == 4


def test_partial_evaluation_preserves_records_and_original_error(monkeypatch, tmp_path) -> None:
    calls = _install_full_mocks(monkeypatch, tmp_path)
    partial = [_fake_record(runner.EVALUATION_CELLS[0], arm="after")]
    monkeypatch.setattr(
        runner,
        "_run_evaluation",
        lambda **kwargs: (partial, "evaluation runtime stopped after first record"),
    )
    report = runner.run(output_dir=tmp_path / "partial")
    assert report["status"] == "FAILED"
    assert report["evaluation"]["new_rollouts"] == 1
    assert len(report["evaluation"]["snapshots"]) == 1
    snapshot = report["evaluation"]["snapshots"][0]
    assert len(snapshot["after_records"]) == 1
    assert snapshot["paired_differences"] == []
    assert snapshot["evaluation_error"] == "evaluation runtime stopped after first record"
    assert report["unresolved"] == ["evaluation runtime stopped after first record"]
    assert (tmp_path / "partial" / "report.json").is_file()
    assert calls["eval_calls"] == []


def test_episodic_return_diagnostics_include_noactor_tail_and_terminal_segments() -> None:
    first = [
        _sequence_step(reward=1.0, actor=True, episode_end=False, terminated=False),
        *[
            _sequence_step(reward=1.0, actor=False, episode_end=False, terminated=False)
            for _ in range(59)
        ],
        _sequence_step(reward=1.0, actor=False, episode_end=True, terminated=True),
    ]
    second = [_sequence_step(reward=2.0, actor=True, episode_end=True, terminated=True)]
    diagnostic = runner._episodic_return_diagnostics(
        first + second,
        {"gamma": 1.0, "gae_lambda": 1.0},
    )
    assert diagnostic["full_episode_mc_return_match"] is True
    assert diagnostic["full_episode_mc_return_max_abs_error"] == pytest.approx(0.0)
    assert diagnostic["returns"]["min"] == pytest.approx(1.0)
    assert diagnostic["returns"]["max"] == pytest.approx(61.0)
    assert diagnostic["full_episode_mc_returns"]["max"] == pytest.approx(61.0)


def test_episodic_preflight_uses_supplied_gamma_one_warmup(monkeypatch, tmp_path) -> None:
    full_calls = _install_full_mocks(monkeypatch, tmp_path)
    calls = {
        "policy": full_calls["policy"],
        "source_policy": full_calls["source_policy"],
    }
    warmup, warmup_sha = _install_episodic_preflight_mocks(monkeypatch, tmp_path, calls)
    config = _config()
    resolved, policy, binding, source_policy, warmup_info, source = runner._episodic_preflight(
        {"dense24": config}, warmup, 1.0,
    )
    assert resolved == tmp_path / "candidate.pt"
    assert policy is calls["policy"]
    assert source_policy is calls["source_policy"]
    assert calls["expected_gamma"] == 1.0
    assert calls["expected_reward_scale"] == 1.0
    assert warmup_info["warmup_sha256"] == warmup_sha
    assert warmup_info["expected_gamma"] == 1.0
    assert source["source_q_digest"] == "q"
    assert binding["checkpoint_sha256"] == runner.pilot.EXPECTED_CANDIDATE_SHA256


def test_episodic_mode_full_mock_chain_uses_gamma_one_and_repair_method(
    monkeypatch, tmp_path,
) -> None:
    calls = _install_full_mocks(monkeypatch, tmp_path)
    warmup, warmup_sha = _install_episodic_preflight_mocks(monkeypatch, tmp_path, {
        "policy": calls["policy"], "source_policy": calls["source_policy"],
    })
    # The helper above stores the expected-gamma observation in its own mapping;
    # the runner contract is asserted through the report and update calls here.
    report = runner.run(
        output_dir=tmp_path / "episodic_full",
        episodic_return=True,
        initial_warmup_checkpoint=warmup,
        reward_scale=0.01,
    )
    assert report["status"] == "COMPLETED"
    assert report["repair_method"] == runner.EPISODIC_REPAIR_METHOD
    assert report["episodic_return"] is True
    assert report["reward_scale"] == pytest.approx(0.01)
    assert report["binding"]["warmup"]["warmup_sha256"] == warmup_sha
    assert report["binding"]["warmup"]["expected_reward_scale"] == pytest.approx(0.01)
    assert report["binding"]["update_kwargs"]["gamma"] == 1.0
    assert report["binding"]["update_kwargs"]["gae_lambda"] == 1.0
    assert len(calls["update_calls"]) == 6
    assert all(
        kwargs["gamma"] == 1.0
        and kwargs["gae_lambda"] == 1.0
        and kwargs["actor_reduction"] == "joint_team"
        for _, kwargs in calls["update_calls"]
    )
    assert all(
        batch["full_episode_mc_return_match"] is True
        for batch in report["training"]["batches"]
    )
    assert report["training"]["batches"][-1]["credit_method"] == "full_episode_mc"


def test_larger_batch_full_mock_chain_uses_one_optimizer_four_updates_and_mb128(
    monkeypatch, tmp_path,
) -> None:
    calls = _install_full_mocks(monkeypatch, tmp_path)
    warmup, warmup_sha = _install_episodic_preflight_mocks(monkeypatch, tmp_path, {
        "policy": calls["policy"], "source_policy": calls["source_policy"],
    })
    profile = dataclasses.replace(
        runner.LARGER_BATCH_PROFILE,
        required_warmup_sha256=warmup_sha,
    )
    monkeypatch.setattr(runner, "LARGER_BATCH_PROFILE", profile)
    report = runner.run(
        output_dir=tmp_path / "larger_full",
        larger_batch=True,
        episodic_return=True,
        initial_warmup_checkpoint=warmup,
        reward_scale=0.01,
    )
    assert report["status"] == "COMPLETED"
    assert report["profile"] == "larger_batch"
    assert len(calls["configure_calls"]) == 1
    assert len(calls["collect_calls"]) == 4
    assert len(calls["update_calls"]) == 4
    assert all(size == 24 for size, _ in calls["update_calls"])
    assert all(
        kwargs["minibatch_size"] == 128
        and kwargs["gamma"] == 1.0
        and kwargs["gae_lambda"] == 1.0
        and kwargs["actor_reduction"] == "joint_team"
        for _, kwargs in calls["update_calls"]
    )
    assert [len(item["cells"]) for item in calls["collect_calls"]] == [24] * 4
    assert calls["weight_calls"] == [24, 48, 72, 96]
    assert [batch for batch, _ in calls["eval_calls"]] == [2, 4]
    assert report["binding"]["training_rollouts_expected"] == 96
    assert report["binding"]["training_requests_expected"] == 2304
    assert report["binding"]["training_max_steps"] == 172800
    assert report["binding"]["evaluation_new_rollouts_expected"] == 8
    assert report["binding"]["evaluation_new_max_steps"] == 14400
    assert report["evaluation"]["new_rollouts"] == 8
    assert [item["anchor_steps"] for item in report["training"]["batches"]] == [24, 48, 72, 96]


def test_warmup_override_without_episodic_mode_is_rejected(monkeypatch, tmp_path) -> None:
    calls = _install_full_mocks(monkeypatch, tmp_path)
    warmup = tmp_path / "override.pt"
    warmup.write_bytes(b"override")
    report = runner.run(
        output_dir=tmp_path / "override_rejected",
        initial_warmup_checkpoint=warmup,
    )
    assert report["status"] == "FAILED"
    assert "requires --episodic-return" in report["unresolved"][0]
    assert calls["collect_calls"] == []
    assert calls["update_calls"] == []


def test_episodic_check_only_has_zero_runtime_and_binds_warmup(
    monkeypatch, tmp_path,
) -> None:
    calls = _install_full_mocks(monkeypatch, tmp_path)
    warmup, warmup_sha = _install_episodic_preflight_mocks(monkeypatch, tmp_path, {
        "policy": calls["policy"], "source_policy": calls["source_policy"],
    })
    report = runner.run(
        output_dir=tmp_path / "episodic_check",
        check_only=True,
        episodic_return=True,
        initial_warmup_checkpoint=warmup,
        reward_scale=0.01,
    )
    assert report["status"] == "CHECK_ONLY_OK"
    assert report["checks"]["runtime_rollouts"] == 0
    assert report["checks"]["update_calls"] == 0
    assert report["checks"]["evaluation_rollouts"] == 0
    assert report["binding"]["warmup"]["warmup_sha256"] == warmup_sha
    assert report["binding"]["warmup"]["expected_reward_scale"] == pytest.approx(0.01)
    assert report["reward_scale"] == pytest.approx(0.01)
    assert report["update_kwargs"]["gamma"] == 1.0
    assert report["update_kwargs"]["gae_lambda"] == 1.0
    assert report["full_evaluation"] is False
    assert calls["collect_calls"] == []
    assert calls["update_calls"] == []


def test_episodic_warmup_gamma_payload_mismatch_is_rejected(monkeypatch, tmp_path) -> None:
    calls = _install_full_mocks(monkeypatch, tmp_path)
    warmup = tmp_path / "gamma_mismatch.pt"
    warmup.write_bytes(b"gamma-mismatch")
    _install_episodic_preflight_mocks(monkeypatch, tmp_path, {
        "policy": calls["policy"], "source_policy": calls["source_policy"],
    })

    def reject_gamma(path, active, config, *, expected_gamma=None, expected_reward_scale=None):
        assert expected_gamma == 1.0
        assert expected_reward_scale == 1.0
        raise runner.LearningCurveError("warmup payload gamma mismatch")

    monkeypatch.setattr(runner.pilot, "_load_initial_warmup", reject_gamma)
    report = runner.run(
        output_dir=tmp_path / "gamma_rejected",
        check_only=True,
        episodic_return=True,
        initial_warmup_checkpoint=warmup,
    )
    assert report["status"] == "FAILED"
    assert "warmup payload gamma mismatch" in report["unresolved"][0]
    assert calls["collect_calls"] == []
