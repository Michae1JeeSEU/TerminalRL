from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from terminal_marl.training import run_p3_task_neutral_critic_refresh_comparison as refresh
from terminal_marl.training import run_p3_task_neutral_critic_warmup as warmup
from terminal_marl.training import run_p3_task_neutral_learning_curve as curve
from terminal_marl.training import run_p3_task_neutral_pilot as pilot


class _TinyCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Parameter(torch.tensor([0.25]))
        self.critic_encoder = nn.Linear(88, 4)
        self.critic_head = nn.Linear(8, 1)


def _policy() -> SimpleNamespace:
    return SimpleNamespace(model=_TinyCritic(), device=torch.device("cpu"))


def _base_payload(policy: SimpleNamespace, config: object) -> dict:
    state = {name: value.detach().cpu().clone() for name, value in policy.model.state_dict().items()}
    return {
        "format": "p3_task_neutral_critic_warmup_v1",
        "stage": warmup.STAGE,
        "curriculum_id": warmup.CURRICULUM_ID,
        "gamma": 1.0,
        "reward_scale": 0.01,
        "task_priority_enabled": False,
        "low_level_priority_time_weight": 0.0,
        "neutral_config_snapshots": {"dense24": {"fixture": True}},
        "parent_checkpoint_sha256": pilot.EXPECTED_CANDIDATE_SHA256,
        "model_state": state,
        "actor_invariance": {
            "non_critic_state_bitwise_unchanged": True,
            "critic_parameter_names": [name for name in state if name.startswith("critic_")],
        },
        "fit_parameters": {"gamma": 1.0, "reward_scale": 0.01, "epochs": 20},
        "fit_metrics": {"before": {}, "after": {}, "before_per_cell": {}, "after_per_cell": {}},
        "trajectory_reports": [],
    }


def _probe_bundle(policy: SimpleNamespace) -> dict:
    critic_state = {
        name: value.detach().cpu().clone() + 0.125
        for name, value in policy.model.state_dict().items()
        if name.startswith("critic_")
    }
    reports = [
        {"index": index, "split": "fit" if index <= 18 else "validation"}
        for index in range(1, 25)
    ]
    return {
        "critic_state": critic_state,
        "critic_input_hash": "context-hash",
        "critic_sha256": refresh.PROBE_LEGACY_CRITIC_SHA256,
        "policy": policy,
        "report_path": Path("probe-report.json"),
        "critic_path": Path("legacy_critic.pt"),
        "report": {
            "fit": {
                "epochs": 20, "minibatch_size": 64, "learning_rate": 3e-4,
                "permutation_seed": 879501, "grad_clip": 1.0,
                "before": {"legacy": {}}, "after": {"legacy": {}},
            },
            "episode_metrics": {"before": {}, "after": {}},
            "trajectory_reports": reports,
        },
    }


def test_comparison_profile_fixed_matrix_and_default_curve_contract_unchanged() -> None:
    profile = refresh.comparison_profile("control", refresh.A82_WARMUP_SHA256)
    assert profile.batch_count == 2
    assert profile.episodes_per_batch == 24
    assert profile.permutation_seeds == (881001, 881002)
    assert profile.training_env_seeds == tuple(range(881101, 881149))
    assert profile.training_policy_seeds == tuple(range(881201, 881249))
    assert profile.evaluation_batches == (2,)
    assert profile.minibatch_size == 128
    assert profile.required_episodic_return is True
    assert curve.DEFAULT_PROFILE.batch_count == 6
    assert curve.LARGER_BATCH_PROFILE.batch_count == 4


def test_arm_summary_reads_curve_training_metrics_from_nested_report() -> None:
    report = {
        "status": "COMPLETED", "profile": "critic_refresh_control",
        "training_rollouts": 48, "training_steps": 999,
        "evaluation_rollouts": 4,
        "training": {"rollouts": 48, "steps": 86400, "updates": 2, "batches_processed": 2},
        "evaluation": {"new_rollouts": 4, "snapshots": [{"batch_index": 2, "after_summary": {"normal": 4}}]},
        "checks": {"batches_completed": True, "updates_completed": True, "source_q_fixed": True},
    }
    summary = refresh._arm_summary("control", Path("arm"), report)
    assert summary["training_rollouts"] == 48
    assert summary["training_steps"] == 86400
    assert summary["ppo_updates"] == 2
    assert summary["evaluation_rollouts"] == 4
    assert summary["evaluation_snapshots"] == 1


def test_merge_refresh_warmup_replaces_only_critic_and_records_probe_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    config = SimpleNamespace()
    monkeypatch.setattr(pilot, "_config_snapshot", lambda value: {"fixture": True})
    base = _base_payload(policy, config)
    bundle = _probe_bundle(policy)
    actor_before = policy.model.actor.detach().clone()
    output = tmp_path / "prepared" / "refresh_warmup.pt"
    payload = refresh._merge_refresh_warmup(base, bundle, config=config, output_path=output)
    assert output.is_file()
    assert payload["refresh_method"] == "probe_legacy_critic_into_a82_warmup"
    assert payload["origin_fit_episodes"] == 18
    assert payload["origin_validation_episodes"] == 6
    assert payload["critic_input_hash"] == "context-hash"
    assert torch.equal(payload["model_state"]["actor"], actor_before)
    assert payload["trajectory_reports"] == bundle["report"]["trajectory_reports"]


def test_merge_refresh_rejects_critic_shape_or_nonfinite_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    config = SimpleNamespace()
    monkeypatch.setattr(pilot, "_config_snapshot", lambda value: {"fixture": True})
    base = _base_payload(policy, config)
    bad_shape = _probe_bundle(policy)
    bad_shape["critic_state"]["critic_head.weight"] = torch.zeros((2, 8))
    with pytest.raises(refresh.CriticRefreshComparisonError, match="shape mismatch"):
        refresh._merge_refresh_warmup(base, bad_shape, config=config, output_path=tmp_path / "shape.pt")
    bad_finite = _probe_bundle(policy)
    bad_finite["critic_state"]["critic_head.weight"] = torch.full((1, 8), float("nan"))
    with pytest.raises(refresh.CriticRefreshComparisonError, match="non-finite"):
        refresh._merge_refresh_warmup(base, bad_finite, config=config, output_path=tmp_path / "finite.pt")


def test_probe_bundle_rejects_wrong_source_or_credit_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report_path = tmp_path / "probe-report.json"
    critic_path = tmp_path / "legacy.pt"
    torch.save({
        "diagnostic_only": True,
        "critic_kind": "legacy_mean_max",
        "state_dict": {"critic_head.weight": torch.zeros((1, 1))},
    }, critic_path)
    critic_sha = refresh._sha256(critic_path)
    monkeypatch.setattr(refresh, "PROBE_LEGACY_CRITIC_SHA256", critic_sha)
    checks = {
        name: True for name in (
            "replay_exact", "terminal", "ledger_alignment", "safety_required_zero",
            "source_actor_model_bitwise_unchanged", "validation_excluded_from_optimizer",
            "expanded_initial_prediction_matches_legacy",
        )
    }
    binding = {
        "source_checkpoint_sha256": refresh.SOURCE_CHECKPOINT_SHA256,
        "warmup_checkpoint_sha256": refresh.A82_WARMUP_SHA256,
        "fit_episodes": 18, "validation_episodes": 6,
        "gamma": 1.0, "reward_scale": 0.01,
    }
    trajectory_reports = [
        {"split": "fit" if index <= 18 else "validation"}
        for index in range(1, 25)
    ]
    report = {"status": "COMPLETED", "binding": binding, "checks": checks,
              "trajectory_reports": trajectory_reports}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(refresh.CriticRefreshComparisonError, match="source checkpoint SHA"):
        bad_source = json.loads(report_path.read_text(encoding="utf-8"))
        bad_source["binding"]["source_checkpoint_sha256"] = "wrong"
        report_path.write_text(json.dumps(bad_source), encoding="utf-8")
        refresh._probe_bundle(report_path, critic_path)
    with pytest.raises(refresh.CriticRefreshComparisonError, match="credit binding"):
        bad_gamma = json.loads(json.dumps(report))
        bad_gamma["binding"]["gamma"] = 0.99
        report_path.write_text(json.dumps(bad_gamma), encoding="utf-8")
        refresh._probe_bundle(report_path, critic_path)


def _prepared_fixture(tmp_path: Path) -> dict:
    control = tmp_path / "control_warmup.pt"
    refresh_path = tmp_path / "refresh_warmup.pt"
    control.write_bytes(b"control")
    refresh_path.write_bytes(b"refresh")
    return {
        "control_warmup": control,
        "control_warmup_check": {},
        "refresh_warmup": refresh_path,
        "refresh_warmup_sha256": "refresh-sha",
        "refresh_warmup_check": {},
        "probe": {
            "report_path": tmp_path / "probe-report.json",
            "critic_path": tmp_path / "legacy.pt",
            "critic_sha256": refresh.PROBE_LEGACY_CRITIC_SHA256,
            "critic_input_hash": "critic-hash",
        },
    }


def test_check_only_runs_both_arms_without_runtime_and_writes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(task_priority_enabled=False, low_level_priority_time_weight=0.0,
                             max_simulation_seconds=1800.0, dt=1.0)
    monkeypatch.setattr(refresh.neutral_eval, "_build_neutral_inputs",
                        lambda: ({}, {"dense24": config}, {}))
    prepared = _prepared_fixture(tmp_path)
    monkeypatch.setattr(refresh, "_prepare_inputs", lambda output, config: prepared)
    calls: list[dict] = []

    def fake_curve_run(**kwargs):
        calls.append(dict(kwargs))
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=False)
        return {
            "status": "CHECK_ONLY_OK", "profile": kwargs["profile_override"].name,
            "training_rollouts": 0, "training_steps": 0, "ppo_updates": 0,
            "evaluation": {"new_rollouts": 0, "snapshots": []}, "training": {"batches_processed": 0},
            "checks": {"source_q_fixed": True, "runtime_rollouts": 0,
                        "update_calls": 0, "evaluation_rollouts": 0},
        }

    monkeypatch.setattr(refresh.curve, "run", fake_curve_run)
    report = refresh.run(tmp_path / "comparison", check_only=True)
    assert report["status"] == "CHECK_ONLY_OK"
    assert len(calls) == 2
    assert [call["profile_override"].name for call in calls] == [
        "critic_refresh_control", "critic_refresh_refresh",
    ]
    assert all(call["check_only"] is True for call in calls)
    assert all(call["episodic_return"] is True and call["reward_scale"] == 0.01 for call in calls)
    assert report["checks"]["check_only_runtime_zero"] is True
    assert (tmp_path / "comparison" / "report.json").is_file()
    assert (tmp_path / "comparison" / "summary.json").is_file()
    assert json.loads((tmp_path / "comparison" / "summary.json").read_text(encoding="utf-8"))["status"] == "CHECK_ONLY_OK"


def test_control_failure_does_not_start_refresh_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(task_priority_enabled=False, low_level_priority_time_weight=0.0,
                             max_simulation_seconds=1800.0, dt=1.0)
    monkeypatch.setattr(refresh.neutral_eval, "_build_neutral_inputs",
                        lambda: ({}, {"dense24": config}, {}))
    monkeypatch.setattr(refresh, "_prepare_inputs", lambda output, config: _prepared_fixture(tmp_path))
    calls: list[str] = []

    def fake_curve_run(**kwargs):
        calls.append(Path(kwargs["output_dir"]).name)
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=False)
        return {"status": "FAILED", "profile": kwargs["profile_override"].name,
                "unresolved": ["fixture failure"], "evaluation": {}, "training": {}, "checks": {}}

    monkeypatch.setattr(refresh.curve, "run", fake_curve_run)
    report = refresh.run(tmp_path / "failure", check_only=False)
    assert report["status"] == "FAILED"
    assert calls == ["control"]
    assert report["unresolved"]
