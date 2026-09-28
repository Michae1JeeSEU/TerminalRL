from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.algo.student_ppo_v4_2 import StudentStepV42
from terminal_marl.core.config import SimConfig
from terminal_marl.training import run_p3_task_neutral_pilot as pilot


def _step(*, actors: int) -> StudentStepV42:
    world = np.zeros((actors, 88), dtype=np.float32)
    return StudentStepV42(
        world=world,
        requested_gate=np.zeros(actors, dtype=np.int64),
        requested_proposal=np.full(actors, -1, dtype=np.int64),
        gate_logprob=np.zeros(actors, dtype=np.float32),
        proposal_logprob=np.full(actors, np.nan, dtype=np.float32),
        actor_mask=np.ones(actors, dtype=np.bool_),
        value=0.0,
        next_value=0.0,
        reward=0.0,
        terminated=True,
        episode_end=True,
    )


def test_actor_world_weights_are_equal_by_world_then_actor() -> None:
    observations, weights = pilot.actor_world_observation_weights([_step(actors=2), _step(actors=1)])
    assert observations.shape == (3, 88)
    assert weights.tolist() == pytest.approx([0.25, 0.25, 0.5])
    assert float(weights.sum()) == pytest.approx(1.0)


def test_training_scope_freezes_actor_gate_proposal_and_temperature() -> None:
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88)
    scope = pilot._configure_trainable_policy(policy)
    names = dict(policy.model.named_parameters())
    assert scope["learning_rate"] == pytest.approx(3e-5)
    assert all(names[name].requires_grad for name in scope["trainable_parameter_names"])
    assert all(not names[name].requires_grad for name in scope["frozen_parameter_names"])
    assert scope["gate_temperature_frozen"] is True
    assert all(name.startswith(pilot.TRAINABLE_PREFIXES) for name in scope["trainable_parameter_names"])


def test_gate_reference_uses_fixed_source_policy_snapshot() -> None:
    source = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88)
    before = pilot._model_digest(source)
    observations = np.zeros((3, 88), dtype=np.float32)
    weights = np.asarray([0.25, 0.25, 0.5], dtype=np.float32)
    reference = pilot._build_gate_reference(source, observations, weights)
    assert reference.observations.shape == (3, 88)
    assert reference.weights.tolist() == pytest.approx(weights.tolist())
    assert reference.coefficient == pytest.approx(0.1)
    assert reference.max_kl == pytest.approx(0.05)
    assert reference.soft_kl_target == pytest.approx(0.02)
    assert reference.soft_kl_gain == pytest.approx(400.0)
    assert pilot._model_digest(source) == before


def test_training_record_policy_seed_is_optional_and_forwarded(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_rollout(policy, config, **kwargs):
        calls.append(dict(kwargs))
        effective = int(kwargs.get("policy_seed", kwargs["seed"]))
        return [], {"action_tape": "tape", "policy_seed": effective, "terminal": True}

    monkeypatch.setattr(pilot, "_model_digest", lambda policy: "digest")
    monkeypatch.setattr(
        pilot.student_rollout_v4_2,
        "rollout",
        fake_rollout,
    )
    monkeypatch.setattr(
        pilot.terminal_lifecycle_metrics,
        "terminal_lifecycle_metrics",
        lambda *args: {"outcomes": {"completed": 0}},
    )
    monkeypatch.setattr(pilot.rank_eval, "_normalize_runtime_record", lambda record, **kwargs: dict(record))
    config = SimpleNamespace(max_simulation_seconds=2.0, dt=1.0)
    common = {
        "policy": object(),
        "checkpoint": tmp_path / "candidate.pt",
        "config": config,
        "scenario": "flow6",
        "output": tmp_path / "rollout",
    }

    _, default_record = pilot._training_record(**common, seed=101)
    _, seeded_record = pilot._training_record(**common, seed=102, policy_seed=909)

    assert "policy_seed" not in calls[0]
    assert calls[1]["policy_seed"] == 909
    assert default_record["policy_seed"] == 101
    assert seeded_record["policy_seed"] == 909
    assert default_record["summary"]["policy_seed"] == 101
    assert seeded_record["summary"]["policy_seed"] == 909


def test_training_record_failure_preserves_effective_policy_seed(tmp_path, monkeypatch) -> None:
    def failing_rollout(*args, **kwargs):
        raise RuntimeError("fixture rollout failure")

    monkeypatch.setattr(pilot, "_model_digest", lambda policy: "digest")
    monkeypatch.setattr(pilot.student_rollout_v4_2, "rollout", failing_rollout)
    monkeypatch.setattr(pilot.rank_eval, "_normalize_runtime_record", lambda record, **kwargs: dict(record))
    config = SimpleNamespace(max_simulation_seconds=2.0, dt=1.0)

    steps, record = pilot._training_record(
        policy=object(),
        checkpoint=tmp_path / "candidate.pt",
        config=config,
        scenario="flow6",
        seed=101,
        policy_seed=909,
        output=tmp_path / "rollout",
    )

    assert steps == []
    assert record["status"] == "failed"
    assert record["policy_seed"] == 909


def test_nonterminal_training_rollout_is_rejected_before_update() -> None:
    called = []
    record = {
        "summary": {"terminal": False},
        "weights_unchanged": True,
        "safety_nonzero": False,
        "lifecycle_valid": False,
    }
    reason = pilot._training_terminal_failure(record, [])
    if reason is None:
        called.append("update")
    assert reason == "training rollout did not reach a terminal episode"
    assert called == []


def test_preflight_failure_writes_failed_report_and_progress(tmp_path, monkeypatch) -> None:
    def fail_inputs():
        raise pilot.TaskNeutralPilotError("fixture preflight failure")

    monkeypatch.setattr(pilot.neutral_eval, "_build_neutral_inputs", fail_inputs)
    result = pilot.run(output_dir=tmp_path / "failed", check_only=True)
    assert result["status"] == "FAILED"
    assert result["check_only"] is True
    assert result["full_evaluation"] is False
    assert (tmp_path / "failed" / "report.json").is_file()
    assert (tmp_path / "failed" / "progress.json").is_file()
    assert __import__("json").loads((tmp_path / "failed" / "progress.json").read_text())["status"] == "FAILED"


def test_training_record_does_not_become_baseline(tmp_path) -> None:
    record = {"scenario": "flow6", "seed": 873101}
    pilot._write_training_record(tmp_path, 1, record)
    assert (tmp_path / "training" / "episode_01" / "record.json").is_file()
    assert not (tmp_path / "full" / "records" / "candidate").exists()


def test_baseline_profile_mismatch_is_rejected(tmp_path, monkeypatch) -> None:
    cells = pilot.neutral_eval._evaluation_cells()
    configs = {
        scenario: SimpleNamespace(task_priority_enabled=False, low_level_priority_time_weight=0.0)
        for scenario in {str(cell["scenario"]) for cell in cells}
    }
    paths = {}
    for cell in cells:
        scenario, seed = str(cell["scenario"]), int(cell["seed"])
        path = tmp_path / f"{scenario}_{seed}.json"
        paths[(seed, scenario)] = path
        config = pilot._config_snapshot(configs[scenario])
        path.write_text(__import__("json").dumps({
            "scenario": scenario,
            "seed": seed,
            "checkpoint_sha256": pilot.EXPECTED_CANDIDATE_SHA256,
            "task_priority_enabled": False,
            "low_level_priority_time_weight": 0.0,
            "weights_unchanged": True,
            "technical_ok": True,
            "lifecycle_valid": True,
            "config": config,
        }))
    wrong = paths[(873102, "flow12")]
    payload = __import__("json").loads(wrong.read_text())
    payload["task_priority_enabled"] = True
    wrong.write_text(__import__("json").dumps(payload))
    monkeypatch.setattr(pilot, "_baseline_path", lambda scenario, seed: paths[(seed, scenario)])
    with pytest.raises(pilot.TaskNeutralPilotError, match="task-neutral flag"):
        pilot._load_baseline_records(configs)


def test_final_evaluation_runs_each_registered_cell_once(tmp_path, monkeypatch) -> None:
    calls = []

    class FakePolicy:
        pass

    def fake_load(path, config):
        return FakePolicy(), {}

    def fake_runtime_record(**kwargs):
        calls.append((kwargs["scenario"], kwargs["seed"]))
        return {
            "scenario": kwargs["scenario"],
            "seed": kwargs["seed"],
            "weights_unchanged": True,
            "safety_nonzero": False,
            "lifecycle_valid": True,
            "technical_ok": True,
        }

    monkeypatch.setattr(pilot.rank_eval, "_load_policy", fake_load)
    monkeypatch.setattr(pilot.rank_eval, "_runtime_record", fake_runtime_record)
    monkeypatch.setattr(pilot, "_record_annotations", lambda record, config: record)
    monkeypatch.setattr(pilot, "_final_eval_fatal_reason", lambda record: None)
    monkeypatch.setattr(pilot, "_validate_pilot_payload", lambda payload, config: {"phase_config_match": True})
    cells = pilot.neutral_eval._evaluation_cells()
    configs = {str(cell["scenario"]): object() for cell in cells}
    records, reason = pilot._final_evaluation(
        output=tmp_path,
        pilot_path=tmp_path / "pilot.pt",
        neutral_configs=configs,
        checkpoint_sha="sha",
        progress_start=3,
    )
    assert reason is None
    assert len(records) == 6
    assert calls == [(str(cell["scenario"]), int(cell["seed"])) for cell in cells]
    assert len(list((tmp_path / "evaluation" / "records").glob("*.json"))) == 6


def test_update_budget_is_exactly_three_and_uses_fixed_arguments() -> None:
    assert len(pilot.TRAINING_CELLS) == 3
    assert [int(cell["seed"]) for cell in pilot.TRAINING_CELLS] == [873101, 873102, 873103]
    assert pilot.UPDATE_KWARGS == {
        "training_authorized": True,
        "epochs": 2,
        "minibatch_size": 32,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.1,
        "value_coef": 0.5,
        "entropy_coef": 0.001,
        "target_kl": 0.01,
    }


def test_main_flow_calls_three_updates_and_six_final_cells(tmp_path, monkeypatch) -> None:
    base = replace(
        SimConfig(),
        action_schema_version="low_action_schema_v4_2_hierarchical",
        observation_schema_version="phaseA_low_obs_v2_1_retry_state",
        task_priority_enabled=True,
        low_level_priority_time_weight=1.0,
    )
    neutral = replace(base, task_priority_enabled=False, low_level_priority_time_weight=0.0)
    configs = {"flow6": base, "flow12": base, "dense24": base}
    neutral_configs = {"flow6": neutral, "flow12": neutral, "dense24": neutral}
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88)
    update_calls = []
    eval_calls = []

    def fake_update(steps, **kwargs):
        update_calls.append(dict(kwargs))
        return {
            "updates": 1,
            "accepted_updates": 1,
            "attempted_updates": 1,
            "rejected_updates": 0,
            "final_rollout_kl": 0.001,
        }

    policy.update = fake_update

    def good_record(scenario, seed, checkpoint):
        requests = int(pilot.neutral_eval.EXPECTED_REQUESTS[scenario])
        rows = [{"wait_seconds": 1.0, "initial_soc": 0.8, "base_priority": 1} for _ in range(requests)]
        return {
            "label": "task_neutral_pilot_training",
            "scenario": scenario,
            "seed": seed,
            "status": "completed",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": pilot.EXPECTED_CANDIDATE_SHA256,
            "requests": requests,
            "summary": {"terminal": True, "reward_total": 1.0, "reward_component_totals": {}, "safety": {}},
            "rollout_summary": {"terminal": True},
            "action_tape": "tape",
            "lifecycle": {
                "scope": "all_requested_terminal_rows",
                "replay_exact": True,
                "all_requests_observed": True,
                "ledger_alignment": True,
                "missing_request_count": 0,
                "row_count": requests,
                "rows": rows,
                "outcomes": {"completed": requests},
            },
            "outcomes": {"completed": requests},
            "lifecycle_valid": True,
            "outcome_counts": {"requests": requests, "normal": requests, "emergency": 0,
                                "failed_energy": 0, "other": 0, "unresolved": 0},
            "normal": requests,
            "emergency": 0,
            "failed_energy": 0,
            "unresolved": 0,
            "weights_unchanged": True,
            "safety_nonzero": False,
            "teacher_called": False,
            "training_or_update_called": False,
            "PPO_ALLOWED": False,
        }

    def fake_training_record(*, policy, checkpoint, config, scenario, seed, output):
        return [_step(actors=1)], good_record(scenario, seed, checkpoint)

    baseline = []
    for cell in pilot.neutral_eval._evaluation_cells():
        baseline.append(good_record(str(cell["scenario"]), int(cell["seed"]), Path("candidate.pt")))
    monkeypatch.setattr(pilot.neutral_eval, "_build_neutral_inputs", lambda: (configs, neutral_configs, {}))
    monkeypatch.setattr(pilot, "_load_binding", lambda config: (Path("candidate.pt"), policy, {
        "checkpoint_sha256": pilot.EXPECTED_CANDIDATE_SHA256,
        "baseline_scope": "fixed",
    }))
    monkeypatch.setattr(pilot, "_load_baseline_records", lambda configs: (baseline, {"records": 6}))
    monkeypatch.setattr(pilot, "_training_record", fake_training_record)
    monkeypatch.setattr(pilot, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(pilot, "_sha256", lambda path: "sha")
    monkeypatch.setattr(pilot.rank_eval, "_load_policy", lambda path, config: (policy, {}))
    monkeypatch.setattr(pilot, "_roundtrip_pilot", lambda *args, **kwargs: {"checkpoint_sha256": "sha", "exact": True})

    def fake_final(**kwargs):
        records = []
        for cell in pilot.neutral_eval._evaluation_cells():
            eval_calls.append((str(cell["scenario"]), int(cell["seed"])))
            records.append(good_record(str(cell["scenario"]), int(cell["seed"]), Path("pilot.pt")))
        return records, None

    monkeypatch.setattr(pilot, "_final_evaluation", fake_final)
    monkeypatch.setattr(pilot, "_pilot_comparison", lambda baseline, evaluation: {"baseline_cells": 6, "evaluation_cells": 6})
    result = pilot.run(output_dir=tmp_path / "pilot")
    assert result["status"] == "COMPLETED"
    assert len(update_calls) == 3
    assert all(call == pilot.UPDATE_KWARGS for call in update_calls)
    assert len(eval_calls) == 6
    progress = __import__("json").loads((tmp_path / "pilot" / "progress.json").read_text())
    assert progress["training_update_calls"] == 3
    assert progress["episodes_processed"] == 3
    assert progress["evaluation_processed"] == 6


def test_cli_help_exposes_required_fresh_output_and_check_only(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        pilot.build_parser().parse_args(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--output-dir" in output
    assert "--check-only" in output
