from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42, StudentStepV42
from terminal_marl.training import run_v4_2_p3_controlled_emergency_admission_v5 as runner


def _step(*, deterministic: bool = False) -> StudentStepV42:
    return StudentStepV42(
        world=np.zeros((1, 88), dtype=np.float32),
        requested_gate=np.asarray([0], dtype=np.int64),
        requested_proposal=np.asarray([-1], dtype=np.int64),
        gate_logprob=np.asarray([-0.7], dtype=np.float32),
        proposal_logprob=np.asarray([np.nan], dtype=np.float32),
        actor_mask=np.asarray([True], dtype=np.bool_),
        value=0.0,
        next_value=0.0,
        reward=0.0,
        terminated=True,
        episode_end=True,
        stochastic_behavior=not deterministic,
    )


def test_emergency_is_accepted_but_energy_and_unknown_outcomes_are_rejected() -> None:
    emergency = runner._outcome_data(
        {"terminal": True, "outcomes": {"1": "completed", "2": "emergency_landed"}},
        2,
    )
    assert emergency["recognized_outcomes"] is True
    assert emergency["emergency"] == 1
    assert emergency["full_resolution"] is True

    energy = runner._outcome_data(
        {"terminal": True, "outcomes": {"1": "failed_energy"}}, 1,
    )
    assert energy["recognized_outcomes"] is True
    assert energy["zero_failed_energy"] is False
    assert energy["full_resolution"] is False

    unknown = runner._outcome_data(
        {"terminal": True, "outcomes": {"1": "diverted_capacity"}}, 1,
    )
    assert unknown["recognized_outcomes"] is False
    assert unknown["zero_other_failures"] is False


def test_entry_binding_refusal_happens_before_any_rollout() -> None:
    config = SimpleNamespace(
        action_schema_version="low_action_schema_v4_2_hierarchical",
        observation_schema_version="phaseA_low_obs_v2_1_retry_state",
        vehicle_profile=SimpleNamespace(profile_version="eh216s_research_v2"),
        scheduler_mode="execution",
        teacher_enabled=False,
        high_level_enabled=False,
        shield_enabled=True,
    )
    report = {
        "stage": "V5_ENTRY",
        "PASS": True,
        "selected_checkpoint_kind": "canonical",
        "selected_checkpoint_sha256": "0" * 64,
        "schema_binding": runner._schema_binding(config),
        "schema_digest": runner._schema_digest(config),
    }
    with pytest.raises(runner.V5InputError, match="SHA mismatch"):
        runner.validate_entry_report(
            report, config=config, checkpoint_kind="canonical",
            checkpoint_sha=runner.CANONICAL_INITIAL_CHECKPOINT_SHA256,
        )


def test_entry_binding_carries_protocol_optimizer_and_all_config_snapshots() -> None:
    config = SimpleNamespace(
        action_schema_version="low_action_schema_v4_2_hierarchical",
        observation_schema_version="phaseA_low_obs_v2_1_retry_state",
        vehicle_profile=SimpleNamespace(profile_version="eh216s_research_v2"),
        scheduler_mode="execution", teacher_enabled=False,
        high_level_enabled=False, shield_enabled=True,
    )
    p = runner.protocol()
    snapshots = {scenario: {"scenario": scenario} for scenario in runner.ENTRY_SCENARIOS}
    report = {
        "stage": "V5_ENTRY", "PASS": True,
        "selected_checkpoint_kind": "canonical",
        "selected_checkpoint_sha256": runner.CANONICAL_INITIAL_CHECKPOINT_SHA256,
        "PPO_ENTRY_ALLOWED": True,
        "bounded_pilot_training_authorized": True,
        **runner._entry_report_binding(
            config, "canonical", runner.CANONICAL_INITIAL_CHECKPOINT_SHA256,
            actual_config_snapshots=snapshots, p=p,
        ),
    }
    runner.validate_entry_report(
        report, config=config, checkpoint_kind="canonical",
        checkpoint_sha=runner.CANONICAL_INITIAL_CHECKPOINT_SHA256,
        p=p, expected_config_snapshots=snapshots,
    )
    report["actual_config_snapshots"]["dense24"]["scenario"] = "changed"
    with pytest.raises(runner.V5InputError, match="actual configuration snapshots"):
        runner.validate_entry_report(
            report, config=config, checkpoint_kind="canonical",
            checkpoint_sha=runner.CANONICAL_INITIAL_CHECKPOINT_SHA256,
            p=p, expected_config_snapshots=snapshots,
        )


def test_deterministic_steps_are_rejected_by_ppo_update() -> None:
    policy = StudentPPOV42(88, learning_rate=3e-5)
    with pytest.raises(ValueError, match="deterministic evaluation"):
        policy.update([_step(deterministic=True)], training_authorized=True, epochs=1)


def test_frozen_scope_equality_and_restore_contract() -> None:
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88, learning_rate=3e-5)
    before = {key: value.detach().clone() for key, value in policy.model.state_dict().items()}
    after = {key: value.detach().clone() for key, value in policy.model.state_dict().items()}
    adapter_key = next(key for key in after if key.startswith("gate_adapter."))
    after[adapter_key] = after[adapter_key] + 1.0
    assert runner._frozen_state_equal(before, after)

    frozen_key = next(key for key in after if key.startswith("actor_encoder."))
    after[frozen_key] = after[frozen_key] + 1.0
    assert runner._frozen_state_equal(before, after) is False


def test_entry_stops_and_keeps_partial_records(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    p = runner.protocol()
    config = SimpleNamespace(
        total_requested_uavs=6,
        max_simulation_seconds=10.0,
        dt=1.0,
        action_schema_version="low_action_schema_v4_2_hierarchical",
        observation_schema_version="phaseA_low_obs_v2_1_retry_state",
        vehicle_profile=SimpleNamespace(profile_version="eh216s_research_v2"),
        scheduler_mode="execution", teacher_enabled=False,
        high_level_enabled=False, shield_enabled=True,
    )
    dummy = object()
    monkeypatch.setattr(runner, "_build_config", lambda scenario, _: config)
    monkeypatch.setattr(runner, "_load_source", lambda *args, **kwargs: (dummy, {}, ("gate_adapter.x",), "a" * 64))
    calls = []

    def fake_record(**kwargs):
        calls.append(kwargs["seed"])
        passed = len(calls) == 1
        return {
            "seed": kwargs["seed"], "scenario": kwargs["scenario"],
            "hard_gate_pass": passed, "full_resolution": passed,
            "status": "completed", "weights_unchanged": True,
        }

    monkeypatch.setattr(runner, "_record_rollout", fake_record)
    result = runner._entry_stage(
        output=tmp_path / "entry", checkpoint=tmp_path / "checkpoint.pt",
        checkpoint_kind="canonical", p=p,
    )
    assert result["PASS"] is False
    assert result["stopped"] is True
    assert calls == [runner.ENTRY_SEEDS[0], runner.ENTRY_SEEDS[1]]
    manifest = json.loads((tmp_path / "entry" / "manifest.json").read_text())
    assert len(manifest["records"]) == 2


def test_before_after_comparison_keeps_same_seed_pairing() -> None:
    before = {
        "records": [{
            "seed": 849301, "scenario": "flow6", "emergency": 1,
            "failed_energy": 0, "full_resolution": True,
            "requests": 1, "completed": 1,
            "lifecycle": {
                "replay_exact": True,
                "rows": [{"uid": 1, "base_priority": 1, "initial_soc": 0.60,
                          "wait_seconds": 3.0, "landing_soc": 0.50}],
            },
        }]
    }
    after = {
        "records": [{
            "seed": 849301, "scenario": "flow6", "emergency": 0,
            "failed_energy": 0, "full_resolution": True,
            "requests": 1, "completed": 1,
            "lifecycle": {
                "replay_exact": True,
                "rows": [{"uid": 1, "base_priority": 1, "initial_soc": 0.60,
                          "wait_seconds": 2.0, "landing_soc": 0.50}],
            },
        }]
    }
    row = runner._performance_comparison(before, after)["paired_records"][0]
    assert row["seed"] == 849301
    assert row["priority_weighted_wait_delta_seconds"] == pytest.approx(-1.0)
    assert row["low_soc_mean_wait_delta_seconds"] == pytest.approx(-1.0)
    assert row["before_lifecycle"]["completed"] == 1
    pooled = runner._performance_comparison(before, after)["pooled_emergency"]
    assert pooled["sparse"]["before_emergency"] == 1
    assert pooled["sparse"]["after_emergency"] == 0
    assert pooled["sparse"]["report_only"] is True


def test_smoke_stage_preserves_existing_gate_failure_and_reward_check(tmp_path, monkeypatch) -> None:
    tape = tmp_path / "tape.json"
    tape.write_text(json.dumps({"action_tape": [{}, {}]}), encoding="utf-8")
    config = SimpleNamespace(
        total_requested_uavs=24, max_simulation_seconds=1800.0, dt=1.0,
        action_schema_version="low_action_schema_v4_2_hierarchical",
        observation_schema_version="phaseA_low_obs_v2_1_retry_state",
        vehicle_profile=SimpleNamespace(profile_version="eh216s_research_v2"),
        scheduler_mode="execution", teacher_enabled=False,
        high_level_enabled=False, shield_enabled=True,
    )
    monkeypatch.setattr(runner, "_build_config", lambda scenario, _: config)
    monkeypatch.setattr(
        runner, "_load_source", lambda *args, **kwargs: (object(), {}, ("gate_adapter.x",), "a" * 64),
    )
    monkeypatch.setattr(
        runner, "_record_rollout", lambda **kwargs: {
            "action_tape": str(tape), "summary": {
                "safety": {"collisions": 0, "nmac": 0, "hard_separation_violations": 0,
                            "uncertified_execution": 0, "SafetyInvariantError": 0},
            },
            "teacher_enabled": False, "label_rows": 0, "PPO_ALLOWED": False,
            "weights_unchanged": True, "reward_telemetry": {
                "components_finite": True, "total_finite": True,
                "components_consistent": True,
            },
            "safety_invariant_error": False, "hard_gate_pass": False,
            "status": "completed", "_steps": [_step(), _step()],
        },
    )
    result = runner._smoke_stage(
        output=tmp_path / "smoke", checkpoint=tmp_path / "checkpoint.pt",
        checkpoint_kind="canonical", p=runner.protocol(),
    )
    assert result["PASS"] is False
    assert result["status"] == "SMOKE_FAILED"
    assert result["records"][0]["hard_gate_pass"] is False
    assert result["records"][0]["smoke_checks"]["reward_components_consistent"] is True


def test_actual_post_checkpoint_sha_uses_file_bytes(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"v5-checkpoint")
    assert runner._actual_checkpoint_sha(checkpoint, "canonical") == runner._sha256(checkpoint)


def test_before_dev_failure_stops_before_any_pilot_update_or_next_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    p = runner.protocol()
    config = SimpleNamespace(
        action_schema_version="low_action_schema_v4_2_hierarchical",
        observation_schema_version="phaseA_low_obs_v2_1_retry_state",
        vehicle_profile=SimpleNamespace(profile_version="eh216s_research_v2"),
        scheduler_mode="execution", teacher_enabled=False,
        high_level_enabled=False, shield_enabled=True,
    )
    dummy = object()
    monkeypatch.setattr(runner, "_build_config", lambda scenario, _: config)
    monkeypatch.setattr(
        runner, "validate_entry_report", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner, "_load_source", lambda *args, **kwargs: (dummy, {}, ("gate_adapter.x",), "a" * 64),
    )
    monkeypatch.setattr(
        runner, "_dev_stage", lambda **kwargs: {"PASS": False, "records": []},
    )
    update_calls: list[object] = []
    monkeypatch.setattr(
        runner, "_record_rollout", lambda **kwargs: update_calls.append(kwargs) or pytest.fail(
            "pilot rollout must not start after before-dev failure"
        ),
    )
    entry_report = {
        "stage": "V5_ENTRY", "PASS": True,
        "selected_checkpoint_kind": "canonical",
        "selected_checkpoint_sha256": runner.CANONICAL_INITIAL_CHECKPOINT_SHA256,
    }
    result = runner._pilot_stage(
        output=tmp_path / "pilot", checkpoint=tmp_path / "checkpoint.pt",
        checkpoint_kind="canonical", entry_report=entry_report, p=p,
    )
    assert result["PASS"] is False
    assert result["ppo_called"] is False
    assert result["records"] == []
    assert result["dev_after"] is None
    assert update_calls == []


def test_smoke_protocol_binds_two_steps_to_full_dense_horizon() -> None:
    p = runner.protocol()
    assert p["smoke"]["steps"] == 2
    assert p["smoke"]["full_horizon_seconds"] == 1800
    assert p["smoke"]["seed"] == 849000
