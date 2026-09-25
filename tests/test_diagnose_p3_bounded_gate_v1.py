from __future__ import annotations

import json
from pathlib import Path

import pytest
from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATION_TEMPERATURE,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import save_checkpoint
from terminal_marl.training import diagnose_p3_bounded_gate_v1 as diagnostic


def _source_fixture(tmp_path: Path, run_index: int, config) -> Path:
    root = tmp_path / "source"
    run = root / f"run{run_index}"
    checkpoint_dir = run / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "checkpoint_episode_0001.pt"
    source = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        88, temperature=CALIBRATION_TEMPERATURE,
    )
    save_checkpoint(checkpoint, source, config, stage="fixture", formal=True)
    manifest = {
        "run_index": run_index,
        "status": "FAILED",
        "accepted_episodes": 1,
        "episodes_recorded": 2,
        "final_checkpoint": f"/home/user/training_outputs/terminal/source/run{run_index}/checkpoints/{checkpoint.name}",
        "train_seeds": [850001, 850002],
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_source_binding_uses_final_checkpoint_and_next_failure_seed(tmp_path):
    config = diagnostic.build_config()
    root = _source_fixture(tmp_path, 0, config)

    binding = diagnostic.source_binding(0, source_root=root)

    assert Path(binding["source_checkpoint"]).name == "checkpoint_episode_0001.pt"
    assert binding["failure_seed"] == 850002
    assert binding["failure_episode_number"] == 2


def test_build_config_applies_canonical_wait():
    config = diagnostic.build_config()

    assert config.v4_2_rejected_release_hold_mode == "canonical_wait"
    assert config.action_schema_version == "low_action_schema_v4_2_hierarchical"
    assert config.observation_schema_version == "phaseA_low_obs_v2_1_retry_state"
    assert config.teacher_enabled is False
    assert config.scheduler_mode == "execution"


def test_run_uses_stochastic_no_label_rollout_and_roundtrips_candidate(tmp_path, monkeypatch):
    config = diagnostic.build_config()
    root = _source_fixture(tmp_path, 0, config)
    output = tmp_path / "diagnostic"
    seen = {}

    def fake_rollout(policy, loaded_config, *, seed, max_steps, output,
                     deterministic, collect_labels):
        seen.update({
            "policy": policy,
            "config": loaded_config,
            "seed": seed,
            "max_steps": max_steps,
            "output": output,
            "deterministic": deterministic,
            "collect_labels": collect_labels,
        })
        def forbidden_update(*args, **kwargs):
            raise AssertionError("diagnostic must not call PPO update")
        policy.update = forbidden_update
        output.mkdir(parents=True, exist_ok=False)
        summary = {
            "seed": seed,
            "steps": 2,
            "terminal": False,
            "outcomes": {},
            "safety": {
                "collisions": 0,
                "nmac": 0,
                "hard_separation_violations": 0,
                "uncertified_execution": 0,
                "SafetyInvariantError": 0,
            },
            "safety_error": None,
            "PPO_ALLOWED": False,
        }
        return [], summary

    monkeypatch.setattr(diagnostic.student_rollout, "rollout", fake_rollout)
    result = diagnostic.run(run_index=0, output=output, max_steps=2, source_root=root)

    assert seen["seed"] == 850002
    assert seen["max_steps"] == 2
    assert seen["deterministic"] is False
    assert seen["collect_labels"] is False
    assert seen["config"].v4_2_rejected_release_hold_mode == "canonical_wait"
    assert float(seen["policy"].model.minimum_gate_probability) == pytest.approx(0.05)
    assert result["ppo_called"] is False
    assert result["training_called"] is False
    assert result["full_outcomes"]["unresolved"] == int(config.total_requested_uavs)
    assert result["migration"]["old_weights_unchanged"] is True
    assert result["checkpoint_restore"]["model_exact"] is True
    assert result["checkpoint_restore"]["optimizer_exact"] is True
    assert result["checkpoint_restore"]["restored_architecture_id"] == diagnostic.BOUNDED_CALIBRATED_ARCHITECTURE_ID
    assert result["checkpoint_restore"]["restored_floor"] == pytest.approx(0.05)
    assert result["checkpoint_restore"]["restored_optimizer_state_entries"] == 0
    assert (output / "bounded_gate_candidate.pt").is_file()
    assert (output / "summary.json").is_file()


def test_run_does_not_overwrite_existing_output(tmp_path):
    config = diagnostic.build_config()
    root = _source_fixture(tmp_path, 0, config)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("keep", encoding="utf-8")

    try:
        diagnostic.run(run_index=0, output=output, max_steps=2, source_root=root)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output directory was overwritten")
    assert marker.read_text(encoding="utf-8") == "keep"
