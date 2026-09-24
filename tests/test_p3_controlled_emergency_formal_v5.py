from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from terminal_marl.training import (
    run_v4_2_p3_controlled_emergency_formal_v5 as runner,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _minimal_context(tmp_path: Path) -> dict:
    formal = runner._formal_protocol()
    configs = {
        scenario: SimpleNamespace(max_simulation_seconds=2.0, dt=1.0)
        for scenario in ("flow6", "flow12", "dense24")
    }
    return {
        "formal": formal,
        "formal_protocol_sha256": "f" * 64,
        "admission_protocol": {},
        "admission_protocol_sha256": "a" * 64,
        "authorization": {},
        "configs": configs,
        "config_snapshots": {name: {"scenario": name} for name in configs},
        "policy": object(),
        "metadata": {},
        "trainable": ("gate_adapter.weight",),
        "loaded_sha": runner.CHECKPOINT_SHA256,
        "run_index": 0,
        "checkpoint": tmp_path / "initial.pt",
    }


def test_schedule_is_exact_and_disjoint() -> None:
    formal = runner._formal_protocol()
    runner._validate_schedule_disjoint(formal)

    assert formal["formal_schedule"]["run_indices"] == [0, 1, 2]
    assert formal["formal_schedule"]["episodes_per_run"] == 240
    assert formal["formal_schedule"]["episode_pattern"] == list(runner.EPISODE_PATTERN)
    assert runner._expected_train_seeds(formal, 0) == list(range(850001, 850241))
    assert runner._expected_train_seeds(formal, 1) == list(range(851001, 851241))
    assert runner._expected_train_seeds(formal, 2) == list(range(852001, 852241))
    assert tuple(formal["development"]["seeds"]) == runner.DEV_SEEDS
    assert set(runner.DEV_SEEDS).isdisjoint(runner.HELDOUT_SEEDS)


def test_formal_config_binds_current_horizons_and_late_slots_off() -> None:
    protocol = runner.admission.protocol()
    configs, _ = runner._config_context(protocol)
    assert configs["flow6"].max_simulation_seconds == 1210.0
    assert configs["flow12"].max_simulation_seconds == 1570.0
    assert configs["dense24"].max_simulation_seconds == 1800.0
    assert all(not config.reconcile_late_merge_slots_enabled for config in configs.values())


def test_admission_runtime_bindings_must_match_current_configs() -> None:
    protocol = runner.admission.protocol()
    configs, snapshots = runner._config_context(protocol)
    report = {
        "actual_config_snapshots": runner._jsonable(snapshots),
        "optimizer_boundary": runner._jsonable(protocol["optimizer"]),
    }
    runner._validate_runtime_bindings(report, configs, snapshots, protocol)
    report["actual_config_snapshots"]["dense24"]["max_simulation_seconds"] = -1
    with pytest.raises(runner.FormalInputError, match="configuration snapshots mismatch"):
        runner._validate_runtime_bindings(report, configs, snapshots, protocol)


def test_false_authorization_is_rejected_before_strict_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    report_path = tmp_path / "admission.json"
    _write_json(report_path, {
        "stage": runner.ADMISSION_REPORT_STAGE,
        "FORMAL_P3_ALLOWED": False,
    })
    calls: list[object] = []
    monkeypatch.setattr(
        runner.admission, "_load_source",
        lambda *args, **kwargs: calls.append(args) or pytest.fail("strict load called"),
    )
    with pytest.raises(runner.FormalInputError, match="authorization is false"):
        runner._preflight(
            run_index=0, admission_report=report_path,
            checkpoint=tmp_path / "checkpoint.pt",
        )
    assert calls == []


def test_mismatched_pilot_sha_is_rejected_before_strict_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    pilot_path = tmp_path / "pilot.json"
    _write_json(pilot_path, {
        "PILOT_TECHNICAL_PASS": True,
        "selected_checkpoint_sha256": runner.CHECKPOINT_SHA256,
    })
    report_path = tmp_path / "admission.json"
    _write_json(report_path, {
        "stage": runner.ADMISSION_REPORT_STAGE,
        "FORMAL_P3_ALLOWED": True,
        "initial_checkpoint_sha256": runner.CHECKPOINT_SHA256,
        "admission_protocol_sha256": runner._sha256(runner.admission.PROTOCOL),
        "formal_protocol_sha256": runner._sha256(runner.FORMAL_PROTOCOL),
        "pilot_manifest_path": str(pilot_path),
        "pilot_manifest_sha256": "0" * 64,
    })
    calls: list[object] = []
    monkeypatch.setattr(
        runner.admission, "_load_source",
        lambda *args, **kwargs: calls.append(args) or pytest.fail("strict load called"),
    )
    with pytest.raises(runner.FormalInputError, match="pilot manifest SHA mismatch"):
        runner._preflight(
            run_index=0, admission_report=report_path,
            checkpoint=tmp_path / "checkpoint.pt",
        )
    assert calls == []


def test_check_only_does_not_create_output_or_call_rollout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    context = _minimal_context(tmp_path)
    monkeypatch.setattr(runner, "_preflight", lambda **kwargs: context)
    monkeypatch.setattr(
        runner, "_dev_evaluation",
        lambda **kwargs: pytest.fail("check-only performed evaluation"),
    )
    output = tmp_path / "formal-output"
    result = runner.run(
        run_index=0, admission_report=tmp_path / "report.json",
        checkpoint=tmp_path / "initial.pt", output=output, check_only=True,
    )
    assert result["status"] == "CHECK_ONLY_OK"
    assert result["strict_load"] is True
    assert result["output_created"] is False
    assert not output.exists()


def test_training_unresolved_is_report_only_but_dev_requires_resolution() -> None:
    record = {
        "status": "completed",
        "hard_gate_pass": True,
        "weights_unchanged": True,
        "failed_energy": 0,
        "other_failure_count": 0,
        "safety_invariant_error": False,
        "reward_telemetry": {"available": True, "components_consistent": True},
        "gates": {"safety": True, "full_resolution": False},
        "full_resolution": False,
    }
    assert runner._training_hard_pass(record) is True
    assert runner._dev_hard_pass(record) is False


def test_partial_training_failure_preserves_progress_and_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    context = _minimal_context(tmp_path)
    monkeypatch.setattr(runner, "_preflight", lambda **kwargs: context)

    def fake_dev(**kwargs):
        kwargs["output"].mkdir(parents=True, exist_ok=False)
        return {
            "stage": "P3_FORMAL_DEV", "accepted_episodes": kwargs["accepted_episodes"],
            "records": [], "complete": True, "full_resolution": True,
            "hard_failure": False, "same_initial_states": True,
            "deterministic": True, "heldout": False,
        }

    monkeypatch.setattr(runner, "_dev_evaluation", fake_dev)
    calls: list[int] = []

    def fake_record(**kwargs):
        calls.append(kwargs["seed"])
        return {
            "seed": kwargs["seed"], "scenario": kwargs["scenario"],
            "status": "completed", "hard_gate_pass": True,
            "weights_unchanged": True, "failed_energy": 0,
            "other_failure_count": 0, "safety_invariant_error": False,
            "reward_telemetry": {"available": True, "components_consistent": True},
            "gates": {"safety": True, "full_resolution": False},
            "action_tape": str(tmp_path / "action_tape.json"),
            "_steps": [object()],
        }

    monkeypatch.setattr(runner.admission, "_record_rollout", fake_record)
    monkeypatch.setattr(
        runner.p3_legacy, "_transactional_update",
        lambda *args, **kwargs: ({"accepted_updates": 0}, {"update_transaction": False}),
    )
    output = tmp_path / "formal-output"
    result = runner.run(
        run_index=0, admission_report=tmp_path / "report.json",
        checkpoint=tmp_path / "initial.pt", output=output,
    )
    assert result["status"] == "FAILED"
    assert result["stopped"] is True
    assert result["episodes_recorded"] == 1
    assert result["ppo_called"] is True
    assert calls == [850001]
    assert (output / "progress.json").is_file()
    assert (output / "manifest.json").is_file()
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert progress["records"][0]["accepted_episode"] is False


def test_success_then_restore_failure_uses_latest_checkpoint_and_counts_only_restored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    context = _minimal_context(tmp_path)
    monkeypatch.setattr(runner, "_preflight", lambda **kwargs: context)

    def fake_dev(**kwargs):
        kwargs["output"].mkdir(parents=True, exist_ok=False)
        return {
            "stage": "P3_FORMAL_DEV", "accepted_episodes": kwargs["accepted_episodes"],
            "records": [], "complete": True, "full_resolution": True,
            "hard_failure": False, "same_initial_states": True,
            "deterministic": True, "heldout": False,
        }

    monkeypatch.setattr(runner, "_dev_evaluation", fake_dev)
    rollout_checkpoints: list[Path] = []
    record_calls = 0

    def fake_record(**kwargs):
        nonlocal record_calls
        record_calls += 1
        rollout_checkpoints.append(Path(kwargs["checkpoint"]))
        return {
            "seed": kwargs["seed"], "scenario": kwargs["scenario"],
            "status": "completed", "hard_gate_pass": True,
            "weights_unchanged": True, "failed_energy": 0,
            "other_failure_count": 0, "safety_invariant_error": False,
            "reward_telemetry": {"available": True, "components_consistent": True},
            "gates": {"safety": True, "full_resolution": False},
            "action_tape": str(tmp_path / f"action_tape_{record_calls}.json"),
            "_steps": [object()],
        }

    monkeypatch.setattr(runner.admission, "_record_rollout", fake_record)
    monkeypatch.setattr(
        runner.p3_legacy, "_transactional_update",
        lambda *args, **kwargs: ({"accepted_updates": 1}, {"update_transaction": True}),
    )
    restore_calls: list[Path] = []

    def fake_save_restore(policy, checkpoint, config, admission_protocol, formal_sha, **kwargs):
        del config, admission_protocol, formal_sha, kwargs
        restore_calls.append(Path(checkpoint))
        return object(), len(restore_calls) == 1

    monkeypatch.setattr(runner, "_save_and_restore", fake_save_restore)
    output = tmp_path / "formal-output"
    result = runner.run(
        run_index=0, admission_report=tmp_path / "report.json",
        checkpoint=tmp_path / "initial.pt", output=output,
    )
    first_checkpoint = output / "checkpoints" / "checkpoint_episode_0001.pt"
    assert result["status"] == "FAILED"
    assert result["accepted_episodes"] == 1
    assert result["episodes_recorded"] == 2
    assert rollout_checkpoints == [context["checkpoint"], first_checkpoint]
    assert restore_calls == [first_checkpoint, output / "checkpoints" / "checkpoint_episode_0002.pt"]
    assert result["records"][0]["accepted_episode"] is True
    assert result["records"][1]["accepted_episode"] is False
