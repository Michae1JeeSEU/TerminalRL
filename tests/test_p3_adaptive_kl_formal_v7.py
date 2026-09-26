from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.algo.gate_reference_v4_2 import GateReference
from terminal_marl.training import run_p3_adaptive_kl_formal_v7 as runner


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _admission(tmp_path: Path, *, stage: str = "P3_ADAPTIVE_KL_FORMAL_ADMISSION_V7") -> Path:
    manifests = []
    for index in range(2):
        manifest = tmp_path / f"adaptive_{index}.json"
        _write_json(manifest, {
            "stage": "P3_ADAPTIVE_KL_PROBE_V1",
            "status": "COMPLETED",
            "run_index": index if index == 0 else 2,
            "source_checkpoint_sha256": runner.SOURCE_CHECKPOINT_SHA256,
            "reference_options": {"soft_kl_target": .02, "soft_kl_gain": 400.0},
            "reference_gate_coefficient": .1,
            "reference_gate_max_kl": .05,
            "reference_soft_settings": {"soft_kl_target": .02, "soft_kl_gain": 400.0},
        })
        manifests.append({"path": str(manifest), "sha256": runner._sha256(manifest)})
    report = tmp_path / "admission.json"
    _write_json(report, {
        "stage": stage,
        "FORMAL_P3_ALLOWED": True,
        "source_checkpoint_sha256": runner.SOURCE_CHECKPOINT_SHA256,
        "protocol_sha256": "protocol-sha",
        "adaptive_probe_manifests": manifests,
    })
    return report


def test_protocol_binds_v7_schedule_soft_penalty_and_evaluation_plan():
    protocol = runner._protocol()
    runner._validate_schedule(protocol)
    assert protocol["optimizer"]["reference_gate_kl_soft_target"] == pytest.approx(.02)
    assert protocol["optimizer"]["reference_gate_kl_soft_gain"] == pytest.approx(400.)
    assert runner._schedule(protocol, 0)[0] == list(range(870001, 870241))
    assert runner._schedule(protocol, 1)[0] == list(range(871001, 871241))
    assert runner._schedule(protocol, 2)[0] == list(range(872001, 872241))
    assert protocol["post_training_evaluation"]["run_in_this_task"] is True


def test_admission_requires_v7_stage_and_two_completed_probe_manifests(tmp_path: Path):
    report = _admission(tmp_path)
    checked = runner._validate_admission(report, "protocol-sha")
    assert len(checked["adaptive_probe_manifests"]) == 2
    with pytest.raises(runner.V7InputError, match="V7"):
        runner._validate_admission(_admission(tmp_path, stage="P3_FORMAL_ADMISSION_V5"), "protocol-sha")


def _context_for_training() -> dict:
    protocol = runner._protocol()
    config = SimpleNamespace(max_simulation_seconds=1.0, dt=1.0)
    policy = SimpleNamespace(model=SimpleNamespace(state_dict=lambda: {"gate_adapter": torch.zeros(1)}))
    reference = SimpleNamespace(coefficient=.1, soft_kl_target=.02, soft_kl_gain=400., max_kl=.05)
    return {
        "protocol": protocol,
        "run_index": 0,
        "protocol_sha256": "protocol-sha",
        "protocol_digest": "protocol-digest",
        "runtime_protocol": {},
        "configs": {"flow6": config, "flow12": config, "dense24": config},
        "policy": policy,
        "checkpoint": Path("source.pt"),
        "source_payload": {"model": {"gate_adapter": torch.zeros(1)}},
        "loaded_sha": runner.SOURCE_CHECKPOINT_SHA256,
        "trainable": ("gate_adapter",),
        "old_rows": [],
        "coverage_rows": [],
        "reference": reference,
        "reference_digest": "reference-digest",
        "output": Path("unused"),
        "authorization": {"adaptive_probe_manifests": []},
        "config_snapshots": {},
    }


def _patch_training(monkeypatch: pytest.MonkeyPatch, *, recovery_status: str):
    def fake_dev(*args, **kwargs):
        return {"hard_failure": False, "full_resolution": True, "complete": True, "records": []}

    monkeypatch.setattr(runner, "_dev_evaluate", fake_dev)
    monkeypatch.setattr(runner.v6, "_training_pass", lambda record: True)
    monkeypatch.setattr(
        runner.v6.admission, "_record_rollout",
        lambda **kwargs: {"status": "completed", "hard_gate_pass": True, "failed_energy": 0,
                           "other_failure_count": 0, "safety_invariant_error": False,
                           "reward_telemetry": {"available": True, "components_consistent": True},
                           "gates": {"safety_all_zero": True}, "weights_unchanged": True,
                           "_steps": [object()], "action_tape": "tape.json"},
    )
    monkeypatch.setattr(runner, "_recovery", lambda *args, **kwargs: {"status": recovery_status,
                                                                         "update": {"accepted_updates": 0}})
    def fake_save(context, checkpoint, config, **kwargs):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return context["policy"], True, True
    monkeypatch.setattr(runner, "_save_and_restore", fake_save)
    monkeypatch.setattr(runner, "_gate_adapter_l2_delta", lambda *args: 1.0)
    monkeypatch.setattr(runner.v6.formal_v5, "_public_record", lambda record: dict(record))


def test_safe_skips_are_counted_and_ten_consecutive_stops_without_after_dev(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    _patch_training(monkeypatch, recovery_status="SKIPPED_KL")
    result = runner._training_run(_context_for_training(), tmp_path / "run0")
    assert result["status"] == "LEARNING_STALLED"
    assert result["processed_episodes"] == 10
    assert result["updated_episodes"] == 0
    assert result["skipped_kl_episodes"] == 10
    assert result["accepted_minibatches"] == 0
    assert result["dev_after"] is None


def test_hard_failed_update_stops_without_success_checkpoint_or_after_dev(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    _patch_training(monkeypatch, recovery_status="HARD_FAILED")
    result = runner._training_run(_context_for_training(), tmp_path / "run0")
    assert result["status"] == "FAILED"
    assert result["processed_episodes"] == 1
    assert result["hard_failed_episodes"] == 1
    assert result["checkpoints"] == []
    assert result["dev_after"] is None


def test_development_cadence_uses_processed_episode_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    _patch_training(monkeypatch, recovery_status="UPDATED")
    calls = []

    def fake_dev(*args, **kwargs):
        calls.append({key: kwargs[key] for key in ("processed_episodes", "accepted_updates")})
        return {"hard_failure": False, "full_resolution": True, "complete": True, "records": []}

    monkeypatch.setattr(runner, "_dev_evaluate", fake_dev)
    monkeypatch.setattr(
        runner, "_schedule",
        lambda protocol, run_index: (list(range(40)), ["flow6"] * 40),
    )
    result = runner._training_run(_context_for_training(), tmp_path / "run0")
    assert result["status"] == "COMPLETED"
    assert calls == [
        {"processed_episodes": 0, "accepted_updates": 0},
        {"processed_episodes": 40, "accepted_updates": 40},
    ]


def test_processed_cadence_does_not_repeat_after_late_update(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    _patch_training(monkeypatch, recovery_status="UPDATED")
    statuses = iter(["UPDATED"] * 39 + ["SKIPPED_KL", "UPDATED"])
    monkeypatch.setattr(
        runner, "_recovery",
        lambda *args, **kwargs: {"status": next(statuses), "update": {"accepted_updates": 0}},
    )
    calls = []

    def fake_dev(*args, **kwargs):
        calls.append(kwargs["processed_episodes"])
        return {"hard_failure": False, "full_resolution": True, "complete": True, "records": []}

    monkeypatch.setattr(runner, "_dev_evaluate", fake_dev)
    monkeypatch.setattr(
        runner, "_schedule",
        lambda protocol, run_index: (list(range(41)), ["flow6"] * 41),
    )
    result = runner._training_run(_context_for_training(), tmp_path / "run0")
    assert result["status"] == "COMPLETED"
    assert result["processed_episodes"] == 41
    assert result["updated_episodes"] == 40
    assert result["skipped_kl_episodes"] == 1
    assert calls == [0, 40]


def test_checkpoint_roundtrip_records_soft_binding_and_real_adam(tmp_path: Path):
    protocol = runner._protocol()
    configs, _ = runner._build_configs(protocol)
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88, learning_rate=3e-5)
    trainable_names = runner.v6._trainable_names(policy)
    assert any(name.startswith("gate_adapter.") for name in trainable_names)
    assert any(name.startswith("critic_encoder.") for name in trainable_names)
    assert any(name.startswith("critic_head.") for name in trainable_names)
    policy.optimizer = torch.optim.Adam(
        [parameter for parameter in policy.model.parameters() if parameter.requires_grad], lr=3e-5,
    )
    one_step_loss = sum(
        parameter.float().square().sum()
        for parameter in policy.model.parameters() if parameter.requires_grad
    )
    one_step_loss.backward()
    policy.optimizer.step()
    policy.optimizer.zero_grad(set_to_none=True)
    assert policy.optimizer.state
    observations = np.zeros((4, 88), dtype=np.float32)
    reference = GateReference.from_policy(
        policy, observations, np.ones(4, dtype=np.float32),
        coefficient=.1, max_kl=.05, soft_kl_target=.02, soft_kl_gain=400.,
    )
    context = {
        "policy": policy,
        "reference": reference,
        "protocol": protocol,
        "protocol_sha256": runner._sha256(runner.PROTOCOL),
        "protocol_digest": runner._protocol_digest(protocol),
        "reference_digest": runner.v6._reference_digest(reference),
        "trainable": runner.v6._trainable_names(policy),
        "source_payload": {"model": policy.model.state_dict()},
        "loaded_sha": runner.SOURCE_CHECKPOINT_SHA256,
    }
    restored, exact, rng_exact = runner._save_and_restore(
        context, tmp_path / "checkpoint.pt", configs["dense24"],
        episode_index=1, timestep=4, ppo_called=True,
        counters={"processed": 1, "updated": 1, "skipped": 0},
        gate_adapter_parameter_delta_norm=0.0,
    )
    assert exact is True
    assert rng_exact is True
    assert restored.optimizer.state
    assert runner.v6.admission._state_equal(
        policy.optimizer.state_dict(), restored.optimizer.state_dict(),
    )
    payload = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert payload["reference_gate_soft_target"] == pytest.approx(.02)
    assert payload["reference_gate_soft_gain"] == pytest.approx(400.)
    assert payload["rng_state"] is not None
    assert payload["processed_updated_skipped_counters"] == {
        "processed": 1, "updated": 1, "skipped": 0,
    }
    assert payload["gate_adapter_parameter_delta_norm"] == pytest.approx(0.0)
