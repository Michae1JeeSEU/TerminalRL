from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.algo.gate_reference_v4_2 import GateReference
from terminal_marl.training import p3_kl_recovery_v1 as recovery
from terminal_marl.training import run_p3_adaptive_kl_probe_v1 as adaptive
from terminal_marl.training import run_p3_kl_recovery_probe_v1 as probe
from terminal_marl.training import run_v4_2_p3_gate_anchored_v6 as v6


def _reference() -> GateReference:
    probabilities = torch.tensor([[.75, .25], [.25, .75]], dtype=torch.float32)
    return GateReference(
        observations=torch.zeros((2, 88), dtype=torch.float32),
        reference_logits=torch.log(probabilities),
        reference_probs=probabilities,
        weights=torch.ones(2),
        coefficient=.1,
        max_kl=.05,
        soft_kl_target=.02,
        soft_kl_gain=400.0,
    )


def _check_context() -> dict:
    reference = _reference()
    policy = SimpleNamespace(optimizer=SimpleNamespace(state={"adam": 1}))
    return {
        "run_index": 0,
        "source_checkpoint": Path("source.pt"),
        "source_checkpoint_sha256": "source-sha",
        "base_checkpoint": Path("stopped.pt"),
        "base_checkpoint_sha256": "stopped-sha",
        "protocol_sha256": "protocol-sha",
        "reference": reference,
        "reference_digest": "reference-sha",
        "legacy_reference_digest": "reference-sha",
        "reference_options": dict(adaptive.REFERENCE_OPTIONS),
        "reference_soft_settings": {
            "soft_kl_target": .02,
            "soft_kl_gain": 400.0,
            "coefficient": .1,
            "max_kl": .05,
        },
        "initial_reference_metrics": {"effective_kl_coefficient": .1},
        "trainable": ("gate_adapter.weight",),
        "policy": policy,
        "adaptive_entry": adaptive._entry(0),
    }


def test_adaptive_plan_marks_first_four_replays_and_last_four_unconsumed():
    run0 = adaptive._entry(0)
    run2 = adaptive._entry(2)

    assert run0["seeds"] == list(range(860080, 860088))
    assert run2["seeds"] == list(range(862049, 862057))
    assert run0["scenarios"] == [
        "dense24", "flow6", "dense24", "flow12",
        "dense24", "flow6", "dense24", "flow12",
    ]
    assert run2["scenarios"] == [
        "flow6", "dense24", "flow12", "dense24",
        "flow6", "dense24", "flow12", "dense24",
    ]
    assert run0["diagnostic_replay_count"] == 4
    assert run2["diagnostic_replay_count"] == 4


def test_soft_reference_keeps_legacy_q_digest_and_recovery_freezes_new_fields():
    torch.manual_seed(31)
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(88)
    observations = np.random.default_rng(31).normal(size=(6, 88)).astype(np.float32)
    weights = np.arange(1, 7, dtype=np.float32)
    legacy = GateReference.from_policy(
        policy, observations, weights, coefficient=.1, max_kl=.05,
    )
    soft = GateReference.from_policy(
        policy, observations, weights, coefficient=.1, max_kl=.05,
        soft_kl_target=.02, soft_kl_gain=400.0,
    )

    assert v6._reference_digest(legacy) == v6._reference_digest(soft)
    assert torch.equal(legacy.observations, soft.observations)
    assert torch.equal(legacy.reference_logits, soft.reference_logits)
    assert torch.equal(legacy.reference_probs, soft.reference_probs)
    assert torch.equal(legacy.weights, soft.weights)
    snapshot = recovery._reference_snapshot(soft)
    assert snapshot["soft_kl_target"] == pytest.approx(.02)
    assert snapshot["soft_kl_gain"] == pytest.approx(400.0)
    assert recovery._reference_equal(soft, snapshot)

    weighted_kl = torch.tensor(.03, requires_grad=True)
    loss = soft.regularization_loss(weighted_kl)
    assert loss.item() == pytest.approx(.043)
    assert torch.autograd.grad(loss, weighted_kl)[0].item() == pytest.approx(8.1)


def test_check_only_does_not_create_output_and_reports_soft_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    context = _check_context()
    monkeypatch.setattr(adaptive, "_context", lambda _run_index: context)
    output = tmp_path / "adaptive_check_only"

    result = adaptive.run(run_index=0, output=output, check_only=True)

    assert result["status"] == "CHECK_ONLY_OK"
    assert result["output_created"] is False
    assert result["reference_options"] == adaptive.REFERENCE_OPTIONS
    assert result["reference_soft_settings"]["soft_kl_target"] == pytest.approx(.02)
    assert result["reference_soft_settings"]["soft_kl_gain"] == pytest.approx(400.0)
    assert result["effective_kl_coefficient"] == pytest.approx(.1)
    assert not output.exists()


def test_checkpoint_sidecar_preserves_soft_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    reference = _reference()
    context = {
        "policy": object(),
        "protocol": {},
        "reference": reference,
        "reference_digest": "reference-sha",
        "trainable": ("gate_adapter.weight",),
        "protocol_sha256": "protocol-sha",
        "stage": adaptive.STAGE,
    }

    def fake_save(policy, checkpoint, *_args, **_kwargs):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"adaptive-checkpoint")
        return policy, True

    monkeypatch.setattr(probe.v6, "_save_and_restore", fake_save)
    output = tmp_path / "run"
    _, exact, checkpoint = probe._checkpoint(
        context, output, 1, 12, object(), ppo_called=True,
        reference_options=dict(adaptive.REFERENCE_OPTIONS),
    )

    sidecar = checkpoint.with_name(checkpoint.name + ".reference.json")
    payload = v6._read_json(sidecar)
    assert exact is True
    assert checkpoint.is_file()
    assert payload["reference_digest"] == "reference-sha"
    assert payload["soft_kl_target"] == pytest.approx(.02)
    assert payload["soft_kl_gain"] == pytest.approx(400.0)
    assert payload["reference_options"] == adaptive.REFERENCE_OPTIONS


def test_adaptive_runner_forwards_eight_episode_entry_and_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    context = _check_context()
    context["stage"] = adaptive.STAGE
    monkeypatch.setattr(adaptive, "_context", lambda _run_index: context)
    captured = {}

    def fake_run(run_context, output, *, entry):
        captured["context"] = run_context
        captured["output"] = output
        captured["entry"] = entry
        return {"status": "COMPLETED", "stage": run_context["stage"]}

    monkeypatch.setattr(adaptive.base, "_run", fake_run)
    result = adaptive.run(run_index=0, output=tmp_path / "adaptive")

    assert result["status"] == "COMPLETED"
    assert captured["context"]["stage"] == adaptive.STAGE
    assert len(captured["entry"]["seeds"]) == 8
    assert captured["entry"]["diagnostic_replay_count"] == 4


def test_adaptive_entry_stops_after_first_rollout_hard_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    context = _check_context()
    config = SimpleNamespace(max_simulation_seconds=1.0, dt=1.0)
    context.update(
        configs={"dense24": config, "flow6": config, "flow12": config},
        runtime_protocol={},
        base_checkpoint=tmp_path / "base.pt",
    )
    context["base_checkpoint"].write_bytes(b"base")
    monkeypatch.setattr(
        probe, "_dev_run",
        lambda *_args, **_kwargs: {
            "hard_failure": False, "full_resolution": True, "records": [],
        },
    )
    monkeypatch.setattr(probe.v6, "_rng_state", lambda: {})
    monkeypatch.setattr(probe.v6, "_restore_rng_state", lambda _state: None)
    monkeypatch.setattr(
        probe.v6.admission, "_record_rollout",
        lambda **kwargs: {
            "scenario": kwargs["scenario"], "seed": kwargs["seed"],
            "action_tape": "tape.json", "_steps": [],
        },
    )
    monkeypatch.setattr(probe.v6, "_training_pass", lambda _record: False)
    monkeypatch.setattr(probe.v6.formal_v5, "_public_record", lambda record: dict(record))
    monkeypatch.setattr(probe.v6, "_jsonable", lambda value: value)

    result = probe._run(
        context,
        tmp_path / "adaptive_run",
        entry={
            "seeds": [860080, 860081],
            "scenarios": ["dense24", "dense24"],
            "diagnostic_replay_count": 2,
        },
    )

    assert result["status"] == "STOPPED"
    assert result["processed_episodes"] == 1
    assert result["records"][0]["diagnostic_seed"] == 860080
    assert result["stop_reason"] == "rollout hard failure"
