from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from terminal_marl.training import run_v4_2_p3_gate_anchored_v6 as runner


def _v6() -> dict:
    return runner._protocol()


def _config() -> SimpleNamespace:
    return SimpleNamespace(max_simulation_seconds=1.0, dt=1.0)


def _context(tmp_path: Path, *, formal: bool = False) -> dict:
    p = _v6()
    config = _config()
    policy = SimpleNamespace(gate_reference=None, optimizer=SimpleNamespace(state={}))
    reference = SimpleNamespace(coefficient=0.1, max_kl=0.05)
    return {
        "stage": "formal" if formal else "pilot",
        "v6": p,
        "runtime_protocol": {},
        "protocol_sha256": "protocol",
        "configs": {"flow6": config, "flow12": config, "dense24": config},
        "config_snapshots": {},
        "policy": policy,
        "source_payload": {},
        "trainable": ("gate_adapter.weight",),
        "loaded_sha": runner.SOURCE_CHECKPOINT_SHA256,
        "old_rows": [],
        "coverage_rows": [],
        "weights": np.ones(1),
        "weight_report": {},
        "reference": reference,
        "reference_digest": "reference",
        "authorization": None,
        "output": tmp_path / "source.pt",
    }


def _fake_record(*, failed_energy: int = 0) -> dict:
    return {
        "status": "completed",
        "requests": 1,
        "completed": 1,
        "normal": 1,
        "emergency": 0,
        "failed_energy": failed_energy,
        "unresolved": 0,
        "other_failure_count": 0,
        "weights_unchanged": True,
        "teacher_enabled": False,
        "label_rows": 0,
        "PPO_ALLOWED": False,
        "hard_gate_pass": True,
        "safety_invariant_error": False,
        "reward_telemetry": {"available": True, "components_consistent": True},
        "gates": {"safety_all_zero": True},
        "action_tape": "tape.json",
        "_steps": [SimpleNamespace()],
        "summary": {},
        "scenario": "dense24",
        "seed": 1,
    }


def test_schedule_is_exact_and_disjoint() -> None:
    p = _v6()
    runner._validate_schedule(p)
    assert runner._schedule(p)[0] == list(range(859201, 859213))
    assert runner._schedule(p)[1] == list(runner.EPISODE_PATTERN) * 3
    assert runner._schedule(p, 0)[0] == list(range(860001, 860241))
    assert runner._schedule(p, 1)[0] == list(range(861001, 861241))
    assert runner._schedule(p, 2)[0] == list(range(862001, 862241))


def test_schedule_overlap_is_rejected() -> None:
    p = _v6()
    p["formal_schedule"]["train_seeds"][1]["start"] = 860001
    p["formal_schedule"]["train_seeds"][1]["end"] = 860240
    with pytest.raises(runner.V6InputError, match="overlap"):
        runner._validate_schedule(p)


def test_optimizer_aliases_use_v6_values() -> None:
    runtime = runner._runtime_protocol(_v6())
    assert runtime["optimizer"]["target_kl"] == 0.01
    assert runtime["optimizer"]["minimum_accepted_updates"] == 1
    assert runtime["optimizer"]["maximum_rejected_updates"] == 1


def test_reference_snapshot_has_fixed_source_digest() -> None:
    policy = SimpleNamespace()

    class _Model:
        def distributions(self, observations):
            logits = torch.zeros((len(observations), 2))
            return torch.distributions.Categorical(logits=logits), None

        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1))])

    policy.model = _Model()
    reference = runner.GateReference.from_policy(
        policy, np.zeros((2, 88), dtype=np.float32), np.ones(2),
    )
    digest = runner._reference_digest(reference)
    assert digest == runner._reference_digest(reference)
    assert reference.max_kl == 0.05


def test_source_sha_is_checked_before_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    called = False

    def _load(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(runner, "_sha256", lambda _: "wrong")
    monkeypatch.setattr(runner, "load_checkpoint", _load)
    with pytest.raises(runner.V6InputError, match="SHA mismatch"):
        runner._load_source(source, _config(), _v6())
    assert called is False


def test_source_loader_uses_fresh_adam_and_frozen_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    monkeypatch.setattr(runner, "_sha256", lambda _: runner.SOURCE_CHECKPOINT_SHA256)
    monkeypatch.setattr(
        runner, "load_checkpoint",
        lambda _path, policy, _config, **kwargs: {"model": policy.model.state_dict()},
    )
    policy, _, names, sha = runner._load_source(source, _config(), _v6())
    assert sha == runner.SOURCE_CHECKPOINT_SHA256
    assert names
    assert all(name.startswith(runner.TRAINABLE_PREFIXES) for name in names)
    assert len(policy.optimizer.state) == 0
    assert all(
        parameter.requires_grad == name.startswith(runner.TRAINABLE_PREFIXES)
        for name, parameter in policy.model.named_parameters()
    )


def test_restore_reinstates_saved_rng_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(101)
    np.random.seed(101)
    random.seed(101)
    saved = runner._rng_state()
    torch.set_rng_state(saved["torch"])
    np.random.set_state(saved["numpy"])
    random.setstate(saved["python"])
    expected_torch = torch.rand(3)
    expected_numpy = np.random.rand(3)
    expected_python = random.random()
    monkeypatch.setattr(runner, "load_checkpoint", lambda *args, **kwargs: {"rng_state": saved})
    restored = runner._restore_policy(tmp_path / "checkpoint.pt", _config(), _v6(),
                                      SimpleNamespace())
    del restored
    assert torch.equal(torch.rand(3), expected_torch)
    assert np.array_equal(np.random.rand(3), expected_numpy)
    assert random.random() == expected_python


def test_admission_false_and_pilot_sha_mismatch_refuse(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"PILOT_TECHNICAL_PASS": True, "episodes_recorded": 12,
                                 "selected_checkpoint_sha256": runner.SOURCE_CHECKPOINT_SHA256}))
    pilot_sha = runner._sha256(pilot)
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({
        "FORMAL_P3_ALLOWED": False,
        "initial_checkpoint_sha256": runner.SOURCE_CHECKPOINT_SHA256,
        "pilot_manifest_path": str(pilot),
        "pilot_manifest_sha256": pilot_sha,
    }))
    with pytest.raises(runner.V6InputError, match="authorization"):
        runner._validate_admission(decision, _v6(), "protocol")
    decision.write_text(json.dumps({
        "FORMAL_P3_ALLOWED": True,
        "initial_checkpoint_sha256": runner.SOURCE_CHECKPOINT_SHA256,
        "protocol_sha256": "protocol",
        "pilot_manifest_path": str(pilot),
        "pilot_manifest_sha256": "bad",
    }))
    with pytest.raises(runner.V6InputError, match="pilot manifest SHA"):
        runner._validate_admission(decision, _v6(), "protocol")


def test_check_only_does_not_create_output(tmp_path: Path) -> None:
    output = tmp_path / "check-only"
    context = _context(tmp_path)
    result = runner._check_only_payload({**context, "output": output})
    assert result["status"] == "CHECK_ONLY_OK"
    assert result["output_created"] is False
    assert not output.exists()


def test_smoke_contract_does_not_require_full_resolution(tmp_path: Path) -> None:
    tape = tmp_path / "tape.json"
    tape.write_text(json.dumps({"action_tape": [{}, {}]}))
    step = SimpleNamespace(world=np.zeros((1, 88), dtype=np.float32), stochastic_behavior=True)
    record = {
        "action_tape": str(tape), "teacher_enabled": False, "label_rows": 0,
        "PPO_ALLOWED": False, "weights_unchanged": True,
        "summary": {"safety": {"SafetyInvariantError": 0},
                    "reward_component_totals": {}},
        "reward_telemetry": {"components_finite": True, "total_finite": True,
                              "components_consistent": True},
    }
    checks = runner._smoke_checks(record, [step, step], 2)
    assert checks["exact_step_count"] is True
    assert "full_resolution" not in checks


def test_failed_rollout_stops_before_update_and_preserves_manifest(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path)
    monkeypatch.setattr(runner, "_schedule", lambda *args, **kwargs: ([1], ["dense24"]))
    monkeypatch.setattr(runner, "_dev_evaluate", lambda **kwargs: {
        "hard_failure": False, "full_resolution": True, "records": []
    })
    monkeypatch.setattr(runner.admission, "_record_rollout", lambda **kwargs: _fake_record(failed_energy=1))
    called = False

    def _update(*args, **kwargs):
        nonlocal called
        called = True
        return {}, {"x": True}, {}

    monkeypatch.setattr(runner, "_update", _update)
    result = runner._training_run(context, tmp_path / "failed", run_index=None)
    assert result["status"] == "FAILED"
    assert called is False
    assert (tmp_path / "failed" / "manifest.json").is_file()


def test_restore_failure_keeps_latest_checkpoint_and_stops(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path, formal=True)
    monkeypatch.setattr(runner, "_schedule", lambda *args, **kwargs: ([1, 2], ["dense24", "dense24"]))
    monkeypatch.setattr(runner, "_dev_evaluate", lambda **kwargs: {
        "hard_failure": False, "full_resolution": True, "records": []
    })
    rollout_checkpoints: list[str] = []

    def _rollout(**kwargs):
        rollout_checkpoints.append(str(kwargs["checkpoint"]))
        return _fake_record()

    monkeypatch.setattr(runner.admission, "_record_rollout", _rollout)
    monkeypatch.setattr(runner, "_training_pass", lambda record: True)
    monkeypatch.setattr(runner, "_update", lambda *args, **kwargs: (
        {"accepted_updates": 1}, {"x": True}, {}
    ))
    calls = 0

    def _save_restore(*args, **kwargs):
        nonlocal calls
        calls += 1
        return args[0], calls == 1

    monkeypatch.setattr(runner, "_save_and_restore", _save_restore)
    result = runner._training_run(context, tmp_path / "restore-failure", run_index=0)
    assert result["status"] == "FAILED"
    assert result["accepted_episodes"] == 1
    assert len(result["records"]) == 2
    assert result["records"][0]["checkpoint_restore_exact"] is True
    assert result["records"][1]["checkpoint_restore_exact"] is False
    assert rollout_checkpoints[1].endswith("checkpoint_episode_0001.pt")
