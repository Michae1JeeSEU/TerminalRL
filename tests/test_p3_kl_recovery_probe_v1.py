from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from terminal_marl.training import run_p3_kl_recovery_probe_v1 as probe


def _context(run_index: int = 0) -> dict:
    policy = SimpleNamespace(optimizer=SimpleNamespace(state={}))
    config = SimpleNamespace(max_simulation_seconds=2.0, dt=1.0)
    return {
        "run_index": run_index,
        "protocol": {"optimizer": {}},
        "protocol_sha256": "protocol-sha",
        "configs": {"flow6": config, "flow12": config, "dense24": config},
        "config_snapshots": {},
        "runtime_protocol": {},
        "source_checkpoint": Path("source.pt"),
        "source_checkpoint_sha256": probe.SOURCE_CHECKPOINT_SHA256,
        "source_payload": {},
        "trainable": ("gate_adapter.weight",),
        "old_rows": [],
        "coverage_rows": [],
        "weight_report": {},
        "reference": object(),
        "reference_digest": "reference-sha",
        "base_checkpoint": Path("base.pt"),
        "base_checkpoint_sha256": "base-sha",
        "policy": policy,
    }


def _patch_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeV6:
        @staticmethod
        def _dev_evaluate(**kwargs):
            return {
                "complete": True,
                "full_resolution": True,
                "hard_failure": False,
                "records": [],
            }

        @staticmethod
        def _restore_rng_state(_state):
            return None

        @staticmethod
        def _rng_state():
            return {"fake": True}

    monkeypatch.setattr(probe.v6, "_dev_evaluate", FakeV6._dev_evaluate)
    monkeypatch.setattr(probe.v6, "_restore_rng_state", FakeV6._restore_rng_state)
    monkeypatch.setattr(probe.v6, "_rng_state", FakeV6._rng_state)
    monkeypatch.setattr(
        probe.v6.admission,
        "_record_rollout",
        lambda **kwargs: {
            "scenario": kwargs["scenario"],
            "seed": kwargs["seed"],
            "action_tape": "tape.json",
            "_steps": [],
        },
    )
    monkeypatch.setattr(probe.v6, "_training_pass", lambda _record: True)
    monkeypatch.setattr(probe.v6.formal_v5, "_public_record", lambda record: dict(record))
    monkeypatch.setattr(probe.v6, "_jsonable", lambda value: value)
    monkeypatch.setattr(probe.v6, "_comparison", lambda before, after: {"report_only": True})

    def fake_checkpoint(context, output, episode_index, timestep, config, *, ppo_called):
        path = output / "checkpoints" / f"checkpoint_episode_{episode_index:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"checkpoint-{episode_index}".encode())
        return context["policy"], True, path

    monkeypatch.setattr(probe, "_checkpoint", fake_checkpoint)


def test_check_only_does_not_create_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    context = _context()
    monkeypatch.setattr(probe, "_protocol_context", lambda _run_index: context)
    output = tmp_path / "check_only"

    result = probe.run(run_index=0, output=output, check_only=True)

    assert result["status"] == "CHECK_ONLY_OK"
    assert result["output_created"] is False
    assert not output.exists()
    assert result["base_checkpoint_sha256"] == "base-sha"


def test_missing_source_checkpoint_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing-source.pt"
    monkeypatch.setattr(probe, "SOURCE_CHECKPOINT", missing)

    with pytest.raises((probe.ProbeInputError, ValueError), match="source checkpoint"):
        probe._protocol_context(0)


def test_kl_skip_is_checkpointed_but_not_counted_as_updated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _patch_rollout(monkeypatch)
    context = _context()
    context["base_checkpoint"] = tmp_path / "base.pt"
    context["base_checkpoint"].write_bytes(b"base")
    monkeypatch.setattr(probe, "_call_recover_update", lambda _context, _record: {"status": "SKIPPED_KL"})

    result = probe._run(context, tmp_path / "run0")

    assert result["status"] == "COMPLETED"
    assert result["processed_episodes"] == 4
    assert result["updated_episodes"] == 0
    assert result["skipped_kl_episodes"] == 4
    assert result["hard_failed_episodes"] == 0
    assert result["accepted_episodes"] == 0
    assert all(row["ppo_accepted"] is False for row in result["records"])
    assert len(result["checkpoints"]) == 4
    saved = json.loads((tmp_path / "run0" / "manifest.json").read_text(encoding="utf-8"))
    assert saved["skipped_kl_episodes"] == 4


def test_hard_failure_stops_after_first_processed_episode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _patch_rollout(monkeypatch)
    context = _context()
    context["base_checkpoint"] = tmp_path / "base.pt"
    context["base_checkpoint"].write_bytes(b"base")
    monkeypatch.setattr(probe.v6, "_training_pass", lambda _record: False)

    result = probe._run(context, tmp_path / "run0")

    assert result["status"] == "STOPPED"
    assert result["processed_episodes"] == 1
    assert result["hard_failed_episodes"] == 1
    assert result["updated_episodes"] == 0
    assert result["skipped_kl_episodes"] == 0
    assert result["stop_reason"] == "rollout hard failure"
    assert result["records"][0]["recovery"]["status"] == "HARD_FAILED"
    assert result["development_after"] is None
    assert result["checkpoints"] == []
    assert not (tmp_path / "run0" / "dev_after").exists()


def test_recovery_hard_failure_stops_without_checkpoint_or_after_dev(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _patch_rollout(monkeypatch)
    context = _context()
    context["base_checkpoint"] = tmp_path / "base.pt"
    context["base_checkpoint"].write_bytes(b"base")
    monkeypatch.setattr(
        probe, "_call_recover_update", lambda _context, _record: {"status": "HARD_FAILED"},
    )

    result = probe._run(context, tmp_path / "run0")

    assert result["status"] == "STOPPED"
    assert result["processed_episodes"] == 1
    assert result["hard_failed_episodes"] == 1
    assert result["updated_episodes"] == 0
    assert result["skipped_kl_episodes"] == 0
    assert result["development_after"] is None
    assert result["checkpoints"] == []
