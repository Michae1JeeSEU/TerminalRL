from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from terminal_marl.visualization import p3_formal_replay as replay


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    run_dir = tmp_path / "run0"
    checkpoint = run_dir / "checkpoints" / "checkpoint_episode_0007.pt"
    tape = run_dir / "episodes" / "episode_0008_dense24_99" / "action_tapes" / "tape.json"
    checkpoint.parent.mkdir(parents=True)
    tape.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint-bytes")
    tape.write_text("{}", encoding="utf-8")
    sha = _sha256(checkpoint)
    server_checkpoint = "/home/user/training_outputs/terminal/example/run0/checkpoints/checkpoint_episode_0007.pt"
    server_tape = "/home/user/training_outputs/terminal/example/run0/episodes/episode_0008_dense24_99/action_tapes/tape.json"
    manifest = {
        "stage": "P3_FORMAL_CONTROLLED_EMERGENCY_V5",
        "run_index": 0,
        "final_checkpoint": server_checkpoint,
        "actual_config_snapshots": {},
        "records": [{
            "checkpoint": server_checkpoint,
            "checkpoint_input_sha256": sha,
            "checkpoint_sha256": sha,
            "action_tape": server_tape,
            "scenario": "dense24",
            "seed": 99,
            "requests": 24,
            "normal": 0,
            "emergency": 0,
            "failed_energy": 1,
            "unresolved": 23,
            "accepted_episode": False,
            "hard_gate_pass": False,
            "full_resolution": False,
            "outcomes": {"0": "failed_energy"},
        }],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    return run_dir, checkpoint, sha


def test_server_paths_map_from_manifest_and_checkpoint_identity_is_checked(
    tmp_path: Path,
) -> None:
    run_dir, checkpoint, sha = _fixture(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["records"][-1]

    resolved_tape = replay._resolve_artifact(record["action_tape"], run_dir, "episodes")
    assert resolved_tape.is_file()
    resolved_checkpoint, actual, fields = replay._bind_checkpoint(manifest, record, run_dir)
    assert resolved_checkpoint == checkpoint
    assert actual == sha
    assert fields["checkpoint_input_sha256"] == sha


def test_checkpoint_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    run_dir, _, _ = _fixture(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][-1]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(replay.ReplayInputError, match="checkpoint_sha256"):
        replay._bind_checkpoint(manifest, manifest["records"][-1], run_dir)


def test_nonexact_replay_writes_report_without_exact_html(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    run_dir, _, _ = _fixture(tmp_path)

    class DummyEnv:
        def set_step_observer(self, observer) -> None:
            self.observer = observer

    monkeypatch.setattr(replay, "_scenario_config", lambda manifest, scenario: SimpleNamespace())
    monkeypatch.setattr(replay, "TerminalParallelEnv", lambda config: DummyEnv())
    monkeypatch.setattr(
        replay.SafetyTraceRecorder,
        "replay_action_tape",
        lambda env, path: {"exact": False, "steps_replayed": 1, "steps_expected": 1},
    )
    output = tmp_path / "replay-output"
    result = replay.replay_run(run_dir, output)

    assert result["exact"] is False
    assert (output / "replay_report.json").is_file()
    assert not (output / "recorded_stochastic_policy_replay.html").exists()


def test_cli_nonexact_replay_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    run_dir, _, _ = _fixture(tmp_path)

    class DummyEnv:
        def set_step_observer(self, observer) -> None:
            self.observer = observer

    monkeypatch.setattr(replay, "_scenario_config", lambda manifest, scenario: SimpleNamespace())
    monkeypatch.setattr(replay, "TerminalParallelEnv", lambda config: DummyEnv())
    monkeypatch.setattr(
        replay.SafetyTraceRecorder,
        "replay_action_tape",
        lambda env, path: {"exact": False, "steps_replayed": 1, "steps_expected": 1},
    )
    output = tmp_path / "cli-replay-output"
    monkeypatch.setattr(
        sys, "argv", ["p3_formal_replay", "--run-dir", str(run_dir), "--output-dir", str(output)],
    )
    with pytest.raises(SystemExit) as exc_info:
        raise SystemExit(replay.main())
    assert exc_info.value.code == 2
