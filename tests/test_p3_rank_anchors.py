from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from terminal_marl.training import collect_p3_rank_anchors as anchors


class _FakeEnv:
    def __init__(self, config):
        self.config = config
        self.step_count = 0
        self.active = {2: object(), 7: object()}

    def _observations(self):
        return {"2": np.full(88, 2.0, dtype=np.float32), "7": np.full(88, 7.0, dtype=np.float32)}

    def step(self, *args, **kwargs):
        self.step_count += 1
        return {}, {}, {}, {}, {}


def test_collects_finite_88d_rows_once_per_train_tape(tmp_path, monkeypatch):
    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"source")
    monkeypatch.setattr(anchors.collector, "SOURCE_CHECKPOINT", source_checkpoint)
    monkeypatch.setattr(anchors.collector, "SOURCE_CHECKPOINT_SHA256", anchors._sha256(source_checkpoint))
    records = [
        {"seed": seed, "scenario": scenario, "split": "train", "action_tape": str(tmp_path / f"{seed}.json")}
        for seed, scenario in zip((873101, 873102, 873103), ("flow6", "flow12", "dense24"))
    ]
    for row in records:
        Path(row["action_tape"]).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        anchors.collector,
        "_load_source_inputs",
        lambda formal_root: ({}, {scenario: object() for scenario in ("flow6", "flow12", "dense24")}, records, tmp_path / "manifest.json"),
    )
    monkeypatch.setattr(anchors.collector, "_summary_outcomes", lambda tape: {"done": "completed"})
    monkeypatch.setattr(anchors.collector, "_replay_outcomes", lambda env: {"done": "completed"})
    replay_calls = []

    def fake_replay(env, tape):
        replay_calls.append((env, Path(tape)))
        env.step({"2": np.zeros(1, dtype=np.float32)})
        return {"exact": True, "steps_replayed": 1, "steps_expected": 1}

    monkeypatch.setattr(anchors.SafetyTraceRecorder, "replay_action_tape", staticmethod(fake_replay))
    monkeypatch.setattr(anchors, "TerminalParallelEnv", _FakeEnv)

    manifest = anchors.collect(output_dir=tmp_path / "out", formal_root=tmp_path / "formal")

    assert len(replay_calls) == 3
    assert manifest["status"] == "COMPLETED"
    assert manifest["exact"] is True
    assert manifest["seeds"] == [873101, 873102, 873103]
    assert manifest["records_count"] == 3
    assert manifest["rows_count"] == 6
    assert manifest["rows_by_seed"] == {"873101": 2, "873102": 2, "873103": 2}
    with np.load(tmp_path / "out" / "anchors.npz") as loaded:
        assert loaded["observations"].shape == (6, 88)
        assert np.isfinite(loaded["observations"]).all()
        assert loaded["seed"].shape == (6,)
        assert loaded["step"].tolist() == [0, 0, 0, 0, 0, 0]
        assert loaded["uid"].tolist() == [2, 7, 2, 7, 2, 7]
    serialized = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert serialized["source_checkpoint_sha256"] == anchors._sha256(source_checkpoint)
    assert json.loads((tmp_path / "out" / "progress.json").read_text(encoding="utf-8"))["status"] == "COMPLETED"
