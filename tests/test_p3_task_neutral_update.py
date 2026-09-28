import json
from pathlib import Path

import numpy as np
import torch

from terminal_marl.algo import student_ppo_v4_2 as ppo
from terminal_marl.training import diagnose_p3_task_neutral_update as diagnostic


class _FakeModel:
    def distributions(self, observations):
        logits = torch.zeros((len(observations), 2), dtype=torch.float32)
        logits[:, 1] = 0.4
        return torch.distributions.Categorical(logits=logits), None


class _FakePolicy:
    device = torch.device("cpu")
    model = _FakeModel()


def _capture(step, *, reward, value, next_value, terminated, episode_end, uid="5", arrival_id=20):
    return {
        "step": step,
        "uids": [uid],
        "world": np.full((1, 88), float(step), dtype=np.float32),
        "requested_gate": np.asarray([1], dtype=np.int64),
        "requested_proposal": np.asarray([2], dtype=np.int64),
        "gate_logprob": np.asarray([0.0], dtype=np.float32),
        "proposal_logprob": np.asarray([0.0], dtype=np.float32),
        "actor_mask": np.asarray([True]),
        "release_probability_by_uid": {uid: 0.25},
        "phase_by_uid": {uid: "ring_inner"},
        "arrival_by_uid": {uid: arrival_id},
        "value": value,
        "post_value": next_value,
        "next_value": next_value,
        "reward": reward,
        "terminated": terminated,
        "episode_end": episode_end,
    }


def test_analysis_uses_production_gae_and_preserves_identity():
    captures = [
        _capture(0, reward=1.0, value=0.2, next_value=0.5, terminated=False, episode_end=False),
        _capture(1, reward=2.0, value=0.3, next_value=0.0, terminated=True, episode_end=True),
    ]
    analysis, rows, steps = diagnostic._analyze_captures(captures, _FakePolicy())
    raw, _ = ppo.team_gae(steps, gamma=diagnostic.GAMMA, gae_lambda=diagnostic.GAE_LAMBDA)
    assert analysis["steps"] == 2
    assert [row["arrival_id"] for row in rows] == [20, 20]
    assert [row["step"] for row in rows] == [0, 1]
    assert np.allclose([row["rawadv"] for row in rows], raw)
    assert rows[0]["p_before"] == 0.25
    assert rows[0]["p_after"] == rows[1]["p_after"]


def test_check_only_does_not_capture_or_reset(tmp_path, monkeypatch):
    root = tmp_path / "diagnostic"
    monkeypatch.setattr(diagnostic, "_preflight", lambda: {
        "record_path": Path("record.json"), "tape_path": Path("tape.json"),
        "before_path": Path("before.pt"), "after_path": Path("after.pt"),
        "expected_steps": 2, "before_sha256": "before", "after_sha256": "after",
        "tape_rows": [], "config": object(),
    })
    monkeypatch.setattr(diagnostic.credit, "_capture_replay", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("replay called")))
    result = diagnostic.run(root, check_only=True)
    assert result["status"] == "CHECK_ONLY_OK"
    assert (root / "report.json").is_file()
    assert not (root / "replay_steps.pt").exists()


def test_replay_serialization_retains_nan_and_json_sanitizes(tmp_path):
    capture = _capture(0, reward=0.0, value=0.0, next_value=0.0, terminated=True, episode_end=True)
    capture["requested_proposal"] = np.asarray([-1], dtype=np.int64)
    capture["proposal_logprob"] = np.asarray([np.nan], dtype=np.float32)
    target = tmp_path / "replay_steps.pt"
    diagnostic._save_replay_steps(target, [capture], tape_path=Path("tape.json"))
    payload = torch.load(target, map_location="cpu", weights_only=False)
    assert torch.isnan(payload["steps"][0]["proposal_logprob"])[0]
    sanitized = diagnostic._jsonable({"proposal_logprob": np.float32(np.nan)})
    json.dumps(sanitized, allow_nan=False)
    assert sanitized["proposal_logprob"] is None
