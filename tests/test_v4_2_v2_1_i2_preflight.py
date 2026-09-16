from dataclasses import replace

import pytest
import torch

from terminal_marl.algo.joint_ppo_v4_2 import JointPPOV42
import terminal_marl.training.run_v4_2_v2_1_i2_preflight as runner


def test_protocol_is_frozen():
    assert runner._sha(runner.PROTOCOL) == runner.PROTOCOL_SHA


def test_wrong_i1_cannot_authorize_rollout(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "validate_r4", lambda _: {})
    path = tmp_path / "i1.json"
    path.write_text('{"I1_V2":"PASS"}')
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.qualified_student(path, path)


def test_smoke_full_output_and_replay_cannot_admit_round1(tmp_path, monkeypatch):
    policy = JointPPOV42(88)
    with torch.no_grad():
        policy.model.gate_head.weight.zero_()
        policy.model.gate_head.bias.copy_(torch.tensor([20., -20.]))
    config = replace(runner._config(), observation_schema_version=runner.OBS_SCHEMA)
    monkeypatch.setattr(runner, "qualified_student", lambda *_: (
        policy, config, {"canonical_checkpoint_sha256": "synthetic"}))
    result = runner.run(tmp_path / "smoke", tmp_path, tmp_path, tmp_path, smoke=True)
    assert result["I2_PREFLIGHT"] == "PASS"
    assert all(result["gates"].values()) and result["labels"] == 1
    assert not result["I2_ROUND1_ALLOWED"] and not result["PPO_ALLOWED"]
