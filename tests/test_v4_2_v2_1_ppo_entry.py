import json

import pytest
import torch

from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42
from terminal_marl.core.checkpoints import save_checkpoint
import terminal_marl.training.run_v4_2_v2_1_ppo as runner


def test_unadmitted_and_changed_code_are_rejected():
    with pytest.raises(PermissionError):
        runner.validate_admission({"PPO_ALLOWED": False}, formal=True)
    p = runner.protocol()
    report = {"PPO_ALLOWED": True, "gates": {"pass": True}, "i3_sha256": p["i3_manifest_sha256"],
        "protocol_sha256": runner.PROTOCOL_SHA, "scope": p["scope"], "source_hashes": {},
        "checkpoint_sha256": runner.i3_protocol()["checkpoint_sha256"]}
    with pytest.raises(ValueError, match="implementation changed"):
        runner.validate_admission(report, formal=True)
    report["source_hashes"] = runner.source_hashes()
    runner.validate_admission(report, formal=True)


def test_full_entry_updates_reloads_and_writes_admission(tmp_path, monkeypatch):
    p = runner.protocol()
    monkeypatch.setattr(runner, "protocol", lambda: {**p, "verification_steps": [2, 2],
        "verification_scenarios": ["warm3", "warm3"]})
    monkeypatch.setattr(runner, "validate_admission", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "qualified_dagger", lambda *args: None)
    monkeypatch.setattr(runner, "historical_registry", lambda *args: {"seeds": set()})
    config = runner.scenario_config("warm3", runner.i3_protocol())
    policy = StudentPPOV42(88)
    with torch.no_grad():
        policy.model.gate_head.weight.zero_()
        policy.model.gate_head.bias.copy_(torch.tensor([20., -20.]))
    checkpoint = tmp_path / "parent.pt"
    save_checkpoint(checkpoint, policy, config)
    dagger = tmp_path / "dagger.json"
    dagger.write_text(json.dumps({"checkpoint": str(checkpoint)}))
    admission = tmp_path / "review.json"
    admission.write_text(json.dumps({"checkpoint_sha256": runner._sha(checkpoint)}))
    result = runner.run(tmp_path / "verify", admission, dagger, tmp_path)
    assert result["PASS"] and result["PPO_ALLOWED"]
    assert len(result["records"]) == 2
    assert all(r["gates"]["checkpoint_reload"] for r in result["records"])
    assert not result["formal_training_called"]
    assert (tmp_path / "verify" / "admission.json").exists()

def test_ratio_check_preserves_rollout_batch_and_detects_real_change():
    import numpy as np
    from types import SimpleNamespace
    from terminal_marl.algo.joint_ppo_v4_2 import hierarchical_joint_logprob
    class BatchSensitiveModel:
        def __init__(self): self.bias=0.; self.sizes=[]
        def distributions(self, x):
            self.sizes.append(len(x))
            logits=torch.zeros((len(x),2));logits[:,1]=len(x)*.01+self.bias
            return torch.distributions.Categorical(logits=logits), torch.distributions.Categorical(logits=torch.zeros((len(x),25)))
    model=BatchSensitiveModel();world=np.zeros((5,88),dtype=np.float32)
    gate=np.array([0,1,0,0,1]);proposal=np.array([-1,12,-1,-1,14])
    gd,pd=model.distributions(torch.as_tensor(world))
    _,_,old=hierarchical_joint_logprob(gd,pd,torch.as_tensor(gate),torch.as_tensor(proposal))
    step=SimpleNamespace(world=world,actor_mask=np.array([False,True,False,False,True]),
        requested_gate=gate,requested_proposal=proposal,requested_logprob=old.numpy())
    policy=SimpleNamespace(model=model,device='cpu')
    assert runner.requested_logprob_error(policy,[step])==0.
    assert model.sizes[-1]==5
    model.bias=.1
    assert runner.requested_logprob_error(policy,[step])>1e-5
