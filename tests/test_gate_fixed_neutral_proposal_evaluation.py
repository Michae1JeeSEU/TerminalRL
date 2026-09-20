import numpy as np
import torch

from terminal_marl.algo.joint_ppo_v4_2 import JointPPOV42
from terminal_marl.training import evaluate_v4_2_p2_gate_fixed_neutral_proposal_v1 as evaluation


def test_fixed_neutral_protocol_is_frozen_without_catalogue_search():
    value = evaluation.protocol()
    assert value["bindings"]["fixed_proposal_index"] == 12
    assert value["bindings"]["runtime_catalogue_search"] is False
    assert value["formal"]["seeds"] == list(evaluation.FORMAL_SEEDS)


def test_fixed_neutral_policy_preserves_gate_and_replaces_release_proposal():
    base = JointPPOV42(88)
    policy = evaluation.FixedNeutralProposalPolicy(base)
    observations = np.zeros((3, 88), dtype=np.float32)
    batch = policy.act_details(observations, observations, deterministic=True)
    assert batch.requested_gate.shape == (3,)
    for gate, proposal in zip(batch.requested_gate, batch.requested_proposal):
        if gate == 1:
            assert proposal == evaluation.FIXED_PROPOSAL
        else:
            assert proposal is None
    assert batch.decoded.shape == (3, 3)
    assert np.isfinite(batch.joint_logprob).all()


def test_fixed_neutral_replacement_lives_at_rollout_model_boundary():
    base = JointPPOV42(88)
    policy = evaluation.FixedNeutralProposalPolicy(base)
    observations = torch.zeros((4, 88))
    original_gate, _ = base.model.distributions(observations)
    replaced_gate, replaced_proposal = policy.model.distributions(observations)
    assert torch.equal(original_gate.logits, replaced_gate.logits)
    assert torch.equal(replaced_proposal.logits.argmax(dim=-1), torch.full((4,), 12))
    assert torch.allclose(replaced_proposal.probs[:, 12], torch.ones(4))


def test_fixed_neutral_v2_successor_uses_fresh_seeds():
    names = ("STAGE", "PROTOCOL", "PROTOCOL_SHA", "FORMAL_SEEDS", "SMOKE_SEED",
             "RESERVATION_MARKER")
    original = {name: getattr(evaluation, name) for name in names}
    try:
        evaluation.configure_successor_v2()
        value = evaluation.protocol()
        assert value["predecessor"]["v1_status"] == "INVALID_IMPLEMENTATION"
        assert value["bindings"]["implementation_boundary"] == "policy_model_distributions"
        assert evaluation.FORMAL_SEEDS == (804001, 804002, 804003, 804004, 804005, 804006)
    finally:
        for name, value in original.items():
            setattr(evaluation, name, value)
