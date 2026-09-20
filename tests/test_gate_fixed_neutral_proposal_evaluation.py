import numpy as np

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
