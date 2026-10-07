import torch
import pytest

from terminal_marl.algo.trajectory_set_actor import TrajectorySetActor
from terminal_marl.algo.joint_ppo_v4_2 import hierarchical_bc_loss
from terminal_marl.training import p3_v25_set_learnability as diagnostic


def inputs():
    torch.manual_seed(19)
    return torch.randn(3, 105), torch.randn(3, 5, 50), torch.tensor(
        [[True, True, True, False, False], [False]*5, [True]*5])


def test_set_permutation_padding_and_batch_invariance():
    model = TrajectorySetActor()
    ego, traffic, mask = inputs()
    expected = model(ego, traffic, mask)
    order = torch.tensor([3, 0, 4, 1, 2])
    for variant in (model(ego, traffic[:, order], mask[:, order]),
                    model(ego, traffic.masked_fill(~mask[..., None], float('nan')), mask),
                    model(ego, torch.cat((traffic, torch.full((3, 7, 50), float('nan'))), 1),
                          torch.cat((mask, torch.zeros(3, 7, dtype=torch.bool)), 1))):
        for a, b in zip(expected, variant):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    for j in (0, 1):
        singles = torch.cat([model(ego[i:i+1], traffic[i:i+1], mask[i:i+1])[j] for i in range(3)])
        torch.testing.assert_close(singles, expected[j], rtol=1e-5, atol=1e-6)


def test_empty_set_equals_all_masked_and_masked_gradient_zero():
    model = TrajectorySetActor()
    ego, traffic, mask = inputs()
    mask.zero_()
    a = model(ego, traffic, mask)
    b = model(ego, traffic[:, :0], mask[:, :0])
    for x, y in zip(a, b): torch.testing.assert_close(x, y)
    traffic.requires_grad_()
    gate, proposal = model(ego, traffic, mask)
    (gate.sum()+proposal.sum()).backward()
    assert torch.isfinite(traffic.grad).all() and not traffic.grad.any()


def test_gate_and_proposal_paths_are_independent_and_wait_has_no_proposal_loss():
    model = TrajectorySetActor()
    x = inputs()
    g, p = model(*x)
    loss, _ = hierarchical_bc_loss(g, p, torch.zeros(3, dtype=torch.long), torch.full((3,), -1))
    loss.backward()
    assert any(p.grad is not None for p in model.gate_encoder.parameters())
    assert all(p.grad is None for p in model.proposal_encoder.parameters())
    model.zero_grad(set_to_none=True)
    _, p = model(*x)
    torch.nn.functional.cross_entropy(p, torch.tensor([0, 12, 24])).backward()
    assert all(p.grad is None for p in model.gate_encoder.parameters())
    assert any(p.grad is not None and p.grad.any() for p in model.proposal_encoder.parameters())


@pytest.mark.parametrize('bad', ['ego', 'mask', 'traffic'])
def test_shape_contract(bad):
    ego, traffic, mask = inputs()
    if bad == 'ego': ego = ego[:, :90]
    if bad == 'mask': mask = mask.float()
    if bad == 'traffic': traffic = traffic[:, :, :49]
    with pytest.raises(ValueError): TrajectorySetActor()(ego, traffic, mask)


def test_old_arm_is_exact_original_information_and_forward_updates_nothing():
    paired = diagnostic.synthetic()
    old = paired.tensors('old_information')
    assert torch.equal(old[0][:, :90], paired.original.x)
    assert not old[0][:, 90:].any() and not old[1].any() and not old[2].any()
    assert paired.new[1].any() and paired.new[2].any()
    assert diagnostic.forward_check(paired)['parameter_updates'] == 0


def test_synthetic_output_and_identical_fit_pair(tmp_path):
    result = diagnostic.run(tmp_path/'unused_root', tmp_path/'smoke', 'smoke')
    assert result['status'] == 'FIT_COMPLETE' and result['parameter_updates'] == 4
    a, b = result['fits']
    assert a['initial_state_hash'] == b['initial_state_hash']
    assert a['batch_order_hash'] == b['batch_order_hash']
    assert a['updates'] == b['updates'] == 2
    assert all(a['changed'].values()) and all(b['changed'].values())
    for arm in diagnostic.ARMS:
        path = tmp_path/'smoke'/f'17_{arm}'
        checkpoint = torch.load(path/'diagnostic_final.pt', weights_only=True)
        assert checkpoint['diagnostic_only'] is True
        assert len(__import__('json').loads((path/'predictions.json').read_text())) == 16
    with pytest.raises(FileExistsError): diagnostic.run(tmp_path, tmp_path/'smoke', 'smoke')
