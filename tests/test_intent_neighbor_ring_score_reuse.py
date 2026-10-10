import pytest
import torch

from terminal_marl.algo.intent_neighbor_actor import IntentNeighborActor
from terminal_marl.algo.ordered_proposal_actor import first_feasible_logits


def _inputs(dtype, kind, neighbor_count):
    generator = torch.Generator().manual_seed(20261010 + neighbor_count)
    ego = torch.randn(4, 159, generator=generator, dtype=dtype) * 0.1
    ego[:, 8] = 1.
    ego[:, 105:] = 0.
    if kind == 'ring':
        ego[:, 10] = 0.
    elif kind == 'final':
        ego[:, 10] = 1.
        for row in range(len(ego)):
            ego[row, 105 + 3] = 1.
            ego[row, 105 + 6] = 1.
            ego[row, 105 + 9 + 3] = 1.
            ego[row, 105 + 9 + 6] = 1.
    elif kind == 'mixed':
        ego[:, 10] = torch.tensor([0., 1., 0., 1.], dtype=dtype)
        for row in (1, 3):
            ego[row, 105 + 3] = 1.
            ego[row, 105 + 6] = 1.
            ego[row, 105 + 18 + 3] = 1.
            ego[row, 105 + 18 + 6] = 1.
    elif kind == 'final_ineligible':
        ego[:, 10] = 1.
    else:
        raise AssertionError(kind)

    traffic = torch.randn(4, neighbor_count, 50, generator=generator, dtype=dtype)
    mask = torch.ones(4, neighbor_count, dtype=torch.bool)
    if neighbor_count:
        mask[0, -1] = False
        if neighbor_count > 1:
            mask[1, 1] = False
        traffic[~mask] = torch.nan
    return ego, traffic, mask


def _reference_relations(model, ego, traffic, mask):
    pair, scene, pads, eligible = model.bases(ego, traffic, mask)
    query = model.intent(model.proposals)[None, :, None, :] + model.destination(pads)[:, None, :, :]
    output = model.pair_head(pair[:, None, None, :, :] + query[:, :, :, None, :]).squeeze(-1)
    risk = output.masked_fill(~mask[:, None, None, :], -torch.inf)
    if traffic.shape[1]:
        maximum, blocker = risk.max(-1)
        blocker = torch.where(
            mask.any(-1)[:, None, None], blocker, torch.full_like(blocker, -1)
        )
    else:
        maximum = torch.full(query.shape[:3], -torch.inf, device=ego.device, dtype=ego.dtype)
        blocker = torch.full(query.shape[:3], -1, device=ego.device, dtype=torch.long)
    scene_risk = model.scene_head(scene[:, None, None, :] + query).squeeze(-1)
    destination_risk = torch.maximum(maximum, scene_risk)
    destination_risk = destination_risk.masked_fill(~eligible[:, None, :], 30.)
    best_risk, best_pad = destination_risk.min(-1)
    return dict(
        candidate_logits=-best_risk,
        pair_logits=risk,
        scene_logits=scene_risk,
        blocker_index=blocker,
        best_pad=best_pad,
        eligible_destinations=eligible,
    )


def _gate_from_candidates(candidate_logits, gate_mode):
    if gate_mode == 'first_feasible_union':
        return first_feasible_logits(candidate_logits)
    _, proposal = first_feasible_logits(candidate_logits)
    best = candidate_logits.max(-1).values
    return torch.stack((torch.zeros_like(best), best), -1), proposal


def _assert_outputs(actual, expected, dtype):
    floating = ('candidate_logits', 'pair_logits', 'scene_logits')
    rtol, atol = ((1e-10, 1e-10) if dtype == torch.float64 else (None, None))
    for key in actual:
        if key in floating:
            if rtol is None:
                torch.testing.assert_close(actual[key], expected[key])
            else:
                torch.testing.assert_close(actual[key], expected[key], rtol=rtol, atol=atol)
        else:
            assert torch.equal(actual[key], expected[key]), key


def _loss(relations, gate, proposal):
    pair = torch.nan_to_num(relations['pair_logits'], nan=0., posinf=0., neginf=0.)
    scene = torch.nan_to_num(relations['scene_logits'], nan=0., posinf=0., neginf=0.)
    pair_term = pair.square().sum() / max(pair.numel(), 1)
    return pair_term + scene.square().mean() + gate.square().mean() + proposal.square().mean()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('gate_mode', ['first_feasible_union', 'max_candidate'])
@pytest.mark.parametrize('kind', ['ring', 'final', 'mixed', 'final_ineligible'])
@pytest.mark.parametrize('neighbor_count', [0, 3, 20])
def test_ring_reuse_matches_full_six_pad_reference_and_gradients(
    dtype, gate_mode, kind, neighbor_count
):
    torch.manual_seed(19)
    optimized = IntentNeighborActor(gate_mode=gate_mode).to(dtype=dtype)
    reference = IntentNeighborActor(gate_mode=gate_mode).to(dtype=dtype)
    reference.load_state_dict(optimized.state_dict())
    for name, value in optimized.state_dict().items():
        assert torch.equal(value, reference.state_dict()[name]), name

    ego, traffic, mask = _inputs(dtype, kind, neighbor_count)
    ego_opt = ego.clone().requires_grad_()
    traffic_opt = traffic.clone().requires_grad_()
    ego_ref = ego.clone().requires_grad_()
    traffic_ref = traffic.clone().requires_grad_()

    actual = optimized.relations(ego_opt, traffic_opt, mask)
    expected = _reference_relations(reference, ego_ref, traffic_ref, mask)
    _assert_outputs(actual, expected, dtype)

    actual_gate, actual_proposal = _gate_from_candidates(actual['candidate_logits'], gate_mode)
    api_gate, api_proposal = optimized(ego_opt, traffic_opt, mask)
    expected_gate, expected_proposal = _gate_from_candidates(expected['candidate_logits'], gate_mode)
    if dtype == torch.float64:
        torch.testing.assert_close(api_gate, actual_gate, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(api_proposal, actual_proposal, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(actual_gate, expected_gate, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(actual_proposal, expected_proposal, rtol=1e-10, atol=1e-10)
    else:
        torch.testing.assert_close(api_gate, actual_gate)
        torch.testing.assert_close(api_proposal, actual_proposal)
        torch.testing.assert_close(actual_gate, expected_gate)
        torch.testing.assert_close(actual_proposal, expected_proposal)

    actual_loss = _loss(actual, actual_gate, actual_proposal)
    expected_loss = _loss(expected, *(_gate_from_candidates(expected['candidate_logits'], gate_mode)))
    assert torch.isfinite(actual_loss)
    assert torch.isfinite(expected_loss)
    actual_loss.backward()
    expected_loss.backward()

    for name, parameter in optimized.named_parameters():
        reference_parameter = dict(reference.named_parameters())[name]
        assert (parameter.grad is None) == (reference_parameter.grad is None), name
        if parameter.grad is not None:
            if dtype == torch.float64:
                torch.testing.assert_close(parameter.grad, reference_parameter.grad,
                                            rtol=1e-10, atol=1e-10)
            else:
                torch.testing.assert_close(parameter.grad, reference_parameter.grad)
    if dtype == torch.float64:
        torch.testing.assert_close(ego_opt.grad, ego_ref.grad, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(traffic_opt.grad, traffic_ref.grad, rtol=1e-10, atol=1e-10)
    else:
        torch.testing.assert_close(ego_opt.grad, ego_ref.grad)
        torch.testing.assert_close(traffic_opt.grad, traffic_ref.grad)


@pytest.mark.parametrize('kind,expected_pad_counts', [
    ('ring', [1]),
    ('final', [6]),
    ('mixed', [1, 6]),
])
def test_ring_reuse_head_shapes_keep_full_api_and_final_six_pads(kind, expected_pad_counts):
    model = IntentNeighborActor()
    ego, traffic, mask = _inputs(torch.float32, kind, 3)
    pair_shapes = []
    scene_shapes = []
    pair_hook = model.pair_head.register_forward_hook(
        lambda _module, inputs, _output: pair_shapes.append(tuple(inputs[0].shape))
    )
    scene_hook = model.scene_head.register_forward_hook(
        lambda _module, inputs, _output: scene_shapes.append(tuple(inputs[0].shape))
    )
    try:
        result = model.relations(ego, traffic, mask)
    finally:
        pair_hook.remove()
        scene_hook.remove()
    assert result['pair_logits'].shape == (4, 25, 6, 3)
    assert result['scene_logits'].shape == (4, 25, 6)
    assert sorted(shape[2] for shape in pair_shapes) == expected_pad_counts
    assert sorted(shape[2] for shape in scene_shapes) == expected_pad_counts
    assert all(shape[-2:] == (3, 32) for shape in pair_shapes)
    assert all(shape[-1] == 32 for shape in scene_shapes)


@pytest.mark.parametrize('gate_mode', ['first_feasible_union', 'max_candidate'])
def test_queried_scores_remains_the_original_single_query_score(gate_mode):
    model = IntentNeighborActor(gate_mode=gate_mode)
    ego, traffic, mask = _inputs(torch.float32, 'mixed', 20)
    proposal = torch.tensor([0, 12, 24, 3])
    pad = torch.tensor([0, 2, 0, 5])
    pair, scene = model.queried_scores(ego, traffic, mask, proposal, pad)
    full = _reference_relations(model, ego, traffic, mask)
    index = torch.arange(len(ego))
    expected_pair = full['pair_logits'][index, proposal, pad]
    valid = mask
    torch.testing.assert_close(pair[valid], expected_pair[valid])
    torch.testing.assert_close(scene, full['scene_logits'][index, proposal, pad])
