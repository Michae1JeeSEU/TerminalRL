import torch
import pytest
from terminal_marl.algo.conditioned_set_actor import ConditionedSetActor


def test_conditioned_actor_is_set_invariant_and_ignores_padding_in_forward_and_gradients():
    torch.manual_seed(7);model=ConditionedSetActor()
    ego=torch.randn(2,159);traffic=torch.randn(2,4,50)
    mask=torch.tensor([[True,True,False,False],[False,False,False,False]])
    traffic[~mask]=float('nan')
    expected=model(ego,traffic,mask)
    actual=model(ego,traffic[:,[3,1,0,2]],mask[:,[3,1,0,2]])
    for a,b in zip(expected,actual):torch.testing.assert_close(a,b)
    assert expected[0].shape==(2,2) and expected[1].shape==(2,25)
    sum(x.square().sum() for x in actual).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    empty=model(ego,torch.zeros(2,0,50),torch.zeros(2,0,dtype=torch.bool))
    for a,b in zip(expected,empty):torch.testing.assert_close(a[1],b[1])
    with pytest.raises(ValueError):model(ego[:,:105],traffic,mask)


def test_ego_state_changes_pair_encoding_before_pooling():
    model=ConditionedSetActor()
    # A fixed neighbor has different learned pair features for two ego states.
    # This dependency is absent from the historical neighbor-only encoder.
    ego=torch.zeros(2,159);ego[1,0]=1
    neighbor=torch.zeros(2,50)
    with torch.no_grad():
        first=model.gate_encoder.neighbor[0]
        first.weight.zero_();first.bias.zero_();first.weight[0,0]=1
        second=model.gate_encoder.neighbor[2]
        second.weight.zero_();second.bias.zero_();second.weight[0,0]=1
        encoded=model.gate_encoder.neighbor(torch.cat((ego,neighbor),-1))
    assert encoded[0,0]==0 and encoded[1,0]>0
