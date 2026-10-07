"""Synthetic-only checks for 90D bounded BC; never reset an environment."""
import pytest
import torch

from terminal_marl.training import p3_v3_bc_learnability as bc
from terminal_marl.training.p3_v3_expert_dataset import inspect_sparse_episode
from tests.test_p3_v3_expert_dataset import fixture_episode, read, write, raw_rows, save_rows


def complete_fixture(tmp_path):
    ep = fixture_episode(tmp_path, joints=(13, 0))
    protocol = read(ep.parent/'protocol.json'); protocol['protocol'] = 'synthetic'
    write(ep.parent/'protocol.json', protocol)
    rows = raw_rows(ep)
    for i, row in enumerate(rows):
        row.update(interface=('o2m','m2i')[i], soc=.6, speed=20., busy_lz=1, retry_active=False)
    save_rows(ep, rows)
    annotation = read(ep/'annotation_result.json')
    result = dict(seed=101, status='COLLECTION_OK', error=None, blockers=[], annotation=annotation,
        rollout=dict(terminal=True,censored=False,error=None,selection_overflow=0,unresolved=0,
            active=0,queue=0,service=0,requested=2,outcome_counts={'completed':2},
            telemetry={k:0 for k in ['hard_separation_violations','nmac','collisions','checked_executed_plan_hash_mismatches']}),
        dataset_check=inspect_sparse_episode(ep, expected_seed=101, expected_partition='train'))
    write(ep.parent/'result.json', result)
    return ep.parent


def test_data_preparation_uses_only_teacher_and_constructs_no_model(tmp_path, monkeypatch):
    complete_fixture(tmp_path)
    def forbidden(*a, **k):
        raise AssertionError('preparation must not construct model or optimizer')
    monkeypatch.setattr(bc, 'HierarchicalActorCentralCriticV42', forbidden)
    monkeypatch.setattr(torch.optim, 'Adam', forbidden)
    data = bc.load_inputs(tmp_path, cases=((101,'case',2),))
    assert data.x.shape == (2,90)
    assert data.gate.tolist() == [1,0]  # Both saved behavior actions were WAIT.
    assert data.proposal.tolist() == [12,-1]


@pytest.mark.parametrize('change', ['status','overflow','unresolved','annotation','safety'])
def test_partial_or_invalid_episode_is_not_accepted(tmp_path, change):
    case = complete_fixture(tmp_path); result = read(case/'result.json')
    if change == 'status': result['status'] = 'INCOMPLETE'
    if change == 'overflow': result['rollout']['selection_overflow'] = 1
    if change == 'unresolved': result['rollout']['unresolved'] = 1
    if change == 'annotation': result['annotation']['entries'][0]['status'] = 'TIMEOUT'
    if change == 'safety': result['rollout']['telemetry']['collisions'] = 1
    write(case/'result.json',result)
    with pytest.raises(ValueError):
        bc.load_inputs(tmp_path, cases=((101,'case',2),))


def test_metrics_do_not_hide_all_wait_or_drop_proposal_errors():
    gate = torch.tensor([0,0,1,1]); proposal = torch.tensor([-1,-1,3,4])
    g = torch.tensor([[2.,0.]]*4)
    p = torch.zeros(4,25); p[2,3] = 2.; p[3,7] = 3.; p[3,4] = 2.
    result = bc.classification(g,p,gate,proposal)
    assert result['gate']['accuracy'] == .5 and result['gate']['RELEASE_recall'] == 0
    assert result['gate']['balanced_accuracy'] == .5
    assert result['proposal']['top1'] == .5 and result['proposal']['top3'] == 1
    assert result['joint_accuracy'] == .5
    assert bc.classification(g[:0],p[:0],gate[:0],proposal[:0])['gate'] is None
    assert bc.classification(g[:2],p[:2],gate[:2],proposal[:2])['proposal'] is None


def test_wait_rows_have_zero_proposal_gradient():
    gate_logits = torch.zeros(3,2,requires_grad=True)
    proposal_logits = torch.zeros(3,25,requires_grad=True)
    loss,_ = bc.hierarchical_bc_loss(gate_logits,proposal_logits,torch.tensor([0,1,0]),torch.tensor([-1,12,-1]))
    loss.backward()
    assert torch.count_nonzero(proposal_logits.grad[[0,2]]) == 0
    assert torch.count_nonzero(proposal_logits.grad[1]) > 0


def test_synthetic_smoke_writes_fixed_final_and_leaves_critic_frozen(tmp_path):
    output = tmp_path/'smoke'
    result = bc.run(tmp_path/'missing_no_real_data', output, 'smoke')
    assert result['status'] == 'SMOKE_OK'
    assert result['seeds'][0]['updates'] == 2
    assert result['seeds'][0]['actor_changed'] and result['seeds'][0]['critic_unchanged']
    assert not result['I0_QUALIFIED'] and not result['PPO_ALLOWED']
    protocol=read(output/'protocol.json')
    assert protocol['updates'] == 2 and protocol['model_seeds'] == [17]
    checkpoint = torch.load(output/'seed_17/final_diagnostic.pt', weights_only=True)
    assert checkpoint['observation_dim'] == 90 and checkpoint['diagnostic_only']
    assert checkpoint['train_seeds'] == [] and not checkpoint['qualified']
    with pytest.raises(FileExistsError):
        bc.run(tmp_path,output,'smoke')


def test_wall_failure_keeps_error_and_does_not_emit_checkpoint(tmp_path, monkeypatch):
    original = bc.fit
    monkeypatch.setattr(bc, 'fit', lambda data,out,seed,**kwargs: original(data,out,seed,updates=2,wall_seconds=-1))
    result=bc.run(tmp_path,tmp_path/'failure','smoke')
    assert result['status']=='ERROR' and 'wall budget' in result['error']['message']
    assert not (tmp_path/'failure/seed_17/final_diagnostic.pt').exists()
