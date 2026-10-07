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


def test_balanced_gate_uses_fixed_weights_and_batch_row_denominator():
    g = torch.tensor([[1.,0.],[0.,2.],[1.,3.]], requires_grad=True)
    p = torch.zeros(3,25,requires_grad=True)
    y, q = torch.tensor([0,0,1]), torch.tensor([-1,-1,12])
    weighted = bc.training_loss(g,p,y,q,bc.GATE_WEIGHTS)
    ce = torch.nn.functional.cross_entropy(g,y,reduction='none')
    expected = (ce * torch.tensor(bc.GATE_WEIGHTS)[y]).sum()/3
    expected += torch.nn.functional.cross_entropy(p[2:],q[2:])
    assert torch.allclose(weighted, expected)
    assert bc.GATE_WEIGHTS[0]*817 == pytest.approx(523)
    assert bc.GATE_WEIGHTS[1]*229 == pytest.approx(523)
    gp = torch.autograd.grad(weighted, (g,p), retain_graph=True)
    original = bc.training_loss(g,p,y,q)
    unweighted_gp = torch.autograd.grad(original, (g,p))
    assert torch.allclose(gp[0], unweighted_gp[0]*torch.tensor(bc.GATE_WEIGHTS)[y,None])
    assert torch.equal(gp[1], unweighted_gp[1])
    assert torch.count_nonzero(gp[1][:2]) == 0
    # Homogeneous WAIT batch keeps its fixed weight, unlike weighted-mean CE.
    only_wait = bc.training_loss(g[:2],p[:2],y[:2],q[:2],bc.GATE_WEIGHTS)
    assert torch.allclose(only_wait, ce[:2].mean()*bc.GATE_WEIGHTS[0])


def test_balanced_smoke_preserves_initialization_and_metric_definition(tmp_path):
    unweighted = bc.run(tmp_path/'missing',tmp_path/'unweighted','smoke')
    balanced = bc.run(tmp_path/'missing',tmp_path/'balanced','smoke',gate_balance=True)
    assert balanced['status'] == 'SMOKE_OK'
    assert balanced['protocol'] == bc.BALANCED_PROTOCOL
    b, u = balanced['seeds'][0], unweighted['seeds'][0]
    assert b['initial'] == u['initial']
    assert b['gate_weights'] == bc.GATE_WEIGHTS
    assert b['critic_unchanged'] and b['actor_changed'] and b['updates'] == 2
    cp = torch.load(tmp_path/'balanced/seed_17/final_diagnostic.pt',weights_only=True)
    assert cp['protocol'] == bc.BALANCED_PROTOCOL and cp['gate_weights'] == bc.GATE_WEIGHTS
    assert cp['diagnostic_only'] and not cp['qualified']


def test_paired_initialization_mismatch_precedes_optimizer(tmp_path,monkeypatch):
    def forbidden(*a,**k):
        raise AssertionError('optimizer must not be created')
    monkeypatch.setattr(torch.optim,'Adam',forbidden)
    with pytest.raises(ValueError,match='initialization'):
        bc.fit(bc.synthetic_data(),tmp_path/'mismatch',17,updates=2,
            gate_weights=bc.GATE_WEIGHTS,reference_initial={})


def test_paired_initialization_accepts_serialized_counter_keys(tmp_path):
    import json
    data=bc.synthetic_data()
    torch.manual_seed(17)
    initial=bc.measure(bc.HierarchicalActorCentralCriticV42(90),data)
    serialized=json.loads(json.dumps(initial))
    assert initial != serialized  # Integer Counter keys become JSON string keys.
    result=bc.fit(data,tmp_path/'paired',17,updates=2,gate_weights=bc.GATE_WEIGHTS,
        protocol=bc.BALANCED_PROTOCOL,reference_initial=serialized)
    assert result['reference_initial_equal'] and result['updates'] == 2
    assert result['critic_unchanged'] and result['actor_changed']


def test_reference_rejects_different_row_identity_and_incomplete_seed(tmp_path):
    data = bc.synthetic_data()
    case = tmp_path/bc.REFERENCE_CASE; case.mkdir(parents=True)
    write(case/'protocol.json',dict(model_seeds=list(bc.MODEL_SEEDS),updates=bc.UPDATES,
        batch_size=bc.BATCH_SIZE,learning_rate=bc.LEARNING_RATE))
    inputs = dict(decision_ids=list(data.ids),episodes=data.inputs,
        counts=dict(rows=1046,WAIT=817,RELEASE=229))
    write(case/'inputs.json',inputs)
    result = dict(status='FIT_COMPLETE',protocol=bc.PROTOCOL,
        seeds=[dict(seed=s,status='FIT_COMPLETE',updates=1000,critic_unchanged=True) for s in bc.MODEL_SEEDS])
    write(case/'result.json',result)
    assert set(bc.reference_results(tmp_path,data)) == set(bc.MODEL_SEEDS)
    inputs['decision_ids'][0] = 'wrong'; write(case/'inputs.json',inputs)
    with pytest.raises(ValueError,match='dataset'):
        bc.reference_results(tmp_path,data)
    inputs['decision_ids'] = list(data.ids); write(case/'inputs.json',inputs)
    result['seeds'][1]['updates'] = 999; write(case/'result.json',result)
    with pytest.raises(ValueError,match='incomplete'):
        bc.reference_results(tmp_path,data)


def test_detach_preserves_forward_and_heads_but_removes_proposal_encoder_gradient():
    torch.manual_seed(17)
    model=bc.HierarchicalActorCentralCriticV42(90)
    data=bc.synthetic_data()
    g,p=bc.training_distributions(model,data.x)
    gd,pd=bc.training_distributions(model,data.x,True)
    assert torch.equal(g.logits,gd.logits) and torch.equal(p.logits,pd.logits)
    original=bc.training_loss(g.logits,p.logits,data.gate,data.proposal)
    detached=bc.training_loss(gd.logits,pd.logits,data.gate,data.proposal)
    assert torch.equal(original,detached)
    _,pieces=bc.hierarchical_bc_loss(gd.logits,pd.logits,data.gate,data.proposal)
    encoder=list(model.actor_encoder.parameters())
    absent=torch.autograd.grad(pieces['proposal'],encoder,allow_unused=True,retain_graph=True)
    assert all(x is None for x in absent)
    combined_grad=torch.autograd.grad(detached,encoder,retain_graph=True)
    gate_grad=torch.autograd.grad(pieces['gate'],encoder,retain_graph=True)
    assert all(torch.equal(a,b) for a,b in zip(combined_grad,gate_grad))
    heads=list(model.gate_head.parameters())+list(model.proposal_head.parameters())
    original_heads=torch.autograd.grad(original,heads,retain_graph=True)
    detached_heads=torch.autograd.grad(detached,heads)
    assert all(torch.equal(a,b) for a,b in zip(original_heads,detached_heads))
    assert any(torch.count_nonzero(x)>0 for x in detached_heads[2:])


def test_detach_smoke_keeps_proposal_head_trainable_and_original_initial_values(tmp_path):
    original=bc.run(tmp_path/'missing',tmp_path/'original','smoke')
    result=bc.run(tmp_path/'missing',tmp_path/'detached','smoke',detach_proposal_encoder=True)
    assert result['status']=='SMOKE_OK' and result['protocol']==bc.DETACHED_PROTOCOL
    seed=result['seeds'][0]
    assert seed['initial']==original['seeds'][0]['initial']
    assert seed['detach_proposal_encoder'] and seed['gate_weights'] is None
    assert all(seed['actor_module_changed'].values()) and seed['critic_unchanged']
    assert 0 <= seed['final_gate_auc']['overall'] <= 1
    cp=torch.load(tmp_path/'detached/seed_17/final_diagnostic.pt',weights_only=True)
    assert cp['detach_proposal_encoder'] and cp['protocol']==bc.DETACHED_PROTOCOL
    assert not cp['qualified'] and cp['diagnostic_only']


def test_detach_and_class_balance_cannot_be_combined(tmp_path):
    with pytest.raises(ValueError,match='one diagnostic factor'):
        bc.run(tmp_path,tmp_path/'invalid','smoke',gate_balance=True,detach_proposal_encoder=True)
    assert not (tmp_path/'invalid').exists()


def test_auc_counts_ties_and_reports_missing_class():
    assert bc.gate_auc(torch.tensor([0.,1.]),torch.tensor([0,1]))==1
    assert bc.gate_auc(torch.tensor([1.,0.]),torch.tensor([0,1]))==0
    assert bc.gate_auc(torch.tensor([.5,.5]),torch.tensor([0,1]))==.5
    assert bc.gate_auc(torch.tensor([.5,.5]),torch.tensor([0,0])) is None
