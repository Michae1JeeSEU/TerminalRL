import json

import pytest
import torch

from terminal_marl.training import p3_v25_imitation_qualification as q
from terminal_marl.training import p3_v26_course_data as course
from terminal_marl.training import p3_v26_supplemental_data as supplemental_data
from terminal_marl.training import p3_v26_train_dataset as train_data
from terminal_marl.training import p3_neighbor_witness_data as witness_data
from terminal_marl.training import p3_neighbor_witness_dataset as witness_dataset
from terminal_marl.training.p3_v25_set_learnability import PairedInputs
from terminal_marl.training.p3_v3_bc_learnability import Data
from terminal_marl.scripts.diagnose_p3_intent_neighbor import run


def _resource_row(seed, width):
    return dict(schema=q.CONTEXT_SCHEMA, observation_schema=course.VERSION, partition='train', seed=seed,
        tick=0, uid='0', decision_id=f'{train_data.DATA_SCHEMA}:{seed}:0:0', actor_mask=True,
        ego_feature_hash=course.EGO_SCHEMA.feature_hash, traffic_feature_hash=q.TRAFFIC_SCHEMA.feature_hash,
        ego=[0.]*159, traffic=[[float(width)]*50]*width, traffic_mask=[True]*width,
        teacher_gate=int(width == 2), teacher_proposal=12 if width == 2 else None,
        original_observation=[0.]*90, context=dict(seed=seed,interface='m2i'))


def test_supplemental_loader_uses_declared_seeds_and_preserves_ragged_rows(monkeypatch,tmp_path):
    paths=[];validated=[]
    for index,(seed,_) in enumerate(supplemental_data.TRAIN_CASES,1):
        root=tmp_path/f'supp_{index}';episode=root/f'seed_{seed}';episode.mkdir(parents=True)
        row=_resource_row(seed,index)
        path=episode/'resource_context_observations.jsonl';path.write_text(json.dumps(row)+'\n')
        paths.append(root)
        validated.append((path,dict(resource_reconstruction=dict(rows=1,sha256=q.sha(path)),reconstruction=dict(rows=1))))
    monkeypatch.setattr(supplemental_data,'validated_case',lambda directory,index: validated[index-1])
    value=train_data.load_supplemental_train(paths)
    assert value.original.ids == tuple(f'{train_data.DATA_SCHEMA}:{seed}:0:0' for seed,_ in supplemental_data.TRAIN_CASES)
    assert [item['source_group'] for item in value.original.inputs] == ['supplemental']*3
    assert value.new[0].shape == (3,159) and value.new[1].shape == (3,3,50)
    assert value.new[2].sum(1).tolist() == [1,2,3]
    bad=json.loads((paths[0]/f'seed_{supplemental_data.TRAIN_CASES[0][0]}'/'resource_context_observations.jsonl').read_text())
    bad['seed']=999999
    (paths[0]/f'seed_{supplemental_data.TRAIN_CASES[0][0]}'/'resource_context_observations.jsonl').write_text(json.dumps(bad)+'\n')
    with pytest.raises(ValueError,match='identity'):
        train_data.load_supplemental_train(paths)


def _real_supplemental_cases(tmp_path):
    paths=[]
    for index,(seed,scenario) in enumerate(supplemental_data.TRAIN_CASES,1):
        root=tmp_path/f'real_supp_{index}';episode=root/f'seed_{seed}';episode.mkdir(parents=True)
        path=episode/'resource_context_observations.jsonl';path.write_text(json.dumps(_resource_row(seed,index))+'\n')
        identity=dict(protocol=supplemental_data.PROTOCOL,profile=course.PROFILE,partition='train',
            seed=seed,scenario=scenario)
        protocol=dict(identity,smoke=False,task=course.task_settings(course.build_config(scenario)),
            learning_observation_schema=course.VERSION)
        result=dict(identity,status='COLLECTION_OK',error=None,blockers=[],reconstruction=dict(rows=1),
            resource_reconstruction=dict(rows=1,sha256=q.sha(path)))
        (root/'protocol.json').write_text(json.dumps(protocol))
        (root/'result.json').write_text(json.dumps(result))
        paths.append(root)
    return paths


@pytest.mark.parametrize('mutation,pattern',(
    ('q','wrong V2.6 train identity'),('wrong_seed','wrong V2.6 train identity'),
    ('incomplete','incomplete V2.6 cycle'),('changed_bytes','resource bytes changed')))
def test_supplemental_validator_rejects_wrong_partition_seed_incomplete_or_changed_bytes(tmp_path,mutation,pattern):
    paths=_real_supplemental_cases(tmp_path)
    root=paths[0];seed=supplemental_data.TRAIN_CASES[0][0]
    protocol=json.loads((root/'protocol.json').read_text());result=json.loads((root/'result.json').read_text())
    if mutation=='q':
        protocol['partition']=result['partition']='Q'
    elif mutation=='wrong_seed':
        protocol['seed']=result['seed']=999999
    elif mutation=='incomplete':
        result['status']='INCOMPLETE'
    else:
        path=root/f'seed_{seed}'/'resource_context_observations.jsonl'
        path.write_text(path.read_text()+'\n')
    (root/'protocol.json').write_text(json.dumps(protocol));(root/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match=pattern):
        train_data.load_supplemental_train(paths)


def _part(prefix, seed, width):
    original=Data(torch.zeros(1,90),torch.tensor([1]),torch.tensor([12]),
        (f'{prefix}:{seed}',),[dict(seed=seed,interface='o2m')],
        [dict(seed=seed,partition='train')])
    traffic=torch.ones(1,width,50);mask=torch.ones(1,width,dtype=torch.bool)
    return PairedInputs(original,(torch.ones(1,159),traffic,mask))


def test_nine_cycle_combine_pads_ragged_traffic_and_rejects_duplicate_ids():
    parts=[_part('original',893511,1),_part('independent',893601,3),_part('supplemental',893604,2)]
    value=train_data.combine_train(parts)
    assert value.original.ids == ('original:893511','independent:893601','supplemental:893604')
    assert value.new[0].shape == (3,159) and value.new[1].shape == (3,3,50)
    assert value.new[2].sum(1).tolist() == [1,3,2]
    with pytest.raises(ValueError,match='duplicate'):
        train_data.combine_train([parts[0],parts[0]])


def _targets(state, proposal, pad):
    return witness_dataset.WitnessTargets(torch.tensor([state]),torch.tensor([proposal]),torch.tensor([pad]),
        torch.zeros(1,2),torch.ones(1,2,dtype=torch.bool),torch.zeros(1),torch.ones(1,dtype=torch.bool),
        torch.tensor([893604]))


def test_witness_target_join_rejects_duplicate_state_proposal_pad():
    first=_targets(0,12,0);second=_targets(1,13,2)
    joined=witness_dataset.combine_witness_targets((first,second))
    assert joined.state_index.tolist() == [0,1] and joined.proposal.tolist() == [12,13]
    with pytest.raises(ValueError,match='duplicate'):
        witness_dataset.combine_witness_targets((first,first))


def test_witness_loader_rejects_a_declared_seed_set_from_another_group(tmp_path):
    root=tmp_path/'wrong_witness';root.mkdir()
    witnesses=root/'witnesses.jsonl';witnesses.write_text('')
    (root/'protocol.json').write_text(json.dumps(dict(
        protocol=witness_data.SUPPLEMENTAL_PROTOCOL,smoke=False,inputs=[dict(seed=893601)],selected_ids=[])))
    (root/'result.json').write_text(json.dumps(dict(status='COMPLETE',states=0,witness_sha256=q.sha(witnesses))))
    with pytest.raises(ValueError,match='source seed set'):
        witness_dataset.load_witness_targets(root,None)


def test_supplemental_witness_collection_uses_separate_protocol_and_budget(monkeypatch,tmp_path):
    rows=[]
    for seed,interface in zip((893604,893605,893606),('o2m','m2i','i2f')):
        row=dict(seed=seed,context={'interface':interface},decision_id=f'{seed}:{interface}')
        rows.append((None,None,row,None))
    calls=[]
    monkeypatch.setattr(witness_data,'selected_inputs',lambda directories, supplemental=False: (rows,[dict(seed=s) for s,_ in supplemental_data.TRAIN_CASES]))
    monkeypatch.setattr(witness_data,'witness_state',lambda episode,entry,row,label,remaining_calls: calls.append(remaining_calls) or dict(decision_id=row['decision_id'],plan_calls=1))
    result=witness_data.collect([],tmp_path/'supp_witness',smoke=True,supplemental=True)
    protocol=json.loads((tmp_path/'supp_witness/protocol.json').read_text())
    assert result['status']=='SMOKE_OK' and protocol['protocol']==witness_data.SUPPLEMENTAL_PROTOCOL
    assert protocol['source_group']=='supplemental' and protocol['max_states']==72
    assert protocol['max_plan_calls']==36 and protocol['wall_seconds']==120 and calls==[36,35,34]
    assert protocol['selected_ids']==[f'{seed}:{interface}' for seed,interface in zip((893604,893605,893606),('o2m','m2i','i2f'))]
    calls.clear()
    full=witness_data.collect([],tmp_path/'supp_witness_full',supplemental=True)
    full_protocol=json.loads((tmp_path/'supp_witness_full/protocol.json').read_text())
    assert full['status']=='COMPLETE' and full_protocol['max_plan_calls']==864
    assert full_protocol['max_states']==72 and full_protocol['wall_seconds']==1800 and calls==[864,863,862]


def test_supplemental_smoke_is_nine_fold_native_only_and_conflicts_with_max_gate(tmp_path):
    output=tmp_path/'supp_cv'
    result=run(None,None,None,output,smoke=True,supplemental=(),supplemental_witness=tmp_path/'supp_witness')
    saved=json.loads((output/'result.json').read_text())
    assert result['status']=='COMPLETE' and len(result['folds'])==9
    assert saved['protocol']['input_seeds']==list(range(1,10))
    assert saved['protocol']['gate_mode']=='first_feasible_union'
    assert saved['protocol']['arms']==['native_pair_auxiliary']
    assert saved['protocol']['source_group']=='old6_plus_supplemental3'
    assert saved['protocol']['supplemental_role']=='train_only_development_not_qualification'
    assert [v['update'] for v in saved['folds'][0]['curve']] == [0,1,2]
    with pytest.raises(ValueError,match='max_candidate_gate'):
        run(None,None,None,tmp_path/'conflict',smoke=True,max_candidate_gate=True,
            supplemental=(),supplemental_witness=tmp_path/'supp_witness')
