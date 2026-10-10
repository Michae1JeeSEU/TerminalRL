from concurrent.futures import Future
import json
from pathlib import Path

import pytest
import torch

from terminal_marl.scripts import diagnose_p3_intent_neighbor as diagnostic
from terminal_marl.training import p3_neighbor_witness_dataset as witness_dataset


def _folds_without_runtime_paths(result):
    cleaned=[]
    for fold in result['folds']:
        cleaned.append({key:value for key,value in fold.items()
            if key not in ('elapsed_seconds','progress_path')})
    return cleaned


def test_synthetic_nine_route_serial_and_spawn_parallel_curves_match(tmp_path):
    serial=diagnostic.run(None,None,None,tmp_path/'serial',smoke=True,supplemental=(),
        supplemental_witness=tmp_path/'supplemental',workers=1)
    parallel=diagnostic.run(None,None,None,tmp_path/'parallel',smoke=True,supplemental=(),
        supplemental_witness=tmp_path/'supplemental',workers=2)
    assert serial['status']=='COMPLETE' and parallel['status']=='COMPLETE'
    assert _folds_without_runtime_paths(serial)==_folds_without_runtime_paths(parallel)
    assert len(serial['folds'])==len(parallel['folds'])==9
    progress=list((tmp_path/'parallel').glob('fold_*.progress.json'))
    assert len(progress)==9
    assert {json.loads(path.read_text())['status'] for path in progress}=={'COMPLETE'}
    assert all('validation_reference_pairs' not in curve
        for fold in serial['folds'] for curve in fold['curve'])


def test_synthetic_nine_split_has_no_held_seed_leakage():
    paired,witness=diagnostic.smoke_inputs(supplemental=True)
    seeds=[int(context['seed']) for context in paired.original.contexts]
    for held_seed in range(1,10):
        train,held,wtrain,wheld=diagnostic.split(paired,witness,held_seed)
        assert set(seeds[index] for index in train.tolist())==set(range(1,10))-{held_seed}
        assert set(seeds[index] for index in held.tolist())=={held_seed}
        assert set(int(value) for value in witness.seeds[wtrain].tolist())==set(range(1,10))-{held_seed}
        assert set(int(value) for value in witness.seeds[wheld].tolist())=={held_seed}


def test_prefix_route_requires_paired_non_smoke_supplemental_inputs(tmp_path):
    with pytest.raises(ValueError,match='paired nine-cycle supplemental route'):
        diagnostic.run(None,None,None,tmp_path/'smoke_prefix',smoke=True,
            prefix_witness=tmp_path/'prefix')
    with pytest.raises(ValueError,match='paired nine-cycle supplemental route'):
        diagnostic.run(None,None,None,tmp_path/'smoke_prefix_supp',smoke=True,supplemental=(),
            supplemental_witness=tmp_path/'supplemental',prefix_witness=tmp_path/'prefix')
    with pytest.raises(ValueError,match='workers must be an integer in 1..4'):
        diagnostic.run(None,None,None,tmp_path/'bad_workers',smoke=True,workers=5)


def test_prefix_route_protocol_and_duplicate_targets_are_checked(tmp_path,monkeypatch):
    paired,witness=diagnostic.smoke_inputs(supplemental=True)
    monkeypatch.setattr(diagnostic,'load_nine_cycle_train',lambda *args:paired)
    monkeypatch.setattr(diagnostic,'load_witness_targets',lambda directory,value:witness)
    monkeypatch.setattr(diagnostic,'combine_witness_targets',lambda parts:witness)
    monkeypatch.setattr(diagnostic,'_execute_folds',lambda *args,**kwargs:dict(
        status='COMPLETE',protocol=kwargs['protocol'],folds=[],error=None))
    result=diagnostic.run((Path('old1'),Path('old2'),Path('old3')),
        (Path('ind1'),Path('ind2'),Path('ind3')),tmp_path/'old',tmp_path/'prefix_route',
        supplemental=(Path('supp1'),Path('supp2'),Path('supp3')),
        supplemental_witness=tmp_path/'supplemental',prefix_witness=tmp_path/'prefix',workers=2)
    protocol=json.loads((tmp_path/'prefix_route'/'protocol.json').read_text())
    assert result['status']=='COMPLETE'
    assert protocol['source_group']=='old6_plus_supplemental3_plus_prefix'
    assert protocol['comparison_control_reference']=='p3_neighbor_nine_cycle_cv_20261009/nine_cycle'
    assert protocol['prefix_witness_protocol']==witness_dataset.PREFIX_COVERAGE_PROTOCOL
    assert protocol['prefix_role']=='new_train_decision_joint_candidates_only_not_qualification'
    assert protocol['arms']==['native_pair_auxiliary']
    first=witness_dataset.WitnessTargets(*(value[:1] for value in witness.__dict__.values()))
    with pytest.raises(ValueError,match='duplicate witness target'):
        witness_dataset.combine_witness_targets((first,first))


def test_prefix_validation_keeps_original_held_reference_separate(tmp_path):
    paired,reference=diagnostic.smoke_inputs(supplemental=True)
    extra=reference.take(torch.tensor([0]))
    extra.pad=extra.pad+1
    expanded=witness_dataset.combine_witness_targets((reference,extra))
    outcome=diagnostic.train_fold(paired,expanded,1,'native_pair_auxiliary',
        'first_feasible_union',(0,),10.,tmp_path/'reference.progress.json',
        reference_witness=reference)
    assert outcome['status']=='COMPLETE'
    curve=outcome['record']['curve'][0]
    assert curve['validation_pairs']['queries']==3
    assert curve['validation_reference_pairs']['queries']==2


def _write_minimal_witness(directory,paired,protocol_name,joint,*,status='COMPLETE',smoke=False,
                           wrong_sha=False):
    directory.mkdir(parents=True)
    index=1
    active=int(paired.new[2][index].sum())
    traffic_uids=list(range(100,100+active))
    gate=int(paired.original.gate[index])
    teacher_joint=int(paired.original.proposal[index])+1 if gate else 0
    candidate=dict(joint=joint,original_feasible=True,original_reason='certified',destinations=[dict(
        pad=None,feasible=True,reason='certified',pairs=[dict(uid=uid,blocked=False,approved=True)
            for uid in traffic_uids])])
    row=dict(decision_id=paired.original.ids[index],seed=int(paired.original.contexts[index]['seed']),
        ego=paired.new[0][index].tolist(),traffic=paired.new[1][index,:active].tolist(),
        traffic_mask=paired.new[2][index,:active].tolist(),traffic_uids=traffic_uids,
        teacher_gate=gate,teacher_joint=teacher_joint,candidates=[candidate])
    witness_path=directory/'witnesses.jsonl'
    witness_path.write_text(json.dumps(row)+'\n',encoding='utf8')
    expected={1,2} if protocol_name==witness_dataset.PROTOCOL else {1,2,3}
    protocol=dict(protocol=protocol_name,inputs=[dict(seed=seed) for seed in sorted(expected)],
        smoke=smoke,selected_ids=[row['decision_id']])
    witness_sha=witness_dataset.sha(witness_path)
    report=dict(status=status,states=1,witness_sha256='bad' if wrong_sha else witness_sha)
    (directory/'protocol.json').write_text(json.dumps(protocol),encoding='utf8')
    (directory/'result.json').write_text(json.dumps(report),encoding='utf8')
    return row


def test_serialized_prefix_loader_accepts_distinct_joint_and_rejects_duplicate(tmp_path,monkeypatch):
    monkeypatch.setattr(witness_dataset.old,'TRAIN_CASES',((1,'old'),))
    monkeypatch.setattr(witness_dataset.independent,'TRAIN_CASES',((2,'independent'),))
    monkeypatch.setattr(witness_dataset.supplemental,'TRAIN_CASES',((3,'supplemental'),))
    paired,_=diagnostic.smoke_inputs(supplemental=True)
    old_dir=tmp_path/'old';prefix_dir=tmp_path/'prefix';duplicate_dir=tmp_path/'duplicate'
    _write_minimal_witness(old_dir,paired,witness_dataset.PROTOCOL,12)
    _write_minimal_witness(prefix_dir,paired,witness_dataset.PREFIX_COVERAGE_PROTOCOL,13)
    old_part=witness_dataset.load_witness_targets(old_dir,paired)
    prefix_part=witness_dataset.load_witness_targets(prefix_dir,paired)
    combined=witness_dataset.combine_witness_targets((old_part,prefix_part))
    assert len(combined.state_index)==2 and combined.proposal.tolist()==[11,12]
    _write_minimal_witness(duplicate_dir,paired,witness_dataset.PREFIX_COVERAGE_PROTOCOL,12)
    duplicate=witness_dataset.load_witness_targets(duplicate_dir,paired)
    with pytest.raises(ValueError,match='duplicate witness target'):
        witness_dataset.combine_witness_targets((old_part,duplicate))


@pytest.mark.parametrize('variant',('smoke','incomplete','sha'))
def test_serialized_loader_rejects_smoke_incomplete_or_bad_sha(tmp_path,monkeypatch,variant):
    monkeypatch.setattr(witness_dataset.old,'TRAIN_CASES',((1,'old'),))
    monkeypatch.setattr(witness_dataset.independent,'TRAIN_CASES',((2,'independent'),))
    monkeypatch.setattr(witness_dataset.supplemental,'TRAIN_CASES',((3,'supplemental'),))
    paired,_=diagnostic.smoke_inputs(supplemental=True)
    directory=tmp_path/variant
    _write_minimal_witness(directory,paired,witness_dataset.PREFIX_COVERAGE_PROTOCOL,13,
        status='INCOMPLETE' if variant=='incomplete' else 'COMPLETE',
        smoke=variant=='smoke',wrong_sha=variant=='sha')
    with pytest.raises(ValueError):
        witness_dataset.load_witness_targets(directory,paired)


def _scheduler_specs(tmp_path,paired):
    return [dict(seed=seed,arm='native_pair_auxiliary',gate_mode='first_feasible_union',marks=(0,),
        wall_seconds=1,progress_path=tmp_path/f'fold_{seed}.progress.json') for seed in (1,2,3,4)]


def test_fold_failure_stops_serial_and_parallel_submission(tmp_path,monkeypatch):
    paired,witness=diagnostic.smoke_inputs(supplemental=True)
    specs=_scheduler_specs(tmp_path/'serial',paired)
    serial_calls=[]
    def failed_serial(*args,**kwargs):
        serial_calls.append(args[2]);return dict(status='INCOMPLETE',record={},error='small failure')
    monkeypatch.setattr(diagnostic,'train_fold',failed_serial)
    serial=diagnostic._execute_folds(paired,witness,specs,workers=1,
        root_path=tmp_path/'serial.json',protocol={})
    assert serial['status']=='INCOMPLETE' and serial_calls==[1]

    specs=_scheduler_specs(tmp_path/'parallel',paired)
    submitted=[]
    class ImmediatePool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def submit(self,fn,*args):
            seed=args[2];submitted.append(seed);future=Future()
            future.set_result(dict(status='INCOMPLETE' if seed==1 else 'COMPLETE',
                record={},error='small failure' if seed==1 else None))
            return future
    monkeypatch.setattr(diagnostic,'ProcessPoolExecutor',ImmediatePool)
    parallel=diagnostic._execute_folds(paired,witness,specs,workers=2,
        root_path=tmp_path/'parallel.json',protocol={})
    assert parallel['status']=='INCOMPLETE' and len(submitted)==2 and set(submitted)=={1,2}
