import json
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.configs.arrival_restructure import build_config, EXTERNAL_AUTHORITY
from terminal_marl.core.lifecycle import OutcomeLedger
from terminal_marl.core.uav import TerminalOutcome
from terminal_marl.scripts.evaluate_arrival_restructure import (
    accounting, accumulate, canonical, finite_json, natural_terminal,
    reference_candidates, replay_run, run_arm, select_candidate,
)


def test_json_finite_and_uuid_normalization():
    result=finite_json({'a': np.array([float('nan'),float('inf'),2.]), 'b': float('-inf')})
    assert json.loads(json.dumps(result,allow_nan=False))=={'a':[None,None,2.], 'b':None}
    assert canonical({'plan_id':'plan-a','trajectory_hash':'abc'})==canonical({'plan_id':'plan-b','trajectory_hash':'abc'})
    assert canonical({'trajectory_hash':'abc'})!=canonical({'trajectory_hash':'def'})


def test_reference_registry_and_config():
    for n in range(3,7):
        rows=reference_candidates(n,tuple(reversed(range(n))))
        assert len(rows)==12
        assert len({(tuple(r['order']),r['variant']) for r in rows})==len(rows)
        assert tuple(reversed(range(n))) in {tuple(r['order']) for r in rows}
        config=build_config(reference_n=n,authority=EXTERNAL_AUTHORITY)
        assert config.warm_start_uavs==config.total_requested_uavs==n
        assert config.initial_soc==.8 and config.fixed_arrival_schedule==(0.,)*n
        assert not config.task_priority_enabled


def test_terminal_and_censor_denominators():
    ledger=OutcomeLedger(); ledger.request(0); ledger.resolve(0,TerminalOutcome.COMPLETED)
    env=SimpleNamespace(config=SimpleNamespace(total_requested_uavs=3),ledger=ledger,
        arrivals=SimpleNamespace(exhausted=lambda:False),active={},service_uavs={},completed_uavs=[],
        admission=SimpleNamespace(queue=SimpleNamespace(items=[])),energy_outcomes=[],now=2.,external_queue_uav_seconds=0.)
    assert not natural_terminal(env)
    result=accounting(env,{})
    assert (result['expected_total_requested'],result['actually_requested'],result['known_unresolved'],result['not_yet_requested'])==(3,1,0,2)
    env.config.total_requested_uavs=1; env.arrivals.exhausted=lambda:True
    assert natural_terminal(env)


def test_deduplicate_cumulative_wait_and_energy():
    obj=SimpleNamespace(arrival_id=7,external_wait_time=2.,internal_wait_time=3.,energy_consumed=4.)
    rows={}; accumulate(rows,[obj,obj]); obj.internal_wait_time=5.; obj.energy_consumed=8.; accumulate(rows,[obj])
    assert rows[7]['external_wait']==2. and rows[7]['internal_wait']==5. and rows[7]['energy']==8.


def test_selection_excludes_invalid_censored_and_error():
    metrics=dict(F=0,D=0,U=0,E=0,total_wait=4.,makespan=5.,energy=6.)
    good=dict(status='COMPLETED',full_valid=True,metrics=metrics)
    assert select_candidate([dict(status='CENSORED',full_valid=True,metrics=metrics),dict(status='COMPLETED',full_valid=False,metrics=metrics),dict(status='ERROR'),good]) is good
    assert select_candidate([dict(status='CENSORED',full_valid=False)]) is None


def test_native_two_ticks_and_order_tape_replay(tmp_path):
    spec=dict(mode='smoke',scenario='flow6_mixed',seed=896091,controller='energy_slack',
              max_ticks=2,source_commit='unit-test',output=str(tmp_path/'run'))
    result=run_arm(spec)
    assert result['status']=='CENSORED',result.get('stop_reason')
    assert result['ticks']==2 and not result['full_valid']
    assert result['metrics']['not_yet_requested']==6
    assert result['metrics']['wait_measurement']=='accrued_partial'
    assert result['metrics']['mean_wait_over_requested'] is None
    from terminal_marl.configs.arrival_restructure import protocol_snapshot
    old_header=protocol_snapshot(mode=spec['mode'],scenario=spec['scenario'],seed=spec['seed'],
        controller=spec['controller'],authority=EXTERNAL_AUTHORITY,source_commit=spec['source_commit'],
        smoke=True,max_ticks=2)
    assert all(result['protocol'][key]==value for key,value in old_header.items())
    replay=replay_run(tmp_path/'run',tmp_path/'replay')
    assert replay['status']=='MATCH',replay
    assert replay['matched_steps']==2 and not replay['whole_cycle']
    tape=tmp_path/'run'/'steps.jsonl'
    entries=[json.loads(line) for line in tape.read_text().splitlines()]
    entries[0]['release_order']=[999]
    tape.write_text(''.join(json.dumps(row)+'\n' for row in entries))
    altered=replay_run(tmp_path/'run',tmp_path/'altered')
    assert altered['status']=='ERROR'
    with pytest.raises(FileExistsError): run_arm(spec)


def test_legacy_course_config_is_preserved():
    from terminal_marl.configs.p3_v25_course import build_config as course
    for scene in ('flow6_mixed','dense24_fixed','dense24_mixed'):
        assert build_config(scene)==course(scene)


def test_external_downstream_run_and_phase2_metadata(tmp_path):
    spec=dict(mode='phase2',scenario='dense24_fixed',seed=896113,
              controller='downstream_soc_external',reference_n=3,max_ticks=1,
              source_commit='unit-test',output=str(tmp_path/'external_rule'),
              protocol_version='phase2_test',comparison_boundary='same external authority',
              registration={'seeds':[896113]},profile='phase2_profile',
              concurrency_budget={'total_max':3,'nested_pools':False},
              protocol_revision_reason='same authority validation',
              control_authority='invalid_override',q_consumed=True,ppo_allowed=True)
    result=run_arm(spec)
    assert result['status']=='CENSORED',result.get('stop_reason')
    assert result['protocol']['control_authority']==EXTERNAL_AUTHORITY
    for key in ('profile','protocol_version','comparison_boundary','registration',
                'concurrency_budget','protocol_revision_reason'):
        assert result['protocol'][key]==spec[key]
    assert not any(key.startswith(('reference_registered_','max_reference_')) for key in result['protocol'])
    assert 'reference_permutation_selection' not in result['protocol']
    assert result['protocol']['seed']==896113
    assert result['protocol']['source_commit']=='unit-test'
    assert result['protocol']['q_consumed'] is False
    assert result['protocol']['ppo_allowed'] is False
    entry=json.loads((tmp_path/'external_rule'/'steps.jsonl').read_text())
    assert set(entry['release_order'])=={0,1,2}
    assert entry['diagnostics']['controller']=='downstream_soc_external'
    assert entry['diagnostics']['ordering_state']=='pre_step_active'
    assert entry['diagnostics']['same_tick_admissions']=='appended_in_uid_order_by_external_interface'
    assert entry['diagnostics']['legacy_stepwise_equivalence'] is False
    assert all(candidates==[[1.,0.,0.],[1.,-1.,0.],[1.,1.,0.]] for candidates in entry['shortlists'].values())
    json.dumps(result,allow_nan=False)
    assert replay_run(tmp_path/'external_rule',tmp_path/'external_replay')['status']=='MATCH'


@pytest.mark.parametrize('controller,candidate', [
    ('unknown',None), ('reference_static',None),
    ('downstream_soc_external',{'order':[],'variant':'all_release'}),
])
def test_invalid_controller_never_falls_back_to_nominal(tmp_path,controller,candidate):
    with pytest.raises(ValueError):
        run_arm(dict(controller=controller,candidate=candidate,output=str(tmp_path/'invalid')))
    assert not (tmp_path/'invalid').exists()


def test_integrity_checks_reject_duplicate_living_and_unchecked_plan():
    from terminal_marl.scripts.evaluate_arrival_restructure import integrity_checks
    ledger=OutcomeLedger(); ledger.request(1)
    u=SimpleNamespace(arrival_id=1,committed_plan=None)
    env=SimpleNamespace(active={1:u},service_uavs={},admission=SimpleNamespace(queue=SimpleNamespace(items=[])),ledger=ledger,telemetry={},safety=SimpleNamespace(checked_trajectory_hashes={}))
    assert integrity_checks(env)['ledger_integrity']
    env.service_uavs[1]=u
    assert not integrity_checks(env)['ledger_integrity']
    env.service_uavs.clear()
    u.committed_plan=SimpleNamespace(plan_id='p',trajectory_hash='hash')
    assert not integrity_checks(env)['active_plan_hashes_match']
    env.safety.checked_trajectory_hashes['p']='hash'
    assert integrity_checks(env)['active_plan_hashes_match']


def test_registered_permutations_span_full_list_and_include_energy():
    import itertools
    from terminal_marl.configs.arrival_restructure import REFERENCE_REGISTERED_PERMUTATIONS, protocol_snapshot
    assert REFERENCE_REGISTERED_PERMUTATIONS==6
    permutations=list(itertools.permutations(range(6)))
    selected=[permutations[i] for i in np.linspace(0,719,6,dtype=int)]
    rows=reference_candidates(6,permutations[1])
    assert [tuple(r['order']) for r in rows[::2]]==selected[:-1]+[permutations[1]]
    header=protocol_snapshot(mode='reference',scenario='dense24_fixed',seed=896116,controller='reference_static',authority=EXTERNAL_AUTHORITY)
    assert header['protocol_version']=='arrival_restructure_protocol_v2'
    assert header['reference_registered_candidates_total']==48
    assert header['max_reference_permutations']==24 and header['max_reference_candidates']==48


@pytest.mark.parametrize('outcomes,joint,normal', [
    ({'N':3,'E':0,'F':0,'D':0},True,True),
    ({'N':2,'E':1,'F':0,'D':0},True,False),
    ({'N':2,'E':0,'F':1,'D':0},False,False),
    ({'N':2,'E':0,'F':0,'D':1},False,False),
])
def test_reference_witness_requires_all_requests_landed(outcomes,joint,normal):
    from terminal_marl.scripts.evaluate_arrival_restructure import reference_evidence
    metrics=dict(expected_total_requested=3,U=0,not_yet_requested=0,**outcomes)
    best=dict(status='COMPLETED',full_valid=True,metrics=metrics)
    replay=dict(status='MATCH',whole_cycle=True)
    for complete in (True,False):
        evidence=reference_evidence(best,replay,complete)
        assert evidence['joint_completion_witness'] is joint
        assert evidence['normal_completion_witness'] is normal
        if not joint: assert evidence['evidence']=='replayed_resolved_rollout'
    assert not reference_evidence(best,dict(status='MATCH',whole_cycle=False),True)['joint_completion_witness']
    assert not reference_evidence(best,dict(status='ERROR'),True)['joint_completion_witness']
    metrics['not_yet_requested']=1
    assert not reference_evidence(best,replay,True)['joint_completion_witness']
