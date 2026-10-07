import json

import pytest
import torch

from terminal_marl.algo.trajectory_set_actor import TrajectorySetActor
from terminal_marl.core.trajectory_context_observation import VERSION
from terminal_marl.training import p3_v25_task_student as student
from terminal_marl.training.p3_v25_imitation_qualification import sha


def complete():
    return dict(status='COMPLETED',terminal=True,censored=False,error=None,replay_exact=True,
        telemetry={k:0 for k in student.SAFETY_KEYS},requested=6,unresolved=0,active=0,queue=0,service=0,
        counts=dict(actor=30),outcome_counts=dict(completed=5,emergency_landed=1))


def test_complete_emergency_is_reported_without_inventing_zero_emergency_gate():
    result=complete();assert student.blockers(result)==[]
    result['outcome_counts']=dict(completed=5,failed_energy=1)
    assert 'failed_or_deferred_outcome' in student.blockers(result)
    result=complete();result['telemetry']['checked_executed_plan_hash_mismatches']=1
    assert 'actual_safety_or_hash' in student.blockers(result)
    result=complete();result['shield_actor_fraction']=.051
    assert 'actor_takeover_fraction' in student.blockers(result)
    result=complete();result.update(censored=True,terminal=False)
    assert 'cycle_incomplete' in student.blockers(result)


def test_student_wrapper_uses_actual_task_canonical_source_without_optimizer(tmp_path,monkeypatch):
    torch.set_num_threads(1);torch.manual_seed(31)
    fit=tmp_path/'fit';fit.mkdir();path=fit/'actor.pt'
    actor=TrajectorySetActor()
    with torch.no_grad():actor.gate_head.weight.zero_();actor.gate_head.bias.copy_(torch.tensor([-20.,20.]))
    torch.save(dict(model=actor.state_dict(),observation_schema=VERSION,arm='new_information'),path)
    manifest=dict(checkpoints=[dict(seed=student.SEEDS[0],path='actor.pt',sha256=sha(path))])
    monkeypatch.setattr(student,'frozen_models',lambda _:manifest)
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k:pytest.fail('inference constructed optimizer'))
    report=student.run(fit,tmp_path/'run',scenario='dense24_mixed',seed=892909,policy_seed=392,
                       smoke=True,deterministic=False)
    assert report['status']=='SMOKE_OK' and report['rollout']['replay_exact']
    assert report['parameter_updates']==0 and not report['PPO_ALLOWED']
    config=json.loads((tmp_path/'run/episode/protocol.json').read_text())['physics_config']
    assert not config['task_priority_enabled'] and config['release_arbitration_mode']=='downstream_soc_priority'
    assert config['low_level_priority_time_weight']==0.
    assert config['merge_gap_prediction_mode']=='committed_arrivals_v1' and config['initial_soc_min']==.55
    assert sha(path)==manifest['checkpoints'][0]['sha256']
    assert report['rollout']['counts']['RELEASE']==3
