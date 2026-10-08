from types import SimpleNamespace

import pytest
import torch

from terminal_marl.training import p3_v25_formal as formal
from terminal_marl.training import p3_v25_ppo_block as block
from tests.test_p3_v25_ppo_block import fixture as pilot_fixture, complete


def admitted_fixture(tmp_path, monkeypatch):
    fit, source, actor, calls = pilot_fixture(tmp_path, monkeypatch)
    pilot_dir = tmp_path/'pilot'
    pilot = block.run(fit, 'q', 'warm', (), ('train0', 'train1', 'train2'), pilot_dir)
    assert pilot['status'] == 'PILOT_COMPLETE'
    monkeypatch.setattr(formal, 'input_review', block.input_review)
    monkeypatch.setattr(formal, 'frozen_models', block.frozen_models)
    monkeypatch.setattr(formal, 'load_train', lambda values: block.load_train(tuple(values)))
    monkeypatch.setattr(formal, 'collect_episode', block.collect_episode)
    manifest = dict(protocol=formal.PROTOCOL, profile=formal.PROFILE,
        formal_allowed=True, candidate_retained=True, review_reason='synthetic unit fixture only',
        fit_directory=str(fit), qualification_directory='q', warmup_directory='warm',
        development_directories=[], train_directories=['train0', 'train1', 'train2'],
        pilot_directory=str(pilot_dir), pilot_result_sha256=formal.sha(pilot_dir/'result.json'),
        candidate_checkpoint_sha256=pilot['training'][-1]['checkpoint_sha256'],
        initialization_seed=39181, campaign_wall_seconds=60,
        episodes=[dict(scenario=s, seed=897801+i, policy_seed=937801+i, update_seed=938801+i)
                  for i, s in enumerate(formal.SCENARIOS)])
    path = tmp_path/'manifest.json'
    formal.write(path, manifest)
    return path, manifest, source, actor, calls


def test_formal_continues_reviewed_model_and_optimizer_without_q_or_teacher(tmp_path, monkeypatch):
    path, manifest, source, actor, calls = admitted_fixture(tmp_path, monkeypatch)
    pilot_path = tmp_path/'pilot/after_03.pt'
    before = torch.load(pilot_path, weights_only=True)
    result = formal.run(path, tmp_path/'formal')
    assert result['status'] == 'TRAINING_COMPLETE' and len(result['episodes']) == 3
    assert result['completed_updates'] > 0 and result['initial_model_hash'] == block.digest_state(before['model'])
    assert len(calls) == 9 and [v['seed'] for v in calls[6:]] == [897801,897802,897803]
    after = torch.load(tmp_path/'formal/after_0003.pt', weights_only=True)
    assert all(torch.equal(v, after['model']['actor.'+k]) for k,v in actor.state_dict().items())
    for k, value in before['optimizer']['state'].items():
        assert after['optimizer']['state'][k]['step'] > value['step']
    assert formal.sha(pilot_path) == manifest['candidate_checkpoint_sha256']
    assert formal.sha(tmp_path/'fit/actor.pt') == source['sha256']
    protocol = formal.read(tmp_path/'formal/protocol.json')
    assert protocol['optimizer_continued_from_pilot'] and protocol['PPO_ALLOWED']
    assert not protocol['Q_observations_accessed'] and not protocol['teacher_enabled'] and not protocol['heartbeat_enabled']
    assert all(not cfg['task_priority_enabled'] and cfg['priority_wait_mode']=='total_wait_v1'
               and cfg['nominal_holder_separation_mode']=='swept_curve_v1' for cfg in protocol['physics'].values())
    digest = formal.sha(tmp_path/'formal/result.json')
    with pytest.raises(ValueError, match='already exists'):
        formal.run(path, tmp_path/'formal')
    assert formal.sha(tmp_path/'formal/result.json') == digest


@pytest.mark.parametrize('case', ['no_review','wrong_profile','changed_pilot','changed_checkpoint',
                                  'q_seed','duplicate_seed','wrong_scenario','bad_budget'])
def test_formal_rejects_unreviewed_or_mixed_evidence_before_optimizer(tmp_path, monkeypatch, case):
    path, manifest, _, _, _ = admitted_fixture(tmp_path, monkeypatch)
    if case == 'no_review': manifest['formal_allowed'] = False
    elif case == 'wrong_profile': manifest['profile'] = 'old'
    elif case == 'changed_pilot': manifest['pilot_result_sha256'] = '0'*64
    elif case == 'changed_checkpoint': manifest['candidate_checkpoint_sha256'] = '0'*64
    elif case == 'q_seed': manifest['episodes'][0]['seed'] = formal.Q_CASES[0][0]
    elif case == 'duplicate_seed': manifest['episodes'][1]['seed'] = manifest['episodes'][0]['seed']
    elif case == 'wrong_scenario': manifest['episodes'][0]['scenario'] = formal.SCENARIOS[1]
    elif case == 'bad_budget': manifest['campaign_wall_seconds'] = float('inf')
    # Direct review also validates non-finite values that JSON serialization rejects.
    monkeypatch.setattr(formal, 'TrajectorySetPPO', lambda *a,**k: pytest.fail('invalid manifest reached optimizer'))
    with pytest.raises(ValueError): formal.review(manifest)
    assert not (tmp_path/'formal').exists()


def test_formal_stops_before_update_after_failed_episode(tmp_path, monkeypatch):
    path, _, _, _, _ = admitted_fixture(tmp_path, monkeypatch)
    calls=[]
    def failed(*args, **kwargs):
        calls.append(kwargs['seed'])
        value = complete(6)
        value.update(status='CENSORED', terminal=False, censored=True, unresolved=1,
                     outcome_counts=dict(completed=5, failed_energy=1))
        return [], value
    monkeypatch.setattr(formal, 'collect_episode', failed)
    monkeypatch.setattr(formal.TrajectorySetPPO, 'update', lambda *a,**k: pytest.fail('failed rollout updated'))
    result = formal.run(path, tmp_path/'formal')
    assert result['status'] == 'BLOCKED' and calls == [897801]
    assert 'episode_1:failed_or_deferred_outcome' in result['blockers']
    assert not list((tmp_path/'formal').glob('after_*.pt'))


def test_formal_retains_checkpoint_on_update_timeout_and_never_extends_budget(tmp_path, monkeypatch):
    path, _, _, _, calls = admitted_fixture(tmp_path, monkeypatch)
    original = formal.TrajectorySetPPO.update
    def expired(self, steps, **kwargs):
        kwargs['wall_seconds'] = 1e-12
        return original(self, steps, **kwargs)
    monkeypatch.setattr(formal.TrajectorySetPPO, 'update', expired)
    result = formal.run(path, tmp_path/'formal')
    assert result['status'] == 'BLOCKED' and result['blockers'] == ['episode_1:update_wall_budget']
    assert len(calls) == 7 and (tmp_path/'formal/after_0001.pt').exists()


def test_formal_preserves_completed_rollout_if_update_throws(tmp_path, monkeypatch):
    path, _, _, _, _ = admitted_fixture(tmp_path, monkeypatch)
    def broken(*args, **kwargs): raise RuntimeError('synthetic optimizer exception')
    monkeypatch.setattr(formal.TrajectorySetPPO, 'update', broken)
    result = formal.run(path, tmp_path/'formal')
    assert result['status'] == 'ERROR' and 'synthetic optimizer exception' in result['error']
    assert len(result['episodes']) == 1 and result['episodes'][0]['rollout']['status'] == 'COMPLETED'


def test_formal_campaign_budget_checked_before_starting_next_episode(tmp_path, monkeypatch):
    path, manifest, _, _, calls = admitted_fixture(tmp_path, monkeypatch)
    manifest['campaign_wall_seconds'] = .5
    formal.write(path, manifest)
    values=iter([0.,1.,2.])
    monkeypatch.setattr(formal, 'time', SimpleNamespace(monotonic=lambda: next(values)))
    result = formal.run(path, tmp_path/'formal')
    assert result['status'] == 'BUDGET_EXHAUSTED' and result['episodes'] == [] and len(calls) == 6
