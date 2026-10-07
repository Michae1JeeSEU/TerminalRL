"""All-actor collection contracts; formal seeds are never reset in tests."""
import json

import pytest

from terminal_marl.configs.p3_integrated_repair import build_config
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.scripts import collect_p3_v3_i0 as i0
from terminal_marl.scripts import collect_p3_v3_sparse_expert as sparse
from terminal_marl.training.p3_v3_expert_dataset import (
    DEVELOPMENT_SCHEMA, DEVELOPMENT_PARTITION, inspect_sparse_episode,
)


def environment():
    return TerminalParallelEnv(build_config(requests=3, warm_start=3, high_soc=True), seed=892901)


def test_frozen_case_order_and_partition_bounds():
    cases = [i0.case_definition('train', index) for index in range(1, 25)]
    assert [case['seed'] for case in cases] == list(range(893001, 893025))
    assert [(case['requests'], case['warm_start'], case['high_soc']) for case in cases] == list(i0.SCENARIOS)*8
    assert [i0.case_definition('audit', index)['seed'] for index in range(1, 7)] == list(range(894001, 894007))
    for partition, index in [('Q', 1), ('train', 0), ('train', 25), ('audit', 7)]:
        with pytest.raises(ValueError):
            i0.case_definition(partition, index)


def test_all_actor_keeps_every_uid_and_joins_three_independent_labels(tmp_path, monkeypatch):
    output = tmp_path / 'case'
    output.mkdir()
    sparse.write_json(output/'protocol.json', dict(schema=DEVELOPMENT_SCHEMA,
                      partition=DEVELOPMENT_PARTITION, seed=892901))
    episode = output/'seed_892901'
    def forbidden(*args, **kwargs):
        raise AssertionError('rollout must not query the teacher')
    with monkeypatch.context() as context:
        context.setattr(sparse, 'teacher_joint_expert_v4_1', forbidden)
        flow, selected = sparse.rollout(environment(), episode, seed=892901,
            max_steps=2, wall_seconds=120, smoke=True, all_actor=True)
    assert flow['status'] == 'SMOKE_OK', flow['error']
    assert flow['rows'] == 6 and flow['row_counts']['actor'] == 3
    assert flow['selection_overflow'] == 0 and flow['selection_mode'] == 'all_actor'
    assert [entry['row']['uid'] for entry in selected] == ['0', '1', '2']
    assert [entry['window'] for entry in selected] == [0, 1, 2]
    assert len({entry['snapshot'] for entry in selected}) == 1
    assert len({entry['snapshot_sha256'] for entry in selected}) == 1
    before = (episode/'rows.jsonl').read_bytes()
    annotation = sparse.annotate(episode, selected, wall_seconds=120)
    assert annotation['status'] == 'COMPLETED' and annotation['labelled'] == 3
    summary = inspect_sparse_episode(episode, expected_seed=892901)
    assert summary['gate_supervision_rows'] == summary['actor_eligible_rows'] == 3
    assert summary['ignored_forced_labels'] == 0
    assert (episode/'rows.jsonl').read_bytes() == before
    coverage = i0.coverage_report(episode)
    assert sum(coverage['contexts']['actor'][f'interface_{key}'] for key in ('o2m', 'm2i', 'i2f')) == 3
    assert i0.rollout_blockers(flow, smoke=True) == []


def test_query_cap_does_not_stop_environment_or_drop_raw_rows(tmp_path):
    flow, selected = sparse.rollout(environment(), tmp_path/'episode', seed=892901,
        max_steps=2, wall_seconds=120, smoke=True, all_actor=True, max_selected=1)
    assert flow['ticks'] == 2 and flow['rows'] == 6
    assert len(selected) == 1 and flow['selection_overflow'] == 2
    assert 'actor_query_budget_exceeded' in i0.rollout_blockers(flow, smoke=True)
    assert len((tmp_path/'episode/rows.jsonl').read_text().splitlines()) == 6


def completed_flow():
    return dict(status='COMPLETED', error=None, first_tick_replay_equal=True,
        telemetry={key: 0 for key in sparse.SAFETY_KEYS}, selection_overflow=0,
        terminal=True, censored=False, unresolved=0, active=0, queue=0, service=0,
        requested=12, outcome_counts={'completed': 10, 'emergency_landed': 2})


def test_business_blocks_failed_deferred_unresolved_and_safety_but_allows_emergency():
    flow = completed_flow()
    assert i0.rollout_blockers(flow) == []
    for outcome in ('failed_energy', 'deferred'):
        changed = dict(flow, outcome_counts={'completed': 11, outcome: 1})
        assert 'failed_deferred_or_missing_outcome' in i0.rollout_blockers(changed)
    assert 'unresolved_cycle' in i0.rollout_blockers(dict(flow, unresolved=1))
    flow['telemetry']['checked_executed_plan_hash_mismatches'] = 1
    assert 'actual_safety_or_hash' in i0.rollout_blockers(flow)


def test_protocol_written_before_reset_and_constructor_failure_is_retained(tmp_path, monkeypatch):
    output = tmp_path/'smoke'
    def forbidden(config, *, seed):
        protocol = json.loads((output/'protocol.json').read_text())
        assert protocol['seed'] == seed == i0.SMOKE_SEED
        assert protocol['selection'].startswith('every actor_mask=true')
        raise RuntimeError('injected constructor failure; no reset')
    monkeypatch.setattr(i0, 'TerminalParallelEnv', forbidden)
    result = i0.run(output, smoke=True)
    assert result['status'] == 'ERROR'
    assert result['error']['message'] == 'injected constructor failure; no reset'
    assert (output/'result.json').exists()
    with pytest.raises(FileExistsError):
        i0.run(output, smoke=True)


def test_rollout_blocker_prevents_annotation(tmp_path, monkeypatch):
    flow = completed_flow()
    flow.update(status='SMOKE_OK', selection_overflow=1)
    monkeypatch.setattr(i0, 'TerminalParallelEnv', lambda *args, **kwargs: object())
    monkeypatch.setattr(sparse, 'rollout', lambda *args, **kwargs: (flow, []))
    def forbidden(*args, **kwargs):
        raise AssertionError('blocked rollout must not start annotation')
    monkeypatch.setattr(sparse, 'annotate', forbidden)
    result = i0.run(tmp_path/'case', smoke=True)
    assert result['status'] == 'INCOMPLETE'
    assert result['annotation']['status'] == 'NOT_RUN'
    assert result['blockers'] == ['actor_query_budget_exceeded']


def test_v2_uses_fresh_dense_cases_without_reinterpreting_v1():
    assert i0.protocol_settings('v1')['max_selected'] == 256
    assert i0.protocol_settings('v2') == dict(protocol=i0.PROTOCOL_V2,
        max_selected=2048, smoke_seed=892601)
    assert [i0.case_definition('train', n, protocol_version='v2') for n in (1, 2)] == [
        dict(partition='train', index=1, seed=893101, requests=24, warm_start=12, high_soc=True),
        dict(partition='train', index=2, seed=893102, requests=24, warm_start=12, high_soc=False)]
    for partition, index in [('audit', 1), ('Q', 1), ('train', 0), ('train', 3)]:
        with pytest.raises(ValueError):
            i0.case_definition(partition, index, protocol_version='v2')
    with pytest.raises(ValueError, match='unknown I0 protocol'):
        i0.protocol_settings('unknown')


def test_v2_forwards_larger_count_but_keeps_wall_budget_and_stop_rule(tmp_path, monkeypatch):
    # Constructor is replaced: no formal seed is reset by this contract test.
    def constructor(config, *, seed):
        protocol = json.loads((tmp_path/'case/protocol.json').read_text())
        assert seed == protocol['seed'] == 893101
        assert protocol['protocol'] == i0.PROTOCOL_V2
        assert protocol['max_selected'] == 2048 and protocol['max_release_intents'] == 51200
        assert protocol['annotation_wall_seconds'] == 7200
        return object()
    def rollout(env, output, **kwargs):
        assert kwargs['max_selected'] == 2048 and kwargs['all_actor']
        assert kwargs['max_steps'] == 1800 and kwargs['wall_seconds'] == 3600
        return dict(completed_flow(), selection_overflow=1), []
    def forbidden(*args, **kwargs):
        raise AssertionError('overflow still prevents annotation in V2')
    monkeypatch.setattr(i0, 'TerminalParallelEnv', constructor)
    monkeypatch.setattr(sparse, 'rollout', rollout)
    monkeypatch.setattr(sparse, 'annotate', forbidden)
    result = i0.run(tmp_path/'case', protocol_version='v2')
    assert result['status'] == 'INCOMPLETE'
    assert result['blockers'] == ['actor_query_budget_exceeded']
    assert result['annotation']['status'] == 'NOT_RUN'


def test_v3_is_one_fresh_mixed_case_with_unchanged_v2_limits():
    assert i0.protocol_settings('v3') == dict(protocol=i0.PROTOCOL_V3,
        max_selected=2048, smoke_seed=892701)
    assert i0.case_definition('train', 1, protocol_version='v3') == dict(
        partition='train', index=1, seed=893201, requests=24, warm_start=12, high_soc=False)
    for partition, index in [('audit', 1), ('Q', 1), ('train', 0), ('train', 2), ('train', True)]:
        with pytest.raises(ValueError):
            i0.case_definition(partition, index, protocol_version='v3')


@pytest.mark.parametrize('smoke,seed,high,steps,wall', [
    (False, 893201, False, 1800, 3600), (True, 892701, True, 2, 180)])
def test_v3_protocol_precedes_reset_and_preserves_failure_stop(
        tmp_path, monkeypatch, smoke, seed, high, steps, wall):
    # No formal reset: inspect frozen metadata at the mocked constructor boundary.
    output = tmp_path/'v3'
    def constructor(config, *, seed: int):
        protocol = json.loads((output/'protocol.json').read_text())
        assert protocol['seed'] == seed
        assert protocol['protocol'] == i0.PROTOCOL_V3
        assert protocol['high_soc'] is high
        assert protocol['max_selected'] == 2048
        assert protocol['max_release_intents'] == 51200
        assert protocol['max_steps'] == steps
        assert protocol['rollout_wall_seconds'] == wall
        assert protocol['annotation_wall_seconds'] == (180 if smoke else 7200)
        return object()
    def rollout(env, episode, **kwargs):
        assert kwargs['seed'] == seed
        assert kwargs['max_selected'] == 2048 and kwargs['all_actor']
        assert kwargs['max_steps'] == steps and kwargs['wall_seconds'] == wall
        return dict(completed_flow(), status='SMOKE_OK' if smoke else 'COMPLETED',
                    selection_overflow=1), []
    def forbidden(*args, **kwargs):
        raise AssertionError('incomplete selection must not start annotation')
    monkeypatch.setattr(i0, 'TerminalParallelEnv', constructor)
    monkeypatch.setattr(sparse, 'rollout', rollout)
    monkeypatch.setattr(sparse, 'annotate', forbidden)
    result = i0.run(output, smoke=smoke, protocol_version='v3')
    assert result['seed'] == seed and result['status'] == 'INCOMPLETE'
    assert result['blockers'] == ['actor_query_budget_exceeded']
    assert result['annotation']['status'] == 'NOT_RUN'


def test_all_actor_serializes_queries_past_old_256_boundary(tmp_path, monkeypatch):
    # Hold a known development fixture still to generate 300 distinct tick/UID
    # records cheaply. This tests storage/selection, not a business trajectory.
    def no_advance(env, tape):
        return dict(outputs=[None, None, None, None, {'action_traces': {}}],
                    after=sparse.canonical(sparse.state_record(env)))
    monkeypatch.setattr(sparse, 'execute_tape', no_advance)
    flow, selected = sparse.rollout(environment(), tmp_path/'episode', seed=892901,
        max_steps=100, wall_seconds=120, smoke=True, all_actor=True,
        max_selected=i0.protocol_settings('v2')['max_selected'])
    assert flow['rows'] == flow['row_counts']['actor'] == len(selected) == 300
    assert flow['selection_overflow'] == 0
    assert [e['window'] for e in selected] == list(range(300))
    assert len({e['row']['decision_id'] for e in selected}) == 300
    assert selected[256]['row']['tick'] == 85 and selected[256]['row']['uid'] == '1'
    entry = tmp_path/'episode/snapshots/window_256.json'
    assert json.loads(entry.read_text()) == selected[256]
    label = sparse.query_snapshot(entry)
    assert label['status'] == 'OK' and label['decision_id'] == selected[256]['row']['decision_id']
    assert label['window'] == 256 and label['actor_mask']
    assert all(e['row']['teacher_label'] is None for e in selected)
