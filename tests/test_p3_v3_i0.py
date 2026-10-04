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
