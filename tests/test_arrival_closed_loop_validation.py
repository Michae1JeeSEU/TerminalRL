import json
from collections import Counter

import pytest

from terminal_marl.configs import arrival_closed_loop_validation as protocol
from terminal_marl.configs.arrival_restructure import EXTERNAL_AUTHORITY, build_config
from terminal_marl.scripts import evaluate_arrival_closed_loop_validation as runner
from terminal_marl.scripts import evaluate_arrival_restructure as native

SHA = 'a' * 40


def test_frozen_matrix_and_authority(tmp_path):
    specs = protocol.build_specs('suite', tmp_path, SHA)
    assert len(specs) == 18
    assert Counter(s['mode'] for s in specs) == {'bridge': 3, 'replicate': 15}
    cohorts = {}
    for spec in specs:
        assert spec['control_authority'] == EXTERNAL_AUTHORITY
        assert spec['max_ticks'] == 2400 and spec['wall_seconds'] == 7200
        assert spec['profile'] == spec['protocol_version'] == 'arrival_closed_loop_validation_v1'
        if spec['mode'] == 'replicate':
            cohorts.setdefault((spec['scenario'], spec['seed']), []).append(spec['controller'])
    assert set(cohorts) == set(protocol.REPLICATE_COHORTS)
    assert all(tuple(cs) == protocol.CONTROLLERS for cs in cohorts.values())
    assert len({s['output'] for s in specs}) == 18
    smoke = protocol.build_specs('smoke', tmp_path, SHA)
    assert len(smoke) == 3 and all(s['max_ticks'] == 2 for s in smoke)
    assert {s['seed'] for s in smoke}.isdisjoint(s['seed'] for s in specs)
    assert protocol.SEARCH_SETTINGS == native.SEARCH == dict(lookahead_steps=4, max_orders=4, replan_interval=10, planning_wall_seconds=60.)
    for scenario in protocol.SMOKE_SEEDS:
        assert build_config(scenario).max_simulation_seconds == build_config(scenario, authority=EXTERNAL_AUTHORITY).max_simulation_seconds


def test_fresh_output_and_input_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'execute', lambda *_: pytest.fail('must not execute'))
    with pytest.raises(FileExistsError):
        runner.run_validation('suite', tmp_path, SHA, 1)
    for sha, workers in [('bad', 1), (SHA, 0), (SHA, 19)]:
        with pytest.raises(ValueError):
            runner.run_validation('suite', tmp_path/'fresh', sha, workers)
    monkeypatch.setattr(runner, 'SEARCH', {})
    with pytest.raises(ValueError, match='SEARCH'):
        runner.run_validation('suite', tmp_path/'fresh', SHA, 1)


@pytest.mark.parametrize('status,expected', [('COMPLETED', 0), ('CENSORED', 1), ('ERROR', 1)])
def test_mock_summary_and_exit(tmp_path, monkeypatch, status, expected):
    root = tmp_path/'run'
    def execute(specs, workers):
        manifest = json.loads((root/'manifest.json').read_text())
        assert len(manifest['specs']) == 18
        assert json.loads((root/'status.json').read_text())['status'] == 'RUNNING'
        return [dict(status=status, output=s['output'], metrics={'distinct_arm': i}) for i,s in enumerate(specs)]
    monkeypatch.setattr(runner, 'execute', execute)
    code = runner.main(['suite', '--output', str(root), '--source-commit', SHA, '--workers', '18'])
    assert code == expected
    summary = json.loads((root/'summary.json').read_text())
    assert summary['status'] == status
    assert len(summary['runs']) == 18 and summary['runs'][-1]['metrics']['distinct_arm'] == 17
    assert summary['pid'] > 0 and summary['elapsed_seconds'] >= 0


@pytest.mark.parametrize('replay_status,steps,expected', [('MATCH', 2, 0), ('ERROR', 0, 1), ('MATCH', 1, 1)])
def test_mock_smoke_replay(tmp_path, monkeypatch, replay_status, steps, expected):
    monkeypatch.setattr(runner, 'execute', lambda specs, workers: [dict(status='CENSORED', stop_reason='tick_cap', ticks=2, integrity_ok=True, output=s['output']) for s in specs])
    monkeypatch.setattr(runner, 'replay_run', lambda source, output: dict(status=replay_status, matched_steps=steps))
    result = runner.run_validation('smoke', tmp_path/'smoke', SHA, 3)
    assert result['exit_code'] == expected
    assert len(result['replays']) == 3
    assert result['status'] != 'COMPLETED'


def test_execute_failure_recorded(tmp_path, monkeypatch):
    def fail(*_):
        raise RuntimeError('mock failure')
    monkeypatch.setattr(runner, 'execute', fail)
    result = runner.run_validation('bridge', tmp_path/'run', SHA, 1)
    assert result['status'] == 'ERROR' and result['exit_code'] == 1
    assert 'mock failure' in result['error']


def test_reuses_native_execute_and_run_arm(tmp_path, monkeypatch):
    assert runner.execute is native.execute
    assert runner.replay_run is native.replay_run
    specs = protocol.build_specs('bridge', tmp_path, SHA)
    calls = []
    monkeypatch.setattr(native, 'run_arm', lambda spec: calls.append(spec) or {'status': 'COMPLETED'})
    assert len(runner.execute(specs, 1)) == 3
    assert calls == specs


def test_pool_failure_preserves_written_results(tmp_path, monkeypatch):
    def partial(specs, workers):
        # Use the exact registered output, including platform-native separators.
        from pathlib import Path
        path = Path(specs[0]['output'])
        path.mkdir()
        runner.write_json(path/'result.json', dict(output=str(path), status='COMPLETED', metrics={'retained': True}))
        raise RuntimeError('mock pool failure')
    monkeypatch.setattr(runner, 'execute', partial)
    result = runner.run_validation('bridge', tmp_path/'run', SHA, 1)
    assert result['status'] == 'ERROR' and result['exit_code'] == 1
    assert len(result['runs']) == 1 and result['runs'][0]['metrics']['retained']
