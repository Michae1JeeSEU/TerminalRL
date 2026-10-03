"""Event selection and real synthetic-environment replay; no training."""
import json
from pathlib import Path

import pytest
import torch

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import TemperatureCalibratedGateResidualAdapterActorCentralCriticV42
from terminal_marl.algo.service_context_runtime_v4_2 import ServiceContextInferencePolicyV42
from terminal_marl.configs.p3_integrated_repair import build_config
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.scripts import diagnose_p3_release_event_contract as event


def policy_and_env(release=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(891921)
        model = TemperatureCalibratedGateResidualAdapterActorCentralCriticV42(90)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.gate_head.bias[1 if release else 0] = 1000.
        model.proposal_head.bias[12] = 1000.  # Nominal speed and immediate timing.
    model.eval().requires_grad_(False)
    policy = ServiceContextInferencePolicyV42(model, 90, torch.device('cpu'))
    env = TerminalParallelEnv(build_config(requests=3, warm_start=3, high_soc=True), seed=891922)
    return policy, env


def test_event_selection_keeps_rejects_separate_and_only_takes_first():
    requested = {'0': 'o2m', '1': 'm2i'}
    assert event.event_labels(requested, {}, set(), set()) == ['first_request:m2i', 'first_request:o2m']
    assert event.event_labels(requested, {'1': 'm2i'}, {'m2i', 'o2m'}, set()) == ['first_execution:m2i']
    assert event.event_labels(requested, {'1': 'm2i'}, {'m2i', 'o2m'}, {'m2i'}) == []


def test_smoke_serializes_real_release_and_independent_replay(tmp_path):
    policy, env = policy_and_env()
    before_rng = torch.random.get_rng_state().clone()
    report = event.collect(policy, env, tmp_path, policy_seed=891923, max_steps=2,
                           wall_seconds=120, smoke=True)
    assert report['status'] == 'SMOKE_OK', report['error']
    assert report['ticks'] == 2
    assert report['policy_requests'] == 2
    assert report['request_counts']['RELEASE'] >= 3
    assert report['replay_ticks'] >= 1
    assert report['first_execution']  # Actually exercises new checked trajectory UUIDs.
    assert report['env_step_calls'] == report['ticks']+report['replay_ticks']
    assert torch.equal(before_rng, torch.random.get_rng_state())
    records = [json.loads(line) for line in (tmp_path/'decisions.jsonl').read_text().splitlines()]
    assert len(records) == 2
    for selected in report['events']:
        folder = tmp_path/selected['directory']
        assert (folder/'before.pkl').stat().st_size > 0
        assert json.loads((folder/'live.json').read_text()) == json.loads((folder/'replay.json').read_text())
        tape = json.loads((folder/'request_tape.json').read_text())
        for row in tape['rows']:
            if row['gate'] == 1:
                assert row['proposal'] == row['shortlist_proposals'][0]
                assert row['likelihood']['proposal'] is not None
    event.write_json(tmp_path/'result.json', report)
    assert json.loads((tmp_path/'result.json').read_text())['status'] == 'SMOKE_OK'


def test_no_release_is_coverage_missing_without_retrying_or_inventing_events(tmp_path):
    policy, env = policy_and_env(False)
    report = event.collect(policy, env, tmp_path, policy_seed=891923, max_steps=2, wall_seconds=120)
    assert report['status'] == 'COVERAGE_MISSING', report['error']
    assert report['stop_reason'] == 'tick_limit'
    assert report['ticks'] == report['env_step_calls'] == report['policy_requests'] == 2
    assert report['events'] == [] and report['replay_ticks'] == 0
    assert report['missing_execution'] == ['i2f', 'm2i', 'o2m']


def test_replay_difference_stops_and_preserves_both_records(tmp_path, monkeypatch):
    policy, env = policy_and_env()
    actual = event.execute_tape
    calls = 0
    def mismatched(copy, tape):
        nonlocal calls
        calls += 1
        result = actual(copy, tape)
        if calls == 2:
            result['outputs'][4]['team_reward'] += 1.0
        return result
    monkeypatch.setattr(event, 'execute_tape', mismatched)
    report = event.collect(policy, env, tmp_path, policy_seed=891923, max_steps=20, wall_seconds=120)
    assert report['status'] == 'ERROR'
    assert report['ticks'] == 1 and report['env_step_calls'] == 2
    assert 'replay differs' in report['error']['message']
    assert (tmp_path/'failure_request.json').exists()
    assert (tmp_path/'failure_state.pkl').exists()
    assert (tmp_path/'event_0001/live.json').exists()
    assert (tmp_path/'event_0001/replay.json').exists()


def test_coverage_stops_at_first_sufficient_tick_and_wall_limit_is_bounded(tmp_path):
    policy, env = policy_and_env()
    report = event.collect(policy, env, tmp_path, policy_seed=891923, max_steps=100, wall_seconds=0)
    assert report['status'] == 'COVERAGE_MISSING'
    assert report['stop_reason'] == 'wall_limit' and report['env_step_calls'] == 0


def test_wrong_checkpoint_reports_error_without_environment(tmp_path):
    checkpoint = tmp_path/'wrong.pt'
    checkpoint.write_bytes(b'not the fixed source')
    output = tmp_path/'output'
    output.mkdir()
    report = event.run(checkpoint, output)
    assert report['status'] == 'ERROR' and report['error']['message'] == 'wrong fixed checkpoint'
    assert not (output/'decisions.jsonl').exists()
    assert json.loads((output/'result.json').read_text()) == report
