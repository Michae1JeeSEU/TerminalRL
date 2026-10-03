"""Synthetic contract/output checks before consuming the three registered states."""
from copy import deepcopy
import json
import pickle

import numpy as np
import pytest
import torch

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import TemperatureCalibratedGateResidualAdapterActorCentralCriticV42
from terminal_marl.algo.service_context_runtime_v4_2 import ServiceContextInferencePolicyV42
from terminal_marl.configs.p3_integrated_repair import (
    _legacy_source_config, build_config, initialize_legacy_source, load_legacy_source,
)
from terminal_marl.core.checkpoints import IncompatibleCheckpointError, checkpoint_metadata
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.scripts import diagnose_p3_direct_contract as diagnostic
from terminal_marl.training.student_rollout_v4_2 import _student_request_details


@pytest.fixture
def checkpoint(tmp_path):
    config = build_config(requests=3, warm_start=3, high_soc=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(891800)
        model = TemperatureCalibratedGateResidualAdapterActorCentralCriticV42(88)
    path = tmp_path / 'synthetic.pt'
    torch.save(dict(model=model.state_dict(), metadata=checkpoint_metadata(_legacy_source_config(config))), path)
    return config, path, model


def test_initializer_creates_no_optimizer_and_preserves_exact_weights(checkpoint, monkeypatch):
    config, path, reference = checkpoint
    def forbidden(*args, **kwargs):
        raise AssertionError('optimizer construction is forbidden')
    monkeypatch.setattr(torch.optim.Optimizer, '__init__', forbidden)
    rng_before = torch.random.get_rng_state().clone()
    source, _ = load_legacy_source(path, config)
    target, metadata = initialize_legacy_source(path, config)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert not hasattr(source, 'optimizer') and not hasattr(target, 'optimizer')
    assert not any(p.requires_grad for p in target.model.parameters())
    for name, tensor in reference.state_dict().items():
        assert torch.equal(tensor, source.model.state_dict()[name])
    assert metadata['optimizer_created'] is False
    observations = np.linspace(-1, 1, 360, dtype=np.float32).reshape(4, 90)
    assert diagnostic.check_migration(source, target, observations)['copied_tensors_exact']


@pytest.mark.parametrize('mutation', ['schema', 'temperature', 'missing_tensor'])
def test_model_only_loader_retains_strict_checks(checkpoint, mutation):
    config, path, _ = checkpoint
    payload = torch.load(path, weights_only=False)
    if mutation == 'schema':
        payload['metadata']['observation_dimension'] = 90
    elif mutation == 'temperature':
        payload['model']['gate_temperature'] = torch.tensor(1.0)
    else:
        payload['model'].pop('proposal_head.weight')
    torch.save(payload, path)
    with pytest.raises((ValueError, RuntimeError, IncompatibleCheckpointError)):
        load_legacy_source(path, config)


@pytest.fixture
def requests():
    env = TerminalParallelEnv(build_config(requests=3, warm_start=3, high_soc=True), seed=891801)
    # Lexical order is deliberately different from numeric order.
    uav = env.active.pop(1)
    uav.uid = 10
    env.active[10] = uav
    env.active[0].safety_retry_after = env.now + 2
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(891802)
        model = TemperatureCalibratedGateResidualAdapterActorCentralCriticV42(90)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.gate_head.bias.copy_(torch.tensor([-100., 100.]))
    policy = ServiceContextInferencePolicyV42(model.eval(), 90, torch.device('cpu'))
    obs = env._observations()
    details = _student_request_details(policy, env, obs, deterministic=True, shortlist_k=3)
    return policy, env, obs, details


def test_request_binding_roundtrip_and_two_independent_steps(requests, tmp_path):
    policy, env, obs, details = requests
    tape = diagnostic.check_requests(policy, env, obs, details)
    assert tape['uids'] == ['0', '10', '2']
    assert tape['counts'] == {'forced_WAIT': 1, 'RELEASE': 2}
    assert tape['rows'][0]['proposal'] is None
    assert tape['rows'][0]['likelihood'] == dict(gate=0., proposal=None, joint=0.)
    assert all(row['shortlist_proposals'] == [0, 1, 2] for row in tape['rows'][1:])
    path = tmp_path / 'request.json'
    diagnostic.write_json(path, tape)
    restored = json.loads(path.read_text())
    assert 'NaN' not in path.read_text()
    data = pickle.dumps(env)
    first = diagnostic.execute_tape(pickle.loads(data), restored)
    second = diagnostic.execute_tape(pickle.loads(data), restored)
    assert first == second
    diagnostic.write_json(tmp_path / 'replay.json', first)
    assert json.loads((tmp_path / 'replay.json').read_text()) == first


@pytest.mark.parametrize('corrupt', ['uid', 'mask', 'likelihood', 'conditional', 'rank1', 'wait_sentinel'])
def test_request_validator_detects_misbindings(requests, corrupt):
    policy, env, obs, original = requests
    details = list(deepcopy(original))
    if corrupt == 'uid':
        details[0].reverse()
    elif corrupt == 'mask':
        details[2][0] = True
    elif corrupt == 'likelihood':
        details[8]['10']['joint'] += .1
    elif corrupt == 'conditional':
        details[8]['10']['proposal'] = None
    elif corrupt == 'rank1':
        details[9]['10'] = tuple(reversed(details[9]['10']))
    else:
        details[5][0] = 0
    with pytest.raises(ValueError):
        diagnostic.check_requests(policy, env, obs, details)


def test_unregistered_input_writes_error_without_loading_pickle(tmp_path, monkeypatch):
    root = tmp_path / 'input' / 'A_source'
    root.mkdir(parents=True)
    (root / 'snapshot_400.pkl').write_bytes(b'not a trusted pickle')
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'not a trusted checkpoint')
    output = tmp_path / 'output'
    output.mkdir()
    def forbidden(*args, **kwargs):
        raise AssertionError('untrusted pickle must not be loaded')
    monkeypatch.setattr(diagnostic.pickle, 'loads', forbidden)
    result = diagnostic.run('A', root.parent, checkpoint, output)
    assert result['status'] == 'ERROR'
    assert result['env_step_calls'] == 0
    assert result['error']['message'] == 'unregistered snapshot hash'
    assert json.loads((output / 'result.json').read_text()) == result
