import copy
import pytest
import torch
import yaml
from terminal_marl.training import run_v4_2_ppo_block2 as runner
from terminal_marl.core.checkpoints import save_checkpoint
from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42


def test_continuation_requires_matching_successful_stage():
    p = yaml.safe_load(runner.PROTOCOL.read_text(encoding='utf-8-sig'))
    report = dict(stage='entry', PASS=True, gates={'passed': True}, source_hashes=runner.source_hashes(),
        protocol_sha256=runner._sha(runner.PROTOCOL), checkpoint_sha256=p['initial_checkpoint_sha256'])
    runner.validate_gate(report, 'entry', p)
    for changed in ({'stage':'smoke'}, {'PASS':False}, {'gates':{}}, {'source_hashes':{}}, {'checkpoint_sha256':'wrong'}):
        with pytest.raises((PermissionError, ValueError)):
            runner.validate_gate({**report, **changed}, 'entry', p)


def test_restore_preserves_optimizer_and_rejects_entry_weights(tmp_path):
    policy = StudentPPOV42(88, learning_rate=.00003)
    policy.optimizer.zero_grad()
    sum(x.sum() for x in policy.model.parameters()).backward()
    policy.optimizer.step()
    config = runner.scenario_config('warm3', runner.i3_protocol())
    path = tmp_path/'checkpoint.pt'
    save_checkpoint(path, policy, config, formal=True, stage='V4_2_V2_1_PPO')
    p = dict(initial_checkpoint_sha256=runner._sha(path), learning_rate=.00003)
    restored = runner.restore(path, p)
    assert runner.state_equal(policy.optimizer.state_dict(), restored.optimizer.state_dict())
    assert runner.state_equal(policy.model.state_dict(), restored.model.state_dict())
    altered = copy.deepcopy(restored.optimizer.state_dict())
    next(iter(altered['state'].values()))['step'] += 1
    assert not runner.state_equal(policy.optimizer.state_dict(), altered)
    save_checkpoint(path, policy, config, formal=False, stage='V4_2_PPO_BLOCK2_ENTRY')
    p['initial_checkpoint_sha256'] = runner._sha(path)
    with pytest.raises(ValueError, match='formal checkpoint'):
        runner.restore(path, p)
