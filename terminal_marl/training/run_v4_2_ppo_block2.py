"""Checkpoint12 continuation after geometry requalification; never reuse entry weights."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42
from terminal_marl.core.checkpoints import load_checkpoint, save_checkpoint
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import _sha
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r3 import _write
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_student_qualification import protocol as i3_protocol, scenario_config, episode_gates
from terminal_marl.training.run_v4_2_v2_1_ppo import requested_logprob_error
from terminal_marl.training.run_v4_2_i2_dagger_preflight import _policy_tape_logprob_exact
from terminal_marl.training.student_rollout_v4_2 import rollout

PROTOCOL = Path('terminal_marl/configs/low_v4_2/ppo_block2_transfer_repair.yaml')
PROTOCOL_SHA = 'e6ec939f19c50f9da1471e5a025f5935211e61110ede3010fb9ab029e39c9ba6'


def source_hashes():
    names = ['core/trajectory_generator.py', 'core/trajectory_time_parameterization.py',
             'core/checkpoints.py', 'envs/parallel_env.py', 'algo/student_ppo_v4_2.py',
             'algo/joint_ppo_v4_2.py', 'training/student_rollout_v4_2.py',
             'training/run_v4_2_v2_1_student_qualification.py', 'training/run_v4_2_v2_1_ppo.py',
             'training/run_v4_2_ppo_block2.py']
    return {n: _sha(Path('terminal_marl') / n) for n in names}


def state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return a == b


def restore(checkpoint, p):
    if _sha(checkpoint) != p['initial_checkpoint_sha256']:
        raise ValueError('checkpoint12 mismatch')
    policy = StudentPPOV42(88, learning_rate=p['learning_rate'])
    payload = load_checkpoint(checkpoint, policy, scenario_config('warm3', i3_protocol()), load_optimizer=True)
    if payload.get('stage') != 'V4_2_V2_1_PPO' or not payload.get('formal') or not payload.get('optimizer', {}).get('state'):
        raise ValueError('formal checkpoint with initialized optimizer required')
    if not state_equal(payload['model'], policy.model.state_dict()) or not state_equal(payload['optimizer'], policy.optimizer.state_dict()):
        raise ValueError('model/optimizer restore mismatch')
    if any(g['lr'] != p['learning_rate'] for g in policy.optimizer.param_groups):
        raise ValueError('checkpoint learning rate differs from continuation protocol')
    return policy


def validate_gate(report, stage, p):
    if report.get('stage') != stage or not report.get('PASS') or not report.get('gates') or not all(report['gates'].values()):
        raise PermissionError('prior stage has not passed')
    if report.get('source_hashes') != source_hashes() or report.get('protocol_sha256') != _sha(PROTOCOL) or report.get('checkpoint_sha256') != p['initial_checkpoint_sha256']:
        raise ValueError('qualification source/protocol/checkpoint changed')


def replay_geometry(config, tape):
    env = TerminalParallelEnv(config)
    metrics = dict(samples=0, reverse_samples=0, max_inner_excursion_m=0.)
    def observe(world):
        for u in world.active.values():
            plan = u.committed_plan
            if plan is None or plan.mode != 'ring_transfer':
                continue
            xy, v = u.position[:2], u.velocity[:2]
            r = float(np.linalg.norm(xy))
            metrics['samples'] += 1
            metrics['reverse_samples'] += int((xy[0]*v[1]-xy[1]*v[0])/max(r, 1e-9) < -.001)
            metrics['max_inner_excursion_m'] = max(metrics['max_inner_excursion_m'], float(np.linalg.norm(plan.points[-1].position[:2]))-r)
    env.set_step_observer(observe)
    replay = SafetyTraceRecorder.replay_action_tape(env, tape)
    return replay['exact'], metrics


def execute(stage, output, checkpoint, history, p, prior=None):
    if stage in ('entry', 'train'):
        validate_gate(prior, 'qualify' if stage == 'entry' else 'entry', p)
    policy = restore(checkpoint, p)  # Entry updates are deliberately discarded before formal training.
    seeds = (p['qualification_seeds'] if stage == 'qualify' else p['entry_seeds'] if stage == 'entry'
             else [p['smoke_seed']] if stage == 'smoke' else list(range(p['seed_start'], p['seed_start']+p['episodes'])))
    output.mkdir(parents=True, exist_ok=False)
    if set(seeds) & historical_registry(history, output)['seeds']:
        raise ValueError('seeds previously consumed')
    _write(output/'pre_reset_registry.json', dict(seeds=seeds, stage=stage, protocol_sha256=_sha(PROTOCOL)))
    np.random.seed(seeds[0])
    records = []
    for index, seed in enumerate(seeds):
        name = p['entry_scenarios'][index] if stage == 'entry' else p['scenarios'][index % 3]
        config = scenario_config(name, i3_protocol())
        horizon = p['smoke_steps'] if stage == 'smoke' else p['entry_steps'][index] if stage == 'entry' else int(p['horizon_seconds']/config.dt)
        steps, summary = rollout(policy, config, seed=seed, max_steps=horizon, output=output/f'seed{seed}',
                                 deterministic=stage == 'qualify' and index < 3, collect_labels=False)
        error = requested_logprob_error(policy, steps)
        summary['replay_exact'], geometry = replay_geometry(config, summary['action_tape'])
        summary['requested_logprob_exact'] = _policy_tape_logprob_exact(policy, config, summary['action_tape'])
        gates = dict(safety=not any(summary['safety'].values()), replay=summary['replay_exact'],
            requested_logprob=summary['requested_logprob_exact'], initial_ratio=error <= 1e-5,
            student_only=summary['label_rows'] == 0 and all(not s.teacher_used for s in steps),
            actor_decisions=summary['counts']['actor_decisions'] > 0,
            geometry=geometry['reverse_samples'] == 0 and geometry['max_inner_excursion_m'] <= .05)
        if stage in ('qualify', 'train'):
            gates.update(episode_gates(summary, config, i3_protocol()))
            gates['geometry_observed'] = geometry['samples'] > 0
        update = None
        if stage != 'qualify' and all(gates.values()):
            gates['stochastic'] = all(s.stochastic_behavior for s in steps)
            if gates['stochastic']:
                keys = ('minibatch_size','gamma','gae_lambda','clip_ratio','value_coef','entropy_coef')
                update = policy.update(steps, training_authorized=True, epochs=p['update_epochs'], **{k:p[k] for k in keys})
                gates['finite_update'] = all(np.isfinite(v) for v in update.values()) and all(torch.isfinite(v).all() for v in policy.model.parameters())
                gates['bounded_KL'] = update['approx_kl'] <= p['maximum_approx_kl']
                saved = output/f'checkpoint_{index+1}.pt'
                save_checkpoint(saved, policy, config, stage='V4_2_V2_1_PPO' if stage == 'train' else 'V4_2_PPO_BLOCK2_ENTRY',
                    formal=stage == 'train', ppo_called=True, protocol_sha256=_sha(PROTOCOL), source_hashes=source_hashes(),
                    initial_checkpoint_sha256=p['initial_checkpoint_sha256'])
                restored = StudentPPOV42(88, learning_rate=p['learning_rate'])
                load_checkpoint(saved, restored, config, load_optimizer=True)
                gates['restore'] = state_equal(policy.model.state_dict(), restored.model.state_dict()) and state_equal(policy.optimizer.state_dict(), restored.optimizer.state_dict())
                policy = restored
        records.append(dict(seed=seed, scenario=name, summary=summary, geometry=geometry, initial_logprob_error=error, gates=gates, update=update))
        _write(output/'progress.json', dict(stage=stage, completed=len(records), total=len(seeds), records=records))
        if not all(gates.values()):
            break
    gates = dict(all_episodes=len(records) == len(seeds), all_episode_gates=all(all(r['gates'].values()) for r in records))
    if stage == 'qualify':
        gates['both_gates'] = all(sum(r['summary']['counts'][k] for r in records) > 0 for k in ('WAIT','RELEASE'))
    result = dict(stage=stage, PASS=all(gates.values()), gates=gates, records=records,
        PPO_ALLOWED=stage == 'entry' and all(gates.values()), protocol_sha256=_sha(PROTOCOL), source_hashes=source_hashes(),
        checkpoint_sha256=p['initial_checkpoint_sha256'], training_called=stage == 'train',
        prior_stage=prior['stage'] if prior else None, scope=p['scope'])
    _write(output/'manifest.json', result)
    return result


def run(args):
    if _sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError('continuation protocol changed')
    p = yaml.safe_load(PROTOCOL.read_text(encoding='utf-8-sig'))
    if _sha(Path('terminal_marl/core/trajectory_generator.py')) != p['generator_sha256']:
        raise ValueError('trajectory repair snapshot changed')
    for key, name in [('training_manifest_sha256','manifest.json'), ('evaluation_manifest_sha256',None)]:
        path = args.checkpoint.parent/name if name else args.evaluation_manifest
        if _sha(path) != p[key] or not json.loads(path.read_text())['PASS']:
            raise ValueError('previous formal training/evaluation evidence mismatch')
    if args.mode == 'smoke':
        return execute('smoke', args.output, args.checkpoint, args.history_root, p)
    args.output.mkdir(parents=True, exist_ok=False)
    prior = None
    for stage in ('qualify','entry','train'):
        prior = execute(stage, args.output/stage, args.checkpoint, args.history_root, p, prior)
        _write(args.output/'progress.json', dict(stage=stage, PASS=prior['PASS']))
        if not prior['PASS']:
            break
    _write(args.output/'manifest.json', prior)
    return prior


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke','pipeline'], required=True)
    for name in ('output','checkpoint','evaluation-manifest','history-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run(args)
    print(json.dumps(dict(stage=result['stage'], PASS=result['PASS'])), flush=True)
