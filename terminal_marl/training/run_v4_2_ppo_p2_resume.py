"""Natural-flow P2 PPO with distance-loop reward disabled only in this curriculum."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
import yaml
from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42
from terminal_marl.core.checkpoints import load_checkpoint, save_checkpoint
from terminal_marl.training.run_v4_2_ppo_block2 import restore, replay_geometry, source_hashes, state_equal
from terminal_marl.training.run_v4_2_curriculum_pilot import config_for
from terminal_marl.training.run_v4_2_v2_1_ppo import requested_logprob_error
from terminal_marl.training.run_v4_2_v2_1_student_qualification import protocol as i3_protocol, episode_gates
from terminal_marl.training.run_v4_2_i2_dagger_preflight import _policy_tape_logprob_exact
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import _sha
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r3 import _write
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.student_rollout_v4_2 import rollout
PROTOCOL=Path('terminal_marl/configs/low_v4_2/ppo_p2_resume_v2.yaml')


def p2_config(n):
    return replace(config_for(n), low_level_progress_reward_per_meter=0.)


def p2_source_hashes():
    return {**source_hashes(), 'p2_runner':_sha(Path(__file__)),
            'p2_config':_sha(Path('terminal_marl/training/run_v4_2_curriculum_pilot.py'))}


def validate_p2_gate(report, stage, p):
    if report.get('stage')!=stage or not report.get('PASS') or not report.get('gates') or not all(report['gates'].values()):
        raise PermissionError('P2 prior stage has not passed')
    if report.get('source_hashes')!=p2_source_hashes() or report.get('protocol_sha256')!=_sha(PROTOCOL) or report.get('checkpoint_sha256')!=p['initial_checkpoint_sha256']:
        raise ValueError('P2 evidence changed')


def execute(stage, output, checkpoint, history, p, prior=None):
    if stage in ('entry', 'train'):
        validate_p2_gate(prior, 'qualify' if stage == 'entry' else 'entry', p)
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
        config = p2_config(int(name[4:]))
        horizon = p['smoke_steps'] if stage == 'smoke' else p['entry_steps'][index] if stage == 'entry' else int(config.max_simulation_seconds/config.dt)
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
            gates.update(episode_gates(summary, config, {**i3_protocol(), 'horizon_seconds': config.max_simulation_seconds}))
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
                    formal=stage == 'train', ppo_called=True, protocol_sha256=_sha(PROTOCOL), source_hashes=p2_source_hashes(),
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
        PPO_ALLOWED=stage == 'entry' and all(gates.values()), protocol_sha256=_sha(PROTOCOL), source_hashes=p2_source_hashes(),
        checkpoint_sha256=p['initial_checkpoint_sha256'], training_called=stage == 'train',
        prior_stage=prior['stage'] if prior else None, scope=p['scope'])
    _write(output/'manifest.json', result)
    return result


def run(args):
    if _sha(PROTOCOL) != '343f4499fe726e7145181a0eb090bd4fb56dac3de8643fe9d5e64cb06bebd837':raise ValueError('P2 resume protocol changed')
    p=yaml.safe_load(PROTOCOL.read_text(encoding='utf-8-sig'))
    if _sha(args.failed_manifest)!=p['failed_manifest_sha256']:
        raise ValueError('failed block evidence mismatch')
    failed=json.loads(args.failed_manifest.read_text())
    records=failed['records']
    if failed['PASS'] or len(records)!=24 or not all(all(r['gates'].values()) for r in records[:23]):
        raise ValueError('expected 23 accepted updates before stopped episode')
    if {k for k,v in records[-1]['gates'].items() if not v}!={'initial_ratio'} or records[-1]['update'] is not None:
        raise ValueError('unexpected failure or unauthorized last update')
    if args.checkpoint.name!='checkpoint_23.pt':raise ValueError('resume only last accepted update')
    if args.mode=='smoke':return execute('smoke',args.output,args.checkpoint,args.history_root,p)
    args.output.mkdir(parents=True,exist_ok=False)
    prior=None
    for stage in ('qualify','entry','train'):
        prior=execute(stage,args.output/stage,args.checkpoint,args.history_root,p,prior)
        _write(args.output/'progress.json',dict(stage=stage,PASS=prior['PASS']))
        if not prior['PASS']:break
    _write(args.output/'manifest.json',prior)
    return prior


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['smoke','pipeline'],required=True)
    for name in ('output','checkpoint','history-root','failed-manifest'):
        p.add_argument('--'+name,type=Path,required=True)
    torch.set_num_threads(1)
    r=run(p.parse_args())
    print(json.dumps(dict(stage=r['stage'],PASS=r['PASS'])),flush=True)
