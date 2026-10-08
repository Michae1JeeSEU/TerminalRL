"""Continue an explicitly reviewed V2.5 PPO pilot for a frozen finite campaign.

The primary writes the manifest only after reviewing real qualification and
paired pilot results. This runner never chooses candidates, changes budgets,
reads Q observations, launches held-out evaluation, or schedules monitoring.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..algo.trajectory_set_ppo import TrajectorySetPPO
from ..configs.p3_v25_course import PROFILE, SCENARIOS, build_config
from .p3_v25_course_data import load_train, TRAIN_CASES, Q_CASES
from .p3_v25_imitation_qualification import sha
from .p3_v25_ppo_block import input_review, OPTIONS, PROTOCOL as PILOT_PROTOCOL
from .p3_v25_ppo_block import TRAIN_SEEDS as PILOT_SEEDS
from .p3_v25_rollout import collect_episode
from .p3_v25_runtime import load_actor
from .p3_v25_set_learnability import digest_state
from .p3_v25_task_imitation import frozen_models
from .p3_v25_task_student import blockers
from .p3_v3_bc_learnability import require, read, write

PROTOCOL = 'p3_v25_neutral_formal_20261008_v1'


def review(manifest):
    require(manifest['protocol'] == PROTOCOL and manifest['profile'] == PROFILE,
            'wrong formal protocol/profile')
    require(manifest['formal_allowed'] is True and manifest['candidate_retained'] is True
            and isinstance(manifest['review_reason'], str) and manifest['review_reason'].strip(),
            'primary review has not admitted this candidate')
    source, _, baselines = input_review(manifest['fit_directory'], manifest['qualification_directory'],
        manifest['warmup_directory'], manifest['development_directories'])
    pilot_dir = Path(manifest['pilot_directory'])
    require(sha(pilot_dir/'result.json') == manifest['pilot_result_sha256'], 'reviewed pilot result changed')
    pilot = read(pilot_dir/'result.json')
    require(pilot['protocol'] == PILOT_PROTOCOL and pilot['profile'] == PROFILE
            and pilot['source'] == source and pilot['status'] == 'PILOT_COMPLETE'
            and not pilot['blockers'] and pilot['accepted_updates'] > 0,
            'pilot is incomplete or has no accepted update')
    require(len(pilot['training']) == len(pilot['evaluation']) == len(SCENARIOS),
            'pilot conditions are incomplete')
    for index, (scenario, baseline) in enumerate(zip(SCENARIOS, baselines)):
        training, evaluation = pilot['training'][index], pilot['evaluation'][index]
        require(training['scenario'] == evaluation['scenario'] == scenario
                and training['seed'] == PILOT_SEEDS[index]
                and not training['blockers'] and not blockers(training['rollout'])
                and not training['update']['budget_exhausted']
                and not evaluation['blockers'] and not blockers(evaluation['candidate'])
                and evaluation['baseline'] == baseline['result'], 'pilot evidence differs from reviewed inputs')
    candidate = pilot['training'][-1]
    require(candidate['checkpoint'] == 'after_03.pt', 'unexpected pilot checkpoint')
    checkpoint = pilot_dir/candidate['checkpoint']
    require(sha(checkpoint) == candidate['checkpoint_sha256'] == manifest['candidate_checkpoint_sha256'],
            'reviewed candidate bytes changed')
    episodes = manifest['episodes']
    require(isinstance(episodes, list) and len(episodes) > 0 and len(episodes) % len(SCENARIOS) == 0,
            'formal campaign must contain complete declared curriculum cycles')
    used = {v[0] for v in (*TRAIN_CASES, *Q_CASES)} | set(PILOT_SEEDS) | {b['seed'] for b in baselines}
    for index, item in enumerate(episodes):
        require(item['scenario'] == SCENARIOS[index % len(SCENARIOS)], 'formal curriculum order differs')
        require(all(type(item[k]) is int and 0 <= item[k] < 2**32
                    for k in ('seed', 'policy_seed', 'update_seed')), 'invalid formal seed')
        require(item['seed'] not in used, 'formal seed overlaps previous cohort or campaign')
        used.add(item['seed'])
    require(type(manifest['initialization_seed']) is int
            and 0 <= manifest['initialization_seed'] < 2**32, 'invalid initialization seed')
    require(type(manifest['campaign_wall_seconds']) in (int, float)
            and np.isfinite(manifest['campaign_wall_seconds'])
            and manifest['campaign_wall_seconds'] > 0, 'invalid formal wall budget')
    return source, checkpoint


def _run(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = read(manifest_path)
    manifest_hash = sha(manifest_path)
    source, checkpoint = review(manifest)
    # Only the strict train loader is used by the optimizer reference.
    reference = load_train(manifest['train_directories']).new
    actor, _ = load_actor(Path(manifest['fit_directory'])/source['path'])
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(payload['protocol'] == PILOT_PROTOCOL and payload['profile'] == PROFILE
            and payload['source'] == source and payload['completed_cycles'] == 3,
            'candidate checkpoint provenance differs')
    for name, value in actor.state_dict().items():
        require(torch.equal(value, payload['model']['actor.'+name]), 'pilot changed frozen actor/proposal')
    torch.manual_seed(manifest['initialization_seed'])
    np.random.seed(manifest['initialization_seed'])
    policy = TrajectorySetPPO(actor, learning_rate=3e-5, reference=reference,
                              reference_limit=.02, reference_coefficient=1.)
    policy.model.load_state_dict(payload['model'])
    policy.optimizer.load_state_dict(payload['optimizer'])
    require(all(g['lr'] == 3e-5 for g in policy.optimizer.param_groups), 'pilot optimizer rate differs')
    require(float(policy.reference_kl().detach()) <= .02, 'candidate exceeds fixed-reference KL')
    frozen_actor = digest_state(policy.model.actor.state_dict())
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write(output/'protocol.json', dict(protocol=PROTOCOL, profile=PROFILE, manifest=manifest,
        manifest_sha256=manifest_hash, source=source, learning_rate=3e-5,
        reference_kl_limit=.02, reference_coefficient=1., options=OPTIONS,
        physics={s: asdict(build_config(s)) for s in SCENARIOS},
        rollout_and_replay_wall_seconds_each=3600, update_wall_seconds=600,
        optimizer_continued_from_pilot=True, Q_observations_accessed=False,
        campaign_budget_check='before each complete episode',
        teacher_enabled=False, heartbeat_enabled=False, PPO_ALLOWED=True))
    report = dict(protocol=PROTOCOL, profile=PROFILE, status='RUNNING', error=None, source=source,
        episodes=[], blockers=[], current_episode=None, requested_episodes=len(manifest['episodes']),
        initial_model_hash=digest_state(policy.model.state_dict()), initial_actor_hash=frozen_actor,
        source_checkpoint_sha256=sha(checkpoint), PPO_ALLOWED=True)
    write(output/'result.json', report)
    for index, case in enumerate(manifest['episodes']):
        if time.monotonic()-started >= manifest['campaign_wall_seconds']:
            report.update(status='BUDGET_EXHAUSTED', blockers=['campaign_wall_budget'])
            break
        report['current_episode'] = index + 1
        write(output/'result.json', report)
        config = build_config(case['scenario'])
        steps, result = collect_episode(policy.model, config, seed=case['seed'], policy_seed=case['policy_seed'],
            output=output/f'episode_{index+1:04d}', max_steps=int(config.max_simulation_seconds/config.dt))
        issues = blockers(result)
        item = dict(**case, rollout=result, blockers=issues)
        report['episodes'].append(item)
        if issues:
            report.update(status='BLOCKED', blockers=[f'episode_{index+1}:{b}' for b in issues])
            break
        write(output/'result.json', report)
        np.random.seed(case['update_seed'])
        item['before_model_hash'] = digest_state(policy.model.state_dict())
        item['update'] = policy.update(steps, training_authorized=True, wall_seconds=600, **OPTIONS)
        require(digest_state(policy.model.actor.state_dict()) == frozen_actor, 'formal PPO changed frozen actor/proposal')
        item['after_model_hash'] = digest_state(policy.model.state_dict())
        path = output/f'after_{index+1:04d}.pt'
        torch.save(dict(protocol=PROTOCOL, profile=PROFILE, source=source, model=policy.model.state_dict(),
            optimizer=policy.optimizer.state_dict(), completed_episodes=index+1,
            manifest_sha256=manifest_hash), path)
        item.update(checkpoint=path.name, checkpoint_sha256=sha(path))
        if item['update']['budget_exhausted']:
            report.update(status='BLOCKED', blockers=[f'episode_{index+1}:update_wall_budget'])
            break
        write(output/'result.json', report)
    else:
        report['status'] = 'TRAINING_COMPLETE'
    report.update(elapsed_seconds=time.monotonic()-started,
        completed_updates=sum(v.get('update', {}).get('accepted_updates', 0) for v in report['episodes']))
    frozen_models(manifest['fit_directory'])
    write(output/'result.json', report)
    return report


def run(manifest, output):
    require(not Path(output).exists(), 'formal output already exists')
    try:
        return _run(manifest, output)
    except Exception as error:
        path = Path(output)/'result.json'
        if not path.exists():
            raise
        report = read(path)
        report.update(status='ERROR', error=repr(error), blockers=['technical_error'])
        write(path, report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = run(args.manifest, args.output)
    print(json.dumps(dict(status=report['status'], blockers=report['blockers'])))
    return 0 if report['status'] == 'TRAINING_COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
