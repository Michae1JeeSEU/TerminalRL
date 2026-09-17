"""Once-only held-out paired evaluation of repaired-geometry PPO block2."""
import argparse
import json
from pathlib import Path
import torch
from terminal_marl.training.run_v4_2_ppo_block2 import restore, replay_geometry, source_hashes
from terminal_marl.training.run_v4_2_v2_1_student_qualification import protocol, scenario_config, episode_gates
from terminal_marl.training.run_v4_2_i2_dagger_preflight import _policy_tape_logprob_exact
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r3 import _write
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import _sha, reserve_once
from terminal_marl.training.student_rollout_v4_2 import rollout

CANDIDATE_SHA = '6420df9a8562135756819459c7682ec897d47acd64e841b75a21c697298ea14f'
BASELINE_SHA = '965c5438dfc42f9898a4059b1225527bebd94f048ae9dfe6253cb49fc0519198'
TRAINING_SHA = '657f28d8a71f88fde650fb193d31bcbba1fd435877a0a5a00771b930beb2b019'


def run(output, checkpoint, baseline, history_root, smoke=False):
    if _sha(checkpoint.parent/'manifest.json') != TRAINING_SHA:
        raise ValueError('block2 training evidence mismatch')
    policies = {name: restore(path, dict(initial_checkpoint_sha256=sha, learning_rate=.00003))
        for name, path, sha in [('candidate',checkpoint,CANDIDATE_SHA),('baseline',baseline,BASELINE_SHA)]}
    output.mkdir(parents=True, exist_ok=False)
    seeds = [740000] if smoke else list(range(710001,710007))
    if set(seeds) & historical_registry(history_root, output)['seeds']:
        raise ValueError('held-out seeds previously consumed')
    if not smoke:
        reserve_once(checkpoint.parent/'heldout710_consumed.json', dict(seeds=seeds, checkpoint_sha256=CANDIDATE_SHA, output=str(output)))
    _write(output/'pre_reset_registry.json', dict(seeds=seeds, paired=True, smoke=smoke))
    results = {k: [] for k in policies}
    for i, seed in enumerate(seeds):
        scenario = ('warm3','flow3','flow6')[i % 3]
        config = scenario_config(scenario, protocol())
        for name, policy in policies.items():
            _, summary = rollout(policy, config, seed=seed, max_steps=32 if smoke else 900,
                output=output/name/f'seed{seed}', deterministic=i < 3, collect_labels=False)
            summary['replay_exact'], geometry = replay_geometry(config, summary['action_tape'])
            summary['requested_logprob_exact'] = _policy_tape_logprob_exact(policy, config, summary['action_tape'])
            gates = episode_gates(summary, config, protocol())
            gates['geometry'] = geometry['samples'] > 0 and geometry['reverse_samples'] == 0 and geometry['max_inner_excursion_m'] <= .05
            results[name].append(dict(scenario=scenario, summary=summary, geometry=geometry, gates=gates))
            _write(output/'progress.json', dict(completed=sum(map(len,results.values())), total=2*len(seeds), results=results))
    pairs = [dict(seed=a['summary']['seed'], scenario=a['scenario'], candidate_seconds=a['summary']['seconds'],
        baseline_seconds=b['summary']['seconds'], delta_seconds=a['summary']['seconds']-b['summary']['seconds'])
        for a,b in zip(results['candidate'],results['baseline'])]
    passed = not smoke and len(results['candidate']) == 6 and all(all(r['gates'].values()) for r in results['candidate'])
    result = dict(PASS=passed, smoke=smoke, stage='BLOCK2_HELDOUT_EVALUATION', results=results, pairs=pairs,
        checkpoint_sha256=CANDIDATE_SHA, baseline_sha256=BASELINE_SHA, training_called=False,
        source_hashes={**source_hashes(), 'evaluation_runner':_sha(Path(__file__))})
    _write(output/'manifest.json', result)
    print(json.dumps(dict(PASS=passed, smoke=smoke, pairs=pairs)), flush=True)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('output','checkpoint','baseline','history-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    torch.set_num_threads(1)
    run(**vars(p.parse_args()))
