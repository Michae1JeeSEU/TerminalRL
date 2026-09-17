"""Targeted transfer regression; no training or historical evidence mutation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from terminal_marl.algo.joint_ppo_v4_2 import JointPPOV42
from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training.run_v4_2_v2_1_student_qualification import protocol, scenario_config, episode_gates
from terminal_marl.training.student_rollout_v4_2 import rollout
from terminal_marl.training.run_v4_2_i2_dagger_preflight import _policy_tape_logprob_exact
from terminal_marl.visualization.snapshot_playback import SnapshotRecorder, write_playback


def run(checkpoint, output, scenario, seed, steps=None):
    p = protocol()
    config = scenario_config(scenario, p)
    policy = JointPPOV42(88)
    load_checkpoint(checkpoint, policy, config, load_optimizer=False)
    _, summary = rollout(policy, config, seed=seed,
        max_steps=steps or int(config.max_simulation_seconds/config.dt), output=output,
        deterministic=True)
    env = TerminalParallelEnv(config)
    recorder = SnapshotRecorder(2)
    metrics = {'samples': 0, 'reverse_samples': 0, 'max_inner_excursion_m': 0., 'plans': {}, 'by_uid': {}}
    def observe(world):
        recorder(world)
        for u in world.active.values():
            plan = u.committed_plan
            if not plan or plan.mode != 'ring_transfer':
                continue
            xy, v = u.position[:2], u.velocity[:2]
            r = float(np.linalg.norm(xy))
            target_r = float(np.linalg.norm(plan.points[-1].position[:2]))
            vt = float((xy[0]*v[1]-xy[1]*v[0])/max(r, 1e-9))
            row = metrics['by_uid'].setdefault(str(u.uid), {'min_tangential_speed': vt, 'max_inner_excursion_m': 0.})
            row['min_tangential_speed'] = min(row['min_tangential_speed'], vt)
            row['max_inner_excursion_m'] = max(row['max_inner_excursion_m'], target_r-r)
            metrics['samples'] += 1
            metrics['reverse_samples'] += int(vt < -1e-3)
            metrics['max_inner_excursion_m'] = max(metrics['max_inner_excursion_m'], target_r-r)
            metrics['plans'][plan.plan_id] = plan.metadata.get('transfer_geometry', 'legacy_cubic')
    env.set_step_observer(observe)
    summary['replay_exact'] = SafetyTraceRecorder.replay_action_tape(env, summary['action_tape'])['exact']
    summary['requested_logprob_exact'] = _policy_tape_logprob_exact(policy, config, summary['action_tape'])
    summary['episode_gates'] = episode_gates(summary, config, p)
    summary['geometry'] = metrics
    summary['scenario'] = scenario
    summary['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    generator = Path(__file__).parents[1]/'core'/'trajectory_generator.py'
    summary['generator_sha256'] = hashlib.sha256(generator.read_bytes()).hexdigest()
    summary['geometry_pass'] = metrics['samples'] > 0 and metrics['reverse_samples'] == 0 and metrics['max_inner_excursion_m'] <= .05
    write_playback(output / 'validation_scene.html', recorder.frames, f'Transfer repair regression {scenario} seed {seed}')
    (output/'regression.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ('completed','seconds','terminal','safety','replay_exact','episode_gates','geometry')}, indent=2))

if __name__ == '__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--checkpoint', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--scenario', choices=['warm3','flow3','flow6'], required=True)
    a.add_argument('--seed', type=int, required=True)
    a.add_argument('--steps', type=int)
    args=a.parse_args()
    torch.set_num_threads(1)
    run(**vars(args))
