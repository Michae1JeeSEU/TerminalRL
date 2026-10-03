"""Bounded, event-sampled RELEASE/K3 development preflight; no learning."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import time
import traceback

import numpy as np
import torch

from terminal_marl.configs.p3_integrated_repair import build_config, load_legacy_source
from terminal_marl.algo.service_context_runtime_v4_2 import initialize_service_context_policy
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.scripts.diagnose_p3_direct_contract import (
    CHECKPOINT_SHA, canonical, check_migration, check_requests, execute_tape,
    require, sha256, tensor_digest, write_json,
)
from terminal_marl.training.student_rollout_v4_2 import _student_request_details, validate_student_config


INTERFACES = frozenset({'o2m', 'm2i', 'i2f'})
ROLE_INTERFACE = {'outer': 'o2m', 'middle': 'm2i', 'inner': 'i2f'}
PROTOCOL = dict(requests=12, warm_start=6, high_soc=True, env_seed=891901,
                policy_seed=891902, max_steps=1800, wall_seconds=1800)
SMOKE = dict(requests=3, warm_start=3, high_soc=True, env_seed=891911,
             policy_seed=891912, max_steps=2, wall_seconds=120)


def event_labels(requested, executed, seen_requested, seen_executed):
    return ([f'first_request:{name}' for name in sorted(set(requested.values()) - seen_requested)] +
            [f'first_execution:{name}' for name in sorted(set(executed.values()) - seen_executed)])


def collect(policy, env, output, *, policy_seed, max_steps, wall_seconds, smoke=False):
    """One policy draw per tick; at most six event ticks get one extra replay."""
    started = time.monotonic()
    result = dict(status='ERROR', error=None, ticks=0, env_step_calls=0, replay_ticks=0,
                  policy_requests=0, terminal=False, censored=True, events=[],
                  first_request={}, first_execution={}, request_counts={},
                  fallback_selected=0, request_rejections=0, stop_reason=None)
    seen_requested, seen_executed = set(), set()
    counts = Counter()
    tape = None
    before_weights = tensor_digest(policy.model)
    try:
        validate_student_config(env.config)
        require(not hasattr(policy, 'optimizer'), 'optimizer present')
        require(not any(p.requires_grad for p in policy.model.parameters()), 'unfrozen policy')
        observations = env._observations()
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(policy_seed)
            for tick in range(max_steps):
                if time.monotonic()-started >= wall_seconds:
                    result['stop_reason'] = 'wall_limit'
                    break
                before_time = float(env.now)
                rng_before = torch.random.get_rng_state().clone()
                details = _student_request_details(policy, env, observations, deterministic=False, shortlist_k=3)
                result['policy_requests'] += 1
                rng_after = torch.random.get_rng_state().clone()
                if observations:
                    tape = check_requests(policy, env, observations, details)
                else:
                    require(not env.active and details[1].shape == (0, 90), 'empty UID binding mismatch')
                    tape = dict(uids=[], world=[], rows=[], actions={}, shortlists={}, counts={})
                # Use the same JSON representation for the live step and replay.
                serialized = json.loads(json.dumps(canonical(tape), allow_nan=False))
                requested = {uid: ROLE_INTERFACE[env.active[int(uid)].ring_role.value]
                             for uid, gate in zip(details[0], details[4]) if gate == 1}
                pre_rows = {uid: dict(arrival_id=env.active[int(uid)].arrival_id,
                                     phase=env.active[int(uid)].phase.value,
                                     interface=requested[uid]) for uid in requested}
                # Preserve the pre-step state while a first execution is still
                # possible. Rejects are logged; they are never discarded/retried.
                may_be_event = bool(set(requested.values()) - seen_requested or
                                    set(requested.values()) - seen_executed)
                snapshot = pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL) if may_be_event else None
                result['env_step_calls'] += 1
                live = execute_tape(env, serialized)
                result['ticks'] = tick+1
                info = live['outputs'][4]
                executed = {}
                for uid, interface in requested.items():
                    trace = info['action_traces'][uid]
                    rank = trace['runtime_shortlist_rank']
                    if rank is not None:
                        require(info['release_decisions'][uid] == 'approved', 'K3 rank without approval')
                        require(uid in live['after']['plans'], 'approved release has no checked active plan')
                        expected_mode = 'final_approach' if interface == 'i2f' else 'ring_transfer'
                        require(live['after']['plans'][uid]['mode'] == expected_mode, 'wrong executed trajectory mode')
                        executed[uid] = interface
                        result['fallback_selected'] += int(rank > 1)
                    else:
                        result['request_rejections'] += 1
                labels = event_labels(requested, executed, seen_requested, seen_executed)
                counts.update(tape['counts'])
                record = dict(tick=tick+1, time_before=before_time, counts=tape['counts'],
                              rows=serialized['rows'], requested_context=pre_rows,
                              release_decisions=info['release_decisions'],
                              action_traces=info['action_traces'], labels=labels,
                              outcomes=live['after']['outcomes'],
                              checked_executed_samples=env.telemetry['checked_executed_plan_samples'])
                with (output / 'decisions.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                if labels:
                    require(snapshot is not None, 'missing event pre-state')
                    event_dir = output / f'event_{tick+1:04d}'
                    event_dir.mkdir()
                    (event_dir / 'before.pkl').write_bytes(snapshot)
                    write_json(event_dir / 'request_tape.json', serialized)
                    write_json(event_dir / 'live.json', live)
                    write_json(event_dir / 'policy_rng.json', dict(seed=policy_seed,
                        before_request=rng_before.numpy().tobytes().hex(),
                        after_request=rng_after.numpy().tobytes().hex()))
                    replay_env = pickle.loads(snapshot)  # Produced by this process, not external input.
                    replay_tape = json.loads((event_dir / 'request_tape.json').read_text(encoding='utf8'))
                    result['env_step_calls'] += 1
                    replay = execute_tape(replay_env, replay_tape)
                    result['replay_ticks'] += 1
                    write_json(event_dir / 'replay.json', replay)
                    require(live == replay, f'event replay differs at tick {tick+1}')
                    require(torch.equal(torch.random.get_rng_state(), rng_after), 'execution consumed policy RNG')
                    event = dict(tick=tick+1, labels=labels, directory=event_dir.name, replay_equal=True,
                                 requested=requested, executed=executed,
                                 actual_ranks={uid: info['action_traces'][uid]['runtime_shortlist_rank']
                                               for uid in requested},
                                 plans={uid: live['after']['plans'][uid] for uid in executed})
                    result['events'].append(event)
                for interface in set(requested.values())-seen_requested:
                    result['first_request'][interface] = tick+1
                for interface in set(executed.values())-seen_executed:
                    result['first_execution'][interface] = tick+1
                seen_requested.update(requested.values())
                seen_executed.update(executed.values())
                require(len(result['events']) <= 6, 'event budget exceeded')
                result['request_counts'] = dict(counts)
                result['terminal'] = (env.arrivals.exhausted() and not env.active and
                                      not env.admission.queue.items and not env.service_uavs)
                require(not any(getattr(value, 'value', value).startswith('failed') or
                                getattr(value, 'value', value) == 'invalid'
                                for value in live['after']['outcomes'].values()), 'terminal failure during preflight')
                if tick == 0 or (tick+1) % 25 == 0 or labels or result['terminal']:
                    write_json(output / 'progress.json', dict(status='running', ticks=tick+1,
                        elapsed_seconds=time.monotonic()-started, request_counts=dict(counts),
                        first_request=result['first_request'], first_execution=result['first_execution'],
                        replay_ticks=result['replay_ticks'], active=len(env.active)))
                observations = {uid: np.asarray(row, dtype=np.float32) for uid, row in live['outputs'][0].items()}
                if not smoke and seen_requested == INTERFACES and seen_executed == INTERFACES:
                    result.update(status='PASS', stop_reason='coverage_reached')
                    break
                if result['terminal']:
                    result['stop_reason'] = 'natural_terminal'
                    break
            if result['stop_reason'] is None:
                result['stop_reason'] = 'tick_limit'
        require(before_weights == tensor_digest(policy.model), 'policy weights changed')
        result['weights_unchanged'] = True
        if smoke:
            result['status'] = 'SMOKE_OK'
        elif result['status'] != 'PASS':
            result['status'] = 'COVERAGE_MISSING'
    except Exception as exc:
        result.update(status='ERROR', stop_reason='exception',
                      error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        if tape is not None:
            write_json(output / 'failure_request.json', tape)
        (output / 'failure_state.pkl').write_bytes(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL))
    result.update(censored=not result['terminal'], elapsed_seconds=time.monotonic()-started,
                  missing_request=sorted(INTERFACES-seen_requested),
                  missing_execution=sorted(INTERFACES-seen_executed),
                  outcomes=dict(env.ledger.outcomes), unresolved=sorted(env.ledger.unresolved),
                  telemetry=dict(env.telemetry))
    return result


def run(checkpoint, output, *, smoke=False):
    spec = SMOKE if smoke else PROTOCOL
    report = dict(protocol=spec, smoke=smoke, PPO_ALLOWED=False, qualification='NOT_QUALIFIED',
                  trained=False, optimizer_created=False, status='ERROR', error=None)
    try:
        require(sha256(checkpoint) == CHECKPOINT_SHA, 'wrong fixed checkpoint')
        config = build_config(requests=spec['requests'], warm_start=spec['warm_start'], high_soc=spec['high_soc'])
        report.update(config=asdict(config), checkpoint=str(checkpoint), checkpoint_sha256=CHECKPOINT_SHA)
        source, payload = load_legacy_source(checkpoint, config)
        policy, migration = initialize_service_context_policy(source, source_checkpoint_metadata=payload['metadata'])
        env = TerminalParallelEnv(config, seed=spec['env_seed'])
        obs = env._observations()
        world = np.asarray([obs[uid] for uid in sorted(obs)], dtype=np.float32)
        report['migration'] = migration
        report['migration_check'] = check_migration(source, policy, world)
        report.update(collect(policy, env, output, policy_seed=spec['policy_seed'],
                              max_steps=spec['max_steps'], wall_seconds=spec['wall_seconds'], smoke=smoke))
        require(sha256(checkpoint) == CHECKPOINT_SHA, 'checkpoint file changed')
    except Exception as exc:
        report.update(status='ERROR', error=dict(type=type(exc).__name__, message=str(exc),
                                                traceback=traceback.format_exc()))
    write_json(output / 'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve earlier evidence')
    args.output.mkdir(parents=True)
    report = run(args.checkpoint, args.output, smoke=args.smoke)
    print(json.dumps({key: report.get(key) for key in
        ('status', 'ticks', 'replay_ticks', 'first_request', 'first_execution', 'stop_reason', 'error')}, ensure_ascii=False))
    return 0 if report['status'] in {'PASS', 'SMOKE_OK'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
