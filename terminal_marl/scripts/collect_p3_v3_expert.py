"""Bounded fresh V3/90D expert pilot; no training or qualification cohort access.

The historical state-only expert proposes labels on a private environment copy.
The live environment receives only its selected canonical requests. All rows,
including masked WAIT and rejected RELEASE, are retained before execution.
"""
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

from terminal_marl.configs.p3_integrated_repair import PROFILE, build_config
from terminal_marl.core.low_level_joint_action_v4_2 import hierarchical_from_joint_index_v4_1
from terminal_marl.core.observations import build_low_observation_v2_4
from terminal_marl.core.safety_trace import _digest, _json
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.scripts.diagnose_p3_direct_contract import (
    SAFETY_KEYS, canonical, execute_tape, require, state_record, write_json,
)
from terminal_marl.training.v4_1_joint_teacher import teacher_joint_expert_v4_1

DATA_SCHEMA = 'p3_v3_90d_expert_pilot_v1'
SMOKE_SEED = 892101
PILOT_CASES = ((892201, True), (892202, False))


def actor_eligible(env, uav):
    interface = {'outer': 'o2m', 'middle': 'm2i', 'inner': 'i2f'}.get(
        getattr(uav.ring_role, 'value', None))
    return bool(uav.committed_plan is None
                and uav.phase.value in {'ring_outer', 'ring_middle', 'ring_inner'}
                and uav.uid not in env.future_release_reservations
                and interface is not None and uav.soc > 0
                and env.now >= uav.safety_retry_after
                and interface not in env.airspace.affected_interfaces())


def labels_from_snapshot(data, observations, *, seed, tick, repeat=False):
    """Candidate telemetry and safety certificate caches never touch live state."""
    probe = pickle.loads(data)  # Bytes made in this process, not external input.
    rows, actions, shortlists = [], {}, {}
    require(set(observations) == {str(uid) for uid in probe.active}, 'observation UID mismatch')
    for uid in sorted(observations):
        uav = probe.active[int(uid)]
        obs = np.asarray(observations[uid], dtype=np.float32)
        require(obs.shape == (90,) and np.isfinite(obs).all(), 'invalid 90D observation')
        require(np.array_equal(obs, build_low_observation_v2_4(uav, probe)),
                '90D observation reconstruction differs')
        certificates = []
        candidate_plan = probe._candidate_plan

        def record_candidate(*args, **kwargs):
            proposal, plan, reason = candidate_plan(*args, **kwargs)
            if proposal is not None and plan is not None and reason is None:
                checked = probe.safety.checked_trajectory_hashes.get(plan.plan_id)
                require(checked == plan.trajectory_hash, 'teacher candidate lacks checked hash')
                certificates.append(dict(mode=plan.mode, trajectory_hash=plan.trajectory_hash,
                                         checked_hash=checked))
            return proposal, plan, reason

        probe._candidate_plan = record_candidate
        try:
            selected, audit = teacher_joint_expert_v4_1(probe, uav, state_only=True)
        finally:
            del probe.__dict__['_candidate_plan']
        if repeat:
            independent = pickle.loads(data)
            repeated, repeated_audit = teacher_joint_expert_v4_1(
                independent, independent.active[int(uid)], state_only=True)
            require(selected.index == repeated.index and audit == repeated_audit,
                    'teacher label depends on prior queries')
        action = hierarchical_from_joint_index_v4_1(selected.index)
        eligible = actor_eligible(probe, uav)
        require(eligible or action.gate == 0, 'forced-WAIT row has RELEASE label')
        require((action.gate == 1) == bool(certificates), 'label/certificate mismatch')
        # Offline expert search is not a student K3 ranking. Execute only its
        # selected request; do not invent two extra proposals or relabel rejects.
        actions[uid] = action.normalized.tolist()
        if action.gate:
            shortlists[uid] = [actions[uid]]
        interface = {'outer': 'o2m', 'middle': 'm2i', 'inner': 'i2f'}.get(
            getattr(uav.ring_role, 'value', None))
        rows.append(dict(schema=DATA_SCHEMA, partition='development_pilot_never_train',
            seed=seed, tick=tick, time=probe.now, uid=uid, arrival_id=uav.arrival_id,
            episode_id=f'{DATA_SCHEMA}:{seed}', lifecycle_id=f'{DATA_SCHEMA}:{seed}:{uid}',
            decision_id=f'{DATA_SCHEMA}:{seed}:{tick}:{uid}', observation=obs.tolist(),
            observation_schema=probe.config.observation_schema_version,
            observation_dim=90, actor_mask=eligible, teacher_gate=int(action.gate),
            teacher_proposal=action.proposal, teacher_joint_index=int(selected.index),
            requested_action=actions[uid], teacher_audit=audit,
            selected_certificate=certificates[-1] if certificates else None,
            teacher_repeat_checked=repeat, interface=interface, phase=uav.phase.value,
            speed=float(uav.speed), soc=float(uav.soc),
            retry_active=probe.now < uav.safety_retry_after,
            committed=uav.committed_plan is not None,
            final_count=sum(u.phase.value == 'final' for u in probe.active.values()),
            busy_lz=sum(z.status.value != 'free' for z in probe.lz_manager.zones),
            retention='all_active_rows_before_step', bc_input_fields=['observation'],
            target_fields=['teacher_gate', 'teacher_proposal'],
            loss_mask_fields=['actor_mask']))
    return rows, dict(actions=actions, shortlists=shortlists)


def natural_terminal(env):
    return env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs


def collect(env, output, *, seed, max_steps=1800, wall_seconds=3600, smoke=False):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'config.json', asdict(env.config))
    started = time.monotonic()
    counts, proposals, contexts, recovery = Counter(), Counter(), Counter(), Counter()
    previous = {}
    ticks = rows_count = 0
    stop_reason, error = 'tick_limit', None
    data, tape = None, None
    replay_equal = False
    try:
        with (output/'rows.jsonl').open('x', encoding='utf8') as row_stream, \
                (output/'steps.jsonl').open('x', encoding='utf8') as step_stream:
            for tick in range(max_steps):
                if time.monotonic()-started >= wall_seconds:
                    stop_reason = 'wall_limit'
                    break
                observations = env._observations()
                data = pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)
                rows, tape = labels_from_snapshot(data, observations, seed=seed, tick=tick,
                                                 repeat=tick == 0)
                require(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL) == data,
                        'expert query mutated live environment')
                pre_hash = _digest(canonical(state_record(env)))
                for row in rows:
                    row['pre_state_hash'] = pre_hash
                    row_stream.write(json.dumps(_json(row), allow_nan=False)+'\n')
                    rows_count += 1
                    kind = 'RELEASE' if row['teacher_gate'] else ('actor_WAIT' if row['actor_mask'] else 'forced_WAIT')
                    counts[kind] += 1
                    if row['teacher_gate']:
                        proposals[str(row['teacher_proposal'])] += 1
                    for key, present in dict(fast_inner=row['interface'] == 'i2f' and row['speed'] >= 15,
                        busy_lz=row['busy_lz'] > 0, retry=row['retry_active'], low_soc=row['soc'] < .55).items():
                        contexts[key] += int(present)
                    prior = previous.get(row['uid'])
                    if prior and not prior['teacher_gate'] and row['teacher_gate']:
                        recovery['WAIT_to_RELEASE'] += 1
                        recovery['retry_to_RELEASE'] += int(prior['retry_active'])
                    previous[row['uid']] = row
                row_stream.flush()  # Labels survive a subsequent execution failure.
                tape = json.loads(json.dumps(tape))
                live = execute_tape(env, tape)
                ticks = tick+1
                step_stream.write(json.dumps(dict(tick=tick, pre_state_hash=pre_hash,
                    tape=tape, action_traces=live['outputs'][4]['action_traces'],
                    release_telemetry=canonical(env.last_release_telemetry),
                    post_state_hash=_digest(live['after']), outcomes=dict(env.ledger.outcomes)),
                    allow_nan=False)+'\n')
                step_stream.flush()
                if tick == 0:
                    (output/'first_before.pkl').write_bytes(data)
                    write_json(output/'first_request.json', tape)
                    replay = execute_tape(pickle.loads(data), tape)
                    write_json(output/'first_live.json', live)
                    write_json(output/'first_replay.json', replay)
                    require(live == replay, 'live/replay differs')
                    replay_equal = True
                outcomes = Counter(getattr(v, 'value', v) for v in env.ledger.outcomes.values())
                require(not outcomes.get('failed_energy'), 'failed_energy during expert pilot')
                if ticks == 1 or ticks % 25 == 0 or natural_terminal(env):
                    write_json(output/'progress.json', dict(ticks=ticks, rows=rows_count,
                        elapsed_seconds=time.monotonic()-started, active=len(env.active),
                        counts=dict(counts), outcomes=dict(outcomes)))
                if natural_terminal(env):
                    stop_reason = 'natural_terminal'
                    break
    except Exception as exc:
        stop_reason = 'error'
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        if data is not None:
            (output/'failure_before.pkl').write_bytes(data)
        write_json(output/'failure_request.json', tape)
    terminal = natural_terminal(env)
    outcomes = Counter(getattr(v, 'value', v) for v in env.ledger.outcomes.values())
    success = error is None and replay_equal and (terminal or (smoke and ticks == max_steps))
    report = dict(schema=DATA_SCHEMA, profile=PROFILE, seed=seed,
        status=('SMOKE_OK' if smoke else 'PILOT_OK') if success else ('ERROR' if error else 'CENSORED'),
        stop_reason=stop_reason, error=error, ticks=ticks, rows=rows_count,
        terminal=bool(terminal), censored=not terminal, elapsed_seconds=time.monotonic()-started,
        row_counts=dict(counts), proposals=dict(proposals), contexts=dict(contexts), recovery=dict(recovery),
        outcome_counts=dict(outcomes), unresolved=len(env.ledger.unresolved),
        requested=len(env.ledger.requested), generated_requests=len(env.request_times),
        active=len(env.active), queue=len(env.admission.queue.items), service=len(env.service_uavs),
        telemetry=dict(env.telemetry), energy_outcomes=env.energy_outcomes,
        first_tick_replay_equal=replay_equal, all_rows_retained=True,
        qualification='NOT_QUALIFIED', I0_QUALIFIED=False, PPO_ALLOWED=False, trained=False,
        optimizer_created=False, checkpoint_loaded=False, heldout_used=False)
    write_json(output/'result.json', report)
    return report


def run(output, *, smoke=False):
    output.mkdir(parents=True, exist_ok=False)
    cases = ((SMOKE_SEED, True),) if smoke else PILOT_CASES
    write_json(output/'protocol.json', dict(schema=DATA_SCHEMA, cases=cases,
        max_steps=2 if smoke else 1800, wall_seconds=180 if smoke else 3600,
        requests=3 if smoke else 12, warm_start=3 if smoke else 6,
        maximum_rows_per_episode=6 if smoke else 21600, historical_rows_used=0,
        partition='development_pilot_never_train', future_train_seeds=[893001,893024],
        future_audit_seeds=[894001,894006], future_Q_reserved=[895001,895012]))
    results = []
    for seed, high_soc in cases:
        env = TerminalParallelEnv(build_config(requests=3 if smoke else 12,
            warm_start=3 if smoke else 6, high_soc=high_soc), seed=seed)
        result = collect(env, output/f'seed_{seed}', seed=seed, max_steps=2 if smoke else 1800,
                         wall_seconds=180 if smoke else 3600, smoke=smoke)
        results.append(result)
        write_json(output/'result.json', dict(results=results, PPO_ALLOWED=False,
            status='RUNNING' if len(results)<len(cases) else result['status']))
        if result['status'] not in {'PILOT_OK', 'SMOKE_OK'}:
            break
    report = dict(status=('SMOKE_OK' if smoke else 'PILOT_OK') if len(results)==len(cases)
        and all(r['status'] in {'PILOT_OK','SMOKE_OK'} for r in results) else 'INCOMPLETE',
        results=results, PPO_ALLOWED=False, I0_QUALIFIED=False)
    write_json(output/'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    report = run(args.output, smoke=args.smoke)
    print(json.dumps(dict(status=report['status'], episodes=len(report['results']))))
    return 0 if report['status'] in {'PILOT_OK','SMOKE_OK'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
