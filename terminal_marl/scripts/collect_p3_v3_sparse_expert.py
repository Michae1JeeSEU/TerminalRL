"""Development-only rule rollout followed by bounded, state-only expert labels.

Raw rows are immutable and unlabelled. Labels join by decision_id; teacher
requests are never substituted for the recorded rule requests or execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

import numpy as np

from terminal_marl.configs.p3_integrated_repair import PROFILE, build_config
from terminal_marl.core.low_level_joint_action_v4_2 import hierarchical_from_joint_index_v4_1
from terminal_marl.core.observations import build_low_observation_v2_4
from terminal_marl.core.safety_trace import _digest, _json
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.scripts.collect_p3_v3_expert import actor_eligible, natural_terminal
from terminal_marl.scripts.diagnose_p3_direct_contract import (
    SAFETY_KEYS, canonical, execute_tape, require, sha256, state_record, write_json,
)
from terminal_marl.training.v4_1_joint_teacher import teacher_joint_expert_v4_1

SCHEMA = 'p3_v3_90d_sparse_expert_development_v1'
MODULE = 'terminal_marl.scripts.collect_p3_v3_sparse_expert'
CASES = {'smoke': (892401, True), 'high': (892402, True), 'mixed': (892403, False)}
WINDOW_TICKS = 75
MAX_WINDOWS = 24


def rule_tape(env):
    """Same nominal K3 rule used in the completed integrated-repair runs."""
    actions, shortlists = {}, {}
    for uid, uav in env.active.items():
        eligible = (uav.committed_plan is None and uav.ring_role is not None
                    and env.now >= uav.safety_retry_after)
        actions[str(uid)] = [1., 0., 0.] if eligible else [0., 0., 0.]
        if eligible:
            shortlists[str(uid)] = [[1., 0., 0.], [1., -1., 0.], [1., 1., 0.]]
    return dict(actions=actions, shortlists=shortlists)


def selected_uid(tick, eligible_uids, selected_windows):
    window = tick // WINDOW_TICKS
    eligible_uids = sorted(eligible_uids, key=int)
    if window >= MAX_WINDOWS or window in selected_windows or not eligible_uids:
        return None
    return eligible_uids[window % len(eligible_uids)]


def contexts(row):
    return dict(fast_inner=row['interface'] == 'i2f' and row['speed'] >= 15.,
                busy_lz=row['busy_lz'] > 0, retry=row['retry_active'], low_soc=row['soc'] < .55)


def raw_rows(env, observations, tape, *, seed, tick, pre_hash):
    require(set(observations) == {str(uid) for uid in env.active}, 'observation UID mismatch')
    rows = []
    for uid in sorted(observations, key=int):
        uav = env.active[int(uid)]
        obs = np.asarray(observations[uid], dtype=np.float32)
        require(obs.shape == (90,) and np.isfinite(obs).all(), 'invalid 90D observation')
        require(np.array_equal(obs, build_low_observation_v2_4(uav, env)),
                '90D observation reconstruction differs')
        rows.append(dict(schema=SCHEMA, partition='development_pilot_never_train',
            seed=seed, tick=tick, time=env.now, uid=uid, arrival_id=uav.arrival_id,
            decision_id=f'{SCHEMA}:{seed}:{tick}:{uid}',
            observation=obs.tolist(), observation_dim=90,
            observation_schema=env.config.observation_schema_version,
            actor_mask=actor_eligible(env, uav), teacher_label=None,
            behavior_request=tape['actions'][uid], behavior_shortlist=tape['shortlists'].get(uid, []),
            pre_state_hash=pre_hash, interface={'outer':'o2m','middle':'m2i','inner':'i2f'}.get(
                getattr(uav.ring_role, 'value', None)), phase=uav.phase.value,
            speed=float(uav.speed), soc=float(uav.soc),
            retry_active=env.now < uav.safety_retry_after,
            busy_lz=sum(z.status.value != 'free' for z in env.lz_manager.zones)))
    return rows


def rollout(env, output, *, seed, max_steps=1800, wall_seconds=3600, smoke=False):
    output.mkdir(parents=True, exist_ok=False)
    (output/'snapshots').mkdir()
    (output/'labels').mkdir()
    write_json(output/'config.json', asdict(env.config))
    start = time.monotonic()
    selected, counts, coverage = [], Counter(), Counter()
    ticks = total_rows = 0
    error, stop, data, tape = None, 'tick_limit', None, None
    replay_equal = False
    try:
        with (output/'rows.jsonl').open('x', encoding='utf8') as rows_file, \
                (output/'steps.jsonl').open('x', encoding='utf8') as steps_file:
            for tick in range(max_steps):
                if time.monotonic()-start >= wall_seconds:
                    stop = 'wall_limit'
                    break
                observations = env._observations()
                data = pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)
                pre_hash = _digest(canonical(state_record(env)))
                tape = rule_tape(env)
                rows = raw_rows(env, observations, tape, seed=seed, tick=tick, pre_hash=pre_hash)
                chosen = selected_uid(tick, [r['uid'] for r in rows if r['actor_mask']],
                                      {entry['window'] for entry in selected})
                if chosen is not None:
                    row = next(r for r in rows if r['uid'] == chosen)
                    window = tick // WINDOW_TICKS
                    snapshot = output/'snapshots'/f'window_{window:02d}.pkl'
                    snapshot.write_bytes(data)
                    entry = dict(row=row, window=window, snapshot=snapshot.name,
                                 snapshot_sha256=sha256(snapshot))
                    write_json(output/'snapshots'/f'window_{window:02d}.json', entry)
                    selected.append(entry)
                    write_json(output/'selected.json', selected)
                for row in rows:
                    rows_file.write(json.dumps(_json(row), allow_nan=False)+'\n')
                    total_rows += 1
                    counts['actor' if row['actor_mask'] else 'forced_WAIT'] += 1
                    coverage.update({k:int(v) for k,v in contexts(row).items()})
                rows_file.flush()
                # No teacher call occurs anywhere in this loop.
                live = execute_tape(env, tape)
                ticks = tick+1
                steps_file.write(json.dumps(dict(tick=tick, pre_state_hash=pre_hash, tape=tape,
                    action_traces=live['outputs'][4]['action_traces'],
                    release_telemetry=canonical(env.last_release_telemetry),
                    post_state_hash=_digest(live['after']), outcomes=dict(env.ledger.outcomes)),
                    allow_nan=False)+'\n')
                steps_file.flush()
                if tick == 0:
                    (output/'first_before.pkl').write_bytes(data)
                    write_json(output/'first_request.json', tape)
                    replay = execute_tape(pickle.loads(data), tape)
                    write_json(output/'first_live.json', live)
                    write_json(output/'first_replay.json', replay)
                    require(live == replay, 'live/replay differs')
                    replay_equal = True
                outcomes = Counter(getattr(v, 'value', v) for v in env.ledger.outcomes.values())
                require(not outcomes.get('failed_energy'), 'failed_energy during rule rollout')
                if ticks == 1 or ticks % 25 == 0 or natural_terminal(env):
                    write_json(output/'progress.json', dict(stage='rollout', ticks=ticks,
                        rows=total_rows, selected=len(selected), outcomes=dict(outcomes),
                        elapsed_seconds=time.monotonic()-start))
                if natural_terminal(env):
                    stop = 'natural_terminal'
                    break
    except Exception as exc:
        stop = 'error'
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        if data is not None:
            (output/'failure_before.pkl').write_bytes(data)
        write_json(output/'failure_request.json', tape)
    terminal = natural_terminal(env)
    outcomes = Counter(getattr(v, 'value', v) for v in env.ledger.outcomes.values())
    ok = error is None and replay_equal and (terminal or (smoke and ticks == max_steps))
    report = dict(status=('SMOKE_OK' if smoke else 'COMPLETED') if ok else
        ('ERROR' if error else 'CENSORED'), error=error, stop_reason=stop, seed=seed,
        ticks=ticks, rows=total_rows, row_counts=dict(counts), raw_contexts=dict(coverage),
        terminal=terminal, censored=not terminal, elapsed_seconds=time.monotonic()-start,
        selected_count=len(selected), selected_windows=[e['window'] for e in selected],
        requested=len(env.ledger.requested), generated_requests=len(env.request_times),
        outcome_counts=dict(outcomes), unresolved=len(env.ledger.unresolved),
        active=len(env.active), queue=len(env.admission.queue.items), service=len(env.service_uavs),
        telemetry=dict(env.telemetry), energy_outcomes=env.energy_outcomes,
        first_tick_replay_equal=replay_equal, trained=False, PPO_ALLOWED=False)
    write_json(output/'selected.json', selected)
    write_json(output/'rollout_result.json', report)
    return report, selected


def query_snapshot(entry_path):
    """Read only a freshly generated, hash-checked development snapshot."""
    entry = json.loads(entry_path.read_text(encoding='utf8'))
    require(Path(entry['snapshot']).name == entry['snapshot'], 'snapshot path must be a basename')
    path = entry_path.parent/entry['snapshot']
    require(sha256(path) == entry['snapshot_sha256'], 'snapshot hash differs')
    probe = pickle.loads(path.read_bytes())
    row = entry['row']
    uav = probe.active[int(row['uid'])]
    require(_digest(canonical(state_record(probe))) == row['pre_state_hash'], 'snapshot state differs')
    require(actor_eligible(probe, uav) and row['actor_mask'], 'selected actor is ineligible')
    obs = np.asarray(row['observation'], dtype=np.float32)
    require(obs.shape == (90,) and np.array_equal(obs, build_low_observation_v2_4(uav, probe)),
            'selected observation differs')
    certificates, plan_calls = [], 0
    original = probe._candidate_plan

    def record_candidate(*args, **kwargs):
        nonlocal plan_calls
        plan_calls += 1
        proposal, plan, reason = original(*args, **kwargs)
        if proposal is not None and plan is not None and reason is None:
            checked = probe.safety.checked_trajectory_hashes.get(plan.plan_id)
            require(checked == plan.trajectory_hash, 'teacher lacks checked hash')
            certificates.append(dict(trajectory_hash=plan.trajectory_hash,
                                     checked_hash=checked, mode=plan.mode))
        return proposal, plan, reason

    probe._candidate_plan = record_candidate
    selected, audit = teacher_joint_expert_v4_1(probe, uav, state_only=True)
    action = hierarchical_from_joint_index_v4_1(selected.index)
    require((action.gate == 1) == bool(certificates), 'label/certificate mismatch')
    intent_checks = sum(r['decoded']['release_request'] > 0 for r in audit)
    require(intent_checks <= 25, 'teacher exceeded intent catalogue')
    return _json(dict(status='OK', decision_id=row['decision_id'], tick=row['tick'],
        uid=row['uid'], window=entry['window'], actor_mask=True, observation=row['observation'],
        pre_state_hash=row['pre_state_hash'], teacher_gate=int(action.gate),
        teacher_proposal=action.proposal, teacher_joint_index=int(selected.index),
        teacher_request=action.normalized.tolist(), teacher_audit=audit,
        selected_certificate=certificates[-1] if certificates else None,
        release_intents_checked=intent_checks, candidate_plan_calls=plan_calls,
        contexts=contexts(row), executed_as_behavior=False))


def annotate(output, selected, *, wall_seconds=1800):
    """A serial subprocess per query provides a real total annotation deadline."""
    start = time.monotonic()
    entries, status = [], 'COMPLETED'
    for entry in selected:
        window, row = entry['window'], entry['row']
        stem = f'window_{window:02d}'
        item = dict(window=window, decision_id=row['decision_id'], status='NOT_ATTEMPTED')
        remaining = wall_seconds-(time.monotonic()-start)
        if status != 'COMPLETED' or remaining <= 0:
            if status == 'COMPLETED':
                status = 'CENSORED'
            entries.append(item)
            continue
        label_path = output/'labels'/f'{stem}.json'
        try:
            with (output/'labels'/f'{stem}.log').open('x', encoding='utf8') as log:
                result = subprocess.run([sys.executable, '-u', '-m', MODULE,
                    '--query-entry', str(output/'snapshots'/f'{stem}.json'),
                    '--query-output', str(label_path)], stdout=log, stderr=subprocess.STDOUT,
                    timeout=remaining, check=False)
            item['exit_code'] = result.returncode
            label = json.loads(label_path.read_text(encoding='utf8'))
            require(result.returncode == 0 and label['status'] == 'OK', 'query subprocess failed')
            require(label['decision_id'] == row['decision_id'], 'query decision ID differs')
            item.update(status='OK', label_file=f'labels/{stem}.json',
                        teacher_gate=label['teacher_gate'], teacher_proposal=label['teacher_proposal'],
                        release_intents_checked=label['release_intents_checked'],
                        candidate_plan_calls=label['candidate_plan_calls'])
        except subprocess.TimeoutExpired:
            # subprocess.run kills and waits for this one child; the child has no grandchildren.
            item['status'], status = 'TIMEOUT', 'CENSORED'
        except Exception as exc:
            item.update(status='ERROR', error=dict(type=type(exc).__name__, message=str(exc)))
            status = 'ERROR'
        entries.append(item)
        write_json(output/'annotation_progress.json', dict(status=status, entries=entries,
            selected=len(selected), elapsed_seconds=time.monotonic()-start))
    report = dict(status=status, selected=len(selected), entries=entries,
                  labelled=sum(x['status'] == 'OK' for x in entries),
                  elapsed_seconds=time.monotonic()-start, wall_seconds=wall_seconds)
    write_json(output/'annotation_result.json', report)
    return report


def run(output, case):
    seed, high_soc = CASES[case]
    smoke = case == 'smoke'
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'protocol.json', dict(schema=SCHEMA, profile=PROFILE, seed=seed,
        case=case, high_soc=high_soc, requests=3 if smoke else 12, warm_start=3 if smoke else 6,
        max_steps=2 if smoke else 1800, rollout_wall_seconds=180 if smoke else 3600,
        annotation_wall_seconds=180 if smoke else 1800, window_ticks=WINDOW_TICKS,
        max_selected=MAX_WINDOWS, max_release_intents=MAX_WINDOWS*25,
        behavior='integrated_repair_rule_nominal_k3', partition='development_pilot_never_train',
        selection='first_eligible_tick_in_window; numeric_sorted_uid[window % eligible_count]',
        label_join='decision_id; only status OK; actor_mask required; null is not WAIT'))
    env = TerminalParallelEnv(build_config(requests=3 if smoke else 12,
        warm_start=3 if smoke else 6, high_soc=high_soc), seed=seed)
    episode = output/f'seed_{seed}'
    flow, selected = rollout(env, episode, seed=seed, max_steps=2 if smoke else 1800,
                             wall_seconds=180 if smoke else 3600, smoke=smoke)
    annotation = dict(status='NOT_RUN', selected=len(selected), labelled=0)
    if flow['status'] in {'SMOKE_OK', 'COMPLETED'}:
        annotation = annotate(episode, selected, wall_seconds=180 if smoke else 1800)
    success = (flow['status'] in {'SMOKE_OK','COMPLETED'}
               and annotation['status'] == 'COMPLETED' and annotation['labelled'] > 0)
    report = dict(status=('SMOKE_OK' if smoke else 'PILOT_OK') if success else 'INCOMPLETE',
        schema=SCHEMA, rollout=flow, annotation=annotation, trained=False, PPO_ALLOWED=False,
        I0_QUALIFIED=False, qualification='NOT_QUALIFIED', checkpoint_loaded=False,
        optimizer_created=False, heldout_used=False)
    write_json(output/'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--case', choices=tuple(CASES))
    parser.add_argument('--query-entry', type=Path)
    parser.add_argument('--query-output', type=Path)
    args = parser.parse_args()
    if args.query_entry is not None:
        if args.query_output is None or args.output is not None or args.case is not None:
            parser.error('query mode requires only entry and query-output')
        if args.query_output.exists():
            parser.error('query output already exists')
        start = time.monotonic()
        try:
            result = query_snapshot(args.query_entry)
        except Exception as exc:
            result = dict(status='ERROR', error=dict(type=type(exc).__name__, message=str(exc),
                                                   traceback=traceback.format_exc()))
        result['elapsed_seconds'] = time.monotonic()-start
        write_json(args.query_output, result)
        return 0 if result['status'] == 'OK' else 1
    if args.output is None or args.case is None or args.query_output is not None:
        parser.error('rollout mode requires output and case')
    result = run(args.output, args.case)
    print(json.dumps(dict(status=result['status'], ticks=result['rollout']['ticks'],
                          labels=result['annotation']['labelled'])))
    return 0 if result['status'] in {'SMOKE_OK', 'PILOT_OK'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
