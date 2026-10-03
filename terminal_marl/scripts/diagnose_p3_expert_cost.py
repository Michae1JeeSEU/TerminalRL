"""Reconstruct one recorded pilot state and profile one unchanged expert query."""
from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import pickle
import pstats
import time
import traceback

from terminal_marl.core.safety_trace import _digest, _json
from terminal_marl.scripts.diagnose_p3_direct_contract import (
    canonical, execute_tape, require, sha256, state_record, write_json,
)
from terminal_marl.training.v4_1_joint_teacher import teacher_joint_expert_v4_1

INPUT_HASHES = {
    'first_before.pkl': 'f5e4d9628d7ad1cd1d0bc0029f76a44da73e0aaddcb1f7769afd542093f47d1d',
    'steps.jsonl': '07009754ffa445429354b506dde5b4668c7cbeea267fb0642e4e5ad165369c79',
    'rows.jsonl': '74746f0603a418af984f1bb954eef9e7177137b4aac1bbd35a61b6f463934a4a',
}


def reconstruct(data, steps, target_tick, output, *, wall_seconds=600):
    env = pickle.loads(data)  # Only hash-bound, locally produced pilot artifacts.
    start = time.monotonic()
    times = []
    for tick in range(target_tick):
        require(time.monotonic()-start < wall_seconds, 'replay wall cap')
        step = steps[tick]
        require(step['tick'] == tick, 'nonconsecutive input tape')
        require(_digest(canonical(state_record(env))) == step['pre_state_hash'], 'pre-state differs')
        begin = time.monotonic()
        live = execute_tape(env, step['tape'])
        times.append(time.monotonic()-begin)
        require(_digest(live['after']) == step['post_state_hash'], 'post-state differs')
        require(live['outputs'][4]['action_traces'] == step['action_traces'], 'action trace differs')
        if tick == 0 or (tick+1) % 25 == 0:
            write_json(output/'progress.json', dict(replayed=tick+1, target=target_tick,
                                                   elapsed_seconds=time.monotonic()-start))
    require(_digest(canonical(state_record(env))) == steps[target_tick]['pre_state_hash'],
            'target pre-state differs')
    return env, times, time.monotonic()-start


def profile_query(env, uid, output):
    probe = pickle.loads(pickle.dumps(env))
    records, components = [], []
    for owner, name in ((probe.generator, 'ring_transfer'), (probe.generator, 'final_approach'),
                        (probe.safety, 'filter_plan')):
        original = getattr(owner, name)
        def wrapped(*args, _original=original, _name=name, **kwargs):
            start = time.monotonic()
            try:
                return _original(*args, **kwargs)
            finally:
                components.append(dict(component=_name, seconds=time.monotonic()-start))
        setattr(owner, name, wrapped)
    original_candidate = probe._candidate_plan
    def candidate(*args, **kwargs):
        start = time.monotonic()
        proposal, plan, reason = original_candidate(*args, **kwargs)
        records.append(dict(seconds=time.monotonic()-start, reason=reason,
            action=list(args[1].intent.normalized) if hasattr(args[1].intent, 'normalized') else
                [args[1].intent.release_request,args[1].intent.speed_residual,args[1].intent.merge_time_residual],
            plan_mode=None if plan is None else plan.mode,
            duration=None if plan is None else plan.end_time-plan.start_time,
            points=None if plan is None else len(plan.points),
            segments=None if plan is None else len(plan.polynomial_segments)))
        return proposal, plan, reason
    probe._candidate_plan = candidate
    profiler = cProfile.Profile()
    start = time.monotonic()
    label, audit = profiler.runcall(teacher_joint_expert_v4_1, probe, probe.active[int(uid)], state_only=True)
    elapsed = time.monotonic()-start
    profiler.dump_stats(str(output/'teacher.prof'))
    stats = pstats.Stats(profiler)
    top = []
    for (file, line, name), (primitive, calls, own, cumulative, _) in sorted(
            stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:40]:
        top.append(dict(file=file,line=line,function=name,calls=calls,primitive_calls=primitive,
                        own_seconds=own,cumulative_seconds=cumulative))
    return _json(dict(uid=uid, joint_index=label.index, audit=audit, elapsed_seconds=elapsed,
                      candidates=records, components=components, profile_top=top))


def run(source, output, *, smoke=False):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='ERROR', error=None, target_tick=0 if smoke else 150,
                  target_uid='0' if smoke else '10', trained=False, PPO_ALLOWED=False)
    start = time.monotonic()
    try:
        for name, expected in INPUT_HASHES.items():
            require(sha256(source/name) == expected, f'input hash differs: {name}')
        steps = [json.loads(line) for line in (source/'steps.jsonl').read_text().splitlines()]
        rows = [json.loads(line) for line in (source/'rows.jsonl').read_text().splitlines()]
        env, times, elapsed = reconstruct((source/'first_before.pkl').read_bytes(), steps,
                                          report['target_tick'], output)
        (output/'target_before.pkl').write_bytes(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL))
        report.update(replay_ticks=len(times), replay_seconds=elapsed, replay_step_seconds=times)
        row = next(r for r in rows if r['tick']==report['target_tick'] and r['uid']==report['target_uid'])
        require(row['pre_state_hash']==_digest(canonical(state_record(env))), 'row state differs')
        before = pickle.dumps(env)
        query = profile_query(env, report['target_uid'], output)
        write_json(output/'query.json', query)
        require(pickle.dumps(env)==before, 'profile mutated original state')
        require(query['joint_index']==row['teacher_joint_index'] and query['audit']==row['teacher_audit'],
                'profiled query differs from recorded label/audit')
        report.update(status='SMOKE_OK' if smoke else 'PROFILE_OK', query=query,
                      recorded_label_exact=True, original_state_unchanged=True)
    except Exception as exc:
        report['error']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    report['elapsed_seconds']=time.monotonic()-start
    write_json(output/'result.json', report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    result=run(args.source,args.output,smoke=args.smoke)
    print(json.dumps({key:result[key] for key in ('status','target_tick','target_uid','error','elapsed_seconds')}))
    raise SystemExit(0 if result['error'] is None else 1)
