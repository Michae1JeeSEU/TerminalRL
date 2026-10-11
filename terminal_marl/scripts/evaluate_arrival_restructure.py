"""Development-only native arrival pilot, bounded reference and independent replay."""
from __future__ import annotations

import os
# Set before importing numpy and before spawning independent arms. Never nest pools.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
from terminal_marl.configs.arrival_restructure import (
    CONTROLLERS, EXTERNAL_AUTHORITY, LEGACY_AUTHORITY, MAX_TICKS, MAX_WALL_SECONDS,
    PILOT_SCENARIOS, PILOT_SEEDS, REFERENCE_REGISTERED_PERMUTATIONS, REFERENCE_SEEDS, REFERENCE_SIZES, SMOKE_SEEDS,
    build_config, config_snapshot, protocol_snapshot,
)
from terminal_marl.core.arrival_dispatch import ArrivalDispatcher, energy_slack_order, nominal_requests
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv

SEARCH = dict(lookahead_steps=4, max_orders=4, replan_interval=10, planning_wall_seconds=60.)


def finite_json(value):
    """Strict JSON; undefined native values are explicit nulls, never NaN tokens."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque, set)):
        return [finite_json(v) for v in (sorted(value) if isinstance(value, set) else value)]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str) and value in {'Infinity', '-Infinity', 'NaN'}:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'unsupported JSON type {type(value).__name__}')


def canonical(value):
    return finite_json(SafetyTraceRecorder._replay_state_value(finite_json(value)))


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(finite_json(value), indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf8')


def objects(env):
    return [*env.active.values(), *env.service_uavs.values(), *env.completed_uavs, *env.admission.queue.items]


def accumulate(rows, values):
    # References captured before step still expose a removed object's final mutation.
    for obj in values:
        aid = obj.arrival_id if hasattr(obj, 'arrival_id') else obj.request.arrival_id
        row = rows.setdefault(aid, dict(external_wait=0., internal_wait=0., energy=0., landing_soc=None))
        for key, attr in [('external_wait', 'external_wait_time'), ('internal_wait', 'internal_wait_time'), ('energy', 'energy_consumed')]:
            row[key] = max(row[key], float(getattr(obj, attr, 0.)))
        if getattr(obj, 'touchdown_time', None) is not None:
            row['landing_soc'] = float(obj.soc)


def accounting(env, rows):
    previously_seen = set(rows)
    for record in env.energy_outcomes:
        row = rows.setdefault(record['arrival_id'], dict(external_wait=0., internal_wait=0., energy=0., landing_soc=None))
        row['energy'] = max(row['energy'], float(record.get('energy_consumed', 0.)))
    outcomes = Counter(v.value for v in env.ledger.outcomes.values())
    requested = len(env.ledger.requested)
    expected = int(env.config.total_requested_uavs)
    # Queue diversions can be born and removed within a tick, before snapshots.
    for aid, outcome in env.ledger.outcomes.items():
        if outcome.value.startswith('diverted') and aid not in previously_seen:
            row = rows.setdefault(aid, dict(external_wait=0., internal_wait=0., energy=0., landing_soc=None))
            if not row['internal_wait']:
                row['external_wait'] = max(row['external_wait'], env.resolution_times.get(aid, env.now)-env.request_times[aid])
    waits = [r['external_wait']+r['internal_wait'] for r in rows.values()]
    resolved_waits = [rows[aid]['external_wait']+rows[aid]['internal_wait'] for aid in env.ledger.outcomes if aid in rows]
    return dict(expected_total_requested=expected, actually_requested=requested,
                known_unresolved=len(env.ledger.unresolved), not_yet_requested=max(0, expected-requested),
                N=outcomes['completed'], E=outcomes['emergency_landed'],
                F=sum(v for k,v in outcomes.items() if k.startswith('failed') or k == 'invalid'),
                D=sum(v for k,v in outcomes.items() if k.startswith('diverted')),
                U=len(env.ledger.unresolved), outcomes=dict(outcomes), per_arrival=rows,
                wait_measurement='full_cycle' if natural_terminal(env) else 'accrued_partial',
                mean_wait_over_requested=sum(waits)/requested if requested else None,
                total_wait=sum(waits), full_cycle_wait=sum(resolved_waits), full_cycle_wait_count=len(resolved_waits), all_request_wait=sum(waits)/expected if expected else None,
                wait_max=max(waits, default=0.), energy=sum(r['energy'] for r in rows.values()),
                makespan=float(env.now), external_queue_uav_seconds=env.external_queue_uav_seconds)


def natural_terminal(env):
    return (env.arrivals.exhausted() and len(env.ledger.requested) == env.config.total_requested_uavs
            and not env.ledger.unresolved and not env.active and not env.service_uavs and not env.admission.queue.items)


def integrity_checks(env):
    living = [*env.active.values(), *env.service_uavs.values(), *env.admission.queue.items]
    ids = [obj.arrival_id if hasattr(obj, 'arrival_id') else obj.request.arrival_id for obj in living]
    plan_pairs = [(u.committed_plan.trajectory_hash, env.safety.checked_trajectory_hashes.get(u.committed_plan.plan_id))
                  for u in env.active.values() if u.committed_plan is not None]
    return dict(ledger_integrity=set(env.ledger.outcomes)<=env.ledger.requested
                and set(ids)==env.ledger.unresolved and len(ids)==len(set(ids)),
                active_plan_hashes_match=all(a==b for a,b in plan_pairs),
                active_plan_hash_samples=len(plan_pairs),
                checked_executed_plan_samples=int(env.telemetry.get('checked_executed_plan_samples',0)),
                checked_executed_plan_hash_mismatches=int(env.telemetry.get('checked_executed_plan_hash_mismatches',0)))


def state_record(env, rows):
    plans = {}
    for uid, uav in sorted(env.active.items()):
        p = uav.committed_plan
        if p is not None:
            plans[uid] = dict(trajectory_hash=p.trajectory_hash,
                             checked_hash=env.safety.checked_trajectory_hashes.get(p.plan_id),
                             start=p.start_time, end=p.end_time, mode=p.mode)
    return canonical(dict(native=SafetyTraceRecorder._state(env), plans=plans,
        service=[asdict(u) for _,u in sorted(env.service_uavs.items())],
        completed=[asdict(u) for u in env.completed_uavs], queue=env.admission.queue.items,
        requested=sorted(env.ledger.requested), outcomes=env.ledger.outcomes,
        energy_outcomes=env.energy_outcomes, metrics=accounting(env, rows), telemetry=env.telemetry,
        release_telemetry=env.last_release_telemetry, action_traces={k:v.as_dict() for k,v in env.last_action_traces.items()}))


def run_arm(spec):
    started = time.perf_counter()
    out = Path(spec['output']); out.mkdir(parents=True, exist_ok=False)
    authority = LEGACY_AUTHORITY if spec['controller'] == 'rule_nominal_k3' else EXTERNAL_AUTHORITY
    header = protocol_snapshot(mode=spec['mode'], scenario=spec['scenario'], seed=spec['seed'],
        controller=spec['controller'], authority=authority, source_commit=spec['source_commit'],
        reference_n=spec.get('reference_n'), smoke=spec['mode']=='smoke',
        max_ticks=spec.get('max_ticks',MAX_TICKS), wall_seconds=spec.get('wall_seconds',MAX_WALL_SECONDS))
    header.update(search_settings=SEARCH, candidate=spec.get('candidate'),
        comparison_boundary='legacy arbitration versus external ordering is a control-authority confound; no algorithm-superiority claim',
        nonfinite_values='null means undefined/nonfinite native field', spec=spec)
    write_json(out/'protocol.json',header)
    env = None; rows = {}; planning = native = 0.; ticks = 0; status='ERROR'; reason=None
    integrity_ok = True; hash_samples = 0; first_tick_seconds = None
    try:
        config = build_config(spec['scenario'], reference_n=spec.get('reference_n'), authority=authority)
        write_json(out/'config.json',config_snapshot(config))
        write_json(out/'progress.json',dict(status='INITIALIZING',elapsed_seconds=time.perf_counter()-started))
        env = TerminalParallelEnv(config,seed=spec['seed'])
        accumulate(rows, objects(env))
        write_json(out/'initial_state.json',state_record(env,rows))
        dispatcher = ArrivalDispatcher(spec['controller'],**SEARCH) if spec['controller'] in ('energy_slack','rolling_reorder') else None
        with (out/'steps.jsonl').open('w',encoding='utf8') as tape:
            while True:
                if natural_terminal(env): status='COMPLETED'; reason='natural_terminal'; break
                if ticks >= header['max_ticks'] or env.now >= config.max_simulation_seconds or time.perf_counter()-started >= header['wall_seconds']:
                    status='CENSORED'; reason='tick_cap' if ticks>=header['max_ticks'] else 'native_time_cap' if env.now>=config.max_simulation_seconds else 'wall_cap'; break
                before = objects(env); t = time.perf_counter(); diagnostics={}
                candidate=spec.get('candidate')
                if candidate:
                    order=tuple(uid for uid in candidate['order'] if uid in env.active)
                    order += tuple(sorted(set(env.active)-set(order)))
                    actions,shortlists=nominal_requests(env)
                    if candidate['variant']=='serial_release':
                        eligible=[uid for uid in order if str(uid) in shortlists]
                        actions,shortlists=nominal_requests(env,eligible[:1])
                elif dispatcher:
                    decision=dispatcher.decide(env)
                    order,actions,shortlists,diagnostics=decision.order,decision.actions,decision.shortlists,decision.diagnostics
                else:
                    order=None; actions,shortlists=nominal_requests(env)
                planning += time.perf_counter()-t
                t=time.perf_counter()
                returned=env.step(actions,shortlists,**({'release_order':order} if authority==EXTERNAL_AUTHORITY else {}))
                native += time.perf_counter()-t; ticks+=1
                if first_tick_seconds is None: first_tick_seconds=time.perf_counter()-started
                checks=integrity_checks(env)
                integrity_ok = integrity_ok and checks['ledger_integrity'] and checks['active_plan_hashes_match']
                hash_samples += checks['active_plan_hash_samples']
                accumulate(rows,[*before,*objects(env)])
                state=state_record(env,rows)
                entry=dict(tick=ticks, actions=actions,shortlists=shortlists,release_order=order,
                    diagnostics=diagnostics, dispatch={k:v for k,v in returned[-1].items() if 'order' in k or k in ('normal_approved_releases','release_decisions','release_intent_telemetry')},
                    state=state,state_hash=digest(state))
                tape.write(json.dumps(finite_json(entry),sort_keys=True,allow_nan=False)+'\n'); tape.flush()
                write_json(out/'progress.json',dict(status='RUNNING',ticks=ticks,time=env.now,elapsed_seconds=time.perf_counter()-started,planning_seconds=planning,native_checks_seconds=native,metrics=accounting(env,rows)))
    except Exception:
        reason=traceback.format_exc()
    result=dict(status=status,stop_reason=reason,ticks=ticks,elapsed_seconds=time.perf_counter()-started,
        planning_seconds=planning,native_checks_seconds=native,first_tick_seconds=first_tick_seconds,
        integrity_ok=integrity_ok,active_plan_hash_samples=hash_samples,protocol=header,output=str(out),
        metrics=accounting(env,rows) if env is not None else None,
        telemetry=env.telemetry if env is not None else None)
    result['full_valid'] = status=='COMPLETED' and integrity_ok and int((result.get('telemetry') or {}).get('checked_executed_plan_samples',0))>0 and not any((result.get('telemetry') or {}).get(k,0) for k in ('collisions','nmac','hard_separation_violations','checked_executed_plan_hash_mismatches'))
    write_json(out/'result.json',result); write_json(out/'progress.json',result)
    return finite_json(result)


def replay_run(source, output):
    started=time.perf_counter(); source=Path(source); out=Path(output); out.mkdir(parents=True,exist_ok=False)
    header=json.loads((source/'protocol.json').read_text(encoding='utf8')); spec=header['spec']
    write_json(out/'protocol.json',dict(mode='independent_replay',source=str(source),source_protocol=header))
    result=dict(status='ERROR',matched_steps=0,elapsed_seconds=0.,native_checks_seconds=0.,planning_seconds=0.)
    try:
        env=TerminalParallelEnv(build_config(spec['scenario'],reference_n=spec.get('reference_n'),authority=header['control_authority']),seed=spec['seed'])
        rows={}; accumulate(rows,objects(env))
        if state_record(env,rows)!=json.loads((source/'initial_state.json').read_text(encoding='utf8')):
            raise ValueError('initial state differs')
        with (source/'steps.jsonl').open(encoding='utf8') as tape:
            for line in tape:
                entry=json.loads(line); before=objects(env)
                if time.perf_counter()-started >= header['wall_seconds']:
                    result.update(status='CENSORED',stop_reason='replay_wall_cap')
                    break
                actions={k:np.asarray(v,dtype=np.float32) for k,v in entry['actions'].items()}
                shortlists={k:[np.asarray(a,dtype=np.float32) for a in v] for k,v in entry['shortlists'].items()}
                native_started=time.perf_counter()
                env.step(actions,shortlists,**({'release_order':entry['release_order']} if header['control_authority']==EXTERNAL_AUTHORITY else {}))
                result['native_checks_seconds']+=time.perf_counter()-native_started
                accumulate(rows,[*before,*objects(env)]); actual=state_record(env,rows)
                if digest(actual)!=entry['state_hash'] or actual!=entry['state']:
                    write_json(out/'mismatch.json',dict(tick=entry['tick'],actual=actual,expected=entry['state']))
                    raise ValueError(f"per-step state mismatch at tick {entry['tick']}")
                result['matched_steps']+=1
                write_json(out/'progress.json',dict(status='RUNNING',matched_steps=result['matched_steps']))
        if result['status']=='CENSORED':
            result['elapsed_seconds']=time.perf_counter()-started
            write_json(out/'result.json',result)
            return finite_json(result)
        original=json.loads((source/'result.json').read_text(encoding='utf8'))
        if result['matched_steps']!=original['ticks'] or canonical(accounting(env,rows))!=original['metrics']:
            raise ValueError('tape length or final accounting mismatch')
        result.update(status='MATCH',source_status=original['status'],whole_cycle=original['status']=='COMPLETED' and natural_terminal(env),metrics=accounting(env,rows))
    except Exception:
        result['error']=traceback.format_exc()
    result['elapsed_seconds']=time.perf_counter()-started
    write_json(out/'result.json',result)
    return finite_json(result)


def reference_candidates(n, energy_order):
    permutations=list(itertools.permutations(range(n)))
    indices=np.linspace(0,len(permutations)-1,min(REFERENCE_REGISTERED_PERMUTATIONS,len(permutations)),dtype=int)
    orders=[permutations[i] for i in indices]
    energy_order=tuple(energy_order)
    if energy_order not in orders: orders[-1]=energy_order
    return [dict(order=list(order),variant=variant) for order in orders for variant in ('all_release','serial_release')]


def select_candidate(results):
    eligible=[r for r in results if r.get('full_valid') and r.get('status')=='COMPLETED']
    return min(eligible,key=lambda r:tuple(r['metrics'][k] for k in ('F','D','U','E','total_wait','makespan','energy'))) if eligible else None


def reference_evidence(best, replay, all_evaluated):
    """A fully resolved rollout may contain failures; it is not a completion witness."""
    matched = bool(best and best.get('full_valid') and best.get('status') == 'COMPLETED'
                   and replay and replay.get('status') == 'MATCH' and replay.get('whole_cycle'))
    metrics = best['metrics'] if best else {}
    joint = matched and metrics.get('N', 0) + metrics.get('E', 0) == metrics.get('expected_total_requested') and all(
        metrics.get(key) == 0 for key in ('F', 'D', 'U', 'not_yet_requested'))
    evidence = 'unknown'
    if joint:
        evidence = 'candidate_set_best_joint_completion_witness' if all_evaluated else 'found_joint_completion_witness_among_evaluated'
    elif matched:
        evidence = 'replayed_resolved_rollout'
    return dict(evidence=evidence, resolved_rollout_replayed=matched,
                joint_completion_witness=bool(joint),
                normal_completion_witness=bool(joint and metrics.get('E') == 0),
                comparison_scope='registered_candidate_set' if all_evaluated else 'evaluated_subset',
                all_registered_candidates_evaluated=bool(all_evaluated))


def execute(specs,workers):
    if workers==1: return [run_arm(s) for s in specs]
    with ProcessPoolExecutor(max_workers=min(24,workers)) as pool:
        return list(pool.map(run_arm,specs))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('smoke','pilot','reference','aggregate','replay'))
    parser.add_argument('--output',required=True); parser.add_argument('--source-commit',required=True)
    parser.add_argument('--input',nargs='+'); parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--scenario',choices=PILOT_SCENARIOS); parser.add_argument('--controller',choices=CONTROLLERS)
    parser.add_argument('--reference-n',type=int,choices=REFERENCE_SIZES)
    args=parser.parse_args(argv)
    if not 1<=args.workers<=24: parser.error('--workers must be 1..24')
    if args.mode in ('replay','aggregate') and not args.input: parser.error('--input required')
    if args.mode=='replay':
        result=replay_run(args.input[0],args.output); print(json.dumps(result)); return 0 if result['status']=='MATCH' else 1
    root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
    if args.mode=='aggregate':
        results=[]
        for source in args.input:
            for path in Path(source).rglob('result.json'):
                row=json.loads(path.read_text(encoding='utf8'))
                if 'protocol' in row: results.append(row)
        write_json(root/'aggregate.json',dict(runs=results,status_counts=dict(Counter(r['status'] for r in results)),comparison_boundary='development only; authority confound retained'))
        return 0
    specs=[]; registry={}
    if args.mode=='reference':
        for n in ([args.reference_n] if args.reference_n else REFERENCE_SIZES):
            env=TerminalParallelEnv(build_config(reference_n=n,authority=EXTERNAL_AUTHORITY),seed=REFERENCE_SEEDS[n])
            candidates=reference_candidates(n,energy_slack_order(env)); registry[str(n)]=candidates
            for i,candidate in enumerate(candidates):
                specs.append(dict(mode='reference',scenario='dense24_fixed',seed=REFERENCE_SEEDS[n],controller='reference_static',reference_n=n,candidate=candidate,output=str(root/f'n{n}_candidate{i:02d}'),source_commit=args.source_commit))
    else:
        for scenario in ([args.scenario] if args.scenario else PILOT_SCENARIOS):
            for controller in ([args.controller] if args.controller else CONTROLLERS):
                specs.append(dict(mode=args.mode,scenario=scenario,controller=controller,
                    seed=(SMOKE_SEEDS if args.mode=='smoke' else PILOT_SEEDS)[scenario],max_ticks=2 if args.mode=='smoke' else MAX_TICKS,
                    source_commit=args.source_commit,output=str(root/f'{scenario}_{controller}')))
    write_json(root/'manifest.json',dict(specs=specs,registry=registry,workers=args.workers,source_commit=args.source_commit))
    results=execute(specs,args.workers)
    selected=[]
    if args.mode=='reference':
        for n in registry:
            cohort=[r for r in results if r['protocol']['reference_n']==int(n)]
            best=select_candidate(cohort)
            replay=replay_run(best['output'],root/f'n{n}_replay') if best else None
            all_evaluated=all(r['status']=='COMPLETED' for r in cohort) and len(cohort)==len(registry[n])
            selected.append(dict(n=int(n),selected=best['output'] if best else None,replay=replay,
                **reference_evidence(best,replay,all_evaluated)))
    elif args.mode=='smoke':
        for result in results:
            if result['status']!='ERROR': selected.append(replay_run(result['output'],Path(result['output']+'_replay')))
    write_json(root/'summary.json',dict(results=results,replays_or_selection=selected))
    return int(any(r['status']=='ERROR' for r in results) or any(r.get('status')=='ERROR' or (r.get('replay') or {}).get('status')=='ERROR' for r in selected))


if __name__=='__main__':
    raise SystemExit(main())
