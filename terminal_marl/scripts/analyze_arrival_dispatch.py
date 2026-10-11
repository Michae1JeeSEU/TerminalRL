"""Read-only, streaming extraction of preexisting arrival pilot traces."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import zip_longest
import json
from pathlib import Path

UNKNOWN = 'unknown'
PHYSICAL_FIELDS = ('uid', 'arrival_id', 'phase', 'position', 'velocity', 'acceleration',
                   'speed', 'committed', 'trajectory_hash', 'soc')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def steps(path):
    with (Path(path) / 'steps.jsonl').open(encoding='utf8') as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def array_state(obj, key):
    if key not in obj:
        return 'missing'
    value = obj[key]
    if value is None:
        return 'null'
    if not isinstance(value, list):
        return 'invalid'
    return 'nonempty' if value else 'empty'


def wait(row):
    a, b = row.get('external_wait'), row.get('internal_wait')
    return a + b if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values)-1)*q
    low = int(pos)
    return values[low] + (values[min(low+1, len(values)-1)]-values[low])*(pos-low)


def physical(state):
    rows = state.get('native', {}).get('uavs')
    if not isinstance(rows, list):
        return None, 'native.uavs missing or not a list'
    if any(not isinstance(r, dict) or any(k not in r for k in ('uid', 'phase', 'position')) for r in rows):
        return None, 'uav uid/phase/position missing'
    if len({str(r['uid']) for r in rows}) != len(rows):
        return None, 'duplicate native uav uid'
    return {str(r['uid']): {k: r.get(k, UNKNOWN) for k in PHYSICAL_FIELDS} for r in rows}, None


def context(entry):
    dispatch = entry.get('dispatch', {})
    native = entry.get('state', {}).get('native', {})
    intents = dispatch.get('release_intent_telemetry')
    fields = ('arrival_id', 'uav_id', 'interface', 'primary_blocker', 'secondary_constraints',
              'runtime_shortlist_candidates_checked', 'runtime_shortlist_rank', 'final_lz_pairs_checked')
    return dict(tick=entry.get('tick'), time=native.get('time', UNKNOWN),
                normal_approved_releases=dispatch.get('normal_approved_releases', UNKNOWN),
                release_decisions=dispatch.get('release_decisions', UNKNOWN),
                release_intents=({uid: {k: r[k] for k in fields if k in r}
                                 for uid, r in intents.items()} if isinstance(intents, dict) else UNKNOWN),
                landing_zones=native.get('landing_zones', UNKNOWN),
                reservations=native.get('reservations', UNKNOWN),
                future_release_reservations=native.get('future_release_reservations', UNKNOWN))


def candidate_summary(candidate):
    if candidate is None:
        return dict(status=UNKNOWN, order=None, score=None)
    return {k: candidate.get(k, None if k in ('order', 'score') else UNKNOWN)
            for k in ('status', 'order', 'score')}


def analyze_arm(path):
    path = Path(path)
    result = read_json(path/'result.json')
    metrics = result.get('metrics', {})
    per_arrival = metrics.get('per_arrival', {})
    waits = [v for r in per_arrival.values() if (v := wait(r)) is not None]
    socs = [r['landing_soc'] for r in per_arrival.values() if r.get('landing_soc') is not None]
    counts = {k: Counter() for k in ('replanned', 'fallback', 'replan_reason', 'fallback_reason',
                                    'candidate_status', 'release_order_state', 'energy_order_state',
                                    'order_comparison', 'release_decisions', 'approved_release_field_state')}
    numerics = {k: dict(observed_ticks=0, sum=None, max=None) for k in ('native_rollout_steps', 'planning_elapsed_seconds', 'budget_elapsed_seconds', 'native_budget_queries')}
    replans, differences = [], []
    first_different = None
    approved_count = 0
    last = None
    count = 0
    for entry in steps(path):
        count += 1
        last = entry
        diag = entry.get('diagnostics') or {}
        for key in ('replanned', 'fallback', 'replan_reason', 'fallback_reason'):
            counts[key][str(diag.get(key, UNKNOWN))] += 1
        for key in numerics:
            if isinstance(diag.get(key), (int, float)):
                stat = numerics[key]
                stat['observed_ticks'] += 1
                stat['sum'] = (stat['sum'] or 0) + diag[key]
                stat['max'] = diag[key] if stat['max'] is None else max(stat['max'], diag[key])
        rs, es = array_state(entry, 'release_order'), array_state(diag, 'energy_order')
        counts['release_order_state'][rs] += 1
        counts['energy_order_state'][es] += 1
        comparable = rs in ('empty', 'nonempty') and es in ('empty', 'nonempty')
        relation = ('same' if entry['release_order'] == diag['energy_order'] else 'different') if comparable else UNKNOWN
        counts['order_comparison'][relation] += 1
        if relation == 'different' and first_different is None:
            first_different = entry['tick']
        candidates = diag.get('candidate_results') or []
        for candidate in candidates:
            counts['candidate_status'][candidate.get('status', UNKNOWN)] += 1
        if diag.get('replanned') is True:
            selected = next((c for c in candidates if c.get('order') == entry.get('release_order') and c.get('status') == 'evaluated'), None)
            base = next((c for c in candidates if es in ('empty', 'nonempty') and c.get('order') == diag['energy_order']), None)
            record = dict(tick=entry['tick'], selected=candidate_summary(selected), base=candidate_summary(base),
                          selected_order=entry.get('release_order'), energy_order=diag.get('energy_order'),
                          relation=relation, fallback=diag.get('fallback', UNKNOWN))
            replans.append(record)
            if relation == 'different':
                differences.append(record)
        dispatch = entry.get('dispatch') or {}
        counts['approved_release_field_state'][array_state(dispatch, 'normal_approved_releases')] += 1
        approved = dispatch.get('normal_approved_releases')
        if isinstance(approved, list):
            approved_count += len(approved)
        decisions = dispatch.get('release_decisions')
        if isinstance(decisions, dict):
            counts['release_decisions'].update(str(v) for v in decisions.values())
    final_state = (last or {}).get('state', {})
    telemetry = result.get('telemetry', final_state.get('telemetry', UNKNOWN))
    return dict(name=path.name, source=str(path.resolve()), protocol=result.get('protocol', {}),
                status=result.get('status', UNKNOWN), metrics={k:v for k,v in metrics.items() if k != 'per_arrival'},
                per_arrival=per_arrival, final_outcomes=final_state.get('outcomes', {}),
                wait_derived=dict(count=len(waits), max=max(waits) if waits else None, p90=percentile(waits, .9), method='linear interpolation'),
                min_landing_soc=min(socs) if socs else None,
                timing={k:result.get(k, UNKNOWN) for k in ('elapsed_seconds','planning_seconds','native_checks_seconds','first_tick_seconds')},
                native_telemetry_final=telemetry, telemetry_semantics='final cumulative snapshot; never sum step snapshots',
                integrity={k:result.get(k, UNKNOWN) for k in ('integrity_ok','full_valid','active_plan_hash_samples')},
                ticks_observed=count, diagnostics={k:dict(v) for k,v in counts.items()},
                numeric_diagnostics=numerics,
                normal_approved_release_event_count=approved_count, first_order_difference_tick=first_different,
                replans=replans, selected_differs_from_base_events=differences)


def delta(a, b):
    return b-a if isinstance(a, (int,float)) and isinstance(b, (int,float)) else None


def pair_outcomes(a, b):
    aa, bb = a['per_arrival'], b['per_arrival']
    ao, bo = a['final_outcomes'], b['final_outcomes']
    rows = []
    for aid in sorted(set(aa)|set(bb)|set(ao)|set(bo)):
        x, y = aa.get(aid, {}), bb.get(aid, {})
        rows.append(dict(arrival_id=aid, energy_outcome=ao.get(aid, UNKNOWN), rolling_outcome=bo.get(aid, UNKNOWN),
                         wait_delta=delta(wait(x), wait(y)), energy_delta=delta(x.get('energy'),y.get('energy')),
                         landing_soc_delta=delta(x.get('landing_soc'),y.get('landing_soc'))))
    return rows


def compare_pair(a, b):
    pa, pb = Path(a['source']), Path(b['source'])
    ai, ae = physical(read_json(pa/'initial_state.json'))
    bi, be = physical(read_json(pb/'initial_state.json'))
    out = dict(energy_arm=a['name'], rolling_arm=b['name'], delta_direction='rolling minus energy',
               initial_physical_equal=ai == bi if not ae and not be else UNKNOWN,
               initial_unknown_reason=ae or be, physical_fields=list(PHYSICAL_FIELDS),
               physical_scope='native active uavs only; SOC is unknown when native uavs omit it',
               per_arrival=pair_outcomes(a,b), first_normal_release_difference=None,
               first_post_step_physical_difference=None, alignment_status='aligned common ticks', unknown=[])
    for x,y in zip_longest(steps(pa), steps(pb)):
        if x is None or y is None:
            out['common_prefix_end_tick'] = (x or y).get('tick')
            out['alignment_status'] = 'unequal tape lengths; compared common ticks only'
            break
        if x.get('tick') is None or x.get('tick') != y.get('tick'):
            out['alignment_status'] = UNKNOWN
            out['unknown'].append('tick alignment missing or mismatched')
            break
        dx,dy = x.get('dispatch', {}), y.get('dispatch', {})
        sx,sy = array_state(dx,'normal_approved_releases'), array_state(dy,'normal_approved_releases')
        if sx not in ('empty','nonempty') or sy not in ('empty','nonempty'):
            if 'normal release field missing/null/invalid' not in out['unknown']:
                out['unknown'].append('normal release field missing/null/invalid')
        elif out['first_normal_release_difference'] is None and dx['normal_approved_releases'] != dy['normal_approved_releases']:
            out['first_normal_release_difference'] = dict(tick=x['tick'], energy=context(x), rolling=context(y),
                approved_membership_differs=set(dx['normal_approved_releases']) != set(dy['normal_approved_releases']))
        px,ex = physical(x.get('state', {})); py,ey = physical(y.get('state', {}))
        if ex or ey:
            if (ex or ey) not in out['unknown']:
                out['unknown'].append(ex or ey)
        elif px != py and out['first_post_step_physical_difference'] is None:
            changed = {uid:dict(energy=px.get(uid),rolling=py.get(uid)) for uid in sorted(set(px)|set(py)) if px.get(uid)!=py.get(uid)}
            out['first_post_step_physical_difference'] = dict(tick=x['tick'], changed_uavs=changed, energy=context(x), rolling=context(y))
    out['E_to_N'] = [r['arrival_id'] for r in out['per_arrival'] if r['energy_outcome']=='emergency_landed' and r['rolling_outcome']=='completed']
    out['N_to_E'] = [r['arrival_id'] for r in out['per_arrival'] if r['energy_outcome']=='completed' and r['rolling_outcome']=='emergency_landed']
    return out


def analyze(input_path, output_path, input_evidence_type='preexisting_artifact'):
    if input_evidence_type not in ('preexisting_artifact', 'newly_run'):
        raise ValueError('input_evidence_type must be preexisting_artifact or newly_run')
    root, output = Path(input_path).resolve(), Path(output_path).resolve()
    if output == root or root in output.parents:
        raise ValueError('output must be outside the preexisting input tree')
    if output.exists():
        raise FileExistsError('output must be a fresh directory')
    arms = [analyze_arm(p.parent) for p in sorted(root.glob('*/result.json'))]
    pairs = []
    for a in arms:
        if a['protocol'].get('controller') == 'energy_slack':
            matching = [b for b in arms if b['protocol'].get('controller') == 'rolling_reorder'
                        and all(b['protocol'].get(k)==a['protocol'].get(k) for k in ('scenario','seed'))]
            if len(matching)==1:
                pairs.append(compare_pair(a, matching[0]))
    summary = dict(input_evidence_type=input_evidence_type, extraction_evidence_type='newly_run',
                   input=str(root), inference_boundary='Observed paired scenario-seed traces; no independent-sample pooling, no counterfactual causality. Concurrent planning wall times are descriptive only.', arms=arms,pairs=pairs)
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n',encoding='utf8')
    lines = ['# Arrival trace extraction', '', summary['inference_boundary'], '',
             f'Input: {input_evidence_type}. Extraction: newly_run. Native cumulative telemetry uses final snapshots.', '',
             '| Arm | N/E/F/D/U | Wait max / p90 | Min landing SOC | Replans | Order differences | Shortlist exhausted |',
             '|---|---|---|---|---|---|---|']
    for a in arms:
        m=a['metrics']; w=a['wait_derived']; t=a['native_telemetry_final']
        lines.append(f"| {a['name']} | {'/'.join(str(m.get(k,UNKNOWN)) for k in ('N','E','F','D','U'))} | {w['max']} / {w['p90']} | {a['min_landing_soc']} | {a['diagnostics']['replanned'].get('True',0)} | {a['diagnostics']['order_comparison'].get('different',0)} | {t.get('runtime_shortlist_exhausted',UNKNOWN) if isinstance(t,dict) else UNKNOWN} |")
    lines += ['', '| Pair (rolling versus energy) | Initial active physical equal | First release difference | First physical difference | E→N IDs | N→E IDs |', '|---|---|---|---|---|---|']
    for p in pairs:
        lines.append(f"| {p['rolling_arm']} | {p['initial_physical_equal']} | {(p['first_normal_release_difference'] or {}).get('tick')} | {(p['first_post_step_physical_difference'] or {}).get('tick')} | {p['E_to_N']} | {p['N_to_E']} |")
    lines += ['', 'Exact per-arrival deltas, selected/base candidate scores for each replan, all selected-order difference events, first-divergence native release reasons/resource snapshots, timing and final safety telemetry are in summary.json.', '', 'Physical comparison uses active native UAV UID, arrival ID, phase, position, velocity, acceleration, speed, committed flag, trajectory hash and SOC where present. Opaque plan UUIDs are excluded. Missing SOC is unknown; equality is limited to recorded fields. Null deltas preserve missing measurements. Release differences compare ordered approved lists and separately flag membership changes.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--input-evidence-type', choices=('preexisting_artifact', 'newly_run'),
                        default='preexisting_artifact')
    args=parser.parse_args()
    summary=analyze(args.input,args.output,args.input_evidence_type)
    print(json.dumps(dict(arms=len(summary['arms']),pairs=len(summary['pairs']),output=str(Path(args.output).resolve()))))


if __name__ == '__main__':
    main()
