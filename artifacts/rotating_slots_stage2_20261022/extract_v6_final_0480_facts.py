"""Read frozen local campaign artifacts and write compact numerical observations."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from terminal_marl.core.rotating_slots.stage2_campaign import Stage2Codec, rescue_comparison_tables
from terminal_marl.core.rotating_slots.world import AIRBORNE

BASE=Path('D:/7_codespace/training_outputs/terminal')
OUT=Path(__file__).with_name('v6_final_0480_facts.json')
METHODS=('energy_slack','rh_cp_sat')
METRICS=('L_normal','L_emergency','L_success','F_energy_total','F_battery','F_peak','N_offered',
    'N_admitted_total','N_airborne_pending','N_upstream_pending','N_rerouted','manifest_offered',
    'initial_airborne_count','initial_ground_count','emergency_declared','emergency_trigger_count',
    'emergency_unresolved','protected_rescue_opportunities','rescue_warning','rescue_search_exhausted',
    'landing_below_reserve','cost_emergency','energy_j','holding_time','pad_utilization_seconds',
    'decision_count','conservation','terminated_invalid')


def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]
def counts(values): return dict(sorted(Counter(values).items(),key=lambda item:str(item[0])))
def selected(value, keys): return {key:value.get(key,'UNKNOWN') for key in keys}
def stats(values):
    values=sorted(float(v) for v in values if v is not None)
    if not values: return dict(count=0,sum=0.,mean=None,p95=None,min=None,max=None)
    def q(p):
        pos=(len(values)-1)*p;lo=math.floor(pos);hi=math.ceil(pos)
        return values[lo]+(values[hi]-values[lo])*(pos-lo)
    return dict(count=len(values),sum=math.fsum(values),mean=math.fsum(values)/len(values),p95=q(.95),min=values[0],max=values[-1])
def timing(summary,name): return selected(summary.get(name,{}),('count','mean','p95','max','p50','over_5s','over_5s_fraction'))
def kind(event,trajectories):
    return event.get('details',{}).get('maneuver_kind',trajectories.get(event.get('details',{}).get('trajectory_hash'),{}).get('kind','UNKNOWN'))


def normal_plan_event(event,trajectories):
    plan=trajectories.get(event['details'].get('trajectory_hash'),{})
    return dict(aircraft_id=event['aircraft_id'],time_s=event['time'],event=event['type'],kind=kind(event,trajectories),
        planned_start_s=plan.get('start_time','UNKNOWN'),planned_end_s=plan.get('end_time','UNKNOWN'))


def event_summary(events, trajectories, legacy=False):
    by_type=defaultdict(list)
    for event in events: by_type[event['type']].append(event)
    starts=by_type['maneuver_started']
    declared=by_type['emergency_declared'];first_decl=min((e['time'] for e in declared),default=None)
    ordinary=[e for e in starts if kind(e,trajectories) in ('TRANSFER','TRANSFER_O_M','TRANSFER_M_I','FINAL_NORMAL')]
    after=[e for e in ordinary if first_decl is not None and e['time']>first_decl+1.e-8]
    def last_event(items):
        if not items: return dict(observed=False)
        e=max(items,key=lambda e:e['time'])
        return dict(observed=True,time_s=e['time'],aircraft_id=e['aircraft_id'],maneuver_kind=kind(e,trajectories))
    result=dict(event_type_counts=counts(e['type'] for e in events),
        first_warning_time_s='UNKNOWN' if legacy else min((e['time'] for e in by_type['rescue_warning']),default=None),
        first_trigger_time_s=min((e['time'] for e in by_type['emergency_triggered']),default=None),
        first_declaration_time_s=first_decl,
        first_actual_emergency_start_time_s=min((e['time'] for e in starts if kind(e,trajectories)=='FINAL_EMERGENCY'),default=None),
        actual_normal_start_counts_by_kind=counts(kind(e,trajectories) for e in ordinary),
        actual_normal_start_counts_strictly_after_first_declaration=counts(kind(e,trajectories) for e in after),
        normal_starts_at_first_declaration_time=sum(first_decl is not None and abs(e['time']-first_decl)<=1.e-8 for e in ordinary),
        last_actual_normal_start=last_event(ordinary),last_actual_normal_final_start=last_event([e for e in ordinary if kind(e,trajectories)=='FINAL_NORMAL']),
        warning_events='UNKNOWN' if legacy else len(by_type['rescue_warning']),
        warning_unique_aircraft='UNKNOWN' if legacy else len({e['aircraft_id'] for e in by_type['rescue_warning']}),
        trigger_events=len(by_type['emergency_triggered']),trigger_unique_aircraft=len({e['aircraft_id'] for e in by_type['emergency_triggered']}),
        declaration_events=len(declared),declaration_unique_aircraft=len({e['aircraft_id'] for e in declared}),
        actual_rescue_start_events=sum(kind(e,trajectories)=='FINAL_EMERGENCY' for e in starts),
        actual_rescue_start_unique_aircraft=len({e['aircraft_id'] for e in starts if kind(e,trajectories)=='FINAL_EMERGENCY'}),
        low_reserve_touchdown_events=sum(bool(e['details'].get('below_reserve',False)) for e in by_type['touchdown']),
        low_reserve_touchdown_unique_aircraft=len({e['aircraft_id'] for e in by_type['touchdown'] if e['details'].get('below_reserve',False)}),
        recorded_soft_plan_cancellation_events=len(by_type['soft_plan_cancelled']),
        recorded_emergency_priority_yielded_events=len(by_type['emergency_priority_yielded']),
        rescue_commitment_protected_events=len(by_type['rescue_commitment_protected']),
        explicit_safe_ordinary_retention_label='UNKNOWN_NOT_LOGGED')
    if not legacy:
        submissions=[e for e in events if e['type'] in ('plan_committed','soft_plan_reserved')
            and kind(e,trajectories) in ('TRANSFER','FINAL_NORMAL') and first_decl is not None and e['time']>first_decl+1.e-8]
        result['actual_normal_starts_after_first_declaration']=[dict(aircraft_id=e['aircraft_id'],time_s=e['time'],kind=kind(e,trajectories)) for e in after]
        result['normal_submission_events_after_first_declaration']=[normal_plan_event(e,trajectories) for e in submissions]
        result['normal_submission_unique_aircraft_after_first_declaration']=len({e['aircraft_id'] for e in submissions})
    return result,by_type


def state_facts(path,event_times):
    boundary_states={};event_states={};last=None
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            state=json.loads(line);t=state['time'];last=state
            boundary=int(round(t/60.))*60
            if 0<=boundary<=480 and abs(t-boundary)<1.e-6: boundary_states[boundary]=state
            if round(t,6) in event_times: event_states[round(t,6)]={a['id']:a for a in state['aircraft']}
    return boundary_states,event_states,last


def soc_rows(state):
    aircraft=[a for a in state['aircraft'] if a['status'] in AIRBORNE]
    return dict(state_time_s=state['time'],airborne_count=len(aircraft),
        reserve_margin_soc=stats(a['soc']-.25 for a in aircraft),
        normal_margin_known_count=sum(a['emergency'].get('normal_margin') is not None for a in aircraft),
        normal_margin_known=stats(a['emergency']['normal_margin'] for a in aircraft if a['emergency'].get('normal_margin') is not None),
        rings={ring:dict(count=len(group),min_soc=min((a['soc'] for a in group),default=None),
            soc_le_035=sum(a['soc']<=.35 for a in group),soc_below_025=sum(a['soc']<.25 for a in group))
            for ring in ('outer','middle','inner',None)
            for group in [[a for a in aircraft if a['ring']==ring]]})


def final_checkpoint_parts(path):
    payload=load(path)['payload'];encoded=dict(payload['world']['items']);codec=Stage2Codec()
    return payload['time'],codec.decode(encoded['aircraft']),codec.decode(encoded['active']),codec.decode(encoded['soft'])


def rescue_sequences(events,by_type,trajectories,event_states,aircraft,active,soft):
    ids={e['aircraft_id'] for typ in ('emergency_triggered','emergency_declared','rescue_opportunity_protected') for e in by_type[typ]}
    starts=defaultdict(list);cancels=defaultdict(list)
    for e in by_type['maneuver_started']: starts[e['details']['trajectory_hash']].append(e)
    for e in by_type['soft_plan_cancelled']: cancels[e['details']['trajectory_hash']].append(e)
    pending={};pending_at_protection={}
    for e in events:
        h=e['details'].get('trajectory_hash');typ=e['type']
        if typ=='soft_plan_reserved' and trajectories.get(h,{}).get('kind')!='FINAL_EMERGENCY': pending[e['aircraft_id']]=h
        elif typ in ('maneuver_started','soft_plan_cancelled') and pending.get(e['aircraft_id'])==h: pending.pop(e['aircraft_id'],None)
        elif typ=='rescue_opportunity_protected': pending_at_protection[(e['aircraft_id'],h)]=list(pending.values())
    result=[]
    for aid in sorted(ids):
        own=[e for e in events if e['aircraft_id']==aid]
        def first(typ,predicate=lambda e:True): return next((e for e in own if e['type']==typ and predicate(e)),None)
        def fact(e):
            if e is None: return dict(observed=False,censored_at_s=480.)
            a=event_states.get(round(e['time'],6),{}).get(aid)
            return dict(observed=True,time_s=e['time'],details=e['details'],
                soc_at_exact_state_time='UNKNOWN' if a is None else a['soc'],
                ring_at_exact_state_time='UNKNOWN' if a is None else a['ring'],
                state_match='UNKNOWN' if a is None else 'SAME_TIME_RECORDED_STATE')
        trigger=first('emergency_triggered');declaration=first('emergency_declared')
        start=first('maneuver_started',lambda e:kind(e,trajectories)=='FINAL_EMERGENCY');touchdown=first('touchdown')
        failure=first('aircraft_energy_failure')
        waits={}
        for name,origin in [('trigger_to_actual_start',trigger),('declaration_to_actual_start',declaration)]:
            waits[name]=dict(status='UNKNOWN',seconds='UNKNOWN') if origin is None else dict(
                status='CENSORED_AT_480' if start is None else 'COMPLETED',seconds=(480. if start is None else start['time'])-origin['time'])
        protected=[]
        for e in own:
            if e['type']!='rescue_opportunity_protected': continue
            h=e['details']['trajectory_hash'];ordinary=pending_at_protection[(aid,h)];outcomes=Counter()
            for ordinary_hash in ordinary:
                next_start=next((s for s in starts[ordinary_hash] if s['time']>=e['time']-1.e-8),None)
                cancelled=any(c['time']>=e['time']-1.e-8 and (next_start is None or c['time']<=next_start['time']) for c in cancels[ordinary_hash])
                outcomes['CANCELLED_BEFORE_START' if cancelled else 'STARTED_WITHOUT_CANCELLATION' if next_start else 'PENDING_NO_CANCELLATION_BY_480']+=1
            active_match=aid in active and active[aid].trajectory_hash==h;soft_match=aid in soft and soft[aid].trajectory_hash==h
            protected.append(dict(protection_time_s=e['time'],**e['details'],
                state_at_480='ACTIVE' if active_match else 'SOFT_RESERVED' if soft_match else 'NEITHER_ACTIVE_NOR_SOFT',
                ordinary_soft_pending_at_protection_count=len(ordinary),ordinary_soft_pending_hashes=ordinary,
                later_ordinary_soft_outcomes=dict(outcomes)))
        a=aircraft[aid]
        result.append(dict(aircraft_id=aid,first_warning=fact(first('rescue_warning')),trigger=fact(trigger),declaration=fact(declaration),
            actual_rescue_start=fact(start),touchdown=fact(touchdown),failure=fact(failure),waits=waits,
            protected_plans=protected,final_state=dict(status=a.status,ring=a.ring,soc=a.soc,
                emergency_state=a.emergency.state,rescue_slack_seconds=a.emergency.rescue_slack_seconds,
                rescue_wait_seconds=a.emergency.rescue_wait_seconds,q_e=a.emergency.q_e)))
    return result


def planning_facts(decisions,summary):
    choices=[c for row in decisions for c in row['aircraft_choices']]
    solver=[s for row in decisions for s in row['solver']]
    eligible=[r for r in decisions if r['move_aircraft_count']>0]
    cuts=Counter();unknown=[]
    for row in decisions:
        grouped=defaultdict(list)
        for s in row['solver']: grouped[s.get('uniform_refinement_restart',0)].append(s)
        for group in grouped.values():
            for key in ('cumulative_pair_cuts','cumulative_single_cuts','cumulative_prefix_nogoods'):
                observed=[s[key] for s in group if key in s]
                if observed: cuts[key]+=max(observed)
        for s in row['solver']:
            if s.get('validation_status')=='UNKNOWN' or s.get('prefix_status')=='UNKNOWN':
                unknown.append(dict(time_s=row['time'],phase=s['phase'],**selected(s,('validation_status','prefix_status','batch_conflict_count',
                    'cumulative_pair_cuts','cumulative_single_cuts','cumulative_prefix_nogoods'))))
    caches=Counter();geometry=Counter()
    for row in decisions:
        caches.update(row.get('validation_cache',{}));geometry.update(row.get('geometry_screening',{}))
    def coverage(items):
        n=len(items);moves=sum(c['move_candidates'] for c in items);full=sum(c['full_normal_chain_count'] for c in items)
        return dict(aircraft_call_observations=n,only_keep_wait_observations=sum(c['only_keep_wait'] for c in items),
            only_keep_wait_fraction=sum(c['only_keep_wait'] for c in items)/n if n else None,
            observations_with_full_chain=sum(c['full_normal_chain_count']>0 for c in items),
            observations_with_full_chain_fraction=sum(c['full_normal_chain_count']>0 for c in items)/n if n else None,
            move_columns=moves,full_chain_columns=full,partial_progress_columns=sum(c['partial_progress_count'] for c in items),
            full_chain_over_move_columns=full/moves if moves else None)
    return dict(normal_calls=len(decisions),move_eligible_calls=len(eligible),solver_entered_calls=sum(r['solver_entered'] for r in decisions),
        solver_entered_when_move_eligible=sum(r['solver_entered'] for r in eligible),
        strict_improvement_over_hint_counts=counts('UNKNOWN' if r['strict_improvement_over_hint'] is None else str(r['strict_improvement_over_hint']) for r in decisions),
        skip_reason_counts=counts(str(r['skip_reason']) for r in decisions),solver_stop_reason_counts=counts(str(r['solver_stop_reason']) for r in decisions),
        solver_phase_calls=len(solver),solver_phase_status_counts=counts(s['status'] for s in solver),
        logged_batch_conflicts=sum(s.get('batch_conflict_count',0) for s in solver),
        sum_per_call_restart_max_cumulative_cuts=dict(cuts),
        cut_count_scope='Logged cumulative counters include initial resource cuts; maxima are taken once per call/restart, then summed.',
        unknown_validation_phase_records=len(unknown),unknown_validation_phase_samples=unknown[:5],
        unknown_cut_proof_in_run_logs='UNKNOWN: absent cumulative counters cannot establish a zero-cut proof',
        validation_cache_sum_logged_context_counters=dict(caches),move_energy_cache_final_cumulative=decisions[-1].get('move_energy_cache','UNKNOWN'),
        candidate_reference_cache_hits_sum_logged_catalog_deltas=sum(r['cache_hits'] for r in decisions),geometry_screening_sum_logged=dict(geometry),
        candidate_coverage=coverage(choices),candidate_coverage_by_ring={ring:coverage([c for c in choices if c['ring']==ring]) for ring in ('outer','middle','inner')},
        candidate_coverage_per_aircraft_source='summary_0480.json:rescue_observations.candidate_coverage')


def pad_facts(decisions,summary,first_declaration):
    pads=[pad for row in decisions for pad in row['pad_opportunities']]
    flags=('grounded_service','ground_idle','approach_reserved','future_calendar','emergency_protected','idle_without_active_approach','offered')
    reason_counts=counts(p['reason'] for p in pads)
    assert reason_counts==summary['rescue_observations']['pad_reason_counts']
    idle=[p for p in pads if p['offered'] and p['idle_without_active_approach']]
    after=[p for p in pads if first_declaration is not None and p['time']>first_declaration+1.e-8]
    not_submitted=[p for p in pads if p['reason']=='PLANNING_NOT_SUBMITTED']
    return dict(denominators=summary['rescue_observations']['denominators'],reason_counts=reason_counts,
        all_sample_flag_counts={key:sum(p[key] for p in pads) for key in flags},
        offered_idle_without_active_approach_reason_counts=counts(p['reason'] for p in idle),
        strictly_after_first_declaration=dict(first_declaration_time_s=first_declaration,
            all_samples=len(after),offered_idle_without_active_approach_samples=sum(p['offered'] and p['idle_without_active_approach'] for p in after),
            flag_counts={key:sum(p[key] for p in after) for key in flags},reason_counts=counts(p['reason'] for p in after)),
        planning_not_submitted_samples=len(not_submitted),
        planning_not_submitted_with_certified_single_count_positive=sum(p['planning_evidence'].get('certified_single_count',0)>0 for p in not_submitted),
        protected_future_overlap=sum(p['emergency_protected'] and p['future_calendar'] for p in pads),
        by_pad={str(pad):dict(samples=sum(p['pad_id']==pad for p in pads),
            flags={key:sum(p[key] for p in pads if p['pad_id']==pad) for key in flags},
            reasons=counts(p['reason'] for p in pads if p['pad_id']==pad)) for pad in range(6)},
        scope='Dispatch samples with overlapping flags; not time-weighted pad utilization or causal availability claims.')


def extract_v6(root):
    campaign=load(root/'campaign.json');provenance=load(root/'provenance.json')
    helper=rescue_comparison_tables(root);assert helper['common_time_s']==480
    result=dict(root=str(root),provenance_path=str(root/'provenance.json'),identity=campaign['identity'],
        status=campaign['status'],common_completed_time_s=campaign['common_completed_time_s'],
        campaign_wall_s=campaign['campaign_wall_s'],campaign_timing=(root/'campaign.timing.txt').read_text(encoding='utf-8').strip(),
        exit_code_raw=(root/'exit_code.txt').read_text().strip(),methods={})
    raw=result['exit_code_raw'];result['exit_code_numeric_prefix']=int(raw.rstrip('n')) if raw.rstrip('n').isdigit() else 'UNKNOWN'
    result['exit_code_file_valid_integer']=raw.isdigit()
    wrapper=(root/'campaign.wrapper.log').read_text(encoding='utf-8')
    footer={key:re.findall(r'^'+key+r'=(.*)$',wrapper,re.MULTILINE)[-1] for key in ('run_exit_code','wrapper_exit_code','run_end_utc','run_elapsed_seconds')}
    result['verified_wrapper_footer']=footer
    result['wrapper_log_path']=str(root/'campaign.wrapper.log')
    result['terminal_budget']=load(root/'budget.json')
    result['elapsed_scope']='Terminal campaign wall and stored budget floor; no post-completion clock subtraction.'
    for method in METHODS:
        directory=root/method;replay=directory/'replay_0480';summary=load(directory/'summary_0480.json')
        events=rows(replay/'events.jsonl');decisions=rows(directory/'decisions.jsonl');trajectories=load(replay/'trajectories.json')
        event_counts,by_type=event_summary(events,trajectories)
        event_times={round(e['time'],6) for e in events if e['type'] in ('rescue_warning','emergency_triggered','emergency_declared','rescue_opportunity_protected','maneuver_started','touchdown','aircraft_energy_failure')}
        boundaries,event_states,last=state_facts(directory/'states.jsonl',event_times)
        t,aircraft,active,soft=final_checkpoint_parts(directory/'checkpoint_0480.json');assert t==480.
        assert set(range(60,481,60))<=set(boundaries)
        timeline=[]
        for boundary in range(60,481,60):
            s=load(directory/f'summary_{boundary:04d}.json')
            timeline.append(dict(time_s=boundary,metrics=selected(s['metrics'],('L_normal','L_emergency','L_success','F_energy_total',
                'N_offered','N_airborne_pending','N_upstream_pending','emergency_declared','emergency_unresolved','rescue_warning')),**soc_rows(boundaries[boundary])))
        observed=summary['rescue_observations'];comparative=next(r for r in helper['rows'] if r['method']==method)
        assert comparative['total']==summary['metrics']['L_success']
        assert len(decisions)==summary['normal_timing']['count']==summary['metrics']['decision_count']
        assert event_counts['declaration_unique_aircraft']==observed['declared_aircraft']
        assert event_counts['actual_rescue_start_unique_aircraft']==observed['rescue_started_aircraft']
        sequences=rescue_sequences(events,by_type,trajectories,event_states,aircraft,active,soft)
        attempts=by_type['rescue_opportunity_attempt']
        below_undeclared=[]
        for observation in observed['aircraft']:
            if observation['status'] not in AIRBORNE or observation['soc']>=.25 or observation['declaration_time'] is not None: continue
            aid=observation['aircraft_id'];search=next((e for e in reversed(by_type['rescue_opportunity_search_exhausted']) if e['aircraft_id']==aid),None)
            below_undeclared.append(dict(aircraft_id=aid,ring=aircraft[aid].ring,**selected(observation,('status','soc','normal_status')),
                rescue_search_exhausted_event_count=sum(e['aircraft_id']==aid for e in by_type['rescue_opportunity_search_exhausted']),
                last_rescue_search_exhausted=dict(observed=False) if search is None else dict(observed=True,time_s=search['time'],details=search['details']),
                normal_plan_events=[normal_plan_event(e,trajectories) for e in events if e['aircraft_id']==aid
                    and e['type'] in ('plan_committed','soft_plan_reserved','maneuver_started') and kind(e,trajectories) in ('TRANSFER','FINAL_NORMAL')]))
        declared_unlanded=[a for a in aircraft.values() if a.emergency.declared and a.touchdown_time is None]
        method_result=dict(provenance_path=str(root/'provenance.json'),
            evidence_paths={name:str(path) for name,path in dict(summary=directory/'summary_0480.json',checkpoint=directory/'checkpoint_0480.json',
                decisions=directory/'decisions.jsonl',dispatch_timings=directory/'dispatch_timings.jsonl',states=directory/'states.jsonl',events=replay/'events.jsonl',trajectories=replay/'trajectories.json').items()},
            final_metrics=selected(summary['metrics'],METRICS),cohorts=summary['metrics']['cohorts'],
            final_airborne_soc_below_reserve_without_declaration=below_undeclared,
            final_declared_not_landed_soc=dict(count=len(declared_unlanded),minimum=min((a.soc for a in declared_unlanded),default=None)),
            terminal_artifact_checks=dict(checkpoint_time_s=t,summary_time_s=summary['time'],last_state_time_s=last['time'],
                checkpoint_bytes=(directory/'checkpoint_0480.json').stat().st_size,state_bytes=(directory/'states.jsonl').stat().st_size,
                error_json_exists=(directory/'error.json').exists(),invalid_json_exists=(directory/'invalid.json').exists()),
            boundaries_60s=timeline,event_observations=event_counts,rescue_sequences=sequences,
            warning_first_times=[dict(aircraft_id=aid,time_s=min(e['time'] for e in by_type['rescue_warning'] if e['aircraft_id']==aid)) for aid in sorted({e['aircraft_id'] for e in by_type['rescue_warning']})],
            rescue_attempt_status_reason_counts=counts(str((e['details'].get('status','UNKNOWN'),e['details'].get('reason','UNKNOWN'))) for e in attempts),
            rescue_commitment_rejection_reason_counts=counts(e['details'].get('reason','UNKNOWN') for e in by_type['rescue_commitment_protected']),
            timing=dict(normal=timing(summary,'normal_timing'),dispatch=timing(summary,'dispatch_timing'),
                candidate_exclusive=selected(observed['candidate_exclusive_timing'],('count','mean','p95','max')),
                safety_validation=selected(observed['safety_validation_timing'],('count','mean','p95','max')),
                selected_validation_overhead=selected(observed['selected_validation_overhead_timing'],('count','mean','p95','max')),
                solver=selected(observed['solver_timing'],('count','mean','p95','max')),
                episode_compute_wall_s=summary['episode_compute_wall_s']),
            rescue_wait_summary=selected(observed,('rescue_wait_completed','rescue_wait_censored')),
            pads=pad_facts(decisions,summary,event_counts['first_declaration_time_s']),planning=planning_facts(decisions,summary),audit=load(directory/'archive_status.json'))
        for waiting in method_result['rescue_wait_summary'].values(): waiting.pop('raw_s',None)
        result['methods'][method]=method_result
        print(root.name,method,{key:summary['metrics'][key] for key in ('L_normal','L_emergency','L_success','F_energy_total','N_airborne_pending')},'declared/started',event_counts['declaration_unique_aircraft'],event_counts['actual_rescue_start_unique_aircraft'])
    return result


def extract_v5(root):
    campaign=load(root/'campaign.json');result=dict(root=str(root),provenance_path=str(root/'provenance.json'),
        identity=campaign['identity'],status=campaign['status'],common_completed_time_s=campaign['common_completed_time_s'],
        frozen_facts_path=str(OUT.with_name('v5_final_0480_facts.json')),methods={})
    frozen=load(OUT.with_name('v5_final_0480_facts.json'))
    helper=rescue_comparison_tables(root)
    for method in METHODS:
        directory=root/method;replay=directory/'replay_0480';summary=load(directory/'summary_0480.json')
        frozen_counts=frozen['methods'][method]['boundaries'][-1]['counts']
        assert all(summary['metrics'][key]==value for key,value in frozen_counts.items() if key in summary['metrics'])
        events=rows(replay/'events.jsonl');trajectories=load(replay/'trajectories.json');observed,_=event_summary(events,trajectories,legacy=True)
        result['methods'][method]=dict(provenance_path=str(root/'provenance.json'),summary_path=str(directory/'summary_0480.json'),
            final_metrics=selected(summary['metrics'],METRICS),cohorts=summary['metrics']['cohorts'],event_observations=observed,
            timing=dict(normal=timing(summary,'normal_timing'),dispatch=timing(summary,'dispatch_timing')),
            v6_only_observation_fields=selected(next(r for r in helper['rows'] if r['method']==method),
                ('warning_aircraft','candidate_exclusive_timing','safety_validation_timing','pad_reason_counts','candidate_coverage')))
    return result


def main():
    if OUT.exists() and '--replace-own-output' not in sys.argv: raise SystemExit('Use --replace-own-output to recompute this owned facts artifact')
    result=dict(schema='stage2_v6_final_0480_facts_v1',recorded_utc=datetime.now(timezone.utc).isoformat(),
        evidence_type='preexisting_artifact',extraction_type='newly_run_local_read_only_aggregation',human_review='HUMAN_PENDING',
        observation_window_s=480,
        definitions=dict(missing_fields='UNKNOWN',unobserved_event='observed=false, censored_at_s=480',
            low_soc='SOC <= .35; below reserve SOC < .25',ring='Recorded source/current ring, including transfer aircraft; None is stored as null key.',
            timing='Cumulative final-summary call distributions, never sums of repeated cumulative 60 s summaries.',
            ordinary_retention='Actual pending soft-plan event sequences only; no causal or physical infeasibility claim.'),
        scenarios={})
    for name in ('stress','diagnostic'):
        result['scenarios'][name]=extract_v6(BASE/f'stage2_rescue_opportunity_v6_{name}_480s_20261022')
    result['v5_comparison']=extract_v5(BASE/'stage2_quality_rescue_480s_v5_20261022')
    encoded=json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n'
    if len(encoded.encode())>150000: raise RuntimeError('Compact facts exceed 150 KB: '+str(len(encoded.encode())))
    OUT.write_text(encoded,encoding='utf-8')
    assert load(OUT)['observation_window_s']==480
    print('OUTPUT',OUT,'BYTES',OUT.stat().st_size)


if __name__=='__main__': main()
