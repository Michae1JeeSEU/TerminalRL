"""Read saved four-cell 600s results and create new derived report/trend files.

Standard-library analysis plus lazy matplotlib plotting. No project runtime,
simulation, solver, inference or training imports/calls. Existing files are never
overwritten; incomplete derived publication is retained for primary review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

CASES=('medium','high')
ARMS=('heuristic_reduced','learned_reduced')
OUTPUTS=('FINAL_REPORT.md','analysis_summary.json','trends.csv','trends.svg','trends.png')
QUALITY_FIELDS=('N_admitted_total','N_airborne_pending','N_upstream_pending','L_normal','L_emergency','L_success',
    'F_energy_total','terminated_invalid','conservation','energy_j','holding_time','airborne_time','upstream_wait_time',
    'emergency_declared','emergency_unresolved','cost_emergency','cost_reroute','K_episode','cohorts')
TREND_FIELDS=('L_normal','N_airborne_pending','energy_j','L_success','L_emergency','N_admitted_total',
    'holding_time','upstream_wait_time','emergency_declared','emergency_unresolved')


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def file_hash(path):
    value=sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def number(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def difference(h,l):return l-h if number(h) and number(l) else None


def quality(q):return {k:q[k] for k in QUALITY_FIELDS if k in q}


def compact_callbacks(source):
    # These are existing all-callback aggregates. No last-selection proxy and
    # no summing repeated cumulative counters are introduced by this reporter.
    result={k:v for k,v in source.items() if k not in ('catalogs','coverage_by_callback','final_selection_only','observer_counts')}
    catalogs=source.get('catalogs',[])
    pruned=[r['pruned_columns'] for r in catalogs if number(r.get('pruned_columns'))]
    columns=[r['final_columns'] for r in catalogs if number(r.get('final_columns'))]
    result['catalog_observations']=dict(count=len(catalogs),pruning_observed_callbacks=len(pruned),
        pruned_total=sum(pruned) if pruned else None,
        final_columns_observed=len(columns),final_columns_min=min(columns) if columns else None,
        final_columns_max=max(columns) if columns else None,final_columns_sum=sum(columns) if columns else None)
    return result


def collect(output):
    """Read-only: returns compact facts and forty actual 60s trend rows."""
    scope=load(output/'evaluation_scope.json');ledger=load(output/'budget_ledger.json')
    cells={};trends=[];inputs={};case_comparisons={};missing=[]
    def read(relative):
        path=output/relative;inputs[str(relative)]=file_hash(path);return load(path)
    read('evaluation_scope.json');read('budget_ledger.json');run=read('run.finished.json')
    for case in CASES:
        roots=read(Path('roots')/case/'root.json');cells[case]={}
        available={arm:{int(p.stem.split('_')[-1]):p for p in (output/'evaluation'/case/arm).glob('boundary_*.json')} for arm in ARMS}
        common=max(set(available[ARMS[0]]) & set(available[ARMS[1]]),default=None)
        for arm in ARMS:
            base=Path('evaluation')/case/arm
            result=read(base/'result.json');terminal=read(base/'terminal_observation.json');finished=read(base/'job.finished.json')
            receipt=ledger['jobs'].get(case+':'+arm)
            a=compact_callbacks(result['all_callbacks'])
            boundary=read(base/f'boundary_{common:04d}.json') if common is not None else None
            common_q=boundary['quality'] if boundary else None
            clocks=dict(boundary_online_wall_s=result['costs'].get('full_online_wall_s'),
                terminal_online_wall_s=result.get('terminal_costs',{}).get('full_online_wall_s'),
                full_job_wall_s=finished.get('full_job_wall_s'),parent_receipt_wall_s=receipt.get('wall_s') if receipt else None,
                dispatch_wall_s=result['costs'].get('full_dispatch_wall_s'),
                callback_total_s=a.get('callback_latency',{}).get('total_s'))
            clocks['dispatch_minus_callback_residual_s']=difference(clocks['callback_total_s'],clocks['dispatch_wall_s'])
            clocks['boundary_minus_dispatch_residual_s']=difference(clocks['dispatch_wall_s'],clocks['boundary_online_wall_s'])
            cells[case][arm]=dict(status=result['status'],result_time=result['time'],common_boundary_s=common,
                common_quality=quality(common_q) if common_q else None,terminal=terminal,job_finished=finished,
                receipt=receipt,clocks=clocks,all_callbacks=a,cpu=result.get('cpu'),
                identity=dict(shared_manifest_hash=result['shared_manifest_hash'],initial_state_hash=result['initial_state_hash'],
                    evaluation_scope=result.get('evaluation_scope'),execution_provenance=result.get('execution_provenance')),
                consistency=dict(manifest_matches_root=result['shared_manifest_hash']==roots['manifest_hash'],
                    initial_state_matches_root=result['initial_state_hash']==roots['initial_state_hash'],
                    terminal_embedded_matches=result.get('terminal_observation')==terminal,
                    terminal_time_matches=terminal.get('observed_time_s')==result['time'],
                    terminal_quality_matches=terminal.get('quality')==result['quality'],
                    result_quality_matches_common=result['quality']==common_q if common_q else None))
            for t in range(60,scope['target_sim_seconds']+1,60):
                path=base/f'boundary_{t:04d}.json'
                if not (output/path).is_file():missing.append(str(path));continue
                saved=read(path);q=saved['quality']
                row=dict(case=case,arm=arm,time_s=saved['time'],**{k:q.get(k) for k in TREND_FIELDS})
                for cohort in ('initial_airborne','new_requests'):
                    value=q.get('cohorts',{}).get(cohort,{})
                    for key in ('admitted','success','airborne_pending','upstream_pending'):
                        row[cohort+'_'+key]=value.get(key)
                trends.append(row)
        h,l=(cells[case][arm] for arm in ARMS);hq,lq=h['common_quality'],l['common_quality']
        phases={}
        hp=h['all_callbacks'].get('exclusive_phase_timings',{});lp=l['all_callbacks'].get('exclusive_phase_timings',{})
        for name in sorted(set(hp)|set(lp)):
            hv=hp.get(name,{}).get('wall_s');lv=lp.get(name,{}).get('wall_s')
            phases[name]=dict(H_wall_s=hv,L_wall_s=lv,H_minus_L_s=hv-lv if number(hv) and number(lv) else None)
        reductions=sorted((dict(phase=k,**v) for k,v in phases.items() if number(v['H_minus_L_s']) and v['H_minus_L_s']>0),
                          key=lambda v:-v['H_minus_L_s'])[:3]
        case_comparisons[case]=dict(common_boundary_s=common,
            quality_L_minus_H={k:difference(hq.get(k),lq.get(k)) for k in TREND_FIELDS} if hq and lq else None,
            boundary_wall_reduction_fraction=1-l['clocks']['boundary_online_wall_s']/h['clocks']['boundary_online_wall_s']
                if number(h['clocks']['boundary_online_wall_s']) and number(l['clocks']['boundary_online_wall_s']) else None,
            phase_differences=phases,largest_common_measured_phase_reductions=reductions,
            clock_reductions_H_minus_L={k:h['clocks'][k]-l['clocks'][k] if number(h['clocks'][k]) and number(l['clocks'][k]) else None for k in h['clocks']},
            manifest_equal=h['identity']['shared_manifest_hash']==l['identity']['shared_manifest_hash'],
            initial_state_equal=h['identity']['initial_state_hash']==l['identity']['initial_state_hash'],
            evaluation_scope_equal=h['identity']['evaluation_scope']==l['identity']['evaluation_scope'])
    validation_path=output/'replay_validation.json'
    replay_validation=read('replay_validation.json') if validation_path.exists() else dict(status='MISSING_NOT_VERIFIED_BY_REPORTER')
    cohort_path=output/'cohort_touchdowns.json'
    cohort_receipt=read('cohort_touchdowns.json') if cohort_path.exists() else dict(status='MISSING_NOT_VERIFIED_BY_REPORTER')
    return dict(schema='stage3_online_v2_600_report_v1',generated_at=datetime.now(timezone.utc).isoformat(),
        evaluation_scope=scope,run_status=run,cells=cells,comparisons=case_comparisons,replay_validation=replay_validation,cohort_touchdowns=cohort_receipt,
        missing_trend_files=missing,input_sha256=inputs,reporter_sha256=file_hash(Path(__file__)),
        interpretations=['H is heuristic candidate ordering on CP-SAT v8, not the separate energy_slack baseline.',
            'One pair per case; cases differ in both initial density and inflow. No significance or independent generalization claim.',
            'All-callback totals are saved aggregates; coverage totals are callback-aircraft observations, not independent aircraft.',
            'Candidate active includes feature/inference work under the fixed 6s clock; fewer attempts are not isolated causal evidence.',
            'Dispatch-minus-callback and boundary-minus-dispatch are unattributed measured residuals, not an emergency-phase attribution.',
            'No new training, labels, model inference or physics were executed by this reporting script.',
            'These new wall-budget rollouts are not spliced with, or guaranteed to reproduce, the older 60s pair.']),trends


def write_trends(output,rows):
    fields=['case','arm','time_s',*TREND_FIELDS]+[c+'_'+k for c in ('initial_airborne','new_requests') for k in ('admitted','success','airborne_pending','upstream_pending')]
    with (output/'trends.csv').open('x',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def plot_trends(output,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'svg.fonttype':'none','font.size':9})
    fig,axes=plt.subplots(2,3,figsize=(11.5,6.4),constrained_layout=True)
    fields=('L_normal','N_airborne_pending','energy_j')
    labels=('Normal landings','Airborne pending','Cumulative energy (GJ)')
    for i,case in enumerate(CASES):
        for j,(field,label) in enumerate(zip(fields,labels)):
            ax=axes[i,j]
            for arm,color,style,marker,name in [('heuristic_reduced','#777777','--','s','H: heuristic order + CP'),
                                                ('learned_reduced','#46758F','-','o','L: learned order + CP')]:
                local=sorted((r for r in rows if r['case']==case and r['arm']==arm),key=lambda r:r['time_s'])
                if not local or any(not number(r.get(field)) for r in local):raise ValueError('actual trend data missing: '+case+':'+arm+':'+field)
                scale=1.e9 if field=='energy_j' else 1.
                ax.plot([r['time_s'] for r in local],[r[field]/scale for r in local],label=name,color=color,
                        linestyle=style,marker=marker,markersize=3.3,linewidth=1.6)
            ax.set(xlabel='Simulated time (s)',ylabel=label,xlim=(0,600),ylim=(0,None),xticks=[0,120,240,360,480,600])
            ax.set_title(f"({chr(97+i*3+j)}) {case}",loc='left',fontsize=10,fontweight='bold')
            ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False,fontsize=9)
    fig.savefig(output/'trends.png',dpi=300);fig.savefig(output/'trends.svg');plt.close(fig)
    return dict(core_conclusion='Scene-dependent outcomes; medium loses normal landings while high improves them with unresolved emergencies remaining.',
        archetype='Two-case time-series grid',panel_mapping=dict(rows=list(CASES),columns=list(fields)),
        evidence='Saved boundary_0060.json through boundary_0600.json only; no synthetic t0 or prior-run prefix.',
        sample_size='One H/L pair per case; no error bars or significance inference',
        exports=['trends.png','trends.svg','trends.csv'],
        qa=dict(actual_saved_points=True,points_by_cell={case+':'+arm:sum(r['case']==case and r['arm']==arm for r in rows) for case in CASES for arm in ARMS},
                axes_start_at_zero=True,png_dpi=300,editable_svg_text=True,distinguished_by_color_linestyle_and_marker=True,
                source_csv=True,visual_review='PENDING_PRIMARY_VIEW_IMAGE',human_replay_acceptance_unchanged=True))


def fmt(value):return 'missing' if value is None else f'{value:.3f}' if isinstance(value,float) else str(value)


def markdown(s,rows):
    text=['# Online-v2 600s: measured two-case report','',
        'The frozen model has scene-dependent outcomes. Medium is faster with fewer normal and new-request landings; high improves landings and has no emergency touchdown, but still has unresolved emergencies. This does not establish universal no-quality-loss acceleration.',
        '', 'H is heuristic candidate ordering on the rh_cp_sat / CP-SAT v8 backend, not the separate energy_slack heuristic baseline. L shares that backend and the progressive catalog. There was no new fitting or algorithm change.',
        '', '## Completion and measured clocks','',
        '| Case | Arm | Common(s) | Terminal(s) | RC | Boundary online(s) | Terminal online(s) | Job finished(s) | Parent receipt(s) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for case in CASES:
        for arm in ARMS:
            r=s['cells'][case][arm];c=r['clocks']
            text.append(f"| {case} | {arm} | {r['common_boundary_s']} | {r['terminal'].get('observed_time_s')} | {(r['receipt'] or {}).get('exit_code')} | {fmt(c['boundary_online_wall_s'])} | {fmt(c['terminal_online_wall_s'])} | {fmt(c['full_job_wall_s'])} | {fmt(c['parent_receipt_wall_s'])} |")
    text += ['', '`result.costs` stops before checkpoint600; `terminal_costs` includes the final checkpoint/boundary publication; job-finished and parent receipt also include replay/publication and process overhead. They are separate clocks. All-callback aggregates below are the saved `result.all_callbacks`, not last-selection data.',
        '', '## Common-boundary quality, cohorts and emergency status','',
        '| Case/arm | Admitted | Normal / emergency / success | Airborne pending | New-request successes | Initial successes | Energy(J) | Holding(s) | Upstream wait(s) | Declared / unresolved |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for case in CASES:
        for arm in ARMS:
            q=s['cells'][case][arm]['common_quality'];co=q.get('cohorts',{})
            text.append(f"| {case}/{arm} | {q.get('N_admitted_total')} | {q.get('L_normal')}/{q.get('L_emergency')}/{q.get('L_success')} | {q.get('N_airborne_pending')} | {co.get('new_requests',{}).get('success')} | {co.get('initial_airborne',{}).get('success')} | {fmt(q.get('energy_j'))} | {fmt(q.get('holding_time'))} | {fmt(q.get('upstream_wait_time'))} | {q.get('emergency_declared')}/{q.get('emergency_unresolved')} |")
    for case in CASES:
        h,l=(s['cells'][case][a]['common_quality'] for a in ARMS);change=s['comparisons'][case]
        text += ['',f"{case}: boundary wall reduction {100*change['boundary_wall_reduction_fraction']:.2f}%; normal {h.get('L_normal')}→{l.get('L_normal')}, emergency landings {h.get('L_emergency')}→{l.get('L_emergency')}, success {h.get('L_success')}→{l.get('L_success')}. "
            f"Emergency declarations {h.get('emergency_declared')}→{l.get('emergency_declared')}; unresolved {h.get('emergency_unresolved')}→{l.get('emergency_unresolved')}. "
            f"Energy failures {h.get('F_energy_total')}→{l.get('F_energy_total')}; invalid {h.get('terminated_invalid')}→{l.get('terminated_invalid')}; conservation {h.get('conservation')}→{l.get('conservation')}."]
    text += ['', 'Medium lower energy coexists with fewer completed landings, so it is not equal-service efficiency evidence. High has zero emergency touchdowns in L but eight unresolved emergencies, exceeding H\'s seven; it is not an emergency-free outcome.',
        '', '## Actual saved trends','', '![Saved 60s trends](trends.png)','',
        'Rows are medium/high; columns are normal landings, airborne pending and cumulative energy. H is gray dashed/squares; L is muted blue solid/circles. Axes start at zero but no t0 point was invented. One pair per case, without error bars or significance claims. Editable SVG and all plotted values plus new-request/cohort counts are in trends.svg and trends.csv.']
    text += ['',f"Missing saved trend files: {len(s['missing_trend_files'])}; missing points are not fabricated."]
    cohort_records=s['cohort_touchdowns'].get('records',[])
    if cohort_records:
        by_cell={(r['case'],r['arm']):r for r in cohort_records}
        for case in CASES:
            h=by_cell.get((case,ARMS[0]));l=by_cell.get((case,ARMS[1]))
            if h and l and all('normal' in r.get(cohort,{}) for r in (h,l) for cohort in ('initial_airborne','new_requests')):
                i=difference(h['initial_airborne']['normal'],l['initial_airborne']['normal'])
                n=difference(h['new_requests']['normal'],l['new_requests']['normal'])
                text += ['',f"Exact-ID touchdown receipt, {case}: initial normal {h['initial_airborne']['normal']}→{l['initial_airborne']['normal']}; "
                    f"new-request normal {h['new_requests']['normal']}→{l['new_requests']['normal']}. Normal-landing difference decomposes as initial {i:+d} plus new requests {n:+d}. "
                    f"Initial emergency counters: `{h['initial_airborne']}` → `{l['initial_airborne']}`. Success cohorts alone combine normal and emergency landings."]
    for case in CASES:
        times=sorted({r['time_s'] for r in rows if r['case']==case})
        selected=[t for t in (60,480,540,600) if t in times]
        points=[]
        for t in selected:
            v={r['arm']:r['L_normal'] for r in rows if r['case']==case and r['time_s']==t}
            points.append(f"t{t:g}: {v.get(ARMS[0])}/{v.get(ARMS[1])}")
        text += ['',case+' normal landings H/L: '+', '.join(points)+'.']
    text += ['', 'The medium600 run begins with t60 H3/L4, unlike the older separate60s pair H4/L3. These are new wall-budget rollouts; checkpoint cadence and concurrent background differ. The curves are not spliced with the old prefix, and later outcome differences are not claimed to persist throughout the entire window.',
        '', '## All-callback costs and work','',
        '| Case/arm | Callback count / total(s) / P95(s) | Dispatch total(s) / P95(s) | Active candidate(s) | Attempts / queries | Catalog rounds / pruned | Prefix hits/misses | Learned / cached rank |',
        '| --- | --- | --- | ---: | --- | --- | --- | --- |']
    for case in CASES:
        for arm in ARMS:
            r=s['cells'][case][arm];a=r['all_callbacks'];c=a.get('callback_latency',{});di=a.get('dispatch_latency',{});p=a.get('prefix_reuse',{})
            text.append(f"| {case}/{arm} | {c.get('count')} / {fmt(c.get('total_s'))} / {fmt(c.get('p95_s'))} | {fmt(r['clocks']['dispatch_wall_s'])} / {fmt(di.get('p95_s'))} | {fmt(a.get('candidate_active_wall_s'))} | {a.get('physical_attempts')} / {a.get('descriptor_queries')} | {a.get('catalog_round_count')} / {a['catalog_observations']['pruned_total']} | {p.get('prefix_hits')}/{p.get('prefix_misses')} | {a.get('rank_learned')} / {a.get('rank_cache_hits')} |")
    text += ['', '| Case/arm | Features | Inference | CP | Trajectory validation | Prefix validation | Final commit phase |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for case in CASES:
        for arm in ARMS:
            p=s['cells'][case][arm]['all_callbacks'].get('exclusive_phase_timings',{})
            text.append('| '+case+'/'+arm+' | '+' | '.join(fmt(p.get(k,{}).get('wall_s')) for k in ('stage3_features','stage3_inference','cp_solve','trajectory_validation','prefix_validation','commit'))+' |')
    for case in CASES:
        comp=s['comparisons'][case]
        text += ['',f"{case} largest common measured phase reductions H−L: "+', '.join(f"{r['phase']} {r['H_minus_L_s']:.3f}s" for r in comp['largest_common_measured_phase_reductions'])+'.',
            f"Distinct wall reductions H−L: callback {fmt(comp['clock_reductions_H_minus_L']['callback_total_s'])}s, dispatch {fmt(comp['clock_reductions_H_minus_L']['dispatch_wall_s'])}s, boundary online {fmt(comp['clock_reductions_H_minus_L']['boundary_online_wall_s'])}s. These cannot be assigned entirely to CP or callback work."]
    text += ['', 'High CP time increases in L, while candidate/trajectory totals fall and feature/inference overhead is added. All observed pruning totals are zero, so similarity pruning is not an observed contributor. Candidate active time includes feature/inference inside the fixed6s clock; fewer searches do not isolate smarter ranking. Phase differences describe measured totals across different callback counts, not a causal ablation.',
        '', '| Case/arm | Callback deadline misses / count | Timeout / watchdog / scheduler fallback | Full-chain / immediate coverage gaps | Prefix invalidations | Advisor fallback |',
        '| --- | --- | --- | --- | ---: | --- |']
    for case in CASES:
        for arm in ARMS:
            a=s['cells'][case][arm]['all_callbacks']
            text.append(f"| {case}/{arm} | {a.get('deadline_miss_callbacks')} / {a.get('callback_latency',{}).get('count')} | {a.get('timeout_callbacks')} / {a.get('watchdog_callbacks')} / {a.get('scheduler_fallback_callbacks')} | {a.get('coverage_missing_full_callback_aircraft')} / {a.get('coverage_missing_immediate_callback_aircraft')} | {a.get('prefix_reuse',{}).get('prefix_invalidations')} | `{a.get('advisor_fallback_counts')}` |")
    text += ['', 'Coverage gaps are callback-aircraft observations, not distinct independent aircraft or physical infeasibility proofs. Every recorded planning callback misses the5s target here. Dispatch rows lack the deadline label; an aggregate dispatch zero is not proof of real-time feasibility. All available phase timings, cumulative energy-cache observations, truncations and commit-status counts remain in compact analysis_summary.json.',
        '', '## Identity, terminal, replay and figure QA','']
    for case in CASES:
        text.append(f"- {case} paired checks: manifest={s['comparisons'][case]['manifest_equal']}, initial state={s['comparisons'][case]['initial_state_equal']}, evaluation scope={s['comparisons'][case]['evaluation_scope_equal']}.")
        for arm in ARMS:text.append(f"- {case}/{arm}: independent terminal t={s['cells'][case][arm]['terminal'].get('observed_time_s')}; consistency `{s['cells'][case][arm]['consistency']}`.")
    records=s['replay_validation'].get('records')
    if records:
        text += [f"- Replay {r.get('case')}/{r.get('arm')}: `{r.get('validation')}`; touchdown normal/emergency={r.get('touchdown_normal')}/{r.get('touchdown_emergency')}; human acceptance `{r.get('human_acceptance')}`." for r in records]
    else:text.append('- Replay validation receipt missing; this reporter does not execute the runtime validator.')
    text += ['', 'Figure contract: 2×3 actual-time grid; compare scene-dependent quality trajectories, not significance. Data are saved60s observations, exported unchanged to CSV; energy is displayed inGJ only for readability. No synthetic t0, smoothing or uncertainty band. PNG300dpi, SVG text remains editable, and line styles/markers distinguish H/L without color alone. Final image visual QA is PENDING primary view_image; replay HUMAN_PENDING is untouched.',
        '', '## Scope and limitations','',f"Frozen core `{s['evaluation_scope']['core_commit']}` / `{s['evaluation_scope']['core_source_hash']}`; model `{s['evaluation_scope']['model_sha256']}`; normalizer `{s['evaluation_scope']['normalizer_sha256']}`. Original model fingerprint is preserved in evaluation_scope.json; this is a new600s evaluation, not a new fit.",
        '', 'Cases differ in both initial density and inflow and each has one pair. Shared CP backend and driver changes cannot all be attributed to learning. No statistical significance, independent generalization, or universal no-quality-loss acceptance is established. Earlier60s and these600s runs remain separate experiments. Source file hashes and missing-field evidence are recorded in analysis_summary.json.', '']
    return '\n'.join(text)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);output=args.output.resolve()
    if any((output/name).exists() for name in OUTPUTS):raise ValueError('derived report exists; no overwrite or automatic rerun')
    summary,rows=collect(output)
    summary['figure_contract']=plot_trends(output,rows)
    write_trends(output,rows)
    if any(file_hash(output/name)!=value for name,value in summary['input_sha256'].items()):raise ValueError('an original input changed during reporting')
    with (output/'analysis_summary.json').open('x',encoding='utf8') as stream:
        json.dump(summary,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    with (output/'FINAL_REPORT.md').open('x',encoding='utf8') as stream:stream.write(markdown(summary,rows))
    print(json.dumps(dict(status='DERIVED_REPORT_WRITTEN',files=list(OUTPUTS),trend_points=len(rows)),sort_keys=True))


if __name__=='__main__':main()
