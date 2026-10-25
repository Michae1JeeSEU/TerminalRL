"""Derive the final online-v2 report from saved artifacts after archive.

No project runtime imports, simulation, solver, training or model inference.
Only new derived files and FINAL_REPORT.md are written; the archive stub is
copied byte-for-byte before replacement. Original inputs and ledger stay intact.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path


ARMS = ('reference_full', 'heuristic_reduced', 'learned_reduced')
EXPECTED_COMMIT = '96c00d49ce8ee9fb7f570e88cec545bfb6a46a5d'
MODEL_SHA = 'dba25b3393a5580bdc3971d9adf7d6e3cff6360404fdb08bc8fbb6f0b36e3b48'
NORMALIZER_SHA = '1e06b066dad2d657acbe16de6706b180dd4f4ed8ba1bc106cefcc2a3d4f4e7be'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def jsonl(path):
    with Path(path).open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def file_hash(path):
    value = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def latency(rows, field):
    values = sorted(float(row[field]) for row in rows if isinstance(row.get(field), (int, float)) and math.isfinite(row[field]))
    return dict(count=len(values), total_s=math.fsum(values),
                p50_s=values[max(0, math.ceil(.5*len(values))-1)] if values else None,
                p95_s=values[max(0, math.ceil(.95*len(values))-1)] if values else None,
                max_s=max(values) if values else None, quantile_rule='NEAREST_RANK')


def aggregate_decisions(records, costs, dispatch=()):
    """All callback records; observer counts already cover the whole job.

    Catalog pruning is cumulative within a callback: use its last round, not
    the sum of successive cumulative counts. Move-energy cache is cumulative
    across callbacks: retain its last observation. Missing phase timings stay
    absent rather than being invented as observed zeros.
    """
    phases = defaultdict(lambda: dict(wall_s=0., cpu_s=0., observations=0))
    prefix, truncations = Counter(), Counter()
    coverage, catalogs = [], []
    for record in records:
        for key, value in record.get('timings', {}).items():
            phases[key]['wall_s'] += value[0]
            phases[key]['cpu_s'] += value[1]
            phases[key]['observations'] += 1
        prefix.update(record.get('prefix_reuse', {}))
        truncations.update(record.get('truncations', []))
        rounds = record.get('catalog_rounds', [])
        catalogs.append(dict(time=record.get('time'), rounds=len(rounds), final_columns=record.get('columns'),
            columns_by_round=[r['columns'] for r in rounds], cp_by_round_s=[r['cp_wall_s'] for r in rounds],
            pruned_columns=rounds[-1]['pruned_columns'] if rounds else None,
            expansion_reasons=rounds[-1].get('expansion_reasons', []) if rounds else []))
        if rounds:
            rows = rounds[-1]['coverage']
            coverage.append(dict(time=record.get('time'), source='FINAL_CATALOG_ROUND', aircraft=len(rows),
                missing_full=[aid for aid, row in rows.items() if not row['full_chain']],
                missing_immediate=[aid for aid, row in rows.items() if not row['immediate']],
                missing_delayed=[aid for aid, row in rows.items() if not row['delayed']],
                distinct_pad_count=len({pad for row in rows.values() for pad in row['pads']})))
        elif record.get('aircraft_choices'):
            rows = record['aircraft_choices']
            coverage.append(dict(time=record.get('time'), source='AIRCRAFT_CHOICES', aircraft=len(rows),
                missing_full=[r['aircraft_id'] for r in rows if not r['full_normal_chain_count']],
                missing_immediate=[r['aircraft_id'] for r in rows if not r['immediate_move_candidates']],
                missing_delayed=None, distinct_pad_count=len({p for r in rows for p in r['candidate_pad_ids']})))
    telemetry = costs.get('online_telemetry', {})
    return dict(callback_latency=latency(records, 'wall_s'), dispatch_latency=latency(dispatch, 'dispatch_wall_s'),
        dispatch_deadline_tagged_records=sum('deadline_miss' in r for r in dispatch),
        full_online_wall_s=costs.get('full_online_wall_s'), full_dispatch_wall_s=costs.get('full_dispatch_wall_s'),
        process_cpu_s=costs.get('process_cpu_s'), exclusive_phase_timings=dict(phases),
        candidate_active_wall_s=math.fsum(r.get('candidate_active_wall_s', 0.) for r in records),
        candidate_clock_regimes=dict(Counter(r.get('candidate_timing_regime', 'MISSING') for r in records)),
        attempts_from_callback_counters=sum(sum(r.get('attempts', {}).values()) for r in records),
        physical_attempts=costs.get('physical_attempts'), descriptor_queries=costs.get('descriptor_queries'),
        generation_wall_s=costs.get('generation_wall_s'), energy_wall_s=costs.get('energy_wall_s'),
        column_validation_wall_s=costs.get('column_validation_wall_s'),
        catalog_round_count=sum(c['rounds'] for c in catalogs), catalogs=catalogs, coverage_by_callback=coverage,
        coverage_missing_full_callback_aircraft=sum(len(r['missing_full']) for r in coverage),
        coverage_missing_immediate_callback_aircraft=sum(len(r['missing_immediate']) for r in coverage),
        prefix_reuse=dict(prefix), last_cumulative_move_energy_cache=records[-1].get('move_energy_cache', {}) if records else {},
        rank_learned=telemetry.get('counts', {}).get('rank:learned', 0),
        rank_cache_hits=telemetry.get('counts', {}).get('rank:learned_cache', 0),
        advisor_fallback_counts=telemetry.get('fallback_counts', {}),
        scheduler_fallback_callbacks=sum(bool(r.get('fallback')) for r in records),
        stale_callbacks=sum(bool(r.get('stale')) for r in records),
        deadline_miss_callbacks=sum(bool(r.get('deadline_miss')) for r in records),
        watchdog_callbacks=sum(bool(r.get('watchdog')) for r in records),
        timeout_callbacks=sum(bool(r.get('watchdog')) or any(r.get(k) in ('TIMEOUT', 'WATCHDOG') for k in
            ('status', 'stop_reason', 'solver_stop_reason')) or bool(set(r.get('truncations', [])) & {'TIMEOUT', 'WATCHDOG'}) for r in records),
        truncations=dict(truncations), commit_statuses=dict(Counter(r.get('commit_status', 'MISSING') for r in records)),
        submitted_prefix_plans=sum(r.get('submitted', 0) for r in records),
        observer_counts=telemetry.get('counts', {}),
        final_selection_only=telemetry.get('last_selection'),
        note='Coverage counts sum callback-aircraft observations, not distinct aircraft; observer counters span the job.')


def verify_replay(path, expected_commit):
    index = load(path/'replay.json')
    files = {}
    for name, expected in index['files'].items():
        member = (path/name).resolve()
        if not member.is_relative_to(path.resolve()):
            raise ValueError('replay member escapes its directory')
        actual = file_hash(member) if member.is_file() else None
        files[name] = dict(expected_sha256=expected, actual_sha256=actual, matches=actual==expected)
    trajectories = load(path/'trajectories.json')
    if not isinstance(trajectories, dict):
        raise ValueError('observed trajectories.json must be a hash-keyed object')
    events = jsonl(path/'events.jsonl')
    states = jsonl(path/'states.jsonl')
    times = [row['time'] for row in states]
    event_counts = Counter(r['type'] for r in events)
    return dict(replay_index_sha256=file_hash(path/'replay.json'), code_commit=index['code_commit'],
        code_commit_matches=index['code_commit']==expected_commit, files=files,
        all_recorded_file_hashes_match=all(r['matches'] for r in files.values()),
        trajectory_count=len(trajectories), trajectory_keys_match=sorted(trajectories)==sorted(index['trajectory_hashes']),
        event_counts=dict(event_counts), touchdown_events=event_counts['touchdown'],
        recorded_states=dict(count=len(states),time_bounds=[times[0],times[-1]] if times else None,
                             strictly_increasing=all(a<b for a,b in zip(times,times[1:]))),
        maneuver_started_events=event_counts['maneuver_started'], plan_committed_events=event_counts['plan_committed'],
        human_acceptance=index.get('human_acceptance'), visual_acceptance=load(path/'visual_acceptance.json'),
        assertion_archive=load(path/'assertions.json'),
        authoritative_members=['states.jsonl', 'events.jsonl', 'trajectories.json'],
        sha_verification_is_not_human_acceptance=True)


def collect(output):
    """Read only. All writes are confined to main after this succeeds."""
    protocol=load(output/'online_v2_protocol.json');binding=load(output/'run_binding.generated.json')
    frozen=load(output/'source_freeze.json');card=load(output/'model/model_card.json')
    history=jsonl(output/'model/training_history.jsonl');ledger=load(output/'budget_ledger.json')
    if len(history)!=card['epochs_completed'] or [r['epoch'] for r in history]!=list(range(1,len(history)+1)):
        raise ValueError('training history epochs do not match the saved model card')
    if card['best_epoch'] not in [r['epoch'] for r in history]:raise ValueError('best epoch absent from actual history')
    dev=[]
    for rid in protocol['development_case_ids']:
        for arm in ARMS:
            path=output/'evaluation/same_state'/rid/arm/'result.json';row=load(path)
            dev.append(dict(root_id=rid,arm=arm,status=row['status'],certified=row['certified'],objective=row['objective'],
                execution_provenance=row['execution_provenance'],source_file=str(path.relative_to(output)),sha256=file_hash(path),
                aggregate=aggregate_decisions(row['scheduler_records'],row['costs'])))
    comparison=load(output/'closed_loop_comparison.json');closed=[]
    physical_root=load(output/'roots'/protocol['closed_loop_case_id']/'root.json')
    by_arm={r['arm']:r for r in comparison['rows']}
    for arm in ARMS[1:]:
        base=output/'evaluation/closed_loop'/protocol['closed_loop_case_id']/arm
        result=load(base/'result.json');terminal=load(base/'terminal_observation.json')
        common=by_arm[arm];boundary=load(base/f"boundary_{int(comparison['common_completed_boundary_s']):04d}.json")
        replay=verify_replay(base/'replay',frozen['source_commit'])
        decisions=jsonl(base/'replay/decisions.jsonl');dispatch=jsonl(base/'replay/dispatch_timings.jsonl')
        replay['decision_count_matches_reported']=len(decisions)==result['costs']['planning_latency']['count']
        replay['dispatch_matches_result']=dispatch==result['dispatch_timings']
        replay['touchdown_count_matches_common_quality']=replay['touchdown_events']==common['quality']['L_success']
        closed.append(dict(arm=arm,status=result['status'],time=result['time'],common_time=common['time'],
            process_receipt=ledger['jobs']['evaluate:'+protocol['closed_loop_case_id']+':'+arm],
            common_quality=common['quality'],terminal=terminal,holding_energy_by_ring=common.get('holding_energy_by_ring'),
            aggregate=aggregate_decisions(decisions,result['costs'],dispatch),replay=replay,
            identity=dict(manifest_hash=result['shared_manifest_hash'],initial_state_hash=result['initial_state_hash'],
                execution_provenance=result['execution_provenance']),
            consistency=dict(common_equals_boundary_quality=common['quality']==boundary['quality'],
                manifest_matches_root=result['shared_manifest_hash']==boundary['shared_manifest_hash']==common['shared_manifest_hash']==physical_root['manifest_hash'],
                initial_state_matches_root=result['initial_state_hash']==boundary['initial_state_hash']==common['initial_state_hash']==physical_root['initial_state_hash'],
                result_equals_common_quality=result['quality']==common['quality'],
                terminal_quality_equals_result=terminal['quality']==result['quality'],
                terminal_times_match=terminal['observed_time_s']==result['time']==common['time'],
                terminal_embedded_equals_independent=result['terminal_observation']==terminal,
                execution_provenance_matches_binding=result['execution_provenance']==binding['execution_provenance']),
            source_file=str((base/'result.json').relative_to(output)),result_sha256=file_hash(base/'result.json')))
    original_labels=[];pair_sources=Counter()
    for spec in protocol['config']['mid_snapshots']:
        rid=spec['root_id']+'_t30';labels=load(output/'labels_reindexed'/rid/'labels.json')
        pair_sources.update(r['label_source'] for r in labels['preference_pairs'])
        original_labels.append(dict(root_id=rid,complete_cost_facts=labels['complete_cost_facts'],
            contribution_labels=sum(bool(r['contribution_mask']) for r in labels['incremental_contribution_evidence']),
            pairs=len(labels['preference_pairs'])))
    model_sha=file_hash(output/'model/model.pt');normalizer_sha=file_hash(output/'model/feature_normalizer.json')
    cost_diagnostics={k:v for k,v in card.get('cost_diagnostics',{}).items() if k!='predictions'}
    repair=load(output/'implementation_repair_pretrain/repair_receipt.json')
    recovery=load(output/'implementation_repair_pretrain/recovery_review.json')
    isolation=load(output/'source_isolation/resume.finished.json')
    full_h,full_l=(r['aggregate']['full_online_wall_s'] for r in closed)
    dispatch_h,dispatch_l=(r['aggregate']['full_dispatch_wall_s'] for r in closed)
    qh,ql=(r['common_quality'] for r in closed)
    regression=comparison['quality_differences']['quality_regression_observed']
    important_inputs=['source_freeze.json','run_binding.generated.json','online_v2_protocol.json','budget_ledger.json',
        'model/model_card.json','model/training_history.jsonl','same_state_comparison.json','closed_loop_comparison.json',
        'pretraining_review.json','implementation_repair_pretrain/repair_receipt.json',
        'implementation_repair_pretrain/recovery_review.json','source_isolation/resume.finished.json']
    return dict(schema='stage3_online_v2_analysis_summary_v1',created_at=datetime.now(timezone.utc).isoformat(),
        source=dict(execution_commit=frozen['source_commit'],execution_source_hash=frozen['source_hash'],
            expected_commit_matches=frozen['source_commit']==EXPECTED_COMMIT,
            collection_source=binding['execution_provenance'].get('collection_source')),
        training=dict(**{k:v for k,v in card.items() if k not in ('fingerprint','cost_diagnostics')},
            model_sha256_actual=model_sha,normalizer_sha256_actual=normalizer_sha,
            model_sha_matches_card=model_sha==card['model_hash'],model_sha_matches_expected=model_sha==MODEL_SHA,
            normalizer_sha_matches_expected=normalizer_sha==NORMALIZER_SHA,
            history=history,history_sha256=file_hash(output/'model/training_history.jsonl'),cost_diagnostics=cost_diagnostics),
        dataset_index=load(output/'dataset_index.json'),added_middle_labels=original_labels,new_pair_sources=dict(pair_sources),
        contribution_or_cost_tiebreak_pairs=sum(n for s,n in pair_sources.items() if s in
            ('equal_quality_incremental_preference','equal_quality_measured_cost_preference')),
        development=dev,development_comparisons=load(output/'same_state_comparison.json')['reference_comparisons'],
        closed=closed,common_completed_boundary_s=comparison['common_completed_boundary_s'],
        quality_differences=comparison['quality_differences'],terminal_observations=comparison['terminal_observations'],
        comparison_identity=dict(manifest_equal=closed[0]['identity']['manifest_hash']==closed[1]['identity']['manifest_hash'],
            initial_state_equal=closed[0]['identity']['initial_state_hash']==closed[1]['identity']['initial_state_hash'],
            execution_provenance_equal=closed[0]['identity']['execution_provenance']==closed[1]['identity']['execution_provenance']),
        outcome=dict(full_wall_reduction_fraction=1.-full_l/full_h,dispatch_wall_reduction_fraction=1.-dispatch_l/dispatch_h,
            energy_reduction_fraction=1.-ql['energy_j']/qh['energy_j'],holding_increase_fraction=ql['holding_time']/qh['holding_time']-1.,
            normal_landing_difference=ql['L_normal']-qh['L_normal'],quality_regression_observed=regression,
            no_quality_loss_acceptance='FAILED' if regression else 'NO_REGRESSION_OBSERVED_IN_THIS_PAIR_ONLY',
            independent_generalization='NOT_ASSESSED',causal_learning_speedup='NOT_ESTABLISHED'),
        recovery=dict(original_repair_status=repair.get('status'),error_type=repair.get('error_type'),error=repair.get('error'),
            recovery_status=recovery.get('status'),pretraining_review_status=load(output/'pretraining_review.json')['status'],
            source_isolation={k:v for k,v in isolation.items() if k!='new_jobs'}),
        ledger=dict(started_epoch=ledger['started_epoch'],phase_caps=ledger['phase_caps'],phase_used=ledger['phase_used'],
            elapsed_floor_s=ledger['elapsed_floor_s'],paused_at=ledger['paused_at'],active_phase=ledger['active_phase'],
            job_statuses={k:dict(status=v['status'],exit_code=v.get('exit_code'),wall_s=v.get('wall_s')) for k,v in ledger['jobs'].items()}),
        input_sha256={name:file_hash(output/name) for name in important_inputs},
        historical_comparison=dict(source='PRIMARY_REVIEW_PROVIDED_ROUNDED_ARCHIVE_VALUES',
            archive_path=protocol['config'].get('encoding_archive_readonly'),
            encoding_L_full_wall_s=532.226,encoding_H_full_wall_s=512.575,
            L_full_wall_change_fraction=full_l/532.226-1.,H_full_wall_change_fraction=full_h/512.575-1.,
            isolated_causal_comparison=False),
        primary_reported_replay_validation=dict(source='PRIMARY_REVIEW_MESSAGE_NOT_RERUN_BY_REPORTER',
            validator='ReplayBundle.validate',execution_commit=EXPECTED_COMMIT,H_state_count=89,L_state_count=86,
            time_bounds=[0.,60.],both_passed=True,human_acceptance_unchanged=True),
        post_execution_maintenance=dict(commit='7d103c9cc5fd84d695e46ac92c5494f839be672a',
            purpose='Repair diagnostic bool counting; execution remains frozen96c; no refit or rerun'),
        limitations=['Two DEVELOPMENT_SEEN states and one paired 60s run; no independent generalization assessment.',
            'R is bounded-catalog reference evidence, not a global optimum.',
            'Historical protocols differ in candidate clock, telemetry and prefix reuse; timing comparison is descriptive.',
            'Feature/inference work inside next(catalogs) consumes the fixed candidate-active budget; only selection is excluded.',
            'Fewer attempts and CP rounds coexist with ranking overhead and quality loss; no isolated causal attribution.',
            'Replay hashes are byte-integrity checks; HUMAN_PENDING remains unchanged.'])


def training_curve(output, training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.fonttype']='none'
    history=training['history'];epochs=[r['epoch'] for r in history]
    fig,axes=plt.subplots(1,2,figsize=(9.,3.4),constrained_layout=True)
    for ax,field,label,color in zip(axes,('train_loss','validation_rank_loss'),
                                   ('Training objective','Validation rank loss'),('#557A95','#995A58')):
        ax.plot(epochs,[r[field] for r in history],color=color,marker='o',markersize=3.,linewidth=1.4)
        ax.axvline(training['best_epoch'],color='#555555',linestyle='--',linewidth=1.,label='Selected epoch')
        ax.set(xlabel='Epoch',ylabel=label,ylim=(0,None),xticks=epochs[::2])
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('One supervised run (seed '+str(training['seed'])+'); actual saved history',fontsize=10)
    fig.savefig(output/'training_curve.png',dpi=300)
    fig.savefig(output/'training_curve.svg')
    plt.close(fig)
    with (output/'training_curve.csv').open('x',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=('epoch','train_loss','validation_rank_loss','elapsed_s'))
        writer.writeheader();writer.writerows(history)
    return dict(files=['training_curve.png','training_curve.svg','training_curve.csv'],
        source='model/training_history.jsonl',source_sha256=training['history_sha256'],
        epochs=len(history),best_epoch=training['best_epoch'],smoothing=False,uncertainty_bands=False,
        qa=dict(data_is_saved_history=True,y_axes_start_at_zero=True,svg_text_editable=True,
                no_synthetic_epochs=True,source_data_exported=True,visual_review='PENDING'))


def fmt(value):
    return 'not recorded' if value is None else f'{value:.3f}' if isinstance(value,float) else str(value)


def report(summary):
    t=summary['training'];outcome=summary['outcome'];closed=summary['closed']
    text=['# Stage3 online v2 — final measured report','',
        f"At common simulated {summary['common_completed_boundary_s']}s, L reduced full wall time by {100*outcome['full_wall_reduction_fraction']:.2f}%; normal landings changed by {outcome['normal_landing_difference']:+d}. "
        f"No-quality-loss acceptance: **{outcome['no_quality_loss_acceptance']}**. Independent generalization and a causal learning speedup are not established.",
        '', '## Collection and the only model fit','',
        f"Seed {t['seed']}; {t['parameter_count']:,} parameters; {t['epochs_completed']} actual epochs; selected epoch {t['best_epoch']}; validation rank loss {t['validation_rank_loss']:.9f}. "
        f"Eligible train groups {t['train_groups']} ({t['train_pairs']} pairs); validation ranking groups {t['validation_groups']} ({t['validation_pairs']} pairs). No second fit or hyperparameter sweep.",
        '', '| Added snapshot | Complete cost facts | Certified contributions | Pairs |','| --- | ---: | ---: | ---: |']
    text += [f"| {r['root_id']} | {r['complete_cost_facts']} | {r['contribution_labels']} | {r['pairs']} |" for r in summary['added_middle_labels']]
    index=summary['dataset_index'];diag=t['cost_diagnostics']
    text += ['',f"Merged dataset: {index['contexts']} contexts, {index['pairs']} pairs. Added-pair sources: `{json.dumps(summary['new_pair_sources'],sort_keys=True)}`. "
        f"Contribution/cost tie-break pairs: {summary['contribution_or_cost_tiebreak_pairs']}; saved contribution evidence alone does not show that contribution preferences drove training.",
        '',f"Cost head: {diag['count']} measured validation examples; log1p-ms MAE {diag['mae_log1p_ms']:.9f}, training-mean baseline {diag['constant_train_mean_mae_log1p_ms']:.9f}; "
        f"ordering changed in {diag['ordering_changed_contexts']}/{diag['comparable_order_context_count']} comparable covered contexts. This offline subset is not an online-benefit test.",
        '', '![Actual training history](training_curve.png)','',
        'Curves use every saved epoch without smoothing or invented uncertainty. Editable SVG and source CSV accompany the PNG.',
        '', '## Six development callbacks','',
        '| State | Arm | Full wall(s) | Objective | Callback(s) | Prefix hits/misses | Catalog rounds |',
        '| --- | --- | ---: | --- | ---: | --- | ---: |']
    for row in summary['development']:
        a=row['aggregate'];p=a['prefix_reuse']
        text.append(f"| {row['root_id']} | {row['arm']} | {fmt(a['full_online_wall_s'])} | `{json.dumps(row['objective'])}` | {fmt(a['callback_latency']['total_s'])} | {p.get('prefix_hits',0)}/{p.get('prefix_misses',0)} | {a['catalog_round_count']} |")
    better=sum(bool(r['bounded_reference_better']) for r in summary['development_comparisons'])
    text += ['',f"The bounded reference was better in {better}/{len(summary['development_comparisons'])} progressive comparisons. "
        'Certification and accepted commits establish the common safety/transaction path, not equal scheduling quality. R is not a global optimum.',
        '', '## Paired closed loop at the common boundary','',
        '| Arm | Full to boundary(s) | Process receipt(s) | Dispatch wall(s) | Callbacks | Callback P95(s) | Dispatch P95(s) | Normal landed | Airborne pending | Energy(J) | Holding(s) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in closed:
        a=row['aggregate'];q=row['common_quality']
        text.append(f"| {row['arm']} | {fmt(a['full_online_wall_s'])} | {fmt(row['process_receipt']['wall_s'])} | {fmt(a['full_dispatch_wall_s'])} | {a['callback_latency']['count']} | {fmt(a['callback_latency']['p95_s'])} | {fmt(a['dispatch_latency']['p95_s'])} | {q['L_normal']} | {q['N_airborne_pending']} | {q['energy_j']:.3f} | {q['holding_time']:.3f} |")
    text += ['',f"Quality differences (L minus H): `{json.dumps(summary['quality_differences']['learned_minus_control'],sort_keys=True)}`. "
        f"Recorded deterioration fields: `{summary['quality_differences']['forbidden_deterioration_fields']}`; lexicographic episode worse: {summary['quality_differences']['lexicographic_episode_worse']}.",
        '',f"Energy was {100*outcome['energy_reduction_fraction']:.2f}% lower while holding time was {100*outcome['holding_increase_fraction']:.2f}% higher. "
        'The lower energy coexists with one fewer normal landing; it is not evidence of better efficiency at equal service.',
        'Full-to-boundary wall starts at job entry and ends after completing the simulated boundary. The parent process receipt also includes process startup and post-boundary artifact/replay work; these are separate measured boundaries.',
        '', '## All-callback computation totals','',
        'Closed totals below use every replay decision, not the final selection. Phase values are exclusive wall seconds; missing phases are not recorded. Other phases and CPU totals are retained in analysis_summary.json.',
        '', '| Arm | Candidate active | Candidate exclusive | Features | Inference | CP | Trajectory validation | Prefix validation | Commit | Attempts | Queries | Learned/cache rank |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in closed:
        a=row['aggregate'];p=a['exclusive_phase_timings'];v=lambda name:fmt(p.get(name,{}).get('wall_s'))
        text.append(f"| {row['arm']} | {fmt(a['candidate_active_wall_s'])} | {v('candidate_generation_and_single_vehicle')} | {v('stage3_features')} | {v('stage3_inference')} | {v('cp_solve')} | {v('trajectory_validation')} | {v('prefix_validation')} | {v('commit')} | {a['physical_attempts']} | {a['descriptor_queries']} | {a['rank_learned']}/{a['rank_cache_hits']} |")
    text += ['', 'Feature and inference work inside next(catalogs) consumes the fixed 6s candidate-active clock; selection is excluded. Lower wall time coexists with fewer searches/CP work and substantial ranking overhead. Fewer attempts cannot be attributed solely to smarter ranking; this is mechanism evidence and observed behavior, not an isolated ablation or causal estimate.',
        '', '| Arm | Catalog rounds | Missing-full callback-aircraft | Missing-immediate callback-aircraft | Prefix hits/misses | Scheduler fallback | Advisor fallback | Stale | Callback>5s | Timeout/watchdog |',
        '| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |']
    for row in closed:
        a=row['aggregate'];p=a['prefix_reuse']
        text.append(f"| {row['arm']} | {a['catalog_round_count']} | {a['coverage_missing_full_callback_aircraft']} | {a['coverage_missing_immediate_callback_aircraft']} | {p.get('prefix_hits',0)}/{p.get('prefix_misses',0)} | {a['scheduler_fallback_callbacks']} | `{a['advisor_fallback_counts']}` | {a['stale_callbacks']} | {a['deadline_miss_callbacks']} | {a['timeout_callbacks']}/{a['watchdog_callbacks']} |")
    text += ['', 'Coverage totals are repeated callback-aircraft observations, not distinct aircraft or infeasibility proofs. Per-callback columns, pruning, coverage gaps, truncations and all phase timings are preserved in analysis_summary.json.',
        'Callback deadline misses use the recorded 5s target. Dispatch rows have no deadline_miss label in this run; a recorded aggregate zero does not demonstrate on-time dispatch.',
        '', '## Independent terminal and replay checks','']
    for row in closed:
        terminal=row['terminal'];q=terminal['quality'];replay=row['replay']
        text += [f"- {row['arm']}: terminal t={terminal['observed_time_s']}; failed={terminal['failed']}; invalid={terminal['terminated_invalid']}; "
            f"admitted={q['N_admitted_total']}, normal={q['L_normal']}, emergency={q['L_emergency']}, energy failures={q['F_energy_total']}, conservation={q['conservation']}. "
            f"Boundary/terminal checks `{row['consistency']}`. Replay file hashes match={replay['all_recorded_file_hashes_match']}; "
            f"trajectory-key check={replay['trajectory_keys_match']}; source commit matches={replay['code_commit_matches']}; "
            f"actual touchdown/maneuver_started/plan_committed events={replay['touchdown_events']}/{replay['maneuver_started_events']}/{replay['plan_committed_events']}. "
            f"Recorded states `{replay['recorded_states']}`. "
            f"Human acceptance remains `{replay['human_acceptance']}`."]
    text += ['',f"Cross-arm physical/source identity: `{summary['comparison_identity']}`. Hash checks verify recorded bytes and do not set human acceptance.",
        'The primary separately reported that frozen96c ReplayBundle.validate passed both bundles, including config/trajectory reconstruction and strictly increasing state time (H89/L86 samples, 0–60s). This reporter does not rerun that validator; its own byte/key/state checks are recorded above.',
        '', '## Provenance, recovery and limits','',
        f"Execution commit `{summary['source']['execution_commit']}`, source hash `{summary['source']['execution_source_hash']}`. "
        f"Model SHA256 `{t['model_sha256_actual']}`; normalizer SHA256 `{t['normalizer_sha256_actual']}`. "
        f"Recorded/expected model checks={t['model_sha_matches_card']}/{t['model_sha_matches_expected']}; expected normalizer check={t['normalizer_sha_matches_expected']}.",
        '',f"Post-execution diagnostic-count maintenance commit `{summary['post_execution_maintenance']['commit']}` is separate from the frozen execution commit; it did not refit or rerun this matrix.",
        '',f"Collection commit `{(summary['source']['collection_source'] or {}).get('source_commit')}`, source hash `{(summary['source']['collection_source'] or {}).get('source_hash')}`; original fingerprint is retained in analysis_summary.json.",
        '',f"Original pretraining repair receipt remains {summary['recovery']['original_repair_status']}: {summary['recovery']['error_type']} ({summary['recovery']['error']}). "
        f"Independent recovery status: {summary['recovery']['recovery_status']}; pretraining review: {summary['recovery']['pretraining_review_status']}. "
        'The diagnostic failure occurred after derived writes; accepted recovery did not add simulations, solves or fits. The t0 R/H results and failed source-guard startup were preserved; frozen-checkout isolation resumed only the four missing development cells.',
        '',f"Same ledger epoch {summary['ledger']['started_epoch']}; phase caps `{summary['ledger']['phase_caps']}`; phase use `{summary['ledger']['phase_used']}`. "
        'No ledger, stage or training guard is reset by this reporting script. The archived stub is preserved separately; original results/raw/model/ledger are read only.',
        '',f"Historical encoding archive values supplied in primary review (rounded): L 532.226s → {closed[1]['aggregate']['full_online_wall_s']:.3f}s "
        f"({100*summary['historical_comparison']['L_full_wall_change_fraction']:+.2f}%); H 512.575s → {closed[0]['aggregate']['full_online_wall_s']:.3f}s "
        f"({100*summary['historical_comparison']['H_full_wall_change_fraction']:+.2f}%). Protocols differ in candidate-clock accounting, telemetry and prefix reuse; these are descriptive historical values, not isolated optimization effects.",
        '']
    text += ['- '+item for item in summary['limitations']]
    text += ['', 'Detailed fields and input SHA256 values: analysis_summary.json. No human pass or quality-loss tolerance is introduced.', '']
    return '\n'.join(text)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);output=args.output.resolve()
    if load(output/'archive.finished.json').get('status')!='COMPLETED':raise ValueError('run frozen CLI archive first')
    if not (output/'FINAL_REPORT.md').is_file():raise ValueError('archive report stub required')
    names=('analysis_summary.json','training_curve.png','training_curve.svg','training_curve.csv','FINAL_REPORT.archive_stub.md')
    if any((output/name).exists() for name in names):raise ValueError('derived report already attempted; preserve existing files')
    summary=collect(output)
    original_report=(output/'FINAL_REPORT.md').read_bytes()
    with (output/'FINAL_REPORT.archive_stub.md').open('xb') as stream:stream.write(original_report)
    summary['archive_stub_sha256']=sha256(original_report).hexdigest()
    summary['training_curve']=training_curve(output,summary['training'])
    # Input hashes include ledger/results/model; this script never mutates them.
    if any(file_hash(output/name)!=value for name,value in summary['input_sha256'].items()):raise ValueError('an input changed during reporting')
    with (output/'analysis_summary.json').open('x',encoding='utf8') as stream:
        json.dump(summary,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    (output/'FINAL_REPORT.md').write_text(report(summary),encoding='utf8')
    print(json.dumps(dict(status='DERIVED_REPORT_WRITTEN',common_boundary_s=summary['common_completed_boundary_s'],
        outcome=summary['outcome'],files=['analysis_summary.json','FINAL_REPORT.md',*names[1:]]),sort_keys=True))


if __name__=='__main__':main()
