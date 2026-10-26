"""Small serialized saved-output fixtures; no simulator, model or solver."""
from copy import deepcopy
import csv
import json

import pytest

from scripts.stage3 import report_online_v2_600 as report


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf8')


def fixture(output):
    scope=dict(target_sim_seconds=600,core_commit='frozen96c',core_source_hash='source',model_sha256='weights',normalizer_sha256='normalizer')
    save(output/'evaluation_scope.json',scope);save(output/'run.finished.json',dict(status='COMPLETED'))
    jobs={}
    for case in report.CASES:
        save(output/'roots'/case/'root.json',dict(manifest_hash='manifest:'+case,initial_state_hash='initial:'+case))
        for ai,arm in enumerate(report.ARMS):
            q=dict(N_admitted_total=20,N_airborne_pending=15+ai,N_upstream_pending=0,L_normal=5-ai,L_emergency=0,L_success=5-ai,
                F_energy_total=0,terminated_invalid=False,conservation=True,energy_j=1.e8+ai,holding_time=100.+ai,
                airborne_time=200.,upstream_wait_time=0.,emergency_declared=2,emergency_unresolved=2,
                cohorts=dict(initial_airborne=dict(admitted=16,success=4,airborne_pending=12),new_requests=dict(admitted=4,success=1-ai,airborne_pending=3+ai)))
            a=dict(callback_latency=dict(count=2,total_s=10.,p95_s=6.),dispatch_latency=dict(count=2,total_s=12.,p95_s=7.),
                candidate_active_wall_s=12.,catalog_round_count=2,physical_attempts=4,descriptor_queries=5,prefix_reuse=dict(prefix_hits=1,prefix_misses=2,prefix_invalidations=1),
                rank_learned=1 if ai else 0,rank_cache_hits=0,exclusive_phase_timings=dict(cp_solve=dict(wall_s=4.,cpu_s=4.,observations=2)),
                deadline_miss_callbacks=2,timeout_callbacks=0,watchdog_callbacks=0,scheduler_fallback_callbacks=0,advisor_fallback_counts={},
                coverage_missing_full_callback_aircraft=3,coverage_missing_immediate_callback_aircraft=2,
                catalogs=[dict(pruned_columns=0,final_columns=4),dict(pruned_columns=0,final_columns=5)],
                coverage_by_callback=[dict(large_payload='must not be copied')],final_selection_only={'last':'not whole-run proxy'})
            terminal=dict(status='OBSERVED',observed_time_s=600.,quality=q,failed=False,terminated_invalid=False)
            result=dict(status='COMPLETE',time=600.,quality=q,cpu=[12],shared_manifest_hash='manifest:'+case,initial_state_hash='initial:'+case,
                evaluation_scope={'case':case},execution_provenance={'source':'frozen'},terminal_observation=terminal,
                costs=dict(full_online_wall_s=20.-ai,full_dispatch_wall_s=12.),terminal_costs=dict(full_online_wall_s=21.-ai),all_callbacks=a)
            base=output/'evaluation'/case/arm
            save(base/'result.json',result);save(base/'terminal_observation.json',terminal)
            save(base/'job.finished.json',dict(status='COMPLETE',full_job_wall_s=22.-ai))
            for t in (60,600):save(base/f'boundary_{t:04d}.json',dict(time=float(t),quality=q))
            jobs[case+':'+arm]=dict(status='completed',exit_code=0,wall_s=23.-ai)
    save(output/'budget_ledger.json',dict(jobs=jobs))


def test_compact_callback_counts_preserve_observed_zero_but_missing_is_not_zero():
    source=dict(catalogs=[dict(pruned_columns=0,final_columns=5)],coverage_by_callback=['large'],physical_attempts=12)
    original=deepcopy(source);result=report.compact_callbacks(source)
    assert source==original and result['physical_attempts']==12
    assert result['catalog_observations']['pruned_total']==0
    assert 'catalogs' not in result and 'coverage_by_callback' not in result
    assert report.compact_callbacks({})['catalog_observations']['pruned_total'] is None
    assert report.difference(None,2) is None


def test_saved_fixture_serialization_exports_no_t0_and_never_modifies_inputs(tmp_path):
    pytest.importorskip('matplotlib')
    fixture(tmp_path)
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    report.main(['--output',str(tmp_path)])
    assert all(p.read_bytes()==value for p,value in before.items())
    assert all((tmp_path/name).is_file() for name in report.OUTPUTS)
    result=report.load(tmp_path/'analysis_summary.json')
    assert 'catalogs' not in result['cells']['medium']['heuristic_reduced']['all_callbacks']
    assert result['cells']['medium']['heuristic_reduced']['clocks']['boundary_online_wall_s']==20.
    assert result['cells']['medium']['heuristic_reduced']['clocks']['terminal_online_wall_s']==21.
    assert result['comparisons']['medium']['clock_reductions_H_minus_L']['boundary_online_wall_s']==1.
    assert result['comparisons']['medium']['manifest_equal'] is True
    assert result['replay_validation']['status']=='MISSING_NOT_VERIFIED_BY_REPORTER'
    assert result['figure_contract']['qa']['png_dpi']==300
    assert result['figure_contract']['qa']['visual_review']=='PENDING_PRIMARY_VIEW_IMAGE'
    assert result['figure_contract']['qa']['points_by_cell']['medium:heuristic_reduced']==2
    with (tmp_path/'trends.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==8 and {float(r['time_s']) for r in rows}=={60.,600.}
    assert all('new_requests_success' in r for r in rows)
    svg=(tmp_path/'trends.svg').read_text()
    assert '<text' in svg
    with pytest.raises(ValueError,match='no overwrite'):report.main(['--output',str(tmp_path)])
