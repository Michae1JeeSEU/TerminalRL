"""Small pure aggregation and replay-integrity checks; no research execution."""
import json

from scripts.stage3 import summarize_online_v2 as summary


def test_all_callbacks_and_cumulative_catalog_counters_are_aggregated():
    coverage={'a':dict(full_chain=False,immediate=True,delayed=False,pads=[])}
    records=[]
    for i in (1,2):
        records.append(dict(time=i,wall_s=float(i),candidate_active_wall_s=6.,attempts={'a':i},
            timings=dict(cp_solve=[i,.5*i],stage3_features=[.1*i,.1*i]),prefix_reuse=dict(prefix_hits=i,prefix_misses=1),
            catalog_rounds=[dict(columns=3,cp_wall_s=1.,pruned_columns=2,coverage=coverage),
                            dict(columns=4,cp_wall_s=1.,pruned_columns=5,coverage=coverage)],
            columns=4,deadline_miss=True,submitted=i,commit_status='ACCEPTED'))
    costs=dict(physical_attempts=3,descriptor_queries=7,online_telemetry=dict(counts={'rank:learned':5,'rank:learned_cache':2}))
    result=summary.aggregate_decisions(records,costs,[dict(dispatch_wall_s=4.)])
    assert result['callback_latency']['count']==2 and result['callback_latency']['total_s']==3.
    assert result['callback_latency']['p95_s']==2.
    assert result['exclusive_phase_timings']['cp_solve']['wall_s']==3.
    assert result['candidate_active_wall_s']==12.
    assert result['prefix_reuse']['prefix_hits']==3 and result['rank_learned']==5
    assert result['rank_cache_hits']==2 and result['catalog_round_count']==4
    assert [r['pruned_columns'] for r in result['catalogs']]==[5,5]
    assert result['coverage_missing_full_callback_aircraft']==2
    assert result['deadline_miss_callbacks']==2 and result['dispatch_deadline_tagged_records']==0
    assert 'stage3_inference' not in result['exclusive_phase_timings']


def test_replay_checks_actual_relative_file_bytes_and_trajectory_object_keys(tmp_path):
    data={'trajectories.json':{'actual-hash':{'kind':'MOVE'}},'visual_acceptance.json':{'researcher_visual_acceptance':'HUMAN_PENDING'},'assertions.json':[]}
    for name,value in data.items():(tmp_path/name).write_text(json.dumps(value))
    (tmp_path/'events.jsonl').write_text('\n'.join(json.dumps(dict(type=t)) for t in ('plan_committed','maneuver_started','touchdown')))
    (tmp_path/'states.jsonl').write_text('{"time":0.0}\n{"time":60.0}\n')
    files={p.name:summary.file_hash(p) for p in tmp_path.iterdir()}
    (tmp_path/'replay.json').write_text(json.dumps(dict(files=files,trajectory_hashes=['actual-hash'],code_commit='commit',human_acceptance='HUMAN_PENDING')))
    result=summary.verify_replay(tmp_path,'commit')
    assert result['all_recorded_file_hashes_match'] and result['trajectory_keys_match']
    assert result['touchdown_events']==result['maneuver_started_events']==result['plan_committed_events']==1
    assert result['human_acceptance']=='HUMAN_PENDING'
    assert result['recorded_states']==dict(count=2,time_bounds=[0.,60.],strictly_increasing=True)
    (tmp_path/'assertions.json').write_text('["changed"]')
    assert not summary.verify_replay(tmp_path,'commit')['all_recorded_file_hashes_match']
