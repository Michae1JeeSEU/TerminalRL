import json
from pathlib import Path

import pytest

from terminal_marl.scripts.analyze_arrival_dispatch import (
    analyze, analyze_arm, array_state, compare_pair, pair_outcomes, physical,
)


def write_arm(tmp_path, name, rows, per=None, outcomes=None, controller='energy_slack'):
    path = tmp_path/name
    path.mkdir()
    state = {'native': {'uavs': [{'uid': 1, 'phase': 'ring', 'position': [0,0,0]}]}, 'outcomes': outcomes or {}}
    (path/'initial_state.json').write_text(json.dumps(state))
    (path/'result.json').write_text(json.dumps({'protocol': {'scenario': 'test', 'seed': 1, 'controller': controller},
        'metrics': {'per_arrival': per or {}}, 'telemetry': {'runtime_shortlist_exhausted': 7}}))
    with (path/'steps.jsonl').open('w') as f:
        for i,row in enumerate(rows):
            f.write(json.dumps({'tick': i, 'state': state, **row})+'\n')
    return path


def test_missing_null_empty_orders_and_scores(tmp_path):
    assert [array_state(x,'a') for x in ({},{'a':None},{'a':[]},{'a':[1]})] == ['missing','null','empty','nonempty']
    path=write_arm(tmp_path,'arm',[
        {}, {'release_order': None, 'diagnostics': {'energy_order': None}},
        {'release_order': [], 'diagnostics': {'energy_order': [], 'replanned':True}},
        {'release_order':[2,1], 'diagnostics': {'energy_order':[1,2], 'replanned':True,
         'candidate_results':[{'order':[1,2],'status':'timeout'}, {'order':[2,1],'status':'evaluated'}]}}])
    result=analyze_arm(path)
    assert result['diagnostics']['order_comparison']=={'unknown':2,'same':1,'different':1}
    assert result['first_order_difference_tick']==3
    assert result['replans'][-1]['selected']['score'] is None
    assert result['replans'][-1]['base']['status']=='timeout'
    assert len(result['selected_differs_from_base_events'])==1


def test_cumulative_telemetry_not_summed_wait_and_numeric_diagnostics(tmp_path):
    state={'native':{'uavs':[]},'telemetry':{'runtime_shortlist_exhausted':7}}
    path=write_arm(tmp_path,'arm',[{'state':state,'diagnostics':{'planning_elapsed_seconds':2}},
                                 {'state':state,'diagnostics':{'planning_elapsed_seconds':3}}],
                   per={'a':{'external_wait':1,'internal_wait':3,'landing_soc':.5},
                        'b':{'external_wait':2,'internal_wait':8,'landing_soc':None}})
    result=analyze_arm(path)
    assert result['native_telemetry_final']['runtime_shortlist_exhausted']==7
    assert result['numeric_diagnostics']['planning_elapsed_seconds']=={'observed_ticks':2,'sum':5,'max':3}
    assert result['wait_derived']['p90']==pytest.approx(9.4)
    assert result['min_landing_soc']==.5


def test_pair_outcomes_align_by_arrival_id_and_keep_unknown_delta():
    a={'per_arrival':{'2':{'external_wait':1,'internal_wait':2,'energy':7},'1':{'energy':10}},
       'final_outcomes':{'2':'emergency_landed','1':'completed'}}
    b={'per_arrival':{'1':{'energy':4},'2':{'external_wait':0,'internal_wait':2,'energy':8,'landing_soc':.4}},
       'final_outcomes':{'1':'emergency_landed','2':'completed','3':'completed'}}
    rows={r['arrival_id']:r for r in pair_outcomes(a,b)}
    assert rows['2']['wait_delta']==-1
    assert rows['2']['energy_delta']==1
    assert rows['1']['energy_delta']==-6
    assert rows['2']['landing_soc_delta'] is None
    assert rows['3']['energy_outcome']=='unknown'


def test_physical_excludes_uuid_and_diagnostics_and_pairs_first_actual_change(tmp_path):
    x={'native':{'uavs':[{'uid':1,'phase':'ring','position':[0,0,0],'trajectory_plan_id':'a'}]}}
    y={'native':{'uavs':[{'uid':1,'phase':'ring','position':[0,0,0],'trajectory_plan_id':'b'}]},'diagnostics':{'x':2}}
    assert physical(x)==physical(y)
    assert physical({})[0] is None
    z={'native':{'uavs':[{'uid':1,'phase':'ring','position':[1,0,0]}]}}
    a=write_arm(tmp_path,'energy',[{'state':x,'dispatch':{'normal_approved_releases':[]}},
                                 {'state':x,'dispatch':{'normal_approved_releases':[1]}}],outcomes={'1':'emergency_landed'})
    b=write_arm(tmp_path,'rolling',[{'state':y,'dispatch':{'normal_approved_releases':[]}},
                                  {'state':z,'dispatch':{'normal_approved_releases':[2]}}],controller='rolling_reorder')
    pair=compare_pair(analyze_arm(a),analyze_arm(b))
    assert pair['initial_physical_equal'] is True
    assert pair['first_normal_release_difference']['tick']==1
    assert pair['first_normal_release_difference']['approved_membership_differs'] is True
    assert pair['first_post_step_physical_difference']['tick']==1


def test_fresh_output_required(tmp_path):
    root=tmp_path/'input';root.mkdir()
    with pytest.raises(ValueError):
        analyze(root, root/'new')
    output=tmp_path/'output';output.mkdir()
    with pytest.raises(FileExistsError):
        analyze(root,output)


@pytest.mark.parametrize('evidence', [None, 'preexisting_artifact', 'newly_run'])
def test_input_evidence_type_in_summary_and_report(tmp_path, evidence):
    root=tmp_path/'input';root.mkdir()
    output=tmp_path/'output'
    kwargs={} if evidence is None else {'input_evidence_type': evidence}
    summary=analyze(root,output,**kwargs)
    expected=evidence or 'preexisting_artifact'
    assert summary['input_evidence_type']==expected
    assert summary['extraction_evidence_type']=='newly_run'
    assert summary['inference_boundary'].startswith('Observed paired scenario-seed traces;')
    report=(output/'report.md').read_text(encoding='utf8')
    assert report.startswith('# Arrival trace extraction\n')
    assert f'Input: {expected}. Extraction: newly_run.' in report


def test_invalid_input_evidence_type_does_not_create_output(tmp_path):
    output=tmp_path/'output'
    with pytest.raises(ValueError, match='input_evidence_type'):
        analyze(tmp_path/'input',output,input_evidence_type='unverified')
    assert not output.exists()


def test_cli_rejects_invalid_input_evidence_type(monkeypatch):
    from terminal_marl.scripts.analyze_arrival_dispatch import main
    monkeypatch.setattr('sys.argv', ['analyzer', '--input', 'input', '--output', 'output',
                                    '--input-evidence-type', 'unverified'])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code==2
