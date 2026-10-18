"""Bounded L0/L1 energy queries; R0 supplies expected values in a fresh child."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from terminal_marl.core.rotating_slots.performance.provenance import (
    R0_COMMIT, R0_SOURCE_SHA256, run_reference,
)

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'artifacts/rotating_slots/performance_reference_r0'

# Identical program, separate interpreters and complete core namespaces.
# The descent container deliberately is not a TrajectoryPlan/approved final.
QUERY=r'''
from dataclasses import asdict
import math
from terminal_marl.core.rotating_slots import energy as en, trajectory as tr, safety as sf
from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.geometry import pad_position
c=load_config()
TIMES=(0.,300.137,720.9445824868908)
LENGTHS=(.031,.2,10.)
MAX_DT=.2
def exact(value):
    if isinstance(value,float):
        if math.isnan(value): return {'float.special':'NaN'}
        if math.isinf(value): return {'float.special':'+Infinity' if value>0 else '-Infinity'}
        return {'float.hex':value.hex()}
    if isinstance(value,dict): return {str(k):exact(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [exact(v) for v in value]
    return value
def cache_info():
    return {'segment_table':en._segment_table.cache_info()._asdict(),
            'integrable_plan':en._integrable_plan.cache_info()._asdict()}
en._segment_table.cache_clear(); en._integrable_plan.cache_clear()
initial_cache=cache_info()
assert initial_cache['segment_table']['currsize']==initial_cache['integrable_plan']['currsize']==0
generated={
    'transfer':tr.generate_transfer('transfer','outer',0,0,0.),
    'normal_final':tr.generate_final('normal_final','inner',0,0,0.),
    'outer_emergency_final':tr.generate_final('outer_emergency_final','outer',0,0,0.,emergency=True),
}
assert all(candidate.accepted and candidate.plan is not None for candidate in generated.values())
plans={name:result.plan for name,result in generated.items()}
assert plans['transfer'].source_ring=='outer' and plans['transfer'].target_ring=='middle'
assert plans['normal_final'].kind=='FINAL_NORMAL'
assert plans['outer_emergency_final'].kind=='FINAL_EMERGENCY'
stable={}
stable_queries=[]
for ring,count in (('outer',37),('middle',28),('inner',18)):
    for slot in range(count):
        key=f'{ring}/{slot}'
        stable[key]=tr.make_stable_plan(key,ring,slot,0.,1000.,config=c)
        for t in TIMES:
            for length in LENGTHS:
                stable_queries.append({'id':f'{key}/{TIMES.index(t)}/{LENGTHS.index(length)}',
                    'plan_id':key,'start':t,'end':t+length,'midpoint':(t+(t+length))/2})
assert len(stable)==83 and len(stable_queries)==747
assert {ring:sum(key.startswith(ring+'/') for key in stable) for ring in ('outer','middle','inner')}=={'outer':37,'middle':28,'inner':18}
class DescentSegmentsOnly:
    fixture_scope='function_level_segments_only_not_approved_trajectory'
    def __init__(self,segments): self.segments=segments
    def intervals(self,start,end):
        for segment in self.segments:
            if segment.start_time<end and segment.end_time>start: yield segment
descent={}; descent_inputs=[]
for pad in range(6):
    for index,t in enumerate(TIMES):
        key=f'pad{pad}/time{index}'
        segments=tr.descent_segments(pad,t,c)
        assert len(segments)==3 and all(s.kind=='DESCENT' for s in segments)
        assert math.dist(segments[0].state(t).p,(*pad_position(pad,c)[:2],15.))<1e-7
        assert math.dist(segments[-1].state(segments[-1].end_time).p,pad_position(pad,c))<1e-7
        for left,right in zip(segments,segments[1:]):
            assert left.end_time==right.start_time
            assert all(math.dist(getattr(left.state(left.end_time),k),getattr(right.state(right.start_time),k))<1e-7
                       for k in ('p','v','a'))
        descent[key]=DescentSegmentsOnly(segments)
        queries=[{'label':'full_three_segments','start':t,'end':segments[-1].end_time}]
        state_times=[]
        for i,s in enumerate(segments):
            queries.append({'label':f'segment{i}_non_grid','start':s.start_time+.137*s.duration,
                            'end':s.start_time+.371*s.duration})
            state_times.extend({'segment':i,'time':when} for when in (s.start_time,(s.start_time+s.end_time)/2,s.end_time))
        for i,s in enumerate(segments[:-1]):
            queries.append({'label':f'junction{i}','start':s.end_time-.031,'end':s.end_time+.037})
        descent_inputs.append({'id':key,'pad_id':pad,'absolute_start':t,
            'fixture_scope':DescentSegmentsOnly.fixture_scope,
            'segments':[s.to_dict() for s in segments],'state_queries':state_times,'energy_queries':queries})
assert len(descent)==18 and {row['pad_id'] for row in descent_inputs}==set(range(6))
arbitrary=[]; remaining_inputs=[]
for key,plan in plans.items():
    first=plan.segments[0]; start,end=plan.start_time,plan.end_time
    width=first.duration/math.ceil(first.duration/MAX_DT)
    queries=[('full',start,end),('at_start',start,start),('at_end',end,end),
        ('short_non_grid',start+.137,start+.168),('non_grid_137_371',start+.137,start+.371),
        ('within_cell',start+3*width+.031,start+3*width+.068),
        ('cross_cell',start+4*width-.031,start+4*width+.037),
        ('trailing_partial',end-.137,end)]
    queries.extend((f'cross_segment_{i}',s.end_time-.031,s.end_time+.037)
                   for i,s in enumerate(plan.segments[:-1]))
    arbitrary.append({'plan_id':key,'cell_width':width,
        'queries':[{'label':label,'start':a,'end':b} for label,a,b in queries]})
    offsets=[0.,.2,.4,.6,.8,1.,.137,.371]
    nows=sorted(set([start+x for x in offsets]+[end-.137,end]+
                    [s.end_time+x for s in plan.segments[:-1] for x in (-.031,.037)]))
    assert all(start<=now<=end for now in nows)
    remaining_inputs.append({'plan_id':key,'binding_scope':'same accepted immutable plan; no world/resource commit',
        'start':start,'end':end,'nows':nows,
        'queries':[{'now':now,'full':[start,end],'used':[start,now],'remaining':[now,end],
                    'increment':[start if i==0 else nows[i-1],now]} for i,now in enumerate(nows)]})
inputs={'max_dt':MAX_DT,'absolute_times':TIMES,'stable_lengths':LENGTHS,
    'stable_plans':{key:p.to_dict() for key,p in stable.items()},'stable_queries':stable_queries,
    'descent_fixtures':descent_inputs,'generated_candidates':{key:asdict(result) for key,result in generated.items()},
    'arbitrary_intervals':arbitrary,'remaining_energy':remaining_inputs}
def energy(plan,a,b):
    result=en.integrate_energy(plan,a,b,max_dt=MAX_DT,config=c)
    assert result.valid,(a,b,asdict(result))
    return asdict(result)
def run_queries():
    c01=[]
    for row in stable_queries:
        plan=stable[row['plan_id']]; st=plan.state(row['midpoint'])
        power=en.evaluate_power(st.v,st.a,c)
        zero_acceleration=en.evaluate_power(st.v,(0.,0.,0.),c)
        assert power.valid and math.hypot(*st.a[:2])>0.
        assert power.battery_w!=zero_acceleration.battery_w
        c01.append({'id':row['id'],'midpoint_state':asdict(st),'power_with_centripetal':asdict(power),
                    'power_zero_acceleration_control':asdict(zero_acceleration),
                    'energy':energy(plan,row['start'],row['end'])})
    c02=[]
    for row in descent_inputs:
        fixture=descent[row['id']]; states=[]
        for query in row['state_queries']:
            state=fixture.segments[query['segment']].state(query['time'])
            power=en.evaluate_power(state.v,state.a,c)
            assert power.valid,(row['id'],query,asdict(power))
            states.append({'segment':query['segment'],'time':query['time'],'state':asdict(state),'power':asdict(power)})
        c02.append({'id':row['id'],'states_and_domain':states,
            'energy':[{'label':q['label'],'result':energy(fixture,q['start'],q['end'])} for q in row['energy_queries']]})
    c03=[{'plan_id':row['plan_id'],'energy':[{'label':q['label'],
        'result':energy(plans[row['plan_id']],q['start'],q['end'])} for q in row['queries']]} for row in arbitrary]
    c04=[]
    for row in remaining_inputs:
        plan=plans[row['plan_id']]; full=energy(plan,row['start'],row['end'])
        series=[]; cumulative_j=cumulative_error=0.
        payload_before=plan.to_dict()
        for q in row['queries']:
            used=energy(plan,*q['used']); remaining=energy(plan,*q['remaining']); increment=energy(plan,*q['increment'])
            cumulative_j+=increment['joules']; cumulative_error+=increment['error_j']
            series.append({'now':q['now'],'used':used,'remaining':remaining,'increment':increment,
                'cumulative_increment_j':cumulative_j,'cumulative_increment_reported_error_j':cumulative_error,
                'used_plus_remaining_minus_full_j':used['joules']+remaining['joules']-full['joules'],
                'partition_reported_error_sum_j':used['error_j']+remaining['error_j']+full['error_j'],
                'cumulative_increment_minus_direct_used_j':cumulative_j-used['joules'],
                'full_minus_used_j':full['joules']-used['joules'],
                'direct_remaining_minus_subtraction_j':remaining['joules']-(full['joules']-used['joules'])})
        assert plan.to_dict()==payload_before
        c04.append({'plan_id':row['plan_id'],'trajectory_hash':plan.trajectory_hash,'full':full,'series':series})
    return {'C01':c01,'C02':c02,'C03':c03,'C04':c04}
before_first=cache_info()
first=run_queries(); after_first=cache_info()
repeat=run_queries(); after_repeat=cache_info()
assert exact(first)==exact(repeat),'same-process repeated query results differ'
query_result=exact({'inputs':inputs,'first':first,'repeat':repeat,
    'cache_observation':{'initial':initial_cache,'before_first':before_first,'after_first':after_first,'after_repeat':after_repeat,
        'scope':'existing energy._segment_table and energy._integrable_plan only; generated candidate validation may warm safety caches'},
    'counts':{'stable_plans':len(stable),'stable_queries_per_pass':len(stable_queries),'pads':6,
              'descent_bindings':len(descent),'descent_segments':sum(len(v.segments) for v in descent.values()),
              'accepted_generated_plans':len(plans),'arbitrary_queries_per_pass':sum(len(r['queries']) for r in arbitrary),
              'remaining_nows_per_pass':sum(len(r['nows']) for r in remaining_inputs)},
    'scope':'bounded L0/L1 characterization only; no catalogue PASS or performance qualification'})
'''


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    output=tmp_path_factory.mktemp('energy_queries')
    def write(name,data):
        path=output/name
        path.write_text(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8')
        return {'path':str(path),'sha256':sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    try:
        reference=run_reference(REFERENCE,code=QUERY,timeout=600)
        assert reference['reference_commit']==R0_COMMIT=='1c5afd37c891a0ed4776f64367fc7264db3c1c54'
        assert reference['source_sha256']==R0_SOURCE_SHA256
        for name,expected in reference['actual_source_files_sha256'].items():
            assert sha256((REFERENCE/name).read_bytes()).hexdigest()==expected
        expected=reference['query_result']
        artifacts={'inputs':write('explicit_inputs.json',expected['inputs']),
                   'r0_results':write('r0_results.json',{k:v for k,v in expected.items() if k!='inputs'})}
        child='import sys,json\nsys.dont_write_bytecode=True\n'+f'sys.path.insert(0,{str(ROOT)!r})\n'+QUERY+'\nprint(json.dumps(query_result,allow_nan=False))\n'
        result=subprocess.run([sys.executable,'-I','-B','-c',child],cwd=ROOT,
                              capture_output=True,text=True,timeout=600)
        (output/'opt_stdout.log').write_text(result.stdout,encoding='utf8')
        (output/'opt_stderr.log').write_text(result.stderr,encoding='utf8')
        (output/'opt_numericexitcode.txt').write_text(str(result.returncode)+'\n',encoding='utf8')
        assert result.returncode==0,result.stderr
        optimized=json.loads(result.stdout)
        artifacts['opt_results']=write('opt_results.json',{k:v for k,v in optimized.items() if k!='inputs'})
        provenance={'schema':'bounded_energy_query_evidence_v1','r0_commit':reference['reference_commit'],
            'r0_source_sha256':reference['source_sha256'],'r0_pid':reference['pid'],
            'r0_actual_source_files_sha256':reference['actual_source_files_sha256'],
            'r0_imported_module_paths':reference['imported_module_paths'],
            'query_source_sha256':sha256(QUERY.encode()).hexdigest(),
            'test_source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'optimized_core_files_sha256':{name:sha256((ROOT/name).read_bytes()).hexdigest()
                for name in reference['actual_source_files_sha256'] if name.endswith('.py')},
            'artifacts':artifacts,'counts':expected['counts'],
            'semantic_exact_match':{k:v for k,v in expected.items() if k!='cache_observation'}==
                                   {k:v for k,v in optimized.items() if k!='cache_observation'},
            'cache_counts_required_equal':False,
            'first_repeat_exact_match':expected['first']==expected['repeat'] and optimized['first']==optimized['repeat'],
            'cache_scope':expected['cache_observation']['scope'],
            'qualification':'NOT_QUALIFIED','human_status':'HUMAN_PENDING',
            'missing_scope':['20 shapes per maneuver','phase wrap/24-bin boundary sides','M-to-I and inbound',
                'middle/inner emergency sources','C04 reserve/qE/event thresholds','complete cache on/off','traffic/qualification']}
        write('provenance.json',provenance)
        return expected,optimized,output
    except Exception as exc:
        def captured(value):
            return value.decode('utf8',errors='replace') if isinstance(value,bytes) else value
        write('failure.json',{'exception_type':type(exc).__name__,'message':str(exc),
                             'stdout':captured(getattr(exc,'stdout',None)),
                             'stderr':captured(getattr(exc,'stderr',None))})
        raise


@pytest.mark.parametrize('section',['inputs','C01','C02','C03','C04','cache_observation','counts'])
def test_isolated_r0_query_equivalence(evidence,section):
    reference,optimized,_=evidence
    if section=='cache_observation':
        for implementation in (reference,optimized):
            observation=implementation[section]
            assert observation['scope']=='existing energy._segment_table and energy._integrable_plan only; generated candidate validation may warm safety caches'
            for name in ('segment_table','integrable_plan'):
                assert all(observation['initial'][name][field]==0 for field in ('hits','misses','currsize'))
                for phase in ('initial','before_first','after_first','after_repeat'):
                    info=observation[phase][name]
                    assert all(type(info[field]) is int and info[field]>=0 for field in ('hits','misses','currsize'))
                    assert type(info['maxsize']) is int and info['maxsize']>0
                    assert info['currsize']<=info['maxsize']
    elif section.startswith('C0'):
        assert optimized['first'][section]==reference['first'][section]
        assert optimized['repeat'][section]==reference['repeat'][section]
    else:
        assert optimized[section]==reference[section]


def test_original_repeat_and_explicit_coverage(evidence):
    reference,optimized,_=evidence
    assert reference['first']==reference['repeat']
    assert optimized['first']==optimized['repeat']
    assert reference['counts']['stable_plans']==83
    assert reference['counts']['stable_queries_per_pass']==747
    assert reference['counts']['pads']==6
    assert reference['counts']['descent_bindings']==18
    assert reference['counts']['descent_segments']==54
    assert reference['counts']['accepted_generated_plans']==3
