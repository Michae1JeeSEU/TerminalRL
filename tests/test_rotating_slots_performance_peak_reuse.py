"""Exact duplicate peak queries; functional evidence, not P3 qualification."""
import ast
from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

from terminal_marl.core.rotating_slots import energy as en, trajectory as tr
from terminal_marl.core.rotating_slots.config import Config, Section, load_config
from terminal_marl.core.rotating_slots.performance import provenance

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'artifacts/rotating_slots/performance_reference_r0'

P2_PEAK=r'''
def _peak_measure(s,a,b,threshold,c,depth=0):
    polar=s.polar_bounds(a,b)
    bounds=power_bounds(_intrinsic_bounds_from_polar(polar,1),_intrinsic_bounds_from_polar(polar,2),c)
    if bounds is not None:
        low,high=bounds['shaft_w']
        if low>threshold: return b-a
        if high<=threshold: return 0.
    if b-a<1e-7 or depth>=26:
        return b-a
    mid=(a+b)/2
    return _peak_measure(s,a,mid,threshold,c,depth+1)+_peak_measure(s,mid,b,threshold,c,depth+1)
'''

QUERY=r'''
from dataclasses import asdict
import math
from terminal_marl.core.rotating_slots import energy as en, trajectory as tr
from terminal_marl.core.rotating_slots.config import load_config, Config, Section
c=load_config()
def exact(x):
    if isinstance(x,float): return {'float.hex':x.hex()}
    if isinstance(x,dict): return {str(k):exact(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [exact(v) for v in x]
    return x
def bounds(s,a,b):
    return en.power_bounds(s.intrinsic_bounds(a,b,1),s.intrinsic_bounds(a,b,2),c)
threshold=float(c.vehicle.rotor_count*c.energy.reference_motor_continuous_power_kW_each*1000)
segments={}; queries=[]
def add(name,s,a,b):
    segments[name]=s
    pb=bounds(s,a,b)
    assert pb is not None
    for label,value in [('all_above',float(pb['shaft_w'][0]-1.)),
                        ('all_below',float(pb['shaft_w'][1]+1.)),('continuous_limit',threshold)]:
        queries.append((name,label,float(a),float(b),value,0))
for pad in range(6):
    for t in (0.,300.137,720.94458):
        for i,s in enumerate(tr.descent_segments(pad,t,c)):
            add(f'descent/{pad}/{t}/{i}',s,s.start_time+.137*s.duration,s.start_time+.371*s.duration)
generated={
    'outer_middle':tr.generate_transfer('transfer','outer',0,0,0.),
    'middle_inner':tr.generate_transfer('transfer2','middle',0,0,300.137),
    'normal_final':tr.generate_final('final','inner',0,0,0.),
    'emergency_final':tr.generate_final('emergency','outer',0,0,0.,emergency=True)}
assert all(row.accepted and row.plan is not None for row in generated.values())
cross=[]
for name,row in generated.items():
    for i,s in enumerate(row.plan.segments):
        add(f'{name}/{i}',s,s.start_time+.137*s.duration,s.start_time+.371*s.duration)
    for i,s in enumerate(row.plan.segments[:-1]):
        cross.append((name,i,float(s.end_time-.031),float(s.end_time+.037)))
# A genuine scalar-power crossing, with its bracketing point powers recorded.
s=tr.descent_segments(0,0.,c)[0]
samples=[(s.start_time+s.duration*k/8,en.evaluate_power(s.state(s.start_time+s.duration*k/8).v,
          s.state(s.start_time+s.duration*k/8).a,c).shaft_w) for k in range(9)]
left,right=max(zip(samples,samples[1:]),key=lambda pair:abs(pair[1][1]-pair[0][1]))
assert abs(left[1]-right[1])>1.
cross_threshold=float((left[1]+right[1])/2)
lo,hi=left[0],right[0]
for _ in range(32):
    mid=(lo+hi)/2; st=s.state(mid); power=en.evaluate_power(st.v,st.a,c).shaft_w
    if (power>cross_threshold)==(left[1]>cross_threshold): lo=mid
    else: hi=mid
center=(lo+hi)/2
a=max(s.start_time,center-.01);b=min(s.end_time,center+.01)
pa=en.evaluate_power(s.state(a).v,s.state(a).a,c).shaft_w
pb=en.evaluate_power(s.state(b).v,s.state(b).a,c).shaft_w
assert (pa-cross_threshold)*(pb-cross_threshold)<0.
segments['real_threshold_crossing']=s
queries.append(('real_threshold_crossing','physical_crossing',a,b,cross_threshold,0))
# Controlled numerical fixtures are not approved flight trajectories.
hover=tr.Segment(0.,1.,(123.,),(0.,),(20.,),'POLAR')
st=hover.state(.137); tangent=en.evaluate_power(st.v,st.a,c).shaft_w
segments['controlled_hover_tangent']=hover
queries.extend([('controlled_hover_tangent','narrow_threshold_tangent',.137,.137+4e-8,tangent,0),
                ('controlled_hover_tangent','depth26_threshold_tangent',.137,.371,tangent,26)])
def run():
    def measure(*args):
        first=en._peak_measure(*args)
        repeated=en._peak_measure(*args)
        assert exact(first)==exact(repeated)
        return first
    rows=[]
    for name,label,a,b,value,depth in queries:
        answer=measure(segments[name],a,b,value,c,depth)
        if label=='all_above': assert answer==b-a
        if label=='all_below': assert answer==0.
        rows.append([name,label,a,b,value,depth,answer])
    crossings=[]
    for name,index,a,b in cross:
        plan=generated[name].plan
        pieces=[(seg,max(a,seg.start_time),min(b,seg.end_time)) for seg in plan.segments
                if seg.start_time<b and seg.end_time>a]
        assert len(pieces)==2
        values=[measure(seg,left,right,threshold,c) for seg,left,right in pieces]
        crossings.append([name,index,a,b,values,sum(values)])
    return exact({'queries':rows,'cross_segment':crossings})
if hasattr(en,'peak_cache_clear'):
    assert en.peak_cache_stats()['entries']==0
    en.peak_cache_clear();en.peak_cache_set_enabled(False)
off=run()
if hasattr(en,'peak_cache_clear'):
    en.peak_cache_clear();en.peak_cache_set_enabled(True)
first=run()
first_stats=en.peak_cache_stats() if hasattr(en,'peak_cache_stats') else None
repeat=run()
repeat_stats=en.peak_cache_stats() if hasattr(en,'peak_cache_stats') else None
assert off==first==repeat
# Controlled unresolved-node fixture checks original left-then-right recursion.
controlled=tr.Segment(0.,1.,(127.,.1,.01),(.2,-.02),(23.,.01),'POLAR')
original=en.power_bounds
controlled_results=[];controlled_traces=[]
try:
    for enabled in (False,True):
        if hasattr(en,'peak_cache_clear'):
            en.peak_cache_clear();en.peak_cache_set_enabled(enabled)
        trace=[]
        def unresolved(v,a,config):
            trace.append(exact([v,a]))
            return None
        en.power_bounds=unresolved
        result=en._peak_measure(controlled,.137,.137+1.5e-7,threshold,c)
        controlled_results.append(exact(result));controlled_traces.append(trace)
        assert len(trace)==3
    assert controlled_results[0]==controlled_results[1]
    assert controlled_traces[0]==controlled_traces[1]
finally:
    en.power_bounds=original
    if hasattr(en,'peak_cache_clear'): en.peak_cache_clear()
query_result={'semantic':{'results':first,'segments':exact({k:s.to_dict() for k,s in segments.items()}),
    'real_crossing_point_powers':exact([a,b,pa,pb,cross_threshold]),
    'controlled_unresolved_fixture':{'result':controlled_results[0],'trace':controlled_traces[0],
        'scope':'synthetic unresolved power bounds; function-level recursion fixture only'}},
    'counts':{'pads':6,'descent_bindings':18,'descent_segments':54,'queries':len(queries),
              'cross_segment_queries':len(cross)},'stats':{'first':first_stats,'repeat':repeat_stats}}
'''


@pytest.fixture(scope='module')
def isolated(tmp_path_factory):
    reference=provenance.run_reference(REFERENCE,code=QUERY,timeout=300)
    child=('import sys\nsys.dont_write_bytecode=True\nimport json\n'
           f'sys.path.insert(0,{str(ROOT)!r})\n'+QUERY+
           '\nprint(json.dumps(query_result,allow_nan=False))\n')
    completed=subprocess.run([sys.executable,'-I','-B','-c',child],cwd=ROOT,
                             capture_output=True,text=True,timeout=300)
    output=tmp_path_factory.mktemp('peak_reuse')
    for name,data in [('opt_stdout.log',completed.stdout),('opt_stderr.log',completed.stderr),
                      ('opt_numericexitcode.txt',str(completed.returncode))]:
        (output/name).write_text(data,encoding='utf8')
    assert completed.returncode==0,completed.stderr
    optimized=json.loads(completed.stdout)
    (output/'evidence.json').write_text(json.dumps({'reference':reference,'optimized':optimized,
        'query_source_sha256':sha256(QUERY.encode()).hexdigest(),
        'qualification':'NOT_QUALIFIED; exact duplicate queries only; no full C07 claim'},allow_nan=False),encoding='utf8')
    return reference,optimized


def test_independent_r0_exact_peak_measure(isolated):
    reference,optimized=isolated
    assert reference['reference_commit']==provenance.R0_COMMIT
    assert reference['source_sha256']==provenance.R0_SOURCE_SHA256
    assert reference['query_result']['semantic']==optimized['semantic']
    assert optimized['counts']['pads']==6 and optimized['counts']['descent_segments']==54
    first,repeat=optimized['stats']['first'],optimized['stats']['repeat']
    assert first['logical_requests']==first['actual_computes']+first['hits']
    # Whole-batch eviction is valid: every query's immediate exact duplicate
    # exercises reuse independently of whether the entire matrix fits the cap.
    assert repeat['actual_computes']>=first['actual_computes']
    assert repeat['hits']>first['hits']
    # Explicit depth26 requests bypass caching on both passes.
    assert repeat['non_top_level_computes']>=first['non_top_level_computes']+1


def test_unchanged_peak_body_defaults_and_semantics():
    def peak(path):
        node=deepcopy(next(n for n in ast.parse(path.read_text(encoding='utf8')).body
                           if isinstance(n,ast.FunctionDef) and n.name=='_peak_measure'))
        node.decorator_list=[]
        return node
    # The P2 shared polar jet is already accepted separately; freeze current P2
    # expressions and limits without replacing R0 as the numeric oracle.
    node=peak(ROOT/'terminal_marl/core/rotating_slots/energy.py')
    assert ast.dump(node,include_attributes=False)==ast.dump(ast.parse(P2_PEAK).body[0],include_attributes=False)
    assert ast.literal_eval(node.args.defaults[0])==0
    assert any(isinstance(n,ast.Constant) and n.value==26 for n in ast.walk(node))
    assert any(isinstance(n,ast.Constant) and n.value==1e-7 for n in ast.walk(node))
    provenance.require_same_semantics(provenance.semantic_fingerprint(REFERENCE),
                                     provenance.semantic_fingerprint(ROOT))
    assert inspect.signature(en._peak_measure)==inspect.signature(en._peak_measure_uncached)


@pytest.fixture
def empty_cache():
    prior=en.peak_cache_stats()['enabled']
    en.peak_cache_clear();en.peak_cache_set_enabled(True)
    yield
    en.peak_cache_clear();en.peak_cache_set_enabled(prior)


def owned_bytes(value,seen=None):
    seen=set() if seen is None else seen
    if id(value) in seen: return 0
    seen.add(id(value));size=sys.getsizeof(value)
    if isinstance(value,dict): return size+sum(owned_bytes(k,seen)+owned_bytes(v,seen) for k,v in value.items())
    if isinstance(value,(tuple,list)): return size+sum(owned_bytes(x,seen) for x in value)
    return size


def fixture():
    c=load_config();s=tr.Segment(0.,1.,(123.,),(0.,),(20.,))
    return s,c


def test_actual_original_calls_and_disable(empty_cache):
    s,c=fixture();observed=[];code=en._peak_measure_uncached.__code__
    def record(frame,event,arg):
        if event=='call' and frame.f_code is code: observed.append(frame.f_locals['depth'])
    previous=sys.getprofile()
    try:
        sys.setprofile(record);en.peak_cache_set_enabled(False)
        off=[en._peak_measure(s,.137,.371,1e9,c) for _ in range(4)];off_calls=len(observed)
        en.peak_cache_clear();observed.clear();en.peak_cache_set_enabled(True)
        on=[en._peak_measure(s,.137,.371,1e9,c) for _ in range(4)];on_calls=len(observed)
    finally: sys.setprofile(previous)
    assert off==on and off_calls==4 and on_calls==1
    assert en.peak_cache_stats()['actual_computes']==1 and en.peak_cache_stats()['hits']==3
    en.peak_cache_set_enabled(False)
    assert en._peak_measure(s,.137,.371,1e9,c)==on[0]
    assert en.peak_cache_stats()['entries']==1 and en.peak_cache_stats()['hits']==3
    with pytest.raises(TypeError): en.peak_cache_set_enabled(1)
    en.peak_cache_clear()
    assert all(en.peak_cache_stats()[k]==0 for k in ('entries','hits','actual_computes','logical_requests','evictions','non_top_level_computes'))


def clone_config(c,data):
    return Config(Section(data),c.raw_hash,c.config_hash,c.profile_hash,c.path)


def test_forged_hash_config_changes_actual_peak_answer(empty_cache):
    s,c=fixture();data=c.to_dict();data['vehicle']['fixed_flight_mass_kg']=621.
    changed=clone_config(c,data);state=s.state(.2)
    p0=en.evaluate_power(state.v,state.a,c).shaft_w
    p1=en.evaluate_power(state.v,state.a,changed).shaft_w
    assert p1>p0
    threshold=float((p0+p1)/2)
    old=en._peak_measure(s,.137,.371,threshold,c)
    new=en._peak_measure(s,.137,.371,threshold,changed)
    assert old==0. and new==.371-.137
    assert en._peak_measure_uncached(s,.137,.371,threshold,changed)==new
    assert en.peak_cache_stats()['actual_computes']==2


def test_all_actual_key_fields_and_forged_hash_mutation(empty_cache):
    from dataclasses import replace
    s,c=fixture();a=.137;b=.371;threshold=1e9
    base=en._peak_measure(s,a,b,threshold,c)
    variants=[(replace(s,start_time=-0.),a,b,threshold,c),
              (replace(s,duration=1.001),a,b,threshold,c),
              (replace(s,r=(124.,)),a,b,threshold,c),
              (replace(s,theta=(-0.,)),a,b,threshold,c),
              (replace(s,z=(21.,)),a,b,threshold,c),
              (replace(s,kind='STABLE'),a,b,threshold,c),
              (s,a+1e-8,b,threshold,c),(s,a,b+1e-8,threshold,c),
              (s,a,b,threshold+1.,c)]
    data=c.to_dict();data['energy']['motor_drive_efficiency']=.89
    forged=clone_config(c,data)
    assert forged.profile_hash==c.profile_hash
    variants.append((s,a,b,threshold,forged))
    for args in variants:
        before=en.peak_cache_stats()['actual_computes']
        assert en._peak_measure(*args)==en._peak_measure_uncached(*args)
        assert en.peak_cache_stats()['actual_computes']==before+1
    mutable=clone_config(c,c.to_dict())
    assert en._peak_measure(s,a,b,threshold,mutable)==base
    before=en.peak_cache_stats()['actual_computes']
    changed=mutable.to_dict();changed['vehicle']['fixed_flight_mass_kg']=621.
    object.__setattr__(mutable,'sections',Section(changed))
    assert en._peak_measure(s,a,b,threshold,mutable)==en._peak_measure_uncached(s,a,b,threshold,mutable)
    assert en.peak_cache_stats()['actual_computes']==before+1
    # Mutation of a previously queried Segment cannot hit its old payload key.
    altered=replace(s);en._peak_measure(altered,a,b,threshold,c)
    before=en.peak_cache_stats()['actual_computes']
    object.__setattr__(altered,'r',(125.,))
    en._peak_measure(altered,a,b,threshold,c)
    assert en.peak_cache_stats()['actual_computes']==before+1
    assert all(type(value) is float for value in en._PEAK_CACHE.values())
    assert en._peak_exact_key(0.)!=en._peak_exact_key(-0.)
    assert en._peak_exact_key(1)!=en._peak_exact_key(1.)!=en._peak_exact_key(True)
    assert en._peak_exact_key([1])!=en._peak_exact_key((1,))
    assert en._peak_exact_key({'a':1,'b':2})!=en._peak_exact_key({'b':2,'a':1})


def test_caps_fifo_churn_and_independent_owned_bytes(empty_cache,monkeypatch):
    s,c=fixture()
    assert type(en._PEAK_CACHE) is OrderedDict
    assert en.peak_cache_stats()['max_entries']==4096
    assert en.peak_cache_stats()['max_bytes']==16*1024*1024
    for i in range(5000):
        en._peak_measure(s,.137,.371,1e9+float(i),c)
        if i%1024==0:
            stats=en.peak_cache_stats()
            assert stats['entries']<=4096
            assert owned_bytes(en._PEAK_CACHE)<=stats['estimated_bytes']<=16*1024*1024
    assert en.peak_cache_stats()['evictions']>0
    before=en.peak_cache_stats()['actual_computes'];en._peak_measure(s,.137,.371,1e9,c)
    assert en.peak_cache_stats()['actual_computes']==before+1
    en.peak_cache_clear();monkeypatch.setattr(en,'_PEAK_CACHE_MAX_ENTRIES',2)
    for value in (1e9,1e9+1.,1e9+2.): en._peak_measure(s,.137,.371,value,c)
    assert en.peak_cache_stats()['entries']==2 and en.peak_cache_stats()['evictions']==1
    en.peak_cache_clear();monkeypatch.setattr(en,'_PEAK_CACHE_MAX_BYTES',sys.getsizeof(OrderedDict())+32)
    for _ in range(2): en._peak_measure(s,.137,.371,1e9,c)
    assert en.peak_cache_stats()['entries']==0 and en.peak_cache_stats()['actual_computes']==2
    assert en.peak_cache_stats()['estimated_bytes']<=en.peak_cache_stats()['max_bytes']


def test_ordered_fifo_hits_do_not_reorder_peak_entries(empty_cache,monkeypatch):
    s,c=fixture();monkeypatch.setattr(en,'_PEAK_CACHE_MAX_ENTRIES',3)
    for threshold in (1e9,1e9+1.,1e9+2.): en._peak_measure(s,.137,.371,threshold,c)
    original_keys=tuple(en._PEAK_CACHE)
    en._peak_measure(s,.137,.371,1e9,c)
    assert tuple(en._PEAK_CACHE)==original_keys
    en._peak_measure(s,.137,.371,1e9+3.,c)
    assert tuple(en._PEAK_CACHE)[:2]==original_keys[1:]
    assert original_keys[0] not in en._PEAK_CACHE
    assert en.peak_cache_stats()['hits']==1 and en.peak_cache_stats()['evictions']==1
    stats=en.peak_cache_stats()
    assert owned_bytes(en._PEAK_CACHE)<=stats['estimated_bytes']<=stats['max_bytes']


def test_unsupported_payload_depth_and_exceptions_fall_back(empty_cache):
    import numpy as np
    from types import SimpleNamespace
    s,c=fixture()
    data=c.to_dict();data['unrelated_unsupported_fixture']=object()
    odd=clone_config(c,data)
    broken_data=c.to_dict();broken_data['vehicle']['fixed_flight_mass_kg']='bad'
    broken=clone_config(c,broken_data)
    class SegmentSubclass(tr.Segment): pass
    subclass=SegmentSubclass(0.,1.,(123.,),(0.,),(20.,))
    fixtures=[(s,0,.371,1e9,c,0),(s,np.float64(.137),.371,1e9,c,0),
              (s,.137,.371,1e9,c,False),(s,.137,.371,1e9,c,26),
              (s,.137,.371,float('inf'),c,0),(s,.137,.371,1e9,odd,0),
              (subclass,.137,.371,1e9,c,0),
              (SimpleNamespace(polar_bounds=s.polar_bounds),.137,.371,1e9,c,0),
              (s,.137,.371,1e9,None,0),(s,.137,.371,1e9,c,'bad'),
              (s,None,.371,1e9,c,0),(s,.137,.371,1e9,broken,0),
              (s,.137,.371,1e9,broken,0)]
    def outcome(function,args):
        try:
            result=function(*args)
            return ('result',result.hex() if type(result) is float else result)
        except Exception as exc: return ('exception',type(exc).__name__,str(exc))
    for args in fixtures:
        assert outcome(en._peak_measure,args)==outcome(en._peak_measure_uncached,args)
    assert en.peak_cache_stats()['entries']==0


def test_optional_cold_observation_keeps_warm_entries(empty_cache,monkeypatch):
    from terminal_marl.core.rotating_slots.performance.windows import _cold_cache_snapshot, _final_cache_observation
    FixtureWorld=type('FixtureWorld',(),{'__module__':'terminal_marl.core.rotating_slots.world'})
    world=FixtureWorld();world._q_cache={};world._energy_cache={};world._reference_templates={}
    assert _cold_cache_snapshot(world)['module_numerical_caches']['energy._PEAK_CACHE']['entries']==0
    s,c=fixture();en._peak_measure(s,.137,.371,1e9,c)
    observed=_cold_cache_snapshot(world)
    assert observed['module_numerical_caches']['energy._PEAK_CACHE']['entries']==1
    assert not observed['all_named_acceleration_caches_empty'] and not observed['module_caches_cleared']
    assert en.peak_cache_stats()['entries']==1
    before=en.peak_cache_stats()
    final=_final_cache_observation(world)
    assert final['status']=='OBSERVED'
    assert final['snapshot_scope']=='after rollout and record flush, before replay export'
    assert final['observation_timing_scope']==final['snapshot_scope']
    assert final['module_numerical_caches']['energy._PEAK_CACHE']==dict(before,present=True)
    assert en.peak_cache_stats()==before and final['module_caches_cleared'] is False
    monkeypatch.delattr(en,'peak_cache_stats')
    assert _cold_cache_snapshot(world)['module_numerical_caches']['energy._PEAK_CACHE']=={'present':False,'entries':0}
    assert _final_cache_observation(world)['module_numerical_caches']['energy._PEAK_CACHE']=={'present':False,'entries':0}
