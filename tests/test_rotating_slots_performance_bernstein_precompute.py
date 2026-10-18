"""Exact Bernstein metadata characterization against the independent frozen R0."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'terminal_marl').is_dir())
sys.path.insert(0,str(ROOT))
from terminal_marl.core.rotating_slots.performance.provenance import R0_SOURCE_SHA256, run_reference
REFERENCE=ROOT/'artifacts/rotating_slots/performance_reference_r0'
DRAFT=os.environ.get('BERNSTEIN_PROPOSED_SOURCE')
DRAFT_WINDOWS=os.environ.get('BERNSTEIN_PROPOSED_WINDOWS')

QUERY=r'''
from dataclasses import asdict
import math
from terminal_marl.core.rotating_slots import trajectory as tr
from terminal_marl.core.rotating_slots.config import load_config
def exact(v):
    if isinstance(v,float): return {'float.hex':v.hex()}
    if isinstance(v,dict): return {k:exact(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)): return [exact(x) for x in v]
    return v
config=load_config()
stable=[tr.make_stable_plan('s'+ring+str(i),ring,i,.137,10.371)
        for ring,n in (('outer',37),('middle',28),('inner',18)) for i in range(n)]
transfer=tr.generate_transfer('transfer','outer',0,0,0.)
final=tr.generate_final('final','inner',0,0,0.)
assert len(stable)==83 and transfer.accepted and final.accepted
segments=[s for p in stable+[transfer.plan,final.plan] for s in p.segments]
segments.extend(s for pad in range(6) for s in tr.descent_segments(pad,300.137))
coefficients=[(),(0.,),(-0.,),(1,),(-1,2,-3),(1.,-0.,.125),(-0.,0.,-0.)]
for degree in (1,2,3,5,8,12,16,32,64,127):
    coefficients.append(tuple(((-1.)**i)*(i+1)/2.**(i%12+4) for i in range(degree+1)))
coefficients.extend(c for s in segments for c in (s.r,s.theta,s.z))
intervals=((0.,0.),(-0.,0.),(0.,1.),(.137,.371),(.371,.571),
           (math.nextafter(0.,-math.inf),math.nextafter(1.,math.inf)),(.999999999,1.))
inputs=[]; values=[]
def outcome(call):
    try: return {'result':call()}
    except Exception as e: return {'error':type(e).__name__,'message':str(e)}
for index,c in enumerate(coefficients):
    for k in range(4):
        for lo,hi in intervals:
            scale=1./13.7**k
            inputs.append([index,c,lo,hi,k,scale])
            values.append(outcome(lambda:tr.poly_bounds(c,lo,hi,k,scale)))
# Repeated queries exercise existing LRU; metadata counts are intentionally absent
# from semantic comparison because equal results do not imply equal internal work.
repeat=[outcome(lambda c=c,lo=lo,hi=hi,k=k,scale=scale:tr.poly_bounds(c,lo,hi,k,scale))
        for index,c,lo,hi,k,scale in inputs]
assert repeat==values
query_result=exact({'inputs':inputs,'values':values,'repeat':repeat,
    'stable_count':len(stable),'segment_inputs':[s.to_dict() for s in segments],
    'candidates':[asdict(transfer),asdict(final)],'plans':[p.to_dict() for p in stable]})
'''

BOOT=r'''
import sys,json,importlib.util,importlib.abc
sys.dont_write_bytecode=True
sys.path.insert(0,ROOT)
if DRAFT:
    class DraftFinder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname=='terminal_marl.core.rotating_slots.trajectory':
                return importlib.util.spec_from_file_location(fullname,DRAFT)
            if fullname=='terminal_marl.core.rotating_slots.performance.windows' and DRAFT_WINDOWS:
                return importlib.util.spec_from_file_location(fullname,DRAFT_WINDOWS)
    sys.meta_path.insert(0,DraftFinder())
from pathlib import Path
from terminal_marl.core.rotating_slots import trajectory as _tr, energy as _en, safety as _sf, world as _world
from terminal_marl.core.rotating_slots.performance import windows as _windows
same_objects={
    'energy.Interval':_en.Interval is _tr.Interval,
    'energy.intrinsic':_en._intrinsic_bounds_from_polar is _tr._intrinsic_bounds_from_polar,
    'safety.Interval':_sf.Interval is _tr.Interval,
    'safety.TrajectoryPlan':_sf.TrajectoryPlan is _tr.TrajectoryPlan,
    'safety.intrinsic':_sf._intrinsic_bounds_from_polar is _tr._intrinsic_bounds_from_polar}
for _name in ('generate_final','generate_inbound','generate_transfer','make_ground_plan','make_stable_plan'):
    same_objects['world.'+_name]=getattr(_world,_name) is getattr(_tr,_name)
assert all(same_objects.values()),same_objects
imported={}
for _name,_module in list(sys.modules.items()):
    if _name=='terminal_marl' or _name.startswith('terminal_marl.'):
        _path=Path(_module.__file__).resolve()
        if _name=='terminal_marl.core.rotating_slots.trajectory' and DRAFT: _expected=Path(DRAFT).resolve()
        elif _name=='terminal_marl.core.rotating_slots.performance.windows' and DRAFT_WINDOWS: _expected=Path(DRAFT_WINDOWS).resolve()
        else:
            _expected=Path(ROOT)/_name.replace('.','/')
            _expected=(_expected/'__init__.py' if _expected.is_dir() else _expected.with_suffix('.py')).resolve()
        assert _path==_expected,(_name,str(_path),str(_expected))
        imported[_name]=str(_path)
Path(EVIDENCE_PATH).write_text(json.dumps({'same_object_assertions':same_objects,'imported_paths':imported},indent=2),encoding='utf8')
'''

def child(code,output,name):
    boot_path=output/(name+'.bootstrap.json')
    command=[sys.executable,'-I','-B','-c','ROOT='+repr(str(ROOT))+'\nDRAFT='+repr(DRAFT)+'\nDRAFT_WINDOWS='+repr(DRAFT_WINDOWS)+'\nEVIDENCE_PATH='+repr(str(boot_path))+'\n'+BOOT+code]
    process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=600)
    (output/(name+'.stdout.json')).write_text(process.stdout,encoding='utf8')
    (output/(name+'.stderr.txt')).write_text(process.stderr,encoding='utf8')
    (output/(name+'.process.json')).write_text(json.dumps({'argv':command,'returncode':process.returncode},indent=2),encoding='utf8')
    assert process.returncode==0,(process.stdout,process.stderr)
    return json.loads(process.stdout)

@pytest.fixture(scope='module')
def comparison(tmp_path_factory):
    output=tmp_path_factory.mktemp('bernstein_equivalence')
    reference=run_reference(REFERENCE,code=QUERY,timeout=600)
    assert reference['source_sha256']==R0_SOURCE_SHA256
    assert len(reference['actual_source_files_sha256'])==24
    optimized=child(QUERY+'\nprint(json.dumps(query_result,allow_nan=False))\n',output,'opt')
    for name,value in (('r0_results',reference['query_result']),('opt_results',optimized)):
        (output/(name+'.json')).write_text(json.dumps(value,sort_keys=True,allow_nan=False),encoding='utf8')
    provenance={'reference_commit':reference['reference_commit'],'r0_source_sha256':reference['source_sha256'],
        'r0_files':reference['actual_source_files_sha256'],'r0_imported_paths':reference['imported_module_paths'],
        'test_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'opt_source_path':DRAFT or str(ROOT/'terminal_marl/core/rotating_slots/trajectory.py'),
        'opt_source_sha256':sha256(Path(DRAFT or ROOT/'terminal_marl/core/rotating_slots/trajectory.py').read_bytes()).hexdigest(),
        'semantic_exact_match':optimized==reference['query_result'],'cache_counts_required_equal':False,
        'bootstrap_evidence':json.loads((output/'opt.bootstrap.json').read_text(encoding='utf8')),
        'query_count':len(optimized['inputs']),'stable_count':optimized['stable_count'],
        'performance_status':'NOT_QUALIFIED','human_status':'HUMAN_PENDING',
        'result_sha256':{name:sha256((output/(name+'.json')).read_bytes()).hexdigest() for name in ('r0_results','opt_results')}}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    return reference['query_result'],optimized

@pytest.mark.parametrize('field',['inputs','values','repeat','stable_count','segment_inputs','candidates','plans'])
def test_independent_r0_exact(comparison,field):
    reference,optimized=comparison
    assert reference[field]==optimized[field]

CACHE_QUERY=r'''
from terminal_marl.core.rotating_slots import trajectory as tr
import math,sys
def hx(v): return tuple(float(x).hex() for x in v)
def owned(value,seen=None):
    # Independent retained-container accounting, not the implementation estimate.
    if seen is None: seen=set()
    if id(value) in seen: return 0
    seen.add(id(value)); size=sys.getsizeof(value)
    if isinstance(value,dict): return size+sum(owned(k,seen)+owned(v,seen) for k,v in value.items())
    if isinstance(value,(tuple,list)): return size+sum(owned(x,seen) for x in value)
    if type(value) is tr._BernsteinMetadataCache:
        return size+sum(owned(getattr(value,n),seen) for n in value.__slots__)
    return size
cache=tr._BERNSTEIN_METADATA
from collections import OrderedDict
assert type(cache.entries) is OrderedDict
tr.poly_bounds.cache_clear(); tr.bernstein_cache_clear()
assert tr.bernstein_cache_stats()['entries']==0
c=(1.,-3.,.25,-0.,.125)
tr.bernstein_cache_enable(False)
disabled=tr.poly_bounds(c,.137,.371,1,1.)
assert tr.bernstein_cache_stats()['entries']==0
tr.bernstein_cache_enable(True); tr.poly_bounds.cache_clear()
enabled=tr.poly_bounds(c,.137,.371,1,1.)
assert hx(enabled)==hx(disabled)==hx(tr._poly_bounds_uncached(c,.137,.371,1,1.))
assert type(enabled) is tuple
before=tr.poly_bounds.cache_info(); tr.poly_bounds(c,.137,.371,1,1.)
assert tr.poly_bounds.cache_info().hits==before.hits+1 and tr.poly_bounds.cache_info().maxsize==16384
# Bypass the unchanged outer LRU so original signed-zero key aliasing cannot mask
# distinct metadata key observations; the public LRU's existing policy is retained.
cache.clear(); cache.get((0.,),0); cache.get((-0.,),0)
cache.get((1,),0); cache.get((1.,),0)
cache.get((2**53,),0); cache.get((2**53+1,),0)
assert len(cache.entries)==6
populated=dict(estimate=cache.owned_bytes(),observed=owned(cache),stats=cache.stats())
assert populated['estimate']>=populated['observed']
# Proposed cold/end observers read the canonical artifact metadata cache without
# clearing it. Explicit absence is the frozen-R0 compatibility representation.
FixtureWorld=type('FixtureWorld',(),{'__module__':'terminal_marl.core.rotating_slots.world'})
world=FixtureWorld();world._q_cache={};world._energy_cache={};world._reference_templates={}
before_observation=cache.stats()
cold=_windows._cold_cache_snapshot(world)
final=_windows._final_cache_observation(world)
assert cold['module_numerical_caches']['trajectory._BERNSTEIN_METADATA']==dict(before_observation,present=True)
assert final['module_numerical_caches']['trajectory._BERNSTEIN_METADATA']==dict(before_observation,present=True)
assert final['snapshot_scope']=='after rollout and record flush, before replay export'
observation_unchanged=cache.stats()==before_observation
assert final['module_caches_cleared'] is False and observation_unchanged
assert cold['all_named_acceleration_caches_empty'] is False
saved_stats=tr.bernstein_cache_stats
try:
    del tr.bernstein_cache_stats
    assert _windows._cold_cache_snapshot(world)['module_numerical_caches']['trajectory._BERNSTEIN_METADATA']=={'present':False,'entries':0}
    assert _windows._final_cache_observation(world)['module_numerical_caches']['trajectory._BERNSTEIN_METADATA']=={'present':False,'entries':0}
finally: tr.bernstein_cache_stats=saved_stats
unsupported=[((True,),.0,1.,0,1.),((1.,),0,1.,0,1.),((1.,),0.,1.,True,1.),
             ((1.,),0.,1.,4,1.),((1.,)*129,0.,1.,0,1.),((math.inf,),0.,1.,0,1.),
             ((1.,),0.,math.inf,0,1.),((1.,),0.,1.,0,1)]
class Float(float): pass
unsupported.append(((Float(1.),),0.,1.,0,1.))
for args in unsupported:
    assert not tr._bernstein_eligible(*args)
    def result(f):
        try: return ('value',hx(f(*args)))
        except Exception as e: return ('error',type(e).__name__,str(e))
    old=cache.stats().copy()
    assert result(tr.poly_bounds.__wrapped__)==result(tr._poly_bounds_uncached)
    assert cache.stats()==old
# Entry cap, FIFO hit does not move insertion order.
small=tr._BernsteinMetadataCache(max_entries=3)
for x in range(3): small.get((float(x),1.),0)
first=next(iter(small.entries)); original_order=tuple(small.entries)
small.get((0.,1.),0)
assert tuple(small.entries)==original_order
small.get((3.,1.),0)
assert first not in small.entries and len(small.entries)==3 and small.evictions==1
# Byte limit uses independently observed owned containers, populated and churned.
probe=tr._BernsteinMetadataCache(); probe.get(c,1)
budget=probe.owned_bytes()+1024
bounded=tr._BernsteinMetadataCache(max_bytes=budget)
for x in range(80):
    bounded.get((float(x),-3.,.25,-0.,.125),x%4)
    assert bounded.owned_bytes()<=budget and len(bounded.entries)<=4096
    assert bounded.owned_bytes()>=owned(bounded)
churn=dict(estimate=bounded.owned_bytes(),observed=owned(bounded),stats=bounded.stats())
assert bounded.evictions>0
tiny=tr._BernsteinMetadataCache(max_bytes=8192)
record=tiny.get(tuple(float(i) for i in range(128)),0)
assert record is None and len(tiny.entries)==0 and tiny.owned_bytes()<=8192
tr.bernstein_cache_clear(); info=tr.poly_bounds.cache_info()
assert info.currsize>0 and tr.bernstein_cache_stats()['entries']==0
query_result=dict(populated=populated,churn=churn,oversize=tiny.stats(),
    disabled_enabled_hex=hx(enabled),lru_info=info._asdict(),unsupported_count=len(unsupported),
    observation_scope=final['snapshot_scope'],metadata_observation_unchanged=observation_unchanged)
print(json.dumps(query_result,allow_nan=False))
'''

def test_cache_scope_limits_accounting_and_fallback(tmp_path):
    evidence=child(CACHE_QUERY,tmp_path,'cache')
    (tmp_path/'cache_evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf8')
    assert evidence['populated']['estimate']>=evidence['populated']['observed']
    assert evidence['churn']['estimate']>=evidence['churn']['observed']
