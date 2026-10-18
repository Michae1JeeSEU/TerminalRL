"""Exact execution-query reuse against the independent frozen R0 oracle."""
from dataclasses import replace
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest

from terminal_marl.core.rotating_slots import safety, trajectory, world
from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.models import KinematicState
from terminal_marl.core.rotating_slots.performance.provenance import R0_COMMIT, R0_SOURCE_SHA256, run_reference

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'artifacts/rotating_slots/performance_reference_r0'

QUERY=r'''
from dataclasses import asdict
from types import SimpleNamespace
from terminal_marl.core.rotating_slots import safety as sf,trajectory as tr,world as wo
from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.models import ValidationResult
def exact(x):
    if isinstance(x,float): return {'float.hex':x.hex()}
    if isinstance(x,dict): return {k:exact(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [exact(v) for v in x]
    return x
c=load_config()
def fixture(name,r,theta,z):
    s=tr.Segment(0.,1.,r,theta,z,'FIXTURE')
    return tr.TrajectoryPlan(name,'FIXTURE',0.,1.,profile_hash=c.profile_hash,
                             segments=(s,),config_hash=c.config_hash)
a=fixture('a',(100.,),(0.,),(0.,))
unsafe=fixture('unsafe',(100.,),(0.,),(80.999,-40.,400.))
origin=fixture('origin',(0.,),(0.,),(0.,))
edge=fixture('edge',(80.,),(.713,),(0.,))
moving=[fixture(str(i),(600.+i*.001,-.01,.001),(i*2*__import__('math').pi/20,-.035),(110.,))
        for i in range(20)]
transfer=tr.generate_transfer('transfer','outer',0,0,0.)
final=tr.generate_final('final','inner',0,0,0.)
assert transfer.accepted and final.accepted
ground=tr.make_ground_plan('ground',3,0.,1000.)
stable=tr.make_stable_plan('stable','outer',18,0.,1000.)
junction=final.plan.segments[0].end_time
cases=[('SAFE',moving[0],moving[1],0.,.2,32),
       ('UNSAFE',a,unsafe,0.,.2,32),('UNKNOWN',origin,edge,0.,1.,0),
       ('transfer_stable',transfer.plan,stable,.137,.371,32),
       ('final_ground_junction',final.plan,ground,junction-.031,junction+.037,32)]
def queries(enabled):
    output=[]
    cls=getattr(sf,'_ExecutionQueryCache',None)
    cache=cls([p for _,a,b,_,_,_ in cases for p in (a,b)]) if enabled and cls else None
    for label,left,right,start,end,depth in cases:
        args={'config':c,'max_depth':depth}
        if cache is not None: args['_query_cache']=cache
        result=sf.validate_separation(left,right,start,end,**args)
        if label in ('SAFE','UNSAFE','UNKNOWN'): assert result.status==label
        output.append({'label':label,'inputs':[left.to_dict(),right.to_dict(),start,end,depth],
                       'result':asdict(result)})
    return output
def execution(mode,enabled):
    bodies=[SimpleNamespace(id=name,status='HOLD_OUTER',p=(i*100.,0.,110.))
            for i,name in enumerate(('z','a','m','b'))]
    bound={a.id:tr.make_stable_plan(a.id,'outer',i,0.,1000.) for i,a in enumerate(bodies)}
    calls=[]; effects=[]; original=wo.validate_separation
    def certificate(x,y,start,end,config,**kwargs):
        calls.append((x.aircraft_id,y.aircraft_id,start,end))
        if mode=='unknown' and (x.aircraft_id,y.aircraft_id)==('z','m'):
            return ValidationResult(False,'UNKNOWN','HARD_SEPARATION',{'interval':[start,end]})
        crossing={('z','a'):.625,('z','m'):.375}.get((x.aircraft_id,y.aircraft_id))
        accepted=not(mode=='failure' and crossing is not None and end>=crossing)
        return ValidationResult(accepted,'SAFE' if accepted else 'UNSAFE','' if accepted else 'HARD_SEPARATION',
                                {'upper_bound_m':100.-len(calls),'lower_bound_m':80.,'witness_time':end})
    holder=SimpleNamespace(aircraft={a.id:a for a in bodies},time=0.,config=c,cache_enabled=enabled,
                           minimum_separation=None,failed=False,_base=lambda a:bound[a.id])
    holder._advance_energy=lambda t:effects.append(('advance_energy',t))
    holder._event=lambda *args,**kwargs:effects.append(('event',args,kwargs))
    holder._fail=lambda *args:effects.append(('fail',args))
    error=None; wo.validate_separation=certificate
    try: wo.RotatingSlotWorld._check_execution(holder,1.)
    except RuntimeError as exc: error=str(exc)
    finally: wo.validate_separation=original
    return {'queries':calls,'minimum':holder.minimum_separation,'effects':effects,'error':error}
live=wo.RotatingSlotWorld(); live.seed_hold('transfer','outer',0)
live.seed_service('service',pad_id=5,remaining=1.,soc=.48)
assert live.submit((transfer.plan,)).accepted
live.advance(.4)
archive=live.to_dict(); del archive['cpu_seconds']
query_result=exact({'queries_off':queries(False),'queries_on':queries(True),
    'execution':{str(enabled):{mode:execution(mode,enabled) for mode in ('safe','unknown','failure')}
                 for enabled in (False,True)},'actual_short_execution':archive})
assert query_result['queries_off']==query_result['queries_on']
'''


@pytest.fixture(scope='module')
def comparison(tmp_path_factory):
    output=tmp_path_factory.mktemp('execution_reuse')
    reference=run_reference(REFERENCE,code=QUERY,timeout=180)
    assert reference['reference_commit']==R0_COMMIT
    assert reference['source_sha256']==R0_SOURCE_SHA256
    code='import sys,json\nsys.dont_write_bytecode=True\n'+f'sys.path.insert(0,{str(ROOT)!r})\n'+QUERY+'\nprint(json.dumps(query_result,allow_nan=False))\n'
    result=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=ROOT,
                          capture_output=True,text=True,timeout=180)
    (output/'opt_stdout.log').write_text(result.stdout,encoding='utf8')
    (output/'opt_stderr.log').write_text(result.stderr,encoding='utf8')
    assert result.returncode==0,result.stderr
    optimized=json.loads(result.stdout)
    artifacts={}
    for name,data in (('r0_results',reference['query_result']),('opt_results',optimized)):
        path=output/(name+'.json')
        path.write_text(json.dumps(data,sort_keys=True,separators=(',',':'))+'\n',encoding='utf8')
        artifacts[name]={'path':str(path),'sha256':sha256(path.read_bytes()).hexdigest()}
    provenance={'r0_commit':reference['reference_commit'],'r0_source_sha256':reference['source_sha256'],
        'r0_pid':reference['pid'],'r0_imported_module_paths':reference['imported_module_paths'],
        'r0_actual_source_files_sha256':reference['actual_source_files_sha256'],
        'test_source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'optimized_source_files_sha256':{name:sha256((ROOT/name).read_bytes()).hexdigest()
            for name in reference['actual_source_files_sha256'] if name.endswith(('/safety.py','/world.py','/trajectory.py'))},
        'artifacts':artifacts,'semantic_exact_match':optimized==reference['query_result'],
        'performance_status':'NOT_QUALIFIED','human_status':'HUMAN_PENDING'}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf8')
    return reference['query_result'],optimized


@pytest.mark.parametrize('key',['queries_off','queries_on','execution','actual_short_execution'])
def test_independent_r0_exact_results(comparison,key):
    reference,optimized=comparison
    assert reference[key]==optimized[key]


def moving_plans():
    c=load_config()
    return [trajectory.TrajectoryPlan(str(i),'FIXTURE',0.,1.,profile_hash=c.profile_hash,
        segments=(trajectory.Segment(0.,1.,(600.+i*.001,-.01,.001),(i*2*math.pi/20,-.035),(110.,),'FIXTURE'),),
        config_hash=c.config_hash) for i in range(20)]


def accounted_owned_bytes(cache):
    """Independent recursive measurement; exclude authoritative plan objects."""
    seen=set()
    def size(value):
        if type(value) is trajectory.TrajectoryPlan or id(value) in seen: return 0
        seen.add(id(value)); total=sys.getsizeof(value)
        if isinstance(value,dict):
            total+=sum(size(k)+size(v) for k,v in value.items())
        elif isinstance(value,(tuple,list)):
            total+=sum(size(v) for v in value)
        elif hasattr(value,'__dict__'):
            total+=size(value.__dict__)
        return total
    return size(cache)


def test_twenty_moving_plans_reduce_original_queries_and_repeat_cold(monkeypatch,tmp_path):
    plans=moving_plans(); counts={'state':0,'bounds':0}
    state=trajectory.TrajectoryPlan.state; bounds=trajectory.TrajectoryPlan.bounds
    def counted_state(plan,t):
        counts['state']+=1; return state(plan,t)
    def counted_bounds(plan,a,b,k=0):
        counts['bounds']+=1; return bounds(plan,a,b,k)
    monkeypatch.setattr(trajectory.TrajectoryPlan,'state',counted_state)
    monkeypatch.setattr(trajectory.TrajectoryPlan,'bounds',counted_bounds)
    def run(cache):
        return [safety.validate_separation(a,b,0.,.2,_query_cache=cache) for a,b in combinations(plans,2)]
    uncached=run(None)
    observations={'uncached_original_method_calls':dict(counts),'pairs':len(uncached),'cold_cached_rounds':[],
                  'memory_scope':'recursive cache-owned bytes only; authoritative plans excluded; no RSS qualification'}
    evidence_path=tmp_path/'method_counts_and_owned_bytes.json'
    evidence_path.write_text(json.dumps(observations,indent=2)+'\n',encoding='utf8')
    assert all(r.accepted for r in uncached) and len(uncached)==190
    assert counts=={'state':380,'bounds':760}
    for _ in range(2):
        counts.update(state=0,bounds=0)
        cache=safety._ExecutionQueryCache(plans)
        cached=run(cache)
        observations['cold_cached_rounds'].append({'original_method_calls':dict(counts),
            'cache_counts':dict(cache.counts),'entries':cache.entries,'estimated_bytes':cache.estimated_bytes,
            'independently_accounted_owned_bytes':accounted_owned_bytes(cache),'results_exact_match':cached==uncached})
        evidence_path.write_text(json.dumps(observations,indent=2)+'\n',encoding='utf8')
        assert cached==uncached
        assert counts=={'state':20,'bounds':40}
        assert cache.counts['state_hits']==360 and cache.counts['bounds_hits']==720
        assert cache.entries==60 and cache.estimated_bytes<=8*1024*1024
        assert accounted_owned_bytes(cache)<=cache.estimated_bytes


def test_entry_and_estimated_byte_limits_recompute(tmp_path):
    plan=moving_plans()[0]; cache=safety._ExecutionQueryCache([plan])
    for i in range(4100):
        t=float(i)/10000
        assert cache.state(plan,t)==plan.state(t)
        assert cache.entries<=4096 and cache.estimated_bytes<=8*1024*1024
    assert cache.counts['evictions']>0
    assert accounted_owned_bytes(cache)<=cache.estimated_bytes
    misses=cache.counts['state_misses']; cache.state(plan,0.)
    assert cache.counts['state_misses']==misses+1
    probe=safety._ExecutionQueryCache([plan]); probe.state(plan,0.)
    small=safety._ExecutionQueryCache([plan],max_bytes=max(8192,probe.estimated_bytes+64))
    for i in range(10):
        t=float(i)/10
        assert small.state(plan,t)==plan.state(t)
        assert small.estimated_bytes<=small.max_bytes
    assert small.counts['evictions']>0 and small.entries<10
    assert accounted_owned_bytes(small)<=small.estimated_bytes
    observations={name:{'entries':q.entries,'counts':dict(q.counts),'estimated_bytes':q.estimated_bytes,
        'independently_accounted_owned_bytes':accounted_owned_bytes(q),'entry_limit':q.max_entries,
        'estimated_byte_limit':q.max_bytes} for name,q in (('default_cap_churn',cache),('small_byte_cap_churn',small))}
    (tmp_path/'entry_and_byte_churn.json').write_text(json.dumps(observations,indent=2)+'\n',encoding='utf8')


def test_exact_times_identity_version_and_strong_references():
    plan=moving_plans()[0]; twin=replace(plan); version=replace(plan,source_version=1,trajectory_hash='')
    assert twin.trajectory_hash==plan.trajectory_hash and twin is not plan
    cache=safety._ExecutionQueryCache([plan,twin,version])
    for p in (plan,twin,version): cache.state(p,.1)
    assert cache.counts['state_misses']==3
    for t in (0.,-0.,math.nextafter(.1,math.inf)): cache.state(plan,t)
    assert cache.counts['state_misses']==6
    cache.bounds(plan,0.,.2,0); cache.bounds(plan,-0.,.2,0); cache.bounds(plan,0.,.2,1)
    assert cache.counts['bounds_misses']==3
    import gc
    def held():
        p=moving_plans()[0]
        return weakref.ref(p),safety._ExecutionQueryCache([p])
    reference,holder=held(); gc.collect(); assert reference() is not None
    del holder; gc.collect(); assert reference() is None


def test_custom_plans_bypass_and_default_audit_path_has_no_cache(monkeypatch):
    class Custom:
        def __init__(self): self.calls=0
        def state(self,t):
            self.calls+=1; return KinematicState((float(self.calls),0.,0.))
    custom=Custom(); cache=safety._ExecutionQueryCache([custom])
    assert cache.state(custom,.1)!=cache.state(custom,.1)
    assert cache.entries==0 and cache.counts['bypasses']==2
    def forbidden(*args,**kwargs): raise AssertionError('default validator constructed a cache')
    monkeypatch.setattr(safety,'_ExecutionQueryCache',forbidden)
    plans=moving_plans()
    assert safety.validate_separation(plans[0],plans[1],0.,.2).accepted


def test_world_cache_is_local_and_respects_cache_off(monkeypatch):
    plans=moving_plans(); objects=[]; original=world._ExecutionQueryCache
    def new_cache(values):
        cache=original(values); objects.append(cache); return cache
    monkeypatch.setattr(world,'_ExecutionQueryCache',new_cache)
    bodies=[SimpleNamespace(id=p.aircraft_id,status='INBOUND',p=p.state(0.).p) for p in plans]
    by_id={p.aircraft_id:p for p in plans}
    holder=SimpleNamespace(aircraft={a.id:a for a in bodies},time=0.,config=load_config(),
        minimum_separation=None,cache_enabled=True,_base=lambda a:by_id[a.id])
    before=set(holder.__dict__)
    world.RotatingSlotWorld._check_execution(holder,.2)
    first_minimum=holder.minimum_separation
    holder.minimum_separation=None
    world.RotatingSlotWorld._check_execution(holder,.2)
    assert len(objects)==2 and objects[0] is not objects[1]
    assert objects[0].counts==objects[1].counts
    assert holder.minimum_separation==first_minimum and set(holder.__dict__)==before
    holder.cache_enabled=False; holder.minimum_separation=None
    world.RotatingSlotWorld._check_execution(holder,.2)
    assert len(objects)==2 and holder.minimum_separation==first_minimum
