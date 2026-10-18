"""Scalar root reuse: exact R0, bounded storage and cold observation.

Run after applying the accompanying patch. These are function-level checks,
not common-window performance qualification.
"""
import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from terminal_marl.core.rotating_slots import energy as en
from terminal_marl.core.rotating_slots.performance import provenance

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'artifacts/rotating_slots/performance_reference_r0'

QUERY = r'''
from dataclasses import asdict
import math
from terminal_marl.core.rotating_slots import energy as en
from terminal_marl.core.rotating_slots.geometry import slot_state
from terminal_marl.core.rotating_slots.config import load_config, Config, Section
c=load_config()
def exact(x):
    if isinstance(x,float): return {'float.hex':x.hex()}
    if isinstance(x,dict): return {str(k):exact(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [exact(v) for v in x]
    return x
states=[]
for ring,count in (('outer',37),('middle',28),('inner',18)):
    for slot in range(count):
        for t in (0.,300.137,720.94458):
            s=slot_state(ring,slot,t,c)
            assert math.hypot(*s.a[:2])>0.
            states.append((ring,slot,t,s))
assert len(states)==249
roots=[(float(i)/7.,float(i%9-4)/5.,float(i+1)*2.137,80) for i in range(31)]
roots += [(0.,0.,20.,80),(-0.,0.,20.,80),(0.,-0.,20.,80),(-0.,-0.,20.,80),
          (2.,-2.,4.,80),(2.,-2.,4.-1e-10,80),(2.,-2.,1.,80),
          (1.,1.,30.,79),(1.,1.,30.,80),(1.,1.,30.,81)]
def changed(**changes):
    data=c.to_dict()
    for dotted,value in changes.items():
        section,key=dotted.split('__'); data[section][key]=value
    # Deliberately forged unchanged hashes: actual arguments remain authoritative.
    return Config(Section(data),c.raw_hash,c.config_hash,c.profile_hash,c.path)
configs=[c,changed(vehicle__fixed_flight_mass_kg=621.),
         changed(energy__effective_horizontal_drag_area_m2=2.12),
         changed(energy__motor_drive_efficiency=.89),
         changed(energy__induced_root_bisections=79)]
power_inputs=[((0.,0.,0.),(0.,0.,-c.coordinates.gravity_mps2)),
              ((math.nan,0.,0.),(0.,0.,0.)),((0.,0.,-20.),(0.,0.,0.)),
              ((-0.,0.,0.),(0.,-0.,0.)),((0.,0.,0.),(0.,0.,30.))]
def run():
    powers=[]; bounds=[]
    for ring,slot,t,s in states:
        powers.append([ring,slot,t,asdict(en.evaluate_power(s.v,s.a,c))])
        v=tuple((float(x)-1e-8,float(x)+1e-8) for x in s.v)
        a=tuple((float(x)-1e-8,float(x)+1e-8) for x in s.a)
        bounds.append([ring,slot,t,en.power_bounds(v,a,c)])
    return exact({'powers':powers,'bounds':bounds,
        'roots':[[args,en._root(*args)] for args in roots],
        'domain':[asdict(en.evaluate_power(v,a,c)) for v,a in power_inputs],
        'forged_hash_configs':[[asdict(en.evaluate_power(states[i][3].v,states[i][3].a,config))
                                for i in (0,50,200)] for config in configs]})
if hasattr(en,'root_cache_clear'):
    assert en.root_cache_stats()['entries']==0  # Fresh child, before any reset.
    en.root_cache_clear(); en.root_cache_set_enabled(False)
off=run()
if hasattr(en,'root_cache_clear'):
    off_stats=en.root_cache_stats()
    assert off_stats['hits']==off_stats['entries']==0
    en.root_cache_clear(); en.root_cache_set_enabled(True)
first=run()
first_stats=en.root_cache_stats() if hasattr(en,'root_cache_stats') else None
repeat=run()
repeat_stats=en.root_cache_stats() if hasattr(en,'root_cache_stats') else None
assert off==first==repeat
query_result={'semantic':first,'counts':{'slots':83,'states':len(states),'roots':len(roots)},
              'stats':{'first':first_stats,'repeat':repeat_stats}}
'''


@pytest.fixture(scope='module')
def isolated(tmp_path_factory):
    reference=provenance.run_reference(REFERENCE,code=QUERY,timeout=180)
    child=('import sys\nsys.dont_write_bytecode=True\nimport json\n'
           f'sys.path.insert(0,{str(ROOT)!r})\n'+QUERY+
           '\nprint(json.dumps(query_result,allow_nan=False))\n')
    result=subprocess.run([sys.executable,'-I','-B','-c',child],cwd=ROOT,
                          capture_output=True,text=True,timeout=180)
    output=tmp_path_factory.mktemp('root_reuse')
    for name,value in [('opt_stdout.log',result.stdout),('opt_stderr.log',result.stderr),
                       ('opt_numericexitcode.txt',str(result.returncode))]:
        (output/name).write_text(value,encoding='utf8')
    assert result.returncode==0,result.stderr
    optimized=json.loads(result.stdout)
    evidence={'reference':reference,'optimized':optimized,
              'query_sha256':sha256(QUERY.encode()).hexdigest(),
              'scope':'function-level exactness and actual root computes only; NOT_QUALIFIED'}
    (output/'evidence.json').write_text(json.dumps(evidence,allow_nan=False),encoding='utf8')
    return reference,optimized


def test_isolated_r0_exact_roots_power_and_bounds(isolated):
    reference,optimized=isolated
    assert reference['reference_commit']==provenance.R0_COMMIT
    assert reference['source_sha256']==provenance.R0_SOURCE_SHA256
    assert reference['query_result']['semantic']==optimized['semantic']
    assert optimized['counts']=={'slots':83,'states':249,'roots':41}
    first,repeat=optimized['stats']['first'],optimized['stats']['repeat']
    assert first['logical_requests']==first['actual_computes']+first['hits']
    assert repeat['actual_computes']==first['actual_computes']
    assert repeat['hits']>first['hits']


def test_original_body_defaults_and_frozen_semantic_fingerprint():
    def scalar(path):
        tree=ast.parse(path.read_text(encoding='utf8'))
        function=deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_root'))
        function.decorator_list=[]
        return ast.dump(function,include_attributes=False)
    assert scalar(ROOT/'terminal_marl/core/rotating_slots/energy.py')==scalar(
        REFERENCE/'terminal_marl/core/rotating_slots/energy.py')
    reference=provenance.semantic_fingerprint(REFERENCE)
    current=provenance.semantic_fingerprint(ROOT)
    provenance.require_same_semantics(reference,current)
    assert current['numerical_policy_hash']==reference['numerical_policy_hash']


@pytest.fixture
def empty_cache():
    enabled=en.root_cache_stats()['enabled']
    en.root_cache_clear(); en.root_cache_set_enabled(True)
    yield
    en.root_cache_clear(); en.root_cache_set_enabled(enabled)


def owned_bytes(value,seen=None):
    """Independent recursive accounting; deduplicate shared immutable objects."""
    seen=set() if seen is None else seen
    if id(value) in seen: return 0
    seen.add(id(value)); total=sys.getsizeof(value)
    if isinstance(value,dict):
        return total+sum(owned_bytes(k,seen)+owned_bytes(v,seen) for k,v in value.items())
    if isinstance(value,(tuple,list)):
        return total+sum(owned_bytes(x,seen) for x in value)
    return total


def test_caps_churn_fifo_and_owned_bytes(empty_cache,monkeypatch):
    assert en.root_cache_stats()['max_entries']==32768
    assert en.root_cache_stats()['max_bytes']==8*1024*1024
    # Smaller limits exercise the same insertion/eviction code with bounded cost.
    monkeypatch.setattr(en,'_ROOT_CACHE_MAX_ENTRIES',17)
    monkeypatch.setattr(en,'_ROOT_CACHE_MAX_BYTES',5000)
    for i in range(500):
        args=(float(i)/7.,.1,float(i+1),3)
        assert en._root(*args)==en._root_uncached(*args)
        stats=en.root_cache_stats()
        assert stats['entries']<=17
        assert owned_bytes(en._ROOT_CACHE)<=stats['estimated_bytes']<=5000
    assert stats['evictions']>0
    before=stats['actual_computes']
    en._root(0.,.1,1.,3)
    assert en.root_cache_stats()['actual_computes']==before+1
    monkeypatch.setattr(en,'_ROOT_CACHE_MAX_BYTES',100)
    en.root_cache_clear()
    for _ in range(2): en._root(1.,.1,20.,3)
    assert en.root_cache_stats()['entries']==0
    assert en.root_cache_stats()['actual_computes']==2


def test_default_caps_under_churn(empty_cache):
    for i in range(40000):
        en._root(float(i)/7.,.1,float(i+1),1)
        if i%4096==0:
            stats=en.root_cache_stats()
            assert stats['entries']<=32768
            assert owned_bytes(en._ROOT_CACHE)<=stats['estimated_bytes']<=8*1024*1024
    stats=en.root_cache_stats()
    assert stats['evictions']>0 and stats['actual_computes']==40000
    assert owned_bytes(en._ROOT_CACHE)<=stats['estimated_bytes']<=stats['max_bytes']


def test_exact_key_invalidation_and_cached_none(empty_cache):
    args=[(0.,0.,20.,80),(-0.,0.,20.,80),(0.,-0.,20.,80),
          (1.,0.,20.,80),(0.,1.,20.,80),(0.,0.,21.,80),(0.,0.,20.,79),
          (2.,-2.,1.,80)]
    expected=[en._root_uncached(*x) for x in args]
    assert expected[-1] is None
    for row,want in zip(args,expected): assert en._root(*row)==want
    assert en.root_cache_stats()['entries']==len(args)
    before=en.root_cache_stats()['actual_computes']
    for row,want in zip(args,expected): assert en._root(*row)==want
    assert en.root_cache_stats()['actual_computes']==before
    assert en.root_cache_stats()['hits']==len(args)
    assert all(type(key) is tuple and (value is None or type(value) is float)
               for key,value in en._ROOT_CACHE.items())
    snapshot=en.root_cache_stats(); snapshot['entries']=-1
    assert en.root_cache_stats()['entries']==len(args)


def test_disable_preserves_exact_results_and_reset_counters(empty_cache):
    import inspect
    assert inspect.signature(en._root)==inspect.signature(en._root_uncached)
    expected=en._root_uncached(vp=1.,vn=0.,rhs=20.,bisections=80)
    assert en._root(rhs=20.,vp=1.,vn=0.)==expected
    en.root_cache_set_enabled(False)
    assert en._root(1.,0.,20.)==expected
    assert en.root_cache_stats()['entries']==1 and en.root_cache_stats()['hits']==0
    en.root_cache_set_enabled(True)
    assert en._root(1.,0.,20.)==expected
    assert en.root_cache_stats()['hits']==1
    en.root_cache_clear()
    stats=en.root_cache_stats()
    assert all(stats[k]==0 for k in ('logical_requests','actual_computes','hits','evictions','entries'))


def test_observed_original_function_calls_reduced_only_on_hits(empty_cache):
    observed=[]
    original_code=en._root_uncached.__code__
    def record(frame,event,arg):
        if event=='call' and frame.f_code is original_code: observed.append(1)
    previous=sys.getprofile()
    try:
        sys.setprofile(record)
        en.root_cache_set_enabled(False)
        off=[en._root(1.,0.,20.) for _ in range(4)]
        off_calls=len(observed)
        en.root_cache_clear(); observed.clear(); en.root_cache_set_enabled(True)
        on=[en._root(1.,0.,20.) for _ in range(4)]
        on_calls=len(observed)
    finally:
        sys.setprofile(previous)
    assert off==on and off_calls==4 and on_calls==1
    assert en.root_cache_stats()['actual_computes']==on_calls
    assert en.root_cache_stats()['logical_requests']==4 and en.root_cache_stats()['hits']==3


def test_unknown_numbers_fallback_and_exceptions(empty_cache):
    from decimal import Decimal
    import numpy as np
    fixtures=[(1,0.,20.,80),(True,0.,20.,80),(np.float64(1),0.,20.,80),
              (Decimal('1'),0.,20.,80),(1.,0.,20.,False),(1.,0.,20.,0),
              (1.,0.,20.,np.int64(4)),(1.,0.,-1.,80),(float('nan'),0.,20.,80),
              (float('inf'),0.,20.,1),(1.,0.,20.,1.)]
    def outcome(function,args):
        try:
            value=function(*args)
            return ('result',value.hex() if hasattr(value,'hex') else value)
        except Exception as exc: return ('exception',type(exc).__name__,str(exc))
    for args in fixtures:
        assert outcome(en._root,args)==outcome(en._root_uncached,args)
    assert en.root_cache_stats()['entries']==0
    with pytest.raises(TypeError): en.root_cache_set_enabled(1)
def test_pre_root_power_failures_do_not_populate_cache(empty_cache):
    from terminal_marl.core.rotating_slots.config import load_config
    c=load_config()
    for v,a in [((0.,0.,0.),(0.,0.,-c.coordinates.gravity_mps2)),
                ((float('nan'),0.,0.),(0.,0.,0.)),((0.,0.,-20.),(0.,0.,0.))]:
        assert en.evaluate_power(v,a,c).status=='ENERGY_MODEL_DOMAIN'
    stats=en.root_cache_stats()
    assert stats['logical_requests']==stats['actual_computes']==stats['entries']==0


def test_cold_observer_does_not_clear_and_supports_missing_r0_cache(empty_cache,monkeypatch):
    from terminal_marl.core.rotating_slots.performance.windows import _cold_cache_snapshot
    FixtureWorld=type('FixtureWorld',(),{'__module__':'terminal_marl.core.rotating_slots.world'})
    world=FixtureWorld()
    world._q_cache={}; world._energy_cache={}; world._reference_templates={}
    initial=_cold_cache_snapshot(world)['module_numerical_caches']['energy._ROOT_CACHE']
    assert initial['present'] and initial['entries']==0
    en._root(1.,0.,20.)
    warm=_cold_cache_snapshot(world)
    assert warm['module_numerical_caches']['energy._ROOT_CACHE']['entries']==1
    assert not warm['all_named_acceleration_caches_empty']
    assert not warm['module_caches_cleared'] and en.root_cache_stats()['entries']==1
    monkeypatch.delattr(en,'root_cache_stats')
    absent=_cold_cache_snapshot(world)['module_numerical_caches']['energy._ROOT_CACHE']
    assert absent=={'present':False,'entries':0}
    monkeypatch.delattr(en,'_segment_table')
    with pytest.raises(AttributeError): _cold_cache_snapshot(world)
