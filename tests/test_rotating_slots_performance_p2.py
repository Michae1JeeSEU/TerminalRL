"""Bounded P2 function equivalence against the frozen, isolated R0 export."""
from hashlib import sha256
import json
from pathlib import Path

import pytest

from terminal_marl.core.rotating_slots import energy, safety, trajectory, world
from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.performance.provenance import R0_SOURCE_SHA256, run_reference

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/rotating_slots/performance_reference_r0"

# This identical query program runs in two separate interpreters. R0 supplies
# every expected value; no OPT-generated golden values or rounded comparisons.
QUERY = r'''
from dataclasses import asdict, replace
from types import SimpleNamespace
from terminal_marl.core.rotating_slots import trajectory as tr, safety as sf, energy as en, world as wo
from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.models import ValidationResult
def exact(value):
    if isinstance(value,float): return {'float.hex':value.hex()}
    if isinstance(value,dict): return {key:exact(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)): return [exact(item) for item in value]
    return value
c=load_config()
transfer=tr.generate_transfer('transfer','outer',0,0,0.)
final=tr.generate_final('final','inner',0,0,0.)
assert transfer.accepted and final.accepted
plans=[tr.make_stable_plan('stable','outer',3,0.,17.3),
       tr.make_ground_plan('ground',2,0.,17.3),transfer.plan,final.plan]
segments=[s for p in plans for s in p.segments]
intrinsic=[]
for s in segments:
    for left,right in ((0.,1.),(.137,.419),(.517,.517)):
        a=s.start_time+left*s.duration; b=s.start_time+right*s.duration
        intrinsic.append([s.kind,a,b,[s.intrinsic_bounds(a,b,k) for k in (1,2,3)]])
validations=[asdict(sf.validate_plan(p,c,use_cache=False)) for p in plans]
bad_stable=replace(plans[0],segments=(replace(plans[0].segments[0],r=(601.,)),),trajectory_hash='')
bad_ground=replace(plans[1],segments=(replace(plans[1].segments[0],z=(1.,)),),trajectory_hash='')
# A one-ulp coefficient change preserves endpoint tolerances but violates the
# canonical descent payload, independently of the outer trajectory hash.
descent=list(final.plan.segments)
z=list(descent[1].z); z[0]=__import__('math').nextafter(z[0],float('inf'))
descent[1]=replace(descent[1],z=tuple(z))
bad_descent=replace(final.plan,segments=tuple(descent),trajectory_hash='')
tampered=replace(transfer.plan)
object.__setattr__(tampered,'end_time',tampered.end_time+.1)
bad_config=replace(transfer.plan,config_hash='wrong',trajectory_hash='')
invalid=[asdict(sf.validate_plan(p,c,use_cache=False))
         for p in (bad_stable,bad_ground,bad_descent,tampered,bad_config)]
assert [v['reason'] for v in invalid]==['NONCANONICAL_STABLE','NONCANONICAL_GROUND',
    'NONCANONICAL_DESCENT','TRAJECTORY_HASH_MISMATCH','PROFILE_HASH_MISMATCH']
peaks=[]
for s in (transfer.plan.segments[0],final.plan.segments[0],final.plan.segments[1]):
    a=s.start_time+.371*s.duration; b=a+1.5e-7
    pb=en.power_bounds(s.intrinsic_bounds(a,b,1),s.intrinsic_bounds(a,b,2),c)
    assert pb is not None
    st=s.state((a+b)/2); point=en.evaluate_power(st.v,st.a,c).shaft_w
    peaks.append([pb,[en._peak_measure(s,a,b,threshold,c)
                      for threshold in (pb['shaft_w'][0]-1.,pb['shaft_w'][1]+1.,point)]])
def execution(mode):
    bodies=[SimpleNamespace(id=name,status='HOLD_OUTER',p=(i*100.,0.,110.))
            for i,name in enumerate(('z','a','m','b'))]
    calls=[]; effects=[]; original=wo.validate_separation
    def certificate(x,y,a,b,config):
        calls.append((x.id,y.id,a,b))
        if mode=='unknown' and (x.id,y.id)==('z','m'):
            return ValidationResult(False,'UNKNOWN','HARD_SEPARATION',{'interval':[a,b]})
        crossing={('z','a'):.625,('z','m'):.375}.get((x.id,y.id))
        accepted=not(mode=='failure' and crossing is not None and b>=crossing)
        return ValidationResult(accepted,'SAFE' if accepted else 'UNSAFE','' if accepted else 'HARD_SEPARATION',
                                {'upper_bound_m':100.-len(calls),'lower_bound_m':80.,'witness_time':b})
    holder=SimpleNamespace(aircraft={a.id:a for a in bodies},time=0.,config=c,
                           minimum_separation=None,failed=False,_base=lambda a:a)
    holder._advance_energy=lambda t:effects.append(('advance_energy',t))
    holder._event=lambda *args,**kwargs:effects.append(('event',args,kwargs))
    holder._fail=lambda *args:effects.append(('fail',args))
    error=None
    wo.validate_separation=certificate
    try:
        wo.RotatingSlotWorld._check_execution(holder,1.)
    except RuntimeError as exc: error=str(exc)
    finally: wo.validate_separation=original
    return {'queries':calls,'minimum':holder.minimum_separation,'effects':effects,'error':error}
from terminal_marl.core.rotating_slots.driver import DeterministicDriver
live=wo.RotatingSlotWorld()
live.seed_hold('transfer','outer',0)
live.seed_service('service',pad_id=5,remaining=1.,soc=.48)
commit=live.submit((transfer.plan,))
assert commit.accepted and live.active['transfer'] is transfer.plan
driver=DeterministicDriver()
live.advance(.4,driver)
archive=live.to_dict()
del archive['cpu_seconds']  # Explicit wall-clock diagnostics only.
l2={'commit':asdict(commit),'archive':archive,'snapshot':live.snapshot().to_dict(),
    'driver_calls':driver.calls,'decision_times':live.decision_times}
assert live.time==.4 and not live.failed and driver.calls==[0.]
assert len(live.records)==3 and live.minimum_separation is not None
query_result=exact({'candidates':[asdict(transfer),asdict(final),
                               asdict(tr.generate_transfer('bad','inner',0,0,0.))],
                    'intrinsic':intrinsic,'validation':validations,'invalid':invalid,'peak':peaks,
                    'execution':{mode:execution(mode) for mode in ('safe','unknown','failure')},'l2':l2})
'''


@pytest.fixture(scope="module")
def comparison(tmp_path_factory):
    import subprocess
    import sys
    reference = run_reference(REFERENCE, code=QUERY, timeout=180)
    assert reference["reference_commit"] == "1c5afd37c891a0ed4776f64367fc7264db3c1c54"
    assert reference["source_sha256"] == R0_SOURCE_SHA256
    for name, expected in reference["actual_source_files_sha256"].items():
        assert sha256((REFERENCE / name).read_bytes()).hexdigest() == expected
    child = (
        "import sys,json\nsys.dont_write_bytecode=True\n"
        f"sys.path.insert(0,{str(ROOT)!r})\n" + QUERY +
        "\nprint(json.dumps(query_result,allow_nan=False))\n"
    )
    result = subprocess.run([sys.executable,"-I","-B","-c",child],cwd=ROOT,
                            capture_output=True,text=True,timeout=180)
    assert result.returncode == 0, result.stderr
    optimized=json.loads(result.stdout)
    output=tmp_path_factory.mktemp('equivalence')
    def write(name,data):
        path=output/name
        path.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf8')
        return {'path':str(path),'sha256':sha256(path.read_bytes()).hexdigest()}
    artifacts={
        'reference':write('r0_results.json',reference['query_result']),
        'optimized':write('opt_results.json',optimized),
    }
    write('comparison_provenance.json',{
        'r0_commit':reference['reference_commit'],'r0_source_sha256':reference['source_sha256'],
        'r0_actual_source_files_sha256':reference['actual_source_files_sha256'],
        'r0_imported_module_paths':reference['imported_module_paths'],
        'test_source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'optimized_source_files_sha256':{name:sha256((ROOT/name).read_bytes()).hexdigest()
            for name in reference['actual_source_files_sha256']
            if name.endswith(('/world.py','/trajectory.py','/safety.py','/energy.py'))},
        'artifacts':artifacts,'exact_match':optimized==reference['query_result'],
        'excluded_l2_fields':['archive.cpu_seconds'],
        'qualification':'NOT_QUALIFIED','human_acceptance':'HUMAN_PENDING',
    })
    return reference['query_result'], optimized


@pytest.mark.parametrize("key",["candidates","intrinsic","validation","invalid","peak","execution","l2"])
def test_isolated_r0_exact_function_results(comparison,key):
    reference, optimized = comparison
    assert optimized[key] == reference[key]


def test_shared_jet_is_read_only_and_exact():
    segment=trajectory.Segment(7.3,13.7,(600.,0.,-3.,-2.,1.),(.7,-.3,.02),
                               (110.,0.,-2.,-.4))
    polar=segment.polar_bounds(8.317,12.271)
    before=tuple(tuple((x.lo.hex(),x.hi.hex()) for x in axis) for axis in polar)
    for derivative in (1,2,3):
        assert trajectory._intrinsic_bounds_from_polar(polar,derivative) == segment.intrinsic_bounds(8.317,12.271,derivative)
    energy.power_bounds(trajectory._intrinsic_bounds_from_polar(polar,1),
                        trajectory._intrinsic_bounds_from_polar(polar,2),load_config())
    assert before == tuple(tuple((x.lo.hex(),x.hi.hex()) for x in axis) for axis in polar)
    with pytest.raises(ValueError,match="derivatives 1,2,3"):
        trajectory._intrinsic_bounds_from_polar(polar,0)


def test_validation_one_polar_jet_per_recursive_node(monkeypatch):
    plan=trajectory.generate_transfer('a','outer',0,0,0.).plan
    calls=[]; original=trajectory.Segment.polar_bounds
    def polar(segment,a,b):
        calls.append((id(segment),a,b))
        return original(segment,a,b)
    monkeypatch.setattr(trajectory.Segment,"polar_bounds",polar)
    result=safety.validate_plan(plan,use_cache=False)
    assert result.accepted
    assert len(calls) == result.details['intervals'] == result.details['power_intervals']
    assert len(calls) == len(set(calls))


def test_peak_one_polar_jet_per_recursive_node(monkeypatch):
    segment=trajectory.descent_segments(0,0.)[0]
    config=load_config()
    initial=energy.power_bounds(segment.intrinsic_bounds(1.137,1.139,1),
                                segment.intrinsic_bounds(1.137,1.139,2),config)
    threshold=sum(initial['shaft_w'])/2
    calls=[]; power_calls=[]
    original=trajectory.Segment.polar_bounds; original_power=energy.power_bounds
    def polar(s,a,b):
        calls.append((a,b))
        return original(s,a,b)
    def power(v,a,c):
        power_calls.append((v,a))
        return original_power(v,a,c)
    monkeypatch.setattr(trajectory.Segment,"polar_bounds",polar)
    monkeypatch.setattr(energy,"power_bounds",power)
    value=energy._peak_measure(segment,1.137,1.139,threshold,config,depth=24)
    assert 0. <= value <= 1.139-1.137
    assert len(calls) == len(power_calls) > 1
    assert len(calls) == len(set(calls))


@pytest.mark.parametrize("n",[0,1,10,30,60,83])
def test_execution_base_constructions_and_pair_coverage(monkeypatch,n):
    w=world.RotatingSlotWorld()
    for ring,count in (('outer',37),('middle',28),('inner',18)):
        for slot in range(count):
            if len(w.aircraft)<n: w.seed_hold(f'{ring}_{slot}',ring,slot)
    original_base=w._base; constructed=[]; pairs=[]
    def base(a):
        constructed.append(a.id)
        before=(w.snapshot(),dict(w.active),dict(w.soft))
        result=original_base(a)
        assert (w.snapshot(),dict(w.active),dict(w.soft)) == before
        return result
    original_separation=world.validate_separation
    def separation(a,b,start,end,config):
        pairs.append((a.aircraft_id,b.aircraft_id,start,end))
        return original_separation(a,b,start,end,config=config)
    monkeypatch.setattr(w,"_base",base)
    monkeypatch.setattr(world,"validate_separation",separation)
    w._check_execution(.2)
    ids=list(w.aircraft)
    assert constructed == (ids if n>=2 else [])
    assert [(a,b) for a,b,_,_ in pairs] == [(a,b) for i,a in enumerate(ids) for b in ids[i+1:]]
    assert len(pairs) == n*(n-1)//2
    assert all((start,end)==(0.,.2) for _,_,start,end in pairs)
    # No cross-step cache: each following interval reconstructs once per body.
    constructed.clear(); pairs.clear()
    w._check_execution(.3)
    assert constructed == (ids if n>=2 else [])


def test_execution_bisection_queries_counted_separately(monkeypatch):
    from terminal_marl.core.rotating_slots.models import ValidationResult
    w=world.RotatingSlotWorld()
    w.seed_hold('a','outer',0); w.seed_hold('b','outer',1)
    original=w._base; bases=[]; full=[]; bisection=[]; effects=[]
    def base(a):
        bases.append(a.id)
        return original(a)
    def separation(a,b,start,end,config):
        (full if end==1. else bisection).append((start,end))
        return ValidationResult(end<.375,'SAFE' if end<.375 else 'UNSAFE',
                                details={'witness_time':end})
    monkeypatch.setattr(w,"_base",base)
    monkeypatch.setattr(world,"validate_separation",separation)
    monkeypatch.setattr(w,"_advance_energy",lambda t:effects.append(('energy',t)))
    monkeypatch.setattr(w,"_event",lambda *args,**kwargs:effects.append(('event',args,kwargs)))
    monkeypatch.setattr(w,"_fail",lambda *args:effects.append(('fail',args)))
    w._check_execution(1.)
    assert bases == ['a','b'] and len(full)==1 and len(bisection)==17
    assert [e[0] for e in effects] == ['energy','event','fail']
    assert effects[0][1] == .375
