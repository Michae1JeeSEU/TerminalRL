"""Exact Cartesian component bounds against the isolated fixed R0 export."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from terminal_marl.core.rotating_slots import trajectory as tr
from terminal_marl.core.rotating_slots.performance.provenance import R0_SOURCE_SHA256, run_reference

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/rotating_slots/performance_reference_r0"


QUERY = r'''
from dataclasses import asdict
from hashlib import sha256
import inspect, math, random
from terminal_marl.core.rotating_slots import trajectory as tr, safety as sf
from terminal_marl.core.rotating_slots.config import load_config
def exact(value):
    if isinstance(value,float): return {'float.hex':value.hex()}
    if isinstance(value,dict): return {key:exact(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)): return [exact(item) for item in value]
    return value
config=load_config()
plans=[]
for ring,n in (('outer',37),('middle',28),('inner',18)):
    for slot in range(n): plans.append(tr.make_stable_plan('stable-'+ring+str(slot),ring,slot,.137,17.437))
for pad in range(6): plans.append(tr.make_ground_plan('ground-'+str(pad),pad,.137,17.437))
transfer=tr.generate_transfer('transfer','outer',0,0,0.)
final=tr.generate_final('final','inner',0,0,0.)
assert transfer.accepted and final.accepted
plans.extend((transfer.plan,final.plan))
segments=[s for p in plans for s in p.segments]
for pad in range(6): segments.extend(tr.descent_segments(pad,17.371,config))
segments.extend((
    tr.Segment(7.3,13.7,(600.,0.,-3.,-2.,1.),(2*math.pi+1e-12,-.4795,.02),(110.,0.,-2.,-.4)),
    tr.Segment(.137,.019,(300.,.01,-.003),(-math.pi+1e-13,2*math.pi-.0001),(50.,-.1)),
    tr.Segment(3.14159,3.7e-6,(1.,.01,-.001),(.713,-.002,.0001),(2.,-.01,.001)),
    tr.Segment(.317,420.,(300.,-200.),(.125,-.125),tr.final_height(50.,420.),'FINAL_HEIGHT_FIXTURE'),
    tr.Segment(-.137,1.23,(0.,),(0.,),(0.,),'ZERO_FIXTURE')))
rng=random.Random(20261018)
for i in range(32):
    duration=10**rng.uniform(-3.,2.)
    start=rng.uniform(-10.,100.)
    r=(rng.uniform(0.,600.),)+tuple(rng.uniform(-.03,.03) for _ in range(rng.randrange(1,7)))
    theta=(rng.uniform(-4*math.pi,4*math.pi),)+tuple(rng.uniform(-.3,.3) for _ in range(rng.randrange(1,7)))
    z=(rng.uniform(0.,110.),)+tuple(rng.uniform(-.03,.03) for _ in range(rng.randrange(1,9)))
    segments.append(tr.Segment(start,duration,r,theta,z,'FINITE_POLYNOMIAL_FIXTURE'))
bounds=[]; states=[]
for index,s in enumerate(segments):
    a=s.start_time; b=s.end_time
    intervals=((a,a),(b,b),(a,b),(a+.137*s.duration,a+.419*s.duration),
               (a+.517*s.duration,a+.517*s.duration),(a,math.nextafter(a,math.inf)),
               (math.nextafter(b,-math.inf),b))
    for left,right in intervals:
        bounds.append([index,left,right,[s.bounds(left,right,k) for k in range(4)]])
    for t in (a,b,a+.317*s.duration,a-.137,b+.137):
        states.append([index,t,asdict(s.state(t))])
def fixture(aircraft_id,z,r=(100.,),theta=(0.,-.5)):
    s=tr.Segment(0.,1.,r,theta,z,'FIXTURE')
    return tr.TrajectoryPlan(aircraft_id,'FIXTURE',0.,1.,profile_hash=config.profile_hash,
                             config_hash=config.config_hash,segments=(s,))
separation=[]
for d in (80.,math.nextafter(80.,-math.inf),math.nextafter(80.,math.inf),79.999,80.5,78.):
    result=sf.validate_separation(fixture('a',(0.,)),fixture('b',(d,)),0.,1.)
    separation.append([d,asdict(result)])
separation.append(['rounded-static-unknown',asdict(sf.validate_separation(
    fixture('a',(0.,),r=(0.,),theta=(0.,)),
    fixture('b',(0.,),r=(80.,),theta=(.713,)),0.,1.,max_depth=0))])
separation.append(['between-samples',asdict(sf.validate_separation(
    fixture('a',(0.,)),fixture('b',(80.999,-40.,400.)),0.,.2))])
statuses={row[1]['status'] for row in separation}
assert statuses=={'SAFE','UNSAFE','UNKNOWN'},statuses
source_functions={
    '_cartesian':tr._cartesian,'Segment.polar_bounds':tr.Segment.polar_bounds,
    'Segment.state':tr.Segment.state,'Segment.to_dict':tr.Segment.to_dict,
    'TrajectoryPlan._payload':tr.TrajectoryPlan._payload,
    'TrajectoryPlan.state':tr.TrajectoryPlan.state,'TrajectoryPlan.to_dict':tr.TrajectoryPlan.to_dict}
source_sha={name:sha256(inspect.getsource(function).encode()).hexdigest()
            for name,function in source_functions.items()}
query_result=exact({'seed':20261018,'segment_inputs':[s.to_dict() for s in segments],
    'bounds':bounds,'states':states,'plans':[p.to_dict() for p in plans],
    'candidates':[asdict(transfer),asdict(final)],'separation':separation,
    'unchanged_function_source_sha256':source_sha})
'''


@pytest.fixture(scope="module")
def comparison(tmp_path_factory):
    output = tmp_path_factory.mktemp("cartesian_component_equivalence")
    reference = run_reference(REFERENCE, code=QUERY, timeout=180)
    assert reference["source_sha256"] == R0_SOURCE_SHA256
    assert len(reference["actual_source_files_sha256"]) == 24
    child = ("import sys,json\nsys.dont_write_bytecode=True\n"
             f"sys.path.insert(0,{str(ROOT)!r})\n" + QUERY +
             "\nprint(json.dumps(query_result,allow_nan=False))\n")
    command = [sys.executable, "-I", "-B", "-c", child]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
    (output / "opt.stdout.json").write_text(process.stdout, encoding="utf8")
    (output / "opt.stderr.txt").write_text(process.stderr, encoding="utf8")
    assert process.returncode == 0, (process.stdout, process.stderr)
    optimized = json.loads(process.stdout)
    def write(name, data):
        path = output / name
        path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf8")
        return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
    artifacts = {"reference": write("r0_results.json", reference["query_result"]),
                 "optimized": write("opt_results.json", optimized)}
    write("comparison_provenance.json", {
        "r0_commit": reference["reference_commit"], "r0_source_sha256": reference["source_sha256"],
        "r0_actual_source_files_sha256": reference["actual_source_files_sha256"],
        "r0_imported_module_paths": reference["imported_module_paths"],
        "optimized_trajectory_source_sha256": sha256((ROOT / "terminal_marl/core/rotating_slots/trajectory.py").read_bytes()).hexdigest(),
        "test_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "optimized_child_command": command, "optimized_child_exit_code": process.returncode,
        "seed": 20261018, "finite_random_segment_count": 32,
        "segment_count": len(optimized["segment_inputs"]), "bound_query_count": len(optimized["bounds"]),
        "state_query_count": len(optimized["states"]), "artifacts": artifacts,
        "exact_match": optimized == reference["query_result"],
        "performance_status": "NOT_QUALIFIED", "human_status": "HUMAN_PENDING"})
    return reference["query_result"], optimized


@pytest.mark.parametrize("key", ["segment_inputs", "bounds", "states", "plans", "candidates", "separation",
                                  "unchanged_function_source_sha256"])
def test_isolated_fixed_r0_float_hex_exact(comparison, key):
    reference, optimized = comparison
    assert optimized[key] == reference[key]


def frozen(polar):
    return tuple(tuple((value.lo.hex(), value.hi.hex()) for value in axis) for axis in polar)


def vector_hex(values):
    return tuple((value.lo.hex(), value.hi.hex()) for value in values)


def test_single_component_exact_same_jet_and_read_only():
    segment = tr.Segment(7.3,13.7,(600.,0.,-3.,-2.,1.),(.7,-.3,.02),(110.,0.,-2.,-.4))
    polar = segment.polar_bounds(8.317,12.271)
    before = frozen(polar)
    expected = tr._cartesian(*polar)
    for derivative in range(4):
        assert vector_hex(tr._cartesian_component(*polar, derivative)) == vector_hex(expected[derivative])
    assert frozen(polar) == before


@pytest.mark.parametrize("derivative", [0,1,2,3,-1,-2,-3,-4,False,True])
def test_tuple_integer_index_and_bool_semantics(derivative):
    segment = tr.Segment(.137,3.17,(300.,-1.,.02),(.713,-.2),(50.,-.3))
    polar = segment.polar_bounds(.371,1.919)
    expected = tr._cartesian(*polar)[derivative]
    assert vector_hex(tr._cartesian_component(*polar, derivative)) == vector_hex(expected)
    assert segment.bounds(.371,1.919,derivative) == tuple(value.pair() for value in expected)


@pytest.mark.parametrize("derivative,error", [(4,IndexError),(-5,IndexError),(1.,TypeError),
    (None,TypeError),("1",TypeError),(slice(0,2),TypeError)])
def test_invalid_derivative_never_clamps(derivative,error):
    polar = tr.Segment(0.,1.,(300.,),(.713,-.2),(50.,)).polar_bounds(0.,1.)
    with pytest.raises(error):
        tr._cartesian_component(*polar, derivative)


def test_integer_index_protocol_and_scalar_jet():
    class Index:
        def __index__(self): return -2
    polar = ((3.,.1,.2,.3),(.713,-.2,.01,.001),(2.,.1,.01,.001))
    assert tr._cartesian_component(*polar,Index()) == tr._cartesian(*polar)[Index()]
    for derivative in range(4):
        assert tuple(x.hex() for x in tr._cartesian_component(*polar,derivative)) == tuple(x.hex() for x in tr._cartesian(*polar)[derivative])


def test_segment_bounds_only_requests_one_polar_jet(monkeypatch):
    segment = tr.Segment(.137,3.17,(300.,-1.,.02),(.713,-.2),(50.,-.3))
    original_polar = tr.Segment.polar_bounds
    original_component = tr._cartesian_component
    calls = []
    def polar(s,a,b):
        calls.append(("polar",s,a,b))
        return original_polar(s,a,b)
    def component(r,theta,z,derivative):
        calls.append(("component",derivative))
        return original_component(r,theta,z,derivative)
    monkeypatch.setattr(tr.Segment,"polar_bounds",polar)
    monkeypatch.setattr(tr,"_cartesian_component",component)
    segment.bounds(.371,1.919,1)
    assert calls == [("polar",segment,.371,1.919),("component",1)]


def test_interval_work_removed_without_changing_rotation_or_input(monkeypatch,tmp_path):
    segment = tr.Segment(7.3,13.7,(600.,0.,-3.,-2.,1.),(.7,-.3,.02),(110.,0.,-2.,-.4))
    polar = segment.polar_bounds(8.317,12.271)
    before = frozen(polar)
    counts = {}
    methods = ("__init__","__add__","__radd__","__sub__","__rsub__","__neg__",
               "__mul__","__rmul__","__truediv__","sin","cos")
    for name in methods:
        original = getattr(tr.Interval,name)
        def counted(self,*args,_name=name,_original=original,**kwargs):
            counts[_name] = counts.get(_name,0) + 1
            return _original(self,*args,**kwargs)
        monkeypatch.setattr(tr.Interval,name,counted)
    counts.clear()
    expected = tr._cartesian(*polar)
    full = {name:counts.get(name,0) for name in methods}
    results = []
    for derivative in range(4):
        counts.clear()
        actual = tr._cartesian_component(*polar,derivative)
        single = {name:counts.get(name,0) for name in methods}
        assert vector_hex(actual) == vector_hex(expected[derivative])
        assert single["__init__"] < full["__init__"]
        assert single["__mul__"] + single["__rmul__"] < full["__mul__"] + full["__rmul__"]
        assert single["cos"] == full["cos"] == 1
        assert single["sin"] == full["sin"] == 2  # cos retains its original shift+sin implementation.
        results.append({"derivative":derivative,"interval_operation_calls":single})
    assert frozen(polar) == before
    (tmp_path / "interval_operation_counts.json").write_text(json.dumps({
        "scope":"one identical already-built polar jet; no wall/CPU benchmark",
        "original_all_four_derivatives":full,"single_derivative":results,"input_jet_unchanged":True,
        "performance_status":"NOT_QUALIFIED","human_status":"HUMAN_PENDING"},indent=2),encoding="utf8")
