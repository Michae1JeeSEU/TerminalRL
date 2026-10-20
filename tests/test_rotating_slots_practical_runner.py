"""Small runner/serialization checks; never execute the practical traffic scene."""
from functools import lru_cache
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace, ModuleType
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/server/compare_rotating_slots_practical.py"
spec = importlib.util.spec_from_file_location("practical_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def synthetic_audit(tmp_path,monkeypatch,*,samples=3,energy_error=.5,peak_error=.005,missing=False,replay_error=False):
    """Small ledger fixture, with no traffic-window run or imported core."""
    root=tmp_path/'fixture'; root.mkdir()
    capacity=85.*252.*3600
    definitions={name:dict(kind=kind,start_time=0.,end_time=10.,rate=rate)
                 for name,kind,rate in (('s','STABLE',10000.),('t','TRANSFER',100000.),('f','FINAL_NORMAL',200000.))}
    (root/'trajectories.json').write_text(json.dumps(definitions))
    frames=[]
    for t in (0.,.2,.4):
        aircraft=[]
        for name,plan in definitions.items():
            joules=plan['rate']*t+energy_error*t/.4
            row=dict(id=name,status='HOLD_OUTER' if name=='s' else 'FINAL_NORMAL' if name=='f' else 'TRANSFER_O_M',
                     trajectory_hash=name,energy_j=joules,soc=.8-joules/capacity,peak_seconds=peak_error*t/.4)
            if missing and name=='t' and t==.2: row.pop('energy_j')
            aircraft.append(row)
        frames.append(dict(time=t,aircraft=aircraft))
    (root/'states.jsonl').write_text(''.join(json.dumps(frame)+'\n' for frame in frames))
    observed=dict(bundle=0,plan=0,separation=0,tolerances=[],energy=[])
    def validate_bundle(path):
        observed['bundle']+=1
        if replay_error: raise ValueError('fixture replay checksum mismatch')
    def validate_plan(plan,**kwargs):
        observed['plan']+=1
        return SimpleNamespace(accepted=True,status='SAFE',reason='',details={})
    def separation(*args,**kwargs):
        observed['separation']+=1
        return SimpleNamespace(accepted=True,status='SAFE',details={'lower_bound_m':100.})
    def minimum(*args,**kwargs):
        observed['tolerances'].append(kwargs['tolerance_m'])
        return dict(status='CERTIFIED',lower_bound_m=100.,upper_bound_m=100.)
    def integrate(plan,start,end,**kwargs):
        observed['energy'].append((plan.kind,start,end))
        joules=plan.rate*(end-start)
        return SimpleNamespace(valid=True,joules=joules,soc_used=joules/capacity,peak_seconds=0.)
    modules={'replay':SimpleNamespace(ReplayBundle=SimpleNamespace(validate=validate_bundle)),
        'config':SimpleNamespace(load_config=lambda path:SimpleNamespace(energy=SimpleNamespace(equivalent_voltage_V=85.,reference_capacity_Ah=252.))),
        'trajectory':SimpleNamespace(TrajectoryPlan=SimpleNamespace(from_dict=lambda value,**kwargs:SimpleNamespace(**value))),
        'safety':SimpleNamespace(validate_plan=validate_plan,validate_separation=separation,certified_minimum_distance=minimum),
        'energy':SimpleNamespace(integrate_energy=integrate)}
    monkeypatch.setattr(runner,'load_core',lambda source:modules)
    args=SimpleNamespace(source_root=str(ROOT),replay=str(root),start=0.,end=.4,
                         output=str(tmp_path/'audit.json'),energy_samples=samples,minimum_tolerance_m=.1)
    code=runner.audit(args)
    return code,json.loads(Path(args.output).read_text()),observed


def test_sampled_audit_merges_ledgers_covers_kinds_and_separates_timing(tmp_path,monkeypatch):
    code,report,observed=synthetic_audit(tmp_path,monkeypatch)
    assert code==0 and report['status']=='PASS'
    assert report['energy_mode']=='sampled_merged_execution'
    assert report['energy_sample_count']==report['checks']['energy_sample_count']==3
    assert report['checks']['energy_candidate_intervals']==3
    assert {row['kind'] for row in report['energy_samples']}=={'STABLE','TRANSFER','FINAL_NORMAL'}
    assert all(row['start']==0. and row['end']==.4 for row in report['energy_samples'])
    assert all(row['budgets']['peak_seconds']==.01 for row in report['energy_samples'])
    assert report['max_peak_error_s']==pytest.approx(.005)
    assert report['max_energy_error_j']==pytest.approx(.5)
    assert report['audit_wall_seconds']>=0 and report['audit_process_seconds']>=0
    assert observed['bundle']==1 and observed['plan']==3 and observed['separation']==3
    assert observed['tolerances']==[.1]
    assert report['checks']['energy_sample_integrations']==3
    assert report['checks']['energy_complete_plan_integrations']==3


def test_sampled_energy_budget_applies_once_to_merged_execution(tmp_path,monkeypatch):
    code,report,_=synthetic_audit(tmp_path,monkeypatch,energy_error=3.,peak_error=0.)
    assert code==1 and report['status']=='FAIL'
    assert {failure['kind'] for failure in report['failures']}=={'energy_sample_budget'}
    assert sorted(row['budgets']['energy_j'] for row in report['energy_samples'])==[1.,1.,2.]
    assert all(row['signed_errors']['energy_j']==pytest.approx(3.) for row in report['energy_samples'])


@pytest.mark.parametrize('failure',('missing','replay_error'))
def test_sampled_audit_missing_evidence_emits_explicit_failure(tmp_path,monkeypatch,failure):
    code,report,_=synthetic_audit(tmp_path,monkeypatch,**{failure:True})
    assert code==1 and report['status']=='FAIL'
    expected='energy_sample_fields' if failure=='missing' else 'replay_validation'
    assert expected in {entry['kind'] for entry in report['failures']}
    assert report['audit_wall_seconds']>=0 and report['audit_process_seconds']>=0


def test_zero_samples_keeps_original_strict_frame_thresholds(tmp_path,monkeypatch):
    code,report,observed=synthetic_audit(tmp_path,monkeypatch,samples=0)
    assert code==1 and report['energy_mode']=='strict_per_frame'
    assert report['checks']['energy_intervals']==6
    assert report['energy_samples']==[] and report['energy_sample_count']==0
    assert {entry['kind'] for entry in report['failures']}=={'energy'}
    assert len(observed['energy'])==6


def test_counter_calls_actual_cache_and_counts_only_misses():
    calls = runner.CallCounts()
    computed = []
    @lru_cache(None)
    def square(value):
        computed.append(value)
        return value*value
    wrapped = calls.wrap(square, "table", lambda args, kwargs: square.cache_info().misses)
    assert wrapped(3) == wrapped(3) == 9
    assert calls.calls == {"table": 2, "table.misses": 1}
    assert computed == [3]
    assert calls.precompute_seconds >= 0


def fake_counter_modules(monkeypatch,energy):
    modules={'energy':energy}
    for name in ('trajectory','safety','world'):
        modules[name]=ModuleType(runner.CORE+'.'+name)
    modules['world'].RotatingSlotWorld=type('FakeWorld',(),{})
    for name,module in modules.items():
        monkeypatch.setitem(sys.modules,runner.CORE+'.'+name,module)
    return modules


def test_initial_and_policy_cache_misses_use_nested_union_timing(monkeypatch):
    energy=ModuleType(runner.CORE+'.energy'); clock=[0.]
    @lru_cache(None)
    def initial(value):
        clock[0]+=2.
        return value
    @lru_cache(None)
    def table(value):
        energy._segment_initial(value)
        clock[0]+=3.
        return value
    @lru_cache(None)
    def policy(value):
        energy._segment_initial(value)
        energy._segment_table(value)
        clock[0]+=1.
        return value
    energy._segment_initial=initial; energy._segment_table=table; energy._plan_numerical_policy=policy
    modules=fake_counter_modules(monkeypatch,energy)
    monkeypatch.setattr(runner.time,'perf_counter',lambda:clock[0])
    calls=runner.CallCounts(); calls.install(modules)
    try:
        assert energy._plan_numerical_policy(1)==energy._plan_numerical_policy(1)==1
        assert calls.precompute_seconds==6.
        energy._segment_table(2)
        assert calls.precompute_seconds==11.
        assert calls.calls['energy._plan_numerical_policy.misses']==1
        assert calls.calls['energy._segment_initial.misses']==2
        assert calls.calls['energy._segment_table.misses']==2
        assert calls.precompute_depth==0
    finally: calls.remove()


def test_absent_new_helpers_preserve_frozen_baseline_wrappers_and_counts(monkeypatch):
    original=subprocess.check_output(['git','show','fcb11c4:scripts/server/compare_rotating_slots_practical.py'],cwd=ROOT,text=True)
    node=next(node for node in ast.parse(original).body if isinstance(node,ast.ClassDef) and node.name=='CallCounts')
    namespace=vars(runner).copy()
    exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen_CallCounts','exec'),namespace)
    baseline_class=namespace['CallCounts']
    def fixture():
        energy=ModuleType(runner.CORE+'.energy')
        for name in ('evaluate_power','integrate_energy','power_bounds','_power_array','_peak_measure','_segment_table','_peak_table'):
            def value(x): return x+1
            setattr(energy,name,lru_cache(None)(value) if name in ('_segment_table','_peak_table') else value)
        return energy
    outcomes=[]
    for cls in (baseline_class,runner.CallCounts):
        energy=fixture(); modules=fake_counter_modules(monkeypatch,energy)
        calls=cls(); calls.install(modules)
        try:
            wrappers=sorted((owner.__name__,name) for owner,name,_ in calls.replacements)
            for name in ('evaluate_power','integrate_energy','power_bounds','_power_array','_peak_measure','_segment_table','_peak_table'):
                assert getattr(energy,name)(3)==getattr(energy,name)(3)==4
            outcomes.append((wrappers,dict(calls.calls)))
        finally: calls.remove()
    assert outcomes[0]==outcomes[1]
    frozen_energy=subprocess.check_output(['git','show','fcb11c4:terminal_marl/core/rotating_slots/energy.py'],cwd=ROOT,text=True)
    names={node.name for node in ast.parse(frozen_energy).body if isinstance(node,ast.FunctionDef)}
    assert not {'_segment_initial','_plan_numerical_policy'} & names


def test_window_intervals_preserve_old_plan_and_clip_prefix():
    frames = [dict(time=t, aircraft=[dict(id="a", status="LANDED_SERVICE", trajectory_hash=h)])
              for t,h in ((0,"old"),(2,"old"),(3,"new"),(4,"new"))]
    intervals = list(runner.window_intervals(iter(frames), 2., 4.))
    assert [(a,b) for a,b,*_ in intervals] == [(2.,3.),(3.,4.)]
    assert intervals[0][2]["a"]["trajectory_hash"] == "old"
    assert "LANDED_SERVICE" in intervals[0][4]


def test_synthetic_cold_restore_point_four_seconds_and_original_audit(tmp_path):
    # Fresh process also proves selected source imports precede all core imports.
    frozen = ROOT / "artifacts/rotating_slots/practical_perf_20261020/before_source"
    if not frozen.is_dir():
        pytest.skip("optional frozen original comparison fixture is absent")
    code = r'''
import importlib.util,json,sys
from pathlib import Path
helper,source,out=sys.argv[1:]
spec=importlib.util.spec_from_file_location('runner',helper)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
m=r.load_core(source)
core='terminal_marl.core.rotating_slots'
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld,AircraftState
from terminal_marl.core.rotating_slots.trajectory import make_stable_plan
from terminal_marl.core.rotating_slots.replay import BufferedStateJSONLSink
c=m['performance.checkpoint'];cfg=m['config'].load_config()
out=Path(out);out.mkdir();(out/'source').mkdir()
w=RotatingSlotWorld(state_sink=BufferedStateJSONLSink(out/'source/states.jsonl'))
w.auto_emergency=False
for aid,slot in [('a',0),('b',5)]:
 p=make_stable_plan(aid,'outer',slot,end_time=100.,config=cfg);s=p.state(0.)
 w.aircraft[aid]=AircraftState(aid,'HOLD_OUTER',.8,s.p,s.v,s.a,s.jerk,
     ring='outer',slot=slot,fixture=True,trajectory_hash=p.trajectory_hash)
 w.trajectories[p.trajectory_hash]=p.to_dict()
w._record()
identity=c.CheckpointIdentity(c.POLICY_ID,cfg.config_hash,cfg.profile_hash,cfg.raw_hash,'r','c','n','original-test')
payload=c.capture_checkpoint(w,identity=identity,cursor=c.AdvanceCursor(.4,'loop_start','test'),driver=False)
c.write_checkpoint(out/'checkpoint.json',payload)
payload,loaded_identity=r.load_payload(c,out/'checkpoint.json')
assert loaded_identity==identity
restored=c.restore_checkpoint(payload,expected_identity=identity,cache_mode='cold',new_attempt_dir=out/'restored')
assert restored.world.time==0. and restored.driver is False
assert not restored.world._q_cache and not restored.world._energy_cache
if hasattr(restored.world,'configure_practical_runtime'):
 restored.world.configure_practical_runtime(True)
r.cold_caches(m)
counts=r.CallCounts();counts.install(m)
restored.world.advance(.4,restored.driver);restored.world.flush_records()
counts.remove()
assert restored.world.time==.4
assert counts.calls['world._check_execution']>0
m['replay'].ReplayBundle.write(out/'replay',world=restored.world,code_commit='synthetic-preflight',config_path=m['config'].DEFAULT_CONFIG,eager=False)
restored.world.state_sink.close();w.state_sink.close()
print(json.dumps(dict(time=restored.world.time,calls=dict(counts.calls))))
'''
    completed = subprocess.run([sys.executable, "-c", code, str(RUNNER), str(ROOT), str(tmp_path / "micro")],
                               capture_output=True, text=True, cwd=ROOT)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["time"] == .4
    audited = subprocess.run([sys.executable, str(RUNNER), "audit", "--source-root", str(frozen),
        "--replay", str(tmp_path / "micro/replay"), "--start", "0", "--end", ".4",
        "--output", str(tmp_path / "audit.json")], capture_output=True, text=True, cwd=ROOT)
    assert audited.returncode == 0, audited.stdout + audited.stderr
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["status"] == "PASS"
    assert report["checks"]["energy_intervals"] >= 2
    assert report["checks"]["separation_runs"] == 1
    assert report["minimum_distance"]["upper_bound_m"] - report["minimum_distance"]["lower_bound_m"] <= .01
