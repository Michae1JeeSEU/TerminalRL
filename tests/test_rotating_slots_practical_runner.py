"""Small runner/serialization checks; never execute the practical traffic scene."""
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/server/compare_rotating_slots_practical.py"
spec = importlib.util.spec_from_file_location("practical_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


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
