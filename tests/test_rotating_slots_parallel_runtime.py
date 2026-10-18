"""Bounded real spawn/reference and physical-output equality checks."""
from copy import deepcopy
import json
import multiprocessing as mp
import math
from pathlib import Path

import pytest

from scripts.server import parallel_rotating_slots_runtime as runtime
from scripts.server import probe_rotating_slots_parallel as probe
from scripts.server import recover_rotating_slots_parallel as launcher
from terminal_marl.core.rotating_slots import diagnostics
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld


@pytest.fixture(scope="module")
def real_pool(tmp_path_factory):
    with runtime.ReferencePool(2, stats_path=tmp_path_factory.mktemp("inner_pool")/"stats.json") as pool:
        yield pool
    assert pool.closed


def test_real_two_spawn_workers_match_exact_reference_payload_and_pad_order(real_pool):
    query = ("inner", 0, 0., 0)
    serial = RotatingSlotWorld()
    expected = probe.reference_payload(serial._reference(*query))
    values = real_pool.references([query, query])
    assert list(values) == [query]
    assert probe.reference_payload(values[query]) == expected
    # A warm repeat remains exact, independently of task completion ordering.
    assert probe.reference_payload(real_pool.references([query])[query]) == expected
    summary = real_pool.summary()
    assert len(summary["worker_pids"]) == 2
    assert all(w["threads"] == {k: "1" for k in runtime.THREADS} for w in summary["workers"])
    assert summary["pad_tasks"] == 12
    assert summary["child_task_cpu_seconds"] > 0
    assert summary["child_measured_components"]["exclusive_cpu_seconds"]


def test_real_world_matches_metrics_states_events_rejections_and_trajectories(real_pool):
    serial = probe.run_world(.2, fixture="seeded_inner")
    parallel = probe.run_world(.2, real_pool, fixture="seeded_inner")
    assert not probe.differences(serial, parallel)
    assert serial == parallel
    assert serial["states"][-1]["time"] == .2
    assert serial["trajectories"] and serial["events"]


def test_query_prefetch_filters_deduplicates_and_preserves_mutation_order(monkeypatch):
    class Pool:
        config = runtime.load_config()
        def references(self, queries):
            self.seen = list(queries)
            return {q: (None, None) for q in dict.fromkeys(queries)}
    pool = Pool()
    world = runtime.ParallelReferenceWorld(reference_pool=pool)
    world.seed_hold("z", "inner", 0, .8)
    world.seed_hold("a", "inner", 1, .8)
    def frozen_update(w):
        assert (*("inner", 0, 0., 0), w.config.profile_hash) in w._q_cache
        assert (*("inner", 1, 0., 0), w.config.profile_hash) in w._q_cache
        return "frozen update called after all cache insertion"
    monkeypatch.setattr(RotatingSlotWorld, "_update_emergency", frozen_update)
    assert world._update_emergency().startswith("frozen update")
    assert pool.seen == [("inner", 1, 0., 0), ("inner", 0, 0., 0)]
    assert world.prefetched_queries == 2
    world._update_emergency()
    assert world.prefetched_queries == 2  # already exact-cached


def test_three_ring_world_multiple_updates_and_emergency_events_exact():
    measured = probe.world_probe(.4, [2], "mixed_3ring")
    variant = measured["variants"][0]
    assert not variant["differences"]
    for serial_stage, parallel_stage in zip(measured["serial_stages"], variant["stages"]):
        assert not parallel_stage["differences"]
        assert serial_stage["record"] == parallel_stage["record"]
    serial = measured["serial_record"]
    assert measured["serial_stages"][0]["record"]["states"][-1]["time"] == .4
    assert serial["states"][-1]["time"] == .8
    assert any(e["type"] == "emergency_declared" for e in serial["events"])
    assert len(serial["states"]) >= 3
    assert len(serial["states"][0]["aircraft"]) == 9
    assert len(variant["pool"]["worker_pids"]) == 2
    assert variant["pool"]["closed_and_joined"]
    assert variant["stages"][1]["timing"]["child_task_cpu_seconds"] > 0
    fixture = probe.mixed_3ring_fixture()
    assert len(fixture) == 9
    assert sum(a["soc"] == .26 for a in fixture) == 1


def test_real_diagnostic_output_and_parallel_metadata_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CODE_COMMIT", launcher.recovery.FROZEN_COMMIT)
    job = {"id": "parallel_independent_output_smoke", "kind": "benchmark", "count": 1,
        "mode": "stable", "repeat": 0, "duration": .2, "pilot": True,
        "output": str(tmp_path), "command": "pytest independent parallel pilot; NOT FORMAL/G09"}
    row = launcher.parallel_job(job, inner_workers=1, recovery_commit="a"*40)
    assert row["status"] == "COMPLETED", row["reason"]
    assert row["simulated_seconds"] == .2 and row["state_count"] >= 2
    assert row["metrics"]["conservation"] is True
    result_path = tmp_path/job["id"]/"result.json"
    assert json.loads(result_path.read_bytes()) == row
    archive = launcher.recovery.ReplayBundle.validate(tmp_path/job["id"]/"replay")
    assert archive.state_count == row["state_count"]
    assert row["metrics"]["pad_busy_seconds"]
    assert all(isinstance(k, str) for k in row["metrics"]["pad_busy_seconds"])
    assert row["metrics"] == json.loads((archive.path/"metrics.json").read_bytes())
    assert row["execution"]["workers"]["pad_tasks"] >= 6
    assert row["execution"]["workers"]["closed_and_joined"]
    assert row["child_task_cpu_seconds"] > 0
    print("REAL_PARALLEL_SMOKE_RESULT="+str(result_path))
    print("REAL_PARALLEL_SMOKE_REPLAY="+str(archive.path/"replay.json"))


def test_worker_exception_propagates_without_fallback_and_joins_children():
    before = {p.pid for p in mp.active_children()}
    pool = runtime.ReferencePool(2)
    with pytest.raises((ValueError, KeyError)):
        pool.references([("unknown_ring", 0, 0., 0)])
    assert pool.closed
    assert pool.failures
    assert {p.pid for p in mp.active_children()} <= before
    with pytest.raises(RuntimeError, match="closed"):
        pool.references([("inner", 0, 0., 0)])


def test_factory_restored_after_exception_and_sink_is_closed(tmp_path):
    original = diagnostics._new_world
    with runtime.ReferencePool(1) as pool:
        with pytest.raises(RuntimeError, match="sentinel"):
            with runtime.diagnostic_world_factory(pool):
                world, sink = diagnostics._new_world(tmp_path)
                assert isinstance(world, runtime.ParallelReferenceWorld)
                sink.close()
                raise RuntimeError("sentinel")
        assert diagnostics._new_world is original
    assert pool.closed


def test_runner_preserves_parent_cpu_scope_and_writes_matching_row(tmp_path, monkeypatch):
    job = {"output": str(tmp_path), "id": "fixture", "command": "fixture only"}
    original = diagnostics._new_world
    def diagnostic_fixture(actual):
        assert diagnostics._new_world is not original
        (tmp_path/"fixture").mkdir()
        return {"status": "COMPLETED", "cpu_seconds": 2., "measured_components": {},
                "measurement_instrumentation": {}, "metrics": {"conservation": True}}
    monkeypatch.setattr(diagnostics, "_diagnostic_job", diagnostic_fixture)
    row = launcher.parallel_job(job, inner_workers=1, recovery_commit="a"*40)
    assert row["cpu_seconds"] == row["cpu_seconds_parent_only"] == 2.
    assert row["child_task_cpu_seconds"] == 0.
    assert row["known_cpu_seconds_parent_plus_child_tasks"] == 2.
    assert row["execution"]["workers"]["closed_and_joined"]
    assert "parent process only" in row["measured_components"]["scope"]
    assert json.loads((tmp_path/"fixture/result.json").read_bytes()) == row
    assert diagnostics._new_world is original


def test_launcher_injects_one_runner_without_bypassing_legacy_recovery(tmp_path, monkeypatch):
    observed = {}
    old = {"id": "old", "metrics": {"unchanged": True}}
    def fake_parallel(job, **kwargs):
        observed["inner"] = kwargs
        return {"id": job["id"], "execution": {"fixture": True}}
    def legacy(source, output, **kwargs):
        observed["legacy"] = kwargs.copy()
        new = kwargs["runner"]({"id": launcher.recovery.INTERRUPTED_ID})
        return {"jobs": [deepcopy(old), new], "recovery": {"fixture": True}}
    monkeypatch.setattr(launcher, "parallel_job", fake_parallel)
    monkeypatch.setattr(launcher.recovery, "recover", legacy)
    result = launcher.recover_parallel("original", tmp_path, code_commit=launcher.recovery.FROZEN_COMMIT,
        recovery_commit="b"*40, inner_workers=20, command="fixture command")
    assert observed["legacy"]["workers"] == 1
    assert observed["inner"]["inner_workers"] == 20
    assert result["jobs"][0] == old
    assert result["execution"] == {"fixture": True}


def test_probe_query_coverage_and_strict_differences():
    queries = probe.representative_queries()
    assert len(queries) == len(set(queries)) == 81
    assert {q[0] for q in queries[:72]} == {"outer", "middle", "inner"}
    assert sum(q[2] == 1235. for q in queries) == 9
    assert runtime.load_config().airspace.common_angular_speed_radps == .035
    for ring in ("outer", "middle", "inner"):
        actual = set()
        for rr, slot, t, version in queries[:72]:
            if rr == ring:
                phase = (-math.atan2(*runtime.slot_state(rr, slot, t).p[1::-1])) % (2*math.pi)
                actual.add(min(23, int(phase/(2*math.pi)*24)))
        assert actual == set(range(24))
    assert probe.differences({"x": 1.}, {"x": 1.00000000001})
    assert probe.differences({"x": 1}, {"x": 1.})


def test_prefetch_does_not_change_cache_when_worker_fails(monkeypatch):
    class Pool:
        config = runtime.load_config()
        def references(self, queries):
            raise RuntimeError("worker error sentinel")
    world = runtime.ParallelReferenceWorld(reference_pool=Pool())
    world.seed_hold("a", "inner", 0, .8)
    events = deepcopy(world.events)
    with pytest.raises(RuntimeError, match="worker error sentinel"):
        world._update_emergency()
    assert not world._q_cache
    assert world.events == events
