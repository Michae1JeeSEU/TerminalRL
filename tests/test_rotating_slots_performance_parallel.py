"""Bounded adapter tests; these do not establish P4 performance qualification."""
from dataclasses import replace
from hashlib import sha256
import multiprocessing as mp
from pathlib import Path
import json
import subprocess
import sys
import time

import pytest

from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
from terminal_marl.core.rotating_slots.performance.parallel import (
    POLICY_ID, ReferencePool, install_parallel_reference, _spawn_module)

CORE = Path(__file__).resolve().parents[1]/"terminal_marl/core/rotating_slots"
SEMANTIC = {"policy_id": POLICY_ID, "effective_rules_hash": "test-rules",
            "candidate_policy_hash": "test-candidates", "numerical_policy_hash": "test-numerics"}


def pool(workers=1, **kwargs):
    values = dict(config=load_config(), expected_core_root=CORE,
        expected_core_manifest={p.name: sha256(p.read_bytes()).hexdigest() for p in CORE.glob("*.py")},
        semantic_identity=SEMANTIC, process_budget=workers+1, core_budget=workers,
        exclusive_process_accounting=True)
    values.update(kwargs)
    return ReferencePool(workers, **values)


def pure_task(task):
    # Reverse finish order and mutate the child-only world. Pad order must win.
    parallel = _spawn_module()
    parallel._WORLD.time = 9123
    time.sleep((6-task["pad"])*.003)
    return {"query": task["query"], "pad": task["pad"], "identity": task["identity"],
        "candidate": (1., f"pad{task['pad']}"), "pid": __import__("os").getpid(),
        "phase_bin": task["pad"], "canonical_duration": 20.,
        "fallback": task["pad"] in (1, 4),
        "reference_counts": {"initial_search": 1, "template_instantiation": 1,
                             "fallback_search": int(task["pad"] in (1, 4))}}


def exception_task(task):
    raise RuntimeError("injected-pad-failure")


def wrong_identity_task(task):
    row = pure_task(task)
    row["identity"] = {"bad": "profile"}
    return row


def wrong_query_task(task):
    row = pure_task(task)
    row["query"] = (*task["query"][:3], 99)
    return row


def death_task(task):
    __import__("os")._exit(17)


@pytest.mark.parametrize("workers", [1, 2, 4, 6])
def test_real_reference_and_emergency_equivalence(workers):
    serial, parallel = RotatingSlotWorld(), RotatingSlotWorld()
    serial.seed_hold("a", "inner", 0, soc=.8)
    parallel.seed_hold("a", "inner", 0, soc=.8)
    expected = serial._reference("inner", 0, 0., 0)
    original = RotatingSlotWorld._reference
    authoritative_fields = set(parallel.__dict__)
    with pool(workers) as executor:
        with install_parallel_reference(parallel, executor):
            actual = parallel._reference("inner", 0, 0., 0)
            assert actual[0] == expected[0]
            assert actual[1].to_dict() == expected[1].to_dict()
            assert actual[1].trajectory_hash == expected[1].trajectory_hash
            assert parallel._last_reference_fallback == serial._last_reference_fallback
            assert parallel._reference_templates == serial._reference_templates
            assert parallel._update_emergency() == serial._update_emergency()
            assert parallel.aircraft == serial.aircraft
            assert parallel.events == serial.events
            assert parallel.records == serial.records
            assert parallel._reference("inner", 0, 0., 0) is actual
            assert executor.tasks == 6
            assert set(parallel.__dict__) == authoritative_fields
        assert RotatingSlotWorld._reference is original
    summary = executor.summary()
    assert summary["closed_and_joined"]
    assert summary["resources"]["cpu_complete"], summary
    assert summary["resources"]["process_tree_cpu_seconds"] >= 0
    assert len(summary["worker_pids"]) == workers
    assert not any(p.pid in executor.worker_pids for p in mp.active_children())
    assert "functions" not in summary


def test_ties_cache_order_last_fallback_and_parent_isolation():
    world, other = RotatingSlotWorld(), RotatingSlotWorld()
    original = RotatingSlotWorld._reference
    with pool(2) as executor:
        executor._task = pure_task
        with install_parallel_reference(world, executor):
            other._q_cache[("inner", 0, 1.25, 3, other.config.profile_hash)] = (7., "other")
            assert world._reference("inner", 0, 1.25, 3) == (1., "pad0")
            assert world.time == 0
            assert world._last_reference_fallback["pad_id"] == 4
            assert world.reference_counts == {"initial_search": 6, "template_instantiation": 6,
                                              "fallback_search": 2}
            assert world.cache_hits == 0
            assert world._reference("inner", 0, 1.25, 3) == (1., "pad0")
            assert world.cache_hits == 1
            assert executor.tasks == 6
            assert other._reference.__func__ is not original  # class patch delegates other instances
            assert other._reference("inner", 0, 1.25, 3) == (7., "other")
            assert executor.tasks == 6
        assert RotatingSlotWorld._reference is original


@pytest.mark.parametrize("task,match", [(exception_task, "injected-pad-failure"),
    (wrong_identity_task, "identity mismatch"), (wrong_query_task, "identity mismatch"),
    (death_task, "terminated abruptly")])
def test_failures_propagate_and_join(task, match):
    world = RotatingSlotWorld()
    executor = pool(2)
    executor._task = task
    with pytest.raises(Exception, match=match):
        with install_parallel_reference(world, executor):
            world._reference("inner", 0, 0.)
    assert executor.closed and executor.joined and executor.aborted
    assert world._q_cache == {}
    assert executor.failures
    assert not any(p.pid in executor.worker_pids for p in mp.active_children())


def test_profile_is_bounded_and_payload_equal(tmp_path):
    expected = RotatingSlotWorld()._reference("inner", 0, 0.)
    with pool(mode="profile", profile_directory=tmp_path, max_profile_tasks=6) as executor:
        world = RotatingSlotWorld()
        with install_parallel_reference(world, executor):
            actual = world._reference("inner", 0, 0.)
            assert actual[0] == expected[0]
            assert actual[1].to_dict() == expected[1].to_dict()
            assert len(list(tmp_path.glob("*.prof"))) == 6
            with pytest.raises(RuntimeError, match="profile task budget"):
                world._reference("inner", 0, 1.)
    assert executor.summary()["profiler_mode"] == "profile"


def test_benchmark_does_not_create_profiler_or_profile_files(tmp_path, monkeypatch):
    # Enforce in-process task branch as well as observing spawned benchmark.
    import cProfile
    parallel = _spawn_module()
    config = load_config()
    world = RotatingSlotWorld(config=config)
    identity = {"test": "benchmark"}
    monkeypatch.setattr(parallel, "_WORLD", world)
    monkeypatch.setattr(parallel, "_IDENTITY", identity)
    monkeypatch.setattr(cProfile, "Profile", lambda *a, **kw: pytest.fail("profiler constructed"))
    row = parallel._pad_task({"query": ("inner", 0, 0., 0), "pad": 0,
        "identity": identity, "mode": "benchmark", "known_template": (False, None)})
    assert "functions" not in row and "components" not in row
    with pool(profile_directory=tmp_path) as executor:
        with install_parallel_reference(RotatingSlotWorld(), executor) as bound:
            bound._reference("inner", 0, 0.)
    assert not list(tmp_path.iterdir())


def test_budget_identity_and_source_fail_closed(tmp_path):
    for kwargs in ({"workers": 7}, {"workers": 0}, {"process_budget": 1},
                   {"core_budget": 0}, {"exclusive_process_accounting": False}, {"kernel": "UNKNOWN"}):
        with pytest.raises(ValueError):
            pool(**kwargs)
    with pytest.raises(ValueError, match="policy_id"):
        pool(semantic_identity={**SEMANTIC, "policy_id": "legacy_f67"})
    with pytest.raises(ValueError, match="hash mismatch"):
        manifest = {p.name: sha256(p.read_bytes()).hexdigest() for p in CORE.glob("*.py")}
        pool(expected_core_manifest={**manifest, "world.py": "wrong"})
    with pool() as executor:
        bad = RotatingSlotWorld(config=replace(load_config(), profile_hash="wrong"))
        with pytest.raises(ValueError, match="configuration"):
            with install_parallel_reference(bad, executor):
                pass


def test_context_exception_restores_class_and_joins():
    original = RotatingSlotWorld._reference
    executor = pool()
    with pytest.raises(KeyboardInterrupt):
        with executor:
            with install_parallel_reference(RotatingSlotWorld(), executor):
                raise KeyboardInterrupt()
    assert RotatingSlotWorld._reference is original
    assert executor.closed and executor.joined
    assert not any(p.pid in executor.worker_pids for p in mp.active_children())


def test_worker_initializer_rejects_changed_profile_and_reaps():
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(Exception):
        pool(config=replace(load_config(), profile_hash="wrong-profile"))
    assert {p.pid for p in mp.active_children()} == before


def test_cache_disabled_repeats_and_pool_restart_preserve_real_payload():
    expected = RotatingSlotWorld(cache_enabled=False)._reference("inner", 0, .125, 4)
    expected_payload = expected[1].to_dict()
    for _ in range(2):
        world = RotatingSlotWorld(cache_enabled=False)
        with pool(2, cache_enabled=False) as executor:
            with install_parallel_reference(world, executor):
                for _ in range(2):
                    actual = world._reference("inner", 0, .125, 4)
                    assert actual[0] == expected[0]
                    assert actual[1].to_dict() == expected_payload
                assert executor.tasks == 12
                assert not world._q_cache


def test_frozen_r0_import_root_with_external_helper():
    reference = CORE.parents[2]/"artifacts/rotating_slots/performance_reference_r0"
    if not reference.exists():
        pytest.skip("P0 R0 source export is not present")
    code = r'''
import hashlib, json, pathlib, sys
sys.dont_write_bytecode=True
root, helper = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(root))
import terminal_marl.core.rotating_slots as package
package.__path__.append(str(helper))
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
from terminal_marl.core.rotating_slots.performance.parallel import ReferencePool, install_parallel_reference, POLICY_ID
core = root / "terminal_marl/core/rotating_slots"
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in core.glob("*.py")}
semantic = {"policy_id": POLICY_ID, "effective_rules_hash": "test-rules",
            "candidate_policy_hash": "test-candidates", "numerical_policy_hash": "test-numerics"}
serial, world = RotatingSlotWorld(), RotatingSlotWorld()
expected = serial._reference("inner", 0, 0.)
with ReferencePool(2, config=world.config, expected_core_root=core,
    expected_core_manifest=manifest, semantic_identity=semantic,
    process_budget=3, core_budget=2, exclusive_process_accounting=True) as pool:
    with install_parallel_reference(world, pool):
        actual = world._reference("inner", 0, 0.)
        assert actual[0] == expected[0]
        assert actual[1].to_dict() == expected[1].to_dict()
import terminal_marl.core.rotating_slots.world as module
print(json.dumps({"core_path": module.__file__, "closed": pool.closed,
                  "cpu_complete": pool.summary()["resources"]["cpu_complete"]}))
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code,
                             str(reference), str(CORE)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)
    assert Path(row["core_path"]).is_relative_to(reference)
    assert row["closed"] and row["cpu_complete"]


def test_resource_samples_are_bounded_and_full_series_is_explicit(tmp_path):
    from terminal_marl.core.rotating_slots.performance.process_metrics import OwnedProcessMetrics
    path = tmp_path/"rss.jsonl"
    metrics = OwnedProcessMetrics(exclusive=True, defer_sampler=True, rss_series_path=path)
    try:
        for _ in range(520):
            metrics._sample()
        count = metrics.sample_count
        metrics.register([])
        assert metrics.sample_count == count
        assert len(metrics.samples) == 512
    finally:
        result = metrics.finish(joined=True)
    assert len(path.read_text().splitlines()) == result["rss_sample_count"]
    assert result["sampled_peak_simultaneous_rss_bytes"] >= max(
        x["simultaneous_rss_bytes"] for x in result["rss_samples_recent"])
    assert result["sampled_peak_rss_sweep_sum_bytes"] == result["sampled_peak_simultaneous_rss_bytes"]
    assert result["lifetime_hwm_sum_estimate_bytes"] == result["sum_lifetime_peak_upper_bound_bytes"]
    assert result["rss_is_strict_bound"] is False
    assert "report-only" in result["rss_limit_gate"] and "neither" in result["rss_limit_gate"]
    assert "strict enclosure" in result["rss_error"]


def test_kernel_label_changes_only_summary_metadata():
    rows=[]
    for kernel in ("R0", "OPT"):
        with pool(kernel=kernel) as executor:
            rows.append(executor.summary())
    assert rows[0]["identity"] == rows[1]["identity"]
    for kernel,row in zip(("R0","OPT"),rows):
        assert row["kernel"]==kernel and row["execution_backend"]==kernel+"_PARALLEL"
        assert row["prefix_execution"]==kernel+"_PARALLEL"


def test_rss_estimates_do_not_assert_hwm_encloses_sweep(monkeypatch):
    from terminal_marl.core.rotating_slots.performance import process_metrics as pm
    # Inject an accounting mismatch; these synthetic bytes are not RSS evidence.
    monkeypatch.setattr(pm,"_rss",lambda pid,peak=False: 1 if peak else 2**60)
    metrics=pm.OwnedProcessMetrics(exclusive=True,defer_sampler=True)
    result=metrics.finish(joined=True)
    assert result["lifetime_hwm_sum_estimate_bytes"]<result["sampled_peak_rss_sweep_sum_bytes"]
    assert result["rss_is_strict_bound"] is False
    assert "report-only" in result["rss_limit_gate"]
    assert result["external_controller_included"] is False
