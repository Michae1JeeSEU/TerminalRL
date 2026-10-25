import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import pytest
from terminal_marl.analysis.service_capacity import (
    constants_table, fcfs_queue, generate_marks, optimistic_waiting_allowance_s,
    pad_workload_moments, rate_latency_bounds, resource_workload_upper,
    ring_max_packing, run_task, run_tasks, scenarios,
)


def test_geometry():
    assert [ring_max_packing(r, 100) for r in (600, 450, 300)] == [37, 28, 18]
    assert sum(ring_max_packing(r, 100) for r in (600, 450, 300)) == 83
    assert sum(ring_max_packing(r, 80) for r in (600, 450, 300)) == 105
    assert ring_max_packing(40, 100) == 1
    assert ring_max_packing(50, 100) == 2
    for r, d in [(0, 100), (100, 0), (math.nan, 100)]:
        with pytest.raises(ValueError):
            ring_max_packing(r, d)


def test_workload_descent_same_mean_variance():
    moments = {n: pad_workload_moments(s.durations_s, s.probabilities, s.pads) for n, s in scenarios().items()}
    half = moments["baseline6half"]
    assert half["ground_mean_s"] == 66
    assert half["occupied_mean_s"] == 79.5
    assert half["all_served_fixed_mix_upper_per_h"] == pytest.approx(6 * 3600 / 79.5)
    assert moments["original6"]["occupied_mean_s"] == 145.5
    assert moments["original6"]["all_served_fixed_mix_upper_per_h"] < half["all_served_fixed_mix_upper_per_h"]
    deterministic, variable = (moments[n] for n in ("same_mean_deterministic", "same_mean_high_variance"))
    assert deterministic["occupied_mean_s"] == variable["occupied_mean_s"] == half["occupied_mean_s"]
    assert deterministic["ground_variance_s2"] == 0
    assert variable["ground_variance_s2"] > half["ground_variance_s2"]
    assert resource_workload_upper([6], [[43.5, 73.5, 133.5]], [.2, .6, .2]) == pytest.approx(6 / 79.5)


def test_rate_latency_assumed_guarantee():
    assert rate_latency_bounds(8, .1, .2, 10) == {"backlog_jobs": 9, "delay_s": 50}
    with pytest.raises(ValueError):
        rate_latency_bounds(8, .3, .2, 10)


@pytest.mark.parametrize("pads,expected", [(1, [0., 23.5, 47.]), (2, [0., 0., 23.5])])
def test_known_schedule(pads, expected):
    events = fcfs_queue([0, 0, 0], [0] * 3, [1000] * 3, durations_s=[10], pads=pads, waiting_buffer=10)
    assert [e["start_s"] for e in events] == expected
    assert [e["touchdown_s"] for e in events] == [s + 13.5 for s in expected]
    assert [e["release_s"] for e in events] == [s + 23.5 for s in expected]


def test_buffer_immediate_and_rejected_reservation():
    e = fcfs_queue([0, 0, 0, 23.5], [0] * 4, [1000] * 4, durations_s=[10], pads=1, waiting_buffer=1)
    assert [x["outcome"] for x in e] == ["normal", "normal", "reject_buffer", "normal"]
    e = fcfs_queue([0, 1, 23.5], [0] * 3, [1000] * 3, durations_s=[10], pads=1, waiting_buffer=0)
    assert [x["outcome"] for x in e] == ["normal", "reject_buffer", "normal"]
    e = fcfs_queue([0, 0, 0], [0] * 3, [1000, 1, 1000], durations_s=[10], pads=1, waiting_buffer=1)
    assert [x["outcome"] for x in e] == ["normal", "reject_deadline", "normal"]


def test_soc_monotonic_fixed_marks_common_uniforms():
    controls = scenarios()
    a, cls, low = generate_marks(11, 300, 500, controls["low_soc6half"])
    a2, cls2, high = generate_marks(11, 300, 500, controls["baseline6half"])
    assert np.array_equal(a, a2) and np.array_equal(cls, cls2)
    assert np.all(high > low) and np.allclose(high - low, .15)
    assert all(optimistic_waiting_allowance_s(h) > optimistic_waiting_allowance_s(l) for h, l in zip(high, low))
    low_events = fcfs_queue([0, 0], [0, 0], [100, 10], durations_s=[10], pads=1, waiting_buffer=1)
    high_events = fcfs_queue([0, 0], [0, 0], [100, 30], durations_s=[10], pads=1, waiting_buffer=1)
    assert low_events[1]["outcome"] == "reject_deadline"
    assert high_events[1]["outcome"] == "normal"


def test_conservation_drain_constants():
    row = run_task(("baseline6half", 11, 5000., 50., 100.))
    assert row["arrivals"] == row["normal"] + row["reject_buffer"] + row["reject_deadline"]
    assert row["calendar_drain_end_s"] > row["cohort_end_s"] and row["end_waiting_backlog"] > 0
    assert row["drained_waiting_backlog"] == row["drained_in_service"] == 0
    assert sum(c["arrivals"] for c in row["class_outcomes"].values()) == row["arrivals"]
    constants = constants_table()
    assert sum(r["configured_slots"] for r in constants["rings"]) == 83
    assert constants["scenarios"]["inner_radius250"]["radius_counterfactual"]
    assert constants["scenarios"]["inner_radius350"]["waiting_buffer"] == 21


def test_serial_parallel_exact_and_visible_failures():
    tasks = [("baseline6half", 11, 300., 10., 50.), ("low_soc6half", 23, 300., 10., 50.)]
    assert run_tasks(tasks, 1) == run_tasks(tasks, 2)
    rows, failed, counts = run_tasks([("missing", 11, 300., 0., 1.)])
    assert rows == [] and len(failed) == 1 and counts["failed"] == counts["missing"] == 1


def test_cli_roundtrip(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "run_service_capacity.py"
    spec = importlib.util.spec_from_file_location("run_service_capacity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--output", str(tmp_path), "--source-commit", "test-source", "--scenario", "baseline6half", "--seeds", "11", "--rates", "300",
                        "--warmup-s", "10", "--duration-s", "50"]) == 0
    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert payload["task_counts"] == {"expected": 1, "completed": 1, "failed": 0, "duplicate": 0, "missing": 0, "duplicate_requested": 0}
    assert payload["rows"][0] == run_task(("baseline6half", 11, 300., 10., 50.))
    assert payload["source_commit"] == "test-source" and payload["source_commit_source"] == "explicit_argument"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["expected"] == progress["completed"] == 1 and progress["failed"] == 0
    assert "constants" in progress
    assert (tmp_path / "summary.csv").read_text(encoding="utf-8").startswith("scenario,seed,")
