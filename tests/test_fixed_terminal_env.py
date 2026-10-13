import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
import pytest
from terminal_marl.core.fixed_terminal.domain import Intent, TerminalConfig
from terminal_marl.core.fixed_terminal.scenarios import Request, Scenario
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv
from terminal_marl.scripts.evaluate_fixed_terminal_stage1 import run


def scenario(*requests):
    return Scenario(1, "custom_test", "custom_test", requests, 1, 2)


def test_landings_censoring_all_request_denominators_and_energy_conservation():
    c = TerminalConfig(horizon=20)
    s = scenario(Request("land", 0, .8, 0, 0), Request("late", 19.5, .7, None, 0))
    env = FixedTerminalEnv(c, s)
    assert env.submit(Intent("land")).status == "feasible"
    while not env.done:
        env.step()
    report = env.report()
    summary = report["summary"]
    assert summary["outcomes"] == {"normal_landed": 1, "emergency_landed": 0, "failed_energy": 0, "censored": 1, "not_arrived": 0}
    assert summary["normal_proportion_all_requests"] == .5
    assert summary["throughput_landings_per_second"] == .05
    assert summary["landing_metric_denominator"] == 1
    assert summary["conservation_ok"]
    for row in report["requests"]:
        assert row["initial_soc"]-row["final_soc"] == pytest.approx(row["energy_consumed_soc"])
    late = report["requests"][1]
    assert late["energy_consumed_soc"] == pytest.approx(.5*c.energy.hold)
    assert late["landing_duration_seconds"] is None
    assert summary["normal_throughput_per_second"] == .05
    assert summary["normal_landing_metric_denominator"] == 1
    assert summary["emergency_landing_metric_denominator"] == 0
    assert summary["censored_state_counts"]["external"] == 1
    assert summary["observed_request_exposure_seconds"] == pytest.approx(report["requests"][0]["completion_time"]+.5)


def test_future_requests_are_not_yet_observed_and_low_energy_admission_rejected():
    env = FixedTerminalEnv(TerminalConfig(horizon=10), scenario(Request("future", 9, .8, None, 0), Request("low", 0, .01, None, 0)))
    report = env.report()
    assert report["summary"]["outcomes"]["not_arrived"] == 1
    assert report["summary"]["arrived_request_count"] == 1
    env.step()
    assert env.aircraft["low"].state == "external"
    assert env.admission_rejection_counts["minimum_route_energy_reserve"] == 1
    assert env.events[-1]["reason"] == "minimum_route_energy_reserve"


def test_explicit_emergency_intent_revalidates_eligibility_and_counts_cost():
    from math import pi
    env = FixedTerminalEnv(TerminalConfig(horizon=40), scenario(Request("a", 0, .09, 2, pi)))
    assert env.submit(Intent("a", emergency=True)).status == "feasible"
    while not env.done:
        env.step()
    result = env.report()
    assert result["summary"]["outcomes"]["emergency_landed"] == 1
    assert result["summary"]["normal_landing_metric_denominator"] == 0
    assert result["summary"]["emergency_landing_metric_denominator"] == 1
    assert result["summary"]["emergency_reserved_resource_seconds"] > 0
    assert result["requests"][0]["final_soc"] >= env.config.energy.reserve
    rich = FixedTerminalEnv(TerminalConfig(horizon=40), scenario(Request("rich", 0, .4, 2, pi)))
    assert rich.submit(Intent("rich", emergency=True)).reason == "normal_plan_exists"
    assert rich.search_candidates == 1


@pytest.mark.parametrize("bad_request", [Request("negative", 0, .8, -1, 0),
                                     Request("outside", 0, .8, 3, 0),
                                     Request("boolean", 0, .8, True, 0),
                                     Request("fractional", 0, .8, 1.5, 0),
                                     Request("future-initial", 1, .8, 0, 0),
                                     Request("nan", 0, .8, 0, float("nan"))])
def test_invalid_initial_requests_rejected(bad_request):
    with pytest.raises(ValueError):
        FixedTerminalEnv(TerminalConfig(), scenario(bad_request))


def test_cutoff_preserves_scheduled_request_as_censored_not_emergency():
    env = FixedTerminalEnv(TerminalConfig(horizon=1), scenario(Request("a", 0, .8, 2, 2)))
    env.exercise_feasibility()
    assert env.aircraft["a"].plan.end > env.config.horizon
    env.step()
    row = env.report()["requests"][0]
    assert row["outcome"] == "censored" and row["state_at_cutoff"] == "scheduled"
    assert env.report()["summary"]["outcomes"]["emergency_landed"] == 0


def test_energy_exhaustion_is_explicit_absorbing_failure():
    env = FixedTerminalEnv(TerminalConfig(horizon=3), scenario(Request("a", 0, .0005, 0, 1)))
    while not env.done:
        env.step()
    row = env.report()["requests"][0]
    assert row["outcome"] == "failed_energy"
    assert row["final_soc"] == 0
    assert row["reason"] == "SOC_exhausted_before_landing"
    assert any(event["event"] == "failed_energy" for event in env.events)


@pytest.mark.parametrize("dt", [.2, 1., 3.])
@pytest.mark.parametrize("external", [False, True])
def test_exact_depletion_time_is_independent_of_tick(dt, external):
    arrival = .15 if external else 0.
    env = FixedTerminalEnv(TerminalConfig(horizon=3, dt=dt),
                           scenario(Request("a", arrival, .0005, None if external else 0, 1)))
    while not env.done:
        env.step()
    row = env.report()["requests"][0]
    expected = arrival+.0005/env.config.energy.hold
    assert row["completion_time"] == pytest.approx(expected)
    assert row["observed_request_seconds"] == pytest.approx(expected-arrival)
    failure = next(event for event in env.events if event["event"] == "failed_energy")
    assert failure["time"] == pytest.approx(expected)
    assert row["energy_consumed_soc"] == pytest.approx(.0005)


def test_admission_cannot_exceed_capacity():
    c = TerminalConfig(horizon=2)
    from math import pi
    outer = c.operational_capacity(2)
    requests = [Request(str(i), 0, .8, 2, 2*pi*(i+.3)/outer) for i in range(outer)]
    requests.append(Request("external", 0, .8, None, 0))
    env = FixedTerminalEnv(c, scenario(*requests))
    env.step()
    assert env.aircraft["external"].state == "external"
    assert env.aircraft["external"].soc == pytest.approx(.8-c.energy.hold)


def test_runner_serializes_frozen_inputs_request_rows_and_events(tmp_path):
    result = run(tmp_path/"run", config=TerminalConfig(horizon=2), search_limit=8)
    for name in ("frozen_input.json", "result.json"):
        assert json.loads((tmp_path/"run"/name).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (tmp_path/"run"/"requests.jsonl").read_text().splitlines()]
    assert len(rows) == result["summary"]["request_count"]
    assert all(row["outcome"] == "censored" for row in rows)
    assert (tmp_path/"run"/"events.jsonl").stat().st_size > 0
    with pytest.raises(FileExistsError):
        run(tmp_path/"run", config=TerminalConfig(horizon=2))


def test_cli_and_import_isolation_and_legacy_lazy_exports():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-m", "terminal_marl.scripts.evaluate_fixed_terminal_stage1", "--help"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0 and "--search-limit" in result.stdout
    code = """
import sys, types
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv
assert 'torch' not in sys.modules and 'gymnasium' not in sys.modules
assert 'terminal_marl.envs.parallel_env' not in sys.modules
import terminal_marl, terminal_marl.envs
legacy = types.ModuleType('terminal_marl.envs.parallel_env')
legacy.TerminalParallelEnv = object()
sys.modules[legacy.__name__] = legacy
assert terminal_marl.TerminalParallelEnv is legacy.TerminalParallelEnv
assert terminal_marl.envs.TerminalParallelEnv is legacy.TerminalParallelEnv
config = types.ModuleType('terminal_marl.core.config')
config.SimConfig = object(); config.RingRole = object()
sys.modules[config.__name__] = config
assert terminal_marl.SimConfig is config.SimConfig
assert terminal_marl.RingRole is config.RingRole
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
