from terminal_marl.config import load_experiment_config
from terminal_marl.core.arrival_process import ArrivalProcess
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.experiments.ablation_runner import apply_demand_profile
from terminal_marl.experiments.safe_retrain_gate import run_finite_with_forensics


def test_runtime_arrival_holds_in_external_queue_when_same_gate_approach_active() -> None:
    config = load_experiment_config("dynamic_hrl")
    config.total_requested_uavs = 2
    config.arrival_rate = 10.0
    config.initial_aircraft_count = 0
    config.formation_rows = 0
    config.formation_cols = 0
    config.max_active_uavs = 10
    config.max_steps = 5
    config.max_unsequenced_approach_per_gate = 1
    env = TerminalParallelEnv(config)
    env.reset(seed=0)

    env._runtime_request_due(2)
    first = env.external_arrival_queue[0]
    second = env.external_arrival_queue[1]

    assert env._admit_one_external_arrival(first, 0, 2)
    assert not env._admit_one_external_arrival(second, 1, 2)
    assert sum(
        not aircraft.finished and aircraft.sequence_number is None
        for aircraft in env.aircraft
    ) == 1


def test_safe_gate_forensics_reproduces_nominal_capacity_without_unresolved() -> None:
    result = run_finite_with_forensics(
        seed=0,
        total_requested=3,
        arrival_rate=0.3,
        max_active=3,
        max_steps=1200,
    )

    summary = result["summary"]
    assert summary["conservation_ok"]
    assert summary["unresolved"] == 0
    assert "crashed_collision" in summary
    assert isinstance(result["collision_rows"], list)


def test_safe_demand_profile_csv_overrides_default_profiles(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "safe_demand_profile_summary.csv"
    csv_path.write_text(
        "\n".join(
            [
                "demand,arrival_rate_per_second,initial_aircraft_count,formation_rows,formation_cols",
                "medium,0.0123,0,0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TERMINAL_MARL_DEMAND_PROFILE_CSV", str(csv_path))

    config = apply_demand_profile(load_experiment_config("event_gated_hrl"), "medium")

    assert config.arrival_rate == 0.0123
    assert config.initial_aircraft_count == 0
    assert config.formation_rows == 0
    assert config.formation_cols == 0


def test_overload_demand_level_is_valid_for_stress_evaluation() -> None:
    process = ArrivalProcess(
        demand_level="overload",
        model="deterministic_interval",
        arrival_rate=0.1,
        max_active_uavs=1,
        total_requested_uavs=1,
    )

    assert process.arrivals_due(10.0) == 1
