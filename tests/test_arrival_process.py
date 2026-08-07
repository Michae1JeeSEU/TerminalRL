import math

import pytest

from terminal_marl.core.arrival_process import (
    ArrivalProcess,
    generate_arrival_schedule,
    schedule_times,
)


@pytest.mark.parametrize("model", ["deterministic_interval", "poisson", "bursty_wave"])
def test_arrival_models_schedule_runtime_traffic(model: str) -> None:
    process = ArrivalProcess("high", model, 1.0, 10, seed=3)
    assert math.isfinite(process.next_arrival_time)
    due = process.arrivals_due(process.next_arrival_time + 1e-6, 0)
    assert due >= 1


def test_max_active_capacity_does_not_censor_external_arrival_request() -> None:
    process = ArrivalProcess("low", "deterministic_interval", 1.0, 2)
    assert process.arrivals_due(100.0, 2) == 1


def test_fixed_arrival_schedule_replays_same_request_times() -> None:
    schedule = generate_arrival_schedule(
        demand_level="medium",
        model="deterministic_interval",
        arrival_rate=0.5,
        seed=7,
        duration_seconds=5.0,
    )
    times = schedule_times(schedule)
    process = ArrivalProcess(
        "medium",
        "deterministic_interval",
        0.5,
        2,
        fixed_schedule=times,
    )
    assert process.arrivals_due(times[0]) == 1
    assert process.arrivals_due(times[1]) == 1


def test_total_requested_uavs_caps_runtime_generation_exactly() -> None:
    process = ArrivalProcess(
        "medium",
        "deterministic_interval",
        10.0,
        4,
        total_requested_uavs=5,
    )
    due = 0
    current = 0.0
    while math.isfinite(process.next_arrival_time):
        current = process.next_arrival_time + 1e-6
        due += process.arrivals_due(current)

    assert due == 5
    assert not math.isfinite(process.next_arrival_time)


def test_bursty_wave_long_run_mean_matches_configured_rate() -> None:
    rows = generate_arrival_schedule(
        demand_level="bursty",
        model="bursty_wave",
        arrival_rate=0.2,
        seed=11,
        duration_seconds=1000.0,
    )

    observed_rate = len(rows) / 1000.0
    assert observed_rate == pytest.approx(0.2, rel=0.03)
