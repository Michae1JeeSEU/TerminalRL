import math

import pytest

from terminal_marl.core.arrival_process import ArrivalProcess


@pytest.mark.parametrize("model", ["deterministic_interval", "poisson", "bursty_wave"])
def test_arrival_models_schedule_runtime_traffic(model: str) -> None:
    process = ArrivalProcess("high", model, 1.0, 10, seed=3)
    assert math.isfinite(process.next_arrival_time)
    due = process.arrivals_due(process.next_arrival_time + 1e-6, 0)
    assert due >= 1


def test_max_active_capacity_blocks_arrival() -> None:
    process = ArrivalProcess("low", "deterministic_interval", 1.0, 2)
    assert process.arrivals_due(100.0, 2) == 0
