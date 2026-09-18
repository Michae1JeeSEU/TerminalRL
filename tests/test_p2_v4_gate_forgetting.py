from __future__ import annotations

import numpy as np

from terminal_marl.training.analyze_v4_2_p2_gate_forgetting import gate_metrics


def _row(label: int, interface: str) -> dict:
    return {"teacher_gate": label, "interface": interface}


def test_gate_metrics_reports_errors_calibration_and_interfaces() -> None:
    rows = [_row(1, "o2m"), _row(1, "m2i"), _row(0, "i2f"), _row(0, "o2m")]
    probabilities = np.asarray([0.1, 0.8, 0.9, 0.2])
    result = gate_metrics(rows, probabilities)
    assert result["teacher"] == {"WAIT": 2, "RELEASE": 2}
    assert result["predicted"] == {"WAIT": 2, "RELEASE": 2}
    assert result["false_WAIT"] == result["false_RELEASE"] == 1
    assert result["false_wait_rate"] == result["false_release_rate"] == 0.5
    assert result["interfaces"]["o2m"]["false_WAIT"] == 1
    assert result["interfaces"]["m2i"]["false_wait_rate"] == 0.0
    assert result["release_probability"]["teacher_RELEASE"]["n"] == 2
    assert np.isfinite(result["brier"]) and np.isfinite(result["binary_nll"])
