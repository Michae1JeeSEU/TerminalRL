"""Pure numeric nominal inner-ring point/swept separation probe."""
import json
from pathlib import Path

import numpy as np

from terminal_marl.core.safety_filter import SafetyFilter

OUT = Path("results/nominal_holder_swept_probe.json")
RADIUS = 200.0
SPEED = 16.0
HORIZON = 30.0


def measure(chord: float, sample_dt: float) -> dict:
    angle = 2.0 * np.arcsin(chord / (2.0 * RADIUS))
    times = np.arange(0.0, HORIZON + 1e-12, sample_dt)
    theta = SPEED * times / RADIUS
    left = np.column_stack((RADIUS * np.cos(theta), RADIUS * np.sin(theta),
                            np.zeros_like(theta)))
    right_theta = theta + angle
    right = np.column_stack((RADIUS * np.cos(right_theta),
                             RADIUS * np.sin(right_theta),
                             np.zeros_like(theta)))
    point, swept = SafetyFilter._track_distances(left, right)
    return {"target_chord_m": chord, "sample_dt_s": sample_dt,
            "sample_count": len(times), "angular_difference_rad": float(angle),
            "min_point_m": float(np.min(point)),
            "min_swept_m": float(np.min(swept)),
            "swept_intervals": len(swept)}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = [measure(chord, dt) for chord in (80.02, 81.0) for dt in (0.2, 1.0)]
    payload = {"ring_radius_m": RADIUS, "speed_mps": SPEED,
               "horizon_s": HORIZON, "method": "SafetyFilter._track_distances",
               "records": rows}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "bytes": OUT.stat().st_size,
                      "records": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
