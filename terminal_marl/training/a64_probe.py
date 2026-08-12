"""Run only the deterministic A6.4 calibration probes, not the short screen."""
import json
from pathlib import Path

from .a64_lever_calibration import capacity_threshold_probe, headway_sensitivity


def main() -> None:
    output = Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a64"
    output.mkdir(parents=True, exist_ok=True)
    result = {"headway_sensitivity": headway_sensitivity(), "capacity_threshold_probe": capacity_threshold_probe()}
    (output / "a64_calibration_probe.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"safe": result["headway_sensitivity"]["protected_safety_passed"], "capacity_after": result["capacity_threshold_probe"]["capacity_after"]}))


if __name__ == "__main__":
    main()
