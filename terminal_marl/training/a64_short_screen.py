"""Execute only the A6.4 calibrated-rule six-seed short-screen funnel."""
import json
from pathlib import Path

from .a64_lever_calibration import run_short_screen, write_report


def main() -> None:
    output = Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a64"
    calibration = json.loads((output / "a64_calibration_probe.json").read_text(encoding="utf-8"))
    short_screen = run_short_screen()
    report = {"stage": "A6.4", **calibration, "short_screen": short_screen}
    write_report(report, output)
    print(json.dumps({"safe": short_screen["all_protected_safe"], "consistency": short_screen["directional_consistency"]}))


if __name__ == "__main__":
    main()
