"""Rewrite the A6.4 summary/gate from completed calibration artifacts only."""
import json
from pathlib import Path

from .a64_lever_calibration import write_report


def main() -> None:
    output = Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a64"
    path = output / "a64_lever_calibration.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report.pop("gate", None)
    write_report(report, output)
    print(json.dumps(report["gate"]))


if __name__ == "__main__":
    main()
