"""Aggregate frozen A6.3 short screens into a no-training bottleneck diagnosis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load(output: Path, scenario: str, seed: int) -> dict:
    return json.loads((output / f"a63_dynamic_{scenario}_seed{seed}.json").read_text(encoding="utf-8"))


def build_report(output: str | Path) -> dict:
    root = Path(output)
    mechanism = json.loads((root / "a63_mechanism_screen.json").read_text(encoding="utf-8"))
    rows = {scenario: [_load(root, scenario, seed) for seed in range(3)] for scenario in ("high", "bursty")}
    summary = {}
    for scenario, values in rows.items():
        delta = {key: [value["delta_dynamic_minus_fixed"][key] for value in values]
                 for key in ("completed", "external_queue_area", "service_delay_p95")}
        dynamic = [value["dynamic_rule"] for value in values]
        summary[scenario] = {
            "protected_safety_passed": all(value["protected_safety_passed"] for value in values),
            "paired_delta_mean": {key: float(np.mean(items)) for key, items in delta.items()},
            "paired_delta_per_seed": delta,
            "effective_geometry_updates_per_seed": [row["effective_geometry_updates"] for row in dynamic],
            "headway_updates_per_seed": [row["headway_speed_updates"] for row in dynamic],
            "capacity_ranges_per_seed": [{"min": row["capacity_min"], "max": row["capacity_max"]} for row in dynamic],
            "minimum_swept_separation": [row["minimum_swept_separation"] for row in dynamic],
        }
    high_actions = rows["high"][0]["dynamic_rule"]
    diagnosis = {
        "decision": "STOP_BEFORE_PPO",
        "reason": [
            "The deterministic headway-only probe establishes a safe operational lever, but the current dynamic_rule applies headway increases while the beneficial probe decreases headways.",
            "At high load, dynamic_rule loses completions on every seed; therefore the rule-policy short screen does not establish stable high-demand terminal benefit.",
            "Applied structural increments leave derived capacity unchanged (outer/middle/inner capacity remains 47/31/15), so the current geometry action envelope is below the capacity quantization threshold in this horizon.",
            "Bursty results are mixed (one negative and two positive completion deltas), insufficient as a stable cross-seed basis for PPO training.",
        ],
        "evidence": {
            "headway_only_probe": mechanism["leverage_vs_nominal"]["headway_only"],
            "high_dynamic_headway_range_seed0": {"min": high_actions["headway_min"], "max": high_actions["headway_max"]},
            "high_dynamic_capacity_range_seed0": {"min": high_actions["capacity_min"], "max": high_actions["capacity_max"]},
        },
        "next_safe_research": [
            "Do not train PPO or evaluate learned checkpoints.",
            "Calibrate a prospective rule action envelope: test bounded headway decreases against queue/priority and LZ/final pressure while retaining protected safety.",
            "Calibrate structural step/dwell choices only when an applied radius/height change crosses a derived-capacity threshold; otherwise treat geometry as non-levered for this demand/horizon.",
            "Repeat deterministic probes after calibration before any new high-level PPO decision.",
        ],
    }
    return {"stage": "A6.3", "mechanism_screen": mechanism, "dynamic_rule_short_screen": summary, "bottleneck_diagnosis": diagnosis}


def write_report(report: dict, output: str | Path) -> None:
    root = Path(output)
    (root / "a63_bottleneck_diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    diagnosis = report["bottleneck_diagnosis"]
    lines = ["# A6.3 mechanism and bottleneck diagnosis", "", f"Decision: **{diagnosis['decision']}**.", "",
             "## Evidence", "", "| Demand | Safety | Mean completed delta | Mean queue-area delta | Mean P95 delay delta |", "|---|---:|---:|---:|---:|"]
    for scenario, value in report["dynamic_rule_short_screen"].items():
        delta = value["paired_delta_mean"]
        lines.append(f"| {scenario} | {value['protected_safety_passed']} | {delta['completed']:.2f} | {delta['external_queue_area']:.1f} | {delta['service_delay_p95']:.1f} |")
    lines.extend(["", "## Diagnosis", *[f"- {item}" for item in diagnosis["reason"]], "", "## Next safe research", *[f"- {item}" for item in diagnosis["next_safe_research"]]])
    (root / "a63_bottleneck_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output = Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a63"
    report = build_report(output)
    write_report(report, output)
    print(json.dumps({"decision": report["bottleneck_diagnosis"]["decision"]}))


if __name__ == "__main__":
    main()
