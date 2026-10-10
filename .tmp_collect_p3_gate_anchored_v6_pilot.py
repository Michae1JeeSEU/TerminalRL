"""Read-only compact collector for an already-produced V6 pilot directory."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "/home/user/training_outputs/terminal/p3_gate_anchored_v6_20260925/pilot"
)


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


def _outcome_counts(record: dict[str, Any]) -> dict[str, int]:
    outcomes = record.get("outcomes")
    if isinstance(outcomes, dict):
        values = outcomes.get("raw_outcomes") if isinstance(outcomes.get("raw_outcomes"), dict) else outcomes
        if isinstance(values, dict):
            counts = defaultdict(int)
            for value in values.values():
                value = getattr(value, "value", value)
                counts[str(value)] += 1
            return {
                "normal": int(counts.get("completed", 0)),
                "emergency": int(counts.get("emergency_landed", 0)),
                "failed_energy": int(counts.get("failed_energy", 0)),
                "unresolved": int(counts.get("unresolved", 0)),
            }
    summary = record.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("outcomes"), dict):
        counts = defaultdict(int)
        for value in summary["outcomes"].values():
            counts[str(getattr(value, "value", value))] += 1
        return {
            "normal": int(counts.get("completed", 0)),
            "emergency": int(counts.get("emergency_landed", 0)),
            "failed_energy": int(counts.get("failed_energy", 0)),
            "unresolved": int(counts.get("unresolved", 0)),
        }
    return {
        "normal": int(record.get("normal", 0) or 0),
        "emergency": int(record.get("emergency", 0) or 0),
        "failed_energy": int(record.get("failed_energy", 0) or 0),
        "unresolved": int(record.get("unresolved", 0) or 0),
    }


def _update_fields(record: dict[str, Any]) -> dict[str, Any]:
    update = record.get("update") if isinstance(record.get("update"), dict) else {}
    reference_before = update.get("reference_gate_before")
    reference_after = update.get("reference_gate_after")
    return {
        "accepted_minibatches": int(update.get("accepted_updates", update.get("updates", 0)) or 0),
        "rejected_minibatches": int(update.get("rejected_updates", 0) or 0),
        "attempted_minibatches": int(update.get("attempted_updates", 0) or 0),
        "approx_kl": _number(update.get("approx_kl")),
        "final_rollout_kl": _number(update.get("final_rollout_kl")),
        "reference_kl_before": _number(
            reference_before.get("reference_kl") if isinstance(reference_before, dict)
            else update.get("reference_kl_before")
        ),
        "reference_kl_after": _number(
            reference_after.get("reference_kl") if isinstance(reference_after, dict)
            else update.get("reference_kl_final")
        ),
        "reference_kl_max_accepted": _number(update.get("reference_kl_max_accepted")),
        "restore_exact": record.get("checkpoint_restore_exact"),
        "update_gates": record.get("update_gates"),
        "timing_rollout_wall_seconds": _number(record.get("timing_rollout_wall_seconds")),
        "timing_update_wall_seconds": _number(record.get("timing_update_wall_seconds")),
        "timing_episode_wall_seconds": _number(record.get("timing_episode_wall_seconds")),
    }


def _sum_records(records: list[dict[str, Any]], *, include_scenarios: bool = True) -> dict[str, Any]:
    totals = {key: 0 for key in ("requests", "normal", "emergency", "failed_energy", "unresolved")}
    update_totals = {key: 0 for key in ("accepted_minibatches", "rejected_minibatches", "attempted_minibatches")}
    restore = {"true": 0, "false": 0, "missing": 0}
    timing_totals = defaultdict(float)
    timing_counts = defaultdict(int)
    kl_values: dict[str, list[float]] = defaultdict(list)
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        counts = _outcome_counts(record)
        totals["requests"] += int(record.get("requests", sum(counts.values())) or 0)
        for key in ("normal", "emergency", "failed_energy", "unresolved"):
            totals[key] += counts[key]
        fields = _update_fields(record)
        for key in update_totals:
            update_totals[key] += fields[key]
        restored = fields["restore_exact"]
        if restored is True:
            restore["true"] += 1
        elif restored is False:
            restore["false"] += 1
        else:
            restore["missing"] += 1
        for key in ("timing_rollout_wall_seconds", "timing_update_wall_seconds", "timing_episode_wall_seconds"):
            if fields[key] is not None:
                timing_totals[key] += float(fields[key])
                timing_counts[key] += 1
        for key in ("approx_kl", "final_rollout_kl", "reference_kl_before", "reference_kl_after", "reference_kl_max_accepted"):
            if fields[key] is not None:
                kl_values[key].append(float(fields[key]))
        scenario = str(record.get("scenario", "unknown"))
        if include_scenarios:
            by_scenario[scenario].append(record)
    return {
        **totals,
        "resolved": totals["normal"] + totals["emergency"],
        "by_scenario": {
            name: _sum_records(rows, include_scenarios=False) for name, rows in by_scenario.items()
        } if include_scenarios else {},
        "updates": update_totals,
        "checkpoint_restore": restore,
        "timing_seconds_total": dict(timing_totals),
        "timing_seconds_mean": {
            key: timing_totals[key] / timing_counts[key] for key in timing_totals
        },
        "kl_values": {key: values for key, values in kl_values.items()},
        "records": len(records),
    }


def _dev_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    rows = []
    priority_values = []
    low_soc_values = []
    for record in value.get("records", []):
        if not isinstance(record, dict):
            continue
        lifecycle = record.get("lifecycle") if isinstance(record.get("lifecycle"), dict) else {}
        priority = _number(lifecycle.get("priority_weighted_wait_seconds"))
        low_soc = lifecycle.get("low_soc") if isinstance(lifecycle.get("low_soc"), dict) else {}
        low_soc_mean = _number(low_soc.get("mean_wait_seconds"))
        if priority is not None:
            priority_values.append(float(priority))
        if low_soc_mean is not None:
            low_soc_values.append(float(low_soc_mean))
        rows.append({
            "seed": record.get("seed"), "scenario": record.get("scenario"),
            "full_resolution": record.get("full_resolution"),
            "priority_weighted_wait_seconds": priority,
            "low_soc_mean_wait_seconds": low_soc_mean,
            "low_soc_count": low_soc.get("count"),
        })
    return {
        "stage": value.get("stage"), "PASS": value.get("PASS"),
        "full_resolution": value.get("full_resolution"),
        "records": rows,
        "priority_weighted_wait_mean_seconds": (
            sum(priority_values) / len(priority_values) if priority_values else None
        ),
        "low_soc_mean_wait_mean_seconds": (
            sum(low_soc_values) / len(low_soc_values) if low_soc_values else None
        ),
    }


def collect(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "manifest.json"
    progress_path = root / "progress.json"
    source = manifest_path if manifest_path.is_file() else progress_path
    if not source.is_file():
        return {"status": "MISSING", "root": str(root), "manifest": str(manifest_path)}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"status": "INVALID", "root": str(root), "source": str(source)}
    records = [item for item in payload.get("records", []) if isinstance(item, dict)]
    return {
        "status": "OBSERVED",
        "root": str(root), "source": str(source),
        "stage": payload.get("stage"), "status_field": payload.get("status"),
        "accepted_episodes": payload.get("accepted_episodes"),
        "episodes_recorded": payload.get("episodes_recorded"),
        "PILOT_TECHNICAL_PASS": payload.get("PILOT_TECHNICAL_PASS"),
        "aggregate": _sum_records(records),
        "dev_before": _dev_summary(payload.get("dev_before")),
        "dev_after": _dev_summary(payload.get("dev_after")),
        "dev_evaluations": [_dev_summary(item) for item in payload.get("dev_evaluations", [])],
        "performance_comparison": payload.get("performance_comparison"),
    }


def _self_test() -> None:
    records = [
        {"scenario": "flow6", "seed": 1, "requests": 2, "normal": 1, "emergency": 1,
         "failed_energy": 0, "unresolved": 0,
         "update": {"accepted_updates": 2, "rejected_updates": 0, "approx_kl": 0.01},
         "checkpoint_restore_exact": True},
        {"scenario": "dense24", "seed": 2, "requests": 2, "normal": 1, "emergency": 0,
         "failed_energy": 1, "unresolved": 0,
         "update": {"accepted_updates": 1, "rejected_updates": 1, "reference_kl_final": 0.02},
         "checkpoint_restore_exact": False},
    ]
    result = _sum_records(records)
    assert result["normal"] == 2 and result["emergency"] == 1
    assert result["failed_energy"] == 1 and result["updates"]["accepted_minibatches"] == 3
    assert result["checkpoint_restore"] == {"true": 1, "false": 1, "missing": 0}
    print(json.dumps({"status": "SELF_TEST_OK", "aggregate": result}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        print(json.dumps(collect(args.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
