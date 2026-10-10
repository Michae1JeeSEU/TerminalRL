"""Read-only failure-inclusive lifecycle metrics for existing V7 dev tapes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

from terminal_marl.core.safety_trace import _digest, _environment_config_match
from terminal_marl.training import evaluate_p3_adaptive_kl_v7 as evaluator
from terminal_marl.training.terminal_lifecycle_metrics import (
    summarize_terminal_rows,
    terminal_lifecycle_metrics,
)


INPUT_ROOT = Path("/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926")
OUTPUT_ROOT = Path("/home/user/training_outputs/terminal/p3_adaptive_kl_terminal_dev_metrics_20260926")
SMOKE_OUTPUT_ROOT = Path("/home/user/training_outputs/terminal/p3_adaptive_kl_terminal_dev_metrics_20260926_smoke")
GROUPS = (
    ("run1_dev_before_training", "run1/dev_before_training"),
    ("run0_dev_after_0080", "run0/dev_after_0080"),
    ("run1_dev_after_0240", "run1/dev_after_0240"),
    ("run2_dev_after_0240", "run2/dev_after_0240"),
)


class DevMetricsError(ValueError):
    pass


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DevMetricsError(f"JSON input missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DevMetricsError(f"JSON object required: {path}")
    return value


def _group_records(input_root: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    result: list[tuple[str, Path, dict[str, Any]]] = []
    for label, relative in GROUPS:
        manifest_path = input_root / relative / "manifest.json"
        manifest = _read_json(manifest_path)
        records = manifest.get("records")
        if not isinstance(records, list) or len(records) != 6:
            raise DevMetricsError(f"{label} expected 6 records: {manifest_path}")
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise DevMetricsError(f"{label} record {index} is not a mapping")
            result.append((label, manifest_path, dict(record)))
    return result


def _config_for_record(record: Mapping[str, Any], configs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    scenario = str(record.get("scenario"))
    if scenario not in configs:
        raise DevMetricsError(f"scenario config unavailable: {scenario}")
    config = configs[scenario]
    config_hash = _digest(vars(config))
    tape = Path(str(record.get("action_tape", "")))
    payload = _read_json(tape)
    tape_hash = str(payload.get("environment_config_hash", ""))
    if not tape_hash:
        raise DevMetricsError(f"tape has no environment_config_hash: {tape}")
    config_match = _environment_config_match(config, tape_hash)
    if not config_match[0] or config_hash != tape_hash:
        raise DevMetricsError(
            f"config binding mismatch for {tape}: config={config_hash} tape={tape_hash} match={config_match}"
        )
    return config, {
        "config_sha256": config_hash,
        "environment_config_hash": tape_hash,
        "config_compatibility": config_match[1],
        "config_snapshot": evaluator._config_snapshot(config),
    }


def _outcome_check(record: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "completed": int(record.get("normal", 0) or 0),
        "emergency_landed": int(record.get("emergency", 0) or 0),
        "failed_energy": int(record.get("failed_energy", 0) or 0),
    }
    other = record.get("other_failures") or {}
    if isinstance(other, Mapping):
        expected.update({str(key): int(value) for key, value in other.items()})
    observed = {str(key): int(value) for key, value in (metrics.get("outcomes") or {}).items()}
    expected = {key: value for key, value in expected.items() if value or key in observed}
    record_requests = int(record.get("requests", 0) or 0)
    record_unresolved = int(record.get("unresolved", 0) or 0)
    metrics_requested = int(metrics.get("requested_denominator", 0) or 0)
    metrics_missing = int(metrics.get("missing_request_count", 0) or 0)
    observed_rows = int(metrics.get("observed_terminal_count", len(metrics.get("rows") or [])) or 0)
    return {
        "expected_terminal_outcomes": dict(sorted(expected.items())),
        "observed_terminal_outcomes": dict(sorted(observed.items())),
        "terminal_counts_match": expected == observed,
        "record_requests": record_requests,
        "record_unresolved": record_unresolved,
        "metrics_requested_denominator": metrics_requested,
        "metrics_observed_terminal_count": observed_rows,
        "metrics_missing_request_count": metrics_missing,
        "requested_denominator_matches_record": metrics_requested == record_requests,
        "terminal_rows_plus_missing_matches_record": observed_rows + metrics_missing == record_requests,
        "missing_denominator_matches_record": metrics_missing == record_unresolved,
        "ledger_alignment": metrics.get("ledger_alignment") is True,
        "replay_exact": metrics.get("replay_exact") is True,
    }


def _record_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy outcome fields so group denominators survive per-tape errors."""
    fields = (
        "requests",
        "normal",
        "emergency",
        "failed_energy",
        "other_failures",
        "unresolved",
        "status",
        "teacher_enabled",
        "ppo_called",
        "weights_unchanged",
    )
    return {key: record.get(key) for key in fields if key in record}


def _validation_errors(item: Mapping[str, Any]) -> list[str]:
    if item.get("error") is not None:
        return ["processing_error"]
    check = item.get("outcome_check") or {}
    failures = []
    if check.get("terminal_counts_match") is not True:
        failures.append("terminal_counts_mismatch")
    if check.get("requested_denominator_matches_record") is not True:
        failures.append("requested_denominator_mismatch")
    if check.get("terminal_rows_plus_missing_matches_record") is not True:
        failures.append("terminal_row_denominator_mismatch")
    if check.get("missing_denominator_matches_record") is not True:
        failures.append("unresolved_denominator_mismatch")
    if check.get("ledger_alignment") is not True:
        failures.append("ledger_alignment_mismatch")
    if check.get("replay_exact") is not True:
        failures.append("replay_not_exact")
    return failures


def _group_summary(records: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    rows: list[Mapping[str, Any]] = []
    total_requests = 0
    missing_requests = 0
    totals = Counter()
    for record in records:
        total_requests += int(record.get("requests", 0) or 0)
        metrics = record.get("full_metrics") or {}
        rows.extend(row for row in metrics.get("rows", []) if isinstance(row, Mapping))
        if metrics:
            missing_requests += int(metrics.get("missing_request_count", 0) or 0)
        else:
            missing_requests += int(record.get("requests", 0) or 0)
        for key in ("normal", "emergency", "failed_energy", "unresolved"):
            totals[key] += int(record.get(key, 0) or 0)
        other = record.get("other_failures") or {}
        if isinstance(other, Mapping):
            totals.update({str(key): int(value) for key, value in other.items()})
    summary = summarize_terminal_rows(
        rows,
        low_soc_threshold=threshold,
        requested_count=total_requests,
        enforce_unique=False,
    )
    summary.update({
        "scope": "group_all_requested_terminal_rows",
        "episodes": len(records),
        "missing_request_count": missing_requests,
        "all_requests_observed": missing_requests == 0,
        "outcome_totals_from_records": dict(sorted(totals.items())),
    })
    return summary


def run(*, input_root: Path, output_root: Path, max_records: int | None = None) -> dict[str, Any]:
    if output_root.exists():
        raise DevMetricsError(f"refusing existing output root: {output_root}")
    protocol = evaluator._protocol()
    configs = evaluator._build_configs(protocol)
    source_records = _group_records(input_root)
    selected = source_records if max_records is None else source_records[: int(max_records)]
    if not selected:
        raise DevMetricsError("no records selected")
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "progress.json", {
        "status": "RUNNING",
        "expected_records": len(selected),
        "processed_records": 0,
    })
    processed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    try:
        for index, (group, manifest_path, record) in enumerate(selected):
            try:
                config, binding = _config_for_record(record, configs)
                tape = Path(str(record["action_tape"]))
                metrics = terminal_lifecycle_metrics(config, tape, .65)
                item = {
                    "group": group,
                    "manifest": str(manifest_path),
                    "seed": int(record["seed"]),
                    "scenario": str(record["scenario"]),
                    "action_tape": str(tape),
                    "binding": binding,
                    **_record_fields(record),
                    "full_metrics": metrics,
                    "outcome_check": _outcome_check(record, metrics),
                }
                item["validation_errors"] = _validation_errors(item)
                processed.append(item)
                _write_json(output_root / "records" / group / f"{record['scenario']}_{int(record['seed'])}.json", item)
            except Exception as exc:
                error = {
                    "index": index,
                    "group": group,
                    "manifest": str(manifest_path),
                    "seed": record.get("seed"),
                    "scenario": record.get("scenario"),
                    "action_tape": record.get("action_tape"),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
                errors.append(error)
                item = {
                    "group": group,
                    "manifest": str(manifest_path),
                    "seed": record.get("seed"),
                    "scenario": record.get("scenario"),
                    "action_tape": record.get("action_tape"),
                    **_record_fields(record),
                    "error": error["error"],
                    "validation_errors": ["processing_error"],
                }
                processed.append(item)
                _write_json(output_root / "errors" / f"error_{index:03d}.json", error)
                _write_json(output_root / "records" / group / f"{record.get('scenario')}_{int(record.get('seed', index))}.json", item)
            _write_json(output_root / "progress.json", {
                "status": "RUNNING",
                "expected_records": len(selected),
                "processed_records": len(processed),
                "error_records": len(errors),
                "validation_error_records": sum(bool(item.get("validation_errors")) for item in processed),
                "last_index": index,
                "last_group": group,
                "last_seed": record.get("seed"),
                "last_scenario": record.get("scenario"),
            })
        grouped = {}
        for group, _manifest, _record in selected:
            group_items = [item for item in processed if item["group"] == group]
            if group_items:
                grouped[group] = _group_summary(group_items, threshold=.65)
        validation_errors = [
            {
                "group": item.get("group"),
                "seed": item.get("seed"),
                "scenario": item.get("scenario"),
                "errors": list(item.get("validation_errors") or []),
            }
            for item in processed
            if item.get("validation_errors")
        ]
        result = {
            "stage": "P3_ADAPTIVE_KL_TERMINAL_DEV_METRICS_V7",
            "status": "COMPLETED" if not errors and not validation_errors and len(processed) == len(selected) else "FAILED",
            "input_root": str(input_root),
            "output": str(output_root),
            "expected_records": len(selected),
            "processed_records": len(processed),
            "error_records": len(errors),
            "validation_errors": validation_errors,
            "groups": grouped,
            "errors": errors,
        }
        _write_json(output_root / "manifest.json", result)
        _write_json(output_root / "progress.json", result)
        return result
    except BaseException:
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    result = run(input_root=args.input_root, output_root=args.output, max_records=args.max_records)
    print(json.dumps({key: result.get(key) for key in ("status", "output", "expected_records", "processed_records", "error_records")}, sort_keys=True))
    return 0 if result["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
