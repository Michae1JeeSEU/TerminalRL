"""Offline formal-coverage audit for the immutable P3 V2 training outputs.

The audit repairs only the reporting error in formal coverage.  It validates
the frozen root/formal manifests and final checkpoint, recomputes coverage
from lifecycle rows, and writes a separate audit manifest.  It never imports
the environment, resets a simulator, reserves seeds, or modifies an input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v2_formal_audit_v1.yaml"
)
# Filled after the protocol bytes are final and before the audit is used.
PROTOCOL_SHA = "fe2422c9af9ff6b3f0e234cb66987a0ff54a051c3975b60b8846e05a8f0d1ce6"
STAGE = "V4_2_P3_MIXED_PRIORITY_SOC_TRAINING_V2"
AUDIT_STAGE = "V4_2_P3_MIXED_PRIORITY_SOC_TRAINING_V2_FORMAL_AUDIT_V1"
FORMAL_STAGE = "P3_FORMAL"
TRAINING_ROOT_MANIFEST_SHA = (
    "27027da894f9caa03a429bc6a617bb84cec0370e0ffc22a4dbc27792ab946aeb"
)
FORMAL_MANIFEST_SHA = (
    "9a391c6e4d70cfe5c76ea5a802e1eddde48fa6efa57a3af0ecee88bc4b6cac38"
)
CHECKPOINT_8_SHA = (
    "bdff62ac94fbbd42ebb0fc571c09b6b2cd7b182058b0bf1a2ef496ab863ca9c6"
)
TRAINING_PROTOCOL_SHA = (
    "4a445d885f0e7662c9f1592f44464d970ece8322aa8a7cf81012d59e7fd916ed"
)
SCHEDULER_SOURCE = Path("terminal_marl/core/merge_scheduler.py")
SCHEDULER_SOURCE_SHA = (
    "58bf3f3a29b86f00855610000a819424e5c90b0c1f1433a780bcb42f86309ce1"
)
TRAINING_PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/p3_mixed_priority_soc_training_v2.yaml"
)


class AuditInputError(ValueError):
    """Raised when an immutable formal audit input violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditInputError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(path: str | Path) -> str:
    return _sha256(path)


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    _require(source.is_file(), f"JSON manifest missing: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditInputError(f"cannot read JSON manifest: {source}") from exc
    _require(isinstance(value, dict), f"JSON object required: {source}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def protocol() -> dict[str, Any]:
    """Read and verify the byte-frozen formal audit protocol."""
    _require(PROTOCOL.is_file(), f"formal audit protocol missing: {PROTOCOL}")
    _require(
        PROTOCOL_SHA != "TO_BE_FROZEN" and _sha(PROTOCOL) == PROTOCOL_SHA,
        "formal audit protocol changed after preregistration",
    )
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "formal audit protocol must be a mapping")
    _require(value.get("protocol") == AUDIT_STAGE, "formal audit protocol id changed")
    bindings = value.get("bindings", {})
    expected_bindings = {
        "training_stage": STAGE,
        "training_root_manifest_sha256": TRAINING_ROOT_MANIFEST_SHA,
        "formal_stage": FORMAL_STAGE,
        "formal_manifest_sha256": FORMAL_MANIFEST_SHA,
        "checkpoint_8_sha256": CHECKPOINT_8_SHA,
        "training_protocol_sha256": TRAINING_PROTOCOL_SHA,
        "scheduler_source": SCHEDULER_SOURCE.as_posix(),
        "scheduler_source_sha256": SCHEDULER_SOURCE_SHA,
    }
    for key, expected in expected_bindings.items():
        _require(bindings.get(key) == expected, f"formal audit binding changed: {key}")

    failure = value.get("formal_failure_contract", {})
    _require(failure.get("root_pass") is False, "root PASS contract changed")
    _require(
        failure.get("root_heldout_evaluation_allowed") is False,
        "root heldout contract changed",
    )
    _require(failure.get("formal_pass") is False, "formal PASS contract changed")
    _require(
        list(failure.get("required_true_gates", [])) == ["all_episodes", "all_episode_gates"],
        "formal required true gates changed",
    )
    _require(
        list(failure.get("allowed_false_gates", []))
        == ["priority_coverage", "soc_in_bounds", "soc_range_observed"],
        "formal allowed false gates changed",
    )
    _require(failure.get("record_count") == 8, "formal record count changed")
    _require(failure.get("record_episode_gates_all_true") is True,
             "formal episode gate contract changed")
    _require(failure.get("record_attributes_replay_exact") is True,
             "formal replay contract changed")

    coverage = value.get("coverage", {})
    _require(list(coverage.get("required_priorities", [])) == [1, 3, 5],
             "required priority coverage changed")
    _require(float(coverage.get("initial_soc_min")) == 0.55,
             "initial SOC lower bound changed")
    _require(float(coverage.get("initial_soc_max")) == 0.90,
             "initial SOC upper bound changed")
    _require(float(coverage.get("observed_soc_min_at_most")) == 0.65,
             "observed SOC minimum threshold changed")
    _require(float(coverage.get("observed_soc_max_at_least")) == 0.80,
             "observed SOC maximum threshold changed")
    authorization = value.get("authorization", {})
    _require(authorization.get("heldout_untouched") is True,
             "heldout immutability contract changed")
    _require(authorization.get("P4_ALLOWED") is False,
             "P4 authorization contract changed")
    _require(authorization.get("checkpoint_promotable") is False,
             "checkpoint promotion contract changed")
    return value


def _resolve_artifact_path(raw: str | Path, manifest_path: Path) -> Path:
    """Resolve an absolute server path against a synced manifest directory."""
    path = Path(str(raw))
    if path.is_file():
        return path
    fallback = manifest_path.parent / path.name
    if fallback.is_file():
        return fallback
    return path


def _same_path(raw: str | Path, expected: Path, manifest_path: Path) -> bool:
    resolved = _resolve_artifact_path(raw, manifest_path)
    if resolved.is_file() and expected.is_file():
        return resolved.resolve() == expected.resolve()
    return Path(str(raw)).as_posix() == expected.as_posix()


def _finite_number(value: Any, label: str) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{label} is not numeric",
    )
    number = float(value)
    _require(math.isfinite(number), f"{label} is not finite")
    return number


def _integer_number(value: Any, label: str) -> int:
    number = _finite_number(value, label)
    _require(number.is_integer(), f"{label} is not integral")
    return int(number)


def _gate_summary(gates: Any, *, label: str, p: Mapping[str, Any]) -> dict[str, bool]:
    _require(isinstance(gates, Mapping), f"{label} gates are missing")
    contract = p["formal_failure_contract"]
    required_true = list(contract["required_true_gates"])
    allowed_false = list(contract["allowed_false_gates"])
    expected_keys = set(required_true + allowed_false)
    _require(set(gates) == expected_keys, f"{label} contains an unexpected gate")
    _require(all(isinstance(value, bool) for value in gates.values()),
             f"{label} gate values are not boolean")
    normalized = {str(key): bool(value) for key, value in gates.items()}
    _require(all(normalized[key] for key in required_true),
             f"{label} required true gate failed")
    _require(all(not normalized[key] for key in allowed_false),
             f"{label} original coverage failure changed")
    return normalized


def validate_root_manifest(path: Path, p: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(_sha(path) == TRAINING_ROOT_MANIFEST_SHA,
             "training root manifest SHA mismatch")
    root = _read_json(path)
    _require(root.get("stage") == p["bindings"]["training_stage"],
             "training root stage changed")
    _require(root.get("PASS") is False, "training root unexpectedly passed")
    _require(root.get("HELDOUT_EVALUATION_ALLOWED") is False,
             "training root unexpectedly authorizes heldout")
    _require(root.get("heldout_untouched") is True,
             "training root heldout flag changed")
    _require(root.get("P4_ALLOWED") is False, "training root unexpectedly authorizes P4")
    _require(root.get("checkpoint_promotable") is False,
             "training root unexpectedly promotes checkpoint")
    formal = root.get("formal")
    _require(isinstance(formal, Mapping), "training root formal summary missing")
    _require(formal.get("PASS") is False, "training root formal summary unexpectedly passed")
    _gate_summary(formal.get("gates"), label="training root formal", p=p)
    return root, dict(formal)


def _validate_row(
    row: Any,
    *,
    index: str,
    p: Mapping[str, Any],
) -> dict[str, Any]:
    _require(isinstance(row, Mapping), f"lifecycle row {index} is not a mapping")
    required = (
        "uid",
        "base_priority",
        "initial_soc",
        "landing_soc",
        "wait_seconds",
        "request_to_touchdown_seconds",
    )
    for field in required:
        _require(field in row, f"lifecycle row {index} is missing {field}")
    uid = _integer_number(row["uid"], f"lifecycle row {index} uid")
    _require(uid >= 0, f"lifecycle row {index} uid is negative")
    priority = _integer_number(row["base_priority"], f"lifecycle row {index} priority")
    required_priorities = [int(value) for value in p["coverage"]["required_priorities"]]
    _require(priority in required_priorities,
             f"lifecycle row {index} priority is outside the frozen set")
    coverage = p["coverage"]
    initial_soc = _finite_number(row["initial_soc"], f"lifecycle row {index} initial SOC")
    lower = float(coverage["initial_soc_min"])
    upper = float(coverage["initial_soc_max"])
    _require(lower <= initial_soc <= upper,
             f"lifecycle row {index} initial SOC is outside bounds")
    _finite_number(row["landing_soc"], f"lifecycle row {index} landing SOC")
    wait = _finite_number(row["wait_seconds"], f"lifecycle row {index} wait")
    request_to_touchdown = _finite_number(
        row["request_to_touchdown_seconds"],
        f"lifecycle row {index} request-to-touchdown",
    )
    _require(wait >= 0.0, f"lifecycle row {index} wait is negative")
    _require(request_to_touchdown >= 0.0,
             f"lifecycle row {index} request-to-touchdown is negative")
    return {
        "uid": uid,
        "base_priority": priority,
        "initial_soc": initial_soc,
        "landing_soc": float(row["landing_soc"]),
        "wait_seconds": wait,
        "request_to_touchdown_seconds": request_to_touchdown,
    }


def _validate_formal_records(
    formal: Mapping[str, Any], p: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _require(formal.get("stage") == p["bindings"]["formal_stage"],
             "formal manifest stage changed")
    _require(formal.get("PASS") is False, "formal manifest unexpectedly passed")
    _gate_summary(formal.get("gates"), label="formal manifest", p=p)
    records = formal.get("records")
    _require(isinstance(records, list) and len(records) == 8,
             "formal manifest does not contain exactly eight records")
    normalized_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        _require(isinstance(record, Mapping), f"formal record {record_index} is not a mapping")
        gates = record.get("gates")
        _require(isinstance(gates, Mapping) and gates,
                 f"formal record {record_index} gates are missing")
        _require(all(isinstance(value, bool) for value in gates.values()),
                 f"formal record {record_index} gate values are not boolean")
        _require(all(bool(value) for value in gates.values()),
                 f"formal record {record_index} contains a failed episode gate")
        attributes = record.get("attributes")
        _require(isinstance(attributes, Mapping),
                 f"formal record {record_index} attributes are missing")
        _require(attributes.get("replay_exact") is True,
                 f"formal record {record_index} lifecycle replay is not exact")
        raw_rows = attributes.get("rows")
        _require(isinstance(raw_rows, list) and raw_rows,
                 f"formal record {record_index} lifecycle rows are missing")
        normalized_rows = [
            _validate_row(row, index=f"{record_index}:{row_index}", p=p)
            for row_index, row in enumerate(raw_rows)
        ]
        if "completed" in attributes:
            _require(_integer_number(attributes["completed"],
                                     f"formal record {record_index} completed")
                     == len(normalized_rows),
                     f"formal record {record_index} completed count changed")
        rows.extend(normalized_rows)
        normalized_records.append({
            "seed": record.get("seed"),
            "scenario": record.get("scenario"),
            "gates": {str(key): bool(value) for key, value in gates.items()},
            "attributes_replay_exact": True,
            "row_count": len(normalized_rows),
            "checkpoint": record.get("checkpoint"),
        })
    return normalized_records, rows


def recompute_coverage(rows: list[dict[str, Any]], p: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute coverage solely from validated lifecycle rows and bindings."""
    _require(isinstance(rows, list) and rows, "no lifecycle rows available for coverage")
    coverage = p["coverage"]
    required = [int(value) for value in coverage["required_priorities"]]
    lower = float(coverage["initial_soc_min"])
    upper = float(coverage["initial_soc_max"])
    observed_min_threshold = float(coverage["observed_soc_min_at_most"])
    observed_max_threshold = float(coverage["observed_soc_max_at_least"])
    priorities = sorted({int(row["base_priority"]) for row in rows})
    soc = [float(row["initial_soc"]) for row in rows]
    observed_min = min(soc)
    observed_max = max(soc)
    return {
        "rows_count": len(rows),
        "priorities_observed": priorities,
        "required_priorities": required,
        "priority_coverage": priorities == required,
        "initial_soc_min": observed_min,
        "initial_soc_max": observed_max,
        "initial_soc_bounds": [lower, upper],
        "soc_in_bounds": all(lower <= value <= upper for value in soc),
        "observed_soc_thresholds": {
            "min_at_most": observed_min_threshold,
            "max_at_least": observed_max_threshold,
        },
        "soc_range_observed": (
            observed_min <= observed_min_threshold
            and observed_max >= observed_max_threshold
        ),
    }


def _validate_final_checkpoint(
    root: Mapping[str, Any],
    root_formal: Mapping[str, Any],
    formal: Mapping[str, Any],
    records: list[dict[str, Any]],
    checkpoint_path: Path,
    formal_manifest_path: Path,
    p: Mapping[str, Any],
) -> dict[str, Any]:
    _require(checkpoint_path.is_file(), f"checkpoint 8 missing: {checkpoint_path}")
    checkpoint_sha = _sha(checkpoint_path)
    _require(checkpoint_sha == p["bindings"]["checkpoint_8_sha256"],
             "checkpoint 8 SHA mismatch")
    final_path = formal.get("final_checkpoint")
    final_sha = formal.get("final_checkpoint_sha256")
    _require(isinstance(final_path, str) and final_path,
             "formal final checkpoint path missing")
    _require(final_sha == p["bindings"]["checkpoint_8_sha256"],
             "formal final checkpoint SHA binding changed")
    _require(_same_path(final_path, checkpoint_path, formal_manifest_path),
             "formal final checkpoint path changed")
    root_final_path = root_formal.get("final_checkpoint")
    root_final_sha = root_formal.get("final_checkpoint_sha256")
    _require(isinstance(root_final_path, str) and root_final_path,
             "root formal final checkpoint path missing")
    _require(root_final_sha == p["bindings"]["checkpoint_8_sha256"],
             "root formal final checkpoint SHA binding changed")
    _require(_same_path(root_final_path, checkpoint_path, formal_manifest_path),
             "root formal final checkpoint path changed")
    last_checkpoint = records[-1].get("checkpoint")
    _require(isinstance(last_checkpoint, str) and last_checkpoint,
             "formal record 8 checkpoint path missing")
    _require(_same_path(last_checkpoint, checkpoint_path, formal_manifest_path),
             "formal record 8 checkpoint path changed")
    return {
        "path": str(checkpoint_path),
        "sha256": checkpoint_sha,
        "formal_manifest_path": final_path,
        "root_manifest_path": root_final_path,
    }


def run(
    *,
    training_root_manifest: Path,
    formal_manifest: Path,
    checkpoint_8: Path,
    output: Path,
) -> dict[str, Any]:
    """Audit immutable P3 V2 formal outputs and write a separate manifest."""
    p = protocol()
    training_root_manifest = Path(training_root_manifest)
    formal_manifest = Path(formal_manifest)
    checkpoint_8 = Path(checkpoint_8)
    _require(_sha(training_root_manifest) == TRAINING_ROOT_MANIFEST_SHA,
             "training root manifest SHA mismatch")
    _require(_sha(formal_manifest) == FORMAL_MANIFEST_SHA,
             "formal manifest SHA mismatch")
    _require(_sha(TRAINING_PROTOCOL) == TRAINING_PROTOCOL_SHA,
             "training protocol SHA mismatch")
    _require(_sha(SCHEDULER_SOURCE) == SCHEDULER_SOURCE_SHA,
             "scheduler source SHA mismatch")
    root, root_formal = validate_root_manifest(training_root_manifest, p)
    formal_value = _read_json(formal_manifest)
    normalized_records, rows = _validate_formal_records(formal_value, p)
    corrected_coverage = recompute_coverage(rows, p)
    checkpoint = _validate_final_checkpoint(
        root,
        root_formal,
        formal_value,
        normalized_records,
        checkpoint_8,
        formal_manifest,
        p,
    )
    passed = bool(
        corrected_coverage["priority_coverage"]
        and corrected_coverage["soc_in_bounds"]
        and corrected_coverage["soc_range_observed"]
    )
    result = {
        "stage": AUDIT_STAGE,
        "PASS": passed,
        "HELDOUT_EVALUATION_ALLOWED": passed,
        "heldout_untouched": True,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
        "protocol_sha256": PROTOCOL_SHA,
        "training_protocol_sha256": TRAINING_PROTOCOL_SHA,
        "scheduler_source": SCHEDULER_SOURCE.as_posix(),
        "scheduler_source_sha256": SCHEDULER_SOURCE_SHA,
        "original_failure_evidence": {
            "training_root_manifest": str(training_root_manifest),
            "training_root_manifest_sha256": TRAINING_ROOT_MANIFEST_SHA,
            "formal_manifest": str(formal_manifest),
            "formal_manifest_sha256": FORMAL_MANIFEST_SHA,
            "root_stage": root.get("stage"),
            "root_PASS": root.get("PASS"),
            "root_HELDOUT_EVALUATION_ALLOWED": root.get("HELDOUT_EVALUATION_ALLOWED"),
            "formal_stage": formal_value.get("stage"),
            "formal_PASS": formal_value.get("PASS"),
            "formal_gates": formal_value.get("gates"),
        },
        "corrected_coverage": corrected_coverage,
        "formal_records": normalized_records,
        "final_checkpoint": checkpoint,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / str(p["output"]["manifest_name"]), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-root-manifest",
        "--training-manifest",
        dest="training_root_manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        training_root_manifest=args.training_root_manifest,
        formal_manifest=args.formal_manifest,
        checkpoint_8=args.checkpoint_8,
        output=args.output,
    )
    print(json.dumps({
        "stage": result["stage"],
        "PASS": result["PASS"],
        "HELDOUT_EVALUATION_ALLOWED": result["HELDOUT_EVALUATION_ALLOWED"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
