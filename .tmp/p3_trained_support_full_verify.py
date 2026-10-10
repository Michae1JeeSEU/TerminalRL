"""Mechanical verifier for the trained proposal-support full artifact.

This utility reads only the registered summary JSON files.  It never creates an
environment and does not interpret the business comparison as a scientific
acceptance decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


class VerificationError(RuntimeError):
    """Raised when the registered artifact set is incomplete or malformed."""


ARMS = ("before_legacy", "after_legacy", "after_support")
CELLS = (
    {"cell_id": 1, "scenario": "dense24", "env_seed": 875101, "policy_seed": 875201, "split": "development_original_dense24"},
    {"cell_id": 2, "scenario": "dense24", "env_seed": 875102, "policy_seed": 875202, "split": "development_original_dense24"},
    {"cell_id": 3, "scenario": "dense24", "env_seed": 875103, "policy_seed": 875203, "split": "development_original_dense24"},
    {"cell_id": 4, "scenario": "dense24", "env_seed": 875104, "policy_seed": 875204, "split": "development_original_dense24"},
)
CHECKPOINTS = {
    "before_legacy": "3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab",
    "after_legacy": "d44990dc7d129573d9145f169df1cbc4f91fa32bc5127b1c58d937260c02a24f",
    "after_support": "d44990dc7d129573d9145f169df1cbc4f91fa32bc5127b1c58d937260c02a24f",
}
SAFETY_FIELDS = (
    "collisions", "nmac", "hard_separation_violations",
    "uncertified_execution", "SafetyInvariantError",
)
FULL_METRIC_FIELDS = (
    "successful", "resolved", "wait_seconds", "request_elapsed_total_seconds",
    "energy_total_consumed", "reward_total", "raw_reward_excluding_rejection_penalty",
    "release_rejection_penalty",
)
AGGREGATE_FIELDS = (
    "generated", "admitted", "normal", "emergency", "failed", "diverted",
    "unresolved", "successful", "resolved", "wait_seconds",
    "request_elapsed_total_seconds", "energy_total_consumed", "reward_total",
    "raw_reward_excluding_rejection_penalty", "release_rejection_penalty",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationError(f"required artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON artifact: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same(left: Any, right: Any, *, atol: float = 1e-8) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        try:
            return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= atol
        except (TypeError, ValueError):
            return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_same(left[key], right[key], atol=atol) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same(a, b, atol=atol) for a, b in zip(left, right))
    return left == right


def _cell_key(cell: Mapping[str, Any]) -> tuple[int, int, int]:
    return (int(cell.get("cell_id", -1)), int(cell.get("env_seed", -1)), int(cell.get("policy_seed", -1)))


def _expected_cell(cell_id: int) -> dict[str, Any]:
    return dict(CELLS[cell_id - 1])


def _metric(arm: Mapping[str, Any]) -> Mapping[str, Any]:
    value = arm.get("metrics")
    return value if isinstance(value, Mapping) else {}


def _aggregate(pair_summaries: list[Mapping[str, Any]], arm_name: str) -> dict[str, Any]:
    metrics = [_metric((row.get("arms") or {}).get(arm_name, {})) for row in pair_summaries]
    known = bool(metrics) and all(
        item.get("complete_scene_metrics_known") is True
        and all(item.get(key) is not None for key in AGGREGATE_FIELDS)
        for item in metrics
    )
    result: dict[str, Any] = {"known": known}
    for key in AGGREGATE_FIELDS:
        result[key] = None if not known else sum(
            float(item[key]) if isinstance(item[key], float) else int(item[key])
            for item in metrics
        )
    return result


def _delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("normal", "successful", "failed", "wait_seconds", "request_elapsed_total_seconds", "energy_total_consumed")
    known = bool(left.get("known") and right.get("known"))
    values = {key: None if not known else float(left[key]) - float(right[key]) for key in keys}
    return {
        "known": known,
        "values": values,
        "normal_not_lower": bool(known and values["normal"] >= 0),
        "successful_not_lower": bool(known and values["successful"] >= 0),
        "wait_not_higher": bool(known and values["wait_seconds"] <= 0),
        "strict_business_gain": bool(known and any(
            values[key] > 0 if key in {"normal", "successful"} else values[key] < 0
            for key in keys
        )),
    }


def _file_inventory(root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    relative = {
        "full": Path("full.json"),
        "preflight": Path("preflight.json"),
        "pilot": Path("pilot.json"),
    }
    for cell in CELLS:
        cell_id = int(cell["cell_id"])
        relative[f"pair_{cell_id}"] = Path("pairs") / f"cell_{cell_id:02d}.json"
        for arm in ARMS:
            relative[f"arm_{cell_id}_{arm}"] = Path("arms") / f"cell_{cell_id:02d}_{arm}.json"
    paths = {name: root / rel for name, rel in relative.items()}
    loaded = {name: _read_json(path) for name, path in paths.items()}
    return loaded, paths


def verify(root: Path) -> dict[str, Any]:
    loaded, paths = _file_inventory(root)
    full = loaded["full"]
    pilot = loaded["pilot"]
    preflight = loaded["preflight"]
    pairs = [loaded[f"pair_{cell['cell_id']}"] for cell in CELLS]
    full_pairs = full.get("pair_summaries")
    pair_list_shape = isinstance(full_pairs, list) and len(full_pairs) == len(CELLS)
    if not isinstance(full_pairs, list):
        full_pairs = []

    pair_checks: list[dict[str, Any]] = []
    for cell, pair in zip(CELLS, pairs):
        pair_cell = pair.get("cell") if isinstance(pair.get("cell"), Mapping) else {}
        full_match = any(
            _cell_key(row.get("cell", {})) == _cell_key(cell)
            and _same(row, pair)
            for row in full_pairs
            if isinstance(row, Mapping)
        )
        expected_cell_ok = _same(pair_cell, cell)
        arm_results: dict[str, Any] = {}
        identity_values: list[Any] = []
        for arm in ARMS:
            arm_payload = loaded[f"arm_{cell['cell_id']}_{arm}"]
            nested = (pair.get("arms") or {}).get(arm)
            nested = nested if isinstance(nested, Mapping) else {}
            arm_metrics = _metric(arm_payload)
            nested_metrics = _metric(nested)
            metrics_match = _same(arm_metrics, nested_metrics)
            safety = arm_metrics.get("metrics", {}).get("safety", {}) if isinstance(arm_metrics.get("metrics"), Mapping) else {}
            safety_known_zero = all(
                key in safety and isinstance(safety[key], (int, float)) and not isinstance(safety[key], bool)
                and math.isfinite(float(safety[key])) and float(safety[key]) == 0.0
                for key in SAFETY_FIELDS
            )
            lifecycle = arm_metrics.get("metrics", {}).get("lifecycle", {}) if isinstance(arm_metrics.get("metrics"), Mapping) else {}
            runtime = arm_metrics.get("metrics", {}).get("runtime_capture", {}) if isinstance(arm_metrics.get("metrics"), Mapping) else {}
            identity = arm_metrics.get("request_identity")
            identity_values.append(identity)
            identity_ok = bool(
                arm_metrics.get("request_identity_exact_24") is True
                and arm_metrics.get("request_identity_ids") == list(range(24))
                and isinstance(identity, list) and len(identity) == 24
                and [row.get("arrival_id") for row in identity if isinstance(row, Mapping)] == list(range(24))
            )
            status = arm_payload.get("status")
            error = arm_payload.get("error") or arm_metrics.get("error")
            full_unknown_on_partial = bool(
                (arm_metrics.get("censored") is True or error is not None)
                and all(arm_metrics.get(key) is None for key in FULL_METRIC_FIELDS)
            )
            arm_results[arm] = {
                "status": status,
                "natural": arm_metrics.get("complete_scene_metrics_known") is True,
                "censored": arm_metrics.get("censored") is True,
                "error": error,
                "metrics_match_group": metrics_match,
                "identity_24": identity_ok,
                "safety5_known_zero": safety_known_zero,
                "source_model_rng_weights": {
                    "model_unchanged": arm_payload.get("model_unchanged") is True,
                    "caller_rng_unchanged": arm_payload.get("caller_rng_unchanged") is True,
                    "weights_unchanged": arm_payload.get("weights_unchanged") is True,
                },
                "ledger_replay_energy": {
                    "replay_exact": lifecycle.get("replay_exact") is True,
                    "ledger_alignment": runtime.get("ledger_alignment") is True,
                    "energy_match": runtime.get("energy_match") is True,
                },
                "full_metrics_none_on_partial": full_unknown_on_partial,
                "checkpoint_sha256": arm_payload.get("checkpoint_sha256"),
                "config_matches_preflight": _same(arm_payload.get("config"), (preflight.get("runtime") or {}).get("config_snapshot")),
            }
        identities_equal = bool(identity_values) and all(_same(identity_values[0], value) for value in identity_values[1:])
        pair_checks.append({
            "cell": cell,
            "pair_file_cell_exact": expected_cell_ok,
            "full_pair_exact": full_match,
            "arms_exact": set((pair.get("arms") or {})) == set(ARMS),
            "request_identity_equal_within_pair": identities_equal,
            "arm_checks": arm_results,
            "status": pair.get("status"),
            "new_issue_ids": {
                key: pair.get(key) for key in ("new_failed_arrival_ids", "new_diverted_arrival_ids", "new_unresolved_arrival_ids")
            },
        })

    computed_aggregates = {arm: _aggregate(pairs, arm) for arm in ARMS}
    reported_review = full.get("business_review") if isinstance(full.get("business_review"), Mapping) else {}
    aggregate_matches = {
        arm: _same(computed_aggregates[arm], reported_review.get(f"{arm}_aggregate"))
        for arm in ARMS
    }
    computed_deltas = {
        "before_to_after_legacy": _delta(computed_aggregates["after_legacy"], computed_aggregates["before_legacy"]),
        "support_to_after_legacy": _delta(computed_aggregates["after_support"], computed_aggregates["after_legacy"]),
    }
    reported_deltas = {
        key: _same(value, reported_review.get(key))
        for key, value in computed_deltas.items()
    }
    pilot_artifacts = pilot.get("pilot_artifacts") if isinstance(pilot.get("pilot_artifacts"), Mapping) else {}
    pilot_checks = {
        "pilot_status_present": pilot.get("status") in {"PILOT_OK", "PILOT_INCOMPLETE"},
        "pilot_preflight_sha_match": pilot.get("preflight_sha256") == _sha256(paths["preflight"]),
        "pilot_runner_binding_match": pilot.get("preflight_runner_sha256") == preflight.get("runner_sha256"),
        "pilot_pair_sha_match": pilot_artifacts.get("pair_sha256") == _sha256(paths["pair_1"]),
        "pilot_arm_sha_match": {
            arm: pilot_artifacts.get("arm_sha256", {}).get(arm) == _sha256(paths[f"arm_1_{arm}"])
            for arm in ARMS
        },
        "pilot_pair_matches_cell1": _same(pilot.get("pair"), pairs[0]),
    }
    source_inventory = {
        str(path.relative_to(root)): _sha256(path)
        for path in paths.values()
    }
    result = {
        "status": "MECHANICAL_OBSERVATIONS",
        "scientific_acceptance_decision": None,
        "full_status": full.get("status"),
        "pair_count": len(full_pairs),
        "pair_list_shape_exact4": pair_list_shape,
        "pair_checks": pair_checks,
        "computed_aggregates": computed_aggregates,
        "reported_aggregate_matches": aggregate_matches,
        "computed_deltas": computed_deltas,
        "reported_delta_matches": reported_deltas,
        "pilot_checks": pilot_checks,
        "source_file_sha256": source_inventory,
        "preflight_bindings_observed": {
            "runner_sha256": preflight.get("runner_sha256"),
            "protocol_sha256": preflight.get("protocol_sha256"),
            "runtime": preflight.get("runtime"),
            "checkpoints": preflight.get("checkpoints"),
        },
    }
    mechanical_ok = bool(
        full.get("status") in {"FULL_OK", "FULL_TECHNICAL_ERROR"}
        and pair_list_shape
        and all(
            row["pair_file_cell_exact"] and row["full_pair_exact"] and row["arms_exact"]
            and row["request_identity_equal_within_pair"]
            and all(item["metrics_match_group"] for item in row["arm_checks"].values())
            for row in pair_checks
        )
        and all(aggregate_matches.values())
        and all(reported_deltas.values())
        and all(pilot_checks["pilot_arm_sha_match"].values())
    )
    result["mechanical_checks_complete"] = mechanical_ok
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(r"D:/7_codespace/training_outputs/terminal/p3_followup_20260930/trained_proposal_support_20261001"),
    )
    parser.add_argument("--proof", type=Path, default=Path(".tmp/p3_trained_support_full_proof.json"))
    args = parser.parse_args(argv)
    try:
        result = verify(args.output_root)
    except VerificationError as exc:
        print(json.dumps({"status": "MISSING_OR_INVALID_INPUT", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    args.proof.parent.mkdir(parents=True, exist_ok=True)
    args.proof.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "mechanical_checks_complete": result["mechanical_checks_complete"], "proof": str(args.proof)}))
    return 0 if result["mechanical_checks_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
