"""P2 gate V5 development-only support supplement.

The V4 combined support artifacts are validated and retained read-only.  This
runner collects only three fresh dev episodes under the same calibrated
candidate and unchanged V2 support gates; train rows are never recollected.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from terminal_marl.training import collect_v4_2_p2_gate_support_v3 as v3
from terminal_marl.training import collect_v4_2_p2_gate_support_v4_supplement as v4


PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/ppo_p2_gate_support_collection_v5_dev_supplement.yaml"
)
PROTOCOL_SHA = "2947aa95f67c9b87b1dfb0ec3d12ba7b9fdbfa5c85280527eb770475453ee314"
STAGE = "V4_2_P2_GATE_SUPPORT_COLLECTION_V5_DEV_SUPPLEMENT"
CANDIDATE_SHA = "61e03e02e3fb204d2db4e814097543905f51e1a3772626fdac1d7bc02ab8ca6b"
CANDIDATE_ARCHITECTURE_ID = "v4_2_hierarchical_gate_temperature_calibrated_residual_adapter_v3"
CANDIDATE_TEMPERATURE = 2.5

V4_MANIFEST_SHA = "d0efbc0bfd2cd2f8c0d3a35898ae850c6cee2f133d987b2bd46b2aa1eba82f93"
V4_COMBINED_TRAIN_SHA = "40002db70c2e0d5d181b53b2f23290c65954f6bbe6282be1066b6f558e6153d4"
V4_COMBINED_DEV_SHA = "2261ef53377706a6ec6e3a735cb0ab4be300663979226ff7eebc97ee7f36fea0"
V4_PROTOCOL_SHA = "d626690c054f61b40c2740ee7fe05bf80e4e2f5c2f8d291f229291897538b8c3"
V4_CONSUMED_MARKER_SHA = "dd969ead1957085667f3ffcf5deb606a33b0d763f86228a9dd0211425ace4eca"
V4_RESERVATION_MARKER = v4.RESERVATION_MARKER
RESERVATION_MARKER = "p2_gate_support_collection_v5_dev_supplement_consumed.json"

V3_PROTOCOL_SHA = v4.V3_PROTOCOL_SHA
V3_CONSUMED_MARKER_SHA = v4.V3_CONSUMED_MARKER_SHA
V3_RESERVATION_MARKER = v4.V3_RESERVATION_MARKER
RETAINED_V3_TRAIN_SEEDS = v4.RETAINED_V3_TRAIN_SEEDS
RETAINED_V3_DEV_SEEDS = v4.RETAINED_V3_DEV_SEEDS
FUTURE_QUALIFICATION_SEEDS = v4.FUTURE_QUALIFICATION_SEEDS
RETAINED_V4_TRAIN_SEEDS = tuple(v4.RETAINED_V3_TRAIN_SEEDS) + tuple(v4.FORMAL_TRAIN_SEEDS)
RETAINED_V4_DEV_SEEDS = tuple(v4.RETAINED_V3_DEV_SEEDS) + tuple(v4.FORMAL_DEV_SEEDS)
FORMAL_DEV_SEEDS = tuple(range(785001, 785004))
SMOKE_SEEDS = (785000,)
SMOKE_SCENARIOS = ("flow6",)
SMOKE_STEPS = 160
SCENARIOS = v4.SCENARIOS
INTERFACES = v4.INTERFACES
SHIELD_ACTOR_FRACTION_MAX = v4.SHIELD_ACTOR_FRACTION_MAX
V4_TRAIN_ROWS = 3786
V4_DEV_ROWS = 1787
FORMAL_AUDIT_SEEDS = set(range(785000, 785004))
FORMAL_DEV_SEED_SET = set(FORMAL_DEV_SEEDS)

_sha = v4._sha
_write = v4._write
_require = v4._require
reserve_once = v4.reserve_once
historical_registry = v4.historical_registry
support_config = v4.support_config
collection_config = support_config
diagnostic_config = support_config
_episode_record = v4._episode_record
filter_actor_rows = v4.filter_actor_rows
_actor_rows = v4._actor_rows
_finite_88d = v4._finite_88d
_thresholds = v4._thresholds
split_metrics = v4.split_metrics
split_gates = v4.split_gates
support_gates = v4.support_gates
formal_authorization = v4.formal_authorization


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), f"support rows missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid support JSONL at {path}:{line_number}") from exc
        _require(isinstance(value, dict), f"support row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def protocol() -> dict[str, Any]:
    """Load and verify the byte-frozen V5 development supplement protocol."""
    _require(PROTOCOL.is_file(), f"V5 dev supplement protocol missing: {PROTOCOL}")
    if PROTOCOL_SHA:
        _require(_sha(PROTOCOL) == PROTOCOL_SHA,
                 "P2 V5 dev supplement protocol changed after preregistration")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "V5 dev supplement protocol must be a mapping")
    _require(value.get("protocol") == STAGE, "V5 dev supplement protocol id changed")
    _require(value.get("candidate_checkpoint_sha256") == CANDIDATE_SHA,
             "V5 candidate checkpoint binding changed")
    _require(value.get("candidate_architecture_id") == CANDIDATE_ARCHITECTURE_ID,
             "V5 candidate architecture binding changed")
    _require(float(value.get("candidate_temperature")) == CANDIDATE_TEMPERATURE,
             "V5 candidate temperature binding changed")

    predecessors = value.get("predecessors", {})
    for key, expected in {
        "v4_manifest_sha256": V4_MANIFEST_SHA,
        "v4_combined_train_sha256": V4_COMBINED_TRAIN_SHA,
        "v4_combined_dev_sha256": V4_COMBINED_DEV_SHA,
        "v4_protocol_sha256": V4_PROTOCOL_SHA,
        "v4_consumed_marker_sha256": V4_CONSUMED_MARKER_SHA,
    }.items():
        _require(predecessors.get(key) == expected,
                 f"V5 predecessor binding changed: {key}")

    formal = value.get("formal", {})
    for key, expected in {
        "retained_v4_train_seeds": RETAINED_V4_TRAIN_SEEDS,
        "retained_v4_dev_seeds": RETAINED_V4_DEV_SEEDS,
        "supplemental_dev_seeds": FORMAL_DEV_SEEDS,
        "future_qualification_seeds": FUTURE_QUALIFICATION_SEEDS,
        "scenarios": SCENARIOS,
    }.items():
        _require(tuple(formal.get(key, ())) == tuple(expected),
                 f"V5 formal binding changed: {key}")
    for key, expected in {
        "scenario_assignment": "cycle_in_list_order_per_split",
        "teacher_mode": "teacher_label_only",
        "scheduler_mode": "execution",
        "retained_v3_marker": V3_RESERVATION_MARKER,
        "retained_v4_marker": V4_RESERVATION_MARKER,
        "consumed_marker": RESERVATION_MARKER,
        "qualification_seed_range_audited": "785000-785003",
        "first_reset_requires": "historical_registry_and_dev_supplement_reservation",
    }.items():
        _require(formal.get(key) == expected, f"V5 formal binding changed: {key}")
    _require(formal.get("deterministic") is False, "V5 collection is not stochastic")
    _require(formal.get("teacher_release_label_enabled") is False,
             "V5 Teacher release-label binding changed")
    _require(formal.get("future_qualification_rollouts") == 0
             and formal.get("future_qualification_markers_mutated") is False,
             "V5 future qualification boundary changed")

    smoke = value.get("smoke", {})
    _require(tuple(smoke.get("seeds", ())) == SMOKE_SEEDS, "V5 smoke seed changed")
    _require(tuple(smoke.get("scenarios", ())) == SMOKE_SCENARIOS,
             "V5 smoke scenario changed")
    _require(smoke.get("steps") == SMOKE_STEPS, "V5 smoke horizon changed")
    _require(smoke.get("reserve_formal_seeds") is False
             and smoke.get("create_consumed_marker") is False
             and smoke.get("authorization") is False,
             "V5 smoke authorization changed")

    bindings = value.get("bindings", {})
    for key, expected in {
        "v4_manifest_sha256": V4_MANIFEST_SHA,
        "v4_combined_train_sha256": V4_COMBINED_TRAIN_SHA,
        "v4_combined_dev_sha256": V4_COMBINED_DEV_SHA,
        "v4_consumed_marker_sha256": V4_CONSUMED_MARKER_SHA,
        "v3_consumed_marker_read_only": True,
        "observation_schema": "phaseA_low_obs_v2_1_retry_state",
        "observation_dimension": 88,
        "action_schema": "low_action_schema_v4_2_hierarchical",
        "source_usage": "v4_rows_retained_dev_rows_new_no_v2_or_qualification_rows",
    }.items():
        _require(bindings.get(key) == expected, f"V5 binding changed: {key}")

    episode_gate = value.get("episode_gate", {})
    _require(episode_gate.get("shield_actor_fraction_max") == SHIELD_ACTOR_FRACTION_MAX,
             "V5 shield actor fraction limit changed")
    _require(episode_gate.get("inherited_from") == "i3_v2_1_qualification.yaml",
             "V5 inherited episode gate changed")

    collection = value.get("collection", {})
    for key, expected in {
        "observation_schema": "phaseA_low_obs_v2_1_retry_state",
        "observation_dimension": 88,
        "action_schema": "low_action_schema_v4_2_hierarchical",
        "output_files": ["dev_support_supplement.jsonl", "train_support.jsonl", "dev_support.jsonl"],
        "combined_row_order": "retained_v4_rows_then_dev_supplement_rows",
        "v4_incomplete_support_rows_retained": True,
        "replay": "exact_action_tape",
        "requested_logprob": "exact",
        "natural_flow": True,
        "no_policy_update": True,
        "student_authoritative": True,
        "teacher_label_only": True,
        "teacher_replaced_behavior": False,
        "calibrated_candidate_distribution": True,
        "support_v2_rows_mixed": False,
        "failed_qualification_rows_mixed": False,
        "failed_qualification_labels_mixed": False,
    }.items():
        _require(collection.get(key) == expected, f"V5 collection binding changed: {key}")
    _require(value.get("gates", {}).get("unchanged_from") ==
             "V4_2_P2_GATE_SUPPORT_COLLECTION_V2",
             "V5 gate provenance changed")
    _require(value.get("authorization", {}).get("always_false") == [
        "GATE_REPAIR_ALLOWED", "PPO_ALLOWED", "curriculum_promotion_allowed",
    ], "V5 authorization boundary changed")
    return value


def _expected_gate_shape(gates: Mapping[str, Any], split: str, *, all_pass: bool) -> None:
    expected_keys = {
        "cardinality", "teacher_class_minimums", "teacher_class_proportions",
        "interface_minimums", "interface_class_minimums", "schema_88d_finite",
        "decision_event_id_unique", "seed_coverage",
        "all_planned_episodes_execution_complete",
        "all_episode_shield_actor_fraction_within_inherited_limit",
        "all_episode_gates", "cross_split_seed_disjoint", "cross_split_event_disjoint",
        "all",
    }
    _require(set(gates) == expected_keys, f"V4 {split} gate shape changed")
    expected_false = set() if all_pass else {"cardinality", "all"}
    for key, flag in gates.items():
        _require(flag is (key not in expected_false), f"V4 {split} gate shape changed: {key}")


def _validate_record(record: Mapping[str, Any]) -> None:
    _require(record.get("exception") in (None, ""), "V4 record contains an exception")
    _require(record.get("episode_execution_complete") is True,
             "V4 record execution gate changed")
    _require(record.get("terminal_or_frozen_horizon") is True,
             "V4 record terminal/frozen gate changed")
    _require(record.get("safety_all_zero") is True, "V4 safety gate changed")
    _require(record.get("shield_actor_count", 0) == 0
             and record.get("shield_actor_fraction", 0.0) == 0.0
             and record.get("shield_actor_fraction_within_inherited_limit") is True,
             "V4 shield gate changed")
    _require(record.get("v1_rows_mixed") is False, "V4 row provenance changed")
    _require(int(record.get("completed", -1)) == int(record.get("requested", -2)),
             "V4 record completion changed")
    gates = record.get("gates", {})
    _require(isinstance(gates, Mapping) and gates and all(value is True for value in gates.values()),
             "V4 record episode gates changed")
    safety = record.get("safety", {})
    _require(isinstance(safety, Mapping) and all(int(value) == 0 for value in safety.values()),
             "V4 record safety counts changed")
    counts = record.get("counts", {})
    _require(isinstance(counts, Mapping)
             and int(counts.get("shield", 0)) == 0
             and int(counts.get("shield_actor", 0)) == 0,
             "V4 record shield counts changed")


def _validate_v4_records(records: Sequence[Mapping[str, Any]]) -> None:
    _require(len(records) == len(RETAINED_V4_TRAIN_SEEDS) + len(RETAINED_V4_DEV_SEEDS),
             "V4 record count changed")
    by_split: dict[str, list[Mapping[str, Any]]] = {"train": [], "dev": []}
    for record in records:
        split = str(record.get("split"))
        _require(split in by_split, "V4 record split changed")
        by_split[split].append(record)
        _validate_record(record)
        _require(str(record.get("scenario")) in SCENARIOS, "V4 record scenario changed")
    for split, expected_seeds in (("train", RETAINED_V4_TRAIN_SEEDS),
                                  ("dev", RETAINED_V4_DEV_SEEDS)):
        _require(len(by_split[split]) == len(expected_seeds), f"V4 {split} record count changed")
        observed = {int(item.get("seed", -1)) for item in by_split[split]}
        _require(observed == set(expected_seeds), f"V4 {split} record seeds changed")
        for index, record in enumerate(by_split[split]):
            _require(record.get("scenario") == SCENARIOS[index % len(SCENARIOS)],
                     f"V4 {split} scenario assignment changed")


def _validate_v4_manifest(path: Path) -> dict[str, Any]:
    """Validate V4 metadata without opening its support rows."""
    _require(path.is_file(), f"V4 manifest missing: {path}")
    _require(_sha(path) == V4_MANIFEST_SHA, "V4 manifest SHA mismatch")
    value = _read_json(path)
    expected = {
        "stage": v4.STAGE,
        "protocol_sha256": V4_PROTOCOL_SHA,
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "candidate_architecture_id": CANDIDATE_ARCHITECTURE_ID,
        "candidate_temperature": CANDIDATE_TEMPERATURE,
        "SUPPORT_COLLECTION": "FAIL",
        "SUPPORT_FROZEN": False,
        "LEARNABILITY_AUDIT_ALLOWED": False,
        "GATE_REPAIR_ALLOWED": False,
        "PPO_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "formal_episode_plan_complete": True,
        "training_called": False,
        "optimizer_called": False,
        "v3_incomplete_support_rows_retained": True,
        "support_v2_rows_mixed": False,
        "failed_qualification_rows_mixed": False,
        "failed_qualification_labels_mixed": False,
        "future_qualification_rollouts": 0,
        "future_qualification_consumed_marker": False,
        "future_qualification_marker_mutated": False,
        "formal_marker_created": True,
    }
    for key, wanted in expected.items():
        _require(value.get(key) == wanted, f"V4 manifest binding changed: {key}")
    _require(value.get("student_authoritative") is True
             and value.get("teacher_label_only") is True
             and value.get("teacher_replaced_behavior") is False,
             "V4 authority provenance changed")
    manifest_bindings = value.get("bindings", {})
    _require(
        manifest_bindings.get("v3_incomplete_support_rows_retained") is True
        or manifest_bindings.get("v4_incomplete_support_rows_retained") is True,
             "V4 retained-row provenance changed")
    for key in (
        "support_v2_rows_mixed", "failed_qualification_rows_mixed",
        "failed_qualification_labels_mixed", "future_qualification_marker_mutated",
    ):
        if key in manifest_bindings:
            _require(manifest_bindings.get(key) is False,
                     f"V4 binding provenance changed: {key}")
    _validate_v4_records(value.get("records", []))
    metrics = value.get("metrics", {})
    gates = value.get("gates", {})
    _require(isinstance(metrics, Mapping) and isinstance(gates, Mapping),
             "V4 metrics/gates are not mappings")
    _require(int(metrics.get("train", {}).get("rows", -1)) == V4_TRAIN_ROWS
             and int(metrics.get("dev", {}).get("rows", -1)) == V4_DEV_ROWS,
             "V4 combined row counts changed")
    for split, all_pass in (("train", True), ("dev", False)):
        split_metrics_value = metrics.get(split, {})
        _require(bool(split_metrics_value.get("schema_88d_finite"))
                 and bool(split_metrics_value.get("decision_event_id_unique"))
                 and bool(split_metrics_value.get("seed_coverage")),
                 f"V4 {split} integrity gate changed")
        completion = split_metrics_value.get("completion_requested", {})
        _require(int(completion.get("requested", -1)) == int(completion.get("completed", -2)),
                 f"V4 {split} completion changed")
        _expected_gate_shape(gates.get(split, {}), split, all_pass=all_pass)
    _require(gates["dev"]["cardinality"] is False and gates["dev"]["all"] is False,
             "V4 dev failure shape changed")
    _require(value.get("authorization", {}).get("GATE_REPAIR_ALLOWED") is False
             and value.get("authorization", {}).get("PPO_ALLOWED") is False,
             "V4 authorization boundary changed")
    return value


def _validate_v3_marker_read_only(checkpoint: Path) -> dict[str, Any]:
    marker = checkpoint.parent / V3_RESERVATION_MARKER
    _require(marker.is_file(), f"V3 consumed marker missing: {marker}")
    _require(_sha(marker) == V3_CONSUMED_MARKER_SHA, "V3 consumed marker SHA mismatch")
    value = _read_json(marker)
    expected = {
        "protocol_sha256": V3_PROTOCOL_SHA,
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "train_support_seeds": list(RETAINED_V3_TRAIN_SEEDS),
        "dev_support_seeds": list(RETAINED_V3_DEV_SEEDS),
        "future_qualification_seeds": list(FUTURE_QUALIFICATION_SEEDS),
    }
    for key, wanted in expected.items():
        _require(value.get(key) == wanted, f"V3 marker binding changed: {key}")
    return {"path": str(marker), "sha256": V3_CONSUMED_MARKER_SHA, "mutated": False}


def _validate_v4_marker(checkpoint: Path, v4_manifest: Path) -> dict[str, Any]:
    marker = checkpoint.parent / V4_RESERVATION_MARKER
    _require(marker.is_file(), f"V4 consumed marker missing: {marker}")
    _require(_sha(marker) == V4_CONSUMED_MARKER_SHA, "V4 consumed marker SHA mismatch")
    value = _read_json(marker)
    expected = {
        "protocol_sha256": V4_PROTOCOL_SHA,
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "v3_manifest_sha256": v4.V3_MANIFEST_SHA,
        "supplemental_train_seeds": list(v4.FORMAL_TRAIN_SEEDS),
        "supplemental_dev_seeds": list(v4.FORMAL_DEV_SEEDS),
        "output": str(v4_manifest.parent),
        "future_qualification_marker_mutated": False,
    }
    for key, wanted in expected.items():
        _require(value.get(key) == wanted, f"V4 consumed marker binding changed: {key}")
    _require("future_qualification_seeds" not in value,
             "V4 marker unexpectedly reserves future qualification seeds")
    return {"path": str(marker), "sha256": V4_CONSUMED_MARKER_SHA, "mutated": False}


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    expected_seeds: Sequence[int],
    expected_count: int,
) -> list[dict[str, Any]]:
    _require(len(rows) == expected_count, f"V4 {split} row count changed")
    result = [dict(row) for row in rows]
    observed: set[int] = set()
    events: list[str] = []
    for row in result:
        _require(row.get("support_split") == split, f"V4 {split} row split changed")
        observed.add(int(row.get("seed", -1)))
        event = str(row.get("decision_event_id", ""))
        _require(event not in {"", "None"}, f"V4 {split} event id missing")
        events.append(event)
        _require(_finite_88d(row), f"V4 {split} row schema changed")
        for key in (
            "support_v2_rows_mixed", "failed_qualification_rows_mixed",
            "failed_qualification_labels_mixed", "v1_rows_mixed",
        ):
            if key in row:
                _require(row.get(key) is False, f"V4 row provenance changed: {key}")
    _require(observed == set(int(seed) for seed in expected_seeds),
             f"V4 {split} seed coverage changed")
    _require(len(events) == len(set(events)), f"V4 {split} event identity changed")
    return result


def _load_retained_v4_rows(train_path: Path, dev_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Open V4 rows only after all V4 metadata and marker checks pass."""
    _require(_sha(train_path) == V4_COMBINED_TRAIN_SHA, "V4 combined train SHA mismatch")
    _require(_sha(dev_path) == V4_COMBINED_DEV_SHA, "V4 combined dev SHA mismatch")
    return {
        "train": _validate_rows(
            _read_jsonl_rows(train_path), split="train",
            expected_seeds=RETAINED_V4_TRAIN_SEEDS, expected_count=V4_TRAIN_ROWS,
        ),
        "dev": _validate_rows(
            _read_jsonl_rows(dev_path), split="dev",
            expected_seeds=RETAINED_V4_DEV_SEEDS, expected_count=V4_DEV_ROWS,
        ),
    }


def _validate_seed_groups(*, smoke: bool) -> None:
    groups = [
        set(RETAINED_V4_TRAIN_SEEDS), set(RETAINED_V4_DEV_SEEDS),
        set(FUTURE_QUALIFICATION_SEEDS), set(FORMAL_DEV_SEEDS),
    ]
    _require(all(not (left & right) for index, left in enumerate(groups)
                 for right in groups[index + 1:]),
             "V5 seed groups are not mutually exclusive")
    if smoke:
        _require(not (set(SMOKE_SEEDS) & set().union(*groups)),
                 "V5 smoke seed overlaps formal seed groups")


def _load_candidate(checkpoint: Path):
    policy, payload = v4._load_candidate(checkpoint)
    _require(payload.get("architecture_id") == CANDIDATE_ARCHITECTURE_ID,
             "V5 candidate architecture changed")
    _require(float(payload.get("temperature")) == CANDIDATE_TEMPERATURE,
             "V5 candidate temperature changed")
    _require(payload.get("STUDENT_QUALIFICATION_ALLOWED") is True,
             "V5 candidate authorization changed")
    return policy, payload


def _planned_episodes(*, smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return [{
            "split": "dev", "seed": SMOKE_SEEDS[0], "scenario": SMOKE_SCENARIOS[0],
            "max_steps": SMOKE_STEPS,
        }]
    plan: list[dict[str, Any]] = []
    for index, seed in enumerate(FORMAL_DEV_SEEDS):
        scenario = SCENARIOS[index % len(SCENARIOS)]
        config = support_config(int(scenario.removeprefix("flow")))
        plan.append({
            "split": "dev", "seed": int(seed), "scenario": scenario,
            "max_steps": int(config.max_simulation_seconds / config.dt),
        })
    return plan


def _validate_v4_inputs(
    checkpoint: Path,
    v4_manifest: Path,
    v4_train_support: Path,
    v4_dev_support: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any], Any, dict[str, Any]]:
    manifest = _validate_v4_manifest(v4_manifest)
    v3_marker = _validate_v3_marker_read_only(checkpoint)
    v4_marker = _validate_v4_marker(checkpoint, v4_manifest)
    support_files = manifest.get("support_files", {})
    _require(Path(str(support_files.get("combined", {}).get("train", ""))).name
             == v4_train_support.name, "V4 train support path binding changed")
    _require(Path(str(support_files.get("combined", {}).get("dev", ""))).name
             == v4_dev_support.name, "V4 dev support path binding changed")
    sha_map = support_files.get("sha256", {})
    _require(sha_map.get("combined_train") == V4_COMBINED_TRAIN_SHA
             and sha_map.get("combined_dev") == V4_COMBINED_DEV_SHA,
             "V4 combined support hash binding changed")
    rows = _load_retained_v4_rows(v4_train_support, v4_dev_support)
    candidate, payload = _load_candidate(checkpoint)
    return {
        "manifest": manifest,
        "v3_marker": v3_marker,
        "v4_marker": v4_marker,
    }, rows, manifest, candidate, payload


def _reserve_formal(
    checkpoint: Path,
    history_root: Path,
    output: Path,
    v4_manifest: Path,
) -> tuple[Path, dict[str, Any]]:
    marker = checkpoint.parent / RESERVATION_MARKER
    _require(not marker.exists(), f"V5 consumed marker already exists: {marker}")
    history = historical_registry(history_root, output)
    consumed = set(int(value) for value in history.get("seeds", set()))
    formal_overlap = sorted(FORMAL_DEV_SEED_SET & consumed)
    _require(not formal_overlap, f"V5 formal_dev_seed_overlap: {formal_overlap}")
    audited_overlap = sorted(FORMAL_AUDIT_SEEDS & consumed)
    smoke_overlap = sorted(set(SMOKE_SEEDS) & consumed)
    reserve_once(marker, {
        "protocol_sha256": PROTOCOL_SHA or _sha(PROTOCOL),
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "v4_manifest_sha256": V4_MANIFEST_SHA,
        "supplemental_dev_seeds": list(FORMAL_DEV_SEEDS),
        "output": str(output),
        "future_qualification_marker_mutated": False,
    })
    return marker, {
        "historical_seed_count": len(consumed),
        "formal_dev_seed_overlap": formal_overlap,
        "audited_seed_range": "785000-785003",
        "audited_seed_overlap": audited_overlap,
        "smoke_seed_overlap": smoke_overlap,
        "future_qualification_markers_mutated": False,
    }


def _write_jsonl_immutable(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
    return _sha(path)


def _combined_identity(rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, bool]:
    rows = list(rows_by_split["train"]) + list(rows_by_split["dev"])
    events = [str(row.get("decision_event_id", "")) for row in rows]
    _require(all(event not in {"", "None"} for event in events),
             "V5 combined support event id missing")
    _require(len(events) == len(set(events)), "V5 combined support event identity changed")
    cross = v4.v3.v2._cross_split_identity(rows)
    _require(cross["train_dev_seed_disjoint"] and cross["train_dev_event_disjoint"],
             "V5 combined support cross-split identity changed")
    return cross


def run(
    output: Path,
    checkpoint: Path,
    v4_manifest: Path,
    v4_train_support: Path,
    v4_dev_support: Path,
    history_root: Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the V5 dev-only smoke or formal support supplement."""
    protocol_value = protocol()
    del protocol_value
    output = Path(output)
    checkpoint = Path(checkpoint)
    v4_manifest = Path(v4_manifest)
    v4_train_support = Path(v4_train_support)
    v4_dev_support = Path(v4_dev_support)
    history_root = Path(history_root)
    _require(not output.exists(), f"V5 output already exists: {output}")
    _validate_seed_groups(smoke=smoke)
    bindings, retained_rows, v4_manifest_value, candidate, checkpoint_payload = _validate_v4_inputs(
        checkpoint, v4_manifest, v4_train_support, v4_dev_support,
    )
    if not smoke:
        _require(history_root.is_dir(), f"V5 history root missing: {history_root}")
        _require(not (checkpoint.parent / RESERVATION_MARKER).exists(),
                 "V5 consumed marker already exists")
    output.mkdir(parents=True, exist_ok=False)
    plan = _planned_episodes(smoke=smoke)
    marker: Path | None = None
    if not smoke:
        marker, registry = _reserve_formal(checkpoint, history_root, output, v4_manifest)
    else:
        registry = {
            "historical_seed_count": None,
            "formal_dev_seed_overlap": [],
            "audited_seed_range": "785000-785003",
            "audited_seed_overlap": [],
            "smoke_seed_overlap": [],
            "future_qualification_markers_mutated": False,
        }
    _write(output / "pre_reset_registry.json", {
        "protocol_sha256": PROTOCOL_SHA or _sha(PROTOCOL),
        "smoke": smoke,
        "planned_episodes": plan,
        "formal_seed_reservation": not smoke,
        "consumed_marker_created": not smoke,
        "historical_registry": registry,
        "v4_manifest_sha256": V4_MANIFEST_SHA,
        "v4_combined_train_sha256": V4_COMBINED_TRAIN_SHA,
        "v4_combined_dev_sha256": V4_COMBINED_DEV_SHA,
        "v3_marker_mutated": False,
        "v4_marker_mutated": False,
        "support_v2_rows_mixed": False,
        "failed_qualification_rows_mixed": False,
        "failed_qualification_labels_mixed": False,
    })

    records: list[dict[str, Any]] = [dict(record) for record in v4_manifest_value["records"]]
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [dict(row) for row in retained_rows["train"]],
        "dev": [dict(row) for row in retained_rows["dev"]],
    }
    supplement_rows: list[dict[str, Any]] = []
    supplement_records: list[dict[str, Any]] = []
    for item in plan:
        split, seed, scenario = str(item["split"]), int(item["seed"]), str(item["scenario"])
        episode_output = output / f"supplement_{split}_{seed}_{scenario}"
        try:
            summary, rows = _episode_record(
                candidate, split=split, seed=seed, scenario=scenario,
                output=episode_output, max_steps=int(item["max_steps"]),
            )
        except Exception as exc:
            summary = {
                "split": split, "seed": seed, "scenario": scenario,
                "requested": int(scenario.removeprefix("flow")), "completed": 0,
                "episode_execution_complete": False, "exception": repr(exc),
                "gates": {"collection_exception": False},
            }
            records.append(summary)
            supplement_records.append(summary)
            _write(output / "progress.json", {
                "completed_supplement_episodes": len(supplement_records),
                "total_supplement_episodes": len(plan),
                "retained_v4_rows": {key: len(value) for key, value in retained_rows.items()},
                "supplement_records": supplement_records,
            })
            break
        summary = dict(summary)
        summary["rows"] = len(rows)
        summary["support_collection_source"] = "v5_dev_supplement"
        records.append(summary)
        supplement_records.append(summary)
        supplement_rows.extend(dict(row) for row in rows)
        rows_by_split["dev"].extend(dict(row) for row in rows)
        _write(output / "progress.json", {
            "completed_supplement_episodes": len(supplement_records),
            "total_supplement_episodes": len(plan),
            "retained_v4_rows": {key: len(value) for key, value in retained_rows.items()},
            "new_supplement_rows": len(supplement_rows),
            "supplement_records": supplement_records,
        })
        if any(int(value) for value in summary.get("safety", {}).values()):
            break

    supplement_hash = _write_jsonl_immutable(output / "dev_support_supplement.jsonl", supplement_rows)
    combined = _combined_identity(rows_by_split)
    combined_hashes = {
        split: _write_jsonl_immutable(output / f"{split}_support.jsonl", rows_by_split[split])
        for split in ("train", "dev")
    }
    planned_train = tuple(RETAINED_V4_TRAIN_SEEDS)
    planned_dev = tuple(RETAINED_V4_DEV_SEEDS) + ((SMOKE_SEEDS[0],) if smoke else tuple(FORMAL_DEV_SEEDS))
    metrics = {
        split: split_metrics(rows_by_split[split], split, records, planned)
        for split, planned in (("train", planned_train), ("dev", planned_dev))
    }
    for split in ("train", "dev"):
        metrics[split]["cross_split_seed_disjoint"] = combined["train_dev_seed_disjoint"]
        metrics[split]["cross_split_event_disjoint"] = combined["train_dev_event_disjoint"]
    gates = {
        split: split_gates(metrics[split], split, formal=not smoke)
        for split in ("train", "dev")
    }
    for split in gates:
        gates[split]["cross_split_seed_disjoint"] = combined["train_dev_seed_disjoint"]
        gates[split]["cross_split_event_disjoint"] = combined["train_dev_event_disjoint"]
        gates[split]["all"] = all(gates[split].values())
    auth = formal_authorization(gates["train"], gates["dev"], smoke=smoke)
    supplement_complete = (not smoke and len(supplement_records) == len(plan)
                           and all(bool(item.get("episode_execution_complete"))
                                    for item in supplement_records))
    result = {
        "stage": STAGE,
        "smoke": smoke,
        "protocol_sha256": PROTOCOL_SHA or _sha(PROTOCOL),
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "candidate_architecture_id": CANDIDATE_ARCHITECTURE_ID,
        "candidate_temperature": CANDIDATE_TEMPERATURE,
        "v4_manifest_sha256": V4_MANIFEST_SHA,
        "v4_combined_train_sha256": V4_COMBINED_TRAIN_SHA,
        "v4_combined_dev_sha256": V4_COMBINED_DEV_SHA,
        "v4_protocol_sha256": V4_PROTOCOL_SHA,
        "v4_consumed_marker_sha256": V4_CONSUMED_MARKER_SHA,
        "checkpoint_payload": {
            "stage": checkpoint_payload.get("stage"),
            "architecture_id": checkpoint_payload.get("architecture_id"),
            "temperature": checkpoint_payload.get("temperature"),
            "qualification_only": checkpoint_payload.get("qualification_only"),
            "checkpoint_promotable": checkpoint_payload.get("checkpoint_promotable"),
        },
        "predecessor_bindings": {
            "v4_manifest_sha256": V4_MANIFEST_SHA,
            "v4_combined_train_sha256": V4_COMBINED_TRAIN_SHA,
            "v4_combined_dev_sha256": V4_COMBINED_DEV_SHA,
            "v4_protocol_sha256": V4_PROTOCOL_SHA,
            "v4_consumed_marker_sha256": V4_CONSUMED_MARKER_SHA,
            "v3_consumed_marker_sha256": V3_CONSUMED_MARKER_SHA,
        },
        "bindings": {
            "v4_manifest": bindings["manifest"],
            "v3_consumed_marker": bindings["v3_marker"],
            "v4_consumed_marker": bindings["v4_marker"],
            "v4_rows_opened_after_validation": True,
            "v4_incomplete_support_rows_retained": True,
            "support_v2_rows_mixed": False,
            "failed_qualification_rows_mixed": False,
            "failed_qualification_labels_mixed": False,
            "v3_marker_mutated": False,
            "v4_marker_mutated": False,
            "future_qualification_markers_mutated": False,
        },
        "records": records,
        "retained_v4_records": len(v4_manifest_value["records"]),
        "supplement_records": len(supplement_records),
        "metrics": metrics,
        "gates": gates,
        "authorization": auth,
        "SUPPORT_COLLECTION": auth["SUPPORT_COLLECTION"],
        "SUPPORT_FROZEN": auth["SUPPORT_FROZEN"],
        "LEARNABILITY_AUDIT_ALLOWED": auth["LEARNABILITY_AUDIT_ALLOWED"],
        "GATE_REPAIR_ALLOWED": False,
        "PPO_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "formal_episode_plan_complete": supplement_complete,
        "formal_reservation_registry": registry,
        "training_called": False,
        "optimizer_called": False,
        "student_authoritative": True,
        "teacher_label_only": True,
        "teacher_replaced_behavior": False,
        "v4_incomplete_support_rows_retained": True,
        "v4_retained_rows": {key: len(value) for key, value in retained_rows.items()},
        "supplement_new_rows": {"train": 0, "dev": len(supplement_rows)},
        "support_v2_rows_mixed": False,
        "failed_qualification_rows_mixed": False,
        "failed_qualification_labels_mixed": False,
        "future_qualification_rollouts": 0,
        "future_qualification_consumed_marker": False,
        "future_qualification_markers_mutated": False,
        "v3_marker_mutated": False,
        "v4_marker_mutated": False,
        "formal_marker": str(marker) if marker is not None else None,
        "formal_marker_created": marker is not None,
        "support_files": {
            "retained_v4": {
                "train": str(v4_train_support), "dev": str(v4_dev_support),
                "rows": {"train": len(retained_rows["train"]), "dev": len(retained_rows["dev"])},
                "sha256": {"train": V4_COMBINED_TRAIN_SHA, "dev": V4_COMBINED_DEV_SHA},
            },
            "supplement": {
                "dev": str(output / "dev_support_supplement.jsonl"),
                "rows": {"train": 0, "dev": len(supplement_rows)},
                "sha256": {"dev": supplement_hash},
            },
            "combined": {
                "train": str(output / "train_support.jsonl"),
                "dev": str(output / "dev_support.jsonl"),
                "rows": {key: len(value) for key, value in rows_by_split.items()},
                "sha256": combined_hashes,
                "row_order": "retained_v4_rows_then_dev_supplement_rows",
            },
            "sha256": {
                "v4_train": V4_COMBINED_TRAIN_SHA,
                "v4_dev": V4_COMBINED_DEV_SHA,
                "supplement_dev": supplement_hash,
                "combined_train": combined_hashes["train"],
                "combined_dev": combined_hashes["dev"],
            },
        },
        "v4_combined_train_output_sha256": combined_hashes["train"],
        "v4_combined_dev_output_sha256": combined_hashes["dev"],
        "supplement_dev_support_sha256": supplement_hash,
        "source_hashes": {
            "support_runner": _sha(Path(__file__)),
            "support_protocol": _sha(PROTOCOL),
            "v4_support_runner": _sha(Path(v4.__file__)),
            "v4_support_protocol": V4_PROTOCOL_SHA,
            "candidate_checkpoint": CANDIDATE_SHA,
            "v4_manifest": V4_MANIFEST_SHA,
            "v4_combined_train": V4_COMBINED_TRAIN_SHA,
            "v4_combined_dev": V4_COMBINED_DEV_SHA,
            "v4_consumed_marker": V4_CONSUMED_MARKER_SHA,
            "v3_consumed_marker": V3_CONSUMED_MARKER_SHA,
        },
    }
    _write(output / "manifest.json", result)
    print(json.dumps({
        "stage": STAGE,
        "SUPPORT_COLLECTION": result["SUPPORT_COLLECTION"],
        "smoke": smoke,
        "retained_rows": result["v4_retained_rows"],
        "supplement_rows": result["supplement_new_rows"],
        "combined_rows": {key: len(value) for key, value in rows_by_split.items()},
    }, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--v4-manifest", type=Path, required=True)
    parser.add_argument("--v4-train-support", type=Path, required=True)
    parser.add_argument("--v4-dev-support", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    torch.set_num_threads(1)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2), flush=True)
