"""P2 gate support V4 supplemental collection.

This runner retains the validated, incomplete V3 support rows and collects a
fresh calibrated-candidate cohort under the unchanged V2 support gates.  V3
and failed-qualification artifacts are read-only bindings; no old V2 or
failed-qualification rows are opened or mixed into the combined support set.
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


PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/ppo_p2_gate_support_collection_v4_supplement.yaml"
)
PROTOCOL_SHA = "d626690c054f61b40c2740ee7fe05bf80e4e2f5c2f8d291f229291897538b8c3"
STAGE = "V4_2_P2_GATE_SUPPORT_COLLECTION_V4_SUPPLEMENT"
CANDIDATE_SHA = "61e03e02e3fb204d2db4e814097543905f51e1a3772626fdac1d7bc02ab8ca6b"
CANDIDATE_ARCHITECTURE_ID = "v4_2_hierarchical_gate_temperature_calibrated_residual_adapter_v3"
CANDIDATE_TEMPERATURE = 2.5

V3_MANIFEST_SHA = "b4e167d57b1b5f637fdcbf87054e62bc7d2620251a463cc6b033a045731df5d6"
V3_TRAIN_SUPPORT_SHA = "4afe9fa3fa04ac5963d4320afe0b2ad8b7c6c66cb07e40f2a12a9f012cfcfc75"
V3_DEV_SUPPORT_SHA = "1f2f12389e85e4a237261319656a6fc602cf8e157f64a94f7e86d86a941c85db"
V3_PROTOCOL_SHA = "88ec0a7b7ef6bcc771f9388490ba43a378e0f99ebc03e1753c6cf58293ed78a1"
V3_CONSUMED_MARKER_SHA = "7568e6e46259f38e066a1688e9a1c78ca6d816f03bd29868c35aa9a0c945f17f"
V3_RESERVATION_MARKER = "p2_gate_support_collection_v3_consumed.json"
RESERVATION_MARKER = "p2_gate_support_collection_v4_supplement_consumed.json"

RETAINED_V3_TRAIN_SEEDS = tuple(range(780001, 780010))
RETAINED_V3_DEV_SEEDS = tuple(range(781001, 781007))
FUTURE_QUALIFICATION_SEEDS = tuple(range(782001, 782007))
FORMAL_TRAIN_SEEDS = tuple(range(783001, 783019))
FORMAL_DEV_SEEDS = tuple(range(784001, 784010))
SMOKE_SEEDS = (783000, 784000)
SMOKE_SCENARIOS = ("flow6", "flow6")
SMOKE_STEPS = 160
SCENARIOS = v3.SCENARIOS
INTERFACES = v3.INTERFACES
SHIELD_ACTOR_FRACTION_MAX = v3.SHIELD_ACTOR_FRACTION_MAX
V3_TRAIN_ROWS = 1150
V3_DEV_ROWS = 908
FORMAL_AUDIT_SEEDS = set(range(783000, 785000))

_sha = v3._sha
_write = v3._write
_require = v3._require
reserve_once = v3.reserve_once
historical_registry = v3.historical_registry
support_config = v3.support_config
collection_config = support_config
diagnostic_config = support_config
_episode_record = v3._episode_record
filter_actor_rows = v3.filter_actor_rows
_actor_rows = v3._actor_rows
_finite_88d = v3._finite_88d
_thresholds = v3._thresholds
split_metrics = v3.split_metrics
split_gates = v3.split_gates
support_gates = v3.support_gates
formal_authorization = v3.formal_authorization


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
    """Load and verify the byte-frozen V4 supplement protocol."""
    _require(PROTOCOL.is_file(), f"V4 supplement protocol missing: {PROTOCOL}")
    if PROTOCOL_SHA:
        _require(_sha(PROTOCOL) == PROTOCOL_SHA,
                 "P2 V4 supplement protocol changed after preregistration")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "V4 supplement protocol must be a mapping")
    _require(value.get("protocol") == STAGE, "V4 supplement protocol id changed")
    _require(value.get("candidate_checkpoint_sha256") == CANDIDATE_SHA,
             "V4 supplement candidate checkpoint binding changed")
    _require(value.get("candidate_architecture_id") == CANDIDATE_ARCHITECTURE_ID,
             "V4 supplement candidate architecture binding changed")
    _require(float(value.get("candidate_temperature")) == CANDIDATE_TEMPERATURE,
             "V4 supplement candidate temperature binding changed")

    predecessors = value.get("predecessors", {})
    for key, expected in {
        "v3_manifest_sha256": V3_MANIFEST_SHA,
        "v3_train_support_sha256": V3_TRAIN_SUPPORT_SHA,
        "v3_dev_support_sha256": V3_DEV_SUPPORT_SHA,
        "v3_protocol_sha256": V3_PROTOCOL_SHA,
        "v3_consumed_marker_sha256": V3_CONSUMED_MARKER_SHA,
    }.items():
        _require(predecessors.get(key) == expected,
                 f"V4 supplement predecessor binding changed: {key}")

    formal = value.get("formal", {})
    for key, expected in {
        "retained_v3_train_seeds": RETAINED_V3_TRAIN_SEEDS,
        "retained_v3_dev_seeds": RETAINED_V3_DEV_SEEDS,
        "supplemental_train_seeds": FORMAL_TRAIN_SEEDS,
        "supplemental_dev_seeds": FORMAL_DEV_SEEDS,
        "future_qualification_seeds": FUTURE_QUALIFICATION_SEEDS,
        "scenarios": SCENARIOS,
    }.items():
        _require(tuple(formal.get(key, ())) == tuple(expected),
                 f"V4 supplement formal binding changed: {key}")
    for key, expected in {
        "scenario_assignment": "cycle_in_list_order_per_split",
        "teacher_mode": "teacher_label_only",
        "scheduler_mode": "execution",
        "retained_v3_marker": V3_RESERVATION_MARKER,
        "consumed_marker": RESERVATION_MARKER,
        "qualification_seed_range_audited": "783000-784999",
    }.items():
        _require(formal.get(key) == expected, f"V4 supplement formal binding changed: {key}")
    _require(formal.get("deterministic") is False, "V4 supplement collection is not stochastic")
    _require(formal.get("teacher_release_label_enabled") is False,
             "V4 supplement Teacher release-label binding changed")
    _require(formal.get("first_reset_requires") ==
             "historical_registry_and_new_supplement_reservation",
             "V4 supplement reset ordering changed")
    _require(formal.get("future_qualification_rollouts") == 0
             and formal.get("future_qualification_marker_mutated") is False,
             "V4 supplement future qualification boundary changed")

    smoke = value.get("smoke", {})
    _require(tuple(smoke.get("seeds", ())) == SMOKE_SEEDS, "V4 supplement smoke seeds changed")
    _require(tuple(smoke.get("scenarios", ())) == SMOKE_SCENARIOS,
             "V4 supplement smoke scenarios changed")
    _require(smoke.get("steps") == SMOKE_STEPS, "V4 supplement smoke horizon changed")
    _require(smoke.get("reserve_formal_seeds") is False
             and smoke.get("create_consumed_marker") is False
             and smoke.get("authorization") is False,
             "V4 supplement smoke authorization changed")

    bindings = value.get("bindings", {})
    for key, expected in {
        "v3_manifest_sha256": V3_MANIFEST_SHA,
        "v3_train_support_sha256": V3_TRAIN_SUPPORT_SHA,
        "v3_dev_support_sha256": V3_DEV_SUPPORT_SHA,
        "v3_consumed_marker_sha256": V3_CONSUMED_MARKER_SHA,
        "observation_schema": "phaseA_low_obs_v2_1_retry_state",
        "observation_dimension": 88,
        "action_schema": "low_action_schema_v4_2_hierarchical",
        "source_usage": "v3_rows_retained_supplement_rows_new_no_v2_or_qualification_rows",
    }.items():
        _require(bindings.get(key) == expected, f"V4 supplement binding changed: {key}")

    episode_gate = value.get("episode_gate", {})
    _require(episode_gate.get("shield_actor_fraction_max") == SHIELD_ACTOR_FRACTION_MAX,
             "V4 supplement shield actor fraction limit changed")
    _require(episode_gate.get("inherited_from") == "i3_v2_1_qualification.yaml",
             "V4 supplement inherited episode gate changed")

    collection = value.get("collection", {})
    for key, expected in {
        "observation_schema": "phaseA_low_obs_v2_1_retry_state",
        "observation_dimension": 88,
        "action_schema": "low_action_schema_v4_2_hierarchical",
        "output_files": [
            "train_support_supplement.jsonl", "dev_support_supplement.jsonl",
            "train_support.jsonl", "dev_support.jsonl",
        ],
        "combined_row_order": "retained_v3_rows_then_supplement_rows",
        "retained_v3_incomplete_support_rows": True,
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
        _require(collection.get(key) == expected, f"V4 supplement collection binding changed: {key}")
    _require(value.get("gates", {}).get("unchanged_from") ==
             "V4_2_P2_GATE_SUPPORT_COLLECTION_V2",
             "V4 supplement gate provenance changed")
    _require(value.get("authorization", {}).get("always_false") == [
        "GATE_REPAIR_ALLOWED", "PPO_ALLOWED", "curriculum_promotion_allowed",
    ], "V4 supplement authorization boundary changed")
    return value


def _expected_gate_shape(gates: Mapping[str, Any], split: str) -> None:
    expected_false = {
        "cardinality", "teacher_class_minimums", "interface_minimums",
        "interface_class_minimums", "all",
    }
    expected_keys = {
        "cardinality", "teacher_class_minimums", "teacher_class_proportions",
        "interface_minimums", "interface_class_minimums", "schema_88d_finite",
        "decision_event_id_unique", "seed_coverage",
        "all_planned_episodes_execution_complete",
        "all_episode_shield_actor_fraction_within_inherited_limit",
        "all_episode_gates", "cross_split_seed_disjoint", "cross_split_event_disjoint",
        "all",
    }
    _require(set(gates) == expected_keys, f"V3 {split} gate shape changed")
    for key, flag in gates.items():
        _require(flag is (key not in expected_false),
                 f"V3 {split} gate failure shape changed: {key}")


def _validate_v3_records(records: Sequence[Mapping[str, Any]]) -> None:
    _require(len(records) == len(RETAINED_V3_TRAIN_SEEDS) + len(RETAINED_V3_DEV_SEEDS),
             "V3 record count changed")
    expected = {
        "train": RETAINED_V3_TRAIN_SEEDS,
        "dev": RETAINED_V3_DEV_SEEDS,
    }
    by_split: dict[str, list[Mapping[str, Any]]] = {"train": [], "dev": []}
    for record in records:
        split = str(record.get("split"))
        _require(split in by_split, "V3 record split changed")
        by_split[split].append(record)
        _require(record.get("exception") in (None, ""), "V3 record contains an exception")
        _require(record.get("episode_execution_complete") is True,
                 "V3 record execution gate changed")
        _require(record.get("terminal_or_frozen_horizon") is True,
                 "V3 record terminal/frozen gate changed")
        _require(record.get("safety_all_zero") is True, "V3 safety gate changed")
        _require(record.get("shield_actor_count", 0) == 0,
                 "V3 shield actor count changed")
        _require(record.get("shield_actor_fraction", 0.0) == 0.0,
                 "V3 shield actor fraction changed")
        _require(record.get("shield_actor_fraction_within_inherited_limit") is True,
                 "V3 shield fraction gate changed")
        _require(record.get("v1_rows_mixed") is False, "V3 row provenance changed")
        gates = record.get("gates", {})
        _require(isinstance(gates, Mapping) and gates and all(value is True for value in gates.values()),
                 "V3 record episode gate changed")
        safety = record.get("safety", {})
        _require(isinstance(safety, Mapping) and all(int(value) == 0 for value in safety.values()),
                 "V3 safety counts changed")
        counts = record.get("counts", {})
        _require(isinstance(counts, Mapping)
                 and int(counts.get("shield", 0)) == 0
                 and int(counts.get("shield_actor", 0)) == 0,
                 "V3 shield counts changed")
        scenario = str(record.get("scenario"))
        _require(scenario in SCENARIOS, "V3 record scenario changed")
        _require(int(record.get("completed", -1)) == int(record.get("requested", -2)),
                 "V3 record completion changed")
    for split, seeds in expected.items():
        _require(len(by_split[split]) == len(seeds), f"V3 {split} record count changed")
        observed = tuple(int(item.get("seed", -1)) for item in by_split[split])
        _require(set(observed) == set(seeds), f"V3 {split} record seeds changed")
        for index, record in enumerate(by_split[split]):
            _require(record.get("scenario") == SCENARIOS[index % len(SCENARIOS)],
                     f"V3 {split} scenario assignment changed")


def _validate_v3_manifest(path: Path) -> dict[str, Any]:
    """Validate V3 metadata without opening its support rows."""
    _require(path.is_file(), f"V3 manifest missing: {path}")
    _require(_sha(path) == V3_MANIFEST_SHA, "V3 manifest SHA mismatch")
    value = _read_json(path)
    expected = {
        "stage": v3.STAGE,
        "protocol_sha256": V3_PROTOCOL_SHA,
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
        "support_v2_rows_mixed": False,
        "failed_qualification_rows_mixed": False,
        "failed_qualification_labels_mixed": False,
        "future_qualification_rollouts": 0,
        "future_qualification_consumed_marker": False,
        "formal_marker_created": True,
    }
    for key, wanted in expected.items():
        _require(value.get(key) == wanted, f"V3 manifest binding changed: {key}")
    _require(value.get("training_called") is False and value.get("optimizer_called") is False,
             "V3 support unexpectedly trained")
    _require(value.get("student_authoritative") is True
             and value.get("teacher_label_only") is True
             and value.get("teacher_replaced_behavior") is False,
             "V3 authority provenance changed")
    formal_marker = value.get("formal_marker")
    _require(formal_marker not in (None, "")
             and Path(str(formal_marker)).name == V3_RESERVATION_MARKER,
             "V3 marker binding changed")
    records = value.get("records", [])
    _require(isinstance(records, list), "V3 records are not a list")
    _validate_v3_records(records)

    metrics = value.get("metrics", {})
    gates = value.get("gates", {})
    _require(isinstance(metrics, Mapping) and isinstance(gates, Mapping),
             "V3 metrics/gates are not mappings")
    for split, expected_rows in (("train", V3_TRAIN_ROWS), ("dev", V3_DEV_ROWS)):
        _require(int(metrics.get(split, {}).get("rows", -1)) == expected_rows,
                 f"V3 {split} row count changed")
        completion = metrics.get(split, {}).get("completion_requested", {})
        _require(int(completion.get("requested", -1)) == int(completion.get("completed", -2)),
                 f"V3 {split} completion changed")
        _require(bool(metrics.get(split, {}).get("schema_88d_finite"))
                 and bool(metrics.get(split, {}).get("decision_event_id_unique"))
                 and bool(metrics.get(split, {}).get("seed_coverage")),
                 f"V3 {split} integrity gate changed")
        _expected_gate_shape(gates.get(split, {}), split)
    _require(int(metrics["train"]["completion_requested"]["requested"])
             + int(metrics["dev"]["completion_requested"]["requested"]) == 130,
             "V3 total requested completion changed")
    _require(value.get("authorization", {}).get("GATE_REPAIR_ALLOWED") is False
             and value.get("authorization", {}).get("PPO_ALLOWED") is False,
             "V3 authorization boundary changed")
    return value


def _validate_v3_marker(checkpoint: Path, v3_manifest: Path) -> dict[str, Any]:
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
        "output": str(v3_manifest.parent),
    }
    for key, wanted in expected.items():
        _require(value.get(key) == wanted, f"V3 consumed marker binding changed: {key}")
    return {
        "path": str(marker),
        "sha256": V3_CONSUMED_MARKER_SHA,
        "protocol_sha256": value["protocol_sha256"],
        "output": value["output"],
        "rows_opened": False,
        "mutated": False,
    }


def _validate_support_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    expected_seeds: Sequence[int],
    expected_count: int,
) -> list[dict[str, Any]]:
    _require(len(rows) == expected_count, f"V3 {split} support row count changed")
    result = [dict(row) for row in rows]
    observed_seeds: set[int] = set()
    event_ids: list[str] = []
    for row in result:
        _require(row.get("support_split") == split, f"V3 {split} row split changed")
        seed = int(row.get("seed", -1))
        observed_seeds.add(seed)
        event = str(row.get("decision_event_id", ""))
        _require(event not in {"", "None"}, f"V3 {split} event id missing")
        event_ids.append(event)
        _require(_finite_88d(row), f"V3 {split} row schema changed")
        for key in (
            "support_v2_rows_mixed", "failed_qualification_rows_mixed",
            "failed_qualification_labels_mixed", "v1_rows_mixed",
        ):
            if key in row:
                _require(row.get(key) is False, f"V3 row provenance changed: {key}")
    _require(observed_seeds == set(int(seed) for seed in expected_seeds),
             f"V3 {split} seed coverage changed")
    _require(len(event_ids) == len(set(event_ids)), f"V3 {split} event identity changed")
    return result


def _load_retained_v3_rows(
    train_path: Path,
    dev_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Open V3 rows only after manifest, hashes, and marker have passed."""
    _require(_sha(train_path) == V3_TRAIN_SUPPORT_SHA, "V3 train support SHA mismatch")
    _require(_sha(dev_path) == V3_DEV_SUPPORT_SHA, "V3 dev support SHA mismatch")
    train = _validate_support_rows(
        _read_jsonl_rows(train_path), split="train",
        expected_seeds=RETAINED_V3_TRAIN_SEEDS, expected_count=V3_TRAIN_ROWS,
    )
    dev = _validate_support_rows(
        _read_jsonl_rows(dev_path), split="dev",
        expected_seeds=RETAINED_V3_DEV_SEEDS, expected_count=V3_DEV_ROWS,
    )
    return {"train": train, "dev": dev}


def _validate_seed_groups(*, smoke: bool) -> None:
    groups = [
        set(RETAINED_V3_TRAIN_SEEDS), set(RETAINED_V3_DEV_SEEDS),
        set(FUTURE_QUALIFICATION_SEEDS), set(FORMAL_TRAIN_SEEDS), set(FORMAL_DEV_SEEDS),
    ]
    _require(all(not (left & right) for index, left in enumerate(groups)
                 for right in groups[index + 1:]),
             "V4 supplement seed groups are not mutually exclusive")
    if smoke:
        _require(not (set(SMOKE_SEEDS) & set().union(*groups)),
                 "V4 supplement smoke seed overlaps formal seed groups")


def _load_candidate(checkpoint: Path):
    policy, payload = v3._load_calibrated_candidate(checkpoint)
    _require(payload.get("architecture_id") == CANDIDATE_ARCHITECTURE_ID,
             "V4 supplement candidate architecture changed")
    _require(float(payload.get("temperature")) == CANDIDATE_TEMPERATURE,
             "V4 supplement candidate temperature changed")
    _require(payload.get("STUDENT_QUALIFICATION_ALLOWED") is True,
             "V4 supplement candidate authorization changed")
    return policy, payload


def _planned_episodes(*, smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return [
            {"split": "train", "seed": SMOKE_SEEDS[0], "scenario": SMOKE_SCENARIOS[0],
             "max_steps": SMOKE_STEPS},
            {"split": "dev", "seed": SMOKE_SEEDS[1], "scenario": SMOKE_SCENARIOS[1],
             "max_steps": SMOKE_STEPS},
        ]
    plan: list[dict[str, Any]] = []
    for split, seeds in (("train", FORMAL_TRAIN_SEEDS), ("dev", FORMAL_DEV_SEEDS)):
        for index, seed in enumerate(seeds):
            scenario = SCENARIOS[index % len(SCENARIOS)]
            config = support_config(int(scenario.removeprefix("flow")))
            plan.append({
                "split": split,
                "seed": int(seed),
                "scenario": scenario,
                "max_steps": int(config.max_simulation_seconds / config.dt),
            })
    return plan


def _validate_v3_inputs(
    checkpoint: Path,
    v3_manifest: Path,
    v3_train_support: Path,
    v3_dev_support: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any], Any, dict[str, Any]]:
    manifest = _validate_v3_manifest(v3_manifest)
    marker = _validate_v3_marker(checkpoint, v3_manifest)
    rows = _load_retained_v3_rows(v3_train_support, v3_dev_support)
    support_files = manifest.get("support_files", {})
    _require(Path(str(support_files.get("train", ""))).name == v3_train_support.name,
             "V3 train support path binding changed")
    _require(Path(str(support_files.get("dev", ""))).name == v3_dev_support.name,
             "V3 dev support path binding changed")
    sha_map = support_files.get("sha256", {})
    _require(sha_map.get("train") == V3_TRAIN_SUPPORT_SHA
             and sha_map.get("dev") == V3_DEV_SUPPORT_SHA,
             "V3 support file hash binding changed")
    candidate, payload = _load_candidate(checkpoint)
    return {"manifest": manifest, "marker": marker}, rows, manifest, candidate, payload


def _reserve_formal(
    checkpoint: Path,
    history_root: Path,
    output: Path,
    v3_manifest: Path,
) -> tuple[Path, dict[str, Any]]:
    marker = checkpoint.parent / RESERVATION_MARKER
    _require(not marker.exists(), f"V4 supplement consumed marker already exists: {marker}")
    history = historical_registry(history_root, output)
    consumed = set(int(value) for value in history.get("seeds", set()))
    _require(not (FORMAL_AUDIT_SEEDS & consumed),
             "V4 supplement audited seed range already contains consumed seeds")
    reserve_once(marker, {
        "protocol_sha256": PROTOCOL_SHA or _sha(PROTOCOL),
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "v3_manifest_sha256": V3_MANIFEST_SHA,
        "supplemental_train_seeds": list(FORMAL_TRAIN_SEEDS),
        "supplemental_dev_seeds": list(FORMAL_DEV_SEEDS),
        "output": str(output),
        "future_qualification_marker_mutated": False,
    })
    return marker, {
        "historical_seed_count": len(consumed),
        "checked_audited_seed_range": "783000-784999",
        "future_qualification_marker_mutated": False,
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
             "combined support event id missing")
    _require(len(events) == len(set(events)), "combined support event identity changed")
    cross = v3.v2._cross_split_identity(rows)
    _require(cross["train_dev_seed_disjoint"] and cross["train_dev_event_disjoint"],
             "combined support cross-split identity changed")
    return cross


def run(
    output: Path,
    checkpoint: Path,
    v3_manifest: Path,
    v3_train_support: Path,
    v3_dev_support: Path,
    history_root: Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the V4 supplemental smoke or formal support collection."""
    protocol_value = protocol()
    del protocol_value
    output = Path(output)
    checkpoint = Path(checkpoint)
    v3_manifest = Path(v3_manifest)
    v3_train_support = Path(v3_train_support)
    v3_dev_support = Path(v3_dev_support)
    history_root = Path(history_root)
    _require(not output.exists(), f"V4 supplement output already exists: {output}")
    _validate_seed_groups(smoke=smoke)
    v3_binding, retained_rows, v3_manifest_value, candidate, checkpoint_payload = _validate_v3_inputs(
        checkpoint, v3_manifest, v3_train_support, v3_dev_support,
    )
    if not smoke:
        _require(history_root.is_dir(), f"V4 supplement history root missing: {history_root}")
        _require(not (checkpoint.parent / RESERVATION_MARKER).exists(),
                 f"V4 supplement consumed marker already exists: {checkpoint.parent / RESERVATION_MARKER}")
    output.mkdir(parents=True, exist_ok=False)

    plan = _planned_episodes(smoke=smoke)
    marker: Path | None = None
    if not smoke:
        marker, registry = _reserve_formal(checkpoint, history_root, output, v3_manifest)
    else:
        registry = {
            "historical_seed_count": None,
            "checked_audited_seed_range": False,
            "future_qualification_marker_mutated": False,
        }
    _write(output / "pre_reset_registry.json", {
        "protocol_sha256": PROTOCOL_SHA or _sha(PROTOCOL),
        "smoke": smoke,
        "planned_episodes": plan,
        "formal_seed_reservation": not smoke,
        "consumed_marker_created": not smoke,
        "historical_registry": registry,
        "v3_manifest_sha256": V3_MANIFEST_SHA,
        "v3_train_support_sha256": V3_TRAIN_SUPPORT_SHA,
        "v3_dev_support_sha256": V3_DEV_SUPPORT_SHA,
        "v3_rows_opened_after_validation": True,
        "support_v2_rows_mixed": False,
        "failed_qualification_rows_mixed": False,
        "failed_qualification_labels_mixed": False,
        "future_qualification_marker_mutated": False,
    })

    records: list[dict[str, Any]] = [dict(record) for record in v3_manifest_value["records"]]
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [dict(row) for row in retained_rows["train"]],
        "dev": [dict(row) for row in retained_rows["dev"]],
    }
    supplement_rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
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
                "split": split,
                "seed": seed,
                "scenario": scenario,
                "requested": int(scenario.removeprefix("flow")),
                "completed": 0,
                "episode_execution_complete": False,
                "exception": repr(exc),
                "gates": {"collection_exception": False},
            }
            records.append(summary)
            supplement_records.append(summary)
            _write(output / "progress.json", {
                "completed_supplement_episodes": len(supplement_records),
                "total_supplement_episodes": len(plan),
                "retained_v3_rows": {key: len(value) for key, value in retained_rows.items()},
                "supplement_records": supplement_records,
            })
            break
        summary = dict(summary)
        summary["rows"] = len(rows)
        summary["support_collection_source"] = "v4_supplement"
        records.append(summary)
        supplement_records.append(summary)
        supplement_rows_by_split[split].extend(dict(row) for row in rows)
        rows_by_split[split].extend(dict(row) for row in rows)
        _write(output / "progress.json", {
            "completed_supplement_episodes": len(supplement_records),
            "total_supplement_episodes": len(plan),
            "retained_v3_rows": {key: len(value) for key, value in retained_rows.items()},
            "new_supplement_rows": {
                key: len(value) for key, value in supplement_rows_by_split.items()
            },
            "supplement_records": supplement_records,
        })
        if any(int(value) for value in summary.get("safety", {}).values()):
            break

    supplement_hashes = {
        split: _write_jsonl_immutable(
            output / f"{split}_support_supplement.jsonl", supplement_rows_by_split[split]
        )
        for split in ("train", "dev")
    }
    combined = _combined_identity(rows_by_split)
    combined_hashes = {
        split: _write_jsonl_immutable(output / f"{split}_support.jsonl", rows_by_split[split])
        for split in ("train", "dev")
    }
    planned_train = tuple(RETAINED_V3_TRAIN_SEEDS) + (SMOKE_SEEDS[0],) if smoke else (
        tuple(RETAINED_V3_TRAIN_SEEDS) + tuple(FORMAL_TRAIN_SEEDS)
    )
    planned_dev = tuple(RETAINED_V3_DEV_SEEDS) + (SMOKE_SEEDS[1],) if smoke else (
        tuple(RETAINED_V3_DEV_SEEDS) + tuple(FORMAL_DEV_SEEDS)
    )
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
        "v3_manifest_sha256": V3_MANIFEST_SHA,
        "v3_train_support_sha256": V3_TRAIN_SUPPORT_SHA,
        "v3_dev_support_sha256": V3_DEV_SUPPORT_SHA,
        "v3_protocol_sha256": V3_PROTOCOL_SHA,
        "v3_consumed_marker_sha256": V3_CONSUMED_MARKER_SHA,
        "checkpoint_payload": {
            "stage": checkpoint_payload.get("stage"),
            "architecture_id": checkpoint_payload.get("architecture_id"),
            "temperature": checkpoint_payload.get("temperature"),
            "qualification_only": checkpoint_payload.get("qualification_only"),
            "checkpoint_promotable": checkpoint_payload.get("checkpoint_promotable"),
        },
        "predecessor_bindings": {
            "v3_manifest_sha256": V3_MANIFEST_SHA,
            "v3_train_support_sha256": V3_TRAIN_SUPPORT_SHA,
            "v3_dev_support_sha256": V3_DEV_SUPPORT_SHA,
            "v3_protocol_sha256": V3_PROTOCOL_SHA,
            "v3_consumed_marker_sha256": V3_CONSUMED_MARKER_SHA,
        },
        "bindings": {
            "v3_manifest": v3_binding["manifest"],
            "v3_consumed_marker": v3_binding["marker"],
            "v3_rows_opened_after_validation": True,
            "v3_incomplete_support_rows_retained": True,
            "support_v2_rows_mixed": False,
            "failed_qualification_rows_mixed": False,
            "failed_qualification_labels_mixed": False,
            "future_qualification_marker_mutated": False,
        },
        "records": records,
        "retained_v3_records": len(v3_manifest_value["records"]),
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
        "training_called": False,
        "optimizer_called": False,
        "student_authoritative": True,
        "teacher_label_only": True,
        "teacher_replaced_behavior": False,
        "v3_incomplete_support_rows_retained": True,
        "v3_retained_rows": {key: len(value) for key, value in retained_rows.items()},
        "supplement_new_rows": {
            key: len(value) for key, value in supplement_rows_by_split.items()
        },
        "supplement_train_support_sha256": supplement_hashes["train"],
        "supplement_dev_support_sha256": supplement_hashes["dev"],
        "combined_train_support_sha256": combined_hashes["train"],
        "combined_dev_support_sha256": combined_hashes["dev"],
        "support_v2_rows_mixed": False,
        "failed_qualification_rows_mixed": False,
        "failed_qualification_labels_mixed": False,
        "future_qualification_rollouts": 0,
        "future_qualification_consumed_marker": False,
        "future_qualification_marker_mutated": False,
        "retained_v3_marker": str(checkpoint.parent / V3_RESERVATION_MARKER),
        "formal_marker": str(marker) if marker is not None else None,
        "formal_marker_created": marker is not None,
        "support_files": {
            "retained_v3": {
                "train": str(v3_train_support),
                "dev": str(v3_dev_support),
                "rows": {"train": len(retained_rows["train"]), "dev": len(retained_rows["dev"])},
                "sha256": {"train": V3_TRAIN_SUPPORT_SHA, "dev": V3_DEV_SUPPORT_SHA},
            },
            "supplement": {
                "train": str(output / "train_support_supplement.jsonl"),
                "dev": str(output / "dev_support_supplement.jsonl"),
                "rows": {key: len(value) for key, value in supplement_rows_by_split.items()},
                "sha256": supplement_hashes,
            },
            "combined": {
                "train": str(output / "train_support.jsonl"),
                "dev": str(output / "dev_support.jsonl"),
                "rows": {key: len(value) for key, value in rows_by_split.items()},
                "sha256": combined_hashes,
                "row_order": "retained_v3_rows_then_supplement_rows",
            },
            "sha256": {
                "v3_train": V3_TRAIN_SUPPORT_SHA,
                "v3_dev": V3_DEV_SUPPORT_SHA,
                "supplement_train": supplement_hashes["train"],
                "supplement_dev": supplement_hashes["dev"],
                "combined_train": combined_hashes["train"],
                "combined_dev": combined_hashes["dev"],
            },
        },
        "source_hashes": {
            "support_runner": _sha(Path(__file__)),
            "support_protocol": _sha(PROTOCOL),
            "v3_support_runner": _sha(Path(v3.__file__)),
            "v3_support_protocol": V3_PROTOCOL_SHA,
            "candidate_checkpoint": CANDIDATE_SHA,
            "v3_manifest": V3_MANIFEST_SHA,
            "v3_train_support": V3_TRAIN_SUPPORT_SHA,
            "v3_dev_support": V3_DEV_SUPPORT_SHA,
            "v3_consumed_marker": V3_CONSUMED_MARKER_SHA,
        },
    }
    _write(output / "manifest.json", result)
    print(json.dumps({
        "stage": STAGE,
        "SUPPORT_COLLECTION": result["SUPPORT_COLLECTION"],
        "smoke": smoke,
        "retained_rows": result["v3_retained_rows"],
        "supplement_rows": result["supplement_new_rows"],
        "combined_rows": {key: len(value) for key, value in rows_by_split.items()},
    }, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--v3-manifest", type=Path, required=True)
    parser.add_argument("--v3-train-support", type=Path, required=True)
    parser.add_argument("--v3-dev-support", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    torch.set_num_threads(1)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2), flush=True)
