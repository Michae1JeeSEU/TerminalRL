"""Bounded offline coverage of the original teacher checked prefixes.

The collector joins the nine validated train cycles with their saved labels and
the two existing native witness streams.  Selection is deterministic and uses
only candidates already present in ``teacher_audit``.  ``--prepare-only``
creates the complete selection and budget record without loading snapshots or
calling the native planner.  Full execution reuses existing witness rows and
delegates new candidate groups to one process per parent seed.

This module is train-only.  It has no Q, checkpoint, optimizer, environment
reset, or environment step entry point.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from . import p3_v25_course_data as old_data
from . import p3_v26_course_data as additional_data
from . import p3_v26_supplemental_data as supplemental_data
from . import p3_neighbor_witness_data as base


PREFIX_PROTOCOL = "p3_train_neighbor_prefix_coverage_20261010_v1"
OLD_TRAIN_CASES = tuple(old_data.TRAIN_CASES)
ADDITIONAL_TRAIN_CASES = tuple(additional_data.TRAIN_CASES)
SUPPLEMENTAL_TRAIN_CASES = tuple(supplemental_data.TRAIN_CASES)
ALL_TRAIN_SEEDS = tuple(seed for seed, _ in (
    OLD_TRAIN_CASES + ADDITIONAL_TRAIN_CASES + SUPPLEMENTAL_TRAIN_CASES
))
INTERFACES = ("o2m", "m2i", "i2f")
NON_FINAL_INTERFACES = {"o2m", "m2i"}
MAX_CANDIDATE_REQUESTS_PER_PARENT = 150
MAX_PLAN_CALLS_PER_PARENT = 900
MAX_CANDIDATE_REQUESTS = 1350
MAX_PLAN_CALLS = 8100
FULL_PARENT_WALL_SECONDS = 1800
FULL_ROOT_WALL_SECONDS = 5700
SMOKE_WALL_SECONDS = 120
SMOKE_OUTER_SECONDS = 180
FULL_OUTER_SECONDS = 6000
PROCESS_STOP_JOIN_SECONDS = 5.0


class PrefixCoverageError(ValueError):
    """Raised when a validated input or deterministic selection is invalid."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise PrefixCoverageError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        _require(isinstance(value, dict), f"{path}:{number}: expected object")
        rows.append(value)
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _safe_relative(episode: Path, relative: str) -> Path:
    value = Path(str(relative))
    _require(not value.is_absolute() and ".." not in value.parts, f"invalid label path: {relative}")
    path = (episode / value).resolve()
    _require(path == episode.resolve() or episode.resolve() in path.parents, f"label escapes episode: {relative}")
    return path


def _case_identity(source_group: str, index: int) -> tuple[int, str]:
    if source_group == "old":
        return OLD_TRAIN_CASES[index]
    if source_group == "additional":
        return ADDITIONAL_TRAIN_CASES[index]
    if source_group == "supplemental":
        return SUPPLEMENTAL_TRAIN_CASES[index]
    raise PrefixCoverageError(f"unknown source group: {source_group}")


def _validated_case(directory: Path, *, source_group: str, index: int) -> dict[str, Any]:
    seed, scenario = _case_identity(source_group, index)
    directory = Path(directory)
    if source_group == "old":
        context_path, report = old_data.validated_case(directory, seed, scenario, "train")
    elif source_group == "additional":
        context_path, report = additional_data.validated_case(directory, index + 1)
    else:
        context_path, report = supplemental_data.validated_case(directory, index + 1)

    episode = directory / f"seed_{seed}"
    selected_path = episode / "selected.json"
    annotation_path = episode / "annotation_result.json"
    _require(selected_path.is_file(), f"selected.json missing: {selected_path}")
    _require(annotation_path.is_file(), f"annotation_result.json missing: {annotation_path}")
    selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
    _require(isinstance(selected_payload, list), f"selected.json is not a list: {selected_path}")
    selected_by_id: dict[str, dict[str, Any]] = {}
    for entry in selected_payload:
        _require(isinstance(entry, dict) and isinstance(entry.get("row"), dict),
                 f"invalid selected entry: {selected_path}")
        row = entry["row"]
        decision_id = str(row.get("decision_id", ""))
        _require(decision_id and decision_id not in selected_by_id, f"duplicate selected decision_id: {decision_id}")
        _require(int(row.get("seed")) == seed and row.get("partition") == "train",
                 f"selected identity mismatch: {decision_id}")
        selected_by_id[decision_id] = entry

    rows = _read_jsonl(Path(context_path))
    row_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        decision_id = str(row.get("decision_id", ""))
        _require(decision_id and decision_id not in row_by_id, f"duplicate context decision_id: {decision_id}")
        _require(int(row.get("seed")) == seed and row.get("partition") == "train",
                 f"context identity mismatch: {decision_id}")
        _require(bool(row.get("actor_mask")), f"non-actor row in validated input: {decision_id}")
        _require(decision_id in selected_by_id, f"context row absent from selected.json: {decision_id}")
        row_by_id[decision_id] = row
    _require(set(row_by_id) == set(selected_by_id), f"selected/context decision set differs for seed {seed}")

    annotation_payload = _read_json(annotation_path)
    annotation_entries = annotation_payload.get("entries")
    _require(annotation_payload.get("status") == "COMPLETED" and isinstance(annotation_entries, list),
             f"annotation is incomplete for seed {seed}")
    annotation_by_id: dict[str, dict[str, Any]] = {}
    label_by_id: dict[str, dict[str, Any]] = {}
    label_files: dict[str, str] = {}
    label_sha256: dict[str, str] = {}
    for annotation_entry in annotation_entries:
        _require(isinstance(annotation_entry, dict), f"invalid annotation entry for seed {seed}")
        decision_id = str(annotation_entry.get("decision_id", ""))
        _require(decision_id in row_by_id and decision_id not in annotation_by_id,
                 f"annotation/context join failed: {decision_id}")
        _require(annotation_entry.get("status") == "OK", f"annotation entry is not OK: {decision_id}")
        label_file = str(annotation_entry.get("label_file", ""))
        label_path = _safe_relative(episode, label_file)
        label = _read_json(label_path)
        _require(label.get("status") == "OK" and str(label.get("decision_id")) == decision_id,
                 f"label identity/status mismatch: {decision_id}")
        row = row_by_id[decision_id]
        _require(label.get("teacher_gate") == row.get("teacher_gate") and
                 label.get("teacher_proposal") == row.get("teacher_proposal"),
                 f"teacher label changed: {decision_id}")
        expected_joint = int(row["teacher_proposal"]) + 1 if row.get("teacher_proposal") is not None else 0
        _require(int(label.get("teacher_joint_index", 0)) == expected_joint,
                 f"teacher joint mismatch: {decision_id}")
        audit = label.get("teacher_audit")
        _require(isinstance(audit, list) and audit, f"missing teacher_audit: {decision_id}")
        seen_joints: set[int] = set()
        for item in audit:
            _require(isinstance(item, dict), f"invalid teacher_audit entry: {decision_id}")
            joint = int(item.get("joint_index", 0))
            _require(0 <= joint <= 25 and joint not in seen_joints,
                     f"teacher_audit joint outside 0..25 or duplicated: {decision_id}")
            _require(isinstance(item.get("feasible"), bool), f"teacher_audit feasibility is not bool: {decision_id}")
            seen_joints.add(joint)
        annotation_by_id[decision_id] = annotation_entry
        label_by_id[decision_id] = label
        label_files[decision_id] = label_file
        label_sha256[decision_id] = _sha(label_path)
    _require(set(annotation_by_id) == set(row_by_id), f"annotation does not cover selected rows for seed {seed}")

    # Keep all rows/labels in memory for deterministic preparation.  Worker
    # specifications later retain only paths and selected IDs.
    return {
        "seed": seed,
        "scenario": scenario,
        "source_group": source_group,
        "directory": str(directory),
        "episode": str(episode),
        "rows_path": str(context_path),
        "selected_path": str(selected_path),
        "annotation_path": str(annotation_path),
        "rows_sha256": _sha(Path(context_path)),
        "selected_sha256": _sha(selected_path),
        "annotation_sha256": _sha(annotation_path),
        "label_files": label_files,
        "label_sha256": label_sha256,
        "row_by_id": row_by_id,
        "selected_by_id": selected_by_id,
        "annotation_by_id": annotation_by_id,
        "label_by_id": label_by_id,
        "rows_count": len(rows),
        "selected_count": len(selected_by_id),
        "annotation_count": len(annotation_by_id),
        "label_count": len(label_by_id),
    }


def load_validated_inputs(
    old_directories: Sequence[Path | str],
    additional_directories: Sequence[Path | str],
    supplemental_directories: Sequence[Path | str],
) -> list[dict[str, Any]]:
    """Validate and load exactly the declared nine train cycles."""
    _require(len(old_directories) == 3, "old train directories require exactly three paths")
    _require(len(additional_directories) == 3, "additional train directories require exactly three paths")
    _require(len(supplemental_directories) == 3, "supplemental train directories require exactly three paths")
    inputs: list[dict[str, Any]] = []
    for source_group, directories in (
        ("old", old_directories),
        ("additional", additional_directories),
        ("supplemental", supplemental_directories),
    ):
        for index, directory in enumerate(directories):
            inputs.append(_validated_case(Path(directory), source_group=source_group, index=index))
    _require(tuple(item["seed"] for item in inputs) == ALL_TRAIN_SEEDS, "validated input seed order differs")
    return inputs


def _witness_rows(path: Path, *, source: str, expected_protocol: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(path)
    protocol_path = path / "protocol.json"
    result_path = path / "result.json"
    witness_path = path / "witnesses.jsonl"
    _require(protocol_path.is_file() and result_path.is_file() and witness_path.is_file(),
             f"witness artifact incomplete: {path}")
    protocol = _read_json(protocol_path)
    result = _read_json(result_path)
    _require(protocol.get("protocol") == expected_protocol and not protocol.get("smoke"),
             f"wrong {source} witness protocol")
    _require(result.get("status") == "COMPLETE" and result.get("error") is None and
             _sha(witness_path) == result.get("witness_sha256"),
             f"incomplete or changed {source} witness")
    rows = _read_jsonl(witness_path)
    _require(len(rows) == int(result.get("states", -1)), f"{source} witness state count differs")
    _require(len({str(row.get("decision_id")) for row in rows}) == len(rows),
             f"duplicate {source} witness decision_id")
    return {"source": source, "directory": str(path), "protocol": protocol, "result": result}, rows


def load_witness_reuse(
    old_witness_directory: Path | str,
    supplemental_witness_directory: Path | str,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    """Load the two completed witness streams and index candidate reuse."""
    streams = (
        _witness_rows(Path(old_witness_directory), source="old", expected_protocol=base.PROTOCOL),
        _witness_rows(Path(supplemental_witness_directory), source="supplemental",
                      expected_protocol=base.SUPPLEMENTAL_PROTOCOL),
    )
    reuse: dict[tuple[str, int], dict[str, Any]] = {}
    summary: dict[str, Any] = {"streams": [], "candidate_keys": 0, "records": 0, "plan_calls": 0}
    for stream, rows in streams:
        source = stream["source"]
        summary["streams"].append({
            "source": source,
            "directory": stream["directory"],
            "protocol_sha256": _sha(Path(stream["directory"]) / "protocol.json"),
            "result_sha256": _sha(Path(stream["directory"]) / "result.json"),
            "witness_sha256": stream["result"]["witness_sha256"],
            "records": len(rows),
            "plan_calls": int(stream["result"].get("plan_calls", 0)),
        })
        summary["records"] += len(rows)
        summary["plan_calls"] += int(stream["result"].get("plan_calls", 0))
        for row in rows:
            did = str(row["decision_id"])
            for candidate in row.get("candidates") or []:
                joint = int(candidate.get("joint", 0))
                _require(1 <= joint <= 25, f"witness candidate joint outside 1..25: {did}")
                key = (did, joint)
                value = {"source": source, "row": row, "candidate": candidate}
                if key in reuse:
                    old_value = json.dumps(reuse[key]["candidate"], sort_keys=True)
                    new_value = json.dumps(candidate, sort_keys=True)
                    _require(old_value == new_value, f"conflicting witness candidate reuse: {key}")
                else:
                    reuse[key] = value
    summary["candidate_keys"] = len(reuse)
    return reuse, summary


def _candidate_entries(inputs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in inputs:
        seed = int(item["seed"])
        for decision_id in sorted(item["annotation_by_id"]):
            row = item["row_by_id"][decision_id]
            label = item["label_by_id"][decision_id]
            interface = str((row.get("context") or {}).get("interface", ""))
            _require(interface in INTERFACES, f"unknown interface in train row: {decision_id}")
            audit = label["teacher_audit"]
            for audit_rank, candidate in enumerate(audit):
                joint = int(candidate["joint_index"])
                if joint == 0:
                    continue
                _require(1 <= joint <= 25,
                         f"teacher_audit joint outside 1..25: {decision_id}")
                entries.append({
                    "parent_seed": seed,
                    "source_group": item["source_group"],
                    "interface": interface,
                    "joint": joint,
                    "original_feasible": bool(candidate["feasible"]),
                    "original_reason": candidate.get("reason"),
                    "decision_id": decision_id,
                    "tick": int(row["tick"]),
                    "uid": int(row["uid"]),
                    "audit_rank": audit_rank,
                    "teacher_gate": row.get("teacher_gate"),
                    "teacher_joint": int(label.get("teacher_joint_index", 0)),
                    "label_file": item["label_files"][decision_id],
                })
    _require(entries, "no teacher_audit candidate entries")
    return entries


def _selection_sort_key(entry: Mapping[str, Any]) -> tuple[int, int, int, str]:
    return (int(entry["tick"]), int(entry["uid"]), int(entry["audit_rank"]), str(entry["decision_id"]))


def select_candidates(
    inputs: Sequence[dict[str, Any]],
    reuse: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one lower-median candidate for each required parent group."""
    entries = _candidate_entries(inputs)
    groups: dict[tuple[int, str, int, bool], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        groups[(int(entry["parent_seed"]), str(entry["interface"]), int(entry["joint"]),
               bool(entry["original_feasible"]))].append(entry)

    selected: list[dict[str, Any]] = []
    for group_key in sorted(groups, key=lambda value: (value[0], value[1], value[2], int(value[3]))):
        values = sorted(groups[group_key], key=_selection_sort_key)
        chosen = values[(len(values) - 1) // 2]
        key = (str(chosen["decision_id"]), int(chosen["joint"]))
        reused = key in reuse
        selected.append({
            **chosen,
            "group_size": len(values),
            "median_policy": "lower_median_after_tick_numeric_uid_audit_order",
            "candidate_key": f"{chosen['decision_id']}|joint={chosen['joint']}",
            "reuse": reused,
            "reuse_source": None if not reused else reuse[key]["source"],
            "action": "reuse_existing_witness" if reused else "new_native_witness",
        })

    per_parent: dict[str, dict[str, int]] = defaultdict(lambda: {
        "candidate_requests": 0,
        "ring_candidate_requests": 0,
        "final_candidate_requests": 0,
        "plan_upper_bound": 0,
        "new_plan_upper_bound": 0,
        "reused_candidate_requests": 0,
        "new_candidate_requests": 0,
    })
    for item in selected:
        seed = str(item["parent_seed"])
        weight = 1 if item["interface"] in NON_FINAL_INTERFACES else 6
        value = per_parent[seed]
        value["candidate_requests"] += 1
        value["plan_upper_bound"] += weight
        value["new_plan_upper_bound"] += weight * int(not item["reuse"])
        value["reused_candidate_requests"] += int(item["reuse"])
        value["new_candidate_requests"] += int(not item["reuse"])
        value["ring_candidate_requests"] += int(item["interface"] in NON_FINAL_INTERFACES)
        value["final_candidate_requests"] += int(item["interface"] == "i2f")
    for seed, value in per_parent.items():
        _require(value["candidate_requests"] <= MAX_CANDIDATE_REQUESTS_PER_PARENT,
                 f"candidate request budget exceeded for seed {seed}")
        _require(value["plan_upper_bound"] <= MAX_PLAN_CALLS_PER_PARENT,
                 f"plan budget exceeded for seed {seed}")
        _require(value["new_plan_upper_bound"] <= MAX_PLAN_CALLS_PER_PARENT,
                 f"new plan budget exceeded for seed {seed}")
    total = {
        "candidate_requests": sum(item["candidate_requests"] for item in per_parent.values()),
        "plan_upper_bound": sum(item["plan_upper_bound"] for item in per_parent.values()),
        "new_plan_upper_bound": sum(item["new_plan_upper_bound"] for item in per_parent.values()),
        "reused_candidate_requests": sum(item["reused_candidate_requests"] for item in per_parent.values()),
        "new_candidate_requests": sum(item["new_candidate_requests"] for item in per_parent.values()),
    }
    _require(total["candidate_requests"] <= MAX_CANDIDATE_REQUESTS, "global candidate request budget exceeded")
    _require(total["plan_upper_bound"] <= MAX_PLAN_CALLS, "global plan budget exceeded")
    _require(total["new_plan_upper_bound"] <= MAX_PLAN_CALLS, "global new plan budget exceeded")
    counts = {
        "candidate_entries": len(entries),
        "selected_candidate_groups": len(selected),
        "selected_state_ids": len({str(item["decision_id"]) for item in selected}),
        "reused_candidate_groups": total["reused_candidate_requests"],
        "new_candidate_groups": total["new_candidate_requests"],
        "per_parent": dict(sorted(per_parent.items())),
        "total": total,
    }
    return selected, counts


def select_smoke_candidates(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select one new non-default candidate per interface for the smoke."""
    choices: list[dict[str, Any]] = []
    for interface in INTERFACES:
        values = [item for item in selected
                  if item["interface"] == interface and not item["reuse"] and int(item["joint"]) not in {1, 13}]
        values.sort(key=lambda item: (int(item["parent_seed"]), int(item["tick"]),
                                     int(item["uid"]), int(item["audit_rank"]), str(item["decision_id"])))
        _require(values, f"smoke preparation found no new non-default candidate for interface {interface}")
        choices.append(dict(values[0]))
    _require(len({str(item["candidate_key"]) for item in choices}) == len(choices),
             "smoke candidate selection duplicated a candidate")
    _require(len(choices) <= 3 and sum(1 if item["interface"] in NON_FINAL_INTERFACES else 6 for item in choices) <= 18,
             "smoke budget exceeded")
    return choices


def _input_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        key: item[key] for key in (
            "seed", "scenario", "source_group", "directory", "episode", "rows_path",
            "selected_path", "annotation_path", "rows_sha256", "selected_sha256",
            "annotation_sha256", "rows_count", "selected_count", "annotation_count", "label_count",
        )
    }
    summary["label_files"] = dict(item["label_files"])
    summary["label_sha256"] = dict(item["label_sha256"])
    return summary


def _selection_summary(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: item[key] for key in (
        "parent_seed", "source_group", "interface", "joint", "original_feasible", "original_reason",
        "decision_id", "tick", "uid", "audit_rank", "teacher_gate", "teacher_joint", "label_file",
        "group_size", "median_policy", "candidate_key", "reuse", "reuse_source", "action",
    )} for item in selected]


def _state_ids_in_output_order(selected: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return unique state IDs in the order used by parent witness streams."""
    return [decision_id for _, decision_id in sorted({
        (int(item["parent_seed"]), str(item["decision_id"])) for item in selected
    })]


def _root_protocol(
    inputs: Sequence[dict[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    counts: Mapping[str, Any],
    witness_summary: Mapping[str, Any],
    *,
    smoke: bool,
    prepare_only: bool,
) -> dict[str, Any]:
    return {
        "protocol": PREFIX_PROTOCOL,
        "source_group": "prefix_coverage",
        "scope": "smoke" if smoke else "full",
        "prepare_only": bool(prepare_only),
        "smoke": bool(smoke),
        "input_seeds": list(ALL_TRAIN_SEEDS),
        "inputs": [_input_summary(item) for item in inputs],
        "witness_inputs": dict(witness_summary),
        "selected_ids": _state_ids_in_output_order(selected),
        "selected_candidate_ids": [str(item["candidate_key"]) for item in selected],
        "selected_candidate_count": len(selected),
        "selection_counts": dict(counts),
        "upper_bound": {
            "per_parent_candidate_requests": MAX_CANDIDATE_REQUESTS_PER_PARENT,
            "per_parent_plan_calls": MAX_PLAN_CALLS_PER_PARENT,
            "global_candidate_requests": MAX_CANDIDATE_REQUESTS,
            "global_plan_calls": MAX_PLAN_CALLS,
            "ring_plan_weight": 1,
            "final_plan_weight": 6,
        },
        "wall_seconds_per_parent": FULL_PARENT_WALL_SECONDS,
        "root_wall_seconds": FULL_ROOT_WALL_SECONDS,
        "outer_timeout_seconds": FULL_OUTER_SECONDS,
        "smoke_wall_seconds": SMOKE_WALL_SECONDS,
        "smoke_outer_timeout_seconds": SMOKE_OUTER_SECONDS,
        "max_workers": 1 if smoke else 4,
        "resets": 0,
        "steps": 0,
        "snapshot_reads": 0 if prepare_only else None,
        "Q_accessed": False,
        "teacher_queries": 0,
        "original_labels_unchanged": True,
        "unknown_is_not_safe_or_blocked": True,
        "candidate_source": "original teacher_audit checked prefix only",
        "unknown_suffix_policy": "unknown suffixes are never selected or fabricated",
        "reuse_policy": "reuse exact (decision_id,joint) from old or supplemental witness; no new query",
        "execution_environment": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "process_model": "spawn"},
    }


def _runtime_bundle(item: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = _read_jsonl(Path(str(item["rows_path"])))
    row_by_id = {str(row["decision_id"]): row for row in rows}
    selected_payload = json.loads(Path(str(item["selected_path"])).read_text(encoding="utf-8"))
    selected_by_id = {str(entry["row"]["decision_id"]): entry for entry in selected_payload}
    labels: dict[str, dict[str, Any]] = {}
    del selected
    for did, label_file in sorted(item["label_files"].items()):
        did = str(did)
        label_path = _safe_relative(Path(str(item["episode"])), str(label_file))
        _require(_sha(label_path) == item["label_sha256"][did], f"runtime label changed: {label_path}")
        labels[did] = _read_json(label_path)
        _require(did in row_by_id and did in selected_by_id, f"runtime decision join missing: {did}")
    _require(_sha(Path(str(item["rows_path"]))) == item["rows_sha256"], f"runtime input changed: {item['rows_path']}")
    _require(_sha(Path(str(item["selected_path"]))) == item["selected_sha256"], f"runtime selected changed: {item['selected_path']}")
    _require(set(row_by_id) == set(selected_by_id) == set(labels),
             f"runtime decision sets differ for seed {item['seed']}")
    return {"item": item, "row_by_id": row_by_id, "selected_by_id": selected_by_id, "labels": labels}


def _merge_state_rows(
    reused_rows: Sequence[Mapping[str, Any]],
    new_state: Mapping[str, Any] | None,
    selected_for_state: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if new_state is not None:
        state = dict(new_state)
    elif reused_rows:
        state = dict(reused_rows[0])
    else:
        raise PrefixCoverageError("cannot merge an empty prefix coverage state")
    expected = {int(item["joint"]): item for item in selected_for_state}
    candidate_map: dict[int, dict[str, Any]] = {}
    for row in reused_rows:
        for candidate in row.get("candidates") or []:
            joint = int(candidate["joint"])
            if joint in expected:
                candidate_map[joint] = dict(candidate)
    if new_state is not None:
        for candidate in new_state.get("candidates") or []:
            joint = int(candidate["joint"])
            _require(joint not in candidate_map, f"duplicate merged candidate: {state.get('decision_id')} joint {joint}")
            candidate_map[joint] = dict(candidate)
    _require(set(candidate_map) == set(expected), f"merged candidate set differs: {state.get('decision_id')}")
    state["candidates"] = [candidate_map[int(item["joint"])]
                            for item in sorted(selected_for_state, key=lambda value: int(value["audit_rank"]))]
    state["plan_calls"] = sum(len(candidate.get("destinations") or []) for candidate in state["candidates"])
    return state


def _run_items(
    output: Path,
    items: Sequence[Mapping[str, Any]],
    inputs_by_seed: Mapping[str, Mapping[str, Any]],
    reuse_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    wall_seconds: int,
    max_candidate_requests: int,
    max_plan_calls: int,
) -> dict[str, Any]:
    if output.exists():
        _require(output.is_dir() and not (output / "result.json").exists(),
                 f"refuse overwrite of parent output: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "status": "RUNNING", "error": None, "states": 0, "candidate_requests": 0,
        "new_candidate_requests": 0, "reused_candidate_requests": 0, "plan_calls": 0,
        "new_plan_calls": 0, "reused_plan_calls": 0, "elapsed_seconds": 0.0,
        "resets": 0, "steps": 0, "teacher_queries": 0,
    }
    _write(output / "result.json", result)
    started = time.monotonic()
    runtime_cache: dict[int, dict[str, Any]] = {}
    try:
        grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
        for item in items:
            grouped[(int(item["parent_seed"]), str(item["decision_id"]))].append(item)
        with (output / "witnesses.jsonl").open("x", encoding="utf-8") as stream:
            for (seed, decision_id), selected_for_state in sorted(grouped.items()):
                _require(time.monotonic() - started < wall_seconds, "parent wall budget exhausted")
                input_item = inputs_by_seed[str(seed)]
                if seed not in runtime_cache:
                    runtime_cache[seed] = _runtime_bundle(input_item, selected_for_state)
                runtime = runtime_cache[seed]
                did = str(decision_id)
                row = dict(runtime["row_by_id"][did])
                entry = runtime["selected_by_id"][did]
                label = runtime["labels"][did]
                reused = list(reuse_candidates.get(did, ()))
                selected_joints = {int(item["joint"]) for item in selected_for_state}
                for record in reused:
                    _require(int(record["joint"]) in selected_joints,
                             f"reuse candidate is outside selected state: {did}")
                reused_rows = [record["row"] for record in reused]
                selected_new = [item for item in selected_for_state
                                if not item["reuse"]]
                selected_new.sort(key=lambda value: int(value["audit_rank"]))
                new_state: dict[str, Any] | None = None
                if selected_new:
                    _require(result["candidate_requests"] + len(selected_new) <= max_candidate_requests,
                             "parent candidate request budget exhausted")
                    row["pre_state_hash"] = entry["row"]["pre_state_hash"]
                    requested_joints = [int(item["joint"]) for item in selected_new]
                    remaining = max_plan_calls - int(result["new_plan_calls"])
                    _require(remaining >= 0, "parent plan budget exhausted")
                    new_state = base.witness_state(
                        Path(str(input_item["episode"])), entry, row, label,
                        remaining_calls=remaining, requested_joints=requested_joints,
                    )
                    _require(time.monotonic() - started <= wall_seconds,
                             "parent wall budget exhausted")
                    result["new_candidate_requests"] += len(selected_new)
                    result["new_plan_calls"] += int(new_state.get("plan_calls", 0))
                result["reused_candidate_requests"] += len(reused)
                _merge_state_rows(reused_rows, new_state, selected_for_state)
                result["candidate_requests"] += len(selected_new)
                result["reused_plan_calls"] += sum(
                    len(record.get("candidate", {}).get("destinations") or [])
                    for record in reused
                )
                result["plan_calls"] = int(result["new_plan_calls"]) + int(result["reused_plan_calls"])
                _require(result["plan_calls"] <= max_plan_calls, "parent plan budget exhausted")
                if selected_new:
                    _require(new_state is not None, "new candidate state missing")
                    result["states"] += 1
                    stream.write(json.dumps(new_state, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                result["elapsed_seconds"] = time.monotonic() - started
                _write(output / "result.json", result)
        _require(time.monotonic() - started <= wall_seconds, "parent wall budget exhausted")
        result["status"] = "COMPLETE"
        result["witness_sha256"] = _sha(output / "witnesses.jsonl")
    except Exception as error:  # preserve the partial stream for diagnosis; never retry here
        result["status"] = "INCOMPLETE"
        result["error"] = repr(error)
    result["elapsed_seconds"] = time.monotonic() - started
    _write(output / "result.json", result)
    return result


def _run_parent(spec: Mapping[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    output = Path(str(spec["output"]))
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(spec["protocol"])
    protocol["parent_seed"] = int(spec["parent_seed"])
    protocol["selected_ids"] = _state_ids_in_output_order(spec["selected"])
    protocol["selected_candidate_ids"] = [str(item["candidate_key"]) for item in spec["selected"]]
    _write(output / "protocol.json", protocol)
    result = _run_items(
        output, spec["selected"], {str(spec["parent_seed"]): spec["input"]},
        spec["reuse_candidates"], wall_seconds=FULL_PARENT_WALL_SECONDS,
        max_candidate_requests=MAX_CANDIDATE_REQUESTS_PER_PARENT,
        max_plan_calls=MAX_PLAN_CALLS_PER_PARENT,
    )
    return {"parent_seed": int(spec["parent_seed"]), "output": str(output), **result}


def merge_parent_results(
    output: Path | str,
    parent_summaries: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge only complete parent streams; propagate any child failure."""
    output = Path(output)
    failed = [
        {"parent_seed": item.get("parent_seed"), "status": item.get("status"), "error": item.get("error")}
        for item in parent_summaries if item.get("status") != "COMPLETE"
    ]
    if failed:
        result = {"status": "INCOMPLETE", "error": f"parent failures: {failed}", "parents": list(parent_summaries),
                  "states": 0, "candidate_requests": 0, "plan_calls": 0, "resets": 0, "steps": 0,
                  "teacher_queries": 0}
        _write(output / "result.json", result)
        return result
    expected_ids = _state_ids_in_output_order(selected)
    rows_by_id: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    totals = {key: 0 for key in ("states", "candidate_requests", "new_candidate_requests",
                                 "reused_candidate_requests", "plan_calls", "new_plan_calls", "reused_plan_calls")}
    for summary in sorted(parent_summaries, key=lambda item: int(item["parent_seed"])):
        path = Path(str(summary["output"])) / "witnesses.jsonl"
        _require(path.is_file(), f"parent witness output missing: {path}")
        parent_rows = _read_jsonl(path)
        for row in parent_rows:
            did = str(row["decision_id"])
            _require(did not in seen_ids, f"duplicate parent state: {did}")
            seen_ids.add(did)
            rows_by_id[did] = row
        for key in totals:
            totals[key] += int(summary.get(key, 0))
    _require(seen_ids == set(expected_ids), "parent state set differs from deterministic selection")
    rows = [rows_by_id[did] for did in expected_ids]
    root_witness = output / "witnesses.jsonl"
    with root_witness.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    result = {
        "status": "COMPLETE", "error": None, **totals, "parents": list(parent_summaries),
        "elapsed_seconds": sum(float(item.get("elapsed_seconds", 0.0)) for item in parent_summaries),
        "resets": 0, "steps": 0, "teacher_queries": 0, "witness_sha256": _sha(root_witness),
    }
    _write(output / "result.json", result)
    return result


def _incomplete_root_result(
    output: Path | str,
    parent_summaries: Sequence[Mapping[str, Any]],
    error: str,
) -> dict[str, Any]:
    """Persist an explicit terminal root failure without creating a witness merge."""
    result = {
        "status": "INCOMPLETE", "error": str(error), "parents": list(parent_summaries),
        "states": 0, "candidate_requests": 0, "plan_calls": 0,
        "resets": 0, "steps": 0, "teacher_queries": 0,
    }
    _write(Path(output) / "result.json", result)
    return result


def _stop_executor(executor: ProcessPoolExecutor) -> None:
    """Stop only this collector's workers and wait for their bounded teardown."""
    processes = list((getattr(executor, "_processes", {}) or {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + PROCESS_STOP_JOIN_SECONDS
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        process.join(timeout=remaining)
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        if process.is_alive():
            process.join(timeout=1.0)
    executor.shutdown(wait=True, cancel_futures=True)


def _worker_spec(
    root: Path,
    item: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    reuse: Mapping[tuple[str, int], Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    grouped_reuse: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for choice in selected:
        key = (str(choice["decision_id"]), int(choice["joint"]))
        if key in reuse:
            grouped_reuse[str(choice["decision_id"])].append({
                "joint": int(choice["joint"]),
                "row": reuse[key]["row"],
                "candidate": reuse[key]["candidate"],
                "source": reuse[key]["source"],
            })
    return {
        "parent_seed": int(item["seed"]),
        "output": str(root / f"seed_{int(item['seed'])}"),
        "input": _input_summary(item),
        "selected": [dict(choice) for choice in selected],
        "reuse_candidates": dict(grouped_reuse),
        "protocol": dict(protocol),
    }


def collect(
    old_directories: Sequence[Path | str],
    additional_directories: Sequence[Path | str],
    supplemental_directories: Sequence[Path | str],
    old_witness_directory: Path | str,
    supplemental_witness_directory: Path | str,
    output: Path | str,
    *,
    smoke: bool = False,
    prepare_only: bool = False,
) -> dict[str, Any]:
    """Prepare, smoke, or run the declared prefix coverage collection."""
    _require(not (smoke and prepare_only), "--smoke and --prepare-only are mutually exclusive")
    inputs = load_validated_inputs(old_directories, additional_directories, supplemental_directories)
    reuse, witness_summary = load_witness_reuse(old_witness_directory, supplemental_witness_directory)
    selected, counts = select_candidates(inputs, reuse)
    new_selected = [item for item in selected if not item["reuse"]]
    reused_selected = [item for item in selected if item["reuse"]]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = _root_protocol(inputs, new_selected, counts, witness_summary, smoke=smoke,
                              prepare_only=prepare_only)
    protocol["all_selected_candidate_count"] = len(selected)
    protocol["new_selected_candidate_count"] = len(new_selected)
    protocol["reused_candidate_count"] = len(reused_selected)
    protocol["reused_candidate_ids"] = [str(item["candidate_key"]) for item in reused_selected]
    _write(output / "protocol.json", protocol)
    _write(output / "selection.json", {
        "selected": _selection_summary(selected),
        "new": _selection_summary(new_selected),
        "reuse": _selection_summary(reused_selected),
        "counts": counts,
    })
    if prepare_only:
        result = {"status": "PREPARED", "error": None, "states": len(protocol["selected_ids"]),
                  "candidate_requests": 0, "plan_calls": 0, "resets": 0, "steps": 0,
                  "teacher_queries": 0, "snapshot_reads": 0}
        _write(output / "result.json", result)
        return result

    if smoke:
        smoke_selected = select_smoke_candidates(selected)
        protocol["selected_ids"] = _state_ids_in_output_order(smoke_selected)
        protocol["selected_candidate_ids"] = [str(item["candidate_key"]) for item in smoke_selected]
        protocol["selected_candidate_count"] = len(smoke_selected)
        protocol["smoke_selection"] = _selection_summary(smoke_selected)
        _write(output / "protocol.json", protocol)
        inputs_by_seed = {str(item["seed"]): item for item in inputs}
        grouped_reuse: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for choice in smoke_selected:
            key = (str(choice["decision_id"]), int(choice["joint"]))
            if key in reuse:
                grouped_reuse[str(choice["decision_id"])].append({"row": reuse[key]["row"],
                    "candidate": reuse[key]["candidate"], "joint": int(choice["joint"]), "source": reuse[key]["source"]})
        smoke_result = _run_items(output / "smoke", smoke_selected, inputs_by_seed, grouped_reuse,
                                  wall_seconds=SMOKE_WALL_SECONDS, max_candidate_requests=3,
                                  max_plan_calls=18)
        if smoke_result.get("status") == "COMPLETE":
            smoke_result["status"] = "SMOKE_OK"
        smoke_result["outer_timeout_seconds"] = SMOKE_OUTER_SECONDS
        _write(output / "result.json", smoke_result)
        return smoke_result

    protocol["selected_ids"] = _state_ids_in_output_order(new_selected)
    _write(output / "protocol.json", protocol)
    initial = {"status": "RUNNING", "error": None, "states": 0, "candidate_requests": 0,
               "plan_calls": 0, "resets": 0, "steps": 0, "teacher_queries": 0}
    _write(output / "result.json", initial)
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for choice in new_selected:
        by_seed[int(choice["parent_seed"])].append(dict(choice))
    input_by_seed = {int(item["seed"]): item for item in inputs}
    specs = [_worker_spec(output, input_by_seed[seed], choices, reuse, protocol)
             for seed, choices in sorted(by_seed.items())]
    parent_summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    context = mp.get_context("spawn")
    max_workers = min(4, len(specs))
    if max_workers == 0:
        result = merge_parent_results(output, [], new_selected)
        result["outer_timeout_seconds"] = FULL_OUTER_SECONDS
        _write(output / "result.json", result)
        return result
    executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=context)
    futures: dict[Any, int] = {}
    spec_iterator = iter(specs)

    def submit_next() -> bool:
        try:
            spec = next(spec_iterator)
        except StopIteration:
            return False
        futures[executor.submit(_run_parent, spec)] = int(spec["parent_seed"])
        return True

    for _ in range(max_workers):
        if not submit_next():
            break
    stop_reason: str | None = None
    while futures and stop_reason is None:
        remaining = FULL_ROOT_WALL_SECONDS - (time.monotonic() - started)
        if remaining <= 0:
            stop_reason = "root wall budget exhausted"
            break
        done, _ = wait(tuple(futures), timeout=remaining, return_when=FIRST_COMPLETED)
        if not done:
            stop_reason = "root wall budget exhausted"
            break
        completed_successes = 0
        for future in done:
            seed = futures.pop(future)
            try:
                summary = future.result()
            except Exception as error:
                summary = {"parent_seed": seed, "status": "INCOMPLETE", "error": repr(error),
                           "output": str(output / f"seed_{seed}")}
            parent_summaries.append(summary)
            if summary.get("status") != "COMPLETE":
                stop_reason = f"parent {seed} returned {summary.get('status')}"
                break
            completed_successes += 1
        if stop_reason is None:
            for _ in range(completed_successes):
                if not submit_next():
                    break

    if stop_reason is not None:
        for future in futures:
            if not future.done():
                future.cancel()
        _stop_executor(executor)
    else:
        executor.shutdown(wait=True, cancel_futures=True)
    recorded_seeds = {int(item.get("parent_seed", -1)) for item in parent_summaries}
    for future, seed in futures.items():
        if seed in recorded_seeds:
            continue
        if future.cancelled():
            summary = {"parent_seed": seed, "status": "INCOMPLETE",
                       "error": f"NOT_RUN: {stop_reason or 'parent cancelled'}",
                       "output": str(output / f"seed_{seed}")}
        else:
            try:
                summary = future.result()
            except Exception as error:
                summary = {"parent_seed": seed, "status": "INCOMPLETE", "error": repr(error),
                           "output": str(output / f"seed_{seed}")}
        parent_summaries.append(summary)
    recorded_seeds = {int(item.get("parent_seed", -1)) for item in parent_summaries}
    if stop_reason is not None:
        for spec in specs:
            seed = int(spec["parent_seed"])
            if seed not in recorded_seeds:
                parent_summaries.append({
                    "parent_seed": seed, "status": "INCOMPLETE",
                    "error": f"NOT_RUN: {stop_reason}", "output": str(spec["output"]),
                })
                recorded_seeds.add(seed)
        result = _incomplete_root_result(output, parent_summaries, stop_reason)
        result["outer_timeout_seconds"] = FULL_OUTER_SECONDS
        _write(output / "result.json", result)
        return result
    if recorded_seeds != {int(spec["parent_seed"]) for spec in specs}:
        stop_reason = "root scheduler ended before all parents completed"
        result = _incomplete_root_result(output, parent_summaries, stop_reason)
        result["outer_timeout_seconds"] = FULL_OUTER_SECONDS
        _write(output / "result.json", result)
        return result
    result = merge_parent_results(output, parent_summaries, new_selected)
    result["outer_timeout_seconds"] = FULL_OUTER_SECONDS
    _write(output / "result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-train-directories", nargs=3, type=Path, required=True)
    parser.add_argument("--additional-train-directories", nargs=3, type=Path, required=True)
    parser.add_argument("--supplemental-train-directories", nargs=3, type=Path, required=True)
    parser.add_argument("--old-witness-directory", type=Path, required=True)
    parser.add_argument("--supplemental-witness-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.smoke and args.prepare_only:
        _parser().error("--smoke and --prepare-only are mutually exclusive")
    try:
        result = collect(
            args.old_train_directories, args.additional_train_directories,
            args.supplemental_train_directories, args.old_witness_directory,
            args.supplemental_witness_directory, args.output,
            smoke=args.smoke, prepare_only=args.prepare_only,
        )
    except Exception as error:
        print(json.dumps({"status": "PREPARE_FAILED", "error": repr(error)}, sort_keys=True))
        return 2
    print(json.dumps({"status": result.get("status"), "error": result.get("error")}, sort_keys=True))
    return 0 if result.get("status") in {"PREPARED", "SMOKE_OK", "COMPLETE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
