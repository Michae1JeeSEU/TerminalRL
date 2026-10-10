"""Read-only coverage summary for the existing P3 neighbor witness records.

This script consumes the already hash-checked nine label JSONL files listed in
the nine-cycle loader manifest and the two existing witness JSONL files.  It
does not load snapshots or call a planner/teacher.  The output is intentionally
limited to coverage and deterministic candidate selection metadata.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / ".tmp" / "p3_nine_cycle_loader_check_20261009.json"
OLD_WITNESS = ROOT / ".tmp" / "p3_neighbor_witness_train_20261009" / "witnesses.jsonl"
SUPP_WITNESS = (
    ROOT.parent
    / "training_outputs"
    / "terminal"
    / "p3_neighbor_supplemental_witness_20261009"
    / "three_cycle"
    / "witnesses.jsonl"
)
OUT = ROOT / ".tmp" / "p3_neighbor_candidate_coverage_20261010.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _reason(value: Any) -> str:
    return "unknown" if value is None else str(value)


def _flag_value(value: Any) -> bool | None:
    if value is True or value == 1 or (isinstance(value, str) and value.lower() in {"1", "true"}):
        return True
    if value is False or value == 0 or (isinstance(value, str) and value.lower() in {"0", "false"}):
        return False
    return None


def _bool_key(value: Any) -> str:
    flag = _flag_value(value)
    if flag is True:
        return "true"
    if flag is False:
        return "false"
    return "unknown"


def _counter(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=str)}


def _new_stats() -> dict[str, Any]:
    return {
        "candidate_records": 0,
        "candidate_feasible_counts": Counter(),
        "candidate_reason_counts": Counter(),
        "teacher_gate_counts": Counter(),
        "candidate_state_ids": set(),
        "target_queries": 0,
        "destination_state_ids": set(),
        "parent_seeds": set(),
        "known_blocked_pairs": 0,
        "known_safe_pairs": 0,
        "other_pair_records": 0,
        "unknown_pair_records": 0,
        "destination_without_pairs": 0,
        "global_no_destination_candidates": 0,
        "global_reason_counts": Counter(),
        "pair_reason_counts": Counter(),
    }


def _add_candidate(
    table: dict[str, dict[str, Any]],
    row: dict[str, Any],
    candidate: dict[str, Any],
    source: str,
) -> None:
    del source  # kept in the caller's event stream; no source inference here
    interface = str(row.get("interface", "unknown"))
    joint = int(candidate.get("joint", 0))
    if joint == 0:
        return
    key = f"{interface}|joint={joint}"
    stats = table.setdefault(key, _new_stats())
    stats["candidate_records"] += 1
    stats["candidate_feasible_counts"][_bool_key(candidate.get("original_feasible"))] += 1
    stats["teacher_gate_counts"][_bool_key(row.get("teacher_gate"))] += 1
    original_reason = _reason(candidate.get("original_reason"))
    stats["candidate_reason_counts"][original_reason] += 1
    decision_id = str(row.get("decision_id", ""))
    stats["candidate_state_ids"].add(decision_id)
    seed = int(row["seed"])
    stats["parent_seeds"].add(seed)

    destinations = candidate.get("destinations") or []
    if not destinations:
        stats["global_no_destination_candidates"] += 1
        global_reason = _reason(candidate.get("global_reason"))
        stats["global_reason_counts"][global_reason] += 1
        return

    for destination in destinations:
        stats["target_queries"] += 1
        stats["destination_state_ids"].add(decision_id)
        pairs = destination.get("pairs") or []
        if not pairs:
            stats["destination_without_pairs"] += 1
            continue
        for pair in pairs:
            blocked = pair.get("blocked")
            approved = pair.get("approved")
            pair_reason = _reason(pair.get("reason"))
            stats["pair_reason_counts"][pair_reason] += 1
            blocked_flag = _flag_value(blocked)
            approved_flag = _flag_value(approved)
            if blocked_flag is True:
                stats["known_blocked_pairs"] += 1
            elif blocked_flag is False and approved_flag is True:
                stats["known_safe_pairs"] += 1
            elif blocked_flag is False and approved_flag is False:
                stats["other_pair_records"] += 1
            else:
                stats["unknown_pair_records"] += 1


def _serial_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_records": int(stats["candidate_records"]),
        "candidate_feasible_counts": _counter(stats["candidate_feasible_counts"]),
        "candidate_reason_counts": _counter(stats["candidate_reason_counts"]),
        "teacher_gate_counts": _counter(stats["teacher_gate_counts"]),
        "candidate_distinct_states": len(stats["candidate_state_ids"]),
        "target_queries": int(stats["target_queries"]),
        "distinct_states": len(stats["destination_state_ids"]),
        "parent_seeds": sorted(stats["parent_seeds"]),
        "known_blocked_pairs": int(stats["known_blocked_pairs"]),
        "known_safe_pairs": int(stats["known_safe_pairs"]),
        "other_pair_records": int(stats["other_pair_records"]),
        "unknown_pair_records": int(stats["unknown_pair_records"]),
        "destination_without_pairs": int(stats["destination_without_pairs"]),
        "global_no_destination_candidates": int(stats["global_no_destination_candidates"]),
        "global_reason_counts": _counter(stats["global_reason_counts"]),
        "pair_reason_counts": _counter(stats["pair_reason_counts"]),
    }


def _merge_stats(parts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    merged = _new_stats()
    for part in parts:
        merged["candidate_records"] += part["candidate_records"]
        merged["candidate_feasible_counts"].update(part["candidate_feasible_counts"])
        merged["candidate_reason_counts"].update(part["candidate_reason_counts"])
        merged["teacher_gate_counts"].update(part["teacher_gate_counts"])
        merged["candidate_state_ids"].update(part["candidate_state_ids"])
        merged["target_queries"] += part["target_queries"]
        merged["destination_state_ids"].update(part["destination_state_ids"])
        merged["parent_seeds"].update(part["parent_seeds"])
        merged["known_blocked_pairs"] += part["known_blocked_pairs"]
        merged["known_safe_pairs"] += part["known_safe_pairs"]
        merged["other_pair_records"] += part["other_pair_records"]
        merged["unknown_pair_records"] += part["unknown_pair_records"]
        merged["destination_without_pairs"] += part["destination_without_pairs"]
        merged["global_no_destination_candidates"] += part["global_no_destination_candidates"]
        merged["global_reason_counts"].update(part["global_reason_counts"])
        merged["pair_reason_counts"].update(part["pair_reason_counts"])
    return merged


def _row_context(row: dict[str, Any]) -> tuple[str, int, int]:
    context = row.get("context") or {}
    interface = str(context.get("interface", "unknown"))
    gate = 1 if _flag_value(row.get("teacher_gate")) is True else 0
    proposal = row.get("teacher_proposal")
    joint = int(proposal) + 1 if proposal is not None else 0
    return interface, gate, joint


def _label_summary(label_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    per_seed: dict[int, dict[str, Any]] = {}
    per_interface: dict[str, Counter[str]] = defaultdict(Counter)
    for row in label_rows:
        seed = int(row["seed"])
        interface, gate, joint = _row_context(row)
        seed_summary = per_seed.setdefault(
            seed,
            {
                "rows": 0,
                "release": 0,
                "wait": 0,
                "interface_counts": defaultdict(Counter),
                "release_joint_counts": Counter(),
            },
        )
        seed_summary["rows"] += 1
        seed_summary["release"] += gate
        seed_summary["wait"] += 1 - gate
        seed_summary["interface_counts"][interface]["release" if gate else "wait"] += 1
        per_interface[interface]["release" if gate else "wait"] += 1
        if gate and joint:
            seed_summary["release_joint_counts"][str(joint)] += 1

    serial_seed: dict[str, Any] = {}
    for seed in sorted(per_seed):
        value = per_seed[seed]
        serial_seed[str(seed)] = {
            "rows": value["rows"],
            "release": value["release"],
            "wait": value["wait"],
            "interface_counts": {
                interface: _counter(counts)
                for interface, counts in sorted(value["interface_counts"].items())
            },
            "release_joint_counts": _counter(value["release_joint_counts"]),
        }
    return serial_seed, {
        interface: _counter(counts) for interface, counts in sorted(per_interface.items())
    }


def _serialize_selection(
    candidate_entries: list[dict[str, Any]], existing_keys: set[tuple[str, int]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[tuple[int, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in candidate_entries:
        key = (
            int(entry["seed"]),
            str(entry["interface"]),
            int(entry["joint"]),
            _bool_key(entry["original_feasible"]),
        )
        groups[key].append(entry)

    selected: list[dict[str, Any]] = []
    reuse_count = 0
    for group_key in sorted(groups):
        entries = sorted(
            groups[group_key],
            key=lambda item: (
                int(item["tick"]),
                int(item["uid"]),
                str(item["decision_id"]),
                str(item["source"]),
            ),
        )
        chosen = entries[(len(entries) - 1) // 2]
        existing_reuse = (str(chosen["decision_id"]), int(chosen["joint"])) in existing_keys
        reuse_count += int(existing_reuse)
        selected.append(
            {
                "parent_seed": int(chosen["seed"]),
                "interface": str(chosen["interface"]),
                "joint": int(chosen["joint"]),
                "original_feasible": chosen["original_feasible"],
                "original_reason": chosen.get("original_reason"),
                "group_size": len(entries),
                "median_policy": "lower_median_after_tick_uid_decision_id_source_sort",
                "selected_source": str(chosen["source"]),
                "selected_decision_id": str(chosen["decision_id"]),
                "selected_tick": int(chosen["tick"]),
                "selected_uid": int(chosen["uid"]),
                "selected_destination_count": int(chosen["destination_count"]),
                "selected_label_file": chosen.get("label_file"),
                "selected_window": chosen.get("window"),
                "selected_audit_rank": chosen.get("audit_rank"),
                "existing_witness_reuse": bool(existing_reuse),
                "action": "reuse_existing_witness" if existing_reuse else "new_teacher_audit_candidate",
            }
        )
    return selected, {
        "groups": len(selected),
        "candidate_entries": len(candidate_entries),
        "groups_with_existing_witness_reuse": reuse_count,
        "groups_without_existing_witness_reuse": len(selected) - reuse_count,
    }


def _new_teacher_stats() -> dict[str, Any]:
    return {
        "candidate_records": 0,
        "feasible_counts": Counter(),
        "reason_counts": Counter(),
        "teacher_gate_counts": Counter(),
        "decision_ids": set(),
        "label_files": set(),
        "parent_seeds": set(),
    }


def _add_teacher_stats(
    table: dict[str, dict[str, Any]], entry: dict[str, Any]
) -> None:
    key = f"{entry['interface']}|joint={int(entry['joint'])}"
    stats = table.setdefault(key, _new_teacher_stats())
    stats["candidate_records"] += 1
    stats["feasible_counts"][_bool_key(entry.get("original_feasible"))] += 1
    stats["reason_counts"][_reason(entry.get("original_reason"))] += 1
    stats["teacher_gate_counts"][_bool_key(entry.get("teacher_gate"))] += 1
    stats["decision_ids"].add(str(entry["decision_id"]))
    stats["label_files"].add(str(entry["label_file"]))
    stats["parent_seeds"].add(int(entry["seed"]))


def _serial_teacher_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_records": int(stats["candidate_records"]),
        "feasible_counts": _counter(stats["feasible_counts"]),
        "reason_counts": _counter(stats["reason_counts"]),
        "teacher_gate_counts": _counter(stats["teacher_gate_counts"]),
        "distinct_decision_ids": len(stats["decision_ids"]),
        "distinct_label_files": len(stats["label_files"]),
        "parent_seeds": sorted(stats["parent_seeds"]),
    }


def _load_teacher_audit_entries(
    input_files: list[dict[str, Any]],
    context_by_key: dict[tuple[int, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    label_status_counts: Counter[str] = Counter()
    audit_count = 0
    zero_joint_count = 0
    mismatches: list[dict[str, Any]] = []
    annotation_file_count = 0
    annotation_entry_count = 0
    for item in input_files:
        seed = int(item["seed"])
        seed_dir = Path(str(item["directory"])) / f"seed_{seed}"
        annotation_path = seed_dir / "annotation_result.json"
        if not annotation_path.exists():
            raise FileNotFoundError(annotation_path)
        annotation_file_count += 1
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        status_counts[str(annotation.get("status"))] += 1
        if annotation.get("status") != "COMPLETED":
            mismatches.append(
                {"seed": seed, "kind": "annotation_status", "value": annotation.get("status")}
            )
        for annotation_entry in annotation.get("entries") or []:
            annotation_entry_count += 1
            label_file = str(annotation_entry["label_file"])
            label_path = seed_dir / label_file
            if not label_path.exists():
                raise FileNotFoundError(label_path)
            label = json.loads(label_path.read_text(encoding="utf-8"))
            label_status = str(label.get("status"))
            label_status_counts[label_status] += 1
            decision_id = str(label.get("decision_id", ""))
            context = context_by_key.get((seed, decision_id))
            if context is None:
                mismatches.append(
                    {"seed": seed, "kind": "missing_context_join", "decision_id": decision_id}
                )
                continue
            context_data = context.get("context") or {}
            checks = {
                "annotation_entry_status": annotation_entry.get("status") == "OK",
                "label_status": label_status == "OK",
                "decision_id": str(annotation_entry.get("decision_id")) == decision_id == str(context.get("decision_id")),
                "teacher_gate": (
                    annotation_entry.get("teacher_gate")
                    == label.get("teacher_gate")
                    == context.get("teacher_gate")
                ),
                "teacher_proposal": (
                    annotation_entry.get("teacher_proposal")
                    == label.get("teacher_proposal")
                    == context.get("teacher_proposal")
                ),
                "teacher_joint_index": (
                    int(label.get("teacher_joint_index", 0))
                    == (int(label["teacher_proposal"]) + 1 if label.get("teacher_proposal") is not None else 0)
                ),
                "interface": bool(context_data.get("interface")),
            }
            for check_name, passed in checks.items():
                if not passed and len(mismatches) < 20:
                    mismatches.append(
                        {
                            "seed": seed,
                            "kind": check_name,
                            "decision_id": decision_id,
                            "label_file": label_file,
                            "annotation_value": annotation_entry.get(check_name),
                            "label_value": label.get(check_name),
                            "context_value": context.get(check_name),
                        }
                    )
            for audit_rank, audit in enumerate(label.get("teacher_audit") or []):
                audit_count += 1
                joint = int(audit.get("joint_index", 0))
                if joint == 0:
                    zero_joint_count += 1
                    continue
                entries.append(
                    {
                        "seed": seed,
                        "interface": str(context_data.get("interface", "unknown")),
                        "joint": joint,
                        "original_feasible": audit.get("feasible"),
                        "original_reason": audit.get("reason"),
                        "decision_id": decision_id,
                        "tick": int(label.get("tick", context.get("tick", 0))),
                        "uid": int(label.get("uid", context.get("uid", 0))),
                        "source": "teacher_audit",
                        "label_file": label_file,
                        "window": int(label.get("window", annotation_entry.get("window", -1))),
                        "audit_rank": audit_rank,
                        "teacher_gate": label.get("teacher_gate"),
                        "teacher_proposal": label.get("teacher_proposal"),
                        "destination_count": 0,
                    }
                )
    consistency = {
        "annotation_files": annotation_file_count,
        "annotation_status_counts": _counter(status_counts),
        "annotation_entries": annotation_entry_count,
        "label_status_counts": _counter(label_status_counts),
        "teacher_audit_entries_total": audit_count,
        "teacher_audit_joint_zero_entries": zero_joint_count,
        "teacher_audit_nonzero_joint_entries": len(entries),
        "context_joins": annotation_entry_count - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:20],
    }
    if mismatches:
        raise ValueError(f"teacher audit consistency checks failed: {mismatches[:3]}")
    return entries, consistency


def _weighted_selection_bounds(selected: list[dict[str, Any]]) -> dict[str, Any]:
    def calculate(items: Iterable[dict[str, Any]]) -> tuple[dict[str, int], dict[str, dict[str, int]], list[str]]:
        per_seed: Counter[str] = Counter()
        by_seed_interface: dict[str, Counter[str]] = defaultdict(Counter)
        unknown_interfaces: list[str] = []
        for item in items:
            interface = str(item["interface"])
            if interface in {"o2m", "m2i"}:
                weight = 1
            elif interface == "i2f":
                weight = 6
            else:
                unknown_interfaces.append(interface)
                continue
            seed = str(item["parent_seed"])
            per_seed[seed] += weight
            by_seed_interface[seed][interface] += weight
        return dict(sorted(per_seed.items())), {
            seed: dict(sorted(counter.items()))
            for seed, counter in sorted(by_seed_interface.items())
        }, sorted(set(unknown_interfaces))

    all_bound, all_by_interface, all_unknown = calculate(selected)
    new_bound, new_by_interface, new_unknown = calculate(
        item for item in selected if not item["existing_witness_reuse"]
    )
    per_parent = {}
    for seed in sorted(set(all_bound) | set(new_bound)):
        total = int(all_bound.get(seed, 0))
        new_total = int(new_bound.get(seed, 0))
        per_parent[seed] = {
            "selected_group_plan_bound": total,
            "new_selected_group_plan_bound": new_total,
            "registered_max_plans_per_parent": 900,
            "selected_bound_le_900": total <= 900,
            "new_bound_le_900": new_total <= 900,
            "selected_weight_by_interface": all_by_interface.get(seed, {}),
            "new_weight_by_interface": new_by_interface.get(seed, {}),
        }
    return {
        "rule": "one plan per selected o2m/m2i group and six plans per selected i2f group",
        "per_parent": per_parent,
        "unknown_interfaces": sorted(set(all_unknown + new_unknown)),
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refuse overwrite: {OUT}")
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)
    if not OLD_WITNESS.exists():
        raise FileNotFoundError(OLD_WITNESS)
    if not SUPP_WITNESS.exists():
        raise FileNotFoundError(SUPP_WITNESS)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    input_files = manifest.get("input_files") or manifest.get("sources")
    if not isinstance(input_files, list) or len(input_files) != 9:
        raise ValueError("loader manifest does not contain exactly nine input files")

    label_rows: list[dict[str, Any]] = []
    context_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    label_sources: list[dict[str, Any]] = []
    for item in input_files:
        source_file = Path(str(item["source_file"]))
        if not source_file.exists():
            raise FileNotFoundError(source_file)
        rows = _read_jsonl(source_file)
        label_rows.extend(rows)
        for row in rows:
            key = (int(row["seed"]), str(row["decision_id"]))
            if key in context_by_key:
                raise ValueError(f"duplicate context decision_id: {key}")
            context_by_key[key] = row
        label_sources.append(
            {
                "seed": int(item.get("seed", rows[0]["seed"] if rows else -1)),
                "source_file": str(source_file),
                "rows": len(rows),
                "source_file_sha256": item.get("source_file_sha256"),
            }
        )

    teacher_entries, teacher_consistency = _load_teacher_audit_entries(
        input_files, context_by_key
    )
    teacher_aggregate: dict[str, dict[str, Any]] = {}
    teacher_per_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for teacher_entry in teacher_entries:
        _add_teacher_stats(teacher_aggregate, teacher_entry)
        _add_teacher_stats(teacher_per_seed[int(teacher_entry["seed"])], teacher_entry)

    old_rows = _read_jsonl(OLD_WITNESS)
    supp_rows = _read_jsonl(SUPP_WITNESS)
    witness_streams = [("old", old_rows), ("supplemental", supp_rows)]

    aggregate: dict[str, dict[str, Any]] = {}
    per_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    candidate_entries: list[dict[str, Any]] = []
    old_keys: set[tuple[str, int]] = set()
    existing_witness_keys: set[tuple[str, int]] = set()
    for source, rows in witness_streams:
        for row in rows:
            interface = str(row.get("interface", "unknown"))
            seed = int(row["seed"])
            row_for_stats = dict(row)
            row_for_stats["interface"] = interface
            for candidate in row.get("candidates") or []:
                joint = int(candidate.get("joint", 0))
                if joint == 0:
                    continue
                _add_candidate(aggregate, row_for_stats, candidate, source)
                seed_table = per_seed[seed]
                _add_candidate(seed_table, row_for_stats, candidate, source)
                entry = {
                    "seed": seed,
                    "interface": interface,
                    "joint": joint,
                    "original_feasible": candidate.get("original_feasible"),
                    "decision_id": str(row.get("decision_id", "")),
                    "tick": int(row.get("tick", 0)),
                    "uid": int(row.get("uid", 0)),
                    "source": source,
                    "destination_count": len(candidate.get("destinations") or []),
                }
                candidate_entries.append(entry)
                existing_witness_keys.add((entry["decision_id"], joint))
                if source == "old":
                    old_keys.add((entry["decision_id"], joint))

    label_per_seed, label_per_interface = _label_summary(label_rows)
    release_total = sum(value["release"] for value in label_per_seed.values())
    wait_total = sum(value["wait"] for value in label_per_seed.values())
    if len(label_rows) != 3149 or release_total != 577 or wait_total != 2572:
        raise ValueError(
            f"label count check failed rows={len(label_rows)} release={release_total} wait={wait_total}"
        )

    # This upper bound uses only the already checked label prefix.  Ring rows
    # contribute one plan and final rows contribute six, with the registered
    # per-parent cap exposed separately.
    upper_bound: dict[str, Any] = {}
    for seed_text, value in label_per_seed.items():
        ring = sum(
            value["interface_counts"].get(interface, {}).get("release", 0)
            for interface in ("o2m", "m2i")
        )
        final = value["interface_counts"].get("i2f", {}).get("release", 0)
        theoretical = ring + 6 * final
        upper_bound[seed_text] = {
            "ring_release_rows_weight1": ring,
            "final_release_rows_weight6": final,
            "actual_theoretical_upper_bound": theoretical,
            "registered_max_plans_per_parent": 900,
            "exceeds_registered_max": theoretical > 900,
        }

    all_seeds = sorted(int(seed) for seed in label_per_seed)
    all_strata = sorted(aggregate)
    held_support: dict[str, Any] = {}
    rare: list[dict[str, Any]] = []
    for held_seed in all_seeds:
        strata: dict[str, Any] = {}
        for key in all_strata:
            held = per_seed[held_seed].get(key, _new_stats())
            other = _merge_stats(
                per_seed[seed][key]
                for seed in all_seeds
                if seed != held_seed and key in per_seed[seed]
            )
            held_serial = _serial_stats(held)
            other_serial = _serial_stats(other)
            strata[key] = {"held_seed": held_serial, "other_eight_seeds": other_serial}
            known_other = other_serial["known_blocked_pairs"] + other_serial["known_safe_pairs"]
            if known_other <= 5:
                rare.append(
                    {
                        "held_seed": held_seed,
                        "stratum": key,
                        "missing": known_other == 0,
                        "rare_le_5": True,
                        "other_eight_known_blocked_pairs": other_serial["known_blocked_pairs"],
                        "other_eight_known_safe_pairs": other_serial["known_safe_pairs"],
                        "other_eight_parent_seeds": other_serial["parent_seeds"],
                    }
                )
        held_support[str(held_seed)] = {"strata": strata}

    teacher_selected, teacher_selection_counts = _serialize_selection(
        teacher_entries, existing_witness_keys
    )
    teacher_by_seed = {
        str(seed): {
            key: _serial_teacher_stats(stats)
            for key, stats in sorted(teacher_per_seed[seed].items())
        }
        for seed in sorted(teacher_per_seed)
    }
    teacher_by_interface = {
        key: _serial_teacher_stats(teacher_aggregate[key])
        for key in sorted(teacher_aggregate)
    }
    selected_plan_bounds = _weighted_selection_bounds(teacher_selected)
    source_record_counts = {source: len(rows) for source, rows in witness_streams}
    source_plan_counts = {
        source: sum(int(row.get("plan_calls", 0)) for row in rows)
        for source, rows in witness_streams
    }
    source_destination_counts = {
        source: sum(
            len(candidate.get("destinations") or [])
            for row in rows
            for candidate in row.get("candidates") or []
            if int(candidate.get("joint", 0)) != 0
        )
        for source, rows in witness_streams
    }

    output = {
        "protocol": "p3_neighbor_candidate_coverage_20261010_v2",
        "scope": {
            "read_only": True,
            "labels_are_input_context_only": False,
            "candidate_feasibility_source": "recorded teacher_audit entries in each annotation-referenced label file",
            "wait_joint_zero_ignored": True,
            "no_snapshot_load": True,
            "teacher_queries": 0,
            "reset_or_step_calls": 0,
            "no_witness_state_calls": True,
            "coverage_limit": (
                "Feasibility and reason counts are read from recorded teacher_audit entries; "
                "no new query or witness_state call was made."
            ),
        },
        "input_manifest": str(MANIFEST),
        "input_files": label_sources,
        "label_counts": {
            "rows_total": len(label_rows),
            "gate_release": release_total,
            "gate_wait": wait_total,
            "per_seed": label_per_seed,
            "per_interface": label_per_interface,
        },
        "witness_inputs": {
            "old": {"path": str(OLD_WITNESS), "records": source_record_counts["old"], "plan_calls": source_plan_counts["old"], "joint_nonzero_destinations": source_destination_counts["old"]},
            "supplemental": {"path": str(SUPP_WITNESS), "records": source_record_counts["supplemental"], "plan_calls": source_plan_counts["supplemental"], "joint_nonzero_destinations": source_destination_counts["supplemental"]},
            "records_total": sum(source_record_counts.values()),
            "plan_calls_total": sum(source_plan_counts.values()),
            "joint_nonzero_destinations_total": sum(source_destination_counts.values()),
        },
        "teacher_audit_consistency": teacher_consistency,
        "teacher_checked_release_prefix_candidate_coverage": {
            "teacher_audit_entries_total": teacher_consistency["teacher_audit_entries_total"],
            "teacher_audit_joint_zero_entries": teacher_consistency["teacher_audit_joint_zero_entries"],
            "candidate_records_total": len(teacher_entries),
            "candidate_records_nonzero_joint_total": len(teacher_entries),
            "by_seed_interface_joint": teacher_by_seed,
            "by_interface_joint": teacher_by_interface,
        },
        "witness_candidate_coverage": {
            "by_interface_joint": {
                key: _serial_stats(aggregate[key]) for key in sorted(aggregate)
            },
            "candidate_records_total": len(candidate_entries),
            "candidate_records_nonzero_joint_total": len(candidate_entries),
        },
        "held_seed_support": held_support,
        "missing_or_rare_other_eight_support": rare,
        "deterministic_candidate_selection": {
            "rule": "one lower-median teacher_audit candidate per parent_seed/interface/joint/original_feasible group after tick, numeric uid, decision_id, source sort",
            "existing_witness_skip_key": "(decision_id, joint) across old and supplemental witness files",
            "selection_counts": teacher_selection_counts,
            "selected": teacher_selected,
        },
        "plan_upper_bound": {
            "selected_teacher_audit_groups": selected_plan_bounds,
            "label_row_reference_bound": upper_bound,
        },
    }
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUT),
                "labels": len(label_rows),
                "release": release_total,
                "wait": wait_total,
                "old_records": len(old_rows),
                "supp_records": len(supp_rows),
                "plan_calls": sum(source_plan_counts.values()),
                "witness_candidate_entries": len(candidate_entries),
                "teacher_audit_candidate_entries": len(teacher_entries),
                "witness_strata": len(aggregate),
                "teacher_strata": len(teacher_aggregate),
                "teacher_selected_groups": teacher_selection_counts["groups"],
                "teacher_new_selected_groups": teacher_selection_counts["groups_without_existing_witness_reuse"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
