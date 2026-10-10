"""Read-only validation of the completed P3 neighbor-prefix witness output.

This validator consumes a local Syncthing copy of the collector's ``full``
directory.  It never calls the collector, ``witness_state``, a planner, a
teacher, or an environment, and performs no query, reset, step, or optimizer
operation.  The official train loader does read and unpickle archived old3
snapshots for SHA-verified 54D resource reconstruction; stored prefix and
traffic arrays are preserved and checked with the official resource tolerances.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from terminal_marl.training.p3_neighbor_prefix_coverage_data import (
    PREFIX_PROTOCOL,
    load_validated_inputs,
)
from terminal_marl.training.p3_v26_train_dataset import load_nine_cycle_train
from terminal_marl.training.p3_neighbor_witness_dataset import (
    PREFIX_COVERAGE_PROTOCOL,
    WitnessTargets,
    combine_witness_targets,
    load_witness_targets,
)
from terminal_marl.training.p3_neighbor_witness_data import PROTOCOL as OLD_WITNESS_PROTOCOL
from terminal_marl.training.p3_neighbor_witness_data import SUPPLEMENTAL_PROTOCOL as SUPP_WITNESS_PROTOCOL


LOCAL_TRAIN_ROOT = PROJECT_ROOT.parent / "training_outputs" / "terminal"
REMOTE_TRAIN_ROOT = "/home/user/training_outputs/terminal"
REMOTE_PROJECT_ROOT = "/home/user/projects/terminal"
MANIFEST = PROJECT_ROOT / ".tmp" / "p3_nine_cycle_loader_check_20261009.json"
EXPECTED_SEEDS = (893511, 893512, 893513, 893601, 893602, 893603, 893604, 893605, 893606)
EXPECTED_SELECTION = {"selected": 733, "reuse": 20, "new": 713}
MAX_PARENT_PLANS = 900
MAX_GLOBAL_PLANS = 8100


class ValidationError(ValueError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"{path}:{number}: expected JSON object")
        rows.append(value)
    return rows


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def local_path(raw: str | Path) -> tuple[Path, str]:
    """Map server and local manifest paths to the shared local checkout."""
    value = str(raw)
    normalized = value.replace("\\", "/")
    mappings = (
        (REMOTE_TRAIN_ROOT, LOCAL_TRAIN_ROOT, "server_training_output_to_local"),
        (REMOTE_PROJECT_ROOT, PROJECT_ROOT, "server_project_to_local"),
        ("D:/7_codespace/training_outputs/terminal", LOCAL_TRAIN_ROOT, "local_training_output"),
        ("D:/7_codespace/terminal", PROJECT_ROOT, "local_project"),
    )
    for prefix, base, rule in mappings:
        if normalized == prefix or normalized.startswith(prefix + "/"):
            remainder = normalized[len(prefix):].lstrip("/")
            return base / Path(*remainder.split("/")) if remainder else base, rule
    candidate = Path(value)
    if not candidate.is_absolute():
        return PROJECT_ROOT / candidate, "project_relative"
    return candidate, "absolute_unmapped"


def remember_path(report: dict[str, Any], raw: str | Path) -> Path:
    mapped, rule = local_path(raw)
    report.setdefault("path_mapping", []).append({
        "raw": str(raw), "local": str(mapped), "rule": rule, "exists": mapped.exists(),
    })
    return mapped


def compare_path(report: dict[str, Any], left: str | Path, right: str | Path, message: str) -> None:
    left_path = remember_path(report, left)
    right_path = remember_path(report, right)
    require(left_path.resolve() == right_path.resolve(), message)


def input_groups(protocol: dict[str, Any], report: dict[str, Any]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {"old": [], "additional": [], "supplemental": []}
    for item in protocol.get("inputs", []):
        source = str(item.get("source_group", ""))
        require(source in groups, f"unknown protocol input source_group: {source}")
        groups[source].append(remember_path(report, item["directory"]))
    require([len(groups[key]) for key in groups] == [3, 3, 3],
            f"protocol input group sizes differ: { {key: len(value) for key, value in groups.items()} }")
    return groups


def compare_input_manifest(protocol: dict[str, Any], loaded: list[dict[str, Any]],
                           report: dict[str, Any]) -> None:
    by_seed = {int(item["seed"]): item for item in loaded}
    protocol_by_seed = {int(item["seed"]): item for item in protocol["inputs"]}
    require(tuple(sorted(protocol_by_seed)) == EXPECTED_SEEDS, "protocol input seeds differ")
    for seed in EXPECTED_SEEDS:
        expected = protocol_by_seed[seed]
        actual = by_seed[seed]
        for key in ("seed", "scenario", "source_group", "rows_sha256", "selected_sha256",
                    "annotation_sha256", "rows_count", "selected_count", "annotation_count",
                    "label_count"):
            require(expected.get(key) == actual.get(key), f"input {seed} {key} differs")
        for key in ("directory", "rows_path", "selected_path", "annotation_path"):
            compare_path(report, expected[key], actual[key], f"input {seed} {key} path differs")
        require(expected.get("label_files") == actual.get("label_files"),
                f"input {seed} label file manifest differs")
        require(expected.get("label_sha256") == actual.get("label_sha256"),
                f"input {seed} label SHA manifest differs")
    mark = {"status": "PASS", "seeds": list(EXPECTED_SEEDS), "inputs": len(loaded)}
    if MANIFEST.is_file():
        manifest = read_json(MANIFEST)
        manifest_by_seed = {int(item["seed"]): item for item in manifest.get("input_files", [])}
        require(set(manifest_by_seed) == set(EXPECTED_SEEDS), "nine-cycle manifest seed set differs")
        group_map = {"original": "old", "independent": "additional", "supplemental": "supplemental"}
        for seed in EXPECTED_SEEDS:
            entry = manifest_by_seed[seed]
            expected = protocol_by_seed[seed]
            compare_path(report, entry["directory"], expected["directory"],
                         f"manifest input {seed} directory differs")
            require(group_map.get(entry.get("group")) == expected.get("source_group"),
                    f"manifest input {seed} source group differs")
            source_hash = entry.get("source_file_sha256")
            require(source_hash == expected.get("rows_sha256"),
                    f"manifest input {seed} context/resource SHA differs")
        mark["manifest"] = str(MANIFEST)
        mark["manifest_status"] = manifest.get("status")
    else:
        mark["manifest"] = str(MANIFEST)
        mark["manifest_status"] = "MISSING"
    report["checks"]["input_hashes_and_protocol"] = mark


def candidate_key(decision_id: str, joint: int) -> tuple[str, int]:
    return str(decision_id), int(joint)


def selection_records(selection: dict[str, Any], report: dict[str, Any]) -> tuple[dict[tuple[str, int], dict],
                                                                                dict[tuple[str, int], dict],
                                                                                dict[tuple[str, int], dict]]:
    all_rows = list(selection.get("selected", []))
    reuse_rows = list(selection.get("reuse", []))
    new_rows = list(selection.get("new", []))
    require(len(all_rows) == EXPECTED_SELECTION["selected"],
            f"selection selected count {len(all_rows)} != {EXPECTED_SELECTION['selected']}")
    require(len(reuse_rows) == EXPECTED_SELECTION["reuse"],
            f"selection reuse count {len(reuse_rows)} != {EXPECTED_SELECTION['reuse']}")
    require(len(new_rows) == EXPECTED_SELECTION["new"],
            f"selection new count {len(new_rows)} != {EXPECTED_SELECTION['new']}")

    def index(rows: list[dict], name: str) -> dict[tuple[str, int], dict]:
        result = {}
        for row in rows:
            key = candidate_key(row["decision_id"], int(row["joint"]))
            require(key not in result, f"duplicate {name} selection candidate: {key}")
            require(str(row.get("candidate_key")) == f"{key[0]}|joint={key[1]}",
                    f"{name} candidate_key differs: {key}")
            result[key] = row
        return result

    all_index, reuse_index, new_index = index(all_rows, "selected"), index(reuse_rows, "reuse"), index(new_rows, "new")
    require(set(reuse_index).isdisjoint(new_index), "reuse and new selection candidates overlap")
    require(set(reuse_index) | set(new_index) == set(all_index),
            "selected/new/reuse candidate sets differ")
    protocol_counts = selection.get("counts", {})
    require(protocol_counts.get("selected_candidate_groups") in (None, EXPECTED_SELECTION["selected"]),
            "selection counts selected groups differ")
    report["checks"]["selection_counts"] = {
        "status": "PASS", "selected": len(all_index), "reuse": len(reuse_index), "new": len(new_index),
    }
    return all_index, reuse_index, new_index


def label_audit_index(loaded: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for item in loaded:
        for did, label in item["label_by_id"].items():
            for audit in label.get("teacher_audit", []):
                joint = int(audit["joint_index"])
                if joint == 0:
                    continue
                key = candidate_key(did, joint)
                require(key not in result, f"duplicate teacher audit candidate: {key}")
                result[key] = audit
    return result


def validate_selection_audits(selection_index: dict[tuple[str, int], dict],
                              audits: dict[tuple[str, int], dict], report: dict[str, Any]) -> None:
    for key, selected in selection_index.items():
        require(key in audits, f"selected candidate missing from teacher_audit: {key}")
        audit = audits[key]
        require(bool(selected.get("original_feasible")) == bool(audit.get("feasible")) and
                selected.get("original_reason") == audit.get("reason"),
                f"selected candidate teacher_audit mismatch: {key}")
    report["checks"]["teacher_audit_feasibility_reason"] = {
        "status": "PASS", "checked_candidates": len(selection_index),
    }


def validate_resource_loader(protocol: dict[str, Any], loaded_inputs: list[dict[str, Any]],
                             paired: Any, report: dict[str, Any]) -> None:
    original, ego, traffic, mask = paired.original, *paired.new
    require(tuple(original.x.shape) == (3149, 90), f"original shape differs: {tuple(original.x.shape)}")
    require(tuple(ego.shape) == (3149, 159), f"resource ego shape differs: {tuple(ego.shape)}")
    require(tuple(traffic.shape) == (3149, 20, 50), f"N50 traffic shape differs: {tuple(traffic.shape)}")
    require(tuple(mask.shape) == (3149, 20), f"traffic mask shape differs: {tuple(mask.shape)}")
    require(torch.isfinite(original.x).all() and torch.isfinite(ego).all() and torch.isfinite(traffic).all(),
            "non-finite loader tensors")
    require(mask.dtype == torch.bool, "traffic mask is not bool")
    by_id = {did: index for index, did in enumerate(original.ids)}
    require(len(by_id) == len(original.ids) == 3149, "loader decision IDs are not unique")
    group_for_seed = {int(item["seed"]): item["source_group"] for item in protocol["inputs"]}
    release = Counter(); wait = Counter(); prefix_checks = 0
    for item in loaded_inputs:
        for did, row in item["row_by_id"].items():
            index = by_id[did]; seed = int(row["seed"])
            raw_original = torch.tensor(row["original_observation"], dtype=torch.float32)
            require(torch.equal(original.x[index], raw_original), f"original observation changed: {did}")
            proposal = -1 if row.get("teacher_proposal") is None else int(row["teacher_proposal"])
            require(int(original.gate[index]) == int(row["teacher_gate"]) and
                    int(original.proposal[index]) == proposal, f"teacher label changed: {did}")
            if int(row["teacher_gate"]):
                release[seed] += 1
            else:
                wait[seed] += 1
            raw_traffic = torch.tensor(row.get("traffic", []), dtype=torch.float32)
            if raw_traffic.numel() == 0:
                raw_traffic = raw_traffic.reshape(0, 50)
            else:
                raw_traffic = raw_traffic.reshape(-1, 50)
            raw_mask = torch.tensor(row.get("traffic_mask", []), dtype=torch.bool)
            require(raw_mask.shape == (len(raw_traffic),), f"traffic mask row differs: {did}")
            require(torch.allclose(traffic[index, :len(raw_traffic)], raw_traffic, rtol=1e-6, atol=1e-7),
                    f"traffic changed: {did}")
            require(torch.equal(mask[index, :len(raw_mask)], raw_mask) and
                    not mask[index, len(raw_mask):].any(), f"traffic mask changed: {did}")
            raw_ego = torch.tensor(row["ego"], dtype=torch.float32)
            if group_for_seed[seed] == "old":
                require(raw_ego.shape == (105,) and
                        torch.allclose(ego[index, :105], raw_ego, rtol=1e-6, atol=1e-7),
                        f"old 105 prefix changed: {did}")
            else:
                require(raw_ego.shape == (159,) and
                        torch.allclose(ego[index], raw_ego, rtol=1e-6, atol=1e-7),
                        f"resource 159 ego changed: {did}")
            prefix_checks += 1
    report["checks"]["loader_159_N50_labels"] = {
        "status": "PASS", "rows": prefix_checks, "shape_original": list(original.x.shape),
        "shape_ego": list(ego.shape), "shape_traffic": list(traffic.shape), "shape_mask": list(mask.shape),
        "release_by_seed": dict(sorted(release.items())), "wait_by_seed": dict(sorted(wait.items())),
    }


def target_counts(target: WitnessTargets) -> dict[str, Any]:
    output = {}
    for seed in sorted(set(int(value) for value in target.seeds.tolist())):
        selected = target.seeds == seed
        known = target.known[selected]
        blocked = known & target.blocked[selected].bool()
        safe = known & ~target.blocked[selected].bool()
        scene_known = target.scene_known[selected].bool()
        scene_blocked = scene_known & target.scene_blocked[selected].bool()
        blocker_targets = blocked.any(-1)
        output[str(seed)] = {
            "queries": int(selected.sum()), "known_pairs": int(known.sum()),
            "blocked_pairs": int(blocked.sum()), "safe_pairs": int(safe.sum()),
            "scene_known": int(scene_known.sum()), "scene_blocked": int(scene_blocked.sum()),
            "blocker_targets": int(blocker_targets.sum()),
        }
    return output


def validate_witness_streams(protocol: dict[str, Any], paired: Any, report: dict[str, Any]):
    streams = (protocol.get("witness_inputs") or {}).get("streams", [])
    by_source = {str(item.get("source")): item for item in streams}
    require(set(by_source) == {"old", "supplemental"}, "old/supplemental witness streams missing")
    paths = {}
    for source, expected_protocol in (("old", OLD_WITNESS_PROTOCOL),
                                      ("supplemental", SUPP_WITNESS_PROTOCOL)):
        stream = by_source[source]
        path = remember_path(report, stream["directory"])
        paths[source] = path
        require(path.is_dir(), f"{source} witness directory missing: {path}")
        for filename, summary_key in (("protocol.json", "protocol_sha256"),
                                      ("result.json", "result_sha256"),
                                      ("witnesses.jsonl", "witness_sha256")):
            file = path / filename
            require(file.is_file(), f"{source} witness file missing: {file}")
            require(sha(file) == stream.get(summary_key), f"{source} {filename} SHA differs")
        witness_protocol = read_json(path / "protocol.json")
        require(witness_protocol.get("protocol") == expected_protocol,
                f"{source} witness protocol differs")
    old_target = load_witness_targets(paths["old"], paired)
    supplemental_target = load_witness_targets(paths["supplemental"], paired)
    reference = combine_witness_targets((old_target, supplemental_target))
    report["checks"]["old_supp_witness_reference"] = {
        "status": "PASS", "old_queries": len(old_target.state_index),
        "supplemental_queries": len(supplemental_target.state_index),
        "reference_counts_by_seed": target_counts(reference),
    }
    return reference, paths


def flatten_witness_rows(rows: list[dict[str, Any]], source: str) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for row in rows:
        did = str(row.get("decision_id"))
        for candidate in row.get("candidates") or []:
            key = candidate_key(did, int(candidate["joint"]))
            require(key not in result, f"duplicate {source} witness candidate: {key}")
            result[key] = {"row": row, "candidate": candidate}
    return result


def validate_parent_outputs(full: Path, root_protocol: dict[str, Any], root_result: dict[str, Any],
                            new_index: dict[tuple[str, int], dict], reuse_index: dict[tuple[str, int], dict],
                            audits: dict[tuple[str, int], dict], report: dict[str, Any]) -> list[dict[str, Any]]:
    parent_summaries = list(root_result.get("parents", []))
    require(len(parent_summaries) == 9, f"parent count {len(parent_summaries)} != 9")
    require(sorted(int(item.get("parent_seed", -1)) for item in parent_summaries) == list(EXPECTED_SEEDS),
            "parent seed set differs")
    all_child_rows = []
    seen_parent_keys = set()
    parent_facts = []
    for summary in sorted(parent_summaries, key=lambda item: int(item["parent_seed"])):
        seed = int(summary["parent_seed"])
        child = remember_path(report, summary.get("output", full / f"seed_{seed}"))
        require(child.is_dir(), f"child output missing: {child}")
        protocol = read_json(child / "protocol.json")
        result = read_json(child / "result.json")
        require(protocol.get("protocol") == PREFIX_PROTOCOL and not protocol.get("smoke"),
                f"child {seed} protocol/status differs")
        require(int(protocol.get("parent_seed")) == seed, f"child {seed} parent identity differs")
        require(result.get("status") == "COMPLETE" and result.get("error") is None,
                f"child {seed} is not complete")
        witness_path = child / "witnesses.jsonl"
        require(witness_path.is_file() and sha(witness_path) == result.get("witness_sha256"),
                f"child {seed} witness SHA differs")
        rows = read_jsonl(witness_path)
        require(len(rows) == int(result.get("states", -1)), f"child {seed} witness count differs")
        require([str(row.get("decision_id")) for row in rows] == list(protocol.get("selected_ids", [])),
                f"child {seed} selected state order differs")
        for row in rows:
            require(int(row.get("seed")) == seed, f"child {seed} row isolation differs")
        child_keys = flatten_witness_rows(rows, f"child {seed}")
        require(set(child_keys).issubset(set(new_index)), f"child {seed} contains non-new candidate")
        require(set(child_keys).isdisjoint(reuse_index), f"child {seed} repeats reuse candidate")
        require(seen_parent_keys.isdisjoint(child_keys), f"child {seed} candidate overlaps another parent")
        seen_parent_keys.update(child_keys)
        for key, payload in child_keys.items():
            audit = audits[key]
            candidate = payload["candidate"]
            require(bool(candidate.get("original_feasible")) == bool(audit.get("feasible")) and
                    candidate.get("original_reason") == audit.get("reason"),
                    f"child {seed} teacher_audit candidate mismatch: {key}")
        for key in ("resets", "steps", "teacher_queries"):
            require(int(result.get(key, 0)) == 0, f"child {seed} {key} is nonzero")
        for key in ("plan_calls", "new_plan_calls", "reused_plan_calls"):
            require(0 <= int(result.get(key, 0)) <= MAX_PARENT_PLANS,
                    f"child {seed} {key} exceeds parent budget")
        require(int(result.get("plan_calls", 0)) == int(result.get("new_plan_calls", 0)) +
                int(result.get("reused_plan_calls", 0)), f"child {seed} plan totals differ")
        for key in ("status", "states", "candidate_requests", "new_candidate_requests", "reused_candidate_requests",
                    "plan_calls", "new_plan_calls", "reused_plan_calls"):
            if key in summary:
                require(summary[key] == result.get(key), f"parent summary {seed} {key} differs")
        parent_facts.append({"seed": seed, "states": int(result.get("states", 0)),
                             "new_candidates": int(result.get("new_candidate_requests", 0)),
                             "new_plans": int(result.get("new_plan_calls", 0)),
                             "plan_calls": int(result.get("plan_calls", 0)),
                             "output": str(child)})
        all_child_rows.extend(rows)
    require(seen_parent_keys == set(new_index),
            f"child new candidate set differs: missing={len(set(new_index)-seen_parent_keys)}")
    report["checks"]["parents_and_child_witnesses"] = {
        "status": "PASS", "parents": parent_facts,
        "new_candidates": len(seen_parent_keys), "nonzero_new_plans": sum(item["new_plans"] > 0 for item in parent_facts),
    }
    return all_child_rows


def validate_root_output(full: Path, root_protocol: dict[str, Any], root_result: dict[str, Any],
                         child_rows: list[dict[str, Any]], new_index: dict[tuple[str, int], dict],
                         reuse_index: dict[tuple[str, int], dict], audits: dict[tuple[str, int], dict],
                         report: dict[str, Any]) -> WitnessTargets:
    require(root_result.get("status") == "COMPLETE" and root_result.get("error") is None,
            "root result is not COMPLETE/error=null")
    require(root_protocol.get("protocol") == PREFIX_PROTOCOL and root_protocol.get("scope") == "full" and
            not root_protocol.get("smoke") and not root_protocol.get("prepare_only"),
            "root protocol is not full non-smoke prefix protocol")
    require(root_protocol.get("input_seeds") == list(EXPECTED_SEEDS), "root input seed set differs")
    require(int(root_protocol.get("all_selected_candidate_count", -1)) == EXPECTED_SELECTION["selected"] and
            int(root_protocol.get("new_selected_candidate_count", -1)) == EXPECTED_SELECTION["new"] and
            int(root_protocol.get("reused_candidate_count", -1)) == EXPECTED_SELECTION["reuse"],
            "root selection count metadata differs")
    for key in ("resets", "steps", "teacher_queries"):
        require(int(root_result.get(key, 0)) == 0, f"root {key} is nonzero")
    require(0 <= int(root_result.get("plan_calls", 0)) <= MAX_GLOBAL_PLANS and
            0 <= int(root_result.get("new_plan_calls", 0)) <= MAX_GLOBAL_PLANS,
            "root plan budget exceeded")
    require(int(root_result.get("candidate_requests", 0)) == EXPECTED_SELECTION["new"] and
            int(root_result.get("new_candidate_requests", 0)) == EXPECTED_SELECTION["new"] and
            int(root_result.get("reused_candidate_requests", 0)) == 0,
            "root new/reuse request totals differ")
    root_witness = full / "witnesses.jsonl"
    require(root_witness.is_file() and sha(root_witness) == root_result.get("witness_sha256"),
            "root witness SHA differs")
    root_rows = read_jsonl(root_witness)
    require(len(root_rows) == int(root_result.get("states", -1)) == len(root_protocol.get("selected_ids", [])),
            "root witness state count differs")
    root_keys = flatten_witness_rows(root_rows, "root")
    child_keys = flatten_witness_rows(child_rows, "children")
    require(root_keys.keys() == child_keys.keys() == new_index.keys(),
            "root/child/new selection candidate sets differ")
    require(set(root_keys).isdisjoint(reuse_index), "root repeats reuse candidate")
    for key, payload in root_keys.items():
        candidate = payload["candidate"]
        audit = audits[key]
        require(bool(candidate.get("original_feasible")) == bool(audit.get("feasible")) and
                candidate.get("original_reason") == audit.get("reason"),
                f"root teacher_audit candidate mismatch: {key}")
    report["checks"]["root_result_and_witness"] = {
        "status": "PASS", "states": len(root_rows), "new_candidates": len(root_keys),
        "actual_new_plans": int(root_result.get("new_plan_calls", 0)),
        "actual_plan_calls": int(root_result.get("plan_calls", 0)),
        "nonzero_states": sum(bool(row.get("candidates")) for row in root_rows),
    }
    return root_keys


def coverage_summary(new_selection: dict[tuple[str, int], dict], report: dict[str, Any]) -> None:
    interfaces = Counter(str(item.get("interface")) for item in new_selection.values())
    joints = Counter(int(item.get("joint")) for item in new_selection.values())
    report["coverage"] = {
        "new_candidates_by_interface": dict(sorted(interfaces.items())),
        "new_candidates_by_joint": dict(sorted(joints.items())),
    }


def validate(full: Path, report: dict[str, Any]) -> None:
    protocol_path, result_path, selection_path = full / "protocol.json", full / "result.json", full / "selection.json"
    require(protocol_path.is_file() and result_path.is_file() and selection_path.is_file(),
            "full protocol/result/selection artifact missing")
    root_protocol, root_result, selection = read_json(protocol_path), read_json(result_path), read_json(selection_path)
    report["protocol_sha256"] = sha(protocol_path); report["result_sha256"] = sha(result_path)
    require(root_protocol.get("protocol") == PREFIX_PROTOCOL, "root protocol name differs")
    require(root_result.get("status") == "COMPLETE" and root_result.get("error") is None,
            "root is not COMPLETE/error=null")
    groups = input_groups(root_protocol, report)
    loaded_inputs = load_validated_inputs(groups["old"], groups["additional"], groups["supplemental"])
    compare_input_manifest(root_protocol, loaded_inputs, report)
    paired = load_nine_cycle_train(groups["old"], groups["additional"], groups["supplemental"])
    validate_resource_loader(root_protocol, loaded_inputs, paired, report)
    all_selected, reuse_index, new_index = selection_records(selection, report)
    audits = label_audit_index(loaded_inputs)
    validate_selection_audits(all_selected, audits, report)
    reference, witness_paths = validate_witness_streams(root_protocol, paired, report)
    prefix_target = load_witness_targets(full, paired)
    combined = combine_witness_targets((reference, prefix_target))
    report["checks"]["prefix_and_combined_witness"] = {
        "status": "PASS", "prefix_counts_by_seed": target_counts(prefix_target),
        "combined_counts_by_seed": target_counts(combined),
        "prefix_queries": len(prefix_target.state_index), "combined_queries": len(combined.state_index),
        "witness_paths": {key: str(value) for key, value in witness_paths.items()},
    }
    child_rows = validate_parent_outputs(full, root_protocol, root_result, new_index, reuse_index, audits, report)
    validate_root_output(full, root_protocol, root_result, child_rows, new_index, reuse_index, audits, report)
    coverage_summary(new_index, report)
    report["facts"] = {
        "selected": len(all_selected), "reuse": len(reuse_index), "new": len(new_index),
        "actual_new_plans": int(root_result.get("new_plan_calls", 0)),
        "actual_plan_calls": int(root_result.get("plan_calls", 0)),
        "nonzero_states": int(report["checks"]["root_result_and_witness"]["nonzero_states"]),
        "validator_runtime_calls": {
            "environment_resets": 0,
            "environment_steps": 0,
            "teacher_queries": 0,
            "planner_calls": 0,
            "optimizer_updates": 0,
            "archived_snapshot_reads": (
                "official load_nine_cycle_train/load_resource_train path; "
                "old3 SHA-verified unpickle/rebuild"
            ),
        },
        "official_resource_tolerance": {
            "rtol": 1e-6,
            "atol": 1e-7,
            "stored_prefix_traffic_preserved": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    full = args.full_directory
    report: dict[str, Any] = {
        "schema": "p3_neighbor_prefix_validation_20261010_v1",
        "status": "RUNNING", "full_directory": str(full), "checks": {}, "path_mapping": [],
        "source": (
            "read-only validator; no collector/witness_state/planner/teacher/"
            "environment/query/reset/step/optimizer calls; official loader may "
            "read archived old3 snapshots for SHA-verified resource rebuild"
        ),
    }
    try:
        if not full.is_dir():
            report["status"] = "MISSING_FULL"
            report["error"] = f"full directory missing: {full}"
            write_json(args.output, report)
            print(json.dumps({"status": report["status"], "error": report["error"]}, sort_keys=True))
            return 2
        validate(full, report)
        report["status"] = "PASS"
        report["error"] = None
        code = 0
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = repr(error)
        code = 1
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "error": report.get("error")}, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
