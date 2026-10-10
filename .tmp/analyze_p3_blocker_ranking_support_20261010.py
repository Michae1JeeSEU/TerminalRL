from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from terminal_marl.training.p3_neighbor_witness_dataset import (
    WitnessTargets,
    combine_witness_targets,
    load_witness_targets,
)
from terminal_marl.training.p3_v26_train_dataset import load_nine_cycle_train


LOCAL_TRAIN_ROOT = PROJECT_ROOT.parent / "training_outputs" / "terminal"
REMOTE_TRAIN_ROOT = "/home/user/training_outputs/terminal"
REMOTE_PROJECT_ROOT = "/home/user/projects/terminal"
MANIFEST_PATH = PROJECT_ROOT / ".tmp" / "p3_neighbor_ring_reuse_cv_proposed_20261010.json"
OUTPUT_PATH = PROJECT_ROOT / ".tmp" / "p3_blocker_ranking_support_20261010.json"
RING_OUTPUT_REMOTE = "/home/user/training_outputs/terminal/p3_neighbor_prefix_ring_reuse_20261010/nine_cycle"
EXPECTED_SEEDS = [893511, 893512, 893513, 893601, 893602, 893603, 893604, 893605, 893606]
MAX_RUNTIME_SECONDS = 120.0
EXPECTED_COUNTS = {
    "reference": {"queries": 473, "known_pairs": 4555, "blocked_pairs": 394},
    "prefix": {"queries": 797, "known_pairs": 9841, "blocked_pairs": 1075},
    "combined": {"queries": 1270, "known_pairs": 14396, "blocked_pairs": 1469},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def map_path(raw: str | Path) -> Path:
    value = str(raw).replace("\\", "/")
    if value == REMOTE_TRAIN_ROOT or value.startswith(REMOTE_TRAIN_ROOT + "/"):
        suffix = value[len(REMOTE_TRAIN_ROOT) :].lstrip("/")
        return LOCAL_TRAIN_ROOT / Path(suffix) if suffix else LOCAL_TRAIN_ROOT
    if value == REMOTE_PROJECT_ROOT or value.startswith(REMOTE_PROJECT_ROOT + "/"):
        suffix = value[len(REMOTE_PROJECT_ROOT) :].lstrip("/")
        return PROJECT_ROOT / Path(suffix) if suffix else PROJECT_ROOT
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def argv_values(argv: list[str], option: str, count: int) -> list[Path]:
    positions = [index for index, token in enumerate(argv) if token == option]
    if len(positions) != 1:
        raise ValueError(f"expected exactly one {option} in frozen launcher argv")
    start = positions[0] + 1
    values = argv[start : start + count]
    if len(values) != count or any(not value or value.startswith("-") for value in values):
        raise ValueError(f"{option} does not have {count} values")
    return [map_path(value) for value in values]


def argv_path(argv: list[str], option: str) -> Path:
    positions = [index for index, token in enumerate(argv) if token == option]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"expected exactly one {option} in frozen launcher argv")
    return map_path(argv[positions[0] + 1])


def histogram(values: list[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[int]) -> dict[str, Any]:
    return {
        "count_queries": len(values),
        "sum": int(sum(values)),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": (sum(values) / len(values)) if values else None,
        "median": percentile(values, 0.5),
        "histogram": histogram(values),
    }


def bool_count(value: torch.Tensor) -> int:
    return int(value.bool().sum().item())


def target_records(
    target: WitnessTargets,
    source: str,
    paired: Any,
    errors: list[str],
) -> list[dict[str, Any]]:
    mask = paired.new[2]
    records: list[dict[str, Any]] = []
    if len(target.state_index) != len(target.proposal):
        raise ValueError(f"{source}: target state/proposal length mismatch")
    for index in range(len(target.state_index)):
        state_index = int(target.state_index[index])
        if not 0 <= state_index < len(paired.original.ids):
            errors.append(f"{source}: state_index {state_index} outside official loader rows")
            continue
        context = paired.original.contexts[state_index]
        seed = int(target.seeds[index])
        context_seed = int(context["seed"])
        if seed != context_seed:
            errors.append(f"{source}: seed join mismatch at target row {index}: {seed} != {context_seed}")
        interface = str(context["interface"])
        joint = int(target.proposal[index]) + 1
        if not 1 <= joint <= 25:
            errors.append(f"{source}: candidate joint {joint} outside 1..25 at target row {index}")
        present = mask[state_index].bool()
        known = target.known[index].bool()
        blocked = target.blocked[index].bool()
        if bool((known & ~present).any()):
            errors.append(f"{source}: known target outside mask at target row {index}")
        if bool((blocked & ~known).any()):
            errors.append(f"{source}: blocked value present without known mask at target row {index}")
        known_blocked = bool_count(known & blocked)
        known_safe = bool_count(known & ~blocked)
        present_count = bool_count(present)
        known_count = known_blocked + known_safe
        unknown_present = bool_count(present & ~known)
        absent_mask = int((~present).sum().item())
        if known_count + unknown_present != present_count:
            errors.append(f"{source}: known/unknown/mask support mismatch at target row {index}")
        scene_known = bool(target.scene_known[index])
        scene_blocked = bool(target.scene_blocked[index])
        records.append(
            {
                "source": source,
                "target_index": index,
                "state_index": state_index,
                "decision_id": str(paired.original.ids[state_index]),
                "parent_seed": seed,
                "interface": interface,
                "joint": joint,
                "present_mask_neighbors": present_count,
                "absent_mask_neighbors": absent_mask,
                "known_pairs": known_count,
                "known_blocked_pairs": known_blocked,
                "known_safe_pairs": known_safe,
                "unknown_but_present_pairs": unknown_present,
                "scene_known": scene_known,
                "scene_blocked": scene_blocked,
            }
        )
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def total(key: str) -> int:
        return int(sum(int(row[key]) for row in records))

    blocked_queries = [row for row in records if row["known_blocked_pairs"] > 0]
    safe_queries = [row for row in records if row["known_safe_pairs"] > 0]
    mixed_queries = [
        row
        for row in records
        if row["known_blocked_pairs"] > 0 and row["known_safe_pairs"] > 0
    ]
    blocked_without_safe = [
        row
        for row in records
        if row["known_blocked_pairs"] > 0 and row["known_safe_pairs"] == 0
    ]
    blocked_with_unknown = [
        row
        for row in blocked_queries
        if row["unknown_but_present_pairs"] > 0
    ]
    no_known = [row for row in records if row["known_pairs"] == 0]
    no_known_present = [row for row in no_known if row["present_mask_neighbors"] > 0]
    no_known_no_present = [row for row in no_known if row["present_mask_neighbors"] == 0]
    return {
        "queries": len(records),
        "known_pairs": total("known_pairs"),
        "blocked_pairs": total("known_blocked_pairs"),
        "safe_pairs": total("known_safe_pairs"),
        "unknown_but_present_pairs": total("unknown_but_present_pairs"),
        "present_mask_neighbors": total("present_mask_neighbors"),
        "absent_mask_neighbors": total("absent_mask_neighbors"),
        "scene_known_queries": sum(bool(row["scene_known"]) for row in records),
        "scene_blocked_queries": sum(bool(row["scene_blocked"]) for row in records),
        "ranking_support": {
            "queries_with_known_blocked_and_safe": len(mixed_queries),
            "known_blocked_pairs_in_mixed_queries": sum(row["known_blocked_pairs"] for row in mixed_queries),
            "known_safe_pairs_in_mixed_queries": sum(row["known_safe_pairs"] for row in mixed_queries),
            "queries_with_blocked_but_no_known_safe": len(blocked_without_safe),
            "blocked_pairs_in_queries_without_known_safe": sum(
                row["known_blocked_pairs"] for row in blocked_without_safe
            ),
            "blocked_queries_with_unknown_contenders": len(blocked_with_unknown),
            "blocked_pairs_in_queries_with_unknown_contenders": sum(
                row["known_blocked_pairs"] for row in blocked_with_unknown
            ),
            "safe_queries": len(safe_queries),
            "blocked_queries": len(blocked_queries),
            "queries_with_known_safe_but_no_blocked": sum(
                row["known_safe_pairs"] > 0 and row["known_blocked_pairs"] == 0 for row in records
            ),
            "allunknown_no_known_targets": len(no_known),
            "allunknown_with_present_neighbors": len(no_known_present),
            "no_known_and_no_present_neighbors": len(no_known_no_present),
        },
        "per_query_distributions": {
            "known_safe_pairs": distribution([row["known_safe_pairs"] for row in records]),
            "known_blocked_pairs": distribution([row["known_blocked_pairs"] for row in records]),
            "unknown_but_present_pairs": distribution([row["unknown_but_present_pairs"] for row in records]),
            "present_mask_neighbors": distribution([row["present_mask_neighbors"] for row in records]),
            "absent_mask_neighbors": distribution([row["absent_mask_neighbors"] for row in records]),
            "known_pairs": distribution([row["known_pairs"] for row in records]),
        },
    }


def group_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["source"], row["parent_seed"], row["interface"], row["joint"])].append(row)
    groups = []
    for (source, seed, interface, joint), rows in sorted(grouped.items()):
        summary = summarize(rows)
        groups.append(
            {
                "source": source,
                "parent_seed": seed,
                "interface": interface,
                "joint": joint,
                "support": summary,
            }
        )
    return groups


def source_file_meta(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "sha256": sha256(path) if path.is_file() else None}


def main() -> None:
    started = time.monotonic()
    errors: list[str] = []
    manifest = read_json(MANIFEST_PATH)
    argv = list(dict(manifest["command"]) ["argv"])
    manifest_output = str(dict(manifest["candidate"])["output_remote"])
    if manifest_output != RING_OUTPUT_REMOTE:
        raise ValueError(f"frozen manifest output {manifest_output!r} differs from ring-reuse output")

    old_dirs = argv_values(argv, "--train-directories", 3)
    additional_dirs = argv_values(argv, "--additional-train-directories", 3)
    supplemental_dirs = argv_values(argv, "--supplemental-train-directories", 3)
    old_witness_dir = argv_path(argv, "--witness-directory")
    supplemental_witness_dir = argv_path(argv, "--supplemental-witness-directory")
    prefix_witness_dir = argv_path(argv, "--prefix-witness-directory")
    all_input_paths = old_dirs + additional_dirs + supplemental_dirs + [
        old_witness_dir,
        supplemental_witness_dir,
        prefix_witness_dir,
    ]
    missing = [str(path) for path in all_input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing frozen loader input(s): " + ", ".join(missing))

    paired = load_nine_cycle_train(old_dirs, additional_dirs, supplemental_dirs)
    old_target = load_witness_targets(old_witness_dir, paired)
    supplemental_target = load_witness_targets(supplemental_witness_dir, paired)
    prefix_target = load_witness_targets(prefix_witness_dir, paired)
    reference_target = combine_witness_targets((old_target, supplemental_target))
    combined_target = combine_witness_targets((reference_target, prefix_target))

    target_parts = {
        "old": old_target,
        "supplemental": supplemental_target,
        "prefix": prefix_target,
    }
    records_by_source = {
        source: target_records(target, source, paired, errors)
        for source, target in target_parts.items()
    }
    records_reference = records_by_source["old"] + records_by_source["supplemental"]
    records_combined = records_reference + records_by_source["prefix"]
    summaries = {
        "old": summarize(records_by_source["old"]),
        "supplemental": summarize(records_by_source["supplemental"]),
        "reference": summarize(records_reference),
        "prefix": summarize(records_by_source["prefix"]),
        "combined": summarize(records_combined),
    }
    observed_counts = {
        name: {
            key: summaries[name][key]
            for key in ("queries", "known_pairs", "blocked_pairs")
        }
        for name in ("reference", "prefix", "combined")
    }
    for name, expected in EXPECTED_COUNTS.items():
        if observed_counts[name] != expected:
            errors.append(f"{name} counts {observed_counts[name]} != expected {expected}")
    if len(records_by_source["old"]) != 317:
        errors.append(f"old query count {len(records_by_source['old'])} != 317")
    if len(records_by_source["supplemental"]) != 156:
        errors.append(f"supplemental query count {len(records_by_source['supplemental'])} != 156")
    if len(records_by_source["prefix"]) != 797:
        errors.append(f"prefix query count {len(records_by_source['prefix'])} != 797")
    if len(combined_target.state_index) != len(records_combined):
        errors.append("combined tensor/query record count mismatch")
    if len(reference_target.state_index) != len(records_reference):
        errors.append("reference tensor/query record count mismatch")

    elapsed = time.monotonic() - started
    if elapsed > MAX_RUNTIME_SECONDS:
        errors.append(f"runtime {elapsed:.3f}s exceeded hard maximum {MAX_RUNTIME_SECONDS:.1f}s")

    output = {
        "schema": "p3_blocker_ranking_support_20261010_v1",
        "status": "PASS" if not errors else "FAIL",
        "runtime_seconds": elapsed,
        "operations": {
            "environment_resets": 0,
            "environment_steps": 0,
            "teacher_queries": 0,
            "planner_calls": 0,
            "optimizer_updates": 0,
            "Q_accessed": False,
            "official_loader": "load_nine_cycle_train + load_witness_targets + combine_witness_targets",
        },
        "inputs": {
            "manifest": source_file_meta(MANIFEST_PATH),
            "manifest_status": manifest.get("status"),
            "frozen_manifest_output_remote": manifest_output,
            "old_train_directories": [str(path) for path in old_dirs],
            "additional_train_directories": [str(path) for path in additional_dirs],
            "supplemental_train_directories": [str(path) for path in supplemental_dirs],
            "old_witness_directory": str(old_witness_dir),
            "supplemental_witness_directory": str(supplemental_witness_dir),
            "prefix_witness_directory": str(prefix_witness_dir),
            "target_rows": {
                "old": len(old_target.state_index),
                "supplemental": len(supplemental_target.state_index),
                "reference": len(reference_target.state_index),
                "prefix": len(prefix_target.state_index),
                "combined": len(combined_target.state_index),
            },
        },
        "expected_counts": EXPECTED_COUNTS,
        "observed_counts": observed_counts,
        "checks": {
            "official_loader_row_count": len(paired.original.ids) == 3149,
            "official_loader_seed_set": sorted({int(context["seed"]) for context in paired.original.contexts}) == EXPECTED_SEEDS,
            "target_rows_old_supp_prefix": {
                "old_317": len(records_by_source["old"]) == 317,
                "supplemental_156": len(records_by_source["supplemental"]) == 156,
                "prefix_797": len(records_by_source["prefix"]) == 797,
            },
            "expected_aggregate_counts": all(observed_counts[name] == expected for name, expected in EXPECTED_COUNTS.items()),
            "join_and_mask_errors": errors,
            "mask_join_checks": {
                "seed_join_mismatches": sum("seed join mismatch" in error for error in errors),
                "known_outside_mask": sum("known target outside mask" in error for error in errors),
                "blocked_without_known": sum("blocked value present without known" in error for error in errors),
            },
        },
        "aggregate_support": summaries,
        "groups_by_source_parent_seed_interface_joint": {
            source: group_records(records) for source, records in records_by_source.items()
        },
        "comparison_facts": {
            "reference_vs_prefix": {
                "reference": summaries["reference"],
                "prefix": summaries["prefix"],
                "combined": summaries["combined"],
                "per_query_distribution_fields": [
                    "known_safe_pairs",
                    "known_blocked_pairs",
                    "unknown_but_present_pairs",
                    "present_mask_neighbors",
                    "absent_mask_neighbors",
                    "known_pairs",
                ],
            },
            "unknown_semantics": "unknown-but-present and absent-mask neighbors remain separate; unknown entries are not assigned blocked or safe labels",
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(OUTPUT_PATH), "runtime_seconds": elapsed, "errors": errors}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
