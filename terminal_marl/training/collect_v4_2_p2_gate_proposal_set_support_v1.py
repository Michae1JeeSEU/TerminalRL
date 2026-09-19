"""Collect fresh set-valued proposal support on Teacher-RELEASE actor states."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.training import diagnose_v4_2_p2_gate_feasibility_v1 as attribution
from terminal_marl.training import diagnose_v4_2_p2_gate_proposal_feasible_set_v1 as feasible_set
from terminal_marl.training import evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import reserve_once


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_proposal_set_support_v1.yaml")
PROTOCOL_SHA = "fa2d764604e05526262549559c718291caeef3ca23202bce38d20d3e038d2972"
STAGE = "V4_2_P2_GATE_PROPOSAL_SET_SUPPORT_V1"
PREDECESSOR_MANIFEST_SHA = "957bc2535fbef767a90fbf6cd299ea5aa38cd2f680099a3c5f79aff97fe4821f"
CANDIDATE_SHA = effectiveness.REPAIRED_CHECKPOINT_SHA
TRAIN_SEEDS = tuple(range(795001, 795010))
DEV_SEEDS = tuple(range(796001, 796007))
SMOKE_SEED = 795000
SMOKE_STEPS = 160
RESERVATION_MARKER = "p2_gate_proposal_set_support_v1_consumed.json"
INTERFACES = attribution.INTERFACES

LIMITS = {
    "train": {
        "min_rows": 400, "max_rows": 1200,
        "interface_min": {"o2m": 150, "m2i": 120, "i2f": 80},
        "unique_sets_min": 8, "candidate_miss_min": 50,
        "top3_miss_min": 10, "late_fraction_min": 0.50,
    },
    "dev": {
        "min_rows": 250, "max_rows": 800,
        "interface_min": {"o2m": 100, "m2i": 80, "i2f": 60},
        "unique_sets_min": 6, "candidate_miss_min": 30,
        "top3_miss_min": 6, "late_fraction_min": 0.50,
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


def protocol() -> dict[str, Any]:
    _require(PROTOCOL.is_file(), f"protocol missing: {PROTOCOL}")
    _require(_sha256(PROTOCOL) == PROTOCOL_SHA, "proposal-set support protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict) and value.get("protocol") == STAGE, "protocol id changed")
    predecessor = value.get("predecessor", {})
    for key, expected in {
        "feasible_set_manifest_sha256": PREDECESSOR_MANIFEST_SHA,
        "feasible_set_protocol_sha256": feasible_set.PROTOCOL_SHA,
        "feasible_set_status": "COMPLETE", "diagnostic_complete": True,
        "repaired_checkpoint_sha256": CANDIDATE_SHA,
    }.items():
        _require(predecessor.get(key) == expected, f"predecessor changed: {key}")
    bindings = value.get("bindings", {})
    for key, expected in {
        "deterministic": True, "fixed_geometry": True, "natural_flow": True,
        "scenario": "flow12", "teacher_mode": "teacher_label_only",
        "student_authoritative": True,
        "catalogue_scope": "actor_events_with_teacher_RELEASE",
        "catalogue_semantics": "all_25_RELEASE_proposals_checked_before_env_step",
        "retained_rows": "actor_events_with_teacher_RELEASE_and_nonempty_feasible_set",
        "no_training": True, "no_optimizer": True, "no_policy_update": True,
    }.items():
        _require(bindings.get(key) == expected, f"binding changed: {key}")
    formal = value.get("formal", {})
    _require(formal.get("train_seeds") == list(TRAIN_SEEDS), "train seeds changed")
    _require(formal.get("dev_seeds") == list(DEV_SEEDS), "dev seeds changed")
    _require(formal.get("scenario") == "flow12", "formal scenario changed")
    _require(value.get("smoke", {}).get("seed") == SMOKE_SEED, "smoke seed changed")
    return value


def validate_predecessor(path: str | Path) -> dict[str, Any]:
    _require(_sha256(path) == PREDECESSOR_MANIFEST_SHA, "feasible-set manifest SHA mismatch")
    value = _read_json(path)
    for key, expected in {
        "stage": feasible_set.STAGE, "status": "COMPLETE", "DIAGNOSTIC_COMPLETE": True,
        "protocol_sha256": feasible_set.PROTOCOL_SHA,
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "rows_authorized_for_training": False, "SET_VALUED_SUPPORT_ALLOWED": False,
        "PROPOSAL_REPAIR_ALLOWED": False, "PPO_ALLOWED": False,
    }.items():
        _require(value.get(key) == expected, f"feasible-set predecessor changed: {key}")
    return value


def retain_set_rows(
    policy: Any, rows: Sequence[Mapping[str, Any]], *, split: str,
) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows
                if bool(row.get("actor_mask")) and int(row.get("teacher_gate", -1)) == 1]
    if not selected:
        return []
    observations = torch.as_tensor(
        np.asarray([row["observation_v2_1"] for row in selected], dtype=np.float32)
    )
    ranks = feasible_set.proposal_audit.base_logits(policy, observations).argsort(
        dim=-1, descending=True,
    )
    result = []
    for index, source in enumerate(selected):
        feasible = sorted({int(value) for value in source.get("teacher_feasible_proposals", ())})
        _require(source.get("teacher_catalogue_complete") is True, "Teacher RELEASE catalogue incomplete")
        _require(source.get("teacher_catalogue_scope") == "teacher_release", "catalogue scope mismatch")
        _require(bool(feasible) and all(0 <= value < 25 for value in feasible), "invalid feasible set")
        teacher = int(source.get("teacher_proposal", -1))
        _require(teacher in feasible, "Teacher selected proposal outside feasible set")
        top1 = int(ranks[index, 0])
        top3 = [int(value) for value in ranks[index, :3]]
        row = dict(source)
        row.update({
            "proposal_set_support_split": split,
            "proposal_set_target": feasible,
            "proposal_set_signature": ",".join(str(value) for value in feasible),
            "proposal_set_size": len(feasible),
            "policy_top1_proposal": top1,
            "policy_top3_proposals": top3,
            "policy_top1_feasible": top1 in feasible,
            "policy_top3_feasible_hit": any(value in feasible for value in top3),
            "support_source": "fresh_flow12_student_visited_teacher_release_catalogue",
        })
        result.append(row)
    return result


def split_metrics(
    rows: Sequence[Mapping[str, Any]], split: str,
    records: Sequence[Mapping[str, Any]], planned_seeds: Sequence[int],
) -> dict[str, Any]:
    selected = [dict(row) for row in rows if row.get("proposal_set_support_split") == split]
    interfaces = Counter(str(row.get("interface")) for row in selected)
    signatures = Counter(str(row.get("proposal_set_signature")) for row in selected)
    sizes = [int(row.get("proposal_set_size", 0)) for row in selected]
    events = [str(row.get("decision_event_id")) for row in selected]
    observed_seeds = {int(row.get("seed")) for row in selected}
    split_records = [record for record in records if record.get("split") == split]
    late = sum(int(row.get("arrival_id", -1)) >= 6 for row in selected)
    top1_hits = sum(bool(row.get("policy_top1_feasible")) for row in selected)
    top3_hits = sum(bool(row.get("policy_top3_feasible_hit")) for row in selected)
    schema_ok = bool(selected) and all(
        isinstance(row.get("observation_v2_1"), list)
        and len(row["observation_v2_1"]) == 88
        and np.isfinite(np.asarray(row["observation_v2_1"], dtype=np.float64)).all()
        for row in selected
    )
    return {
        "split": split, "rows": len(selected),
        "interface_counts": {name: int(interfaces[name]) for name in INTERFACES},
        "unique_feasible_sets": len(signatures),
        "dominant_feasible_set_fraction": max(signatures.values(), default=0) / max(1, len(selected)),
        "feasible_set_size": {
            "min": min(sizes, default=0), "max": max(sizes, default=0),
            "mean": float(np.mean(sizes)) if sizes else 0.0,
            "median": float(np.median(sizes)) if sizes else 0.0,
        },
        "policy_top1_feasible": int(top1_hits),
        "policy_top1_infeasible": len(selected) - int(top1_hits),
        "policy_top3_feasible_hit": int(top3_hits),
        "policy_top3_miss": len(selected) - int(top3_hits),
        "late_arrival_rows": int(late),
        "late_arrival_fraction": float(late / max(1, len(selected))),
        "catalogue_complete": bool(selected) and all(row.get("teacher_catalogue_complete") is True for row in selected),
        "feasible_set_nonempty": bool(selected) and all(bool(row.get("proposal_set_target")) for row in selected),
        "teacher_selected_proposal_feasible": bool(selected) and all(
            int(row["teacher_proposal"]) in set(row["proposal_set_target"]) for row in selected
        ),
        "schema_88d_finite": schema_ok,
        "decision_event_id_unique": bool(events) and len(events) == len(set(events))
        and all(value not in {"", "None"} for value in events),
        "observed_seeds": sorted(observed_seeds),
        "planned_seeds": sorted(int(seed) for seed in planned_seeds),
        "seed_coverage": observed_seeds == set(int(seed) for seed in planned_seeds),
        "episodes": len(split_records),
        "all_episode_gates": bool(split_records)
        and all(record.get("gates", {}).get("all") is True for record in split_records),
        "completed_requests": sum(int(record.get("completed", 0)) for record in split_records),
        "requested": 12 * len(split_records),
    }


def split_gates(metrics: Mapping[str, Any], split: str, *, formal: bool = True) -> dict[str, bool]:
    limits = LIMITS[split]
    interface = metrics.get("interface_counts", {})
    gates = {
        "cardinality": limits["min_rows"] <= int(metrics.get("rows", 0)) <= limits["max_rows"] if formal else True,
        "interface_coverage": all(int(interface.get(name, 0)) >= limits["interface_min"][name]
                                  for name in INTERFACES) if formal else True,
        "feasible_set_diversity": int(metrics.get("unique_feasible_sets", 0)) >= limits["unique_sets_min"] if formal else True,
        "candidate_infeasible_support": int(metrics.get("policy_top1_infeasible", 0)) >= limits["candidate_miss_min"] if formal else True,
        "top3_miss_support": int(metrics.get("policy_top3_miss", 0)) >= limits["top3_miss_min"] if formal else True,
        "late_arrival_fraction": float(metrics.get("late_arrival_fraction", 0.0)) >= limits["late_fraction_min"] if formal else True,
        "catalogue_complete": bool(metrics.get("catalogue_complete")),
        "feasible_set_nonempty": bool(metrics.get("feasible_set_nonempty")),
        "teacher_selected_proposal_feasible": bool(metrics.get("teacher_selected_proposal_feasible")),
        "schema_88d_finite": bool(metrics.get("schema_88d_finite")),
        "decision_event_id_unique": bool(metrics.get("decision_event_id_unique")),
        "seed_coverage": bool(metrics.get("seed_coverage")),
        "all_episode_gates": bool(metrics.get("all_episode_gates")),
        "all_requests_complete": int(metrics.get("completed_requests", -1)) == int(metrics.get("requested", -2)),
    }
    gates["all"] = all(gates.values())
    return gates


def cross_split_checks(rows: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    train = [row for row in rows if row.get("proposal_set_support_split") == "train"]
    dev = [row for row in rows if row.get("proposal_set_support_split") == "dev"]
    train_seeds, dev_seeds = {int(row["seed"]) for row in train}, {int(row["seed"]) for row in dev}
    train_events = {str(row["decision_event_id"]) for row in train}
    dev_events = {str(row["decision_event_id"]) for row in dev}
    return {
        "train_dev_seed_disjoint": not bool(train_seeds & dev_seeds),
        "train_dev_event_disjoint": not bool(train_events & dev_events),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True, separators=(",", ":")) + "\n")
    return _sha256(path)


def _reserve(marker: Path, history_root: Path, output: Path) -> dict[str, Any]:
    registry = historical_registry(history_root, output)
    seeds = set(TRAIN_SEEDS) | set(DEV_SEEDS)
    overlap = sorted(seeds & set(registry.get("seeds", set())))
    _require(not overlap, f"formal proposal-set support seeds already consumed: {overlap}")
    payload = {
        "protocol": STAGE, "protocol_sha256": PROTOCOL_SHA,
        "feasible_set_manifest_sha256": PREDECESSOR_MANIFEST_SHA,
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "train_seeds": list(TRAIN_SEEDS), "dev_seeds": list(DEV_SEEDS),
        "output": str(output), "reserved_before_first_reset": True,
    }
    reserve_once(marker, payload)
    return {"historical": registry, "payload": payload, "marker": str(marker)}


def run(
    output: str | Path, feasible_set_manifest: str | Path,
    candidate_checkpoint: str | Path, history_root: str | Path, *, smoke: bool = False,
) -> dict[str, Any]:
    protocol()
    output_path = Path(output)
    _require(not output_path.exists(), f"output already exists: {output_path}")
    validate_predecessor(feasible_set_manifest)
    policy, payload = effectiveness.load_candidate_checkpoint(candidate_checkpoint)
    output_path.mkdir(parents=True, exist_ok=False)
    reservation = None
    if not smoke:
        history_path = Path(history_root)
        _require(history_path.is_dir(), f"history root missing: {history_path}")
        marker = Path(candidate_checkpoint).parent / RESERVATION_MARKER
        _require(not marker.exists(), f"proposal-set support marker already exists: {marker}")
        reservation = _reserve(marker, history_path, output_path)
    plan = [("smoke", SMOKE_SEED)] if smoke else [
        *(("train", seed) for seed in TRAIN_SEEDS),
        *(("dev", seed) for seed in DEV_SEEDS),
    ]
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    exception_text = None
    for index, (split, seed) in enumerate(plan):
        try:
            config = attribution.diagnostic_config()
            max_steps = SMOKE_STEPS if smoke else int(config.max_simulation_seconds / config.dt)
            record, episode_rows = attribution._episode(
                policy, seed=int(seed), output=output_path / f"{split}_{seed}_flow12",
                max_steps=max_steps, smoke=smoke, collect_catalogue_diagnostics=True,
                catalogue_scope="teacher_release",
            )
            retained = retain_set_rows(policy, episode_rows, split=split)
            teacher_release_rows = [row for row in episode_rows
                                    if bool(row.get("actor_mask")) and int(row.get("teacher_gate", -1)) == 1]
            record = dict(record)
            record["split"] = split
            record["proposal_set_rows"] = len(retained)
            record["teacher_release_rows"] = len(teacher_release_rows)
            record["gates"] = dict(record["gates"])
            record["gates"].update({
                "catalogue_complete": len(retained) == len(teacher_release_rows) and bool(retained),
                "feasible_set_nonempty": bool(retained) and all(row["proposal_set_target"] for row in retained),
                "teacher_selected_proposal_feasible": bool(retained) and all(
                    int(row["teacher_proposal"]) in set(row["proposal_set_target"]) for row in retained
                ),
            })
            record["gates"]["all"] = all(
                bool(value) for key, value in record["gates"].items() if key != "all"
            )
            records.append(record)
            rows.extend(retained)
        except Exception as exc:
            exception_text = f"{type(exc).__name__}: {exc}"
            records.append({"split": split, "seed": int(seed), "exception": exception_text,
                            "gates": {"all": False}})
            break
        _write_json(output_path / "progress.json", {
            "stage": STAGE, "smoke": bool(smoke), "episodes_completed": index + 1,
            "episodes_planned": len(plan), "proposal_set_rows": len(rows),
            "last_split": split, "last_seed": int(seed), "last_gates": records[-1]["gates"],
        })
        if not record["gates"]["all"]:
            exception_text = "episode validity failure"
            break
    train_rows = [row for row in rows if row.get("proposal_set_support_split") == "train"]
    dev_rows = [row for row in rows if row.get("proposal_set_support_split") == "dev"]
    smoke_rows = [row for row in rows if row.get("proposal_set_support_split") == "smoke"]
    train_path, dev_path = output_path / "proposal_set_train.jsonl", output_path / "proposal_set_dev.jsonl"
    train_sha, dev_sha = _write_jsonl(train_path, train_rows), _write_jsonl(dev_path, dev_rows)
    train_metrics = split_metrics(rows, "train", records, TRAIN_SEEDS)
    dev_metrics = split_metrics(rows, "dev", records, DEV_SEEDS)
    train_gates = split_gates(train_metrics, "train", formal=not smoke)
    dev_gates = split_gates(dev_metrics, "dev", formal=not smoke)
    cross = cross_split_checks(rows)
    passed = bool(
        not smoke and exception_text is None and train_gates["all"] and dev_gates["all"]
        and all(cross.values())
    )
    result = {
        "stage": STAGE, "status": "PASS" if passed else ("INCOMPLETE" if smoke else "FAIL"),
        "evidence_type": "newly_run", "smoke": bool(smoke),
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA,
        "feasible_set_manifest": str(feasible_set_manifest),
        "feasible_set_manifest_sha256": PREDECESSOR_MANIFEST_SHA,
        "candidate_checkpoint": str(candidate_checkpoint), "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "candidate_payload": {"stage": payload.get("stage"), "architecture_id": payload.get("architecture_id"),
                              "temperature": payload.get("temperature"), "optimizer_loaded": False},
        "train_seeds": list(TRAIN_SEEDS), "dev_seeds": list(DEV_SEEDS),
        "records": records, "rows": len(rows), "smoke_rows": len(smoke_rows),
        "train": {"path": str(train_path), "sha256": train_sha,
                  "metrics": train_metrics, "gates": train_gates},
        "dev": {"path": str(dev_path), "sha256": dev_sha,
                "metrics": dev_metrics, "gates": dev_gates},
        "cross_split_checks": cross, "exception": exception_text, "reservation": reservation,
        "training_called": False, "optimizer_called": False, "policy_update": False,
        "teacher_called": True, "teacher_label_only": True, "student_authoritative": True,
        "PROPOSAL_SET_SUPPORT_COLLECTION": "PASS" if passed else "FAIL",
        "PROPOSAL_SET_SUPPORT_FROZEN": passed,
        "PROPOSAL_SET_LEARNABILITY_AUDIT_ALLOWED": passed,
        "PROPOSAL_REPAIR_ALLOWED": False, "GATE_REPAIR_ALLOWED": False,
        "PPO_ALLOWED": False, "P3_ALLOWED": False,
        "curriculum_promotion_allowed": False, "checkpoint_promotable": False,
        "source_hashes": {"runner": _sha256(Path(__file__)), "protocol": _sha256(PROTOCOL)},
    }
    manifest_path = output_path / "manifest.json"
    result["manifest"] = str(manifest_path)
    _write_json(manifest_path, result)
    print(json.dumps({
        "stage": STAGE, "status": result["status"], "smoke": bool(smoke),
        "train_rows": len(train_rows), "dev_rows": len(dev_rows), "smoke_rows": len(smoke_rows),
        "PROPOSAL_SET_LEARNABILITY_AUDIT_ALLOWED": passed, "PPO_ALLOWED": False,
    }, sort_keys=True), flush=True)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feasible-set-manifest", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    torch.set_num_threads(1)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    _main()
