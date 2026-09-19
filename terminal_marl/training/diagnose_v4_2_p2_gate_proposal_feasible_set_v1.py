"""Fresh set-valued feasibility diagnostic for conditional proposals."""
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

from terminal_marl.training import audit_v4_2_p2_gate_proposal_learnability_v1 as proposal_audit
from terminal_marl.training import diagnose_v4_2_p2_gate_feasibility_v1 as attribution
from terminal_marl.training import diagnose_v4_2_p2_gate_proposal_failure_v1 as failure
from terminal_marl.training import evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import reserve_once


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_proposal_feasible_set_v1.yaml")
PROTOCOL_SHA = "dbe387cc134741f87cde8caa2b8910410e42e5ad756dcb197bc2c04c208eabca"
STAGE = "V4_2_P2_GATE_PROPOSAL_FEASIBLE_SET_V1"
FAILURE_MANIFEST_SHA = "3c8e0980f4f970bb66ca207ceb3305dc3fbe9e3eb79e9e7a827a1b086acf11f5"
CANDIDATE_SHA = effectiveness.REPAIRED_CHECKPOINT_SHA
FORMAL_SEEDS = (794001, 794002, 794003)
SMOKE_SEED = 794000
SMOKE_STEPS = 160
RESERVATION_MARKER = "p2_gate_proposal_feasible_set_v1_consumed.json"


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
    if PROTOCOL_SHA:
        _require(_sha256(PROTOCOL) == PROTOCOL_SHA, "feasible-set protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict) and value.get("protocol") == STAGE, "protocol id changed")
    predecessor = value.get("predecessor", {})
    for key, expected in {
        "failure_attribution_manifest_sha256": FAILURE_MANIFEST_SHA,
        "failure_attribution_protocol_sha256": failure.PROTOCOL_SHA,
        "failure_attribution_status": "COMPLETE",
        "classification": "observation_or_label_learnability_limit_supported",
        "repaired_checkpoint_sha256": CANDIDATE_SHA,
    }.items():
        _require(predecessor.get(key) == expected, f"predecessor changed: {key}")
    bindings = value.get("bindings", {})
    for key, expected in {
        "scenario": "flow12", "deterministic": True, "fixed_geometry": True,
        "student_authoritative": True, "teacher_mode": "teacher_label_only",
        "catalogue_scope": "actor_requested_RELEASE_only",
        "catalogue_semantics": "all_25_RELEASE_proposals_checked_before_env_step",
        "no_training": True, "no_optimizer": True, "no_policy_update": True,
    }.items():
        _require(bindings.get(key) == expected, f"binding changed: {key}")
    formal = value.get("formal", {})
    _require(formal.get("seeds") == list(FORMAL_SEEDS), "formal seeds changed")
    _require(formal.get("scenarios") == ["flow12"] * len(FORMAL_SEEDS), "formal scenarios changed")
    _require(value.get("smoke", {}).get("seed") == SMOKE_SEED, "smoke seed changed")
    return value


def validate_failure_manifest(path: str | Path) -> dict[str, Any]:
    _require(_sha256(path) == FAILURE_MANIFEST_SHA, "failure manifest SHA mismatch")
    value = _read_json(path)
    for key, expected in {
        "stage": failure.STAGE, "status": "COMPLETE",
        "classification": "observation_or_label_learnability_limit_supported",
        "standalone_mlp_pass": False,
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "PROPOSAL_REPAIR_ALLOWED": False, "PPO_ALLOWED": False,
    }.items():
        _require(value.get(key) == expected, f"failure manifest changed: {key}")
    return value


def annotate_policy_ranks(policy: Any, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows if int(row.get("requested_gate", -1)) == 1]
    if not selected:
        return []
    observations = torch.as_tensor(
        np.asarray([row["observation_v2_1"] for row in selected], dtype=np.float32)
    )
    logits = proposal_audit.base_logits(policy, observations)
    ranks = logits.argsort(dim=-1, descending=True)
    for index, row in enumerate(selected):
        feasible = {int(value) for value in row.get("teacher_feasible_proposals", ())}
        row["policy_top1_proposal"] = int(ranks[index, 0])
        row["policy_top3_proposals"] = [int(value) for value in ranks[index, :3]]
        row["policy_top1_feasible"] = int(ranks[index, 0]) in feasible
        row["policy_top3_feasible_hit"] = any(int(value) in feasible for value in ranks[index, :3])
        row["candidate_proposal_feasible"] = int(row["requested_proposal"]) in feasible
        row["teacher_proposal_feasible"] = (
            int(row.get("teacher_gate", -1)) == 1
            and int(row.get("teacher_proposal")) in feasible
        )
        row["nominal_proposal_12_feasible"] = 12 in feasible
        row["feasible_set_size"] = len(feasible)
        row["execution_accepted"] = row.get("release_decision") == "approved"
    return selected


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    sizes = [int(row.get("feasible_set_size", 0)) for row in values]
    cross = Counter(
        f"candidate_feasible={bool(row.get('candidate_proposal_feasible'))}|accepted={bool(row.get('execution_accepted'))}"
        for row in values
    )
    teacher_release = [row for row in values if int(row.get("teacher_gate", -1)) == 1]
    return {
        "requested_release_events": len(values),
        "teacher_release_events": len(teacher_release),
        "empty_feasible_set_events": sum(size == 0 for size in sizes),
        "feasible_set_size": {
            "min": min(sizes, default=0), "max": max(sizes, default=0),
            "mean": float(np.mean(sizes)) if sizes else 0.0,
            "median": float(np.median(sizes)) if sizes else 0.0,
        },
        "candidate_proposal_feasible": sum(bool(row.get("candidate_proposal_feasible")) for row in values),
        "teacher_proposal_feasible": sum(bool(row.get("teacher_proposal_feasible")) for row in teacher_release),
        "nominal_proposal_12_feasible": sum(bool(row.get("nominal_proposal_12_feasible")) for row in values),
        "policy_top1_feasible": sum(bool(row.get("policy_top1_feasible")) for row in values),
        "policy_top3_feasible_hit": sum(bool(row.get("policy_top3_feasible_hit")) for row in values),
        "execution_accepted": sum(bool(row.get("execution_accepted")) for row in values),
        "execution_cross_candidate_feasibility": dict(sorted(cross.items())),
        "catalogue_complete": bool(values) and all(row.get("teacher_catalogue_complete") is True for row in values),
    }


def sliced_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    return {
        "overall": summarize(values),
        "by_interface": {
            name: summarize([row for row in values if row.get("interface") == name])
            for name in attribution.INTERFACES
        },
        "by_arrival_half": {
            "early_0_5": summarize([row for row in values if int(row.get("arrival_id", -1)) < 6]),
            "late_6_11": summarize([row for row in values if int(row.get("arrival_id", -1)) >= 6]),
        },
    }


def _reserve(marker: Path, history_root: Path, output: Path) -> dict[str, Any]:
    registry = historical_registry(history_root, output)
    overlap = sorted(set(FORMAL_SEEDS) & set(registry.get("seeds", set())))
    _require(not overlap, f"formal feasible-set seeds already consumed: {overlap}")
    payload = {
        "protocol": STAGE, "protocol_sha256": PROTOCOL_SHA or _sha256(PROTOCOL),
        "failure_attribution_manifest_sha256": FAILURE_MANIFEST_SHA,
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "formal_seeds": list(FORMAL_SEEDS), "output": str(output),
        "reserved_before_first_reset": True,
    }
    reserve_once(marker, payload)
    return {"historical": registry, "payload": payload, "marker": str(marker)}


def run(
    output: str | Path, failure_manifest: str | Path,
    candidate_checkpoint: str | Path, history_root: str | Path, *, smoke: bool = False,
) -> dict[str, Any]:
    protocol()
    output_path = Path(output)
    _require(not output_path.exists(), f"output already exists: {output_path}")
    validate_failure_manifest(failure_manifest)
    policy, _payload = effectiveness.load_candidate_checkpoint(candidate_checkpoint)
    output_path.mkdir(parents=True, exist_ok=False)
    reservation = None
    if not smoke:
        marker = Path(candidate_checkpoint).parent / RESERVATION_MARKER
        _require(not marker.exists(), f"feasible-set marker already exists: {marker}")
        reservation = _reserve(marker, Path(history_root), output_path)
    seeds = (SMOKE_SEED,) if smoke else FORMAL_SEEDS
    records, rows = [], []
    exception_text = None
    for index, seed in enumerate(seeds):
        try:
            config = attribution.diagnostic_config()
            max_steps = SMOKE_STEPS if smoke else int(config.max_simulation_seconds / config.dt)
            record, episode_rows = attribution._episode(
                policy, seed=int(seed), output=output_path / f"seed{seed}_flow12",
                max_steps=max_steps, smoke=smoke, collect_catalogue_diagnostics=True,
            )
            release_rows = annotate_policy_ranks(policy, episode_rows)
            catalogue_complete = bool(release_rows) and all(
                row.get("teacher_catalogue_complete") is True for row in release_rows
            )
            teacher_consistent = all(
                int(row.get("teacher_gate", -1)) == 0 or bool(row.get("teacher_proposal_feasible"))
                for row in release_rows
            )
            record = dict(record)
            record["catalogue_rows"] = len(release_rows)
            record["catalogue_summary"] = sliced_summary(release_rows)
            record["gates"] = dict(record["gates"])
            record["gates"]["catalogue_complete"] = catalogue_complete
            record["gates"]["teacher_selected_proposal_feasible"] = teacher_consistent
            record["gates"]["all"] = all(
                bool(value) for key, value in record["gates"].items() if key != "all"
            )
            records.append(record)
            rows.extend(release_rows)
        except Exception as exc:
            exception_text = f"{type(exc).__name__}: {exc}"
            records.append({"seed": int(seed), "exception": exception_text, "gates": {"all": False}})
            break
        _write_json(output_path / "progress.json", {
            "stage": STAGE, "smoke": bool(smoke), "episodes_completed": index + 1,
            "episodes_planned": len(seeds), "catalogue_rows": len(rows),
            "last_seed": int(seed), "last_gates": records[-1]["gates"],
        })
        if not record["gates"]["all"]:
            exception_text = "episode validity failure"
            break
    rows_path = output_path / "catalogue_rows.jsonl"
    with rows_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True, separators=(",", ":")) + "\n")
    complete = bool(
        not smoke and exception_text is None and len(records) == len(FORMAL_SEEDS)
        and all(record.get("gates", {}).get("all") is True for record in records)
    )
    result = {
        "stage": STAGE, "status": "COMPLETE" if complete else "INCOMPLETE",
        "evidence_type": "newly_run", "smoke": bool(smoke),
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA or _sha256(PROTOCOL),
        "failure_manifest": str(failure_manifest), "failure_manifest_sha256": FAILURE_MANIFEST_SHA,
        "candidate_checkpoint": str(candidate_checkpoint), "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "formal_seeds": list(FORMAL_SEEDS), "records": records, "catalogue_rows": len(rows),
        "catalogue_rows_path": str(rows_path), "catalogue_rows_sha256": _sha256(rows_path),
        "summary": sliced_summary(rows), "exception": exception_text, "reservation": reservation,
        "DIAGNOSTIC_COMPLETE": complete,
        "training_called": False, "optimizer_called": False, "policy_update": False,
        "teacher_called": True, "teacher_label_only": True, "student_authoritative": True,
        "rows_authorized_for_training": False,
        "SET_VALUED_SUPPORT_ALLOWED": False, "PROPOSAL_REPAIR_ALLOWED": False,
        "GATE_REPAIR_ALLOWED": False, "PPO_ALLOWED": False, "P3_ALLOWED": False,
        "curriculum_promotion_allowed": False, "checkpoint_promotable": False,
        "source_hashes": {"runner": _sha256(Path(__file__)), "protocol": _sha256(PROTOCOL)},
    }
    manifest_path = output_path / "manifest.json"
    result["manifest"] = str(manifest_path)
    _write_json(manifest_path, result)
    print(json.dumps({
        "stage": STAGE, "status": result["status"], "smoke": bool(smoke),
        "episodes": len(records), "catalogue_rows": len(rows),
        "DIAGNOSTIC_COMPLETE": complete, "PPO_ALLOWED": False,
    }, sort_keys=True), flush=True)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-manifest", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    torch.set_num_threads(1)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    _main()
