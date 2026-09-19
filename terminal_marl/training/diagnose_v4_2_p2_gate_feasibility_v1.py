"""Fresh, descriptive attribution of P2 gate/proposal feasibility in flow12.

The repaired student remains execution-authoritative.  The Teacher is a
label-only sidecar, and no row produced here is authorized for training.
"""
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

from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import collect_v4_2_p2_gate_support as support
from terminal_marl.training import evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import reserve_once
from terminal_marl.training.student_rollout_v4_2 import rollout


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_feasibility_attribution_v1.yaml")
PROTOCOL_SHA = "50b2cb7dc40160a54ecfe150122d7bcc3407b638a40e887b20045120e220b1fd"
STAGE = "V4_2_P2_GATE_FEASIBILITY_ATTRIBUTION_V1"
EFFECTIVENESS_MANIFEST_SHA = "6aaebeac4dc8dcf425ce250df6c6abcd991921263605b3dd71d3d717f0783e88"
EFFECTIVENESS_PROTOCOL_SHA = "5882361adcabaff4c0017cae73a8b95dc8029e204942bf4a83879981bd321d84"
CANDIDATE_SHA = effectiveness.REPAIRED_CHECKPOINT_SHA
FORMAL_SEEDS = (789001, 789002, 789003, 789004, 789005, 789006)
FORMAL_SCENARIOS = ("flow12",) * len(FORMAL_SEEDS)
SMOKE_SEED = 789000
SMOKE_STEPS = 160
RESERVATION_MARKER = "p2_gate_feasibility_attribution_v1_consumed.json"
INTERFACES = support.INTERFACES


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
        _require(_sha256(PROTOCOL) == PROTOCOL_SHA, "attribution protocol changed after preregistration")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict) and value.get("protocol") == STAGE, "protocol id changed")
    predecessor = value.get("predecessor", {})
    _require(predecessor.get("effectiveness_manifest_sha256") == EFFECTIVENESS_MANIFEST_SHA,
             "effectiveness manifest binding changed")
    _require(predecessor.get("effectiveness_protocol_sha256") == EFFECTIVENESS_PROTOCOL_SHA,
             "effectiveness protocol binding changed")
    _require(predecessor.get("effectiveness_status") == "FAIL", "effectiveness terminal status changed")
    formal = value.get("formal", {})
    _require(formal.get("seeds") == list(FORMAL_SEEDS), "formal seeds changed")
    _require(formal.get("scenarios") == list(FORMAL_SCENARIOS), "formal scenarios changed")
    smoke = value.get("smoke", {})
    _require(smoke.get("seed") == SMOKE_SEED and smoke.get("steps") == SMOKE_STEPS,
             "smoke binding changed")
    bindings = value.get("bindings", {})
    for key, expected in {
        "deterministic": True,
        "teacher_mode": "teacher_label_only",
        "student_authoritative": True,
        "no_training": True,
        "no_optimizer": True,
        "no_policy_update": True,
    }.items():
        _require(bindings.get(key) == expected, f"binding changed: {key}")
    return value


def validate_effectiveness_manifest(path: str | Path) -> dict[str, Any]:
    _require(_sha256(path) == EFFECTIVENESS_MANIFEST_SHA, "effectiveness manifest SHA mismatch")
    value = _read_json(path)
    _require(value.get("status") == "FAIL", "effectiveness result is not terminal FAIL")
    _require(value.get("protocol_sha256") == EFFECTIVENESS_PROTOCOL_SHA,
             "effectiveness protocol mismatch")
    _require(value.get("candidate_checkpoint_sha256") == CANDIDATE_SHA,
             "effectiveness candidate mismatch")
    _require(value.get("hard_gates", {}).get("formal_shape") is True,
             "effectiveness hard evidence is incomplete")
    _require(value.get("effectiveness_gates", {}).get("wait_gap_closure_ge_50pct") is False,
             "expected failed effectiveness gate missing")
    _require(value.get("P2_PPO_ADMISSION_REVIEW_ALLOWED") is False
             and value.get("PPO_ALLOWED") is False, "effectiveness authorization boundary changed")
    return value


def diagnostic_config():
    return support.support_config(12)


def _release_rejected(row: Mapping[str, Any]) -> bool:
    return int(row.get("requested_gate", -1)) == 1 and row.get("release_decision") != "approved"


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [dict(row) for row in rows]
    requested_release = [row for row in selected if int(row.get("requested_gate", -1)) == 1]
    both_release = [row for row in requested_release if int(row.get("teacher_gate", -1)) == 1]
    rejected = [row for row in requested_release if _release_rejected(row)]
    rejected_teacher_release = [row for row in rejected if int(row.get("teacher_gate", -1)) == 1]
    proposal_match = [
        row for row in both_release
        if row.get("requested_proposal") is not None
        and row.get("requested_proposal") == row.get("teacher_proposal")
    ]
    mismatch_rejected = [
        row for row in rejected_teacher_release
        if row.get("requested_proposal") != row.get("teacher_proposal")
    ]
    decisions = Counter(str(row.get("release_decision")) for row in requested_release)
    blockers = Counter(str(row.get("release_primary_blocker")) for row in rejected)
    retry_values = [int(row.get("release_retry_count") or 0) for row in requested_release]
    result = {
        "actor_events": len(selected),
        "requested_gate": {
            "WAIT": sum(int(row.get("requested_gate", -1)) == 0 for row in selected),
            "RELEASE": len(requested_release),
        },
        "teacher_gate": {
            "WAIT": sum(int(row.get("teacher_gate", -1)) == 0 for row in selected),
            "RELEASE": sum(int(row.get("teacher_gate", -1)) == 1 for row in selected),
        },
        "gate_disagreement": {
            "false_WAIT": sum(int(row.get("requested_gate", -1)) == 0 and int(row.get("teacher_gate", -1)) == 1 for row in selected),
            "false_RELEASE": sum(int(row.get("requested_gate", -1)) == 1 and int(row.get("teacher_gate", -1)) == 0 for row in selected),
        },
        "both_release": len(both_release),
        "proposal_match": len(proposal_match),
        "proposal_mismatch": len(both_release) - len(proposal_match),
        "release_attempts": len(requested_release),
        "release_accepted": int(decisions["approved"]),
        "release_rejected": len(rejected),
        "release_decisions": dict(sorted(decisions.items())),
        "rejected_primary_blockers": dict(sorted(blockers.items())),
        "rejected_while_teacher_release": len(rejected_teacher_release),
        "proposal_mismatch_and_rejected_while_teacher_release": len(mismatch_rejected),
        "retry_events": sum(value > 0 for value in retry_values),
        "retry_count_sum": sum(retry_values),
        "release_telemetry_complete": all(
            int(row.get("requested_gate", -1)) == 0 or bool(row.get("release_diagnostic_present"))
            for row in selected
        ),
    }
    return result


def attribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    by_interface = {
        interface: summarize_rows([row for row in values if row.get("interface") == interface])
        for interface in INTERFACES
    }
    by_arrival_half = {
        "early_0_5": summarize_rows([row for row in values if int(row.get("arrival_id", -1)) < 6]),
        "late_6_11": summarize_rows([row for row in values if int(row.get("arrival_id", -1)) >= 6]),
    }
    by_interface_and_half = {
        interface: {
            "early_0_5": summarize_rows([
                row for row in values
                if row.get("interface") == interface and int(row.get("arrival_id", -1)) < 6
            ]),
            "late_6_11": summarize_rows([
                row for row in values
                if row.get("interface") == interface and int(row.get("arrival_id", -1)) >= 6
            ]),
        }
        for interface in INTERFACES
    }
    return {
        "overall": summarize_rows(values),
        "by_interface": by_interface,
        "by_arrival_half": by_arrival_half,
        "by_interface_and_arrival_half": by_interface_and_half,
    }


def _episode(
    policy: Any, *, seed: int, output: Path, max_steps: int, smoke: bool,
    collect_catalogue_diagnostics: bool = False,
    catalogue_scope: str = "requested_release",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = diagnostic_config()
    _, summary = rollout(
        policy, config, seed=seed, max_steps=max_steps, output=output,
        deterministic=True, collect_labels=True, collect_diagnostics=True,
        collect_catalogue_diagnostics=collect_catalogue_diagnostics,
        catalogue_scope=catalogue_scope,
    )
    labels_path = output / "labels.jsonl"
    raw = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = support._actor_rows(raw, split="diagnostic", scenario="flow12")
    replay, geometry, lifecycle = effectiveness._geometry_replay(config, summary["action_tape"])
    likelihood = support._policy_tape_logprob_exact(policy, config, summary["action_tape"])
    flags = support._tape_flags(Path(summary["action_tape"]))
    wait_seconds, request_to_touchdown = lifecycle
    safety_zero = not any(bool(value) for value in summary.get("safety", {}).values())
    formal_complete = bool(summary.get("terminal") and summary.get("completed") == 12)
    geometry_ok = bool(
        geometry.get("samples", 0) > 0
        and geometry.get("reverse_samples", 0) == 0
        and geometry.get("max_inner_excursion_m", float("inf")) <= 0.05
    )
    row_integrity = bool(rows) and all(
        row.get("interface") in INTERFACES
        and "arrival_id" in row
        and (int(row.get("requested_gate", -1)) == 0 or row.get("release_diagnostic_present") is True)
        for row in rows
    )
    gates = {
        "formal_all_requests_complete": formal_complete if not smoke else True,
        "safety_all_zero": safety_zero,
        "exact_action_tape_replay": bool(replay.get("exact")),
        "exact_requested_likelihood": bool(likelihood),
        "geometry": geometry_ok,
        "teacher_label_only": bool(flags["teacher_label_only"]),
        "student_authoritative": bool(flags["student_authoritative"]),
        "diagnostic_rows_complete": row_integrity,
        "no_training": True,
    }
    gates["all"] = all(gates.values())
    record = {
        "seed": int(seed), "scenario": "flow12", "smoke": bool(smoke),
        "requests": 12, "completed": int(summary.get("completed", 0)),
        "terminal": bool(summary.get("terminal")), "seconds": float(summary.get("seconds", 0.0)),
        "actor_rows": len(rows), "safety": summary.get("safety", {}),
        "geometry": geometry, "replay_exact": bool(replay.get("exact")),
        "requested_logprob_exact": bool(likelihood),
        "weighted_wait_seconds": wait_seconds,
        "request_to_touchdown_seconds": request_to_touchdown,
        "attribution": attribution(rows), "gates": gates,
    }
    return record, rows


def _reserve(marker: Path, history_root: Path, output: Path) -> dict[str, Any]:
    registry = historical_registry(history_root, output)
    overlap = sorted(set(FORMAL_SEEDS) & set(registry.get("seeds", set())))
    _require(not overlap, f"formal attribution seeds already consumed: {overlap}")
    payload = {
        "protocol": STAGE,
        "protocol_sha256": PROTOCOL_SHA or _sha256(PROTOCOL),
        "effectiveness_manifest_sha256": EFFECTIVENESS_MANIFEST_SHA,
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "formal_seeds": list(FORMAL_SEEDS),
        "formal_scenarios": list(FORMAL_SCENARIOS),
        "output": str(output),
        "reserved_before_first_reset": True,
    }
    reserve_once(marker, payload)
    return {"historical": registry, "payload": payload, "marker": str(marker)}


def run(
    output: str | Path, effectiveness_manifest: str | Path,
    candidate_checkpoint: str | Path, history_root: str | Path, *, smoke: bool = False,
) -> dict[str, Any]:
    protocol()
    output_path = Path(output)
    _require(not output_path.exists(), f"output already exists: {output_path}")
    validate_effectiveness_manifest(effectiveness_manifest)
    policy, payload = effectiveness.load_candidate_checkpoint(candidate_checkpoint)
    output_path.mkdir(parents=True, exist_ok=False)
    reservation = None
    if not smoke:
        history_path = Path(history_root)
        _require(history_path.is_dir(), f"history root missing: {history_path}")
        marker = Path(candidate_checkpoint).parent / RESERVATION_MARKER
        _require(not marker.exists(), f"formal attribution marker already exists: {marker}")
        reservation = _reserve(marker, history_path, output_path)
    seeds = (SMOKE_SEED,) if smoke else FORMAL_SEEDS
    records: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    exception_text = None
    for index, seed in enumerate(seeds):
        try:
            config = diagnostic_config()
            max_steps = SMOKE_STEPS if smoke else int(config.max_simulation_seconds / config.dt)
            record, rows = _episode(
                policy, seed=int(seed), output=output_path / f"seed{seed}_flow12",
                max_steps=max_steps, smoke=smoke,
            )
            records.append(record)
            all_rows.extend(rows)
        except Exception as exc:
            exception_text = f"{type(exc).__name__}: {exc}"
            records.append({"seed": int(seed), "scenario": "flow12", "exception": exception_text,
                            "gates": {"all": False}})
            break
        _write_json(output_path / "progress.json", {
            "stage": STAGE, "smoke": bool(smoke), "episodes_completed": index + 1,
            "episodes_planned": len(seeds), "rows": len(all_rows),
            "last_seed": int(seed), "last_gates": records[-1]["gates"],
        })
        if not record["gates"]["all"]:
            exception_text = "episode validity failure"
            break
    rows_path = output_path / "diagnostic_rows.jsonl"
    with rows_path.open("x", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True, separators=(",", ":")) + "\n")
    formal_shape = bool(
        not smoke and len(records) == len(FORMAL_SEEDS)
        and [record.get("seed") for record in records] == list(FORMAL_SEEDS)
        and all(record.get("gates", {}).get("all") is True for record in records)
    )
    diagnostic_complete = bool(formal_shape and exception_text is None)
    result = {
        "stage": STAGE,
        "status": "COMPLETE" if diagnostic_complete else "INCOMPLETE",
        "evidence_type": "newly_run",
        "smoke": bool(smoke),
        "protocol": str(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA or _sha256(PROTOCOL),
        "effectiveness_manifest": str(effectiveness_manifest),
        "effectiveness_manifest_sha256": EFFECTIVENESS_MANIFEST_SHA,
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "candidate_payload": {
            "stage": payload.get("stage"), "architecture_id": payload.get("architecture_id"),
            "temperature": payload.get("temperature"), "optimizer_loaded": False,
        },
        "formal_seeds": list(FORMAL_SEEDS), "formal_scenarios": list(FORMAL_SCENARIOS),
        "records": records, "rows": len(all_rows),
        "diagnostic_rows": str(rows_path), "diagnostic_rows_sha256": _sha256(rows_path),
        "attribution": attribution(all_rows),
        "formal_shape": formal_shape, "exception": exception_text,
        "reservation": reservation,
        "training_called": False, "optimizer_called": False, "policy_update": False,
        "teacher_called": True, "teacher_label_only": True, "student_authoritative": True,
        "rows_authorized_for_training": False,
        "DIAGNOSTIC_COMPLETE": diagnostic_complete,
        "SUPPORT_COLLECTION_ALLOWED": False, "GATE_REPAIR_ALLOWED": False,
        "PROPOSAL_REPAIR_ALLOWED": False, "PPO_ALLOWED": False, "P3_ALLOWED": False,
        "curriculum_promotion_allowed": False, "checkpoint_promotable": False,
        "source_hashes": {"runner": _sha256(Path(__file__)), "protocol": _sha256(PROTOCOL)},
    }
    manifest_path = output_path / "manifest.json"
    result["manifest"] = str(manifest_path)
    _write_json(manifest_path, result)
    print(json.dumps({
        "stage": STAGE, "status": result["status"], "smoke": bool(smoke),
        "episodes": len(records), "rows": len(all_rows),
        "DIAGNOSTIC_COMPLETE": diagnostic_complete, "PPO_ALLOWED": False,
    }, sort_keys=True), flush=True)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effectiveness-manifest", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    torch.set_num_threads(1)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    _main()
