"""Compare state-only Teacher slot witnesses with late-slot reconciliation toggled."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
import math
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from terminal_marl.core.low_level_joint_action_v4_2 import hierarchical_from_joint_index_v4_1
from terminal_marl.core.merge_scheduler import LowLevelIntent, RepairCandidate
from terminal_marl.core.safety_trace import SafetyTraceRecorder, _digest
from terminal_marl.core.config import RingRole
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot
from terminal_marl.training import evaluate_p3_adaptive_kl_v7 as evaluator
from terminal_marl.training.v4_1_joint_teacher import teacher_joint_expert_v4_1


STAGE = "P3_V7_RUN0_SLOT_STATE_COMPARE_V1"
TARGET_UID = 72
TARGET_ARRIVAL_ID = 15
EXPECTED_SEED = 873106
EXPECTED_STEPS = 897
SAMPLE_TIMES = (136, 152, 200, 250, 297, 500)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(_jsonable(value), sort_keys=True, allow_nan=False))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rng_snapshot() -> tuple[Any, tuple[Any, ...], torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _rng_equal(before: tuple[Any, tuple[Any, ...], torch.Tensor], after: tuple[Any, tuple[Any, ...], torch.Tensor]) -> bool:
    py_before, np_before, torch_before = before
    py_after, np_after, torch_after = after
    return (
        py_before == py_after
        and np_before[0] == np_after[0]
        and np.array_equal(np_before[1], np_after[1])
        and np_before[2:] == np_after[2:]
        and bool(torch.equal(torch_before, torch_after))
    )


def _find_uav(env: TerminalParallelEnv, uid: int) -> Any | None:
    if int(uid) in env.active:
        return env.active[int(uid)]
    if int(uid) in env.service_uavs:
        return env.service_uavs[int(uid)]
    for uav in env.completed_uavs:
        if int(uav.uid) == int(uid):
            return uav
    return None


def _state(env: TerminalParallelEnv, uid: int) -> dict[str, Any]:
    uav = _find_uav(env, uid)
    if uav is None:
        return {"uid": int(uid), "present": False}
    return {
        "uid": int(uav.uid), "arrival_id": int(uav.arrival_id), "present": True,
        "phase": getattr(uav.phase, "value", uav.phase),
        "ring_role": None if uav.ring_role is None else getattr(uav.ring_role, "value", uav.ring_role),
        "soc": float(uav.soc), "energy_remaining": float(uav.energy_remaining),
        "committed": uav.committed_plan is not None,
        "safety_retry_after": float(uav.safety_retry_after),
        "emergency_retry_after": float(uav.emergency_retry_after),
    }


def _restore_shadow_methods(shadow: TerminalParallelEnv) -> None:
    cls = type(shadow)
    for name in ("step", "_release", "_try_emergency_landing", "_candidate_energy_guard_reason"):
        setattr(shadow, name, getattr(cls, name).__get__(shadow, cls))
    safety_cls = type(shadow.safety)
    shadow.safety.filter_plan = safety_cls.filter_plan.__get__(shadow.safety, safety_cls)


def _prefix_check(tape: Path, config: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = json.loads(tape.read_text(encoding="utf-8"))
    payload["action_tape"] = rows[:2]
    with tempfile.TemporaryDirectory(prefix="p3_v7_slot_compare_") as directory:
        prefix = Path(directory) / "prefix.json"
        prefix.write_text(json.dumps(_jsonable(payload), sort_keys=True, allow_nan=False), encoding="utf-8")
        replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), prefix)
    return {
        "rows": 2, "json_roundtrip": True, "exact": bool(replay.get("exact")),
        "steps_expected": replay.get("steps_expected"), "steps_replayed": replay.get("steps_replayed"),
    }


def _candidate_context(uav: Any) -> tuple[RingRole | str, str] | None:
    phase = getattr(uav.phase, "value", uav.phase)
    if phase == "ring_outer":
        return RingRole.MIDDLE, "o2m"
    if phase == "ring_middle":
        return RingRole.INNER, "m2i"
    if phase == "ring_inner":
        return "final", "i2f"
    return None


def _energy_fields(shadow: TerminalParallelEnv) -> dict[str, Any]:
    value = getattr(shadow, "energy_reachability_diagnostics", {}).get(TARGET_UID)
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): _copy_json(item)
        for key, item in value.items()
        if str(key) in {"slack_energy", "current_energy", "current_soc", "required_energy", "candidate_energy", "selected_lz_id", "pad_wait_seconds"}
    }


def _catalogue(shadow: TerminalParallelEnv) -> dict[str, Any]:
    uav = shadow.active[TARGET_UID]
    selected, audit = teacher_joint_expert_v4_1(
        shadow, uav, audit_catalogue=True, state_only=True,
    )
    release_rows = [
        row for row in audit
        if float(row.get("decoded", {}).get("release_request", 0.0)) > 0.0
    ]
    _require(len(release_rows) == 25, f"expected 25 RELEASE proposals, got {len(release_rows)}")
    context = _candidate_context(uav)
    _require(context is not None, "target is not in a ring holder phase")
    target, _interface = context
    snapshot = WorldSnapshot(shadow.now, shadow.active, shadow.airspace)
    candidates: list[dict[str, Any]] = []
    for row in release_rows:
        decoded = row.get("decoded", {})
        intent = LowLevelIntent(
            float(decoded.get("release_request", 0.0)),
            float(decoded.get("speed_residual", 0.0)),
            float(decoded.get("merge_time_residual", 0.0)),
        )
        repair = RepairCandidate("slot_state_compare", intent, 0.0)
        proposal, plan, reason = shadow._candidate_plan(uav, repair, target, snapshot, None)
        metadata = {} if plan is None else getattr(plan, "metadata", {})
        candidates.append({
            "rank": int(row.get("rank", -1)),
            "joint_index": int(row.get("joint_index", -1)),
            "proposal": None if decoded.get("proposal") is None else int(decoded["proposal"]),
            "decoded": _copy_json(decoded),
            "teacher_feasible": bool(row.get("feasible")),
            "teacher_reason": row.get("reason"),
            "candidate_reason": reason,
            "requested_merge_time": None if proposal is None else float(proposal.merge_time),
            "actual_endpoint": None if plan is None else float(plan.end_time),
            "plan_end_time": None if plan is None else float(plan.end_time),
            "metadata_requested_merge_time": metadata.get("requested_merge_time"),
            "metadata_merge_time": metadata.get("merge_time"),
            "metadata_requested_merge_late_seconds": metadata.get("requested_merge_late_seconds"),
            "energy": _energy_fields(shadow),
        })
    reasons = Counter(str(row.get("reason")) for row in release_rows)
    certified = [
        int(row.get("joint_index", -1))
        for row in release_rows if bool(row.get("feasible"))
    ]
    certified_proposals = [
        int(hierarchical_from_joint_index_v4_1(index).proposal)
        for index in certified
        if hierarchical_from_joint_index_v4_1(index).proposal is not None
    ]
    return {
        "selected_joint_index": int(selected.index),
        "release_catalogue_count": len(release_rows),
        "reason_distribution": dict(sorted(reasons.items())),
        "certified_release_joint_indices": certified,
        "certified_release_proposals": certified_proposals,
        "candidates": candidates,
    }


def _summary_outcomes(summary: Mapping[str, Any]) -> dict[str, Any]:
    value = summary.get("outcomes") or {}
    return {str(key): item for key, item in value.items()} if isinstance(value, Mapping) else {}


def _replay_outcomes(env: TerminalParallelEnv) -> dict[str, Any]:
    return {str(key): getattr(value, "value", value) for key, value in env.ledger.outcomes.items()}


def compare(*, tape: Path, output: Path) -> dict[str, Any]:
    _require(tape.is_file(), f"ActionTape missing: {tape}")
    _require(not output.exists(), f"refusing to overwrite output: {output}")
    payload = json.loads(tape.read_text(encoding="utf-8"))
    rows = list(payload.get("action_tape", []))
    _require(len(rows) == EXPECTED_STEPS, f"expected {EXPECTED_STEPS} rows, got {len(rows)}")
    _require(int(payload.get("episode_seed")) == EXPECTED_SEED, "tape seed changed")
    protocol = evaluator._protocol()
    config = evaluator._build_configs(protocol)["dense24"]
    _require(not bool(config.reconcile_late_merge_slots_enabled), "live V7 config late-slot flag is not false")
    prefix = _prefix_check(tape, config, rows)
    _require(prefix["exact"] is True, f"prefix replay failed: {prefix}")
    summary_path = tape.parent.parent / "summary.json"
    _require(summary_path.is_file(), f"summary missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    original_outcomes = _summary_outcomes(summary)
    prior_path = Path("/home/user/training_outputs/terminal/p3_adaptive_kl_run0_state_witness_20260926/state_witness.json")
    prior = None if not prior_path.is_file() else json.loads(prior_path.read_text(encoding="utf-8"))
    cross_check = {
        "path": str(prior_path), "present": prior is not None,
        "exact": None if prior is None else bool(prior.get("exact")),
        "sample_times": None if prior is None else [row.get("time") for row in prior.get("samples", [])],
    }

    env = TerminalParallelEnv(config)
    original_step = env.step
    comparisons: list[dict[str, Any]] = []
    sampled: set[int] = set()
    rng_before_replay = _rng_snapshot()
    live_digest_before = _digest(SafetyTraceRecorder._state(env))

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        step = int(env.step_count)
        if step in SAMPLE_TIMES and step not in sampled:
            sampled.add(step)
            target = env.active.get(TARGET_UID)
            base_state = _state(env, TARGET_UID)
            row: dict[str, Any] = {
                "step": step, "time": float(env.now), "state": base_state,
                "target_uid": TARGET_UID, "target_arrival_id": TARGET_ARRIVAL_ID,
            }
            if target is None or target.committed_plan is not None or getattr(target.phase, "value", target.phase) not in {"ring_outer", "ring_middle", "ring_inner"}:
                row["status"] = "unavailable_target_not_holder"
            else:
                base_digest = _digest(SafetyTraceRecorder._state(env))
                sample_rng_before = _rng_snapshot()
                shadow_off = copy.deepcopy(env)
                shadow_on = copy.deepcopy(env)
                _restore_shadow_methods(shadow_off)
                _restore_shadow_methods(shadow_on)
                shadow_off.config = replace(shadow_off.config, reconcile_late_merge_slots_enabled=False)
                shadow_on.config = replace(shadow_on.config, reconcile_late_merge_slots_enabled=True)
                off = _catalogue(shadow_off)
                on = _catalogue(shadow_on)
                sample_rng_after = _rng_snapshot()
                after_digest = _digest(SafetyTraceRecorder._state(env))
                _require(base_digest == after_digest, f"live state changed at t={env.now}")
                _require(_rng_equal(sample_rng_before, sample_rng_after), f"RNG changed at t={env.now}")
                row.update({
                    "status": "evaluated",
                    "live_state_digest_before": base_digest,
                    "live_state_digest_after": after_digest,
                    "rng_unchanged": True,
                    "late_merge_off": off,
                    "late_merge_on": on,
                    "certified_release_set_equal": off["certified_release_joint_indices"] == on["certified_release_joint_indices"],
                })
            comparisons.append(row)
        return original_step(*args, **kwargs)

    env.step = observed_step  # type: ignore[method-assign]
    try:
        replay = SafetyTraceRecorder.replay_action_tape(env, tape)
    finally:
        env.step = original_step  # type: ignore[method-assign]
    _require(replay.get("exact") is True, f"full replay failed: {replay}")
    outcomes_match = original_outcomes == _replay_outcomes(env)
    _require(outcomes_match, "replayed outcomes differ from summary")
    _require(tuple(sorted(sampled)) == SAMPLE_TIMES, f"sample times missing: {sorted(sampled)}")
    live_digest_after = _digest(SafetyTraceRecorder._state(env))
    rng_after_replay = _rng_snapshot()
    _require(_rng_equal(rng_before_replay, rng_after_replay), "replay changed Python/NumPy/Torch RNG state")
    _require(not bool(config.reconcile_late_merge_slots_enabled), "live config was changed")
    report = {
        "stage": STAGE,
        "scenario": "dense24",
        "seed": EXPECTED_SEED,
        "target_uid": TARGET_UID,
        "target_arrival_id": TARGET_ARRIVAL_ID,
        "action_tape": str(tape),
        "action_tape_sha256": _sha256(tape),
        "summary": str(summary_path),
        "prefix_check": prefix,
        "replay": {key: value for key, value in replay.items() if key != "steps"},
        "exact": True,
        "outcomes_match": True,
        "samples": comparisons,
        "cross_check_prior_off_report": cross_check,
        "live_state_digest_before_after": [live_digest_before, live_digest_after],
        "rng_unchanged": True,
        "evaluation": {
            "same_live_state_deepcopies": True,
            "only_reconcile_late_merge_slots_enabled_varied": True,
            "state_only_teacher_catalogue": True,
            "release_catalogue_expected": 25,
            "policy_sampling": False,
            "policy_update": False,
            "live_env_config_modified": False,
            "safety_filter_conditions_modified": False,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "slot_state_compare.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(tape=args.tape, output=args.output)
    print(json.dumps({
        "stage": report["stage"], "exact": report["exact"],
        "outcomes_match": report["outcomes_match"],
        "prefix_exact": report["prefix_check"]["exact"],
        "samples": len(report["samples"]),
        "output": str(args.output),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
