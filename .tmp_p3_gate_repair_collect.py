"""Collect sparse state-only gate-repair labels from the P3 failure tapes.

This temporary collector is inference/replay only.  The original arm consumes
the recorded formal ActionTape.  The bounded arm consumes the already exact
bounded-gate diagnostic ActionTape; it never reconstructs the missing forced
actor-step sidecar and never samples a policy.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from terminal_marl.core.low_level_joint_action_v4_2 import (
    ACTION_SCHEMA_V4_2,
    hierarchical_from_normalized_v4_2,
    hierarchical_from_joint_index_v4_1,
)
from terminal_marl.core.safety_trace import SafetyTraceRecorder, _digest
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import diagnose_trained_dense30_v2 as diagnostic
from terminal_marl.training import diagnose_p3_bounded_gate_v1 as bounded_diagnostic
from terminal_marl.training import student_rollout_v4_2 as student


STAGE = "P3_GATE_COVERAGE_REPAIR_COLLECTION_V2"
SCENARIO = "dense24"
LABEL_STRIDE_STEPS = 20
OBSERVATION_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
ACTION_SCHEMA = ACTION_SCHEMA_V4_2
RUNTIME_SHORTLIST_K = 3
FORMAL_ROOT_NAME = "p3_controlled_emergency_formal_v5_restart1_20260925"
BOUNDED_ROOT_NAME = "p3_bounded_gate_diagnostic_v1_20260925"
LOCAL_OUTPUT_ROOT = Path(r"D:\7_codespace\training_outputs\terminal")
SERVER_OUTPUT_ROOT = Path("/home/user/training_outputs/terminal")


class CollectionError(ValueError):
    """Raised when a replay or label-collection contract fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CollectionError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_root_candidates() -> tuple[Path, ...]:
    return (
        LOCAL_OUTPUT_ROOT,
        SERVER_OUTPUT_ROOT,
    )


def _resolve_local_or_server(path: str | Path) -> Path:
    value = Path(path)
    if value.is_file():
        return value
    raw = str(path)
    for root in _source_root_candidates():
        if raw.startswith("/home/user/training_outputs/terminal/"):
            candidate = root / raw.removeprefix("/home/user/training_outputs/terminal/")
            if candidate.is_file():
                return candidate
    return value


def _source_binding(run_index: int) -> dict[str, Any]:
    for root in _source_root_candidates():
        candidate = root / FORMAL_ROOT_NAME
        if (candidate / f"run{run_index}" / "manifest.json").is_file():
            return bounded_diagnostic.source_binding(run_index, source_root=candidate)
    return bounded_diagnostic.source_binding(run_index)


def _source_tape(binding: Mapping[str, Any], arm: str) -> Path:
    if arm == "bounded":
        for root in _source_root_candidates():
            candidate = root / BOUNDED_ROOT_NAME / f"run{int(binding['run_index'])}" / "rollout" / "action_tapes"
            tapes = sorted(candidate.glob("*.json"))
            if len(tapes) == 1:
                return tapes[0]
        raise CollectionError(f"bounded diagnostic tape missing for run {binding['run_index']}")
    manifest = Path(str(binding["manifest"]))
    episode = int(binding["failure_episode_number"])
    seed = int(binding["failure_seed"])
    episode_dir = manifest.parent / "episodes" / f"episode_{episode:04d}_{SCENARIO}_{seed}"
    tapes = sorted((episode_dir / "action_tapes").glob("*.json"))
    _require(len(tapes) == 1, f"expected one original tape, found {len(tapes)} in {episode_dir}")
    return tapes[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(isinstance(value, dict), f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _build_config(arm: str):
    if arm == "bounded":
        config = bounded_diagnostic.build_config()
    else:
        config = diagnostic.build_scenario_config(SCENARIO, reconcile_late_merge_slots=False)
    expected_mode = "canonical_wait" if arm == "bounded" else "legacy_requested"
    _require(config.v4_2_rejected_release_hold_mode == expected_mode,
             f"{arm} label replay has unexpected rejected-release mode")
    _require(config.action_schema_version == ACTION_SCHEMA,
             "label replay requires V4.2 hierarchical actions")
    _require(config.observation_schema_version == OBSERVATION_SCHEMA,
             "label replay requires 88D V2.1 observations")
    _require(config.teacher_enabled is False and config.scheduler_mode == "execution",
             "label replay requires Teacher-off execution config")
    return config


def _actor_mask(env: TerminalParallelEnv, uids: list[str]) -> np.ndarray:
    return np.asarray([
        env.active[int(uid)].committed_plan is None
        and env.active[int(uid)].phase.value in {"ring_outer", "ring_middle", "ring_inner"}
        and int(uid) not in env.future_release_reservations
        for uid in uids
    ], dtype=bool)


def _requested_gate_proposal(actions: Mapping[str, Any], uid: str) -> tuple[int, int | None]:
    value = actions.get(uid, actions.get(str(uid)))
    _require(value is not None, f"missing requested action for uid {uid}")
    decoded = hierarchical_from_normalized_v4_2(np.asarray(value, dtype=np.float32))
    return int(decoded.gate), None if decoded.proposal is None else int(decoded.proposal)


def _tape_payload(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"tape payload must be an object: {path}")
    rows = payload.get("action_tape", payload.get("records", []))
    _require(isinstance(rows, list) and rows, f"tape has no action rows: {path}")
    return payload, [dict(row) for row in rows]


def _actions_and_shortlists(row: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, ...]]]:
    actions = {
        str(uid): np.asarray(action, dtype=np.float32)
        for uid, action in (row.get("a_requested") or {}).items()
    }
    shortlists = {
        str(uid): tuple(np.asarray(action, dtype=np.float32) for action in values)
        for uid, values in (row.get("runtime_proposal_shortlists") or {}).items()
    }
    return actions, shortlists


def _label_uids(*, step: int, uids: list[str], actor_mask: np.ndarray,
                previous_actor: dict[str, bool]) -> set[str]:
    periodic = int(step) % LABEL_STRIDE_STEPS == 0
    selected: set[str] = set()
    for index, uid in enumerate(uids):
        uid_key = str(uid)
        actor = bool(actor_mask[index])
        if periodic or (actor and not previous_actor.get(uid_key, False)):
            selected.add(uid_key)
        previous_actor[uid_key] = actor
    current_uids = {str(uid) for uid in uids}
    for uid_key in tuple(previous_actor):
        if uid_key not in current_uids:
            previous_actor[uid_key] = False
    return selected


def _teacher_row(env: TerminalParallelEnv, *, uid: str, observation: np.ndarray,
                 step: int, gate: int, proposal: int | None, arm: str,
                 run_index: int, seed: int, reason: str, tape_path: Path) -> dict[str, Any]:
    uav = env.active[int(uid)]
    before_hash = _digest(SafetyTraceRecorder._state(env))
    shadow = copy.deepcopy(env)
    shadow_uav = shadow.active[int(uid)]
    target, _ = student._teacher_label(
        shadow, shadow_uav, audit_catalogue=False, state_only=True,
    )
    hierarchical_target = hierarchical_from_joint_index_v4_1(target.index)
    after_hash = _digest(SafetyTraceRecorder._state(env))
    _require(before_hash == after_hash, "state-only label query changed original environment")
    return {
        "observation_schema": OBSERVATION_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "observation_v2_1": np.asarray(observation, dtype=np.float32).tolist(),
        "teacher_gate": int(hierarchical_target.gate),
        "teacher_proposal": None if hierarchical_target.proposal is None else int(hierarchical_target.proposal),
        "teacher_joint_index": int(target.index),
        "teacher_joint_index_v4_1": int(target.index),
        "source": "p3_gate_coverage_repair_v2_state_only_teacher",
        "source_arm": arm,
        "run_index": int(run_index),
        "seed": int(seed),
        "step": int(step),
        "sim_time_before": float(env.now),
        "phase": str(uav.phase.value),
        "observed_soc": float(uav.soc),
        "base_priority": int(uav.base_priority),
        "arrival_id": int(uav.arrival_id),
        "active_uavs_before": int(len(env.active)),
        "completed_uavs_before": int(len(env.completed_uavs)),
        "episode_id": f"p3_gate_repair:{arm}:run{run_index}:seed{seed}",
        "lifecycle_event_id": f"p3_gate_repair:{arm}:run{run_index}:seed{seed}:uav{uid}",
        "decision_event_id": f"p3_gate_repair:{arm}:run{run_index}:seed{seed}:step{step}:uid{uid}",
        "observation_hash": _digest(np.asarray(observation, dtype=np.float32)),
        "state_hash": before_hash,
        "uav_id": str(uid),
        "actor_mask": True,
        "requested_gate": int(gate),
        "requested_proposal": proposal,
        "label_sampling_reason": reason,
        "teacher_state_only": True,
        "source_tape": str(tape_path),
    }


def _prefix_tape(path: Path, output: Path, max_steps: int | None) -> tuple[Path, list[dict[str, Any]]]:
    payload, action_rows = _tape_payload(path)
    if max_steps is None:
        return path, action_rows
    prefix_rows = action_rows[:int(max_steps)]
    _require(prefix_rows, "preflight tape prefix is empty")
    prefix_payload = dict(payload)
    prefix_payload["action_tape"] = prefix_rows
    prefix_payload.pop("records", None)
    prefix_path = output / ".preflight_action_tape.json"
    prefix_path.write_text(json.dumps(_jsonable(prefix_payload), sort_keys=True), encoding="utf-8")
    return prefix_path, prefix_rows


def _replay_with_labels(env: TerminalParallelEnv, *, action_rows: list[dict[str, Any]],
                        replay_tape: Path, arm: str, run_index: int, seed: int,
                        source_tape: Path, output: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Collect labels from the same exact replay used for validation.

    ``SafetyTraceRecorder.replay_action_tape`` owns reset and transition
    validation.  The wrapped ``step`` observes the pre-step state and invokes
    only state-only Teacher queries before delegating to the real step.
    """
    previous_actor: dict[str, bool] = {}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    _write_json(output / "progress.json", {
        "stage": STAGE, "status": "RUNNING", "arm": arm, "run_index": int(run_index),
        "seed": int(seed), "processed_steps": 0, "total_steps": len(action_rows),
        "label_rows": 0, "elapsed_seconds": 0.0,
    })
    step_index = 0
    original_step = env.step

    def observed_step(actions, shortlists):
        nonlocal step_index
        _require(step_index < len(action_rows), "replay supplied more steps than the source tape")
        row = action_rows[step_index]
        step = int(row.get("step", step_index))
        observations = env._observations()
        uids = sorted(str(uid) for uid in observations)
        actor_mask = _actor_mask(env, uids)
        decoded_actions = {
            str(uid): np.asarray(value, dtype=np.float32)
            for uid, value in actions.items()
        }
        gates = np.asarray([
            _requested_gate_proposal(decoded_actions, uid)[0] for uid in uids
        ], dtype=int)
        proposals = {
            uid: _requested_gate_proposal(decoded_actions, uid)[1] for uid in uids
        }
        selected = _label_uids(
            step=step, uids=uids, actor_mask=actor_mask,
            previous_actor=previous_actor,
        )
        for uid in sorted(selected, key=int):
            if not bool(actor_mask[uids.index(uid)]):
                continue
            rows.append(_teacher_row(
                env, uid=uid, observation=observations[uid], step=step,
                gate=int(gates[uids.index(uid)]), proposal=proposals[uid],
                arm=arm, run_index=run_index, seed=seed,
                reason=("stride" if step % LABEL_STRIDE_STEPS == 0 else "first_actor_or_gate_change"),
                tape_path=source_tape,
            ))
        result = original_step(actions, shortlists)
        step_index += 1
        if step_index % LABEL_STRIDE_STEPS == 0 or step_index == len(action_rows):
            _write_json(output / "progress.json", {
                "stage": STAGE, "status": "RUNNING", "arm": arm,
                "run_index": int(run_index), "seed": int(seed),
                "processed_steps": step_index, "total_steps": len(action_rows),
                "label_rows": len(rows), "elapsed_seconds": time.perf_counter() - started,
            })
        return result

    env.step = observed_step
    try:
        replay = SafetyTraceRecorder.replay_action_tape(env, replay_tape)
    finally:
        env.step = original_step
    return replay, rows, {
        "processed_steps": step_index,
        "label_rows": len(rows), "elapsed_seconds": time.perf_counter() - started,
    }


def _collect_tape(*, run_index: int, arm: str, binding: Mapping[str, Any], output: Path,
                  max_steps: int | None) -> dict[str, Any]:
    config = _build_config(arm)
    tape_path = _source_tape(binding, arm)
    replay_tape, action_rows = _prefix_tape(tape_path, output, max_steps)
    payload, _ = _tape_payload(tape_path)
    _require(int(payload.get("episode_seed")) == int(binding["failure_seed"]),
             f"{arm} tape seed does not match source failure seed")
    env = TerminalParallelEnv(config)
    replay, rows, progress = _replay_with_labels(
        env, action_rows=action_rows, replay_tape=replay_tape,
        arm=arm, run_index=run_index, seed=int(binding["failure_seed"]),
        source_tape=tape_path, output=output,
    )
    if not bool(replay.get("exact")):
        _write_json(output / "progress.json", {
            "stage": STAGE, "status": "FAILED", "arm": arm,
            "run_index": int(run_index), "seed": int(binding["failure_seed"]),
            "processed_steps": int(progress["processed_steps"]),
            "label_rows": int(progress["label_rows"]),
            "replay_exact": False,
            "reset_state_match": bool(replay.get("reset_state_match")),
            "config_match": bool(replay.get("config_match")),
            "config_compatibility": replay.get("config_compatibility"),
            "steps_replayed": int(replay.get("steps_replayed", 0)),
            "steps_expected": int(replay.get("steps_expected", 0)),
            "elapsed_seconds": float(progress["elapsed_seconds"]),
        })
    _require(bool(replay.get("exact")), f"{arm} tape replay was not exact: {replay}")
    expected_compatibility = "legacy_requested_default_field" if arm == "original" else None
    _require(replay.get("config_compatibility") == expected_compatibility,
             f"{arm} config compatibility changed: {replay.get('config_compatibility')!r}")
    return {
        "replay": replay, "tape": str(tape_path), "tape_sha256": _sha256(tape_path),
        "tape_rows": len(action_rows), "rows": rows, "progress": progress,
        "source_kind": "formal_failed_action_tape" if arm == "original" else "bounded_gate_diagnostic_action_tape",
    }


def run(*, run_index: int, arm: str, output: str | Path,
        max_steps: int | None = None) -> dict[str, Any]:
    _require(arm in {"original", "bounded"}, "arm must be original or bounded")
    _require(int(run_index) in (0, 1, 2), "run_index must be 0, 1, or 2")
    output_path = Path(output)
    _require(not output_path.exists(), f"refusing to overwrite existing output: {output_path}")
    if max_steps is not None:
        _require(int(max_steps) >= 1, "max_steps must be positive")
    binding = _source_binding(int(run_index))
    output_path.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    collected = _collect_tape(
        run_index=int(run_index), arm=arm, binding=binding,
        output=output_path, max_steps=max_steps,
    )
    label_rows = collected.pop("rows")
    labels_path = output_path / "labels.jsonl"
    with labels_path.open("w", encoding="utf-8") as handle:
        for row in label_rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    counts = Counter(int(row["teacher_gate"]) for row in label_rows)
    summary = {
        "stage": STAGE,
        "arm": arm,
        "run_index": int(run_index),
        "scenario": SCENARIO,
        "seed": int(binding["failure_seed"]),
        "source_binding": binding,
        "source_checkpoint_sha256": binding["source_checkpoint_sha256"],
        "config_compatibility": collected["replay"].get("config_compatibility"),
        "label_stride_steps": LABEL_STRIDE_STEPS,
        "first_actor_extra": True,
        "teacher": "state_only_deepcopy",
        "policy_sampling": False,
        "training_called": False,
        "ppo_called": False,
        "heldout_consumed": False,
        "max_steps": max_steps,
        "label_rows": len(label_rows),
        "class_counts": {"WAIT": int(counts.get(0, 0)), "RELEASE": int(counts.get(1, 0))},
        "labels_path": str(labels_path),
        "collection_wall_elapsed_seconds": time.perf_counter() - started,
        **collected,
    }
    _write_json(output_path / "summary.json", summary)
    _write_json(output_path / "progress.json", {
        "stage": STAGE, "status": "COMPLETED", "arm": arm,
        "run_index": int(run_index), "seed": int(binding["failure_seed"]),
        "processed_steps": int(summary["progress"]["processed_steps"]),
        "label_rows": len(label_rows),
        "elapsed_seconds": summary["collection_wall_elapsed_seconds"],
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--arm", choices=("original", "bounded"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="bounded preflight prefix; omit for full source episode")
    args = parser.parse_args()
    result = run(
        run_index=args.run_index, arm=args.arm, output=args.output,
        max_steps=args.max_steps,
    )
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
