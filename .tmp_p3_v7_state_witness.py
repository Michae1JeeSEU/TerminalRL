"""Bounded state-only witness collection on one frozen V7 ActionTape.

The original tape is replayed exactly.  Teacher queries run only on deep
copies of the replay state, and policy queries are forward-only.  No teacher
label or policy action is passed to the live environment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from terminal_marl.core.low_level_joint_action_v4_2 import (
    hierarchical_from_joint_index_v4_1,
)
from terminal_marl.core.safety_trace import SafetyTraceRecorder, _digest
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import evaluate_p3_adaptive_kl_v7 as evaluator
from terminal_marl.training.v4_1_joint_teacher import (
    teacher_joint_expert_v4_1,
    teacher_joint_label_v4_1,
)


STAGE = "P3_V7_RUN0_STATE_WITNESS_V1"
TARGET_ARRIVAL_ID = 15
TARGET_UID = 72
EXPECTED_SEED = 873106
EXPECTED_STEPS = 897
SAMPLE_TIMES = (136, 152, 200, 250, 297, 500)
FILTER_CAPTURE_TIMES = frozenset((152, 297, 302, 497, 500, 502))


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


def _rng_snapshot() -> tuple[Any, tuple[Any, ...], torch.Tensor]:
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    return py_state, np_state, torch_state


def _rng_equal(before: tuple[Any, tuple[Any, ...], torch.Tensor],
               after: tuple[Any, tuple[Any, ...], torch.Tensor]) -> bool:
    py_before, np_before, torch_before = before
    py_after, np_after, torch_after = after
    np_equal = (
        np_before[0] == np_after[0]
        and np.array_equal(np_before[1], np_after[1])
        and np_before[2:] == np_after[2:]
    )
    return py_before == py_after and np_equal and bool(torch.equal(torch_before, torch_after))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_uav(env: TerminalParallelEnv, uid: int) -> Any | None:
    if int(uid) in env.active:
        return env.active[int(uid)]
    if int(uid) in env.service_uavs:
        return env.service_uavs[int(uid)]
    for uav in env.completed_uavs:
        if int(uav.uid) == int(uid):
            return uav
    return None


def _uav_state(env: TerminalParallelEnv, uid: int) -> dict[str, Any]:
    uav = _find_uav(env, uid)
    if uav is None:
        return {"uid": int(uid), "present": False}
    plan = uav.committed_plan
    return {
        "uid": int(uav.uid),
        "arrival_id": int(uav.arrival_id),
        "present": True,
        "phase": getattr(uav.phase, "value", uav.phase),
        "ring_role": None if uav.ring_role is None else getattr(uav.ring_role, "value", uav.ring_role),
        "soc": float(uav.soc),
        "energy_remaining": float(uav.energy_remaining),
        "committed": plan is not None,
        "assigned_lz": None if uav.assigned_lz is None else int(uav.assigned_lz),
        "emergency_landing": bool(uav.emergency_landing),
        "safety_retry_after": float(uav.safety_retry_after),
        "emergency_retry_after": float(uav.emergency_retry_after),
        "terminal_outcome": None if uav.terminal_outcome is None else getattr(uav.terminal_outcome, "value", uav.terminal_outcome),
        "terminal_reason": uav.terminal_reason,
    }


def _restore_shadow_methods(shadow: TerminalParallelEnv) -> None:
    cls = type(shadow)
    for name in (
        "step", "_release", "_try_emergency_landing",
        "_candidate_energy_guard_reason",
    ):
        setattr(shadow, name, getattr(cls, name).__get__(shadow, cls))
    safety_cls = type(shadow.safety)
    shadow.safety.filter_plan = safety_cls.filter_plan.__get__(shadow.safety, safety_cls)


def _intent_payload(intent: Any) -> dict[str, float]:
    return {
        "release_request": float(intent.release_request),
        "speed_residual": float(intent.speed_residual),
        "merge_time_residual": float(intent.merge_time_residual),
    }


def _state_only_teacher(env: TerminalParallelEnv) -> dict[str, Any]:
    uav = env.active[TARGET_UID]
    before_hash = _digest(SafetyTraceRecorder._state(env))
    rng_before = _rng_snapshot()
    shadow = copy.deepcopy(env)
    _restore_shadow_methods(shadow)
    shadow_uav = shadow.active[TARGET_UID]
    label = teacher_joint_label_v4_1(shadow, shadow_uav, state_only=True)
    _, audit = teacher_joint_expert_v4_1(
        shadow, shadow_uav, audit_catalogue=True, state_only=True,
    )
    after_hash = _digest(SafetyTraceRecorder._state(env))
    rng_after = _rng_snapshot()
    _require(before_hash == after_hash, "state-only Teacher query changed live replay state")
    _require(_rng_equal(rng_before, rng_after), "state-only Teacher query changed RNG state")
    selected = hierarchical_from_joint_index_v4_1(label.index)
    certified = [
        {
            "rank": int(row.get("rank", -1)),
            "joint_index": int(row.get("joint_index", -1)),
            "decoded": _copy_json(row.get("decoded", {})),
            "reason": row.get("reason"),
        }
        for row in audit
        if bool(row.get("feasible"))
        and float(row.get("decoded", {}).get("release_request", 0.0)) > 0.0
    ]
    reasons = Counter(str(row.get("reason")) for row in audit)
    return {
        "selected_label": {
            "gate": int(selected.gate),
            "proposal": None if selected.proposal is None else int(selected.proposal),
            "joint_index_v4_1": int(label.index),
        },
        "audit_count": int(len(audit)),
        "audit_reason_counts": dict(sorted(reasons.items())),
        "certified_release_count": int(len(certified)),
        "certified_release_candidates": certified,
        "live_state_digest_unchanged": True,
        "python_numpy_torch_rng_unchanged": True,
    }


def _policy_forward(policy: Any, observation: np.ndarray) -> dict[str, Any]:
    values = torch.as_tensor(observation, dtype=torch.float32, device=policy.device).reshape(1, -1)
    _require(int(values.shape[-1]) == 88, "witness observation is not 88D")
    with torch.no_grad():
        gate_dist, proposal_dist = policy.model.distributions(values)
        gate_probs = torch.softmax(gate_dist.logits, dim=-1)[0]
        proposal_probs = torch.softmax(proposal_dist.logits, dim=-1)[0]
    return {
        "observation_dimension": int(values.shape[-1]),
        "p_release": float(gate_probs[1].detach().cpu().item()),
        "gate_distribution": [float(x) for x in gate_probs.detach().cpu().tolist()],
        "proposal_distribution": [float(x) for x in proposal_probs.detach().cpu().tolist()],
    }


def _actor_eligible(env: TerminalParallelEnv, uav: Any | None) -> bool:
    return bool(
        uav is not None
        and uav.committed_plan is None
        and getattr(uav.phase, "value", uav.phase) in {"ring_outer", "ring_middle", "ring_inner"}
        and int(uav.uid) not in env.future_release_reservations
        and not uav.emergency_landing
    )


def _prefix_check(tape_path: Path, config: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = json.loads(tape_path.read_text(encoding="utf-8"))
    payload["action_tape"] = rows[:2]
    with tempfile.TemporaryDirectory(prefix="p3_v7_state_witness_") as directory:
        prefix_path = Path(directory) / "prefix.json"
        prefix_path.write_text(
            json.dumps(_jsonable(payload), sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), prefix_path)
    return {
        "rows": 2,
        "json_roundtrip": True,
        "exact": bool(replay.get("exact")),
        "steps_expected": replay.get("steps_expected"),
        "steps_replayed": replay.get("steps_replayed"),
    }


def _summary_outcomes(summary: Mapping[str, Any]) -> dict[str, Any]:
    outcomes = summary.get("outcomes") or {}
    return {str(key): value for key, value in outcomes.items()} if isinstance(outcomes, Mapping) else {}


def _replay_outcomes(env: TerminalParallelEnv) -> dict[str, Any]:
    return {
        str(arrival_id): getattr(outcome, "value", outcome)
        for arrival_id, outcome in env.ledger.outcomes.items()
    }


def _load_policy_pair(checkpoint: Path, config: Any, protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_sha = evaluator._sha256(evaluator.PROTOCOL)
    optimizer = protocol.get("optimizer") or {}
    learning_rate = float(optimizer.get("learning_rate", 3e-4))
    source_path = Path(str((protocol.get("lineage") or {}).get("source_checkpoint_path")))
    _require(source_path.is_file(), f"source checkpoint missing: {source_path}")
    source_policy, source_payload, source_sha = evaluator._load_eval_policy(
        source_path, config=config, expected_checkpoint_sha256=evaluator.SOURCE_CHECKPOINT_SHA256,
        protocol_sha256=protocol_sha, final_checkpoint=False, learning_rate=learning_rate,
    )
    checkpoint_policy, checkpoint_payload, checkpoint_sha = evaluator._load_eval_policy(
        checkpoint, config=config, expected_checkpoint_sha256=_sha256(checkpoint),
        protocol_sha256=protocol_sha, final_checkpoint=True, learning_rate=learning_rate,
    )
    return (
        {
            "label": "source_03c0", "policy": source_policy, "payload": source_payload,
            "path": str(source_path), "sha256": source_sha,
            "model_digest_before": evaluator._model_digest(source_policy),
        },
        {
            "label": "run0_checkpoint_episode_0080", "policy": checkpoint_policy,
            "payload": checkpoint_payload, "path": str(checkpoint), "sha256": checkpoint_sha,
            "model_digest_before": evaluator._model_digest(checkpoint_policy),
        },
    )


def replay_witness(*, tape: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    _require(tape.is_file(), f"ActionTape missing: {tape}")
    _require(checkpoint.is_file(), f"checkpoint missing: {checkpoint}")
    _require(not output.exists(), f"refusing to overwrite output: {output}")
    payload = json.loads(tape.read_text(encoding="utf-8"))
    rows = list(payload.get("action_tape", []))
    _require(len(rows) == EXPECTED_STEPS, f"expected {EXPECTED_STEPS} tape rows, got {len(rows)}")
    _require(int(payload.get("episode_seed")) == EXPECTED_SEED, "tape seed changed")
    protocol = evaluator._protocol()
    configs = evaluator._build_configs(protocol)
    config = configs["dense24"]
    prefix = _prefix_check(tape, config, rows)
    _require(prefix["exact"] is True, f"2-step prefix replay failed: {prefix}")
    summary_path = tape.parent.parent / "summary.json"
    _require(summary_path.is_file(), f"summary missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    original_outcomes = _summary_outcomes(summary)
    source, candidate = _load_policy_pair(checkpoint, config, protocol)
    checkpoint_sha_before = _sha256(checkpoint)
    source_sha_before = _sha256(Path(source["path"]))

    samples: list[dict[str, Any]] = []
    sampled_steps: set[int] = set()
    filter_events: list[dict[str, Any]] = []
    env = TerminalParallelEnv(config)
    original_step = env.step
    live_filter_plan: Any | None = None

    def observed_filter_plan(plan: Any, committed: Any, snapshot: Any) -> Any:
        _require(live_filter_plan is not None, "live SafetyFilter was not bound")
        result = live_filter_plan(plan, committed, snapshot)
        if (
            int(getattr(plan, "uav_id", -1)) == TARGET_UID
            and int(round(float(env.now))) in FILTER_CAPTURE_TIMES
        ):
            metadata = getattr(plan, "metadata", {})
            filter_events.append({
                "time": float(env.now), "step": int(env.step_count),
                "uav_id": int(plan.uav_id), "mode": str(getattr(plan, "mode", "")),
                "target_phase": metadata.get("target_phase"),
                "approved": bool(getattr(result, "approved", False)),
                "reason": getattr(result, "reason", None),
                "plan_id": getattr(plan, "plan_id", None),
            })
        return result

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal live_filter_plan
        if live_filter_plan is None:
            live_filter_plan = env.safety.filter_plan
            env.safety.filter_plan = observed_filter_plan
        step = int(env.step_count)
        time = float(env.now)
        if step in SAMPLE_TIMES and step not in sampled_steps:
            sampled_steps.add(step)
            uav = env.active.get(TARGET_UID)
            state = _uav_state(env, TARGET_UID)
            row: dict[str, Any] = {
                "step": step, "time": time, "arrival_id": TARGET_ARRIVAL_ID,
                "target_uid": TARGET_UID, "state": state,
                "actor_eligible": _actor_eligible(env, uav),
            }
            if not row["actor_eligible"]:
                row["status"] = "unavailable_target_not_holder"
            else:
                observation = env._observations().get(str(TARGET_UID))
                _require(observation is not None, f"target observation missing at t={time}")
                observation = np.asarray(observation, dtype=np.float32)
                row["observation_dimension"] = int(observation.size)
                row["observation_sha256"] = hashlib.sha256(observation.tobytes()).hexdigest()
                row["state_only_teacher"] = _state_only_teacher(env)
                row["source_03c0"] = _policy_forward(source["policy"], observation)
                row["run0_checkpoint_episode_0080"] = _policy_forward(candidate["policy"], observation)
                row["status"] = "evaluated"
            samples.append(row)
        return original_step(*args, **kwargs)

    env.step = observed_step  # type: ignore[method-assign]
    try:
        replay = SafetyTraceRecorder.replay_action_tape(env, tape)
    finally:
        env.step = original_step  # type: ignore[method-assign]
        if live_filter_plan is not None:
            env.safety.filter_plan = live_filter_plan  # type: ignore[method-assign]
    _require(replay.get("exact") is True, f"full replay was not exact: {replay}")
    replay_outcomes = _replay_outcomes(env)
    outcomes_match = original_outcomes == replay_outcomes
    _require(outcomes_match, "replayed outcomes differ from summary outcomes")
    _require(tuple(sorted(sampled_steps)) == SAMPLE_TIMES, f"sample times missing: {sorted(sampled_steps)}")

    source_sha_after = _sha256(Path(source["path"]))
    checkpoint_sha_after = _sha256(checkpoint)
    source_digest_after = evaluator._model_digest(source["policy"])
    candidate_digest_after = evaluator._model_digest(candidate["policy"])
    _require(source_sha_before == source_sha_after, "source checkpoint changed")
    _require(checkpoint_sha_before == checkpoint_sha_after, "run0 checkpoint changed")
    _require(source["model_digest_before"] == source_digest_after, "source model changed during witness")
    _require(candidate["model_digest_before"] == candidate_digest_after, "candidate model changed during witness")

    report = {
        "stage": STAGE,
        "scenario": "dense24",
        "seed": EXPECTED_SEED,
        "target_arrival_id": TARGET_ARRIVAL_ID,
        "target_uid": TARGET_UID,
        "action_tape": str(tape),
        "action_tape_sha256": _sha256(tape),
        "summary": str(summary_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "source_checkpoint": source["path"],
        "source_checkpoint_sha256_before": source_sha_before,
        "source_checkpoint_sha256_after": source_sha_after,
        "policy_model_digest_before_after": {
            "source_03c0": [source["model_digest_before"], source_digest_after],
            "run0_checkpoint_episode_0080": [candidate["model_digest_before"], candidate_digest_after],
        },
        "replay": {key: value for key, value in replay.items() if key != "steps"},
        "exact": True,
        "outcomes_match": True,
        "prefix_check": prefix,
        "samples": samples,
        "filter_plan_events": filter_events,
        "evaluation": {
            "policy_sampling": False,
            "policy_update": False,
            "online_teacher_substitution": False,
            "state_only_teacher_on_deepcopy": True,
            "filter_plan_return_values_preserved": True,
            "guards_modified": False,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "state_witness.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def replay_filter_witness(*, tape: Path, output: Path) -> dict[str, Any]:
    """Replay one tape while passively recording target filter return values."""
    _require(tape.is_file(), f"ActionTape missing: {tape}")
    _require(not output.exists(), f"refusing to overwrite output: {output}")
    payload = json.loads(tape.read_text(encoding="utf-8"))
    rows = list(payload.get("action_tape", []))
    _require(len(rows) == EXPECTED_STEPS, f"expected {EXPECTED_STEPS} tape rows, got {len(rows)}")
    _require(int(payload.get("episode_seed")) == EXPECTED_SEED, "tape seed changed")
    protocol = evaluator._protocol()
    config = evaluator._build_configs(protocol)["dense24"]
    prefix = _prefix_check(tape, config, rows)
    _require(prefix["exact"] is True, f"2-step prefix replay failed: {prefix}")
    summary_path = tape.parent.parent / "summary.json"
    _require(summary_path.is_file(), f"summary missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    original_outcomes = _summary_outcomes(summary)
    filter_events: list[dict[str, Any]] = []
    env = TerminalParallelEnv(config)
    original_step = env.step
    live_filter_plan: Any | None = None

    def observed_filter_plan(plan: Any, committed: Any, snapshot: Any) -> Any:
        _require(live_filter_plan is not None, "live SafetyFilter was not bound")
        result = live_filter_plan(plan, committed, snapshot)
        if (
            int(getattr(plan, "uav_id", -1)) == TARGET_UID
            and int(round(float(env.now))) in FILTER_CAPTURE_TIMES
        ):
            metadata = getattr(plan, "metadata", {})
            filter_events.append({
                "time": float(env.now),
                "step": int(env.step_count),
                "uav_id": int(plan.uav_id),
                "mode": str(getattr(plan, "mode", "")),
                "target_phase": metadata.get("target_phase"),
                "approved": bool(getattr(result, "approved", False)),
                "reason": getattr(result, "reason", None),
                "plan_id": getattr(plan, "plan_id", None),
                "state": _uav_state(env, TARGET_UID),
            })
        return result

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal live_filter_plan
        if live_filter_plan is None:
            live_filter_plan = env.safety.filter_plan
            env.safety.filter_plan = observed_filter_plan
        return original_step(*args, **kwargs)

    env.step = observed_step  # type: ignore[method-assign]
    try:
        replay = SafetyTraceRecorder.replay_action_tape(env, tape)
    finally:
        env.step = original_step  # type: ignore[method-assign]
        if live_filter_plan is not None:
            env.safety.filter_plan = live_filter_plan  # type: ignore[method-assign]
    _require(replay.get("exact") is True, f"full replay was not exact: {replay}")
    replay_outcomes = _replay_outcomes(env)
    outcomes_match = original_outcomes == replay_outcomes
    _require(outcomes_match, "replayed outcomes differ from summary outcomes")
    emergency_events = [row for row in filter_events if row["mode"] == "emergency_landing"]
    _require(emergency_events, "no target emergency filter records captured")
    report = {
        "stage": "P3_V7_RUN0_FILTER_WITNESS_V1",
        "scenario": "dense24",
        "seed": EXPECTED_SEED,
        "target_arrival_id": TARGET_ARRIVAL_ID,
        "target_uid": TARGET_UID,
        "action_tape": str(tape),
        "action_tape_sha256": _sha256(tape),
        "summary": str(summary_path),
        "replay": {key: value for key, value in replay.items() if key != "steps"},
        "prefix_check": prefix,
        "exact": True,
        "outcomes_match": True,
        "filter_capture_times": sorted(FILTER_CAPTURE_TIMES),
        "filter_plan_events": filter_events,
        "target_emergency_filter_events": emergency_events,
        "evaluation": {
            "passive_filter_wrapper": True,
            "filter_plan_return_values_preserved": True,
            "policy_sampling": False,
            "policy_update": False,
            "teacher_substitution": False,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "filter_witness.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-only", action="store_true")
    parser.add_argument("--filter-only", action="store_true")
    args = parser.parse_args()
    _require(not (args.prefix_only and args.filter_only), "prefix-only and filter-only are exclusive")
    if args.filter_only:
        report = replay_filter_witness(tape=args.tape, output=args.output)
        print(json.dumps({
            "stage": report["stage"], "exact": report["exact"],
            "outcomes_match": report["outcomes_match"],
            "prefix_exact": report["prefix_check"]["exact"],
            "filter_plan_events": len(report["filter_plan_events"]),
            "target_emergency_filter_events": len(report["target_emergency_filter_events"]),
            "output": str(args.output),
        }, sort_keys=True), flush=True)
        return 0
    if args.prefix_only:
        payload = json.loads(args.tape.read_text(encoding="utf-8"))
        rows = list(payload.get("action_tape", []))
        protocol = evaluator._protocol()
        config = evaluator._build_configs(protocol)["dense24"]
        result = _prefix_check(args.tape, config, rows)
        _require(result["exact"] is True, f"prefix replay failed: {result}")
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    _require(args.checkpoint is not None, "--checkpoint is required unless --filter-only is used")
    report = replay_witness(tape=args.tape, checkpoint=args.checkpoint, output=args.output)
    print(json.dumps({
        "stage": report["stage"], "exact": report["exact"],
        "outcomes_match": report["outcomes_match"],
        "prefix_exact": report["prefix_check"]["exact"],
        "samples": len(report["samples"]),
        "evaluated_samples": sum(row.get("status") == "evaluated" for row in report["samples"]),
        "filter_plan_events": len(report["filter_plan_events"]),
        "output": str(args.output),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
