"""Replay the four completed after-legacy tapes for matched actor comparison.

The tape is the sole source of executed actions.  This diagnostic evaluates the
before and after policies on the same replayed worlds without sampling,
stepping extra transitions, computing returns, or changing the environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from terminal_marl.core.safety_trace import SafetyTraceRecorder, _digest
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import run_p3_trained_proposal_support_20261001 as trained_support
from terminal_marl.training import student_rollout_v4_2 as student


STAGE = "P3_MATCHED_GATE_SHIFT_20261001_V1"
SCENARIO = "dense24"
OBSERVATION_DIM = 88
SHORTLIST_K = 3
LP_TOLERANCE = 1e-5
DIST_TOLERANCE = 1e-7
FULL_SHA256 = "3f65446f49474452d31c3e1498c8019f04f4250cad3e8594c75107a123525919"
PILOT_SHA256 = "b9133fdbe1c94b7e2a2d5225c052c103339bb55ca2d59d6df3e885a871a55cda"
PREFLIGHT_SHA256 = "0911e9a9edafc17b2108ea9c56f770f7c54cbd1dfbc6b3c72b17974fc2a2fe38"
BEFORE_SHA256 = "3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab"
AFTER_SHA256 = "d44990dc7d129573d9145f169df1cbc4f91fa32bc5127b1c58d937260c02a24f"
WARMUP_SHA256 = "a82ed7a287035fb1fc868fc955e056c0481de75bcb99880d5c79f35588dde710"
INPUT_ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_followup_20260930/trained_proposal_support_20261001"
)
OUTPUT_ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_followup_20260930/matched_gate_shift_20261001"
)
PROTOCOL = Path(__file__).resolve().parents[2] / "docs/evidence/v4_2/p3_matched_gate_shift_protocol_20261001.md"


class MatchedGateShiftError(ValueError):
    """Raised for a source, replay, state, or serialization mismatch."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatchedGateShiftError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, value: Mapping[str, Any], *, refuse_existing: bool = False) -> None:
    if refuse_existing and path.exists():
        raise MatchedGateShiftError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON input missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatchedGateShiftError(f"invalid JSON input: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = _sha256(path)
    _require(actual == expected, f"{label} SHA mismatch: {actual}")
    return actual


def _synced(path: Path) -> Path:
    value = trained_support._sync_path(path)
    if value.is_file():
        return value
    text = str(path).replace("\\", "/")
    prefix = "/home/user/training_outputs/terminal/"
    if text.startswith(prefix):
        mirror = Path(__file__).resolve().parents[3] / "training_outputs" / "terminal" / text[len(prefix):]
        if mirror.is_file():
            return mirror
    return value


def _arm_path(cell_id: int) -> Path:
    return _synced(INPUT_ROOT / "arms" / f"cell_{cell_id:02d}_after_legacy.json")


def _expected_cells() -> list[dict[str, Any]]:
    return [dict(item) for item in trained_support.CELL_SPECS]


def _policy_binding(config: Any, path: Path, expected_sha: str, label: str) -> tuple[Any, dict[str, Any]]:
    resolved = _synced(path)
    _require(resolved.is_file(), f"{label} checkpoint missing: {resolved}")
    _require(_sha256(resolved) == expected_sha, f"{label} checkpoint SHA mismatch")
    policy, _payload = rank_eval._load_policy(resolved, config)
    _require(int(policy.observation_dim) == OBSERVATION_DIM, f"{label} policy dimension changed")
    model = policy.model
    model.eval()
    optimizer = getattr(policy, "optimizer", None)
    _require(not getattr(optimizer, "state", {}), f"{label} optimizer state is not empty")
    return policy, {
        "label": label,
        "path": str(resolved),
        "checkpoint_sha256": _sha256(resolved),
        "model_digest": rank_eval._model_digest(policy),
        "observation_dim": int(policy.observation_dim),
        "training": bool(model.training),
        "optimizer_state_empty": True,
    }


def _frozen_binding(before: Any, after: Any, config: Any) -> dict[str, Any]:
    source = before.model.state_dict()
    candidate = after.model.state_dict()
    _require(set(source) == set(candidate), "before/after state keys changed")
    groups = (
        "actor_encoder", "proposal_head", "gate_head", "gate_temperature",
        "gate_adapter", "critic_encoder", "critic_head",
    )
    equalities: dict[str, bool] = {}
    for group in groups:
        names = [name for name in source if name == group or name.startswith(group + ".")]
        equalities[group] = bool(names) and all(torch.equal(source[name], candidate[name]) for name in names)
    changed = sorted(group for group in groups if not equalities[group])
    allowed = {"gate_adapter", "critic_encoder", "critic_head"}
    _require(set(changed) <= allowed, f"unexpected frozen groups changed: {changed}")
    zero = torch.zeros((1, OBSERVATION_DIM), dtype=torch.float32, device=before.device)
    with torch.no_grad():
        before_gate, before_q = before.model.distributions(zero)
        after_gate, after_q = after.model.distributions(zero)
    return {
        **{f"{key}_bitwise_unchanged": value for key, value in equalities.items()},
        "observed_changed_groups": changed,
        "allowed_changed_groups": sorted(allowed),
        "proposal_logits_zero_input_bitwise_unchanged": bool(torch.equal(before_q.logits, after_q.logits)),
        "proposal_probabilities_zero_input_bitwise_unchanged": bool(torch.equal(before_q.probs, after_q.probs)),
        "gate_logits_zero_input_bitwise_unchanged": bool(torch.equal(before_gate.logits, after_gate.logits)),
        "config_observation_schema": str(config.observation_schema_version),
    }


def _load_arm_inputs() -> list[dict[str, Any]]:
    cells = _expected_cells()
    result: list[dict[str, Any]] = []
    for cell in cells:
        cell_id = int(cell["cell_id"])
        arm_path = _arm_path(cell_id)
        arm = _read_json(arm_path)
        _require(arm.get("scenario") == SCENARIO, f"cell {cell_id} scenario mismatch")
        _require(int(arm.get("seed", -1)) == int(cell["env_seed"]), f"cell {cell_id} environment seed mismatch")
        _require(int(arm.get("policy_seed", -1)) == int(cell["policy_seed"]), f"cell {cell_id} policy seed mismatch")
        _require(arm.get("status") == "completed", f"cell {cell_id} source arm is not completed")
        _require(arm.get("checkpoint_sha256") == AFTER_SHA256, f"cell {cell_id} source checkpoint mismatch")
        tape_value = arm.get("action_tape") or (arm.get("summary") or {}).get("action_tape")
        _require(isinstance(tape_value, str) and tape_value, f"cell {cell_id} action tape path missing")
        tape_path = _synced(Path(tape_value))
        _require(tape_path.is_file(), f"cell {cell_id} action tape missing: {tape_path}")
        payload = _read_json(tape_path)
        rows = payload.get("action_tape", payload.get("records", []))
        _require(isinstance(rows, list) and rows, f"cell {cell_id} action tape rows missing")
        expected_steps = int((arm.get("summary") or {}).get("steps", len(rows)))
        _require(len(rows) == expected_steps, f"cell {cell_id} tape step count mismatch")
        lifecycle = arm.get("lifecycle") if isinstance(arm.get("lifecycle"), Mapping) else {}
        summary = arm.get("summary") if isinstance(arm.get("summary"), Mapping) else {}
        _require(lifecycle.get("replay_exact") is True and summary.get("terminal") is True,
                 f"cell {cell_id} source lifecycle is not replay-exact terminal")
        outcome_by_uid: dict[int, dict[str, Any]] = {}
        for row in lifecycle.get("rows", []):
            if isinstance(row, Mapping) and row.get("uid") is not None:
                outcome_by_uid[int(row["uid"])] = dict(row)
        _require(len(outcome_by_uid) == 24, f"cell {cell_id} lifecycle UID rows are incomplete")
        result.append({
            "cell": cell,
            "arm_path": arm_path,
            "arm_sha256": _sha256(arm_path),
            "arm": arm,
            "tape_path": tape_path,
            "tape_sha256": _sha256(tape_path),
            "tape_payload": payload,
            "tape_rows": rows,
            "outcome_by_uid": outcome_by_uid,
        })
    return result


def _preflight_payload() -> tuple[dict[str, Any], list[dict[str, Any]], Any, Any, dict[str, Any]]:
    full_path = _synced(INPUT_ROOT / "full.json")
    pilot_path = _synced(INPUT_ROOT / "pilot.json")
    preflight_path = _synced(INPUT_ROOT / "preflight.json")
    _require_sha(full_path, FULL_SHA256, "full input")
    _require_sha(pilot_path, PILOT_SHA256, "pilot input")
    _require_sha(preflight_path, PREFLIGHT_SHA256, "preflight input")
    full = _read_json(full_path)
    pilot = _read_json(pilot_path)
    preflight = _read_json(preflight_path)
    _require(full.get("status") == "FULL_OK", "full input is not terminal FULL_OK")
    _require(pilot.get("status") in {"PILOT_OK", "PILOT_INCOMPLETE"}, "pilot input status is not registered")
    _require(preflight.get("status") == "PREFLIGHT_OK", "preflight input is not successful")
    _require(preflight.get("PPO_ALLOWED") is False and preflight.get("heldout_used") is False
             and preflight.get("teacher_used") is False and preflight.get("no_training") is True
             and preflight.get("optimizer_used") is False and preflight.get("off_policy") is True,
             "source preflight has disallowed evaluation flags")
    config, _configs = trained_support._build_inputs()
    _require(float(config.max_simulation_seconds) == 1800.0 and float(config.dt) == 1.0,
             "current config horizon changed")
    _require(int(config.runtime_proposal_shortlist_k) == SHORTLIST_K, "current shortlist K changed")
    _require(str(config.observation_schema_version) == "phaseA_low_obs_v2_1_retry_state", "observation schema changed")
    current_config_snapshot = trained_support._config_snapshot(config)
    runtime = preflight.get("runtime") if isinstance(preflight.get("runtime"), Mapping) else {}
    _require(runtime.get("config_snapshot") == current_config_snapshot, "current config binding changed")
    _require(runtime.get("max_steps") == 1800 and runtime.get("deterministic") is False
             and runtime.get("native_rollout") == "evaluate_p3_runtime_rank_repair._runtime_record",
             "source runtime binding changed")
    schema = preflight.get("schema") if isinstance(preflight.get("schema"), Mapping) else {}
    _require(schema.get("observation") == "phaseA_low_obs_v2_1_retry_state"
             and int(schema.get("observation_dim", -1)) == OBSERVATION_DIM
             and schema.get("runtime_shortlist_k") == SHORTLIST_K,
             "source schema binding changed")
    checkpoints = preflight.get("checkpoints") if isinstance(preflight.get("checkpoints"), Mapping) else {}
    _require(isinstance(checkpoints.get("before"), Mapping)
             and checkpoints["before"].get("checkpoint_sha256") == BEFORE_SHA256,
             "source before checkpoint binding changed")
    _require(isinstance(checkpoints.get("after"), Mapping)
             and checkpoints["after"].get("checkpoint_sha256") == AFTER_SHA256,
             "source after checkpoint binding changed")
    _require(isinstance(checkpoints.get("warmup"), Mapping)
             and checkpoints["warmup"].get("sha256") == WARMUP_SHA256,
             "source warmup binding changed")
    caller_rng = trained_support._rng_snapshot()
    try:
        before, before_binding = _policy_binding(config, trained_support.BEFORE_CHECKPOINT, BEFORE_SHA256, "before")
        after, after_binding = _policy_binding(config, trained_support.AFTER_CHECKPOINT, AFTER_SHA256, "after")
        freeze = _frozen_binding(before, after, config)
    finally:
        trained_support._restore_rng(caller_rng)
    _require(trained_support._rng_equal(caller_rng, trained_support._rng_snapshot()), "caller RNG changed during preflight")
    arms = _load_arm_inputs()
    source = {
        "before": before_binding,
        "after": after_binding,
        "warmup_sha256": WARMUP_SHA256,
        "freeze": freeze,
    }
    input_binding = {
        "full_path": str(full_path), "full_sha256": _sha256(full_path),
        "pilot_path": str(pilot_path), "pilot_sha256": _sha256(pilot_path),
        "preflight_path": str(preflight_path), "preflight_sha256": _sha256(preflight_path),
        "arms": [
            {"cell_id": item["cell"]["cell_id"], "arm_path": str(item["arm_path"]),
             "arm_sha256": item["arm_sha256"], "tape_path": str(item["tape_path"]),
             "tape_sha256": item["tape_sha256"], "steps": len(item["tape_rows"])}
            for item in arms
        ],
        "config_snapshot": trained_support._config_snapshot(config),
        "runtime": preflight.get("runtime"),
        "source": source,
        "runner_sha256": _sha256(Path(__file__)),
        "protocol_sha256": _sha256(PROTOCOL) if PROTOCOL.is_file() else None,
    }
    return input_binding, arms, before, after, config


def _uid_value(mapping: Mapping[Any, Any], uid: str, default: Any = None) -> Any:
    if uid in mapping:
        return mapping[uid]
    if str(uid) in mapping:
        return mapping[str(uid)]
    return default


def _phase_value(actor: Any) -> str:
    phase = getattr(actor, "phase", None)
    return str(getattr(phase, "value", phase))


def _group_for_phase(phase: str) -> str:
    if "outer" in phase:
        return "outer"
    if "middle" in phase:
        return "middle"
    if "inner" in phase:
        return "inner"
    return "other"


def _actor_context(env: Any, uid: int, outcome_by_uid: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    actor = getattr(env, "active", {}).get(uid)
    if actor is None:
        actor = getattr(env, "active", {}).get(str(uid))
    _require(actor is not None, f"active actor missing for uid={uid}")
    arrival_id = getattr(actor, "arrival_id", None)
    _require(arrival_id is not None, f"arrival_id missing for uid={uid}")
    phase = _phase_value(actor)
    retry_after = float(getattr(actor, "safety_retry_after", 0.0))
    return {
        "uid": int(uid),
        "arrival_id": int(arrival_id),
        "phase": phase,
        "group": _group_for_phase(phase),
        "ring_role": str(getattr(getattr(actor, "ring_role", None), "value", getattr(actor, "ring_role", None))),
        "soc": float(getattr(actor, "soc")),
        "safety_retry_after": retry_after,
        "retry_blocked": bool(float(getattr(env, "now", 0.0)) < retry_after),
        "outcome": (dict(outcome_by_uid[int(uid)]) if int(uid) in outcome_by_uid else None),
    }


def _world_from_observations(observations: Mapping[Any, Any]) -> tuple[list[str], np.ndarray]:
    uids = sorted(str(uid) for uid in observations)
    if not uids:
        return [], np.zeros((0, OBSERVATION_DIM), dtype=np.float32)
    world = np.asarray([
        observations[uid] if uid in observations else observations[int(uid)] for uid in uids
    ], dtype=np.float32)
    _require(world.ndim == 2 and world.shape[1] == OBSERVATION_DIM, "world is not an 88D matrix")
    _require(bool(np.isfinite(world).all()), "world contains non-finite values")
    return uids, world


def _distribution_snapshot(policy: Any, world: np.ndarray) -> dict[str, np.ndarray]:
    tensor = torch.as_tensor(world, dtype=torch.float32, device=policy.device)
    with torch.no_grad():
        gate, proposal = policy.model.distributions(tensor)
        return {
            "gate_probs": gate.probs.detach().cpu().numpy().astype(np.float64),
            "gate_logits": gate.logits.detach().cpu().numpy().astype(np.float64),
            "gate_entropy": gate.entropy().detach().cpu().numpy().astype(np.float64),
            "proposal_q": proposal.probs.detach().cpu().numpy().astype(np.float64),
            "proposal_logits": proposal.logits.detach().cpu().numpy().astype(np.float64),
            "proposal_entropy": proposal.entropy().detach().cpu().numpy().astype(np.float64),
        }


def _categorical_logprob(logits: np.ndarray, index: int, value: int) -> float:
    row = torch.as_tensor(logits[int(index)], dtype=torch.float32)
    with torch.no_grad():
        result = torch.distributions.Categorical(logits=row).log_prob(
            torch.as_tensor(int(value), dtype=torch.long)
        )
    number = float(result.detach().cpu())
    _require(math.isfinite(number), "analytic log probability is non-finite")
    return number


def _validate_tape_logprob(actual: float | None, stored: Any, *, label: str,
                           tolerance: float = LP_TOLERANCE) -> None:
    _require(actual is not None and math.isfinite(float(actual)), f"{label} analytic log probability missing")
    _require(stored is not None and math.isfinite(float(stored)), f"{label} tape log probability missing")
    _require(abs(float(actual) - float(stored)) <= float(tolerance),
             f"{label} log probability mismatch: actual={actual!r} stored={stored!r}")


def _require_replay_exact(replay: Mapping[str, Any], expected_steps: int) -> None:
    _require(replay.get("exact") is True, "action tape replay was not exact")
    _require(int(replay.get("steps_replayed", -1)) == int(expected_steps),
             "action tape replay step count changed")


def _row_quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p05": None, "p50": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {key: float(np.quantile(array, q)) for key, q in (("p05", .05), ("p50", .50), ("p95", .95))}


def _categorical_kl_from_log_probs(before: Sequence[float], after: Sequence[float]) -> float:
    before_tensor = torch.as_tensor(before, dtype=torch.float64)
    after_tensor = torch.as_tensor(after, dtype=torch.float64)
    _require(before_tensor.shape == after_tensor.shape and before_tensor.ndim == 1,
             "gate KL inputs have incompatible shapes")
    with torch.no_grad():
        before_dist = torch.distributions.Categorical(logits=before_tensor)
        after_dist = torch.distributions.Categorical(logits=after_tensor)
        value = torch.distributions.kl_divergence(before_dist, after_dist)
    result = float(value.detach().cpu())
    _require(math.isfinite(result), "gate KL is non-finite")
    return result


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    outcomes: Counter[str] = Counter()
    for row in rows:
        groups[str(row["group"])].append(row)
        if row.get("outcome"):
            outcomes[str(row["outcome"].get("outcome", "unknown"))] += 1

    def group_summary(group_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        before = [float(row["gate_release_before"]) for row in group_rows]
        after = [float(row["gate_release_after"]) for row in group_rows]
        delta = [float(row["gate_release_delta"]) for row in group_rows]
        eligible = [row for row in group_rows if bool(row["actor_mask"])]
        eligible_before = [float(row["gate_release_before"]) for row in eligible]
        eligible_after = [float(row["gate_release_after"]) for row in eligible]
        eligible_delta = [float(row["gate_release_delta"]) for row in eligible]
        eligible_kl_forward = [
            _categorical_kl_from_log_probs(row["gate_logits_before"], row["gate_logits_after"])
            for row in eligible
        ]
        eligible_kl_reverse = [
            _categorical_kl_from_log_probs(row["gate_logits_after"], row["gate_logits_before"])
            for row in eligible
        ]
        return {
            "rows": len(group_rows),
            "eligible_rows": sum(bool(row["actor_mask"]) for row in group_rows),
            "gate_statistics_scope": "all_rows; actor_eligible_fields_are_explicit_below",
            "gate_release_before_mean": float(np.mean(before)) if before else None,
            "gate_release_after_mean": float(np.mean(after)) if after else None,
            "gate_release_delta_mean": float(np.mean(delta)) if delta else None,
            "gate_release_delta_quantiles": _row_quantiles(delta),
            "actor_eligible_gate_release_before_mean": float(np.mean(eligible_before)) if eligible_before else None,
            "actor_eligible_gate_release_after_mean": float(np.mean(eligible_after)) if eligible_after else None,
            "actor_eligible_gate_release_delta_mean": float(np.mean(eligible_delta)) if eligible_delta else None,
            "actor_eligible_gate_release_delta_quantiles": _row_quantiles(eligible_delta),
            "actor_eligible_gate_kl_before_to_after_mean": float(np.mean(eligible_kl_forward)) if eligible_kl_forward else None,
            "actor_eligible_gate_kl_before_to_after_max": max(eligible_kl_forward, default=None),
            "actor_eligible_gate_kl_before_to_after_quantiles": _row_quantiles(eligible_kl_forward),
            "actor_eligible_gate_kl_after_to_before_mean": float(np.mean(eligible_kl_reverse)) if eligible_kl_reverse else None,
            "actor_eligible_gate_kl_after_to_before_max": max(eligible_kl_reverse, default=None),
            "actor_eligible_gate_kl_after_to_before_quantiles": _row_quantiles(eligible_kl_reverse),
            "proposal_qmax_before_mean": float(np.mean([row["proposal_qmax_before"] for row in group_rows])) if group_rows else None,
            "proposal_qmax_after_mean": float(np.mean([row["proposal_qmax_after"] for row in group_rows])) if group_rows else None,
            "proposal_entropy_before_mean": float(np.mean([row["proposal_entropy_before"] for row in group_rows])) if group_rows else None,
            "proposal_entropy_after_mean": float(np.mean([row["proposal_entropy_after"] for row in group_rows])) if group_rows else None,
            "proposal_equal_max_abs": max((float(row["proposal_max_abs_diff"]) for row in group_rows), default=None),
            "argmax_gate_difference_count": sum(bool(row["gate_argmax_diff"]) for row in group_rows),
            "argmax_proposal_difference_count": sum(bool(row["proposal_argmax_diff"]) for row in group_rows),
            "retry_blocked_rows": sum(bool(row["retry_blocked"]) for row in group_rows),
            "retry_unblocked_rows": sum(not bool(row["retry_blocked"]) for row in group_rows),
        }

    return {
        "rows": len(rows),
        "eligible_rows": sum(bool(row["actor_mask"]) for row in rows),
        "groups": {group: group_summary(groups.get(group, [])) for group in ("outer", "middle", "inner", "other")},
        "actual_outcomes": dict(sorted(outcomes.items())),
        "repeated_actor_rows_are_not_independent_trials": True,
    }


def _save_npz(path: Path, rows: Sequence[Mapping[str, Any]], expected_steps: int,
              metadata: Mapping[str, Any], *, team_rewards: Sequence[float]) -> None:
    _require(len(team_rewards) == int(expected_steps), "raw team reward step count changed")
    _require(all(math.isfinite(float(value)) for value in team_rewards),
             "raw team reward contains a non-finite value")
    # Keep the native per-step observation order (string UID order from
    # ``_world_from_observations``); numeric UID sorting would silently move
    # rows such as "10" ahead of "2" in the persisted world matrix.
    ordered = [row for _index, row in sorted(
        enumerate(rows), key=lambda pair: (int(pair[1]["step"]), pair[0])
    )]
    counts = Counter(int(row["step"]) for row in ordered)
    offsets = [0]
    for step in range(expected_steps):
        offsets.append(offsets[-1] + int(counts.get(step, 0)))
    def array(name: str, default: Any = 0.0) -> np.ndarray:
        return np.asarray([row.get(name, default) for row in ordered])
    worlds = np.asarray([row["world"] for row in ordered], dtype=np.float32)
    proposal_before = np.asarray([row["proposal_q_before"] for row in ordered], dtype=np.float32)
    proposal_after = np.asarray([row["proposal_q_after"] for row in ordered], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        world=worlds,
        step=array("step", -1).astype(np.int64),
        uid=array("uid", -1).astype(np.int64),
        arrival_id=array("arrival_id", -1).astype(np.int64),
        phase=np.asarray([str(row.get("phase", "")) for row in ordered], dtype="U64"),
        soc=array("soc", np.nan).astype(np.float64),
        actor_mask=array("actor_mask", False).astype(bool),
        retry_blocked=array("retry_blocked", False).astype(bool),
        before_gate_probs=np.asarray([row["gate_probs_before"] for row in ordered], dtype=np.float32),
        after_gate_probs=np.asarray([row["gate_probs_after"] for row in ordered], dtype=np.float32),
        before_gate_logits=np.asarray([row["gate_logits_before"] for row in ordered], dtype=np.float32),
        after_gate_logits=np.asarray([row["gate_logits_after"] for row in ordered], dtype=np.float32),
        proposal_q_before=proposal_before,
        proposal_q_after=proposal_after,
        requested_gate=array("requested_gate", -1).astype(np.int64),
        requested_proposal=array("requested_proposal", -1).astype(np.int64),
        gate_logprob=np.asarray([
            np.nan if row.get("gate_logprob_tape") is None else float(row["gate_logprob_tape"])
            for row in ordered
        ], dtype=np.float64),
        proposal_logprob=np.asarray([
            np.nan if row.get("proposal_logprob_tape") is None else float(row["proposal_logprob_tape"])
            for row in ordered
        ], dtype=np.float64),
        team_reward_raw=np.asarray([float(value) for value in team_rewards], dtype=np.float64),
        step_offsets=np.asarray(offsets, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(_jsonable(metadata), sort_keys=True, allow_nan=False)),
    )


def _replay_cell(item: Mapping[str, Any], before_policy: Any, after_policy: Any, config: Any,
                 output: Path, *, progress_callback: Any | None = None) -> dict[str, Any]:
    tape_path = Path(item["tape_path"])
    tape_rows = list(item["tape_rows"])
    replay_before = trained_support._rng_snapshot()
    try:
        environment = TerminalParallelEnv(config)
    except BaseException:
        trained_support._restore_rng(replay_before)
        raise
    if not trained_support._rng_equal(replay_before, trained_support._rng_snapshot()):
        trained_support._restore_rng(replay_before)
        raise MatchedGateShiftError("environment construction changed caller RNG")
    original_step = environment.step
    rows: list[dict[str, Any]] = []
    eligible_rows_count = 0
    team_rewards: list[float] = []
    before_digest = rank_eval._model_digest(before_policy)
    after_digest = rank_eval._model_digest(after_policy)
    by_step = {int(row["step"]): row for row in tape_rows}
    analytic_rng_failures = 0
    hook_restored = False

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal analytic_rng_failures, eligible_rows_count
        step = int(getattr(environment, "step_count", -1))
        source_row = by_step.get(step)
        if source_row is None:
            raise MatchedGateShiftError(f"tape step missing for replay step {step}")
        observations = environment._observations()
        _require(_digest({key: value for key, value in sorted(observations.items())}) == source_row.get("observation_hash"),
                 f"pre-step observation hash mismatch at step {step}")
        uids, world = _world_from_observations(observations)
        rng_before = trained_support._rng_snapshot()
        before_details = student._student_request_details(before_policy, environment, observations,
                                                          deterministic=True, shortlist_k=SHORTLIST_K)
        after_details = student._student_request_details(after_policy, environment, observations,
                                                         deterministic=True, shortlist_k=SHORTLIST_K)
        _require(before_details[0] == uids and after_details[0] == uids, f"UID ordering mismatch at step {step}")
        before_mask = np.asarray(before_details[2], dtype=bool)
        after_mask = np.asarray(after_details[2], dtype=bool)
        _require(np.array_equal(before_mask, after_mask), f"actor eligibility mask changed at step {step}")
        if not uids:
            _require(trained_support._rng_equal(rng_before, trained_support._rng_snapshot()),
                     f"analytic observer changed caller RNG at empty step {step}")
            if progress_callback is not None and (step + 1) % 50 == 0:
                progress_callback(step + 1, eligible_rows_count, source_row.get("post_state_hash"))
            result = original_step(*args, **kwargs)
            _require(isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], Mapping),
                     f"step reward mapping missing at step {step}")
            team_reward = float(sum(float(value) for value in result[1].values()))
            _require(math.isfinite(team_reward), f"team reward is non-finite at step {step}")
            team_rewards.append(team_reward)
            return result
        before_dist = _distribution_snapshot(before_policy, world)
        after_dist = _distribution_snapshot(after_policy, world)
        _require(float(np.max(np.abs(before_dist["proposal_q"] - after_dist["proposal_q"]))) <= DIST_TOLERANCE,
                 f"proposal probabilities changed at step {step}")
        _require(float(np.max(np.abs(before_dist["proposal_logits"] - after_dist["proposal_logits"]))) <= DIST_TOLERANCE,
                 f"proposal logits changed at step {step}")
        if not trained_support._rng_equal(rng_before, trained_support._rng_snapshot()):
            analytic_rng_failures += 1
            raise MatchedGateShiftError(f"analytic observer changed caller RNG at step {step}")
        requested_gates = source_row.get("requested_gate", {})
        requested_proposals = source_row.get("requested_proposal", {})
        tape_gate_lp = source_row.get("gate_logprob", {})
        tape_proposal_lp = source_row.get("proposal_logprob", {})
        for index, uid_text in enumerate(uids):
            uid = int(uid_text)
            context = _actor_context(environment, uid, item["outcome_by_uid"])
            gate_raw = _uid_value(requested_gates, uid_text, None)
            proposal_raw = _uid_value(requested_proposals, uid_text, None)
            actor_mask = bool(after_mask[index])
            if gate_raw is not None:
                _require(not isinstance(gate_raw, bool) and int(gate_raw) in {0, 1},
                         f"invalid tape gate at step={step} uid={uid}")
            if proposal_raw is not None:
                _require(not isinstance(proposal_raw, bool),
                         f"invalid tape proposal at step={step} uid={uid}")
            if actor_mask:
                eligible_rows_count += 1
                _require(gate_raw is not None, f"eligible tape gate missing at step={step} uid={uid}")
                gate = int(gate_raw)
                if gate == trained_support.RELEASE_GATE:
                    _require(proposal_raw is not None and 0 <= int(proposal_raw) < 25,
                             f"eligible RELEASE proposal missing/invalid at step={step} uid={uid}")
                    proposal = int(proposal_raw)
                else:
                    _require(proposal_raw is None or int(proposal_raw) == -1,
                             f"eligible WAIT proposal sentinel changed at step={step} uid={uid}")
                    proposal = None
                after_gate_lp = _categorical_logprob(after_dist["gate_logits"], index, gate)
                stored_gate_lp = _uid_value(tape_gate_lp, uid_text, None)
                _validate_tape_logprob(after_gate_lp, stored_gate_lp,
                                       label=f"after gate step={step} uid={uid}")
                proposal_lp = None
                stored_proposal_lp = _uid_value(tape_proposal_lp, uid_text, None)
                if gate == trained_support.RELEASE_GATE and proposal is not None:
                    proposal_lp = _categorical_logprob(after_dist["proposal_logits"], index, proposal)
                    _validate_tape_logprob(proposal_lp, stored_proposal_lp,
                                           label=f"after proposal step={step} uid={uid}")
            else:
                gate = None if gate_raw is None else int(gate_raw)
                proposal = None if proposal_raw is None else int(proposal_raw)
                proposal_lp = None
            rows.append({
                **context,
                "step": step,
                "world": world[index].astype(np.float32).tolist(),
                "actor_mask": actor_mask,
                "requested_gate": gate,
                "requested_proposal": proposal if proposal is not None else -1,
                "gate_logprob_tape": None if not actor_mask else float(_uid_value(tape_gate_lp, uid_text, np.nan)),
                "proposal_logprob_tape": None if not actor_mask or proposal is None else float(_uid_value(tape_proposal_lp, uid_text, np.nan)),
                "gate_probs_before": before_dist["gate_probs"][index].tolist(),
                "gate_probs_after": after_dist["gate_probs"][index].tolist(),
                "gate_logits_before": before_dist["gate_logits"][index].tolist(),
                "gate_logits_after": after_dist["gate_logits"][index].tolist(),
                "gate_release_before": float(before_dist["gate_probs"][index, trained_support.RELEASE_GATE]),
                "gate_release_after": float(after_dist["gate_probs"][index, trained_support.RELEASE_GATE]),
                "gate_release_delta": float(after_dist["gate_probs"][index, trained_support.RELEASE_GATE] - before_dist["gate_probs"][index, trained_support.RELEASE_GATE]),
                "proposal_q_before": before_dist["proposal_q"][index].tolist(),
                "proposal_q_after": after_dist["proposal_q"][index].tolist(),
                "proposal_qmax_before": float(np.max(before_dist["proposal_q"][index])),
                "proposal_qmax_after": float(np.max(after_dist["proposal_q"][index])),
                "proposal_entropy_before": float(before_dist["proposal_entropy"][index]),
                "proposal_entropy_after": float(after_dist["proposal_entropy"][index]),
                "proposal_max_abs_diff": float(np.max(np.abs(before_dist["proposal_q"][index] - after_dist["proposal_q"][index]))),
                "gate_argmax_diff": int(np.argmax(before_dist["gate_probs"][index])) != int(np.argmax(after_dist["gate_probs"][index])),
                "proposal_argmax_diff": int(np.argmax(before_dist["proposal_q"][index])) != int(np.argmax(after_dist["proposal_q"][index])),
            })
        if progress_callback is not None and (step + 1) % 50 == 0:
            progress_callback(step + 1, eligible_rows_count, source_row.get("post_state_hash"))
        result = original_step(*args, **kwargs)
        _require(isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], Mapping),
                 f"step reward mapping missing at step {step}")
        team_reward = float(sum(float(value) for value in result[1].values()))
        _require(math.isfinite(team_reward), f"team reward is non-finite at step {step}")
        team_rewards.append(team_reward)
        return result

    environment.step = observed_step  # type: ignore[method-assign]
    replay: dict[str, Any]
    try:
        replay = SafetyTraceRecorder.replay_action_tape(environment, tape_path)
    except BaseException:
        trained_support._restore_rng(replay_before)
        raise
    finally:
        environment.step = original_step  # type: ignore[method-assign]
        hook_restored = True
    after_replay_rng = trained_support._rng_snapshot()
    trained_support._restore_rng(replay_before)
    after_model_digest = rank_eval._model_digest(after_policy)
    before_model_digest = rank_eval._model_digest(before_policy)
    _require(trained_support._rng_equal(replay_before, after_replay_rng), "caller RNG changed during replay")
    _require(trained_support._rng_equal(replay_before, trained_support._rng_snapshot()),
             "caller RNG was not restored after replay")
    _require(before_digest == before_model_digest and after_digest == after_model_digest, "model digest changed during replay")
    _require_replay_exact(replay, len(tape_rows))
    _require(not analytic_rng_failures, "analytic observer RNG check failed")
    _require(len(team_rewards) == len(tape_rows), "raw team reward rows are incomplete")
    arm_summary = item.get("arm", {}).get("summary", {}) if isinstance(item.get("arm"), Mapping) else {}
    expected_reward_total = arm_summary.get("reward_total")
    _require(expected_reward_total is not None and math.isfinite(float(expected_reward_total)),
             "recorded arm reward_total is missing")
    _require(abs(float(sum(team_rewards)) - float(expected_reward_total)) <= 1e-6,
             "raw replay team reward total changed")
    source_summary = _summary_rows(rows)
    cell = item["cell"]
    npz_path = output / f"cell_{int(cell['cell_id']):02d}_matched_states.npz"
    _save_npz(npz_path, rows, len(tape_rows), {
        "cell": cell, "tape_sha256": item["tape_sha256"], "format": "matched_gate_shift_v1",
        "reward_source": "exact_original_step_reward_sum", "reward_scale_applied": False,
    }, team_rewards=team_rewards)
    npz_sha256 = _sha256(npz_path)
    return {
        "cell": cell,
        "status": "COMPLETED",
        "source_arm_path": str(item["arm_path"]),
        "action_tape": str(tape_path),
        "action_tape_sha256": item["tape_sha256"],
        "replay": {
            "exact": replay.get("exact"),
            "reset_state_match": replay.get("reset_state_match"),
            "config_match": replay.get("config_match"),
            "steps_expected": replay.get("steps_expected"),
            "steps_replayed": replay.get("steps_replayed"),
            "step_checks": replay.get("steps"),
        },
        "invariants": {
            "caller_rng_unchanged": True,
            "analytic_observer_rng_unchanged": True,
            "before_model_unchanged": before_digest == before_model_digest,
            "after_model_unchanged": after_digest == after_model_digest,
            "instance_step_hook_restored": hook_restored,
            "policy_sampling": False,
            "extra_env_steps": 0,
            "synthetic_advantage_or_counterfactual_reward_computed": False,
            "observed_raw_team_reward_captured": True,
        },
        "observations": source_summary,
        "counts": {"replayed_steps": len(tape_rows), "matched_rows": len(rows),
                   "eligible_rows": source_summary["eligible_rows"],
                   "team_reward_steps": len(team_rewards),
                   "team_reward_raw_total": float(sum(team_rewards))},
        "artifacts": {"matched_states_npz": str(npz_path), "matched_states_npz_sha256": npz_sha256},
    }


def _write_progress(path: Path, *, phase: str, status: str, cell_id: int | None = None,
                    replayed_steps: int = 0, eligible_rows: int = 0,
                    hashflags_sofar: Mapping[str, Any] | None = None,
                    error: str | None = None) -> None:
    payload: dict[str, Any] = {
        "stage": STAGE, "phase": phase, "status": status,
        "cell_id": cell_id, "replayed_steps": int(replayed_steps),
        "eligible_rows": int(eligible_rows), "PPO_ALLOWED": False,
        "policy_sampling": False, "optimizer_used": False,
    }
    payload["hashflags_sofar"] = dict(hashflags_sofar or {})
    if error is not None:
        payload["error"] = error
    _write_json(path, payload)


def run_preflight(output: Path) -> dict[str, Any]:
    _require(not output.exists(), f"output directory must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    _write_progress(output / "progress.json", phase="preflight", status="STARTING")
    try:
        binding, arms, _before, _after, _config = _preflight_payload()
        payload = {
            "stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK",
            "input_binding": binding,
            "cells": [item["cell"] for item in arms],
            "scope": {"env_created": False, "replay_calls": 0, "policy_sampling": False,
                       "optimizer_used": False, "teacher_used": False, "PPO_ALLOWED": False},
        }
        _write_json(output / "preflight.json", payload, refuse_existing=True)
        _write_progress(output / "progress.json", phase="preflight", status="PREFLIGHT_OK")
        return payload
    except BaseException as exc:
        _write_json(output / "error.json", {"stage": STAGE, "phase": "preflight", "status": "FAILED",
                                               "error": {"type": type(exc).__name__, "message": str(exc)}})
        _write_progress(output / "progress.json", phase="preflight", status="FAILED", error=str(exc))
        raise


def _load_and_validate_phase(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Any, Any, Any]:
    stored = _read_json(output / "preflight.json")
    _require(stored.get("status") == "PREFLIGHT_OK", "matched preflight is not successful")
    binding, arms, before, after, config = _preflight_payload()
    _require(stored.get("input_binding") == binding, "matched preflight binding changed")
    return stored, arms, before, after, config


def run_pilot(output: Path) -> dict[str, Any]:
    preflight, arms, before, after, config = _load_and_validate_phase(output)
    _require(not (output / "pilot.json").exists(), "pilot artifact already exists")
    item = arms[0]
    _write_progress(output / "progress.json", phase="pilot", status="RUNNING", cell_id=1)
    summary = _replay_cell(item, before, after, config, output,
                           progress_callback=lambda steps, eligible, _hash: _write_progress(
                               output / "progress.json", phase="pilot", status="RUNNING", cell_id=1,
                               replayed_steps=steps, eligible_rows=eligible,
                               hashflags_sofar={"post_state_hash_observed": bool(_hash)}))
    _write_json(output / "cell_01.json", summary, refuse_existing=True)
    payload = {
        "stage": STAGE, "phase": "pilot", "status": "PILOT_OK",
        "preflight_sha256": _sha256(output / "preflight.json"),
        "cell": summary, "cell_sha256": _sha256(output / "cell_01.json"),
        "scope": {"one_existing_tape": True, "policy_sampling": False, "PPO_ALLOWED": False},
    }
    _write_json(output / "pilot.json", payload, refuse_existing=True)
    _write_progress(output / "progress.json", phase="pilot", status="PILOT_OK", cell_id=1,
                    replayed_steps=int(summary["counts"]["replayed_steps"]), eligible_rows=int(summary["counts"]["eligible_rows"]))
    return payload


def run_full(output: Path) -> dict[str, Any]:
    preflight, arms, before, after, config = _load_and_validate_phase(output)
    pilot = _read_json(output / "pilot.json")
    _require(pilot.get("status") == "PILOT_OK", "pilot is not reusable")
    _require(pilot.get("preflight_sha256") == _sha256(output / "preflight.json"), "pilot preflight changed")
    _require(pilot.get("cell_sha256") == _sha256(output / "cell_01.json"), "pilot cell changed")
    existing = _read_json(output / "cell_01.json")
    _require(existing.get("cell", {}).get("cell_id") == 1, "pilot cell identity changed")
    pilot_cell = pilot.get("cell") if isinstance(pilot.get("cell"), Mapping) else {}
    _require(pilot_cell.get("artifacts") == existing.get("artifacts"), "pilot cell artifact binding changed")
    npz_artifact = existing.get("artifacts", {}).get("matched_states_npz")
    npz_sha = existing.get("artifacts", {}).get("matched_states_npz_sha256")
    _require(isinstance(npz_artifact, str) and Path(npz_artifact).is_file(), "pilot NPZ artifact missing")
    _require(isinstance(npz_sha, str) and _sha256(Path(npz_artifact)) == npz_sha,
             "pilot NPZ artifact SHA changed")
    cells = [existing]
    for item in arms[1:]:
        cell_id = int(item["cell"]["cell_id"])
        _write_progress(output / "progress.json", phase="full", status="RUNNING", cell_id=cell_id)
        summary = _replay_cell(item, before, after, config, output,
                               progress_callback=lambda steps, eligible, _hash, cid=cell_id: _write_progress(
                               output / "progress.json", phase="full", status="RUNNING", cell_id=cid,
                               replayed_steps=steps, eligible_rows=eligible,
                               hashflags_sofar={"post_state_hash_observed": bool(_hash)}))
        _write_json(output / f"cell_{cell_id:02d}.json", summary, refuse_existing=True)
        cells.append(summary)
    payload = {
        "stage": STAGE, "phase": "full", "status": "FULL_OK",
        "preflight_sha256": _sha256(output / "preflight.json"),
        "pilot_sha256": _sha256(output / "pilot.json"), "cells": cells,
        "scope": {"existing_tapes_only": True, "policy_sampling": False,
                   "optimizer_used": False, "teacher_used": False, "PPO_ALLOWED": False},
    }
    _write_json(output / "full.json", payload, refuse_existing=True)
    _write_progress(output / "progress.json", phase="full", status="FULL_OK")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "pilot", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.phase == "preflight":
            result = run_preflight(args.output_dir)
        elif args.phase == "pilot":
            result = run_pilot(args.output_dir)
        else:
            result = run_full(args.output_dir)
    except Exception as exc:
        if args.output_dir.exists():
            _write_json(args.output_dir / "error.json", {"stage": STAGE, "phase": args.phase, "status": "FAILED",
                                                           "error": {"type": type(exc).__name__, "message": str(exc)}})
            _write_progress(args.output_dir / "progress.json", phase=args.phase, status="FAILED", error=str(exc))
        print(json.dumps({"stage": STAGE, "status": "FAILED", "error": str(exc)}, sort_keys=True), flush=True)
        return 2
    print(json.dumps({"stage": STAGE, "status": result.get("status"), "PPO_ALLOWED": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
