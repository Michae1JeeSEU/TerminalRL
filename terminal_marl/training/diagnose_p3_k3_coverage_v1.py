"""Bounded, inference-only K3 coverage diagnostic on frozen V7 tapes.

This temporary runner replays existing tapes and observes pre-action states.  It
never supplies Teacher or policy actions to the live replay environment.
"""
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
from terminal_marl.core.safety_trace import SafetyTraceRecorder, _digest, _environment_config_match
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import evaluate_p3_adaptive_kl_v7 as evaluator
from terminal_marl.training.k3_gate_feasibility import k3_certified_mass
from terminal_marl.training.v4_1_joint_teacher import teacher_joint_expert_v4_1


STAGE = "P3_K3_COVERAGE_DIAGNOSTIC_V1"
FORMAL_ROOT = Path("/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926")
FULL_OUTPUT = Path("/home/user/training_outputs/terminal/p3_k3_coverage_diagnostic_v1_20260926")
SMOKE_OUTPUT = Path("/home/user/training_outputs/terminal/p3_k3_coverage_diagnostic_v1_20260926_smoke")
SOURCE_SEEDS = tuple(range(873101, 873107))
SAMPLE_FRACTIONS = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75)
WITNESS_SEED = 873106
WITNESS_ARRIVAL_ID = 15
WITNESS_STEPS = (136, 152, 200, 250, 297, 500)
SOURCE_CHECKPOINT_SHA256 = evaluator.SOURCE_CHECKPOINT_SHA256
FINAL_CHECKPOINT_SHAS = {
    "run0_episode_0080": "ad42e9a72cac4943834041482e93adc34f123c851c4297f7462381fe0d52ff4e",
    "run1_episode_0240": "db74245759653bb61f8d17b0071cf9a04c2e3334e019a63cd99dff988cc23b3a",
    "run2_episode_0240": "4797ae2eea5da0e60a4b24e2fd0a60a6860c37b8efe1c77c0366a33a2d8fb5f9",
}


class DiagnosticError(ValueError):
    pass


class _StopAfterFirstSample(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON input missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


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


def _target_steps(step_count: int) -> list[int]:
    _require(step_count > 0, "tape must contain at least one step")
    return [int(math.floor((step_count - 1) * fraction)) for fraction in SAMPLE_FRACTIONS]


def _state(env: TerminalParallelEnv, uid: int) -> dict[str, Any]:
    uav = env.active.get(int(uid))
    if uav is None:
        return {"uid": int(uid), "present": False}
    return {
        "uid": int(uav.uid),
        "arrival_id": int(uav.arrival_id),
        "base_priority": int(uav.base_priority),
        "phase": getattr(uav.phase, "value", uav.phase),
        "ring_role": None if uav.ring_role is None else getattr(uav.ring_role, "value", uav.ring_role),
        "soc": float(uav.soc),
        "energy_remaining": float(uav.energy_remaining),
        "committed": uav.committed_plan is not None,
        "future_release_reserved": int(uav.uid) in env.future_release_reservations,
    }


def _eligible_actors(env: TerminalParallelEnv) -> list[Any]:
    actors = [
        uav for uav in env.active.values()
        if uav.committed_plan is None
        and getattr(uav.phase, "value", uav.phase) in {"ring_outer", "ring_middle", "ring_inner"}
        and int(uav.uid) not in env.future_release_reservations
    ]
    return sorted(actors, key=lambda uav: (int(uav.arrival_id), int(uav.uid)))


def _find_arrival(env: TerminalParallelEnv, arrival_id: int) -> Any | None:
    for uav in env.active.values():
        if int(uav.arrival_id) == int(arrival_id):
            return uav
    return None


def _restore_shadow_methods(shadow: TerminalParallelEnv) -> None:
    cls = type(shadow)
    for name in ("step", "_release", "_try_emergency_landing", "_candidate_energy_guard_reason"):
        setattr(shadow, name, getattr(cls, name).__get__(shadow, cls))
    safety_cls = type(shadow.safety)
    shadow.safety.filter_plan = safety_cls.filter_plan.__get__(shadow.safety, safety_cls)


def _catalogue(shadow: TerminalParallelEnv, uid: int) -> dict[str, Any]:
    uav = shadow.active.get(int(uid))
    _require(uav is not None, f"shadow actor missing: uid={uid}")
    selected, audit = teacher_joint_expert_v4_1(shadow, uav, audit_catalogue=True, state_only=True)
    release_rows = [
        row for row in audit
        if float(row.get("decoded", {}).get("release_request", 0.0)) > 0.0
    ]
    _require(len(audit) == 26, f"state-only catalogue size changed: {len(audit)}")
    _require(len(release_rows) == 25, f"RELEASE catalogue size changed: {len(release_rows)}")
    proposal_catalogue = [
        hierarchical_from_joint_index_v4_1(int(row.get("joint_index", -1))).proposal
        for row in release_rows
    ]
    _require(
        len(proposal_catalogue) == 25
        and all(index is not None for index in proposal_catalogue)
        and set(int(index) for index in proposal_catalogue) == set(range(25)),
        "RELEASE proposal mapping is not exactly 0..24",
    )
    certified_joint = [int(row.get("joint_index", -1)) for row in release_rows if bool(row.get("feasible"))]
    certified_proposals = [
        int(decoded.proposal)
        for decoded in (hierarchical_from_joint_index_v4_1(index) for index in certified_joint)
        if decoded.proposal is not None
    ]
    compact = []
    for row in release_rows:
        decoded = row.get("decoded", {})
        compact.append({
            "rank": int(row.get("rank", -1)),
            "joint_index": int(row.get("joint_index", -1)),
            "proposal": None if decoded.get("proposal") is None else int(decoded["proposal"]),
            "feasible": bool(row.get("feasible")),
            "reason": row.get("reason"),
        })
    return {
        "selected_joint_index": int(selected.index),
        "audit_count": len(audit),
        "release_catalogue_count": len(release_rows),
        "proposal_catalogue_indices": [int(index) for index in proposal_catalogue],
        "proposal_mapping_complete": True,
        "certified_release_count": len(certified_proposals),
        "certified_joint_indices": certified_joint,
        "certified_proposal_indices": certified_proposals,
        "reason_counts": dict(sorted(Counter(str(row.get("reason")) for row in release_rows).items())),
        "release_catalogue": compact,
    }


def _policy_forward(policy: Any, observation: np.ndarray, certified: set[int]) -> dict[str, Any]:
    values = torch.as_tensor(observation, dtype=torch.float32, device=policy.device).reshape(1, -1)
    _require(int(values.shape[-1]) == 88, "policy observation is not 88D")
    with torch.no_grad():
        gate_dist, proposal_dist = policy.model.distributions(values)
        gate_probs = torch.softmax(gate_dist.logits, dim=-1)[0]
        gate_logits = gate_dist.logits[0].detach().cpu().clone()
        proposal_logits = proposal_dist.logits[0].detach().cpu().clone()
    mass = k3_certified_mass(proposal_logits, certified, shortlist_k=3)
    p_release = float(gate_probs[1].detach().cpu().item())
    q_k3 = float(mass["certified_mass"])
    return {
        "observation_dimension": int(values.shape[-1]),
        "p_release": p_release,
        "gate_logits": _jsonable(gate_dist.logits[0]),
        "proposal_logits": _jsonable(proposal_logits),
        "gate_distribution": _jsonable(gate_probs),
        "q_k3": q_k3,
        "k3_certified_mass": q_k3,
        "p_release_q_k3": p_release * q_k3,
        "k3_mass_detail": mass,
    }


def _prefix_replay(tape: Path, config: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = _read_json(tape)
    payload["action_tape"] = rows
    with tempfile.TemporaryDirectory(prefix="p3_k3_coverage_prefix_") as directory:
        prefix = Path(directory) / "prefix.json"
        _write_json(prefix, payload)
        replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), prefix)
    return {
        "prefix_rows": len(rows),
        "json_roundtrip": True,
        "exact": bool(replay.get("exact")),
        "steps_expected": replay.get("steps_expected"),
        "steps_replayed": replay.get("steps_replayed"),
    }


def _row_step_inputs(row: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, ...]]]:
    actions = {
        str(uid): np.asarray(action, dtype=np.float32)
        for uid, action in (row.get("a_requested") or {}).items()
    }
    shortlists = {
        str(uid): tuple(np.asarray(action, dtype=np.float32) for action in values)
        for uid, values in (row.get("runtime_proposal_shortlists") or {}).items()
    }
    return actions, shortlists


def _first_eligible_prefix_step(tape: Path, config: Any) -> int:
    """Find the first source prefix containing a runtime-eligible actor.

    The replay implementation catches exceptions raised by an env.step wrapper,
    so smoke uses this bounded prefix scan and then verifies that prefix with
    SafetyTraceRecorder.replay_action_tape.
    """
    payload = _read_json(tape)
    rows = list(payload.get("action_tape", []))
    header = payload
    rng_before = _rng_snapshot()
    try:
        env = TerminalParallelEnv(config)
        observations, _ = env.reset(int(header["episode_seed"]))
        recipe = header.get("reset_recipe", {"kind": "generic_reset", "steps": []})
        SafetyTraceRecorder._apply_reset_recipe(env, recipe)
        for index, row in enumerate(rows):
            if _eligible_actors(env):
                return int(index)
            actions, shortlists = _row_step_inputs(row)
            observations, _, _, _, _ = env.step(actions, shortlists)
    finally:
        random.setstate(rng_before[0])
        np.random.set_state(rng_before[1])
        torch.set_rng_state(rng_before[2])
    raise DiagnosticError(f"smoke found no runtime-eligible actor: {tape}")


def _load_inputs(formal_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    protocol = evaluator._protocol()
    configs = evaluator._build_configs(protocol)
    source_manifest_path = formal_root / "run1" / "dev_before_training" / "manifest.json"
    source_manifest = _read_json(source_manifest_path)
    records = [dict(row) for row in source_manifest.get("records", []) if isinstance(row, Mapping)]
    _require(len(records) == 6, f"source baseline manifest expected 6 records, got {len(records)}")
    _require(sorted(int(row.get("seed", -1)) for row in records) == list(SOURCE_SEEDS), "source seed set changed")
    _require(all(str(row.get("scenario")) in {"flow6", "flow12", "dense24"} for row in records), "source scenario changed")
    witness_manifest = _read_json(formal_root / "run0" / "dev_after_0080" / "manifest.json")
    witness = next((dict(row) for row in witness_manifest.get("records", [])
                    if int(row.get("seed", -1)) == WITNESS_SEED and str(row.get("scenario")) == "dense24"), None)
    _require(witness is not None, "run0 dev80 witness record missing")
    source_path = Path(str((protocol.get("lineage") or {}).get("source_checkpoint_path")))
    _require(source_path.is_file(), f"source checkpoint missing: {source_path}")
    checkpoints = {
        "source_03c0": (source_path, SOURCE_CHECKPOINT_SHA256, False),
        "run0_episode_0080": (Path(str(witness["checkpoint"])), FINAL_CHECKPOINT_SHAS["run0_episode_0080"], True),
    }
    for run_index, label in ((1, "run1_episode_0240"), (2, "run2_episode_0240")):
        manifest_path = formal_root / f"run{run_index}" / "manifest.json"
        manifest = _read_json(manifest_path)
        final_value = manifest.get("final_checkpoint")
        _require(final_value, f"run{run_index} final checkpoint binding missing")
        final_path = Path(str(final_value))
        if not final_path.is_absolute():
            final_path = manifest_path.parent / final_path
        checkpoints[label] = (final_path, FINAL_CHECKPOINT_SHAS[label], True)
    for label, (path, expected, _final) in checkpoints.items():
        _require(path.is_file(), f"{label} checkpoint missing: {path}")
        _require(_sha256(path) == expected, f"{label} checkpoint SHA mismatch")
    return protocol, configs, records, {"witness": witness, "checkpoints": checkpoints, "source_manifest": source_manifest_path}


def _load_policies(protocol: Mapping[str, Any], config: Any, checkpoints: Mapping[str, tuple[Path, str, bool]]) -> dict[str, dict[str, Any]]:
    protocol_sha = evaluator._sha256(evaluator.PROTOCOL)
    learning_rate = float((protocol.get("optimizer") or {}).get("learning_rate", 3e-4))
    result = {}
    for label, (path, expected_sha, final_checkpoint) in checkpoints.items():
        policy, payload, actual_sha = evaluator._load_eval_policy(
            path,
            config=config,
            expected_checkpoint_sha256=expected_sha,
            protocol_sha256=protocol_sha,
            final_checkpoint=final_checkpoint,
            learning_rate=learning_rate,
        )
        result[label] = {
            "label": label,
            "path": str(path),
            "sha256": actual_sha,
            "policy": policy,
            "payload": payload,
            "model_digest_before": evaluator._model_digest(policy),
        }
    return result


def _config_binding(config: Any, tape: Path) -> dict[str, Any]:
    payload = _read_json(tape)
    tape_hash = str(payload.get("environment_config_hash", ""))
    config_hash = _digest(vars(config))
    match = _environment_config_match(config, tape_hash)
    _require(tape_hash and config_hash == tape_hash and bool(match[0]),
             f"tape config binding mismatch: {tape} config={config_hash} tape={tape_hash} match={match}")
    return {"config_sha256": config_hash, "environment_config_hash": tape_hash, "compatibility": match[1]}


def _summary_outcomes(tape: Path) -> dict[str, Any]:
    summary = _read_json(tape.parent.parent / "summary.json")
    outcomes = summary.get("outcomes") or {}
    return {str(key): value for key, value in outcomes.items()} if isinstance(outcomes, Mapping) else {}


def _replay_outcomes(env: TerminalParallelEnv) -> dict[str, Any]:
    return {str(key): getattr(value, "value", value) for key, value in env.ledger.outcomes.items()}


def _evaluate_state(env: TerminalParallelEnv, uid: int, policies: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    observation = env._observations().get(str(uid))
    _require(observation is not None, f"observation missing for uid={uid}")
    observation = np.asarray(observation, dtype=np.float32)
    before_digest = _digest(SafetyTraceRecorder._state(env))
    rng_before = _rng_snapshot()
    shadow_off = copy.deepcopy(env)
    shadow_on = copy.deepcopy(env)
    _restore_shadow_methods(shadow_off)
    _restore_shadow_methods(shadow_on)
    shadow_off.config = replace(shadow_off.config, reconcile_late_merge_slots_enabled=False)
    shadow_on.config = replace(shadow_on.config, reconcile_late_merge_slots_enabled=True)
    off = _catalogue(shadow_off, uid)
    on = _catalogue(shadow_on, uid)
    policy_rows = {}
    for label, item in policies.items():
        policy_rows[label] = _policy_forward(item["policy"], observation, set(off["certified_proposal_indices"]))
        policy_rows[label]["proposal_logits_equal_to_source"] = (
            label == "source_03c0"
            or policy_rows[label]["proposal_logits"] == policy_rows["source_03c0"]["proposal_logits"]
        )
        policy_rows[label]["q_k3_off"] = policy_rows[label]["q_k3"]
        policy_rows[label]["q_k3_on"] = float(
            k3_certified_mass(
                torch.as_tensor(policy_rows[label]["proposal_logits"], dtype=torch.float32),
                set(on["certified_proposal_indices"]), shortlist_k=3,
            )["certified_mass"]
        )
        policy_rows[label]["p_release_q_k3_on"] = policy_rows[label]["p_release"] * policy_rows[label]["q_k3_on"]
    after_digest = _digest(SafetyTraceRecorder._state(env))
    rng_after = _rng_snapshot()
    _require(before_digest == after_digest, "live environment changed during state diagnostic")
    _require(_rng_equal(rng_before, rng_after), "global RNG changed during state diagnostic")
    actor = env.active[int(uid)]
    state = _state(env, uid)
    return {
        "status": "evaluated",
        "uid": int(uid),
        "state": state,
        "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
        "observation_dimension": int(observation.size),
        "late_merge_off": off,
        "late_merge_on": on,
        "certified_release_set_equal": off["certified_proposal_indices"] == on["certified_proposal_indices"],
        "policies": policy_rows,
        "live_state_digest_before": before_digest,
        "live_state_digest_after": after_digest,
        "rng_unchanged": True,
        "actor_eligible": True,
        "ring_role": state.get("ring_role"),
        "soc_bucket": "le_0.65" if float(actor.soc) <= 0.65 else "gt_0.65",
        "priority": int(actor.base_priority),
    }


def _replay_tape(*, record: Mapping[str, Any], config: Any, policies: Mapping[str, Mapping[str, Any]], witness: bool = False,
                 stop_after_first: bool = False, targets_override: list[int] | None = None,
                 validate_outcomes: bool = True) -> dict[str, Any]:
    tape = Path(str(record["action_tape"]))
    payload = _read_json(tape)
    rows = list(payload.get("action_tape", []))
    targets = list(targets_override) if targets_override is not None else (_target_steps(len(rows)) if not witness else list(WITNESS_STEPS))
    _require(rows, f"empty tape: {tape}")
    config_binding = _config_binding(config, tape)
    sample_rows: list[dict[str, Any]] = []
    witness_missing: dict[int, dict[str, Any]] = {}
    pending = list(enumerate(targets))
    sampled_slots: set[int] = set()
    rng_before = _rng_snapshot()
    policy_digests_before = {label: item["model_digest_before"] for label, item in policies.items()}
    env = TerminalParallelEnv(config)
    original_step = env.step
    first_sample_step: int | None = None

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal first_sample_step
        step = int(env.step_count)
        selected_uid: int | None = None
        slot_index: int | None = None
        if pending and not witness:
            target_slot, target_step = pending[0]
            if step >= int(target_step):
                actors = _eligible_actors(env)
                if actors:
                    selected_uid = int(actors[target_slot % len(actors)].uid)
                    slot_index = int(target_slot)
                    pending.pop(0)
                    sampled_slots.add(slot_index)
        elif witness and step in targets and step not in sampled_slots:
            target = _find_arrival(env, WITNESS_ARRIVAL_ID)
            eligible_ids = {int(actor.uid) for actor in _eligible_actors(env)}
            if target is not None and int(target.uid) in eligible_ids:
                selected_uid = int(target.uid)
                slot_index = int(targets.index(step))
                sampled_slots.add(slot_index)
            else:
                slot_index = int(targets.index(step))
                witness_missing[slot_index] = {
                    "status": "missing_runtime_actor",
                    "arrival_id": WITNESS_ARRIVAL_ID,
                    "state": None if target is None else _state(env, int(target.uid)),
                    "reason": "arrival_missing" if target is None else "arrival_not_runtime_eligible",
                }
        if selected_uid is not None:
            row = {
                "slot_index": slot_index,
                "target_step": int(targets[int(slot_index)]),
                "observed_step": step,
                "time": float(env.now),
                "selection_rule": "sorted(arrival_id,uid)[slot_index % actor_count]" if not witness else "arrival_id_exact",
            }
            try:
                evaluated = _evaluate_state(env, selected_uid, policies)
                row.update(evaluated)
                first_sample_step = step if first_sample_step is None else first_sample_step
            except Exception as exc:
                row.update({"status": "error", "uid": selected_uid,
                            "error": {"type": type(exc).__name__, "message": str(exc)}})
                sample_rows.append(row)
                raise
            sample_rows.append(row)
            if stop_after_first:
                first_sample_step = step
                raise _StopAfterFirstSample
        return original_step(*args, **kwargs)

    env.step = observed_step  # type: ignore[method-assign]
    stopped = False
    stop_error = None
    try:
        replay = SafetyTraceRecorder.replay_action_tape(env, tape)
    except _StopAfterFirstSample:
        stopped = True
        replay = {"exact": False, "stopped_after_first_sample": True}
    except Exception as exc:
        stop_error = {"type": type(exc).__name__, "message": str(exc)}
        replay = {"exact": False, "error": stop_error}
    finally:
        env.step = original_step  # type: ignore[method-assign]
    if not stopped and stop_error is None and validate_outcomes:
        _require(replay.get("exact") is True, f"tape replay was not exact: {tape}")
        _require(_summary_outcomes(tape) == _replay_outcomes(env), f"tape outcomes changed: {tape}")
    if not witness and pending:
        for slot_index, target_step in pending:
            sample_rows.append({
                "slot_index": int(slot_index),
                "target_step": int(target_step),
                "status": "missing_runtime_actor",
                "error": None,
            })
    if witness:
        for slot_index, target_step in enumerate(targets):
            if slot_index not in sampled_slots:
                sample_rows.append({
                    "slot_index": int(slot_index), "target_step": int(target_step),
                    **witness_missing.get(slot_index, {
                        "status": "missing_arrival_state", "arrival_id": WITNESS_ARRIVAL_ID,
                    }),
                })
    rng_after = _rng_snapshot()
    if not stopped and stop_error is None and validate_outcomes:
        _require(_rng_equal(rng_before, rng_after), f"global RNG changed during tape replay: {tape}")
    for label, item in policies.items():
        _require(policy_digests_before[label] == evaluator._model_digest(item["policy"]), f"policy changed: {label}")
    return {
        "scenario": str(record.get("scenario")),
        "seed": int(record.get("seed")),
        "action_tape": str(tape),
        "action_tape_sha256": _sha256(tape),
        "steps_expected": len(rows),
        "target_steps": targets,
        "config_binding": config_binding,
        "witness": bool(witness),
        "stopped_after_first_sample": bool(stopped),
        "first_sample_step": first_sample_step,
        "replay": replay,
        "exact": bool(replay.get("exact")) if not stopped else False,
        "outcomes_match": False if stopped else (stop_error is None if validate_outcomes else None),
        "samples": sorted(sample_rows, key=lambda row: (int(row.get("slot_index", 0)), int(row.get("observed_step", row.get("target_step", 0))))),
        "sample_count": len(sample_rows),
        "missing_sample_count": sum(1 for row in sample_rows if str(row.get("status", "")).startswith("missing")),
        "rng_unchanged": _rng_equal(rng_before, rng_after),
        "policy_weights_unchanged": True,
        "error": stop_error,
    }


def _aggregate_scope(reports: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    missing = 0
    for report in reports:
        for row in report.get("samples", []):
            if row.get("status") != "evaluated":
                missing += 1
                continue
            key = "|".join((
                str(report.get("scenario")), str(row.get("ring_role")),
                str(row.get("soc_bucket")), str(row.get("priority")),
            ))
            groups.setdefault(key, []).append(row)
    summary = {}
    for key, rows in sorted(groups.items()):
        policy_summary = {}
        for label in ("source_03c0", "run0_episode_0080", "run1_episode_0240", "run2_episode_0240"):
            values = [row["policies"][label] for row in rows]
            def stats(field: str) -> dict[str, Any]:
                numbers = [float(item[field]) for item in values]
                return {"n": len(numbers), "min": min(numbers), "mean": sum(numbers) / len(numbers), "max": max(numbers)}
            certified_counts_off = [int(row["late_merge_off"]["certified_release_count"]) for row in rows]
            certified_counts_on = [int(row["late_merge_on"]["certified_release_count"]) for row in rows]
            catalogue_has_off = [int(bool(row["late_merge_off"]["certified_release_count"])) for row in rows]
            catalogue_has_on = [int(bool(row["late_merge_on"]["certified_release_count"])) for row in rows]
            policy_summary[label] = {
                "n": len(values),
                "q_k3_off": stats("q_k3"),
                "q_k3_on": stats("q_k3_on"),
                "p_release": stats("p_release"),
                "p_release_q_k3_off": stats("p_release_q_k3"),
                "p_release_q_k3_on": stats("p_release_q_k3_on"),
                "catalogue_certified_count_off": {
                    "n": len(certified_counts_off), "min": min(certified_counts_off),
                    "mean": sum(certified_counts_off) / len(certified_counts_off), "max": max(certified_counts_off),
                },
                "catalogue_certified_count_on": {
                    "n": len(certified_counts_on), "min": min(certified_counts_on),
                    "mean": sum(certified_counts_on) / len(certified_counts_on), "max": max(certified_counts_on),
                },
                "catalogue_has_certified_off": {
                    "n": len(catalogue_has_off), "count": sum(catalogue_has_off),
                },
                "catalogue_has_certified_on": {
                    "n": len(catalogue_has_on), "count": sum(catalogue_has_on),
                },
            }
        summary[key] = {"n": len(rows), "policies": policy_summary}
    return {"groups": summary, "evaluated_states": sum(len(rows) for rows in groups.values()), "missing_or_error_states": missing}


def _aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    source = [report for report in reports if not bool(report.get("witness"))]
    witness = [report for report in reports if bool(report.get("witness"))]
    return {"source_aggregate": _aggregate_scope(source), "witness_aggregate": _aggregate_scope(witness)}


def _completion_status(mode: str, reports: list[dict[str, Any]]) -> str:
    if mode == "smoke":
        if len(reports) != 1:
            return "FAILED"
        report = reports[0]
        sample = report.get("samples") or []
        smoke_ok = (
            report.get("error") is None
            and report.get("stopped_after_first_sample") is True
            and report.get("first_sample_step") is not None
            and report.get("exact") is True
            and report.get("rng_unchanged") is True
            and report.get("policy_weights_unchanged") is True
            and report.get("prefix_check", {}).get("exact") is True
            and len(sample) == 1
            and sample[0].get("status") == "evaluated"
        )
        return "SMOKE_COMPLETED" if smoke_ok else "FAILED"
    valid = all(
        report.get("error") is None
        and report.get("exact") is True
        and report.get("outcomes_match") is True
        and report.get("rng_unchanged") is True
        and report.get("policy_weights_unchanged") is True
        for report in reports
    )
    return "COMPLETED" if reports and valid else "FAILED"


def run(*, mode: str, formal_root: Path, output: Path) -> dict[str, Any]:
    _require(mode in {"smoke", "full"}, f"unsupported mode: {mode}")
    _require(not output.exists(), f"refusing existing output: {output}")
    if mode == "smoke":
        _require(not FULL_OUTPUT.exists(), f"full output already exists: {FULL_OUTPUT}")
    protocol, configs, records, paths = _load_inputs(formal_root)
    policies = _load_policies(protocol, configs[str(records[0]["scenario"])], paths["checkpoints"])
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "progress.json", {"stage": STAGE, "status": "RUNNING", "mode": mode})
    reports: list[dict[str, Any]] = []
    try:
        if mode == "smoke":
            first = records[0]
            tape = Path(str(first["action_tape"]))
            payload = _read_json(tape)
            config = configs[str(first["scenario"])]
            targets = _target_steps(len(payload.get("action_tape", [])))
            _write_json(output / "preflight.json", {
                "protocol_sha256": evaluator._sha256(evaluator.PROTOCOL),
                "source_manifest": str(paths["source_manifest"]),
                "source_record": {"scenario": first["scenario"], "seed": first["seed"], "action_tape": first["action_tape"]},
                "target_steps": targets,
                "checkpoints": {label: {"path": item["path"], "sha256": item["sha256"]} for label, item in policies.items()},
            })
            first_sample_step = _first_eligible_prefix_step(tape, config)
            prefix_rows = int(first_sample_step) + 1
            prefix_payload = dict(payload)
            prefix_payload["action_tape"] = payload["action_tape"][:prefix_rows]
            with tempfile.TemporaryDirectory(prefix="p3_k3_coverage_smoke_") as directory:
                prefix_path = Path(directory) / "prefix.json"
                _write_json(prefix_path, prefix_payload)
                prefix_record = dict(first)
                prefix_record["action_tape"] = str(prefix_path)
                report = _replay_tape(
                    record=prefix_record,
                    config=config,
                    policies=policies,
                    targets_override=[0],
                    validate_outcomes=False,
                )
            _require(report["first_sample_step"] == first_sample_step, "smoke first sample step changed")
            _require(report["exact"] is True, f"smoke prefix replay was not exact: {report.get('replay')}")
            report["action_tape"] = str(tape)
            report["prefix_rows"] = prefix_rows
            report["stopped_after_first_sample"] = True
            report["prefix_check"] = {
                "prefix_rows": prefix_rows,
                "json_roundtrip": True,
                "exact": True,
                "steps_expected": prefix_rows,
                "steps_replayed": prefix_rows,
            }
            report["smoke_schema_serializable"] = True
            report["smoke_25_proposal_mapping"] = (
                report["samples"][0]["late_merge_off"]["audit_count"] == 26
                and report["samples"][0]["late_merge_off"]["release_catalogue_count"] == 25
                and report["samples"][0]["late_merge_off"]["proposal_mapping_complete"] is True
                and len(report["samples"][0]["late_merge_off"]["proposal_catalogue_indices"]) == 25
                and set(report["samples"][0]["late_merge_off"]["proposal_catalogue_indices"]) == set(range(25))
            )
            _require(report["smoke_25_proposal_mapping"], "smoke proposal mapping incomplete")
            _require(report["error"] is None and report["rng_unchanged"] is True and report["policy_weights_unchanged"] is True,
                     "smoke state integrity check failed")
            reports.append(report)
        else:
            for index, record in enumerate(records):
                report = _replay_tape(record=record, config=configs[str(record["scenario"])], policies=policies)
                reports.append(report)
                _write_json(output / "records" / f"{record['scenario']}_{int(record['seed'])}.json", report)
                _write_json(output / "progress.json", {"stage": STAGE, "status": "RUNNING", "mode": mode, "processed": index + 1, "expected": len(records) + 1})
            witness_record = paths["witness"]
            witness_report = _replay_tape(record=witness_record, config=configs["dense24"], policies=policies, witness=True)
            reports.append(witness_report)
            _write_json(output / "records" / "witness_dense24_873106.json", witness_report)
        completion_status = _completion_status(mode, reports)
        _require(completion_status != "FAILED", "diagnostic integrity or replay validation failed")
        result = {
            "stage": STAGE,
            "status": completion_status,
            "mode": mode,
            "protocol_sha256": evaluator._sha256(evaluator.PROTOCOL),
            "formal_root": str(formal_root),
            "output": str(output),
            "source_tapes": len(records) if mode == "full" else 1,
            "witness_states": 6 if mode == "full" else 0,
            "reports": reports,
            "aggregate": _aggregate(reports),
            "model_bindings": {
                label: {
                    "path": item["path"],
                    "sha256": item["sha256"],
                    "model_digest_before": item["model_digest_before"],
                    "model_digest_after": evaluator._model_digest(item["policy"]),
                }
                for label, item in policies.items()
            },
            "P3_QUALIFIED": False,
            "P4_ALLOWED": False,
            "PPO_ALLOWED": False,
            "teacher_online": False,
            "policy_sampling": False,
            "policy_update": False,
            "live_action_tape_modified": False,
        }
        _write_json(output / "manifest.json", result)
        _write_json(output / "progress.json", {"stage": STAGE, "status": result["status"], "mode": mode, "reports": len(reports)})
        print(json.dumps({"stage": STAGE, "status": result["status"], "output": str(output), "reports": len(reports)}, sort_keys=True), flush=True)
        return result
    except Exception as exc:
        failure = {
            "stage": STAGE, "status": "FAILED", "mode": mode, "output": str(output),
            "reports": reports, "error": {"type": type(exc).__name__, "message": str(exc)},
            "P3_QUALIFIED": False, "P4_ALLOWED": False, "PPO_ALLOWED": False,
        }
        _write_json(output / "manifest.json", failure)
        _write_json(output / "progress.json", failure)
        print(json.dumps(failure, sort_keys=True), flush=True)
        return failure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--formal-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (SMOKE_OUTPUT if args.mode == "smoke" else FULL_OUTPUT)
    result = run(mode=args.mode, formal_root=args.formal_root, output=output)
    return 0 if result["status"] in {"SMOKE_COMPLETED", "COMPLETED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
