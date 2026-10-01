"""Run the frozen sampled-first prefix-fallback candidate arm.

The candidate keeps the native student sampler, policy, gate, actor mask, and
public K=3 request unchanged.  It records the 25-way order produced by the
same native distribution forward and installs the opt-in prefix controller on
the rollout and exact lifecycle-replay environments.  No training,
resampling, Teacher, PPO, or held-out data is used.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import run_p3_forced_wait_frozen_20260930 as frozen
from terminal_marl.training import run_p3_trained_proposal_support_20261001 as support
from terminal_marl.training import student_rollout_v4_2 as student
from terminal_marl.training import terminal_lifecycle_metrics
from terminal_marl.core.low_level_joint_action_v4_2 import hierarchical_from_normalized_v4_2
from terminal_marl.training.p3_prefix_fallback_control import (
    CONTROLLER_MAX_PROPOSALS,
    POLICY_REQUESTED_K,
    PrefixFallbackContractError,
    prefix_fallback_controller,
)
from terminal_marl.training.run_p3_forced_wait_frozen_20260930 import _CapturedTerminalEnv


STAGE = "P3_PREFIX_FALLBACK_20261001_V1"
PPO_ALLOWED = False
HELDOUT_USED = False
TEACHER_USED = False
OPTIMIZER_USED = False
OFF_POLICY = True
NO_TRAINING = True
MAX_STEPS = int(support.MAX_STEPS)
OBSERVATION_SCHEMA = support.OBSERVATION_SCHEMA
ACTION_SCHEMA = support.ACTION_SCHEMA
OUTPUT_DIR = Path(
    "/home/user/training_outputs/terminal/p3_followup_20260930/prefix_fallback_20261001"
)
BASELINE_ROOT = Path(
    "/home/user/training_outputs/terminal/p3_followup_20260930/trained_proposal_support_20261001"
)
BASELINE_FULL_SHA256 = "3f65446f49474452d31c3e1498c8019f04f4250cad3e8594c75107a123525919"
AFTER_CHECKPOINT = support.AFTER_CHECKPOINT
AFTER_CHECKPOINT_SHA256 = support.AFTER_CHECKPOINT_SHA256
PROTOCOL = Path(__file__).resolve().parents[2] / "docs/evidence/v4_2/p3_prefix_fallback_protocol_20261001.md"
HELPER_PATH = Path(__file__).with_name("p3_prefix_fallback_control.py")
ARMS = ("after_prefix_fallback",)
CANDIDATE_ARM = ARMS[0]
CELL_SPECS = tuple(dict(item) for item in support.CELL_SPECS)
REFERENCE_ARM_NAME = "after_legacy"
DEPENDENCY_FILES = {
    "student_rollout": Path(student.__file__),
    "runtime_rank": Path(rank_eval.__file__),
    "lifecycle_metrics": Path(terminal_lifecycle_metrics.__file__),
    "safety_trace": Path(__import__("terminal_marl.core.safety_trace", fromlist=["__file__"]).__file__),
    "helper": HELPER_PATH,
}


class PrefixFallbackRunnerError(ValueError):
    """Raised when a frozen prefix-fallback contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrefixFallbackRunnerError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    return value


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    _require(target.is_file(), f"JSON input missing: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrefixFallbackRunnerError(f"invalid JSON input: {target}") from exc
    _require(isinstance(value, dict), f"JSON object required: {target}")
    return dict(value)


def _write_json(path: Path, value: Mapping[str, Any], *, refuse_existing: bool = False) -> None:
    if refuse_existing and path.exists():
        raise PrefixFallbackRunnerError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_path(path: str | Path) -> Path:
    return support._sync_path(path)


def _resolve_checkpoint(path: Path, expected_sha: str) -> Path:
    resolved = _sync_path(path)
    _require(resolved.is_file(), f"checkpoint missing: {path}")
    actual = _sha256(resolved)
    _require(actual.lower() == expected_sha.lower(), f"checkpoint SHA mismatch for {path}: {actual}")
    return resolved


def _runtime_dependency_shas() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, path in DEPENDENCY_FILES.items():
        resolved = _sync_path(path)
        _require(resolved.is_file(), f"runtime dependency missing: {path}")
        result[name] = _sha256(resolved)
    return result


def _config_snapshot(config: Any) -> dict[str, Any]:
    return support._config_snapshot(config)


def _build_inputs() -> tuple[Any, dict[str, Any]]:
    config, configs = support._build_inputs()
    _require(config.observation_schema_version == OBSERVATION_SCHEMA, "observation schema changed")
    _require(config.action_schema_version == ACTION_SCHEMA, "action schema changed")
    _require(int(config.runtime_proposal_shortlist_k) == POLICY_REQUESTED_K, "runtime K changed")
    _require(bool(config.teacher_enabled) is False, "Teacher must be disabled")
    _require(str(config.scheduler_mode) == "execution", "scheduler mode changed")
    _require(float(config.max_simulation_seconds) == 1800.0, "simulation cap changed")
    _require(bool(getattr(config, "rule_flow_coordination_enabled", False)) is False,
             "rule-flow coordination must be disabled")
    return config, configs


def _candidate_binding(config: Any) -> dict[str, Any]:
    return support._policy_binding(AFTER_CHECKPOINT, AFTER_CHECKPOINT_SHA256, config, "after_candidate")


def _historical_checkpoint_binding() -> dict[str, Any]:
    """Bind the historical checkpoints without loading them into the candidate arm."""

    before = _resolve_checkpoint(support.BEFORE_CHECKPOINT, support.BEFORE_CHECKPOINT_SHA256)
    warmup = _resolve_checkpoint(support.WARMUP_CHECKPOINT, support.WARMUP_CHECKPOINT_SHA256)
    return {
        "before": {
            "declared": str(support.BEFORE_CHECKPOINT),
            "resolved": str(before), "sha256": _sha256(before),
            "role": "historical_input_binding_only",
        },
        "warmup": {
            "declared": str(support.WARMUP_CHECKPOINT),
            "resolved": str(warmup), "sha256": _sha256(warmup),
            "role": "historical_input_binding_only",
        },
    }


def _validate_baseline_arm(
    cell: Mapping[str, Any], arm: Mapping[str, Any], expected_summary: Mapping[str, Any],
) -> None:
    """Validate one after-legacy arm against its immutable full-summary row."""

    cell_id = int(cell["cell_id"])
    recomputed_metrics = support._record_metrics(arm)
    _require(arm.get("status") == "completed" and
             recomputed_metrics.get("censored") is False and
             recomputed_metrics.get("complete_scene_metrics_known") is True,
             f"baseline arm {cell_id} is not natural")
    _require(arm.get("cell") == dict(cell), f"baseline cell {cell_id} identity changed")
    _require(int(arm.get("seed", -1)) == int(cell["env_seed"]),
             f"baseline cell {cell_id} environment seed changed")
    _require(int(arm.get("policy_seed", -1)) == int(cell["policy_seed"]),
             f"baseline cell {cell_id} policy seed changed")
    _require(arm.get("model_unchanged") is True and arm.get("caller_rng_unchanged") is True,
             f"baseline arm {cell_id} model/RNG binding changed")
    _require(arm.get("safety_nonzero") is False, f"baseline arm {cell_id} safety binding changed")
    _require(arm.get("checkpoint_sha256") == AFTER_CHECKPOINT_SHA256,
             f"baseline arm {cell_id} checkpoint binding changed")
    expected_metrics = expected_summary.get("arms", {}).get(REFERENCE_ARM_NAME, {}).get("metrics")
    _require(isinstance(expected_metrics, Mapping), f"baseline full metrics missing for cell {cell_id}")
    _require(recomputed_metrics == expected_metrics,
             f"baseline arm {cell_id} metrics differ from immutable full")
    _require(arm.get("metrics") == expected_metrics,
             f"baseline arm {cell_id} stored metrics differ from immutable full")


def _baseline_guard() -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    root = _sync_path(BASELINE_ROOT)
    full_path = root / "full.json"
    _require(full_path.is_file(), f"baseline full artifact missing: {full_path}")
    full_sha = _sha256(full_path)
    _require(full_sha.lower() == BASELINE_FULL_SHA256.lower(), "baseline full artifact SHA changed")
    full = _read_json(full_path)
    _require(full.get("status") == "FULL_OK", "baseline full artifact is not FULL_OK")
    _require(int(full.get("pair_count", -1)) == len(CELL_SPECS), "baseline cell count changed")
    summaries = {
        int(item.get("cell", {}).get("cell_id", -1)): item
        for item in full.get("pair_summaries", ())
        if isinstance(item, Mapping)
    }
    _require(set(summaries) == {int(item["cell_id"]) for item in CELL_SPECS},
             "baseline cell grid changed")
    arms: dict[int, dict[str, Any]] = {}
    metadata: dict[str, Any] = {
        "root": str(root), "full_path": str(full_path), "full_sha256": full_sha,
        "pair_count": len(CELL_SPECS), "cells": [dict(item) for item in CELL_SPECS],
        "arms": {},
    }
    for cell in CELL_SPECS:
        cell_id = int(cell["cell_id"])
        path = root / "arms" / f"cell_{cell_id:02d}_{REFERENCE_ARM_NAME}.json"
        arm = _read_json(path)
        expected_summary = summaries[cell_id]
        _validate_baseline_arm(cell, arm, expected_summary)
        arms[cell_id] = arm
        metadata["arms"][str(cell_id)] = {
            "path": str(path), "sha256": _sha256(path),
            "env_seed": int(cell["env_seed"]), "policy_seed": int(cell["policy_seed"]),
            "metrics": arm.get("metrics"),
        }
    return metadata, arms


def _build_preflight() -> dict[str, Any]:
    config, _ = _build_inputs()
    _require(PROTOCOL.is_file(), f"protocol document missing: {PROTOCOL}")
    baseline, _arms = _baseline_guard()
    baseline_preflight_path = _sync_path(BASELINE_ROOT / "preflight.json")
    baseline_preflight = _read_json(baseline_preflight_path)
    candidate = _candidate_binding(config)
    historical = _historical_checkpoint_binding()
    dependencies = _runtime_dependency_shas()
    config_snapshot = _config_snapshot(config)
    base_snapshot = baseline_preflight.get("runtime", {}).get("config_snapshot")
    _require(base_snapshot == config_snapshot, "current config differs from immutable baseline preflight")
    return {
        "stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK",
        "PPO_ALLOWED": PPO_ALLOWED, "heldout_used": HELDOUT_USED,
        "teacher_used": TEACHER_USED, "optimizer_used": OPTIMIZER_USED,
        "off_policy": OFF_POLICY, "no_training": NO_TRAINING,
        "candidate": {
            "arm": CANDIDATE_ARM, "checkpoint": candidate,
            "public_shortlist_k": POLICY_REQUESTED_K,
            "controller_max_proposals": CONTROLLER_MAX_PROPOSALS,
            "sampled_first": True, "source_logits_stable_rank": True,
            "native_forward_capture": "same_model.distributions_call; no_second_forward",
        },
        "historical_checkpoints": historical,
        "schema": {"observation": OBSERVATION_SCHEMA, "observation_dim": 88, "action": ACTION_SCHEMA},
        "runtime": {
            "config_snapshot": config_snapshot,
            "max_steps": MAX_STEPS,
            "native_rollout": "evaluate_p3_runtime_rank_repair._runtime_record",
            "deterministic": False,
            "replay": "terminal_lifecycle_metrics.terminal_lifecycle_metrics",
        },
        "cells": [dict(item) for item in CELL_SPECS], "requests_per_cell": 24,
        "baseline": baseline,
        "baseline_preflight_sha256": _sha256(baseline_preflight_path),
        "original_base_preflight_config_equal": True,
        "protocol_sha256": _sha256(PROTOCOL),
        "helper_sha256": _sha256(HELPER_PATH),
        "runtime_dependency_shas": dependencies,
        "runner_sha256": _sha256(__file__),
        "qualification": "not_a_qualification; fixed sampled-first prefix-fallback development candidate",
    }


def _validate_preflight(payload: Mapping[str, Any]) -> None:
    _require(payload.get("stage") == STAGE and payload.get("status") == "PREFLIGHT_OK",
             "prefix-fallback preflight is not successful")
    for key, expected in (("PPO_ALLOWED", False), ("heldout_used", False),
                          ("teacher_used", False), ("optimizer_used", False),
                          ("off_policy", True), ("no_training", True)):
        _require(payload.get(key) is expected, f"preflight flag {key} changed")
    _require(payload.get("runner_sha256") == _sha256(__file__), "runner changed after preflight")
    _require(payload.get("helper_sha256") == _sha256(HELPER_PATH), "helper changed after preflight")
    _require(payload.get("protocol_sha256") == _sha256(PROTOCOL), "protocol changed after preflight")
    _require(payload.get("runtime_dependency_shas") == _runtime_dependency_shas(),
             "runtime dependency changed after preflight")
    config, _ = _build_inputs()
    runtime = payload.get("runtime", {})
    _require(runtime.get("config_snapshot") == _config_snapshot(config), "config changed after preflight")
    _require(int(runtime.get("max_steps", -1)) == MAX_STEPS, "max_steps changed after preflight")
    _require(payload.get("cells") == [dict(item) for item in CELL_SPECS], "cell grid changed after preflight")
    _require(payload.get("schema", {}).get("observation_dim") == 88, "observation dimension changed")
    _require(payload.get("candidate", {}).get("public_shortlist_k") == POLICY_REQUESTED_K,
             "public shortlist K changed")
    _require(payload.get("candidate", {}).get("controller_max_proposals") == CONTROLLER_MAX_PROPOSALS,
             "controller proposal bound changed")
    baseline, _ = _baseline_guard()
    _require(payload.get("baseline") == baseline, "immutable baseline binding changed")
    base_path = _sync_path(BASELINE_ROOT / "preflight.json")
    _require(payload.get("baseline_preflight_sha256") == _sha256(base_path), "baseline preflight changed")
    base_preflight = _read_json(base_path)
    _require(base_preflight.get("runtime", {}).get("config_snapshot") == _config_snapshot(config),
             "original baseline preflight config changed")
    candidate = _candidate_binding(config)
    stored = payload.get("candidate", {}).get("checkpoint")
    _require(stored == candidate, "candidate checkpoint/model binding changed")
    _require(payload.get("historical_checkpoints") == _historical_checkpoint_binding(),
             "historical checkpoint binding changed")
    _require(payload.get("original_base_preflight_config_equal") is True,
             "original baseline config equality binding changed")


def _phase_inputs(output: Path) -> tuple[dict[str, Any], Any, dict[int, dict[str, Any]]]:
    preflight = _read_json(output / "preflight.json")
    _validate_preflight(preflight)
    config, _ = _build_inputs()
    _baseline, arms = _baseline_guard()
    return preflight, config, arms


def _clone_distribution(dist: Any) -> dict[str, torch.Tensor]:
    try:
        logits = dist.logits.detach().clone()
        probs = dist.probs.detach().clone()
    except Exception as exc:
        raise PrefixFallbackRunnerError("native distributions lack cloneable logits/probs") from exc
    _require(torch.isfinite(logits).all().item() and torch.isfinite(probs).all().item(),
             "native distributions are non-finite")
    return {"logits": logits, "probs": probs}


def _install_distribution_capture(model: Any, capture: Any):
    original = getattr(model, "distributions")
    had_instance = "distributions" in getattr(model, "__dict__", {})
    previous = getattr(model, "__dict__", {}).get("distributions")

    def wrapped(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        _require(isinstance(result, tuple) and len(result) == 2,
                 "native distributions must return gate/proposal pair")
        gate_dist, proposal_dist = result
        capture.append({
            "gate": _clone_distribution(gate_dist),
            "proposal": _clone_distribution(proposal_dist),
            "gate_ref": gate_dist,
            "proposal_ref": proposal_dist,
        })
        return result

    try:
        object.__setattr__(model, "distributions", wrapped)
    except Exception:
        setattr(model, "distributions", wrapped)

    def restore() -> None:
        if had_instance:
            object.__setattr__(model, "distributions", previous)
        else:
            try:
                object.__delattr__(model, "distributions")
            except AttributeError:
                pass

    return restore


def _register_forward(audit: dict[str, Any], env: Any, result: Any, forward: Mapping[str, Any]) -> None:
    _require(isinstance(result, tuple) and len(result) == 10, "native request-details result shape changed")
    uids, _world, actor_mask, _actions, gates, proposals, _gate_lp, _proposal_lp, _likelihoods, shortlists = result
    gate_logits = forward["gate"]["logits"]
    proposal_logits = forward["proposal"]["logits"]
    _require(torch.equal(forward["gate_ref"].logits, gate_logits), "native gate logits changed after forward")
    _require(torch.equal(forward["gate_ref"].probs, forward["gate"]["probs"]),
             "native gate probabilities changed after forward")
    _require(torch.equal(forward["proposal_ref"].logits, proposal_logits),
             "native proposal logits changed after forward")
    _require(torch.equal(forward["proposal_ref"].probs, forward["proposal"]["probs"]),
             "native proposal probabilities changed after forward")
    audit["source_distributions_immutable"] = True
    _require(len(uids) == int(gate_logits.shape[0]) == int(proposal_logits.shape[0]),
             "native distribution row count changed")
    step = int(getattr(env, "step_count", 0))
    now = float(getattr(env, "now", step))
    forward_id = f"{audit['arm_id']}:forward:{len(audit['forwards']):06d}"
    for index, uid_raw in enumerate(uids):
        uid = int(uid_raw)
        if not bool(actor_mask[index]) or int(gates[index]) != 1:
            continue
        sampled = int(proposals[index])
        _require(0 <= sampled < CONTROLLER_MAX_PROPOSALS, "native sampled proposal is outside catalogue")
        logits = proposal_logits[index].detach().cpu()
        ranked = torch.argsort(logits, descending=True, stable=True).tolist()
        order = (sampled,) + tuple(int(value) for value in ranked if int(value) != sampled)
        _require(len(order) == CONTROLLER_MAX_PROPOSALS and set(order) == set(range(CONTROLLER_MAX_PROPOSALS)),
                 "native stable proposal order is not a complete 25-way order")
        shortlist = shortlists.get(uid, shortlists.get(str(uid), ())) if isinstance(shortlists, Mapping) else ()
        _require(len(shortlist) == POLICY_REQUESTED_K, "native RELEASE shortlist is not K=3")
        decoded_prefix_values = []
        for item in shortlist:
            try:
                decoded = hierarchical_from_normalized_v4_2(np.asarray(item, dtype=np.float32))
            except (TypeError, ValueError) as exc:
                raise PrefixFallbackRunnerError("native K=3 shortlist contains an invalid action") from exc
            _require(decoded.gate == 1 and decoded.proposal is not None,
                     "native RELEASE shortlist contains a non-RELEASE action")
            decoded_prefix_values.append(int(decoded.proposal))
        decoded_prefix = tuple(decoded_prefix_values)
        _require(decoded_prefix == order[:POLICY_REQUESTED_K],
                 "native K=3 shortlist is not the sampled-first stable prefix")
        key = f"{step}|{uid}"
        _require(key not in audit["registrations"], f"duplicate native registration {key}")
        audit["registrations"][key] = {
            "step": step, "env_now": now, "uid": uid,
            "order": list(order), "prefix": list(decoded_prefix),
            "metadata": {
                "sampled_proposal": sampled,
                "source_logits": [float(value) for value in logits.tolist()],
                "requested_k": POLICY_REQUESTED_K,
                "source_forward_id": forward_id,
            },
            "forward_id": forward_id,
        }
    progress_path = audit.get("progress_path")
    if progress_path is not None and step % 50 == 0:
        captured_rows = getattr(env, "_captured_request_rows", None)
        generated_request_rows = len(captured_rows) if isinstance(captured_rows, list) else None
        _write_json(Path(progress_path), {
            "stage": STAGE, "phase": "arm", "status": "RUNNING",
            "cell_id": audit.get("cell_id"), "arm": CANDIDATE_ARM,
            "absolute_step": step, "env_now": now,
            "active_observation_count": len(uids),
            "generated_request_rows": generated_request_rows,
            "native_forward_count": len(audit.get("forwards", [])),
        })
    audit["forwards"][-1]["registered_count"] = sum(
        1 for row in audit["registrations"].values() if row.get("forward_id") == forward_id
    )


@contextlib.contextmanager
def _prefix_sampler(policy: Any, audit: dict[str, Any]) -> Iterator[None]:
    """Capture the one native distribution forward without changing its output."""

    original_details = student._student_request_details
    forwards: list[dict[str, Any]] = []
    audit.setdefault("forwards", forwards)
    audit.setdefault("registrations", {})
    restore_distribution = _install_distribution_capture(policy.model, forwards)

    def wrapped_details(policy_arg: Any, env: Any, observations: dict[str, Any], *, deterministic: bool, shortlist_k: int):
        _require(policy_arg is policy, "native request-details policy alias changed")
        before = len(forwards)
        try:
            result = original_details(
                policy_arg, env, observations, deterministic=deterministic, shortlist_k=shortlist_k,
            )
        except BaseException:
            audit["source_distributions_immutable"] = False
            raise
        after = len(forwards)
        if after == before:
            _require(not observations, "native nonempty request-details must use exactly one distribution forward")
            return result
        _require(after - before == 1, "native request-details used more than one distribution forward")
        try:
            _register_forward(audit, env, result, forwards[-1])
        except BaseException:
            audit["source_distributions_immutable"] = False
            raise
        return result

    original_student_details = student._student_request_details
    student._student_request_details = wrapped_details
    try:
        yield
    finally:
        student._student_request_details = original_student_details
        restore_distribution()
        audit["sampler_aliases_restored"] = True


def _registration_lookup(audit: Mapping[str, Any], env: Any, uid: int) -> Mapping[str, Any]:
    step = int(getattr(env, "step_count", -1))
    direct = audit.get("registrations", {}).get(f"{step}|{int(uid)}")
    now = float(getattr(env, "now", float("nan")))
    if isinstance(direct, Mapping) and math.isfinite(now) and math.isclose(
        float(direct.get("env_now", float("nan"))), now, rel_tol=0.0, abs_tol=1e-9,
    ):
        return direct
    candidates = [
        row for row in audit.get("registrations", {}).values()
        if int(row.get("uid", -1)) == int(uid)
        and math.isfinite(now)
        and math.isclose(float(row.get("env_now")), now, rel_tol=0.0, abs_tol=1e-9)
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise PrefixFallbackContractError(f"missing unique saved proposal order for step={step}, uid={uid}")


@contextlib.contextmanager
def _prefix_runtime_scope(audit: dict[str, Any], capture: dict[str, Any]) -> Iterator[None]:
    """Attach the controller to rollout and lifecycle-replay env instances."""

    original_student_factory = student.TerminalParallelEnv
    original_lifecycle_factory = terminal_lifecycle_metrics.TerminalParallelEnv
    contexts: list[Any] = []
    audit.setdefault("controllers", [])
    audit["runtime_aliases_restored"] = False
    capture.setdefault("rollout_factory_calls", 0)
    capture.setdefault("replay_factory_calls", 0)

    def order_provider(env: Any, _uav: Any, uid: int):
        return _registration_lookup(audit, env, uid)["order"]

    def metadata_provider(env: Any, _uav: Any, uid: int):
        return _registration_lookup(audit, env, uid)["metadata"]

    def make_env(kind: str, *args: Any, **kwargs: Any) -> Any:
        env = _CapturedTerminalEnv(*args, **kwargs)
        if kind == "rollout":
            capture["rollout_factory_calls"] = int(capture.get("rollout_factory_calls", 0)) + 1
            capture["rollout_env"] = env
        else:
            capture["replay_factory_calls"] = int(capture.get("replay_factory_calls", 0)) + 1
            capture["replay_env"] = env
        context = prefix_fallback_controller(
            env, order_provider, source_metadata=metadata_provider,
        )
        controller = context.__enter__()
        contexts.append((context, controller))
        audit["controllers"].append(controller)
        return env

    def rollout_factory(*args: Any, **kwargs: Any) -> Any:
        return make_env("rollout", *args, **kwargs)

    def replay_factory(*args: Any, **kwargs: Any) -> Any:
        return make_env("replay", *args, **kwargs)

    student.TerminalParallelEnv = rollout_factory
    terminal_lifecycle_metrics.TerminalParallelEnv = replay_factory
    try:
        yield
    finally:
        try:
            for context, _controller in reversed(contexts):
                context.__exit__(None, None, None)
        finally:
            student.TerminalParallelEnv = original_student_factory
            terminal_lifecycle_metrics.TerminalParallelEnv = original_lifecycle_factory
            audit["runtime_aliases_restored"] = True


def _sidecar_payload(audit: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": STAGE, "cell": dict(cell), "arm": CANDIDATE_ARM,
        "keying": "absolute_step|uid with env_now witness",
        "registrations": {str(key): dict(value) for key, value in audit.get("registrations", {}).items()},
        "registration_count": len(audit.get("registrations", {})),
    }


def _binding_checks(
    record: Mapping[str, Any], *, config_before: Any, config_after: Any,
    dependencies_before: Any, dependencies_after: Any,
) -> dict[str, bool]:
    """Return the immutable arm checks, including before/after runtime bindings."""

    runtime = record.get("runtime_capture", {})
    control = record.get("prefix_fallback_control", {})
    return {
        "checkpoint_sha256": record.get("checkpoint_sha256") == AFTER_CHECKPOINT_SHA256,
        "model_unchanged": record.get("model_unchanged") is True,
        "caller_rng_unchanged": record.get("caller_rng_unchanged") is True,
        "observer_rng_unchanged": record.get("observer_rng_unchanged") is True,
        "source_distributions_immutable": control.get("source_distributions_immutable") is True,
        "sampler_aliases_restored": control.get("sampler_aliases_restored") is True,
        "runtime_aliases_restored": control.get("runtime_aliases_restored") is True,
        "rollout_factory_calls": runtime.get("factory_calls") == 1,
        "replay_factory_calls": runtime.get("replay_factory_calls") == 1,
        "config_snapshot_unchanged": config_before is not None and config_after is not None
        and config_before == config_after,
        "runtime_dependency_shas_unchanged": dependencies_before is not None
        and dependencies_after is not None and dependencies_before == dependencies_after,
    }


def _arm_record(*, config: Any, cell: Mapping[str, Any], output: Path) -> dict[str, Any]:
    caller_before = support._rng_snapshot()
    resolved = AFTER_CHECKPOINT
    policy = None
    before_digest = None
    audit: dict[str, Any] = {
        "arm_id": f"cell_{int(cell['cell_id']):02d}", "cell_id": int(cell["cell_id"]),
        "registrations": {}, "source_distributions_immutable": True,
        "progress_path": output / "progress.json",
    }
    capture: dict[str, Any] = {}
    record_output = output / "rollouts" / f"cell_{int(cell['cell_id']):02d}_{CANDIDATE_ARM}"
    wall_start = time.monotonic()
    try:
        config_before = _jsonable(_config_snapshot(config))
    except BaseException:
        config_before = None
    try:
        dependencies_before = dict(_runtime_dependency_shas())
    except BaseException:
        dependencies_before = None
    try:
        resolved = _resolve_checkpoint(AFTER_CHECKPOINT, AFTER_CHECKPOINT_SHA256)
        policy, _payload = rank_eval._load_policy(resolved, config)
        before_digest = rank_eval._model_digest(policy)
        with _prefix_runtime_scope(audit, capture):
            with _prefix_sampler(policy, audit):
                record = rank_eval._runtime_record(
                    policy=policy, checkpoint=resolved, config=config,
                    scenario="dense24", seed=int(cell["env_seed"]),
                    policy_seed=int(cell["policy_seed"]), output=record_output, smoke=False,
                )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        record = rank_eval._failure_record(
            scenario="dense24", seed=int(cell["env_seed"]), checkpoint=resolved,
            requests=24, error=exc, weights_before=before_digest,
            weights_unchanged=(
                None if policy is None or before_digest is None
                else before_digest == rank_eval._model_digest(policy)
            ), policy_seed=int(cell["policy_seed"]),
        )
    finally:
        support._restore_rng(caller_before)
    record = support._reclassify_incomplete_lifecycle(record)
    after_digest = None if policy is None else rank_eval._model_digest(policy)
    runtime_input = {
        "factory_calls": int(capture.get("rollout_factory_calls", 0)),
        "env": capture.get("rollout_env"),
    }
    try:
        runtime_snapshot = support._live_runtime_snapshot(runtime_input, config)
    except BaseException as exc:
        runtime_snapshot = {
            "available": False,
            "factory_calls": int(capture.get("rollout_factory_calls", 0)),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    runtime_snapshot["replay_factory_calls"] = int(capture.get("replay_factory_calls", 0))
    try:
        config_after = _jsonable(_config_snapshot(config))
    except BaseException:
        config_after = None
    try:
        dependencies_after = dict(_runtime_dependency_shas())
    except BaseException:
        dependencies_after = None
    record.update({
        "cell": dict(cell), "arm": CANDIDATE_ARM,
        "checkpoint_sha256": None if not resolved or not Path(resolved).is_file() else _sha256(resolved),
        "checkpoint_sha256_expected": AFTER_CHECKPOINT_SHA256,
        "model_digest_before": before_digest,
        "model_digest_after": after_digest,
        "model_unchanged": None if before_digest is None or after_digest is None else before_digest == after_digest,
        "weights_unchanged": None if before_digest is None or after_digest is None else before_digest == after_digest,
        "runtime_capture": runtime_snapshot,
        "prefix_fallback_control": {
            "name": "sampled_first_prefix_fallback",
            "policy_requested_k": POLICY_REQUESTED_K,
            "controller_max": CONTROLLER_MAX_PROPOSALS,
            "native_forward_count": len(audit.get("forwards", [])),
            "registration_count": len(audit.get("registrations", {})),
            "public_k3_prefix_validated": True,
            "no_second_forward": True,
            "no_second_sample": True,
            "controller_metadata": [controller.metadata() for controller in audit.get("controllers", [])],
            "source_distributions_immutable": bool(audit.get("source_distributions_immutable")),
            "sampler_aliases_restored": bool(audit.get("sampler_aliases_restored")),
            "runtime_aliases_restored": bool(audit.get("runtime_aliases_restored")),
        },
        "observer_rng_used": False,
        "observer_rng_unchanged": True,
        "wall_time_seconds": float(time.monotonic() - wall_start),
        "runtime_dependency_shas_before": dependencies_before,
        "runtime_dependency_shas": dependencies_after,
        "config_snapshot_before": config_before,
        "config_snapshot": config_after,
        "policy_seed": int(cell["policy_seed"]),
        "caller_rng_unchanged": support._rng_equal(caller_before, support._rng_snapshot()),
        "teacher_called": False, "training_or_update_called": False, "PPO_ALLOWED": False,
    })
    sidecar_path = output / "sidecars" / f"cell_{int(cell['cell_id']):02d}_{CANDIDATE_ARM}_orders.json"
    sidecar = _sidecar_payload(audit, cell)
    _write_json(sidecar_path, sidecar, refuse_existing=True)
    record["prefix_fallback_control"]["orders_sidecar"] = str(sidecar_path)
    record["prefix_fallback_control"]["orders_sidecar_sha256"] = _sha256(sidecar_path)
    record["prefix_fallback_control"]["registration_count"] = int(sidecar["registration_count"])
    try:
        record["metrics"] = support._record_metrics(record)
    except BaseException as exc:
        record["metrics"] = {"technical_ok": False, "complete_scene_metrics_known": False,
                              "status": record.get("status"),
                              "error": {"type": type(exc).__name__, "message": str(exc)}}
    binding_checks = _binding_checks(
        record, config_before=config_before, config_after=config_after,
        dependencies_before=dependencies_before, dependencies_after=dependencies_after,
    )
    record["binding_checks"] = binding_checks
    record["metrics"]["binding_checks"] = dict(binding_checks)
    record["metrics"]["technical_ok"] = bool(
        record["metrics"].get("technical_ok") is True and all(binding_checks.values())
    )
    return record


def _issue_ids(record: Mapping[str, Any], kinds: set[str]) -> list[int] | None:
    return support._issue_ids(record, kinds)


def _pair_summary(cell: Mapping[str, Any], candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    candidate_metrics = candidate.get("metrics") or support._record_metrics(candidate)
    reference_metrics = reference.get("metrics") or support._record_metrics(reference)
    known = bool(candidate_metrics.get("complete_scene_metrics_known") and reference_metrics.get("complete_scene_metrics_known"))
    issue_keys = (("failed", {"failed_energy", "energy_depleted", "failed"}),
                  ("diverted", {"diverted"}), ("unresolved", {"unresolved"}))
    new_ids: dict[str, list[int] | None] = {}
    for name, kinds in issue_keys:
        candidate_ids = _issue_ids(candidate, kinds)
        reference_ids = _issue_ids(reference, kinds)
        new_ids[name] = None if not known or candidate_ids is None or reference_ids is None else sorted(set(candidate_ids) - set(reference_ids))
    values: dict[str, Any] = {}
    for key in ("normal", "successful", "failed", "wait_seconds", "request_elapsed_total_seconds", "energy_total_consumed"):
        left, right = candidate_metrics.get(key), reference_metrics.get(key)
        values[key] = None if not known or left is None or right is None else float(left) - float(right)
    status = str(candidate.get("status"))
    pair_status = "PAIR_TECHNICAL_ERROR" if status in {"failed", "technical_error", "safety_violation"} or candidate.get("error") else (
        "PAIR_CENSORED" if candidate_metrics.get("censored") else "PAIR_OK"
    )
    return {
        "stage": STAGE, "cell": dict(cell), "status": pair_status,
        "candidate_metrics": candidate_metrics, "reference_metrics": reference_metrics,
        "request_identity_match": candidate_metrics.get("request_identity") == reference_metrics.get("request_identity"),
        "request_identity_exact_24": candidate_metrics.get("request_identity_exact_24") is True,
        "technical_ok": candidate_metrics.get("technical_ok") is True,
        "hard_safety_clear": candidate_metrics.get("hard_safety_clear") is True,
        "model_unchanged": candidate.get("model_unchanged"),
        "caller_rng_unchanged": candidate.get("caller_rng_unchanged"),
        "observer_rng_unchanged": candidate.get("observer_rng_unchanged"),
        "sampler_aliases_restored": candidate.get("prefix_fallback_control", {}).get("sampler_aliases_restored"),
        "runtime_aliases_restored": candidate.get("prefix_fallback_control", {}).get("runtime_aliases_restored"),
        "rollout_factory_calls": candidate.get("runtime_capture", {}).get("factory_calls"),
        "replay_factory_calls": candidate.get("runtime_capture", {}).get("replay_factory_calls"),
        "new_failed_arrival_ids": new_ids["failed"],
        "new_diverted_arrival_ids": new_ids["diverted"],
        "new_unresolved_arrival_ids": new_ids["unresolved"],
        "candidate_minus_after_legacy": {"known": known, "values": values},
        "costs": {
            key: {"candidate": candidate_metrics.get(key), "after_legacy": reference_metrics.get(key), "delta": values.get(key)}
            for key in ("wait_seconds", "request_elapsed_total_seconds", "energy_total_consumed")
        },
        "arm_path": None,
    }


def _run_cell(output: Path, config: Any, cell: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    _write_json(output / "progress.json", {
        "stage": STAGE, "phase": "arm", "status": "RUNNING", "cell_id": int(cell["cell_id"]),
        "arm": CANDIDATE_ARM,
    })
    try:
        candidate = _arm_record(config=config, cell=cell, output=output)
    except BaseException as exc:
        candidate = {
            "cell": dict(cell), "arm": CANDIDATE_ARM, "status": "technical_error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "metrics": {"technical_ok": False, "complete_scene_metrics_known": False,
                         "status": "technical_error"},
        }
    arm_path = output / "arms" / f"cell_{int(cell['cell_id']):02d}_{CANDIDATE_ARM}.json"
    _write_json(arm_path, candidate, refuse_existing=True)
    summary = _pair_summary(cell, candidate, reference)
    summary["arm_path"] = str(arm_path)
    _write_json(output / "pairs" / f"cell_{int(cell['cell_id']):02d}.json", summary, refuse_existing=True)
    _write_json(output / "progress.json", {
        "stage": STAGE, "phase": "arm", "status": summary["status"],
        "cell_id": int(cell["cell_id"]), "arm": CANDIDATE_ARM,
        "technical_ok": summary.get("technical_ok"),
    })
    return summary


def _aggregate(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [row.get("candidate_metrics", {}).get(key) for row in rows]
    if not values or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) for value in values):
        return None
    return float(sum(float(value) for value in values))


def _business_review(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(summaries)
    natural = len(rows) == len(CELL_SPECS) and all(
        row.get("status") == "PAIR_OK" and row.get("candidate_metrics", {}).get("complete_scene_metrics_known") is True
        for row in rows
    )
    technical = len(rows) == len(CELL_SPECS) and all(
        row.get("technical_ok") is True and row.get("hard_safety_clear") is True
        for row in rows
    )
    exact = len(rows) == len(CELL_SPECS) and all(
        row.get("request_identity_match") is True and row.get("request_identity_exact_24") is True
        for row in rows
    )
    no_new = len(rows) == len(CELL_SPECS) and all(
        isinstance(row.get(key), list) and not row.get(key)
        for row in rows for key in ("new_failed_arrival_ids", "new_diverted_arrival_ids", "new_unresolved_arrival_ids")
    )
    candidate = {key: _aggregate(rows, key) if natural else None for key in (
        "normal", "successful", "failed", "wait_seconds", "request_elapsed_total_seconds", "energy_total_consumed",
    )}
    reference = {}
    for key in candidate:
        values = [row.get("reference_metrics", {}).get(key) for row in rows]
        reference[key] = float(sum(float(value) for value in values)) if natural and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) for value in values
        ) else None
    normal_ok = candidate["normal"] is not None and reference["normal"] is not None and candidate["normal"] >= reference["normal"]
    successful_ok = candidate["successful"] is not None and reference["successful"] is not None and candidate["successful"] >= reference["successful"]
    costs_ok = all(
        candidate[key] is not None and reference[key] is not None and candidate[key] <= reference[key] + 1e-8
        for key in ("wait_seconds", "request_elapsed_total_seconds", "energy_total_consumed")
    )
    strict_gain = bool(
        candidate["normal"] is not None and reference["normal"] is not None and candidate["normal"] > reference["normal"]
        or candidate["failed"] is not None and reference["failed"] is not None and candidate["failed"] < reference["failed"]
    )
    criterion = bool(natural and technical and exact and no_new and normal_ok and successful_ok and costs_ok and strict_gain)
    return {
        "pair_count": len(rows), "all_four_natural": natural, "all_technical_ok": technical,
        "all_request_identity_exact_24": exact, "no_new_failed_diverted_unresolved": no_new,
        "candidate_aggregate": candidate, "after_legacy_aggregate": reference,
        "normal_not_lower": normal_ok, "successful_not_lower": successful_ok,
        "wait_elapsed_energy_not_higher": costs_ok, "strict_business_gain": strict_gain,
        "criterion_pass": criterion, "development_candidate": criterion,
        "PPO_ALLOWED": False, "qualification": "not_a_qualification",
    }


def _technical_error(summary: Mapping[str, Any]) -> str | None:
    if summary.get("status") == "PAIR_TECHNICAL_ERROR":
        return str(summary.get("candidate_metrics", {}).get("error") or "candidate technical error")
    if summary.get("technical_ok") is not True:
        return "candidate technical checks are incomplete"
    if summary.get("hard_safety_clear") is not True:
        return "candidate hard-safety check is not clear"
    for key in (
        "model_unchanged", "caller_rng_unchanged", "observer_rng_unchanged",
        "sampler_aliases_restored", "runtime_aliases_restored",
    ):
        if summary.get(key) is not True:
            return f"candidate binding check {key} is not true"
    if summary.get("rollout_factory_calls") != 1 or summary.get("replay_factory_calls") != 1:
        return "candidate environment factory counts are incomplete"
    return None


def run_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    _require(not (output / "preflight.json").exists(), "preflight artifact already exists")
    payload = _build_preflight()
    _write_json(output / "preflight.json", payload, refuse_existing=True)
    _write_json(output / "progress.json", {
        "stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK", "environment_created": False,
    })
    return payload


def run_pilot(output: Path) -> dict[str, Any]:
    preflight, config, references = _phase_inputs(output)
    _require(not (output / "pilot.json").exists(), "pilot artifact already exists")
    summary = _run_cell(output, config, CELL_SPECS[0], references[1])
    technical = _technical_error(summary)
    payload = {
        "stage": STAGE, "phase": "pilot",
        "status": "PILOT_TECHNICAL_ERROR" if technical else ("PILOT_OK" if summary["status"] == "PAIR_OK" else "PILOT_INCOMPLETE"),
        "PPO_ALLOWED": False, "preflight_sha256": _sha256(output / "preflight.json"),
        "preflight_runner_sha256": preflight["runner_sha256"], "pair": summary,
        "pilot_artifacts": {
            "pair_path": str(output / "pairs" / "cell_01.json"),
            "pair_sha256": _sha256(output / "pairs" / "cell_01.json"),
            "arm_path": str(output / "arms" / f"cell_01_{CANDIDATE_ARM}.json"),
            "arm_sha256": _sha256(output / "arms" / f"cell_01_{CANDIDATE_ARM}.json"),
        },
        "technical_error": technical, "qualification": "not_a_qualification",
    }
    _write_json(output / "pilot.json", payload, refuse_existing=True)
    _write_json(output / "progress.json", payload)
    if technical:
        raise PrefixFallbackRunnerError(f"pilot stopped after technical failure: {technical}")
    return payload


def _validate_pilot_reuse(output: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    pilot = _read_json(output / "pilot.json")
    _require(pilot.get("status") in {"PILOT_OK", "PILOT_INCOMPLETE"}, "pilot is not reusable")
    _require(pilot.get("preflight_sha256") == _sha256(output / "preflight.json"), "pilot preflight changed")
    _require(pilot.get("preflight_runner_sha256") == preflight.get("runner_sha256"), "pilot runner changed")
    pair_path = Path(str(pilot.get("pilot_artifacts", {}).get("pair_path", "")))
    arm_path = Path(str(pilot.get("pilot_artifacts", {}).get("arm_path", "")))
    _require(pair_path.is_file() and arm_path.is_file(), "pilot artifacts missing")
    _require(pilot["pilot_artifacts"]["pair_sha256"] == _sha256(pair_path), "pilot pair changed")
    _require(pilot["pilot_artifacts"]["arm_sha256"] == _sha256(arm_path), "pilot arm changed")
    pair = _read_json(pair_path)
    arm = _read_json(arm_path)
    _require(pair == pilot.get("pair"), "pilot pair summary changed")
    control = arm.get("prefix_fallback_control", {})
    sidecar_path = Path(str(control.get("orders_sidecar", "")))
    _require(sidecar_path.is_file(), "pilot orders sidecar missing")
    _require(control.get("orders_sidecar_sha256") == _sha256(sidecar_path),
             "pilot orders sidecar changed")
    sidecar = _read_json(sidecar_path)
    _require(sidecar.get("cell", {}).get("cell_id") == 1, "pilot orders sidecar cell changed")
    _require(sidecar.get("arm") == CANDIDATE_ARM, "pilot orders sidecar arm changed")
    _require(int(sidecar.get("registration_count", -1)) == int(control.get("registration_count", -2)),
             "pilot orders sidecar registration count changed")
    return pilot


def run_full(output: Path) -> dict[str, Any]:
    preflight, config, references = _phase_inputs(output)
    pilot = _validate_pilot_reuse(output, preflight)
    _require(not (output / "full.json").exists(), "full artifact already exists")
    summaries = [dict(pilot["pair"])]
    for cell in CELL_SPECS[1:]:
        summary = _run_cell(output, config, cell, references[int(cell["cell_id"])])
        summaries.append(summary)
        technical = _technical_error(summary)
        if technical:
            payload = {
                "stage": STAGE, "phase": "full", "status": "FULL_TECHNICAL_ERROR",
                "pair_count": len(summaries), "pair_summaries": summaries,
                "error": technical, "PPO_ALLOWED": False,
            }
            _write_json(output / "full.json", payload, refuse_existing=True)
            _write_json(output / "progress.json", payload)
            raise PrefixFallbackRunnerError(f"full stopped after technical failure: {technical}")
    payload = {
        "stage": STAGE, "phase": "full", "status": "FULL_OK", "pair_count": len(summaries),
        "pair_summaries": summaries, "business_review": _business_review(summaries),
        "PPO_ALLOWED": False, "qualification": "not_a_qualification; frozen candidate comparison",
    }
    _write_json(output / "full.json", payload, refuse_existing=True)
    _write_json(output / "progress.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "pilot", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_preflight(args.output_dir) if args.phase == "preflight" else (
            run_pilot(args.output_dir) if args.phase == "pilot" else run_full(args.output_dir)
        )
    except Exception as exc:
        _write_json(args.output_dir / "progress.json", {
            "stage": STAGE, "phase": args.phase, "status": "FAILED",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        })
        print(json.dumps({"stage": STAGE, "status": "FAILED", "error": str(exc)}, sort_keys=True), flush=True)
        return 2
    print(json.dumps({"stage": STAGE, "status": result.get("status"), "PPO_ALLOWED": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
