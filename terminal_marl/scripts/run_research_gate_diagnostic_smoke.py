"""Run the bounded real-artifact paired gate diagnostic smoke.

This entry point is intentionally fixed to the registered development
artifacts and to a two-tick branch budget.  It only loads policies, resets one
development environment, evaluates the diagnostic branches, and writes a
strict JSON report.  No optimizer or training call is made here.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval
from terminal_marl.training import paired_gate_diagnostic as diagnostic
from terminal_marl.training import run_p3_task_neutral_pilot as pilot


REPORT_SCHEMA = "research_gate_diagnostic_real_artifact_smoke_v1"
FINAL_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_larger_batch_20260929/full/training/batch_04/policy.pt"
)
FINAL_CHECKPOINT_SHA256 = (
    "d44990dc7d129573d9145f169df1cbc4f91fa32bc5127b1c58d937260c02a24f"
)
WARMUP_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_episodic_learning_20260929/warmup_scaled/warmup.pt"
)
WARMUP_CHECKPOINT_SHA256 = (
    "a82ed7a287035fb1fc868fc955e056c0481de75bcb99880d5c79f35588dde710"
)
ENV_SEED = 877101
CONTINUATION_SEED = 878201
MAX_STEPS = 2
EXPECTED_GAMMA = 1.0
EXPECTED_REWARD_SCALE = 0.01
OBSERVATION_DIM = 88
PROPOSAL_COUNT = 25
EXPECTED_BRANCH_COUNT = PROPOSAL_COUNT + 1
DEFAULT_OUTPUT = Path(
    "/home/user/training_outputs/terminal/"
    "research_review_repairs_20260930/real_artifact_smoke.json"
)


class SmokeError(ValueError):
    """Raised when the fixed smoke contract is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


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
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _rng_snapshot() -> tuple[Any, tuple[Any, ...], torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def _rng_equal(
    left: tuple[Any, tuple[Any, ...], torch.Tensor],
    right: tuple[Any, tuple[Any, ...], torch.Tensor],
) -> bool:
    return bool(
        left[0] == right[0]
        and left[1][0] == right[1][0]
        and np.array_equal(left[1][1], right[1][1])
        and left[1][2:] == right[1][2:]
        and torch.equal(left[2], right[2])
    )


def _state_snapshot(policy: Any) -> dict[str, torch.Tensor]:
    return {
        str(key): value.detach().cpu().clone()
        for key, value in policy.model.state_dict().items()
    }


def _state_unchanged(before: Mapping[str, torch.Tensor], policy: Any) -> bool:
    after = _state_snapshot(policy)
    return list(before) == list(after) and all(
        before[key].shape == after[key].shape and torch.equal(before[key], after[key])
        for key in before
    )


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, bool) or isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _config_report(config: Any, runtime_protocol: Mapping[str, Any]) -> dict[str, Any]:
    legacy_snapshot = neutral_eval._config_snapshot(config)
    return {
        "snapshot": legacy_snapshot,
        "snapshot_sha256": _canonical_sha256(legacy_snapshot),
        "student_decision_mask_mode": str(getattr(config, "student_decision_mask_mode")),
        "observation_schema_version": str(getattr(config, "observation_schema_version")),
        "observation_dimension": OBSERVATION_DIM,
        "action_schema_version": str(getattr(config, "action_schema_version")),
        "runtime_protocol": dict(_jsonable(runtime_protocol)),
    }


def _load_warm_policy(config: Any) -> tuple[Any, dict[str, Any]]:
    resolved, policy, binding = pilot._load_binding(config)
    _require(_sha256(resolved) == pilot.EXPECTED_CANDIDATE_SHA256,
             "source candidate checkpoint SHA mismatch")
    source_model_digest = pilot._model_digest(policy)
    _require(WARMUP_CHECKPOINT.is_file(), f"warmup checkpoint missing: {WARMUP_CHECKPOINT}")
    warmup_sha = _sha256(WARMUP_CHECKPOINT)
    _require(warmup_sha == WARMUP_CHECKPOINT_SHA256,
             "warmup checkpoint SHA mismatch")

    payload = torch.load(WARMUP_CHECKPOINT, map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), "warmup checkpoint payload must be a mapping")
    fit_parameters = payload.get("fit_parameters")
    _require(isinstance(fit_parameters, Mapping), "warmup fit_parameters are missing")
    gamma = float(fit_parameters.get("gamma"))
    reward_scale = float(fit_parameters.get("reward_scale"))
    _require(math.isclose(gamma, EXPECTED_GAMMA, rel_tol=0.0, abs_tol=1e-12),
             "warmup gamma is not 1.0")
    _require(math.isclose(reward_scale, EXPECTED_REWARD_SCALE, rel_tol=0.0, abs_tol=1e-12),
             "warmup reward_scale is not 0.01")
    _require(payload.get("parent_checkpoint_sha256") == pilot.EXPECTED_CANDIDATE_SHA256,
             "warmup parent checkpoint SHA mismatch")

    # Keep this call compatible with the registered three-argument loader.
    resolved_warmup, warmup_binding = pilot._load_initial_warmup(
        WARMUP_CHECKPOINT, policy, config,
    )
    policy.model.eval()
    _require(str(getattr(policy, "device", "cpu")) == "cpu",
             "warm-source policy is not on CPU")
    _require(len(policy.optimizer.state) == 0,
             "warm-source optimizer state is not empty")
    return policy, {
        "source_binding": dict(_jsonable(binding)),
        "source_checkpoint": str(resolved),
        "source_checkpoint_sha256": _sha256(resolved),
        "source_model_digest_before_warmup": source_model_digest,
        "warmup_checkpoint": str(resolved_warmup),
        "warmup_checkpoint_sha256": _sha256(resolved_warmup),
        "warmup_fit_parameters": {"gamma": gamma, "reward_scale": reward_scale},
        "warmup_binding": dict(_jsonable(warmup_binding)),
        "warmup_model_digest_after_load": pilot._model_digest(policy),
        "optimizer_state_entries_after_load": len(policy.optimizer.state),
        "loader": "run_p3_task_neutral_pilot._load_binding + _load_initial_warmup",
    }


def _load_final_policy(config: Any) -> tuple[Any, dict[str, Any]]:
    _require(FINAL_CHECKPOINT.is_file(), f"final checkpoint missing: {FINAL_CHECKPOINT}")
    final_sha = _sha256(FINAL_CHECKPOINT)
    _require(final_sha == FINAL_CHECKPOINT_SHA256,
             "final checkpoint SHA mismatch")
    policy, payload = rank_eval._load_policy(FINAL_CHECKPOINT, config)
    policy.model.eval()
    _require(str(getattr(policy, "device", "cpu")) == "cpu",
             "final policy is not on CPU")
    _require(int(policy.observation_dim) == OBSERVATION_DIM,
             "final policy observation dimension is not 88")
    _require(len(policy.optimizer.state) == 0,
             "final policy optimizer state was restored")
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    _require(isinstance(metadata, Mapping), "final checkpoint metadata is missing")
    _require(int(metadata.get("observation_dimension")) == OBSERVATION_DIM,
             "final checkpoint metadata observation dimension changed")
    return policy, {
        "checkpoint": str(FINAL_CHECKPOINT),
        "checkpoint_sha256": final_sha,
        "payload_stage": payload.get("stage"),
        "payload_keys": sorted(str(key) for key in payload),
        "metadata": dict(_jsonable(metadata)),
        "model_digest": rank_eval._model_digest(policy),
        "optimizer_state_entries_after_load": len(policy.optimizer.state),
        "loader": "evaluate_p3_runtime_rank_repair._load_policy -> calibrated loader (load_optimizer=False)",
    }


def _run(output: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    _, neutral_configs, runtime_protocol = neutral_eval._build_neutral_inputs()
    config = neutral_configs["dense24"]
    config_report = _config_report(config, runtime_protocol)
    _require(config_report["student_decision_mask_mode"] == "legacy",
             "dense24 smoke config is not legacy mask mode")

    warm_policy, warm_binding = _load_warm_policy(config)
    final_policy, final_binding = _load_final_policy(config)
    warm_before = _state_snapshot(warm_policy)
    final_before = _state_snapshot(final_policy)

    env = TerminalParallelEnv(config)
    observations, reset_info = env.reset(ENV_SEED)
    _require(isinstance(observations, Mapping), "environment reset observations are not a mapping")
    setattr(env, "_paired_seed", CONTINUATION_SEED)
    env_before_request = diagnostic._state_fingerprint(env)
    request_rng_before = _rng_snapshot()
    request = diagnostic.keyed_request(final_policy, env)
    request_rng_after = _rng_snapshot()
    env_after_request = diagnostic._state_fingerprint(env)
    details = request.details
    uids = [str(uid) for uid in details["uids"]]
    actor_mask = [bool(value) for value in np.asarray(details["actor_mask"], dtype=bool).tolist()]
    active_uids = {str(uid) for uid in getattr(env, "active", {})}
    eligible_uids = [uid for uid, allowed in zip(uids, actor_mask) if allowed and uid in active_uids]
    _require(eligible_uids, "reset state has no eligible active UID")
    target_uid = eligible_uids[0]
    _require(target_uid in request.actions, "target UID has no frozen first action")

    diagnostic_rng_before = _rng_snapshot()
    paired = diagnostic.paired_gate_value(
        final_policy,
        env,
        request.actions,
        request.shortlists,
        target_uid,
        [CONTINUATION_SEED],
        MAX_STEPS,
    )
    alignment = diagnostic.gate_update_alignment(
        warm_policy,
        final_policy,
        details["world"],
    )
    diagnostic_rng_after = _rng_snapshot()
    warm_unchanged = _state_unchanged(warm_before, warm_policy)
    final_unchanged = _state_unchanged(final_before, final_policy)
    global_rng_unchanged = _rng_equal(diagnostic_rng_before, diagnostic_rng_after)
    env_unchanged = diagnostic._state_fingerprint(env) == env_before_request

    branches = paired.get("branches", [])
    branch_errors = [
        {"branch": row.get("branch"), "error": row.get("error")}
        for row in branches if isinstance(row, Mapping) and row.get("error")
    ]
    _require(len(branches) == EXPECTED_BRANCH_COUNT,
             f"expected {EXPECTED_BRANCH_COUNT} branches, got {len(branches)}")
    _require(len(paired.get("wait", [])) == 1, "expected one WAIT branch")
    _require(len(paired.get("releases", [])) == PROPOSAL_COUNT,
             f"expected {PROPOSAL_COUNT} RELEASE branches")
    _require(not branch_errors, f"paired diagnostic branch errors: {branch_errors}")
    _require(paired.get("censored") is True, "two-tick paired diagnostic was not censored")
    _require("qualified_delta_Q" not in paired,
             "censored paired diagnostic unexpectedly emitted qualified_delta_Q")
    _require(all(paired.get(key) is True for key in (
        "model_unchanged", "env_snapshot_unchanged", "rng_unchanged",
    )), "paired diagnostic invariance check failed")
    _require(alignment.get("model_unchanged") is True,
             "alignment model invariance check failed")
    _require(alignment.get("rng_unchanged") is True,
             "alignment RNG invariance check failed")
    _require(warm_unchanged and final_unchanged and global_rng_unchanged and env_unchanged,
             "parent model, environment, or global RNG changed during diagnostics")

    proposal_before = alignment.get("proposal_probabilities_before", [])
    proposal_after = alignment.get("proposal_probabilities_after", [])
    _require(_all_finite(paired.get("proposal_probabilities")),
             "paired proposal probabilities are not finite")
    for key in (
        "before_gate_release_probability", "after_gate_release_probability",
        "linear_approximation", "actual_probability_delta",
        "proposal_probabilities_before", "proposal_probabilities_after",
    ):
        _require(_all_finite(alignment.get(key)), f"alignment field is not finite: {key}")
    _require(len(alignment.get("changed_parameter_names", [])) > 0,
             "warm-source and final policies have no changed parameters")
    proposal_delta = [
        [float(after) - float(before) for before, after in zip(left, right)]
        for left, right in zip(proposal_before, proposal_after)
    ]
    _require(_all_finite(proposal_delta), "proposal probability deltas are not finite")

    return {
        "schema": REPORT_SCHEMA,
        "status": "completed",
        "output": str(output),
        "artifacts": {
            "final": final_binding,
            "warm_source": warm_binding,
        },
        "config": config_report,
        "diagnostic": {
            "schema": diagnostic.DIAGNOSTIC_SCHEMA,
            "proposal_count": PROPOSAL_COUNT,
            "max_steps": MAX_STEPS,
            "max_steps_semantics": "branch_tick_budget_from_reset_snapshot",
            "environment_seed": ENV_SEED,
            "continuation_seed": CONTINUATION_SEED,
            "reset_info": _jsonable(reset_info),
            "snapshot_start_step": int(getattr(env, "step_count", 0)),
            "active_uids": sorted(active_uids, key=lambda value: int(value)),
            "actor_uids": uids,
            "actor_mask": actor_mask,
            "target_uid": target_uid,
            "first_actions": _jsonable(request.actions),
            "first_shortlists": _jsonable(request.shortlists),
            "request_details": _jsonable(details),
            "keyed_request_rng_unchanged": _rng_equal(request_rng_before, request_rng_after),
            "keyed_request_env_unchanged": env_before_request == env_after_request,
            "paired_gate_value": _jsonable(paired),
            "paired_branch_count": len(branches),
            "paired_branch_error_count": len(branch_errors),
            "paired_branch_errors": branch_errors,
            "censored": bool(paired.get("censored")),
            "qualified_delta_Q_present": "qualified_delta_Q" in paired,
            "alignment": _jsonable(alignment),
            "proposal_probability_delta": proposal_delta,
            "changed_parameter_names": list(alignment.get("changed_parameter_names", [])),
            "changed_parameter_count": len(alignment.get("changed_parameter_names", [])),
        },
        "invariance": {
            "paired_model_unchanged": bool(paired.get("model_unchanged")),
            "paired_env_snapshot_unchanged": bool(paired.get("env_snapshot_unchanged")),
            "paired_rng_unchanged": bool(paired.get("rng_unchanged")),
            "alignment_model_unchanged": bool(alignment.get("model_unchanged")),
            "alignment_rng_unchanged": bool(alignment.get("rng_unchanged")),
            "warm_source_model_unchanged": warm_unchanged,
            "final_model_unchanged": final_unchanged,
            "global_rng_unchanged": global_rng_unchanged,
            "parent_env_unchanged": env_unchanged,
        },
        "checks": {
            "branch_count_expected": EXPECTED_BRANCH_COUNT,
            "branch_count_observed": len(branches),
            "all_branches_error_free": not branch_errors,
            "censored_true": paired.get("censored") is True,
            "qualified_delta_Q_absent": "qualified_delta_Q" not in paired,
            "all_probabilities_and_deltas_finite": True,
            "real_parameter_changes": bool(alignment.get("changed_parameter_names")),
            "all_invariance_true": all((
                bool(paired.get("model_unchanged")),
                bool(paired.get("env_snapshot_unchanged")),
                bool(paired.get("rng_unchanged")),
                bool(alignment.get("model_unchanged")),
                bool(alignment.get("rng_unchanged")),
                warm_unchanged,
                final_unchanged,
                global_rng_unchanged,
                env_unchanged,
            )),
        },
        "errors": [],
        "deviations": [],
        "unresolved": [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = Path(args.output)
    record: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "starting",
        "output": str(output),
        "errors": [],
        "deviations": [],
        "unresolved": [],
    }
    try:
        record = _run(output)
        _write_json(output, record)
        return 0
    except Exception as exc:
        record["status"] = "failed"
        record["errors"] = [{"type": type(exc).__name__, "message": str(exc)}]
        try:
            _write_json(output, record)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
