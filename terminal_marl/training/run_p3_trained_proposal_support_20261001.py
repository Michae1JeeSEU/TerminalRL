"""Run the registered native-sampler proposal-support comparison.

This is a development-only three-arm comparison on the four original dense24
cells.  The before and after legacy arms use the production student rollout
unchanged.  The after-support arm temporarily replaces only the native
request-details function so RELEASE proposals are sampled from the fixed
epsilon mixture.  No optimizer, Teacher, held-out data, or PPO admission is
used here.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from terminal_marl.algo.joint_ppo_v4_2 import hierarchical_joint_logprob
from terminal_marl.core.low_level_joint_action_v4_2 import (
    HierarchicalActionV42,
    RELEASE_GATE,
    WAIT_GATE,
    decode_hierarchical_actions_v4_2,
)
from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval
from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval
from terminal_marl.training import paired_gate_diagnostic as paired
from terminal_marl.training.run_p3_forced_wait_frozen_20260930 import _CapturedTerminalEnv
from terminal_marl.training import student_rollout_v4_2 as student


STAGE = "P3_TRAINED_PROPOSAL_SUPPORT_20261001_V1"
PPO_ALLOWED = False
HELDOUT_USED = False
TEACHER_USED = False
OPTIMIZER_USED = False
OFF_POLICY = True
NO_TRAINING = True
EPSILON = 0.10
Q_FLOOR_MIN = 0.004
PROPOSAL_COUNT = 25
OBSERVATION_SCHEMA = "phaseA_low_obs_v2_1_retry_state"
ACTION_SCHEMA = "low_action_schema_v4_2_hierarchical"
MAX_STEPS = 1800

BEFORE_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt"
)
BEFORE_CHECKPOINT_SHA256 = "3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab"
AFTER_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_larger_batch_20260929/full/training/batch_04/policy.pt"
)
AFTER_CHECKPOINT_SHA256 = "d44990dc7d129573d9145f169df1cbc4f91fa32bc5127b1c58d937260c02a24f"
WARMUP_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_task_neutral_episodic_learning_20260929/warmup_scaled/warmup.pt"
)
WARMUP_CHECKPOINT_SHA256 = "a82ed7a287035fb1fc868fc955e056c0481de75bcb99880d5c79f35588dde710"
OUTPUT_DIR = Path(
    "/home/user/training_outputs/terminal/"
    "p3_followup_20260930/trained_proposal_support_20261001"
)
PROTOCOL = Path(__file__).resolve().parents[2] / "docs/evidence/v4_2/p3_trained_proposal_support_protocol_20261001.md"

CELL_SPECS = tuple(
    {
        "cell_id": index,
        "scenario": "dense24",
        "env_seed": 875100 + index,
        "policy_seed": 875200 + index,
        "split": "development_original_dense24",
    }
    for index in range(1, 5)
)
ARMS = ("before_legacy", "after_legacy", "after_support")
CONTROL_NAME = "native_release_epsilon_uniform_proposal_support"


class TrainedProposalSupportError(ValueError):
    """Raised when the fixed native-sampler contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainedProposalSupportError(message)


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
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        try:
            return _jsonable(value.value)
        except Exception:
            pass
    return value


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    _require(target.is_file(), f"JSON input missing: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainedProposalSupportError(f"invalid JSON input: {target}") from exc
    _require(isinstance(value, dict), f"JSON object required: {target}")
    return dict(value)


def _write_json(path: Path, value: Mapping[str, Any], *, refuse_existing: bool = False) -> None:
    if refuse_existing and path.exists():
        raise TrainedProposalSupportError(f"refusing to overwrite existing artifact: {path}")
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
    target = Path(path)
    if target.is_file():
        return target
    value = str(target).replace("\\", "/")
    prefix = "/home/user/training_outputs/terminal/"
    if value.startswith(prefix):
        mirror = Path(__file__).resolve().parents[3] / "training_outputs" / "terminal" / value[len(prefix):]
        if mirror.is_file():
            return mirror
    return target


def _resolve_checkpoint(path: Path, expected_sha: str) -> Path:
    resolved = _sync_path(path)
    _require(resolved.is_file(), f"checkpoint missing: {path}")
    actual = _sha256(resolved)
    _require(actual == expected_sha, f"checkpoint SHA mismatch for {path}: {actual}")
    return resolved


def _rng_snapshot() -> Any:
    return paired._rng_snapshot()


def _restore_rng(value: Any) -> None:
    paired._restore_rng(value)


def _rng_equal(left: Any, right: Any) -> bool:
    return paired._rng_equal(left, right)


def _proposal_mixture(q: Sequence[float], *, epsilon: float = EPSILON) -> np.ndarray:
    source = np.asarray(q, dtype=np.float64)
    _require(source.ndim == 1 and source.size == PROPOSAL_COUNT, "proposal q must contain 25 entries")
    _require(np.all(np.isfinite(source)) and np.all(source >= 0.0), "proposal q must be finite and nonnegative")
    _require(math.isfinite(float(epsilon)) and 0.0 <= float(epsilon) <= 1.0, "epsilon is outside [0, 1]")
    total = float(source.sum())
    _require(total > 0.0 and math.isfinite(total), "proposal q has no positive mass")
    if float(epsilon) == 0.0:
        return source.copy()
    mixed = (1.0 - float(epsilon)) * (source / total) + float(epsilon) / PROPOSAL_COUNT
    _require(np.all(np.isfinite(mixed)), "proposal mixture is non-finite")
    _require(float(mixed.min()) >= Q_FLOOR_MIN - 1e-12, "proposal mixture violates q floor")
    _require(abs(float(mixed.sum()) - 1.0) <= 1e-12, "proposal mixture is not normalized")
    return mixed.copy()


def _proposal_mixture_tensor(q: torch.Tensor, *, epsilon: float = EPSILON) -> torch.Tensor:
    _require(q.ndim == 2 and q.shape[-1] == PROPOSAL_COUNT, "proposal tensor must have shape [N,25]")
    rows = [_proposal_mixture(row.detach().cpu().numpy(), epsilon=epsilon) for row in q]
    return torch.as_tensor(np.asarray(rows), dtype=q.dtype, device=q.device)


def _native_support_request_details(
    policy: Any,
    env: Any,
    observations: dict,
    *,
    deterministic: bool,
    shortlist_k: int,
    epsilon: float = EPSILON,
    original: Any | None = None,
    audit: dict[str, Any] | None = None,
):
    """Native request-details logic with only conditional RELEASE q changed."""

    if float(epsilon) == 0.0:
        _require(original is not None, "epsilon-zero parity requires the original function")
        return original(policy, env, observations, deterministic=deterministic, shortlist_k=shortlist_k)
    if shortlist_k not in {1, 3}:
        raise ValueError("student runtime shortlist must have K=1 or K=3")
    uids = sorted(observations)
    world = np.asarray([observations[uid] for uid in uids], dtype=np.float32).reshape(-1, policy.observation_dim)
    decision_mask_mode = getattr(env.config, "student_decision_mask_mode", "legacy")
    affected_interfaces = None
    actor_values = []
    for uid in uids:
        uav = env.active[int(uid)]
        actor = (
            uav.committed_plan is None
            and uav.phase.value in {"ring_outer", "ring_middle", "ring_inner"}
            and int(uid) not in env.future_release_reservations
        )
        if actor and decision_mask_mode == "pre_action_forced_wait":
            if affected_interfaces is None:
                affected_interfaces = set(env.airspace.affected_interfaces())
            ring_role = getattr(uav.ring_role, "value", uav.ring_role)
            interface = {"outer": "o2m", "middle": "m2i", "inner": "i2f"}.get(ring_role)
            actor = (
                interface is not None
                and float(uav.soc) > 0.0
                and float(env.now) >= float(uav.safety_retry_after)
                and interface not in affected_interfaces
            )
        actor_values.append(actor)
    actor_mask = np.asarray(actor_values, dtype=bool)
    if not uids:
        return (uids, world, actor_mask, {}, np.zeros(0, dtype=int), np.zeros(0, dtype=int),
                np.zeros(0), np.zeros(0), {}, {})
    tensor = torch.as_tensor(world, device=policy.device)
    with torch.no_grad():
        gates = torch.zeros(len(uids), dtype=torch.long, device=policy.device)
        proposals = torch.full((len(uids),), -1, dtype=torch.long, device=policy.device)
        gate_dist, proposal_dist = policy.model.distributions(tensor)
        mask = torch.as_tensor(actor_mask, device=policy.device)
        source_gate_logits = gate_dist.logits.detach().clone()
        source_proposal_logits = proposal_dist.logits.detach().clone()
        source_q = proposal_dist.probs.detach().clone()
        if mask.any():
            logits = gate_dist.logits[mask]
            gates[mask] = logits.argmax(-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
        release = gates == RELEASE_GATE
        mixed_dist = None
        if release.any():
            mixed_probs = _proposal_mixture_tensor(proposal_dist.probs, epsilon=epsilon)
            mixed_dist = torch.distributions.Categorical(probs=mixed_probs)
            release_dist = torch.distributions.Categorical(probs=mixed_probs[release])
            proposals[release] = (
                release_dist.probs.argmax(-1)
                if deterministic else release_dist.sample()
            )
        else:
            mixed_dist = torch.distributions.Categorical(probs=_proposal_mixture_tensor(proposal_dist.probs, epsilon=epsilon))
        gate_lp, _source_proposal_lp, _source_joint_lp = hierarchical_joint_logprob(
            gate_dist, proposal_dist, gates, proposals,
        )
        _gate_mixed_lp, proposal_lp, joint_lp = hierarchical_joint_logprob(
            gate_dist, mixed_dist, gates, proposals,
        )
        if decision_mask_mode == "pre_action_forced_wait":
            forced_wait = ~mask
            gate_lp = torch.where(forced_wait, torch.zeros_like(gate_lp), gate_lp)
            joint_lp = torch.where(forced_wait, torch.zeros_like(joint_lp), joint_lp)
    _require(torch.equal(source_gate_logits, gate_dist.logits), "source gate logits changed")
    _require(torch.equal(source_proposal_logits, proposal_dist.logits), "source proposal logits changed")
    _require(torch.equal(source_q, proposal_dist.probs), "source proposal probabilities changed")
    gate_np, proposal_np = gates.cpu().numpy(), proposals.cpu().numpy()
    decoded = decode_hierarchical_actions_v4_2(
        gate_np, np.asarray([None if p == -1 else int(p) for p in proposal_np], dtype=object),
    )
    actions = {uid: decoded[i] for i, uid in enumerate(uids)}
    likelihoods = {
        uid: {
            "gate": float(gate_lp[i]),
            "proposal": None if proposal_np[i] == -1 else float(proposal_lp[i]),
            "joint": float(joint_lp[i]),
        }
        for i, uid in enumerate(uids)
    }
    shortlists: dict[str, tuple[np.ndarray, ...]] = {}
    if shortlist_k > 1:
        proposal_logits = proposal_dist.logits.detach().cpu()
        for index, uid in enumerate(uids):
            if not actor_mask[index] or gate_np[index] != RELEASE_GATE:
                continue
            selected = int(proposal_np[index])
            ranked = torch.argsort(proposal_logits[index], descending=True, stable=True).tolist()
            ordered = [selected] + [int(value) for value in ranked if int(value) != selected]
            shortlists[uid] = tuple(
                HierarchicalActionV42(RELEASE_GATE, proposal).normalized.copy()
                for proposal in ordered[:shortlist_k]
            )
    if audit is not None:
        audit["calls"] = int(audit.get("calls", 0)) + 1
        audit["release_rows"] = int(audit.get("release_rows", 0)) + int(release.sum().item())
        audit["source_q_immutable"] = bool(audit.get("source_q_immutable", True))
        audit["source_logits_immutable"] = bool(audit.get("source_logits_immutable", True))
        audit["source_gate_unchanged"] = bool(audit.get("source_gate_unchanged", True))
        audit["k3_source_logits_ranked"] = bool(audit.get("k3_source_logits_ranked", True))
        audit["likelihood_distribution"] = "epsilon_mixture_conditional_release"
    return (
        uids, world, actor_mask, actions, gate_np, proposal_np, gate_lp.cpu().numpy(),
        proposal_lp.cpu().numpy(), likelihoods, shortlists,
    )


@contextlib.contextmanager
def _support_sampler(audit: dict[str, Any], *, epsilon: float = EPSILON) -> Iterator[None]:
    original = student._student_request_details

    def support_details(policy: Any, env: Any, observations: dict, *, deterministic: bool, shortlist_k: int):
        return _native_support_request_details(
            policy, env, observations, deterministic=deterministic,
            shortlist_k=shortlist_k, epsilon=epsilon, original=original, audit=audit,
        )

    student._student_request_details = support_details
    try:
        yield
    finally:
        student._student_request_details = original


@contextlib.contextmanager
def _capture_runtime_env(capture: dict[str, Any]) -> Iterator[None]:
    """Capture one rollout's passive request-observing environment object.

    The factory is wrapped only for the duration of one arm and returns the
    already-qualified `_CapturedTerminalEnv`; no step, action, or observation
    path is added to the rollout. The student factory is restored in `finally`.
    """

    original = student.TerminalParallelEnv
    capture["factory_calls"] = 0

    def factory(*args: Any, **kwargs: Any) -> Any:
        capture["factory_calls"] = int(capture.get("factory_calls", 0)) + 1
        # This already-qualified passive subclass wraps only ArrivalProcess
        # request generation and returns the native env semantics unchanged.
        env = _CapturedTerminalEnv(*args, **kwargs)
        capture["env"] = env
        return env

    student.TerminalParallelEnv = factory
    try:
        yield
    finally:
        student.TerminalParallelEnv = original


def _live_runtime_snapshot(capture: Mapping[str, Any], config: Any) -> dict[str, Any]:
    """Serialize live ledger/energy facts without retaining the env object."""

    result: dict[str, Any] = {
        "factory_calls": int(capture.get("factory_calls", 0) or 0),
        "available": False,
    }
    env = capture.get("env")
    if env is None:
        result["error"] = "native environment was not captured"
        return result
    ledger = getattr(env, "ledger", None)
    requested = getattr(ledger, "requested", None)
    outcomes = getattr(ledger, "outcomes", None)
    requested_ids = sorted(int(value) for value in requested) if isinstance(requested, set) else None
    outcome_map: dict[str, Any] | None = None
    if isinstance(outcomes, Mapping):
        outcome_map = {
            str(int(key)): getattr(value, "value", str(value))
            for key, value in outcomes.items()
        }
    object_ids: set[int] = set()
    for collection in (
        getattr(env, "active", {}),
        getattr(env, "service_uavs", {}),
        getattr(env, "completed_uavs", []),
    ):
        values = collection.values() if isinstance(collection, Mapping) else collection
        for item in values or ():
            arrival_id = getattr(item, "arrival_id", None)
            if arrival_id is not None:
                object_ids.add(int(arrival_id))
    captured_rows = [dict(row) for row in (getattr(env, "_captured_request_rows", []) or [])]
    captured_ids = sorted(int(row["arrival_id"]) for row in captured_rows if row.get("arrival_id") is not None)
    lifecycle_ids = {
        int(item.get("arrival_id"))
        for item in (getattr(env, "energy_outcomes", []) or [])
        if isinstance(item, Mapping)
        and item.get("arrival_id") is not None
        and item.get("uid") is not None
    }
    outcome_ids = set(int(key) for key in (outcome_map or {}))
    admitted_ids = sorted(object_ids | lifecycle_ids)
    energy_rows = [dict(item) for item in (getattr(env, "energy_outcomes", []) or []) if isinstance(item, Mapping)]
    energy_by_arrival: dict[str, float] = {}
    energy_duplicate_ids: set[str] = set()
    for row in energy_rows:
        arrival_id = row.get("arrival_id")
        value = neutral_eval._finite(row.get("energy_consumed"))
        if arrival_id is None or value is None:
            continue
        key = str(int(arrival_id))
        if key in energy_by_arrival:
            energy_duplicate_ids.add(key)
        energy_by_arrival[key] = float(value)
    telemetry = getattr(env, "telemetry", {})
    telemetry = dict(telemetry) if isinstance(telemetry, Mapping) else {}
    rejection_count = int(telemetry.get("release_rejected", 0) or 0)
    penalty_rate = neutral_eval._finite(getattr(config, "low_level_release_rejected_penalty", None))
    rejection_penalty = None if penalty_rate is None else -float(rejection_count) * float(penalty_rate)
    ledger_alignment = bool(
        requested_ids is not None
        and outcome_map is not None
        and outcome_ids.issubset(set(requested_ids))
        and set(energy_by_arrival).issuperset({str(value) for value in outcome_ids})
        and not energy_duplicate_ids
    )
    result.update({
        "available": True,
        "generated_requests": int(getattr(getattr(env, "arrivals", None), "generated", 0)),
        "captured_request_rows": captured_rows,
        "captured_request_ids": captured_ids,
        "requested_arrival_ids": requested_ids,
        "outcomes": outcome_map,
        "outcome_arrival_ids": sorted(outcome_ids),
        "admitted_arrival_ids": admitted_ids,
        "admitted_count": len(admitted_ids),
        "telemetry_arrivals_admitted": int(telemetry.get("arrivals_admitted", 0) or 0),
        "energy_outcome_arrival_ids": sorted(int(value) for value in energy_by_arrival),
        "energy_by_arrival": energy_by_arrival,
        "energy_duplicate_arrival_ids": sorted(int(value) for value in energy_duplicate_ids),
        "energy_total_consumed": float(sum(energy_by_arrival.values())) if energy_by_arrival else None,
        "telemetry": _jsonable(telemetry),
        "release_rejected": rejection_count,
        "release_rejection_penalty": rejection_penalty,
        "ledger_alignment": ledger_alignment,
    })
    return result


def _config_snapshot(config: Any) -> dict[str, Any]:
    return neutral_eval._config_snapshot(config)


def _build_inputs() -> tuple[Any, dict[str, Any]]:
    _original, neutral_configs, _runtime = neutral_eval._build_neutral_inputs()
    _require("dense24" in neutral_configs, "neutral dense24 config is missing")
    config = neutral_configs["dense24"]
    _require(config.action_schema_version == ACTION_SCHEMA, "action schema changed")
    _require(config.observation_schema_version == OBSERVATION_SCHEMA, "observation schema changed")
    _require(int(config.runtime_proposal_shortlist_k) == 3, "runtime shortlist K changed")
    _require(bool(config.teacher_enabled) is False, "Teacher must be disabled")
    _require(str(config.scheduler_mode) == "execution", "scheduler mode changed")
    _require(float(config.max_simulation_seconds) == 1800.0, "simulation cap changed")
    _require(float(config.max_simulation_seconds) <= MAX_STEPS * float(config.dt), "simulation cap exceeds protocol")
    _require(bool(getattr(config, "task_priority_enabled", False)) is False,
             "task priority must be disabled")
    _require(float(getattr(config, "low_level_priority_time_weight", 0.0)) == 0.0,
             "task priority reward weight must be zero")
    _require(bool(getattr(config, "rule_flow_coordination_enabled", False)) is False,
             "rule flow coordination must be disabled")
    return config, neutral_configs


def _policy_binding(path: Path, expected_sha: str, config: Any, label: str) -> dict[str, Any]:
    resolved = _resolve_checkpoint(path, expected_sha)
    policy, _payload = rank_eval._load_policy(resolved, config)
    _require(int(policy.observation_dim) == 88, f"{label} policy observation dimension changed")
    model = policy.model
    state = model.state_dict()
    _require(bool(state), f"{label} model state is empty")
    optimizer = getattr(policy, "optimizer", None)
    _require(isinstance(getattr(optimizer, "state", None), Mapping) and len(optimizer.state) == 0,
             f"{label} optimizer state is not empty")
    return {
        "label": label,
        "declared_checkpoint": str(path),
        "resolved_checkpoint": str(resolved),
        "checkpoint_sha256": _sha256(resolved),
        "model_digest": rank_eval._model_digest(policy),
        "observation_dim": int(policy.observation_dim),
        "state_keys": sorted(str(key) for key in state),
        "state_shapes": {str(key): list(value.shape) for key, value in state.items()},
        "model_training": bool(getattr(model, "training", True)),
        "gate_temperature": _jsonable(getattr(model, "gate_temperature", None)),
        "optimizer_state_empty": True,
    }


def _qualified_frozen_binding(config: Any) -> dict[str, Any]:
    """Compare only the registered before/after frozen groups.

    The d449 checkpoint is not a proposal-head-only update: critic and gate
    adapter tensors may differ.  This binding therefore records the explicit
    actor/proposal/gate-head/temperature groups used by the proposal-support
    protocol.
    """

    before_path = _resolve_checkpoint(BEFORE_CHECKPOINT, BEFORE_CHECKPOINT_SHA256)
    after_path = _resolve_checkpoint(AFTER_CHECKPOINT, AFTER_CHECKPOINT_SHA256)
    before, _ = rank_eval._load_policy(before_path, config)
    after, _ = rank_eval._load_policy(after_path, config)
    source_state = before.model.state_dict()
    candidate_state = after.model.state_dict()
    _require(set(source_state) == set(candidate_state), "before/after model state keys changed")

    def group_equal(prefix: str) -> bool:
        names = [name for name in source_state if name == prefix or name.startswith(prefix + ".")]
        return bool(names) and all(torch.equal(source_state[name], candidate_state[name]) for name in names)

    zero = torch.zeros((1, 88), dtype=torch.float32)
    with torch.no_grad():
        source_gate, source_proposal = before.model.distributions(zero)
        candidate_gate, candidate_proposal = after.model.distributions(zero)
    groups = (
        "actor_encoder", "proposal_head", "gate_head", "gate_temperature",
        "gate_adapter", "critic_encoder", "critic_head",
    )
    equalities = {f"{group}_bitwise_unchanged": group_equal(group) for group in groups}
    allowed_changed = {"gate_adapter", "critic_encoder", "critic_head"}
    observed_changed = [group for group in groups if not equalities[f"{group}_bitwise_unchanged"]]
    unexpected_changed = sorted(set(observed_changed) - allowed_changed)
    _require(not unexpected_changed, f"unexpected before/after state groups changed: {unexpected_changed}")
    return {
        **equalities,
        "proposal_logits_zero_input_bitwise_unchanged": torch.equal(source_proposal.logits, candidate_proposal.logits),
        "proposal_probabilities_zero_input_bitwise_unchanged": torch.equal(source_proposal.probs, candidate_proposal.probs),
        "gate_logits_zero_input_bitwise_unchanged": torch.equal(source_gate.logits, candidate_gate.logits),
        "observed_changed_groups": observed_changed,
        "allowed_changed_groups": sorted(allowed_changed),
        "unexpected_changed_groups": unexpected_changed,
        "qualified_frozen_groups": ["actor_encoder", "proposal_head", "gate_head", "gate_temperature"],
    }


def _build_preflight() -> dict[str, Any]:
    config, _configs = _build_inputs()
    before = _policy_binding(BEFORE_CHECKPOINT, BEFORE_CHECKPOINT_SHA256, config, "before")
    after = _policy_binding(AFTER_CHECKPOINT, AFTER_CHECKPOINT_SHA256, config, "after")
    qualified_freeze = _qualified_frozen_binding(config)
    warmup = _resolve_checkpoint(WARMUP_CHECKPOINT, WARMUP_CHECKPOINT_SHA256)
    cells = [dict(item) for item in CELL_SPECS]
    return {
        "stage": STAGE,
        "phase": "preflight",
        "status": "PREFLIGHT_OK",
        "PPO_ALLOWED": PPO_ALLOWED,
        "heldout_used": HELDOUT_USED,
        "teacher_used": TEACHER_USED,
        "optimizer_used": OPTIMIZER_USED,
        "off_policy": OFF_POLICY,
        "no_training": NO_TRAINING,
        "epsilon": EPSILON,
        "q_floor_min": Q_FLOOR_MIN,
        "proposal_count": PROPOSAL_COUNT,
        "proposal_support": {
            "name": CONTROL_NAME,
            "source_gate": "native student gate distribution unchanged",
            "source_proposal_q": "native after policy proposal probabilities",
            "mixture": "0.9*(q/sum(q))+0.1/25 on sampled RELEASE proposals",
            "shortlist": "sampled proposal rank 1; source proposal logits top-2 fill",
            "epsilon_zero": "direct original _student_request_details parity helper only",
            "likelihood": "mixed conditional RELEASE likelihood",
        },
        "schema": {
            "observation": OBSERVATION_SCHEMA,
            "observation_dim": 88,
            "action": ACTION_SCHEMA,
            "runtime_shortlist_k": 3,
        },
        "runtime": {
            "config_snapshot": _config_snapshot(config),
            "max_steps": MAX_STEPS,
            "native_rollout": "evaluate_p3_runtime_rank_repair._runtime_record",
            "deterministic": False,
            "caller_rng_restored_per_arm": True,
        },
        "cells": cells,
        "requests_per_cell": 24,
        "checkpoints": {
            "before": before,
            "after": after,
            "qualified_frozen_groups": qualified_freeze,
            "warmup": {
                "path": str(WARMUP_CHECKPOINT), "resolved": str(warmup),
                "sha256": WARMUP_CHECKPOINT_SHA256,
                "role": "training_input_binding_only",
                "loaded_for_current_evaluation": False,
            },
        },
        "runner_sha256": _sha256(__file__),
        "protocol_sha256": _sha256(PROTOCOL) if PROTOCOL.is_file() else None,
        "qualification": "not_a_qualification; fixed native stochastic development comparison",
    }


def _validate_preflight(payload: Mapping[str, Any]) -> None:
    _require(payload.get("stage") == STAGE and payload.get("status") == "PREFLIGHT_OK", "preflight is not successful")
    for key, expected in (("PPO_ALLOWED", False), ("heldout_used", False), ("teacher_used", False),
                          ("optimizer_used", False), ("off_policy", True), ("no_training", True)):
        _require(payload.get(key) is expected, f"preflight flag {key} changed")
    _require(payload.get("runner_sha256") == _sha256(__file__), "runner changed after preflight")
    _require(float(payload.get("epsilon")) == EPSILON, "epsilon changed after preflight")
    _require(float(payload.get("q_floor_min")) == Q_FLOOR_MIN, "q floor changed after preflight")
    _require(int(payload.get("proposal_count")) == PROPOSAL_COUNT, "proposal count changed after preflight")
    _require(payload.get("cells") == [dict(item) for item in CELL_SPECS], "cell grid changed after preflight")
    _require(payload.get("protocol_sha256") == (_sha256(PROTOCOL) if PROTOCOL.is_file() else None),
             "protocol changed after preflight")
    config, _ = _build_inputs()
    runtime = payload.get("runtime")
    _require(isinstance(runtime, Mapping), "runtime binding missing")
    _require(runtime.get("config_snapshot") == _config_snapshot(config), "current config differs from preflight")
    _require(int(runtime.get("max_steps", -1)) == MAX_STEPS, "runtime max_steps binding changed")
    _require(runtime.get("native_rollout") == "evaluate_p3_runtime_rank_repair._runtime_record",
             "native rollout binding changed")
    _require(runtime.get("deterministic") is False, "native stochastic binding changed")
    _require(float(config.dt) == 1.0 and float(config.max_simulation_seconds) == 1800.0,
             "current dt or simulation cap changed")
    _require(int(config.runtime_proposal_shortlist_k) == 3, "current shortlist K changed")
    _require(bool(config.teacher_enabled) is False, "Teacher binding changed")
    _require(str(config.scheduler_mode) == "execution", "scheduler binding changed")
    _require(bool(getattr(config, "task_priority_enabled", False)) is False,
             "task priority binding changed")
    _require(float(getattr(config, "low_level_priority_time_weight", 0.0)) == 0.0,
             "task priority reward binding changed")
    _require(bool(getattr(config, "rule_flow_coordination_enabled", False)) is False,
             "rule flow binding changed")
    _require(payload.get("schema", {}).get("observation") == OBSERVATION_SCHEMA, "observation schema binding changed")
    _require(payload.get("schema", {}).get("action") == ACTION_SCHEMA, "action schema binding changed")
    checkpoints = payload.get("checkpoints")
    _require(isinstance(checkpoints, Mapping), "checkpoint binding missing")
    for label, path, expected_sha in (("before", BEFORE_CHECKPOINT, BEFORE_CHECKPOINT_SHA256),
                                      ("after", AFTER_CHECKPOINT, AFTER_CHECKPOINT_SHA256)):
        current = _policy_binding(path, expected_sha, config, label)
        stored = checkpoints.get(label)
        _require(isinstance(stored, Mapping), f"{label} checkpoint binding missing")
        for key in ("checkpoint_sha256", "model_digest", "observation_dim", "state_keys", "state_shapes",
                    "model_training", "optimizer_state_empty", "gate_temperature"):
            _require(stored.get(key) == current.get(key), f"{label} checkpoint binding changed for {key}")
    warmup = checkpoints.get("warmup")
    _require(isinstance(warmup, Mapping), "warmup binding missing")
    _require(warmup.get("sha256") == WARMUP_CHECKPOINT_SHA256, "warmup SHA binding changed")
    _require(warmup.get("role") == "training_input_binding_only" and warmup.get("loaded_for_current_evaluation") is False,
             "warmup evaluation role changed")
    live_warmup = _resolve_checkpoint(WARMUP_CHECKPOINT, WARMUP_CHECKPOINT_SHA256)
    _require(_sha256(live_warmup) == warmup.get("sha256"), "warmup checkpoint changed")
    current_freeze = _qualified_frozen_binding(config)
    _require(checkpoints.get("qualified_frozen_groups") == current_freeze,
             "qualified before/after frozen-group binding changed")


def _record_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    summary = record.get("summary") if isinstance(record.get("summary"), Mapping) else {}
    lifecycle = record.get("lifecycle") if isinstance(record.get("lifecycle"), Mapping) else {}
    runtime = record.get("runtime_capture") if isinstance(record.get("runtime_capture"), Mapping) else {}
    rows = neutral_eval._record_rows(record)
    expected_count = int(record.get("requests") or lifecycle.get("requested_count") or 24)
    _require(expected_count > 0, "request count must be positive")
    expected_ids = list(range(expected_count))
    requested_ids = _id_values(runtime.get("requested_arrival_ids"))
    if requested_ids is None:
        requested_ids = _id_values(
            lifecycle.get("requested_arrival_ids", lifecycle.get("ledger_requested_ids"))
        )
    if requested_ids is None:
        requested_ids = sorted({int(row["arrival_id"]) for row in rows if "arrival_id" in row})
    observed_ids = _id_values(lifecycle.get("observed_arrival_ids"))
    if observed_ids is None:
        observed_ids = sorted({int(row["arrival_id"]) for row in rows if "arrival_id" in row})
    row_ids = [int(row["arrival_id"]) for row in rows if "arrival_id" in row]
    row_ids_unique = len(row_ids) == len(set(row_ids))
    identity_rows = runtime.get("captured_request_rows") if isinstance(runtime.get("captured_request_rows"), list) else rows
    identity = _request_identity(identity_rows)
    captured_ids = _id_values(runtime.get("captured_request_ids"))
    request_identity_ids = captured_ids if captured_ids is not None else requested_ids
    identity_exact = bool(
        requested_ids == expected_ids
        and row_ids_unique
        and sorted(row_ids) == expected_ids
        and observed_ids == expected_ids
        and (captured_ids is None or captured_ids == expected_ids)
    )
    identity_exact_24 = bool(expected_count == 24 and identity_exact)
    runtime_present = bool(runtime)
    runtime_generated = int(runtime.get("generated_requests", -1)) if runtime_present else expected_count
    runtime_requested = _id_values(runtime.get("requested_arrival_ids")) if runtime_present else expected_ids
    runtime_captured = _id_values(runtime.get("captured_request_ids")) if runtime_present else expected_ids
    runtime_outcomes = _id_values(runtime.get("outcome_arrival_ids")) if runtime_present else []
    runtime_energy = _id_values(runtime.get("energy_outcome_arrival_ids")) if runtime_present else expected_ids
    runtime_partial_integrity = bool(
        not runtime_present
        or (
            runtime.get("available") is True
            and int(runtime.get("factory_calls", 0) or 0) == 1
            and runtime_generated >= 0
            and runtime_captured == list(range(runtime_generated))
            and len(runtime.get("captured_request_rows") or []) == runtime_generated
            and runtime_requested is not None
            and set(runtime_requested).issubset(set(runtime_captured or ()))
            and runtime_outcomes is not None
            and set(runtime_outcomes).issubset(set(runtime_requested))
            and runtime_energy is not None
            and set(runtime_energy).issuperset(set(runtime_outcomes))
            and not runtime.get("energy_duplicate_arrival_ids")
            and bool(runtime.get("ledger_alignment"))
        )
    )
    runtime_complete_integrity = bool(
        runtime_partial_integrity
        and (not runtime_present or (
            runtime_generated == expected_count
            and runtime_requested == expected_ids
            and runtime_captured == expected_ids
        ))
    )
    outcomes_observed = _observed_outcomes(record, rows)
    lifecycle_complete = bool(
        lifecycle.get("scope") == "all_requested_terminal_rows"
        and lifecycle.get("replay_exact") is True
        and lifecycle.get("all_requests_observed") is True
        and lifecycle.get("ledger_alignment") is True
        and int(lifecycle.get("missing_request_count", 0) or 0) == 0
        and int(lifecycle.get("row_count", 0) or 0) == expected_count
        and identity_exact
        and runtime_complete_integrity
    )
    terminal_flag = bool(
        summary.get("terminal") is True
        or ("terminal" not in summary and not runtime_present and record.get("status") == "completed")
    )
    complete_scene_metrics_known = False
    # Runtime rank evaluation stores the safety snapshot under ``summary``;
    # use the shared normalizer so a partial/censored prefix still exposes the
    # observed hard-safety counters.
    safety = neutral_eval._safety_snapshot(record)
    safety_fields = ("collisions", "nmac", "hard_separation_violations", "uncertified_execution", "SafetyInvariantError")
    safety_known = all(neutral_eval._finite(safety.get(key)) is not None for key in safety_fields)
    safety_nonzero = bool(record.get("safety_nonzero")) or any(
        int(neutral_eval._finite(safety.get(key)) or 0) != 0 for key in safety_fields
    )
    technical_ok = bool(
        not record.get("error")
        and record.get("weights_unchanged", record.get("model_unchanged")) is True
        and record.get("teacher_called", False) is False
        and record.get("training_or_update_called", False) is False
        and safety_known
        and not safety_nonzero
        and isinstance(lifecycle, Mapping)
        and row_ids_unique
        and runtime_partial_integrity
    )
    reward_components = summary.get("reward_component_totals", {})
    rejection_penalty = neutral_eval._finite(runtime.get("release_rejection_penalty"))
    if rejection_penalty is None and isinstance(reward_components, Mapping):
        # The native ``release`` component includes approved-release reward;
        # it is not a rejection-only penalty and is intentionally not used.
        rejection_penalty = None
    reward_total = neutral_eval._finite(summary.get("reward_total"))
    raw_reward = (
        None if reward_total is None or rejection_penalty is None
        else float(reward_total) - float(rejection_penalty)
    )
    row_waits = [neutral_eval._finite(row.get("wait_seconds")) for row in rows]
    observed_wait = float(sum(value for value in row_waits if value is not None)) if any(value is not None for value in row_waits) else None
    elapsed_values = []
    energy_values = []
    for row in rows:
        request_time = neutral_eval._finite(row.get("request_time"))
        terminal_time = neutral_eval._finite(row.get("terminal_time"))
        if request_time is not None and terminal_time is not None:
            elapsed_values.append(terminal_time - request_time)
        energy = neutral_eval._finite(row.get("energy_consumed"))
        if energy is not None:
            energy_values.append((int(row["arrival_id"]), energy))
    energy_unique = len({item[0] for item in energy_values}) == len(energy_values)
    observed_elapsed = float(sum(elapsed_values)) if elapsed_values else None
    observed_energy = float(sum(value for _, value in energy_values)) if energy_values and energy_unique else None
    lifecycle_energy = observed_energy
    if runtime.get("energy_total_consumed") is not None:
        observed_energy = neutral_eval._finite(runtime.get("energy_total_consumed"))
    row_metric_valid = bool(
        len(rows) == expected_count
        and row_ids_unique
        and sorted(row_ids) == expected_ids
        and all(
            (wait := neutral_eval._finite(row.get("wait_seconds"))) is not None
            and wait >= 0.0
            and (request_time := neutral_eval._finite(row.get("request_time"))) is not None
            and (terminal_time := neutral_eval._finite(row.get("terminal_time"))) is not None
            and terminal_time >= request_time
            and (energy := neutral_eval._finite(row.get("energy_consumed"))) is not None
            and energy >= 0.0
            for row in rows
        )
    )
    runtime_energy_match = bool(
        not runtime_present
        or (
            lifecycle_energy is not None
            and neutral_eval._finite(runtime.get("energy_total_consumed")) is not None
            and math.isclose(
                float(lifecycle_energy), float(runtime.get("energy_total_consumed")),
                rel_tol=0.0, abs_tol=1e-8,
            )
        )
    )
    complete_metric_valid = bool(row_metric_valid and runtime_energy_match)
    complete_scene_metrics_known = bool(
        lifecycle_complete and terminal_flag and not record.get("error") and complete_metric_valid
    )
    natural_metrics_invalid = bool(lifecycle_complete and terminal_flag and not complete_metric_valid)
    technical_ok = bool(technical_ok and not natural_metrics_invalid)
    outcome_counts = {key: outcomes_observed.get(key) for key in ("normal", "emergency", "failed", "diverted", "unresolved")}
    successful_observed = (
        int(outcome_counts["normal"] or 0) + int(outcome_counts["emergency"] or 0)
        if outcome_counts["normal"] is not None and outcome_counts["emergency"] is not None else None
    )
    resolved_observed = (
        sum(int(outcome_counts[key] or 0) for key in ("normal", "emergency", "failed", "diverted"))
        if all(outcome_counts[key] is not None for key in ("normal", "emergency", "failed", "diverted")) else None
    )
    generated = runtime.get("generated_requests") if runtime_present else expected_count
    admitted = runtime.get("admitted_count") if runtime_present else record.get("admitted_count", len(observed_ids) if observed_ids else None)
    full_or_none = lambda value: value if complete_scene_metrics_known else None
    return {
        "status": record.get("status"),
        "technical_ok": technical_ok,
        "natural_terminal": bool(complete_scene_metrics_known),
        "censored": bool(
            not complete_scene_metrics_known
            and not record.get("error")
            and not natural_metrics_invalid
        ),
        "complete_scene_metrics_known": complete_scene_metrics_known,
        "requests": expected_count,
        "generated": int(generated) if generated is not None else None,
        "admitted": int(admitted) if admitted is not None else None,
        "admitted_telemetry": (
            int(runtime.get("telemetry_arrivals_admitted"))
            if runtime.get("telemetry_arrivals_admitted") is not None else None
        ),
        "normal": outcome_counts["normal"],
        "emergency": outcome_counts["emergency"],
        "failed": outcome_counts["failed"],
        "diverted": outcome_counts["diverted"],
        "unresolved": outcome_counts["unresolved"],
        "outcome_counts_observed": outcome_counts,
        "successful": full_or_none(successful_observed),
        "resolved": full_or_none(resolved_observed),
        "request_identity": identity,
        "request_identity_ids": request_identity_ids,
        "request_identity_exact_24": identity_exact_24,
        "wait_seconds": full_or_none(observed_wait),
        "wait_seconds_observed": observed_wait,
        "request_elapsed_total_seconds": full_or_none(observed_elapsed),
        "request_elapsed_total_seconds_observed": observed_elapsed,
        "energy_total_consumed": full_or_none(observed_energy),
        "energy_total_consumed_observed": observed_energy,
        "simulation_seconds": neutral_eval._finite(summary.get("seconds")),
        "reward_total": reward_total if complete_scene_metrics_known else None,
        "reward_total_observed": reward_total,
        "raw_reward_excluding_rejection_penalty": raw_reward if complete_scene_metrics_known else None,
        "raw_reward_excluding_rejection_penalty_observed": raw_reward,
        "release_rejection_penalty": rejection_penalty if complete_scene_metrics_known else None,
        "release_rejection_penalty_observed": rejection_penalty,
        "hard_safety_clear": bool(safety_known and not safety_nonzero),
        "error": record.get("error"),
        "lifecycle_qualification_error": record.get("lifecycle_qualification_error"),
        "metrics": {
            "safety": dict(safety),
            "runtime_capture": {
                "available": runtime.get("available"),
                "factory_calls": runtime.get("factory_calls"),
                "ledger_alignment": runtime.get("ledger_alignment"),
                "partial_integrity": runtime_partial_integrity,
                "complete_integrity": runtime_complete_integrity,
                "energy_match": runtime_energy_match,
            },
            "lifecycle": {
                "scope": lifecycle.get("scope"),
                "replay_exact": lifecycle.get("replay_exact"),
                "all_requests_observed": lifecycle.get("all_requests_observed"),
                "row_count": lifecycle.get("row_count"),
                "missing_request_count": lifecycle.get("missing_request_count"),
            },
        },
    }


def _id_values(value: Any) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        values = [int(item) for item in value]
    except (TypeError, ValueError):
        return None
    return values if len(values) == len(set(values)) else None


def _request_identity(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    identity = []
    for row in sorted(rows, key=lambda item: int(item.get("arrival_id", -1))):
        identity.append({
            "arrival_id": _jsonable(row.get("arrival_id")),
            "request_time": _jsonable(row.get("request_time")),
            "mission_type": _jsonable(row.get("mission_type", row.get("mission"))),
            "initial_soc": _jsonable(row.get("initial_soc")),
            "energy_capacity": _jsonable(row.get("energy_capacity")),
        })
    return identity


def _observed_outcomes(record: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, int | None]:
    counts = {key: 0 for key in ("normal", "emergency", "failed", "diverted", "unresolved")}
    runtime = record.get("runtime_capture")
    runtime_outcomes = runtime.get("outcomes") if isinstance(runtime, Mapping) else None
    if isinstance(runtime_outcomes, Mapping):
        values = list(runtime_outcomes.values())
        for value in values:
            outcome = str(getattr(value, "value", value))
            if outcome == "completed":
                counts["normal"] += 1
            elif outcome == "emergency_landed":
                counts["emergency"] += 1
            elif outcome in {"failed_energy", "energy_depleted"} or outcome.startswith("failed"):
                counts["failed"] += 1
            elif outcome.startswith("diverted"):
                counts["diverted"] += 1
            elif outcome == "unresolved":
                counts["unresolved"] += 1
        requested_ids = _id_values(runtime.get("requested_arrival_ids"))
        outcome_ids = _id_values(runtime.get("outcome_arrival_ids"))
        if requested_ids is not None and outcome_ids is not None:
            # Requests that exist in the live ledger but have no terminal
            # outcome are observed pending/unresolved. Requests that have not
            # been generated yet remain outside this count.
            counts["unresolved"] += len(set(requested_ids) - set(outcome_ids))
        return counts
    seen = False
    for row in rows:
        if "outcome" not in row:
            continue
        seen = True
        outcome = str(row.get("outcome"))
        if outcome == "completed":
            counts["normal"] += 1
        elif outcome == "emergency_landed":
            counts["emergency"] += 1
        elif outcome in {"failed_energy", "energy_depleted"} or outcome.startswith("failed"):
            counts["failed"] += 1
        elif outcome.startswith("diverted"):
            counts["diverted"] += 1
        elif outcome == "unresolved":
            counts["unresolved"] += 1
    if seen:
        requested_ids = _id_values(
            (record.get("lifecycle") or {}).get("requested_arrival_ids")
            if isinstance(record.get("lifecycle"), Mapping) else None
        )
        terminal_ids = {
            int(row["arrival_id"]) for row in rows
            if "arrival_id" in row and "outcome" in row
        }
        if requested_ids is not None:
            counts["unresolved"] += len(set(requested_ids) - terminal_ids)
        return counts
    raw = record.get("outcome_counts")
    if not isinstance(raw, Mapping):
        return {key: None for key in counts}
    return {
        "normal": int(raw.get("normal", 0) or 0),
        "emergency": int(raw.get("emergency", 0) or 0),
        "failed": int(raw.get("failed_energy", raw.get("failed", 0)) or 0),
        "diverted": int(raw.get("diverted", 0) or 0),
        "unresolved": int(raw.get("unresolved", 0) or 0),
    }


def _issue_ids(record: Mapping[str, Any], kinds: set[str]) -> list[int] | None:
    rows = neutral_eval._record_rows(record)
    if not rows:
        return None
    values: list[int] = []
    for row in rows:
        outcome = str(row.get("outcome", row.get("outcome_label", "")))
        if outcome in kinds:
            try:
                values.append(int(row["arrival_id"]))
            except (KeyError, TypeError, ValueError):
                return None
    return sorted(set(values))


def _reclassify_incomplete_lifecycle(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep evaluator qualification errors separate from a valid prefix censor.

    ``_runtime_record`` adds an ``IncompleteLifecycle`` error for a nonterminal
    prefix even when the replay and hard-safety checks are valid.  That marker
    describes missing full-scene qualification metrics; it is not an
    independent replay failure.  Only this exact nonterminal case is
    reclassified, and the original error is retained verbatim.
    """

    normalized = dict(record)
    error = normalized.get("error")
    summary = normalized.get("summary")
    lifecycle = normalized.get("lifecycle")
    if not (
        isinstance(error, Mapping)
        and str(error.get("type")) == "IncompleteLifecycle"
        and isinstance(summary, Mapping)
        and summary.get("terminal") is False
        and isinstance(lifecycle, Mapping)
        and lifecycle.get("replay_exact") is True
        and normalized.get("weights_unchanged") is not False
        and normalized.get("teacher_called", False) is False
        and normalized.get("training_or_update_called", False) is False
    ):
        return normalized
    safety = neutral_eval._safety_snapshot(normalized)
    safety_fields = (
        "collisions", "nmac", "hard_separation_violations",
        "uncertified_execution", "SafetyInvariantError",
    )
    if (
        any(neutral_eval._finite(safety.get(key)) is None for key in safety_fields)
        or any(int(neutral_eval._finite(safety.get(key)) or 0) != 0 for key in safety_fields)
        or normalized.get("safety_nonzero") is True
    ):
        return normalized
    preserved_error = dict(error)
    normalized["lifecycle_qualification_error"] = preserved_error
    normalized["error"] = None
    normalized["lifecycle_error_reclassified"] = "censored_nonterminal_prefix"
    return normalized


def _arm_record(
    *,
    config: Any,
    cell: Mapping[str, Any],
    arm: str,
    output: Path,
) -> dict[str, Any]:
    if arm == "before_legacy":
        checkpoint, expected_sha = BEFORE_CHECKPOINT, BEFORE_CHECKPOINT_SHA256
    else:
        checkpoint, expected_sha = AFTER_CHECKPOINT, AFTER_CHECKPOINT_SHA256
    caller_before = _rng_snapshot()
    resolved = checkpoint
    policy = None
    before_digest = None
    runtime_capture: dict[str, Any] = {}
    audit: dict[str, Any] = {
        "calls": 0, "release_rows": 0, "source_q_immutable": True,
        "source_logits_immutable": True, "source_gate_unchanged": True,
        "k3_source_logits_ranked": True,
    }
    record_output = output / "rollouts" / f"cell_{int(cell['cell_id']):02d}_{arm}"
    try:
        resolved = _resolve_checkpoint(checkpoint, expected_sha)
        policy, _payload = rank_eval._load_policy(resolved, config)
        before_digest = rank_eval._model_digest(policy)
        if arm == "after_support":
            with _capture_runtime_env(runtime_capture):
                with _support_sampler(audit, epsilon=EPSILON):
                    record = rank_eval._runtime_record(
                        policy=policy, checkpoint=resolved, config=config,
                        scenario="dense24", seed=int(cell["env_seed"]),
                        policy_seed=int(cell["policy_seed"]), output=record_output, smoke=False,
                    )
        else:
            with _capture_runtime_env(runtime_capture):
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
            ),
            policy_seed=int(cell["policy_seed"]),
        )
    finally:
        _restore_rng(caller_before)
        caller_unchanged = _rng_equal(caller_before, _rng_snapshot())
    record = _reclassify_incomplete_lifecycle(record)
    after_digest = None if policy is None else rank_eval._model_digest(policy)
    try:
        runtime_snapshot = _live_runtime_snapshot(runtime_capture, config)
    except BaseException as exc:
        runtime_snapshot = {
            "factory_calls": int(runtime_capture.get("factory_calls", 0) or 0),
            "available": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    record = dict(record)
    record.update({
        "arm": arm,
        "cell": dict(cell),
        "checkpoint_declared": str(checkpoint),
        "checkpoint_sha256_expected": expected_sha,
        "checkpoint_sha256": _sha256(resolved),
        "config": _config_snapshot(config),
        "model_digest_before": before_digest,
        "model_digest_after": after_digest,
        "model_unchanged": bool(before_digest is not None and before_digest == after_digest),
        "caller_rng_unchanged": bool(caller_unchanged),
        "teacher_called": False,
        "training_or_update_called": False,
        "PPO_ALLOWED": False,
        "runtime_capture": runtime_snapshot,
        "sampler_intervention": ({
            "name": CONTROL_NAME,
            "epsilon": EPSILON,
            "q_floor_min": Q_FLOOR_MIN,
            "proposal_count": PROPOSAL_COUNT,
            "sampled_first": True,
            "source_gate_unchanged": bool(audit["source_gate_unchanged"]),
            "source_q_immutable": bool(audit["source_q_immutable"]),
            "source_logits_immutable": bool(audit["source_logits_immutable"]),
            "k3_source_logits_ranked": bool(audit["k3_source_logits_ranked"]),
            "likelihood_distribution": "epsilon_mixture_conditional_release",
            "off_policy": True,
            "training": False,
            **audit,
        } if arm == "after_support" else {
            "name": "native_legacy_student_request_details",
            "epsilon": 0.0,
            "source_gate_unchanged": True,
            "source_q_immutable": True,
            "source_logits_immutable": True,
            "k3_source_logits_ranked": True,
            "off_policy": False,
            "training": False,
        }),
    })
    record["metrics"] = _record_metrics(record)
    return record


def _pair_summary(cell: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(records) != set(ARMS):
        return {
            "cell": dict(cell),
            "status": "PAIR_TECHNICAL_ERROR",
            "arms": {
                key: {
                    "status": row.get("status"),
                    "metrics": _record_metrics(row),
                    "checkpoint_sha256": row.get("checkpoint_sha256"),
                    "model_unchanged": row.get("model_unchanged"),
                    "caller_rng_unchanged": row.get("caller_rng_unchanged"),
                    "safety_nonzero": row.get("safety_nonzero"),
                }
                for key, row in records.items()
            },
            "request_identity_match": None,
            "before_to_after_legacy": {"known": False, "values": {}},
            "support_to_after_legacy": {"known": False, "values": {}},
            "new_failed_arrival_ids": None,
            "new_diverted_arrival_ids": None,
            "new_unresolved_arrival_ids": None,
            "technical_error": "arm execution stopped before all registered arms completed",
            "source_model_and_physics_unchanged_scope": True,
        }
    before = records["before_legacy"]
    after = records["after_legacy"]
    support = records["after_support"]
    metrics = {name: _record_metrics(row) for name, row in records.items()}

    def delta(left: str, right: str) -> dict[str, Any]:
        left_m, right_m = metrics[left], metrics[right]
        known = bool(left_m.get("complete_scene_metrics_known") and right_m.get("complete_scene_metrics_known"))
        values: dict[str, Any] = {}
        for key in ("normal", "successful", "failed", "wait_seconds", "request_elapsed_total_seconds", "energy_total_consumed"):
            a, b = left_m.get(key), right_m.get(key)
            values[key] = None if not known or a is None or b is None else float(a) - float(b)
        return {"known": known and all(value is not None for value in values.values()), "values": values}

    def new_ids(left: Mapping[str, Any], right: Mapping[str, Any], kinds: set[str]) -> list[int] | None:
        if not (_record_metrics(left).get("complete_scene_metrics_known") and
                _record_metrics(right).get("complete_scene_metrics_known")):
            return None
        a, b = _issue_ids(left, kinds), _issue_ids(right, kinds)
        if a is None or b is None:
            return None
        return sorted(set(a) - set(b))

    return {
        "cell": dict(cell),
        "status": "PAIR_OK" if all(metrics[key].get("complete_scene_metrics_known") for key in records) else "PAIR_INCOMPLETE",
        "arms": {key: {"status": row.get("status"), "metrics": metrics[key],
                        "checkpoint_sha256": row.get("checkpoint_sha256"),
                        "model_unchanged": row.get("model_unchanged"),
                        "caller_rng_unchanged": row.get("caller_rng_unchanged"),
                        "safety_nonzero": row.get("safety_nonzero")} for key, row in records.items()},
        "request_identity_match": (
            all(metrics[key].get("request_identity_exact_24") for key in records)
            and len({json.dumps(metrics[key].get("request_identity"), sort_keys=True) for key in records}) == 1
        ),
        "before_to_after_legacy": delta("after_legacy", "before_legacy"),
        "support_to_after_legacy": delta("after_support", "after_legacy"),
        "new_failed_arrival_ids": new_ids(support, after, {"failed_energy", "failed", "energy_depleted"}),
        "new_diverted_arrival_ids": new_ids(support, after, {"diverted", "diverted_low_soc"}),
        "new_unresolved_arrival_ids": new_ids(support, after, {"unresolved"}),
        "source_model_and_physics_unchanged_scope": True,
    }


def _business_review(pair_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {(int(item["cell_id"]), int(item["env_seed"]), int(item["policy_seed"])) for item in CELL_SPECS}
    observed = {
        (int(row.get("cell", {}).get("cell_id", -1)), int(row.get("cell", {}).get("env_seed", -1)),
         int(row.get("cell", {}).get("policy_seed", -1)))
        for row in pair_summaries
    }
    grid_ok = len(pair_summaries) == len(CELL_SPECS) and observed == expected
    natural = bool(grid_ok and all(
        row.get("status") == "PAIR_OK"
        and set(row.get("arms", {})) == set(ARMS)
        and all(arm.get("metrics", {}).get("complete_scene_metrics_known") is True for arm in row.get("arms", {}).values())
        for row in pair_summaries
    ))
    technical = bool(grid_ok and all(
        set(row.get("arms", {})) == set(ARMS)
        and all(arm.get("metrics", {}).get("technical_ok") is True
            and arm.get("model_unchanged") is True
            and arm.get("caller_rng_unchanged") is True
            and arm.get("safety_nonzero") is not True
            and arm.get("metrics", {}).get("admitted") == 24
            and arm.get("metrics", {}).get("request_identity_exact_24") is True
            for arm in row.get("arms", {}).values())
        and row.get("request_identity_match") is True
        for row in pair_summaries
    ))
    no_new = bool(grid_ok and all(
        row.get(key) == []
        for row in pair_summaries
        for key in ("new_failed_arrival_ids", "new_diverted_arrival_ids", "new_unresolved_arrival_ids")
    ))

    def aggregate(arm_name: str) -> dict[str, Any]:
        arms = [row.get("arms", {}).get(arm_name, {}).get("metrics", {}) for row in pair_summaries]
        keys = (
            "generated", "admitted", "normal", "emergency", "failed", "diverted", "unresolved",
            "successful", "resolved", "wait_seconds", "request_elapsed_total_seconds", "energy_total_consumed",
        )
        known = bool(arms) and all(
            item.get("complete_scene_metrics_known") is True
            and all(item.get(key) is not None for key in keys)
            for item in arms
        )
        return {
            "known": known,
            "generated": None if not known else sum(int(item["generated"]) for item in arms),
            "admitted": None if not known else sum(int(item["admitted"]) for item in arms),
            "normal": None if not known else sum(int(item.get("normal", 0) or 0) for item in arms),
            "emergency": None if not known else sum(int(item.get("emergency", 0) or 0) for item in arms),
            "successful": None if not known else sum(int(item.get("successful", 0) or 0) for item in arms),
            "failed": None if not known else sum(int(item.get("failed", 0) or 0) for item in arms),
            "diverted": None if not known else sum(int(item.get("diverted", 0) or 0) for item in arms),
            "unresolved": None if not known else sum(int(item.get("unresolved", 0) or 0) for item in arms),
            "resolved": None if not known else sum(int(item.get("resolved", 0) or 0) for item in arms),
            "wait_seconds": None if not known else sum(float(item["wait_seconds"]) for item in arms),
            "request_elapsed_total_seconds": None if not known else sum(float(item["request_elapsed_total_seconds"]) for item in arms),
            "energy_total_consumed": None if not known else sum(float(item["energy_total_consumed"]) for item in arms),
            "reward_total": None if not known or any(item.get("reward_total") is None for item in arms)
            else sum(float(item["reward_total"]) for item in arms),
            "raw_reward_excluding_rejection_penalty": (
                None if not known or any(item.get("raw_reward_excluding_rejection_penalty") is None for item in arms)
                else sum(float(item["raw_reward_excluding_rejection_penalty"]) for item in arms)
            ),
            "release_rejection_penalty": (
                None if not known or any(item.get("release_rejection_penalty") is None for item in arms)
                else sum(float(item["release_rejection_penalty"]) for item in arms)
            ),
        }

    before_aggregate = aggregate("before_legacy")
    after_aggregate = aggregate("after_legacy")
    support_aggregate = aggregate("after_support")

    def compare(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
        keys = ("normal", "successful", "failed", "wait_seconds", "request_elapsed_total_seconds", "energy_total_consumed")
        known = bool(left.get("known") and right.get("known"))
        values = {key: (None if not known else float(left[key]) - float(right[key])) for key in keys}
        return {
            "known": known,
            "values": values,
            "normal_not_lower": bool(known and values["normal"] >= 0),
            "successful_not_lower": bool(known and values["successful"] >= 0),
            "wait_not_higher": bool(known and values["wait_seconds"] <= 0),
            "strict_business_gain": bool(known and any(
                values[key] > 0 if key in {"normal", "successful"} else values[key] < 0
                for key in keys
            )),
        }

    legacy_delta = compare(after_aggregate, before_aggregate)
    support_delta = compare(support_aggregate, after_aggregate)
    aggregate_pass = bool(
        support_delta["known"]
        and support_delta["normal_not_lower"]
        and support_delta["successful_not_lower"]
        and support_delta["wait_not_higher"]
        and support_delta["strict_business_gain"]
    )
    return {
        "pair_count": len(pair_summaries),
        "all_four_unique_registered_cells": grid_ok,
        "all_four_natural": natural,
        "all_technical_ok": technical,
        "no_new_failed_diverted_unresolved": no_new,
        "before_legacy_aggregate": before_aggregate,
        "after_legacy_aggregate": after_aggregate,
        "after_support_aggregate": support_aggregate,
        "before_to_after_legacy": legacy_delta,
        "support_to_after_legacy": support_delta,
        "aggregate_criterion_pass": aggregate_pass,
        "development_candidate": bool(natural and technical and no_new and aggregate_pass),
        "PPO_ALLOWED": False,
        "qualification": "not_a_qualification",
    }


def _progress(output: Path, payload: Mapping[str, Any]) -> None:
    _write_json(output / "progress.json", dict(payload))


def run_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    _require(not (output / "preflight.json").exists(), "preflight artifact already exists")
    payload = _build_preflight()
    _write_json(output / "preflight.json", payload, refuse_existing=True)
    _progress(output, {"stage": STAGE, "phase": "preflight", "status": "PREFLIGHT_OK", "environment_created": False})
    return payload


def _phase_inputs(output: Path) -> tuple[dict[str, Any], Any]:
    preflight_path = output / "preflight.json"
    preflight = _read_json(preflight_path)
    _validate_preflight(preflight)
    config, _configs = _build_inputs()
    return preflight, config


def _write_arm(output: Path, cell: Mapping[str, Any], arm: str, record: Mapping[str, Any]) -> Path:
    path = output / "arms" / f"cell_{int(cell['cell_id']):02d}_{arm}.json"
    payload = dict(record)
    payload.setdefault("metrics", _record_metrics(payload))
    _write_json(path, payload, refuse_existing=True)
    return path


def _run_cell(output: Path, config: Any, cell: Mapping[str, Any]) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        record = _arm_record(config=config, cell=cell, arm=arm, output=output)
        records[arm] = record
        _write_arm(output, cell, arm, record)
        technical_error = _technical_failure_arm(record)
        if technical_error:
            summary = _pair_summary(cell, records)
            summary["technical_error"] = technical_error
            _write_json(output / "pairs" / f"cell_{int(cell['cell_id']):02d}.json", summary, refuse_existing=True)
            return summary
    summary = _pair_summary(cell, records)
    _write_json(output / "pairs" / f"cell_{int(cell['cell_id']):02d}.json", summary, refuse_existing=True)
    return summary


def _technical_failure(summary: Mapping[str, Any]) -> str | None:
    if summary.get("technical_error"):
        return str(summary["technical_error"])
    for arm, payload in (summary.get("arms") or {}).items():
        if payload.get("status") in {"failed", "technical_error", "safety_violation"}:
            return f"{arm} status={payload.get('status')}"
        if payload.get("metrics", {}).get("error"):
            return f"{arm} metrics error={payload['metrics']['error']}"
        if payload.get("model_unchanged") is False or payload.get("caller_rng_unchanged") is False:
            return f"{arm} source or caller RNG changed"
        if payload.get("safety_nonzero") is True:
            return f"{arm} hard-safety flag is nonzero"
        metrics = payload.get("metrics") or {}
        if payload.get("status") == "completed" and metrics.get("technical_ok") is not True:
            return f"{arm} completed arm technical checks are incomplete"
    return None


def _technical_failure_arm(record: Mapping[str, Any]) -> str | None:
    """Check one arm before the next registered arm is started."""

    if record.get("error"):
        error = record.get("error")
        return str(error.get("message", "arm record error")) if isinstance(error, Mapping) else str(error)
    if record.get("status") in {"failed", "technical_error", "safety_violation"}:
        return f"arm status={record.get('status')}"
    if record.get("model_unchanged") is not True or record.get("caller_rng_unchanged") is not True:
        return "source model or caller RNG changed"
    if record.get("safety_nonzero") is True:
        return "hard-safety flag is nonzero"
    metrics = record.get("metrics") or _record_metrics(record)
    if metrics.get("technical_ok") is not True:
        return "arm technical fields are incomplete or invalid"
    return None


def run_pilot(output: Path) -> dict[str, Any]:
    preflight, config = _phase_inputs(output)
    _require(not (output / "pilot.json").exists(), "pilot artifact already exists")
    summary = _run_cell(output, config, CELL_SPECS[0])
    arm_paths = {
        arm: str(output / "arms" / f"cell_01_{arm}.json")
        for arm in (summary.get("arms") or {})
    }
    artifacts = {
        "pair_path": str(output / "pairs" / "cell_01.json"),
        "pair_sha256": _sha256(output / "pairs" / "cell_01.json"),
        "arm_paths": arm_paths,
        "arm_sha256": {arm: _sha256(path) for arm, path in arm_paths.items()},
    }
    technical_error = _technical_failure(summary)
    status = "PILOT_TECHNICAL_ERROR" if technical_error else (
        "PILOT_OK" if summary["status"] == "PAIR_OK" else "PILOT_INCOMPLETE"
    )
    payload = {
        "stage": STAGE, "phase": "pilot", "status": status,
        "PPO_ALLOWED": False, "preflight_sha256": _sha256(output / "preflight.json"),
        "preflight_runner_sha256": preflight["runner_sha256"],
        "preflight_binding": {key: preflight.get(key) for key in ("cells", "runtime", "epsilon", "q_floor_min", "proposal_count", "checkpoints")},
        "pair": summary, "pilot_artifacts": artifacts,
        "technical_error": technical_error,
        "qualification": "not_a_qualification",
    }
    _write_json(output / "pilot.json", payload, refuse_existing=True)
    _progress(output, payload)
    if technical_error:
        raise TrainedProposalSupportError(f"pilot stopped after technical failure: {technical_error}")
    return payload


def _validate_pilot_reuse(output: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    pilot = _read_json(output / "pilot.json")
    _require(pilot.get("status") in {"PILOT_OK", "PILOT_INCOMPLETE"}, "pilot is not reusable")
    _require(pilot.get("preflight_sha256") == _sha256(output / "preflight.json"), "pilot preflight changed")
    _require(pilot.get("preflight_runner_sha256") == preflight.get("runner_sha256"), "pilot runner binding changed")
    expected_binding = {key: preflight.get(key) for key in ("cells", "runtime", "epsilon", "q_floor_min", "proposal_count", "checkpoints")}
    _require(pilot.get("preflight_binding") == expected_binding, "pilot binding changed")
    artifacts = pilot.get("pilot_artifacts", {})
    pair_path = Path(str(artifacts.get("pair_path", "")))
    _require(pair_path.is_file(), "pilot pair artifact missing")
    _require(artifacts.get("pair_sha256") == _sha256(pair_path), "pilot pair artifact changed")
    for arm in ARMS:
        path = Path(str((artifacts.get("arm_paths") or {}).get(arm, "")))
        _require(path.is_file(), f"pilot {arm} artifact missing")
        _require((artifacts.get("arm_sha256") or {}).get(arm) == _sha256(path), f"pilot {arm} artifact changed")
        arm_payload = _read_json(path)
        _require(arm_payload.get("cell", {}).get("cell_id") == 1, "pilot arm cell changed")
        _require(arm_payload.get("metrics") == pilot.get("pair", {}).get("arms", {}).get(arm, {}).get("metrics"),
                 f"pilot {arm} metrics changed")
    return pilot


def run_full(output: Path) -> dict[str, Any]:
    preflight, config = _phase_inputs(output)
    pilot = _validate_pilot_reuse(output, preflight)
    _require(not (output / "full.json").exists(), "full artifact already exists")
    summaries = [dict(pilot["pair"])]
    for cell in CELL_SPECS[1:]:
        summary = _run_cell(output, config, cell)
        summaries.append(summary)
        technical_error = _technical_failure(summary)
        if technical_error:
            payload = {
                "stage": STAGE, "phase": "full", "status": "FULL_TECHNICAL_ERROR",
                "pair_count": len(summaries), "pair_summaries": summaries,
                "error": technical_error, "PPO_ALLOWED": False,
                "qualification": "not_a_qualification",
            }
            _write_json(output / "full.json", payload, refuse_existing=True)
            _progress(output, payload)
            raise TrainedProposalSupportError(f"full stopped after technical failure: {technical_error}")
    review = _business_review(summaries)
    payload = {
        "stage": STAGE, "phase": "full", "status": "FULL_OK",
        "pair_count": len(summaries), "pair_summaries": summaries,
        "business_review": review, "PPO_ALLOWED": False,
        "qualification": "not_a_qualification; current d449 native stochastic development comparison",
    }
    _write_json(output / "full.json", payload, refuse_existing=True)
    _progress(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "pilot", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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
