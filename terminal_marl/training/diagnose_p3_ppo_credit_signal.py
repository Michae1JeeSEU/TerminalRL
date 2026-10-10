"""Bounded PPO team-credit diagnostic for the frozen source03c0 replay.

This module deliberately replays one existing dense24 action tape (seed
873103) and evaluates the frozen critic and the existing ``team_gae`` helper.
It never samples a policy action, calls an optimizer, queries Teacher labels, or
starts a training loop.  The small pure helpers are kept public enough for
fixture tests to exercise the credit arithmetic without constructing the full
simulator.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from terminal_marl.algo import student_ppo_v4_2 as ppo
from terminal_marl.core.low_level_joint_action_v4_2 import hierarchical_from_normalized_v4_2
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import collect_p3_proposal_head_probe_v1 as collector
from terminal_marl.training import diagnose_p3_terminal_credit as terminal_credit
from terminal_marl.training import student_rollout_v4_2
from terminal_marl.training import terminal_lifecycle_metrics


STAGE = "P3_PPO_CREDIT_SIGNAL_DIAGNOSTIC_V1"
SOURCE_SEED = 873103
SOURCE_SCENARIO = "dense24"
SOURCE_SPLIT = "train"
MAX_REPLAY_STEPS = 1000
EXPECTED_REQUESTS = 24
OBSERVATION_DIM = 88
CRITICAL_UID = 42
CRITICAL_ARRIVAL_ID = 12
CRITICAL_STEPS = (102, 103, 104)
GAMMA_099 = 0.99
GAMMA_1 = 1.0
GAE_LAMBDA = 0.95


class PPOCreditSignalError(ValueError):
    """Raised when the bounded diagnostic input/output contract fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PPOCreditSignalError(message)


def _jsonable(value: Any) -> Any:
    """Convert diagnostics to strict JSON values, including numpy scalars."""
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
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    _require(path.is_file(), f"JSON input missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PPOCreditSignalError(f"invalid JSON input: {path}") from exc


def _map_value(mapping: Mapping[Any, Any] | None, uid: Any, default: Any = None) -> Any:
    if not isinstance(mapping, Mapping):
        return default
    if uid in mapping:
        return mapping[uid]
    text = str(uid)
    if text in mapping:
        return mapping[text]
    try:
        integer = int(uid)
    except (TypeError, ValueError):
        return default
    return mapping.get(integer, default)


def _phase_value(actor: Any) -> str:
    phase = getattr(actor, "phase", None)
    return str(getattr(phase, "value", phase))


def actor_mask_for_env(env: Any, uids: Sequence[str]) -> np.ndarray:
    """Return the same eligibility predicate used by ``student_rollout_v4_2``."""
    reservations = getattr(env, "future_release_reservations", {})
    active = getattr(env, "active", {})
    return np.asarray(
        [
            bool(
                (actor := active.get(int(uid))) is not None
                and getattr(actor, "committed_plan", None) is None
                and _phase_value(actor) in {"ring_outer", "ring_middle", "ring_inner"}
                and int(uid) not in reservations
            )
            for uid in uids
        ],
        dtype=bool,
    )


def actor_step_normalize(values: Sequence[float] | np.ndarray,
                         actor_mask: Sequence[bool] | np.ndarray) -> dict[str, Any]:
    """Normalize with statistics from actor steps only.

    The returned vector applies the actor-step mean/std to every timestep, the
    same convention used by the PPO update.  With zero actor steps it is left
    unchanged and explicitly marked as unavailable for actor normalization.
    """
    # ``team_gae`` returns float32 and the PPO update computes its actor
    # statistics in that dtype.  Keep this diagnostic numerically aligned with
    # that production path.
    array = np.asarray(values, dtype=np.float32)
    mask = np.asarray(actor_mask, dtype=bool)
    _require(array.ndim == 1 and mask.shape == array.shape, "values and actor_mask must be aligned vectors")
    actor_values = array[mask]
    if actor_values.size == 0:
        return {
            "values": array.astype(np.float32),
            "actor_count": 0,
            "mean": None,
            "std": None,
            "available": False,
        }
    mean = float(actor_values.mean())
    std = float(actor_values.std())
    normalized = (array - np.float32(mean)) / (np.float32(std) + np.float32(1e-8))
    return {
        "values": normalized.astype(np.float32),
        "actor_count": int(actor_values.size),
        "mean": mean,
        "std": std,
        "available": True,
    }


def mc_returns(rewards: Sequence[float] | np.ndarray, gamma: float) -> np.ndarray:
    """Compute complete-episode Monte Carlo returns without bootstrapping."""
    gamma = float(gamma)
    _require(0.0 <= gamma <= 1.0 and math.isfinite(gamma), "gamma must be finite in [0, 1]")
    values = np.asarray(rewards, dtype=np.float64)
    _require(values.ndim == 1 and np.isfinite(values).all(), "rewards must be a finite vector")
    result = np.zeros(values.size, dtype=np.float64)
    following = 0.0
    for index in range(values.size - 1, -1, -1):
        following = float(values[index]) + gamma * following
        result[index] = following
    return result


def discount_half_life(gamma: float) -> float | None:
    gamma = float(gamma)
    if gamma == 0.0:
        return 0.0
    if gamma == 1.0:
        return None
    _require(0.0 < gamma < 1.0, "gamma must be in [0, 1]")
    return float(math.log(0.5) / math.log(gamma))


def _world_from_observations(observations: Mapping[Any, Any]) -> tuple[list[str], np.ndarray]:
    uids = sorted(str(uid) for uid in observations)
    if not uids:
        return [], np.zeros((0, OBSERVATION_DIM), dtype=np.float32)
    world = np.asarray([observations[uid] if uid in observations else observations[int(uid)] for uid in uids], dtype=np.float32)
    _require(world.ndim == 2 and world.shape[1] == OBSERVATION_DIM, "world must be an 88D matrix")
    _require(bool(np.isfinite(world).all()), "world contains non-finite values")
    return uids, world


def _policy_release_probabilities(policy: Any, world: np.ndarray) -> np.ndarray:
    if len(world) == 0:
        return np.zeros(0, dtype=np.float32)
    with torch.no_grad():
        tensor = torch.as_tensor(world, dtype=torch.float32, device=policy.device)
        gate_dist, _ = policy.model.distributions(tensor)
        probabilities = gate_dist.probs[:, 1].detach().cpu().numpy().astype(np.float32)
    _require(probabilities.shape == (len(world),), "source release probabilities are misaligned")
    _require(bool(np.isfinite(probabilities).all()), "source release probabilities are non-finite")
    return probabilities


def _global_terminal(env: Any, info: Mapping[str, Any] | None = None) -> bool:
    """Parse complete simulator terminal state, with a fixture fallback."""
    arrivals = getattr(env, "arrivals", None)
    active = getattr(env, "active", None)
    admission = getattr(env, "admission", None)
    service = getattr(env, "service_uavs", None)
    if arrivals is not None and active is not None and admission is not None and service is not None:
        exhausted = arrivals.exhausted() if callable(getattr(arrivals, "exhausted", None)) else bool(getattr(arrivals, "exhausted", False))
        queue = getattr(getattr(admission, "queue", None), "items", ())
        return bool(exhausted and not active and not queue and not service)
    return bool((info or {}).get("terminal", False))


def _episode_end(env: Any, terminal: bool, info: Mapping[str, Any] | None = None) -> bool:
    if terminal:
        return True
    info = info or {}
    truncated = info.get("truncated", {})
    truncated_flag = bool(truncated) if isinstance(truncated, bool) else (
        any(bool(item) for item in truncated.values()) if isinstance(truncated, Mapping) else False
    )
    if bool(info.get("episode_end")) or truncated_flag:
        return True
    now = getattr(env, "now", None)
    config = getattr(env, "config", None)
    limit = getattr(config, "max_simulation_seconds", None)
    return bool(now is not None and limit is not None and float(now) >= float(limit))


def _action_details(tape_row: Mapping[str, Any], uid: str, raw_action: Any) -> tuple[int, int, float, float]:
    if raw_action is None:
        raw_action = np.zeros(3, dtype=np.float32)
    decoded = hierarchical_from_normalized_v4_2(np.asarray(raw_action, dtype=np.float32))
    gate = int(decoded.gate)
    proposal = -1 if decoded.proposal is None else int(decoded.proposal)
    stored_gate = _map_value(tape_row.get("requested_gate"), uid, gate)
    stored_proposal = _map_value(tape_row.get("requested_proposal"), uid, proposal)
    if stored_proposal is None:
        stored_proposal = -1
    _require(int(stored_gate) == gate, f"requested gate identity mismatch at uid={uid}")
    _require(int(stored_proposal) == proposal, f"requested proposal identity mismatch at uid={uid}")
    gate_lp = _map_value(tape_row.get("gate_logprob"), uid, 0.0)
    proposal_lp = _map_value(tape_row.get("proposal_logprob"), uid, None)
    gate_lp = float(gate_lp)
    proposal_lp = float(proposal_lp) if proposal == -1 and proposal_lp is not None else (float(proposal_lp) if proposal != -1 and proposal_lp is not None else (float("nan") if proposal == -1 else 0.0))
    _require(math.isfinite(gate_lp), f"gate logprob is non-finite at uid={uid}")
    if proposal != -1:
        _require(math.isfinite(proposal_lp), f"proposal logprob is non-finite at uid={uid}")
    return gate, proposal, gate_lp, proposal_lp


def _capture_reward_components(info: Mapping[str, Any], rewards: Mapping[Any, Any]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    by_uav: dict[str, dict[str, float]] = {}
    raw_components = info.get("reward_components", {})
    if isinstance(raw_components, Mapping):
        for uid, values in raw_components.items():
            if isinstance(values, Mapping):
                by_uav[str(uid)] = {str(name): float(value) for name, value in values.items()}
    # Keep a complete per-UAV row even if a fixture only supplies rewards.
    for uid, value in rewards.items():
        by_uav.setdefault(str(uid), {}).setdefault("total", float(value))
    totals: dict[str, float] = {}
    for values in by_uav.values():
        for name, value in values.items():
            if name == "total":
                continue
            totals[name] = totals.get(name, 0.0) + float(value)
    return totals, by_uav


def _step_from_capture(capture: Mapping[str, Any]) -> ppo.StudentStepV42:
    """Construct the real StudentStep container from one captured transition."""
    return ppo.StudentStepV42(
        np.asarray(capture["world"], dtype=np.float32),
        np.asarray(capture["requested_gate"], dtype=np.int64),
        np.asarray(capture["requested_proposal"], dtype=np.int64),
        np.asarray(capture["gate_logprob"], dtype=np.float32),
        np.asarray(capture["proposal_logprob"], dtype=np.float32),
        np.asarray(capture["actor_mask"], dtype=bool),
        float(capture["value"]),
        float(capture["next_value"]),
        float(capture["reward"]),
        bool(capture["terminated"]),
        bool(capture["episode_end"]),
        teacher_used=False,
        stochastic_behavior=False,
    )


def analyze_captures(captures: Sequence[Mapping[str, Any]], *, dt: float = 1.0) -> dict[str, Any]:
    """Analyze already-captured transitions without touching an environment."""
    ordered = [dict(row) for row in sorted(captures, key=lambda item: int(item["step"]))]
    for expected, row in enumerate(ordered):
        _require(int(row["step"]) == expected, "captured steps must start at zero and be contiguous")
    steps = [_step_from_capture(row) for row in ordered]
    actor_mask = np.asarray([bool(np.asarray(row["actor_mask"], dtype=bool).any()) for row in ordered], dtype=bool)
    # Keep MC and aggregate reward arithmetic in float64; StudentStep fields
    # remain float32 to match the production PPO container.
    rewards = np.asarray([float(row["reward"]) for row in ordered], dtype=np.float64)
    _require(bool(np.isfinite(rewards).all()), "captured rewards are non-finite")

    gae: dict[str, Any] = {}
    for label, gamma in (("gamma_0_99", GAMMA_099), ("gamma_1_0", GAMMA_1)):
        # This is intentionally the production helper, not a local GAE copy.
        raw_advantage, raw_return = ppo.team_gae(steps, gamma=gamma, gae_lambda=GAE_LAMBDA)
        norm_advantage = actor_step_normalize(raw_advantage, actor_mask)
        gae[label] = {
            "gamma": gamma,
            "lambda": GAE_LAMBDA,
            "raw": {"advantage": raw_advantage, "return": raw_return},
            "actor_step_normalized": {
                "advantage": norm_advantage["values"],
                "advantage_stats": {key: value for key, value in norm_advantage.items() if key != "values"},
            },
        }
    mc_discounted = mc_returns(rewards, GAMMA_099)
    mc_undiscounted = mc_returns(rewards, GAMMA_1)
    component_totals: dict[str, float] = {}
    reward_by_uav: dict[str, float] = {}
    for row in ordered:
        for name, value in (row.get("reward_components") or {}).items():
            component_totals[str(name)] = component_totals.get(str(name), 0.0) + float(value)
        for uid, value in (row.get("rewards_by_uav") or {}).items():
            reward_by_uav[str(uid)] = reward_by_uav.get(str(uid), 0.0) + float(value)
    means = {name: float(value) / len(ordered) for name, value in component_totals.items()} if ordered else {}
    critical: list[dict[str, Any]] = []
    for step_index in CRITICAL_STEPS:
        row = next((item for item in ordered if int(item["step"]) == step_index), None)
        _require(row is not None, f"critical step missing: {step_index}")
        identity = next((item for item in row.get("identities", []) if int(item.get("uid", -1)) == CRITICAL_UID), None)
        _require(identity is not None, f"critical uid missing at step {step_index}: uid={CRITICAL_UID}")
        _require(int(identity.get("arrival_id", -1)) == CRITICAL_ARRIVAL_ID, f"critical arrival mismatch at step {step_index}")
        gae_row = {
            "seed": SOURCE_SEED,
            "step": step_index,
            "uid": CRITICAL_UID,
            "arrival_id": CRITICAL_ARRIVAL_ID,
            "requested_gate": _map_value(row.get("requested_gate_by_uid"), str(CRITICAL_UID)),
            "source_release_probability": _map_value(row.get("release_probability_by_uid"), str(CRITICAL_UID)),
            "phase": _map_value(row.get("phase_by_uid"), str(CRITICAL_UID)),
            "actor_count": int(row.get("actor_count", 0)),
            "critic_value_pre": float(row["value"]),
            "critic_value_post": float(row["post_value"]),
            "mc_discounted_return_0_99": float(mc_discounted[step_index]),
            "mc_undiscounted_return": float(mc_undiscounted[step_index]),
            "gae_gamma_0_99_raw_advantage": float(gae["gamma_0_99"]["raw"]["advantage"][step_index]),
            "gae_gamma_0_99_normalized_advantage": float(gae["gamma_0_99"]["actor_step_normalized"]["advantage"][step_index]),
            "gae_gamma_0_99_raw_return": float(gae["gamma_0_99"]["raw"]["return"][step_index]),
            "gae_gamma_1_0_raw_advantage": float(gae["gamma_1_0"]["raw"]["advantage"][step_index]),
            "gae_gamma_1_0_normalized_advantage": float(gae["gamma_1_0"]["actor_step_normalized"]["advantage"][step_index]),
            "gae_gamma_1_0_raw_return": float(gae["gamma_1_0"]["raw"]["return"][step_index]),
        }
        critical.append(gae_row)
    return {
        "steps": len(ordered),
        "actor_steps": int(actor_mask.sum()),
        "reward_total": float(rewards.sum()),
        "reward_mean": None if not rewards.size else float(rewards.mean()),
        "reward_totals_by_uav": reward_by_uav,
        "reward_components": {
            "totals": component_totals,
            "means_per_step": means,
            "sum": float(sum(component_totals.values())),
            "sum_matches_reward_total": bool(np.isclose(sum(component_totals.values()), rewards.sum(), rtol=1e-6, atol=1e-6)),
        },
        "discount_half_life": {
            "gamma_0_99_steps": discount_half_life(GAMMA_099),
            "gamma_0_99_seconds": None if discount_half_life(GAMMA_099) is None else discount_half_life(GAMMA_099) * float(dt),
            "gamma_1_0_steps": discount_half_life(GAMMA_1),
        },
        "gae": gae,
        "mc_returns": {"discounted_gamma_0_99": mc_discounted, "undiscounted_gamma_1_0": mc_undiscounted},
        "critical_rows": critical,
        "terminal_steps": [int(row["step"]) for row in ordered if bool(row.get("terminated"))],
        "episode_end_steps": [int(row["step"]) for row in ordered if bool(row.get("episode_end"))],
    }


def _load_tape(
    path: Path,
    *,
    max_replay_steps: int = MAX_REPLAY_STEPS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        max_replay_steps = int(max_replay_steps)
    except (TypeError, ValueError) as exc:
        raise PPOCreditSignalError("max_replay_steps must be a positive integer") from exc
    _require(max_replay_steps > 0, "max_replay_steps must be a positive integer")
    payload = _read_json(path)
    _require(isinstance(payload, Mapping), "source tape must be a JSON object")
    rows = payload.get("action_tape", payload.get("records", []))
    _require(isinstance(rows, list) and rows, "source tape has no action rows")
    _require(len(rows) <= max_replay_steps, f"source tape exceeds {max_replay_steps} steps")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"source tape row {index} is not an object")
        value = dict(row)
        if "step" in value:
            _require(int(value["step"]) == index, f"source tape step mismatch at index {index}")
        else:
            value["step"] = index
        _require(isinstance(value.get("a_requested", {}), Mapping), f"source tape actions missing at step {index}")
        normalized.append(value)
    return dict(payload), normalized


def _source_record(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [dict(row) for row in records if int(row.get("seed", -1)) == SOURCE_SEED]
    _require(len(selected) == 1, "source seed873103 record is not unique")
    record = selected[0]
    _require(str(record.get("scenario")) == SOURCE_SCENARIO, "source scenario is not dense24")
    _require(str(record.get("split", SOURCE_SPLIT)) == SOURCE_SPLIT, "source seed is not in train split")
    _require(record.get("action_tape"), "source action tape path missing")
    return record


def _request_lifecycle(env: Any, captured: Sequence[Mapping[str, Any]], first_seen: Mapping[int, Mapping[str, Any]], requests: Sequence[Mapping[str, Any]], config: Any) -> dict[str, Any]:
    rows = terminal_lifecycle_metrics._merge_terminal_rows(
        captured,
        terminal_lifecycle_metrics._completed_rows(env, first_seen),
    )
    requested_count = int(getattr(config, "total_requested_uavs", EXPECTED_REQUESTS))
    _require(requested_count > 0, "request lifecycle must contain a positive request count")
    expected_ids = set(range(requested_count))
    ledger = getattr(env, "ledger", None)
    ledger_requested = {int(value) for value in getattr(ledger, "requested", set())}
    ledger_outcomes = {int(key): str(terminal_lifecycle_metrics._enum_value(value)) for key, value in getattr(ledger, "outcomes", {}).items()}
    observed_ids = {int(row["arrival_id"]) for row in rows}
    missing_ids = sorted(expected_ids - observed_ids)
    unexpected_ids = sorted(observed_ids - expected_ids)
    unresolved = sorted(ledger_requested - set(ledger_outcomes))
    rows_without_ledger = sorted(observed_ids - set(ledger_outcomes))
    mismatches = sorted(
        arrival_id for arrival_id, outcome in ledger_outcomes.items()
        if arrival_id in observed_ids
        and str(next(row for row in rows if int(row["arrival_id"]) == arrival_id).get("outcome")) != outcome
    )
    summary = terminal_lifecycle_metrics.summarize_terminal_rows(
        rows,
        low_soc_threshold=0.65,
        requested_count=requested_count,
        missing_arrival_ids=missing_ids,
        enforce_unique=True,
    )
    request_rows = [dict(item) for item in requests]
    by_arrival: dict[int, dict[str, Any]] = {}
    for item in request_rows:
        if "arrival_id" in item:
            by_arrival[int(item["arrival_id"])] = item
    for row in rows:
        arrival_id = int(row["arrival_id"])
        entry = by_arrival.setdefault(arrival_id, {"arrival_id": arrival_id})
        if row.get("mission_type") is not None:
            entry.setdefault("mission", row.get("mission_type"))
            entry.setdefault("mission_type", row.get("mission_type"))
    request_rows = [by_arrival[key] for key in sorted(by_arrival)]
    summary.update({
        "replay_exact": True,
        "terminal": _global_terminal(env),
        "requested_arrival_ids": sorted(expected_ids),
        "observed_arrival_ids": sorted(observed_ids),
        "ledger_requested_ids": sorted(ledger_requested),
        "ledger_outcomes": ledger_outcomes,
        "missing_arrival_ids": missing_ids,
        "unexpected_arrival_ids": unexpected_ids,
        "ledger_unresolved_requested_ids": unresolved,
        "rows_without_ledger_outcome": rows_without_ledger,
        "outcome_mismatch_arrival_ids": mismatches,
        "all_requests_observed": not missing_ids and not unexpected_ids and len(request_rows) == requested_count,
        "ledger_alignment": ledger_requested == expected_ids and not (missing_ids or unexpected_ids or unresolved or rows_without_ledger or mismatches),
        "request_rows": request_rows,
        "mission_field_present": all(row.get("mission_type") is not None for row in rows),
    })
    return summary


def _capture_replay(
    policy: Any,
    config: Any,
    tape_path: Path,
    tape_rows: Sequence[Mapping[str, Any]],
    *,
    progress_callback: Callable[[int], None] | None = None,
    state_observer: Callable[[Any], Mapping[str, Any]] | None = None,
    max_replay_steps: int = MAX_REPLAY_STEPS,
) -> dict[str, Any]:
    try:
        max_replay_steps = int(max_replay_steps)
    except (TypeError, ValueError) as exc:
        raise PPOCreditSignalError("max_replay_steps must be a positive integer") from exc
    _require(max_replay_steps > 0, "max_replay_steps must be a positive integer")
    _require(len(tape_rows) <= max_replay_steps, f"source tape exceeds {max_replay_steps} steps")
    env = TerminalParallelEnv(config)
    captured_terminal, first_seen = terminal_lifecycle_metrics._capture_terminal_rows(env)
    request_rows: list[dict[str, Any]] = []
    try:
        request_rows = terminal_credit._request_capture(env)
    except (AttributeError, TypeError):
        request_rows = []
    source_step = env.step
    captures: list[dict[str, Any]] = []

    def observed_step(*args: Any, **kwargs: Any):
        step_index = int(env.step_count)
        _require(step_index < max_replay_steps, f"replay exceeded {max_replay_steps} steps")
        _require(step_index < len(tape_rows), f"replay step has no source row: {step_index}")
        tape_row = tape_rows[step_index]
        pre_observations = env._observations()
        time_before = float(getattr(env, "now", 0.0))
        uids, world = _world_from_observations(pre_observations)
        actor_mask = actor_mask_for_env(env, uids)
        release_probabilities = _policy_release_probabilities(policy, world)
        value = float(student_rollout_v4_2.world_value(policy, world))
        state_context = None
        if state_observer is not None:
            state_context = state_observer(env)
            _require(isinstance(state_context, Mapping), "state_observer must return a mapping")
            state_context = dict(state_context)
        raw_actions = args[0] if args else kwargs.get("actions", {})
        raw_actions = raw_actions or {}
        gates: list[int] = []
        proposals: list[int] = []
        gate_lps: list[float] = []
        proposal_lps: list[float] = []
        identities: list[dict[str, Any]] = []
        phases: dict[str, str] = {}
        arrivals: dict[str, int] = {}
        for index, uid in enumerate(uids):
            action = _map_value(raw_actions, uid)
            tape_action = _map_value(tape_row.get("a_requested"), uid)
            if action is not None and tape_action is not None:
                _require(np.array_equal(np.asarray(action, dtype=np.float32), np.asarray(tape_action, dtype=np.float32)), f"source action mismatch at step={step_index}, uid={uid}")
            gate, proposal, gate_lp, proposal_lp = _action_details(tape_row, uid, action if action is not None else tape_action)
            gates.append(gate); proposals.append(proposal); gate_lps.append(gate_lp); proposal_lps.append(proposal_lp)
            actor = getattr(env, "active", {}).get(int(uid))
            if actor is not None:
                phase = _phase_value(actor)
                arrival_id = int(actor.arrival_id)
                phases[uid] = phase
                arrivals[uid] = arrival_id
                identities.append({"uid": int(uid), "arrival_id": arrival_id, "phase": phase})
        following, rewards, _terminated, _truncated, info = source_step(*args, **kwargs)
        post_uids, post_world = _world_from_observations(following)
        terminal = _global_terminal(env, info)
        episode_end = _episode_end(env, terminal, info)
        post_value = float(student_rollout_v4_2.world_value(policy, post_world))
        reward_total = float(sum(float(value) for value in rewards.values()))
        component_totals, components_by_uav = _capture_reward_components(info, rewards)
        _require(int(env.step_count) == step_index + 1, f"env step_count mismatch after step {step_index}")
        captures.append({
            "step": step_index,
            "time_before": time_before,
            "time_after": float(getattr(env, "now", 0.0)),
            "uids": uids,
            "world": world,
            "post_uids": post_uids,
            "post_world": post_world,
            "actor_mask": actor_mask,
            "actor_count": int(actor_mask.sum()),
            "requested_gate": np.asarray(gates, dtype=np.int64),
            "requested_proposal": np.asarray(proposals, dtype=np.int64),
            "gate_logprob": np.asarray(gate_lps, dtype=np.float32),
            "proposal_logprob": np.asarray(proposal_lps, dtype=np.float32),
            "requested_gate_by_uid": {uid: int(value) for uid, value in zip(uids, gates, strict=True)},
            "requested_proposal_by_uid": {uid: int(value) for uid, value in zip(uids, proposals, strict=True)},
            "release_probability_by_uid": {uid: float(value) for uid, value in zip(uids, release_probabilities, strict=True)},
            "phase_by_uid": phases,
            "arrival_by_uid": arrivals,
            "identities": identities,
            "value": value,
            "post_value": post_value,
            "next_value": 0.0 if terminal else post_value,
            "rewards_by_uav": {str(uid): float(item) for uid, item in rewards.items()},
            "reward": reward_total,
            "reward_components": component_totals,
            "reward_components_by_uav": components_by_uav,
            "terminated": terminal,
            "episode_end": episode_end,
        })
        if state_context is not None:
            captures[-1]["state_context"] = state_context
        if progress_callback is not None and (len(captures) % 25 == 0 or len(captures) == len(tape_rows)):
            progress_callback(len(captures))
        return following, rewards, _terminated, _truncated, info

    env.step = observed_step  # type: ignore[method-assign]
    replay = SafetyTraceRecorder.replay_action_tape(env, tape_path)
    env.step = source_step  # type: ignore[method-assign]
    _require(len(captures) == len(tape_rows), f"replay capture count mismatch: {len(captures)} != {len(tape_rows)}")
    _require(replay.get("exact") is True, f"source action-tape replay was not exact: {replay}")
    _require(int(replay.get("steps_replayed", -1)) == len(tape_rows), "source replay step count was not exact")
    lifecycle = _request_lifecycle(env, captured_terminal, first_seen, request_rows, config)
    return {"env": env, "replay": replay, "captures": captures, "lifecycle": lifecycle, "request_rows": request_rows}


def _model_digest(policy: Any) -> str:
    return str(collector.evaluator._model_digest(policy))


def _failed_report(output_dir: Path, error: BaseException, *, processed: int = 0) -> dict[str, Any]:
    result = {
        "stage": STAGE,
        "status": "FAILED",
        "output_dir": str(output_dir),
        "processed_steps": int(processed),
        "error": {"type": type(error).__name__, "message": str(error)},
        "PPO_ALLOWED": False,
        "training_called": False,
        "policy_sampling": False,
        "optimizer_used": False,
        "teacher_used": False,
    }
    _write_json(output_dir / "report.json", result)
    _write_json(output_dir / "progress.json", result)
    return result


def run(output_dir: Path, formal_root: Path = collector.FORMAL_ROOT) -> dict[str, Any]:
    """Run the single frozen replay and write trace/report/progress JSON."""
    output_dir = Path(output_dir)
    _require(not output_dir.exists(), f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "progress.json", {"stage": STAGE, "status": "RUNNING", "processed_steps": 0})
    parent_rng = terminal_credit._rng_snapshot()
    processed = 0
    try:
        protocol, configs, records, manifest_path = collector._load_source_inputs(Path(formal_root))
        record = _source_record(records)
        config = configs[SOURCE_SCENARIO]
        _require(int(getattr(config, "total_requested_uavs", -1)) == EXPECTED_REQUESTS, "dense24 config must contain exactly 24 requests")
        source_policy = collector._load_source_policy(protocol, configs[SOURCE_SCENARIO])
        policy = source_policy["policy"]
        digest_before = str(
            source_policy["model_digest_before"]
            if "model_digest_before" in source_policy
            else _model_digest(policy)
        )
        tape_path = Path(str(record["action_tape"]))
        _payload, tape_rows = _load_tape(tape_path)
        def update_progress(count: int) -> None:
            _write_json(output_dir / "progress.json", {
                "stage": STAGE,
                "status": "RUNNING",
                "processed_steps": int(count),
                "expected_steps": len(tape_rows),
                "source_seed": SOURCE_SEED,
            })

        replay_result = _capture_replay(
            policy,
            config,
            tape_path,
            tape_rows,
            progress_callback=update_progress,
        )
        captures = replay_result["captures"]
        processed = len(captures)
        analysis = analyze_captures(captures, dt=float(config.dt))
        _write_json(output_dir / "trace.json", {
            "stage": STAGE,
            "source_seed": SOURCE_SEED,
            "scenario": SOURCE_SCENARIO,
            "split": SOURCE_SPLIT,
            "action_tape": str(tape_path),
            "steps_expected": len(tape_rows),
            "steps_captured": len(captures),
            "replay_exact": bool(replay_result["replay"].get("exact")),
            "steps": captures,
        })
        digest_after = _model_digest(policy)
        rng_restored = False
        terminal_credit._restore_rng(parent_rng)
        rng_restored = terminal_credit._rng_equal(parent_rng, terminal_credit._rng_snapshot())
        lifecycle = replay_result["lifecycle"]
        _require(bool(lifecycle.get("terminal")), "replay did not reach complete terminal state")
        _require(bool(lifecycle.get("all_requests_observed")), "not all request lifecycle rows were observed")
        _require(bool(lifecycle.get("ledger_alignment")), "ledger/request lifecycle alignment failed")
        _require(len(lifecycle.get("request_rows", [])) == EXPECTED_REQUESTS, "24 request lifecycle rows are required")
        _require(all(row.get("mission_type") is not None or row.get("mission") is not None for row in lifecycle.get("request_rows", [])), "request mission field missing")
        _require(all(int(item["step"]) in {102, 103, 104} for item in analysis["critical_rows"]), "critical rows are not fixed steps")
        status = "COMPLETED" if digest_before == digest_after and rng_restored and analysis["steps"] == len(tape_rows) else "FAILED"
        report = {
            "stage": STAGE,
            "status": status,
            "source": {
                "seed": SOURCE_SEED,
                "scenario": SOURCE_SCENARIO,
                "split": SOURCE_SPLIT,
                "action_tape": str(tape_path),
                "source_manifest": str(manifest_path),
                "steps_expected": len(tape_rows),
                "steps_replayed": replay_result["replay"].get("steps_replayed"),
                "replay_exact": bool(replay_result["replay"].get("exact")),
            },
            "source_model_digest_before": digest_before,
            "source_model_digest_after": digest_after,
            "source_model_digest_unchanged": digest_before == digest_after,
            "rng_restored": rng_restored,
            "source_checkpoint": source_policy.get("checkpoint"),
            "source_checkpoint_sha256": source_policy.get("sha256", collector.SOURCE_CHECKPOINT_SHA256),
            "safety_snapshot": terminal_credit._safety_snapshot(replay_result["env"]),
            "lifecycle": lifecycle,
            "analysis": analysis,
            "scope": {
                "one_source_tape": True,
                "complete_replay_only": True,
                "team_gae_is_source_replay_diagnostic": True,
                "historical_training_tensor_claim": False,
                "individual_counterfactual_advantage_claim": False,
                "policy_sampling": False,
                "optimizer_used": False,
                "teacher_used": False,
                "training_called": False,
                "heldout_used": False,
                "PPO_ALLOWED": False,
            },
        }
        _write_json(output_dir / "report.json", report)
        _write_json(output_dir / "progress.json", {
            "stage": STAGE,
            "status": status,
            "processed_steps": len(captures),
            "expected_steps": len(tape_rows),
            "source_seed": SOURCE_SEED,
            "replay_exact": bool(replay_result["replay"].get("exact")),
        })
        return report
    except BaseException as exc:
        terminal_credit._restore_rng(parent_rng)
        return _failed_report(output_dir, exc, processed=processed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, default=collector.FORMAL_ROOT)
    args = parser.parse_args()
    result = run(output_dir=args.output_dir, formal_root=args.formal_root)
    print(json.dumps(_jsonable({"stage": STAGE, "status": result.get("status"), "output_dir": str(args.output_dir)}), sort_keys=True), flush=True)
    return 0 if result.get("status") == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
