"""Deterministic paired counterfactual diagnostics for the V4.2 gate.

The helpers in this module are deliberately diagnostic only.  They use a
frozen policy, keyed (UID based) draws, and deep copied environments.  No
optimizer or policy update is called.  The implementation keeps the first
action of every counterfactual explicit and reports censored continuations
when the environment does not reach its real terminal condition.
"""
from __future__ import annotations

import copy
import hashlib
import math
import pickle
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from terminal_marl.core.low_level_joint_action_v4_2 import (
    HierarchicalActionV42,
    N_RELEASE_PROPOSALS_V4_2,
    RELEASE_GATE,
    WAIT_GATE,
)
from terminal_marl.training import p3_action_credit
from terminal_marl.training import student_rollout_v4_2


PROPOSAL_COUNT = int(N_RELEASE_PROPOSALS_V4_2)
DIAGNOSTIC_SCHEMA = "paired_gate_diagnostic_v1"


def _jsonable(value: Any) -> Any:
    """Convert diagnostic values to strict JSON compatible values."""

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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _rng_snapshot() -> tuple[Any, tuple[Any, ...], torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def _restore_rng(snapshot: tuple[Any, tuple[Any, ...], torch.Tensor]) -> None:
    random.setstate(snapshot[0])
    np.random.set_state(snapshot[1])
    torch.random.set_rng_state(snapshot[2].clone())


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


def keyed_uniform(seed: int, step: int, uid: Any, channel: str) -> float:
    """Return a stable UID keyed uniform variate in ``[0, 1)``.

    The digest input includes the channel so a gate draw can never consume or
    shift a proposal draw.  The helper intentionally does not touch any
    process wide random number generator.
    """

    if channel not in {"gate", "proposal"}:
        raise ValueError("channel must be 'gate' or 'proposal'")
    payload = f"{int(seed)}|{int(step)}|{uid!s}|{channel}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
    # Use the same 53-bit construction as a double precision RNG.  A direct
    # float(value / 2**64) can round the largest uint64 to exactly one.
    return float((value >> 11) / float(1 << 53))


def _policy_model(policy: Any) -> Any:
    return getattr(policy, "model", policy)


def _ensure_cpu(policy: Any) -> None:
    """Reject CUDA policies so the diagnostic has one reproducible RNG path."""

    device = getattr(policy, "device", None)
    if device is not None and torch.device(device).type != "cpu":
        raise ValueError("paired gate diagnostic supports CPU policies only; CUDA is not supported")
    model = _policy_model(policy)
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            if parameter.device.type != "cpu":
                raise ValueError("paired gate diagnostic supports CPU policies only; CUDA is not supported")


def _seed_from_env(env: Any) -> int:
    for name in ("_paired_seed", "paired_seed", "policy_seed", "episode_seed", "seed"):
        value = getattr(env, name, None)
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return int(value)
    return 0


def _step_from_env(env: Any) -> int:
    for name in ("step_count", "absolute_step", "tick"):
        value = getattr(env, name, None)
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return int(value)
    return 0


def _observations(env: Any) -> dict[Any, np.ndarray]:
    method = getattr(env, "_observations", None)
    if callable(method):
        value = method()
    else:
        value = getattr(env, "observations", {})
    if not isinstance(value, Mapping):
        raise TypeError("environment observations must be a mapping")
    return {key: np.asarray(item, dtype=np.float32).copy() for key, item in value.items()}


def _details_context(policy: Any, env: Any) -> dict[str, Any]:
    """Get the canonical pre-action mask/world and deterministic source view."""

    _ensure_cpu(policy)
    observations = _observations(env)
    details = student_rollout_v4_2._student_request_details(
        policy,
        env,
        dict(observations),
        deterministic=True,
        shortlist_k=3,
    )
    if not isinstance(details, Sequence) or len(details) < 3:
        raise ValueError("_student_request_details returned an invalid result")
    uids = [str(uid) for uid in details[0]]
    raw_world = np.asarray(details[1], dtype=np.float32)
    dimension = int(
        getattr(policy, "observation_dim", 0)
        or (raw_world.shape[-1] if raw_world.ndim == 2 and raw_world.shape[-1] else 0)
        or np.asarray(next(iter(observations.values()), np.zeros(0))).size
    )
    if raw_world.size == 0:
        world = np.empty((0, dimension), dtype=np.float32)
    elif raw_world.ndim == 1:
        world = raw_world.reshape(len(uids), -1)
    else:
        world = raw_world.reshape(len(uids), -1)
    actor_mask = np.asarray(details[2], dtype=bool).reshape(-1)
    if len(uids) != len(actor_mask) or world.shape[0] != len(uids):
        raise ValueError("_student_request_details UID, world, and actor mask lengths differ")
    if not uids:
        return {
            "observations": observations,
            "details": details,
            "uids": [],
            "world": world,
            "actor_mask": actor_mask,
            "gate_logits": np.empty((0, 2), dtype=np.float64),
            "proposal_logits": np.empty((0, PROPOSAL_COUNT), dtype=np.float64),
            "gate_probabilities": np.empty((0, 2), dtype=np.float64),
            "proposal_probabilities": np.empty((0, PROPOSAL_COUNT), dtype=np.float64),
            "step": _step_from_env(env),
            "seed": _seed_from_env(env),
        }
    model = _policy_model(policy)
    distributions = getattr(model, "distributions", None)
    if not callable(distributions):
        raise TypeError("policy model must expose distributions(observations)")
    tensor = torch.as_tensor(world, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        gate_dist, proposal_dist = distributions(tensor)
        gate_logits = gate_dist.logits.detach().cpu().numpy().astype(np.float64, copy=True)
        proposal_logits = proposal_dist.logits.detach().cpu().numpy().astype(np.float64, copy=True)
        gate_probabilities = gate_dist.probs.detach().cpu().numpy().astype(np.float64, copy=True)
        proposal_probabilities = proposal_dist.probs.detach().cpu().numpy().astype(np.float64, copy=True)
    if gate_logits.shape != (len(uids), 2):
        raise ValueError(f"gate logits must have shape ({len(uids)}, 2)")
    if proposal_logits.shape != (len(uids), PROPOSAL_COUNT):
        raise ValueError(f"proposal logits must have shape ({len(uids)}, {PROPOSAL_COUNT})")
    return {
        "observations": observations,
        "details": details,
        "uids": uids,
        "world": world,
        "actor_mask": actor_mask,
        "gate_logits": gate_logits,
        "proposal_logits": proposal_logits,
        "gate_probabilities": gate_probabilities,
        "proposal_probabilities": proposal_probabilities,
        "step": _step_from_env(env),
        "seed": _seed_from_env(env),
    }


def _cdf_sample(probabilities: Sequence[float], uniform: float) -> int:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("categorical probabilities must be finite and one dimensional")
    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("categorical probabilities must have positive mass")
    cdf = np.cumsum(values / total)
    return int(min(np.searchsorted(cdf, float(uniform), side="right"), len(values) - 1))


def _shortlist_from_logits(selected: int, logits: Sequence[float]) -> tuple[np.ndarray, ...]:
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (PROPOSAL_COUNT,) or not np.isfinite(values).all():
        raise ValueError(f"proposal logits must have shape ({PROPOSAL_COUNT},) and be finite")
    ranked = np.argsort(-values, kind="stable").tolist()
    remaining = [int(value) for value in ranked if int(value) != int(selected)]
    ordered = [int(selected), remaining[0], remaining[1]]
    return tuple(HierarchicalActionV42(RELEASE_GATE, index).normalized.copy() for index in ordered)


@dataclass
class KeyedRequest:
    """Keyed action request; two-item iteration preserves a small call API."""

    actions: dict[str, np.ndarray]
    shortlists: dict[str, tuple[np.ndarray, ...]]
    details: dict[str, Any]

    def __iter__(self):
        yield self.actions
        yield self.shortlists

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> Any:
        if index == 0:
            return self.actions
        if index == 1:
            return self.shortlists
        raise IndexError(index)


def keyed_request(policy: Any, env: Any) -> KeyedRequest:
    """Generate a policy request with draws keyed by absolute tick and UID."""

    caller_rng = _rng_snapshot()
    try:
        context = _details_context(policy, env)
        actions: dict[str, np.ndarray] = {}
        shortlists: dict[str, tuple[np.ndarray, ...]] = {}
        seed = int(context["seed"])
        step = int(context["step"])
        for index, uid in enumerate(context["uids"]):
            if not bool(context["actor_mask"][index]):
                actions[uid] = HierarchicalActionV42(WAIT_GATE, None).normalized.copy()
                continue
            gate = _cdf_sample(
                context["gate_probabilities"][index],
                keyed_uniform(seed, step, uid, "gate"),
            )
            if gate == WAIT_GATE:
                actions[uid] = HierarchicalActionV42(WAIT_GATE, None).normalized.copy()
                continue
            proposal = _cdf_sample(
                context["proposal_probabilities"][index],
                keyed_uniform(seed, step, uid, "proposal"),
            )
            actions[uid] = HierarchicalActionV42(RELEASE_GATE, proposal).normalized.copy()
            shortlists[uid] = _shortlist_from_logits(proposal, context["proposal_logits"][index])
        details = {
            "schema": DIAGNOSTIC_SCHEMA,
            "uids": list(context["uids"]),
            "world": context["world"].copy(),
            "actor_mask": context["actor_mask"].copy(),
            "step": step,
            "seed": seed,
            "gate_logits": context["gate_logits"].copy(),
            "proposal_logits": context["proposal_logits"].copy(),
            "gate_probabilities": context["gate_probabilities"].copy(),
            "proposal_probabilities": context["proposal_probabilities"].copy(),
        }
        return KeyedRequest(actions, shortlists, details)
    finally:
        _restore_rng(caller_rng)


def _uid_lookup(mapping: Mapping[Any, Any], uid: Any, default: Any = None) -> Any:
    if uid in mapping:
        return mapping[uid]
    if str(uid) in mapping:
        return mapping[str(uid)]
    try:
        integer = int(uid)
    except (TypeError, ValueError):
        integer = None
    if integer is not None and integer in mapping:
        return mapping[integer]
    return default


def _request_signature(request: Any) -> dict[str, Any]:
    mission = getattr(request, "mission_type", None)
    return {
        "arrival_id": int(getattr(request, "arrival_id")),
        "request_time": float(getattr(request, "request_time")),
        "initial_soc": float(getattr(request, "initial_soc")),
        "base_priority": int(getattr(request, "base_priority")),
        "gate_id": int(getattr(request, "gate_id")),
        "mission": getattr(mission, "value", mission),
    }


def _install_request_capture(env: Any) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Install the existing request capture hook and retain per-step groups."""

    try:
        from terminal_marl.training import diagnose_p3_terminal_credit as terminal_credit

        flat = terminal_credit._request_capture(env)
    except (AttributeError, ImportError, TypeError):
        flat = []
        arrivals = getattr(env, "arrivals", None)
        original = getattr(arrivals, "requests_until", None)
        if not callable(original):
            return flat, []

        def requests_until(now: float):
            requests = original(now)
            flat.extend(_request_signature(request) for request in requests)
            return requests

        arrivals.requests_until = requests_until
    return flat, []


def _capture_tick_requests(flat: list[dict[str, Any]], before: int) -> list[dict[str, Any]]:
    return [copy.deepcopy(item) for item in flat[before:]]


def _real_terminal(env: Any) -> bool:
    arrivals = getattr(env, "arrivals", None)
    exhausted = getattr(arrivals, "exhausted", False)
    exhausted_value = bool(exhausted() if callable(exhausted) else exhausted)
    queue = getattr(getattr(env, "admission", None), "queue", None)
    queue_items = getattr(queue, "items", [])
    return bool(exhausted_value and not getattr(env, "active", {}) and not queue_items and not getattr(env, "service_uavs", {}))


def _truncation_true(value: Any) -> bool:
    """Read scalar, mapping, sequence, and ndarray truncation conventions."""

    if isinstance(value, Mapping):
        return any(_truncation_true(item) for item in value.values())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_truncation_true(item) for item in value)
    if value is None:
        return False
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _environment_time_limit_reached(env: Any) -> bool:
    config = getattr(env, "config", None)
    limit = getattr(config, "max_simulation_seconds", None)
    now = getattr(env, "now", None)
    try:
        return bool(
            limit is not None
            and now is not None
            and math.isfinite(float(limit))
            and float(now) >= float(limit) - 1e-9
        )
    except (TypeError, ValueError):
        return False


def _continuation_stop_reason(
    env: Any,
    truncated: Any,
    steps: int,
    max_steps: int,
) -> tuple[str | None, bool]:
    """Return ``(reason, actual_terminal)`` for the current branch state."""

    if _real_terminal(env):
        return "terminal", True
    if _environment_time_limit_reached(env):
        return "environment_time_limit", False
    if _truncation_true(truncated):
        return "environment_truncated", False
    if int(steps) >= int(max_steps):
        return "max_steps", False
    return None, False


def _call_step(env: Any, actions: Mapping[Any, Any], shortlists: Mapping[Any, Any]):
    try:
        return env.step(actions, proposal_shortlists=shortlists)
    except TypeError as exc:
        if "proposal_shortlists" not in str(exc):
            raise
        return env.step(actions, shortlists)


def _step_reward(rewards: Any) -> float:
    if isinstance(rewards, Mapping):
        return float(sum(float(value) for value in rewards.values()))
    return float(rewards)


def _decision_for(info: Any, uid: Any) -> Any:
    if not isinstance(info, Mapping):
        return None
    decisions = info.get("release_decisions", {})
    return _uid_lookup(decisions, uid)


def _runtime_rank_for(info: Any, uid: Any) -> Any:
    if not isinstance(info, Mapping):
        return None
    telemetry = _uid_lookup(info.get("release_intent_telemetry", {}), uid, {})
    if isinstance(telemetry, Mapping):
        return telemetry.get("runtime_shortlist_rank")
    return None


def _telemetry_snapshot(env: Any) -> dict[str, float]:
    values = getattr(env, "telemetry", {})
    if not isinstance(values, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, value in values.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _telemetry_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    return {
        key: float(after.get(key, 0.0) - before.get(key, 0.0))
        for key in sorted(set(before) | set(after))
    }


def _outcome_counts(env: Any) -> dict[str, int]:
    counts = {"normal": 0, "emergency": 0, "failed": 0, "diverted": 0, "unresolved": 0}
    ledger = getattr(env, "ledger", None)
    outcomes = getattr(ledger, "outcomes", {})
    if isinstance(outcomes, Mapping):
        for value in outcomes.values():
            name = str(getattr(value, "value", value)).lower()
            if name == "completed":
                counts["normal"] += 1
            elif "emergency" in name:
                counts["emergency"] += 1
            elif "failed" in name:
                counts["failed"] += 1
            elif "divert" in name:
                counts["diverted"] += 1
    requested = getattr(ledger, "requested", set())
    if isinstance(requested, (set, frozenset, list, tuple)):
        counts["unresolved"] = max(0, len(requested) - len(outcomes))
    return counts


def _request_elapsed_integral(env: Any) -> float:
    request_times = getattr(env, "request_times", {})
    resolution_times = getattr(env, "resolution_times", {})
    now = float(getattr(env, "now", 0.0))
    if not isinstance(request_times, Mapping):
        return 0.0
    total = 0.0
    for arrival_id, request_time in request_times.items():
        try:
            start = float(request_time)
            end = float(resolution_times.get(arrival_id, now))
            total += max(0.0, end - start)
        except (TypeError, ValueError):
            continue
    return float(total)


def _state_fingerprint(value: Any) -> tuple[str, str]:
    """Return a digest and its evidence source for an environment snapshot."""

    try:
        from terminal_marl.core.safety_trace import SafetyTraceRecorder

        state = SafetyTraceRecorder._state(value)
        ledger = getattr(value, "ledger", None)
        queue = getattr(getattr(getattr(value, "admission", None), "queue", None), "items", [])
        request_times = getattr(value, "request_times", {})
        resolution_times = getattr(value, "resolution_times", {})
        payload = {
            "safety_trace_state": state,
            "request_times": _jsonable(request_times),
            "resolution_times": _jsonable(resolution_times),
            "ledger_requested": _jsonable(getattr(ledger, "requested", set())),
            "ledger_outcomes": _jsonable(getattr(ledger, "outcomes", {})),
            "queue_arrival_ids": [
                int(getattr(getattr(item, "request", item), "arrival_id"))
                for item in queue
                if hasattr(getattr(item, "request", item), "arrival_id")
            ],
            "service_uav_ids": sorted(int(key) for key in getattr(value, "service_uavs", {})),
            "energy_outcomes": _jsonable(getattr(value, "energy_outcomes", [])),
        }
        return hashlib.sha256(pickle.dumps(payload, protocol=4)).hexdigest(), "safety_trace_plus_ledgers"
    except Exception:
        try:
            return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest(), "pickle"
        except Exception:
            attrs = getattr(value, "__dict__", None)
            if isinstance(attrs, Mapping):
                compact = {
                    key: _jsonable(item)
                    for key, item in attrs.items()
                    if key not in {"_step_observer"}
                }
                return hashlib.sha256(repr(compact).encode("utf-8")).hexdigest(), "repr_fallback"
            return hashlib.sha256(repr(value).encode("utf-8")).hexdigest(), "repr_fallback"


def _state_snapshot(model: Any) -> dict[str, torch.Tensor]:
    state = getattr(model, "state_dict", None)
    if not callable(state):
        return {}
    return {str(key): value.detach().cpu().clone() for key, value in state().items()}


def _state_unchanged(before: Mapping[str, torch.Tensor], model: Any) -> bool:
    after = _state_snapshot(model)
    return list(before) == list(after) and all(
        before[key].shape == after[key].shape and torch.equal(before[key], after[key])
        for key in before
    )


def _context_target(context: Mapping[str, Any], target_uid: Any) -> tuple[int, str]:
    uid_text = str(target_uid)
    for index, uid in enumerate(context["uids"]):
        if str(uid) == uid_text:
            return index, uid
    raise ValueError(f"target UID {target_uid!r} is absent from the actor context")


def _branch_outcome(
    *,
    policy: Any,
    env_snapshot: Any,
    first_actions: Mapping[Any, Any],
    first_shortlists: Mapping[Any, Any],
    target_uid: Any,
    proposal: int | None,
    logits: Sequence[float],
    seed: int,
    max_steps: int,
    branch_name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "branch": branch_name,
        "proposal": proposal,
        "first_step": {},
        "terminal": False,
        "censored": False,
        "suffix_undiscounted_reward": 0.0,
        "undiscounted_return": 0.0,
        "steps": 0,
        "external_request_signature": [],
        "external_requests_by_tick": [],
    }
    branch_env = None
    try:
        branch_env = copy.deepcopy(env_snapshot)
        # Copied bound methods can retain a source instance in diagnostics;
        # use the existing repair hook when it is available.
        try:
            from terminal_marl.training import diagnose_p3_k3_coverage_v1 as coverage

            coverage._restore_shadow_methods(branch_env)
        except (AttributeError, ImportError, TypeError):
            pass
        setattr(branch_env, "_paired_seed", int(seed))
        flat, _ = _install_request_capture(branch_env)
        branch_actions, branch_shortlists = p3_action_credit.branch_inputs(
            first_actions,
            first_shortlists,
            int(target_uid),
            proposal,
            logits,
            shortlist_k=3,
        )
        safety_before = _telemetry_snapshot(branch_env)
        elapsed_before = _request_elapsed_integral(branch_env)
        tick_before = len(flat)
        following, rewards, terminated, truncated, info = _call_step(branch_env, branch_actions, branch_shortlists)
        first_requests = _capture_tick_requests(flat, tick_before)
        first_reward = _step_reward(rewards)
        result["first_step"] = {
            "decision": _decision_for(info, target_uid),
            "runtime_shortlist_rank": _runtime_rank_for(info, target_uid),
            "reward": first_reward,
            "terminated": _jsonable(terminated),
            "truncated": _jsonable(truncated),
            "actions": _jsonable(branch_actions),
            "proposal_shortlists": _jsonable(branch_shortlists),
        }
        result["external_request_signature"].extend(first_requests)
        result["external_requests_by_tick"].append({"step": _step_from_env(branch_env), "requests": first_requests})
        result["steps"] = 1
        result["undiscounted_return"] = first_reward
        current = following
        stop_reason, actual_terminal = _continuation_stop_reason(
            branch_env, truncated, result["steps"], int(max_steps),
        )
        while stop_reason is None:
            request = keyed_request(policy, branch_env)
            tick_before = len(flat)
            current, rewards, terminated, truncated, info = _call_step(
                branch_env, request.actions, request.shortlists,
            )
            tick_requests = _capture_tick_requests(flat, tick_before)
            reward = _step_reward(rewards)
            result["suffix_undiscounted_reward"] += reward
            result["undiscounted_return"] += reward
            result["steps"] += 1
            result["external_request_signature"].extend(tick_requests)
            result["external_requests_by_tick"].append({"step": _step_from_env(branch_env), "requests": tick_requests})
            stop_reason, actual_terminal = _continuation_stop_reason(
                branch_env, truncated, result["steps"], int(max_steps),
            )
        result["terminal"] = bool(actual_terminal)
        result["censored"] = not result["terminal"]
        if result["censored"] and stop_reason is not None:
            result["censor_reason"] = stop_reason
        result["outcome_counts"] = _outcome_counts(branch_env)
        result["request_elapsed_integral_seconds"] = _request_elapsed_integral(branch_env)
        result["request_elapsed_integral"] = result["request_elapsed_integral_seconds"]
        result["request_elapsed_integral_delta_seconds"] = max(
            0.0, result["request_elapsed_integral_seconds"] - elapsed_before,
        )
        result["safety_counter_delta"] = _telemetry_delta(safety_before, _telemetry_snapshot(branch_env))
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["censored"] = True
        result.setdefault("outcome_counts", {"normal": 0, "emergency": 0, "failed": 0, "diverted": 0, "unresolved": 0})
    return result


def _common_external_prefix(branches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [branch for branch in branches if not branch.get("error")]
    if not valid:
        return {"valid": False, "length": 0, "mismatch_index": None}
    tick_rows = [branch.get("external_requests_by_tick", []) for branch in valid]
    common = min((len(rows) for rows in tick_rows), default=0)
    mismatch = None
    for index in range(common):
        reference = _jsonable(tick_rows[0][index])
        if any(_jsonable(rows[index]) != reference for rows in tick_rows[1:]):
            mismatch = index
            break
    return {
        "valid": mismatch is None,
        "length": common,
        "mismatch_index": mismatch,
        "branch_lengths": [len(rows) for rows in tick_rows],
    }


def _weighted(values: Sequence[float], weights: Sequence[float]) -> float:
    values_array = np.asarray(values, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    if values_array.shape != weights_array.shape:
        raise ValueError("value and weight arrays must have equal shape")
    return float(np.dot(values_array, weights_array / weights_array.sum()))


def paired_gate_value(
    policy: Any,
    env_snapshot: Any,
    first_actions: Mapping[Any, Any],
    first_shortlists: Mapping[Any, Any],
    target_uid: Any,
    continuation_seeds: Sequence[int],
    max_steps: int,
) -> dict[str, Any]:
    """Evaluate WAIT and all 25 RELEASE counterfactuals on paired copies."""

    _ensure_cpu(policy)
    if isinstance(max_steps, bool) or int(max_steps) < 1:
        raise ValueError("max_steps must be a positive integer")
    seeds = [int(seed) for seed in continuation_seeds]
    if not seeds:
        raise ValueError("continuation_seeds must contain at least one seed")
    caller_rng = _rng_snapshot()
    model_before = _state_snapshot(_policy_model(policy))
    env_before, fingerprint_method = _state_fingerprint(env_snapshot)
    try:
        context_env = copy.deepcopy(env_snapshot)
        try:
            from terminal_marl.training import diagnose_p3_k3_coverage_v1 as coverage

            coverage._restore_shadow_methods(context_env)
        except (AttributeError, ImportError, TypeError):
            pass
        context = _details_context(policy, context_env)
    except BaseException:
        _restore_rng(caller_rng)
        raise
    try:
        target_index, target_key = _context_target(context, target_uid)
    except BaseException:
        _restore_rng(caller_rng)
        raise
    logits = context["proposal_logits"][target_index]
    probabilities = context["proposal_probabilities"][target_index]
    reports: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            for proposal, name in [(None, "wait"), *[(index, f"release_{index}") for index in range(PROPOSAL_COUNT)]]:
                reports.append(
                    _branch_outcome(
                        policy=policy,
                        env_snapshot=env_snapshot,
                        first_actions=first_actions,
                        first_shortlists=first_shortlists,
                        target_uid=target_uid,
                        proposal=proposal,
                        logits=logits,
                        seed=seed,
                        max_steps=int(max_steps),
                        branch_name=name,
                    )
                )
    finally:
        _restore_rng(caller_rng)
    grouped: dict[str, list[dict[str, Any]]] = {"wait": [], "releases": []}
    for report in reports:
        if report["branch"] == "wait":
            grouped["wait"].append(report)
        else:
            grouped["releases"].append(report)
    # Each continuation seed has one WAIT and one instance of every proposal.
    wait_values = [float(item["suffix_undiscounted_reward"]) for item in grouped["wait"]]
    release_weights = np.asarray(probabilities, dtype=np.float64)
    release_weights /= float(release_weights.sum())
    expected_per_repeat = []
    expected_full_per_repeat = []
    for repeat_index in range(len(seeds)):
        values = [
            float(grouped["releases"][repeat_index * PROPOSAL_COUNT + proposal]["suffix_undiscounted_reward"])
            for proposal in range(PROPOSAL_COUNT)
        ]
        full_values = [
            float(grouped["releases"][repeat_index * PROPOSAL_COUNT + proposal]["undiscounted_return"])
            for proposal in range(PROPOSAL_COUNT)
        ]
        expected_per_repeat.append(_weighted(values, release_weights))
        expected_full_per_repeat.append(_weighted(full_values, release_weights))
    all_branches = reports
    prefix = _common_external_prefix(all_branches)
    model_unchanged = _state_unchanged(model_before, _policy_model(policy))
    env_after, _ = _state_fingerprint(env_snapshot)
    env_snapshot_unchanged = env_after == env_before
    fingerprint_reliable = fingerprint_method != "repr_fallback"
    rng_unchanged = _rng_equal(caller_rng, _rng_snapshot())
    complete = bool(
        prefix["valid"]
        and all(item.get("terminal") is True and not item.get("error") for item in all_branches)
        and model_unchanged
        and env_snapshot_unchanged
        and fingerprint_reliable
        and rng_unchanged
    )
    release_full_values = [float(item["undiscounted_return"]) for item in grouped["releases"]]
    result: dict[str, Any] = {
        "schema": DIAGNOSTIC_SCHEMA,
        "target_uid": str(target_key),
        "continuation_seeds": seeds,
        "max_steps": int(max_steps),
        "max_steps_semantics": "branch_tick_budget_from_snapshot",
        "snapshot_start_step": _step_from_env(env_snapshot),
        "first_action_other_uids_fixed": True,
        "proposal_probabilities": [float(value) for value in release_weights],
        "wait": grouped["wait"],
        "releases": grouped["releases"],
        "branches": reports,
        "expected_release_suffix_undiscounted_reward": float(np.mean(expected_per_repeat)),
        "expected_release_suffix_by_repeat": expected_per_repeat,
        "expected_release_undiscounted_return": float(np.mean(expected_full_per_repeat)),
        "expected_release_undiscounted_return_by_repeat": expected_full_per_repeat,
        "weighted_release_is_max": bool(
            expected_full_per_repeat
            and math.isclose(float(np.mean(expected_full_per_repeat)), max(release_full_values), rel_tol=0.0, abs_tol=1e-12)
        ),
        "external_common_prefix": prefix,
        "terminal_complete": complete,
        "censored": not complete,
        "model_unchanged": model_unchanged,
        "env_snapshot_unchanged": env_snapshot_unchanged,
        "fingerprint_method": fingerprint_method,
        "env_snapshot_fingerprint_reliable": fingerprint_reliable,
        "rng_unchanged": rng_unchanged,
    }
    if complete:
        wait_full_mean = float(np.mean([float(item["undiscounted_return"]) for item in grouped["wait"]]))
        release_full_mean = float(np.mean(expected_full_per_repeat))
        result["qualified_delta_Q"] = float(release_full_mean - wait_full_mean)
    return _jsonable(result)


def _world_array(world: Any, dimension: int | None = None) -> np.ndarray:
    if isinstance(world, Mapping):
        rows = [world[key] for key in sorted(world, key=str)]
        if not rows:
            width = int(dimension or 0)
            return np.empty((0, width), dtype=np.float32)
        value = np.asarray(rows, dtype=np.float32)
    else:
        value = np.asarray(world, dtype=np.float32)
    if value.ndim == 1:
        value = value.reshape(1, -1)
    if value.ndim != 2:
        raise ValueError("world must be a 2D array or UID mapping")
    if dimension is not None and value.shape[1] != int(dimension):
        raise ValueError(f"world width must be {dimension}")
    return value


def gate_update_alignment(
    before_policy: Any,
    after_policy: Any,
    world: Any,
    delta_q: float | None = None,
) -> dict[str, Any]:
    """Compare a policy update with its local gate probability prediction."""

    _ensure_cpu(before_policy)
    _ensure_cpu(after_policy)
    before_model = _policy_model(before_policy)
    after_model = _policy_model(after_policy)
    before_state = _state_snapshot(before_model)
    after_state = _state_snapshot(after_model)
    if list(before_state) != list(after_state):
        raise ValueError("before and after state_dict keys are not aligned")
    if any(before_state[key].shape != after_state[key].shape for key in before_state):
        raise ValueError("before and after state_dict shapes are not aligned")
    dimension = getattr(before_policy, "observation_dim", None)
    observations = _world_array(world, int(dimension) if dimension is not None else None)
    if observations.shape[0] == 0:
        caller_rng = _rng_snapshot()
        return {
            "schema": DIAGNOSTIC_SCHEMA,
            "rows": [],
            "before_gate_release_probability": [],
            "after_gate_release_probability": [],
            "proposal_probabilities_before": [],
            "proposal_probabilities_after": [],
            "linear_approximation": [],
            "actual_probability_delta": [],
            "sign_match": [],
            "changed_parameter_names": [key for key in before_state if not torch.equal(before_state[key], after_state[key])],
            "delta_q": delta_q,
            "model_unchanged": _state_unchanged(before_state, before_model) and _state_unchanged(after_state, after_model),
            "rng_unchanged": _rng_equal(caller_rng, _rng_snapshot()),
        }
    caller_rng = _rng_snapshot()
    try:
        tensor = torch.as_tensor(observations, dtype=torch.float32, device="cpu")
        with torch.no_grad():
            before_gate_det, before_proposal_det = before_model.distributions(tensor)
            after_gate_det, after_proposal_det = after_model.distributions(tensor)
            before_gate = before_gate_det.probs[:, RELEASE_GATE].detach().cpu().clone()
            after_gate = after_gate_det.probs[:, RELEASE_GATE].detach().cpu().clone()
            before_proposal = before_proposal_det.probs.detach().cpu().clone()
            after_proposal = after_proposal_det.probs.detach().cpu().clone()
        # Compute the derivative on an isolated clone.  This keeps frozen
        # flags, caller ``.grad`` fields, and the source model entirely
        # untouched while retaining the exact before-policy parameter values.
        gradient_model = copy.deepcopy(before_model)
        gradient_parameters = [(name, parameter) for name, parameter in gradient_model.named_parameters()]
        for _, parameter in gradient_parameters:
            parameter.requires_grad_(True)
        deltas = {
            name: after_state[name].to(parameter.device, dtype=parameter.dtype) - parameter.detach()
            for name, parameter in gradient_parameters
            if name in after_state
        }
        linear_values: list[float] = []
        for row in range(tensor.shape[0]):
            gate_dist, _ = gradient_model.distributions(tensor)
            probability = gate_dist.probs[row, RELEASE_GATE]
            if not gradient_parameters:
                linear_values.append(0.0)
                continue
            gradients = torch.autograd.grad(
                probability,
                [parameter for _, parameter in gradient_parameters],
                retain_graph=True,
                allow_unused=True,
            )
            linear_values.append(float(sum(
                torch.sum(gradient * deltas[name]).detach().cpu()
                for (name, _), gradient in zip(gradient_parameters, gradients)
                if gradient is not None and name in deltas
            )))
    finally:
        _restore_rng(caller_rng)
    actual_delta = (after_gate - before_gate).tolist()
    sign_match = [
        bool((abs(linear) <= 1e-12 and abs(actual) <= 1e-12) or (linear * actual > 0.0))
        for linear, actual in zip(linear_values, actual_delta)
    ]
    changed = [key for key in before_state if not torch.equal(before_state[key], after_state[key])]
    result = {
        "schema": DIAGNOSTIC_SCHEMA,
        "rows": int(tensor.shape[0]),
        "before_gate_release_probability": [float(value) for value in before_gate.tolist()],
        "after_gate_release_probability": [float(value) for value in after_gate.tolist()],
        "proposal_probabilities_before": before_proposal.tolist(),
        "proposal_probabilities_after": after_proposal.tolist(),
        "linear_approximation": linear_values,
        "actual_probability_delta": [float(value) for value in actual_delta],
        "sign_match": sign_match,
        "changed_parameter_names": changed,
        "delta_q": None if delta_q is None else float(delta_q),
        "model_unchanged": _state_unchanged(before_state, before_model) and _state_unchanged(after_state, after_model),
        "rng_unchanged": _rng_equal(caller_rng, _rng_snapshot()),
    }
    return _jsonable(result)


__all__ = [
    "KeyedRequest",
    "DIAGNOSTIC_SCHEMA",
    "gate_update_alignment",
    "keyed_request",
    "keyed_uniform",
    "paired_gate_value",
]
