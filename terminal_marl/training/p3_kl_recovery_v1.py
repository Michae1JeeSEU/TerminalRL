"""Bounded retry wrapper for pure KL rejections in the V6 PPO update.

This module is deliberately outside the V6 runner.  It retries the exact same
rollout only after the runner reports the narrow, rollback-safe rejection
shape.  A retry never changes the PPO limits or the reference object; it only
scales the existing Adam learning rates for that update call.
"""
from __future__ import annotations

import copy
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from terminal_marl.training import run_v4_2_p3_gate_anchored_v6 as v6


LR_MULTIPLIERS: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
KL_REJECTION_REASONS = frozenset(
    {"reference_kl_exceeds_max", "rollout_kl_exceeds_target"}
)
REQUIRED_UPDATE_GATES = frozenset({
    "finite_update", "finite_model", "finite_optimizer", "accepted_updates",
    "update_accounting", "rejected_updates", "stopped", "target_kl", "bounded_KL",
    "frozen_proposal_subgraph", "runtime_shortlist_stability", "reference_source_fixed",
    "reference_gate_kl_cap", "proposal_temperature_frozen",
})
_MISSING = object()


def _exact_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_exact_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_exact_equal(a, b) for a, b in zip(left, right))
        )
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _rng_snapshot() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state().detach().clone(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "python": copy.deepcopy(random.getstate()),
    }


def _restore_rng(snapshot: Mapping[str, Any]) -> None:
    torch.set_rng_state(torch.as_tensor(snapshot["torch"], dtype=torch.uint8).cpu().clone())
    numpy_state = snapshot["numpy"]
    np.random.set_state(copy.deepcopy(numpy_state))
    random.setstate(copy.deepcopy(snapshot["python"]))


def _reference_snapshot(reference: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in ("observations", "reference_logits", "reference_probs", "weights"):
        value = getattr(reference, name, _MISSING)
        if value is _MISSING:
            raise TypeError(f"reference is missing {name}")
        values[name] = torch.as_tensor(value).detach().clone()
    for name in ("coefficient", "max_kl"):
        value = getattr(reference, name, _MISSING)
        if value is _MISSING:
            raise TypeError(f"reference is missing {name}")
        values[name] = float(value)
    return values


def _reference_equal(reference: Any, snapshot: Mapping[str, Any]) -> bool:
    try:
        return all(
            _exact_equal(torch.as_tensor(getattr(reference, name)).detach(), snapshot[name])
            for name in ("observations", "reference_logits", "reference_probs", "weights")
        ) and float(reference.coefficient) == float(snapshot["coefficient"]) \
            and float(reference.max_kl) == float(snapshot["max_kl"])
    except (AttributeError, TypeError, ValueError):
        return False


def _restore_reference(reference: Any, snapshot: Mapping[str, Any]) -> None:
    with torch.no_grad():
        for name in ("observations", "reference_logits", "reference_probs", "weights"):
            target = getattr(reference, name)
            source = torch.as_tensor(snapshot[name], device=target.device, dtype=target.dtype)
            target.copy_(source)


def _state_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_state_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_state_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _optimizer_lrs(optimizer: Any) -> list[float]:
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or not groups:
        raise ValueError("optimizer has no parameter groups")
    result = [float(group["lr"]) for group in groups]
    if not result or not all(np.isfinite(value) and value > 0.0 for value in result):
        raise ValueError("optimizer learning rates must be finite and positive")
    return result


def _set_lrs(optimizer: Any, values: Sequence[float]) -> None:
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or len(groups) != len(values):
        raise ValueError("optimizer parameter-group count changed")
    for group, value in zip(groups, values):
        group["lr"] = float(value)


def _restore_policy(
    policy: Any,
    model_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    lrs: Sequence[float],
    rng: Mapping[str, Any] | None,
    *,
    restore_rng: bool,
) -> None:
    policy.model.load_state_dict(copy.deepcopy(model_state))
    policy.optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    _set_lrs(policy.optimizer, lrs)
    if restore_rng and rng is not None:
        _restore_rng(rng)


def _raw_update_fields(update: Mapping[str, Any]) -> dict[str, Any]:
    def value(name: str, default: Any = None) -> Any:
        item = update.get(name, default)
        return item if isinstance(item, (str, int, float, bool)) or item is None else None

    return {
        "accepted_updates": value("accepted_updates", 0),
        "attempted_updates": value("attempted_updates", 0),
        "rejected_updates": value("rejected_updates", 0),
        "updates": value("updates", 0),
        "rejection_reason": value("rejection_reason"),
        "stopped": value("stopped"),
        "final_rollout_kl": value("final_rollout_kl"),
        "max_accepted_rollout_kl": value("max_accepted_rollout_kl"),
        "reference_kl_before": value("reference_kl_before"),
        "reference_kl_final": value("reference_kl_final"),
        "reference_kl_max_accepted": value("reference_kl_max_accepted"),
        "reference_rejected_updates": value("reference_rejected_updates"),
        "team_steps": value("team_steps"),
        "actor_decisions": value("actor_decisions"),
    }


def _attempt_report(
    *,
    index: int,
    multiplier: float,
    lrs: Sequence[float],
    update: Mapping[str, Any],
    gates: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    fields = _raw_update_fields(update)
    return {
        "attempt": int(index),
        "lr_multiplier": float(multiplier),
        "effective_learning_rates": [float(value) for value in lrs],
        "status": status,
        "update": fields,
        "gates": {str(key): bool(value) for key, value in gates.items()},
        "cost": {
            "attempted_updates": int(fields["attempted_updates"] or 0),
            "rejected_updates": int(fields["rejected_updates"] or 0),
            "accepted_updates": int(fields["accepted_updates"] or 0),
            "team_steps": int(fields["team_steps"] or 0),
            "actor_decisions": int(fields["actor_decisions"] or 0),
        },
    }


def _cost(attempts: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        key: sum(int((attempt.get("cost") or {}).get(key, 0)) for attempt in attempts)
        for key in ("attempted_updates", "rejected_updates", "accepted_updates", "team_steps", "actor_decisions")
    } | {"attempts": len(attempts)}


def _result(
    *,
    status: str,
    update: Any,
    original_update: Any,
    gates: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    error: str | None = None,
    processed: int = 1,
    updated: int = 0,
    skipped: int = 0,
) -> dict[str, Any]:
    result = {
        "status": status,
        "processed": int(processed),
        "updated": int(updated),
        "skipped": int(skipped),
        "update": update,
        "original_update": original_update,
        "gates": dict(gates),
        "attempts": list(attempts),
        "cumulative_cost": _cost(attempts),
    }
    if error is not None:
        result["error"] = str(error)
    return result


def recover_update(
    policy: Any,
    steps: list[Any],
    config: Any,
    tape_path: str,
    runtime_protocol: Mapping[str, Any],
    reference: Any,
    old_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run V6 once and retry only a rollback-safe pure KL rejection.

    The return value keeps the actual PPO update untouched.  In particular,
    ``SKIPPED_KL`` returns an update with ``accepted_updates == 0`` and the
    retry counters are separate from PPO acceptance counters.
    """

    optimizer = getattr(policy, "optimizer", None)
    model = getattr(policy, "model", None)
    if optimizer is None or model is None:
        return _result(
            status="HARD_FAILED", update=None, original_update=None, gates={}, attempts=[],
            error="policy must expose model and optimizer",
        )
    original_gate_reference = getattr(policy, "gate_reference", _MISSING)
    try:
        base_model = copy.deepcopy(model.state_dict())
        base_optimizer = copy.deepcopy(optimizer.state_dict())
        base_lrs = _optimizer_lrs(optimizer)
        base_rng = _rng_snapshot()
        reference_state = _reference_snapshot(reference)
        policy.gate_reference = reference
        initial_metrics = reference.metrics(policy)
        initial_reference_kl = float(initial_metrics["reference_kl"])
        if not np.isfinite(initial_reference_kl) or initial_reference_kl > float(reference.max_kl):
            _restore_policy(policy, base_model, base_optimizer, base_lrs, base_rng, restore_rng=True)
            _restore_reference(reference, reference_state)
            if original_gate_reference is _MISSING:
                delattr(policy, "gate_reference")
            else:
                policy.gate_reference = original_gate_reference
            return _result(
                status="HARD_FAILED", update=None, original_update=None,
                gates={"reference_gate_kl_cap": False}, attempts=[],
                error="preexisting reference KL exceeds max_kl",
            )
        _restore_rng(base_rng)
        _set_lrs(optimizer, base_lrs)
    except Exception as exc:
        try:
            _restore_policy(policy, base_model, base_optimizer, base_lrs, base_rng, restore_rng=True)
            if "reference_state" in locals():
                _restore_reference(reference, reference_state)
        except Exception:
            pass
        return _result(
            status="HARD_FAILED", update=None, original_update=None, gates={}, attempts=[],
            error=f"preflight {type(exc).__name__}: {exc}",
        )

    attempts: list[dict[str, Any]] = []
    original_update: Mapping[str, Any] | None = None
    last_update: Mapping[str, Any] | None = None
    last_gates: dict[str, Any] = {}
    hard_error: str | None = None

    for attempt_index, multiplier in enumerate(LR_MULTIPLIERS, start=1):
        if attempt_index > 1:
            try:
                _restore_policy(policy, base_model, base_optimizer, base_lrs, base_rng, restore_rng=True)
                policy.gate_reference = reference
                _set_lrs(policy.optimizer, [value * multiplier for value in base_lrs])
            except Exception as exc:
                hard_error = f"retry restore {type(exc).__name__}: {exc}"
                break
        attempt_model = copy.deepcopy(policy.model.state_dict())
        attempt_optimizer = copy.deepcopy(policy.optimizer.state_dict())
        effective_lrs = [float(value) for value in _optimizer_lrs(policy.optimizer)]
        try:
            update_value, gates_value, _metrics = v6._update(
                policy, steps, config, tape_path, runtime_protocol, reference,
                old_rows, coverage_rows,
            )
            update = dict(update_value) if isinstance(update_value, Mapping) else {"value": update_value}
            gates = {str(key): bool(value) for key, value in (gates_value or {}).items()}
        except Exception as exc:
            hard_error = f"update {type(exc).__name__}: {exc}"
            update = {"exception": type(exc).__name__, "message": str(exc)}
            gates = {"update_transaction": False}
            attempt = _attempt_report(
                index=attempt_index, multiplier=multiplier, lrs=effective_lrs,
                update=update, gates=gates, status="HARD_FAILED",
            )
            attempts.append(attempt)
            last_update, last_gates = update, gates
            break

        if original_update is None:
            original_update = copy.deepcopy(update)
        last_update, last_gates = update, gates
        false_gates = {key for key, value in gates.items() if not value}
        accepted = int(update.get("accepted_updates", 0) or 0)
        rejection_reason = update.get("rejection_reason")
        reference_ok = (
            getattr(policy, "gate_reference", None) is reference
            and _reference_equal(reference, reference_state)
        )
        current_model = policy.model.state_dict()
        current_optimizer = policy.optimizer.state_dict()
        exact_rollback = (
            _exact_equal(current_model, attempt_model)
            and _exact_equal(current_optimizer, attempt_optimizer)
        )
        finite_gates = all(bool(gates.get(key, False)) for key in (
            "finite_model", "finite_optimizer", "finite_update",
        ))
        all_other_gates = (
            REQUIRED_UPDATE_GATES.issubset(gates)
            and false_gates == {"accepted_updates"}
        )
        pure_rejection = (
            accepted == 0
            and rejection_reason in KL_REJECTION_REASONS
            and all_other_gates
            and finite_gates
            and _state_finite(update)
            and update.get("exception") is None
            and reference_ok
            and exact_rollback
        )
        accepted_update = (
            accepted > 0
            and REQUIRED_UPDATE_GATES.issubset(gates)
            and not false_gates
            and (rejection_reason is None or rejection_reason in KL_REJECTION_REASONS)
            and update.get("exception") is None
            and finite_gates
            and _state_finite(update)
            and reference_ok
            and _state_finite(current_model)
            and _state_finite(current_optimizer)
        )
        if accepted_update:
            _set_lrs(policy.optimizer, base_lrs)
            attempts.append(_attempt_report(
                index=attempt_index, multiplier=multiplier, lrs=effective_lrs,
                update=update, gates=gates, status="UPDATED",
            ))
            return _result(
                status="UPDATED", update=update, original_update=original_update,
                gates=gates, attempts=attempts, updated=1,
            )
        if not pure_rejection:
            hard_error = (
                "update did not satisfy pure KL rollback contract: "
                f"accepted={accepted}, reason={rejection_reason!r}, "
                f"false_gates={sorted(false_gates)}, reference_ok={reference_ok}, "
                f"exact_rollback={exact_rollback}"
            )
            attempts.append(_attempt_report(
                index=attempt_index, multiplier=multiplier, lrs=effective_lrs,
                update=update, gates=gates, status="HARD_FAILED",
            ))
            break
        attempts.append(_attempt_report(
            index=attempt_index, multiplier=multiplier, lrs=effective_lrs,
            update=update, gates=gates, status="REJECTED_KL",
        ))
        if attempt_index == len(LR_MULTIPLIERS):
            _restore_policy(policy, base_model, base_optimizer, base_lrs, None, restore_rng=False)
            return _result(
                status="SKIPPED_KL", update=update, original_update=original_update,
                gates=gates, attempts=attempts, skipped=1,
            )

    _restore_policy(policy, base_model, base_optimizer, base_lrs, base_rng, restore_rng=True)
    try:
        _restore_reference(reference, reference_state)
    except Exception as exc:
        hard_error = f"reference rollback {type(exc).__name__}: {exc}"
    if original_gate_reference is _MISSING:
        delattr(policy, "gate_reference")
    else:
        policy.gate_reference = original_gate_reference
    return _result(
        status="HARD_FAILED", update=last_update, original_update=original_update,
        gates=last_gates, attempts=attempts, error=hard_error,
    )


__all__ = ["KL_REJECTION_REASONS", "LR_MULTIPLIERS", "recover_update"]
