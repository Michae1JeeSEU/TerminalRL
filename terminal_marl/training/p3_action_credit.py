"""Pure helpers for the bounded P3 one-step action-credit diagnostic.

These helpers do not access an environment, checkpoint, filesystem, or
training loop.  They only construct the requested V4.2 branch inputs and
summarize the source-probability-weighted one-step critic proxy.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

from terminal_marl.core.low_level_joint_action_v4_2 import (
    HierarchicalActionV42,
    N_RELEASE_PROPOSALS_V4_2,
    RELEASE_GATE,
    WAIT_GATE,
)


def _uid_key(mapping: Mapping[Any, Any], uid: int) -> Any:
    """Return the existing key spelling for a UID, preserving other keys."""
    if uid in mapping:
        return uid
    text = str(uid)
    if text in mapping:
        return text
    return text


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _branch_action(proposal: int | None) -> np.ndarray:
    if proposal is None:
        return HierarchicalActionV42(WAIT_GATE, None).normalized.copy()
    if isinstance(proposal, (bool, np.bool_)) or not isinstance(proposal, Integral):
        raise ValueError("proposal must be an integer in [0, 25)")
    proposal_index = int(proposal)
    if not 0 <= proposal_index < N_RELEASE_PROPOSALS_V4_2:
        raise ValueError("proposal must be an integer in [0, 25)")
    return HierarchicalActionV42(RELEASE_GATE, proposal_index).normalized.copy()


def branch_inputs(
    actions: Mapping[Any, Any],
    shortlists: Mapping[Any, Sequence[Any]],
    uid: int,
    proposal: int | None,
    logits: Sequence[float] | np.ndarray,
    *,
    shortlist_k: int = 3,
) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """Build one WAIT or RELEASE branch while preserving all other requests.

    A RELEASE branch places ``proposal`` first and appends the two highest
    remaining proposal logits.  ``np.argsort(..., kind="stable")`` makes tie
    handling deterministic.  Inputs are deep-copied before the target action
    and shortlist are changed.
    """
    if shortlist_k not in {1, 3}:
        raise ValueError("shortlist_k must be 1 or 3")
    if not isinstance(actions, Mapping) or not isinstance(shortlists, Mapping):
        raise TypeError("actions and shortlists must be mappings")
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size != N_RELEASE_PROPOSALS_V4_2:
        raise ValueError(f"logits must have shape ({N_RELEASE_PROPOSALS_V4_2},)")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")

    branch_actions = deepcopy(dict(actions))
    branch_shortlists = deepcopy(dict(shortlists))
    action_key = _uid_key(branch_actions, int(uid))
    shortlist_key = _uid_key(branch_shortlists, int(uid))
    branch_actions[action_key] = _branch_action(proposal)

    if proposal is None:
        branch_shortlists.pop(shortlist_key, None)
        # A caller may supply both integer and string spellings for the same
        # UID.  Removing both keeps WAIT semantically shortlist-free.
        branch_shortlists.pop(int(uid), None)
        branch_shortlists.pop(str(uid), None)
    else:
        proposal_index = int(proposal)
        if shortlist_k == 1:
            ordered = [proposal_index]
        else:
            ranked = np.argsort(-values, kind="stable").tolist()
            remaining = [int(index) for index in ranked if int(index) != proposal_index]
            if len(remaining) < 2:
                raise ValueError("at least two remaining proposals are required for K3")
            ordered = [proposal_index, remaining[0], remaining[1]]
        branch_shortlists[shortlist_key] = tuple(
            HierarchicalActionV42(RELEASE_GATE, index).normalized.copy()
            for index in ordered
        )
    return branch_actions, branch_shortlists


def _require_fields(value: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(f"{label} missing fields: {missing}")


def summarize_branches(
    wait: Mapping[str, Any],
    releases: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float] | np.ndarray,
    gamma: float = 0.99,
) -> dict[str, float]:
    """Summarize source-probability-weighted one-step Q proxies.

    ``next_value`` is replaced by zero for terminal branches.  The returned
    deltas compare the probability-weighted RELEASE expectation with WAIT;
    this is a one-step critic proxy and does not reconstruct PPO GAE.
    """
    _require_fields(wait, ("reward", "next_value"), "WAIT branch")
    gamma_value = _finite_float(gamma, "gamma")
    if not 0.0 <= gamma_value <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    wait_reward = _finite_float(wait["reward"], "wait.reward")
    wait_next = _finite_float(wait["next_value"], "wait.next_value")
    wait_terminated = wait.get("terminated", False)
    if not isinstance(wait_terminated, (bool, np.bool_)):
        raise ValueError("wait.terminated must be boolean")
    wait_bootstrap = 0.0 if bool(wait_terminated) else wait_next
    wait_q = wait_reward + gamma_value * wait_bootstrap

    if len(releases) != N_RELEASE_PROPOSALS_V4_2:
        raise ValueError(f"expected {N_RELEASE_PROPOSALS_V4_2} release branches")
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or probs.shape != (N_RELEASE_PROPOSALS_V4_2,):
        raise ValueError(f"probabilities must have shape ({N_RELEASE_PROPOSALS_V4_2},)")
    if not np.isfinite(probs).all():
        raise ValueError("probabilities must be finite")
    if np.any(probs < 0.0):
        raise ValueError("probabilities must be non-negative")
    probability_mass = float(probs.sum())
    if not np.isfinite(probability_mass) or probability_mass <= 0.0:
        raise ValueError("probabilities must have positive finite mass")
    if abs(probability_mass - 1.0) > 1e-6:
        raise ValueError("probabilities must sum to one within 1e-6")

    rewards = np.empty(N_RELEASE_PROPOSALS_V4_2, dtype=np.float64)
    bootstrapped_values = np.empty(N_RELEASE_PROPOSALS_V4_2, dtype=np.float64)
    approved_mass = 0.0
    for index, branch in enumerate(releases):
        _require_fields(branch, ("reward", "next_value", "terminated", "decision"), f"release[{index}]")
        rewards[index] = _finite_float(branch["reward"], f"release[{index}].reward")
        next_value = _finite_float(branch["next_value"], f"release[{index}].next_value")
        terminated = branch["terminated"]
        if not isinstance(terminated, (bool, np.bool_)):
            raise ValueError(f"release[{index}].terminated must be boolean")
        bootstrapped_values[index] = 0.0 if bool(terminated) else next_value
        if branch["decision"] == "approved":
            approved_mass += float(probs[index] / probability_mass)

    weights = probs / probability_mass
    expected_reward = float(np.dot(weights, rewards))
    expected_next = float(np.dot(weights, bootstrapped_values))
    delta_reward = expected_reward - wait_reward
    discounted_delta_value = gamma_value * (expected_next - wait_bootstrap)
    delta_q = (expected_reward + gamma_value * expected_next) - wait_q
    return {
        "expected_release_reward": expected_reward,
        "delta_reward": float(delta_reward),
        "discounted_delta_value": float(discounted_delta_value),
        "delta_q": float(delta_q),
        "approved_mass": float(approved_mass),
        "wait_q": float(wait_q),
    }
