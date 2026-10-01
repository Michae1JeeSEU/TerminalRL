"""Prepare uncertainty-softened counterfactual gate targets from a frozen JSON artifact.

This module deliberately has no environment, Torch, model, or optimizer imports.  It
only validates and reads the completed normal gate-value artifact, recomputes the
four-seed full-episode-MC deltas from every proposal probability, and writes a
small JSON target artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "p3_counterfactual_gate_targets_20261001_v1"
STAGE = "P3_PREFIX_NORMAL_GATE_VALUE_20261001_V1"
EXPECTED_INPUT_SHA256 = "1280dc0a5697913749f22982cff89aa7f0ff2a8075f4bb05b02331033ff7cdba"
EXPECTED_PREFLIGHT_SHA256 = "a247e7cdbaee8f9686b67cf6506ba2ad5e7d59f5d38df09a0088f7c6629cd5f8"
EXPECTED_PILOT_SHA256 = "b5138ee3493fbfeaf22172f0107251a1897175878914727c21bd52e424a85332"
EXPECTED_BEFORE_CHECKPOINT_SHA256 = "3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab"
EXPECTED_AFTER_CHECKPOINT_SHA256 = "d44990dc7d129573d9145f169df1cbc4f91fa32bc5127b1c58d937260c02a24f"
EXPECTED_SEEDS = (9303401, 9303402, 9303403, 9303404)
EXPECTED_CELLS = (1, 2, 3, 4)
BRANCHES_PER_SEED = 26
BRANCHES_PER_STATE = 104
TOTAL_BRANCHES = 416
PROPOSAL_COUNT = 25
ETA = 0.02
KL_CAP = 0.001
ALPHA_INITIAL = 1.0
BisectionSteps = 60
VERIFY_TOL = 1e-10


class InputValidationError(ValueError):
    """Raised when the frozen input binding or top-level schema is invalid."""


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputValidationError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise InputValidationError(f"{name} is not finite")
    return result


def _optional_finite(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, name=name)


def _json_copy(value: Any) -> Any:
    """Return JSON-compatible data while rejecting non-finite numeric values."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return _finite_number(value, name="json value")
    if isinstance(value, list):
        return [_json_copy(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_copy(item) for key, item in value.items()}
    raise InputValidationError(f"unsupported JSON value type: {type(value).__name__}")


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InputValidationError(f"{name} must be an object")
    return value


def _require_list(value: Any, *, name: str, length: int | None = None) -> list[Any]:
    if not isinstance(value, list):
        raise InputValidationError(f"{name} must be an array")
    if length is not None and len(value) != length:
        raise InputValidationError(f"{name} length {len(value)} != {length}")
    return value


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalise_probabilities(values: Sequence[Any], *, name: str) -> tuple[list[float], float]:
    raw = [_finite_number(value, name=f"{name}[{index}]") for index, value in enumerate(values)]
    if any(value < 0.0 for value in raw):
        raise InputValidationError(f"{name} contains a negative probability")
    total = float(sum(raw))
    if not math.isfinite(total) or total <= 0.0:
        raise InputValidationError(f"{name} has non-positive sum")
    return [value / total for value in raw], total


def weighted_release_delta(
    proposal_q_before: Sequence[Any],
    release_returns: Sequence[Any],
    wait_return: Any,
) -> tuple[float, float]:
    """Return ``sum(normalised(q) * release_return) - wait_return`` and q sum."""
    q, q_sum = _normalise_probabilities(proposal_q_before, name="proposal_q_before")
    returns = [_finite_number(value, name=f"release_returns[{index}]") for index, value in enumerate(release_returns)]
    if len(q) != PROPOSAL_COUNT or len(returns) != PROPOSAL_COUNT:
        raise InputValidationError("full proposal weighting requires exactly 25 releases")
    wait = _finite_number(wait_return, name="wait_return")
    return float(sum(weight * value for weight, value in zip(q, returns)) - wait), q_sum


def sample_mean_and_se(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise InputValidationError("cannot compute statistics for an empty sequence")
    mean = float(fmean(values))
    if len(values) < 2:
        raise InputValidationError("sample standard error requires at least two values")
    variance = sum((value - mean) ** 2 for value in values) / float(len(values) - 1)
    return mean, math.sqrt(variance) / math.sqrt(float(len(values)))


def reliability_weight(mu: float, sample_se: float) -> float:
    if mu == 0.0 and sample_se == 0.0:
        return 0.0
    denominator = mu * mu + sample_se * sample_se
    if denominator == 0.0:
        return 0.0
    return float((mu * mu) / denominator)


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("logit requires a strict interior probability")
    return math.log(probability) - math.log1p(-probability)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def bernoulli_kl(p: float, q: float) -> float:
    """Compute KL(Bernoulli(p) || Bernoulli(q)) with endpoint handling."""
    p = float(p)
    q = float(q)
    if not (0.0 <= p <= 1.0 and 0.0 <= q <= 1.0):
        raise ValueError("Bernoulli probabilities must be in [0, 1]")
    if p == q:
        return 0.0
    if q == 0.0 or q == 1.0:
        return math.inf
    result = 0.0
    if p > 0.0:
        result += p * math.log(p / q)
    if p < 1.0:
        result += (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
    return float(result)


def _target_probability(
    p0: float,
    effective_d: float,
    *,
    eta: float = ETA,
    kl_cap: float = KL_CAP,
    alpha_initial: float = ALPHA_INITIAL,
    bisection_steps: int = BisectionSteps,
    probability_bounds: tuple[float, float] = (0.0, 1.0),
) -> dict[str, float | int | str | bool | None]:
    if not 0.0 <= p0 <= 1.0:
        raise InputValidationError(f"p0={p0} is outside [0, 1]")
    bound_low, bound_high = probability_bounds
    if not 0.0 <= bound_low <= bound_high <= 1.0:
        raise InputValidationError("probability bounds must satisfy 0 <= low <= high <= 1")
    if not bound_low <= p0 <= bound_high:
        raise InputValidationError(f"p0={p0} is outside probability bounds {probability_bounds}")
    if p0 in (0.0, 1.0):
        return {
            "alpha": 0.0,
            "pstar": p0,
            "target_delta_p": 0.0,
            "kl_pstar_to_p0": 0.0,
            "kl_capped": False,
            "bounds_capped": False,
            "bisection_steps": 0,
            "endpoint_preserved": True,
        }
    shift = float(eta * effective_d)
    base_logit = _logit(p0)

    def probability(alpha: float) -> float:
        return _sigmoid(base_logit + alpha * shift)

    def within_bounds(value: float) -> bool:
        return bound_low <= value <= bound_high

    initial_pstar = probability(alpha_initial)
    initial_kl = bernoulli_kl(initial_pstar, p0)
    initial_allowed = initial_kl <= kl_cap and within_bounds(initial_pstar)
    if initial_allowed:
        alpha = float(alpha_initial)
        kl_capped = False
        bounds_capped = False
        steps = 0
    else:
        low = 0.0
        high = float(alpha_initial)
        for _ in range(int(bisection_steps)):
            middle = (low + high) / 2.0
            middle_probability = probability(middle)
            if bernoulli_kl(middle_probability, p0) <= kl_cap and within_bounds(middle_probability):
                low = middle
            else:
                high = middle
        alpha = low
        kl_capped = initial_kl > kl_cap
        bounds_capped = not within_bounds(initial_pstar)
        steps = int(bisection_steps)
    pstar = probability(alpha)
    if not within_bounds(pstar):
        # This is only reachable for a numerically inconsistent custom bound.
        raise InputValidationError(f"computed pstar={pstar} is outside probability bounds {probability_bounds}")
    return {
        "alpha": float(alpha),
        "pstar": float(pstar),
        "target_delta_p": float(pstar - p0),
        "kl_pstar_to_p0": float(bernoulli_kl(pstar, p0)),
        "kl_capped": bool(kl_capped),
        "bounds_capped": bool(bounds_capped),
        "bisection_steps": steps,
        "endpoint_preserved": False,
    }


def _probability_bounds(payload: Mapping[str, Any]) -> tuple[float, float]:
    raw = payload.get("target_probability_bounds", payload.get("probability_bounds", [0.0, 1.0]))
    bounds = _require_list(raw, name="target_probability_bounds", length=2)
    low = _finite_number(bounds[0], name="target_probability_bounds[0]")
    high = _finite_number(bounds[1], name="target_probability_bounds[1]")
    if not 0.0 <= low <= high <= 1.0:
        raise InputValidationError("target probability bounds must satisfy 0 <= low <= high <= 1")
    return low, high


def _validate_top_level(payload: Mapping[str, Any], input_sha256: str) -> None:
    if payload.get("stage") != STAGE:
        raise InputValidationError(f"stage mismatch: {payload.get('stage')!r}")
    if payload.get("status") not in ("FULL_OK", "COMPLETED"):
        raise InputValidationError(f"full artifact is not a completed FULL artifact: {payload.get('status')!r}")
    if payload.get("phase") != "full":
        raise InputValidationError(f"full artifact phase mismatch: {payload.get('phase')!r}")
    if payload.get("PPO_ALLOWED") is not False:
        raise InputValidationError("PPO_ALLOWED must be false")
    if payload.get("preflight_sha256") != EXPECTED_PREFLIGHT_SHA256:
        raise InputValidationError("preflight SHA binding mismatch")
    if payload.get("pilot_sha256") != EXPECTED_PILOT_SHA256:
        raise InputValidationError("pilot SHA binding mismatch")
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise InputValidationError(
            f"input full SHA mismatch: expected {EXPECTED_INPUT_SHA256}, got {input_sha256}"
        )
    cells = _require_list(payload.get("cells"), name="cells", length=4)
    ids = sorted(_require_mapping(cell, name="cell").get("cell_id") for cell in cells)
    if ids != list(EXPECTED_CELLS):
        raise InputValidationError(f"cell IDs mismatch: {ids!r}")


def _branch_unknown_reasons(cell: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if cell.get("all_branches_natural") is not True:
        reasons.append("all_branches_natural_not_true")
    branches = cell.get("branch_metrics")
    if not isinstance(branches, list):
        reasons.append("branch_metrics_missing")
        return reasons
    if len(branches) != BRANCHES_PER_STATE:
        reasons.append(f"branch_count_{len(branches)}_not_{BRANCHES_PER_STATE}")
    for index, branch_value in enumerate(branches):
        if not isinstance(branch_value, Mapping):
            reasons.append(f"branch_{index}_not_object")
            continue
        if branch_value.get("terminal") is not True:
            reasons.append(f"branch_{index}_not_terminal")
        if branch_value.get("censored") is not False:
            reasons.append(f"branch_{index}_censored_or_unknown")
        if branch_value.get("error") is not None:
            reasons.append(f"branch_{index}_error")
        if branch_value.get("safety_violation") is not False:
            reasons.append(f"branch_{index}_safety_violation_or_unknown")
        if branch_value.get("termination_classification") != "natural_terminal":
            reasons.append(f"branch_{index}_not_natural_terminal")
    return reasons


def _technical_unknown_reasons(cell: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    replay = cell.get("replay")
    if not isinstance(replay, Mapping):
        reasons.append("replay_missing")
    else:
        for key in ("exact", "config_match", "base_reset_state_match", "reset_recipe_match", "reset_state_match"):
            if replay.get(key) is not True:
                reasons.append(f"replay_{key}_not_true")
        if replay.get("steps_expected") != replay.get("steps_replayed"):
            reasons.append("replay_step_count_mismatch")
    for section_name in ("source_policy", "after_policy"):
        section = cell.get(section_name)
        if not isinstance(section, Mapping) or section.get("model_unchanged") is not True:
            reasons.append(f"{section_name}_model_changed_or_unknown")
    if cell.get("caller_rng_unchanged") is not True:
        reasons.append("caller_rng_not_unchanged")
    invariants = cell.get("invariants")
    if not isinstance(invariants, Mapping):
        reasons.append("invariants_missing")
    else:
        required_false = ("native_policy_sampler_used", "optimizer_used", "teacher_used")
        for key in required_false:
            if invariants.get(key) is not False:
                reasons.append(f"invariant_{key}_not_false")
        for key in ("prefix_sampling", "PPO_ALLOWED", "D_includes_first_reward"):
            if invariants.get(key) is not (True if key == "prefix_sampling" or key == "D_includes_first_reward" else False):
                reasons.append(f"invariant_{key}_mismatch")
        if invariants.get("controller_max") != PROPOSAL_COUNT:
            reasons.append("controller_max_mismatch")
        if invariants.get("public_k") != 3:
            reasons.append("public_k_mismatch")
        if invariants.get("max_steps") != 1800:
            reasons.append("max_steps_mismatch")
    source_actual = cell.get("source_actual")
    if not isinstance(source_actual, Mapping):
        reasons.append("source_actual_missing")
    else:
        if source_actual.get("decision") != "approved":
            reasons.append("source_actual_decision_not_approved")
        if source_actual.get("last_decision") != "approved":
            reasons.append("source_actual_last_decision_not_approved")
    selection = cell.get("selection")
    if not isinstance(selection, Mapping):
        reasons.append("selection_missing")
    else:
        context = selection.get("context")
        if not isinstance(context, Mapping):
            reasons.append("selection_context_missing")
        else:
            for key, expected in (("actor_mask", True), ("retry_free", True), ("committed_plan", False), ("phase", "ring_inner")):
                if context.get(key) != expected:
                    reasons.append(f"selection_{key}_mismatch")
    return reasons


def _seed_rows(
    cell: Mapping[str, Any], q: Sequence[Any] | None, reasons: list[str]
) -> tuple[list[dict[str, Any]], list[float]]:
    raw_rows = cell.get("per_seed")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(EXPECTED_SEEDS):
        reasons.append("per_seed_rows_missing_or_wrong_count")
        return [], []
    rows: list[dict[str, Any]] = []
    recomputed: list[float] = []
    seen: list[int] = []
    for row_index, row_value in enumerate(raw_rows):
        if not isinstance(row_value, Mapping):
            reasons.append(f"per_seed_{row_index}_not_object")
            continue
        seed_value = row_value.get("seed")
        if not isinstance(seed_value, int) or isinstance(seed_value, bool):
            reasons.append(f"per_seed_{row_index}_seed_invalid")
            continue
        seen.append(seed_value)
        returns = row_value.get("release_returns")
        wait_return = row_value.get("wait_return")
        row_reasons: list[str] = []
        if row_value.get("complete") is not True:
            row_reasons.append("complete_not_true")
        if not isinstance(returns, list) or len(returns) != PROPOSAL_COUNT:
            row_reasons.append("release_returns_missing_or_wrong_count")
        if wait_return is None:
            row_reasons.append("wait_return_missing")
        if q is None:
            row_reasons.append("q_normalization_unavailable")
        recomputed_value: float | None = None
        q_sum: float | None = None
        if not row_reasons:
            try:
                recomputed_value, q_sum = weighted_release_delta(q, returns, wait_return)
            except InputValidationError as exc:
                row_reasons.append(str(exc))
        recorded_value: float | None
        try:
            recorded_value = _optional_finite(row_value.get("D"), name=f"per_seed[{seed_value}].D")
        except InputValidationError as exc:
            recorded_value = None
            row_reasons.append(str(exc))
        if recorded_value is None:
            row_reasons.append("recorded_D_missing")
        if recomputed_value is not None and recorded_value is not None:
            if abs(recomputed_value - recorded_value) > VERIFY_TOL:
                row_reasons.append("recomputed_D_mismatch")
            recomputed.append(recomputed_value)
        else:
            recomputed.append(float("nan"))
        rows.append(
            {
                "seed": seed_value,
                "complete": row_value.get("complete"),
                "recorded_D": recorded_value,
                "recomputed_D": recomputed_value,
                "D_error": None if recomputed_value is None or recorded_value is None else recomputed_value - recorded_value,
                "release_returns": _json_copy(returns) if isinstance(returns, list) else None,
                "wait_return": _optional_finite(wait_return, name=f"per_seed[{seed_value}].wait_return") if wait_return is not None else None,
                "q_sum": q_sum,
                "business_metrics": _json_copy(row_value.get("business_metrics")) if "business_metrics" in row_value else None,
                "unknown_reasons": row_reasons,
            }
        )
        if row_reasons:
            reasons.extend(f"seed_{seed_value}_{reason}" for reason in row_reasons)
    if sorted(seen) != list(EXPECTED_SEEDS):
        reasons.append(f"seed_set_mismatch:{sorted(seen)!r}")
    return rows, recomputed


def _cell_target(cell: Mapping[str, Any], bounds: tuple[float, float]) -> dict[str, Any]:
    cell_id = cell.get("cell_id")
    reasons: list[str] = []
    q_raw = cell.get("proposal_q_before")
    if not isinstance(q_raw, list) or len(q_raw) != PROPOSAL_COUNT:
        reasons.append("proposal_q_before_missing_or_wrong_count")
        q_raw = []
        q_normalized: list[float] = []
        q_sum = None
    else:
        try:
            q_normalized, q_sum = _normalise_probabilities(q_raw, name="proposal_q_before")
        except InputValidationError as exc:
            reasons.append(str(exc))
            q_normalized, q_sum = [], None
    reasons.extend(_branch_unknown_reasons(cell))
    reasons.extend(_technical_unknown_reasons(cell))
    rows: list[dict[str, Any]] = []
    recomputed: list[float] = []
    if isinstance(cell.get("per_seed"), list):
        rows, recomputed = _seed_rows(cell, q_raw if q_normalized else None, reasons)
    elif q_normalized:
        reasons.append("per_seed_rows_missing_or_wrong_count")

    # The seed rows are always retained.  Only a fully known cell can receive a target.
    complete_recompute = len(recomputed) == len(EXPECTED_SEEDS) and all(math.isfinite(value) for value in recomputed)
    if complete_recompute and not reasons:
        mu, sample_se = sample_mean_and_se(recomputed)
        recorded_cell_d = _optional_finite(cell.get("D"), name=f"cell_{cell_id}.D")
        if recorded_cell_d is None or abs(mu - recorded_cell_d) > VERIFY_TOL:
            reasons.append("cell_mean_D_mismatch")
        if cell.get("qualified_D") is not True:
            reasons.append("qualified_D_not_true")
    else:
        mu = None
        sample_se = None

    if mu is not None and sample_se is not None:
        weight = reliability_weight(mu, sample_se)
        effective_d = weight * mu
    else:
        weight = None
        effective_d = None

    p0 = _optional_finite(cell.get("p_before"), name=f"cell_{cell_id}.p_before")
    p_after = _optional_finite(cell.get("p_after"), name=f"cell_{cell_id}.p_after")
    observed_delta = _optional_finite(cell.get("delta_p"), name=f"cell_{cell_id}.delta_p")
    exact_delta = None if p0 is None or p_after is None else p_after - p0
    if observed_delta is None or exact_delta is None:
        reasons.append("observed_probability_fields_missing")
    elif abs(observed_delta - exact_delta) > VERIFY_TOL:
        reasons.append("observed_delta_p_mismatch")
    if p0 is None:
        reasons.append("p_before_missing")
    elif not bounds[0] <= p0 <= bounds[1]:
        raise InputValidationError(f"cell {cell_id} p_before={p0} outside explicit bounds {bounds}")

    # Any post-recompute validation failure makes the qualified statistics unknown.
    # The per-seed rows remain available as raw evidence, but they must not be
    # exposed as an eligible cell estimate.
    if reasons:
        mu = None
        sample_se = None
        weight = None
        effective_d = None

    target = None
    if not reasons and mu is not None and sample_se is not None and effective_d is not None and p0 is not None:
        target = _target_probability(p0, effective_d, probability_bounds=bounds)
    sign_counts = {
        "negative": sum(value < 0.0 for value in recomputed if math.isfinite(value)),
        "zero": sum(value == 0.0 for value in recomputed if math.isfinite(value)),
        "positive": sum(value > 0.0 for value in recomputed if math.isfinite(value)),
    }
    selection = cell.get("selection")
    source_actual = cell.get("source_actual")
    return {
        "cell_id": cell_id,
        "status": cell.get("status"),
        "selection": _json_copy(selection) if selection is not None else None,
        "source_actual": _json_copy(source_actual) if source_actual is not None else None,
        "proposal_q_before": _json_copy(q_raw),
        "proposal_q_before_normalized": q_normalized if q_normalized else None,
        "proposal_q_sum": q_sum,
        "per_seed": rows,
        "recorded_cell_D": _optional_finite(cell.get("D"), name=f"cell_{cell_id}.D") if cell.get("D") is not None else None,
        "mean_D": mu,
        "sample_se_D": sample_se,
        "sign_counts": sign_counts,
        "reliability_weight": weight,
        "effective_D": effective_d,
        "p0": p0,
        "p_after": p_after,
        "exact_delta_p": exact_delta,
        "observed_delta_p": observed_delta,
        "pstar": None if target is None else target["pstar"],
        "alpha": None if target is None else target["alpha"],
        "kl_pstar_to_p0": None if target is None else target["kl_pstar_to_p0"],
        "target_delta_p": None if target is None else target["target_delta_p"],
        "target_details": target,
        "local_m_original": _optional_finite(cell.get("local_m"), name=f"cell_{cell_id}.local_m") if cell.get("local_m") is not None else None,
        "unknown_business": {
            "linear_residual": cell.get("linear_residual"),
            "per_seed_business_metrics": [row.get("business_metrics") for row in rows],
        },
        "known": target is not None,
        "unknown_reasons": sorted(set(reasons)),
    }


def build_targets(payload: Mapping[str, Any], *, input_sha256: str) -> dict[str, Any]:
    """Validate a completed full artifact and return the target JSON object."""
    _validate_top_level(payload, input_sha256)
    bounds = _probability_bounds(payload)
    cells = sorted((_require_mapping(cell, name="cell") for cell in payload["cells"]), key=lambda cell: cell["cell_id"])
    output_cells = [_cell_target(cell, bounds) for cell in cells]
    known_count = sum(1 for cell in output_cells if cell["known"])
    return {
        "schema": SCHEMA,
        "status": "TARGETS_READY" if known_count == len(output_cells) else "TARGETS_PARTIAL_UNKNOWN",
        "stage": STAGE,
        "input_status": payload.get("status"),
        "input_binding": {
            "input_full_sha256": input_sha256,
            "expected_input_full_sha256": EXPECTED_INPUT_SHA256,
            "preflight_sha256": payload.get("preflight_sha256"),
            "pilot_sha256": payload.get("pilot_sha256"),
            "before_checkpoint_sha256": EXPECTED_BEFORE_CHECKPOINT_SHA256,
            "after_checkpoint_sha256": EXPECTED_AFTER_CHECKPOINT_SHA256,
            "source_label": "BEFORE3fe",
            "after_label": "AFTERd449",
        },
        "method": {
            "q_normalization": "standard_float_sum_then_divide",
            "D_definition": "sum(normalized_proposal_q_before * release_return) - wait_return",
            "full_episode_mc_includes_first_reward": True,
            "eta": ETA,
            "alpha_initial": ALPHA_INITIAL,
            "kl_cap": KL_CAP,
            "bisection_steps": BisectionSteps,
            "reliability_weight": "mu^2 / (mu^2 + sample_se^2), with 0 when mu=sample_se=0",
            "effective_D": "reliability_weight * mu",
            "target_logit_shift": "alpha * eta * effective_D",
            "target_kl": "Bernoulli(pstar) || Bernoulli(p0)",
            "probability_bounds": [bounds[0], bounds[1]],
        },
        "scope": {
            "cells_expected": list(EXPECTED_CELLS),
            "cells_retained": [cell["cell_id"] for cell in output_cells],
            "seeds": list(EXPECTED_SEEDS),
            "branches_per_seed": BRANCHES_PER_SEED,
            "branches_per_state": BRANCHES_PER_STATE,
            "total_branches": TOTAL_BRANCHES,
            "PPO_ALLOWED": False,
            "optimizer_used": False,
            "teacher_used": False,
            "heldout_used": False,
            "weights_changed": False,
        },
        "known_cell_count": known_count,
        "unknown_cell_count": len(output_cells) - known_count,
        "cells": output_cells,
    }


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"input JSON is invalid: {exc}") from exc
    return _require_mapping(payload, name="input JSON"), digest


def write_targets(input_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    payload, input_sha256 = _read_json(input_path)
    output = build_targets(payload, input_sha256=input_sha256)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    # Exclusive creation keeps the no-overwrite contract even if a caller races us.
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-full", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        output = write_targets(args.input_full, args.output)
    except (FileExistsError, FileNotFoundError, InputValidationError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        json.dumps(
            {
                "status": output["status"],
                "output": str(args.output),
                "input_full_sha256": output["input_binding"]["input_full_sha256"],
                "known_cell_count": output["known_cell_count"],
                "unknown_cell_count": output["unknown_cell_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
