"""Frozen six-seed development evaluation for the runtime rank repair.

This evaluator is deliberately separate from the historical PPO admission
path.  It loads a source checkpoint and a proposal-head rank-repair
checkpoint, then calls :func:`student_rollout_v4_2.rollout` directly.  The
rollout is followed by an exact action-tape lifecycle replay so that terminal
failures remain in the request denominator.

The module is development-only.  It never updates a model, consumes a
held-out seed, invokes a Teacher, or sets a PPO/admission flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from terminal_marl.training import evaluate_p3_proposal_head_probe_v1 as legacy_eval
from terminal_marl.training import run_p3_adaptive_kl_formal_v7 as formal_v7
from terminal_marl.training import student_rollout_v4_2
from terminal_marl.training import terminal_lifecycle_metrics


STAGE = "P3_RUNTIME_RANK_REPAIR_V1"
OBSERVATION_DIM = 88
LOW_SOC_THRESHOLD = 0.65

# These bindings intentionally come from the already registered development
# evaluation.  Do not add a seed here: the six cells are the only development
# cells available to this candidate.
SOURCE_CHECKPOINT = legacy_eval.SOURCE_CHECKPOINT
SOURCE_CHECKPOINT_SHA256 = legacy_eval.SOURCE_CHECKPOINT_SHA256
# The fit/evaluation stage has its own frozen record.  The simulation config
# itself is still obtained from legacy_eval._build_eval_inputs below; this
# path is only used for the rank-stage digest in the output binding.
PROTOCOL = Path(__file__).resolve().parents[2] / "docs/evidence/v4_2/p3_runtime_rank_repair_20260926.md"
BASELINE_RECORD_ROOT = legacy_eval.BASELINE_RECORD_ROOT
TRAIN_SEEDS = legacy_eval.TRAIN_SEEDS
VALIDATION_SEEDS = legacy_eval.VALIDATION_SEEDS
SEED_SCENARIO = legacy_eval.SEED_SCENARIO
EXPECTED_REQUESTS = legacy_eval.EXPECTED_REQUESTS

# The rank repair is still a proposal-head-only candidate.  The runtime
# shortlist and all execution semantics stay in the frozen V4.2 policy.
EXPECTED_TRAINABLE_NAMES = frozenset({"proposal_head.weight", "proposal_head.bias"})
EXPECTED_TRAINABLE_COUNT = 2425
ACCEPTED_FIT_STATUSES = frozenset({"COMPLETED", "STOPPED_KL_CAP", "COMPLETED_WITH_KL_CAP"})
ACCEPTED_SCOPE_STRINGS = frozenset({
    "proposal_head",
    "proposal_head_only",
    "proposal_head_rank_only",
    "runtime_rank_repair",
    "runtime_rank_repair_proposal_head",
    "rank_repair_proposal_head",
})


class RuntimeRankEvaluationError(ValueError):
    """Raised when a rank-repair evaluation binding is invalid."""


# A compatibility spelling is useful to callers that mirror the older
# proposal-head evaluator's exception name.
RuntimeRankRepairEvaluationError = RuntimeRankEvaluationError
ProposalHeadEvaluationError = RuntimeRankEvaluationError


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeRankEvaluationError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError, RuntimeError):
            pass
    return value


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    _require(target.is_file(), f"JSON input missing: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeRankEvaluationError(f"invalid JSON: {target}") from exc
    _require(isinstance(value, dict), f"JSON object required: {target}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _protocol_sha256() -> str:
    """Return the registered proposal evaluation protocol digest.

    The rank fit has its own stage and report.  This digest only binds the
    evaluation configuration and is not used to masquerade as the rank-fit
    protocol.
    """

    _require(PROTOCOL.is_file(), f"evaluation protocol missing: {PROTOCOL}")
    return _sha256(PROTOCOL)


def _evaluation_cells() -> list[dict[str, Any]]:
    """Use the exact six cells registered by the proposal-head evaluator."""

    cells = [dict(cell) for cell in legacy_eval._evaluation_cells()]
    expected = {
        (int(seed), str(scenario), str(split))
        for seed, scenario, split in (
            (873101, "flow6", "train"),
            (873102, "flow12", "train"),
            (873103, "dense24", "train"),
            (873104, "flow6", "validation"),
            (873105, "flow12", "validation"),
            (873106, "dense24", "validation"),
        )
    }
    observed = {
        (int(cell.get("seed")), str(cell.get("scenario")), str(cell.get("split")))
        for cell in cells
    }
    _require(observed == expected, "registered development evaluation cells changed")
    _require(len(cells) == 6, "rank evaluation must contain six development cells")
    return cells


def _split_records(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return legacy_eval._split_records(records)


def normalize_record(
    record: Mapping[str, Any],
    *,
    expected_scenario: str,
    expected_seed: int,
    expected_requests: int,
    strict: bool = True,
) -> dict[str, Any]:
    """Expose the registered lifecycle normalization unchanged."""

    return legacy_eval.normalize_record(
        record,
        expected_scenario=expected_scenario,
        expected_seed=expected_seed,
        expected_requests=expected_requests,
        strict=strict,
    )


def load_baseline_records(
    root: str | Path = BASELINE_RECORD_ROOT,
) -> list[dict[str, Any]]:
    """Load the six existing source records through the registered helper."""

    return legacy_eval.load_baseline_records(root)


def _baseline_summaries(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return legacy_eval._baseline_summaries(records)


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return legacy_eval._aggregate(records)


def _fit_coverage_improved(fit_report: Mapping[str, Any]) -> bool:
    return legacy_eval._fit_coverage_improved(fit_report)


def compare_screen(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    fit_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep the registered development screen definition unchanged.

    The runtime rank-repair fit report stores qK3 metrics under the ``train``
    split.  Adapt that nested schema to the legacy helper's flat
    ``coverage_improved`` input without changing any of the screen gates.
    """

    fit_for_legacy = fit_report
    metrics_before = fit_report.get("metrics_before")
    metrics_after = fit_report.get("metrics_after")
    if isinstance(metrics_before, Mapping) and isinstance(metrics_after, Mapping):
        train_before = metrics_before.get("train")
        train_after = metrics_after.get("train")
        before_qk3 = train_before.get("qK3_mean") if isinstance(train_before, Mapping) else None
        after_qk3 = train_after.get("qK3_mean") if isinstance(train_after, Mapping) else None
        if isinstance(before_qk3, (int, float)) and isinstance(after_qk3, (int, float)):
            fit_for_legacy = dict(fit_report)
            fit_for_legacy["coverage_improved"] = float(after_qk3) - float(before_qk3) >= 1e-6

    return legacy_eval.compare_screen(baseline, candidate, fit_for_legacy)


def _strict_leq(candidate: Any, baseline: Any) -> dict[str, Any]:
    if candidate is None or baseline is None:
        return {
            "status": "UNAVAILABLE",
            "candidate": candidate,
            "baseline": baseline,
            "passed": False,
        }
    try:
        candidate_value = float(candidate)
        baseline_value = float(baseline)
    except (TypeError, ValueError):
        return {
            "status": "INVALID",
            "candidate": candidate,
            "baseline": baseline,
            "passed": False,
        }
    return {
        "status": "CHECKED",
        "candidate": candidate_value,
        "baseline": baseline_value,
        "passed": bool(math.isfinite(candidate_value) and candidate_value <= baseline_value),
    }


def _strict_normal(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    candidate_value = int(candidate.get("normal", 0) or 0)
    baseline_value = int(baseline.get("normal", 0) or 0)
    return {
        "candidate": candidate_value,
        "baseline": baseline_value,
        "passed": candidate_value >= baseline_value,
    }


def no_regression_strict(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Report strict no-regression checks for validation and overall splits.

    This is an additional report for the primary agent.  It does not alter the
    registered ``compare_screen`` 1.05x wait screen.
    """

    checks: dict[str, Any] = {}
    for split in ("validation", "overall"):
        base = baseline.get(split, {})
        cand = candidate.get(split, {})
        priority = _strict_leq(
            cand.get("priority_weighted_wait_seconds"),
            base.get("priority_weighted_wait_seconds"),
        )
        low_soc = _strict_leq(
            cand.get("low_soc_wait_seconds"),
            base.get("low_soc_wait_seconds"),
        )
        normal = _strict_normal(cand, base)
        checks[split] = {
            "normal": normal,
            "priority_wait": priority,
            "low_soc_wait": low_soc,
            "passed": bool(normal["passed"] and priority["passed"] and low_soc["passed"]),
        }
    checks["passed"] = bool(checks["validation"]["passed"] and checks["overall"]["passed"])
    return checks


# Alias spelling retained for callers that use the noun-first form.
strict_no_regression = no_regression_strict


def _scope_value(report: Mapping[str, Any]) -> Any:
    for key in ("scope", "fit_scope", "training_scope", "trainable_scope"):
        if key in report:
            return report[key]
    invariants = report.get("invariants")
    if isinstance(invariants, Mapping):
        for key in ("scope", "fit_scope", "training_scope", "trainable_scope"):
            if key in invariants:
                return invariants[key]
        # The rank-fit runner records the concrete parameter names in
        # invariants and the human-readable scope in metadata.
        if "trainable_parameter_names" in invariants:
            return invariants["trainable_parameter_names"]
    metadata = report.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("scope", "fit_scope", "training_scope", "trainable_scope"):
            if key in metadata:
                return metadata[key]
    return None


def _scope_is_proposal_rank(value: Any) -> bool:
    if isinstance(value, Mapping):
        names = value.get("trainable_parameter_names", value.get("parameter_names"))
        return _scope_is_proposal_rank(names)
    if isinstance(value, (list, tuple, set, frozenset)):
        names = {str(item) for item in value}
        return names == set(EXPECTED_TRAINABLE_NAMES)
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in ACCEPTED_SCOPE_STRINGS:
        return True
    # Allow the explicit two-parameter scope emitted by fit implementations
    # while rejecting broad PPO/gate/critic scopes.
    if "proposal_head" in normalized and (
        "rank" in normalized or normalized in {"proposal_head", "proposal_head_only"}
    ):
        return True
    return False


def _scope_names(report: Mapping[str, Any]) -> list[str] | None:
    candidates: list[Any] = [
        report.get("trainable_parameter_names"),
        report.get("trainable_parameters"),
    ]
    invariants = report.get("invariants")
    if isinstance(invariants, Mapping):
        candidates.extend((
            invariants.get("trainable_parameter_names"),
            invariants.get("trainable_parameters"),
        ))
    for value in candidates:
        if isinstance(value, (list, tuple, set, frozenset)):
            return [str(item) for item in value]
    return None


def _restore_exact(report: Mapping[str, Any]) -> bool:
    """Read the rank fit's checkpoint restore exactness binding."""

    invariants = report.get("invariants")
    nested = invariants if isinstance(invariants, Mapping) else {}
    # A single explicit restore flag is the canonical binding.  If the fit
    # reports separate model/optimizer checks, require every supplied check.
    explicit_keys = (
        "checkpoint_restore_exact",
        "restore_exact",
        "checkpoint_roundtrip_exact",
        "checkpoint_roundtrip_allstate_exact",
        "allstate_restore_exact",
        "checkpoint_allstate_restore_exact",
        "checkpoint_roundtrip_model_exact",
        "model_restore_exact",
        "checkpoint_model_restore_exact",
    )
    optimizer_keys = (
        "checkpoint_optimizer_restore_exact",
        "optimizer_restore_exact",
        "checkpoint_roundtrip_optimizer_exact",
    )
    values: list[bool] = []
    found = False
    for key in explicit_keys + optimizer_keys:
        if key in report:
            found = True
            values.append(report[key] is True)
        if key in nested:
            found = True
            values.append(nested[key] is True)
    return bool(found and values and all(values))


def _load_fit_binding(fit_dir: str | Path) -> tuple[dict[str, Any], Path, str]:
    """Validate the new rank-fit report and resolve its candidate checkpoint."""

    root = Path(fit_dir)
    _require(root.is_dir(), f"fit directory missing: {root}")
    report_path = root / "fit_report.json"
    report = _read_json(report_path)
    _require(report.get("stage") == STAGE,
             f"fit report stage must be {STAGE}")
    _require(report.get("status") in ACCEPTED_FIT_STATUSES,
             "fit report status is not an accepted terminal status")
    _require(report.get("source_checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256,
             "fit source checkpoint SHA mismatch")

    scope = _scope_value(report)
    _require(scope is not None, "fit report rank-repair scope is missing")
    _require(_scope_is_proposal_rank(scope),
             "fit report scope is not proposal-head runtime rank repair")
    names = _scope_names(report)
    if names is not None:
        _require(set(names) == set(EXPECTED_TRAINABLE_NAMES) and len(names) == len(EXPECTED_TRAINABLE_NAMES),
                 "fit trainable parameter scope is not proposal_head only")
    count = report.get("trainable_parameter_count")
    if count is None and isinstance(report.get("invariants"), Mapping):
        count = report["invariants"].get("trainable_parameter_count")
    if count is not None:
        _require(int(count) == EXPECTED_TRAINABLE_COUNT,
                 "fit trainable parameter count is not 2425")
    _require(_restore_exact(report),
             "fit checkpoint restore exactness is not true")
    if "effective_fit" in report:
        _require(report.get("effective_fit") is True, "fit report effective_fit is not true")

    checkpoint_value = report.get("checkpoint") or report.get("candidate_checkpoint")
    _require(isinstance(checkpoint_value, str) and checkpoint_value,
             "fit report candidate checkpoint is missing")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_file():
        checkpoint = root / checkpoint_value
    _require(checkpoint.is_file(), f"candidate checkpoint missing: {checkpoint}")
    checkpoint_sha = _sha256(checkpoint)
    declared_sha = report.get("checkpoint_sha256") or report.get("candidate_checkpoint_sha256")
    _require(isinstance(declared_sha, str), "fit report checkpoint_sha256 is missing")
    _require(declared_sha == checkpoint_sha, "candidate checkpoint SHA mismatch")
    return report, checkpoint, _sha256(report_path)


def _load_policy(checkpoint: Path, config: Any) -> tuple[Any, dict[str, Any]]:
    """Delegate loading to the registered calibrated-policy loader."""

    return legacy_eval._load_policy(checkpoint, config)


def validate_candidate_freeze(source: Any, candidate: Any) -> dict[str, Any]:
    """Delegate the frozen source/candidate state check to the registered helper."""

    return legacy_eval.validate_candidate_freeze(source, candidate)


def _tensor_digest(state: Mapping[str, torch.Tensor]) -> str:
    return legacy_eval._tensor_digest(state)


def _model_digest(policy: Any) -> str:
    return _tensor_digest(policy.model.state_dict())


def _optimizer_empty(policy: Any) -> bool:
    optimizer = getattr(policy, "optimizer", None)
    state = getattr(optimizer, "state", None)
    return isinstance(state, Mapping) and len(state) == 0


def _horizon(config: Any) -> int:
    try:
        value = int(round(float(config.max_simulation_seconds) / float(config.dt)))
    except (AttributeError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise RuntimeRankEvaluationError("evaluation config has no valid simulation horizon") from exc
    _require(value > 0, "evaluation horizon must be positive")
    return value


def _empty_lifecycle(*, requests: int, error: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "scope": "incomplete_no_lifecycle_rows",
        "replay_exact": False,
        "all_requests_observed": False,
        "ledger_alignment": False,
        "requested_count": int(requests),
        "observed_terminal_count": 0,
        "row_count": 0,
        "missing_request_count": int(requests),
        "outcomes": {"unresolved": int(requests)},
        "rows": [],
    }
    if error is not None:
        value["error"] = dict(error)
    return value


def _summary_outcomes(summary: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(summary, Mapping):
        return {}
    nested_summary = summary.get("summary")
    if isinstance(nested_summary, Mapping):
        nested_outcomes = nested_summary.get("outcomes")
        if isinstance(nested_outcomes, Mapping):
            return nested_outcomes
    outcomes = summary.get("outcomes")
    return outcomes if isinstance(outcomes, Mapping) else {}


def _failure_record(
    *,
    scenario: str,
    seed: int,
    checkpoint: Path,
    requests: int,
    error: BaseException,
    summary: Mapping[str, Any] | None = None,
    weights_unchanged: bool | None = None,
    weights_before: str | None = None,
    weights_after: str | None = None,
    action_tape: Any = None,
    lifecycle: Mapping[str, Any] | None = None,
    policy_seed: int | None = None,
) -> dict[str, Any]:
    error_value = {"type": type(error).__name__, "message": str(error)}
    outcomes = dict(_summary_outcomes(summary))
    return {
        "label": "runtime_rank_candidate",
        "scenario": scenario,
        "seed": int(seed),
        "policy_seed": int(seed if policy_seed is None else policy_seed),
        "status": "failed",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
        "requests": int(requests),
        "summary": dict(summary) if isinstance(summary, Mapping) else None,
        "rollout_summary": dict(summary) if isinstance(summary, Mapping) else None,
        "action_tape": action_tape,
        # Keep an unavailable replay as ``None`` so normalization can retain
        # the rollout summary's observed ledger outcomes and add only the
        # missing request rows as unresolved.
        "lifecycle": dict(lifecycle) if isinstance(lifecycle, Mapping) else None,
        "outcomes": outcomes,
        "normal": int(outcomes.get("completed", 0) or 0),
        "emergency": int(outcomes.get("emergency_landed", 0) or 0),
        "failed_energy": int(outcomes.get("failed_energy", 0) or 0),
        "unresolved": int(outcomes.get("unresolved", 0) or 0),
        "other_failures": {},
        "error": error_value,
        "weights_unchanged": weights_unchanged,
        "weights_digest_before": weights_before,
        "weights_digest_after": weights_after,
        "weights": {
            "before": weights_before,
            "after": weights_after,
            "unchanged": weights_unchanged,
        },
        "teacher_enabled": False,
        "teacher_called": False,
        "training_or_update_called": False,
        "PPO_ALLOWED": False,
    }


def _normalize_runtime_record(
    record: Mapping[str, Any],
    *,
    scenario: str,
    seed: int,
    requests: int,
) -> dict[str, Any]:
    """Normalize while retaining records whose lifecycle replay failed."""

    value = dict(record)
    value.setdefault("scenario", scenario)
    value.setdefault("seed", int(seed))
    value.setdefault("requests", int(requests))
    value.setdefault("teacher_enabled", False)
    value.setdefault("teacher_called", False)
    value.setdefault("training_or_update_called", False)
    try:
        normalized = normalize_record(
            value,
            expected_scenario=scenario,
            expected_seed=seed,
            expected_requests=requests,
            strict=False,
        )
    except (RuntimeRankEvaluationError, legacy_eval.ProposalHeadEvaluationError) as exc:
        # A malformed secondary replay must not discard the rollout's ledger
        # summary.  Recreate the failure-inclusive denominator explicitly.
        lifecycle = value.get("lifecycle")
        lifecycle_map = lifecycle if isinstance(lifecycle, Mapping) else {}
        outcomes = dict(_summary_outcomes(value))
        if isinstance(lifecycle_map.get("outcomes"), Mapping):
            lifecycle_outcomes = dict(lifecycle_map["outcomes"])
            lifecycle_complete = bool(
                lifecycle_map.get("scope") == "all_requested_terminal_rows"
                and lifecycle_map.get("replay_exact") is True
                and lifecycle_map.get("all_requests_observed") is True
            )
            if lifecycle_complete or not outcomes:
                outcomes = lifecycle_outcomes
            else:
                # A failed replay may expose only successful rows.  Preserve
                # the rollout summary's failure outcomes and use the partial
                # replay only to fill outcome classes it observed.
                for key, item in lifecycle_outcomes.items():
                    outcomes[key] = max(int(outcomes.get(key, 0) or 0), int(item or 0))
        counts = {
            "requests": int(requests),
            "normal": int(value.get("normal", outcomes.get("completed", 0)) or 0),
            "emergency": int(value.get("emergency", outcomes.get("emergency_landed", 0)) or 0),
            "failed_energy": int(value.get("failed_energy", outcomes.get("failed_energy", 0)) or 0),
            "other": 0,
            "unresolved": int(value.get("unresolved", outcomes.get("unresolved", 0)) or 0),
        }
        raw_other = value.get("other_failures")
        if isinstance(raw_other, Mapping):
            counts["other"] = sum(int(item or 0) for item in raw_other.values())
        observed = sum(counts[key] for key in ("normal", "emergency", "failed_energy", "other", "unresolved"))
        non_unresolved = sum(counts[key] for key in ("normal", "emergency", "failed_energy", "other"))
        counts["unresolved"] = max(0, int(requests) - non_unresolved)
        value.update({
            "lifecycle_valid": False,
            "lifecycle_scope": "invalid",
            "lifecycle_error": str(exc),
            "outcome_counts": counts,
            "requests": int(requests),
            "normal": counts["normal"],
            "emergency": counts["emergency"],
            "failed_energy": counts["failed_energy"],
            "unresolved": counts["unresolved"],
            "safety_nonzero": bool(value.get("safety_invariant_error")) or legacy_eval._safety_nonzero(value),
        })
        normalized = value
    if normalized.get("lifecycle_valid") is not True:
        # legacy_eval.normalize_record intentionally prefers a partial
        # lifecycle's outcomes.  For this evaluator, retain the rollout
        # summary's failure classes as well when that lifecycle is explicitly
        # incomplete.
        summary_outcomes = _summary_outcomes(value)
        if isinstance(summary_outcomes, Mapping):
            counts = dict(normalized.get("outcome_counts") or {})
            for source_key, count_key in (
                ("completed", "normal"),
                ("emergency_landed", "emergency"),
                ("failed_energy", "failed_energy"),
                ("unresolved", "unresolved"),
            ):
                counts[count_key] = max(
                    int(counts.get(count_key, 0) or 0),
                    int(summary_outcomes.get(source_key, 0) or 0),
                )
            non_unresolved = sum(int(counts.get(key, 0) or 0) for key in (
                "normal", "emergency", "failed_energy", "other",
            ))
            counts["unresolved"] = max(0, int(requests) - non_unresolved)
            counts["requests"] = int(requests)
            normalized["outcome_counts"] = counts
    normalized["scenario"] = scenario
    normalized["seed"] = int(seed)
    normalized["requests"] = int(requests)
    counts = normalized.get("outcome_counts")
    if isinstance(counts, Mapping):
        for key in ("normal", "emergency", "failed_energy", "unresolved"):
            normalized[key] = int(counts.get(key, 0) or 0)
        other_count = int(counts.get("other", 0) or 0)
        normalized["other_failures"] = (
            {} if other_count == 0 else {"unknown": other_count}
        )
        normalized["outcomes"] = dict(
            (normalized.get("lifecycle") or {}).get("outcomes", {})
        ) if isinstance(normalized.get("lifecycle"), Mapping) else dict(
            normalized.get("outcomes") or {}
        )
    normalized["PPO_ALLOWED"] = False
    normalized["teacher_called"] = False
    normalized["training_or_update_called"] = False
    return normalized


def _runtime_record(
    *,
    policy: Any,
    checkpoint: Path,
    config: Any,
    scenario: str,
    seed: int,
    output: Path,
    smoke: bool = False,
    policy_seed: int | None = None,
) -> dict[str, Any]:
    """Run one direct student rollout and construct its full ledger record."""

    requests = int(EXPECTED_REQUESTS[scenario])
    before = _model_digest(policy)
    summary: Mapping[str, Any] | None = None
    action_tape: Any = None
    lifecycle: Mapping[str, Any] | None = None
    try:
        # Keep this call explicit: no adaptive_eval._episode_record,
        # admission helper, Teacher, diagnostics, or policy update is allowed.
        rollout_horizon = 2 if smoke else _horizon(config)
        rollout_kwargs = {
            "seed": int(seed),
            "max_steps": rollout_horizon,
            "output": output,
            "deterministic": False,
            "collect_labels": False,
        }
        # Keep old monkeypatched/test rollout call signatures compatible when
        # the optional policy-seed split is unused.
        if policy_seed is not None:
            rollout_kwargs["policy_seed"] = int(policy_seed)
        _steps, raw_summary = student_rollout_v4_2.rollout(
            policy,
            config,
            **rollout_kwargs,
        )
        _require(isinstance(raw_summary, Mapping), "student rollout summary must be a mapping")
        summary = dict(raw_summary)
        effective_policy_seed = int(seed if policy_seed is None else policy_seed)
        summary.setdefault("policy_seed", effective_policy_seed)
        action_tape = summary.get("action_tape")
        _require(action_tape is not None, "student rollout action_tape is missing")
        lifecycle = terminal_lifecycle_metrics.terminal_lifecycle_metrics(
            config, action_tape, LOW_SOC_THRESHOLD,
        )
        _require(isinstance(lifecycle, Mapping), "terminal lifecycle metrics must be a mapping")
        after = _model_digest(policy)
        unchanged = bool(before == after)
        record: dict[str, Any] = {
            "label": "runtime_rank_candidate",
            "scenario": scenario,
            "seed": int(seed),
            "policy_seed": effective_policy_seed,
            "status": "completed" if unchanged else "failed",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
            "summary": dict(summary),
            "rollout_summary": dict(summary),
            "action_tape": action_tape,
            "lifecycle": dict(lifecycle),
            "outcomes": dict(lifecycle.get("outcomes") or {}),
            "weights_unchanged": unchanged,
            "weights_digest_before": before,
            "weights_digest_after": after,
            "weights": {
                "before": before,
                "after": after,
                "unchanged": unchanged,
            },
            "teacher_enabled": False,
            "teacher_called": False,
            "training_or_update_called": False,
            "PPO_ALLOWED": False,
        }
        if not unchanged:
            record["error"] = {
                "type": "WeightsChangedDuringEvaluation",
                "message": "candidate model weights changed during direct rollout",
            }
        # The lifecycle helper derives all outcome counts from the replayed
        # ledger; do not replace those counts with successful-only summary data.
        normalized = _normalize_runtime_record(
            record, scenario=scenario, seed=seed, requests=requests,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        after = _model_digest(policy)
        normalized = _normalize_runtime_record(
            _failure_record(
                scenario=scenario,
                seed=seed,
                checkpoint=checkpoint,
                requests=requests,
                error=exc,
                summary=summary,
                weights_unchanged=(before == after),
                weights_before=before,
                weights_after=after,
                action_tape=action_tape,
                lifecycle=lifecycle,
                policy_seed=policy_seed,
            ),
            scenario=scenario,
            seed=seed,
            requests=requests,
        )
    if not smoke and normalized.get("lifecycle_valid") is not True and not normalized.get("error"):
        normalized["error"] = {
            "type": "IncompleteLifecycle",
            "message": "full-request lifecycle metrics were not available",
        }
    if smoke:
        # A two-step prefix is useful for plumbing checks only.  Force the
        # lifecycle status to remain incomplete even if a test double returns
        # a full-looking lifecycle object.
        normalized["smoke_prefix"] = True
        normalized["lifecycle_valid"] = False
        normalized["lifecycle_scope"] = "smoke_prefix"
        normalized["lifecycle_error"] = "smoke prefix is not a full-request evaluation"
        normalized["status"] = "smoke_prefix" if not normalized.get("error") else "failed"
    return normalized


def _candidate_summaries(
    records: Sequence[Mapping[str, Any]],
    *,
    smoke: bool,
) -> dict[str, dict[str, Any]]:
    if smoke:
        by_seed = {int(row["seed"]): dict(row) for row in records}
        train = [by_seed[873101]] if 873101 in by_seed else []
        return {
            "train": _aggregate(train),
            "validation": _aggregate([]),
            "overall": _aggregate(list(records)),
        }
    return {name: _aggregate(rows) for name, rows in _split_records(records).items()}


def _progress(
    output: Path,
    *,
    status: str,
    processed: int,
    expected: int,
    smoke: bool,
    error: str | None = None,
) -> None:
    value: dict[str, Any] = {
        "stage": STAGE,
        "status": status,
        "processed": int(processed),
        "expected": int(expected),
        "smoke": bool(smoke),
        "development_screen": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
    }
    if error:
        value["error"] = str(error)
    _write_json(output / "progress.json", value)


def _build_eval_inputs(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    # This is the exact adapter used by the registered proposal evaluator,
    # including its 1500/.1/radius/K3/late-off/safety-reward bindings.
    return legacy_eval._build_eval_inputs(protocol)


def _full_candidate_record(
    *,
    policy: Any,
    checkpoint: Path,
    config: Any,
    scenario: str,
    seed: int,
    output: Path,
    protocol: Mapping[str, Any] | None = None,
    runtime_protocol: Mapping[str, Any] | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Compatibility wrapper matching the older evaluator's record hook."""

    del protocol, runtime_protocol
    return _runtime_record(
        policy=policy,
        checkpoint=checkpoint,
        config=config,
        scenario=scenario,
        seed=seed,
        output=output,
        smoke=smoke,
    )


def run(
    *,
    fit_dir: str | Path,
    output_dir: str | Path,
    check_only: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    _require(not output.exists(), f"evaluation output must be fresh: {output}")
    cells = _evaluation_cells()
    selected_cells = cells[:1] if smoke else cells

    # Validate the fit and baseline before creating output artifacts.  The
    # check-only path intentionally performs no rollout.
    fit_report, candidate_checkpoint, fit_report_sha = _load_fit_binding(fit_dir)
    baseline_records = load_baseline_records()
    baseline = _baseline_summaries(baseline_records)
    protocol = formal_v7._protocol()
    configs, runtime_protocol = _build_eval_inputs(protocol)
    del runtime_protocol  # retained in the helper binding; direct rollout needs no p dict
    _require(SOURCE_CHECKPOINT.is_file(), f"source checkpoint missing: {SOURCE_CHECKPOINT}")
    _require(_sha256(SOURCE_CHECKPOINT) == SOURCE_CHECKPOINT_SHA256,
             "source checkpoint SHA mismatch")
    source_config = configs["dense24"]
    source_policy, _source_payload = _load_policy(SOURCE_CHECKPOINT, source_config)
    candidate_policy, _candidate_payload = _load_policy(candidate_checkpoint, source_config)
    freeze = validate_candidate_freeze(source_policy, candidate_policy)
    source_digest_before = _model_digest(source_policy)
    candidate_digest_before = _model_digest(candidate_policy)
    binding = {
        "stage": STAGE,
        "evaluation_protocol_sha256": _protocol_sha256(),
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": _sha256(candidate_checkpoint),
        "fit_report": str(Path(fit_dir) / "fit_report.json"),
        "fit_report_sha256": fit_report_sha,
        "fit_stage": fit_report.get("stage"),
        "fit_scope": _scope_value(fit_report),
        "fit_restore_exact": _restore_exact(fit_report),
        "freeze": freeze,
        "source_optimizer_state_empty": _optimizer_empty(source_policy),
        "candidate_optimizer_state_empty": _optimizer_empty(candidate_policy),
        "source_model_digest_before": source_digest_before,
        "candidate_model_digest_before": candidate_digest_before,
        "teacher_called": False,
        "training_or_update_called": False,
        "heldout_consumed": False,
        "smoke": bool(smoke),
        "registered_cells": cells,
    }
    _require(binding["source_optimizer_state_empty"] is True,
             "source optimizer state is not empty")
    _require(binding["candidate_optimizer_state_empty"] is True,
             "candidate optimizer state is not empty")

    if check_only:
        return {
            **binding,
            "status": "CHECK_ONLY_OK",
            "check_only": True,
            "baseline": baseline,
            "candidate": None,
            "comparison": None,
            "no_regression_strict": None,
            "records_processed": 0,
            "records_expected": len(selected_cells),
            "development_screen": False,
            "P3_QUALIFIED": False,
            "P4_ALLOWED": False,
            "checkpoint_promotable": False,
        }

    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "binding_manifest.json", {**binding, "status": "RUNNING"})
    _progress(output, status="RUNNING", processed=0, expected=len(selected_cells), smoke=smoke)
    records: list[dict[str, Any]] = []
    for index, cell in enumerate(selected_cells, start=1):
        scenario = str(cell["scenario"])
        seed = int(cell["seed"])
        record_output = output / f"rollout_{scenario}_{seed}"
        record = _runtime_record(
            policy=candidate_policy,
            checkpoint=candidate_checkpoint,
            config=configs[scenario],
            scenario=scenario,
            seed=seed,
            output=record_output,
            smoke=smoke,
        )
        records.append(record)
        _write_json(output / "records" / f"{scenario}_{seed}.json", record)
        record_status = str(record.get("status", "completed"))
        print(json.dumps({
            "stage": STAGE,
            "status": record_status,
            "scenario": scenario,
            "seed": seed,
            "processed": index,
            "expected": len(selected_cells),
            "error": record.get("error"),
        }, sort_keys=True, default=_jsonable), flush=True)
        _progress(
            output,
            status="RUNNING",
            processed=index,
            expected=len(selected_cells),
            smoke=smoke,
            error=(record.get("error") or {}).get("message") if isinstance(record.get("error"), Mapping) else None,
        )

    candidate = _candidate_summaries(records, smoke=smoke)
    candidate_digest_after = _model_digest(candidate_policy)
    candidate_weights_unchanged = bool(candidate_digest_before == candidate_digest_after)
    binding["candidate_model_digest_after"] = candidate_digest_after
    binding["candidate_weights_unchanged"] = candidate_weights_unchanged
    binding["weights"] = {
        "before": candidate_digest_before,
        "after": candidate_digest_after,
        "unchanged": candidate_weights_unchanged,
    }
    comparison = compare_screen(baseline, candidate, fit_report)
    strict = no_regression_strict(baseline, candidate)
    if smoke:
        # Smoke is a prefix plumbing check and cannot satisfy a full-request
        # or performance screen even when a test double returns good values.
        comparison = dict(comparison)
        comparison["development_screen"] = False
        comparison["smoke_prefix_only"] = True
        strict = dict(strict)
        strict["passed"] = False
        strict["smoke_prefix_only"] = True
        for split in ("validation", "overall"):
            if isinstance(strict.get(split), Mapping):
                strict[split] = dict(strict[split])
                strict[split]["passed"] = False
    any_record_error = any(bool(record.get("error")) for record in records)
    status = "FAILED" if any_record_error or not candidate_weights_unchanged else "COMPLETED"
    result = {
        **binding,
        "status": status,
        "check_only": False,
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
        "no_regression_strict": strict,
        "records": records,
        "records_processed": len(records),
        "records_expected": len(selected_cells),
        "requests_expected": 6 if smoke else 84,
        "requests_observed": int(candidate["overall"]["counts"]["requests"]),
        "development_only": True,
        "development_screen": bool(comparison.get("development_screen")) if not smoke else False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
    }
    _write_json(output / "binding_manifest.json", {**binding, "status": status})
    _write_json(output / "manifest.json", result)
    _progress(
        output,
        status=status,
        processed=len(records),
        expected=len(selected_cells),
        smoke=smoke,
        error="one or more records failed or weights changed" if status == "FAILED" else None,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(
            fit_dir=args.fit_dir,
            output_dir=args.output_dir,
            check_only=args.check_only,
            smoke=args.smoke,
        )
    except (RuntimeRankEvaluationError, FileNotFoundError, ValueError, OSError) as exc:
        print(json.dumps({"stage": STAGE, "status": "ERROR", "error": str(exc)}, sort_keys=True), flush=True)
        return 2
    print(json.dumps({
        "stage": STAGE,
        "status": result.get("status"),
        "records_processed": result.get("records_processed", 0),
        "development_screen": result.get("development_screen", False),
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
    }, sort_keys=True), flush=True)
    return 0 if result.get("status") in {"CHECK_ONLY_OK", "COMPLETED"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
