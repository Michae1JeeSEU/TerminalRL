"""Candidate-only evaluation for the registered P3 proposal-head probe.

The evaluator consumes one proposal-head fit checkpoint and runs the candidate
on the six existing development seeds.  The source baseline is read from the
already-computed full-request lifecycle records; it is never replayed here.
This module does not fit, update, invoke Teacher, use heldout seeds, or promote
a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATED_ARCHITECTURE_ID,
    CALIBRATION_TEMPERATURE,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.training import evaluate_p3_adaptive_kl_v7 as adaptive_eval
from terminal_marl.training import run_p3_adaptive_kl_formal_v7 as formal_v7
from terminal_marl.training import terminal_lifecycle_metrics as lifecycle_metrics


STAGE = "P3_PROPOSAL_HEAD_EVAL_V1"
PROTOCOL = Path("docs/evidence/v4_2/p3_proposal_head_probe_v1_20260926.md")
SOURCE_CHECKPOINT_SHA256 = (
    "03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba"
)
SOURCE_CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_gate_coverage_repair_v2_20260925/fit853401/p3_gate_coverage_repair_v2.pt"
)
BASELINE_RECORD_ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_adaptive_kl_terminal_dev_metrics_20260926/records/run1_dev_before_training"
)
OBSERVATION_DIM = 88
LOW_SOC_THRESHOLD = 0.65
TRAIN_SEEDS = frozenset({873101, 873102, 873103})
VALIDATION_SEEDS = frozenset({873104, 873105, 873106})
SEED_SCENARIO = {
    873101: "flow6",
    873102: "flow12",
    873103: "dense24",
    873104: "flow6",
    873105: "flow12",
    873106: "dense24",
}
EXPECTED_REQUESTS = {"flow6": 6, "flow12": 12, "dense24": 24}
EXPECTED_PROPOSAL_NAMES = frozenset({"proposal_head.weight", "proposal_head.bias"})
# These are the safety violation counters emitted by the student rollout.  In
# particular, shield use/rejections and shortlist/check counts are telemetry,
# not safety failures for this evaluator.
SAFETY_VIOLATION_KEYS = frozenset({
    "collisions",
    "nmac",
    "hardseparationviolations",
    "uncertifiedexecution",
    "safetyinvarianterror",
})


class ProposalHeadEvaluationError(ValueError):
    """Raised when a fit, baseline, or evaluation binding is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProposalHeadEvaluationError(message)


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
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    return value


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    _require(target.is_file(), f"JSON input missing: {target}")
    value = json.loads(target.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {target}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _protocol_sha256() -> str:
    _require(PROTOCOL.is_file(), f"proposal protocol missing: {PROTOCOL}")
    return _sha256(PROTOCOL)


def _evaluation_cells() -> list[dict[str, Any]]:
    return [
        {"seed": seed, "scenario": SEED_SCENARIO[seed], "split": (
            "train" if seed in TRAIN_SEEDS else "validation"
        )}
        for seed in sorted(SEED_SCENARIO)
    ]


def _split_records(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Partition the exact six registered seeds without silently reassigning rows."""
    by_seed: dict[int, dict[str, Any]] = {}
    for value in records:
        if not isinstance(value, Mapping):
            raise ProposalHeadEvaluationError("evaluation record must be a mapping")
        try:
            seed = int(value["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProposalHeadEvaluationError("evaluation record seed is missing") from exc
        if seed in by_seed:
            raise ProposalHeadEvaluationError(f"duplicate evaluation seed: {seed}")
        if seed not in SEED_SCENARIO:
            raise ProposalHeadEvaluationError(f"unexpected development seed: {seed}")
        expected_scenario = SEED_SCENARIO[seed]
        if str(value.get("scenario")) != expected_scenario:
            raise ProposalHeadEvaluationError(
                f"seed {seed} scenario mismatch: {value.get('scenario')} != {expected_scenario}"
            )
        by_seed[seed] = dict(value)
    missing = sorted(set(SEED_SCENARIO) - set(by_seed))
    if missing:
        raise ProposalHeadEvaluationError(f"missing development seeds: {missing}")
    return {
        "train": [by_seed[seed] for seed in sorted(TRAIN_SEEDS)],
        "validation": [by_seed[seed] for seed in sorted(VALIDATION_SEEDS)],
        "overall": [by_seed[seed] for seed in sorted(SEED_SCENARIO)],
    }


def _full_lifecycle(value: Mapping[str, Any]) -> Mapping[str, Any]:
    lifecycle = value.get("lifecycle")
    if not isinstance(lifecycle, Mapping):
        lifecycle = value.get("full_metrics")
    _require(isinstance(lifecycle, Mapping), "full-request lifecycle metrics are required")
    _require(lifecycle.get("scope") == "all_requested_terminal_rows",
             "success_only_legacy lifecycle records cannot pass development screening")
    _require(lifecycle.get("replay_exact") is True, "lifecycle replay_exact is not true")
    _require(lifecycle.get("all_requests_observed") is True,
             "lifecycle does not observe all requested rows")
    _require(int(lifecycle.get("missing_request_count", 0) or 0) == 0,
             "lifecycle has missing request rows")
    return lifecycle


def _outcome_counts(value: Mapping[str, Any], lifecycle: Mapping[str, Any]) -> dict[str, int]:
    outcomes = lifecycle.get("outcomes") or value.get("outcomes") or {}
    if not isinstance(outcomes, Mapping):
        outcomes = {}
    normal = int(value.get("normal", outcomes.get("completed", 0)) or 0)
    emergency = int(value.get("emergency", outcomes.get("emergency_landed", 0)) or 0)
    failed_energy = int(value.get("failed_energy", outcomes.get("failed_energy", 0)) or 0)
    unresolved = int(value.get("unresolved", outcomes.get("unresolved", 0)) or 0)
    known = {"completed", "emergency_landed", "failed_energy", "unresolved"}
    outcome_other = sum(
        int(item or 0) for key, item in outcomes.items() if str(key) not in known
    )
    raw_other = value.get("other_failures") or {}
    raw_other_count = sum(int(item or 0) for item in raw_other.values()) if isinstance(raw_other, Mapping) else 0
    other = max(outcome_other, raw_other_count)
    return {
        "requests": int(lifecycle.get("requested_count", value.get("requests", 0)) or 0),
        "normal": normal,
        "emergency": emergency,
        "failed_energy": failed_energy,
        "other": other,
        "unresolved": unresolved,
    }


def normalize_record(
    record: Mapping[str, Any],
    *,
    expected_scenario: str,
    expected_seed: int,
    expected_requests: int,
    strict: bool = True,
) -> dict[str, Any]:
    """Normalize a baseline or rollout record while retaining lifecycle errors."""
    normalized = dict(record)
    normalized["scenario"] = str(record.get("scenario", expected_scenario))
    normalized["seed"] = int(record.get("seed", expected_seed))
    if normalized["scenario"] != expected_scenario or normalized["seed"] != int(expected_seed):
        raise ProposalHeadEvaluationError("record scenario/seed binding mismatch")
    try:
        lifecycle = _full_lifecycle(record)
    except ProposalHeadEvaluationError as exc:
        if strict:
            raise
        # Preserve the rollout's observed outcomes even when its secondary
        # lifecycle replay is incomplete.  The incomplete rows remain a hard
        # screening failure, while an observed failed_energy row must not be
        # erased by the normalization fallback.
        lifecycle_candidate = record.get("lifecycle")
        if not isinstance(lifecycle_candidate, Mapping):
            lifecycle_candidate = record.get("full_metrics")
        if not isinstance(lifecycle_candidate, Mapping):
            lifecycle_candidate = {}
        observed_counts = _outcome_counts(record, lifecycle_candidate)
        for key in ("normal", "emergency", "failed_energy", "other", "unresolved"):
            observed_counts[key] = max(0, int(observed_counts.get(key, 0) or 0))
        observed = sum(observed_counts[key] for key in (
            "normal", "emergency", "failed_energy", "other", "unresolved"
        ))
        observed_counts["requests"] = int(expected_requests)
        observed_counts["unresolved"] += max(0, int(expected_requests) - observed)
        normalized.update({
            "lifecycle_scope": "invalid",
            "lifecycle_valid": False,
            "lifecycle_error": str(exc),
            "outcome_counts": observed_counts,
            "safety_nonzero": bool(record.get("safety_invariant_error")) or _safety_nonzero(record),
        })
        return normalized
    requests = int(lifecycle.get("requested_count", 0) or 0)
    row_count = int(lifecycle.get("row_count", lifecycle.get("observed_terminal_count", 0)) or 0)
    _require(requests == int(expected_requests),
             f"request count mismatch for {expected_scenario}/{expected_seed}: {requests}")
    _require(row_count == requests, "full-request lifecycle row count does not match requests")
    counts = _outcome_counts(record, lifecycle)
    observed = sum(counts[key] for key in ("normal", "emergency", "failed_energy", "other", "unresolved"))
    _require(observed == requests, "outcome counts do not match full request denominator")
    normalized.update({
        "lifecycle": dict(lifecycle),
        "lifecycle_scope": "all_requested_terminal_rows",
        "lifecycle_valid": True,
        "outcome_counts": counts,
        "safety_nonzero": bool(record.get("safety_invariant_error")) or _safety_nonzero(record),
    })
    return normalized


def _safety_nonzero(record: Mapping[str, Any]) -> bool:
    if bool(record.get("safety_invariant_error")):
        return True
    sources: list[Mapping[str, Any]] = []
    for value in (record.get("safety"), record.get("safety_counters")):
        if isinstance(value, Mapping):
            sources.append(value)
    summary = record.get("summary")
    if isinstance(summary, Mapping):
        for value in (summary.get("safety"), summary.get("safety_counters")):
            if isinstance(value, Mapping):
                sources.append(value)
    for safety in sources:
        # Accept capitalization and underscores used by rollout summaries
        # without broadening the set to routine telemetry fields.
        normalized = {
            str(name).lower().replace("_", ""): value
            for name, value in safety.items()
        }
        for key in SAFETY_VIOLATION_KEYS:
            value = normalized.get(key)
            if isinstance(value, bool):
                if value:
                    return True
                continue
            try:
                if float(value) != 0.0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def load_baseline_records(
    root: str | Path = BASELINE_RECORD_ROOT,
) -> list[dict[str, Any]]:
    """Load exactly the six existing full-request source records."""
    target = Path(root)
    _require(target.is_dir(), f"baseline record root missing: {target}")
    files = sorted(target.glob("*.json"))
    _require(len(files) == 6, f"expected six baseline records, found {len(files)}")
    records: list[dict[str, Any]] = []
    for path in files:
        raw = _read_json(path)
        seed = int(raw.get("seed"))
        scenario = str(raw.get("scenario"))
        normalized = normalize_record(
            raw,
            expected_scenario=SEED_SCENARIO[seed],
            expected_seed=seed,
            expected_requests=EXPECTED_REQUESTS[scenario],
            strict=True,
        )
        normalized["baseline_path"] = str(path)
        records.append(normalized)
    _split_records(records)
    return records


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {key: 0 for key in ("requests", "normal", "emergency", "failed_energy", "other", "unresolved")}
    rows: list[Mapping[str, Any]] = []
    invalid_lifecycle = 0
    safety_nonzero = 0
    safety_unknown = 0
    errors = 0
    for record in records:
        outcome = record.get("outcome_counts") or {}
        for key in counts:
            counts[key] += int(outcome.get(key, 0) or 0)
        if record.get("lifecycle_valid") is not True:
            invalid_lifecycle += 1
        if "safety_nonzero" not in record or record.get("safety_nonzero") is None:
            safety_unknown += 1
        elif record.get("safety_nonzero"):
            safety_nonzero += 1
        if record.get("error") or record.get("validation_errors"):
            errors += 1
        lifecycle = record.get("lifecycle")
        if isinstance(lifecycle, Mapping) and isinstance(lifecycle.get("rows"), list):
            rows.extend(row for row in lifecycle["rows"] if isinstance(row, Mapping))
    requested = counts["requests"]
    lifecycle = lifecycle_metrics.summarize_terminal_rows(
        rows,
        low_soc_threshold=LOW_SOC_THRESHOLD,
        requested_count=requested,
        enforce_unique=False,
    ) if rows else {
        "scope": "incomplete_no_lifecycle_rows",
        "requested_count": requested,
        "missing_request_count": requested,
        "all_requests_observed": False,
        "priority_weighted_wait_seconds": None,
        "low_soc": {"mean_wait_seconds": None, "count": 0},
    }
    lifecycle = dict(lifecycle)
    lifecycle["all_requests_observed"] = bool(
        int(lifecycle.get("missing_request_count", 0) or 0) == 0
        and int(lifecycle.get("observed_terminal_count", 0) or 0) == requested
    )
    lifecycle["replay_exact"] = bool(
        invalid_lifecycle == 0
        and all(
            isinstance(record.get("lifecycle"), Mapping)
            and record["lifecycle"].get("replay_exact") is True
            for record in records
        )
    )
    return {
        "episodes": len(records),
        "counts": counts,
        "normal": counts["normal"],
        "emergency": counts["emergency"],
        "failed_energy": counts["failed_energy"],
        "other": counts["other"],
        "unresolved": counts["unresolved"],
        "lifecycle": lifecycle,
        "lifecycle_scope": lifecycle.get("scope"),
        "full_request_complete": (
            invalid_lifecycle == 0
            and lifecycle.get("scope") == "all_requested_terminal_rows"
            and lifecycle.get("replay_exact") is True
            and lifecycle.get("all_requests_observed") is True
            and int(lifecycle.get("missing_request_count", 0) or 0) == 0
        ),
        "invalid_lifecycle_records": invalid_lifecycle,
        "safety_nonzero_episodes": safety_nonzero,
        "safety_status_unknown_episodes": safety_unknown,
        "error_records": errors,
        "priority_weighted_wait_seconds": lifecycle.get("priority_weighted_wait_seconds"),
        "low_soc_wait_seconds": (lifecycle.get("low_soc") or {}).get("mean_wait_seconds"),
        "missing_request_count": int(lifecycle.get("missing_request_count", requested) or 0),
    }


def _metric_gate(candidate: Any, baseline: Any) -> dict[str, Any]:
    if candidate is None or baseline is None:
        return {"status": "NA", "passed": None, "candidate": candidate, "baseline": baseline}
    candidate_value = float(candidate)
    baseline_value = float(baseline)
    limit = baseline_value * 1.05
    return {
        "status": "CHECKED",
        "passed": bool(math.isfinite(candidate_value) and candidate_value <= limit),
        "candidate": candidate_value,
        "baseline": baseline_value,
        "limit_1_05x": limit,
    }


def _normal_gate(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    candidate_value = int(candidate.get("normal", 0))
    baseline_value = int(baseline.get("normal", 0))
    return {
        "candidate": candidate_value,
        "baseline": baseline_value,
        "passed": candidate_value >= baseline_value,
    }


def _fit_coverage_improved(fit_report: Mapping[str, Any]) -> bool:
    for key in ("qK3_coverage_improved", "qk3_coverage_improved", "coverage_improved"):
        if key in fit_report:
            return bool(fit_report[key])
    metrics_before = fit_report.get("metrics_before") or {}
    metrics_after = fit_report.get("metrics_after") or {}
    for key in ("qK3", "qk3", "qK3_coverage", "qk3_coverage", "coverage"):
        before = metrics_before.get(key) if isinstance(metrics_before, Mapping) else None
        after = metrics_after.get(key) if isinstance(metrics_after, Mapping) else None
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            return float(after) - float(before) >= 1e-6
    return False


def compare_screen(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    fit_report: Mapping[str, Any],
) -> dict[str, Any]:
    split_checks: dict[str, Any] = {}
    for split in ("train", "validation", "overall"):
        base = baseline[split]
        cand = candidate[split]
        priority = _metric_gate(
            cand.get("priority_weighted_wait_seconds"),
            base.get("priority_weighted_wait_seconds"),
        )
        low_soc = _metric_gate(cand.get("low_soc_wait_seconds"), base.get("low_soc_wait_seconds"))
        normal = _normal_gate(cand, base)
        split_checks[split] = {
            "full_request_complete": bool(cand.get("full_request_complete")),
            "safety_energy_other_unresolved_zero": all(
                int(cand.get(key, 0) or 0) == 0
                for key in ("failed_energy", "other", "unresolved")
            ) and int(cand.get("safety_nonzero_episodes", 0) or 0) == 0
            and int(cand.get("safety_status_unknown_episodes", 0) or 0) == 0,
            "normal": normal,
            "priority_wait": priority,
            "low_soc_wait": low_soc,
        }
    closed_loop_improvement = any(
        (
            split_checks[split]["normal"]["passed"]
            and int(candidate[split].get("normal", 0)) > int(baseline[split].get("normal", 0))
        )
        or (
            split_checks[split]["priority_wait"]["passed"] is True
            and candidate[split].get("priority_weighted_wait_seconds") is not None
            and baseline[split].get("priority_weighted_wait_seconds") is not None
            and float(candidate[split]["priority_weighted_wait_seconds"]) <
            float(baseline[split]["priority_weighted_wait_seconds"])
        )
        or (
            split_checks[split]["low_soc_wait"]["passed"] is True
            and candidate[split].get("low_soc_wait_seconds") is not None
            and baseline[split].get("low_soc_wait_seconds") is not None
            and float(candidate[split]["low_soc_wait_seconds"]) <
            float(baseline[split]["low_soc_wait_seconds"])
        )
        for split in ("train", "validation", "overall")
    )
    coverage_improved = _fit_coverage_improved(fit_report)
    for split in ("validation", "overall"):
        check = split_checks[split]
        wait_ok = all(
            item["passed"] is not False
            for item in (check["priority_wait"], check["low_soc_wait"])
        )
        check["screen_pass"] = bool(
            check["full_request_complete"]
            and check["safety_energy_other_unresolved_zero"]
            and check["normal"]["passed"]
            and wait_ok
        )
    development_screen = bool(
        split_checks["overall"]["screen_pass"]
        and split_checks["validation"]["screen_pass"]
        and (coverage_improved or closed_loop_improvement)
    )
    return {
        "development_screen": development_screen,
        "split_checks": split_checks,
        "coverage_improved": coverage_improved,
        "closed_loop_improvement": closed_loop_improvement,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
    }


def _baseline_summaries(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    split = _split_records(records)
    return {name: _aggregate(values) for name, values in split.items()}


def _load_fit_binding(fit_dir: str | Path) -> tuple[dict[str, Any], Path, str]:
    root = Path(fit_dir)
    _require(root.is_dir(), f"fit directory missing: {root}")
    report_path = root / "fit_report.json"
    report = _read_json(report_path)
    _require(report.get("status") in {"COMPLETED", "STOPPED_KL_CAP"},
             "fit report status is not an accepted terminal status")
    _require(report.get("source_checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256,
             "fit source checkpoint SHA mismatch")
    _require(report.get("protocol_sha256") == _protocol_sha256(),
             "fit proposal protocol SHA mismatch")
    _require(report.get("effective_fit") is True,
             "fit report must declare effective_fit=true")
    checkpoint_value = report.get("checkpoint")
    _require(isinstance(checkpoint_value, str) and checkpoint_value,
             "fit report candidate checkpoint is missing")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_file():
        checkpoint = root / checkpoint_value
    _require(checkpoint.is_file(), f"candidate checkpoint missing: {checkpoint}")
    checkpoint_sha = _sha256(checkpoint)
    _require(isinstance(report.get("checkpoint_sha256"), str),
             "fit report checkpoint_sha256 is missing")
    _require(report["checkpoint_sha256"] == checkpoint_sha,
             "candidate checkpoint SHA mismatch")
    invariants = report.get("invariants")
    _require(isinstance(invariants, Mapping), "fit report invariants are missing")
    trainable_names = invariants.get("trainable_parameter_names")
    _require(isinstance(trainable_names, list)
             and set(trainable_names) == EXPECTED_PROPOSAL_NAMES
             and len(trainable_names) == len(EXPECTED_PROPOSAL_NAMES),
             "fit trainable parameter names are not proposal_head only")
    _require(invariants.get("trainable_parameter_count") == 2425,
             "fit trainable parameter count is not 2425")
    for key in (
        "nonproposal_state_bitwise_unchanged",
        "gate_logits_bitwise_unchanged",
        "all_nonproposal_parameters_frozen",
        "checkpoint_roundtrip_model_exact",
        "checkpoint_roundtrip_optimizer_exact",
    ):
        _require(invariants.get(key) is True, f"fit invariant failed: {key}")
    _require(report.get("checkpoint_roundtrip_model_exact") is True,
             "fit checkpoint model roundtrip is not exact")
    _require(report.get("checkpoint_roundtrip_optimizer_exact") is True,
             "fit checkpoint optimizer roundtrip is not exact")
    binding = {
        "fit_dir": str(root),
        "fit_report": str(report_path),
        "fit_report_sha256": _sha256(report_path),
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "protocol_sha256": str(report["protocol_sha256"]),
        "status": str(report["status"]),
        "effective_fit": True,
        "trainable_parameter_names": list(trainable_names),
        "trainable_parameter_count": 2425,
    }
    return report, checkpoint, binding["fit_report_sha256"]


def _new_policy(*, learning_rate: float = 3e-5) -> Any:
    return TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM,
        temperature=CALIBRATION_TEMPERATURE,
        learning_rate=float(learning_rate),
    )


def _load_policy(checkpoint: Path, config: Any) -> tuple[Any, dict[str, Any]]:
    policy = _new_policy()
    payload = load_checkpoint(checkpoint, policy, config, load_optimizer=False)
    _require(getattr(policy.model, "architecture_id", None) == CALIBRATED_ARCHITECTURE_ID,
             "candidate architecture changed")
    _require(int(policy.observation_dim) == OBSERVATION_DIM, "candidate observation dimension changed")
    _require(float(policy.model.gate_temperature.detach().cpu().item()) == CALIBRATION_TEMPERATURE,
             "candidate temperature changed")
    _require(len(policy.optimizer.state) == 0, "candidate optimizer state was restored")
    policy.model.eval()
    return policy, dict(payload) if isinstance(payload, Mapping) else {}


def _tensor_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_candidate_freeze(source: Any, candidate: Any) -> dict[str, Any]:
    source_state = source.model.state_dict()
    candidate_state = candidate.model.state_dict()
    _require(set(source_state) == set(candidate_state), "source/candidate state keys changed")
    proposal_names = {name for name in candidate_state if name.startswith("proposal_head.")}
    _require(proposal_names == EXPECTED_PROPOSAL_NAMES,
             f"proposal head state scope changed: {sorted(proposal_names)}")
    frozen_names = sorted(set(candidate_state) - proposal_names - {"gate_temperature"})
    frozen_unchanged = all(torch.equal(source_state[name], candidate_state[name]) for name in frozen_names)
    _require(frozen_unchanged, "non-proposal candidate state changed")
    proposal_changed = any(not torch.equal(source_state[name], candidate_state[name]) for name in proposal_names)
    _require(proposal_changed, "candidate proposal head has no nonzero parameter update")
    observation = torch.zeros((1, OBSERVATION_DIM), dtype=torch.float32)
    with torch.no_grad():
        source_gate, _ = source.model.distributions(observation)
        candidate_gate, _ = candidate.model.distributions(observation)
    gate_unchanged = torch.equal(source_gate.logits, candidate_gate.logits)
    _require(gate_unchanged, "candidate gate logits changed")
    return {
        "proposal_parameter_names": sorted(proposal_names),
        "proposal_parameter_changed": proposal_changed,
        "frozen_parameter_bitwise_unchanged": frozen_unchanged,
        "gate_logits_bitwise_unchanged": gate_unchanged,
        "source_model_digest": _tensor_digest(source_state),
        "candidate_model_digest": _tensor_digest(candidate_state),
    }


def _full_candidate_record(
    *,
    policy: Any,
    checkpoint: Path,
    protocol: Mapping[str, Any],
    runtime_protocol: Mapping[str, Any],
    config: Any,
    scenario: str,
    seed: int,
    output: Path,
) -> dict[str, Any]:
    raw = adaptive_eval._episode_record(
        policy=policy,
        label="proposal_head_candidate",
        checkpoint=checkpoint,
        config=config,
        protocol=protocol,
        runtime_protocol=runtime_protocol,
        scenario=scenario,
        seed=int(seed),
        output=output,
        require_full_resolution=True,
    )
    return normalize_record(
        raw,
        expected_scenario=scenario,
        expected_seed=seed,
        expected_requests=EXPECTED_REQUESTS[scenario],
        strict=False,
    )


def _build_eval_inputs(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build configs separately from the V6 runtime protocol adapter."""
    configs, _config_snapshots = formal_v7._build_configs(protocol)
    runtime_protocol = adaptive_eval.v6._runtime_protocol(protocol)
    return dict(configs), runtime_protocol


def _lifecycle_outcome_counts(
    lifecycle: Mapping[str, Any], *, expected_requests: int,
) -> dict[str, int]:
    outcomes = lifecycle.get("outcomes") or {}
    _require(isinstance(outcomes, Mapping), "lifecycle outcomes are missing")
    known = {"completed", "emergency_landed", "failed_energy", "unresolved"}
    counts = {
        "requests": int(expected_requests),
        "normal": int(outcomes.get("completed", 0) or 0),
        "emergency": int(outcomes.get("emergency_landed", 0) or 0),
        "failed_energy": int(outcomes.get("failed_energy", 0) or 0),
        "other": int(sum(int(value or 0) for key, value in outcomes.items()
                         if str(key) not in known)),
        "unresolved": int(outcomes.get("unresolved", 0) or 0),
    }
    observed = sum(counts[key] for key in (
        "normal", "emergency", "failed_energy", "other", "unresolved"
    ))
    _require(observed == int(expected_requests),
             "lifecycle outcomes do not match the requested denominator")
    return counts


def _safety_snapshot(record: Mapping[str, Any]) -> tuple[dict[str, Any] | None, bool | None]:
    summary = record.get("summary")
    safety = summary.get("safety") if isinstance(summary, Mapping) else None
    if not isinstance(safety, Mapping):
        return None, None
    snapshot = {str(key): value for key, value in safety.items()}
    return snapshot, _safety_nonzero({"summary": {"safety": snapshot}})


def summarize_existing_eval(
    *, input_root: str | Path, output_dir: str | Path,
) -> dict[str, Any]:
    """Correct only the reporting fields of one known failed evaluation.

    This path reads the exact lifecycle artifacts already produced by the
    evaluation.  It never replays tapes, loads a policy, or performs an
    environment step, and it always retains the original evaluation error.
    """
    source_root = Path(input_root)
    output = Path(output_dir)
    _require(source_root.is_dir(), f"existing evaluation root missing: {source_root}")
    _require(not output.exists(), f"corrected output must be fresh: {output}")
    _require(output.resolve() != source_root.resolve(),
             "corrected output must not overwrite the source evaluation")
    manifest_path = source_root / "manifest.json"
    original_manifest = _read_json(manifest_path)
    original_manifest_sha = _sha256(manifest_path)
    record_paths = sorted((source_root / "records").glob("*.json"))
    _require(len(record_paths) == len(SEED_SCENARIO),
             f"expected six original records, found {len(record_paths)}")
    corrected_records: list[dict[str, Any]] = []
    record_shas: dict[str, str] = {}
    for path in record_paths:
        raw = _read_json(path)
        record_sha = _sha256(path)
        record_shas[path.name] = record_sha
        try:
            seed = int(raw["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProposalHeadEvaluationError(f"record seed missing: {path}") from exc
        _require(seed in SEED_SCENARIO, f"unexpected original seed: {seed}")
        scenario = str(raw.get("scenario"))
        _require(scenario == SEED_SCENARIO[seed],
                 f"seed {seed} scenario mismatch: {scenario}")
        error = raw.get("error")
        _require(isinstance(error, Mapping)
                 and error.get("type") == "KeyError"
                 and error.get("message") == "'hard_gates'",
                 f"record {seed} has an unexpected original error")
        _require(raw.get("status") == "failed", f"record {seed} status changed")
        _require(raw.get("weights_unchanged") is True,
                 f"record {seed} weights_unchanged is not true")
        summary = raw.get("summary")
        _require(isinstance(summary, Mapping) and summary.get("terminal") is True,
                 f"record {seed} summary is not terminal")
        lifecycle = _full_lifecycle(raw)
        _require(lifecycle.get("ledger_alignment") is True,
                 f"record {seed} lifecycle ledger alignment is not true")
        _require(int(lifecycle.get("row_count", 0) or 0)
                 == EXPECTED_REQUESTS[scenario],
                 f"record {seed} lifecycle row count mismatch")
        counts = _lifecycle_outcome_counts(
            lifecycle, expected_requests=EXPECTED_REQUESTS[scenario],
        )
        safety, safety_nonzero = _safety_snapshot(raw)
        corrected = dict(raw)
        corrected.update({
            "original_evaluation_error": dict(error),
            "original_record_sha256": record_sha,
            "reporting_error_corrected": True,
            "training_pass": False,
            "admission_gates_reconstructed": False,
            "admission_gate_limitation": (
                "rollout admission gates were not reconstructed after the hard_gates error"
            ),
            "outcome_counts": counts,
            "requests": counts["requests"],
            "normal": counts["normal"],
            "emergency": counts["emergency"],
            "failed_energy": counts["failed_energy"],
            "other_failures": {} if counts["other"] == 0 else {"unknown": counts["other"]},
            "unresolved": counts["unresolved"],
            "safety_counters": safety,
            "safety_nonzero": safety_nonzero,
            "lifecycle_valid": True,
            "lifecycle_scope": "all_requested_terminal_rows",
        })
        corrected_records.append(corrected)
    _require(len({int(record["seed"]) for record in corrected_records}) == len(SEED_SCENARIO),
             "original records contain duplicate seeds")
    corrected_records.sort(key=lambda record: int(record["seed"]))
    split = _split_records(corrected_records)
    baseline = _baseline_summaries(load_baseline_records())
    candidate = {name: _aggregate(values) for name, values in split.items()}
    comparison = compare_screen(baseline, candidate, {"coverage_improved": False})
    comparison.update({
        "development_screen": False,
        "checkpoint_promotable": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "reporting_error_corrected": True,
        "admission_gates_reconstructed": False,
        "screen_blocked_by_original_evaluation_errors": True,
    })
    output.mkdir(parents=True, exist_ok=False)
    for record in corrected_records:
        _write_json(output / "records" / f"{record['scenario']}_{int(record['seed'])}.json", record)
    result = {
        "stage": STAGE,
        "status": "REPORTING_ERROR_CORRECTED",
        "source_evaluation_root": str(source_root),
        "original_manifest_sha256": original_manifest_sha,
        "original_record_sha256": record_shas,
        "original_manifest_status": original_manifest.get("status"),
        "reporting_error_corrected": True,
        "original_error": {"type": "KeyError", "message": "'hard_gates'"},
        "admission_gates_reconstructed": False,
        "admission_gate_limitation": (
            "The post-rollout admission gates were not reconstructed; this report is not a training pass."
        ),
        "environment_replayed": False,
        "policy_forward_executed": False,
        "training_or_update_called": False,
        "baseline": baseline,
        "candidate": candidate,
        "records_processed": len(corrected_records),
        "records_expected": len(SEED_SCENARIO),
        "comparison": comparison,
        "development_screen": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
    }
    _write_json(output / "manifest.json", result)
    _write_json(output / "progress.json", {
        "stage": STAGE, "status": result["status"], "records_processed": len(corrected_records),
        "reporting_error_corrected": True, "P3_QUALIFIED": False, "P4_ALLOWED": False,
    })
    return result


def run(
    *,
    fit_dir: str | Path,
    output_dir: str | Path,
    check_only: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    _require(not output.exists(), f"evaluation output must be fresh: {output}")
    protocol_sha = _protocol_sha256()
    fit_report, candidate_checkpoint, fit_report_sha = _load_fit_binding(fit_dir)
    baseline_records = load_baseline_records()
    baseline = _baseline_summaries(baseline_records)
    protocol = formal_v7._protocol()
    configs, runtime_protocol = _build_eval_inputs(protocol)
    source_config = configs["dense24"]
    _require(SOURCE_CHECKPOINT.is_file(), f"source checkpoint missing: {SOURCE_CHECKPOINT}")
    _require(_sha256(SOURCE_CHECKPOINT) == SOURCE_CHECKPOINT_SHA256,
             "source checkpoint SHA mismatch")
    source_policy, source_payload = _load_policy(SOURCE_CHECKPOINT, source_config)
    candidate_policy, candidate_payload = _load_policy(candidate_checkpoint, source_config)
    freeze = validate_candidate_freeze(source_policy, candidate_policy)
    binding = {
        "stage": STAGE,
        "protocol_sha256": protocol_sha,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": _sha256(candidate_checkpoint),
        "fit_report_sha256": fit_report_sha,
        "freeze": freeze,
        "source_optimizer_state_empty": len(source_policy.optimizer.state) == 0,
        "candidate_optimizer_state_empty": len(candidate_policy.optimizer.state) == 0,
        "teacher_called": False,
        "training_or_update_called": False,
        "heldout_consumed": False,
    }
    if check_only:
        return {
            **binding,
            "status": "CHECK_ONLY_OK",
            "check_only": True,
            "baseline": baseline,
            "candidate": None,
            "comparison": None,
            "development_screen": False,
            "P3_QUALIFIED": False,
            "P4_ALLOWED": False,
        }

    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "binding_manifest.json", {**binding, "status": "RUNNING"})
    candidate_records: list[dict[str, Any]] = []
    for cell in _evaluation_cells():
        scenario = str(cell["scenario"])
        seed = int(cell["seed"])
        record = _full_candidate_record(
            policy=candidate_policy,
            checkpoint=candidate_checkpoint,
            protocol=protocol,
            runtime_protocol=runtime_protocol,
            config=configs[scenario],
            scenario=scenario,
            seed=seed,
            output=output / f"candidate_{scenario}_{seed}",
        )
        candidate_records.append(record)
        _write_json(output / "records" / f"{scenario}_{seed}.json", record)
        _write_json(output / "progress.json", {
            "stage": STAGE,
            "status": "RUNNING",
            "processed": len(candidate_records),
            "expected": len(SEED_SCENARIO),
            "P3_QUALIFIED": False,
            "P4_ALLOWED": False,
        })
    candidate = _baseline_summaries(candidate_records)
    comparison = compare_screen(baseline, candidate, fit_report)
    result = {
        **binding,
        "status": "COMPLETED",
        "check_only": False,
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
        "records": candidate_records,
        "records_processed": len(candidate_records),
        "records_expected": len(SEED_SCENARIO),
        "development_only": True,
        "development_screen": bool(comparison["development_screen"]),
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
    }
    _write_json(output / "binding_manifest.json", {**binding, "status": "COMPLETED"})
    _write_json(output / "manifest.json", result)
    _write_json(output / "progress.json", {
        "stage": STAGE,
        "status": "COMPLETED",
        "processed": len(candidate_records),
        "expected": len(SEED_SCENARIO),
        "development_screen": bool(comparison["development_screen"]),
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
    })
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--fit-dir", type=Path)
    inputs.add_argument("--summarize-existing-eval", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.summarize_existing_eval is not None:
            _require(not args.check_only, "offline summary does not support --check-only")
            result = summarize_existing_eval(
                input_root=args.summarize_existing_eval,
                output_dir=args.output_dir,
            )
        else:
            result = run(fit_dir=args.fit_dir, output_dir=args.output_dir, check_only=args.check_only)
    except (ProposalHeadEvaluationError, FileNotFoundError, ValueError, OSError) as exc:
        print(json.dumps({"stage": STAGE, "status": "ERROR", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({
        "stage": STAGE,
        "status": result.get("status"),
        "records_processed": result.get("records_processed", 0),
        "development_screen": result.get("development_screen", False),
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
    }, sort_keys=True), flush=True)
    return 0 if result.get("status") in {
        "CHECK_ONLY_OK", "COMPLETED", "REPORTING_ERROR_CORRECTED"
    } else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
