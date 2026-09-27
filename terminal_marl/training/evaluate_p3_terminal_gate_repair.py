"""Evaluate one frozen gate-adapter repair on the six consumed development cells.

The evaluator reuses the registered V4.2 source records and the direct student
rollout/lifecycle path from ``evaluate_p3_runtime_rank_repair``.  It never
updates either policy, reruns the source baseline, consumes held-out data, or
changes PPO/P3 admission state.
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

from terminal_marl.training import collect_p3_proposal_head_probe_v1 as collector
from terminal_marl.training import evaluate_p3_runtime_rank_repair as rank_eval


STAGE = "P3_TERMINAL_GATE_REPAIR_V1"
SOURCE_CHECKPOINT = rank_eval.SOURCE_CHECKPOINT
SOURCE_CHECKPOINT_SHA256 = rank_eval.SOURCE_CHECKPOINT_SHA256
FORMAL_ROOT = collector.FORMAL_ROOT
LOW_SOC_THRESHOLD = rank_eval.LOW_SOC_THRESHOLD
GATE_ADAPTER_PREFIX = "gate_adapter."
EXPECTED_TRAINABLE_SCOPE = frozenset({"gate_adapter"})


class TerminalGateRepairEvaluationError(ValueError):
    """Raised when the gate-repair fit or evaluation binding is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalGateRepairEvaluationError(message)


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
        raise TerminalGateRepairEvaluationError(f"invalid JSON: {target}") from exc
    _require(isinstance(value, dict), f"JSON object required: {target}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


# These wrappers keep the source of the registered six-cell and lifecycle
# implementation explicit while allowing the gate evaluator to own its fit
# binding and retention screen.
def _evaluation_cells() -> list[dict[str, Any]]:
    return rank_eval._evaluation_cells()


def load_baseline_records(root: str | Path = rank_eval.BASELINE_RECORD_ROOT) -> list[dict[str, Any]]:
    return rank_eval.load_baseline_records(root)


def _baseline_summaries(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return rank_eval._baseline_summaries(records)


def _runtime_record(
    *,
    policy: Any,
    checkpoint: Path,
    config: Any,
    scenario: str,
    seed: int,
    output: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    return rank_eval._runtime_record(
        policy=policy,
        checkpoint=checkpoint,
        config=config,
        scenario=scenario,
        seed=seed,
        output=output,
        smoke=smoke,
    )


def _candidate_summaries(
    records: Sequence[Mapping[str, Any]],
    *,
    smoke: bool = False,
) -> dict[str, dict[str, Any]]:
    return rank_eval._candidate_summaries(records, smoke=smoke)


def _load_policy(checkpoint: Path, config: Any) -> tuple[Any, dict[str, Any]]:
    return rank_eval._load_policy(checkpoint, config)


def _load_fit_report(fit_report_path: str | Path, checkpoint: Path) -> tuple[dict[str, Any], str]:
    report_path = Path(fit_report_path)
    report = _read_json(report_path)
    _require(report.get("status") == "COMPLETED", "fit report status must be COMPLETED")
    _require(
        report.get("source_checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256,
        "fit source checkpoint SHA mismatch",
    )
    metadata = report.get("metadata")
    invariants = report.get("invariants")
    _require(isinstance(metadata, Mapping), "fit report metadata is missing")
    _require(metadata.get("training_scope") == "gate_adapter_only", "fit report training scope is not gate_adapter_only")
    _require(isinstance(invariants, Mapping), "fit report invariants are missing")
    names = invariants.get("trainable_parameter_names")
    _require(isinstance(names, (list, tuple)) and names, "fit report trainable parameter names are missing")
    _require(all(str(name).startswith(GATE_ADAPTER_PREFIX) for name in names), "fit report trainable scope is not gate_adapter-only")
    _require(invariants.get("non_adapter_state_bitwise_unchanged") is True, "fit report non-adapter invariant is false")
    _require(invariants.get("proposal_logits_bitwise_unchanged") is True, "fit report proposal invariant is false")
    _require(invariants.get("checkpoint_roundtrip_model_exact") is True, "fit report checkpoint roundtrip invariant is false")
    _require(invariants.get("adapter_state_changed") is True, "fit report adapter change invariant is false")
    _require(checkpoint.is_file(), f"candidate checkpoint missing: {checkpoint}")
    checkpoint_sha = _sha256(checkpoint)
    declared_sha = report.get("checkpoint_sha256")
    _require(isinstance(declared_sha, str), "fit report checkpoint_sha256 is missing")
    _require(declared_sha == checkpoint_sha, "candidate checkpoint SHA mismatch")
    return report, _sha256(report_path)


def _optimizer_empty(policy: Any) -> bool:
    optimizer = getattr(policy, "optimizer", None)
    state = getattr(optimizer, "state", None)
    return isinstance(state, Mapping) and len(state) == 0


def _tensor_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validate_freeze(source: Any, candidate: Any) -> dict[str, Any]:
    source_state = source.model.state_dict()
    candidate_state = candidate.model.state_dict()
    _require(set(source_state) == set(candidate_state), "source/candidate state keys changed")
    adapter_names = sorted(name for name in candidate_state if name.startswith(GATE_ADAPTER_PREFIX))
    _require(adapter_names, "gate_adapter parameter scope is missing")
    changed = sorted(
        name for name in candidate_state
        if not torch.equal(source_state[name].detach().cpu(), candidate_state[name].detach().cpu())
    )
    nonadapter_changed = sorted(name for name in changed if not name.startswith(GATE_ADAPTER_PREFIX))
    adapter_changed = sorted(name for name in changed if name.startswith(GATE_ADAPTER_PREFIX))
    proposal_names = sorted(name for name in candidate_state if name.startswith("proposal_head."))
    proposal_changed = sorted(name for name in proposal_names if name in changed)
    _require(not nonadapter_changed, f"non-adapter parameters changed: {nonadapter_changed}")
    _require(not proposal_changed, f"proposal parameters changed: {proposal_changed}")
    _require(adapter_changed, "candidate has no gate_adapter parameter change")
    _require(source.model.training is False and candidate.model.training is False, "source/candidate model must be eval")
    _require(_optimizer_empty(source) and _optimizer_empty(candidate), "source/candidate optimizer state must be empty")
    return {
        "gate_adapter_parameter_names": adapter_names,
        "gate_adapter_parameter_changed": bool(adapter_changed),
        "changed_parameter_names": changed,
        "nonadapter_parameter_bitwise_unchanged": not nonadapter_changed,
        "proposal_parameter_bitwise_unchanged": not proposal_changed,
        "source_optimizer_state_empty": _optimizer_empty(source),
        "candidate_optimizer_state_empty": _optimizer_empty(candidate),
        "source_model_digest": _tensor_digest(source_state),
        "candidate_model_digest": _tensor_digest(candidate_state),
    }


def _pair_key(record: Mapping[str, Any]) -> tuple[int, str]:
    try:
        return int(record["seed"]), str(record["scenario"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TerminalGateRepairEvaluationError("evaluation record seed/scenario is missing") from exc


def _pair_records(
    baseline_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
) -> list[tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]]:
    cells = _evaluation_cells()
    expected = {(int(cell["seed"]), str(cell["scenario"])) for cell in cells}

    def index(records: Sequence[Mapping[str, Any]], label: str) -> dict[tuple[int, str], Mapping[str, Any]]:
        result: dict[tuple[int, str], Mapping[str, Any]] = {}
        for record in records:
            key = _pair_key(record)
            _require(key in expected, f"{label} has unexpected cell: {key}")
            _require(key not in result, f"{label} has duplicate cell: {key}")
            result[key] = record
        _require(set(result) == expected, f"{label} does not contain exactly the six evaluation cells")
        return result

    baseline = index(baseline_records, "baseline")
    candidate = index(candidate_records, "candidate")
    return [(dict(cell), baseline[(int(cell["seed"]), str(cell["scenario"]))], candidate[(int(cell["seed"]), str(cell["scenario"]))]) for cell in cells]


def _validate_baseline_records(records: Sequence[Mapping[str, Any]]) -> None:
    """Check the reused source records before any candidate rollout."""

    _pair_records(records, records)
    for record in records:
        _counts(record)
        _require(record.get("lifecycle_valid") is True, "source baseline lifecycle is not complete")
        _require(record.get("safety_nonzero") is not None, "source baseline safety status is missing")
        _require(
            _lifecycle_wait(record, "priority_weighted_wait_seconds") is not None,
            "source baseline priority wait is missing",
        )
        # A cell without low-SOC requests remains a valid N/A observation.
        _lifecycle_wait(record, "low_soc_wait_seconds")


def _counts(record: Mapping[str, Any]) -> dict[str, int]:
    raw = record.get("outcome_counts")
    _require(isinstance(raw, Mapping), "record outcome_counts is missing")
    result: dict[str, int] = {}
    for key in ("normal", "failed_energy", "other", "unresolved"):
        _require(key in raw, f"record outcome_counts.{key} is missing")
        try:
            result[key] = int(raw[key])
        except (TypeError, ValueError) as exc:
            raise TerminalGateRepairEvaluationError(f"record outcome_counts.{key} is invalid") from exc
    result["failed"] = result["failed_energy"] + result["other"]
    return result


def _lifecycle_wait(record: Mapping[str, Any], name: str) -> float | None:
    lifecycle = record.get("lifecycle")
    if not isinstance(lifecycle, Mapping):
        return None
    if name == "priority_weighted_wait_seconds":
        value = lifecycle.get(name)
    else:
        low_soc = lifecycle.get("low_soc")
        if not isinstance(low_soc, Mapping):
            return None
        count = low_soc.get("count")
        observed = low_soc.get("observed_wait_count", count)
        value = low_soc.get("mean_wait_seconds")
        if count is None or int(count or 0) <= 0 or observed is None or int(observed or 0) <= 0:
            return None
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _wait_check(candidate: float | None, baseline: float | None) -> dict[str, Any]:
    if candidate is None and baseline is None:
        return {"status": "N/A", "candidate": None, "baseline": None, "passed": True, "strict_improvement": False}
    if candidate is None or baseline is None:
        return {"status": "UNAVAILABLE", "candidate": candidate, "baseline": baseline, "passed": False, "strict_improvement": False}
    return {
        "status": "CHECKED",
        "candidate": float(candidate),
        "baseline": float(baseline),
        "passed": bool(candidate <= baseline),
        "strict_improvement": bool(candidate < baseline),
    }


def _cell_comparison(cell: Mapping[str, Any], baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    base_counts = _counts(baseline)
    cand_counts = _counts(candidate)
    lifecycle_ok = candidate.get("lifecycle_valid") is True and not candidate.get("error")
    base_safety = baseline.get("safety_nonzero")
    cand_safety = candidate.get("safety_nonzero")
    safety_available = base_safety is not None and cand_safety is not None
    safety_passed = bool(safety_available and (not bool(cand_safety) or bool(base_safety)))
    checks = {
        "normal": {"baseline": base_counts["normal"], "candidate": cand_counts["normal"], "passed": cand_counts["normal"] >= base_counts["normal"]},
        "failed": {"baseline": base_counts["failed"], "candidate": cand_counts["failed"], "passed": cand_counts["failed"] <= base_counts["failed"]},
        "unresolved": {"baseline": base_counts["unresolved"], "candidate": cand_counts["unresolved"], "passed": cand_counts["unresolved"] <= base_counts["unresolved"]},
        "safety": {"baseline": base_safety, "candidate": cand_safety, "status": "CHECKED" if safety_available else "UNAVAILABLE", "passed": safety_passed},
        "priority_wait": _wait_check(
            _lifecycle_wait(candidate, "priority_weighted_wait_seconds"),
            _lifecycle_wait(baseline, "priority_weighted_wait_seconds"),
        ),
        "low_soc_wait": _wait_check(_lifecycle_wait(candidate, "low_soc_wait_seconds"), _lifecycle_wait(baseline, "low_soc_wait_seconds")),
    }
    return {
        "seed": int(cell["seed"]),
        "scenario": str(cell["scenario"]),
        "split": str(cell["split"]),
        "lifecycle_valid": lifecycle_ok,
        "checks": checks,
        "passed": bool(
            lifecycle_ok
            and all(bool(checks[name]["passed"]) for name in ("normal", "failed", "unresolved", "safety"))
        ),
    }


def _aggregate_wait_comparison(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    priority = _wait_check(
        candidate.get("priority_weighted_wait_seconds"),
        baseline.get("priority_weighted_wait_seconds"),
    )
    low_soc = _wait_check(candidate.get("low_soc_wait_seconds"), baseline.get("low_soc_wait_seconds"))
    return {
        "priority_wait": priority,
        "low_soc_wait": low_soc,
        "passed": bool(priority["passed"] and low_soc["passed"]),
        "strict_improvement": bool(priority["strict_improvement"] or low_soc["strict_improvement"]),
    }


def _compare_records(
    baseline_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    baseline_summary: Mapping[str, Mapping[str, Any]],
    candidate_summary: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    pairs = _pair_records(baseline_records, candidate_records)
    cells = [_cell_comparison(cell, baseline, candidate) for cell, baseline, candidate in pairs]
    aggregate = _aggregate_wait_comparison(
        baseline_summary["overall"],
        candidate_summary["overall"],
    )
    return {
        "cells": cells,
        "all_cells_passed": bool(all(item["passed"] for item in cells)),
        "aggregate_wait": aggregate,
        "candidate_retained": bool(all(item["passed"] for item in cells) and aggregate["passed"] and aggregate["strict_improvement"]),
    }


def _progress(output: Path, *, status: str, processed: int, expected: int, error: str | None = None) -> None:
    value: dict[str, Any] = {
        "stage": STAGE,
        "status": status,
        "processed": int(processed),
        "expected": int(expected),
        "development_only": True,
        "source_rollouts": 0,
        "candidate_rollouts": int(processed),
        "heldout_consumed": False,
        "PPO_ALLOWED": False,
        "P3_QUALIFIED": False,
    }
    if error:
        value["error"] = str(error)
    _write_json(output / "progress.json", value)


def _failure_record(*, scenario: str, seed: int, checkpoint: Path, error: BaseException) -> dict[str, Any]:
    return {
        "label": "terminal_gate_repair_candidate",
        "scenario": scenario,
        "seed": int(seed),
        "status": "failed",
        "checkpoint": str(checkpoint),
        "error": {"type": type(error).__name__, "message": str(error)},
        "lifecycle_valid": False,
        "outcome_counts": {"requests": int(rank_eval.EXPECTED_REQUESTS[scenario]), "normal": 0, "emergency": 0, "failed_energy": 0, "other": 0, "unresolved": int(rank_eval.EXPECTED_REQUESTS[scenario])},
        "safety_nonzero": None,
        "lifecycle": None,
        "teacher_called": False,
        "training_or_update_called": False,
    }


def run(
    *,
    checkpoint: str | Path,
    fit_report: str | Path,
    output_dir: str | Path,
    check_only: bool = False,
) -> dict[str, Any]:
    candidate_checkpoint = Path(checkpoint)
    output = Path(output_dir)
    _require(not output.exists(), f"evaluation output must be fresh: {output}")
    fit_binding, fit_report_sha = _load_fit_report(fit_report, candidate_checkpoint)
    cells = _evaluation_cells()
    baseline_records = load_baseline_records()
    _validate_baseline_records(baseline_records)
    baseline_summary = _baseline_summaries(baseline_records)
    protocol, configs, source_records, formal_manifest = collector._load_source_inputs(Path(FORMAL_ROOT))
    del protocol, source_records
    source_config = configs["dense24"]
    _require(SOURCE_CHECKPOINT.is_file(), f"source checkpoint missing: {SOURCE_CHECKPOINT}")
    _require(_sha256(SOURCE_CHECKPOINT) == SOURCE_CHECKPOINT_SHA256, "source checkpoint SHA mismatch")
    source_policy, _source_payload = _load_policy(SOURCE_CHECKPOINT, source_config)
    candidate_policy, _candidate_payload = _load_policy(candidate_checkpoint, source_config)
    freeze = _validate_freeze(source_policy, candidate_policy)
    candidate_digest_before = freeze["candidate_model_digest"]
    binding: dict[str, Any] = {
        "stage": STAGE,
        "fit_report": str(Path(fit_report)),
        "fit_report_sha256": fit_report_sha,
        "fit_status": fit_binding.get("status"),
        "fit_scope": fit_binding["metadata"]["training_scope"],
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": _sha256(candidate_checkpoint),
        "source_manifest": str(formal_manifest),
        "freeze": freeze,
        "source_optimizer_state_empty": True,
        "candidate_optimizer_state_empty": True,
        "source_rollouts": 0,
        "candidate_rollouts": 0,
        "teacher_called": False,
        "training_or_update_called": False,
        "heldout_consumed": False,
        "registered_cells": cells,
    }
    if check_only:
        return {
            **binding,
            "status": "CHECK_ONLY_OK",
            "check_only": True,
            "baseline": baseline_summary,
            "candidate": None,
            "comparison": None,
            "candidate_retained": False,
            "records_processed": 0,
            "records_expected": 0,
            "PPO_ALLOWED": False,
            "P3_QUALIFIED": False,
        }

    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "binding_manifest.json", {**binding, "status": "RUNNING"})
    _progress(output, status="RUNNING", processed=0, expected=len(cells))
    records: list[dict[str, Any]] = []
    fatal_error: str | None = None
    for index, cell in enumerate(cells, start=1):
        scenario = str(cell["scenario"])
        seed = int(cell["seed"])
        record_output = output / f"rollout_{scenario}_{seed}"
        try:
            record = _runtime_record(
                policy=candidate_policy,
                checkpoint=candidate_checkpoint,
                config=configs[scenario],
                scenario=scenario,
                seed=seed,
                output=record_output,
                smoke=False,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            record = _failure_record(
                scenario=scenario,
                seed=seed,
                checkpoint=candidate_checkpoint,
                error=exc,
            )
        records.append(dict(record))
        _write_json(output / "records" / f"{scenario}_{seed}.json", dict(record))
        _progress(
            output,
            status="RUNNING",
            processed=index,
            expected=len(cells),
            error=(record.get("error") or {}).get("message") if isinstance(record.get("error"), Mapping) else None,
        )
        weights = record.get("weights")
        weights_changed = (
            record.get("weights_unchanged") is False
            or isinstance(weights, Mapping) and weights.get("unchanged") is False
        )
        if record.get("error") or str(record.get("status", "")).lower() == "failed" or weights_changed:
            if record.get("error") and isinstance(record["error"], Mapping):
                fatal_error = str(record["error"].get("message", "candidate record failed"))
            elif weights_changed:
                fatal_error = "candidate weights changed during evaluation"
            else:
                fatal_error = "candidate record failed"
            break

    candidate_digest_after = _tensor_digest(candidate_policy.model.state_dict())
    candidate_weights_unchanged = candidate_digest_before == candidate_digest_after
    record_errors = any(bool(record.get("error")) for record in records)
    if fatal_error is not None or len(records) != len(cells):
        candidate_summary: dict[str, Any] = {"status": "ABORTED", "records": len(records)}
        comparison: dict[str, Any] = {
            "status": "ABORTED_ON_RECORD_ERROR",
            "error": fatal_error or "candidate evaluation did not complete all six cells",
            "candidate_retained": False,
        }
        candidate_retained = False
        status = "FAILED"
        technical_ok = False
    else:
        candidate_summary = _candidate_summaries(records, smoke=False)
        comparison = _compare_records(baseline_records, records, baseline_summary, candidate_summary)
        technical_ok = bool(not record_errors and candidate_weights_unchanged and all(
            record.get("teacher_called") is False and record.get("training_or_update_called") is False
            for record in records
        ))
        candidate_retained = bool(technical_ok and comparison["candidate_retained"])
        status = "COMPLETED" if not record_errors and candidate_weights_unchanged else "FAILED"
    result = {
        **binding,
        "status": status,
        "check_only": False,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "comparison": comparison,
        "candidate_retained": candidate_retained,
        "selected_checkpoint": str(candidate_checkpoint if candidate_retained else SOURCE_CHECKPOINT),
        "candidate_model_digest_after": candidate_digest_after,
        "candidate_weights_unchanged": candidate_weights_unchanged,
        "technical_ok": technical_ok,
        "evaluation_error": fatal_error,
        "records": records,
        "records_processed": len(records),
        "records_expected": len(cells),
        "requests_expected": int(sum(rank_eval.EXPECTED_REQUESTS[str(cell["scenario"])] for cell in cells)),
        "requests_observed": (
            int(candidate_summary["overall"]["counts"]["requests"])
            if isinstance(candidate_summary.get("overall"), Mapping)
            and isinstance(candidate_summary["overall"].get("counts"), Mapping)
            else None
        ),
        "development_only": True,
        "PPO_ALLOWED": False,
        "P3_QUALIFIED": False,
    }
    _write_json(output / "binding_manifest.json", {**binding, "status": status, "candidate_retained": candidate_retained})
    _write_json(output / "manifest.json", result)
    _progress(output, status=status, processed=len(records), expected=len(cells), error=("candidate evaluation failed" if status == "FAILED" else None))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(
            checkpoint=args.checkpoint,
            fit_report=args.fit_report,
            output_dir=args.output_dir,
            check_only=args.check_only,
        )
    except (TerminalGateRepairEvaluationError, FileNotFoundError, OSError, ValueError) as exc:
        print(json.dumps({"stage": STAGE, "status": "ERROR", "error": str(exc)}, sort_keys=True), flush=True)
        return 2
    print(json.dumps({
        "stage": STAGE,
        "status": result.get("status"),
        "candidate_retained": result.get("candidate_retained", False),
        "records_processed": result.get("records_processed", 0),
        "PPO_ALLOWED": False,
        "P3_QUALIFIED": False,
    }, sort_keys=True), flush=True)
    return 0 if result.get("status") in {"CHECK_ONLY_OK", "COMPLETED"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
