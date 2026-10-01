"""Run the frozen prefix-fallback candidate with the BEFORE checkpoint.

This module is a scoped adapter over ``run_p3_prefix_fallback_20261001``.  It
keeps the native sampler, controller, lifecycle replay, and metric machinery
in one implementation while changing the frozen candidate and comparison
labels to the pre-training (BEFORE) checkpoint.  Every temporary mutation of
the base module is restored, including when a phase raises.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from terminal_marl.training import run_p3_prefix_fallback_20261001 as _base
from terminal_marl.training import run_p3_trained_proposal_support_20261001 as support


_BASE_PAIR_SUMMARY = _base._pair_summary
_BASE_BUSINESS_REVIEW = _base._business_review
_BASE_BUILD_PREFLIGHT = _base._build_preflight
_BASE_VALIDATE_PREFLIGHT = _base._validate_preflight


STAGE = "P3_BEFORE_PREFIX_FALLBACK_20261001_V1"
PPO_ALLOWED = False
HELDOUT_USED = False
TEACHER_USED = False
OPTIMIZER_USED = False
OFF_POLICY = True
NO_TRAINING = True

MAX_STEPS = _base.MAX_STEPS
OBSERVATION_SCHEMA = _base.OBSERVATION_SCHEMA
ACTION_SCHEMA = _base.ACTION_SCHEMA
OUTPUT_DIR = Path(
    "/home/user/training_outputs/terminal/"
    "p3_followup_20260930/before_prefix_fallback_20261001"
)
BASELINE_ROOT = _base.BASELINE_ROOT
BASELINE_FULL_SHA256 = _base.BASELINE_FULL_SHA256
BASELINE_FULL_PATH = BASELINE_ROOT / "full.json"
AFTER_PREFIX_BASELINE_ROOT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_followup_20260930/prefix_fallback_20261001_v2"
)
AFTER_PREFIX_BASELINE_FULL_PATH = AFTER_PREFIX_BASELINE_ROOT / "full.json"
AFTER_PREFIX_BASELINE_FULL_SHA256 = (
    "020312813fc2c90a445bd3fe30e94e6739797151fe46efb65f760809a9361c50"
)
BEFORE_CHECKPOINT = support.BEFORE_CHECKPOINT
BEFORE_CHECKPOINT_SHA256 = support.BEFORE_CHECKPOINT_SHA256
WARMUP_CHECKPOINT = support.WARMUP_CHECKPOINT
WARMUP_CHECKPOINT_SHA256 = support.WARMUP_CHECKPOINT_SHA256
PROTOCOL = Path(__file__).resolve().parents[2] / (
    "docs/evidence/v4_2/p3_before_prefix_fallback_protocol_20261001.md"
)
HELPER_PATH = _base.HELPER_PATH
BASE_RUNNER_PATH = Path(_base.__file__).resolve()
WRAPPER_PATH = Path(__file__).resolve()
ARMS = ("before_prefix_fallback",)
CANDIDATE_ARM = ARMS[0]
REFERENCE_ARM_NAME = "before_legacy"
CELL_SPECS = tuple(dict(item) for item in _base.CELL_SPECS)

# The base runner's dependency list is extended rather than replaced.  The
# immutable baseline full summary and both Python entrypoints are therefore
# covered by the fresh preflight hash set.
DEPENDENCY_FILES = {
    **dict(_base.DEPENDENCY_FILES),
    "prefix_fallback_base_runner": BASE_RUNNER_PATH,
    "before_prefix_fallback_wrapper": WRAPPER_PATH,
    "immutable_base_full": BASELINE_FULL_PATH,
    "after_prefix_fallback_full": AFTER_PREFIX_BASELINE_FULL_PATH,
}

PrefixFallbackRunnerError = _base.PrefixFallbackRunnerError
_require = _base._require
_jsonable = _base._jsonable
_read_json = _base._read_json
_write_json = _base._write_json
_sha256 = _base._sha256
_sync_path = _base._sync_path


def _candidate_binding(config: Any) -> dict[str, Any]:
    """Bind the BEFORE checkpoint with a BEFORE-specific candidate label."""

    return support._policy_binding(
        BEFORE_CHECKPOINT, BEFORE_CHECKPOINT_SHA256, config, "before_candidate"
    )


def _historical_checkpoint_binding() -> dict[str, Any]:
    """Bind only the warmup history and existing AFTER comparison input.

    The BEFORE checkpoint is the loaded candidate and is therefore represented
    by ``candidate.checkpoint`` rather than mislabelled as a historical-only
    input.
    """

    after = _base._resolve_checkpoint(
        support.AFTER_CHECKPOINT, support.AFTER_CHECKPOINT_SHA256
    )
    warmup = _base._resolve_checkpoint(
        WARMUP_CHECKPOINT, WARMUP_CHECKPOINT_SHA256
    )
    return {
        "after_existing_comparison": {
            "declared": str(support.AFTER_CHECKPOINT),
            "resolved": str(after),
            "sha256": _base._sha256(after),
            "role": "existing_comparison_only",
        },
        "warmup": {
            "declared": str(WARMUP_CHECKPOINT),
            "resolved": str(warmup),
            "sha256": _base._sha256(warmup),
            "role": "historical_input_binding_only",
        },
    }


def _auxiliary_after_prefix_binding() -> dict[str, Any]:
    path = _base._sync_path(AFTER_PREFIX_BASELINE_FULL_PATH)
    _base._require(path.is_file(), f"AFTER prefix baseline full artifact missing: {path}")
    actual = _base._sha256(path)
    _base._require(
        actual.lower() == AFTER_PREFIX_BASELINE_FULL_SHA256.lower(),
        f"AFTER prefix baseline full artifact SHA changed: {actual}",
    )
    return {
        "path": str(path),
        "sha256": actual,
        "role": "existing_comparison_only",
        "loaded_for_candidate": False,
    }


def _rewrite_pair_labels(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Rename the base pair's AFTER comparison labels for this adapter."""

    result = dict(summary)
    old_delta = result.pop("candidate_minus_after_legacy", None)
    if old_delta is not None:
        result["candidate_minus_before_legacy"] = old_delta
    costs = result.get("costs")
    if isinstance(costs, Mapping):
        rewritten_costs: dict[str, Any] = {}
        for key, value in costs.items():
            if isinstance(value, Mapping):
                row = dict(value)
                if "after_legacy" in row:
                    row["before_legacy"] = row.pop("after_legacy")
                rewritten_costs[str(key)] = row
            else:
                rewritten_costs[str(key)] = value
        result["costs"] = rewritten_costs
    return result


def _pair_summary(
    cell: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    return _rewrite_pair_labels(_BASE_PAIR_SUMMARY(cell, candidate, reference))


def _business_review(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = dict(_BASE_BUSINESS_REVIEW(summaries))
    if "after_legacy_aggregate" in result:
        result["before_legacy_aggregate"] = result.pop("after_legacy_aggregate")
    return result


@contextlib.contextmanager
def _before_base_scope() -> Iterator[None]:
    """Temporarily bind the base runner to the BEFORE comparison contract."""

    overrides: dict[str, Any] = {
        "__file__": str(WRAPPER_PATH),
        "STAGE": STAGE,
        "OUTPUT_DIR": OUTPUT_DIR,
        "BASELINE_ROOT": BASELINE_ROOT,
        "BASELINE_FULL_SHA256": BASELINE_FULL_SHA256,
        "AFTER_CHECKPOINT": BEFORE_CHECKPOINT,
        "AFTER_CHECKPOINT_SHA256": BEFORE_CHECKPOINT_SHA256,
        "PROTOCOL": PROTOCOL,
        "ARMS": ARMS,
        "CANDIDATE_ARM": CANDIDATE_ARM,
        "REFERENCE_ARM_NAME": REFERENCE_ARM_NAME,
        "DEPENDENCY_FILES": dict(DEPENDENCY_FILES),
        "_candidate_binding": _candidate_binding,
        "_historical_checkpoint_binding": _historical_checkpoint_binding,
        "_pair_summary": _pair_summary,
        "_business_review": _business_review,
        "_build_preflight": _build_preflight,
        "_validate_preflight": _validate_preflight,
    }
    previous = {name: getattr(_base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(_base, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(_base, name, value)


def _baseline_guard() -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    with _before_base_scope():
        return _base._baseline_guard()


def _build_preflight() -> dict[str, Any]:
    with _before_base_scope():
        payload = _BASE_BUILD_PREFLIGHT()
        payload["auxiliary_after_prefix_fallback"] = _auxiliary_after_prefix_binding()
        return payload


def _validate_preflight(payload: Mapping[str, Any]) -> None:
    with _before_base_scope():
        _base._require(
            payload.get("auxiliary_after_prefix_fallback")
            == _auxiliary_after_prefix_binding(),
            "AFTER prefix auxiliary binding changed",
        )
        _BASE_VALIDATE_PREFLIGHT(payload)


def _phase_inputs(output: Path) -> tuple[dict[str, Any], Any, dict[int, dict[str, Any]]]:
    with _before_base_scope():
        return _base._phase_inputs(output)


def _arm_record(*, config: Any, cell: Mapping[str, Any], output: Path) -> dict[str, Any]:
    with _before_base_scope():
        return _base._arm_record(config=config, cell=cell, output=output)


def _run_cell(
    output: Path,
    config: Any,
    cell: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    with _before_base_scope():
        return _base._run_cell(output, config, cell, reference)


def run_preflight(output: Path) -> dict[str, Any]:
    with _before_base_scope():
        return _base.run_preflight(output)


def run_pilot(output: Path) -> dict[str, Any]:
    with _before_base_scope():
        return _base.run_pilot(output)


def _validate_pilot_reuse(output: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    with _before_base_scope():
        return _base._validate_pilot_reuse(output, preflight)


def run_full(output: Path) -> dict[str, Any]:
    with _before_base_scope():
        return _base.run_full(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "pilot", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = (
            run_preflight(args.output_dir)
            if args.phase == "preflight"
            else run_pilot(args.output_dir)
            if args.phase == "pilot"
            else run_full(args.output_dir)
        )
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        error = {
            "stage": STAGE,
            "phase": args.phase,
            "status": "FAILED",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        args.output_dir.joinpath("progress.json").write_text(
            json.dumps(error, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"stage": STAGE, "status": "FAILED", "error": str(exc)}, sort_keys=True), flush=True)
        return 2
    print(json.dumps({"stage": STAGE, "status": result.get("status"), "PPO_ALLOWED": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
