"""Bounded adaptive-soft-KL diagnostic probe for the stopped V6 runs.

This runner reuses the frozen V6 source/checkpoint bindings and the bounded
recovery wrapper.  It is a diagnostic continuation with eight episodes per
run; it does not consume heldout seeds or resume the formal schedule.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from terminal_marl.training import run_p3_kl_recovery_probe_v1 as base


STAGE = "P3_ADAPTIVE_KL_PROBE_V1"
RUNNER_MODULE = "terminal_marl.training.run_p3_adaptive_kl_probe_v1"
OUTPUT_ROOT = Path(
    "/home/user/training_outputs/terminal/p3_adaptive_kl_probe_v1_20260926"
)
REFERENCE_OPTIONS: dict[str, float] = {
    "soft_kl_target": 0.02,
    "soft_kl_gain": 400.0,
}
DEV_SEEDS = list(base.DEV_SEEDS)
DEV_SCENARIOS = list(base.DEV_SCENARIOS)
RUNS: dict[int, dict[str, Any]] = {
    0: {
        "seeds": list(range(860080, 860088)),
        "scenarios": [
            "dense24", "flow6", "dense24", "flow12",
            "dense24", "flow6", "dense24", "flow12",
        ],
        "diagnostic_replay_count": 4,
    },
    2: {
        "seeds": list(range(862049, 862057)),
        "scenarios": [
            "flow6", "dense24", "flow12", "dense24",
            "flow6", "dense24", "flow12", "dense24",
        ],
        "diagnostic_replay_count": 4,
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise base.ProbeInputError(message)


def _entry(run_index: int) -> dict[str, Any]:
    _require(run_index in RUNS, "run-index must be 0 or 2")
    entry = dict(RUNS[run_index])
    _require(len(entry["seeds"]) == 8, "adaptive probe must contain 8 seeds")
    _require(len(entry["scenarios"]) == 8, "adaptive probe must contain 8 scenarios")
    _require(int(entry["diagnostic_replay_count"]) == 4,
             "adaptive probe diagnostic replay count must be 4")
    return entry


def _context(run_index: int) -> dict[str, Any]:
    entry = _entry(run_index)
    context = base._protocol_context(
        run_index,
        reference_options=dict(REFERENCE_OPTIONS),
    )
    context["stage"] = STAGE
    context["reference_options"] = dict(REFERENCE_OPTIONS)
    context["report_gate_adapter_delta"] = True
    context["adaptive_entry"] = entry

    # The V6 digest intentionally covers the q tensors and legacy .1/.05
    # bounds.  Replacing only the soft fields must leave that digest equal to
    # the old source reference bound into the stopped checkpoint.
    soft_reference = context["reference"]
    legacy_digest = context["legacy_reference_digest"]
    _require(legacy_digest == context["reference_digest"],
             "soft reference changed the legacy q reference digest")
    context["legacy_reference_digest"] = legacy_digest
    context["reference_soft_settings"] = {
        "soft_kl_target": getattr(soft_reference, "soft_kl_target", None),
        "soft_kl_gain": float(getattr(soft_reference, "soft_kl_gain", 0.0)),
        "coefficient": float(soft_reference.coefficient),
        "max_kl": float(soft_reference.max_kl),
    }
    return context


def _check_only_payload(context: Mapping[str, Any], output: Path) -> dict[str, Any]:
    entry = context["adaptive_entry"]
    soft = dict(context["reference_soft_settings"])
    metrics = context.get("initial_reference_metrics") or {}
    return {
        "stage": f"{STAGE}_CHECK_ONLY",
        "status": "CHECK_ONLY_OK",
        "check_only": True,
        "run_index": int(context["run_index"]),
        "output_created": False,
        "output_existed": bool(output.exists()),
        "output": str(output),
        "source_checkpoint": str(context["source_checkpoint"]),
        "source_checkpoint_sha256": context["source_checkpoint_sha256"],
        "base_checkpoint": str(context["base_checkpoint"]),
        "base_checkpoint_sha256": context["base_checkpoint_sha256"],
        "protocol_sha256": context["protocol_sha256"],
        "reference_digest": context["reference_digest"],
        "legacy_reference_digest": context["legacy_reference_digest"],
        "reference_options": dict(REFERENCE_OPTIONS),
        "reference_soft_settings": soft,
        "effective_kl_coefficient": metrics.get("effective_kl_coefficient"),
        "trainable_parameters": list(context["trainable"]),
        "restored_adam": bool(context["policy"].optimizer.state),
        "seeds": list(entry["seeds"]),
        "scenarios": list(entry["scenarios"]),
        "diagnostic_replay_count": int(entry["diagnostic_replay_count"]),
        "seed_consumption_plan": [
            {
                "episode_index": index,
                "seed": int(seed),
                "scenario": scenario,
                "classification": (
                    "diagnostic_replay_reproduction"
                    if index <= int(entry["diagnostic_replay_count"])
                    else "previously_unconsumed_planned"
                ),
            }
            for index, (seed, scenario) in enumerate(
                zip(entry["seeds"], entry["scenarios"]), start=1
            )
        ],
        "dev_seeds": list(DEV_SEEDS),
        "dev_scenarios": list(DEV_SCENARIOS),
        "heldout_executed": False,
        "formal_continuation": False,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "PPO_ALLOWED": False,
    }


def run(*, run_index: int, output: str | Path, check_only: bool = False) -> dict[str, Any]:
    entry = _entry(run_index)
    target = Path(output)
    context = _context(run_index)
    if check_only:
        return _check_only_payload(context, target)
    _require(not target.exists(), f"output already exists: {target}")
    return base._run(context, target, entry=entry)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=int, choices=(0, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(run_index=args.run_index, output=args.output,
                     check_only=args.check_only)
    except (base.ProbeInputError, FileNotFoundError, ValueError) as exc:
        print({"stage": STAGE, "status": "ERROR", "error": str(exc)})
        return 2
    print(json.dumps(base.v6._jsonable(result), indent=2, sort_keys=True))
    return 0 if result.get("status") in {"CHECK_ONLY_OK", "COMPLETED", "STOPPED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
