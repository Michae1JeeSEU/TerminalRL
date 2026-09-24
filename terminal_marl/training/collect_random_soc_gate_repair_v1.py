"""Collect bounded label-only rows for the random-SOC gate repair dataset.

The collector validates a frozen checkpoint against the ordinary execution
configuration, then changes only the Teacher labeler fields for a deterministic
student rollout.  It writes the raw visited-label sidecar and a separate
actor-only sidecar for later supervised use; this module never trains or
promotes a policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from numbers import Integral
from pathlib import Path
import time
from typing import Any

from terminal_marl.training import diagnose_trained_dense30_v2 as diagnostic
from terminal_marl.training import student_rollout_v4_2 as student


SCENARIOS = ("flow6", "flow12", "dense24")
DEFAULT_LABEL_STRIDE = 10
STAGE = "V4_2_RANDOM_SOC_GATE_REPAIR_COLLECTION_V1"


def _validate_scenario(scenario: str) -> str:
    value = str(scenario)
    if value not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
    return value


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row must be an object: {path}")
                rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_collection_configs(scenario: str):
    """Return the validated execution config for the collection scenario."""

    scenario = _validate_scenario(scenario)
    config = diagnostic.build_scenario_config(
        scenario, reconcile_late_merge_slots=False,
    )
    return config


def _actor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if bool(row.get("actor_mask", False))]


def run(
    *,
    checkpoint: str | Path,
    output: str | Path,
    scenario: str,
    seed: int,
    max_steps: int,
    label_stride: int = DEFAULT_LABEL_STRIDE,
) -> dict[str, Any]:
    """Run one bounded deterministic label-only collection episode."""

    scenario = _validate_scenario(scenario)
    max_steps = _positive_int(max_steps, name="max_steps")
    label_stride = _positive_int(label_stride, name="label_stride")
    checkpoint = Path(checkpoint)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    config = build_collection_configs(scenario)
    # Strict checkpoint validation remains on the fully legal execution
    # contract.  External state-only label queries do not alter this config.
    policy, checkpoint_info = diagnostic.load_trained_checkpoint(
        checkpoint, config,
    )
    weight_digest_before = diagnostic._policy_weight_digest(policy)
    started = time.perf_counter()
    _, rollout_summary = student.rollout(
        policy,
        config,
        seed=int(seed),
        max_steps=max_steps,
        output=output,
        deterministic=True,
        collect_labels=True,
        collect_diagnostics=True,
        collect_catalogue_diagnostics=False,
        label_stride_steps=label_stride,
        state_only_labels=True,
    )
    elapsed = time.perf_counter() - started
    weight_digest_after = diagnostic._policy_weight_digest(policy)

    raw_path = output / "labels.jsonl"
    raw_rows = _read_jsonl(raw_path)
    actor_rows = _actor_rows(raw_rows)
    actor_path = output / "actor_labels.jsonl"
    _write_jsonl(actor_path, actor_rows)

    source_sha = checkpoint_info.get("checkpoint_sha256")
    if source_sha is None and checkpoint.is_file():
        source_sha = _sha256(checkpoint)
    summary = dict(rollout_summary)
    summary.update({
        "stage": STAGE,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": source_sha,
        "scenario": scenario,
        "seed": int(seed),
        "max_steps": max_steps,
        "label_stride_steps": label_stride,
        "execution_config": diagnostic._config_payload(config),
        "label_config": diagnostic._config_payload(config),
        "config": diagnostic._config_payload(config),
        "teacher_enabled": bool(config.teacher_enabled),
        "teacher_mode": config.teacher_mode,
        "external_state_only_labeler": True,
        "labeler_called": True,
        "state_only_labels": True,
        "all_terminal": bool(rollout_summary.get("terminal", False)),
        "weights_unchanged": bool(weight_digest_before == weight_digest_after),
        "weight_digest_before": weight_digest_before,
        "weight_digest_after": weight_digest_after,
        "PPO_ALLOWED": False,
        "training_called": False,
        "catalogue_collected": False,
        "raw_label_rows": len(raw_rows),
        "actor_label_rows": len(actor_rows),
        "labels_path": str(raw_path),
        "actor_labels_path": str(actor_path),
        "training_rows_source": "actor_labels.jsonl",
        "checkpoint_info": checkpoint_info,
        "collector_wall_elapsed_seconds": float(elapsed),
    })
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=_positive_int_arg, required=True)
    parser.add_argument("--label-stride", type=_positive_int_arg, default=DEFAULT_LABEL_STRIDE)
    args = parser.parse_args()
    result = run(
        checkpoint=args.checkpoint,
        output=args.output,
        scenario=args.scenario,
        seed=args.seed,
        max_steps=args.max_steps,
        label_stride=args.label_stride,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
