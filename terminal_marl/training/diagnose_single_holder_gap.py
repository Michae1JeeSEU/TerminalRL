"""Paired rule-only diagnostic for the single-holder merge-gap fix.

The default variant uses the current antipode rule.  ``--legacy-single-holder-gap``
temporarily restores the historical same-angle behavior only when the target
ring contains exactly one holder; empty and multi-holder gap calculations
continue through the authoritative scheduler method.  The patched method is
restored before this wrapper returns.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from terminal_marl.core.merge_scheduler import MergeScheduler
from terminal_marl.training.diagnose_energy_reachability_guard_smoke import run_smoke


def _legacy_single_holder_gap_center(self, uav, target, time, snapshot, speed):
    """Historical same-angle branch used only for the paired diagnostic."""
    states = [item for item in snapshot.active.values() if item.ring_role is target]
    if len(states) == 1:
        horizon = max(0.0, time - snapshot.now)
        ring = snapshot.airspace.actual_ring(target)
        predicted = ((states[0].ring_progress or 0.0) + speed * horizon / ring.radius) % (2.0 * math.pi)
        return predicted
    return _legacy_single_holder_gap_center.original(self, uav, target, time, snapshot, speed)


def run_variant(
    *,
    seed: int,
    legacy_single_holder_gap: bool,
    initial_soc_min: float,
    initial_soc_max: float,
    max_steps: int,
    progress_every_steps: int,
    output: Path,
    dense_warm_start_uavs: int | None = None,
    final_approach_geometry_mode: str = "brake_bezier_v1",
    release_arbitration_mode: str = "downstream_priority",
    reconcile_late_merge_slots_enabled: bool = False,
) -> dict:
    """Run one variant and write a variant-labelled, non-overwriting output."""
    output = Path(output)
    progress_path = output.with_name(f"{output.name}.progress.json")
    if output.exists() or progress_path.exists():
        raise FileExistsError(f"output or progress path already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    variant = "legacy_same_angle" if legacy_single_holder_gap else "antipode_fix"
    temporary = output.with_name(f".{output.name}.{variant}.tmp")
    temporary_progress = temporary.with_name(f"{temporary.name}.progress.json")
    if temporary.exists() or temporary_progress.exists():
        raise FileExistsError(f"temporary output path already exists: {temporary}")

    original = MergeScheduler._gap_center
    if legacy_single_holder_gap:
        _legacy_single_holder_gap_center.original = original
        MergeScheduler._gap_center = _legacy_single_holder_gap_center
    try:
        result = run_smoke(
            seed=seed,
            guard_on=True,
            initial_soc_min=initial_soc_min,
            initial_soc_max=initial_soc_max,
            max_steps=max_steps,
            progress_every_steps=progress_every_steps,
            output=temporary,
            dense_warm_start_uavs=dense_warm_start_uavs,
            final_approach_geometry_mode=final_approach_geometry_mode,
            release_arbitration_mode=release_arbitration_mode,
            reconcile_late_merge_slots_enabled=reconcile_late_merge_slots_enabled,
        )
    finally:
        MergeScheduler._gap_center = original

    payload = json.loads(temporary.read_text(encoding="utf-8"))
    payload["single_holder_gap_variant"] = variant
    payload["legacy_single_holder_gap"] = bool(legacy_single_holder_gap)
    payload["progress_path"] = str(progress_path)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.unlink()
    temporary_progress.replace(progress_path)
    result["single_holder_gap_variant"] = variant
    result["legacy_single_holder_gap"] = bool(legacy_single_holder_gap)
    result["progress_path"] = str(progress_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initial-soc-min", type=float, default=0.55)
    parser.add_argument("--initial-soc-max", type=float, default=0.90)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--progress-every-steps", type=int, default=100)
    parser.add_argument("--dense-warm-start-uavs", type=int, default=30)
    parser.add_argument(
        "--final-approach-geometry-mode",
        choices=("brake_bezier_v1", "continuous_inward_v1"),
        default="continuous_inward_v1",
    )
    parser.add_argument(
        "--release-arbitration-mode",
        choices=("downstream_priority", "downstream_soc_priority"),
        default="downstream_priority",
    )
    parser.add_argument(
        "--reconcile-late-merge-slots",
        action="store_true",
        dest="reconcile_late_merge_slots_enabled",
    )
    parser.add_argument(
        "--legacy-single-holder-gap",
        action="store_true",
        help="Reproduce the historical same-angle single-holder branch.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_variant(
        seed=args.seed,
        legacy_single_holder_gap=bool(args.legacy_single_holder_gap),
        initial_soc_min=args.initial_soc_min,
        initial_soc_max=args.initial_soc_max,
        max_steps=args.max_steps,
        progress_every_steps=args.progress_every_steps,
        output=args.output,
        dense_warm_start_uavs=args.dense_warm_start_uavs,
        final_approach_geometry_mode=args.final_approach_geometry_mode,
        release_arbitration_mode=args.release_arbitration_mode,
        reconcile_late_merge_slots_enabled=bool(args.reconcile_late_merge_slots_enabled),
    )
    print(json.dumps({
        "output": str(args.output),
        "variant": result["single_holder_gap_variant"],
        "steps": result["steps_executed"],
        "terminal": result["terminal"],
        "failed_energy": result["failed_energy"],
        "emergency_committed": result["emergency_committed"],
        "release_approved": result["release_approved"],
        "safety_error": result["safety_error"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
