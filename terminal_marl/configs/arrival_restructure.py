"""Frozen configuration and seed protocol for the arrival restructuring pilot.

This module contains protocol data only.  The native environment, trajectory
generator, safety filter, energy model, and landing-zone lifecycle remain the
authoritative execution path in :mod:`terminal_marl.envs.parallel_env`.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from .p3_v25_course import SCENARIOS as COURSE_SCENARIOS
from .p3_v25_course import build_config as _build_course_config


PROFILE = "arrival_restructure_v1"
PROTOCOL_VERSION = "arrival_restructure_protocol_v2"

PILOT_SCENARIOS = tuple(COURSE_SCENARIOS)
REFERENCE_SIZES = (3, 4, 5, 6)

CONTROLLERS = (
    "rule_nominal_k3",
    "energy_slack",
    "rolling_reorder",
)

SMOKE_SEEDS = {
    "flow6_mixed": 896091,
    "dense24_fixed": 896092,
    "dense24_mixed": 896093,
}
PILOT_SEEDS = {
    "flow6_mixed": 896101,
    "dense24_fixed": 896102,
    "dense24_mixed": 896103,
}
REFERENCE_SEEDS = {3: 896113, 4: 896114, 5: 896115, 6: 896116}

REFERENCE_REGISTERED_PERMUTATIONS = 6
MAX_REFERENCE_PERMUTATIONS = 24
MAX_REFERENCE_CANDIDATES = 48
MAX_TICKS = 2400
MAX_WALL_SECONDS = 7200.0

LEGACY_AUTHORITY = "legacy"
EXTERNAL_AUTHORITY = "external_dispatch_v1"


def _validate_scenario(scenario: str) -> None:
    if scenario not in PILOT_SCENARIOS:
        raise ValueError(f"unknown arrival restructuring scenario: {scenario}")


def build_config(
    scenario: str = "flow6_mixed",
    *,
    reference_n: int | None = None,
    authority: str = LEGACY_AUTHORITY,
):
    """Build the native task-neutral configuration for one protocol arm.

    ``reference_n`` selects the closed warm-start reference cohort.  The
    regular pilot keeps the existing course builder unchanged apart from the
    explicit release-order authority.  The external authority value is
    accepted here because the corresponding native config validation is owned
    by the core worker.
    """

    _validate_scenario(scenario)
    if authority not in {LEGACY_AUTHORITY, EXTERNAL_AUTHORITY}:
        raise ValueError(f"unsupported control authority: {authority}")

    if reference_n is None:
        config = _build_course_config(scenario)
    else:
        if reference_n not in REFERENCE_SIZES:
            raise ValueError(f"reference_n must be one of {REFERENCE_SIZES}")
        # The small reference uses the current dense fixed-SOC physics profile
        # and only changes cohort size, warm start, fixed schedule, and SOC.
        base = _build_course_config("dense24_fixed")
        config = replace(
            base,
            total_requested_uavs=reference_n,
            warm_start_uavs=reference_n,
            fixed_arrival_schedule=(0.0,) * reference_n,
            initial_soc_mode="fixed",
            initial_soc=0.8,
            initial_soc_min=0.8,
            initial_soc_max=0.8,
            task_priority_enabled=False,
        )

    release_mode = (
        config.release_arbitration_mode
        if authority == LEGACY_AUTHORITY
        else "external_dispatch_v1"
    )
    return replace(config, release_arbitration_mode=release_mode)


def protocol_snapshot(
    *,
    mode: str,
    scenario: str,
    seed: int,
    controller: str,
    authority: str,
    source_commit: str | None = None,
    reference_n: int | None = None,
    smoke: bool = False,
    max_ticks: int = MAX_TICKS,
    wall_seconds: float = MAX_WALL_SECONDS,
) -> dict[str, Any]:
    """Return a serializable protocol header written before execution."""

    return {
        "profile": PROFILE,
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "scenario": scenario,
        "seed": int(seed),
        "controller": controller,
        "control_authority": authority,
        "source_commit": source_commit,
        "reference_n": reference_n,
        "smoke": bool(smoke),
        "max_ticks": int(max_ticks),
        "wall_seconds": float(wall_seconds),
        "reference_registered_permutations": REFERENCE_REGISTERED_PERMUTATIONS,
        "reference_registered_candidates_per_n": 2 * REFERENCE_REGISTERED_PERMUTATIONS,
        "reference_registered_candidates_total": 2 * REFERENCE_REGISTERED_PERMUTATIONS * len(REFERENCE_SIZES),
        "reference_permutation_selection": "six evenly spaced lexicographic indices; replace last with initial energy order if absent",
        "protocol_revision_reason": "bounded mechanism/reference first batch; dense two-tick performance only, no full-rollout outcome data used",
        "concurrency_budget": {"pilot_workers": 9, "reference_workers": 15, "total_max": 24, "nested_pools": False},
        "max_reference_permutations": MAX_REFERENCE_PERMUTATIONS,
        "max_reference_candidates": MAX_REFERENCE_CANDIDATES,
        "q_consumed": False,
        "ppo_allowed": False,
    }


def config_snapshot(config: Any) -> dict[str, Any]:
    """Expose the native config without mutating it."""

    if hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    raise TypeError("config must be a dataclass or object with __dict__")


__all__ = [
    "CONTROLLERS",
    "EXTERNAL_AUTHORITY",
    "LEGACY_AUTHORITY",
    "MAX_REFERENCE_CANDIDATES",
    "MAX_REFERENCE_PERMUTATIONS",
    "MAX_TICKS",
    "MAX_WALL_SECONDS",
    "PILOT_SCENARIOS",
    "PILOT_SEEDS",
    "PROFILE",
    "PROTOCOL_VERSION",
    "REFERENCE_REGISTERED_PERMUTATIONS",
    "REFERENCE_SEEDS",
    "REFERENCE_SIZES",
    "SMOKE_SEEDS",
    "build_config",
    "config_snapshot",
    "protocol_snapshot",
]
