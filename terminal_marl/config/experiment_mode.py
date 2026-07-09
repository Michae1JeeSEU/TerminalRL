"""Unified experiment modes used to route terminal-area ablations."""

from __future__ import annotations

from enum import Enum


class ExperimentMode(str, Enum):
    """Configuration switches for the main experiment families."""

    LEGACY_RESIDUAL = "legacy_residual"
    FIXED_RULE = "fixed_rule"
    FIXED_RESIDUAL = "fixed_residual"
    DYNAMIC_RULE = "dynamic_rule"
    DYNAMIC_HRL = "dynamic_hrl"
    CRAZYFLIE_LAB = "crazyflie_lab"

    @classmethod
    def parse(cls, value: "ExperimentMode | str") -> "ExperimentMode":
        """Parse an enum or its stable string value."""

        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            supported = ", ".join(mode.value for mode in cls)
            raise ValueError(f"unsupported experiment mode {value!r}; choose one of: {supported}") from exc

    @property
    def dynamic_airspace(self) -> bool:
        return self in {
            ExperimentMode.DYNAMIC_RULE,
            ExperimentMode.DYNAMIC_HRL,
            ExperimentMode.CRAZYFLIE_LAB,
        }

    @property
    def action_mode(self) -> str:
        if self is ExperimentMode.LEGACY_RESIDUAL:
            return "legacy_residual"
        if self in {ExperimentMode.FIXED_RULE, ExperimentMode.DYNAMIC_RULE}:
            return "rule"
        return "intention_residual"

    @property
    def includes_airspace_observation(self) -> bool:
        """Legacy checkpoints retain their original observation dimension."""

        return self is not ExperimentMode.LEGACY_RESIDUAL

    @property
    def simulation_only(self) -> bool:
        """All current modes remain simulation/dry-run modes in this phase."""

        return True
