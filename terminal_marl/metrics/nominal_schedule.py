"""Reusable nominal landing schedule estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..core.config import SimConfig


@dataclass
class NominalScheduleEstimator:
    config: SimConfig
    mode: str = "analytic"
    rule_only_reference: Mapping[int, float] | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"analytic", "rule-only-reference"}:
            raise ValueError("mode must be analytic or rule-only-reference")
        if self.mode == "rule-only-reference" and self.rule_only_reference is None:
            raise ValueError("rule-only-reference mode requires reference times")

    def estimate(
        self,
        position: np.ndarray,
        *,
        spawn_time: float,
        sequence_index: int,
    ) -> float:
        if self.mode == "rule-only-reference":
            reference = self.rule_only_reference or {}
            if sequence_index in reference:
                return spawn_time + float(reference[sequence_index])
            if reference:
                return spawn_time + float(np.mean(list(reference.values())))
        center = np.asarray(self.config.center, dtype=float)
        position = np.asarray(position, dtype=float)
        horizontal = float(np.linalg.norm(position[:2] - center[:2]))
        descent = max(0.0, float(position[2] - center[2]))
        return (
            spawn_time
            + horizontal / max(self.config.nominal_speed, 1e-6)
            + descent / max(self.config.max_descent_rate, 1e-6)
            + max(0, sequence_index - 1) * self.config.final_min_time_gap_s
        )
