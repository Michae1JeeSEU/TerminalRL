"""Teacher-off A5-v2 qualification gate and report schema."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..core.config import SimConfig
from .low_v2_protocol import validate_low_v2_config


REQUIRED_N_MATRIX = frozenset((1, 5, 10, 20, 40, 60))


@dataclass(frozen=True)
class StudentQualificationRecord:
    n_uavs: int
    split: str  # id or ood
    collisions: int
    unique_nmac: int
    hard_separation_violations: int
    release_intents: int
    shield_interventions: int
    completed: int
    unresolved: int


@dataclass(frozen=True)
class StudentQualificationReport:
    passed: bool
    hard_safety_passed: bool
    independence_passed: bool
    lifecycle_passed: bool
    missing_n: tuple[int, ...]
    id_interventions_per_100: float
    ood_interventions_per_100: float


def qualify_student_records(config: SimConfig, records: Iterable[StudentQualificationRecord]) -> StudentQualificationReport:
    validate_low_v2_config(config, for_training=False)
    rows = tuple(records)
    seen = {row.n_uavs for row in rows}
    missing = tuple(sorted(REQUIRED_N_MATRIX - seen))
    hard_safe = bool(rows) and all(row.collisions == row.unique_nmac == row.hard_separation_violations == 0 for row in rows)
    lifecycle = bool(rows) and all(row.unresolved == 0 for row in rows)
    def rate(split: str) -> float:
        intents = sum(row.release_intents for row in rows if row.split == split)
        interventions = sum(row.shield_interventions for row in rows if row.split == split)
        return 0.0 if intents == 0 else 100.0 * interventions / intents
    id_rate, ood_rate = rate("id"), rate("ood")
    # Zero release intents is a policy-collapse diagnostic, never an
    # independence pass.
    id_intents = sum(row.release_intents for row in rows if row.split == "id")
    ood_intents = sum(row.release_intents for row in rows if row.split == "ood")
    independence = id_intents > 0 and ood_intents > 0 and id_rate <= config.shield_id_interventions_per_100_intents and ood_rate <= config.shield_ood_interventions_per_100_intents
    passed = not missing and hard_safe and lifecycle and independence
    return StudentQualificationReport(passed, hard_safe, independence, lifecycle, missing, id_rate, ood_rate)
