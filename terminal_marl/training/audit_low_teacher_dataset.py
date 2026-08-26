"""Fail-closed audit for a teacher dataset before any optional BC warm start."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


REQUIRED_FIELDS = frozenset((
    "observation", "requested_action", "teacher_action", "execution_action", "shield_action",
    "teacher_used", "teacher_kind", "teacher_delay_seconds", "teacher_witness_plan_id", "teacher_verified",
    "correction_magnitude", "predicted_min_sep_before", "d_cpa_before", "t_cpa_before",
    "predicted_min_sep_after", "release_decision", "reward", "terminated", "truncated",
))


@dataclass(frozen=True)
class TeacherDatasetAudit:
    passed: bool
    rows: int
    corrected_rows: int
    correction_fraction: float
    errors: tuple[str, ...]


def audit_teacher_dataset(path: str | Path, *, min_corrected_fraction: float = 0.05) -> TeacherDatasetAudit:
    """Validate shape/provenance and reject collapsed teacher datasets."""
    if not 0.0 <= min_corrected_fraction < 1.0:
        raise ValueError("min_corrected_fraction must be in [0, 1)")
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    errors: list[str] = []
    corrected = 0
    for index, row in enumerate(rows):
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            errors.append(f"row {index}: missing {sorted(missing)}")
            continue
        if len(row["observation"]) != 87:
            errors.append(f"row {index}: observation is not 87D")
        for name in ("requested_action", "execution_action"):
            if row[name] is None or len(row[name]) != 3:
                errors.append(f"row {index}: {name} is not 3D")
        if row["teacher_action"] is not None and len(row["teacher_action"]) != 3:
            errors.append(f"row {index}: teacher_action is not 3D")
        if bool(row["teacher_used"]):
            corrected += 1
            if row["teacher_action"] is None or not row["teacher_verified"] or not row["teacher_witness_plan_id"]:
                errors.append(f"row {index}: teacher correction lacks verified witness provenance")
    fraction = 0.0 if not rows else corrected / len(rows)
    if not rows:
        errors.append("dataset is empty")
    if fraction < min_corrected_fraction:
        errors.append(f"teacher correction fraction {fraction:.3f} < required {min_corrected_fraction:.3f}")
    return TeacherDatasetAudit(not errors, len(rows), corrected, fraction, tuple(errors))
