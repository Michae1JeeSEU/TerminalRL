"""Reproducible L0 teacher-dataset collection for A5-v2.

L0 precedes all PPO or behaviour-cloning optimisation.  It freezes the seed
partitions, verifies legal warm-start states, and records Full-Teacher labels
for the nominal fixed-airspace curriculum using independent CPU workers.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ..core.config import SimConfig
from ..envs.parallel_env import TerminalParallelEnv
from ..policies import RuleLowLevelPolicy
from .audit_low_teacher_dataset import TeacherDatasetAudit, audit_teacher_dataset
from .build_low_teacher_dataset import TeacherDatasetCohort, build_low_teacher_dataset
from .low_v2_seed_partitions import LowV2SeedPartition, freeze_seed_partitions


L0_WARM_EPISODES = (4, 4, 5)
L0_EMPTY_EPISODES = (2, 2, 3)
L0_COHORT_SIZES = (5, 10, 15)
L0_WARM_FRACTION = 0.65
L0_CLEARENCE_SECONDS = 1800.0
L0_BALANCED_CORRECTION_FRACTION = 0.10


@dataclass(frozen=True)
class L0TeacherDatasetRun:
    mu_hat: float
    medium_rate: float
    output_path: str
    rows: int
    corrected_rows: int
    correction_fraction: float
    warm_start_episodes: int
    from_empty_episodes: int
    probe_episodes: int
    cohorts: tuple[str, ...]


def load_frozen_mu(c0_report: str | Path) -> float:
    """Read only a successfully frozen C0 result, never a provisional scan."""
    payload = json.loads(Path(c0_report).read_text(encoding="utf-8"))
    frozen = payload.get("frozen_capacity", {})
    mu_hat = frozen.get("mu_hat")
    if not frozen.get("frozen") or not isinstance(mu_hat, (float, int)) or mu_hat <= 0.0:
        raise RuntimeError("L0 is blocked until C0 freezes a positive mu_hat")
    return float(mu_hat)


def _l0_config(n_uavs: int, *, warm_start: bool, medium_rate: float) -> SimConfig:
    if n_uavs not in L0_COHORT_SIZES:
        raise ValueError("L0 supports only the predeclared 5/10/15-UAV cohorts")
    interval = 1.0 / medium_rate
    schedule = tuple(0.0 for _ in range(n_uavs)) if warm_start else tuple(index * interval for index in range(n_uavs))
    horizon = L0_CLEARENCE_SECONDS if warm_start else schedule[-1] + L0_CLEARENCE_SECONDS
    return SimConfig(
        experiment_mode="fixed_teacher_heavy", high_level_enabled=False,
        reset_mode="warm_start_rings" if warm_start else "flow_from_empty",
        warm_start_uavs=n_uavs if warm_start else 0,
        fixed_arrival_schedule=schedule, total_requested_uavs=n_uavs,
        arrival_model="finite_cohort", arrival_rate_per_second=medium_rate,
        max_simulation_seconds=horizon, scheduler_mode="teacher_full",
        teacher_enabled=True, teacher_mode="teacher_full", shield_enabled=True,
        initial_soc_mode="uniform", initial_soc_min=0.8, initial_soc_max=0.9,
        gate_probabilities=(0.25, 0.25, 0.25, 0.25),
        mission_probabilities=(1.0, 0.0, 0.0, 0.0),
    )


def l0_cohorts(mu_hat: float) -> tuple[TeacherDatasetCohort, ...]:
    """Return the fixed 65/35 warm-start/from-empty L0 matrix."""
    if mu_hat <= 0.0:
        raise ValueError("mu_hat must be positive")
    medium_rate = 0.5 * mu_hat
    cohorts: list[TeacherDatasetCohort] = []
    for n_uavs, episodes in zip(L0_COHORT_SIZES, L0_WARM_EPISODES):
        cohorts.append(TeacherDatasetCohort(f"warm_n{n_uavs}", _l0_config(n_uavs, warm_start=True, medium_rate=medium_rate), episodes))
    for n_uavs, episodes in zip(L0_COHORT_SIZES, L0_EMPTY_EPISODES):
        cohorts.append(TeacherDatasetCohort(f"empty_n{n_uavs}", _l0_config(n_uavs, warm_start=False, medium_rate=medium_rate), episodes))
    return tuple(cohorts)


def l0_minor_correction_probe(mu_hat: float) -> TeacherDatasetCohort:
    """A short, bounded near-endpoint probe that exposes minor repairs.

    The requested action stays within the canonical [-1, 1] intention box.
    Teacher/SafetyFilter still decide whether any alternate action is legal.
    """
    return TeacherDatasetCohort("probe_minor", _l0_config(15, warm_start=True, medium_rate=0.5 * mu_hat), 2, max_steps=10)


def validate_l0_warm_starts(cohorts: Iterable[TeacherDatasetCohort], *, seed: int) -> list[dict]:
    """Record the initial legal-state checks required before L0 collection."""
    checks: list[dict] = []
    for index, cohort in enumerate(cohorts):
        if cohort.config.reset_mode != "warm_start_rings":
            continue
        env = TerminalParallelEnv(cohort.config)
        env.reset(seed + index)
        positions = [uav.position for uav in env.active.values()]
        distances = [float(np.linalg.norm(left - right)) for number, left in enumerate(positions) for right in positions[number + 1:]]
        minimum = min(distances, default=float("inf"))
        passed = (len(positions) == cohort.config.warm_start_uavs
                  and minimum >= 1.2 * cohort.config.safe_distance - 1e-6
                  and all(uav.merge_reservation_id is None and uav.committed_plan is None for uav in env.active.values())
                  and env.runtime_traffic.telemetry["shield_interventions"] == 0)
        if not passed:
            raise RuntimeError(f"unsafe L0 warm start for {cohort.name}")
        checks.append({"cohort": cohort.name, "seed": seed + index, "active_uavs": len(positions),
                       "minimum_initial_separation": minimum, "required_separation": 1.2 * cohort.config.safe_distance,
                       "passed": passed})
    return checks


class _NearEndpointProbePolicy:
    """Requests a near-endpoint residual to label bounded minor repair cases."""
    def act(self, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {agent_id: np.asarray((1.0, 0.9, 0.9), dtype=np.float32) for agent_id in observations}


def _collect_task(task: tuple[TeacherDatasetCohort, int, str, str]) -> tuple[str, dict]:
    cohort, seed, output_path, policy_kind = task
    policy = _NearEndpointProbePolicy() if policy_kind == "near_endpoint" else None
    result = build_low_teacher_dataset(cohort.config, output_path=output_path, episodes=1, max_steps=cohort.max_steps,
                                       seed=seed, policy=policy)
    return output_path, asdict(result)


def _balanced_rows(rows: list[dict], *, target_fraction: float, seed: int) -> tuple[list[dict], dict]:
    """Keep every teacher witness and stratify ordinary rows by release result."""
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("target_fraction must be in (0, 1)")
    corrected = [row for row in rows if row["teacher_used"]]
    uncorrected = [row for row in rows if not row["teacher_used"]]
    maximum = int(len(corrected) * (1.0 - target_fraction) / target_fraction)
    if not corrected or maximum <= 0:
        raise RuntimeError("L0 balancing requires at least one teacher correction")
    groups: dict[str, list[dict]] = {}
    for row in uncorrected:
        groups.setdefault(str(row["release_decision"]), []).append(row)
    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    leftovers: list[dict] = []
    # Preserve every observed decision family before filling the remaining
    # budget.  This prevents rare safe/unsafe-release witnesses from being
    # erased by common plan-active rows.
    for key in sorted(groups):
        values = groups[key]
        order = rng.permutation(len(values))
        if len(selected) < maximum:
            selected.append(values[order[0]])
            leftovers.extend(values[index] for index in order[1:])
        else:
            leftovers.extend(values[index] for index in order)
    if len(selected) < maximum:
        order = rng.permutation(len(leftovers))
        selected.extend(leftovers[index] for index in order[:maximum - len(selected)])
    balanced = corrected + selected
    order = rng.permutation(len(balanced))
    balanced = [balanced[index] for index in order]
    correction_magnitudes = [float(row["correction_magnitude"]) for row in corrected]
    coverage = {
        "raw_rows": len(rows), "balanced_rows": len(balanced), "corrected_rows": len(corrected),
        "correction_fraction": len(corrected) / len(balanced),
        "teacher_kind_counts": {key: sum(row["teacher_kind"] == key for row in corrected) for key in sorted({row["teacher_kind"] for row in corrected})},
        "release_decision_counts": {key: sum(row["release_decision"] == key for row in balanced) for key in sorted({row["release_decision"] for row in balanced})},
        "minor_corrections": sum(0.0 < value <= 0.25 for value in correction_magnitudes),
        "major_corrections": sum(value > 0.25 for value in correction_magnitudes),
    }
    required = (coverage["minor_corrections"] > 0 and coverage["major_corrections"] > 0
                and coverage["release_decision_counts"].get("approved", 0) > 0
                and coverage["release_decision_counts"].get("safety_wait", 0) > 0
                and coverage["teacher_kind_counts"].get("speed_repair", 0) > 0
                and coverage["teacher_kind_counts"].get("merge_time_repair", 0) > 0)
    if not required:
        raise RuntimeError("L0 dataset lacks required correction/release coverage")
    return balanced, coverage


def run_l0_teacher_dataset(
    *, c0_report: str | Path, output_dir: str | Path, workers: int = 1,
    partition: LowV2SeedPartition | None = None,
) -> L0TeacherDatasetRun:
    """Collect and audit the complete L0 dataset using disjoint train seeds."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    mu_hat = load_frozen_mu(c0_report)
    cohorts = l0_cohorts(mu_hat)
    probe = l0_minor_correction_probe(mu_hat)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    partition = partition or LowV2SeedPartition()
    freeze_seed_partitions(destination / "seed_partitions.json", partition)
    checks = validate_l0_warm_starts(cohorts, seed=partition.train[0])
    episode_count = sum(cohort.episodes for cohort in cohorts) + probe.episodes
    if episode_count > len(partition.train):
        raise ValueError("L0 cohort count exceeds the frozen train seed partition")
    tasks: list[tuple[TeacherDatasetCohort, int, str, str]] = []
    offset = 0
    for cohort in cohorts:
        for episode in range(cohort.episodes):
            seed = partition.train[offset]
            path = destination / f"l0_teacher_dataset.{cohort.name}.seed{seed}.jsonl"
            tasks.append((cohort, seed, str(path), "rule"))
            offset += 1
    for _ in range(probe.episodes):
        seed = partition.train[offset]
        path = destination / f"l0_teacher_dataset.{probe.name}.seed{seed}.jsonl"
        tasks.append((probe, seed, str(path), "near_endpoint"))
        offset += 1
    if workers == 1:
        parts = tuple(_collect_task(task) for task in tasks)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            parts = tuple(executor.map(_collect_task, tasks))
    raw_rows = [json.loads(line) for path, _ in parts for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    raw = destination / "l0_teacher_dataset.raw.jsonl"
    raw.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in raw_rows), encoding="utf-8")
    rows, coverage = _balanced_rows(raw_rows, target_fraction=L0_BALANCED_CORRECTION_FRACTION, seed=partition.train[0])
    combined = destination / "l0_teacher_dataset.jsonl"
    combined.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    audit = audit_teacher_dataset(combined)
    if not audit.passed:
        raise RuntimeError("L0 teacher dataset audit failed: " + "; ".join(audit.errors))
    result = L0TeacherDatasetRun(mu_hat, 0.5 * mu_hat, str(combined), audit.rows, audit.corrected_rows,
                                 audit.correction_fraction, sum(L0_WARM_EPISODES), sum(L0_EMPTY_EPISODES),
                                 probe.episodes, tuple(cohort.name for cohort in cohorts) + (probe.name,))
    manifest = {"stage": "L0_teacher_dataset", "schema_version": "A5V2-TRAINING-CURRICULUM-v3",
                "c0_report": str(c0_report), "warm_start_fraction": L0_WARM_FRACTION,
                "warm_start_checks": checks, "part_summaries": [summary for _, summary in parts],
                "raw_dataset_path": str(raw), "coverage": coverage,
                "run": asdict(result)}
    (destination / "l0_teacher_dataset.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description="Collect the A5-v2 L0 Full-Teacher dataset; no PPO is trained.")
    parser.add_argument("--c0-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(asdict(run_l0_teacher_dataset(c0_report=args.c0_report, output_dir=args.output, workers=args.workers)), indent=2))


if __name__ == "__main__":
    _main()
