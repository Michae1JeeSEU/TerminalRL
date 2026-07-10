"""Recommended paper experiment matrix for terminal-airspace HRL studies."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentMatrixEntry:
    config: str
    policy: str
    airspace: str
    low_level: str
    safety: str
    purpose: str


RECOMMENDED_EXPERIMENT_MATRIX: tuple[ExperimentMatrixEntry, ...] = (
    ExperimentMatrixEntry(
        "fixed_rule",
        "rule-only",
        "fixed",
        "rule-only",
        "enabled",
        "fixed-airspace rule baseline",
    ),
    ExperimentMatrixEntry(
        "fixed_residual",
        "random-policy",
        "fixed",
        "intention residual placeholder",
        "enabled",
        "low-level residual baseline without dynamic airspace",
    ),
    ExperimentMatrixEntry(
        "dynamic_rule",
        "dynamic-rule",
        "dynamic heuristic",
        "rule-only",
        "enabled",
        "heuristic dynamic airspace reconfiguration baseline",
    ),
    ExperimentMatrixEntry(
        "dynamic_hrl",
        "high-level-ppo",
        "dynamic learned",
        "rule/intention placeholder",
        "enabled",
        "full high-level dynamic airspace policy",
    ),
    ExperimentMatrixEntry(
        "dynamic_hrl",
        "high-level-random",
        "dynamic random",
        "rule-only",
        "enabled",
        "random high-level ablation",
    ),
    ExperimentMatrixEntry(
        "dynamic_hrl",
        "high-level-static",
        "static high-level",
        "rule-only",
        "enabled",
        "static high-level baseline",
    ),
    ExperimentMatrixEntry(
        "full_hrl_no_safety",
        "high-level-ppo",
        "dynamic learned",
        "rule/intention placeholder",
        "disabled, simulation_only",
        "safety-filter ablation",
    ),
)


def recommended_experiment_matrix() -> list[dict[str, str]]:
    return [asdict(entry) for entry in RECOMMENDED_EXPERIMENT_MATRIX]


def export_experiment_matrix(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = recommended_experiment_matrix()
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/experiment_matrix.csv"))
    args = parser.parse_args()
    print(export_experiment_matrix(args.output))


if __name__ == "__main__":
    main()

