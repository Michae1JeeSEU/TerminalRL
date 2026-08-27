"""Frozen, disjoint seed partitions for the A5-v2 curriculum."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class LowV2SeedPartition:
    train: tuple[int, ...] = tuple(range(1001, 1025))
    validation: tuple[int, ...] = tuple(range(2001, 2013))
    qualification: tuple[int, ...] = tuple(range(3001, 3013))
    ood: tuple[int, ...] = tuple(range(4001, 4013))

    def __post_init__(self) -> None:
        groups = (self.train, self.validation, self.qualification, self.ood)
        if any(not group for group in groups):
            raise ValueError("every A5-v2 seed partition must be non-empty")
        values = [seed for group in groups for seed in group]
        if len(values) != len(set(values)):
            raise ValueError("A5-v2 train/validation/qualification/OOD seeds must be disjoint")


def freeze_seed_partitions(path: str | Path, partition: LowV2SeedPartition | None = None) -> LowV2SeedPartition:
    partition = partition or LowV2SeedPartition()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps({"schema_version": "A5V2-TRAINING-CURRICULUM-v3", "seed_partition": asdict(partition)}))
    if destination.exists() and json.loads(destination.read_text(encoding="utf-8")) != payload:
        raise RuntimeError("refusing to overwrite an already frozen A5-v2 seed partition")
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return partition
